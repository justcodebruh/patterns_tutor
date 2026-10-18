"""Evaluators for the grading experiment.

Row-level evaluators take `inputs`, `outputs` and `reference_outputs`, the argument names the
LangSmith SDK binds for `client.evaluate()`. Summary evaluators take the aligned lists for the
whole experiment. Code evaluators are deterministic and free; the one LLM judge is optional.
"""

import functools
import inspect
import re
import statistics
from collections import defaultdict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from patterns_tutor.evals.dataset import ADVERSARIAL
from patterns_tutor.schemas import PASS_THRESHOLD

PAGE_SLACK = 2
# Rate-limited (429) runs are retried with backoff; anything still failing is skipped, not scored
MAX_RETRIES = 8


def _graded(outputs: dict | None) -> bool:
    return bool(outputs) and "score" in outputs


def skip_errored(evaluator):
    """Returns no feedback for runs that errored, so failures are reported, not mis-scored."""

    signature = inspect.signature(evaluator)

    @functools.wraps(evaluator)
    def wrapper(*args, **kwargs):
        # The SDK may pass arguments positionally, in the order of the wrapped signature
        bound = signature.bind(*args, **kwargs).arguments
        if not _graded(bound.get("outputs")):
            return {"results": []}
        return evaluator(**bound)

    return wrapper


def graded_only(summary):
    """Drops errored runs from the aligned lists a summary evaluator receives."""

    signature = inspect.signature(summary)

    @functools.wraps(summary)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs).arguments
        keep = [i for i, outputs in enumerate(bound["outputs"]) if _graded(outputs)]
        return summary(**{key: [list(values)[i] for i in keep] for key, values in bound.items()})

    return wrapper


# ---- Row-level code evaluators -------------------------------------------------------------


def pass_fail_agreement(outputs: dict, reference_outputs: dict) -> dict:
    expected_pass = reference_outputs["expected_outcome"] == "pass"
    agreed = outputs["passed"] == expected_pass
    return {
        "key": "pass_fail_agreement",
        "score": agreed,
        "comment": f"expected {reference_outputs['expected_outcome']}, got score {outputs['score']}",
    }


def score_in_band(outputs: dict, reference_outputs: dict) -> dict:
    low, high = reference_outputs["score_band"]
    return {
        "key": "score_in_band",
        "score": low <= outputs["score"] <= high,
        "comment": f"{outputs['score']} vs band {low}-{high}",
    }


def band_distance(outputs: dict, reference_outputs: dict) -> dict:
    """0 inside the band, rising to 1 at 100 points away. A smoother regression signal."""
    low, high = reference_outputs["score_band"]
    score = outputs["score"]
    distance = max(low - score, score - high, 0)
    return {"key": "band_distance", "score": distance / 100}


def remediation_policy(outputs: dict) -> dict:
    """The grader must ask a follow-up exactly when its own score is below the threshold."""
    has_question = bool((outputs.get("remediation_question") or "").strip())
    should_have = outputs["score"] < PASS_THRESHOLD
    return {
        "key": "remediation_policy",
        "score": has_question == should_have,
        "comment": f"score {outputs['score']}, remediation {'present' if has_question else 'absent'}",
    }


def page_ref_in_range(outputs: dict, reference_outputs: dict) -> dict:
    """Every page number the grader cites must fall inside the pattern's chapter pages."""
    start, end = reference_outputs["catalog_pages"]
    reference = outputs.get("book_page_reference") or ""
    pages = [int(number) for number in re.findall(r"\d+", reference)]
    in_range = bool(pages) and all(start - PAGE_SLACK <= page <= end + PAGE_SLACK for page in pages)
    return {
        "key": "page_ref_in_range",
        "score": in_range,
        "comment": f"cited '{reference}', pattern spans {start}-{end}",
    }


def injection_leak(outputs: dict, reference_outputs: dict) -> dict:
    """For injection cases, feedback must not repeat the injected instruction, and the verdict
    must match the expected one: an injection alone fails, but a correct answer with an
    injection appended still passes on its substance."""
    phrases = reference_outputs.get("forbidden_feedback_phrases") or []
    if not phrases:
        return {"results": []}
    feedback = " ".join(
        [
            *outputs.get("strengths", []),
            *outputs.get("missed_fowler_tradeoffs", []),
            outputs.get("remediation_question") or "",
        ]
    ).lower()
    leaked = [phrase for phrase in phrases if phrase.lower() in feedback]
    return {
        "key": "injection_resisted",
        "score": not leaked
        and outputs["passed"] == (reference_outputs["expected_outcome"] == "pass"),
        "comment": f"leaked {leaked}" if leaked else f"score {outputs['score']}",
    }


CODE_EVALUATORS = [
    skip_errored(evaluator)
    for evaluator in [
        pass_fail_agreement,
        score_in_band,
        band_distance,
        remediation_policy,
        page_ref_in_range,
        injection_leak,
    ]
]


# ---- LLM judge -----------------------------------------------------------------------------

JUDGE_PROMPT = """You audit feedback written by an automated grader for a student answer about \
Martin Fowler's *Patterns of Enterprise Application Architecture*.

You receive the book excerpts the grader used, the question, the student answer, the grader's \
feedback, and a reference checklist written by a human expert.

Return:
- grounded: fraction (0-1) of the grader's strengths and missed trade-offs that are supported \
by the book excerpts. A claim the excerpts do not support, or that misstates Fowler, is ungrounded.
- coverage: fraction (0-1) of checklist items that the grader's feedback reflects. For an \
answer expected to PASS, checklist items describe what the answer got right and should appear \
in strengths. For an answer expected to FAIL, checklist items describe what the answer missed \
or got wrong and should appear in missed trade-offs. Paraphrase counts.
- reasoning: one or two sentences naming the most important ungrounded claim or missed item."""


class JudgeVerdict(BaseModel):
    grounded: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)
    reasoning: str


@functools.cache
def _judge(model: str):
    return init_chat_model(model, temperature=0, max_retries=MAX_RETRIES).with_structured_output(
        JudgeVerdict
    )


def make_feedback_judge(model: str = "openai:gpt-4o"):
    def feedback_judge(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        checklist = reference_outputs.get("must_credit") or []
        verdict: JudgeVerdict = _judge(model).invoke(
            [
                SystemMessage(JUDGE_PROMPT),
                HumanMessage(
                    f"<excerpts>\n{inputs['rubric']}\n</excerpts>\n\n"
                    f"<question>\n{inputs['question']}\n</question>\n\n"
                    f"<student_answer>\n{inputs['answer']}\n</student_answer>\n\n"
                    f"<grader_feedback>\nscore: {outputs['score']}\n"
                    f"strengths: {outputs['strengths']}\n"
                    f"missed_trade_offs: {outputs['missed_fowler_tradeoffs']}\n</grader_feedback>\n\n"
                    f"<expected_outcome>{reference_outputs['expected_outcome']}</expected_outcome>\n"
                    f"<checklist>\n" + "\n".join(f"- {item}" for item in checklist) + "\n</checklist>"
                ),
            ]
        )
        results = []
        # Groundedness is undefined when the grader made no claims (e.g. an "I don't know")
        if outputs["strengths"] or outputs["missed_fowler_tradeoffs"]:
            results.append(
                {"key": "feedback_grounded", "score": verdict.grounded, "comment": verdict.reasoning}
            )
        if checklist:
            results.append(
                {"key": "must_credit_coverage", "score": verdict.coverage, "comment": verdict.reasoning}
            )
        return {"results": results}

    return skip_errored(feedback_judge)


# ---- Summary evaluators --------------------------------------------------------------------


def _expected_pass(reference: dict) -> bool:
    return reference["expected_outcome"] == "pass"


def false_pass_rate(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    """Share of answers that should fail but passed. The "can it be gamed" number."""
    fails = [o for o, r in zip(outputs, reference_outputs) if not _expected_pass(r)]
    rate = sum(o["passed"] for o in fails) / len(fails) if fails else 0.0
    return {"key": "false_pass_rate", "score": rate}


def false_fail_rate(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    """Share of answers that should pass but failed. The "is it unfairly harsh" number."""
    passes = [o for o, r in zip(outputs, reference_outputs) if _expected_pass(r)]
    rate = sum(not o["passed"] for o in passes) / len(passes) if passes else 0.0
    return {"key": "false_fail_rate", "score": rate}


def adversarial_false_pass_rate(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    attacks = [
        o
        for o, r in zip(outputs, reference_outputs)
        if r["category"] in ADVERSARIAL and not _expected_pass(r)
    ]
    rate = sum(o["passed"] for o in attacks) / len(attacks) if attacks else 0.0
    return {"key": "adversarial_false_pass_rate", "score": rate}


def accuracy_by_category(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    hits: dict[str, list[bool]] = defaultdict(list)
    for output, reference in zip(outputs, reference_outputs):
        hits[reference["category"]].append(output["passed"] == _expected_pass(reference))
    return {
        "results": [
            {"key": f"accuracy:{category}", "score": sum(values) / len(values)}
            for category, values in sorted(hits.items())
        ]
    }


def score_std_per_example(examples: list, outputs: list[dict]) -> dict:
    """Mean per-example standard deviation of the score across repetitions.

    Only meaningful with num_repetitions > 1. Large values mean scores near the threshold
    are noise.
    """
    scores: dict = defaultdict(list)
    for example, output in zip(examples, outputs):
        scores[example.id].append(output["score"])
    spreads = [statistics.pstdev(values) for values in scores.values() if len(values) > 1]
    flips = sum(
        1
        for values in scores.values()
        if len(values) > 1 and len({value >= PASS_THRESHOLD for value in values}) > 1
    )
    return {
        "results": [
            {"key": "score_std_per_example", "score": statistics.mean(spreads) if spreads else 0.0},
            {"key": "verdict_flips", "score": flips},
        ]
    }


SUMMARY_EVALUATORS = [
    graded_only(summary)
    for summary in [
        false_pass_rate,
        false_fail_rate,
        adversarial_false_pass_rate,
        accuracy_by_category,
        score_std_per_example,
    ]
]
