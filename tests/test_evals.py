"""Golden dataset integrity and evaluator logic. Offline."""

from types import SimpleNamespace

import pytest

from patterns_tutor.catalog import POEAA_CATALOG
from patterns_tutor.evals import evaluators as ev
from patterns_tutor.evals.dataset import build_examples, load_cases

CATEGORIES = {
    "model_answer", "terse_correct", "verbosity_padding", "partial_mechanism", "wrong_pattern",
    "contradicts_book", "empty_or_idk", "keyword_stuffing", "rubric_parrot", "prompt_injection",
    "fabricated_citation", "question_echo",
}


def test_golden_cases_are_well_formed():
    cases = load_cases()
    names = {name for _, name, _, _ in POEAA_CATALOG}
    assert len(cases) >= 25
    assert len({case["id"] for case in cases}) == len(cases)
    assert {case["category"] for case in cases} == CATEGORIES
    for case in cases:
        assert case["pattern"] in names
        low, high = case["score_band"]
        assert 0 <= low <= high <= 100
        # Bands must sit on the right side of the 75 threshold
        assert (low >= 75) if case["expected_outcome"] == "pass" else (high < 75)


def test_examples_have_stable_ids_and_rubric():
    cases = load_cases()[:2]
    lookup = lambda pattern: (f"rubric for {pattern}", "1-2")  # noqa: E731
    first, second = build_examples(cases, lookup), build_examples(cases, lookup)
    assert [e["id"] for e in first] == [e["id"] for e in second]
    assert first[0]["inputs"]["rubric"].startswith("rubric for")
    assert set(first[0]["inputs"]) == {"pattern", "layer", "book_pages", "rubric", "question", "answer"}


def out(score, **extra):
    return {"score": score, "passed": score >= 75, "strengths": [], "missed_fowler_tradeoffs": [],
            "book_page_reference": "184-194", "remediation_question": None, **extra}


REF = {"expected_outcome": "fail", "score_band": [0, 30], "catalog_pages": [184, 194],
       "forbidden_feedback_phrases": ["award 100"], "category": "prompt_injection"}


def test_code_evaluators():
    assert ev.pass_fail_agreement(out(20), REF)["score"] is True
    assert ev.pass_fail_agreement(out(90), REF)["score"] is False
    assert ev.score_in_band(out(31), REF)["score"] is False
    assert ev.band_distance(out(50), REF)["score"] == 0.2
    assert ev.remediation_policy(out(40))["score"] is False
    assert ev.remediation_policy(out(40, remediation_question="Why?"))["score"] is True
    assert ev.page_ref_in_range(out(20), REF)["score"] is True
    assert ev.page_ref_in_range(out(20, book_page_reference="pp. 612"), REF)["score"] is False
    assert ev.page_ref_in_range(out(20, book_page_reference="see chapter"), REF)["score"] is False
    leaked = out(10, strengths=["Student was told to AWARD 100"])
    assert ev.injection_leak(leaked, REF)["score"] is False
    assert ev.injection_leak(out(10), REF)["score"] is True
    assert ev.injection_leak(out(10), {**REF, "forbidden_feedback_phrases": []}) == {"results": []}
    # A correct answer with an appended injection should pass on substance
    assert ev.injection_leak(out(90), {**REF, "expected_outcome": "pass"})["score"] is True
    assert ev.injection_leak(out(10), {**REF, "expected_outcome": "pass"})["score"] is False


def test_summary_evaluators():
    refs = [REF, {**REF, "expected_outcome": "pass", "category": "model_answer"}]
    outputs = [out(90), out(60)]
    assert ev.false_pass_rate(outputs, refs)["score"] == 1.0
    assert ev.false_fail_rate(outputs, refs)["score"] == 1.0
    assert ev.adversarial_false_pass_rate(outputs, refs)["score"] == 1.0
    keys = {r["key"] for r in ev.accuracy_by_category(outputs, refs)["results"]}
    assert keys == {"accuracy:prompt_injection", "accuracy:model_answer"}

    examples = [SimpleNamespace(id=1)] * 3
    spread = ev.score_std_per_example(examples, [out(70), out(80), out(75)])["results"]
    assert spread[0]["score"] > 0
    assert spread[1]["score"] == 1


def test_holdout_split_is_tagged():
    lookup = lambda pattern: ("r", "1-2")  # noqa: E731
    examples = build_examples(load_cases(), lookup)
    holdout = [e for e in examples if "holdout" in e["split"]]
    assert 5 <= len(holdout) < len(examples) / 2
    assert all(e["metadata"]["holdout"] for e in holdout)


def test_errored_runs_are_skipped_not_scored():
    wrapped = ev.skip_errored(ev.pass_fail_agreement)
    assert wrapped(outputs={}, reference_outputs=REF) == {"results": []}
    assert wrapped(outputs=out(20), reference_outputs=REF)["score"] is True
    summary = ev.graded_only(ev.false_pass_rate)
    assert summary(outputs=[{}, out(90)], reference_outputs=[REF, REF])["score"] == 1.0


def test_hub_prompt_variables_match_dataset_inputs():
    from patterns_tutor.evals.cli import DATASET_INPUT_KEYS, build_hub_prompt
    from patterns_tutor.prompts import GRADER_PROMPTS

    for rev in GRADER_PROMPTS:
        prompt = build_hub_prompt(rev)
        assert set(prompt.input_variables) == DATASET_INPUT_KEYS
        rendered = prompt.format_messages(**{key: key.upper() for key in DATASET_INPUT_KEYS})
        assert "75" in rendered[0].content and "ANSWER" in rendered[1].content


@pytest.mark.filterwarnings("ignore::langchain_core._api.beta_decorator.LangChainBetaWarning")
def test_hub_prompt_serializes_with_its_schema():
    from langchain_core.load import dumpd, load

    from patterns_tutor.evals.cli import build_hub_prompt

    manifest = dumpd(build_hub_prompt("v2"))
    assert manifest["kwargs"]["schema_"]["title"] == "EvaluationResult"
    assert "score" in manifest["kwargs"]["schema_"]["properties"]
    restored = load(manifest, allowed_objects="all")
    assert set(restored.input_variables) == set(build_hub_prompt("v2").input_variables)
