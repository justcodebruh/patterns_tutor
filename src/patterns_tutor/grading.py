"""Answer grading as a pure, independently evaluable function.

`grade_answer` has no graph, checkpointer or store dependencies, so the same code path runs
inside the tutor graph and inside LangSmith evaluation experiments.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import get_current_run_tree, traceable

from patterns_tutor.prompts import (
    DEFAULT_GRADER_REV,
    DELIMITED_REVS,
    GRADER_HUMAN_TEMPLATES,
    GRADER_PROMPTS,
    neutralize_answer,
    prompt_sha,
)
from patterns_tutor.schemas import PASS_THRESHOLD, EvaluationResult


def _trace_inputs(inputs: dict) -> dict:
    # The model object is not useful in a trace; the rubric is kept so traces can seed datasets
    return {key: value for key, value in inputs.items() if key != "llm"}


def build_grader_messages(
    *,
    pattern: str,
    layer: str,
    book_pages: str,
    rubric: str,
    question: str,
    answer: str,
    prompt_rev: str = DEFAULT_GRADER_REV,
) -> list:
    system = GRADER_PROMPTS[prompt_rev].format(
        threshold=PASS_THRESHOLD,
        pattern=pattern,
        layer=layer,
        book_pages=book_pages,
        rubric=rubric,
    )
    if prompt_rev in DELIMITED_REVS:
        answer = neutralize_answer(answer)
    human = GRADER_HUMAN_TEMPLATES[prompt_rev].format(question=question, answer=answer)
    return [SystemMessage(system), HumanMessage(human)]


@traceable(run_type="chain", name="grade_answer", process_inputs=_trace_inputs)
def grade_answer(
    *,
    pattern: str,
    layer: str,
    book_pages: str,
    rubric: str,
    question: str,
    answer: str,
    llm: BaseChatModel,
    prompt_rev: str = DEFAULT_GRADER_REV,
) -> EvaluationResult:
    """Grades one answer against Fowler's text. Pass/fail is left to the caller."""
    run = get_current_run_tree()
    if run is not None:
        run.add_metadata(
            {
                "pattern": pattern,
                "layer": layer,
                "grader_prompt_rev": prompt_rev,
                "grader_prompt_sha": prompt_sha(prompt_rev),
            }
        )
    messages = build_grader_messages(
        pattern=pattern,
        layer=layer,
        book_pages=book_pages,
        rubric=rubric,
        question=question,
        answer=answer,
        prompt_rev=prompt_rev,
    )
    return llm.with_structured_output(EvaluationResult).invoke(messages)
