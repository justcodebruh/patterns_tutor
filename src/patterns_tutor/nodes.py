from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from patterns_tutor.retrieval import get_llm, pick_next_pattern, retrieve_pattern_rubric
from patterns_tutor.schemas import (
    MAX_ATTEMPTS,
    PASS_THRESHOLD,
    SKIP_COMMAND,
    EvaluationResult,
    InterruptPayload,
    TutorContext,
    TutorState,
)

QUESTION_SYSTEM_PROMPT = """You are a Socratic tutor examining a student on Martin Fowler's \
*Patterns of Enterprise Application Architecture*.

Write ONE question about the pattern "{pattern}" ({layer}). Prefer a concrete architecture \
scenario or code-design trade-off that forces the student to reason about when and why to use \
the pattern, how it works, and how it compares to the related patterns Fowler cites. Ground \
the question strictly in the book excerpts below; do not reveal the answer.

Reply with the question text only.

Book excerpts:
{rubric}"""

GRADER_SYSTEM_PROMPT = """You are a strict examiner grading a student's answer about Martin \
Fowler's *Patterns of Enterprise Application Architecture*. Grade ONLY against the book \
excerpts below, not general industry opinion.

Scoring: 0-100. {threshold}+ means the answer captures Fowler's core mechanism and the key \
trade-offs for this question. Credit correct reasoning in the student's own words; penalize \
claims that contradict the book.

- strengths: what the answer got right, per Fowler.
- missed_fowler_tradeoffs: forces or trade-offs from the excerpts the answer missed or got wrong.
- book_page_reference: the book page range to reread (the excerpts cover pages {book_pages}).
- remediation_question: only if score < {threshold}, one Socratic follow-up question that \
leads the student toward the most important missed trade-off without giving it away.

Pattern: {pattern} ({layer})

Book excerpts:
{rubric}"""


def select_and_challenge(
    state: TutorState, runtime: Runtime[TutorContext]
) -> Command[Literal["select_and_challenge", "evaluate_and_route", "__end__"]]:
    """Prepares a question (phase 1), then pauses for the user's answer (phase 2).

    Phase 1 checkpoints the generated question and loops back to this node, because
    code before `interrupt()` re-runs on resume and would otherwise call the LLM twice.
    """
    if not state.get("active_question"):
        return _prepare_question(state, runtime)

    answer = interrupt(
        InterruptPayload(
            layer=state["current_layer"],
            pattern=state["current_pattern"],
            question=state["active_question"],
            attempt=state.get("attempts", 0) + 1,
            max_attempts=MAX_ATTEMPTS,
            last_feedback=state.get("last_feedback"),
        )
    )

    if answer.strip() == SKIP_COMMAND:
        return Command(
            update={
                "skipped_patterns": state.get("skipped_patterns", []) + [state["current_pattern"]],
                "active_question": "",
                "remediation_hint": None,
                "last_feedback": f"Skipped **{state['current_pattern']}**.",
            },
            goto="select_and_challenge",
        )

    return Command(
        update={"user_answer": answer, "messages": [HumanMessage(answer)]},
        goto="evaluate_and_route",
    )


def _prepare_question(
    state: TutorState, runtime: Runtime[TutorContext]
) -> Command[Literal["select_and_challenge", "__end__"]]:
    # Remediation branch: stay on the same pattern with the grader's follow-up
    if state.get("remediation_hint"):
        question = state["remediation_hint"]
        return Command(
            update={"active_question": question, "messages": [AIMessage(question)]},
            goto="select_and_challenge",
        )

    # New challenge branch
    skipped = set(state.get("skipped_patterns", []))
    current = state.get("current_pattern")
    picked = pick_next_pattern(
        runtime.store,
        runtime.context.user_id,
        runtime.context.target_layer,
        skipped | {current} if current else skipped,
    )
    if picked is None and current:
        # The pattern just left may be the only one still failing in scope; allow it again
        picked = pick_next_pattern(
            runtime.store, runtime.context.user_id, runtime.context.target_layer, skipped
        )
    if picked is None:
        scope = runtime.context.target_layer or "every layer"
        message = f"Every pattern in {scope} is passed"
        if skipped:
            message += f" or skipped ({', '.join(sorted(skipped))})"
        return Command(update={"last_feedback": message + ". 🎉"}, goto="__end__")

    layer, pattern = picked
    rubric, book_pages = retrieve_pattern_rubric(pattern)
    response = get_llm(runtime).invoke(
        [
            SystemMessage(
                QUESTION_SYSTEM_PROMPT.format(pattern=pattern, layer=layer, rubric=rubric)
            ),
            HumanMessage(f"Ask me about {pattern}."),
        ]
    )
    question = response.text.strip()
    return Command(
        update={
            "current_layer": layer,
            "current_pattern": pattern,
            "book_pages": book_pages,
            "book_rubric": rubric,
            "active_question": question,
            "attempts": 0,
            "remediation_hint": None,
            "messages": [AIMessage(question)],
        },
        goto="select_and_challenge",
    )


def evaluate_and_route(
    state: TutorState, runtime: Runtime[TutorContext]
) -> Command[Literal["select_and_challenge"]]:
    """Grades the answer, records mastery, and decides between remediation and moving on."""
    grader = get_llm(runtime).with_structured_output(EvaluationResult)
    grade: EvaluationResult = grader.invoke(
        [
            SystemMessage(
                GRADER_SYSTEM_PROMPT.format(
                    threshold=PASS_THRESHOLD,
                    pattern=state["current_pattern"],
                    layer=state["current_layer"],
                    book_pages=state["book_pages"],
                    rubric=state["book_rubric"],
                )
            ),
            HumanMessage(
                f"Question:\n{state['active_question']}\n\nStudent answer:\n{state['user_answer']}"
            ),
        ]
    )

    passed = grade.score >= PASS_THRESHOLD
    attempts = state.get("attempts", 0) + 1
    book_pages = grade.book_page_reference or state["book_pages"]

    runtime.store.put(
        ("mastery", runtime.context.user_id),
        state["current_pattern"],
        {
            "layer": state["current_layer"],
            "score": grade.score,
            "passed": passed,
            "attempts": attempts,
            "missed_tradeoffs": grade.missed_fowler_tradeoffs,
            "book_pages": book_pages,
        },
    )

    remediate = not passed and attempts < MAX_ATTEMPTS and bool(grade.remediation_question)
    feedback = _format_feedback(state["current_pattern"], grade, passed, book_pages)
    if not passed and not remediate:
        feedback += (
            f"\n\nMoving on for now; **{state['current_pattern']}** will come back "
            "after the patterns you haven't tried yet."
        )

    return Command(
        update={
            "attempts": attempts,
            "active_question": "",
            "remediation_hint": grade.remediation_question if remediate else None,
            "last_feedback": feedback,
            "messages": [AIMessage(feedback)],
        },
        goto="select_and_challenge",
    )


def _format_feedback(
    pattern: str, grade: EvaluationResult, passed: bool, book_pages: str
) -> str:
    verdict = "✅ Passed" if passed else "❌ Not yet"
    lines = [f"### {pattern}: {grade.score}/100 ({verdict})"]
    if grade.strengths:
        lines.append("**Strengths**")
        lines.extend(f"- {item}" for item in grade.strengths)
    if grade.missed_fowler_tradeoffs:
        lines.append("**Missed Fowler trade-offs**")
        lines.extend(f"- {item}" for item in grade.missed_fowler_tradeoffs)
    lines.append(f"📖 Reread pp. {book_pages}")
    return "\n".join(lines)
