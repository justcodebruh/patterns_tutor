from dataclasses import dataclass
from typing import Optional, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

PASS_THRESHOLD = 75
# First question plus up to 2 remediation follow-ups per pattern
MAX_ATTEMPTS = 3
# Resume value that skips the current pattern without grading it
SKIP_COMMAND = "/skip"


@dataclass
class TutorContext:
    """Per-run configuration passed to the graph via `context=`."""

    user_id: str = "default_user"
    target_layer: Optional[str] = None
    # Test injection point; None means use the default chat model
    llm: Optional[BaseChatModel] = None


class TutorState(MessagesState):
    """Thread state. `messages` holds the question/answer transcript."""

    current_pattern: str
    current_layer: str
    book_pages: str
    book_rubric: str
    active_question: str
    user_answer: str
    attempts: int
    skipped_patterns: list[str]
    last_feedback: Optional[str]
    remediation_hint: Optional[str]


class EvaluationResult(BaseModel):
    """Structured grade of a user's answer against Fowler's text.

    Pass/fail is not part of the model output; callers derive it from
    `score >= PASS_THRESHOLD` so it can never contradict the score.
    """

    score: int = Field(ge=0, le=100, description="0-100 grade against the book rubric.")
    strengths: list[str] = Field(description="Points the answer got right, per Fowler.")
    missed_fowler_tradeoffs: list[str] = Field(
        description="Trade-offs or forces from the book that the answer missed or got wrong."
    )
    book_page_reference: str = Field(
        description="Book page(s) the user should reread, e.g. '184-194'."
    )
    remediation_question: Optional[str] = Field(
        default=None,
        description=(
            f"Only if score < {PASS_THRESHOLD}: one Socratic follow-up question "
            "targeting the most important missed trade-off."
        ),
    )


class InterruptPayload(TypedDict):
    """Value passed to `interrupt()` and surfaced via `result.interrupts[0].value`."""

    layer: str
    pattern: str
    question: str
    attempt: int
    max_attempts: int
    last_feedback: Optional[str]
