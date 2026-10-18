"""Deterministic tests: no network calls, no OPENAI_API_KEY needed."""

import os
from typing import Optional

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from patterns_tutor import main, nodes, retrieval
from patterns_tutor.graph import build_tutor_graph
from patterns_tutor.schemas import (
    MAX_ATTEMPTS,
    SKIP_COMMAND,
    EvaluationResult,
    InterruptPayload,
    TutorContext,
)

USER = "test_user"
MASTERY = ("mastery", USER)


class FakeTutorLLM:
    """Stands in for the chat model: fixed questions, scripted grades."""

    def __init__(self) -> None:
        self.grades: list[EvaluationResult] = []
        self.question_calls = 0

    def invoke(self, messages) -> AIMessage:
        self.question_calls += 1
        return AIMessage(f"Question {self.question_calls}?")

    def with_structured_output(self, schema) -> RunnableLambda:
        return RunnableLambda(lambda _: self.grades.pop(0))


def grade(score: int, remediation_question: Optional[str] = None) -> EvaluationResult:
    return EvaluationResult(
        score=score,
        strengths=["Named the core mechanism"],
        missed_fowler_tradeoffs=["Ignored the cost of the extra layer"],
        book_page_reference="",
        remediation_question=remediation_question,
    )


@pytest.fixture
def fake_rubric(monkeypatch):
    """Keeps graph tests independent of the Chroma index."""
    monkeypatch.setattr(
        nodes, "retrieve_pattern_rubric", lambda pattern: (f"Fowler on {pattern}.", "1-9")
    )


@pytest.fixture
def tutor(fake_rubric):
    llm = FakeTutorLLM()
    store = InMemoryStore()
    graph = build_tutor_graph(InMemorySaver(), store)
    config = {"configurable": {"thread_id": "test-thread"}}
    context = TutorContext(user_id=USER, llm=llm)

    def step(graph_input) -> Optional[InterruptPayload]:
        result = graph.invoke(graph_input, config=config, context=context, version="v2")
        return result.interrupts[0].value if result.interrupts else None

    return llm, store, step, graph, config, context


# Test 1: Retrieval filter (real index, no embeddings)
@pytest.mark.skipif(not os.path.isdir(retrieval.CHROMA_DIR), reason="vector store not built")
def test_retrieve_pattern_rubric_unit_of_work():
    rubric, book_pages = retrieval.retrieve_pattern_rubric("Unit of Work")

    assert book_pages == "184-194"
    assert "Pages 184-194" in rubric
    for section in retrieval.RUBRIC_SECTIONS:
        assert f"Section: {section}" in rubric
    assert "Section: code_example" not in rubric
    assert rubric == retrieval.retrieve_pattern_rubric("Unit of Work")[0]


@pytest.mark.skipif(not os.path.isdir(retrieval.CHROMA_DIR), reason="vector store not built")
def test_retrieve_pattern_rubric_unknown_pattern():
    with pytest.raises(ValueError):
        retrieval.retrieve_pattern_rubric("Not A Pattern")


# Test 2: Interrupt & resume cycle
def test_first_invoke_halts_with_payload(tutor):
    _, _, step, _, _, _ = tutor

    payload = step({})

    assert set(payload) == set(InterruptPayload.__annotations__)
    assert payload["pattern"] == "Transaction Script"
    assert payload["question"] == "Question 1?"
    assert payload["attempt"] == 1


# Test 3: Store mastery update
def test_answer_updates_mastery(tutor):
    llm, store, step, _, _, _ = tutor
    step({})
    llm.grades = [grade(82)]

    step(Command(resume="It wraps each business transaction in one procedure."))

    entry = store.get(MASTERY, "Transaction Script")
    assert entry is not None
    assert 0 <= entry.value["score"] <= 100
    assert entry.value["passed"] is True
    assert entry.value["book_pages"] == "1-9"


# Test 4: Routing
def test_failing_answer_remediates_same_pattern(tutor):
    llm, _, step, _, _, _ = tutor
    step({})
    llm.grades = [grade(40, "What happens when two scripts share logic?")]

    payload = step(Command(resume="Not sure."))

    assert payload["pattern"] == "Transaction Script"
    assert payload["attempt"] == 2
    assert payload["question"] == "What happens when two scripts share logic?"
    assert "40/100" in payload["last_feedback"]


def test_passing_answer_advances(tutor):
    llm, _, step, _, _, _ = tutor
    step({})
    llm.grades = [grade(90)]

    payload = step(Command(resume="A good answer."))

    assert payload["pattern"] == "Domain Model"
    assert payload["attempt"] == 1


def test_exhausted_attempts_move_on(tutor):
    llm, store, step, _, _, _ = tutor
    payload = step({})
    for attempt in range(MAX_ATTEMPTS):
        assert payload["pattern"] == "Transaction Script"
        llm.grades = [grade(30, f"Follow-up {attempt}?")]
        payload = step(Command(resume="Still wrong."))

    assert payload["pattern"] == "Domain Model"
    assert "come back" in payload["last_feedback"]
    assert store.get(MASTERY, "Transaction Script").value["attempts"] == MAX_ATTEMPTS


def test_question_generated_once_per_pattern(tutor):
    llm, _, step, _, _, _ = tutor
    step({})
    llm.grades = [grade(30, "Follow-up?")]
    step(Command(resume="Wrong."))
    llm.grades = [grade(90)]
    step(Command(resume="Right."))

    # One generated question per new pattern; remediation reuses the grader's question,
    # and resuming an interrupt never regenerates
    assert llm.question_calls == 2


def test_all_patterns_passed_ends_graph(tutor):
    llm, _, step, _, _, context = tutor
    context.target_layer = "Distribution"
    assert step({})["pattern"] == "Remote Facade"
    llm.grades = [grade(80)]
    assert step(Command(resume="a"))["pattern"] == "Data Transfer Object"
    llm.grades = [grade(80)]

    assert step(Command(resume="a")) is None


def test_only_failing_pattern_left_is_reoffered(tutor):
    llm, store, step, _, _, context = tutor
    context.target_layer = "Distribution"
    store.put(MASTERY, "Remote Facade", {"layer": "Distribution", "score": 90})
    assert step({})["pattern"] == "Data Transfer Object"
    llm.grades = [grade(10)]  # fail with no remediation question

    payload = step(Command(resume="?"))

    assert payload["pattern"] == "Data Transfer Object"
    assert payload["attempt"] == 1


# Test 5: Skip
def test_skip_advances_without_grading(tutor):
    _, store, step, _, _, _ = tutor
    step({})

    payload = step(Command(resume=SKIP_COMMAND))

    assert payload["pattern"] == "Domain Model"
    assert "Skipped" in payload["last_feedback"]
    assert store.search(MASTERY) == []


# Test 6: Selection (pure function)
def test_pick_next_pattern_priorities():
    store = InMemoryStore()
    scores = {
        "Table Data Gateway": 90,
        "Row Data Gateway": 40,
        "Active Record": 60,
        "Data Mapper": 80,
    }
    for name, score in scores.items():
        store.put(MASTERY, name, {"layer": "Data Source", "score": score})

    assert retrieval.pick_next_pattern(store, USER, None, set()) == (
        "Domain Logic",
        "Transaction Script",
    )
    assert retrieval.pick_next_pattern(store, USER, "Data Source", set()) == (
        "Data Source",
        "Row Data Gateway",
    )
    assert retrieval.pick_next_pattern(store, USER, "Data Source", {"Row Data Gateway"}) == (
        "Data Source",
        "Active Record",
    )
    assert (
        retrieval.pick_next_pattern(
            store, USER, "Data Source", {"Row Data Gateway", "Active Record"}
        )
        is None
    )


def test_pick_next_pattern_reads_beyond_default_search_limit():
    store = InMemoryStore()
    for layer, name, _, _ in retrieval.POEAA_CATALOG:
        if layer != "Narratives":
            store.put(MASTERY, name, {"layer": layer, "score": 100})

    assert retrieval.pick_next_pattern(store, USER, None, set()) is None


# CLI
def scripted(lines: list[str]):
    remaining = iter(lines)

    def read_line(prompt: str) -> str:
        try:
            return next(remaining)
        except StopIteration:
            raise EOFError from None

    return read_line


def test_cli_session_commands_and_resume(tutor, capsys):
    llm, store, _, graph, config, context = tutor
    llm.grades = [grade(85)]

    main.run_session(
        graph,
        store,
        config,
        context,
        read_line=scripted(
            [
                "/ledger",
                "/layer Nonsense",
                "/layer data source",
                "Procedures per transaction,",
                "simple but duplicates logic.",
                "",
                "/skip",
                "/quit",
            ]
        ),
    )
    output = capsys.readouterr().out

    assert "No patterns attempted yet." in output
    assert "Unknown layer 'Nonsense'" in output
    assert context.target_layer == "Data Source"
    assert "85/100" in output
    # The layer change leaves the current question alone and applies to the next new pattern
    assert "[Domain Logic | Transaction Script | Attempt 1/3]" in output
    assert "[Data Source | Table Data Gateway | Attempt 1/3]" in output
    assert "[Data Source | Row Data Gateway | Attempt 1/3]" in output
    assert "Domain Model" not in output
    assert "Progress saved" in output
    assert store.get(MASTERY, "Transaction Script").value["score"] == 85

    # Relaunch on the same thread resumes the pending question without regenerating it
    main.run_session(graph, store, config, context, read_line=scripted(["/quit"]))
    output = capsys.readouterr().out
    assert "Resuming your previous session." in output
    assert "[Data Source | Row Data Gateway | Attempt 1/3]" in output
    assert llm.question_calls == 3


def test_format_ledger_lists_entries_in_catalog_order():
    store = InMemoryStore()
    store.put(MASTERY, "Unit of Work", {"layer": "OR Behavioral", "score": 50, "attempts": 3})
    store.put(MASTERY, "Domain Model", {"layer": "Domain Logic", "score": 80, "attempts": 1})

    ledger = main.format_ledger(store, USER).splitlines()

    assert ledger[0].split(" | ")[0].strip() == "Pattern"
    assert "Domain Model" in ledger[2] and "Passed" in ledger[2]
    assert "Unit of Work" in ledger[3] and "Needs work" in ledger[3]


def test_turn_config_adds_thread_and_pattern_metadata(tutor):
    _, _, step, graph, config, context = tutor

    start = main.turn_config(graph, config, context, tags=["session-start"])
    assert start["metadata"]["thread_id"] == "test-thread"
    assert start["metadata"]["user_id"] == USER
    assert "pattern" not in start["metadata"]
    assert start["run_name"] == "tutor_turn"
    assert "session-start" in start["tags"]

    step({})
    answering = main.turn_config(graph, config, context)
    assert answering["metadata"]["pattern"] == "Transaction Script"
    assert answering["metadata"]["attempt"] == 1
    assert answering["metadata"]["is_remediation"] is False
    assert answering["configurable"] == config["configurable"]
