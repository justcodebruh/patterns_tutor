import argparse
import os
from importlib.metadata import version
from typing import Any, Callable, Optional, Sequence

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore
from langgraph.types import Command

from patterns_tutor.catalog import POEAA_CATALOG
from patterns_tutor.graph import build_tutor_graph
from patterns_tutor.retrieval import MASTERY_SEARCH_LIMIT, catalog_layers, load_vectorstore
from patterns_tutor.schemas import PASS_THRESHOLD, SKIP_COMMAND, InterruptPayload, TutorContext

CHECKPOINT_DB = "poeaa_tutor_checkpoints.sqlite"
MASTERY_DB = "poeaa_tutor_mastery.sqlite"
__version__ = version("patterns-tutor")
HELP_TEXT = "(Answer, then an empty line to submit. Commands: /ledger, /layer <name>, /skip, /quit)"


def format_ledger(store: BaseStore, user_id: str) -> str:
    """Renders the user's mastery entries as a table, in catalog order."""
    items = store.search(("mastery", user_id), limit=MASTERY_SEARCH_LIMIT)
    if not items:
        return "No patterns attempted yet."

    order = {name: index for index, (_, name, _, _) in enumerate(POEAA_CATALOG)}
    rows = [
        (
            item.key,
            item.value["layer"],
            str(item.value["score"]),
            str(item.value.get("attempts", "")),
            "Passed" if item.value["score"] >= PASS_THRESHOLD else "Needs work",
        )
        for item in sorted(items, key=lambda item: order.get(item.key, len(order)))
    ]
    header = ("Pattern", "Layer", "Score", "Attempts", "Status")
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    lines = [
        " | ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
        for row in [header, *rows]
    ]
    lines.insert(1, "-+-".join("-" * width for width in widths))
    return "\n".join(lines)


def render_challenge(payload: InterruptPayload) -> None:
    if payload["last_feedback"]:
        print(f"\n{payload['last_feedback']}")
    print(
        f"\n[{payload['layer']} | {payload['pattern']} | "
        f"Attempt {payload['attempt']}/{payload['max_attempts']}]"
    )
    print(payload["question"])
    print(HELP_TEXT)


def read_answer(read_line: Callable[[str], str]) -> Optional[str]:
    """Reads a multi-line answer ending on an empty line. Slash commands are single-line.

    Returns None on EOF or Ctrl-C, which the caller treats as /quit.
    """
    try:
        first = ""
        while not first.strip():
            first = read_line("> ")
        if first.strip().startswith("/"):
            return first.strip()
        lines = [first]
        while line := read_line("  "):
            lines.append(line)
        return "\n".join(lines).strip()
    except (EOFError, KeyboardInterrupt):
        return None


def set_target_layer(context: TutorContext, argument: str) -> None:
    if not argument:
        context.target_layer = None
        print("Layer filter cleared; questions will come from every layer.")
        return

    layers = {layer.lower(): layer for layer in catalog_layers()}
    layer = layers.get(argument.lower())
    if layer is None:
        print(f"Unknown layer '{argument}'. Choose one of: {', '.join(layers.values())}")
        return
    context.target_layer = layer
    print(f"Layer set to {layer}. It applies from the next new pattern, after the current question.")


def turn_config(
    graph: CompiledStateGraph, config: dict, context: TutorContext, tags: Sequence[str] = ()
) -> dict:
    """Adds LangSmith metadata and tags for one graph invocation (one trace).

    `thread_id` in metadata groups every turn of a session into one LangSmith thread. Pattern
    and attempt describe the question being answered in this turn, read from the checkpoint
    before the invocation, so grades can be filtered by pattern and attempt in the UI.
    """
    thread_id = config["configurable"]["thread_id"]
    metadata = {
        "thread_id": thread_id,
        "user_id": context.user_id,
        "target_layer": context.target_layer or "all",
        "app_version": __version__,
    }
    values = graph.get_state(config).values
    if values.get("active_question"):
        metadata.update(
            pattern=values["current_pattern"],
            layer=values["current_layer"],
            attempt=values.get("attempts", 0) + 1,
            is_remediation=values.get("attempts", 0) > 0,
        )
    return {
        **config,
        "run_name": "tutor_turn",
        "metadata": {**config.get("metadata", {}), **metadata},
        "tags": [*config.get("tags", []), "poeaa-tutor", *tags],
    }


def advance(
    graph: CompiledStateGraph,
    graph_input: Any,
    config: dict,
    context: TutorContext,
    tags: Sequence[str] = (),
) -> Optional[InterruptPayload]:
    """Runs the graph to its next interrupt, or returns None when the graph has ended."""
    print("\nThinking...")
    result = graph.invoke(
        graph_input,
        config=turn_config(graph, config, context, tags),
        context=context,
        version="v2",
    )
    if result.interrupts:
        return result.interrupts[0].value
    if result.value.get("last_feedback"):
        print(f"\n{result.value['last_feedback']}")
    return None


def run_session(
    graph: CompiledStateGraph,
    store: BaseStore,
    config: dict,
    context: TutorContext,
    read_line: Callable[[str], str] = input,
) -> None:
    """Drives the tutor until the user quits or every pattern in scope is passed."""
    pending = graph.get_state(config).interrupts
    if pending:
        print("Resuming your previous session.")
        payload = pending[0].value
    else:
        payload = advance(graph, {}, config, context, tags=["session-start"])

    while payload is not None:
        render_challenge(payload)
        while True:
            answer = read_answer(read_line)
            if answer is None or answer == "/quit":
                print("\nProgress saved. Run `uv run patterns-tutor` again to pick up where you left off.")
                return
            if answer == "/ledger":
                print(format_ledger(store, context.user_id))
            elif answer == "/layer" or answer.startswith("/layer "):
                set_target_layer(context, answer.removeprefix("/layer").strip())
            elif answer.startswith("/") and answer != SKIP_COMMAND:
                print(f"Unknown command '{answer}'. {HELP_TEXT}")
            else:
                break
        tags = ["skip"] if answer == SKIP_COMMAND else ["answer"]
        payload = advance(graph, Command(resume=answer), config, context, tags=tags)

    print(f"\n{format_ledger(store, context.user_id)}")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="PoEAA Fowler Architect tutor.")
    parser.add_argument("--user", default="default_user", help="User whose mastery is tracked")
    parser.add_argument("--thread", help="Conversation thread id (default: <user>-session)")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Add it to .env in the project root.")
        return 1
    try:
        load_vectorstore()
    except RuntimeError as error:
        print(error)
        return 1

    config = {"configurable": {"thread_id": args.thread or f"{args.user}-session"}, "tags": ["cli"]}
    context = TutorContext(user_id=args.user)
    with (
        SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer,
        SqliteStore.from_conn_string(MASTERY_DB) as store,
    ):
        store.setup()
        graph = build_tutor_graph(checkpointer, store)
        run_session(graph, store, config, context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
