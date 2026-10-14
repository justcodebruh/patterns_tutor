from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from patterns_tutor.nodes import evaluate_and_route, select_and_challenge
from patterns_tutor.schemas import TutorContext, TutorState


def build_tutor_graph(checkpointer: BaseCheckpointSaver, store: BaseStore) -> CompiledStateGraph:
    """Compiles the tutor graph. Routing between nodes comes from their `Command` return types."""
    builder = StateGraph(TutorState, context_schema=TutorContext)
    builder.add_node(select_and_challenge)
    builder.add_node(evaluate_and_route)
    builder.add_edge(START, "select_and_challenge")
    return builder.compile(checkpointer=checkpointer, store=store)
