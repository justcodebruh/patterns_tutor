"""Question generation as a pure, independently evaluable function."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import get_current_run_tree, traceable

from patterns_tutor.prompts import QUESTION_SYSTEM_PROMPT


@traceable(
    run_type="chain",
    name="generate_question",
    process_inputs=lambda inputs: {k: v for k, v in inputs.items() if k != "llm"},
)
def generate_question(*, pattern: str, layer: str, rubric: str, llm: BaseChatModel) -> str:
    """Writes one Socratic question about `pattern`, grounded in its rubric."""
    run = get_current_run_tree()
    if run is not None:
        run.add_metadata({"pattern": pattern, "layer": layer})
    response = llm.invoke(
        [
            SystemMessage(
                QUESTION_SYSTEM_PROMPT.format(pattern=pattern, layer=layer, rubric=rubric)
            ),
            HumanMessage(f"Ask me about {pattern}."),
        ]
    )
    return response.text.strip()
