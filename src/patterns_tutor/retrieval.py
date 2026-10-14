import functools
import os
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

from patterns_tutor.catalog import POEAA_CATALOG
from patterns_tutor.schemas import PASS_THRESHOLD, TutorContext

CHROMA_DIR = "./poeaa_chroma"
COLLECTION_NAME = "poeaa_patterns"
RUBRIC_SECTIONS = ["overview", "how_it_works", "when_to_use"]
# Mastery entries per user never exceed the catalog size; the store's default limit is 10
MASTERY_SEARCH_LIMIT = 1000


@functools.cache
def load_vectorstore() -> Chroma:
    """Opens the existing PoEAA collection read-only. Never re-indexes.

    No embedding function is attached: rubric lookup is a pure metadata fetch,
    so it needs no OpenAI key and makes no API calls.
    """
    # Check before constructing Chroma, which would otherwise create an empty directory
    if not os.path.isdir(CHROMA_DIR):
        raise RuntimeError(
            f"No vector store at '{CHROMA_DIR}'. Run `uv run patterns-tutor-ingest` first."
        )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=None,
    )
    if not vectorstore.get(limit=1, include=[])["ids"]:
        raise RuntimeError(
            f"Vector store at '{CHROMA_DIR}' is empty. Run `uv run patterns-tutor-ingest` first."
        )
    return vectorstore


def retrieve_pattern_rubric(pattern_name: str) -> tuple[str, str]:
    """Returns (rubric_text, book_pages) built from every non-code chunk of a pattern."""
    results = load_vectorstore().get(
        where={
            "$and": [
                {"pattern_name": {"$eq": pattern_name}},
                {"section": {"$in": RUBRIC_SECTIONS}},
            ]
        },
        include=["documents", "metadatas"],
    )
    chunks = list(zip(results["documents"], results["metadatas"]))
    if not chunks:
        raise ValueError(f"No rubric chunks found for pattern '{pattern_name}'.")

    # Stable sort keeps storage order within each section
    chunks.sort(key=lambda chunk: RUBRIC_SECTIONS.index(chunk[1]["section"]))
    book_pages = chunks[0][1]["book_pages"]
    rubric = "\n\n".join(
        f"[Pages {meta['book_pages']} | Related: {meta['related_patterns'] or 'none'}]\n{text}"
        for text, meta in chunks
    )
    return rubric, book_pages


def pick_next_pattern(
    store: BaseStore,
    user_id: str,
    target_layer: Optional[str],
    exclude: set[str],
) -> Optional[tuple[str, str]]:
    """Picks the next (layer, pattern_name) to quiz, or None if everything in scope is passed.

    Priority: first untested pattern in catalog order, then the lowest failing score.
    """
    mastery = {
        item.key: item.value
        for item in store.search(("mastery", user_id), limit=MASTERY_SEARCH_LIMIT)
    }
    candidates = [
        (layer, name)
        for layer, name, _, _ in POEAA_CATALOG
        if layer != "Narratives"
        and name not in exclude
        and (target_layer is None or layer == target_layer)
    ]

    for layer, name in candidates:
        if name not in mastery:
            return layer, name

    failing = [
        (mastery[name]["score"], index, layer, name)
        for index, (layer, name) in enumerate(candidates)
        if mastery[name]["score"] < PASS_THRESHOLD
    ]
    if failing:
        _, _, layer, name = min(failing)
        return layer, name
    return None


@functools.cache
def _default_llm() -> BaseChatModel:
    return init_chat_model("openai:gpt-4o", temperature=0.2)


def get_llm(runtime: Runtime[TutorContext]) -> BaseChatModel:
    """Returns the injected model from the runtime context, or the cached default."""
    if runtime.context.llm is not None:
        return runtime.context.llm
    return _default_llm()


def catalog_layers() -> list[str]:
    """Distinct pattern layers in catalog order, excluding the narrative chapters."""
    return list(dict.fromkeys(layer for layer, *_ in POEAA_CATALOG if layer != "Narratives"))
