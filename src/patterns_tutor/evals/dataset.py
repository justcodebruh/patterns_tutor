"""Golden grading dataset: load from the repo, enrich with the book rubric, sync to LangSmith.

The JSON files under `evals/golden/grading/` are the source of truth and are reviewed in PRs.
They hold no book text. The rubric is fetched from the local Chroma index at push or eval time,
so the LangSmith dataset is self-contained (and usable from the UI Playground) while the repo
stays free of copyrighted content.
"""

import datetime
import json
import pathlib
import uuid
from typing import Optional

from langsmith import Client
from langsmith.schemas import Example

from patterns_tutor.catalog import POEAA_CATALOG

GRADING_DATASET = "poeaa-grading-golden"
GRADING_DIR = pathlib.Path("evals/golden/grading")
ADVERSARIAL = {
    "prompt_injection",
    "keyword_stuffing",
    "rubric_parrot",
    "fabricated_citation",
    "contradicts_book",
    "wrong_pattern",
    "question_echo",
}
# Stable example ids: re-pushing updates examples in place instead of duplicating them
_ID_NAMESPACE = uuid.UUID("6f1d3a52-6a8b-4f1e-9a51-2b0e8a7c4d11")

CATALOG = {name: (layer, start, end) for layer, name, start, end in POEAA_CATALOG}


def load_cases(directory: pathlib.Path = GRADING_DIR) -> list[dict]:
    """Flattens the per-pattern JSON files into one case dict per example."""
    if not directory.is_dir():
        raise RuntimeError(f"No golden dataset at '{directory}'. Run from the project root.")
    cases = []
    for path in sorted(directory.glob("*.json")):
        spec = json.loads(path.read_text())
        for case in spec["cases"]:
            cases.append({"pattern": spec["pattern"], "question": spec["question"], **case})
    return cases


def case_to_example(case: dict, rubric_lookup) -> dict:
    """Splits a case into LangSmith inputs, reference outputs, metadata and split."""
    layer, start, end = CATALOG[case["pattern"]]
    rubric, book_pages = rubric_lookup(case["pattern"])
    split = ["adversarial"] if case["category"] in ADVERSARIAL else ["standard"]
    split.append("holdout" if case.get("holdout") else "tuning")
    return {
        "id": uuid.uuid5(_ID_NAMESPACE, case["id"]),
        "inputs": {
            "pattern": case["pattern"],
            "layer": layer,
            "book_pages": book_pages,
            "rubric": rubric,
            "question": case["question"],
            "answer": case["answer"],
        },
        "outputs": {
            "expected_outcome": case["expected_outcome"],
            "score_band": case["score_band"],
            "must_credit": case["must_credit"],
            "remediation_expected": case["remediation_expected"],
            "forbidden_feedback_phrases": case["forbidden_feedback_phrases"],
            "catalog_pages": [start, end],
            "category": case["category"],
        },
        "metadata": {
            "case_id": case["id"],
            "category": case["category"],
            "pattern": case["pattern"],
            "authored_by": case["authored_by"],
            "notes": case["notes"],
            "holdout": bool(case.get("holdout")),
        },
        "split": split,
    }


def build_examples(cases: list[dict], rubric_lookup) -> list[dict]:
    rubrics: dict = {}

    def cached_lookup(pattern: str):
        if pattern not in rubrics:
            rubrics[pattern] = rubric_lookup(pattern)
        return rubrics[pattern]

    return [case_to_example(case, cached_lookup) for case in cases]


def local_examples(examples: list[dict]) -> list[Example]:
    """Example objects for running `evaluate(..., upload_results=False)` without LangSmith."""
    dataset_id = uuid.uuid5(_ID_NAMESPACE, GRADING_DATASET)
    now = datetime.datetime.now(datetime.timezone.utc)
    return [
        Example(
            id=example["id"],
            dataset_id=dataset_id,
            inputs=example["inputs"],
            outputs=example["outputs"],
            metadata=example["metadata"],
            created_at=now,
        )
        for example in examples
    ]


def push_dataset(
    examples: list[dict], client: Optional[Client] = None, name: str = GRADING_DATASET
) -> dict:
    """Creates or updates the LangSmith dataset so it matches the repo files exactly."""
    client = client or Client()
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
    else:
        dataset = client.create_dataset(
            name,
            description=(
                "Golden set for the PoEAA tutor's answer grader: hand-written student answers "
                "with expected pass/fail, score band and the points a grade must credit. "
                "Source of truth: evals/golden/grading/*.json in the patterns_tutor repo."
            ),
        )
    existing = {example.id for example in client.list_examples(dataset_id=dataset.id)}
    to_create = [e for e in examples if e["id"] not in existing]
    to_update = [e for e in examples if e["id"] in existing]
    stale = existing - {e["id"] for e in examples}
    if to_create:
        client.create_examples(dataset_id=dataset.id, examples=to_create)
    if to_update:
        client.update_examples(dataset_id=dataset.id, updates=to_update)
    if stale:
        client.delete_examples(list(stale))
    return {
        "dataset": name,
        "url": getattr(dataset, "url", None),
        "created": len(to_create),
        "updated": len(to_update),
        "deleted": len(stale),
    }
