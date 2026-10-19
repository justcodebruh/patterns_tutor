"""`patterns-tutor-eval`: sync datasets, run grading experiments, push the grader prompt.

Examples (from the project root):

    uv run patterns-tutor-eval push                      # sync golden set to LangSmith
    uv run patterns-tutor-eval grading                   # SDK experiment, uploaded to LangSmith
    uv run patterns-tutor-eval grading --local           # same evaluators, nothing uploaded
    uv run patterns-tutor-eval grading --prompt-rev v2 --repetitions 3
    uv run patterns-tutor-eval push-prompt --prompt-rev v2   # for the UI Playground experiment
"""

import argparse
import json
import os
import pathlib
import statistics
import sys
from collections import defaultdict
from datetime import datetime

from dotenv import load_dotenv

RESULTS_DIR = pathlib.Path("evals/results")


def _grading_target(model: str, prompt_rev: str, trace: bool = True):
    from langchain.chat_models import init_chat_model
    from langsmith import tracing_context

    from patterns_tutor.grading import grade_answer
    from patterns_tutor.schemas import PASS_THRESHOLD

    from patterns_tutor.evals.evaluators import MAX_RETRIES

    llm = init_chat_model(model, temperature=0.2, max_retries=MAX_RETRIES)

    def grade(inputs: dict) -> dict:
        # In local mode the SDK sets tracing to "local", which in langsmith 0.14.8 still posts
        # child runs (and 401s without a key). Turning tracing off inside the target stops that.
        with tracing_context(enabled=None if trace else False):
            result = grade_answer(**inputs, llm=llm, prompt_rev=prompt_rev)
        return {**result.model_dump(), "passed": result.score >= PASS_THRESHOLD}

    return grade


def _examples(category: str | None, split: str | None = None):
    from patterns_tutor.evals.dataset import build_examples, load_cases
    from patterns_tutor.retrieval import retrieve_pattern_rubric

    cases = load_cases()
    if category:
        cases = [case for case in cases if case["category"] == category]
    if split:
        examples = build_examples(cases, retrieve_pattern_rubric)
        return [example for example in examples if split in example["split"]]
    return build_examples(cases, retrieve_pattern_rubric)


def cmd_push(args) -> int:
    from patterns_tutor.evals.dataset import push_dataset

    print(json.dumps(push_dataset(_examples(None)), indent=2, default=str))
    return 0


def cmd_grading(args) -> int:
    from langsmith import Client, evaluate

    from patterns_tutor.evals.dataset import GRADING_DATASET, local_examples
    from patterns_tutor.evals.evaluators import (
        CODE_EVALUATORS,
        SUMMARY_EVALUATORS,
        make_feedback_judge,
    )
    from patterns_tutor.prompts import prompt_sha

    local = args.local or not os.environ.get("LANGSMITH_API_KEY")
    if local and not args.local:
        print("LANGSMITH_API_KEY is not set; running locally without uploading results.")

    evaluators = list(CODE_EVALUATORS)
    if not args.no_judge:
        evaluators.append(make_feedback_judge(args.judge_model))

    prefix = args.experiment_prefix or f"grader-{args.prompt_rev}"
    metadata = {
        "model": args.model,
        "grader_prompt_rev": args.prompt_rev,
        "grader_prompt_sha": prompt_sha(args.prompt_rev),
        "judge_model": None if args.no_judge else args.judge_model,
        "repetitions": args.repetitions,
    }
    common = dict(
        evaluators=evaluators,
        summary_evaluators=SUMMARY_EVALUATORS,
        experiment_prefix=prefix,
        metadata=metadata,
        num_repetitions=args.repetitions,
        max_concurrency=args.concurrency,
    )
    target = _grading_target(args.model, args.prompt_rev, trace=not local)
    if local:
        results = evaluate(
            target,
            data=local_examples(_examples(args.category, args.split)),
            upload_results=False,
            disable_evaluator_tracing=True,
            **common,
        )
    else:
        client = Client()
        if args.category or args.split:
            data = client.list_examples(
                dataset_name=GRADING_DATASET,
                metadata={"category": args.category} if args.category else None,
                splits=[args.split] if args.split else None,
            )
        else:
            data = GRADING_DATASET
        results = client.evaluate(target, data=data, **common)

    report = _report(results, prefix, metadata)
    _print_report(report)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"{prefix}-{stamp}.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"\nSaved {path}")
    if not local:
        print(f"Experiment: {results.experiment_name}")
    return 0


def _report(results, prefix: str, metadata: dict) -> dict:
    rows = []
    metric_values: dict[str, list[float]] = defaultdict(list)
    for row in results:
        example, run = row["example"], row["run"]
        feedback = {}
        for result in row["evaluation_results"]["results"]:
            if result.score is None:
                continue
            feedback[result.key] = float(result.score)
            metric_values[result.key].append(float(result.score))
        outputs = run.outputs or {}
        rows.append(
            {
                "case_id": example.metadata["case_id"],
                "category": example.metadata["category"],
                "expected": example.outputs["expected_outcome"],
                "band": example.outputs["score_band"],
                "score": outputs.get("score"),
                "passed": outputs.get("passed"),
                "page_ref": outputs.get("book_page_reference"),
                "remediation": outputs.get("remediation_question"),
                "strengths": outputs.get("strengths"),
                "missed": outputs.get("missed_fowler_tradeoffs"),
                "error": run.error,
                "feedback": feedback,
            }
        )
    summary = {
        result.key: result.score for result in results._summary_results.get("results", [])
    }
    means = {key: statistics.mean(values) for key, values in sorted(metric_values.items())}
    return {
        "experiment_prefix": prefix,
        "errored_runs": sum(1 for row in rows if row["error"]),
        "metadata": metadata,
        "row_metric_means": means,
        "summary": summary,
        "rows": rows,
    }


def _print_report(report: dict) -> None:
    print("\nRow metrics (mean):")
    for key, value in report["row_metric_means"].items():
        print(f"  {key:<24} {value:.3f}")
    print("Summary metrics:")
    for key, value in report["summary"].items():
        print(f"  {key:<34} {value:.3f}")
    wrong = [r for r in report["rows"] if r["feedback"].get("pass_fail_agreement") == 0.0]
    if wrong:
        print("\nPass/fail disagreements:")
        for row in wrong:
            print(f"  {row['case_id']:<26} expected {row['expected']:<4} got {row['score']}")
    errors = [r for r in report["rows"] if r["error"]]
    if errors:
        print(f"\n{len(errors)} runs errored and were not scored, e.g. {errors[0]['error'][:200]}")


DATASET_INPUT_KEYS = {"pattern", "layer", "book_pages", "rubric", "question", "answer"}


def build_hub_prompt(prompt_rev: str):
    """The grader as a StructuredPrompt whose variables are exactly the dataset input keys.

    The UI Playground binds dataset inputs to prompt variables by name, and the attached
    schema makes the Playground return the same EvaluationResult the app uses. The
    threshold is fixed in code, so it is baked in. Student answers are neutralized in code
    for delimited revisions; in the Playground they are inserted as-is.
    """
    from langchain_core.prompts.structured import StructuredPrompt

    from patterns_tutor.prompts import GRADER_HUMAN_TEMPLATES, GRADER_PROMPTS
    from patterns_tutor.schemas import PASS_THRESHOLD, EvaluationResult

    system = GRADER_PROMPTS[prompt_rev].replace("{threshold}", str(PASS_THRESHOLD))
    # A JSON Schema dict, not the Pydantic class: the class serializes to null in the hub
    # manifest, and the pulled prompt then fails to load
    return StructuredPrompt.from_messages_and_schema(
        [("system", system), ("human", GRADER_HUMAN_TEMPLATES[prompt_rev])],
        EvaluationResult.model_json_schema(),
    )


def cmd_push_prompt(args) -> int:
    from langchain_openai import ChatOpenAI
    from langsmith import Client

    from patterns_tutor.prompts import prompt_sha

    prompt = build_hub_prompt(args.prompt_rev)
    if set(prompt.input_variables) != DATASET_INPUT_KEYS:
        print(f"Prompt variables {sorted(prompt.input_variables)} do not match the dataset inputs.")
        return 1
    client = Client()
    model_name = args.model.removeprefix("openai:")
    # Tags are set separately: push_prompt's commit_tags fails with 409 once a tag exists,
    # after the commit has already been created
    url = client.push_prompt(
        args.name,
        object=prompt | ChatOpenAI(model=model_name, temperature=0.2),
        description="PoEAA tutor answer grader. Inputs match the poeaa-grading-golden dataset.",
    )
    commit = tag_latest_commit(client, args.name, args.prompt_rev)
    print(
        f"Pushed {args.name}:{args.prompt_rev} (prompt sha {prompt_sha(args.prompt_rev)}, "
        f"commit {commit[:8]}): {url}"
    )
    return 0


def tag_latest_commit(client, name: str, tag: str) -> str:
    """Points `tag` at the prompt's newest commit, creating or moving it. Returns the hash."""
    newest = client.request_with_retries(
        "GET", f"/commits/-/{name}/", params={"limit": 1}
    ).json()["commits"][0]
    existing = client.request_with_retries("GET", f"/repos/-/{name}/tags").json()
    if any(item["tag_name"] == tag for item in existing):
        response = client.session.request(
            "PATCH",
            f"{client.api_url}/repos/-/{name}/tags/{tag}",
            json={"commit_id": newest["id"]},
            headers=client._headers,
        )
    else:
        response = client.session.request(
            "POST",
            f"{client.api_url}/repos/-/{name}/tags",
            json={"tag_name": tag, "commit_id": newest["id"]},
            headers=client._headers,
        )
    response.raise_for_status()
    return newest["commit_hash"]


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="patterns-tutor-eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("push", help="Create or update the golden grading dataset in LangSmith")

    grading = sub.add_parser("grading", help="Run a grading experiment on the golden set")
    grading.add_argument("--model", default="openai:gpt-4o")
    grading.add_argument("--prompt-rev", default="v1")
    grading.add_argument("--repetitions", type=int, default=1)
    grading.add_argument("--concurrency", type=int, default=2)
    grading.add_argument("--category", help="Only run one category, e.g. prompt_injection")
    grading.add_argument(
        "--split", choices=["tuning", "holdout", "adversarial", "standard"],
        help="Only run one dataset split",
    )
    grading.add_argument("--experiment-prefix")
    grading.add_argument("--judge-model", default="openai:gpt-4o")
    grading.add_argument("--no-judge", action="store_true", help="Skip the LLM judge")
    grading.add_argument("--local", action="store_true", help="Do not upload to LangSmith")

    push_prompt = sub.add_parser("push-prompt", help="Push the grader prompt to the Prompt Hub")
    push_prompt.add_argument("--prompt-rev", default="v1")
    push_prompt.add_argument("--name", default="poeaa-grader")
    push_prompt.add_argument("--model", default="openai:gpt-4o")

    args = parser.parse_args(argv)
    if args.command == "grading":
        from patterns_tutor.prompts import GRADER_PROMPTS

        if args.prompt_rev not in GRADER_PROMPTS:
            parser.error(f"--prompt-rev must be one of {sorted(GRADER_PROMPTS)}")
    if not os.environ.get("OPENAI_API_KEY") and args.command == "grading":
        print("OPENAI_API_KEY is not set. Add it to .env in the project root.")
        return 1
    handler = {"push": cmd_push, "grading": cmd_grading, "push-prompt": cmd_push_prompt}
    return handler[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
