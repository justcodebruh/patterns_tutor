# Running the grading experiment from the LangSmith UI

This is the UI half of the take-home's requirement 3. The SDK half is
`uv run patterns-tutor-eval grading`. Both run against the same dataset,
`poeaa-grading-golden`, so the experiments sit side by side in the dataset's Experiments tab.

## 0. One-time setup (SDK, about 1 minute)

Add to `.env`:

```
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=poeaa-tutor
```

Then, from the project root:

```bash
uv run patterns-tutor-eval push                         # creates/updates the dataset (37 examples)
uv run patterns-tutor-eval push-prompt --prompt-rev v1  # commit tagged v1
uv run patterns-tutor-eval push-prompt --prompt-rev v2  # commit tagged v2
uv run patterns-tutor-eval grading --prompt-rev v1      # SDK baseline experiment
uv run patterns-tutor-eval grading --prompt-rev v2      # SDK fixed experiment
```

`push-prompt` uploads the grader as a structured prompt. Its variables are exactly the dataset
input keys (`pattern`, `layer`, `book_pages`, `rubric`, `question`, `answer`), which a unit test
enforces, so the Playground can bind every example without any mapping.

## 1. Add the evaluators to the dataset (UI)

**Datasets & Experiments → poeaa-grading-golden → Evaluators → + Evaluator.** Evaluators
attached here run automatically on every new experiment for this dataset, whether it started
from the SDK or the Playground.

### Code evaluator: `pass_fail_agreement`

Choose **Custom code**, name it `pass_fail_agreement`, and paste:

```python
def perform_eval(run, example):
    out = run["outputs"]
    out = out.get("output", out)  # Playground runs may nest the structured result
    score = out.get("score")
    if score is None:
        return {"pass_fail_agreement": None}
    expected_pass = example["outputs"]["expected_outcome"] == "pass"
    return {"pass_fail_agreement": (score >= 75) == expected_pass}
```

Playground runs return the raw structured grade, without the `passed` flag that the app derives
in code, so this evaluator recomputes it from `score` with the same threshold of 75.

### Code evaluator: `score_in_band`

```python
def perform_eval(run, example):
    out = run["outputs"]
    out = out.get("output", out)
    score = out.get("score")
    low, high = example["outputs"]["score_band"]
    return {"score_in_band": score is not None and low <= score <= high}
```

### Code evaluator: `page_ref_in_range`

```python
import re

def perform_eval(run, example):
    out = run["outputs"]
    out = out.get("output", out)
    start, end = example["outputs"]["catalog_pages"]
    pages = [int(n) for n in re.findall(r"\d+", out.get("book_page_reference") or "")]
    return {"page_ref_in_range": bool(pages) and all(start - 2 <= p <= end + 2 for p in pages)}
```

### LLM-as-judge: `feedback_grounded`

Choose **LLM-as-judge**, model `gpt-4o`, score type continuous 0 to 1, and use this prompt.
Map `{{rubric}}`, `{{answer}}` to the example inputs and `{{output}}` to the run output.

```
You audit feedback written by an automated grader for a student answer about Martin Fowler's
Patterns of Enterprise Application Architecture.

Book excerpts the grader used:
{{rubric}}

Student answer:
{{answer}}

Grader output:
{{output}}

Score the fraction (0-1) of the grader's strengths and missed trade-offs that are supported by
the book excerpts. A claim the excerpts do not support, or that misstates Fowler, is ungrounded.
If the grader made no claims, return 1.
```

## 2. Run the experiment from the Playground (UI)

1. **Prompts → poeaa-grader**, pick the commit tagged `v1`, and click **Playground**.
2. Confirm the model settings show `gpt-4o`, temperature `0.2`, and the output schema.
3. Click **Switch to dataset**, pick `poeaa-grading-golden`, and set repetitions to 3.
4. Click **Start**. The Playground creates an experiment with the evaluators from step 1.
5. Repeat with the commit tagged `v2`.

## 3. Compare and review (UI)

1. **poeaa-grading-golden → Experiments.** Select the four experiments: SDK v1, SDK v2,
   Playground v1 and Playground v2. Click **Compare**.
2. Sort by `pass_fail_agreement`. Two rows should be red for v1 and green for v2:
   `sti-empty-idk` and `sti-prompt-injection`.
3. Filter by metadata `holdout = true` to see the held-out cases written after v2.
4. Select any rows where SDK and Playground disagree and choose **Add to annotation queue**,
   creating a queue named `poeaa-grade-review`. Score two or three by hand. A human score that
   differs from the dataset's band is a signal to correct the golden example, not the grader.

## Notes on SDK and UI differences to expect

* **Answer neutralization.** The SDK target goes through `grade_answer`, which neutralizes
  `</student_answer>` tags inside answers for v2. The Playground inserts the answer as-is, so the
  tag-closing attack (`ool-prompt-injection`) runs unprotected there. A difference on
  that row is expected and shows why the defense lives in code, not only in the prompt.
* **The pass flag.** The pass/fail flag is computed in code, which is why the UI code evaluators
  recompute it from `score`.
