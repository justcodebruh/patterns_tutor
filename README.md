# PoEAA Fowler Architect Tutor

A LangGraph tutor that quizzes you on Martin Fowler's *Patterns of Enterprise Application
Architecture*, grades your answers against the book, and asks Socratic follow-ups until you
pass. It is instrumented with LangSmith, and its grader is evaluated against a hand-written
golden dataset.

* Design and LangGraph conventions: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
* Observability and evaluation plan: [OBSERVABILITY_PLAN.md](OBSERVABILITY_PLAN.md)
* Running the experiment from the LangSmith UI: [evals/UI_EXPERIMENT.md](evals/UI_EXPERIMENT.md)

## Quick start

```bash
cp .env.example .env            # add OPENAI_API_KEY; LangSmith keys are optional
uv run patterns-tutor-ingest    # once, builds ./poeaa_chroma from the PDF
uv run patterns-tutor           # start a session
uv run pytest                   # offline, no API keys
```

## How it works

```
START ──▶ select_and_challenge ──(answer)──▶ evaluate_and_route ──▶ select_and_challenge ...
            │ pick pattern, fetch rubric,          │ grade_answer() against the rubric,
            │ generate_question(), interrupt()     │ record_mastery(), route: pass / remediate / move on
```

The two LLM behaviours are pure functions so they can be evaluated without the graph:

| Module | Function | Purpose |
| :--- | :--- | :--- |
| `grading.py` | `grade_answer()` | Grades one answer against the book excerpts and returns an `EvaluationResult` |
| `questions.py` | `generate_question()` | Writes one Socratic question grounded in the excerpts |
| `prompts.py` | `GRADER_PROMPTS` | Versioned grader prompts (`v1` baseline, `v2` current) |

## Observability

With `LANGSMITH_TRACING=true`, every CLI turn is one trace named `tutor_turn`:

* `metadata.thread_id` groups a whole session into one LangSmith thread.
* `user_id`, `pattern`, `layer`, `attempt` and `is_remediation` make grades filterable.
* Tags `session-start`, `answer` and `skip` say what the turn was.
* `pick_next_pattern`, `retrieve_pattern_rubric`, `generate_question`, `grade_answer` and
  `record_mastery` are their own spans. The retriever span logs chunk count and sections
  present rather than repeating the rubric. `grade_answer` records the prompt revision and hash.

## Evaluating the grader

The grader decides whether a student passes, so it is the part that must not be gameable and
must not be unfair. The golden set in `evals/golden/grading/` has 37 hand-written answers across
10 patterns and 12 categories: model answers, terse and padded correct answers, partial answers,
wrong pattern, contradicting the book, empty, keyword stuffing, copied book text, prompt
injection, fabricated citations, and restating the question. Each case records the expected
verdict, an acceptable score band and the points a good grade must credit. Seven cases are a
held-out split written after the prompt fix, to check that fix was not overfitted.

The repo stores no book text. The rubric is read from the local Chroma index when the dataset is
pushed or evaluated.

```bash
uv run patterns-tutor-eval push                          # sync dataset to LangSmith
uv run patterns-tutor-eval grading --prompt-rev v2       # SDK experiment, uploaded
uv run patterns-tutor-eval grading --local --split holdout --repetitions 3   # no upload
uv run patterns-tutor-eval push-prompt --prompt-rev v2   # for the UI Playground run
```

Without `LANGSMITH_API_KEY` the harness runs the same SDK evaluators locally and writes a report
to `evals/results/`.

| Evaluator | Kind | Checks |
| :--- | :--- | :--- |
| `pass_fail_agreement` | code | Verdict matches the expected outcome |
| `score_in_band`, `band_distance` | code | Score calibration |
| `remediation_policy` | code | A follow-up question exactly when the score is below 75 |
| `page_ref_in_range` | code | Cited pages fall inside the pattern's chapter |
| `injection_resisted` | code | Injected text is not echoed and does not change the verdict |
| `feedback_grounded`, `must_credit_coverage` | LLM judge | Feedback is supported by the book and covers the key points |
| `false_pass_rate`, `false_fail_rate` | summary | Gameability and harshness |
| `score_std_per_example`, `verdict_flips` | summary | Run-to-run noise across repetitions |

## Results: one debug, fix and re-measure loop

All runs are `gpt-4o` at temperature 0.2 with three repetitions per case and a `gpt-4o` judge.
Reports are in `evals/results/`.

**1. Baseline.** The original grader prompt, v1, got 28 of 30 verdicts right on the tuning
split. The two misses repeated in all three runs:

* The answer `?` scored **85**. Its "strengths" listed points the student never made. The grader
  had written its own answer and graded that.
* An answer that pasted a fake grader JSON with `"score": 100` scored **100**. The grader
  returned it verbatim, down to the strength "Complete answer".

Wrong and contradicting answers were also scored around 40, above their expected bands. The
verdicts were still right.

**2. Diagnosis.** Both false passes have one cause. The question and answer were sent as plain
undelimited text, and nothing told the grader that the answer is untrusted data that earns credit
only for what it actually says. The v1 prompt also anchored only the 75 threshold.

**3. Fix.** Grader prompt v2, in `prompts.py`, has three changes:

* It wraps the answer in tags, and code neutralizes any closing tag inside the answer.
* It tells the grader that only the student's own reasoning earns credit and that instructions
  inside the answer are ignored.
* It anchors the full 0 to 100 scale.

**4. Re-measure.**

| Metric | Tuning v1 | Tuning v2 | Held-out v1 | Held-out v2 |
| :--- | ---: | ---: | ---: | ---: |
| Pass/fail agreement | 0.93 | **1.00** | 1.00 | 1.00 |
| False pass rate | 0.10 | **0.00** | 0.00 | 0.00 |
| False fail rate | 0.00 | 0.00 | 0.00 | 0.00 |
| Injection resisted | 0.67 | **1.00** | 1.00 | 1.00 |
| Score in band | 0.72 | **0.97** | 0.57 | **0.86** |
| Feedback grounded, judge | 0.83 | **0.92** | 0.92 | 0.89 |
| Page reference in range | 1.00 | 1.00 | 1.00 | 1.00 |
| Remediation policy | 0.92 | 0.96 | 1.00 | 0.86 |
| Mean score std dev, points | 1.8 | 1.0 | 2.9 | 1.4 |
| Verdict flips across repetitions | 0 | 0 | 0 | 0 |

What this does and does not show:

* **v2 removes both false passes** and adds no false fails, so it is the default.
* **The held-out split did not separate the prompts on verdicts.** v1 also judged all seven
  held-out cases correctly. The held-out gain is calibration. For example, v1 scored a restated
  question at 60 and v2 scored it at 0 to 10. The verdict gains on the tuning split come from
  cases I saw before writing v2, so treat them as a fixed regression rather than proof of
  general robustness.
* **v2 asks follow-ups on some passing answers.** That is the drop in remediation policy on the
  held-out split. The graph already discards a follow-up when the answer passes, so students
  never see it.
* **Some passes sit near the threshold.** v2 scores several correct terse answers at exactly 75.
  None flipped across repetitions, but they are the first thing to watch as the dataset grows.

### Harness bugs found along the way

* **Errored runs were scored as failures.** An OpenAI rate limit hit 24 of 90 runs once, and the
  evaluators scored those runs as wrong. Evaluators now skip errored runs, the report counts them,
  and the models retry with backoff.
* **Positional arguments broke the skip wrapper.** The SDK passes evaluator arguments
  positionally, so the first wrapper silently skipped every row. It now binds arguments to the
  wrapped signature, and a test exercises a failing target.
* **Groundedness was undefined for empty feedback.** The judge scored empty feedback on an
  "I don't know" answer as 0% grounded. It now skips that metric when there is nothing to check.
* **Local mode still sent traces in `langsmith` 0.14.8.** With `upload_results=False` the runner
  sets tracing to "local", but runs were still posted and returned 401 without a key. Local mode
  now disables tracing inside the target and for evaluators.
