# Final evaluation plan: official Banking77 test split, run once

Written before the test split has been scored by any arm. Everything below is fixed at the
commit that adds this file. The analysis implementation is frozen at `73a6808`. The commit that
adds this plan contains only this file and `scripts/audit_runs.py`: it changes no analysis
semantics (no change to routing, calibration, cost, statistics, hypotheses or the report).
Provenance chain: `73a6808` is the frozen implementation, the plan commit is the frozen
protocol, and the held-out run follows both.

## What is frozen

- **Data.** Banking77 at the pinned revision in `src/classicalnlp/tickets.py`. Train
  `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` (10,003 rows), test
  `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d` (3,080 rows). The
  loader refuses a file whose checksum or row count differs. Validation is the seeded stratified
  20% of train; the router is fitted on the other 80%.
- **Arms.** A: TF-IDF (word) + logistic regression. B: local model
  `qwen2.5:7b-instruct-q3_K_M` (Ollama digest `29492a928341`, Ollama version `0.34.4`, both as
  observed before this plan was committed), temperature 0, seed 20260830, one response per
  ticket feeding both variants, through the existing persistent cache. The model is not pulled,
  updated or replaced before the evaluation. B-logprob is the primary B arm and B-verbal
  is an ablation. C is not evaluated and is reported as such.
- **Calibration.** Top-label isotonic regression fitted on all validation rows, applied to the
  test predictions. Raw confidences are reported alongside.
- **Cost.** Routing decision cost per 1,000 tickets, excluding compute. Misroute : handoff
  ratios 2, 5, 10 and 20; a handoff costs 1.0. No currency.
- **Headline.** (A, isotonic) against (B-logprob, isotonic) at 10:1. Paired bootstrap of
  per-ticket cost, 2,000 resamples, seed 20260830, 95% percentile interval, one shared
  resample matrix for both arms.
- **Hypotheses.** H1: A has lower cost than B at the headline ratio, read from the headline
  interval. H2: A has lower ECE than B, read from the A-isotonic against B-logprob-isotonic
  interval. Status text is mechanical: "A lower", "B lower" or "no clear difference".
- **Other statistics.** Exact McNemar with abstentions counted as wrong; top-label ECE over
  all rows; break-even compute sensitivity in handoff units.

From this point nothing above changes because of test results: no change to thresholds,
calibration, prompt, model or settings, confidence or cost definitions, statistical method,
headline comparison or hypothesis definitions.

## Not allowed before the registered run

No test smoke run, partial test run (any `--limit`, any subset), exploratory test command, or
manual inspection of the test dataset's rows or labels. Existing validation files stay
validation files. The registered evaluation itself necessarily reads the held-out labels; the
prohibition is on looking at or using them, or the results, before it, and on altering the
method afterwards.

## Before running (record each in the evaluation log below)

1. `git status` is clean, `HEAD` is the commit that added this file, and it equals `origin/main`
   with CI green on that SHA.
2. The offline test suite passes.
3. Both data files match the checksums above.
4. `ollama list` shows `qwen2.5:7b-instruct-q3_K_M` with digest `29492a928341`; record the
   Ollama version.
5. Record the size and SHA-256 of `data/interim/llm_cache.jsonl`. It currently holds the 2,001
   validation responses: 1,186,610 bytes,
   `806465c0981a85a3bbea9dc9b74e881b384f4bffe273e1395a14e0f8e68fb98c`. The audit's cache check
   verifies that those first 1,186,610 bytes are still byte-for-byte unchanged after the run, so
   the validation entries are untouched and test responses have only been appended.
6. `data/interim/final/` is created empty; `runs_test_A.jsonl` and `runs_test_B.jsonl` do not exist.
7. This plan has been reviewed once, with nothing run. No test-split smoke run of any size.

## Commands, in order

Arm A:

```
uv run classical-nlp route --arm A --split test --data-dir data/raw/banking77 --save data/interim/final/runs_test_A.jsonl | Tee-Object -FilePath data/interim/final/run_test_A.log
```

Arm B (the slow step; about 3,080 uncached tickets at a few seconds each):

```
uv run classical-nlp route --arm B --split test --data-dir data/raw/banking77 --model qwen2.5:7b-instruct-q3_K_M --cache data/interim/llm_cache.jsonl --save data/interim/final/runs_test_B.jsonl | Tee-Object -FilePath data/interim/final/run_test_B.log
```

Each `route` command prints the frozen results table when it finishes. That output is saved, and
no decision is made from it. The audit runs after both prediction runs and before
`route-report --final`; it is strictly an integrity checker and computes no accuracy, cost,
ECE, hypothesis status, winner or any other test-result statistic. It must show 0 failed before
the report is run or any interpretation starts:

```
uv run python scripts/audit_runs.py --split test --data-dir data/raw/banking77 --runs data/interim/final/runs_test_A.jsonl data/interim/final/runs_test_B.jsonl --cache data/interim/llm_cache.jsonl --cache-prefix-bytes 1186610 --cache-prefix-sha256 806465c0981a85a3bbea9dc9b74e881b384f4bffe273e1395a14e0f8e68fb98c
```

Final report:

```
uv run classical-nlp route-report --runs data/interim/final/runs_test_A.jsonl data/interim/final/runs_test_B.jsonl --out data/interim/final/report_test --source-note "Banking77 official test split, run once" --final
```

Expected artifacts: the two run files (3,080 rows per arm and calibration, six runs in all),
the saved console logs `run_test_A.log` and `run_test_B.log`, and nine report files:
`sensitivity.csv`, `usage.csv`, `abstentions.csv`, `comparison_cost.csv`,
`comparison_accuracy.csv`, `comparison_ece.csv`, `breakeven.csv`, `reliability.png`,
`report.md`. After the audit passes, record the SHA-256 of all of them in the evaluation log.

## If something goes wrong

- Any failure is a stop: diagnose, and record what was found. Inference is not rerun or
  repaired automatically. A rerun needs a written reason in the log saying why the original
  run was invalid, made before the rerun.
- The one continuation allowed is finishing a killed or timed-out Arm B run with the same
  command: the cache keeps every response received, and temperature 0 with a fixed seed means
  a cached response is the one the model gave. Log the interruption first. No setting changes.
- If the cause is a defect in the study code, the test split counts as seen: say so in the
  write-up rather than quietly re-running.

## Reporting rules

- Report whatever the frozen analysis produces, including an unfavourable or inconclusive
  result. Validation numbers are never substituted for test numbers.
- A bootstrap p-value shown as `0.000` is written `p < 0.001` (the floor of 2,000 resamples is
  about 1/2,000) in all prose; never `p = 0`.
- B-logprob is primary; B-verbal is a labelled ablation; B-logprob and B-verbal share one
  response per ticket, so their comparisons are one comparison, not two.
- Arm C is stated as not evaluated. Costs are routing decision cost excluding compute.
- The README, interview notes and resume line are written only after the audit and the report
  have passed review.

## Evaluation log (filled in as each step is done)

| Step | Recorded value |
|---|---|
| Plan commit SHA | |
| Ollama version, model digest | |
| Cache size and SHA-256 before the run | |
| Arm A finished (time, console output saved) | |
| Arm B finished (time, interruptions if any) | |
| Audit result | |
| SHA-256 of run files and report files | |
