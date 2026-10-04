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
| Plan commit SHA | `ad6ccdd39e16856d8df25fa3c5ba6ac2064b8728` (frozen implementation `73a6808`); tree clean, equals origin/main, CI success on that SHA; offline tests 227 passed; data checksums match; recorded 2026-10-04 00:06 UTC |
| Ollama version, model digest | Ollama 0.34.4; `qwen2.5:7b-instruct-q3_K_M` digest `29492a928341` |
| Cache size and SHA-256 before the run | 1,186,610 bytes; `806465c0981a85a3bbea9dc9b74e881b384f4bffe273e1395a14e0f8e68fb98c` |
| Arm A finished (time, console output saved) | 2026-10-04 05:36:55 to 05:37:08 IST, exit 0, 6,160 rows (3,080 x 2 calibrations); console saved to `run_test_A.log` |
| Arm B finished (time, interruptions if any) | Started 05:37:18 IST. INTERRUPTION 1: the run had been launched as a background process with a 2-hour time limit and was stopped at that limit (about 07:37 IST), not by any study error. Cache at that point: 3,140 lines = 2,001 validation + 1,139 test responses, 1,858,616 bytes; first 1,186,610 bytes still hash to `806465c0...8fb98c`. `runs_test_B.jsonl` not written (save happens at the end). No setting changed. Continuation under the protocol: same command, same model, same cache. RESUMING at 2026-10-04 09:32 IST after the recorded 2-hour interruption: the identical registered command is re-run in a terminal with no time limit. The 1,139 completed test responses are reused byte-for-byte from the append-only cache; the remaining 1,941 tickets get their first recorded responses under the same frozen configuration. Only difference from the registered text: that terminal does not accept pipes, so the console log is captured with `*> data/interim/final/run_test_B.log` instead of `| Tee-Object`; every `route` argument is identical. COMPLETED: runs_test_B.jsonl written 2026-10-04 12:53 IST (about 7h15m after the first start, including the interruption); 12,320 rows (4 runs x 3,080); cache 5,081 lines = 2,001 validation + 3,080 test, 3,003,486 bytes, sha256 `f00ef74b34e6bdd5415be4dc2c7e7c673a6bedab6f404f94af99a06e6f3b9fd2`; first 1,186,610 bytes still hash to the registered value. The redirect did not capture an exit code; evidence of a clean finish: the prompt returned, the save file exists in full, and the log has no traceback or error text (counted, not read). |
| Audit result | Run once, 2026-10-04 after Arm B completed: 14 passed, 0 failed (exit 0). Checks: JSON keys, split, the six runs, 3,080 rows with dense indices per run, y_true equals the pinned test labels, raw/isotonic predictions agree, confidences finite in [0, 1], abstention confidence 0.0, positive seconds, B token counts present, A token counts absent, cache prefix byte-identical, cache lines parse, cache keys unique. |
| SHA-256 of run files and report files | Full hashes in `results/final/SHA256SUMS`, committed unchanged from `data/interim/final/SHA256SUMS`. The nine report files are committed as byte-identical copies under `results/final/report_test/`; `sha256sum -c --ignore-missing SHA256SUMS` run inside `results/final/` checks them. The two run files, the two console logs and the response cache stay untracked; the manifest holds the hashes of the run files and logs, and the Arm B row above holds the cache's, so the exact files the committed report was built from stay identifiable. Summary: `run_test_A.log` `d29235cfb9fa1bf8...`; `run_test_B.log` `99480303e91b1d10...`; `runs_test_A.jsonl` `e6c1873fa173dde3...`; `runs_test_B.jsonl` `96df85d0fe171ae1...`; `report_test/abstentions.csv` `06e6d904aac60b01...`; `report_test/breakeven.csv` `3d17d074e132ae0b...`; `report_test/comparison_accuracy.csv` `2f105ef2d33ac4a4...`; `report_test/comparison_cost.csv` `5d7673aaf9926e61...`; `report_test/comparison_ece.csv` `55a57cb976d5825b...`; `report_test/reliability.png` `77e3ddd690bd0e4c...`; `report_test/report.md` `8d937914fbbda08f...`; `report_test/sensitivity.csv` `3c7dd94cfcb950af...`; `report_test/usage.csv` `776a34c7b6df049d...`. Report written 2026-10-04 13:26 IST by `route-report --final`, exit 0, 9 files. |
