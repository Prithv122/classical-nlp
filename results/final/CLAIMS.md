# Canonical claims: router study, Banking77 official test split

This sheet is the only source for result claims about the router study in the README,
interview notes and resume line. Every number is taken from `results/final/report_test/`
(the output of `route-report --final`, hashes in `results/final/SHA256SUMS`), produced under
the protocol in `FINAL_EVAL.md` (frozen at `ad6ccdd`, implementation frozen at `73a6808`).
Before this sheet was written, the nine report files were checked against their hashes and
every CSV row was re-derived from its own counts (cost, coverage, auto accuracy, differences,
break-even, abstention share): no mismatches.

## Definitions the claims depend on

- **Routing rule.** A ticket is auto-routed iff (1 − confidence) × ratio < 1, i.e. confidence
  above 0.9 at 10:1; otherwise it is handed to a human.
- **Cost.** A handoff costs 1 unit; a misroute costs `ratio` units. Cost per 1,000 tickets =
  (handoffs + ratio × misroutes) / 3,080 × 1,000. Handing off every ticket costs 1,000 by
  construction. No currency, and compute is excluded.
- **Arm A.** Word TF-IDF + logistic regression, fitted on 8,002 training tickets (the other 80%
  of the 10,003-row train split).
- **Arm B.** `qwen2.5:7b-instruct` at 3-bit quantisation (q3_K_M), run locally, zero-shot, with
  all 77 intents listed in the prompt. Parsing is strict: a reply that does not name a listed
  intent exactly is an abstention (no retry, case folding or constrained decoding). An
  abstention counts as a handoff for cost and as wrong for accuracy.
- **Calibration.** Top-label isotonic regression fitted on the 2,001 validation tickets and
  applied to the test predictions.

## 1. Primary result (H1)

- On 3,080 held-out Banking77 tickets, at a 10:1 misroute-to-handoff cost ratio, isotonic:
  **A 589.0 vs B-logprob 939.0** routing cost per 1,000 tickets.
- **A − B = −350.0, 95% CI [−397.7, −299.3], `p < 0.001`**, excluding compute. H1 status:
  "A lower".
- Mechanism: A auto-routed 1,686 tickets (54.7%) with 42 misroutes; B auto-routed 338 (11.0%)
  with 15. **A wins by automating far more tickets, not by misrouting fewer.**
- Sensitivity (reported comparisons, not hypotheses; not corrected for multiplicity): A is
  lower than B-logprob at all four ratios under both calibrations, and all eight intervals
  exclude zero. Isotonic: 2:1 −375.6 [−405.2, −346.1]; 5:1 −385.7 [−430.2, −339.6];
  20:1 −261.0 [−334.7, −186.4].

## 2. H2: calibration

- Isotonic ECE: **A 0.007 vs B-logprob 0.026; difference −0.019, 95% CI [−0.030, −0.002],
  `p = 0.027`.** Wording: "the interval narrowly excluded zero". H2 status: "A lower". Both
  arms are well calibrated after isotonic in absolute terms.
- **Raw ECE runs the other way:** A 0.404 (under-confident) vs B-logprob 0.244
  (over-confident), difference +0.160 [+0.140, +0.179]. Raw B-logprob was better calibrated;
  A became better calibrated only after the frozen isotonic step.
- Calibration changed outcomes (descriptive, no interval): A's 10:1 cost went from 970.1 raw to
  589.0 isotonic; raw B-logprob at 5, 10 and 20:1 cost more than handing everything off
  (1,255.8, 1,934.7 and 2,965.9 against 1,000).

## 3. Supporting accuracy evidence

- Exact McNemar, abstentions counted as wrong: **937 tickets only A got right, 135 only B got
  right**; exact p ≈ 3.0 × 10⁻¹⁴⁸ (a computed value, not a bootstrap floor).
- A was right on 802 more tickets (26.0 percentage points of 3,080).

## 4. Operational measurements

- B abstained on **167 of 3,080 tickets (5.4%)**; A never abstains.
- B model-side time: mean 4.15 s, p50 4.07 s, p95 5.59 s (model load excluded, one local
  machine). About 499 prompt tokens per ticket (481 cached) and about 14 output tokens.
- A's timing is in-process wall-clock and B's is model-side, so the two are not compared
  numerically.
- Break-even: B was not cheaper before compute in any of the 16 settings, so there is no
  compute price at which it breaks even.

## 5. B-verbal ablation

- B-verbal reads the same responses as B-logprob: identical predictions (937 / 135) and
  identical abstentions (167). Only the confidence signal differs. It is not a second,
  independent model comparison.
- Isotonic ECE 0.031; 10:1 isotonic cost 954.5.
- The only comparison with no clear difference: A vs raw B-verbal at 20:1, −5.5 [−52.3, +38.0],
  p = 0.879.
- B-logprob and B-verbal were never compared with each other statistically.

## 6. Limitations to state

- Compute is excluded: neither arm's inference is priced, and A's compute is not subtracted.
- Calibrators were not refitted inside bootstrap resamples; calibrated confidences are treated
  as fixed.
- One LLM configuration: one 3-bit 7B model, zero-shot, one prompt, strict parsing.
- Unequal supervision: A learned from 8,002 labelled tickets; B saw no examples.
- One dataset (Banking77: English, short queries, 77 intents).
- The 10:1 ratio is an assumption, not a measured cost.
- Arm C (dedicated decision model) was not evaluated.

## Process facts that may be stated

- The evaluation protocol was committed before the test split was scored.
- The test split was run once. One interruption was logged and resumed under the protocol's
  continuation rule; the integrity audit passed 14 of 14 checks.
- Arm B results can be replayed from the response cache.

## Claims that must not be made

- "TF-IDF beats LLMs" or "classical beats LLMs" in general.
- "A misroutes less" (42 vs 15 at the headline setting), or "A is more accurate when it
  auto-routes" as a comparison: the arms auto-route different numbers of tickets.
- "A is better calibrated" without "after isotonic calibration".
- `p = 0`, `p < 1/2000` or `p < 0.0005`; "borderline", "marginal" or "highly significant" for H2.
- Any relative percentage such as "37% cheaper": no interval was computed for a ratio.
- Any currency or "saves $".
- Any compute-inclusive conclusion, such as "including compute, A wins by more".
- Any latency multiple, such as "2,500× faster".
- An overall test accuracy percentage for either arm: none is in the artifacts.
- B-verbal as a second, independent LLM.
- "Deterministic" or "reproducible" LLM outputs beyond "replayable from the response cache".
- Anything about Arm C beyond "not evaluated".
- Validation numbers presented as results.
- "Pre-registered" in a way that implies an external registry; say "protocol committed before
  the test run".
- Generalisation to production traffic.
