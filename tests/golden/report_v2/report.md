# Router study report

Source: synthetic fixture: tests the code, not results

Split: test; n = 12 tickets.

**FINAL: official test split**

## Routing decision cost per 1,000 tickets, excluding compute

ratio is misroute : handoff. ECE is top-label, 10 equal-width bins, over all rows; an abstention has confidence 0.0 and is never correct.

| arm | calibration | ratio | n | auto routed | handed off | misrouted | abstained | coverage | auto accuracy | routing cost per 1000 | ece | brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | raw | 2 | 12 | 0 | 12 | 0 | 0 | 0.000 |  | 1000.0 | 0.643 | 0.433 |
| A | raw | 5 | 12 | 0 | 12 | 0 | 0 | 0.000 |  | 1000.0 | 0.643 | 0.433 |
| A | raw | 10 | 12 | 0 | 12 | 0 | 0 | 0.000 |  | 1000.0 | 0.643 | 0.433 |
| A | raw | 20 | 12 | 0 | 12 | 0 | 0 | 0.000 |  | 1000.0 | 0.643 | 0.433 |
| A | isotonic | 2 | 12 | 11 | 1 | 0 | 0 | 0.917 | 1.000 | 83.3 | 0.069 | 0.025 |
| A | isotonic | 5 | 12 | 11 | 1 | 0 | 0 | 0.917 | 1.000 | 83.3 | 0.069 | 0.025 |
| A | isotonic | 10 | 12 | 9 | 3 | 0 | 0 | 0.750 | 1.000 | 250.0 | 0.069 | 0.025 |
| A | isotonic | 20 | 12 | 9 | 3 | 0 | 0 | 0.750 | 1.000 | 250.0 | 0.069 | 0.025 |
| B-logprob | raw | 2 | 12 | 5 | 7 | 3 | 0 | 0.417 | 0.400 | 1083.3 | 0.314 | 0.260 |
| B-logprob | raw | 5 | 12 | 1 | 11 | 1 | 0 | 0.083 | 0.000 | 1333.3 | 0.314 | 0.260 |
| B-logprob | raw | 10 | 12 | 0 | 12 | 0 | 0 | 0.000 |  | 1000.0 | 0.314 | 0.260 |
| B-logprob | raw | 20 | 12 | 0 | 12 | 0 | 0 | 0.000 |  | 1000.0 | 0.314 | 0.260 |
| B-logprob | isotonic | 2 | 12 | 0 | 12 | 0 | 0 | 0.000 |  | 1000.0 | 0.233 | 0.277 |
| B-logprob | isotonic | 5 | 12 | 0 | 12 | 0 | 0 | 0.000 |  | 1000.0 | 0.233 | 0.277 |
| B-logprob | isotonic | 10 | 12 | 0 | 12 | 0 | 0 | 0.000 |  | 1000.0 | 0.233 | 0.277 |
| B-logprob | isotonic | 20 | 12 | 0 | 12 | 0 | 0 | 0.000 |  | 1000.0 | 0.233 | 0.277 |
| B-verbal | raw | 2 | 12 | 9 | 3 | 6 | 0 | 0.750 | 0.333 | 1250.0 | 0.325 | 0.367 |
| B-verbal | raw | 5 | 12 | 2 | 10 | 2 | 0 | 0.167 | 0.000 | 1666.7 | 0.325 | 0.367 |
| B-verbal | raw | 10 | 12 | 0 | 12 | 0 | 0 | 0.000 |  | 1000.0 | 0.325 | 0.367 |
| B-verbal | raw | 20 | 12 | 0 | 12 | 0 | 0 | 0.000 |  | 1000.0 | 0.325 | 0.367 |
| B-verbal | isotonic | 2 | 12 | 0 | 12 | 0 | 0 | 0.000 |  | 1000.0 | 0.262 | 0.296 |
| B-verbal | isotonic | 5 | 12 | 0 | 12 | 0 | 0 | 0.000 |  | 1000.0 | 0.262 | 0.296 |
| B-verbal | isotonic | 10 | 12 | 0 | 12 | 0 | 0 | 0.000 |  | 1000.0 | 0.262 | 0.296 |
| B-verbal | isotonic | 20 | 12 | 0 | 12 | 0 | 0 | 0.000 |  | 1000.0 | 0.262 | 0.296 |

## Abstentions

| arm | abstained | n | share |
|---|---|---|---|
| A | 0 | 12 | 0.000 |
| B-logprob | 0 | 12 | 0.000 |
| B-verbal | 0 | 12 | 0.000 |

## Headline comparison

A - B in routing decision cost per 1,000 tickets, excluding compute; negative means A is lower. Paired bootstrap, 500 resamples, seed 20260830, 95% percentile interval.

| pair | arm a | arm b | calibration | ratio | cost a | cost b | difference | ci low | ci high | p value | headline |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A vs B-logprob | A | B-logprob | isotonic | 10 | 250.0 | 1000.0 | -750.0 | -916.7 | -500.0 | 0.000 | true |

## Other cost comparisons

| pair | arm a | arm b | calibration | ratio | cost a | cost b | difference | ci low | ci high | p value | headline |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A vs B-logprob | A | B-logprob | raw | 2 | 1000.0 | 1083.3 | -83.3 | -416.7 | +333.3 | 0.872 | false |
| A vs B-logprob | A | B-logprob | raw | 5 | 1000.0 | 1333.3 | -333.3 | -1000.0 | +0.0 | 0.656 | false |
| A vs B-logprob | A | B-logprob | raw | 10 | 1000.0 | 1000.0 | +0.0 | +0.0 | +0.0 | 1.000 | false |
| A vs B-logprob | A | B-logprob | raw | 20 | 1000.0 | 1000.0 | +0.0 | +0.0 | +0.0 | 1.000 | false |
| A vs B-logprob | A | B-logprob | isotonic | 2 | 83.3 | 1000.0 | -916.7 | -1000.0 | -750.0 | 0.000 | false |
| A vs B-logprob | A | B-logprob | isotonic | 5 | 83.3 | 1000.0 | -916.7 | -1000.0 | -750.0 | 0.000 | false |
| A vs B-logprob | A | B-logprob | isotonic | 20 | 250.0 | 1000.0 | -750.0 | -916.7 | -500.0 | 0.000 | false |
| A vs B-verbal | A | B-verbal | raw | 2 | 1000.0 | 1250.0 | -250.0 | -750.0 | +250.0 | 0.396 | false |
| A vs B-verbal | A | B-verbal | raw | 5 | 1000.0 | 1666.7 | -666.7 | -1666.7 | +0.0 | 0.228 | false |
| A vs B-verbal | A | B-verbal | raw | 10 | 1000.0 | 1000.0 | +0.0 | +0.0 | +0.0 | 1.000 | false |
| A vs B-verbal | A | B-verbal | raw | 20 | 1000.0 | 1000.0 | +0.0 | +0.0 | +0.0 | 1.000 | false |
| A vs B-verbal | A | B-verbal | isotonic | 2 | 83.3 | 1000.0 | -916.7 | -1000.0 | -750.0 | 0.000 | false |
| A vs B-verbal | A | B-verbal | isotonic | 5 | 83.3 | 1000.0 | -916.7 | -1000.0 | -750.0 | 0.000 | false |
| A vs B-verbal | A | B-verbal | isotonic | 10 | 250.0 | 1000.0 | -750.0 | -916.7 | -500.0 | 0.000 | false |
| A vs B-verbal | A | B-verbal | isotonic | 20 | 250.0 | 1000.0 | -750.0 | -916.7 | -500.0 | 0.000 | false |

## Accuracy (McNemar, exact)

Abstentions count as wrong.

| arm a | arm b | only a | only b | p value |
|---|---|---|---|---|
| A | B-logprob | 8 | 1 | 0.039 |
| A | B-verbal | 8 | 1 | 0.039 |

## Calibration (ECE) comparison

| arm a | arm b | calibration | ece a | ece b | difference | ci low | ci high | p value |
|---|---|---|---|---|---|---|---|---|
| A | B-logprob | raw | 0.643 | 0.314 | +0.329 | +0.103 | +0.429 | 0.000 |
| A | B-logprob | isotonic | 0.069 | 0.233 | -0.165 | -0.437 | +0.043 | 0.172 |
| A | B-verbal | raw | 0.643 | 0.325 | +0.318 | +0.039 | +0.473 | 0.020 |
| A | B-verbal | isotonic | 0.069 | 0.262 | -0.193 | -0.460 | +0.015 | 0.084 |

## Hypotheses

- H1 (routing cost per 1,000, A - B, isotonic, 10:1): A lower; difference -750.0, 95% CI [-916.7, -500.0]
- H2 (ECE, A - B, isotonic): no clear difference; difference -0.165, 95% CI [-0.437, +0.043]

## Break-even compute sensitivity

Advantage per ticket is mean(cost A - cost B); positive means B is cheaper before compute. The break-even is that advantage when it is positive and blank otherwise; per second it is divided by B's mean model-side seconds.

Costs are in handoff units (one human handoff = 1.0); multiply by your own handoff cost. Arm A's own compute is not subtracted.

| arm b | calibration | ratio | advantage per ticket | breakeven per ticket | b mean seconds | breakeven per second | status |
|---|---|---|---|---|---|---|---|
| B-logprob | raw | 2 | -0.0833 |  | 1.000 |  | A cheaper before compute |
| B-logprob | raw | 5 | -0.3333 |  | 1.000 |  | A cheaper before compute |
| B-logprob | raw | 10 | +0.0000 |  | 1.000 |  | equal before compute |
| B-logprob | raw | 20 | +0.0000 |  | 1.000 |  | equal before compute |
| B-logprob | isotonic | 2 | -0.9167 |  | 1.000 |  | A cheaper before compute |
| B-logprob | isotonic | 5 | -0.9167 |  | 1.000 |  | A cheaper before compute |
| B-logprob | isotonic | 10 | -0.7500 |  | 1.000 |  | A cheaper before compute |
| B-logprob | isotonic | 20 | -0.7500 |  | 1.000 |  | A cheaper before compute |
| B-verbal | raw | 2 | -0.2500 |  | 1.000 |  | A cheaper before compute |
| B-verbal | raw | 5 | -0.6667 |  | 1.000 |  | A cheaper before compute |
| B-verbal | raw | 10 | +0.0000 |  | 1.000 |  | equal before compute |
| B-verbal | raw | 20 | +0.0000 |  | 1.000 |  | equal before compute |
| B-verbal | isotonic | 2 | -0.9167 |  | 1.000 |  | A cheaper before compute |
| B-verbal | isotonic | 5 | -0.9167 |  | 1.000 |  | A cheaper before compute |
| B-verbal | isotonic | 10 | -0.7500 |  | 1.000 |  | A cheaper before compute |
| B-verbal | isotonic | 20 | -0.7500 |  | 1.000 |  | A cheaper before compute |

## Usage

| arm | n | latency p50 | latency p95 | prompt tokens mean | cached prompt tokens mean | output tokens mean |
|---|---|---|---|---|---|---|
| A | 12 | 0.007 | 0.011 |  |  |  |
| B-logprob | 12 | 1.000 | 1.000 | 600.0 | 0.0 | 14.0 |
| B-verbal | 12 | 1.000 | 1.000 | 600.0 | 0.0 | 14.0 |

Arm A's latency is wall-clock per ticket in-process; arm B's is the model-side time Ollama reports with model load excluded, so the two are not directly comparable.

Arm C (dedicated decision model): not evaluated (not available).

## Limits

- Isotonic calibrators are not refitted inside a bootstrap resample, and cross-fitted confidences are treated as fixed.
- Inference is not priced: routing cost excludes compute, and the break-even is arithmetic, not a price.
- Abstentions count as wrong in McNemar and as handoffs in cost.
