# Router study report

Source: Banking77 official test split, run once

Split: test; n = 3080 tickets.

**FINAL: official test split**

## Routing decision cost per 1,000 tickets, excluding compute

ratio is misroute : handoff. ECE is top-label, 10 equal-width bins, over all rows; an abstention has confidence 0.0 and is never correct.

| arm | calibration | ratio | n | auto routed | handed off | misrouted | abstained | coverage | auto accuracy | routing cost per 1000 | ece | brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | raw | 2 | 3080 | 1352 | 1728 | 22 | 0 | 0.439 | 0.984 | 575.3 | 0.404 | 0.274 |
| A | raw | 5 | 3080 | 372 | 2708 | 3 | 0 | 0.121 | 0.992 | 884.1 | 0.404 | 0.274 |
| A | raw | 10 | 3080 | 102 | 2978 | 1 | 0 | 0.033 | 0.990 | 970.1 | 0.404 | 0.274 |
| A | raw | 20 | 3080 | 27 | 3053 | 0 | 0 | 0.009 | 1.000 | 991.2 | 0.404 | 0.274 |
| A | isotonic | 2 | 3080 | 2934 | 146 | 325 | 0 | 0.953 | 0.889 | 258.4 | 0.007 | 0.092 |
| A | isotonic | 5 | 3080 | 2465 | 615 | 163 | 0 | 0.800 | 0.934 | 464.3 | 0.007 | 0.092 |
| A | isotonic | 10 | 3080 | 1686 | 1394 | 42 | 0 | 0.547 | 0.975 | 589.0 | 0.007 | 0.092 |
| A | isotonic | 20 | 3080 | 1478 | 1602 | 30 | 0 | 0.480 | 0.980 | 714.9 | 0.007 | 0.092 |
| B-logprob | raw | 2 | 3080 | 2781 | 299 | 928 | 167 | 0.903 | 0.666 | 699.7 | 0.244 | 0.241 |
| B-logprob | raw | 5 | 3080 | 2342 | 738 | 626 | 167 | 0.760 | 0.733 | 1255.8 | 0.244 | 0.241 |
| B-logprob | raw | 10 | 3080 | 2121 | 959 | 500 | 167 | 0.689 | 0.764 | 1934.7 | 0.244 | 0.241 |
| B-logprob | raw | 20 | 3080 | 1925 | 1155 | 399 | 167 | 0.625 | 0.793 | 2965.9 | 0.244 | 0.241 |
| B-logprob | isotonic | 2 | 3080 | 1925 | 1155 | 399 | 167 | 0.625 | 0.793 | 634.1 | 0.026 | 0.162 |
| B-logprob | isotonic | 5 | 3080 | 722 | 2358 | 52 | 167 | 0.234 | 0.928 | 850.0 | 0.026 | 0.162 |
| B-logprob | isotonic | 10 | 3080 | 338 | 2742 | 15 | 167 | 0.110 | 0.956 | 939.0 | 0.026 | 0.162 |
| B-logprob | isotonic | 20 | 3080 | 194 | 2886 | 6 | 167 | 0.063 | 0.969 | 976.0 | 0.026 | 0.162 |
| B-verbal | raw | 2 | 3080 | 2907 | 173 | 1036 | 167 | 0.944 | 0.644 | 728.9 | 0.241 | 0.264 |
| B-verbal | raw | 5 | 3080 | 2463 | 617 | 712 | 167 | 0.800 | 0.711 | 1356.2 | 0.241 | 0.264 |
| B-verbal | raw | 10 | 3080 | 573 | 2507 | 108 | 167 | 0.186 | 0.812 | 1164.6 | 0.241 | 0.264 |
| B-verbal | raw | 20 | 3080 | 270 | 2810 | 13 | 167 | 0.088 | 0.952 | 996.8 | 0.241 | 0.264 |
| B-verbal | isotonic | 2 | 3080 | 2463 | 617 | 712 | 167 | 0.800 | 0.711 | 662.7 | 0.031 | 0.188 |
| B-verbal | isotonic | 5 | 3080 | 270 | 2810 | 13 | 167 | 0.088 | 0.952 | 933.4 | 0.031 | 0.188 |
| B-verbal | isotonic | 10 | 3080 | 270 | 2810 | 13 | 167 | 0.088 | 0.952 | 954.5 | 0.031 | 0.188 |
| B-verbal | isotonic | 20 | 3080 | 0 | 3080 | 0 | 167 | 0.000 |  | 1000.0 | 0.031 | 0.188 |

## Abstentions

| arm | abstained | n | share |
|---|---|---|---|
| A | 0 | 3080 | 0.000 |
| B-logprob | 167 | 3080 | 0.054 |
| B-verbal | 167 | 3080 | 0.054 |

## Headline comparison

A - B in routing decision cost per 1,000 tickets, excluding compute; negative means A is lower. Paired bootstrap, 2000 resamples, seed 20260830, 95% percentile interval.

| pair | arm a | arm b | calibration | ratio | cost a | cost b | difference | ci low | ci high | p value | headline |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A vs B-logprob | A | B-logprob | isotonic | 10 | 589.0 | 939.0 | -350.0 | -397.7 | -299.3 | 0.000 | true |

## Other cost comparisons

| pair | arm a | arm b | calibration | ratio | cost a | cost b | difference | ci low | ci high | p value | headline |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A vs B-logprob | A | B-logprob | raw | 2 | 575.3 | 699.7 | -124.4 | -156.8 | -90.9 | 0.000 | false |
| A vs B-logprob | A | B-logprob | raw | 5 | 884.1 | 1255.8 | -371.8 | -438.0 | -305.2 | 0.000 | false |
| A vs B-logprob | A | B-logprob | raw | 10 | 970.1 | 1934.7 | -964.6 | -1092.5 | -843.2 | 0.000 | false |
| A vs B-logprob | A | B-logprob | raw | 20 | 991.2 | 2965.9 | -1974.7 | -2212.0 | -1735.7 | 0.000 | false |
| A vs B-logprob | A | B-logprob | isotonic | 2 | 258.4 | 634.1 | -375.6 | -405.2 | -346.1 | 0.000 | false |
| A vs B-logprob | A | B-logprob | isotonic | 5 | 464.3 | 850.0 | -385.7 | -430.2 | -339.6 | 0.000 | false |
| A vs B-logprob | A | B-logprob | isotonic | 20 | 714.9 | 976.0 | -261.0 | -334.7 | -186.4 | 0.000 | false |
| A vs B-verbal | A | B-verbal | raw | 2 | 575.3 | 728.9 | -153.6 | -187.0 | -119.2 | 0.000 | false |
| A vs B-verbal | A | B-verbal | raw | 5 | 884.1 | 1356.2 | -472.1 | -546.8 | -399.7 | 0.000 | false |
| A vs B-verbal | A | B-verbal | raw | 10 | 970.1 | 1164.6 | -194.5 | -252.9 | -134.4 | 0.000 | false |
| A vs B-verbal | A | B-verbal | raw | 20 | 991.2 | 996.8 | -5.5 | -52.3 | +38.0 | 0.879 | false |
| A vs B-verbal | A | B-verbal | isotonic | 2 | 258.4 | 662.7 | -404.2 | -436.7 | -370.5 | 0.000 | false |
| A vs B-verbal | A | B-verbal | isotonic | 5 | 464.3 | 933.4 | -469.2 | -509.1 | -426.9 | 0.000 | false |
| A vs B-verbal | A | B-verbal | isotonic | 10 | 589.0 | 954.5 | -365.6 | -409.4 | -318.2 | 0.000 | false |
| A vs B-verbal | A | B-verbal | isotonic | 20 | 714.9 | 1000.0 | -285.1 | -353.9 | -212.3 | 0.000 | false |

## Accuracy (McNemar, exact)

Abstentions count as wrong.

| arm a | arm b | only a | only b | p value |
|---|---|---|---|---|
| A | B-logprob | 937 | 135 | 0.000 |
| A | B-verbal | 937 | 135 | 0.000 |

## Calibration (ECE) comparison

| arm a | arm b | calibration | ece a | ece b | difference | ci low | ci high | p value |
|---|---|---|---|---|---|---|---|---|
| A | B-logprob | raw | 0.404 | 0.244 | +0.160 | +0.140 | +0.179 | 0.000 |
| A | B-logprob | isotonic | 0.007 | 0.026 | -0.019 | -0.030 | -0.002 | 0.027 |
| A | B-verbal | raw | 0.404 | 0.241 | +0.162 | +0.143 | +0.182 | 0.000 |
| A | B-verbal | isotonic | 0.007 | 0.031 | -0.023 | -0.035 | -0.001 | 0.042 |

## Hypotheses

- H1 (routing cost per 1,000, A - B, isotonic, 10:1): A lower; difference -350.0, 95% CI [-397.7, -299.3]
- H2 (ECE, A - B, isotonic): A lower; difference -0.019, 95% CI [-0.030, -0.002]

## Break-even compute sensitivity

Advantage per ticket is mean(cost A - cost B); positive means B is cheaper before compute. The break-even is that advantage when it is positive and blank otherwise; per second it is divided by B's mean model-side seconds.

Costs are in handoff units (one human handoff = 1.0); multiply by your own handoff cost. Arm A's own compute is not subtracted.

| arm b | calibration | ratio | advantage per ticket | breakeven per ticket | b mean seconds | breakeven per second | status |
|---|---|---|---|---|---|---|---|
| B-logprob | raw | 2 | -0.1244 |  | 4.147 |  | A cheaper before compute |
| B-logprob | raw | 5 | -0.3718 |  | 4.147 |  | A cheaper before compute |
| B-logprob | raw | 10 | -0.9646 |  | 4.147 |  | A cheaper before compute |
| B-logprob | raw | 20 | -1.9747 |  | 4.147 |  | A cheaper before compute |
| B-logprob | isotonic | 2 | -0.3756 |  | 4.147 |  | A cheaper before compute |
| B-logprob | isotonic | 5 | -0.3857 |  | 4.147 |  | A cheaper before compute |
| B-logprob | isotonic | 10 | -0.3500 |  | 4.147 |  | A cheaper before compute |
| B-logprob | isotonic | 20 | -0.2610 |  | 4.147 |  | A cheaper before compute |
| B-verbal | raw | 2 | -0.1536 |  | 4.147 |  | A cheaper before compute |
| B-verbal | raw | 5 | -0.4721 |  | 4.147 |  | A cheaper before compute |
| B-verbal | raw | 10 | -0.1945 |  | 4.147 |  | A cheaper before compute |
| B-verbal | raw | 20 | -0.0055 |  | 4.147 |  | A cheaper before compute |
| B-verbal | isotonic | 2 | -0.4042 |  | 4.147 |  | A cheaper before compute |
| B-verbal | isotonic | 5 | -0.4692 |  | 4.147 |  | A cheaper before compute |
| B-verbal | isotonic | 10 | -0.3656 |  | 4.147 |  | A cheaper before compute |
| B-verbal | isotonic | 20 | -0.2851 |  | 4.147 |  | A cheaper before compute |

## Usage

| arm | n | latency p50 | latency p95 | prompt tokens mean | cached prompt tokens mean | output tokens mean |
|---|---|---|---|---|---|---|
| A | 3080 | 0.002 | 0.003 |  |  |  |
| B-logprob | 3080 | 4.072 | 5.589 | 498.6 | 480.9 | 13.7 |
| B-verbal | 3080 | 4.072 | 5.589 | 498.6 | 480.9 | 13.7 |

Arm A's latency is wall-clock per ticket in-process; arm B's is the model-side time Ollama reports with model load excluded, so the two are not directly comparable.

Arm C (dedicated decision model): not evaluated (not available).

## Limits

- Isotonic calibrators are not refitted inside a bootstrap resample, and cross-fitted confidences are treated as fixed.
- Inference is not priced: routing cost excludes compute, and the break-even is arithmetic, not a price.
- Abstentions count as wrong in McNemar and as handoffs in cost.
