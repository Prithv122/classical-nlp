# Arm C protocol: three-arm comparison on a new Banking77 partition

Written on 2026-10-14. No functional arm C code exists yet (only the stub at `73a6808`),
no decision-model API has been called, and no arm has scored a row of the new held-out set.
Everything below is fixed at the commit that adds this file. A change made before the
held-out set is first scored goes in a dated amendment at the bottom; this text is never
edited in place. After the held-out set is scored, nothing below changes because of results.

## What this study is, and what it is not

**Question.** Does a dedicated decision model (TypeSafe Jev) route Banking77 tickets at lower
decision cost than the calibrated TF-IDF baseline and the local LLM, and do the vendor's
calibration and latency claims hold on this task?

**This is a new partition-based A/B/C comparison.** Its held-out rows come from the Banking77
training split, and those rows were used to fit arm A in v2. It is therefore **not** an
independent external validation of the shipped v2 model. Its numbers are not pooled with,
and not compared against, the v2 results on the official test split (`results/final/`), as
though the two were one experiment. The official test split is not read by anything in this
study.

## Data and partition

- **Source.** Banking77 train at the pinned revision in `src/classicalnlp/tickets.py`,
  `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b`, 10,003 rows, 77 intents.
  Index space throughout: the 0-based row index in that file.
- **Step 1, unchanged from v2.** `train_test_split(range(10003), test_size=0.2,
  stratify=<train labels>, random_state=20260830)` gives the old fit indices F (8,002) and the
  validation indices V (2,001). This is the call `validation_split` makes, and the order of V is
  its order.
- **Step 2, candidate pool.** Normalise each text by lowercasing and collapsing whitespace. The
  pool is every index in F whose normalised text occurs exactly once in the whole training split,
  **sorted ascending** (7,997 indices). The 5 excluded rows stay in the fit set. The check is for
  exact matches after normalisation only; near-duplicates between H and F' are not removed.
- **Step 3, held-out set H.** `train_test_split(pool, test_size=2002, stratify=<labels of pool,
  in pool order>, random_state=20261014)`; the second return value, sorted ascending, is H. The
  new fit set F' is F minus H (6,000 indices).
- **Registered result of the recipe.** It was run once, read-only, before this file was written:
  H has 2,002 rows (7 to 37 per intent), F' has 6,000 rows (at least 21 per intent), and no text
  in H appears in F' or V. Written as `train_index,label` with a header row, one row per index in
  ascending order, LF line endings and integer label ids, the file has SHA-256
  `0b64a728f7f10a633bc8a8fd482f9993302299f8eb823318f98142a343a1e814`.
- **Why proportional, not a fixed count per intent.** The smallest intents have 28 and 33 rows in
  F. A fixed 26 per intent would leave arm A 2 and 7 training rows for them, so the refit would be
  weaker for reasons unrelated to the question. Proportional stratification keeps the training
  distribution in both F' and H.
- **Freeze of the IDs.** The partition script and `data/partitions/abc_heldout.csv` are committed
  in the implementation commit (gate G1). The file must hash to the value above, and the loader
  refuses a file whose checksum or row count differs.
- **Prior exposure, stated in the report.** H comes from the public Banking77 training split, so
  any of the three models may have seen these texts in pretraining; that is not controllable
  here. v2's B cache holds responses for V and the test split only (5,081 = 2,001 + 3,080), so B
  never scored an H row in v2. Whether individual fit rows were read during v2 development is not
  recorded.

## Arms

All three arms are evaluated on exactly the same 2,002 tickets in H and calibrated on the same
2,001 tickets in V.

- **A: TF-IDF (word) and logistic regression.** Features, hyperparameters and seed are those of
  the frozen v2 implementation (`73a6808`). It is refit on F' only, never on any row of H. Raw
  confidence is the top-label probability.
- **B: local LLM, unchanged.** Model `qwen2.5:7b-instruct-q3_K_M`, Ollama digest `29492a928341`,
  the v2 prompt, temperature 0, seed 20260830, one response per ticket feeding B-logprob (primary)
  and B-verbal (ablation), with the v2 confidence and abstention definitions. **Runtime:** Ollama
  0.40.2 (observed 2026-10-14; v2 ran on 0.34.4). So that the calibrator and the held-out responses
  come from one runtime, B's responses for V and H are produced fresh into a new cache file. v2's
  cache is not used as a response source; it is read only to report agreement between B's new V
  intents and v2's V intents, a descriptive drift figure and not a gate.
- **C: dedicated decision model (Jev).** One `choice` question per ticket:
  - `instructions`: the first two sentences of B's prompt, verbatim ("You route
    customer-support messages for a bank. Pick the single intent that best matches the
    message.");
  - `criteria`: each of the 77 intent names mapped to `null`, which gives C the same label
    information B has (names only);
  - `state`: the ticket text, unmodified;
  - requested model revision `jev-1.13.0`. Every response's reported model version must equal
    it, or the run stops.

  **C-prob (primary):** label = `choice`, raw confidence = `probabilities[choice]`. **C-vendor
  (ablation):** the same label, raw confidence = the response's `confidence` field, which tests the
  vendor's own number. One response feeds both. The report states how often `choice` differs from
  the argmax of `probabilities`, ties included.

  **Abstention, per variant, mirroring B.** If `choice` is not one of the 77 names, or
  `probabilities` does not cover all 77 names, or they do not sum to 1 within 1e-3 (no
  renormalisation), both variants abstain: `ABSTAIN` with confidence 0.0. If only the `confidence`
  field is missing or outside [0, 1], C-vendor abstains and C-prob does not.

  **Repeat calls.** The first response stored in the cache is the record for that ticket. Jev's
  self-agreement is measured once (gate G2) and reported; it is not used to change any record.

## Access and endpoints

- **Routes.** Either the TypeSafe API (`api.typesafe.ai`, `/v1/systemone`) or Cloudflare Workers
  AI (model `typesafe/jev`), whichever gives working access first. Account creation and any
  billing setup are the owner's steps.
- **One endpoint for all of C.** The endpoint is recorded in the pre-run commit. C's V responses
  and its H responses both come from it. Endpoint identity is part of every cache key and every
  run row.
- **Probe.** On the chosen endpoint, with the committed client, 20 rows of V: positions
  `numpy.random.default_rng(20261014).choice(2001, 20, replace=False)` into V in its Step 1 order,
  which are train indices 2520, 2375, 9033, 1513, 7455, 6821, 7355, 8960, 4286, 6389, 3639, 8747,
  5790, 8017, 9651, 2891, 9449, 3053, 2516, 9554. The probe records the reported model version,
  the answer schema, `input_tokens` and latency, and its responses enter the cache, so the V run
  reuses them. Then the same 20 requests are sent once more, bypassing the cache, to measure
  self-agreement (same `choice`, largest probability difference); those second responses are
  logged and never used as records. V is a calibration set, so the probe touches no evaluation data.
- **No silent switch.** A change of endpoint, or of the reported model version, during the study
  stops the run. Resuming needs a dated amendment in this file. The second route counts as
  interchangeable only if, on the same 20 probe rows, it reports the same model version string,
  returns the same `choice` on 20 of 20, and differs on every probability by no more than the
  first route's own self-agreement difference or 0.01, whichever is larger. Even then, C's V and H
  are both re-run on the new endpoint, within the total budget, and endpoints are never mixed
  between V and H.

## Budget, caching and credentials

- **Price.** $0.042 per million input tokens, output free, as published in TypeSafe's launch post
  (2026-10-13) and the Cloudflare model listing. It is re-checked on the day of the run. If it is
  higher, the figures below are recomputed and re-approved before any call.
- **Run.** One invocation of the route command for one arm on one split. C has two runs, C-V and
  C-H; the probe and self-agreement calls are logged as a third.
- **Calls.** First attempts for one full C pass: V 2,001 + H 2,002 + 20 self-agreement = 4,023
  (the 20 probe calls are part of V). Retries happen on `429`, `500`, `502`, `503`, `504`, `529` and
  on a timeout of 30 s, with exponential backoff and at most 5 attempts per ticket; any other error
  is not retried. Every billed attempt counts in the ledger.
- **Per-call guard.** A response reporting more than 4,000 input tokens stops the run. At that
  guard, first attempts bound one pass at 4,023 × 4,000 × $0.042 / 10^6 = $0.68. With every
  attempt retried the theoretical ceiling is five times that, so the hard caps below are what
  actually limit spend.
- **Hard caps.** $2.00 per run, computed cumulatively from the usage reported in responses; the run
  aborts before the next call once it is reached. $5.00 in total for this study, including the probe
  and any re-runs. Each run is logged in `NOTES.md` (calls, input tokens, dollars, total remaining),
  and the provider dashboard figure is added after the run.
- **Cache.** Append-only JSONL, keyed by the SHA-256 of the endpoint identity, the requested revision
  `jev-1.13.0` and the canonical request JSON (`sort_keys=True`, separators `(",", ":")`). Each record
  stores the response, the reported model version, the usage and the measured seconds. The client
  reads the cache first, so re-running a finished run costs nothing. B uses the same pattern with its
  existing key. For each of B and C, the cache byte length and SHA-256 are recorded after its V run
  and before its H run; the audit checks that this prefix is unchanged at the end.
- **Credentials.** The client reads the environment first, then Windows Credential Manager. A key
  never appears in the repository, a log, the cache or a run row. Tests run with the credential store
  disabled and a fake provider, and CI never calls the API.
- **Data sent.** Only Banking77 ticket text (public, CC-BY-4.0, no personal data) and the intent
  names. The Cloudflare listing states zero data retention; TypeSafe's direct terms were not reviewed
  for this study.

## Calibration, cost and statistics

These are unchanged from v2 (`FINAL_EVAL.md`) except for the arms and the pairs.

- **Calibration.** Top-label isotonic regression per arm and variant, fitted on all of V and applied
  to H. Raw confidences are reported alongside.
- **Cost.** Routing decision cost per 1,000 tickets, excluding compute. Misroute-to-handoff ratios 2,
  5, 10 and 20, with a handoff costing 1.0. No currency. The break-even stays in handoff units, as in
  v2, and is never converted to dollars. C's actual dollars per 1,000 tickets, from reported usage,
  are reported separately.
- **Headline.** (A, isotonic) against (C-prob, isotonic) at 10:1. A is the comparator because it was
  v2's lower-cost arm. Difference is A minus C, so negative means A is lower. Paired bootstrap of
  per-ticket cost, 2,000 resamples, seed 20260830, 95% percentile interval, one shared resample matrix
  for every arm and every interval in this study.
- **Secondary pairs.** (B-logprob, isotonic) against (C-prob, isotonic), difference B minus C, which
  needs new pairing code; and (A, isotonic) against (B-logprob, isotonic), difference A minus B, which
  repeats v2's question on this partition. Ablations: C-vendor and B-verbal. No multiple-comparison
  correction; only H1 is the headline, and the report says so.
- **Hypotheses.** Status text is mechanical.
  - **H1 (cost).** A against C-prob at 10:1, isotonic: "A lower", "C lower" or "no clear difference",
    read from the headline interval.
  - **H2 (calibration after isotonic).** ECE of A against C-prob, isotonic, read the same way. ECE
    here is v2's definition: top-label, over all rows, abstentions included.
  - **H3 (vendor calibration claim).** The vendor publishes no calibration number, so 0.05 is this
    study's threshold, not a vendor figure. Raw ECE of C-prob over the H rows where C did not abstain,
    with a 95% interval from the shared resample matrix: "supported" if the upper bound is at most
    0.05, "not supported" if the lower bound is above 0.05, otherwise "inconclusive". ECE is biased
    upward in small bins and its percentile interval inherits that bias, which the report notes.
    C-vendor is read the same way, as an ablation. Accuracy by raw-confidence decile is reported for
    the vendor's qualitative claim that higher confidence means higher accuracy.
  - **H4 (vendor latency claim, 70 to 500 ms).** Client-side wall-clock of the successful attempt
    for each H ticket, stored in its cache record, so a resumed run keeps the original timings.
    Requests are sequential over one persistent HTTPS session, and the first request of each run is
    excluded as a cold start. Retries are counted separately. Status: "within claim" if p95 is at most
    500 ms, "median within, tail outside" if p50 is at most 500 ms and p95 is above it, and "outside
    claim" if p50 is above 500 ms. p50, p95, p99 and the share at or under 500 ms are reported. Before
    the C-H run, the median round-trip time of 20 unbilled requests to the same host (a models-list
    call where the route offers one, otherwise the TLS connection time) is recorded, and any
    server-side timing the response carries is stored. The measurement is from one machine in India;
    the vendor measured from the US West Coast. The report states both.
- **Other statistics.** Exact McNemar for each pair, with abstentions counted as wrong; reliability
  diagrams for every arm and variant.

## Gates, in order

- **G1, implementation freeze.** One commit with the partition script and `abc_heldout.csv` (hashing
  to the registered value), the arm C client, the A refit on F', the report and audit extensions, and
  fake-provider tests. Lint is clean and CI is green on that commit.
- **G2, access and probe.** A working key on one route; the probe passes with the committed client:
  all 77 names accepted, probabilities for all 77 summing to 1 within 1e-3, reported version
  `jev-1.13.0`. Self-agreement is measured. If this fails, or there is no working access by
  2026-10-28, C is reported as not evaluated and the study closes as "checked, not accessible".
- **G3, pre-run freeze.** One commit recording the chosen endpoint, the probe and self-agreement
  record, and the day's price check. No arm scores any row of H before this commit.
- **G4, run order.** A on V and H, then B on V and H, then C on V and then H. Each run finishes, or
  stops under the failure rules.
- **G5, completion.** A transport failure is a request that still fails after its last allowed
  attempt. It is scored `ABSTAIN` with confidence 0.0, flagged in the row, and counted separately from
  model abstentions, which are results. At most 1% of a run's tickets may be transport failures (20 for
  V, 20 for H). Above that the run is void and may be resumed once with the identical command, reusing
  every cached success.
- **G6, audit.** The extended audit script passes every check before the report is generated: H
  matches the frozen file, every arm covers the same tickets, run row indices are 0 to 2,001 with none
  missing or repeated, one endpoint and one reported version per C run, confidences lie in [0, 1],
  abstentions have confidence 0.0, and each cache prefix is unchanged. The audit computes no result
  statistic.
- **G7, report.** The report is generated on H with the disclaimer below. A claims sheet with the
  allowed wording is fixed before any README text is written.

## Pass rule

The study **passes** when G1 to G7 all pass and the report and claims sheet are committed, whatever the
direction of H1 to H4. C losing to A is a pass.

The study **closes without a result**, recorded as "checked, not evaluated", if G2 fails, if there is no
access by 2026-10-28, or if a run would exceed its cap or the total budget. Nothing is estimated for C in
that case.

## Failure and deviation rules

- **Interruption.** The identical command resumes the run, and completed responses are reused
  byte-for-byte from the append-only cache. The interruption is logged in the evaluation log below.
- **Defect found after H has been scored.** H counts as seen. There is no second held-out partition:
  the pool would have to be re-sampled, and that is not allowed. The defect, the fix and the re-run are
  reported as a deviation, not hidden.
- **Vendor change.** A new model version, a price change or a schema change during the study stops the
  run, as under "No silent switch".

## Reporting rules

- The report and every public text carry this sentence verbatim: "This is a new partition-based
  comparison on 2,002 Banking77 training rows held out for this study. Those rows were used to fit arm A
  in v2, so this is not an independent external validation of the shipped v2 model, and its numbers are
  not comparable with the v2 results on the official test split."
- The prior-exposure caveat from "Data and partition" is stated.
- Vendor figures are labelled as vendor claims. Latency is stated as measured from one machine.
- B's runtime difference from v2 (Ollama 0.40.2 against 0.34.4, same weights) is stated.
- Numbers in the README, interview notes and resume bullet come only from the claims sheet.

## Evaluation log

| Item | Record |
|---|---|
| Implementation commit (G1) and CI run | |
| Access route and date | |
| Probe (version, schema, median input tokens, p50 latency) | |
| Self-agreement (same choice n/20, largest probability difference) | |
| Price on the day of the run | |
| Pre-run commit (G3) | |
| Arm A finished | |
| Arm B finished (V and H cache prefix bytes and SHA-256) | |
| Arm C finished (calls, input tokens, dollars, transport failures; V cache prefix) | |
| Baseline round-trip time | |
| Audit result | |
| Total spent / remaining of $5.00 | |

## Amendments

None.
