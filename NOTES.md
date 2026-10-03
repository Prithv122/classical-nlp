# Build Notes — Classical NLP Baselines

---

## Log

### 2026-08-30 — build

- **The bug this project is really about.** My first punctuation strip was the obvious
  `re.sub(r"[^\w\s]", " ", text)`. It passes every English test and silently destroys
  Devanagari: Python's `\w` matches Unicode letters and digits but **not combining marks**
  (category Mn/Mc), so every matra and virama is treated as punctuation. `क़िला` came out
  as `क ल`, and `यह एक परीक्षण है और यह अच्छा है` tokenised to
  `['क', 'षण', 'ह', 'अच', 'छ', 'ह']`. Nothing raised. The output was still text, the
  pipeline still trained, and the score would just have been mysteriously bad.

  Fixed by classifying characters with `unicodedata.category()` and keeping L, N and M,
  rather than by adding the `regex` package for `\p{M}`. There is a regression test, and
  a note in `GUIDELINES.md` telling future-me not to "simplify" it back.

- **ZWNJ is not junk.** The standard advice is to strip zero-width characters. ZWSP
  (U+200B), the BOM and soft hyphens are indeed junk. ZWNJ (U+200C) is not — in Devanagari
  it is what keeps a half-form from joining into a conjunct, so removing it changes the
  word. The module removes one set and keeps the other, deliberately.

- **NFC, not NFKC.** `क़` exists as a precomposed codepoint and as `क` + nukta. They render
  identically and hash differently, which quietly doubles vocabulary entries. NFC merges
  them. NFKC would also rewrite characters that merely look similar, which is a different
  and lossier decision.

- **Evaluation choices that changed the story.** Accuracy on the four-newsgroup task looks
  respectable while `talk.religion.misc` sits at F1 ≈ 0.2 — it overlaps heavily with
  `soc.religion.christian` and every model mostly gives up on it. Macro-F1 and the
  per-class table make that visible; accuracy hides it. Also loaded 20 Newsgroups with
  `remove=('headers','footers','quotes')`, because the headers contain the newsgroup name
  and a classifier trained on the raw version is a label reader.

- **Two significance tests, and they disagreed — which is the interesting part.** On the
  800-document subset, tfidf-char beat lsa by +0.019 macro-F1 with a bootstrap 95% CI of
  [-0.011, +0.050] (p = 0.22), while McNemar said tfidf-char was alone correct on 63 items
  against 30 (p = 0.0008). Not a contradiction: McNemar tests per-item correctness, where
  the win is real and consistent; the bootstrap tests *macro*-F1, which is dominated by the
  small hard class where both models are noisy. Reporting only the one that agrees with you
  is how model comparisons go wrong.

- **BERTopic, deliberately not used.** The catalogue entry names it; I implemented the part
  that matters instead — NMF for the soft clustering, c-TF-IDF for the topic labels, NPMI
  coherence for the score. This is a Tier 1 project whose stated purpose is "fundamentals
  below the API layer", and `pip install bertopic` would have replaced all three with one
  call. Noted as an open question rather than pretending it was not in the brief.

- **sentence-transformers stayed optional.** 90 MB of weights and a torch dependency for a
  comparison that is mostly about the cheap methods. It is an extra
  (`uv sync --extra embeddings`), the wrapper imports lazily, and the error message when it
  is missing tells you the exact command.

### 2026-09-30 — v2 router study, T1: data, calibration and cost layer

- **Built.** `tickets.py` (Banking77 loader, stratified seeded 20% validation split of the
  train split, synthetic fixture, checksum-verified download), `costs.py` (the routing cost
  rule and expected-cost accounting), top-label calibration helpers appended to
  `calibrate.py`, and two CLI subcommands, `tickets-download` and `tickets-summary`. Tests
  for all of it; none touch the network.
- **Vendored file.** `calibrate.py` comes from fraud-calibrated @ ebf6770,
  `src/fraud_calibrated/calibration.py`, MIT. The code is unchanged; only the module
  docstring and the `fit_isotonic` docstring were adapted, and the top-label helpers at the
  bottom are new here.
- **Banking77.** PolyAI, 13,083 queries, 77 intents, official split 10,003 train / 3,080
  test. Licence CC-BY-4.0; cite Casanueva et al. 2020, arXiv:2003.04807. The download uses
  the upstream CSVs pinned to commit `57ec275d8078af65b7731c2a98be812d844a6d6b`, with
  SHA-256 `b06e26ac...c664b` (train) and `d12d6e3b...eb474d` (test) enforced, rather than
  Hugging Face: that repository holds only a loading script and no data files, and current
  `datasets` refuses to run scripts.
- **Fixture.** `fixtures/tickets_sample.csv` is 60 hand-written synthetic rows (10 each for
  6 intents). It exists to test the code and is not Banking77 data.
- **Not yet measured.** No router arm exists yet and no Banking77 number of any kind has
  been computed. The 10:1 misroute:handoff ratio is an assumption, not a measurement.

### 2026-09-30 — v2 router study, T2a: router contract, TF-IDF arm, calibrated study runner

- **Built.** The `router/` package: the `Router` protocol (`predict_top(texts)` returns
  `(labels, confidence)`, deliberately not `predict_proba`), `ABSTAIN = -1` with confidence
  always 0.0, `check_output` that every arm returns through, and arm A (TF-IDF word +
  logistic regression, inference cost 0). `study.run_arm` fits an arm, calibrates its
  top-label confidence and prices the decisions at every ratio in `SENSITIVITY_RATIOS`.
  New `route --arm A --split {val,test}` subcommand, with `--fixture` or `--data-dir`.
- **Calibration design.** Calibration is a study step applied identically to every arm, not
  part of any arm; each run reports a `raw` and an `isotonic` row. On the validation split
  the isotonic confidence is 5-fold cross-fitted, so no row is scored by a calibrator that
  saw it. On the test split the calibrator is fitted once on all validation rows. Abstained
  rows are excluded from every calibrator fit and keep confidence 0.0, so they are always
  handed off.
- **Fit split.** Arm A is fitted on the 80% fit part of the official train split and is not
  refitted on the full train split, so the calibrator calibrates the same model it was
  fitted against.
- **Fixture behaviour.** On the 60-row fixture the raw arm A confidences sit below every
  threshold in the sweep, so the raw row hands everything off. This is about the code path
  on a synthetic set, not a result.
- **Not yet measured.** No Banking77 number of any kind has been computed and no LLM arm
  exists yet.

### 2026-09-30 — v2 router study, T2b: LLM arm, Ollama client, response cache, usage accounting

- **Arm B.** `LLMArm` scores each ticket with one model response that feeds two variants,
  `B-logprob` and `B-verbal`. The prompt lists the 77 intents first and puts the ticket last,
  so consecutive prompts share the whole prefix and Ollama can reuse it. The reply must be
  two lines, `intent: <name>` then `confidence: <0..1>`; the intent must match a label name
  exactly, case-sensitive. Anything else abstains: no repair, no fuzzy matching, no retries.
- **Confidence.** `B-logprob` is `exp` of the summed log-probs of the tokens spelling the
  chosen intent, clipped to [0, 1]. `B-verbal` is the stated number, used only if it is a
  plain decimal in [0, 1]. Abstention is independent per variant: an unusable intent abstains
  both, an unusable stated number abstains only `B-verbal`, unusable log-probs only
  `B-logprob`.
- **Ollama client.** `router/ollama.py`, standard library only, one non-streaming request
  per prompt with log-probs. It records the reply text, the per-token log-probs,
  `prompt_eval_count` and `prompt_eval_cached_count` raw, `eval_count`, and model-side seconds
  as `(total_duration - load_duration) / 1e9`. A missing field raises a named error; nothing
  is invented.
- **Response cache.** JSONL keyed by sha256 of the provider identity plus the prompt, where
  the identity covers the model and the generation settings. One line is written after every
  response, so an interrupted run resumes from what it already has; a half-written last line
  is skipped and counted on load. `route --arm B` requires `--cache`.
- **Test double.** `FakeLLM` and `fake_completion` exist so everything is tested without a
  model. Their numbers test the code and are not model behaviour.
- **Usage.** `Usage` carries per-ticket seconds and token counts for every arm; arm A, which
  reports none, is timed per ticket. `route` prints latency and tokens beside the table, and
  `--save` writes one JSON line per run and ticket (it overwrites). `--limit N` is a smoke aid
  on the validation split only and labels its output as not a result.
- **Ratio guard.** `run_arm` rejects cost ratios below 1 before the arm is fitted or called.
  Below 1 the cost rule auto-routes every ticket, including a confidence-0.0 abstention, and
  the study would then count it as a misroute. At ratio 1 the rule still hands off 0.0.
- **API behaviour observed on the owner's machine on 2026-09-30, Ollama 0.34.4.** Log-probs
  are returned without `top_logprobs`. The token strings concatenate to the response.
  `prompt_eval_count` is the full prompt and `prompt_eval_cached_count` the reused part. Load
  time is inside `total_duration` and is excluded from latency.
- **Limits.** Inference cost is 0 in the cost rows: there is no price assumption yet, and
  latency and tokens are reported beside the table instead. The `B-logprob` confidence is the
  probability of the intent string, not normalised over the 77 intents.
- **Not yet measured.** No real-model call has been made, no Banking77 number of any kind has
  been computed, and the prompt has not been tried against the real model.

### 2026-10-04 — correction: the routing rule's boundary, and what it changed

- **The bug.** The rule is documented as strict: a ticket whose expected misroute cost equals
  the handoff cost is handed off. In floating point `1 - 0.9` is not `0.1`, so `(1 - c) * m < h`
  auto-routed a confidence of exactly 0.9 at 10:1 and exactly 0.8 at 5:1, while handing off 0.95
  at 20:1. The old tests used only dyadic values (0.75 at 4:1), where the arithmetic is exact,
  and so never saw it. It matters for `B-verbal`, which states coarse values (0.8, 0.9, 0.95,
  1.0) that sit on those thresholds.
- **The fix.** `costs.auto_route` now treats a confidence within a relative `1e-9` of the
  threshold as on it, and so handed off (`BOUNDARY_TOLERANCE`). The decision rule, the
  thresholds and the headline ratio are unchanged: the code now does what the rule already
  said. New tests cover 0.5@2:1, 0.8@5:1, 0.9@10:1, 0.95@20:1 and 0.99@100:1 at the boundary
  and 1e-6 either side of it; the 0.8@5:1 and 0.9@10:1 cases fail on the previous code.
- **Nothing was re-run.** The model was not called again. Predicted labels, `B-logprob` and
  `B-verbal` confidences, calibration, the prompt, the cache and the saved per-ticket files are
  unchanged. Only the conversion from a recorded confidence to route or hand off was corrected,
  and the validation rows were re-derived from the saved per-ticket files.
- **Impact on the validation split (a development-time audit, not a result).** Four of the 24
  arm/calibration/ratio rows changed; the other 20 are identical.

  | Row | Auto-routed before → after | Cost per 1,000 before → after |
  |---|---|---|
  | A isotonic, 5:1 | 1,619 → 1,618 | 460.8 → 458.8 |
  | B-logprob isotonic, 5:1 | 429 → 424 | 898.1 → 895.6 |
  | B-verbal raw, 5:1 | 1,897 → 1,636 | 1871.1 → 1504.2 |
  | B-verbal raw, 10:1 | 1,636 → 395 | 2826.1 → 1297.4 |

  The 10:1 headline rows, A isotonic and B-logprob isotonic, are unchanged. This was checked by
  recomputing both from the saved files, not assumed. B-verbal raw is the row materially
  affected: its 0.9 and 0.8 answers had been auto-routed at the threshold.
- **Status.** At the T2b merge the "Not yet measured" line above was true. Since then the
  30-ticket smoke and the full validation run (2,001 tickets, prompt frozen before it started)
  have been made, on the validation split only. The official test split has not been touched.

### 2026-10-03 — v2 router study, T3: route-report, paired comparisons, break-even compute sensitivity, arm-C stub

- **Built.** `classical-nlp route-report --runs PATH [PATH ...] --out DIR` reads the per-ticket
  files written by `route --save` and writes `sensitivity.csv`, `usage.csv`, `abstentions.csv`,
  `comparison_cost.csv`, `comparison_accuracy.csv`, `comparison_ece.csv`, `breakeven.csv` (the
  comparison files and the break-even only when arm A and a B variant are both present), an
  optional `reliability.png` and `report.md`. Options: `--source-note`, `--iterations` (at least
  100), `--no-plots`, `--final`. Every number is recomputed from the saved rows with
  `costs.expected_cost` and `calibrate`. `compare.py` holds the per-ticket cost, the paired
  bootstrap for cost and for ECE, the break-even and the interval reading; `plots.py` draws the
  reliability grid; `report.py` loads, validates and writes. matplotlib is the optional `router`
  extra (`uv sync --extra router`), imported inside the drawing function only, and CI now
  installs it so the plot tests run.
- **Reporting decisions.** Cost is the routing decision cost per 1,000 tickets, excluding
  compute, in handoff units (one human handoff = 1.0); no currency, per-token or per-second
  price appears anywhere. Pairs are A against each B variant within the same calibration; the
  headline is (A, isotonic) against (B-logprob, isotonic) at the 10:1 ratio and exactly one row
  carries it. Differences are A minus B, so negative means A is lower. The bootstrap is paired:
  one set of resample indices serves both arms, with a fixed seed, 95% percentile interval and a
  two-sided p-value taken from the resampled differences. McNemar is exact and counts an
  abstention as wrong. ECE is top-label over all rows, abstentions included. Break-even:
  advantage per ticket is the mean of (cost A - cost B); the break-even per ticket is that value
  when it is positive and blank otherwise; per second it is divided by B's mean model-side
  seconds. H1 is read from the headline cost interval and H2 from the headline ECE interval,
  stated mechanically ("A lower", "B lower" or "no clear difference") with no commentary and no
  winner picked. The ticket did not define H1 and H2; this is the reading chosen here.
- **Guards.** A file on the official test split is refused unless `--final` is passed. The
  loader refuses bad JSON, missing keys, unknown arms, out-of-range confidence, an abstention
  with confidence above 0, non-dense indices, a run present twice, two splits at once,
  calibrations that disagree on predictions, and arms that do not share the same tickets. The
  report states "Arm C (dedicated decision model): not evaluated (not available)."
- **Limits.** Isotonic calibrators are not refitted inside a bootstrap resample, and the
  cross-fitted confidences are treated as fixed. Arm A's own compute is not subtracted in the
  break-even. The two latency clocks differ (A is wall-clock per ticket in-process; B is the
  model-side time Ollama reports, load excluded), so they are not directly comparable. CSV floats
  are written with ten significant digits. The README is unchanged; `report.md` stands in for the
  placeholder results skeleton.
- **Not built.** The arm-C interface and stub were not built: the size cap for this ticket was
  reached once `route-report` was complete. There is no `DecisionArm`, no `NotAvailable`, no
  `"C"` entry in `ARMS`, no test for them, and no change to `tests/test_router.py`. Until that
  commit lands, `route --arm C` fails at argument parsing as an invalid choice, not with a
  "not available" message.
- **Not yet measured.** `route-report` has not been run on real per-ticket files, no comparison
  has been computed on Banking77, and the official test split has not been touched.

---

## Rejected approaches

| Approach | Why rejected |
|---|---|
| `re` with `[^\w\s]` for punctuation | Deletes Devanagari/Kannada combining marks. The whole reason this project exists |
| The `regex` package for `\p{L}\p{M}\p{N}` | Correct, but a dependency to do what `unicodedata.category` already does in four lines |
| NFKC normalisation | Over-normalises — folds characters that merely look alike |
| Stripping all zero-width characters | ZWNJ is semantic in Devanagari |
| A borrowed 500-word Hindi stoplist | Imports someone else's judgement about a language this project handles at the surface. Wrote a short, honest one instead |
| Reporting accuracy | Hides that one class is at F1 0.2 |
| Fitting the vectorizer once, then cross-validating the classifier | Leaks test-fold vocabulary and IDF. Measured what it is worth rather than just asserting it |
| BERTopic | See above — the mechanism is the point here |

## Open questions

- [ ] A real Indic **classification** benchmark. The multilingual samples exercise the text-handling path but no accuracy number is computed from them, and a dozen sentences cannot support one. IndicNLP-Suite or a Kannada news corpus would give the project a second, genuinely multilingual results table.
- [ ] BERTopic as an optional backend alongside the hand-written NMF + c-TF-IDF path, so the two can be compared on the same coherence metric.
- [ ] Character n-grams win here; is that a morphology effect or a robustness-to-typos effect? Splitting that apart needs a corpus with controlled noise.
- [ ] Coherence is measured on the same corpus the topics were fitted on. An external reference corpus would be the stricter test.
