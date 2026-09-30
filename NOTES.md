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
