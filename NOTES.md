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
