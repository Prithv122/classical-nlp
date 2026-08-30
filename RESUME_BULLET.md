# Resume Bullets — Classical NLP Baselines

---

## Bullets

- Benchmarked three text representations (word TF-IDF, 3–5 character n-grams, TF-IDF+SVD)
  on a 2,082-document 20 Newsgroups task with leakage-free 5-fold pipelines, paired
  bootstrap confidence intervals and McNemar tests, showing the dense representation's
  +0.009 macro-F1 edge was not significant (95% CI [−0.005, +0.022], p = 0.22) — and
  quantifying that supervised feature selection fitted outside the folds inflates scores
  10× more than the commonly warned-about unsupervised vectorizer leak (+0.029 vs +0.001 at
  n = 400).
- Built an Indic-aware normalisation layer (NFC composition, Unicode-category tokenisation,
  digit folding, ZWNJ-preserving cleanup) after finding that Python's `\w` excludes
  combining marks and silently deletes every Devanagari vowel sign; raised NMF topic
  coherence from NPMI 0.213 to 0.361 by correcting the c-TF-IDF weighting, turning topic
  labels from "the, of, to" into "god, jesus, christ" / "space, nasa, launch". 38 tests,
  CI green.

## Which roles this supports

- [x] Data Scientist / ML
- [x] AI Engineer (LLM/NLP/CV)
- [ ] Data Engineer
- [x] Data Analyst / Python Developer

## Keywords this project earns

TF-IDF · character n-grams · LSA / TruncatedSVD · scikit-learn pipelines · stratified
cross-validation · macro-F1 and per-class evaluation · data leakage (supervised vs
unsupervised) · paired bootstrap · McNemar's test · NMF topic modelling · c-TF-IDF ·
NPMI coherence · Unicode normalisation (NFC/NFKC) · Devanagari/Kannada text handling ·
script detection · pytest · ruff

---

_Note to self: lead with the negative result. "The embeddings did not beat TF-IDF, and here
is the confidence interval" is a stronger signal than any accuracy number, because almost
nobody's portfolio contains a comparison that was allowed to come out that way._
