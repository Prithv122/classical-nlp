# Classical NLP Baselines — G1

**Tier:** 1 · **Category:** G - NLP & AI engineering · **Wave:** 2

Root rules in `../GUIDELINES.md` apply. This file is project-specific only — keep it under 40 lines.

## What this is

TF-IDF (word and character n-gram) vs LSA vs sentence embeddings on 20 Newsgroups, with
Indic-aware text normalisation, NMF + c-TF-IDF topic modelling scored by NPMI coherence,
and an evaluation harness built to avoid the three usual ways these comparisons lie.

## Stack

Python 3.12 · scikit-learn · numpy · scipy · pytest · ruff · GitHub Actions.
`sentence-transformers` is an **optional extra** (`uv sync --extra embeddings`) — a 90 MB
model download has no place in the required dependency set for this project.

## Acceptance criteria

- [x] TF-IDF vs embeddings, compared with a significance test (CATALOG G1)
- [x] Multilingual/Indic text handling — normalisation, script detection, tokenisation
- [x] Topic modelling with a coherence number rather than eyeballed word lists
- [x] Correct evaluation: fit inside folds, macro-F1, per-class F1, leakage measured
- [ ] Ship gate passes (`/ship`)

## Project-specific notes

- **The `\w` trap.** Python's `re` excludes Unicode combining marks from `\w`, so a
  `[^\w\s]` punctuation strip silently deletes every Devanagari/Kannada vowel sign. This
  code matches on Unicode *general category* (L/N/M) instead. `tests/test_normalize.py`
  has the regression test — do not "simplify" it back to a regex.
- ZWNJ (U+200C) is kept, ZWSP/BOM are stripped. They are not interchangeable.
- Topic modelling is NMF + c-TF-IDF written out rather than BERTopic imported: the catalog
  names BERTopic, and the deviation is deliberate and documented in the README.
- 20 Newsgroups is loaded with `remove=('headers','footers','quotes')` — the headers name
  the newsgroup, so keeping them means training a label reader.
- Tests needing the download are marked `network`: `uv run pytest -m "not network"`.
