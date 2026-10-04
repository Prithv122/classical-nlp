# Classical NLP Baselines — G1

**Tier:** 1 · **Category:** G - NLP & AI engineering · **Wave:** 2

Root rules in `../GUIDELINES.md` apply. This file is project-specific only — keep it under 40 lines.

## What this is

Two studies. (1) TF-IDF vs LSA vs sentence embeddings on 20 Newsgroups, with Indic-aware text
normalisation and NMF + c-TF-IDF topics scored by NPMI. (2) A cost-priced router study on
Banking77: calibrated TF-IDF vs a calibrated local 7B LLM (Ollama), scored once on the held-out
test split under the protocol in `FINAL_EVAL.md`.

## Stack

Python 3.12 · scikit-learn · numpy · scipy · pytest · ruff · GitHub Actions. Optional extras:
`embeddings` (`sentence-transformers`) and `router` (matplotlib). Arm B needs a local Ollama server.

## Acceptance criteria

- [x] TF-IDF vs embeddings with a significance test; Indic text handling; topics with NPMI
- [x] Correct evaluation: fit inside folds, macro-F1, per-class F1, leakage measured
- [x] Router study: costed decisions, calibration, paired bootstrap, held-out test run once
- [x] Ship gate passes

## Project-specific notes

- **The `\w` trap.** `re` excludes combining marks from `\w`; match on Unicode general
  category (L/N/M). `tests/test_normalize.py` guards it — do not "simplify" it back to a regex.
- ZWNJ (U+200C) is kept, ZWSP/BOM are stripped. 20 Newsgroups loads with headers removed.
- Topics are NMF + c-TF-IDF written out, not BERTopic imported — deliberate, see the README.
- **The test split is spent.** `results/final/` is frozen evidence (hashes in `SHA256SUMS`,
  stored byte-for-byte via `.gitattributes`). Never re-run or overwrite it; a new router
  experiment needs a new protocol. Result wording comes only from `results/final/CLAIMS.md`.
- Tests needing the download are marked `network`: `uv run pytest -m "not network"`.
