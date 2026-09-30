"""Arm A: TF-IDF (word) + logistic regression."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .. import vectorizers
from ..data import Corpus
from .base import check_output


class TfidfArm:
    name = "A"
    description = "TF-IDF (word) + logistic regression"

    def __init__(self) -> None:
        self._pipeline = vectorizers.build("tfidf-word")
        self._fitted = False

    def fit(self, corpus: Corpus) -> TfidfArm:
        self._pipeline.fit(corpus.documents, corpus.labels)
        self._fitted = True
        return self

    def predict_top(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise ValueError("TfidfArm is not fitted; call fit() first")
        if len(texts) == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)
        proba = self._pipeline.predict_proba(list(texts))
        top = proba.argmax(axis=1)
        # Class ids, not column indices: the ids need not be 0..k-1.
        labels = self._pipeline.classes_[top]
        confidence = np.clip(proba[np.arange(len(top)), top], 0.0, 1.0)
        return check_output(labels, confidence, len(texts))
