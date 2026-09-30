"""What a router is in this study.

A router reads a ticket and returns its top intent with a confidence in [0, 1]; the cost
rule in :mod:`classicalnlp.costs` then decides whether to auto-route or hand off. The
method is ``predict_top``, not ``predict_proba``: scikit-learn's ``predict_proba`` returns
a matrix, and one name for two shapes invites mistakes.

A ticket the router could not classify gets the label :data:`ABSTAIN` and confidence 0.0,
so the cost rule always hands it off and it can never count as correct.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from ..data import Corpus

ABSTAIN = -1


@runtime_checkable
class Router(Protocol):
    name: str
    description: str

    def fit(self, corpus: Corpus) -> Router: ...

    def predict_top(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """``(labels, confidence)``: one class id and one confidence per text."""
        ...


def check_output(labels, confidence, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Validate a router's output and return it as int and float arrays."""
    labels = np.asarray(labels, dtype=int)
    confidence = np.asarray(confidence, dtype=float)
    if len(labels) != n or len(confidence) != n:
        raise ValueError(
            f"length mismatch: expected {n}, got {len(labels)} labels, "
            f"{len(confidence)} confidences"
        )
    if np.any(np.isnan(confidence)):
        raise ValueError("confidence contains nan")
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError("confidence must lie in [0, 1]")
    if np.any((labels == ABSTAIN) & (confidence != 0.0)):
        raise ValueError("an ABSTAIN row must have confidence 0.0")
    return labels, confidence
