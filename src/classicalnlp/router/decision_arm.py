"""Arm C: a dedicated decision model. Not available in this study.

The arm exists so that the router registry, the command line and the report all name it, and
so a request for it fails loudly. It is reported as not evaluated and is never estimated.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..data import Corpus

_MESSAGE = "arm C is not available: it is reported as not evaluated, never estimated"


class NotAvailable(ValueError):
    """Raised when an arm that has no implementation is asked to run."""


class DecisionArm:
    name = "C"
    description = "dedicated decision model (not available)"

    def fit(self, corpus: Corpus) -> DecisionArm:
        raise NotAvailable(_MESSAGE)

    def predict_top(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        raise NotAvailable(_MESSAGE)
