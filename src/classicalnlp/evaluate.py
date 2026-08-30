"""Evaluation that is allowed to disappoint you.

Three habits, each of which changes a conclusion:

1. **Macro-F1, not accuracy.** With unbalanced classes, accuracy is mostly a report on the
   largest class. Per-class F1 is printed alongside so a model that has quietly given up
   on the hard class cannot hide inside a good average.
2. **Fit inside the fold.** The vectorizer is part of the pipeline, so vocabulary and IDF
   come from the training fold only. :func:`leaky_vs_honest` measures what the usual
   shortcut is worth -- fitting the vectorizer on everything first -- because "it is a bit
   optimistic" is not an argument and a number is.
3. **A difference needs an interval.** Two models scoring 0.81 and 0.83 on one split is
   not evidence that one is better. :func:`paired_bootstrap` resamples the shared test
   predictions to put a confidence interval on the *difference*, and McNemar's test asks
   whether the disagreements are one-sided.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

RANDOM_STATE = 20260830


@dataclass
class Result:
    name: str
    macro_f1: float
    accuracy: float
    fold_scores: list[float] = field(default_factory=list)
    predictions: np.ndarray | None = None
    per_class_f1: dict[str, float] = field(default_factory=dict)

    @property
    def fold_std(self) -> float:
        return float(np.std(self.fold_scores)) if self.fold_scores else 0.0


def folds(y: Sequence[int], n_splits: int = 5) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def evaluate(
    name: str,
    pipeline: Pipeline,
    documents: Sequence[str],
    labels: Sequence[int],
    label_names: Sequence[str] | None = None,
    n_splits: int = 5,
) -> Result:
    """Cross-validated, leakage-free evaluation of one pipeline."""
    y = np.asarray(labels)
    splitter = folds(y, n_splits)

    predictions = cross_val_predict(clone(pipeline), list(documents), y, cv=splitter, n_jobs=1)

    fold_scores = []
    for _, test_index in splitter.split(np.zeros(len(y)), y):
        fold_scores.append(float(f1_score(y[test_index], predictions[test_index], average="macro")))

    per_class = {}
    if label_names is not None:
        scores = f1_score(y, predictions, average=None, labels=range(len(label_names)))
        per_class = {name_: float(score) for name_, score in zip(label_names, scores, strict=True)}

    return Result(
        name=name,
        macro_f1=float(f1_score(y, predictions, average="macro")),
        accuracy=float((predictions == y).mean()),
        fold_scores=fold_scores,
        predictions=predictions,
        per_class_f1=per_class,
    )


def leaky_vs_honest(
    documents: Sequence[str],
    labels: Sequence[int],
    pipeline: Pipeline,
    n_splits: int = 5,
) -> tuple[float, float]:
    """Return (leaky macro-F1, honest macro-F1) for the same pipeline and splits.

    The leaky version fits the TF-IDF stage on every document before cross-validating the
    classifier on the resulting matrix -- the shortcut that shows up in a lot of notebooks
    because it is faster and looks equivalent.
    """
    y = np.asarray(labels)
    splitter = folds(y, n_splits)

    vectorizer: TfidfVectorizer = clone(pipeline.named_steps["tfidf"])
    matrix = vectorizer.fit_transform(list(documents))  # <- sees the test folds
    tail = Pipeline(pipeline.steps[1:])
    leaky = cross_val_predict(clone(tail), matrix, y, cv=splitter, n_jobs=1)

    honest = cross_val_predict(clone(pipeline), list(documents), y, cv=splitter, n_jobs=1)

    return (
        float(f1_score(y, leaky, average="macro")),
        float(f1_score(y, honest, average="macro")),
    )


def leaky_supervised_selection(
    documents: Sequence[str],
    labels: Sequence[int],
    k: int = 2000,
    n_splits: int = 5,
) -> tuple[float, float]:
    """The leak that actually costs you: **supervised** feature selection outside the folds.

    Fitting a TF-IDF vectorizer on everything leaks very little, because the vectorizer
    never sees a label -- it only learns vocabulary and IDF. Chi-squared feature selection
    does see the labels, so choosing the top-k features on the full dataset hands the
    classifier a feature set that was picked using the test folds' answers.
    """
    from sklearn.feature_selection import SelectKBest, chi2

    y = np.asarray(labels)
    splitter = folds(y, n_splits)

    vectorizer = TfidfVectorizer(min_df=2, sublinear_tf=True)
    matrix = vectorizer.fit_transform(list(documents))
    selector = SelectKBest(chi2, k=min(k, matrix.shape[1]))
    selected = selector.fit_transform(matrix, y)  # <- uses every label, including test folds

    from .vectorizers import classifier

    leaky = cross_val_predict(classifier(), selected, y, cv=splitter, n_jobs=1)

    honest_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(min_df=2, sublinear_tf=True)),
            ("select", SelectKBest(chi2, k=k)),
            ("clf", classifier()),
        ]
    )
    honest = cross_val_predict(honest_pipeline, list(documents), y, cv=splitter, n_jobs=1)

    return (
        float(f1_score(y, leaky, average="macro")),
        float(f1_score(y, honest, average="macro")),
    )


def paired_bootstrap(
    y_true: Sequence[int],
    predictions_a: Sequence[int],
    predictions_b: Sequence[int],
    *,
    iterations: int = 2000,
    seed: int = RANDOM_STATE,
) -> dict[str, float]:
    """Bootstrap the macro-F1 difference between two models on the same test items.

    Paired, because both models predicted the same documents -- resampling them
    independently would throw away that pairing and widen the interval for no reason.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    a = np.asarray(predictions_a)
    b = np.asarray(predictions_b)
    size = len(y_true)

    differences = np.empty(iterations)
    for i in range(iterations):
        index = rng.integers(0, size, size)
        differences[i] = f1_score(y_true[index], a[index], average="macro") - f1_score(
            y_true[index], b[index], average="macro"
        )

    observed = f1_score(y_true, a, average="macro") - f1_score(y_true, b, average="macro")
    low, high = np.percentile(differences, [2.5, 97.5])
    # Two-sided: how often does the resampled difference land on the other side of zero?
    p_value = 2 * min((differences <= 0).mean(), (differences >= 0).mean())
    return {
        "difference": float(observed),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(min(1.0, p_value)),
    }


def mcnemar(
    y_true: Sequence[int], predictions_a: Sequence[int], predictions_b: Sequence[int]
) -> dict[str, float]:
    """Exact McNemar on the discordant pairs (the items exactly one model got right)."""
    from scipy import stats

    y_true = np.asarray(y_true)
    correct_a = np.asarray(predictions_a) == y_true
    correct_b = np.asarray(predictions_b) == y_true

    only_a = int((correct_a & ~correct_b).sum())
    only_b = int((~correct_a & correct_b).sum())
    total = only_a + only_b
    p_value = 1.0 if total == 0 else float(stats.binomtest(only_a, total, 0.5).pvalue)
    return {"only_a": only_a, "only_b": only_b, "p_value": p_value}


def report(result: Result, label_names: Sequence[str], y_true: Sequence[int]) -> str:
    lines = [
        f"{result.name}: macro-F1 {result.macro_f1:.4f} "
        f"(fold sd {result.fold_std:.4f})  accuracy {result.accuracy:.4f}",
        classification_report(y_true, result.predictions, target_names=list(label_names), digits=3),
        "confusion matrix (rows = true):",
        str(confusion_matrix(y_true, result.predictions)),
    ]
    return "\n".join(lines)
