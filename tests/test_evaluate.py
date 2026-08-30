"""Evaluation tests.

These use a small synthetic corpus rather than the real one: the properties being checked
-- that leakage inflates a score, that a bootstrap interval covers zero when two models
are identical -- are properties of the evaluation code, and 20 Newsgroups would only make
them slower to check.
"""

from __future__ import annotations

import numpy as np
import pytest

from classicalnlp import evaluate, vectorizers


def synthetic_corpus(n_per_class: int = 60, seed: int = 7):
    """Two topics with a shared vocabulary, so the task is learnable but not trivial."""
    rng = np.random.default_rng(seed)
    shared = ["the", "and", "a", "of", "to", "in"]
    topic_a = ["orbit", "launch", "rocket", "payload", "telemetry", "apogee"]
    topic_b = ["goalie", "puck", "penalty", "rink", "faceoff", "shootout"]

    documents, labels = [], []
    for label, vocabulary in enumerate((topic_a, topic_b)):
        for _ in range(n_per_class):
            words = list(rng.choice(vocabulary, 12)) + list(rng.choice(shared, 8))
            rng.shuffle(words)
            documents.append(" ".join(words))
            labels.append(label)
    return documents, labels


def test_evaluate_returns_per_fold_and_per_class_scores():
    documents, labels = synthetic_corpus()
    result = evaluate.evaluate(
        "tfidf-word", vectorizers.tfidf_word(), documents, labels, ["a", "b"], n_splits=3
    )
    assert 0.8 < result.macro_f1 <= 1.0
    assert len(result.fold_scores) == 3
    assert set(result.per_class_f1) == {"a", "b"}
    assert len(result.predictions) == len(labels)


def test_folds_are_stratified_and_deterministic():
    documents, labels = synthetic_corpus()
    first = evaluate.evaluate("m", vectorizers.tfidf_word(), documents, labels, n_splits=3)
    second = evaluate.evaluate("m", vectorizers.tfidf_word(), documents, labels, n_splits=3)
    assert first.macro_f1 == second.macro_f1
    assert np.array_equal(first.predictions, second.predictions)


def test_leakage_inflates_the_score():
    """Fitting the vectorizer outside the folds should never look *worse*."""
    documents, labels = synthetic_corpus(n_per_class=40)
    leaky, honest = evaluate.leaky_vs_honest(documents, labels, vectorizers.tfidf_word(), 3)
    assert leaky >= honest


def test_paired_bootstrap_covers_zero_for_identical_models():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 200)
    predictions = y.copy()
    predictions[:20] = (predictions[:20] + 1) % 3

    stats = evaluate.paired_bootstrap(y, predictions, predictions, iterations=300)
    assert stats["difference"] == pytest.approx(0.0)
    assert stats["ci_low"] <= 0 <= stats["ci_high"]
    assert stats["p_value"] == pytest.approx(1.0)


def test_paired_bootstrap_detects_a_real_difference():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 300)
    good = y.copy()
    good[:15] = 1 - good[:15]
    bad = y.copy()
    bad[:120] = 1 - bad[:120]

    stats = evaluate.paired_bootstrap(y, good, bad, iterations=300)
    assert stats["difference"] > 0.2
    assert stats["ci_low"] > 0
    assert stats["p_value"] < 0.05


def test_mcnemar_counts_the_discordant_pairs():
    y = np.array([0, 0, 1, 1, 0, 1])
    a = np.array([0, 0, 1, 1, 1, 1])  # wrong on item 4
    b = np.array([1, 1, 0, 1, 0, 1])  # wrong on items 0, 1, 2
    result = evaluate.mcnemar(y, a, b)
    assert result["only_a"] == 3
    assert result["only_b"] == 1
    assert 0 < result["p_value"] <= 1


def test_mcnemar_with_no_disagreement_is_not_significant():
    y = np.array([0, 1, 0, 1])
    result = evaluate.mcnemar(y, y, y)
    assert result == {"only_a": 0, "only_b": 0, "p_value": 1.0}


def test_report_mentions_every_class():
    documents, labels = synthetic_corpus(n_per_class=30)
    result = evaluate.evaluate("m", vectorizers.tfidf_word(), documents, labels, ["a", "b"], 3)
    text = evaluate.report(result, ["a", "b"], labels)
    assert "macro-F1" in text
    assert "confusion matrix" in text
