"""Router contract and arm A. Fixture or tiny hand-made corpora only; no network."""

from __future__ import annotations

import numpy as np
import pytest

from classicalnlp import tickets
from classicalnlp.data import Corpus
from classicalnlp.router import (
    ABSTAIN,
    ARMS,
    DecisionArm,
    LLMArm,
    Router,
    TfidfArm,
    check_output,
)


def _corpus(documents, labels):
    return Corpus(documents=documents, labels=labels, label_names=list(tickets.BANKING77_INTENTS))


def test_arm_satisfies_the_contract_and_is_registered():
    assert isinstance(TfidfArm(), Router)
    assert TfidfArm.name == "A"
    assert set(ARMS) == {"A", "B", "C"}
    assert ARMS["A"] is TfidfArm and ARMS["B"] is LLMArm and ARMS["C"] is DecisionArm


def test_predict_top_on_the_fixture():
    fit, val = tickets.validation_split(tickets.load_fixture())
    labels, confidence = TfidfArm().fit(fit).predict_top(val.documents)
    assert len(val) == 12
    assert labels.shape == confidence.shape == (12,)
    assert np.issubdtype(labels.dtype, np.integer)
    assert set(labels) <= set(fit.labels)
    assert np.all((confidence >= 0.0) & (confidence <= 1.0))


def test_labels_are_class_ids_not_column_indices():
    texts = [
        "where is my card post",
        "card delivery post",
        "where is my card",
        "card post delivery",
        "exchange rate euro",
        "euro rate today",
        "exchange rate today",
        "euro exchange rate",
    ]
    corpus = _corpus(texts, [3] * 4 + [40] * 4)
    labels, _ = TfidfArm().fit(corpus).predict_top(texts)
    assert set(labels) == {3, 40}


def test_predict_top_before_fit_raises():
    with pytest.raises(ValueError, match="not fitted"):
        TfidfArm().predict_top(["hello"])


def test_predict_top_of_nothing_is_empty():
    fit, _ = tickets.validation_split(tickets.load_fixture())
    labels, confidence = TfidfArm().fit(fit).predict_top([])
    assert len(labels) == 0
    assert len(confidence) == 0


@pytest.mark.parametrize(
    ("labels", "confidence", "n"),
    [
        ([0, 1], [0.5, 1.5], 2),
        ([0, 1], [0.5, -0.1], 2),
        ([0, 1], [0.5, float("nan")], 2),
        ([0, 1], [0.5, 0.5], 3),
        ([0, 1], [0.5], 2),
        ([0, ABSTAIN], [0.5, 0.3], 2),
    ],
)
def test_check_output_rejects_bad_output(labels, confidence, n):
    with pytest.raises(ValueError):
        check_output(labels, confidence, n)


def test_check_output_accepts_abstain_at_zero():
    labels, confidence = check_output([0, ABSTAIN], [0.5, 0.0], 2)
    assert labels.tolist() == [0, ABSTAIN]
    assert confidence.tolist() == [0.5, 0.0]


def test_fresh_arms_on_the_same_split_agree():
    fit, val = tickets.validation_split(tickets.load_fixture())
    first = TfidfArm().fit(fit).predict_top(val.documents)
    second = TfidfArm().fit(fit).predict_top(val.documents)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
