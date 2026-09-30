"""Calibration tests on hand-made cases whose answers can be worked out on paper."""

from __future__ import annotations

import numpy as np
import pytest

from classicalnlp import calibrate


def test_ece_is_zero_when_confidence_matches_hit_rate():
    y_true = np.array([1, 1, 1, 0])
    assert calibrate.expected_calibration_error(y_true, np.full(4, 0.75))[0] == pytest.approx(0.0)


def test_ece_of_overconfident_predictions():
    y_true = np.array([1, 1, 0, 0])
    assert calibrate.expected_calibration_error(y_true, np.full(4, 1.0))[0] == pytest.approx(0.5)


def test_ece_weights_bins_by_count():
    y_true = np.array([1, 0, 0, 0, 1, 1, 0, 0])
    prob = np.array([0.25] * 4 + [0.9] * 4)
    assert calibrate.expected_calibration_error(y_true, prob, n_bins=10)[0] == pytest.approx(0.2)


def test_top_label_ece_matches_ece_on_the_correctness_vector():
    y_true = np.array([0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 1, 1, 2, 0, 0])
    confidence = np.array([0.9, 0.8, 0.6, 0.7, 0.5, 0.95])
    correct = np.array([1, 1, 0, 1, 0, 1], dtype=float)
    assert calibrate.top_label_correct(y_true, y_pred).tolist() == correct.tolist()
    expected = calibrate.expected_calibration_error(correct, confidence)[0]
    assert calibrate.top_label_ece(y_true, y_pred, confidence) == pytest.approx(expected)


def test_brier_score():
    assert calibrate.brier_score([1, 0], [0.75, 0.25]) == pytest.approx(0.0625)


def test_top_label_isotonic_is_bounded_and_monotone():
    rng = np.random.default_rng(0)
    confidence = rng.uniform(0.2, 1.0, 200)
    y_true = rng.integers(0, 3, 200)
    y_pred = np.where(rng.uniform(size=200) < confidence, y_true, (y_true + 1) % 3)
    calibrator = calibrate.fit_top_label_isotonic(y_true, y_pred, confidence)
    out = calibrate.apply_isotonic(calibrator, np.linspace(0.0, 1.0, 101))
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert np.all(np.diff(out) >= 0.0)
