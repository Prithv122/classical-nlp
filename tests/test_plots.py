"""Reliability bins and the diagram. Hand-made data; the numbers test the code."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from classicalnlp import calibrate, plots

PNG = b"\x89PNG\r\n\x1a\n"

Y_TRUE = np.array([0, 1, 2, 3, 4, 5])
Y_PRED = np.array([0, 1, 0, 3, -1, 5])
CONFIDENCE = np.array([0.95, 0.85, 0.75, 0.55, 0.0, 0.15])


def _panels(arms):
    return [
        (f"{arm} {cal}", Y_TRUE, Y_PRED, CONFIDENCE) for arm in arms for cal in ("raw", "isotonic")
    ]


def test_reliability_bins_match_calibrate_with_an_abstention():
    got = plots.reliability_bins(Y_TRUE, Y_PRED, CONFIDENCE)
    want = calibrate.expected_calibration_error(
        calibrate.top_label_correct(Y_TRUE, Y_PRED), CONFIDENCE
    )
    assert got[0] == want[0]
    for g, w in zip(got[1:], want[1:], strict=True):
        np.testing.assert_array_equal(g, w)
    assert got[3].sum() == 6


@pytest.mark.parametrize("arms", [["A"], ["A", "B-logprob", "B-verbal"]])
def test_save_reliability_writes_a_png(tmp_path, arms):
    path = tmp_path / "new" / "dir" / "reliability.png"
    plots.save_reliability(_panels(arms), path)
    data = path.read_bytes()
    assert data[:8] == PNG
    assert len(data) > 1000


def test_save_reliability_accepts_a_source_note(tmp_path):
    path = tmp_path / "r.png"
    plots.save_reliability(_panels(["A"]), path, "synthetic fixture: tests the code")
    assert path.read_bytes()[:8] == PNG
