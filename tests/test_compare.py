"""Paired comparison statistics. Hand-made arrays only; the values test the code."""

from __future__ import annotations

import numpy as np
import pytest

from classicalnlp import calibrate, compare, costs
from classicalnlp.costs import RoutingCost

Y_TRUE = np.array([0, 1, 2, 3, 4])
Y_PRED = np.array([0, 0, 2, 3, -1])
CONFIDENCE = np.array([0.95, 0.95, 0.91, 0.5, 0.0])


def test_ticket_costs_by_hand_at_ratio_10():
    cost = RoutingCost.from_ratio(10)
    out = compare.ticket_costs(Y_TRUE, Y_PRED, CONFIDENCE, cost)
    assert out.tolist() == [0.0, 10.0, 0.0, 1.0, 1.0]
    assert out.mean() * 1000 == 2400.0
    assert out.mean() * 1000 == costs.expected_cost(Y_TRUE, Y_PRED, CONFIDENCE, cost).cost_per_1000


def test_ticket_costs_by_hand_at_ratio_2():
    cost = RoutingCost.from_ratio(2)
    out = compare.ticket_costs(Y_TRUE, Y_PRED, CONFIDENCE, cost)
    assert out.tolist() == [0.0, 2.0, 0.0, 1.0, 1.0]
    assert out.mean() * 1000 == 800.0
    assert out.mean() * 1000 == costs.expected_cost(Y_TRUE, Y_PRED, CONFIDENCE, cost).cost_per_1000


def test_ticket_costs_rejects_inference_cost_and_length_mismatch():
    with pytest.raises(ValueError, match="routing cost only"):
        compare.ticket_costs(Y_TRUE, Y_PRED, CONFIDENCE, RoutingCost(10.0, 1.0, 0.1))
    with pytest.raises(ValueError, match="length mismatch"):
        compare.ticket_costs(Y_TRUE, Y_PRED[:4], CONFIDENCE, RoutingCost.from_ratio(10))


def test_paired_bootstrap_cost_constant_difference():
    out = compare.paired_bootstrap_cost(np.full(10, 1.5), np.full(10, 1.0))
    assert out == {"difference": 500.0, "ci_low": 500.0, "ci_high": 500.0, "p_value": 0.0}


def test_paired_bootstrap_cost_identical_arrays():
    same = np.full(10, 1.5)
    out = compare.paired_bootstrap_cost(same, same)
    assert out == {"difference": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}


def test_paired_bootstrap_cost_rejects_bad_shapes():
    with pytest.raises(ValueError, match="length mismatch"):
        compare.paired_bootstrap_cost(np.ones(3), np.ones(4))
    with pytest.raises(ValueError, match="zero tickets"):
        compare.paired_bootstrap_cost(np.ones(0), np.ones(0))


def test_paired_bootstrap_cost_on_random_data():
    rng = np.random.default_rng(0)
    a, b = rng.random(200) * 10, rng.random(200) * 10
    out = compare.paired_bootstrap_cost(a, b)
    assert out["difference"] == np.mean(a - b) * 1000
    assert out["ci_low"] <= out["difference"] <= out["ci_high"]
    assert compare.paired_bootstrap_cost(a, b) == out
    other = compare.paired_bootstrap_cost(a, b, seed=1)
    assert (other["ci_low"], other["ci_high"]) != (out["ci_low"], out["ci_high"])
    swapped = compare.paired_bootstrap_cost(b, a)
    assert swapped["difference"] == pytest.approx(-out["difference"], abs=1e-9)
    assert swapped["ci_low"] == pytest.approx(-out["ci_high"], abs=1e-9)
    assert swapped["ci_high"] == pytest.approx(-out["ci_low"], abs=1e-9)


def _ece_data(seed=0, n=200):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 5, n)
    pred_a = np.where(rng.random(n) < 0.7, y_true, (y_true + 1) % 5)
    pred_b = np.where(rng.random(n) < 0.5, y_true, (y_true + 1) % 5)
    return y_true, pred_a, rng.random(n), pred_b, rng.random(n)


def test_paired_bootstrap_ece_identical_arms():
    y_true, pred, conf, _, _ = _ece_data()
    out = compare.paired_bootstrap_ece(y_true, pred, conf, pred, conf, iterations=200)
    assert (out["difference"], out["ci_low"], out["ci_high"]) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_ece_on_random_data():
    y_true, pred_a, conf_a, pred_b, conf_b = _ece_data()
    out = compare.paired_bootstrap_ece(y_true, pred_a, conf_a, pred_b, conf_b, iterations=300)
    expected = calibrate.top_label_ece(y_true, pred_a, conf_a) - calibrate.top_label_ece(
        y_true, pred_b, conf_b
    )
    assert out["difference"] == pytest.approx(expected, abs=1e-12)
    assert out["ci_low"] <= out["difference"] <= out["ci_high"]


def test_breakeven_by_hand():
    a = np.array([0.0, 10.0, 0.0, 1.0, 1.0])
    b = np.ones(5)
    secs = np.full(5, 2.0)
    out = compare.breakeven(a, b, secs)
    assert out["advantage_per_ticket"] == pytest.approx(1.4)
    assert out["breakeven_per_ticket"] == pytest.approx(1.4)
    assert out["breakeven_per_second"] == pytest.approx(0.7)
    assert out["status"] == "B cheaper before compute"

    swapped = compare.breakeven(b, a, secs)
    assert swapped["advantage_per_ticket"] == pytest.approx(-1.4)
    assert swapped["breakeven_per_ticket"] is None
    assert swapped["breakeven_per_second"] is None
    assert swapped["status"] == "A cheaper before compute"

    assert compare.breakeven(a, b, None)["breakeven_per_second"] is None
    assert compare.breakeven(b, b, secs)["status"] == "equal before compute"


@pytest.mark.parametrize(
    ("low", "high", "status"),
    [
        (-5, -1, "A lower"),
        (1, 5, "B lower"),
        (-1, 1, "no clear difference"),
        (-1, 0, "no clear difference"),
        (0, 1, "no clear difference"),
    ],
)
def test_interval_status(low, high, status):
    assert compare.interval_status(low, high) == status
