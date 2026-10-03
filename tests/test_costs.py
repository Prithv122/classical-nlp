"""Cost-rule tests. Boundary values are dyadic (0.75, 4, 1) so that
``(1 - confidence) * misroute`` is exact in floating point: ``1 - 0.9`` is not ``0.1``."""

from __future__ import annotations

import numpy as np
import pytest

from classicalnlp import costs

COST = costs.RoutingCost(misroute=4, handoff=1)

# (true, pred, confidence)
TABLE = [(0, 0, 0.9), (1, 1, 0.8), (2, 0, 0.95), (1, 2, 0.5), (0, 0, 0.75), (2, 2, 0.6)]


def _columns():
    y_true, y_pred, confidence = (np.array(c) for c in zip(*TABLE, strict=True))
    return y_true, y_pred, confidence


def test_equality_at_the_boundary_hands_off():
    assert (1 - 0.75) * 4 == 1.0
    assert not costs.auto_route(np.array([0.75]), COST)[0]


@pytest.mark.parametrize(
    ("confidence", "ratio"), [(0.5, 2), (0.8, 5), (0.9, 10), (0.95, 20), (0.99, 100)]
)
def test_decimal_valued_boundary_hands_off_at_every_ratio(confidence, ratio):
    # Not dyadic: 0.8 at 5:1, 0.9 at 10:1 and 0.99 at 100:1 sit within ~1e-16 of the boundary,
    # and a bare `<` auto-routed them while handing off 0.95 at 20:1.
    assert not costs.auto_route(np.array([confidence]), costs.RoutingCost.from_ratio(ratio))[0]


@pytest.mark.parametrize(("confidence", "ratio"), [(0.8, 5), (0.9, 10), (0.95, 20)])
def test_just_either_side_of_a_decimal_boundary(confidence, ratio):
    cost = costs.RoutingCost.from_ratio(ratio)
    above, below = confidence + 1e-6, confidence - 1e-6
    assert costs.auto_route(np.array([above, below]), cost).tolist() == [True, False]


def test_either_side_of_the_boundary():
    assert costs.auto_route(np.array([0.8, 0.5]), COST).tolist() == [True, False]


def test_threshold_agrees_with_auto_route():
    threshold = costs.confidence_threshold(COST)
    assert threshold == 0.75
    assert costs.auto_route(np.array([0.875]), COST)[0]  # above
    assert not costs.auto_route(np.array([0.625]), COST)[0]  # below
    assert not costs.auto_route(np.array([threshold]), COST)[0]  # on it


def test_confidence_outside_unit_interval_is_rejected():
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            costs.auto_route(np.array([bad]), COST)


def test_expected_cost_on_the_six_row_table():
    result = costs.expected_cost(*_columns(), COST)
    assert result.n == 6
    assert result.auto_routed == 3
    assert result.handed_off == 3
    assert result.misrouted == 1
    assert result.correct_auto == 2
    assert result.total_cost == 7.0
    assert result.cost_per_1000 == pytest.approx(1166.667, abs=1e-3)
    assert result.coverage == 0.5
    assert result.auto_accuracy == pytest.approx(2 / 3)


def test_inference_cost_is_charged_on_every_row():
    priced = costs.RoutingCost(misroute=4, handoff=1, inference_per_ticket=0.01)
    assert costs.expected_cost(*_columns(), priced).total_cost == pytest.approx(7.06)


def test_auto_accuracy_is_nan_when_nothing_is_auto_routed():
    result = costs.expected_cost([0, 1], [0, 1], np.array([0.1, 0.2]), COST)
    assert result.auto_routed == 0
    assert np.isnan(result.auto_accuracy)
    assert result.total_cost == 2.0


def test_expected_cost_rejects_length_mismatch_and_empty_input():
    with pytest.raises(ValueError, match="length"):
        costs.expected_cost([0, 1], [0], np.array([0.5, 0.5]), COST)
    with pytest.raises(ValueError, match="zero"):
        costs.expected_cost([], [], np.array([]), COST)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"misroute": 0, "handoff": 1},
        {"misroute": -1, "handoff": 1},
        {"misroute": 4, "handoff": -1},
        {"misroute": 4, "handoff": 1, "inference_per_ticket": -0.01},
    ],
)
def test_routing_cost_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        costs.RoutingCost(**kwargs)


def test_ratio_constants():
    assert costs.HEADLINE_RATIO == 10
    assert costs.SENSITIVITY_RATIOS == (2, 5, 10, 20)
    ten = costs.RoutingCost.from_ratio(10)
    assert (ten.misroute, ten.handoff) == (10, 1)
