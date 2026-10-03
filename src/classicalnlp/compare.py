"""Paired comparisons of two routers scored on the same tickets.

Every statistic here is paired: both arms are evaluated on the same tickets, in the same
order, and a resample picks the same ticket positions for both.

- :func:`ticket_costs`: the routing cost of each ticket (0 for a correct auto-route, the
  misroute cost for a wrong one, the handoff cost for a handoff).
- :func:`paired_bootstrap_cost`: the mean per-ticket cost difference, scaled to 1,000 tickets,
  with a 95% percentile interval and a two-sided bootstrap p-value.
- :func:`paired_bootstrap_ece`: the same for the difference in top-label ECE.
- :func:`breakeven`: how much compute would have to cost before the cost advantage of one
  arm is used up. It is arithmetic on the cost difference, not a price.
- :func:`interval_status`: what a difference interval says, mechanically.

Sign convention: every difference is A minus B, so a negative cost difference means A is
cheaper. Costs are in handoff units (one human handoff = 1.0). Nothing here prices compute:
routing cost excludes it, and the break-even only reports the arithmetic.

The bootstrap resamples tickets; it does not refit an isotonic calibrator inside a resample.
"""

from __future__ import annotations

import numpy as np

from . import calibrate, costs
from .costs import RoutingCost
from .evaluate import RANDOM_STATE


def ticket_costs(
    y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray, cost: RoutingCost
) -> np.ndarray:
    """Routing cost of each ticket under the rule in :mod:`classicalnlp.costs`.

    Its mean times 1000 equals ``costs.expected_cost(...).cost_per_1000``.
    """
    if cost.inference_per_ticket != 0:
        raise ValueError("routing cost only: compute is reported separately")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence, dtype=float)
    n = len(y_true)
    if len(y_pred) != n or len(confidence) != n:
        raise ValueError(
            f"length mismatch: y_true {n}, y_pred {len(y_pred)}, confidence {len(confidence)}"
        )
    routed = costs.auto_route(confidence, cost)
    wrong = y_true != y_pred
    return np.where(routed, np.where(wrong, cost.misroute, 0.0), cost.handoff).astype(float)


def _p_value(difference: np.ndarray) -> float:
    """Two-sided bootstrap p-value, as in :func:`classicalnlp.evaluate.paired_bootstrap`."""
    return float(min(1.0, 2 * min((difference <= 0).mean(), (difference >= 0).mean())))


def _interval(observed: float, resampled: np.ndarray) -> dict[str, float]:
    low, high = np.percentile(resampled, [2.5, 97.5])
    return {
        "difference": float(observed),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": _p_value(resampled),
    }


def _indices(n: int, iterations: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, (iterations, n))


def paired_bootstrap_cost(
    cost_a: np.ndarray,
    cost_b: np.ndarray,
    *,
    iterations: int = 2000,
    seed: int = RANDOM_STATE,
) -> dict[str, float]:
    """Mean of ``cost_a - cost_b`` per 1,000 tickets, with a paired bootstrap interval."""
    cost_a = np.asarray(cost_a, dtype=float)
    cost_b = np.asarray(cost_b, dtype=float)
    if len(cost_a) != len(cost_b):
        raise ValueError(f"length mismatch: {len(cost_a)} vs {len(cost_b)} tickets")
    if len(cost_a) == 0:
        raise ValueError("cannot compare zero tickets")
    diff = cost_a - cost_b
    resampled = diff[_indices(len(diff), iterations, seed)].mean(axis=1) * 1000
    return _interval(diff.mean() * 1000, resampled)


def paired_bootstrap_ece(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    conf_a: np.ndarray,
    pred_b: np.ndarray,
    conf_b: np.ndarray,
    *,
    iterations: int = 2000,
    seed: int = RANDOM_STATE,
    n_bins: int = 10,
) -> dict[str, float]:
    """``ece_a - ece_b`` (top-label, all rows) with a paired bootstrap interval."""
    y_true = np.asarray(y_true)
    pred_a, conf_a = np.asarray(pred_a), np.asarray(conf_a, dtype=float)
    pred_b, conf_b = np.asarray(pred_b), np.asarray(conf_b, dtype=float)
    n = len(y_true)
    if not (len(pred_a) == len(conf_a) == len(pred_b) == len(conf_b) == n):
        raise ValueError("length mismatch between y_true and the two arms")
    if n == 0:
        raise ValueError("cannot compare zero tickets")

    def ece(idx: np.ndarray, pred: np.ndarray, conf: np.ndarray) -> float:
        return calibrate.top_label_ece(y_true[idx], pred[idx], conf[idx], n_bins=n_bins)

    everyone = np.arange(n)
    observed = ece(everyone, pred_a, conf_a) - ece(everyone, pred_b, conf_b)
    resampled = np.array(
        [
            ece(idx, pred_a, conf_a) - ece(idx, pred_b, conf_b)
            for idx in _indices(n, iterations, seed)
        ]
    )
    return _interval(observed, resampled)


def breakeven(
    cost_a: np.ndarray, cost_b: np.ndarray, seconds_b: np.ndarray | None = None
) -> dict[str, float | str | None]:
    """How much per-ticket compute B could cost before its routing-cost advantage is gone.

    ``advantage_per_ticket`` is ``mean(cost_a - cost_b)`` in handoff units; positive means B is
    cheaper before compute. The break-even is that advantage when it is positive, else None;
    per second it is divided by B's mean model-side seconds. A's own compute is not subtracted.
    """
    advantage = float(np.mean(np.asarray(cost_a, dtype=float) - np.asarray(cost_b, dtype=float)))
    per_ticket = advantage if advantage > 0 else None
    per_second = None
    if per_ticket is not None and seconds_b is not None:
        mean_seconds = float(np.mean(seconds_b))
        if mean_seconds > 0:
            per_second = per_ticket / mean_seconds
    if advantage > 0:
        status = "B cheaper before compute"
    elif advantage < 0:
        status = "A cheaper before compute"
    else:
        status = "equal before compute"
    return {
        "advantage_per_ticket": advantage,
        "breakeven_per_ticket": per_ticket,
        "breakeven_per_second": per_second,
        "status": status,
    }


def interval_status(ci_low: float, ci_high: float) -> str:
    """Read a difference interval (A - B): entirely below 0, entirely above 0, or neither."""
    if ci_high < 0:
        return "A lower"
    if ci_low > 0:
        return "B lower"
    return "no clear difference"
