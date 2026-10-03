"""The routing cost rule and expected-cost accounting.

A router either auto-routes a ticket to the predicted intent or hands it to a human.
With ``confidence`` the probability of the predicted class, the rule is::

    auto-route  iff  (1 - confidence) * misroute_cost < handoff_cost

The inequality is strict, so a ticket whose expected misroute cost equals the handoff
cost is handed off; "equals" means within :data:`BOUNDARY_TOLERANCE` (relative), so the
decision does not depend on floating-point rounding. Every handed-off ticket costs
``handoff``; every auto-routed ticket that is wrong costs ``misroute``; the router's own
inference cost is charged on every ticket, because it runs on the tickets it hands off too.

The headline cost ratio ``misroute : handoff`` of 10 : 1 is an **assumption**, not a
measurement. :data:`SENSITIVITY_RATIOS` lists the ratios swept around it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HEADLINE_RATIO = 10
SENSITIVITY_RATIOS = (2, 5, 10, 20)

# A confidence within this relative distance of the threshold is treated as equal to it, and
# so handed off. It makes the strict inequality hold for decimal-looking values such as the
# 0.8, 0.9 and 0.95 a model states, whose floating-point products are off by about 1e-16.
BOUNDARY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class RoutingCost:
    """Unit costs: a wrong auto-route, a human handoff, and the router's own inference."""

    misroute: float
    handoff: float
    inference_per_ticket: float = 0.0

    def __post_init__(self) -> None:
        if self.misroute <= 0:
            raise ValueError(f"misroute cost must be positive, got {self.misroute}")
        if self.handoff < 0:
            raise ValueError(f"handoff cost must not be negative, got {self.handoff}")
        if self.inference_per_ticket < 0:
            raise ValueError(
                f"inference cost must not be negative, got {self.inference_per_ticket}"
            )

    @classmethod
    def from_ratio(cls, ratio: float, handoff: float = 1.0) -> RoutingCost:
        """``misroute = ratio * handoff``, e.g. ``from_ratio(10)`` for the 10 : 1 headline."""
        return cls(misroute=ratio * handoff, handoff=handoff)


def auto_route(confidence: np.ndarray, cost: RoutingCost) -> np.ndarray:
    """Boolean mask, True where the ticket is auto-routed (False: hand off to a human)."""
    confidence = np.asarray(confidence, dtype=float)
    if np.any((confidence < 0.0) | (confidence > 1.0)) or np.any(np.isnan(confidence)):
        raise ValueError("confidence must lie in [0, 1]")
    # Strict, with a relative tolerance: ``1 - 0.9`` is not exactly ``0.1`` in floating point,
    # so a bare ``<`` would auto-route a confidence of exactly 0.9 at 10 : 1 but hand off 0.95
    # at 20 : 1. Anything within BOUNDARY_TOLERANCE of the threshold counts as on it.
    return (1.0 - confidence) * cost.misroute < cost.handoff * (1.0 - BOUNDARY_TOLERANCE)


def confidence_threshold(cost: RoutingCost) -> float:
    """Auto-route iff ``confidence > threshold``. Informational: :func:`auto_route` is the
    source of truth, because it evaluates the inequality itself."""
    return max(0.0, 1.0 - cost.handoff / cost.misroute)


@dataclass(frozen=True)
class CostBreakdown:
    """Outcome of applying the rule to one set of predictions.

    ``auto_accuracy`` is the share of auto-routed tickets that were right, and is ``nan``
    when nothing was auto-routed.
    """

    n: int
    auto_routed: int
    handed_off: int
    misrouted: int
    correct_auto: int
    total_cost: float
    cost_per_1000: float
    coverage: float
    auto_accuracy: float


def expected_cost(
    y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray, cost: RoutingCost
) -> CostBreakdown:
    """Apply the rule and total the cost.

    Misroutes are counted among auto-routed tickets only. Handed-off tickets cost
    ``cost.handoff`` whether or not the prediction was right. Inference is charged on all
    ``n`` tickets. ``cost_per_1000 = total_cost / n * 1000``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence, dtype=float)
    n = len(y_true)
    if len(y_pred) != n or len(confidence) != n:
        raise ValueError(
            f"length mismatch: y_true {n}, y_pred {len(y_pred)}, confidence {len(confidence)}"
        )
    if n == 0:
        raise ValueError("cannot compute expected cost of zero tickets")

    routed = auto_route(confidence, cost)
    correct = y_true == y_pred
    auto_routed = int(routed.sum())
    misrouted = int((routed & ~correct).sum())
    correct_auto = int((routed & correct).sum())
    handed_off = n - auto_routed
    total = misrouted * cost.misroute + handed_off * cost.handoff + n * cost.inference_per_ticket
    return CostBreakdown(
        n=n,
        auto_routed=auto_routed,
        handed_off=handed_off,
        misrouted=misrouted,
        correct_auto=correct_auto,
        total_cost=float(total),
        cost_per_1000=float(total / n * 1000),
        coverage=auto_routed / n,
        auto_accuracy=correct_auto / auto_routed if auto_routed else float("nan"),
    )
