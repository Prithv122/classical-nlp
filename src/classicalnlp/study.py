"""Study runner: fit -> predict -> calibrate -> apply the cost rule at each ratio.

Calibration is a study step applied identically to every arm. Each run reports a ``raw``
row (the arm's own confidence) and an ``isotonic`` row. On the validation split the
isotonic confidence is cross-fitted: each fold is scored by a calibrator fitted on the
other folds, so no row is scored by a calibrator that saw it. On the test split one
calibrator is fitted on all validation rows and applied to the test rows. Abstained rows
are left out of every calibrator fit and keep confidence 0.0.

The arm is fitted on the fit split only and is not refitted afterwards, so the calibrator
calibrates the same model it was fitted against. Numbers produced on the synthetic fixture
test this code; they are not results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold

from . import calibrate, costs
from .costs import CostBreakdown, RoutingCost
from .data import Corpus
from .evaluate import RANDOM_STATE
from .router import ABSTAIN, Router


def cross_fit_isotonic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = RANDOM_STATE,
) -> np.ndarray:
    """Isotonic-calibrated confidence where each fold is scored by the other folds' fit."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    confidence = np.asarray(confidence, dtype=float)
    if len(confidence) < n_splits:
        raise ValueError(f"need at least {n_splits} rows to cross-fit, got {len(confidence)}")
    out = np.empty(len(confidence))
    for train, held in KFold(n_splits, shuffle=True, random_state=seed).split(confidence):
        calibrator = calibrate.fit_top_label_isotonic(
            y_true[train], y_pred[train], confidence[train]
        )
        out[held] = calibrate.apply_isotonic(calibrator, confidence[held])
    return np.clip(out, 0.0, 1.0)


@dataclass(frozen=True, eq=False)
class ArmRun:
    arm: str
    split: str  # "val" | "test"
    calibration: str  # "raw" | "isotonic"
    y_true: np.ndarray
    y_pred: np.ndarray
    confidence: np.ndarray
    ece: float
    brier: float
    costs: dict[int, CostBreakdown]


def _run(arm, split, calibration, y_true, y_pred, confidence, ratios) -> ArmRun:
    correct = calibrate.top_label_correct(y_true, y_pred)
    return ArmRun(
        arm=arm,
        split=split,
        calibration=calibration,
        y_true=y_true,
        y_pred=y_pred,
        confidence=confidence,
        ece=calibrate.top_label_ece(y_true, y_pred, confidence),
        brier=calibrate.brier_score(correct, confidence),
        costs={
            r: costs.expected_cost(y_true, y_pred, confidence, RoutingCost.from_ratio(r))
            for r in ratios
        },
    )


def run_arm(
    router: Router,
    fit: Corpus,
    val: Corpus,
    test: Corpus | None = None,
    *,
    ratios=costs.SENSITIVITY_RATIOS,
    n_splits: int = 5,
    seed: int = RANDOM_STATE,
) -> list[ArmRun]:
    """Fit on ``fit``; evaluate on ``val`` (cross-fitted) or, if given, on ``test``."""
    router.fit(fit)
    val_true = np.asarray(val.labels)
    val_pred, val_conf = router.predict_top(val.documents)
    val_ok = val_pred != ABSTAIN

    if test is None:
        split, y_true, y_pred, raw = "val", val_true, val_pred, val_conf
        calibrated = np.zeros(len(raw))
        calibrated[val_ok] = cross_fit_isotonic(
            val_true[val_ok], val_pred[val_ok], val_conf[val_ok], n_splits=n_splits, seed=seed
        )
    else:
        if not val_ok.any():
            raise ValueError("every validation row was abstained; nothing to calibrate on")
        calibrator = calibrate.fit_top_label_isotonic(
            val_true[val_ok], val_pred[val_ok], val_conf[val_ok]
        )
        split, y_true = "test", np.asarray(test.labels)
        y_pred, raw = router.predict_top(test.documents)
        calibrated = np.zeros(len(raw))
        ok = y_pred != ABSTAIN
        calibrated[ok] = np.clip(calibrate.apply_isotonic(calibrator, raw[ok]), 0.0, 1.0)

    return [
        _run(router.name, split, "raw", y_true, y_pred, raw, ratios),
        _run(router.name, split, "isotonic", y_true, y_pred, calibrated, ratios),
    ]


def format_runs(runs: list[ArmRun]) -> str:
    """Plain-text table, one line per (run, ratio)."""
    lines = [
        f"{'calibration':<12}{'ratio':>6}{'threshold':>11}{'coverage':>10}"
        f"{'auto acc':>10}{'cost/1000':>11}{'ECE':>8}{'Brier':>8}"
    ]
    for run in runs:
        for ratio, b in run.costs.items():
            threshold = costs.confidence_threshold(RoutingCost.from_ratio(ratio))
            acc = "n/a" if np.isnan(b.auto_accuracy) else f"{b.auto_accuracy:.3f}"
            lines.append(
                f"{run.calibration:<12}{f'{ratio}:1':>6}{threshold:>11.3f}{b.coverage:>10.3f}"
                f"{acc:>10}{b.cost_per_1000:>11.1f}{run.ece:>8.3f}{run.brier:>8.3f}"
            )
    return "\n".join(lines)
