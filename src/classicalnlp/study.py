"""Study runner: fit -> predict -> calibrate -> apply the cost rule at each ratio.

Calibration is a study step applied identically to every arm. Each run reports a ``raw``
row (the arm's own confidence) and an ``isotonic`` row. On the validation split the
isotonic confidence is cross-fitted: each fold is scored by a calibrator fitted on the
other folds, so no row is scored by a calibrator that saw it. On the test split one
calibrator is fitted on all validation rows and applied to the test rows. Abstained rows
are left out of every calibrator fit and keep confidence 0.0.

Cost ratios below 1 are refused before the arm is fitted or called: when a wrong auto-route
costs less than a handoff, the rule auto-routes every ticket, including abstentions
(confidence 0.0), which would then count as misroutes.

The arm is fitted on the fit split only and is not refitted afterwards, so the calibrator
calibrates the same model it was fitted against. Numbers produced on the synthetic fixture
test this code; they are not results.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from . import calibrate, costs
from .costs import CostBreakdown, RoutingCost
from .data import Corpus
from .evaluate import RANDOM_STATE
from .router import ABSTAIN, Router
from .router.base import Usage


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
    usage: Usage | None = None


def _run(arm, split, calibration, y_true, y_pred, confidence, ratios, usage=None) -> ArmRun:
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
        usage=usage,
    )


def _usage_of(router, texts) -> Usage:
    """The usage the router set for the call just made, else a per-ticket wall-clock timing."""
    usage = getattr(router, "usage", None)
    if usage is not None:
        return usage
    seconds = np.empty(len(texts))
    for i, text in enumerate(texts):
        start = time.perf_counter()
        router.predict_top([text])
        seconds[i] = time.perf_counter() - start
    return Usage(seconds=seconds)


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
    ratios = tuple(ratios)
    if any(ratio < 1 for ratio in ratios):
        raise ValueError(
            "cost ratios below 1 are not supported: when a wrong auto-route costs less than a "
            "handoff the rule auto-routes every ticket, including abstentions (confidence 0.0), "
            "which would then count as misroutes"
        )
    router.fit(fit)
    val_true = np.asarray(val.labels)
    val_pred, val_conf = router.predict_top(val.documents)
    usage = _usage_of(router, val.documents) if test is None else None
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
        usage = _usage_of(router, test.documents)
        calibrated = np.zeros(len(raw))
        ok = y_pred != ABSTAIN
        calibrated[ok] = np.clip(calibrate.apply_isotonic(calibrator, raw[ok]), 0.0, 1.0)

    return [
        _run(router.name, split, "raw", y_true, y_pred, raw, ratios, usage),
        _run(router.name, split, "isotonic", y_true, y_pred, calibrated, ratios, usage),
    ]


def format_runs(runs: list[ArmRun]) -> str:
    """Plain-text table, one line per (run, ratio)."""
    lines = [
        f"{'arm':<11}{'calibration':<12}{'ratio':>6}{'threshold':>11}{'coverage':>10}"
        f"{'auto acc':>10}{'abstained':>10}{'cost/1000':>11}{'ECE':>8}{'Brier':>8}"
    ]
    for run in runs:
        abstained = int((np.asarray(run.y_pred) == ABSTAIN).sum())
        for ratio, b in run.costs.items():
            threshold = costs.confidence_threshold(RoutingCost.from_ratio(ratio))
            acc = "n/a" if np.isnan(b.auto_accuracy) else f"{b.auto_accuracy:.3f}"
            lines.append(
                f"{run.arm:<11}{run.calibration:<12}{f'{ratio}:1':>6}{threshold:>11.3f}"
                f"{b.coverage:>10.3f}{acc:>10}{abstained:>10}{b.cost_per_1000:>11.1f}"
                f"{run.ece:>8.3f}{run.brier:>8.3f}"
            )
    return "\n".join(lines)


def format_usage(runs: list[ArmRun]) -> str:
    """One line per distinct arm: latency percentiles and mean token counts.

    Runs of one arm share their usage, so the first run of each arm stands for it. Tokens are
    reported only when all three token arrays are present.
    """
    lines = []
    seen = set()
    for run in runs:
        if run.arm in seen or run.usage is None:
            continue
        seen.add(run.arm)
        u = run.usage
        p50, p95 = np.percentile(u.seconds, [50, 95])
        line = f"{run.arm}: latency p50 {p50:.3f}s p95 {p95:.3f}s per ticket; "
        if u.prompt_tokens is None or u.cached_prompt_tokens is None or u.output_tokens is None:
            line += "tokens n/a"
        else:
            line += (
                f"prompt tokens mean {np.mean(u.prompt_tokens):.0f} "
                f"(cached mean {np.mean(u.cached_prompt_tokens):.0f}); "
                f"output tokens mean {np.mean(u.output_tokens):.1f}"
            )
        lines.append(line)
    return "\n".join(lines)


def save_runs(runs: list[ArmRun], path: str | Path) -> int:
    """Write one JSON line per (run, ticket), overwriting ``path``; return the line count."""

    def cell(array, i, kind):
        return None if array is None else kind(array[i])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for run in runs:
            u = run.usage
            for i in range(len(run.y_true)):
                row = {
                    "arm": run.arm,
                    "split": run.split,
                    "calibration": run.calibration,
                    "index": i,
                    "y_true": int(run.y_true[i]),
                    "y_pred": int(run.y_pred[i]),
                    "confidence": float(run.confidence[i]),
                    "seconds": cell(u.seconds if u else None, i, float),
                    "prompt_tokens": cell(u.prompt_tokens if u else None, i, int),
                    "cached_prompt_tokens": cell(u.cached_prompt_tokens if u else None, i, int),
                    "output_tokens": cell(u.output_tokens if u else None, i, int),
                }
                handle.write(json.dumps(row) + "\n")
                written += 1
    return written
