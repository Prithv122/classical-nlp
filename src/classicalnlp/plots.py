"""Reliability diagrams for the router study.

matplotlib is an optional extra (``uv sync --extra router``) and is imported inside the
function that draws, never at module level, so the rest of the package does not need it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import calibrate

NO_MATPLOTLIB = (
    "route-report plots need matplotlib: install the router extra "
    "(uv sync --extra router) or pass --no-plots"
)


def reliability_bins(
    y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray, n_bins: int = 10
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Top-label ``(ece, bin_true_rate, bin_pred_mean, bin_count)`` over all rows,
    abstentions (confidence 0.0, never correct) included."""
    correct = calibrate.top_label_correct(y_true, y_pred)
    return calibrate.expected_calibration_error(correct, confidence, n_bins=n_bins)


def save_reliability(panels: list, path: str | Path, source_note: str | None = None) -> None:
    """Write one PNG: a grid of reliability panels, two columns (raw, isotonic).

    ``panels`` is a list of ``(label, y_true, y_pred, confidence)`` in display order, two per
    arm. Each panel draws the diagonal and one marker per non-empty bin at (mean confidence,
    observed accuracy), sized by the bin count.
    """
    try:
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise ImportError(NO_MATPLOTLIB) from exc

    rows = max(1, -(-len(panels) // 2))
    fig = Figure(figsize=(8.0, 3.8 * rows), layout="constrained")
    axes = fig.subplots(rows, 2, squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, (label, y_true, y_pred, confidence) in zip(axes.ravel(), panels, strict=False):
        ece, true_rate, pred_mean, count = reliability_bins(y_true, y_pred, confidence)
        shown = count > 0
        ax.set_visible(True)
        ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
        ax.scatter(
            pred_mean[shown], true_rate[shown], s=20 + 300 * count[shown] / count.max(), alpha=0.8
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("mean confidence")
        ax.set_ylabel("observed accuracy")
        ax.set_title(f"{label}  ECE {ece:.3f}  n={len(y_true)}", fontsize=10)
    if source_note:
        fig.suptitle(source_note, fontsize=9)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, format="png")
