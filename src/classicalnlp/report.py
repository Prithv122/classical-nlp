"""Read the per-ticket files written by ``route --save`` and write the study's tables.

Everything is recomputed from the saved rows, never read from another file: the sensitivity
table, the paired comparisons (cost and ECE bootstrap, McNemar), the break-even compute
sensitivity, usage and abstention tables, an optional reliability diagram and a Markdown report.
The statistics themselves live in :mod:`classicalnlp.compare`.

Costs are routing decision costs per 1,000 tickets, excluding compute, in handoff units (one
human handoff = 1.0). No price of any kind is used. The report states results mechanically and
picks no winner.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import calibrate, compare, costs
from . import plots as plot_module
from .costs import RoutingCost
from .evaluate import RANDOM_STATE, mcnemar
from .router.base import ABSTAIN

ARM_ORDER = ("A", "B-logprob", "B-verbal")
CALIBRATIONS = ("raw", "isotonic")


def _spec(text: str) -> dict[str, str]:
    """Columns of one table as ``name`` or ``name:format`` words; shared by CSV and Markdown."""
    return {w.partition(":")[0]: w.partition(":")[2] for w in text.split()}


KEYS = list(
    _spec(
        "arm split calibration index y_true y_pred confidence "
        "seconds prompt_tokens cached_prompt_tokens output_tokens"
    )
)
OPTIONAL = KEYS[7:]


SENSITIVITY = _spec(
    "arm calibration ratio n auto_routed handed_off misrouted abstained coverage:.3f "
    "auto_accuracy:.3f routing_cost_per_1000:.1f ece:.3f brier:.3f"
)
COST = _spec(
    "pair arm_a arm_b calibration ratio cost_a:.1f cost_b:.1f difference:+.1f ci_low:+.1f "
    "ci_high:+.1f p_value:.3f headline"
)
ACCURACY = _spec("arm_a arm_b only_a only_b p_value:.3f")
ECE = _spec(
    "arm_a arm_b calibration ece_a:.3f ece_b:.3f difference:+.3f ci_low:+.3f ci_high:+.3f "
    "p_value:.3f"
)
BREAKEVEN = _spec(
    "arm_b calibration ratio advantage_per_ticket:+.4f breakeven_per_ticket:.4f "
    "b_mean_seconds:.3f breakeven_per_second:.4f status"
)
USAGE = _spec(
    "arm n latency_p50:.3f latency_p95:.3f prompt_tokens_mean:.1f "
    "cached_prompt_tokens_mean:.1f output_tokens_mean:.1f"
)
ABSTENTIONS = _spec("arm abstained n share:.3f")


@dataclass(frozen=True, eq=False)
class RunData:
    arm: str
    split: str
    calibration: str
    y_true: np.ndarray
    y_pred: np.ndarray
    confidence: np.ndarray
    seconds: np.ndarray | None
    prompt_tokens: np.ndarray | None
    cached_prompt_tokens: np.ndarray | None
    output_tokens: np.ndarray | None


# -- loading ----------------------------------------------------------------------------


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _parse_row(line: str, where: str) -> dict:
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        raise ValueError(f"{where}: not valid JSON") from None
    if not isinstance(row, dict):
        raise ValueError(f"{where}: not valid JSON (expected an object)")
    for key in KEYS:
        if key not in row:
            raise ValueError(f"{where}: missing key {key!r}")
    if row["arm"] not in ARM_ORDER:
        raise ValueError(f"{where}: unknown arm {row['arm']!r}")
    if row["calibration"] not in CALIBRATIONS:
        raise ValueError(f"{where}: calibration must be raw or isotonic")
    if row["split"] not in ("val", "test"):
        raise ValueError(f"{where}: split must be val or test")
    if not all(_is_int(row[k]) for k in ("index", "y_true", "y_pred")):
        raise ValueError(f"{where}: index, y_true and y_pred must be integers")
    confidence = row["confidence"]
    if not _is_number(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{where}: confidence outside [0, 1]")
    if row["y_pred"] == ABSTAIN and confidence != 0.0:
        raise ValueError(f"{where}: an abstention (y_pred -1) must have confidence 0.0")
    if not all(row[k] is None or _is_number(row[k]) for k in OPTIONAL):
        raise ValueError(f"{where}: seconds and token counts must be numbers or null")
    return row


def _column(rows: list[dict], key: str, label: str) -> np.ndarray | None:
    values = [row[key] for row in rows]
    if all(v is None for v in values):
        return None
    if any(v is None for v in values):
        raise ValueError(f"{label}: {key} mixes null and non-null values")
    return np.array(values, dtype=float)


def load_runs(paths, *, final: bool = False) -> list[RunData]:
    """Read and validate the saved runs, ordered by arm then raw before isotonic."""
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("no run files given: pass at least one file written by route --save")
    groups: dict[tuple[str, str, str], dict] = {}
    for source, path in enumerate(paths):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"cannot read run file {path}: {exc}") from exc
        for lineno, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            row = _parse_row(line, f"{path}:{lineno}")
            key = (row["arm"], row["split"], row["calibration"])
            group = groups.setdefault(key, {"source": source, "path": path, "rows": []})
            if group["source"] != source:
                raise ValueError(
                    f"duplicate run: {key[0]} {key[1]} {key[2]} is in both {group['path']} "
                    f"and {path}"
                )
            group["rows"].append(row)
    if not groups:
        raise ValueError("no runs found in the given files")
    split = next(iter(groups))[1]
    if any(key[1] != split for key in groups):
        raise ValueError("runs span two splits: val and test rows cannot be reported together")
    if split == "test" and not final:
        raise ValueError("pass --final: these runs are on the official test split")

    runs = []
    for (arm, _, calibration), group in groups.items():
        label = f"{arm} {calibration}"
        rows = sorted(group["rows"], key=lambda r: r["index"])
        if [r["index"] for r in rows] != list(range(len(rows))):
            raise ValueError(f"{label}: indices must be exactly 0..n-1, each once")
        runs.append(
            RunData(
                arm=arm,
                split=split,
                calibration=calibration,
                y_true=np.array([r["y_true"] for r in rows], dtype=int),
                y_pred=np.array([r["y_pred"] for r in rows], dtype=int),
                confidence=np.array([r["confidence"] for r in rows], dtype=float),
                **{key: _column(rows, key, label) for key in OPTIONAL},
            )
        )
    runs.sort(key=lambda r: (ARM_ORDER.index(r.arm), CALIBRATIONS.index(r.calibration)))
    for run in runs[1:]:
        if not np.array_equal(run.y_true, runs[0].y_true):
            raise ValueError("runs do not share the same tickets (different y_true or n)")
    first = _first_runs(runs)
    if any(not np.array_equal(r.y_pred, first[r.arm].y_pred) for r in runs):
        raise ValueError("calibration changed the predictions")
    return runs


# -- tables -----------------------------------------------------------------------------


def _first_runs(runs: list[RunData]) -> dict[str, RunData]:
    """One run per arm (the raw one), for what calibration does not change."""
    out: dict[str, RunData] = {}
    for run in runs:
        out.setdefault(run.arm, run)
    return out


def _cost_of(run: RunData, ratio: int) -> costs.CostBreakdown:
    return costs.expected_cost(
        run.y_true, run.y_pred, run.confidence, RoutingCost.from_ratio(ratio)
    )


def sensitivity_rows(runs: list[RunData]) -> list[dict]:
    rows = []
    for run in runs:
        correct = calibrate.top_label_correct(run.y_true, run.y_pred)
        for ratio in costs.SENSITIVITY_RATIOS:
            b = _cost_of(run, ratio)
            rows.append(
                {
                    "arm": run.arm,
                    "calibration": run.calibration,
                    "ratio": ratio,
                    "n": b.n,
                    "auto_routed": b.auto_routed,
                    "handed_off": b.handed_off,
                    "misrouted": b.misrouted,
                    "abstained": int((run.y_pred == ABSTAIN).sum()),
                    "coverage": b.coverage,
                    "auto_accuracy": b.auto_accuracy,
                    "routing_cost_per_1000": b.cost_per_1000,
                    "ece": calibrate.top_label_ece(run.y_true, run.y_pred, run.confidence),
                    "brier": calibrate.brier_score(correct, run.confidence),
                }
            )
    return rows


def _pairs(runs: list[RunData]):
    """(A run, B run) for every B variant and calibration both arms have."""
    a_runs = {r.calibration: r for r in runs if r.arm == "A"}
    return [(a_runs[b.calibration], b) for b in runs if b.arm != "A" and b.calibration in a_runs]


def _ticket_costs(run: RunData, ratio: int) -> np.ndarray:
    cost = RoutingCost.from_ratio(ratio)
    return compare.ticket_costs(run.y_true, run.y_pred, run.confidence, cost)


def cost_rows(runs: list[RunData], iterations: int = 2000) -> list[dict]:
    rows = []
    for a, b in _pairs(runs):
        for ratio in costs.SENSITIVITY_RATIOS:
            cost_a, cost_b = _ticket_costs(a, ratio), _ticket_costs(b, ratio)
            stats = compare.paired_bootstrap_cost(cost_a, cost_b, iterations=iterations)
            headline = (b.arm, a.calibration, ratio) == ("B-logprob", "isotonic", 10)
            rows.append(
                {
                    "pair": f"{a.arm} vs {b.arm}",
                    "arm_a": a.arm,
                    "arm_b": b.arm,
                    "calibration": a.calibration,
                    "ratio": ratio,
                    "cost_a": float(cost_a.mean() * 1000),
                    "cost_b": float(cost_b.mean() * 1000),
                    **stats,
                    "headline": headline,
                }
            )
    return rows


def accuracy_rows(runs: list[RunData]) -> list[dict]:
    first = _first_runs(runs)
    if "A" not in first:
        return []
    a = first["A"]
    return [
        {"arm_a": "A", "arm_b": arm, **mcnemar(a.y_true, a.y_pred, first[arm].y_pred)}
        for arm in ARM_ORDER[1:]
        if arm in first
    ]


def ece_rows(runs: list[RunData], iterations: int = 2000) -> list[dict]:
    rows = []
    for a, b in _pairs(runs):
        stats = compare.paired_bootstrap_ece(
            a.y_true, a.y_pred, a.confidence, b.y_pred, b.confidence, iterations=iterations
        )
        rows.append(
            {
                "arm_a": a.arm,
                "arm_b": b.arm,
                "calibration": a.calibration,
                "ece_a": calibrate.top_label_ece(a.y_true, a.y_pred, a.confidence),
                "ece_b": calibrate.top_label_ece(b.y_true, b.y_pred, b.confidence),
                **stats,
            }
        )
    return rows


def breakeven_rows(runs: list[RunData]) -> list[dict]:
    rows = []
    for a, b in _pairs(runs):
        mean_seconds = None if b.seconds is None else float(np.mean(b.seconds))
        for ratio in costs.SENSITIVITY_RATIOS:
            out = compare.breakeven(_ticket_costs(a, ratio), _ticket_costs(b, ratio), b.seconds)
            rows.append(
                {
                    "arm_b": b.arm,
                    "calibration": a.calibration,
                    "ratio": ratio,
                    "b_mean_seconds": mean_seconds,
                    **out,
                }
            )
    return rows


def usage_rows(runs: list[RunData]) -> list[dict]:
    def mean(array):
        return None if array is None else float(np.mean(array))

    rows = []
    for arm, run in _first_runs(runs).items():
        p50 = p95 = None
        if run.seconds is not None:
            p50, p95 = (float(x) for x in np.percentile(run.seconds, [50, 95]))
        rows.append(
            {
                "arm": arm,
                "n": len(run.y_true),
                "latency_p50": p50,
                "latency_p95": p95,
                "prompt_tokens_mean": mean(run.prompt_tokens),
                "cached_prompt_tokens_mean": mean(run.cached_prompt_tokens),
                "output_tokens_mean": mean(run.output_tokens),
            }
        )
    return rows


def abstention_rows(runs: list[RunData]) -> list[dict]:
    rows = []
    for arm, run in _first_runs(runs).items():
        abstained = int((run.y_pred == ABSTAIN).sum())
        n = len(run.y_pred)
        rows.append({"arm": arm, "abstained": abstained, "n": n, "share": abstained / n})
    return rows


# -- writing ----------------------------------------------------------------------------


def _cell(value, spec: str = "") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return format(value, spec or ".10g") if isinstance(value, float) else str(value)


def _write_csv(path: Path, columns: dict[str, str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows([_cell(row[c]) for c in columns] for row in rows)


def _table(rows: list[dict], columns: dict[str, str]) -> list[str]:
    """Markdown table over the same columns as the CSV, with the column's number format."""
    lines = ["| " + " | ".join(c.replace("_", " ") for c in columns) + " |"]
    lines.append("|" + "---|" * len(columns))
    lines += ["| " + " | ".join(_cell(r[c], f) for c, f in columns.items()) + " |" for r in rows]
    return [*lines, ""]


def _hypothesis(tag: str, name: str, rows: list[dict], spec: str) -> str:
    if not rows:
        return f"- {tag} ({name}): not computed (needs A and B-logprob, isotonic)"
    r = rows[0]
    status = compare.interval_status(r["ci_low"], r["ci_high"])
    return (
        f"- {tag} ({name}): {status}; difference {r['difference']:{spec}}, "
        f"95% CI [{r['ci_low']:{spec}}, {r['ci_high']:{spec}}]"
    )


def _markdown(runs, tables, source_note, iterations) -> str:
    first = runs[0]
    note = source_note or "source not recorded in the run files"
    lines = ["# Router study report", "", f"Source: {note}", ""]
    lines += [f"Split: {first.split}; n = {len(first.y_true)} tickets.", ""]
    if first.split == "test":
        lines += ["**FINAL: official test split**", ""]
    lines += [
        "## Routing decision cost per 1,000 tickets, excluding compute",
        "",
        "ratio is misroute : handoff. ECE is top-label, 10 equal-width bins, over all rows; an "
        "abstention has confidence 0.0 and is never correct.",
        "",
        *_table(tables["sensitivity"], SENSITIVITY),
        "## Abstentions",
        "",
        *_table(tables["abstentions"], ABSTENTIONS),
    ]
    if tables["cost"]:
        headline = [r for r in tables["cost"] if r["headline"]]
        ece_head = [
            r for r in tables["ece"] if r["arm_b"] == "B-logprob" and r["calibration"] == "isotonic"
        ]
        lines += [
            "## Headline comparison",
            "",
            "A - B in routing decision cost per 1,000 tickets, excluding compute; negative means "
            f"A is lower. Paired bootstrap, {iterations} resamples, seed {RANDOM_STATE}, 95% "
            "percentile interval.",
            "",
            *_table(headline, COST),
            "## Other cost comparisons",
            "",
            *_table([r for r in tables["cost"] if not r["headline"]], COST),
            "## Accuracy (McNemar, exact)",
            "",
            "Abstentions count as wrong.",
            "",
            *_table(tables["accuracy"], ACCURACY),
            "## Calibration (ECE) comparison",
            "",
            *_table(tables["ece"], ECE),
            "## Hypotheses",
            "",
            _hypothesis("H1", "routing cost per 1,000, A - B, isotonic, 10:1", headline, "+.1f"),
            _hypothesis("H2", "ECE, A - B, isotonic", ece_head, "+.3f"),
            "",
            "## Break-even compute sensitivity",
            "",
            "Advantage per ticket is mean(cost A - cost B); positive means B is cheaper before "
            "compute. The break-even is that advantage when it is positive and blank otherwise; "
            "per second it is divided by B's mean model-side seconds.",
            "",
            "Costs are in handoff units (one human handoff = 1.0); multiply by your own handoff "
            "cost. Arm A's own compute is not subtracted.",
            "",
            *_table(tables["breakeven"], BREAKEVEN),
        ]
    else:
        lines += ["comparison skipped: needs arm A and a B variant", ""]
    lines += [
        "## Usage",
        "",
        *_table(tables["usage"], USAGE),
        "Arm A's latency is wall-clock per ticket in-process; arm B's is the model-side time "
        "Ollama reports with model load excluded, so the two are not directly comparable.",
        "",
        "Arm C (dedicated decision model): not evaluated (not available).",
        "",
        "## Limits",
        "",
        "- Isotonic calibrators are not refitted inside a bootstrap resample, and cross-fitted "
        "confidences are treated as fixed.",
        "- Inference is not priced: routing cost excludes compute, and the break-even is "
        "arithmetic, not a price.",
        "- Abstentions count as wrong in McNemar and as handoffs in cost.",
        "",
    ]
    return "\n".join(lines)


def write_report(
    runs: list[RunData],
    out_dir: str | Path,
    *,
    source_note: str | None = None,
    iterations: int = 2000,
    plots: bool = True,
) -> list[Path]:
    """Write the CSVs, the optional reliability diagram and ``report.md``; return the paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "sensitivity": sensitivity_rows(runs),
        "usage": usage_rows(runs),
        "abstentions": abstention_rows(runs),
        "cost": cost_rows(runs, iterations),
        "accuracy": accuracy_rows(runs),
        "ece": ece_rows(runs, iterations),
        "breakeven": breakeven_rows(runs),
    }
    if plots:
        # Drawn before anything else is written, so a missing matplotlib leaves no half-report.
        panels = [(f"{r.arm} {r.calibration}", r.y_true, r.y_pred, r.confidence) for r in runs]
        plot_module.save_reliability(panels, out / "reliability.png", source_note)
    files = [
        ("sensitivity.csv", SENSITIVITY, "sensitivity"),
        ("usage.csv", USAGE, "usage"),
        ("abstentions.csv", ABSTENTIONS, "abstentions"),
        ("comparison_cost.csv", COST, "cost"),
        ("comparison_accuracy.csv", ACCURACY, "accuracy"),
        ("comparison_ece.csv", ECE, "ece"),
        ("breakeven.csv", BREAKEVEN, "breakeven"),
    ]
    written = []
    for name, columns, key in files:
        if tables[key]:
            _write_csv(out / name, columns, tables[key])
            written.append(out / name)
    written += [out / "reliability.png"] if plots else []
    (out / "report.md").write_text(
        _markdown(runs, tables, source_note, iterations), encoding="utf-8"
    )
    return [*written, out / "report.md"]
