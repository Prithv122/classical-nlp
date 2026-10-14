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
from .router.decision_arm import DEFAULT_PRICE_PER_MTOK, EXTRA_KEYS

ARM_ORDER = ("A", "B-logprob", "B-verbal", "C-prob", "C-vendor")
C_ARMS = ("C-prob", "C-vendor")
CALIBRATIONS = ("raw", "isotonic")
SPLITS = ("val", "test", "heldout")


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
ABC_BREAKEVEN = {"arm_a": "", **BREAKEVEN}
TRANSPORT = _spec("arm transport_failures model_abstentions n")
CALIBRATION_CLAIM = _spec("arm n_scored ece:.3f ci_low:.3f ci_high:.3f status")
CONFIDENCE_BINS = _spec("arm bin n accuracy:.3f")


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
    extras: dict[str, list] | None = None  # the six extra row keys, arm C only


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
    if row["arm"] in C_ARMS and row["split"] != "heldout":
        raise ValueError(f"{where}: arm C is reported on the held-out partition only")
    if row["split"] not in SPLITS:
        raise ValueError(f"{where}: split must be val, test or heldout")
    if not all(_is_int(row[k]) for k in ("index", "y_true", "y_pred")):
        raise ValueError(f"{where}: index, y_true and y_pred must be integers")
    confidence = row["confidence"]
    if not _is_number(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{where}: confidence outside [0, 1]")
    if row["y_pred"] == ABSTAIN and confidence != 0.0:
        raise ValueError(f"{where}: an abstention (y_pred -1) must have confidence 0.0")
    if not all(row[k] is None or _is_number(row[k]) for k in OPTIONAL):
        raise ValueError(f"{where}: seconds and token counts must be numbers or null")
    if row["arm"] in C_ARMS:
        _check_c_row(row, where)
    return row


def _check_c_row(row: dict, where: str) -> None:
    for key in EXTRA_KEYS:
        if key not in row:
            raise ValueError(f"{where}: missing key {key!r}")
    if not (isinstance(row["endpoint"], str) and isinstance(row["transport_failure"], bool)):
        raise ValueError(f"{where}: endpoint must be a string and transport_failure a boolean")
    if not (isinstance(row["cold_start"], bool) and _is_int(row["attempts"])):
        raise ValueError(f"{where}: cold_start must be a boolean and attempts an integer")
    if row["model_version"] is not None and not isinstance(row["model_version"], str):
        raise ValueError(f"{where}: model_version must be a string or null")
    if row["argmax_status"] not in (None, "unique", "tied", "differs"):
        raise ValueError(f"{where}: argmax_status must be unique, tied, differs or null")
    if row["transport_failure"] and (row["y_pred"] != ABSTAIN or row["confidence"] != 0.0):
        raise ValueError(f"{where}: a transport failure must be ABSTAIN with confidence 0.0")


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
                extras={k: [r[k] for r in rows] for k in EXTRA_KEYS} if arm in C_ARMS else None,
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


V2_HEADLINE = ("A", "B-logprob", "isotonic", 10)


def cost_rows(runs: list[RunData], iterations: int = 2000, pairs=None, headline=V2_HEADLINE):
    rows = []
    for a, b in _pairs(runs) if pairs is None else pairs:
        for ratio in costs.SENSITIVITY_RATIOS:
            cost_a, cost_b = _ticket_costs(a, ratio), _ticket_costs(b, ratio)
            stats = compare.paired_bootstrap_cost(cost_a, cost_b, iterations=iterations)
            is_headline = (a.arm, b.arm, a.calibration, ratio) == headline
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
                    "headline": is_headline,
                }
            )
    return rows


def accuracy_rows(runs: list[RunData], arm_pairs=None) -> list[dict]:
    first = _first_runs(runs)
    if arm_pairs is None:
        arm_pairs = [("A", arm) for arm in ARM_ORDER[1:]]
    return [
        {"arm_a": x, "arm_b": y, **mcnemar(first[x].y_true, first[x].y_pred, first[y].y_pred)}
        for x, y in arm_pairs
        if x in first and y in first
    ]


def ece_rows(runs: list[RunData], iterations: int = 2000, pairs=None) -> list[dict]:
    rows = []
    for a, b in _pairs(runs) if pairs is None else pairs:
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


def breakeven_rows(runs: list[RunData], pairs=None) -> list[dict]:
    """Break-even per pair. With explicit ``pairs`` the rows also name arm A of the pair and
    the status names the arms instead of "A" and "B"."""
    rows = []
    for a, b in _pairs(runs) if pairs is None else pairs:
        mean_seconds = None if b.seconds is None else float(np.mean(b.seconds))
        for ratio in costs.SENSITIVITY_RATIOS:
            out = compare.breakeven(_ticket_costs(a, ratio), _ticket_costs(b, ratio), b.seconds)
            if pairs is not None:
                out["status"] = out["status"].replace("B cheaper", f"{b.arm} cheaper")
                out["status"] = out["status"].replace("A cheaper", f"{a.arm} cheaper")
            rows.append(
                {
                    "arm_a": a.arm,
                    "arm_b": b.arm,
                    "calibration": a.calibration,
                    "ratio": ratio,
                    "b_mean_seconds": mean_seconds,
                    **out,
                }
            )
    return rows


def usage_rows(runs: list[RunData]) -> list[dict]:
    rows = []
    for arm, run in _first_runs(runs).items():
        # Arm C: a transport failure has no timing or tokens, so it is left out of the means.
        keep = np.ones(len(run.y_true), dtype=bool)
        if run.extras is not None:
            keep = ~np.array(run.extras["transport_failure"], dtype=bool)

        def mean(array, keep=keep):
            return None if array is None or not keep.any() else float(np.mean(array[keep]))

        p50 = p95 = None
        if run.seconds is not None and keep.any():
            p50, p95 = (float(x) for x in np.percentile(run.seconds[keep], [50, 95]))
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


# -- arm C study: the abc layout ---------------------------------------------------------

DISCLAIMER = (
    "This is a new partition-based comparison on 2,002 Banking77 training rows held out for "
    "this study. Those rows were used to fit arm A in v2, so this is not an independent "
    "external validation of the shipped v2 model, and its numbers are not comparable with the "
    "v2 results on the official test split."
)
ABC_HEADLINE = ("A", "C-prob", "isotonic", 10)
ABC_GROUPS = (
    ("Headline", (("A", "C-prob"),)),
    ("Secondary", (("B-logprob", "C-prob"), ("A", "B-logprob"))),
    ("Ablations", (("A", "C-vendor"), ("A", "B-verbal"))),
)
ABC_ARM_PAIRS = [pair for _, pairs in ABC_GROUPS for pair in pairs]
LATENCY_CLAIM_SECONDS = 0.5
H3_THRESHOLD = 0.05


def _abc_pairs(runs: list[RunData]):
    """(first run, second run) for every study pair and calibration both arms have."""
    by_key = {(r.arm, r.calibration): r for r in runs}
    return [
        (by_key[x, cal], by_key[y, cal])
        for x, y in ABC_ARM_PAIRS
        for cal in CALIBRATIONS
        if (x, cal) in by_key and (y, cal) in by_key
    ]


def _raw(runs: list[RunData], arm: str) -> RunData | None:
    return next((r for r in runs if r.arm == arm and r.calibration == "raw"), None)


def _status(low: float, high: float, first: str, second: str) -> str:
    """:func:`compare.interval_status` with the arm names of this pair."""
    status = compare.interval_status(low, high)
    return {"A lower": f"{first} lower", "B lower": f"{second} lower"}.get(status, status)


def h3_status(low: float, high: float) -> str:
    if high <= H3_THRESHOLD:
        return "supported"
    return "not supported" if low > H3_THRESHOLD else "inconclusive"


def latency_status(p50: float, p95: float) -> str:
    if p95 <= LATENCY_CLAIM_SECONDS:
        return "within claim"
    return "median within, tail outside" if p50 <= LATENCY_CLAIM_SECONDS else "outside claim"


def transport_rows(runs: list[RunData]) -> list[dict]:
    rows = []
    for arm in C_ARMS:
        if (run := _raw(runs, arm)) is not None:
            failed = np.array(run.extras["transport_failure"], dtype=bool)
            model = (run.y_pred == ABSTAIN) & ~failed
            rows.append(
                {
                    "arm": arm,
                    "transport_failures": int(failed.sum()),
                    "model_abstentions": int(model.sum()),
                    "n": len(failed),
                }
            )
    return rows


def calibration_claim_rows(runs: list[RunData], iterations: int = 2000) -> list[dict]:
    """H3: raw ECE over the rows C did not abstain on, interval from the shared resamples.

    A transport failure is scored ABSTAIN, so it is outside this ECE too; it is counted in the
    transport table, not here.
    """
    rows = []
    for arm in C_ARMS:
        if (run := _raw(runs, arm)) is None:
            continue
        kept = run.y_pred != ABSTAIN
        if not kept.any():
            continue

        def ece(idx, run=run):
            idx = idx[run.y_pred[idx] != ABSTAIN]
            if len(idx) == 0:
                return np.nan
            return calibrate.top_label_ece(run.y_true[idx], run.y_pred[idx], run.confidence[idx])

        resampled = np.array(
            [ece(idx) for idx in compare._indices(len(run.y_true), iterations, RANDOM_STATE)]
        )
        low, high = (float(x) for x in np.nanpercentile(resampled, [2.5, 97.5]))
        rows.append(
            {
                "arm": arm,
                "n_scored": int(kept.sum()),
                "ece": ece(np.arange(len(run.y_true))),
                "ci_low": low,
                "ci_high": high,
                "status": h3_status(low, high),
            }
        )
    return rows


def confidence_bin_rows(runs: list[RunData]) -> list[dict]:
    """Accuracy by raw confidence in the ECE bins: 10 equal-width bins of [0, 1], the first
    closed on the left and every bin closed on the right, over the rows not abstained."""
    rows = []
    for arm in C_ARMS:
        if (run := _raw(runs, arm)) is None:
            continue
        kept = run.y_pred != ABSTAIN
        correct = calibrate.top_label_correct(run.y_true[kept], run.y_pred[kept])
        _, rate, _, count = calibrate.expected_calibration_error(correct, run.confidence[kept])
        for b in range(10):
            label = f"{'[' if b == 0 else '('}{b / 10:.1f}, {(b + 1) / 10:.1f}]"
            rows.append(
                {
                    "arm": arm,
                    "bin": label,
                    "n": int(count[b]),
                    "accuracy": float(rate[b]) if count[b] else None,
                }
            )
    return rows


def latency_summary(runs: list[RunData]) -> dict | None:
    """H4 from C-prob: successful attempts of H tickets, cold starts and failures excluded."""
    run = _raw(runs, "C-prob")
    if run is None or run.seconds is None:
        return None
    failed = np.array(run.extras["transport_failure"], dtype=bool)
    cold = np.array(run.extras["cold_start"], dtype=bool)
    seconds = run.seconds[~failed & ~cold]
    retries = int(sum(a - 1 for a in run.extras["attempts"]))
    if len(seconds) == 0:
        return {"n": 0, "retries": retries, "cold_excluded": int(cold.sum())}
    p50, p95, p99 = (float(x) for x in np.percentile(seconds, [50, 95, 99]))
    return {
        "n": len(seconds),
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "share_within": float(np.mean(seconds <= LATENCY_CLAIM_SECONDS)),
        "status": latency_status(p50, p95),
        "retries": retries,
        "cold_excluded": int(cold.sum()),
        "failures_excluded": int(failed.sum()),
    }


def dollars_per_1000(
    runs: list[RunData], price_per_mtok: float = DEFAULT_PRICE_PER_MTOK
) -> float | None:
    """C's cost from reported input tokens, transport failures excluded; never in handoff units.

    ``price_per_mtok`` must be the price the runs were made at (``route --price-per-mtok``).
    """
    run = _raw(runs, "C-prob")
    if run is None or run.prompt_tokens is None:
        return None
    ok = ~np.array(run.extras["transport_failure"], dtype=bool)
    if not ok.any():
        return None
    return float(np.mean(run.prompt_tokens[ok])) * price_per_mtok / 1e6 * 1000


def argmax_counts(runs: list[RunData]) -> dict[str, int] | None:
    run = _raw(runs, "C-prob")
    if run is None:
        return None
    statuses = run.extras["argmax_status"]
    return {k: statuses.count(k) for k in ("unique", "tied", "differs")} | {
        "unusable": statuses.count(None)
    }


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


def _hypotheses_abc(runs: list[RunData], tables: dict) -> list[str]:
    def interval(rows, spec, first, second):
        r = rows[0]
        status = _status(r["ci_low"], r["ci_high"], first, second)
        return (
            f"{status}; difference {r['difference']:{spec}}, "
            f"95% CI [{r['ci_low']:{spec}}, {r['ci_high']:{spec}}]"
        )

    head = [r for r in tables["cost"] if r["headline"]]
    ece = [
        r
        for r in tables["ece"]
        if (r["arm_a"], r["arm_b"], r["calibration"]) == ("A", "C-prob", "isotonic")
    ]
    none = "not computed (needs A and C-prob, isotonic)"
    lines = [
        "- H1 (routing cost per 1,000, A - C-prob, isotonic, 10:1): "
        + (interval(head, "+.1f", "A", "C") if head else none),
        "- H2 (ECE after isotonic, A - C-prob, all rows, abstentions included): "
        + (interval(ece, "+.3f", "A", "C") if ece else none),
    ]
    for r in tables["claim"]:
        label = "primary" if r["arm"] == "C-prob" else "ablation"
        lines.append(
            f"- H3 (vendor calibration claim, {r['arm']}, {label}): {r['status']}; raw ECE "
            f"{r['ece']:.3f} over {r['n_scored']} rows not abstained, 95% CI "
            f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]; the {H3_THRESHOLD} threshold is this "
            "study's, not a vendor figure"
        )
    if not tables["claim"]:
        lines.append("- H3 (vendor calibration claim): not computed (needs C-prob raw)")
    lat = tables["latency"]
    if lat and lat["n"]:
        lines.append(
            f"- H4 (vendor latency claim, 70 to 500 ms): {lat['status']}; p50 {lat['p50']:.3f} s, "
            f"p95 {lat['p95']:.3f} s, p99 {lat['p99']:.3f} s, share at or under 0.5 s "
            f"{lat['share_within']:.3f}, over {lat['n']} tickets"
        )
    else:
        lines.append("- H4 (vendor latency claim, 70 to 500 ms): not computed (no timed tickets)")
    return lines


def _markdown_abc(runs, tables, source_note, iterations) -> str:
    first = runs[0]
    note = source_note or "source not recorded in the run files"
    lines = ["# Arm C study report: three arms on a new Banking77 partition", ""]
    lines += [DISCLAIMER, "", f"Source: {note}", ""]
    lines += [f"Split: {first.split}; n = {len(first.y_true)} tickets.", ""]
    lines += [
        "## Caveats",
        "",
        "- Prior exposure: the held-out rows come from the public Banking77 training split, so "
        "any of the three models may have seen these texts in pretraining; that is not "
        "controllable here. v2's B cache holds responses for the validation and official test "
        "rows only, so B never scored one of these rows in v2. Whether individual fit rows were "
        "read during v2 development is not recorded.",
        "- The 70 to 500 ms latency range and the claim that higher confidence means higher "
        "accuracy are vendor claims, tested here and not established by this report. The "
        "vendor publishes no calibration number.",
        "- Latency is client-side wall-clock of the successful attempt, measured from one "
        "machine (India); the vendor measured from the US West Coast.",
        "- Arm B ran on Ollama 0.40.2; v2 ran on 0.34.4. The weights are the same.",
        "- Only H1 is the headline. There is no correction for multiple comparisons.",
        "",
        "## Routing decision cost per 1,000 tickets, excluding compute",
        "",
        "ratio is misroute : handoff. ECE is top-label, 10 equal-width bins, over all rows; an "
        "abstention has confidence 0.0 and is never correct.",
        "",
        *_table(tables["sensitivity"], SENSITIVITY),
        "## Abstentions",
        "",
        "Includes transport failures, which are scored ABSTAIN.",
        "",
        *_table(tables["abstentions"], ABSTENTIONS),
        "## Transport failures, separate from model abstentions",
        "",
        "A transport failure is a request that still failed after its last allowed attempt. "
        "A model abstention is a result.",
        "",
        *_table(tables["transport"], TRANSPORT),
        "## Pairwise comparisons",
        "",
        "Every difference is the first arm minus the second, in routing decision cost per 1,000 "
        f"tickets, excluding compute; negative means the first arm is lower. Paired bootstrap, "
        f"{iterations} resamples, seed {RANDOM_STATE}, one resample matrix shared by every "
        "interval, 95% percentile interval.",
        "",
    ]
    for name, arm_pairs in ABC_GROUPS:
        rows = [r for r in tables["cost"] if (r["arm_a"], r["arm_b"]) in arm_pairs]
        lines += [f"### {name}", "", *_table(rows, COST)]
    lines += [
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
        *_hypotheses_abc(runs, tables),
        "",
        "## H3 detail: raw ECE where C did not abstain",
        "",
        "ECE is biased upward in small bins, and a percentile interval over resamples inherits "
        "that bias.",
        "",
        *_table(tables["claim"], CALIBRATION_CLAIM),
        "### Accuracy by raw-confidence bin",
        "",
        "10 equal-width bins of [0, 1], the same bins as ECE, over the rows not abstained.",
        "",
        *_table(tables["bins"], CONFIDENCE_BINS),
        "## H4 detail: latency and retries",
        "",
    ]
    lat = tables["latency"]
    if lat:
        lines += [
            f"Retries (sum of attempts - 1, all C-prob rows): {lat['retries']}. Cold starts "
            f"excluded: {lat['cold_excluded']}; transport failures excluded: "
            f"{lat.get('failures_excluded', 0)}.",
            "",
        ]
    counts = tables["argmax"]
    if counts:
        lines += [
            "## argmax of probabilities against choice",
            "",
            f"unique {counts['unique']}, tied {counts['tied']}, differs {counts['differs']}, "
            f"probabilities unusable {counts['unusable']} (C-prob rows, raw).",
            "",
        ]
    lines += [
        "## Break-even compute sensitivity",
        "",
        "Advantage per ticket is mean(cost first - cost second); positive means the second arm "
        "is cheaper before compute. The break-even is that advantage when it is positive and "
        "blank otherwise; per second it is divided by the second arm's mean seconds. Costs are "
        "in handoff units and are never converted to dollars.",
        "",
        *_table(tables["breakeven"], ABC_BREAKEVEN),
        "## Usage",
        "",
        *_table(tables["usage"], USAGE),
        "Latency is wall-clock per ticket for A, the model-side time Ollama reports for B, and "
        "client-side wall-clock of the successful attempt for C; the three are not directly "
        "comparable. Arm C rows exclude transport failures.",
        "",
    ]
    dollars = tables["dollars"]
    if dollars is not None:
        lines += [
            "## Arm C spend",
            "",
            f"${dollars:.4f} per 1,000 tickets: mean reported input tokens x "
            f"${tables['price_per_mtok']} per million (output free), transport failures excluded. "
            "This is reported separately from routing cost and is not converted into handoff "
            "units.",
            "",
        ]
    lines += [
        "## Limits",
        "",
        "- Isotonic calibrators are not refitted inside a bootstrap resample, and cross-fitted "
        "confidences are treated as fixed.",
        "- Inference is not priced in routing cost; the break-even is arithmetic, not a price.",
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
    price_per_mtok: float = DEFAULT_PRICE_PER_MTOK,
) -> list[Path]:
    """Write the CSVs, the optional reliability diagram and ``report.md``; return the paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    abc = runs[0].split == "heldout"
    pairs = _abc_pairs(runs) if abc else None
    tables = {
        "sensitivity": sensitivity_rows(runs),
        "usage": usage_rows(runs),
        "abstentions": abstention_rows(runs),
        "cost": cost_rows(runs, iterations, pairs, ABC_HEADLINE if abc else V2_HEADLINE),
        "accuracy": accuracy_rows(runs, ABC_ARM_PAIRS if abc else None),
        "ece": ece_rows(runs, iterations, pairs),
        "breakeven": breakeven_rows(runs, pairs),
    }
    if abc:
        tables |= {
            "transport": transport_rows(runs),
            "claim": calibration_claim_rows(runs, iterations),
            "bins": confidence_bin_rows(runs),
            "latency": latency_summary(runs),
            "argmax": argmax_counts(runs),
            "dollars": dollars_per_1000(runs, price_per_mtok),
            "price_per_mtok": price_per_mtok,
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
        ("breakeven.csv", ABC_BREAKEVEN if abc else BREAKEVEN, "breakeven"),
    ]
    if abc:
        files += [
            ("transport_failures.csv", TRANSPORT, "transport"),
            ("calibration_claim.csv", CALIBRATION_CLAIM, "claim"),
            ("confidence_bins.csv", CONFIDENCE_BINS, "bins"),
        ]
    written = []
    for name, columns, key in files:
        if tables[key]:
            _write_csv(out / name, columns, tables[key])
            written.append(out / name)
    written += [out / "reliability.png"] if plots else []
    render = _markdown_abc if abc else _markdown
    (out / "report.md").write_text(render(runs, tables, source_note, iterations), encoding="utf-8")
    return [*written, out / "report.md"]
