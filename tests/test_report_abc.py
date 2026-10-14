"""The arm C report layout: loading, tables, hypothesis statuses. Hand-made rows only; every
value is scripted to test the code and none is a result. No network."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pytest

from classicalnlp import cli, report
from classicalnlp.router.decision_arm import ENDPOINT

PROTOCOL = Path(__file__).parents[1] / "ARM_C_PROTOCOL.md"
N = 40
Y = tuple(i % 4 for i in range(N))
RIGHT = Y
HANDED = (-1,) * N


def _row(arm, cal, i, y_pred, conf, **extra):
    base = {
        "arm": arm, "split": "heldout", "calibration": cal, "index": i, "y_true": Y[i],
        "y_pred": y_pred[i], "confidence": conf[i], "seconds": 0.3, "prompt_tokens": None,
        "cached_prompt_tokens": None, "output_tokens": None,
    }  # fmt: skip
    return base | extra


def _plain(arm, y_pred, conf):
    return [_row(arm, cal, i, y_pred, conf) for cal in report.CALIBRATIONS for i in range(N)]


def _c_rows(arm, y_pred, conf, *, seconds=None, attempts=None, cold=(0,), failed=(), argmax=None):
    pred = [-1 if i in failed else y_pred[i] for i in range(N)]
    confidence = [0.0 if i in failed else conf[i] for i in range(N)]
    rows = []
    for cal in report.CALIBRATIONS:
        for i in range(N):
            ok = i not in failed
            extra = {
                "seconds": seconds[i] if seconds and ok else (0.2 if ok else 0.0),
                "prompt_tokens": 1000 if ok else 0,
                "output_tokens": 5 if ok else 0,
                "endpoint": ENDPOINT,
                "model_version": "jev-1.13.0" if ok else None,
                "transport_failure": not ok,
                "cold_start": i in cold,
                "attempts": (attempts or {}).get(i, 1 if ok else 5),
                "argmax_status": (argmax or {}).get(i, "unique") if ok else None,
            }
            rows.append(_row(arm, cal, i, pred, confidence, **extra))
    return rows


def _write(tmp_path, *row_lists):
    paths = []
    for k, rows in enumerate(row_lists):
        path = tmp_path / f"f{k}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        paths.append(path)
    return paths


def _scenario(tmp_path, a=None, **c_kwargs):
    """A hands everything off by default; C-prob is right on every row at confidence 1.0."""
    a_pred = a or (HANDED, (0.0,) * N)
    files = _write(
        tmp_path,
        _plain("A", *a_pred),
        _plain("B-logprob", RIGHT, (0.9,) * N),
        _plain("B-verbal", RIGHT, (0.8,) * N),
        _c_rows("C-prob", RIGHT, (1.0,) * N, **c_kwargs),
        _c_rows("C-vendor", RIGHT, (0.7,) * N, **c_kwargs),
    )
    return files


def _read(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _report(tmp_path, files=None, **kwargs):
    files = files or _scenario(tmp_path)
    out = tmp_path / "out"
    report.write_report(report.load_runs(files), out, plots=False, iterations=300, **kwargs)
    return out


def test_the_disclaimer_is_the_protocols_sentence_verbatim(tmp_path):
    text = " ".join(PROTOCOL.read_text(encoding="utf-8").split())
    quoted = re.search(r'sentence verbatim: "(This is a new partition-based.*?split\.)"', text)
    assert quoted and quoted.group(1) == report.DISCLAIMER
    md = (_report(tmp_path) / "report.md").read_text()
    assert md.splitlines()[2] == report.DISCLAIMER


def test_the_caveats_are_stated(tmp_path):
    md = (_report(tmp_path) / "report.md").read_text()
    for phrase in [
        "may have seen these texts in pretraining",
        "vendor claims",
        "measured from one machine",
        "Ollama 0.40.2",
        "0.34.4",
        "Only H1 is the headline",
        "not converted into handoff units",
    ]:
        assert phrase in md
    assert "not evaluated" not in md


def test_pairs_signs_and_the_headline_flag(tmp_path):
    out = _report(tmp_path)
    rows = _read(out / "comparison_cost.csv")
    pairs = [(r["arm_a"], r["arm_b"]) for r in rows]
    expected = [
        ("A", "C-prob"), ("B-logprob", "C-prob"), ("A", "B-logprob"), ("A", "C-vendor"),
        ("A", "B-verbal"),
    ]  # fmt: skip
    assert sorted(set(pairs), key=pairs.index) == expected and len(rows) == 5 * 8
    assert {(r["calibration"], r["ratio"]) for r in rows} == {
        (c, str(x)) for c in ("raw", "isotonic") for x in (2, 5, 10, 20)
    }
    (head,) = [r for r in rows if r["headline"] == "true"]
    assert (head["arm_a"], head["arm_b"], head["calibration"], head["ratio"]) == (
        "A", "C-prob", "isotonic", "10",
    )  # fmt: skip
    assert float(head["cost_a"]) == 1000.0 and float(head["cost_b"]) == 0.0
    assert float(head["difference"]) == 1000.0  # first minus second
    assert len(_read(out / "comparison_accuracy.csv")) == 5
    assert len(_read(out / "comparison_ece.csv")) == 10
    breakeven = _read(out / "breakeven.csv")
    assert len(breakeven) == 40 and {"arm_a", "arm_b"} <= set(breakeven[0])
    assert {r["status"] for r in breakeven if r["arm_a"] == "A" and r["arm_b"] == "C-prob"} == {
        "C-prob cheaper before compute"
    }


@pytest.mark.parametrize(
    ("a_conf", "a_pred", "status"),
    [((0.0,) * N, HANDED, "C lower"), ((1.0,) * N, RIGHT, "no clear difference")],
)
def test_h1_reads_the_headline_interval(tmp_path, a_conf, a_pred, status):
    md = (_report(tmp_path, _scenario(tmp_path, (a_pred, a_conf))) / "report.md").read_text()
    (line,) = [x for x in md.splitlines() if x.startswith("- H1")]
    assert f": {status}; difference" in line


def test_h1_says_a_lower_when_c_is_the_costly_arm(tmp_path):
    files = _write(
        tmp_path,
        _plain("A", RIGHT, (1.0,) * N),
        _c_rows("C-prob", HANDED, (0.0,) * N),
    )
    md = (_report(tmp_path, files) / "report.md").read_text()
    (line,) = [x for x in md.splitlines() if x.startswith("- H1")]
    assert ": A lower; difference -1000.0" in line


@pytest.mark.parametrize(
    ("low", "high", "status"),
    [
        (0.0, 0.05, "supported"),
        (0.0, 0.0500001, "inconclusive"),
        (0.05, 0.2, "inconclusive"),
        (0.0500001, 0.2, "not supported"),
    ],
)
def test_h3_status_at_the_boundaries(low, high, status):
    assert report.h3_status(low, high) == status


@pytest.mark.parametrize(
    ("p50", "p95", "status"),
    [
        (0.1, 0.5, "within claim"),
        (0.1, 0.5001, "median within, tail outside"),
        (0.5, 0.9, "median within, tail outside"),
        (0.5001, 0.9, "outside claim"),
    ],
)
def test_h4_status_at_the_boundaries(p50, p95, status):
    assert report.latency_status(p50, p95) == status


def test_h3_uses_the_non_abstained_rows_and_the_shared_resamples(tmp_path):
    out = _report(tmp_path)
    claim = {r["arm"]: r for r in _read(out / "calibration_claim.csv")}
    assert claim["C-prob"]["status"] == "supported" and float(claim["C-prob"]["ece"]) == 0.0
    assert claim["C-vendor"]["status"] == "not supported"  # right every time, says 0.7
    assert float(claim["C-vendor"]["ece"]) == pytest.approx(0.3)
    (tmp_path / "w").mkdir()
    wrong = _write(
        tmp_path / "w", _plain("A", HANDED, (0.0,) * N), _c_rows("C-prob", (1,) * N, (1.0,) * N)
    )
    rows = report.calibration_claim_rows(report.load_runs(wrong), 200)
    assert rows[0]["status"] == "not supported" and rows[0]["n_scored"] == N


def test_h3_leaves_abstained_and_failed_rows_out(tmp_path):
    pred = list(RIGHT)
    pred[1] = -1  # a model abstention
    conf = [1.0] * N
    conf[1] = 0.0
    files = _write(
        tmp_path,
        _plain("A", HANDED, (0.0,) * N),
        _c_rows("C-prob", pred, conf, failed=(2, 3)),
    )
    runs = report.load_runs(files)
    (row,) = report.calibration_claim_rows(runs, 200)
    assert row["n_scored"] == N - 3
    (transport,) = report.transport_rows(runs)
    assert (transport["transport_failures"], transport["model_abstentions"]) == (2, 1)


def test_accuracy_by_confidence_uses_the_ece_bins():
    conf = np.array([0.0, 0.1, 0.15, 0.95, 1.0, 0.5])
    run = report.RunData(
        "C-prob", "heldout", "raw", np.zeros(6, int), np.array([0, 1, 0, 0, 1, 0]), conf,
        None, None, None, None, extras={"transport_failure": [False] * 6},
    )  # fmt: skip
    rows = report.confidence_bin_rows([run])
    assert [r["n"] for r in rows] == [2, 1, 0, 0, 1, 0, 0, 0, 0, 2]
    assert rows[0]["bin"] == "[0.0, 0.1]" and rows[1]["bin"] == "(0.1, 0.2]"
    assert rows[0]["accuracy"] == 0.5 and rows[2]["accuracy"] is None
    assert rows[9]["accuracy"] == 0.5 and len(rows) == 10


def test_h4_excludes_cold_starts_and_failures_and_counts_retries(tmp_path):
    seconds = [0.1 * (1 + i % 3) for i in range(N)]  # 0.1, 0.2, 0.3 repeating
    seconds[0] = 9.0  # the cold start
    files = _scenario(tmp_path, seconds=seconds, failed=(5,), attempts={7: 3, 8: 2})
    runs = report.load_runs(files)
    lat = report.latency_summary(runs)
    kept = np.array([s for i, s in enumerate(seconds) if i not in (0, 5)])
    assert lat["n"] == N - 2
    for key, q in (("p50", 50), ("p95", 95), ("p99", 99)):
        assert lat[key] == pytest.approx(np.percentile(kept, q))
    assert lat["share_within"] == 1.0 and lat["status"] == "within claim"
    assert lat["retries"] == 2 + 1 + 4  # attempts - 1: rows 7 and 8, and the failed row's 5 - 1
    md = (_report(tmp_path, files) / "report.md").read_text()
    assert "Retries (sum of attempts - 1, all C-prob rows): 7" in md
    (h4,) = [x for x in md.splitlines() if x.startswith("- H4")]
    assert ": within claim; p50" in h4


def test_argmax_counts_transport_rows_and_dollars(tmp_path):
    argmax = {1: "tied", 2: "tied", 3: "differs"}
    files = _scenario(tmp_path, argmax=argmax, failed=(4,))
    runs = report.load_runs(files)
    assert report.argmax_counts(runs) == {"unique": N - 4, "tied": 2, "differs": 1, "unusable": 1}
    assert report.dollars_per_1000(runs) == pytest.approx(1000 * 0.042 / 1e6 * 1000)
    md = (_report(tmp_path, files) / "report.md").read_text()
    assert "$0.0420 per 1,000 tickets" in md
    assert "unique 36, tied 2, differs 1, probabilities unusable 1" in md
    transport = _read(tmp_path / "out" / "transport_failures.csv")
    assert {r["arm"]: r["transport_failures"] for r in transport} == {
        "C-prob": "1",
        "C-vendor": "1",
    }


def test_the_abc_files_are_written_and_the_cli_needs_no_final(tmp_path, capsys):
    files = _scenario(tmp_path)
    args = ["route-report", "--runs", *map(str, files), "--out", str(tmp_path / "o"), "--no-plots"]
    assert cli.main(args) == 0
    names = {p.name for p in (tmp_path / "o").iterdir()}
    assert names == {
        "sensitivity.csv", "usage.csv", "abstentions.csv", "comparison_cost.csv",
        "comparison_accuracy.csv", "comparison_ece.csv", "breakeven.csv",
        "transport_failures.csv", "calibration_claim.csv", "confidence_bins.csv", "report.md",
    }  # fmt: skip
    again = tmp_path / "again"
    assert cli.main([*args[:-3], "--out", str(again), "--no-plots"]) == 0
    for name in names:
        assert (tmp_path / "o" / name).read_bytes() == (again / name).read_bytes()


def test_usage_for_arm_c_leaves_transport_failures_out(tmp_path):
    runs = report.load_runs(_scenario(tmp_path, failed=(5,)))
    rows = {r["arm"]: r for r in report.usage_rows(runs)}
    assert rows["C-prob"]["prompt_tokens_mean"] == 1000.0 and rows["C-prob"]["n"] == N


@pytest.mark.parametrize("split", ["val", "test"])
def test_arm_c_rows_on_another_split_are_refused(tmp_path, split):
    rows = [{**r, "split": split} for r in _c_rows("C-prob", RIGHT, (1.0,) * N)]
    with pytest.raises(ValueError, match="arm C is reported on the held-out partition only"):
        report.load_runs(_write(tmp_path, rows))


def test_heldout_a_only_runs_are_accepted_and_mixed_splits_are_not(tmp_path):
    assert report.load_runs(_write(tmp_path, _plain("A", RIGHT, (0.9,) * N)))
    mixed = [{**r, "split": "val"} for r in _plain("B-logprob", RIGHT, (0.9,) * N)]
    with pytest.raises(ValueError, match="runs span two splits"):
        report.load_runs(_write(tmp_path, _plain("A", RIGHT, (0.9,) * N), mixed))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"endpoint": None}, "endpoint must be a string"),
        ({"attempts": True}, "attempts an integer"),
        ({"argmax_status": "maybe"}, "argmax_status must be"),
        ({"model_version": 3}, "model_version must be"),
        ({"transport_failure": True}, "transport failure must be ABSTAIN"),
    ],
)
def test_c_rows_are_validated(tmp_path, change, message):
    rows = _c_rows("C-prob", RIGHT, (1.0,) * N)
    rows[0] = {**rows[0], **change}
    with pytest.raises(ValueError, match=message):
        report.load_runs(_write(tmp_path, rows))
    gone = {k: v for k, v in _c_rows("C-prob", RIGHT, (1.0,) * N)[0].items() if k != "endpoint"}
    with pytest.raises(ValueError, match="missing key 'endpoint'"):
        report.load_runs(_write(tmp_path, [gone]))


# -- review fixes: the run's price, H2's line, H3 against the shared resample matrix --------


def test_c_dollars_use_the_price_the_runs_were_made_at(tmp_path):
    files = _scenario(tmp_path)
    assert report.dollars_per_1000(report.load_runs(files), 0.05) == pytest.approx(0.05)
    md = (_report(tmp_path, files, price_per_mtok=0.05) / "report.md").read_text()
    assert "$0.0500 per 1,000 tickets" in md and "$0.05 per million" in md
    args = ["route-report", "--runs", *map(str, files), "--out", str(tmp_path / "o")]
    assert cli.main([*args, "--no-plots", "--price-per-mtok", "0.05"]) == 0
    assert "$0.0500 per 1,000 tickets" in (tmp_path / "o" / "report.md").read_text()


def test_h2_reads_the_a_minus_c_prob_isotonic_ece_interval(tmp_path):
    files = _scenario(tmp_path, a=(RIGHT, (0.5,) * N))  # A right at 0.5, C-prob right at 1.0
    md = (_report(tmp_path, files) / "report.md").read_text()
    line = next(x for x in md.splitlines() if x.startswith("- H2 "))
    assert line.endswith("C lower; difference +0.500, 95% CI [+0.500, +0.500]")


def test_h3_interval_is_the_shared_resample_matrix_over_non_abstained_rows(tmp_path):
    from classicalnlp import calibrate, compare
    from classicalnlp.evaluate import RANDOM_STATE

    pred = tuple((y + 1) % 4 if i % 5 == 0 else y for i, y in enumerate(Y))
    conf = tuple(0.6 if i % 3 == 0 else 1.0 for i in range(N))
    files = _write(
        tmp_path, _plain("A", HANDED, (0.0,) * N), _c_rows("C-prob", pred, conf, failed=(7,))
    )
    runs = report.load_runs(files)
    row = report.calibration_claim_rows(runs, 200)[0]
    run = next(r for r in runs if r.arm == "C-prob" and r.calibration == "raw")
    values = []
    for idx in compare._indices(N, 200, RANDOM_STATE):
        idx = idx[run.y_pred[idx] != -1]
        values.append(
            calibrate.top_label_ece(run.y_true[idx], run.y_pred[idx], run.confidence[idx])
        )
    low, high = np.percentile(values, [2.5, 97.5])
    assert row["n_scored"] == N - 1
    assert (row["ci_low"], row["ci_high"]) == (pytest.approx(low), pytest.approx(high))
    assert row["ci_low"] < row["ci_high"]
