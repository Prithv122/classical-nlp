"""route-report: loading, the tables and the CLI. Hand-made rows and the synthetic fixture only;
values test the code and are not results. No network."""

from __future__ import annotations

import csv
import json
import sys

import numpy as np
import pytest

from classicalnlp import cli, plots, report, study, tickets
from classicalnlp.router import FakeLLM, TfidfArm, fake_completion

PNG = b"\x89PNG\r\n\x1a\n"
NOTE = "synthetic fixture: tests the code, not results"


def _rows(arm="A", y_true=(0, 1, 2, 3), y_pred=(0, 1, 2, 0), conf=(0.9, 0.8, 0.7, 0.6), **kw):
    split, cals, seconds = kw.get("split", "val"), kw.get("cals", ("raw", "isotonic")), kw.get("s")
    return [
        {
            "arm": arm,
            "split": split,
            "calibration": cal,
            "index": i,
            "y_true": y_true[i],
            "y_pred": y_pred[i],
            "confidence": conf[i],
            "seconds": seconds,
            "prompt_tokens": None,
            "cached_prompt_tokens": None,
            "output_tokens": None,
        }
        for cal in cals
        for i in range(len(y_true))
    ]


def _files(tmp_path, *row_lists):
    paths = []
    for k, rows in enumerate(row_lists):
        path = tmp_path / f"f{k}.jsonl"
        path.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows))
        paths.append(path)
    return paths


def _read(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


REJECTED = {
    "no paths": (lambda t: [], "no run files"),
    "missing path": (lambda t: [t / "nope.jsonl"], "cannot read run file .*nope.jsonl"),
    "bad json": (lambda t: _files(t, ["{nope"]), r"f0.jsonl:1: not valid JSON"),
    "missing key": (
        lambda t: _files(t, [{k: v for k, v in _rows()[0].items() if k != "seconds"}]),
        "f0.jsonl:1: missing key 'seconds'",
    ),
    "unknown arm": (lambda t: _files(t, _rows(arm="Z")), "unknown arm"),
    "bad calibration": (
        lambda t: _files(t, [{**r, "calibration": "platt"} for r in _rows()]),
        "raw or isotonic",
    ),
    "confidence": (lambda t: _files(t, _rows(conf=(0.9, 0.8, 0.7, 1.5))), r"outside \[0, 1\]"),
    "abstention": (lambda t: _files(t, _rows(y_pred=(0, 1, 2, -1))), "must have confidence 0.0"),
    "gap": (lambda t: _files(t, [r for r in _rows() if r["index"] != 1]), "indices"),
    "repeated index": (lambda t: _files(t, _rows() + _rows()[:1]), "indices"),
    "duplicate run": (lambda t: _files(t, _rows(), _rows()), "duplicate run"),
    "two splits": (
        lambda t: _files(t, _rows(), _rows(arm="B-logprob", split="test")),
        "runs span two splits",
    ),
    "calibration changes predictions": (
        lambda t: _files(t, _rows(cals=("raw",)) + _rows(y_pred=(1, 1, 2, 0), cals=("isotonic",))),
        "calibration changed the predictions",
    ),
    "different y_true": (
        lambda t: _files(t, _rows(), _rows(arm="B-logprob", y_true=(0, 1, 2, 2))),
        "do not share the same tickets",
    ),
    "different n": (
        lambda t: _files(
            t, _rows(), _rows(arm="B-logprob", y_true=(0, 1, 2), y_pred=(0, 1, 2), conf=(1, 1, 1))
        ),
        "do not share the same tickets",
    ),
}


@pytest.mark.parametrize("case", REJECTED)
def test_load_runs_rejects(tmp_path, case):
    make, message = REJECTED[case]
    with pytest.raises(ValueError, match=message):
        report.load_runs(make(tmp_path))


def test_test_split_needs_final(tmp_path):
    paths = _files(tmp_path, _rows(split="test"))
    with pytest.raises(ValueError, match="pass --final: these runs are on the official test split"):
        report.load_runs(paths)
    runs = report.load_runs(paths, final=True)
    assert [(r.arm, r.calibration) for r in runs] == [("A", "raw"), ("A", "isotonic")]


def test_mcnemar_by_hand_through_the_report(tmp_path):
    y = (0,) * 6
    paths = _files(
        tmp_path,
        _rows(y_true=y, y_pred=(0, 0, 0, 1, 0, 1), conf=(0.5,) * 6),
        _rows(arm="B-logprob", y_true=y, y_pred=(1, 1, 1, 0, 0, 1), conf=(0.5,) * 6),
    )
    report.write_report(report.load_runs(paths), tmp_path / "out", plots=False)
    (row,) = _read(tmp_path / "out" / "comparison_accuracy.csv")
    assert (row["arm_b"], row["only_a"], row["only_b"]) == ("B-logprob", "3", "1")
    assert float(row["p_value"]) == pytest.approx(0.625)


def test_sensitivity_csv_equals_the_study(tmp_path):
    fit, val = tickets.validation_split(tickets.load_fixture())
    runs = study.run_arm(TfidfArm(), fit, val)
    study.save_runs(runs, tmp_path / "a.jsonl")
    report.write_report(report.load_runs([tmp_path / "a.jsonl"]), tmp_path / "out", plots=False)
    rows = _read(tmp_path / "out" / "sensitivity.csv")
    assert len(rows) == 8
    for run in runs:
        for ratio, b in run.costs.items():
            (row,) = [
                r for r in rows if r["calibration"] == run.calibration and r["ratio"] == str(ratio)
            ]
            for key, want in [
                ("routing_cost_per_1000", b.cost_per_1000),
                ("coverage", b.coverage),
                ("auto_routed", b.auto_routed),
                ("misrouted", b.misrouted),
                ("ece", run.ece),
                ("brier", run.brier),
            ]:
                assert float(row[key]) == pytest.approx(want, rel=1e-9)
            assert int(row["abstained"]) == int((run.y_pred == -1).sum())


@pytest.fixture(scope="module")
def two_arm(tmp_path_factory):
    """Arm A and arm B through the route command (B against a scripted fake), then report."""
    tmp = tmp_path_factory.mktemp("two_arm")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            cli,
            "_make_provider",
            lambda args: FakeLLM(lambda q: fake_completion("card_arrival", 0.9, logprob=-0.2)),
        )
        route = ["route", "--split", "val", "--fixture"]
        assert cli.main([*route, "--arm", "A", "--save", str(tmp / "a.jsonl")]) == 0
        b = ["--arm", "B", "--cache", str(tmp / "c.jsonl"), "--save", str(tmp / "b.jsonl")]
        assert cli.main([*route, *b]) == 0
    assert len((tmp / "a.jsonl").read_text().splitlines()) == 24
    assert len((tmp / "b.jsonl").read_text().splitlines()) == 48
    args = ["--runs", str(tmp / "a.jsonl"), str(tmp / "b.jsonl"), "--no-plots"]
    assert cli.main(["route-report", *args, "--out", str(tmp / "out"), "--source-note", NOTE]) == 0
    return tmp


def test_two_arm_files(two_arm):
    out = two_arm / "out"
    counts = {
        "sensitivity": 24,
        "comparison_cost": 16,
        "comparison_accuracy": 2,
        "comparison_ece": 4,
        "breakeven": 16,
        "usage": 3,
        "abstentions": 3,
    }
    for name, count in counts.items():
        assert len(_read(out / f"{name}.csv")) == count, name
    assert (out / "report.md").exists() and not (out / "reliability.png").exists()
    (head,) = [r for r in _read(out / "comparison_cost.csv") if r["headline"] == "true"]
    assert (head["arm_b"], head["calibration"], head["ratio"]) == ("B-logprob", "isotonic", "10")


def test_two_arm_report_is_deterministic(two_arm):
    args = ["--runs", str(two_arm / "a.jsonl"), str(two_arm / "b.jsonl"), "--no-plots"]
    assert (
        cli.main(["route-report", *args, "--out", str(two_arm / "again"), "--source-note", NOTE])
        == 0
    )
    for path in (two_arm / "out").iterdir():
        assert path.read_bytes() == (two_arm / "again" / path.name).read_bytes()


def test_two_arm_report_text(two_arm):
    text = (two_arm / "out" / "report.md").read_text()
    assert NOTE in "\n".join(text.splitlines()[:5])
    for phrase in [
        "excluding compute",
        "Arm C (dedicated decision model): not evaluated",
        "handoff units",
        "arm B's is the model-side time Ollama reports with model load excluded",
        "- H1 (",
        "- H2 (",
    ]:
        assert phrase in text
    assert "$" not in text and "total cost" not in text.lower()


def test_a_only_run_skips_the_comparison(two_arm, tmp_path):
    runs = report.load_runs([two_arm / "a.jsonl"])
    written = report.write_report(runs, tmp_path, plots=False)
    names = sorted(p.name for p in written)
    assert names == ["abstentions.csv", "report.md", "sensitivity.csv", "usage.csv"]
    assert len(_read(tmp_path / "sensitivity.csv")) == 8
    assert "comparison skipped" in (tmp_path / "report.md").read_text()
    assert "source not recorded in the run files" in (tmp_path / "report.md").read_text()


def test_a_only_run_with_plots(two_arm, tmp_path):
    pytest.importorskip("matplotlib")
    report.write_report(report.load_runs([two_arm / "a.jsonl"]), tmp_path)
    assert (tmp_path / "reliability.png").read_bytes()[:8] == PNG


def test_breakeven_by_hand_and_swapped(tmp_path):
    y = (0, 1, 2, 3, 4)
    handed = {"y_pred": (-1,) * 5, "conf": (0.0,) * 5}
    right = {"y_pred": y, "conf": (1.0,) * 5}
    for a, b, status in [
        (handed, right, "B cheaper before compute"),
        (right, handed, "A cheaper before compute"),
    ]:
        paths = _files(tmp_path, _rows("A", y, **a), _rows("B-logprob", y, **b, s=2.0))
        report.write_report(report.load_runs(paths), tmp_path / status, plots=False)
        rows = _read(tmp_path / status / "breakeven.csv")
        assert len(rows) == 8 and {r["status"] for r in rows} == {status}
        for r in rows:
            if status.startswith("B"):
                assert [float(r[k]) for k in ("advantage_per_ticket", "breakeven_per_ticket")] == [
                    1.0,
                    1.0,
                ]
                assert float(r["breakeven_per_second"]) == 0.5
            else:
                assert float(r["advantage_per_ticket"]) == -1.0
                assert r["breakeven_per_ticket"] == r["breakeven_per_second"] == ""


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing", "nope.jsonl"),
        ("test", "--final"),
        ("iterations", "--iterations"),
        ("twice", "duplicate run"),
    ],
)
def test_cli_errors_exit_2(tmp_path, capsys, case, message):
    split = "test" if case == "test" else "val"
    (path,) = _files(tmp_path, _rows(split=split))
    runs = {"missing": [tmp_path / "nope.jsonl"], "twice": [path, path]}.get(case, [path])
    extra = ["--iterations", "10"] if case == "iterations" else []
    code = cli.main(
        ["route-report", "--runs", *map(str, runs), "--out", str(tmp_path / "o"), *extra]
    )
    assert code == 2
    assert message in capsys.readouterr().err


def test_missing_matplotlib_is_a_clear_error(tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(ImportError, match="router extra"):
        plots.save_reliability(
            [("A raw", np.array([0]), np.array([0]), np.array([0.5]))], tmp_path / "r.png"
        )
    (path,) = _files(tmp_path, _rows())
    assert cli.main(["route-report", "--runs", str(path), "--out", str(tmp_path / "o")]) == 2
    assert "router extra" in capsys.readouterr().err
