"""The audit script's abc mode and its unchanged v2 mode, on hand-made run files. The audit
computes no statistic, and these rows are scripted to exercise it. No network."""

from __future__ import annotations

import hashlib
import json

import pytest

N = 25
LABELS = [i % 5 for i in range(N)]
CALS = ("raw", "isotonic")
EXTRA = {
    "endpoint": "typesafe:https://api.typesafe.ai/v1/systemone",
    "model_version": "jev-1.13.0",
    "transport_failure": False,
    "cold_start": False,
    "attempts": 1,
    "argmax_status": "unique",
}


@pytest.fixture
def audit(load_script):
    return load_script("audit_runs")


def _rows(arm, split="heldout", labels=LABELS):
    rows = []
    for cal in CALS:
        for i in range(N):
            row = {
                "arm": arm, "split": split, "calibration": cal, "index": i, "y_true": labels[i],
                "y_pred": labels[i], "confidence": 0.9, "seconds": 0.2, "prompt_tokens": None,
                "cached_prompt_tokens": None, "output_tokens": None,
            }  # fmt: skip
            if arm.startswith("B"):
                row |= {"prompt_tokens": 600, "cached_prompt_tokens": 0, "output_tokens": 14}
            if arm.startswith("C"):
                row |= {"prompt_tokens": 120, "output_tokens": 8, **EXTRA}
            rows.append(row)
    return rows


def _all(split="heldout", arms=("A", "B-logprob", "B-verbal", "C-prob", "C-vendor")):
    return [r for arm in arms for r in _rows(arm, split)]


FAILED = {
    "transport_failure": True, "model_version": None, "y_pred": -1, "confidence": 0.0,
    "seconds": 0.0, "prompt_tokens": 0, "output_tokens": 0, "argmax_status": None, "attempts": 5,
}  # fmt: skip


def _write(tmp_path, rows):
    path = tmp_path / "runs.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _prefix_checks(tmp_path):
    """Valid --cache-check entries for arm B and arm C, which a held-out abc audit requires."""
    checks = []
    for name in ("b_default.jsonl", "c_default.jsonl"):
        path = tmp_path / name
        path.write_text(json.dumps({"key": name}) + "\n")
        data = path.read_bytes()
        checks += ["--cache-check", str(path), str(len(data)), hashlib.sha256(data).hexdigest()]
    return checks


def _run(audit, tmp_path, rows, monkeypatch, *extra, split="heldout", partition="abc"):
    monkeypatch.setattr(audit, "expected_labels", lambda *args, **kwargs: LABELS)
    argv = ["--partition", partition, "--split", split, "--data-dir", "d"]
    if partition == "abc" and split == "heldout" and "--cache-check" not in extra:
        extra = (*extra, *_prefix_checks(tmp_path))
    return audit.main([*argv, "--runs", str(_write(tmp_path, rows)), *extra])


def test_a_heldout_audit_needs_both_cache_prefix_checks(audit, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(audit, "expected_labels", lambda *args, **kwargs: LABELS)
    argv = ["--partition", "abc", "--split", "heldout", "--data-dir", "d"]
    argv += ["--runs", str(_write(tmp_path, _all()))]
    one = _prefix_checks(tmp_path)[:4]
    for extra in ([], one):
        with pytest.raises(SystemExit):
            audit.main([*argv, *extra])
        assert "needs --cache-check for arm B and arm C" in capsys.readouterr().err


def test_a_clean_abc_run_set_passes(audit, tmp_path, monkeypatch, capsys):
    assert _run(audit, tmp_path, _all(), monkeypatch) == 0
    out = capsys.readouterr().out
    assert "0 failed" in out and "exactly the ten runs" in out and "FAIL" not in out


def test_the_audit_uses_the_partition_labels_for_the_requested_split(audit, monkeypatch):
    seen = []
    monkeypatch.setattr(audit.partition, "load_train", lambda directory: "train")
    corpora = tuple(type("C", (), {"labels": [k]})() for k in (1, 2, 3))
    monkeypatch.setattr(audit.partition, "abc_corpora", lambda train: seen.append(train) or corpora)
    assert audit.expected_labels("val", "d", "abc") == [2]
    assert audit.expected_labels("heldout", "d", "abc") == [3]
    assert seen == ["train", "train"]


def _mutations():
    base = _all()
    c = [i for i, r in enumerate(base) if r["arm"] == "C-prob" and r["calibration"] == "raw"]

    def edit(at, **change):
        rows = [dict(r) for r in base]
        rows[at].update(change)
        return rows

    def failures(count):
        rows = [dict(r) for r in base]
        for r in rows:
            if r["arm"] == "C-prob" and r["index"] < count:
                r |= FAILED
        return rows

    def without(key):
        rows = [dict(r) for r in base]
        del rows[c[0]][key]
        return rows

    return {
        "missing extra key on a C row": (without("endpoint"), "exactly the expected keys"),
        "extra key on an A row": (edit(0, endpoint="x"), "exactly the expected keys"),
        "a run is missing": (
            [r for r in base if (r["arm"], r["calibration"]) != ("C-vendor", "isotonic")],
            "exactly the ten runs",
        ),
        "an index is repeated": (edit(c[1], index=0), "each once"),
        "wrong y_true": (edit(c[0], y_true=4), "y_true equals"),
        "wrong split": (edit(c[0], split="val"), "expected split"),
        "wrong model version": (edit(c[3], model_version="jev-1.14.0"), "one reported model"),
        "null model version on a success": (edit(c[3], model_version=None), "one reported model"),
        "two endpoints": (edit(c[3], endpoint="cloudflare:other"), "one endpoint"),
        "failure row not abstained": (
            edit(c[2], transport_failure=True, model_version=None),
            "transport failure is ABSTAIN",
        ),
        "failure with confidence": (
            edit(c[2], transport_failure=True, model_version=None, y_pred=-1, confidence=0.5),
            "transport failure is ABSTAIN",
        ),
        "too many transport failures": (failures(21), "at most 20 transport failures"),
        "confidence above one": (edit(c[0], confidence=1.5), "finite number in [0, 1]"),
        "abstention with confidence": (edit(0, y_pred=-1, confidence=0.2), "abstention has"),
        "bad attempts": (edit(c[0], attempts=0), "right types"),
        "bad argmax status": (edit(c[0], argmax_status="maybe"), "right types"),
        "cached tokens on C": (edit(c[0], cached_prompt_tokens=0), "no cached count"),
        "tokens on A": (edit(0, prompt_tokens=5), "arm A rows carry no token counts"),
        "calibration changed predictions": (edit(c[0], y_pred=3), "predict identically"),
    }


@pytest.mark.parametrize("case", list(_mutations()))
def test_abc_failures_are_caught(audit, tmp_path, monkeypatch, capsys, case):
    rows, fragment = _mutations()[case]
    assert _run(audit, tmp_path, rows, monkeypatch) == 1
    failed = [x for x in capsys.readouterr().out.splitlines() if x.startswith("FAIL")]
    assert any(fragment in line for line in failed), failed


def test_twenty_transport_failures_are_allowed_and_need_no_timing(audit, tmp_path, monkeypatch):
    rows = [dict(r) for r in _all()]
    for r in rows:
        if r["arm"] == "C-prob" and r["index"] < 20:
            r |= FAILED
    assert _run(audit, tmp_path, rows, monkeypatch) == 0


def test_abc_split_choices(audit, tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _run(audit, tmp_path, _all(), monkeypatch, split="test")
    with pytest.raises(SystemExit):
        _run(audit, tmp_path, _all(), monkeypatch, split="heldout", partition="v2")


def test_the_v2_mode_is_unchanged(audit, tmp_path, monkeypatch, capsys):
    rows = [r for arm in ("A", "B-logprob", "B-verbal") for r in _rows(arm, "test")]
    assert _run(audit, tmp_path, rows, monkeypatch, split="test", partition="v2") == 0
    out = capsys.readouterr().out
    assert "exactly the six runs" in out and "0 failed" in out
    with_c = [*rows, *_rows("C-prob", "test")]
    assert _run(audit, tmp_path, with_c, monkeypatch, split="test", partition="v2") == 1


def _cache(tmp_path, name, lines):
    path = tmp_path / name
    path.write_text("".join(json.dumps({"key": k}) + "\n" for k in lines))
    return path


def test_cache_checks_are_repeatable_and_catch_a_changed_prefix(
    audit, tmp_path, monkeypatch, capsys
):
    b = _cache(tmp_path, "b.jsonl", ["a", "b"])
    c = _cache(tmp_path, "c.jsonl", ["x", "y"])

    def spec(path):
        data = path.read_bytes()
        return [str(path), str(len(data)), hashlib.sha256(data).hexdigest()]

    for path in (b, c):  # append after recording: still a valid prefix
        path.write_text(path.read_text() + json.dumps({"key": "later"}) + "\n")
    good = ["--cache-check", *spec(_cache(tmp_path, "b0.jsonl", ["a", "b"]))]
    b0 = tmp_path / "b0.jsonl"
    b0.write_text(b0.read_text() + json.dumps({"key": "later"}) + "\n")
    checks = [*good, "--cache-check", *spec(_cache(tmp_path, "c0.jsonl", ["x", "y"]))]
    assert _run(audit, tmp_path, _all(), monkeypatch, *checks) == 0
    out = capsys.readouterr().out
    assert "b0.jsonl: cache is append-only" in out and "c0.jsonl: cache is append-only" in out
    b0.write_text(json.dumps({"key": "tampered"}) + "\n")
    assert _run(audit, tmp_path, _all(), monkeypatch, *checks) == 1
    assert "FAIL  " in capsys.readouterr().out
    with pytest.raises(SystemExit):
        _run(audit, tmp_path, _all(), monkeypatch, "--cache-check", "p", "many", "d")


def test_the_original_cache_flags_still_work(audit, tmp_path, monkeypatch, capsys):
    cache = _cache(tmp_path, "b.jsonl", ["a"])
    data = cache.read_bytes()
    flags = ["--cache", str(cache), "--cache-prefix-bytes", str(len(data))]
    flags += ["--cache-prefix-sha256", hashlib.sha256(data).hexdigest()]
    assert _run(audit, tmp_path, _all(), monkeypatch, *flags) == 0
    assert "PASS  cache is append-only" in capsys.readouterr().out
