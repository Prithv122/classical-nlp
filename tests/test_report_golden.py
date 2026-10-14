"""Golden test: the v2 report layout must stay byte-identical when new layouts are added.

The inputs are built deterministically from the synthetic fixture (arm A, and arm B against a
scripted fake), so every number here tests the code and none is a result. To regenerate the
committed outputs after an intended v2 report change, run this module directly:
``uv run python tests/test_report_golden.py``.
"""

from __future__ import annotations

import json
import shutil
import sys
import zlib
from pathlib import Path

from classicalnlp import report, study, tickets
from classicalnlp.router import FakeLLM, LLMArm, LLMClient, TfidfArm, fake_completion

GOLDEN = Path(__file__).parent / "golden" / "report_v2"
NOTE = "synthetic fixture: tests the code, not results"
INTENTS = ("card_arrival", "exchange_rate", "top_up_failed", "age_limit", "atm_support")


def _scripted(query: str):
    h = zlib.crc32(query.encode("utf-8"))
    verbal = 0.5 + (h % 5) / 10
    return fake_completion(INTENTS[h % len(INTENTS)], verbal, logprob=-0.1 - (h % 7) / 10)


def build_runs(directory: Path) -> list[Path]:
    """Six v2-shaped runs (A, B-logprob, B-verbal x raw, isotonic) with split "test"."""
    fit_all, test = tickets.validation_split(tickets.load_fixture())
    fit, val = tickets.validation_split(fit_all)
    a_runs = study.run_arm(TfidfArm(), fit, val, test)
    client = LLMClient(FakeLLM(_scripted))
    b_runs = []
    for variant in ("logprob", "verbal"):
        b_runs += study.run_arm(LLMArm(client, variant), fit, val, test)
    paths = [directory / "runs_A.jsonl", directory / "runs_B.jsonl"]
    study.save_runs(a_runs, paths[0])
    study.save_runs(b_runs, paths[1])
    # Arm A's seconds are wall-clock; replace them so the golden files are reproducible.
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    for row in rows:
        row["seconds"] = 0.001 * (row["index"] + 1)
    paths[0].write_text("".join(json.dumps(row) + "\n" for row in rows))
    return paths


def write_outputs(directory: Path) -> Path:
    paths = build_runs(directory)
    out = directory / "out"
    runs = report.load_runs(paths, final=True)
    report.write_report(runs, out, source_note=NOTE, iterations=500, plots=False)
    return out


def _bytes(path: Path) -> bytes:
    # report.md is written in text mode, so its line endings depend on the platform.
    return path.read_bytes().replace(b"\r\n", b"\n")


def test_v2_report_matches_the_golden_files(tmp_path):
    out = write_outputs(tmp_path)
    expected = sorted(p.name for p in GOLDEN.iterdir())
    assert sorted(p.name for p in out.iterdir()) == expected
    assert "report.md" in expected and len(expected) == 8
    for name in expected:
        assert _bytes(out / name) == _bytes(GOLDEN / name), name


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = write_outputs(Path(tmp))
        shutil.rmtree(GOLDEN, ignore_errors=True)
        GOLDEN.mkdir(parents=True)
        for path in sorted(out.iterdir()):
            (GOLDEN / path.name).write_bytes(_bytes(path))
    print(f"wrote {len(list(GOLDEN.iterdir()))} files to {GOLDEN}", file=sys.stderr)
