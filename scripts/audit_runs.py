"""Integrity audit of saved per-ticket router runs. Reads files; changes nothing.

Prints PASS or FAIL per check and exits 1 on any FAIL. It reports no accuracy, cost or
calibration figure: those belong to the report, which is read only after these checks pass.

    uv run python scripts/audit_runs.py --split val --data-dir data/raw/banking77 \
        --runs data/interim/runs_val_A.jsonl data/interim/runs_val_B.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from classicalnlp import tickets

EXPECTED_RUNS = {
    (arm, calibration)
    for arm in ("A", "B-logprob", "B-verbal")
    for calibration in ("raw", "isotonic")
}
KEYS = {
    "arm",
    "split",
    "calibration",
    "index",
    "y_true",
    "y_pred",
    "confidence",
    "seconds",
    "prompt_tokens",
    "cached_prompt_tokens",
    "output_tokens",
}
ABSTAIN = -1


def expected_labels(split: str, data_dir: str) -> list[int]:
    train, official_test = tickets.load_banking77(data_dir)
    if split == "test":
        return list(official_test.labels)
    return list(tickets.validation_split(train)[1].labels)


def load_rows(paths: list[str]) -> tuple[list[dict], list[str]]:
    rows, problems = [], []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    problems.append(f"{path}:{number} is not JSON")
                    continue
                if not isinstance(row, dict) or set(row) != KEYS:
                    problems.append(f"{path}:{number} has the wrong keys")
                    continue
                rows.append(row)
    return rows, problems


def check_runs(rows: list[dict], split: str, labels: list[int]) -> list[tuple[str, bool, str]]:
    n = len(labels)
    results = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))

    record("every row is on the expected split", all(r["split"] == split for r in rows))
    by_run: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_run[(row["arm"], row["calibration"])].append(row)
    record(
        "exactly the six runs A, B-logprob, B-verbal x raw, isotonic",
        set(by_run) == EXPECTED_RUNS,
        f"found {sorted(by_run)}",
    )
    record(
        f"each run has {n} rows with indices 0..{n - 1}, each once",
        all(sorted(r["index"] for r in rs) == list(range(n)) for rs in by_run.values()),
    )
    ordered = {key: sorted(rs, key=lambda r: r["index"]) for key, rs in by_run.items()}
    record(
        "y_true equals the pinned split's labels in every run",
        all([r["y_true"] for r in rs] == labels for rs in ordered.values()),
    )
    record(
        "raw and isotonic runs of an arm predict identically",
        all(
            [r["y_pred"] for r in ordered[(arm, "raw")]]
            == [r["y_pred"] for r in ordered[(arm, "isotonic")]]
            for arm in ("A", "B-logprob", "B-verbal")
            if (arm, "raw") in ordered and (arm, "isotonic") in ordered
        ),
    )
    confidences = [r["confidence"] for r in rows]
    record(
        "every confidence is a finite number in [0, 1]",
        all(isinstance(c, float | int) and math.isfinite(c) and 0 <= c <= 1 for c in confidences),
    )
    record(
        "every abstention has confidence 0.0",
        all(r["confidence"] == 0.0 for r in rows if r["y_pred"] == ABSTAIN),
    )
    record(
        "seconds is a positive finite number in every row",
        all(isinstance(r["seconds"], float) and 0 < r["seconds"] < math.inf for r in rows),
    )
    tokens = ("prompt_tokens", "cached_prompt_tokens", "output_tokens")
    record(
        "arm B rows carry integer token counts",
        all(isinstance(r[k], int) and r[k] >= 0 for r in rows if r["arm"] != "A" for k in tokens),
    )
    record(
        "arm A rows carry no token counts",
        all(r[k] is None for r in rows if r["arm"] == "A" for k in tokens),
    )
    return results


def check_cache(path: str, prefix_bytes: int, prefix_sha256: str) -> list[tuple[str, bool, str]]:
    data = Path(path).read_bytes()
    lines = [line for line in data.decode("utf-8", errors="replace").splitlines() if line.strip()]
    bad = 0
    keys = []
    for line in lines:
        try:
            keys.append(json.loads(line)["key"])
        except (json.JSONDecodeError, KeyError, TypeError):
            bad += 1
    prefix_ok = (
        len(data) >= prefix_bytes
        and hashlib.sha256(data[:prefix_bytes]).hexdigest() == prefix_sha256
    )
    return [
        ("cache is append-only: the recorded prefix is byte-identical", prefix_ok, ""),
        ("every cache line parses and has a key", bad == 0, f"{bad} bad lines"),
        ("cache keys are unique", len(keys) == len(set(keys)), ""),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--cache", help="Arm B response cache to check")
    parser.add_argument("--cache-prefix-bytes", type=int)
    parser.add_argument("--cache-prefix-sha256")
    args = parser.parse_args(argv)

    rows, problems = load_rows(args.runs)
    results = [
        ("every line is JSON with exactly the expected keys", not problems, "; ".join(problems[:3]))
    ]
    results += check_runs(rows, args.split, expected_labels(args.split, args.data_dir))
    if args.cache:
        if args.cache_prefix_bytes is None or args.cache_prefix_sha256 is None:
            parser.error("--cache needs --cache-prefix-bytes and --cache-prefix-sha256")
        results += check_cache(args.cache, args.cache_prefix_bytes, args.cache_prefix_sha256)

    for name, ok, detail in results:
        note = f"  ({detail})" if detail and not ok else ""
        print(f"{'PASS' if ok else 'FAIL'}  {name}{note}")
    failed = sum(not ok for _, ok, _ in results)
    print(f"{len(results) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
