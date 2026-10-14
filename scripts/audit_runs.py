"""Integrity audit of saved per-ticket router runs. Reads files; changes nothing.

Prints PASS or FAIL per check and exits 1 on any FAIL. It reports no accuracy, cost or
calibration figure: those belong to the report, which is read only after these checks pass.

    uv run python scripts/audit_runs.py --split val --data-dir data/raw/banking77 \
        --runs data/interim/runs_val_A.jsonl data/interim/runs_val_B.jsonl

The arm C study adds ``--partition abc`` (``--split val`` or ``heldout``, five arms) and
``--cache-check PATH BYTES SHA256`` (repeatable) for the B and C response caches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from classicalnlp import partition, tickets

V2_ARMS = ("A", "B-logprob", "B-verbal")
ABC_ARMS = (*V2_ARMS, "C-prob", "C-vendor")
CALIBRATIONS = ("raw", "isotonic")
EXPECTED_RUNS = {(arm, calibration) for arm in V2_ARMS for calibration in CALIBRATIONS}
C_KEYS = {
    "endpoint",
    "model_version",
    "transport_failure",
    "cold_start",
    "attempts",
    "argmax_status",
}
REQUESTED_MODEL = "jev-1.13.0"
MAX_TRANSPORT_FAILURES = 20
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


def expected_labels(split: str, data_dir: str, layout: str = "v2") -> list[int]:
    if layout == "abc":
        # Train file only; abc_corpora also verifies the frozen held-out file's hash.
        _, val, heldout = partition.abc_corpora(partition.load_train(data_dir))
        return list((val if split == "val" else heldout).labels)
    train, official_test = tickets.load_banking77(data_dir)
    if split == "test":
        return list(official_test.labels)
    return list(tickets.validation_split(train)[1].labels)


def load_rows(paths: list[str], abc: bool = False) -> tuple[list[dict], list[str]]:
    rows, problems = [], []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    problems.append(f"{path}:{number} is not JSON")
                    continue
                is_c = abc and isinstance(row, dict) and str(row.get("arm")).startswith("C-")
                if not isinstance(row, dict) or set(row) != (KEYS | C_KEYS if is_c else KEYS):
                    problems.append(f"{path}:{number} has the wrong keys")
                    continue
                rows.append(row)
    return rows, problems


def check_runs(
    rows: list[dict], split: str, labels: list[int], arms: tuple[str, ...] = V2_ARMS
) -> list[tuple[str, bool, str]]:
    n = len(labels)
    abc = arms == ABC_ARMS
    expected_runs = {(arm, calibration) for arm in arms for calibration in CALIBRATIONS}
    results = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))

    record("every row is on the expected split", all(r["split"] == split for r in rows))
    by_run: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_run[(row["arm"], row["calibration"])].append(row)
    record(
        "exactly the ten runs A, B-logprob, B-verbal, C-prob, C-vendor x raw, isotonic"
        if abc
        else "exactly the six runs A, B-logprob, B-verbal x raw, isotonic",
        set(by_run) == expected_runs,
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
            for arm in arms
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
    timed = [r for r in rows if not r.get("transport_failure")]  # a failure has no timing
    record(
        "seconds is a positive finite number in every row",
        all(isinstance(r["seconds"], float) and 0 < r["seconds"] < math.inf for r in timed),
    )
    tokens = ("prompt_tokens", "cached_prompt_tokens", "output_tokens")
    is_b = [r for r in rows if r["arm"] != "A" and not r["arm"].startswith("C")]
    record(
        "arm B rows carry integer token counts",
        all(isinstance(r[k], int) and r[k] >= 0 for r in is_b for k in tokens),
    )
    record(
        "arm A rows carry no token counts",
        all(r[k] is None for r in rows if r["arm"] == "A" for k in tokens),
    )
    if abc:
        results += check_c_runs(rows, ordered)
    return results


def check_c_runs(rows: list[dict], ordered: dict) -> list[tuple[str, bool, str]]:
    """The arm C checks: one endpoint and version per run, failure rows, the G5 limit."""
    c_runs = {key: rs for key, rs in ordered.items() if key[0].startswith("C-")}
    c_rows = [r for rs in c_runs.values() for r in rs]

    def per_run(test) -> bool:
        return all(test(rs) for rs in c_runs.values())

    def one_version(rs: list[dict]) -> bool:
        done = [r for r in rs if not r["transport_failure"]]
        return {r["model_version"] for r in done} == {REQUESTED_MODEL}

    int_tokens = (
        isinstance(r["prompt_tokens"], int) and isinstance(r["output_tokens"], int) for r in c_rows
    )
    return [
        (
            "arm C rows carry integer prompt and output tokens and no cached count",
            all(int_tokens) and all(r["cached_prompt_tokens"] is None for r in c_rows),
            "",
        ),
        (
            "arm C extra fields have the right types",
            all(
                isinstance(r["transport_failure"], bool)
                and isinstance(r["cold_start"], bool)
                and isinstance(r["attempts"], int)
                and not isinstance(r["attempts"], bool)
                and r["attempts"] >= 1
                and r["argmax_status"] in (None, "unique", "tied", "differs")
                for r in c_rows
            ),
            "",
        ),
        (
            "each arm C run has one endpoint",
            per_run(lambda rs: len({r["endpoint"] for r in rs}) == 1)
            and all(isinstance(r["endpoint"], str) and r["endpoint"] for r in c_rows),
            "",
        ),
        (
            f"each arm C run has one reported model version, {REQUESTED_MODEL}",
            per_run(one_version)
            and all(r["model_version"] is None for r in c_rows if r["transport_failure"]),
            "",
        ),
        (
            "every transport failure is ABSTAIN with confidence 0.0",
            all(
                r["y_pred"] == ABSTAIN and r["confidence"] == 0.0
                for r in c_rows
                if r["transport_failure"] is True
            ),
            "",
        ),
        (
            f"at most {MAX_TRANSPORT_FAILURES} transport failures per arm C run",
            per_run(
                lambda rs: sum(r["transport_failure"] is True for r in rs) <= MAX_TRANSPORT_FAILURES
            ),
            "",
        ),
    ]


def check_cache(
    path: str, prefix_bytes: int, prefix_sha256: str, label: str = ""
) -> list[tuple[str, bool, str]]:
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
        (f"{label}cache is append-only: the recorded prefix is byte-identical", prefix_ok, ""),
        (f"{label}every cache line parses and has a key", bad == 0, f"{bad} bad lines"),
        (f"{label}cache keys are unique", len(keys) == len(set(keys)), ""),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--partition", choices=("v2", "abc"), default="v2")
    parser.add_argument("--split", choices=("val", "test", "heldout"), required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--cache", help="Arm B response cache to check")
    parser.add_argument("--cache-prefix-bytes", type=int)
    parser.add_argument("--cache-prefix-sha256")
    parser.add_argument(
        "--cache-check",
        action="append",
        nargs=3,
        default=[],
        metavar=("PATH", "BYTES", "SHA256"),
        help="A response cache and its recorded prefix; repeat for arm B and arm C",
    )
    args = parser.parse_args(argv)
    abc = args.partition == "abc"
    if args.split not in (("val", "heldout") if abc else ("val", "test")):
        parser.error(f"--split {args.split} is not available with --partition {args.partition}")
    for _, size, _ in args.cache_check:
        if not size.isdigit():
            parser.error(f"--cache-check BYTES must be a whole number, got {size!r}")
    if abc and args.split == "heldout" and len(args.cache_check) < 2:
        # G6: each cache prefix (arm B's and arm C's) must be shown unchanged on the held-out run.
        parser.error("--partition abc --split heldout needs --cache-check for arm B and arm C")

    rows, problems = load_rows(args.runs, abc)
    results = [
        ("every line is JSON with exactly the expected keys", not problems, "; ".join(problems[:3]))
    ]
    labels = expected_labels(args.split, args.data_dir, args.partition)
    results += check_runs(rows, args.split, labels, ABC_ARMS if abc else V2_ARMS)
    if args.cache:
        if args.cache_prefix_bytes is None or args.cache_prefix_sha256 is None:
            parser.error("--cache needs --cache-prefix-bytes and --cache-prefix-sha256")
        results += check_cache(args.cache, args.cache_prefix_bytes, args.cache_prefix_sha256)
    for path, size, digest in args.cache_check:
        results += check_cache(path, int(size), digest, f"{path}: ")

    for name, ok, detail in results:
        note = f"  ({detail})" if detail and not ok else ""
        print(f"{'PASS' if ok else 'FAIL'}  {name}{note}")
    failed = sum(not ok for _, ok, _ in results)
    print(f"{len(results) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
