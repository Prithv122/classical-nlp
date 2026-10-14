"""Write (or check) the frozen held-out file for the arm C study. Reports no result statistic.

uv run python scripts/make_abc_partition.py --data-dir data/raw/banking77
uv run python scripts/make_abc_partition.py --data-dir data/raw/banking77 --check
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

from classicalnlp import partition


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", default=str(partition.ABC_HELDOUT_PATH))
    parser.add_argument("--check", action="store_true", help="Compare with --out; write nothing")
    args = parser.parse_args(argv)

    train = partition.load_train(args.data_dir)
    parts = partition.abc_partition(train)
    payload = partition.heldout_csv_bytes(parts.heldout, train.labels)
    digest = hashlib.sha256(payload).hexdigest()
    per_intent = Counter(train.labels[i] for i in parts.heldout)
    print(f"rows: fit {len(parts.fit)}, val {len(parts.val)}, heldout {len(parts.heldout)}")
    print(f"heldout per intent: min {min(per_intent.values())}, max {max(per_intent.values())}")
    print(f"sha256 {digest}")
    if digest != partition.ABC_HELDOUT_SHA256:
        print("MISMATCH: not the registered value", partition.ABC_HELDOUT_SHA256, file=sys.stderr)
        return 1
    if args.check:
        same = Path(args.out).is_file() and Path(args.out).read_bytes() == payload
        print(f"{args.out}: {'identical' if same else 'DIFFERENT or missing'}")
        return 0 if same else 1
    partition.write_atomic(args.out, payload)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
