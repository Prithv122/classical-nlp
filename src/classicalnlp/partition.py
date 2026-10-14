"""The arm C study's Banking77 partition: fit F', validation V and held-out H.

Everything here follows ``ARM_C_PROTOCOL.md``, "Data and partition". Index space is the 0-based
row index in the pinned Banking77 train file. Step 1 is the v2 validation split (F, V); the
candidate pool is the F rows whose normalised text is unique in the whole train split; H is a
seeded proportional sample of the pool; F' is F minus H.

The H indices are frozen in ``data/partitions/abc_heldout.csv`` and the loader refuses a file
whose SHA-256 or row count differs from the registered values. This module reads the train file
only: the official test split is not read by this study.
"""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

from . import tickets
from .data import Corpus
from .evaluate import RANDOM_STATE

ABC_SEED = 20261014
ABC_HELDOUT_SIZE = 2002
ABC_HELDOUT_SHA256 = "0b64a728f7f10a633bc8a8fd482f9993302299f8eb823318f98142a343a1e814"
ABC_HELDOUT_PATH = Path("data/partitions/abc_heldout.csv")


def normalise_for_duplicates(text: str) -> str:
    return " ".join(text.lower().split())


@dataclass(frozen=True)
class AbcPartition:
    fit: list[int]  # F', ascending
    val: list[int]  # V, in validation_split order
    heldout: list[int]  # H, ascending


def abc_partition(train: Corpus, *, heldout_size: int | None = None) -> AbcPartition:
    """Protocol Steps 1-3. ``heldout_size`` differs from the registered value only in tests."""
    size = ABC_HELDOUT_SIZE if heldout_size is None else heldout_size
    old_fit, val = train_test_split(
        list(range(len(train))), test_size=0.2, stratify=train.labels, random_state=RANDOM_STATE
    )
    normalised = [normalise_for_duplicates(text) for text in train.documents]
    occurrences = Counter(normalised)
    pool = sorted(i for i in old_fit if occurrences[normalised[i]] == 1)
    _, heldout = train_test_split(
        pool,
        test_size=size,
        stratify=[train.labels[i] for i in pool],
        random_state=ABC_SEED,
    )
    held = {int(i) for i in heldout}
    return AbcPartition(
        fit=sorted(int(i) for i in old_fit if int(i) not in held),
        val=[int(i) for i in val],
        heldout=sorted(held),
    )


def heldout_csv_bytes(indices: list[int], labels: list[int]) -> bytes:
    """``train_index,label`` with a header, one row per index ascending, LF, integer label ids."""
    lines = ["train_index,label", *(f"{i},{labels[i]}" for i in sorted(indices))]
    return ("\n".join(lines) + "\n").encode("utf-8")


def load_abc_heldout(path: str | Path = ABC_HELDOUT_PATH) -> list[int]:
    """H's train indices from the frozen file, refusing a wrong checksum or row count."""
    path = Path(path)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{path}: cannot read the frozen held-out file: {exc}") from exc
    if hashlib.sha256(payload).hexdigest() != ABC_HELDOUT_SHA256:
        raise ValueError(f"{path}: sha256 does not match the registered held-out file")
    rows = payload.decode("utf-8").splitlines()[1:]
    if len(rows) != ABC_HELDOUT_SIZE:
        raise ValueError(f"{path}: {len(rows)} rows, expected {ABC_HELDOUT_SIZE}")
    return [int(row.split(",")[0]) for row in rows]


def load_train(directory: str | Path = tickets.DEFAULT_DIRECTORY) -> Corpus:
    """The pinned train file, checked like ``tickets.load_banking77`` but without opening test."""
    path = Path(directory) / "train.csv"
    fix = "run: classical-nlp tickets-download"
    if not path.is_file():
        raise ValueError(f"{path} not found; {fix}")
    if tickets.sha256_file(path) != tickets.BANKING77_SHA256["train"]:
        raise ValueError(f"{path}: sha256 does not match the pinned train file; {fix}")
    corpus = tickets.read_csv(path)
    if len(corpus) != tickets.BANKING77_ROWS["train"]:
        raise ValueError(f"{path}: {len(corpus)} rows, expected {tickets.BANKING77_ROWS['train']}")
    return corpus


def abc_corpora(
    train: Corpus, path: str | Path = ABC_HELDOUT_PATH
) -> tuple[Corpus, Corpus, Corpus]:
    """``(F', V, H)`` as corpora. Refuses unless the frozen file and the recipe agree."""
    frozen = load_abc_heldout(path)
    recipe = abc_partition(train)
    if recipe.heldout != frozen:
        raise ValueError(f"{path}: the recipe gives a different held-out set than the frozen file")
    if heldout_csv_bytes(frozen, train.labels) != Path(path).read_bytes():
        raise ValueError(f"{path}: the label column does not match the train labels")

    def take(indices: list[int]) -> Corpus:
        return Corpus(
            documents=[train.documents[i] for i in indices],
            labels=[train.labels[i] for i in indices],
            label_names=train.label_names,
        )

    return take(recipe.fit), take(recipe.val), take(recipe.heldout)


def write_atomic(path: str | Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
