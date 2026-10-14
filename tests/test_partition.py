"""The arm C partition: recipe, frozen file and loader. Synthetic corpora, one network test."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from classicalnlp import partition, tickets
from classicalnlp.data import Corpus

ROOT = Path(__file__).parents[1]
PROBE_TRAIN_INDICES = [
    2520, 2375, 9033, 1513, 7455, 6821, 7355, 8960, 4286, 6389,
    3639, 8747, 5790, 8017, 9651, 2891, 9449, 3053, 2516, 9554,
]  # fmt: skip


def _synthetic(per_intent=60, intents=10):
    documents, labels = [], []
    for label in range(intents):
        for j in range(per_intent):
            documents.append(f"ticket about intent {label} number {j}")
            labels.append(label)
    # Two exact duplicates after normalisation (case and whitespace), spread over the corpus.
    documents[3], documents[301] = "Where is  my card", " where is my CARD "
    documents[7], documents[9] = "same text", "same text"
    return Corpus(documents=documents, labels=labels, label_names=list(tickets.BANKING77_INTENTS))


@pytest.fixture(scope="module")
def synthetic():
    train = _synthetic()
    return train, partition.abc_partition(train, heldout_size=60)


def test_validation_set_is_the_v2_validation_split_in_order(synthetic):
    train, parts = synthetic
    assert [train.documents[i] for i in parts.val] == tickets.validation_split(train)[1].documents


def test_sets_are_disjoint_cover_everything_and_are_ordered(synthetic):
    train, parts = synthetic
    assert parts.fit == sorted(parts.fit) and parts.heldout == sorted(parts.heldout)
    assert len(parts.heldout) == 60
    everything = [*parts.fit, *parts.val, *parts.heldout]
    assert sorted(everything) == list(range(len(train)))


def test_duplicates_are_never_held_out_and_stay_in_fit(synthetic):
    train, parts = synthetic
    norm = partition.normalise_for_duplicates
    assert norm(" Where is  my CARD ") == "where is my card"
    duplicated = {3, 301, 7, 9}
    assert not duplicated & set(parts.heldout)
    held = {norm(train.documents[i]) for i in parts.heldout}
    others = {norm(train.documents[i]) for i in parts.fit + parts.val}
    assert not held & others
    old_fit = set(range(len(train))) - set(parts.val)
    assert (duplicated & old_fit) <= set(parts.fit)


def test_held_out_set_is_stratified_in_proportion(synthetic):
    train, parts = synthetic
    counts = Counter(train.labels[i] for i in parts.heldout)
    assert set(counts) == set(range(10))
    assert max(counts.values()) - min(counts.values()) <= 1


def test_csv_bytes_are_exact():
    payload = partition.heldout_csv_bytes([5, 2], [0, 0, 7, 0, 0, 9])
    assert payload == b"train_index,label\n2,7\n5,9\n"


def _frozen(tmp_path, monkeypatch, indices, labels, size=None):
    path = tmp_path / "h.csv"
    payload = partition.heldout_csv_bytes(indices, labels)
    path.write_bytes(payload)
    monkeypatch.setattr(partition, "ABC_HELDOUT_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(partition, "ABC_HELDOUT_SIZE", len(indices) if size is None else size)
    return path


def test_loader_refuses_a_wrong_hash_or_row_count(tmp_path, monkeypatch):
    path = tmp_path / "h.csv"
    path.write_bytes(partition.heldout_csv_bytes([1, 2, 3], [0, 0, 0, 0]))
    with pytest.raises(ValueError, match=r"h\.csv.*sha256"):
        partition.load_abc_heldout(path)
    path = _frozen(tmp_path, monkeypatch, [1, 2, 3], [0, 0, 0, 0], size=4)
    with pytest.raises(ValueError, match=r"h\.csv.*3 rows, expected 4"):
        partition.load_abc_heldout(path)
    with pytest.raises(ValueError, match="cannot read"):
        partition.load_abc_heldout(tmp_path / "missing.csv")


def test_abc_corpora_returns_the_three_sets_and_refuses_a_different_recipe(
    tmp_path, monkeypatch, synthetic
):
    train, parts = synthetic
    path = _frozen(tmp_path, monkeypatch, parts.heldout, train.labels)
    fit, val, held = partition.abc_corpora(train, path)
    assert (len(fit), len(val), len(held)) == (len(parts.fit), 120, 60)
    assert held.documents == [train.documents[i] for i in parts.heldout]
    other = sorted([*parts.heldout[1:], parts.fit[0]])
    path = _frozen(tmp_path, monkeypatch, other, train.labels)
    with pytest.raises(ValueError, match="different held-out set"):
        partition.abc_corpora(train, path)


def test_the_committed_file_has_the_registered_hash_and_rows():
    path = ROOT / partition.ABC_HELDOUT_PATH
    assert hashlib.sha256(path.read_bytes()).hexdigest() == partition.ABC_HELDOUT_SHA256
    assert b"\r" not in path.read_bytes()
    indices = partition.load_abc_heldout(path)
    assert len(indices) == 2002 and indices == sorted(set(indices))


def test_the_script_writes_then_checks(tmp_path, monkeypatch, load_script, synthetic, capsys):
    train, parts = synthetic
    script = load_script("make_abc_partition")
    monkeypatch.setattr(partition, "load_train", lambda directory: train)
    monkeypatch.setattr(partition, "ABC_HELDOUT_SIZE", 60)
    payload = partition.heldout_csv_bytes(parts.heldout, train.labels)
    monkeypatch.setattr(partition, "ABC_HELDOUT_SHA256", hashlib.sha256(payload).hexdigest())
    out = tmp_path / "sub" / "h.csv"
    assert script.main(["--data-dir", "d", "--out", str(out)]) == 0
    assert out.read_bytes() == payload
    assert script.main(["--data-dir", "d", "--out", str(out), "--check"]) == 0
    out.write_bytes(payload + b"x")
    assert script.main(["--data-dir", "d", "--out", str(out), "--check"]) == 1
    assert "min" in capsys.readouterr().out


def test_load_train_refuses_a_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        partition.load_train(tmp_path)


@pytest.mark.network
def test_abc_partition_on_real_banking77_hashes_to_the_registered_value(tmp_path):
    tickets.download_banking77(tmp_path)
    train = partition.load_train(tmp_path)
    parts = partition.abc_partition(train)
    payload = partition.heldout_csv_bytes(parts.heldout, train.labels)
    assert hashlib.sha256(payload).hexdigest() == partition.ABC_HELDOUT_SHA256
    positions = np.random.default_rng(partition.ABC_SEED).choice(2001, 20, replace=False)
    assert [parts.val[i] for i in positions] == PROBE_TRAIN_INDICES
