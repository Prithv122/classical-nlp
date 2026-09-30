"""Ticket data-layer tests.

Everything runs on the synthetic fixture or on tiny files written to ``tmp_path``. Nothing
here touches the network: the download is exercised through a fake opener.
"""

from __future__ import annotations

import hashlib
import io

import pytest

from classicalnlp import cli, tickets


def _write_csv(path, rows):
    lines = ["text,category", *rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class FakeOpener:
    """Stands in for urllib.request.urlopen; records the URLs it was asked for."""

    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        return io.BytesIO(self.payloads[url])


def _payloads_and_hashes():
    payloads = {
        tickets.BANKING77_URLS["train"]: b"text,category\nhello,card_arrival\n",
        tickets.BANKING77_URLS["test"]: b"text,category\nworld,exchange_rate\n",
    }
    hashes = {s: hashlib.sha256(payloads[u]).hexdigest() for s, u in tickets.BANKING77_URLS.items()}
    return payloads, hashes


def test_fixture_loads_60_rows_10_per_intent():
    corpus = tickets.load_fixture()
    assert len(corpus) == 60
    present = {name: n for name, n in corpus.class_counts.items() if n}
    assert len(present) == 6
    assert set(present.values()) == {10}


def test_intent_list_is_77_unique_sorted_and_covers_the_fixture():
    intents = tickets.BANKING77_INTENTS
    assert len(intents) == 77
    assert len(set(intents)) == 77
    assert list(intents) == sorted(intents)
    fixture = tickets.load_fixture()
    assert {n for n, c in fixture.class_counts.items() if c} <= set(intents)


def test_read_csv_handles_quoted_commas_and_maps_category_to_index(tmp_path):
    path = _write_csv(tmp_path / "t.csv", ['"Hello, my card, please",card_arrival'])
    corpus = tickets.read_csv(path)
    assert corpus.documents == ["Hello, my card, please"]
    assert corpus.labels == [tickets.BANKING77_INTENTS.index("card_arrival")]
    assert corpus.label_names == list(tickets.BANKING77_INTENTS)


def test_unknown_category_names_the_category(tmp_path):
    path = _write_csv(tmp_path / "t.csv", ["hi,not_an_intent"])
    with pytest.raises(ValueError, match="not_an_intent"):
        tickets.read_csv(path)


def test_missing_column_is_rejected(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("text,label\nhi,card_arrival\n", encoding="utf-8")
    with pytest.raises(ValueError, match="category"):
        tickets.read_csv(path)


def test_validation_split_is_48_12_with_every_intent_in_both_halves():
    fit, val = tickets.validation_split(tickets.load_fixture())
    assert (len(fit), len(val)) == (48, 12)
    assert sum(1 for c in fit.class_counts.values() if c) == 6
    assert sum(1 for c in val.class_counts.values() if c) == 6
    assert fit.label_names == val.label_names == list(tickets.BANKING77_INTENTS)


def test_validation_split_is_seeded():
    corpus = tickets.load_fixture()
    a = tickets.validation_split(corpus, seed=1)
    b = tickets.validation_split(corpus, seed=1)
    c = tickets.validation_split(corpus, seed=2)
    assert a[1].documents == b[1].documents
    assert a[1].documents != c[1].documents


def test_validation_halves_are_disjoint_and_cover_the_corpus():
    corpus = tickets.load_fixture()
    fit, val = tickets.validation_split(corpus)
    assert not set(fit.documents) & set(val.documents)
    assert sorted(fit.documents + val.documents) == sorted(corpus.documents)


def test_download_writes_both_files(tmp_path):
    payloads, hashes = _payloads_and_hashes()
    paths = tickets.download_banking77(tmp_path, opener=FakeOpener(payloads), sha256=hashes)
    assert set(paths) == {"train", "test"}
    assert paths["train"].read_bytes() == payloads[tickets.BANKING77_URLS["train"]]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["test.csv", "train.csv"]


def test_download_hash_mismatch_raises_and_leaves_no_file(tmp_path):
    payloads, hashes = _payloads_and_hashes()
    hashes["test"] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        tickets.download_banking77(tmp_path, opener=FakeOpener(payloads), sha256=hashes)
    assert not (tmp_path / "test.csv").exists()
    assert not list(tmp_path.glob("*.part"))


def test_download_skips_a_file_that_is_already_correct(tmp_path):
    payloads, hashes = _payloads_and_hashes()
    tickets.download_banking77(tmp_path, opener=FakeOpener(payloads), sha256=hashes)
    second = FakeOpener(payloads)
    tickets.download_banking77(tmp_path, opener=second, sha256=hashes)
    assert second.calls == []


def test_load_banking77_rejects_a_wrong_checksum(tmp_path):
    _write_csv(tmp_path / "train.csv", ["hi,card_arrival"])
    _write_csv(tmp_path / "test.csv", ["hi,card_arrival"])
    with pytest.raises(ValueError, match="tickets-download"):
        tickets.load_banking77(tmp_path)


def test_load_banking77_accepts_matching_checksum_and_rows(tmp_path, monkeypatch):
    train = _write_csv(tmp_path / "train.csv", ["a,card_arrival", "b,exchange_rate"])
    test = _write_csv(tmp_path / "test.csv", ["c,card_arrival"])
    monkeypatch.setattr(
        tickets,
        "BANKING77_SHA256",
        {"train": tickets.sha256_file(train), "test": tickets.sha256_file(test)},
    )
    monkeypatch.setattr(tickets, "BANKING77_ROWS", {"train": 2, "test": 1})
    fit, held_out = tickets.load_banking77(tmp_path)
    assert (len(fit), len(held_out)) == (2, 1)
    monkeypatch.setattr(tickets, "BANKING77_ROWS", {"train": 3, "test": 1})
    with pytest.raises(ValueError, match="rows"):
        tickets.load_banking77(tmp_path)


def test_cli_tickets_summary_on_the_fixture(capsys):
    assert cli.main(["tickets-summary", "--fixture"]) == 0
    out = capsys.readouterr().out
    assert "rows: 60" in out
    assert "48/12" in out


def test_cli_tickets_summary_requires_a_source():
    with pytest.raises(SystemExit):
        cli.main(["tickets-summary"])


def test_cli_tickets_download_uses_the_downloader(tmp_path, monkeypatch, capsys):
    made = {}

    def fake_download(directory):
        made["directory"] = directory
        paths = {}
        for split in ("train", "test"):
            paths[split] = tmp_path / f"{split}.csv"
            paths[split].write_bytes(b"x")
        return paths

    monkeypatch.setattr(tickets, "download_banking77", fake_download)
    assert cli.main(["tickets-download", "--dest", str(tmp_path)]) == 0
    assert made["directory"] == str(tmp_path)
    out = capsys.readouterr().out
    assert hashlib.sha256(b"x").hexdigest() in out
    assert tickets.BANKING77_LICENCE in out
