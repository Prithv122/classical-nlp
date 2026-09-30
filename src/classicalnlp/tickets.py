"""Banking77 customer-service queries: loader, validation split, fixture and download.

Banking77 (PolyAI) is 13,083 short customer-service queries from the banking domain,
labelled with one of 77 intents. The official split is 10,003 train / 3,080 test.

* **Licence:** CC-BY-4.0 (Hugging Face dataset card and the upstream GitHub repository).
* **Citation:** Casanueva et al. 2020, "Efficient Intent Detection with Dual Sentence
  Encoders", arXiv:2003.04807.

The Hugging Face repository holds only a loading script and no data files, and current
``datasets`` releases refuse to run such scripts, so the data comes from the upstream CSVs.
Those URLs are pinned to one commit and every downloaded file is checked against a SHA-256
digest and a row count, so a moved branch or a truncated download fails loudly instead of
quietly changing the data under a result.

The committed fixture, ``fixtures/tickets_sample.csv``, is **synthetic**: 60 hand-written
queries for 6 intents, used to test the code. None of its rows come from Banking77 and no
number computed from it says anything about Banking77.

Label ids are indices into :data:`BANKING77_INTENTS`, for the fixture and the real data
alike, so ids are stable across both.
"""

from __future__ import annotations

import csv
import hashlib
import os
import urllib.request
from collections.abc import Callable
from importlib import resources
from pathlib import Path

from sklearn.model_selection import train_test_split

from .data import Corpus
from .evaluate import RANDOM_STATE

BANKING77_LICENCE = "CC-BY-4.0"
BANKING77_SOURCE = "https://github.com/PolyAI-LDN/task-specific-datasets (banking_data)"

_PINNED = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data"
)
BANKING77_URLS = {"train": f"{_PINNED}/train.csv", "test": f"{_PINNED}/test.csv"}
BANKING77_SHA256 = {
    "train": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}
BANKING77_ROWS = {"train": 10003, "test": 3080}

DEFAULT_DIRECTORY = Path("data/raw/banking77")

#: The 77 intent names in ``sorted()`` order. Note the capital R in the first entry and
#: the trailing "?" in ``reverted_card_payment?`` -- both are real, keep them verbatim.
BANKING77_INTENTS: tuple[str, ...] = (
    "Refund_not_showing_up",
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "request_refund",
    "reverted_card_payment?",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
)

_INTENT_ID = {name: i for i, name in enumerate(BANKING77_INTENTS)}
_FIXTURE_PATH = "fixtures/tickets_sample.csv"


def _parse(handle, origin: str) -> Corpus:
    reader = csv.DictReader(handle)
    for column in ("text", "category"):
        if column not in (reader.fieldnames or []):
            raise ValueError(f"{origin}: missing required column {column!r}")
    documents, labels = [], []
    for row in reader:
        category = row["category"]
        if category not in _INTENT_ID:
            raise ValueError(f"{origin}: unknown category {category!r}")
        documents.append(row["text"])
        labels.append(_INTENT_ID[category])
    return Corpus(documents=documents, labels=labels, label_names=list(BANKING77_INTENTS))


def read_csv(path: str | Path) -> Corpus:
    """Read a ``text,category`` CSV. Uses the csv module: many texts contain commas."""
    with open(path, encoding="utf-8", newline="") as handle:
        return _parse(handle, str(path))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_banking77(directory: str | Path = DEFAULT_DIRECTORY) -> tuple[Corpus, Corpus]:
    """Load ``(train, test)`` from ``directory``, verifying checksum and row count first."""
    directory = Path(directory)
    fix = "run: classical-nlp tickets-download"
    corpora = []
    for split in ("train", "test"):
        path = directory / f"{split}.csv"
        if not path.is_file():
            raise ValueError(f"{path} not found; {fix}")
        if sha256_file(path) != BANKING77_SHA256[split]:
            raise ValueError(f"{path}: sha256 does not match the pinned {split} file; {fix}")
        corpus = read_csv(path)
        if len(corpus) != BANKING77_ROWS[split]:
            raise ValueError(f"{path}: {len(corpus)} rows, expected {BANKING77_ROWS[split]}; {fix}")
        corpora.append(corpus)
    return corpora[0], corpora[1]


def download_banking77(
    directory: str | Path = DEFAULT_DIRECTORY,
    *,
    opener: Callable = urllib.request.urlopen,
    sha256: dict[str, str] = BANKING77_SHA256,
) -> dict[str, Path]:
    """Fetch the pinned CSVs into ``directory``; return ``{split: path}``.

    A file already present with the right digest is skipped without touching the network.
    Fetched bytes are hashed in memory and only written (via a temp name and a rename) when
    they match, so a mismatch leaves no file behind.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, url in BANKING77_URLS.items():
        path = directory / f"{split}.csv"
        paths[split] = path
        if path.is_file() and sha256_file(path) == sha256[split]:
            continue
        with opener(url) as response:
            payload = response.read()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != sha256[split]:
            raise ValueError(
                f"{url}: sha256 {actual} does not match the pinned {sha256[split]}; "
                "nothing was written"
            )
        temporary = path.with_name(path.name + ".part")
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    return paths


def load_fixture() -> Corpus:
    """The committed 60-row synthetic fixture (works from an installed package)."""
    resource = resources.files("classicalnlp").joinpath(_FIXTURE_PATH)
    with resource.open("r", encoding="utf-8", newline="") as handle:
        return _parse(handle, _FIXTURE_PATH)


def validation_split(
    corpus: Corpus, *, fraction: float = 0.2, seed: int = RANDOM_STATE
) -> tuple[Corpus, Corpus]:
    """Stratified, seeded ``(fit, val)`` split. Both halves keep the full label names."""
    fit_idx, val_idx = train_test_split(
        list(range(len(corpus))), test_size=fraction, stratify=corpus.labels, random_state=seed
    )

    def take(indices: list[int]) -> Corpus:
        return Corpus(
            documents=[corpus.documents[i] for i in indices],
            labels=[corpus.labels[i] for i in indices],
            label_names=corpus.label_names,
        )

    return take(fit_idx), take(val_idx)
