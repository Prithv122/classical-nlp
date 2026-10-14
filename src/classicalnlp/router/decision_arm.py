"""Arm C: a dedicated decision model (TypeSafe Jev) behind a cache, a ledger and spending caps.

One ``choice`` question per ticket; one response feeds two arms:

* ``C-prob`` (primary): confidence is ``probabilities[choice]``.
* ``C-vendor`` (ablation): confidence is the response's own ``confidence`` field.

Abstention follows ``ARM_C_PROTOCOL.md``. A ``choice`` that is not one of the 77 names, or
``probabilities`` that miss a name, hold a non-number or do not sum to 1 within 0.02 (no
renormalisation; the vendor rounds to two decimals, amendment 7), abstain both variants. A
missing or out-of-range ``confidence`` abstains ``C-vendor`` only.

The client does the work around a provider that makes one HTTP attempt per call: the cache
lookup, up to five attempts with exponential backoff, the transport-failure rule, the version
and usage checks and the two spending caps. A transport failure (a request that still fails
after its last attempt) is scored ABSTAIN with confidence 0.0, flagged, and never cached, so
the identical command retries it. Nothing in this module opens a connection; the transport is
in :mod:`classicalnlp.router.typesafe`.

:class:`FakeDecisionProvider` and :func:`fake_decision_response` are test doubles. Their numbers
exist to exercise this code and are not model behaviour.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..data import Corpus
from ..tickets import BANKING77_INTENTS
from .base import ABSTAIN, Usage, check_output
from .llm_arm import ProviderError

ENDPOINT = "typesafe:https://api.typesafe.ai/v1/systemone"
REQUESTED_MODEL = "jev-1.13.0"
INSTRUCTIONS = (
    "You route customer-support messages for a bank. "
    "Pick the single intent that best matches the message."
)
DEFAULT_PRICE_PER_MTOK = 0.042  # USD per million input tokens; output is free
RUN_CAP_USD = 2.00
STUDY_CAP_USD = 5.00
MAX_INPUT_TOKENS = 4000
MAX_ATTEMPTS = 5
TIMEOUT_SECONDS = 30.0
BACKOFF_SECONDS = (1, 2, 4, 8)
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 529})
PROBABILITY_TOLERANCE = 0.02  # amendment 7; the protocol's original 1e-3 rejected rounding
EXTRA_KEYS = (
    "endpoint",
    "model_version",
    "transport_failure",
    "cold_start",
    "attempts",
    "argmax_status",
)


class DecisionError(ProviderError):
    """A condition that stops the run (a ValueError, so the CLI exits 2)."""


class BudgetExceeded(DecisionError):
    pass


class VersionMismatch(DecisionError):
    pass


class UsageError(DecisionError):
    pass


class AuthError(DecisionError):
    pass


class AttemptFailed(Exception):
    """One HTTP attempt failed. ``status`` is the HTTP code, or None with an ``error`` class."""

    def __init__(self, error: str, status: int | None = None) -> None:
        super().__init__(f"{error} (HTTP {status})" if status else error)
        self.error, self.status = error, status
        self.retryable = status in RETRY_STATUSES or error in ("timeout", "connection")


@dataclass(frozen=True)
class RawResponse:
    """What a provider reports for one successful attempt."""

    body: dict
    seconds: float  # client-side wall-clock of the attempt
    cold_start: bool  # first request on a fresh connection
    attempts: int = 1


@dataclass(frozen=True)
class DecisionResponse:
    """What the client hands an arm. ``body`` is None on a transport failure."""

    body: dict | None
    model_version: str | None
    input_tokens: int
    output_tokens: int
    seconds: float
    cold_start: bool
    attempts: int
    transport_failure: bool = False


@runtime_checkable
class DecisionProvider(Protocol):
    identity: str

    def decide(self, request: dict) -> RawResponse: ...


def build_request(text: str, names: Sequence[str] = BANKING77_INTENTS) -> dict:
    """The request for one ticket: its text unmodified and the 77 names mapped to null."""
    return {
        "state": text,
        "model": REQUESTED_MODEL,
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": dict.fromkeys(names),
            }
        },
    }


def cache_key(identity: str, request: dict) -> str:
    canonical = json.dumps(request, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{identity}\n{REQUESTED_MODEL}\n{canonical}".encode()).hexdigest()


def append_line(path: str | Path, text: str) -> None:
    """Append one flushed line, first ending a partial last line left by a killed run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = ""
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as handle:
            handle.seek(-1, 2)
            prefix = "" if handle.read(1) == b"\n" else "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(prefix + text + "\n")
        handle.flush()


def _records(path: Path | None):
    if path is None or not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                yield line


class DecisionCache:
    """Append-only JSONL of first responses, keyed by :func:`cache_key`.

    A record keeps the response, the reported version, the usage and the original seconds, so a
    cached ticket costs nothing and keeps its first timing. Lines that do not parse are skipped
    and counted in :attr:`skipped`. The file is created at the first write, never on open.
    """

    def __init__(self, path: str | Path | None) -> None:
        self.path = None if path is None else Path(path)
        self.skipped = 0
        self._items: dict[str, DecisionResponse] = {}
        for line in _records(self.path):
            try:
                r = json.loads(line)
                # The first response stored for a ticket is its record; a later duplicate is not.
                self._items.setdefault(
                    r["key"],
                    DecisionResponse(
                        body=r["response"],
                        model_version=r["model_version"],
                        input_tokens=int(r["input_tokens"]),
                        output_tokens=int(r["output_tokens"]),
                        seconds=float(r["seconds"]),
                        cold_start=bool(r["cold_start"]),
                        attempts=int(r["attempts"]),
                    ),
                )
            except (ValueError, KeyError, TypeError):
                self.skipped += 1

    def get(self, key: str) -> DecisionResponse | None:
        return self._items.get(key)

    def put(self, key: str, identity: str, r: DecisionResponse) -> None:
        self._items[key] = r
        if self.path is not None:
            record = {
                "key": key,
                "endpoint": identity,
                "requested_model": REQUESTED_MODEL,
                "response": r.body,
                "model_version": r.model_version,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "seconds": r.seconds,
                "cold_start": r.cold_start,
                "attempts": r.attempts,
            }
            append_line(self.path, json.dumps(record))


class Ledger:
    """Append-only JSONL, one line per HTTP attempt; cache hits write nothing.

    The study total is the sum of every line's dollars, across runs; the run total is this
    object's own. Dollars are input tokens times the price; output is free.
    """

    def __init__(self, path: str | Path, run_id: str | None = None) -> None:
        self.path = Path(path)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.run_total = 0.0
        self.study_total = 0.0
        for line in _records(self.path):
            with contextlib.suppress(ValueError, KeyError, TypeError):
                self.study_total += float(json.loads(line)["dollars"])

    def add(self, kind, key, attempt, status, input_tokens, output_tokens, dollars) -> None:
        row = {
            "run_id": self.run_id,
            "kind": kind,
            "key": key,
            "attempt": attempt,
            "status": status,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "dollars": dollars,
        }
        append_line(self.path, json.dumps(row))
        self.run_total += dollars
        self.study_total += dollars

    def check_budget(self) -> None:
        if self.run_total >= RUN_CAP_USD:
            raise BudgetExceeded(f"run cap of ${RUN_CAP_USD:.2f} reached; stopping before a call")
        if self.study_total >= STUDY_CAP_USD:
            raise BudgetExceeded(
                f"study cap of ${STUDY_CAP_USD:.2f} reached; stopping before a call"
            )


class DecisionClient:
    """A provider behind a cache, a ledger and the retry, version and budget rules."""

    def __init__(
        self,
        provider: DecisionProvider,
        cache: DecisionCache,
        ledger: Ledger,
        *,
        price_per_mtok: float = DEFAULT_PRICE_PER_MTOK,
        sleep: Callable[[float], None] = time.sleep,
        kind: str = "record",
    ) -> None:
        self.provider, self.cache, self.ledger = provider, cache, ledger
        self.price_per_mtok, self.sleep, self.kind = price_per_mtok, sleep, kind
        self.calls = self.hits = self.attempts = self.transport_failures = 0
        self._failed: dict[str, DecisionResponse] = {}  # this process only, never persisted

    def decide(
        self, request: dict, *, kind: str | None = None, use_cache: bool = True
    ) -> DecisionResponse:
        key = cache_key(self.provider.identity, request)
        if use_cache:
            known = self.cache.get(key) or self._failed.get(key)
            if known is not None:
                self.hits += 1
                return known
        kind = kind or self.kind
        self.calls += 1
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.ledger.check_budget()
            self.attempts += 1
            try:
                raw = self.provider.decide(request)
            except AttemptFailed as failure:
                status = failure.status or failure.error
                self.ledger.add(kind, key, attempt, status, None, None, 0.0)
                if failure.status in (401, 403):
                    raise AuthError(
                        f"HTTP {failure.status}: check TYPESAFE_API_KEY (environment variable "
                        "or the keyring entry)"
                    ) from None
                if not failure.retryable:
                    break
                if attempt < MAX_ATTEMPTS:
                    self.sleep(BACKOFF_SECONDS[attempt - 1])
                continue
            result = self._accept(raw, key, attempt, kind)
            if use_cache:
                self.cache.put(key, self.provider.identity, result)
            return result
        self.transport_failures += 1
        failed = DecisionResponse(None, None, 0, 0, 0.0, False, attempt, transport_failure=True)
        if use_cache:
            self._failed[key] = failed
        return failed

    def _accept(self, raw: RawResponse, key: str, attempt: int, kind: str) -> DecisionResponse:
        body = raw.body
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        tokens_in = usage.get("input_tokens")
        tokens_out = usage.get("output_tokens")
        has_input = _number(tokens_in) and isinstance(tokens_in, int) and tokens_in >= 0
        tokens_out = tokens_out if isinstance(tokens_out, int) and tokens_out >= 0 else 0
        dollars = tokens_in * self.price_per_mtok / 1e6 if has_input else 0.0
        self.ledger.add(kind, key, attempt, 200, tokens_in, tokens_out, dollars)
        version = body.get("model")
        if version != REQUESTED_MODEL:
            raise VersionMismatch(f"response reports model {version!r}, expected {REQUESTED_MODEL}")
        if not has_input:
            raise UsageError("response has no usable usage.input_tokens")
        if tokens_in > MAX_INPUT_TOKENS:
            raise UsageError(f"{tokens_in} input tokens exceeds the {MAX_INPUT_TOKENS} guard")
        return DecisionResponse(
            body=body,
            model_version=version,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            seconds=raw.seconds,
            cold_start=raw.cold_start,
            attempts=attempt,
        )


def _number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def parse_answer(body: dict | None, names: Sequence[str]):
    """``(label, probability, vendor, argmax_status)`` for one response.

    ``label`` is None when both variants abstain; ``vendor`` is None when only C-vendor does;
    ``probability`` is None when ``probabilities`` are unusable. ``argmax_status`` is None
    unless ``probabilities`` and ``choice`` are both usable.
    """
    answer = ((body or {}).get("answers") or {}).get("intent")
    if not isinstance(answer, dict):
        return None, None, None, None
    choice, probs = answer.get("choice"), answer.get("probabilities")
    # Only the 77 names count: their probabilities must each lie in [0, 1] and sum to 1, and
    # the argmax is taken over them. Any other key is ignored, never summed or ranked.
    usable = (
        isinstance(probs, dict)
        and all(n in probs and _number(probs[n]) and 0.0 <= probs[n] <= 1.0 for n in names)
        # The 1e-9 only makes the bound inclusive: 0.98 is a deviation of 0.02 plus float noise.
        and abs(sum(probs[n] for n in names) - 1.0) <= PROBABILITY_TOLERANCE + 1e-9
    )
    if not usable or not isinstance(choice, str) or choice not in names:
        return None, None, None, None
    top = max(probs[n] for n in names)
    status = "differs" if probs[choice] < top else "unique"
    if status == "unique" and sum(1 for n in names if probs[n] == top) > 1:
        status = "tied"
    vendor = answer.get("confidence")
    vendor = float(vendor) if _number(vendor) and 0.0 <= vendor <= 1.0 else None
    return list(names).index(choice), float(probs[choice]), vendor, status


class DecisionArm:
    description = "dedicated decision model (TypeSafe Jev, 77 intents)"

    def __init__(self, client: DecisionClient, variant: str = "prob") -> None:
        if variant not in ("prob", "vendor"):
            raise ValueError(f"variant must be 'prob' or 'vendor', got {variant!r}")
        self.client, self.variant, self.name = client, variant, f"C-{variant}"
        self.usage: Usage | None = None
        self._names: list[str] | None = None

    def fit(self, corpus: Corpus) -> DecisionArm:
        if tuple(corpus.label_names) != BANKING77_INTENTS:
            raise ValueError("arm C sends the 77 Banking77 intent names; the corpus has others")
        self._names = list(corpus.label_names)
        return self

    def predict_top(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        if self._names is None:
            raise ValueError("DecisionArm is not fitted; call fit() first")
        labels, confidence, responses, extras = [], [], [], []
        for text in texts:
            r = self.client.decide(build_request(text, self._names))
            label, prob, vendor, status = parse_answer(r.body, self._names)
            value = prob if self.variant == "prob" else vendor
            labels.append(ABSTAIN if value is None else label)
            confidence.append(0.0 if value is None else value)
            responses.append(r)
            extras.append(
                {
                    "endpoint": self.client.provider.identity,
                    "model_version": r.model_version,
                    "transport_failure": r.transport_failure,
                    "cold_start": r.cold_start,
                    "attempts": r.attempts,
                    "argmax_status": status,
                }
            )
        self.usage = Usage(
            seconds=np.array([r.seconds for r in responses], dtype=float),
            prompt_tokens=np.array([r.input_tokens for r in responses], dtype=int),
            output_tokens=np.array([r.output_tokens for r in responses], dtype=int),
            row_extras=tuple(extras),
        )
        return check_output(labels, confidence, len(texts))


def decision_arms(client: DecisionClient) -> list[DecisionArm]:
    """Both variants over one client, so one response per ticket feeds both."""
    return [DecisionArm(client, "prob"), DecisionArm(client, "vendor")]


class FakeDecisionProvider:
    """Test double. ``replies`` is a callable taking the request, or a list consumed in order
    (the last item repeats). An item is a response body, a :class:`RawResponse`, or an exception
    instance to raise. Its outputs are not model behaviour."""

    identity = ENDPOINT

    def __init__(self, replies) -> None:
        self.replies = replies
        self.calls = 0
        self.requests: list[dict] = []
        self.pings = 0

    def decide(self, request: dict) -> RawResponse:
        self.calls += 1
        self.requests.append(request)
        if callable(self.replies):
            out = self.replies(request)
        else:
            out = self.replies[min(self.calls, len(self.replies)) - 1]
        if isinstance(out, Exception):
            raise out
        if isinstance(out, RawResponse):
            return out
        return RawResponse(out, seconds=0.2, cold_start=self.calls == 1)

    def ping(self) -> tuple[int, float]:
        self.pings += 1
        return 200, 0.05


def fake_decision_response(
    choice: str,
    p_choice: float = 0.9,
    confidence: float | None = 0.8,
    *,
    probabilities: dict[str, float] | None = None,
    model: str | None = REQUESTED_MODEL,
    input_tokens: int | None = 120,
    output_tokens: int = 8,
) -> dict:
    """A made-up response in the documented shape: ``choice`` gets ``p_choice`` and the other
    names share the rest equally, unless ``probabilities`` is given. ``None`` drops a field."""
    if probabilities is None:
        rest = (1.0 - p_choice) / (len(BANKING77_INTENTS) - 1)
        probabilities = {n: (p_choice if n == choice else rest) for n in BANKING77_INTENTS}
    answer = {"choice": choice, "probabilities": probabilities, "confidence": confidence}
    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    body = {"model": model, "answers": {"intent": _present(answer)}, "usage": _present(usage)}
    return _present(body)


def _present(fields: dict) -> dict:
    return {k: v for k, v in fields.items() if v is not None}
