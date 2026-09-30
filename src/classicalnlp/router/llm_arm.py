"""Arm B: a local LLM asked to pick one of the 77 intents, with two confidence variants.

One prompt per ticket, one response per prompt, and two arms read that same response:

* ``B-logprob``: confidence is ``exp`` of the summed token log-probabilities of the tokens
  that spell the chosen intent, clipped to [0, 1]. It is the probability of the intent
  string, not normalised over the 77 intents.
* ``B-verbal``: confidence is the number the model states on its second line, used only if
  it is a plain decimal in [0, 1].

The reply must be exactly two lines, ``intent: <name>`` then ``confidence: <0..1>``. The
intent must equal a label name exactly (case-sensitive). There is no repair, fuzzy matching,
case folding, retry, JSON mode or constrained decoding: anything unusable abstains
(:data:`~classicalnlp.router.base.ABSTAIN`, confidence 0.0). Abstention is decided per
variant: an unusable intent abstains both; a usable intent with an unusable stated number
abstains only ``B-verbal``; a usable intent with unusable log-probabilities abstains only
``B-logprob``.

The ticket text is the last thing in the prompt, so consecutive prompts share the whole
prefix and a server that caches prompt prefixes can reuse it.

:class:`FakeLLM` and :func:`fake_completion` are test doubles. Their numbers exist to
exercise this code and are not model behaviour.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..data import Corpus
from .base import ABSTAIN, Usage, check_output


@dataclass(frozen=True)
class Completion:
    """One model response. Counts and seconds are stored as the provider reported them."""

    text: str
    logprobs: tuple[tuple[str, float], ...] | None
    prompt_tokens: int
    output_tokens: int
    seconds: float
    cached_prompt_tokens: int = 0


class ProviderError(ValueError):
    """A provider could not produce a usable response (a ValueError so the CLI exits 2)."""


@runtime_checkable
class Provider(Protocol):
    identity: str

    def complete(self, prompt: str) -> Completion: ...


PROMPT_TEMPLATE = (
    "You route customer-support messages for a bank. "
    "Pick the single intent that best matches the message.\n"
    "\n"
    "Intents:\n"
    "{intents}\n"
    "\n"
    "Reply with exactly two lines and nothing else:\n"
    "intent: <one intent from the list, copied exactly as written, "
    "including capitals and punctuation>\n"
    "confidence: <your probability that the intent is correct, a number from 0 to 1>\n"
    "\n"
    "Message: {query}"
)

_QUERY_MARKER = "\nMessage: "


def build_prompt(names: Sequence[str], query: str) -> str:
    """The prompt for one ticket. ``str.format`` does not re-scan the query for braces."""
    return PROMPT_TEMPLATE.format(intents="\n".join(names), query=query)


def split_prompt(prompt: str) -> tuple[str, str]:
    """``(prefix, query)``: the part shared by every ticket, and the ticket text."""
    prefix, _, query = prompt.partition(_QUERY_MARKER)
    return prefix, query


_INTENT_LINE = re.compile(r"\s*intent:[ \t]*(\S+)[ \t]*(?:\n|$)")
_VERBAL_LINE = re.compile(r"\s*confidence:[ \t]*((?:0|1)(?:\.\d+)?|\.\d+)[ \t]*(?:\n|$)")


def parse_reply(
    text: str, names: Sequence[str]
) -> tuple[int | None, float | None, tuple[int, int] | None]:
    """``(intent_label, verbal_confidence, intent_char_span)``.

    The intent is read from the first line only and must equal a name exactly. The stated
    confidence is read from the next line and must be a plain decimal in [0, 1]. Lines after
    the second are ignored. An unusable intent gives ``(None, None, None)``; an unusable
    stated number gives ``(label, None, span)``.
    """
    match = _INTENT_LINE.match(text)
    if match is None or match.group(1) not in names:
        return None, None, None
    label = list(names).index(match.group(1))
    span = match.span(1)
    verbal = _VERBAL_LINE.match(text[match.end() :])
    if verbal is None:
        return label, None, span
    value = float(verbal.group(1))
    return label, (value if 0.0 <= value <= 1.0 else None), span


def intent_logprob_confidence(
    text: str, logprobs: Sequence[tuple[str, float]] | None, span: tuple[int, int]
) -> float | None:
    """Probability of the intent string: ``exp`` of the summed log-probs of its tokens.

    Every token whose character range overlaps ``span`` counts. A token that straddles the
    span boundary (e.g. ``"_arrival\\n"``) is included whole; this is documented and
    accepted. ``None`` if there are no log-probs, the tokens do not concatenate to exactly
    ``text``, or any log-prob is nan.
    """
    if logprobs is None or "".join(token for token, _ in logprobs) != text:
        return None
    if any(math.isnan(logprob) for _, logprob in logprobs):
        return None
    start, end = span
    total = 0.0
    position = 0
    for token, logprob in logprobs:
        first, position = position, position + len(token)
        if position > first and first < end and position > start:
            total += logprob
    return math.exp(min(total, 0.0))


def cache_key(identity: str, prompt: str) -> str:
    return hashlib.sha256((identity + "\n" + prompt).encode("utf-8")).hexdigest()


class ResponseCache:
    """Responses keyed by :func:`cache_key`, optionally persisted as JSONL.

    Each ``put`` appends one line and flushes before returning, so a run that is killed
    keeps every response it had received. On open, lines that do not parse or lack a field
    are skipped and counted in :attr:`skipped`. The file and its directory are created at
    the first write, never on open. ``path=None`` keeps everything in memory.
    """

    def __init__(self, path: str | Path | None) -> None:
        self.path = None if path is None else Path(path)
        self.skipped = 0
        self._items: dict[str, Completion] = {}
        if self.path is not None and self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    logprobs = record["logprobs"]
                    self._items[record["key"]] = Completion(
                        text=record["text"],
                        logprobs=None
                        if logprobs is None
                        else tuple((str(token), float(value)) for token, value in logprobs),
                        prompt_tokens=int(record["prompt_tokens"]),
                        cached_prompt_tokens=int(record["cached_prompt_tokens"]),
                        output_tokens=int(record["output_tokens"]),
                        seconds=float(record["seconds"]),
                    )
                except (ValueError, KeyError, TypeError):
                    self.skipped += 1

    def get(self, key: str) -> Completion | None:
        return self._items.get(key)

    def put(self, key: str, completion: Completion) -> None:
        self._items[key] = completion
        if self.path is None:
            return
        record = {
            "key": key,
            "text": completion.text,
            "logprobs": None
            if completion.logprobs is None
            else [[token, value] for token, value in completion.logprobs],
            "prompt_tokens": completion.prompt_tokens,
            "cached_prompt_tokens": completion.cached_prompt_tokens,
            "output_tokens": completion.output_tokens,
            "seconds": completion.seconds,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        prefix = ""
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("rb") as handle:
                handle.seek(-1, 2)
                # A killed run can leave a partial last line; start the new record on its own.
                prefix = "" if handle.read(1) == b"\n" else "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(prefix + json.dumps(record) + "\n")
            handle.flush()


class LLMClient:
    """A provider behind a cache. Always memoises in memory, which is what lets the two
    variants share one set of responses."""

    def __init__(self, provider: Provider, cache: ResponseCache | None = None) -> None:
        self.provider = provider
        self.cache = cache if cache is not None else ResponseCache(None)
        self.calls = 0
        self.hits = 0

    def complete(self, prompt: str) -> Completion:
        key = cache_key(self.provider.identity, prompt)
        cached = self.cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        completion = self.provider.complete(prompt)
        self.cache.put(key, completion)
        self.calls += 1
        return completion


class LLMArm:
    description = "local LLM (zero-shot, 77 intents)"

    def __init__(self, client: LLMClient, variant: str = "logprob") -> None:
        if variant not in ("logprob", "verbal"):
            raise ValueError(f"variant must be 'logprob' or 'verbal', got {variant!r}")
        self.client = client
        self.variant = variant
        self.name = f"B-{variant}"
        self.usage: Usage | None = None
        self._names: list[str] | None = None

    def fit(self, corpus: Corpus) -> LLMArm:
        self._names = list(corpus.label_names)
        return self

    def predict_top(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        if self._names is None:
            raise ValueError("LLMArm is not fitted; call fit() first")
        labels: list[int] = []
        confidence: list[float] = []
        completions: list[Completion] = []
        for text in texts:
            completion = self.client.complete(build_prompt(self._names, text))
            completions.append(completion)
            label, verbal, span = parse_reply(completion.text, self._names)
            value = None
            if label is not None:
                if self.variant == "verbal":
                    value = verbal
                else:
                    value = intent_logprob_confidence(completion.text, completion.logprobs, span)
            if value is None:
                labels.append(ABSTAIN)
                confidence.append(0.0)
            else:
                labels.append(label)
                confidence.append(value)
        self.usage = Usage(
            seconds=np.array([c.seconds for c in completions], dtype=float),
            prompt_tokens=np.array([c.prompt_tokens for c in completions], dtype=int),
            cached_prompt_tokens=np.array([c.cached_prompt_tokens for c in completions], dtype=int),
            output_tokens=np.array([c.output_tokens for c in completions], dtype=int),
        )
        return check_output(labels, confidence, len(texts))


def llm_arms(client: LLMClient) -> list[LLMArm]:
    """Both variants over one client, so one response per ticket feeds both."""
    return [LLMArm(client, "logprob"), LLMArm(client, "verbal")]


class FakeLLM:
    """Test double. ``replies`` maps a ticket text to a :class:`Completion`, or is a callable
    taking the ticket text. Its outputs are not model behaviour."""

    identity = "fake"

    def __init__(self, replies: Mapping[str, Completion] | Callable[[str], Completion]) -> None:
        self.replies = replies
        self.calls = 0
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> Completion:
        self.calls += 1
        self.prompts.append(prompt)
        _, query = split_prompt(prompt)
        if callable(self.replies):
            return self.replies(query)
        try:
            return self.replies[query]
        except KeyError:
            raise KeyError(f"FakeLLM has no reply for the query {query!r}") from None


def fake_completion(
    intent: str,
    verbal: float | str | None = 0.9,
    *,
    logprob: float = -0.1,
    seconds: float = 1.0,
    prompt_tokens: int = 600,
    cached_prompt_tokens: int = 0,
    output_tokens: int = 14,
) -> Completion:
    """A made-up reply in the requested format. The tokens concatenate exactly to the text;
    the intent tokens each carry ``logprob`` and every other token carries 0.0."""
    parts = intent.split("_")
    tokens = [("intent", 0.0), (":", 0.0), (" " + parts[0], logprob)]
    tokens += [("_" + part, logprob) for part in parts[1:]]
    text = f"intent: {intent}"
    if verbal is not None:
        text += f"\nconfidence: {verbal}"
        tokens += [("\n", 0.0), ("confidence", 0.0), (":", 0.0), (" ", 0.0), (str(verbal), 0.0)]
    return Completion(
        text=text,
        logprobs=tuple(tokens),
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        seconds=seconds,
        cached_prompt_tokens=cached_prompt_tokens,
    )
