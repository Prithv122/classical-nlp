"""The Ollama client, driven through a fake opener. No socket is ever opened."""

from __future__ import annotations

import copy
import email.message
import io
import json
import math
import urllib.error

import pytest

from classicalnlp import tickets
from classicalnlp.data import Corpus
from classicalnlp.evaluate import RANDOM_STATE
from classicalnlp.router import LLMArm, LLMClient
from classicalnlp.router.llm_arm import Provider, ProviderError
from classicalnlp.router.ollama import DEFAULT_HOST, DEFAULT_MODEL, OllamaProvider

NAMES = list(tickets.BANKING77_INTENTS)

# Shaped like a real reply; values rounded; not a measurement.
CANNED = {
    "model": "qwen2.5:7b-instruct-q3_K_M",
    "response": "intent: card_arrival\nconfidence: 0.8",
    "done": True,
    "done_reason": "stop",
    "logprobs": [
        {"token": "intent", "logprob": -0.0005},
        {"token": ":", "logprob": 0},
        {"token": " card", "logprob": -0.0000066757},
        {"token": "_arr", "logprob": -0.00001},
        {"token": "ival", "logprob": -0.00002},
        {"token": "\n", "logprob": -0.003},
        {"token": "confidence", "logprob": 0},
        {"token": ":", "logprob": 0},
        {"token": " ", "logprob": 0},
        {"token": "0", "logprob": 0},
        {"token": ".", "logprob": 0},
        {"token": "8", "logprob": -0.46},
    ],
    "total_duration": 16837220300,
    "load_duration": 12932650100,
    "prompt_eval_count": 70,
    "prompt_eval_cached_count": 0,
    "eval_count": 13,
}


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._body


class FakeOpener:
    def __init__(self, body: bytes | None = None, error: Exception | None = None) -> None:
        self.body = json.dumps(CANNED).encode() if body is None else body
        self.error = error
        self.requests: list = []
        self.timeouts: list = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        self.timeouts.append(timeout)
        if self.error is not None:
            raise self.error
        return _Response(self.body)


def _without(field):
    reply = copy.deepcopy(CANNED)
    del reply[field]
    return json.dumps(reply).encode()


def _provider(opener, **kwargs):
    return OllamaProvider(opener=opener, **kwargs)


# 13 -- the request ----------------------------------------------------------------------


@pytest.mark.parametrize("host", [DEFAULT_HOST, DEFAULT_HOST + "/"])
def test_request_shape(host):
    opener = FakeOpener()
    _provider(opener, host=host, timeout=12.5).complete("a prompt")
    (request,) = opener.requests
    assert request.full_url == DEFAULT_HOST + "/api/generate"
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data) == {
        "model": DEFAULT_MODEL,
        "prompt": "a prompt",
        "stream": False,
        "logprobs": True,
        "keep_alive": "30m",
        "options": {"temperature": 0, "seed": RANDOM_STATE, "num_predict": 40},
    }
    assert opener.timeouts == [12.5]


# 14 -- a reply shaped like a real one ---------------------------------------------------


def test_reply_becomes_a_completion():
    c = _provider(FakeOpener()).complete("p")
    assert c.text == "intent: card_arrival\nconfidence: 0.8"
    assert c.seconds == (16837220300 - 12932650100) / 1e9
    assert (c.prompt_tokens, c.cached_prompt_tokens, c.output_tokens) == (70, 0, 13)
    assert len(c.logprobs) == 12  # one fewer than eval_count: the stop token has no entry
    assert "".join(token for token, _ in c.logprobs) == c.text


def test_both_variants_read_the_one_reply():
    corpus = Corpus(documents=["x"], labels=[0], label_names=list(NAMES))
    client = LLMClient(_provider(FakeOpener()))
    logprob = LLMArm(client, "logprob").fit(corpus)
    verbal = LLMArm(client, "verbal").fit(corpus)
    labels, confidence = logprob.predict_top(["where is my card"])
    # Only the three intent tokens " card", "_arr", "ival" count. The "intent" token's
    # -0.0005, the "\n" and the verbal tokens do not.
    assert labels.tolist() == [NAMES.index("card_arrival")]
    assert confidence[0] == pytest.approx(math.exp(-0.0000066757 - 0.00001 - 0.00002))
    labels, confidence = verbal.predict_top(["where is my card"])
    assert labels.tolist() == [NAMES.index("card_arrival")]
    assert confidence.tolist() == [0.8]
    assert client.calls == 1


# 15 -- fail loud, never invent ----------------------------------------------------------


@pytest.mark.parametrize(
    "field", ["response", "logprobs", "total_duration", "prompt_eval_count", "eval_count"]
)
def test_a_missing_field_is_named(field):
    with pytest.raises(ProviderError, match=field):
        _provider(FakeOpener(_without(field))).complete("p")


def test_a_missing_logprobs_field_points_at_the_server_version():
    with pytest.raises(ProviderError, match="too old"):
        _provider(FakeOpener(_without("logprobs"))).complete("p")


def test_optional_fields_default_to_zero():
    no_load = _provider(FakeOpener(_without("load_duration"))).complete("p")
    assert no_load.seconds == 16837220300 / 1e9
    no_cached = _provider(FakeOpener(_without("prompt_eval_cached_count"))).complete("p")
    assert no_cached.cached_prompt_tokens == 0


# 16 -- failures -------------------------------------------------------------------------


def test_an_unreachable_server_names_the_host():
    error = urllib.error.URLError("connection refused")
    with pytest.raises(ProviderError, match="could not be reached at http://localhost:11434"):
        _provider(FakeOpener(error=error)).complete("p")


def test_a_timeout_is_a_provider_error():
    with pytest.raises(ProviderError, match="could not be reached"):
        _provider(FakeOpener(error=TimeoutError("timed out"))).complete("p")


def test_an_http_error_carries_the_servers_message():
    body = io.BytesIO(b'{"error": "model \'x\' not found"}')
    error = urllib.error.HTTPError(
        DEFAULT_HOST + "/api/generate", 404, "Not Found", email.message.Message(), body
    )
    with pytest.raises(ProviderError, match="not found"):
        _provider(FakeOpener(error=error)).complete("p")


def test_a_reply_that_is_not_json_is_a_provider_error():
    with pytest.raises(ProviderError, match="JSON"):
        _provider(FakeOpener(b"<html>nope</html>")).complete("p")


# 17 -- identity -------------------------------------------------------------------------


def test_identity_tracks_the_model_and_the_generation_settings():
    base = OllamaProvider().identity
    assert base == OllamaProvider().identity
    assert base != OllamaProvider("another-model").identity
    assert base != OllamaProvider(num_predict=41).identity
    assert OllamaProvider(host="http://elsewhere:1").identity == base  # where it runs is not


def test_provider_satisfies_the_protocol_without_connecting():
    provider = OllamaProvider()
    assert isinstance(provider, Provider)
    assert isinstance(provider.identity, str)
    assert callable(provider.complete)
