"""A provider for a local Ollama server, over the standard library only.

One ``POST /api/generate`` per prompt with ``"stream": false`` and ``"logprobs": true``. The
client records what Ollama reports and invents nothing: a missing field raises
:class:`ProviderError` naming it, and there are no retries. Nothing here runs until
:meth:`OllamaProvider.complete` is called, so constructing a provider makes no connection.

Recorded per response: the reply text, one ``(token, logprob)`` pair per generated token,
``prompt_eval_count`` (the full prompt length) and ``prompt_eval_cached_count`` (the part
Ollama reused) stored raw, ``eval_count``, and model-side seconds
``(total_duration - load_duration) / 1e9`` (Ollama reports durations in nanoseconds; load
time is excluded). ``eval_count`` can exceed the number of log-probs by one, the stop token.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ..evaluate import RANDOM_STATE
from .llm_arm import Completion, ProviderError

DEFAULT_MODEL = "qwen2.5:7b-instruct-q3_K_M"
DEFAULT_HOST = "http://localhost:11434"

_REQUIRED = ("response", "logprobs", "total_duration", "prompt_eval_count", "eval_count")


class OllamaProvider:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        *,
        num_predict: int = 40,
        timeout: float = 300.0,
        opener=None,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._opener = opener if opener is not None else urllib.request.urlopen
        self.options = {"temperature": 0, "seed": RANDOM_STATE, "num_predict": num_predict}
        self.identity = "ollama:" + json.dumps(
            {"model": model, "logprobs": True, "options": self.options}, sort_keys=True
        )

    def complete(self, prompt: str) -> Completion:
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "logprobs": True,
            "keep_alive": "30m",
            "options": self.options,
        }
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"Ollama returned HTTP {exc.code}{_error_text(exc)}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f"Ollama could not be reached at {self.host}: {exc}") from exc

        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise ProviderError("Ollama returned a reply that is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ProviderError("Ollama returned JSON that is not an object")
        for field in _REQUIRED:
            if payload.get(field) is None:
                hint = (
                    ": Ollama may be too old for log-probs, or the model does not support them"
                    if field == "logprobs"
                    else ""
                )
                raise ProviderError(f"Ollama reply has no '{field}' field{hint}")
        try:
            return Completion(
                text=str(payload["response"]),
                logprobs=tuple(
                    (str(entry["token"]), float(entry["logprob"])) for entry in payload["logprobs"]
                ),
                prompt_tokens=int(payload["prompt_eval_count"]),
                cached_prompt_tokens=int(payload.get("prompt_eval_cached_count") or 0),
                output_tokens=int(payload["eval_count"]),
                seconds=(int(payload["total_duration"]) - int(payload.get("load_duration") or 0))
                / 1e9,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError(f"Ollama reply has an unexpected shape: {exc!r}") from exc


def _error_text(exc: urllib.error.HTTPError) -> str:
    """``": <message>"`` from an error response body that parses, else an empty string."""
    try:
        message = json.loads(exc.read())["error"]
    except Exception:  # the body is optional detail; never let it mask the HTTP error
        return ""
    return f": {message}"
