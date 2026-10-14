"""The TypeSafe direct route: ``POST https://api.typesafe.ai/v1/systemone`` over the stdlib.

One persistent ``http.client.HTTPSConnection`` is reused across tickets and reopened after a
drop. :meth:`TypeSafeProvider.decide` makes exactly one HTTP attempt and raises
:class:`~classicalnlp.router.decision_arm.AttemptFailed` when it fails; retries, the ledger and
the spending caps belong to :class:`~classicalnlp.router.decision_arm.DecisionClient`.

Constructing a provider opens no connection and reads no credential. The key is read when the
first request is about to be sent: from ``TYPESAFE_API_KEY``, else from the keyring entry
``classical-nlp`` / ``TYPESAFE_API_KEY`` (skipped when ``SECRETS_NO_KEYRING=1`` or the optional
``keyring`` package is absent). The key is used only in the request header and never appears in
a repr, log, exception message, cache, ledger or run row.

The wire format here follows the documented schema; it has not been exercised against the real
service by this module's tests, which use an injected connection factory.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import os
import time
from collections.abc import Callable, Mapping

from .decision_arm import (
    ENDPOINT,
    TIMEOUT_SECONDS,
    AttemptFailed,
    DecisionError,
    RawResponse,
)

HOST = "api.typesafe.ai"
PATH = "/v1/systemone"
MODELS_PATH = "/v1/models"
KEY_NAME = "TYPESAFE_API_KEY"
KEYRING_SERVICE = "classical-nlp"


class MissingCredential(DecisionError):
    pass


def _checked(key: str) -> str:
    """The key without surrounding whitespace. A key with anything but printable ASCII left
    inside is refused, because ``http.client`` would quote it in its error message."""
    key = key.strip()
    if not all(33 <= ord(c) <= 126 for c in key):
        raise MissingCredential(
            f"{KEY_NAME} contains whitespace or control characters; store the key again"
        )
    return key


def resolve_api_key(environ: Mapping[str, str] | None = None) -> str:
    """The key from the environment, else the keyring; raise naming ``TYPESAFE_API_KEY``."""
    environ = os.environ if environ is None else environ
    if (environ.get(KEY_NAME) or "").strip():
        return _checked(environ[KEY_NAME])
    if environ.get("SECRETS_NO_KEYRING") != "1":
        try:
            import keyring

            stored = keyring.get_password(KEYRING_SERVICE, KEY_NAME)
        except Exception:  # no package, no backend or a locked store: same as no entry
            stored = None
        if stored and stored.strip():
            return _checked(stored)
    raise MissingCredential(
        f"no API key: set {KEY_NAME} in the environment, or store it with keyring under "
        f"service {KEYRING_SERVICE!r} (pip extra 'secrets')"
    )


class TypeSafeProvider:
    identity = ENDPOINT

    def __init__(
        self,
        *,
        host: str = HOST,
        path: str = PATH,
        timeout: float = TIMEOUT_SECONDS,
        connection_factory: Callable[[str, float], object] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.host, self.path, self.timeout = host, path, timeout
        self._factory = connection_factory or self._https
        self._environ = environ
        self._key: str | None = None
        self._connection = None

    @staticmethod
    def _https(host: str, timeout: float):
        return http.client.HTTPSConnection(host, timeout=timeout)

    def __repr__(self) -> str:
        return f"TypeSafeProvider(host={self.host!r}, path={self.path!r})"

    def _drop(self) -> None:
        if self._connection is not None:
            with contextlib.suppress(Exception):  # already broken; the next call reconnects
                self._connection.close()
        self._connection = None

    def _exchange(self, method, path, body):
        """One request: ``(status, payload, seconds, cold_start)``, or AttemptFailed."""
        if self._key is None:  # read at the first request, once
            self._key = resolve_api_key(self._environ)
        headers = {"Authorization": f"Bearer {self._key}"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        cold = self._connection is None
        if cold:
            self._connection = self._factory(self.host, self.timeout)
        start = time.perf_counter()
        rejected = False
        try:
            self._connection.request(method, path, body=body, headers=headers)
            response = self._connection.getresponse()
            payload = response.read()
        except TimeoutError:
            self._drop()
            raise AttemptFailed("timeout") from None
        except (OSError, http.client.HTTPException):
            self._drop()
            raise AttemptFailed("connection") from None
        except ValueError:
            # http.client quotes a rejected header value, which holds the key. Raising here would
            # keep that error as __context__ even with ``from None``, so raise after the block.
            rejected = True
        if rejected:
            self._drop()
            raise DecisionError("the request was rejected before sending (invalid header)")
        seconds = time.perf_counter() - start
        if getattr(response, "will_close", False):
            self._drop()
        return response.status, payload, seconds, cold

    def decide(self, request: dict) -> RawResponse:
        body = json.dumps(request).encode("utf-8")
        status, payload, seconds, cold = self._exchange("POST", self.path, body)
        if status != 200:
            raise AttemptFailed(f"http_{status}", status)
        try:
            parsed = json.loads(payload)
        except ValueError:
            parsed = None
        if not isinstance(parsed, dict):
            raise AttemptFailed("non_json", 200)
        return RawResponse(parsed, seconds=seconds, cold_start=cold)

    def ping(self) -> tuple[int, float]:
        """``(HTTP status, seconds)`` of an unbilled ``GET /v1/models`` on the same connection."""
        status, _, seconds, _ = self._exchange("GET", MODELS_PATH, None)
        return status, seconds
