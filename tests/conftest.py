"""Test-wide guards: no credential store, no key, and no real HTTPS connection."""

from __future__ import annotations

import http.client
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _no_credentials(monkeypatch):
    monkeypatch.setenv("SECRETS_NO_KEYRING", "1")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def _no_real_https(monkeypatch, request):
    """A test that wants a transport injects its own connection factory.

    Tests marked ``network`` (the Banking77 download) are the only ones that may connect.
    """
    if request.node.get_closest_marker("network"):
        return

    class Refused(http.client.HTTPSConnection):
        def __init__(self, *args, **kwargs):
            raise AssertionError("a test tried to open a real HTTPS connection")

    monkeypatch.setattr(http.client, "HTTPSConnection", Refused)


@pytest.fixture
def load_script():
    """Import a file from scripts/ (it is not a package) by name."""

    def load(name: str):
        path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load
