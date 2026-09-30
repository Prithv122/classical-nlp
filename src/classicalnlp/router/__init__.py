"""Ticket routers: the contract and the arms that implement it."""

from __future__ import annotations

from .base import ABSTAIN, Router, Usage, check_output
from .llm_arm import (
    Completion,
    FakeLLM,
    LLMArm,
    LLMClient,
    ProviderError,
    ResponseCache,
    fake_completion,
    llm_arms,
)
from .tfidf_arm import TfidfArm

ARMS: dict[str, type] = {"A": TfidfArm, "B": LLMArm}

__all__ = [
    "ABSTAIN",
    "ARMS",
    "Completion",
    "FakeLLM",
    "LLMArm",
    "LLMClient",
    "ProviderError",
    "ResponseCache",
    "Router",
    "TfidfArm",
    "Usage",
    "check_output",
    "fake_completion",
    "llm_arms",
]
