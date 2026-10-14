"""Ticket routers: the contract and the arms that implement it."""

from __future__ import annotations

from .base import ABSTAIN, Router, Usage, check_output
from .decision_arm import (
    DecisionArm,
    DecisionClient,
    DecisionError,
    FakeDecisionProvider,
    decision_arms,
    fake_decision_response,
)
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

ARMS: dict[str, type] = {"A": TfidfArm, "B": LLMArm, "C": DecisionArm}

__all__ = [
    "ABSTAIN",
    "ARMS",
    "Completion",
    "DecisionArm",
    "DecisionClient",
    "DecisionError",
    "FakeDecisionProvider",
    "FakeLLM",
    "LLMArm",
    "LLMClient",
    "ProviderError",
    "ResponseCache",
    "Router",
    "TfidfArm",
    "Usage",
    "check_output",
    "decision_arms",
    "fake_completion",
    "fake_decision_response",
    "llm_arms",
]
