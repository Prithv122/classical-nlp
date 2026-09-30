"""Ticket routers: the contract and the arms that implement it."""

from __future__ import annotations

from .base import ABSTAIN, Router, check_output
from .tfidf_arm import TfidfArm

ARMS: dict[str, type] = {"A": TfidfArm}

__all__ = ["ABSTAIN", "ARMS", "Router", "TfidfArm", "check_output"]
