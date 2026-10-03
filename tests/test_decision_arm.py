"""Arm C is a stub: it is registered, and every attempt to run it fails loudly. No network."""

from __future__ import annotations

import pytest

from classicalnlp import cli, tickets
from classicalnlp.router import ARMS, DecisionArm, NotAvailable, Router


def test_stub_satisfies_the_contract_and_refuses_to_run():
    arm = DecisionArm()
    assert isinstance(arm, Router)
    assert issubclass(NotAvailable, ValueError)
    corpus = tickets.load_fixture()
    for call in (lambda: arm.fit(corpus), lambda: arm.predict_top(["hello"])):
        with pytest.raises(NotAvailable) as raised:
            call()
        assert "not available" in str(raised.value)
        assert "never estimated" in str(raised.value)


def test_route_arm_c_exits_two_with_not_available(capsys):
    assert ARMS["C"] is DecisionArm
    assert cli.main(["route", "--arm", "C", "--split", "val", "--fixture"]) == 2
    captured = capsys.readouterr()
    assert "not available" in captured.err
