"""The abc route mode, arm C in route, jev-probe and jev-rtt, against fakes only.

The abc corpora are replaced by small synthetic ones, so nothing here touches Banking77 and every
number is scripted to test the code. No network.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from classicalnlp import cli, tickets
from classicalnlp.data import Corpus
from classicalnlp.router import FakeDecisionProvider, fake_decision_response

NAMES = list(tickets.BANKING77_INTENTS)


def _fixture_corpora():
    fit_all, held = tickets.validation_split(tickets.load_fixture())
    fit, val = tickets.validation_split(fit_all)
    return fit, val, held


def _use_corpora(monkeypatch, corpora=None):
    corpora = corpora or _fixture_corpora()
    monkeypatch.setattr(cli, "_abc_corpora", lambda args: corpora)
    return corpora


def _use_provider(monkeypatch, replies):
    provider = FakeDecisionProvider(replies)
    monkeypatch.setattr(cli, "_make_decision_provider", lambda args: provider)
    return provider


def _reply(request):
    return fake_decision_response(NAMES[len(request["state"]) % 5], 0.6, 0.7)


def _first(tmp_path):
    """``--new-ledger`` while this test has no study ledger yet; later runs reuse it."""
    return [] if (tmp_path / "l.jsonl").exists() else ["--new-ledger"]


def _route(tmp_path, split, *extra, arm="C"):
    argv = ["route", "--arm", arm, "--partition", "abc", "--split", split, "--data-dir", "d"]
    if arm == "C":
        argv += ["--cache", str(tmp_path / "c.jsonl"), "--ledger", str(tmp_path / "l.jsonl")]
        argv += _first(tmp_path)
    return [*argv, *extra]


def _lines(path):
    return path.read_text().splitlines()


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--arm", "A", "--partition", "abc", "--split", "test", "--data-dir", "d"], "not read"),
        (["--arm", "A", "--partition", "abc", "--split", "val", "--fixture"], "fixture"),
        (["--arm", "A", "--split", "heldout", "--fixture"], "needs --partition abc"),
        (["--arm", "C", "--split", "val", "--fixture", "--cache", "c", "--ledger", "l"], "abc"),
        (["--arm", "C", "--partition", "abc", "--data-dir", "d", "--ledger", "l"], "--cache"),
        (["--arm", "C", "--partition", "abc", "--data-dir", "d", "--cache", "c"], "--ledger"),
        (["--arm", "A", "--partition", "abc", "--data-dir", "d", "--ledger", "l"], "arm C only"),
        (["--arm", "A", "--partition", "abc", "--split", "heldout", "--limit", "5",
          "--data-dir", "d"], "H is run whole"),
    ],
)  # fmt: skip
def test_route_refusals_come_before_any_data_or_provider(monkeypatch, capsys, argv, message):
    def boom(args):
        raise AssertionError("must not be reached")

    monkeypatch.setattr(cli, "_abc_corpora", boom)
    monkeypatch.setattr(cli, "_make_decision_provider", boom)
    assert cli.main(["route", *argv]) == 2
    captured = capsys.readouterr()
    assert message in captured.err and "source:" not in captured.out


def test_arm_a_is_fit_on_the_refit_set_and_the_heldout_label_is_written(
    monkeypatch, tmp_path, capsys
):
    fit, val, held = _use_corpora(monkeypatch)
    save = tmp_path / "a.jsonl"
    assert cli.main(_route(tmp_path, "heldout", "--save", str(save), arm="A")) == 0
    out = capsys.readouterr().out
    assert f"fit {len(fit)} rows, evaluated {len(held)} rows" in out
    rows = [json.loads(line) for line in _lines(save)]
    assert {r["split"] for r in rows} == {"heldout"} and len(rows) == 2 * len(held)
    assert [r["y_true"] for r in rows[: len(held)]] == held.labels
    assert cli.main(_route(tmp_path, "val", "--save", str(save), arm="A")) == 0
    assert f"fit {len(fit)} rows, evaluated {len(val)} rows" in capsys.readouterr().out
    assert {json.loads(line)["split"] for line in _lines(save)} == {"val"}


def test_v2_still_refuses_arm_c_with_a_cache_and_keeps_its_arm_a_check(capsys):
    argv = ["route", "--arm", "A", "--fixture", "--cache", "x.jsonl"]
    assert cli.main(argv) == 2
    assert "arm B only" in capsys.readouterr().err


def test_arm_c_runs_val_then_heldout_without_a_second_v_call(monkeypatch, tmp_path, capsys):
    _, val, held = _use_corpora(monkeypatch)
    provider = _use_provider(monkeypatch, _reply)
    assert cli.main(_route(tmp_path, "val")) == 0
    out = capsys.readouterr().out
    assert f"jev requests: {len(set(val.documents))} sent" in out and "C-vendor" in out
    v_calls = provider.calls
    assert v_calls == len(_lines(tmp_path / "c.jsonl")) == len(set(val.documents))
    assert len(_lines(tmp_path / "l.jsonl")) == v_calls

    save = tmp_path / "h.jsonl"
    assert cli.main(_route(tmp_path, "heldout", "--save", str(save))) == 0
    assert provider.calls == v_calls + len(set(held.documents))
    assert all(r["split"] == "heldout" for r in map(json.loads, _lines(save)))
    assert len(_lines(save)) == 4 * len(held)
    assert cli.main(_route(tmp_path, "heldout")) == 0  # a finished run costs nothing
    assert provider.calls == v_calls + len(set(held.documents))
    assert "jev requests: 0 sent" in capsys.readouterr().out


def test_heldout_is_refused_before_any_call_unless_v_is_fully_cached(monkeypatch, tmp_path, capsys):
    _use_corpora(monkeypatch)
    provider = _use_provider(monkeypatch, _reply)
    assert cli.main(_route(tmp_path, "heldout")) == 2
    assert "no V call" in capsys.readouterr().err
    assert provider.calls == 0
    assert not (tmp_path / "c.jsonl").exists() and not (tmp_path / "l.jsonl").exists()


def test_a_missing_key_exits_two_naming_it_and_creates_no_file(monkeypatch, tmp_path, capsys):
    _use_corpora(monkeypatch)  # the default provider is the real one; the guard forbids HTTPS
    assert cli.main(_route(tmp_path, "val")) == 2
    assert "TYPESAFE_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "c.jsonl").exists() and not (tmp_path / "l.jsonl").exists()


def test_a_run_that_hits_a_stop_condition_exits_two_and_keeps_what_it_fetched(
    monkeypatch, tmp_path, capsys
):
    _use_corpora(monkeypatch)
    wrong = fake_decision_response("card_arrival", model="jev-9.9.9")
    _use_provider(monkeypatch, [fake_decision_response("card_arrival"), wrong])
    assert cli.main(_route(tmp_path, "val")) == 2
    assert "jev-9.9.9" in capsys.readouterr().err
    assert len(_lines(tmp_path / "c.jsonl")) == 1 and len(_lines(tmp_path / "l.jsonl")) == 2


def test_price_flag_sets_the_ledger_dollars(monkeypatch, tmp_path):
    _use_corpora(monkeypatch)
    _use_provider(monkeypatch, _reply)
    assert cli.main(_route(tmp_path, "val", "--price-per-mtok", "1000")) == 0
    first = json.loads(_lines(tmp_path / "l.jsonl")[0])
    assert first["dollars"] == pytest.approx(0.12)


# -- jev-probe and jev-rtt ------------------------------------------------------------------


def _synthetic_v():
    docs = [f"ticket {i}" for i in range(2001)]
    v = Corpus(documents=docs, labels=[0] * 2001, label_names=NAMES)
    small = Corpus(documents=["x"], labels=[0], label_names=NAMES)
    return small, v, small


def _probe(tmp_path):
    paths = [str(tmp_path / name) for name in ("c.jsonl", "l.jsonl", "log.jsonl")]
    argv = ["jev-probe", "--data-dir", "d"]
    return [*argv, "--cache", paths[0], "--ledger", paths[1], "--log", paths[2], *_first(tmp_path)]


def test_probe_positions_are_the_protocols_seeded_draw():
    expected = np.random.default_rng(20261014).choice(2001, 20, replace=False)
    assert cli.probe_positions(2001).tolist() == expected.tolist()
    assert len(set(cli.probe_positions(2001).tolist())) == 20


def test_probe_sends_the_protocol_rows_then_repeats_them_outside_the_cache(
    monkeypatch, tmp_path, capsys
):
    _use_corpora(monkeypatch, _synthetic_v())
    provider = _use_provider(monkeypatch, lambda request: fake_decision_response("age_limit"))
    assert cli.main(_probe(tmp_path)) == 0
    out = capsys.readouterr().out
    rows = np.random.default_rng(20261014).choice(2001, 20, replace=False)
    sent = [r["state"] for r in provider.requests]
    assert sent == [f"ticket {i}" for i in rows] * 2 and provider.calls == 40
    assert len(_lines(tmp_path / "c.jsonl")) == 20  # the repeat is never a record
    assert len(_lines(tmp_path / "log.jsonl")) == 20
    kinds = [json.loads(line)["kind"] for line in _lines(tmp_path / "l.jsonl")]
    assert kinds == ["probe"] * 20 + ["self_agreement"] * 20
    for expected in (
        "reported version: jev-1.13.0",
        "schema check: 20/20 first pass, 20/20 second pass",
        "median input_tokens: 120",
        "same choice: 20/20",
        "largest absolute probability difference: 0.000000",
        "G2 checks: pass",
    ):
        assert expected in out
    assert "p50 latency" in out
    assert cli.main(_probe(tmp_path)) == 0  # the first pass is now free
    assert provider.calls == 60


def test_probe_reports_disagreement_without_failing_g2(monkeypatch, tmp_path, capsys):
    _use_corpora(monkeypatch, _synthetic_v())
    seen = {}

    def reply(request):
        n = seen[request["state"]] = seen.get(request["state"], 0) + 1
        return fake_decision_response("age_limit" if n == 1 else "atm_support", 0.5 + n / 10)

    _use_provider(monkeypatch, reply)
    assert cli.main(_probe(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "same choice: 0/20" in out and "difference: 0.000000" not in out


def test_probe_fails_g2_on_a_bad_schema_and_stops_on_a_wrong_version(monkeypatch, tmp_path, capsys):
    _use_corpora(monkeypatch, _synthetic_v())
    short = {n: 0.01 for n in NAMES}
    _use_provider(monkeypatch, lambda r: fake_decision_response("age_limit", probabilities=short))
    assert cli.main(_probe(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "schema check: 0/20 first pass" in out and "G2 checks: FAIL" in out
    _use_provider(monkeypatch, lambda r: fake_decision_response("age_limit", model="jev-2.0"))
    assert cli.main(_probe(tmp_path / "other")) == 2


def _rtt(tmp_path, *extra):
    return ["jev-rtt", "--ledger", str(tmp_path / "l.jsonl"), "--log", str(tmp_path / "r.jsonl"),
            *_first(tmp_path), *extra]  # fmt: skip


def test_rtt_logs_unbilled_requests_and_prints_the_median(monkeypatch, tmp_path, capsys):
    provider = _use_provider(monkeypatch, [])
    assert cli.main(_rtt(tmp_path, "--n", "5")) == 0
    assert provider.pings == 5
    ledger = [json.loads(line) for line in _lines(tmp_path / "l.jsonl")]
    assert [r["kind"] for r in ledger] == ["rtt"] * 5
    assert {r["dollars"] for r in ledger} == {0.0} and len(_lines(tmp_path / "r.jsonl")) == 5
    assert "median round trip over 5 of 5 requests" in capsys.readouterr().out


def test_rtt_stops_on_an_unauthorised_status_and_refuses_n_zero(monkeypatch, tmp_path, capsys):
    class Unauthorised(FakeDecisionProvider):
        def ping(self):
            return 401, 0.01

    monkeypatch.setattr(cli, "_make_decision_provider", lambda args: Unauthorised([]))
    assert cli.main(_rtt(tmp_path)) == 2
    assert "TYPESAFE_API_KEY" in capsys.readouterr().err
    assert cli.main(_rtt(tmp_path, "--n", "0")) == 2


def test_every_paid_command_must_use_the_one_study_ledger(monkeypatch, tmp_path, capsys):
    _use_corpora(monkeypatch)
    provider = _use_provider(monkeypatch, _reply)
    base = ["route", "--arm", "C", "--partition", "abc", "--split", "val", "--data-dir", "d"]
    base += ["--cache", str(tmp_path / "c.jsonl"), "--ledger", str(tmp_path / "l.jsonl")]
    assert cli.main(base) == 2  # a ledger path that does not exist: a typo, or a fresh total
    assert "does not exist" in capsys.readouterr().err
    assert provider.calls == 0 and not (tmp_path / "l.jsonl").exists()
    assert cli.main([*base, "--new-ledger"]) == 0
    assert "study total so far $0.0000 of $5.00" in capsys.readouterr().out
    assert cli.main([*base, "--new-ledger"]) == 2  # never restart an existing ledger
    assert "already exists" in capsys.readouterr().err
    calls = provider.calls
    missing = str(tmp_path / "typo.jsonl")
    probe = ["jev-probe", "--data-dir", "d", "--cache", str(tmp_path / "c.jsonl")]
    probe += ["--ledger", missing, "--log", str(tmp_path / "log.jsonl")]
    rtt = ["jev-rtt", "--ledger", missing, "--log", str(tmp_path / "r.jsonl")]
    for argv in (probe, rtt):
        assert cli.main(argv) == 2
        assert "does not exist" in capsys.readouterr().err
    assert (provider.calls, provider.pings) == (calls, 0)  # refused before any request
    assert not (tmp_path / "typo.jsonl").exists()
