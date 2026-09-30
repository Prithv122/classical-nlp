"""Usage accounting, the cost-ratio guard and saved runs. Fixture runs assert invariants only,
never coverage, cost or latency values; scripted numbers exist to test the code."""

from __future__ import annotations

import json

import numpy as np
import pytest

from classicalnlp import cli, costs, study, tickets
from classicalnlp.costs import RoutingCost
from classicalnlp.data import Corpus
from classicalnlp.router import (
    ABSTAIN,
    FakeLLM,
    ProviderError,
    ResponseCache,
    TfidfArm,
    fake_completion,
)
from classicalnlp.router.base import Usage

SAVE_KEYS = {
    "arm",
    "split",
    "calibration",
    "index",
    "y_true",
    "y_pred",
    "confidence",
    "seconds",
    "prompt_tokens",
    "cached_prompt_tokens",
    "output_tokens",
}


class ScriptedRouter:
    name = "S"
    description = "scripted"

    def __init__(self, script):
        self.script = script
        self.fits = 0
        self.predicts = 0

    def fit(self, corpus):
        self.fits += 1
        return self

    def predict_top(self, texts):
        self.predicts += 1
        pairs = [self.script[t] for t in texts]
        return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs], dtype=float)


class UsageRouter(ScriptedRouter):
    """A scripted router that reports usage scaled to the size of each call."""

    name = "U"
    usage = None

    def predict_top(self, texts):
        out = super().predict_top(texts)
        n = len(texts)
        self.usage = Usage(
            seconds=np.full(n, float(n)),
            prompt_tokens=np.full(n, 600),
            cached_prompt_tokens=np.full(n, 500),
            output_tokens=np.full(n, 14),
        )
        return out


def _corpus(documents, labels):
    return Corpus(documents=documents, labels=labels, label_names=list(tickets.BANKING77_INTENTS))


def _scripted(n=12, abstained=(0, 3, 6)):
    docs = [f"d{i}" for i in range(n)]
    script = {d: (ABSTAIN, 0.0) if i in abstained else (0, 0.25) for i, d in enumerate(docs)}
    return script, _corpus(docs, [0] * n)


@pytest.fixture(scope="module")
def fixture_runs():
    fit, val = tickets.validation_split(tickets.load_fixture())
    return study.run_arm(TfidfArm(), fit, val)


def test_the_rule_auto_routes_everything_below_ratio_one():
    # Why ratios below 1 are refused: the rule then auto-routes even a confidence-0.0 row.
    zero = np.array([0.0])
    assert costs.auto_route(zero, RoutingCost.from_ratio(0.5)).tolist() == [True]
    assert costs.auto_route(zero, RoutingCost.from_ratio(1)).tolist() == [False]


@pytest.mark.parametrize("ratios", [(0.5,), (2, 0.99)])
def test_ratios_below_one_are_rejected_before_the_router_runs(ratios):
    script, corpus = _scripted()
    router = ScriptedRouter(script)
    with pytest.raises(ValueError, match="below 1"):
        study.run_arm(router, corpus, corpus, ratios=ratios)
    assert router.fits == 0
    assert router.predicts == 0


def test_abstentions_are_handed_off_at_ratio_one_and_above():
    script, corpus = _scripted()
    idx = [0, 3, 6]
    for run in study.run_arm(ScriptedRouter(script), corpus, corpus, ratios=(1, 2)):
        assert run.confidence[idx].tolist() == [0.0] * 3
        for ratio in (1, 2):
            routed = costs.auto_route(run.confidence, RoutingCost.from_ratio(ratio))
            assert not routed[idx].any()


def test_arm_without_usage_is_timed_per_ticket(fixture_runs):
    for run in fixture_runs:
        assert len(run.usage.seconds) == 12
        assert np.all(run.usage.seconds >= 0)
        assert run.usage.prompt_tokens is None
        assert run.usage.cached_prompt_tokens is None
        assert run.usage.output_tokens is None


def test_router_usage_is_carried_through_on_both_runs():
    script, corpus = _scripted()
    router = UsageRouter(script)
    raw, isotonic = study.run_arm(router, corpus, corpus)
    assert raw.usage is router.usage
    assert isotonic.usage is router.usage


def test_in_test_mode_the_usage_belongs_to_the_test_predictions():
    script = {f"d{i}": (0, 0.5) for i in range(7)}
    val = _corpus([f"d{i}" for i in range(4)], [0] * 4)
    test = _corpus([f"d{i}" for i in range(4, 7)], [0] * 3)
    raw, isotonic = study.run_arm(UsageRouter(script), val, val, test)
    for run in (raw, isotonic):
        assert run.usage.seconds.tolist() == [3.0] * 3


def test_format_runs_has_the_arm_and_an_abstained_count_on_every_line():
    script, corpus = _scripted()
    lines = study.format_runs(study.run_arm(ScriptedRouter(script), corpus, corpus)).splitlines()
    assert "arm" in lines[0] and "abstained" in lines[0]
    assert len(lines) == 1 + 2 * len(costs.SENSITIVITY_RATIOS)
    for line in lines[1:]:
        fields = line.split()
        assert fields[0] == "S"
        assert fields[6] == "3"


def test_format_usage_for_tokens_and_for_an_arm_without_them(fixture_runs):
    script, corpus = _scripted()
    text = study.format_usage(study.run_arm(UsageRouter(script), corpus, corpus))
    assert len(text.splitlines()) == 1
    for expected in ("U:", "latency", "p50", "p95", "prompt tokens", "output tokens"):
        assert expected in text
    plain = study.format_usage(fixture_runs)
    assert plain.startswith("A: latency p50")
    assert "tokens n/a" in plain


def test_save_runs_on_arm_a(fixture_runs, tmp_path):
    path = tmp_path / "nested" / "runs.jsonl"
    assert study.save_runs(fixture_runs, path) == 24
    assert study.save_runs(fixture_runs, path) == 24  # overwrites, does not append
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 24
    for run_number, run in enumerate(fixture_runs):
        chunk = rows[run_number * 12 : (run_number + 1) * 12]
        assert [r["index"] for r in chunk] == list(range(12))
        assert all(set(r) == SAVE_KEYS for r in chunk)
        assert [r["y_pred"] for r in chunk] == run.y_pred.tolist()
        assert [r["y_true"] for r in chunk] == run.y_true.tolist()
        assert [r["confidence"] for r in chunk] == run.confidence.tolist()
        assert [r["calibration"] for r in chunk] == [run.calibration] * 12
        assert all(r["arm"] == "A" and r["split"] == "val" for r in chunk)
        assert all(r["seconds"] is not None for r in chunk)
        assert all(r[k] is None for r in chunk for k in ("prompt_tokens", "cached_prompt_tokens"))
        assert all(r["output_tokens"] is None for r in chunk)


def test_save_runs_fills_tokens_from_usage(tmp_path):
    script, corpus = _scripted(n=10, abstained=(2,))
    runs = study.run_arm(UsageRouter(script), corpus, corpus)
    path = tmp_path / "runs.jsonl"
    assert study.save_runs(runs, path) == 20
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert {r["seconds"] for r in rows} == {10.0}
    assert {r["prompt_tokens"] for r in rows} == {600}
    assert {r["cached_prompt_tokens"] for r in rows} == {500}
    assert {r["output_tokens"] for r in rows} == {14}
    assert rows[2]["y_pred"] == ABSTAIN and rows[2]["confidence"] == 0.0


# -- the route command, arm B, against fakes only ----------------------------------------


def _use_fake_provider(monkeypatch, make):
    made = []

    def factory(args):
        provider = make()
        made.append(provider)
        return provider

    monkeypatch.setattr(cli, "_make_provider", factory)
    return made


def _scripted_reply():
    return FakeLLM(lambda query: fake_completion("card_arrival", 0.9, logprob=-0.2))


def _route_b(tmp_path, *extra):
    return [
        "route",
        "--arm",
        "B",
        "--split",
        "val",
        "--fixture",
        "--cache",
        str(tmp_path / "c.jsonl"),
        *extra,
    ]


def _lines(path):
    return path.read_text().splitlines()


def test_route_arm_b_makes_one_call_per_ticket_for_both_variants(monkeypatch, tmp_path, capsys):
    made = _use_fake_provider(monkeypatch, _scripted_reply)
    save = tmp_path / "r.jsonl"
    assert cli.main(_route_b(tmp_path, "--save", str(save))) == 0
    out = capsys.readouterr().out
    for expected in ("synthetic", "B-logprob", "B-verbal", "raw", "isotonic", "2:1", "20:1"):
        assert expected in out
    assert "latency" in out
    assert "llm calls: 12 made" in out
    assert "saved 48 rows" in out
    assert made[0].calls == 12
    assert len(_lines(tmp_path / "c.jsonl")) == 12
    assert len(_lines(save)) == 48


def test_a_second_run_over_the_same_cache_makes_no_calls(monkeypatch, tmp_path, capsys):
    made = _use_fake_provider(monkeypatch, _scripted_reply)
    assert cli.main(_route_b(tmp_path)) == 0
    first = capsys.readouterr().out
    assert cli.main(_route_b(tmp_path)) == 0
    second = capsys.readouterr().out
    assert "llm calls: 0 made" in second
    assert made[1].calls == 0

    def table(out):
        return [line for line in out.splitlines() if line.startswith(("B-logprob", "B-verbal"))]

    assert len(table(first)) == len(table(second)) > 0
    assert table(first) == table(second)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--arm", "B", "--fixture"], "--cache"),
        (["--arm", "A", "--fixture", "--cache", "x.jsonl"], "arm B only"),
        (["--arm", "A", "--fixture", "--model", "m"], "arm B only"),
        (["--arm", "A", "--fixture", "--limit", "0"], "at least 1"),
        (["--arm", "A", "--fixture", "--split", "test", "--limit", "5"], "test split"),
    ],
)
def test_route_validation_fails_before_anything_runs(monkeypatch, capsys, argv, message):
    def boom():
        raise AssertionError("a provider must not be built")

    _use_fake_provider(monkeypatch, boom)
    assert cli.main(["route", *argv]) == 2
    captured = capsys.readouterr()
    assert message in captured.err
    assert "source:" not in captured.out  # no data was loaded


def test_limit_truncates_the_validation_rows(capsys):
    argv = ["route", "--arm", "A", "--split", "val", "--fixture", "--limit", "6"]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "LIMITED to the first 6" in out
    assert "evaluated 6 rows" in out


class _AnswersThenFails:
    identity = "three"

    def __init__(self):
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        if self.calls > 3:
            raise ProviderError("could not reach Ollama")
        return fake_completion("card_arrival")


def test_an_interrupted_run_keeps_the_responses_it_received(monkeypatch, tmp_path, capsys):
    _use_fake_provider(monkeypatch, _AnswersThenFails)
    assert cli.main(_route_b(tmp_path)) == 2
    assert "could not reach Ollama" in capsys.readouterr().err
    path = tmp_path / "c.jsonl"
    assert len(_lines(path)) == 3
    assert ResponseCache(path).skipped == 0


def test_route_arm_a_saves_per_ticket_rows(tmp_path, capsys):
    save = tmp_path / "r.jsonl"
    argv = ["route", "--arm", "A", "--split", "val", "--fixture", "--save", str(save)]
    assert cli.main(argv) == 0
    assert "saved 24 rows" in capsys.readouterr().out
    assert len(_lines(save)) == 24
