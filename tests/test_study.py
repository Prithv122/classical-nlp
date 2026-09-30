"""Study runner tests. ScriptedRouter gives hand-computable numbers and shows the study
is arm-agnostic. Fixture runs assert invariants only, never coverage or cost values."""

from __future__ import annotations

import numpy as np
import pytest

from classicalnlp import cli, costs, study, tickets
from classicalnlp.costs import RoutingCost
from classicalnlp.data import Corpus
from classicalnlp.router import ABSTAIN, TfidfArm


class ScriptedRouter:
    name = "S"
    description = "scripted"

    def __init__(self, script):
        self.script = script

    def fit(self, corpus):
        return self

    def predict_top(self, texts):
        pairs = [self.script[t] for t in texts]
        return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs], dtype=float)


def _corpus(documents, labels):
    return Corpus(documents=documents, labels=labels, label_names=list(tickets.BANKING77_INTENTS))


def test_cross_fit_leave_one_out_by_hand():
    out = study.cross_fit_isotonic([0, 0, 0, 0, 0], [0, 0, 0, 1, 1], np.full(5, 0.5), n_splits=5)
    np.testing.assert_array_equal(out, [0.5, 0.5, 0.5, 0.75, 0.75])


def test_cross_fit_output_shape_range_and_determinism():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, 40)
    y_pred = rng.integers(0, 3, 40)
    conf = rng.uniform(0, 1, 40)
    first = study.cross_fit_isotonic(y_true, y_pred, conf, seed=7)
    second = study.cross_fit_isotonic(y_true, y_pred, conf, seed=7)
    assert len(first) == 40
    assert np.all((first >= 0.0) & (first <= 1.0))
    np.testing.assert_array_equal(first, second)
    with pytest.raises(ValueError, match="at least 5 rows"):
        study.cross_fit_isotonic([0, 1, 0], [0, 1, 1], [0.5, 0.6, 0.7])


@pytest.fixture(scope="module")
def fixture_runs():
    fit, val = tickets.validation_split(tickets.load_fixture())
    return study.run_arm(TfidfArm(), fit, val)


def test_run_arm_on_the_fixture(fixture_runs):
    raw, isotonic = fixture_runs
    assert [r.calibration for r in fixture_runs] == ["raw", "isotonic"]
    assert {r.split for r in fixture_runs} == {"val"}
    for run in fixture_runs:
        assert tuple(run.costs) == costs.SENSITIVITY_RATIOS
        for breakdown in run.costs.values():
            assert breakdown.n == 12
            assert breakdown.auto_routed + breakdown.handed_off == 12
    np.testing.assert_array_equal(raw.y_pred, isotonic.y_pred)


def test_breakdowns_agree_with_the_rule_and_coverage_falls_with_ratio(fixture_runs):
    for run in fixture_runs:
        for ratio, breakdown in run.costs.items():
            routed = costs.auto_route(run.confidence, RoutingCost.from_ratio(ratio))
            assert breakdown.auto_routed == routed.sum()
        coverage = [run.costs[r].coverage for r in costs.SENSITIVITY_RATIOS]
        assert coverage == sorted(coverage, reverse=True)


def _scripted_test_mode(test_label):
    val_docs = ["v0", "v1", "v2", "v3"]
    test_docs = ["t0", "t1", "t2"]
    script = {d: (0, 0.5) for d in val_docs + test_docs}
    script["v3"] = (1, 0.5)
    val = _corpus(val_docs, [0] * 4)
    test = _corpus(test_docs, [test_label] * 3)
    return study.run_arm(ScriptedRouter(script), val, val, test)


def test_test_mode_calibrates_on_val_only():
    raw, isotonic = _scripted_test_mode(0)
    assert raw.split == isotonic.split == "test"
    np.testing.assert_array_equal(raw.confidence, [0.5, 0.5, 0.5])
    np.testing.assert_array_equal(isotonic.confidence, [0.75, 0.75, 0.75])
    assert isotonic.costs[2].auto_routed == 3
    assert isotonic.costs[2].cost_per_1000 == 0.0
    assert isotonic.costs[5].auto_routed == 0
    assert isotonic.costs[5].cost_per_1000 == 1000.0
    _, relabelled = _scripted_test_mode(1)
    np.testing.assert_array_equal(relabelled.confidence, isotonic.confidence)


@pytest.mark.parametrize("use_test", [False, True])
def test_abstentions_keep_zero_confidence_and_are_never_routed(use_test):
    docs = [f"d{i}" for i in range(12)]
    abstained = {0, 3, 6}
    script = {d: (ABSTAIN, 0.0) if i in abstained else (0, 0.25) for i, d in enumerate(docs)}
    corpus = _corpus(docs, [0] * 12)
    runs = study.run_arm(ScriptedRouter(script), corpus, corpus, corpus if use_test else None)
    for run in runs:
        idx = sorted(abstained)
        assert run.y_pred[idx].tolist() == [ABSTAIN] * 3
        assert run.confidence[idx].tolist() == [0.0] * 3
        for ratio in run.costs:
            routed = costs.auto_route(run.confidence, RoutingCost.from_ratio(ratio))
            assert not routed[idx].any()


def test_route_command_on_the_fixture(capsys):
    assert cli.main(["route", "--arm", "A", "--split", "val", "--fixture"]) == 0
    out = capsys.readouterr().out
    for expected in ("synthetic", "raw", "isotonic", "2:1", "5:1", "10:1", "20:1"):
        assert expected in out


def test_route_command_has_no_fixture_test_split(capsys):
    assert cli.main(["route", "--arm", "A", "--split", "test", "--fixture"]) == 2
    assert "test split" in capsys.readouterr().err


def test_route_command_with_missing_data_directory(tmp_path, capsys):
    assert cli.main(["route", "--arm", "A", "--data-dir", str(tmp_path)]) == 2
    assert capsys.readouterr().err
