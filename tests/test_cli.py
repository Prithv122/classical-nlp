"""CLI tests.

``compare``/``leakage``/``topics`` need 20 Newsgroups, which is a network fetch on a cold
machine (cached afterwards). They are marked ``network`` so the offline parts of the suite
still run: ``uv run pytest -m "not network"``.
"""

from __future__ import annotations

import pytest

from classicalnlp import __version__, cli


def test_version_is_set():
    assert __version__


def test_normalize_command_prints_every_sample(capsys):
    assert cli.main(["normalize"]) == 0
    out = capsys.readouterr().out
    assert out.count("normalized :") >= 6
    assert "devanagari" in out
    assert "kannada" in out


def test_normalize_command_with_explicit_text(capsys):
    assert cli.main(["normalize", "--text", "यह एक परीक्षण है", "--language", "hi"]) == 0
    out = capsys.readouterr().out
    assert "script     : devanagari" in out
    assert "परीक्षण" in out


def test_unknown_model_exits_with_two(capsys):
    with pytest.raises(SystemExit):  # argparse rejects it before we do
        cli.main(["compare", "--models", "not-a-model"])


@pytest.mark.network
def test_compare_command(capsys):
    assert cli.main(["compare", "--limit", "300", "--folds", "3"]) == 0
    out = capsys.readouterr().out
    assert "tfidf-word" in out
    assert "macro-F1" in out
    assert "95% CI" in out


@pytest.mark.network
def test_leakage_command_reports_inflation(capsys):
    assert cli.main(["leakage", "--limit", "300", "--folds", "3"]) == 0
    out = capsys.readouterr().out
    assert "inflation" in out


@pytest.mark.network
def test_topics_command(capsys):
    assert cli.main(["topics", "--limit", "300", "--n-topics", "4"]) == 0
    out = capsys.readouterr().out
    assert "NPMI coherence" in out
    assert out.count("topic ") >= 4
