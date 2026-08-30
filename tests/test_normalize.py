"""Normalisation tests.

The first test is a regression test for a bug this project actually had: the punctuation
strip was written as ``re.sub(r"[^\\w\\s]", " ", text)``, and Python's ``\\w`` does not
include Unicode combining marks. Every Devanagari vowel sign was silently deleted --
"क़िला" came out as "क ल" -- and nothing errored, because the output was still text.
"""

from __future__ import annotations

import pytest

from classicalnlp import normalize as n

ZWNJ = "‌"
ZWSP = "​"
NUKTA = "़"


def test_combining_marks_survive_normalisation():
    """The bug: matras are category Mn, and Python's \\w excludes them."""
    assert n.normalize("क़िला") == "क़िला"
    assert n.tokenize("क़िला") == ["क़िला"]
    assert n.normalize("ಪರೀಕ್ಷೆ") == "ಪರೀಕ್ಷೆ"


def test_precomposed_and_decomposed_forms_unify():
    precomposed = "क़"  # क़
    decomposed = "क" + NUKTA  # क + nukta
    assert precomposed != decomposed
    assert n.normalize(precomposed) == n.normalize(decomposed)


def test_junk_invisibles_go_and_meaningful_ones_stay():
    assert n.normalize(f"a{ZWSP}b") == "ab"
    assert ZWNJ in n.normalize(f"क{ZWNJ}ष")


def test_indic_digits_fold_to_ascii():
    assert n.fold_digits("०१२") == "012"
    assert n.fold_digits("೧೨೩") == "123"
    assert n.fold_digits("௧௨") == "12"
    assert "012" in n.normalize("०१२ books")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello world", "latin"),
        ("नमस्ते दुनिया", "devanagari"),
        ("ಕನ್ನಡ ಪಠ್ಯ", "kannada"),
        ("தமிழ் உரை", "tamil"),
        ("hello नमस्ते", "mixed"),
        ("123 !!!", "unknown"),
    ],
)
def test_script_detection(text, expected):
    assert n.detect_script(text) == expected


def test_punctuation_goes_and_case_folds():
    assert n.normalize("The Quick, Brown Fox -- jumped!") == "the quick brown fox jumped"


def test_stopwords_are_language_specific():
    tokens = n.tokenize("यह एक परीक्षण है और यह अच्छा है", language="hi")
    assert tokens == ["परीक्षण", "अच्छा"]
    # Without the language, nothing is dropped.
    assert len(n.tokenize("यह एक परीक्षण है और यह अच्छा है")) == 8


def test_kannada_stopwords():
    tokens = n.tokenize("ಇದು ಒಂದು ಪರೀಕ್ಷೆ ಮತ್ತು ಇದು ಚೆನ್ನಾಗಿದೆ", language="kn")
    assert tokens == ["ಪರೀಕ್ಷೆ", "ಚೆನ್ನಾಗಿದೆ"]


def test_normalisation_is_idempotent():
    for text, language in [("क़िला ०१२ Hello!", "hi"), ("ಪರೀಕ್ಷೆ ೧೨೩", "kn")]:
        once = n.normalize(text)
        assert n.normalize(once) == once
        assert n.tokenize(once, language=language) == n.tokenize(text, language=language)


def test_analyzer_is_a_scikit_learn_compatible_callable():
    analyze = n.analyzer("hi")
    assert analyze("यह एक परीक्षण है") == ["परीक्षण"]
