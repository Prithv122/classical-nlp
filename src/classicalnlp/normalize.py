"""Text normalisation, with the Indic-script cases that Latin-only pipelines get wrong.

Three things break when a pipeline written for English meets Devanagari or Kannada text:

1. **The same word has several byte sequences.** Devanagari nukta characters can be a
   single precomposed codepoint (क़ U+0958) or a base plus a combining nukta (क + U+093C).
   They render identically and compare unequal, so one spelling becomes two vocabulary
   entries. NFC composition fixes it, and NFKC would go too far -- it also rewrites things
   like ligatures and superscripts that carry meaning elsewhere.

2. **Digits are not just 0-9.** ० १ २ (Devanagari), ೦ ೧ ೨ (Kannada) and ௦ ௧ ௨ (Tamil) are
   digits, and a tokenizer that only knows ASCII treats them as words.

3. **Zero-width characters are not all noise.** ZWSP (U+200B) and the BOM are junk and get
   stripped. ZWNJ (U+200C) is *not*: in Devanagari it is what keeps a half-form from
   joining into a conjunct, so stripping it changes the word. This module removes the
   first kind and keeps the second, which is the opposite of what "strip all invisible
   characters" advice produces.

Lowercasing is applied unconditionally because the Indic scripts here are unicase -- it is
a no-op for them and necessary for Latin.
"""

from __future__ import annotations

import re
import unicodedata

#: Codepoint blocks used for script detection.
SCRIPT_RANGES = {
    "latin": ((0x0041, 0x024F),),
    "devanagari": ((0x0900, 0x097F),),
    "kannada": ((0x0C80, 0x0CFF),),
    "tamil": ((0x0B80, 0x0BFF),),
    "arabic": ((0x0600, 0x06FF),),
}

#: Removed outright: they carry no meaning and only split vocabulary.
JUNK_INVISIBLES = "​‎‏﻿­"  # ZWSP, LRM, RLM, BOM, soft hyphen

#: Kept: ZWNJ and ZWJ change how a Devanagari conjunct renders, so they are part of the word.
MEANINGFUL_INVISIBLES = "‌‍"  # ZWNJ, ZWJ

_WHITESPACE = re.compile(r"\s+")

#: Unicode general-category prefixes that count as part of a word: Letter, Number, Mark.
#:
#: **Mark is the one that matters.** Python's ``re`` module excludes combining marks from
#: ``\w`` -- so a "[^\w\s]" punctuation strip deletes every Devanagari vowel sign and
#: virama, turning "क़िला" into "क ल". It looks like it works because the ASCII path is
#: fine and the Devanagari path is still *text*, just silently mangled. Matching on
#: category rather than on ``\w`` avoids it without adding a regex-engine dependency.
_WORD_CATEGORIES = ("L", "N", "M")


def is_word_character(char: str) -> bool:
    if char in MEANINGFUL_INVISIBLES:
        return True
    return unicodedata.category(char)[0] in _WORD_CATEGORIES


def _english_stopwords() -> frozenset[str]:
    """scikit-learn ships one; there is no reason to retype it.

    The Indic lists below are hand-written and short on purpose -- a borrowed 500-word
    list would import someone else's judgement about a language this project handles at
    the surface.
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return frozenset(ENGLISH_STOP_WORDS)


STOPWORDS: dict[str, frozenset[str] | set[str]] = {
    "en": _english_stopwords(),
    "hi": {
        "और",
        "का",
        "की",
        "के",
        "है",
        "हैं",
        "में",
        "से",
        "को",
        "पर",
        "यह",
        "वह",
        "एक",
        "कि",
        "था",
        "थी",
        "थे",
        "हो",
        "ने",
        "भी",
        "नहीं",
        "तो",
        "ही",
    },
    "kn": {
        "ಮತ್ತು",
        "ಈ",
        "ಆ",
        "ಅವರು",
        "ಇದು",
        "ಅದು",
        "ಒಂದು",
        "ಇದೆ",
        "ಇವೆ",
        "ಆಗಿ",
        "ಎಂದು",
        "ಗೆ",
        "ಗಳ",
        "ನಲ್ಲಿ",
        "ನ",
        "ಅಥವಾ",
        "ಎಲ್ಲಾ",
    },
}


def detect_script(text: str) -> str:
    """Return the dominant script, or 'mixed' when no script holds a majority."""
    counts = dict.fromkeys(SCRIPT_RANGES, 0)
    total = 0
    for char in text:
        if not char.isalpha():
            continue
        point = ord(char)
        for script, ranges in SCRIPT_RANGES.items():
            if any(low <= point <= high for low, high in ranges):
                counts[script] += 1
                total += 1
                break
    if not total:
        return "unknown"
    script, count = max(counts.items(), key=lambda item: item[1])
    return script if count / total >= 0.6 else "mixed"


def fold_digits(text: str) -> str:
    """Map Indic digits onto ASCII. ``unicodedata.digit`` knows every script's mapping."""
    out = []
    for char in text:
        if char.isdigit() and not char.isascii():
            try:
                out.append(str(unicodedata.digit(char)))
                continue
            except ValueError:  # pragma: no cover - a digit-like char with no numeric value
                pass
        out.append(char)
    return "".join(out)


def normalize(
    text: str,
    *,
    compose: bool = True,
    fold_indic_digits: bool = True,
    strip_punctuation: bool = True,
    lowercase: bool = True,
) -> str:
    """Canonical form for one document."""
    if compose:
        # NFC, not NFKC: NFC merges the encodings of the same character, NFKC also
        # rewrites characters that merely look similar.
        text = unicodedata.normalize("NFC", text)

    text = text.translate({ord(char): None for char in JUNK_INVISIBLES})

    if fold_indic_digits:
        text = fold_digits(text)
    if strip_punctuation:
        text = "".join(char if is_word_character(char) else " " for char in text)
    if lowercase:
        text = text.lower()

    return _WHITESPACE.sub(" ", text).strip()


def tokenize(text: str, *, language: str | None = None, drop_stopwords: bool = True) -> list[str]:
    """Script-agnostic word tokenizer over normalised text."""
    tokens = normalize(text).split()
    if drop_stopwords:
        stops = STOPWORDS.get(language or "", set())
        tokens = [token for token in tokens if token not in stops]
    return tokens


def analyzer(language: str | None = None):
    """A callable for scikit-learn's ``analyzer=`` argument.

    Passing the analyzer keeps normalisation inside the vectorizer, which keeps it inside
    the cross-validation pipeline -- normalising up front, outside the folds, is one of the
    quieter ways to leak information between train and test.
    """

    def analyze(document: str) -> list[str]:
        return tokenize(document, language=language)

    return analyze
