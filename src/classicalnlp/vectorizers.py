"""The representations under comparison.

Four of them, all built as scikit-learn pipelines so that *fitting happens inside the
cross-validation fold*. That is not a stylistic choice: a vectorizer fitted on the whole
corpus has seen the test documents' vocabulary and IDF weights, which inflates the score
by a margin large enough to change which model you would ship (measured in
``evaluate.leaky_vs_honest``).

* **tfidf-word** -- the baseline everyone should beat before reaching for anything else.
* **tfidf-char** -- character n-grams (3-5, word-bounded). For morphologically rich
  languages, where a word appears in a dozen inflected forms, this is often the strongest
  cheap representation, and it needs no tokenizer that understands the language.
* **lsa** -- TF-IDF followed by truncated SVD: dense "embeddings" learned from this corpus
  alone, no pretraining. The honest middle ground between counting and downloading.
* **sentence-transformer** -- optional, and it stays optional. It is a 90 MB download and
  a torch dependency for a project whose point is what happens below the API layer.
"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from .normalize import analyzer

RANDOM_STATE = 20260830


def classifier() -> LogisticRegression:
    """One classifier for every representation, so the comparison is of the features."""
    return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)


def tfidf_word(language: str | None = None, min_df: int = 2) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(analyzer=analyzer(language), min_df=min_df, sublinear_tf=True),
            ),
            ("clf", classifier()),
        ]
    )


def tfidf_char(min_df: int = 2) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=min_df, sublinear_tf=True
                ),
            ),
            ("clf", classifier()),
        ]
    )


def lsa(language: str | None = None, dimensions: int = 300, min_df: int = 2) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(analyzer=analyzer(language), min_df=min_df, sublinear_tf=True),
            ),
            ("svd", TruncatedSVD(n_components=dimensions, random_state=RANDOM_STATE)),
            ("norm", Normalizer()),
            ("clf", classifier()),
        ]
    )


class SentenceTransformerVectorizer(BaseEstimator, TransformerMixin):
    """Lazy wrapper so importing this module never imports torch.

    Stateless by construction -- a pretrained encoder has nothing to learn from the
    training fold -- which is exactly why it is the one representation here that *cannot*
    leak, and worth saying out loud when comparing it to the fitted ones.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise ImportError(
                    "sentence-transformers is an optional extra. Install it with:\n"
                    "  uv sync --extra embeddings"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, X, y=None):  # scikit-learn's parameter names, kept as-is
        return self

    def transform(self, X):
        return self._load().encode(list(X), show_progress_bar=False, normalize_embeddings=True)


def sentence_transformer(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Pipeline:
    return Pipeline([("encode", SentenceTransformerVectorizer(model_name)), ("clf", classifier())])


#: name -> factory. The CLI and the tests iterate over this.
BUILDERS: dict[str, Callable[..., Pipeline]] = {
    "tfidf-word": tfidf_word,
    "tfidf-char": lambda language=None: tfidf_char(),
    "lsa": lsa,
    "sentence-transformer": lambda language=None: sentence_transformer(),
}

#: Everything that runs with no optional extras and no model download.
OFFLINE_MODELS = ("tfidf-word", "tfidf-char", "lsa")


def build(name: str, language: str | None = None) -> Pipeline:
    if name not in BUILDERS:
        raise ValueError(f"Unknown model {name!r}. Known: {sorted(BUILDERS)}")
    return BUILDERS[name](language=language)
