"""Topic modelling, and a number to judge it by.

BERTopic is the tool this catalogue entry names, and it is three steps in a trench coat:
embed, cluster, then describe each cluster with a class-based TF-IDF. The interesting step
-- the one that decides whether a topic is readable -- is the third, so it is implemented
here rather than imported: NMF gives the soft clustering over TF-IDF, and ``c-TF-IDF``
re-weights terms *per topic against the other topics*, which is what stops every topic
being labelled with the corpus's most common words.

Stopwords matter here in a way they do not for classification. TF-IDF *weights* a common
word down, so a classifier copes with "the" being in every document; c-TF-IDF counts raw
frequencies within a topic, so an unfiltered run labels all eight topics "the, of, to".
That is why ``fit_topics`` is called with a language and its stoplist, and why the
vectorizers here also cap ``max_df``.

Topics are then scored with **NPMI coherence**, computed from document co-occurrence in
this corpus. Eyeballing top-10 word lists is how topic models get shipped broken; a number
lets you compare 8 topics against 20 without deciding by vibe.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .normalize import analyzer

RANDOM_STATE = 20260830


@dataclass(frozen=True)
class Topic:
    index: int
    terms: tuple[str, ...]
    weight: float
    size: int

    def label(self, n: int = 4) -> str:
        return ", ".join(self.terms[:n])


@dataclass(frozen=True)
class TopicModel:
    topics: tuple[Topic, ...]
    assignments: np.ndarray
    coherence: float

    def __len__(self) -> int:
        return len(self.topics)


def c_tf_idf(
    documents: Sequence[str], assignments: np.ndarray, language: str | None = None, top_k: int = 10
) -> tuple[list[tuple[str, ...]], CountVectorizer]:
    """Class-based TF-IDF: treat all documents in a topic as one document.

    A term scores highly for a topic when it is frequent *in that topic* and rare across
    the others -- which is why the labels come out specific instead of every topic being
    described as "people, time, good".
    """
    vectorizer = CountVectorizer(analyzer=analyzer(language), min_df=2, max_df=0.5)
    counts = vectorizer.fit_transform(list(documents))
    vocabulary = np.asarray(vectorizer.get_feature_names_out())

    labels = np.unique(assignments)
    per_topic = np.vstack(
        [np.asarray(counts[assignments == label].sum(axis=0)).ravel() for label in labels]
    )

    topic_totals = per_topic.sum(axis=1, keepdims=True)
    topic_totals[topic_totals == 0] = 1
    term_frequency = per_topic / topic_totals

    # log(1 + A / f), where A is the average number of words per topic and f the term's
    # frequency across every topic. Using the corpus total instead of the average makes
    # the weight far too flat, and every topic comes back labelled "the, of, to".
    average_words = float(per_topic.sum(axis=1).mean())
    term_totals = per_topic.sum(axis=0).astype(float)
    term_totals[term_totals == 0] = 1.0
    inverse = np.log(1 + average_words / term_totals)

    scores = term_frequency * inverse
    terms = [tuple(vocabulary[np.argsort(row)[::-1][:top_k]]) for row in scores]
    return terms, vectorizer


def npmi_coherence(
    documents: Sequence[str],
    topic_terms: Sequence[Sequence[str]],
    language: str | None = None,
    top_n: int = 10,
) -> float:
    """Mean pairwise NPMI over each topic's top terms, averaged across topics.

    NPMI is in [-1, 1]: 1 means two words always co-occur, 0 means independence, negative
    means they avoid each other. Anything around 0.05-0.15 is typical for short-document
    corpora, so the number is only useful compared against another run on the same corpus.
    """
    vectorizer = CountVectorizer(analyzer=analyzer(language), binary=True)
    matrix = vectorizer.fit_transform(list(documents))
    index = {term: i for i, term in enumerate(vectorizer.get_feature_names_out())}
    total_documents = matrix.shape[0]

    scores = []
    for terms in topic_terms:
        present = [term for term in terms[:top_n] if term in index]
        pair_scores = []
        for i, first in enumerate(present):
            for second in present[i + 1 :]:
                a = matrix[:, index[first]]
                b = matrix[:, index[second]]
                count_a = a.sum()
                count_b = b.sum()
                joint = int(a.multiply(b).sum())
                if joint == 0 or count_a == 0 or count_b == 0:
                    pair_scores.append(-1.0)
                    continue
                p_joint = joint / total_documents
                p_a = count_a / total_documents
                p_b = count_b / total_documents
                pair_scores.append(float(np.log(p_joint / (p_a * p_b)) / -np.log(p_joint)))
        if pair_scores:
            scores.append(float(np.mean(pair_scores)))
    return float(np.mean(scores)) if scores else 0.0


def fit_topics(
    documents: Sequence[str],
    n_topics: int = 8,
    language: str | None = None,
    min_df: int = 2,
    top_k: int = 10,
) -> TopicModel:
    """NMF over TF-IDF, labelled with c-TF-IDF and scored with NPMI coherence."""
    tfidf = TfidfVectorizer(
        analyzer=analyzer(language), min_df=min_df, max_df=0.5, sublinear_tf=True
    )
    matrix = tfidf.fit_transform(list(documents))

    model = NMF(n_components=n_topics, random_state=RANDOM_STATE, init="nndsvda", max_iter=400)
    weights = model.fit_transform(matrix)
    assignments = weights.argmax(axis=1)

    terms, _ = c_tf_idf(documents, assignments, language=language, top_k=top_k)
    sizes = np.bincount(assignments, minlength=n_topics)
    labels = np.unique(assignments)

    topics = tuple(
        Topic(
            index=int(label),
            terms=terms[position],
            weight=float(weights[assignments == label, label].mean()),
            size=int(sizes[label]),
        )
        for position, label in enumerate(labels)
    )

    return TopicModel(
        topics=topics,
        assignments=assignments,
        coherence=npmi_coherence(documents, [topic.terms for topic in topics], language=language),
    )
