from __future__ import annotations

import pytest

from classicalnlp import topics, vectorizers

CORPUS = [
    "rocket launch orbit payload telemetry apogee mission control",
    "orbit insertion burn telemetry rocket stage separation",
    "launch window payload fairing rocket apogee orbit",
    "telemetry downlink orbit rocket launch mission",
    "goalie puck penalty rink faceoff shootout hockey",
    "hockey rink puck goalie save penalty box",
    "faceoff circle puck goalie shootout hockey win",
    "penalty kill hockey puck rink goalie period",
] * 4


def test_topics_separate_the_two_subjects():
    model = topics.fit_topics(CORPUS, n_topics=2)
    assert len(model) == 2
    vocabularies = [set(topic.terms) for topic in model.topics]
    space = {"rocket", "orbit", "launch", "telemetry"}
    hockey = {"puck", "goalie", "rink", "hockey"}
    assert any(space & vocabulary for vocabulary in vocabularies)
    assert any(hockey & vocabulary for vocabulary in vocabularies)
    # c-TF-IDF should not label both topics with the same words.
    assert vocabularies[0] != vocabularies[1]


def test_assignments_cover_every_document():
    model = topics.fit_topics(CORPUS, n_topics=2)
    assert len(model.assignments) == len(CORPUS)
    assert sum(topic.size for topic in model.topics) == len(CORPUS)


def test_coherence_is_in_range_and_rewards_the_coherent_corpus():
    model = topics.fit_topics(CORPUS, n_topics=2)
    assert -1.0 <= model.coherence <= 1.0
    assert model.coherence > 0  # words within a topic genuinely co-occur here


def test_incoherent_topics_score_lower():
    """A topic of words that never co-occur should score below a real one."""
    real = topics.fit_topics(CORPUS, n_topics=2)
    nonsense = topics.npmi_coherence(
        CORPUS, [["rocket", "puck", "goalie", "telemetry"], ["orbit", "rink", "faceoff", "launch"]]
    )
    assert nonsense < real.coherence


def test_c_tf_idf_labels_are_topic_specific():
    model = topics.fit_topics(CORPUS, n_topics=2)
    terms, _ = topics.c_tf_idf(CORPUS, model.assignments)
    assert len(terms) == 2
    assert set(terms[0]) != set(terms[1])


def test_topic_label_is_readable():
    model = topics.fit_topics(CORPUS, n_topics=2)
    label = model.topics[0].label(3)
    assert label.count(",") == 2


def test_unknown_model_name_is_refused():
    with pytest.raises(ValueError, match="Unknown model"):
        vectorizers.build("word2vec-but-invented")


def test_optional_backend_fails_with_an_instruction_not_a_traceback():
    """sentence-transformers is an extra; asking for it uninstalled must say how to fix it."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="uv sync --extra embeddings"):
            vectorizers.SentenceTransformerVectorizer().transform(["hello"])
    else:  # pragma: no cover - only when the optional extra is installed
        pytest.skip("sentence-transformers is installed")
