"""Arm B, its prompt, reply parsing, response cache and the FakeLLM double. No network: every
reply here is scripted, and none of these numbers describe a real model."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from classicalnlp import costs, study, tickets
from classicalnlp.costs import RoutingCost
from classicalnlp.data import Corpus
from classicalnlp.router import ABSTAIN, Router, check_output
from classicalnlp.router.llm_arm import (
    PROMPT_TEMPLATE,
    Completion,
    FakeLLM,
    LLMArm,
    LLMClient,
    ProviderError,
    ResponseCache,
    build_prompt,
    cache_key,
    fake_completion,
    intent_logprob_confidence,
    llm_arms,
    parse_reply,
    split_prompt,
)

NAMES = list(tickets.BANKING77_INTENTS)
CARD = NAMES.index("card_arrival")
HALF = math.log(0.5)


def _corpus(documents, labels):
    return Corpus(documents=documents, labels=labels, label_names=list(NAMES))


def _span(text, name):
    start = text.index(name)
    return (start, start + len(name))


# 1 -- the prompt ------------------------------------------------------------------------


def test_prompt_layout():
    assert PROMPT_TEMPLATE.count("\nMessage: ") == 1
    prompt = build_prompt(NAMES, "where is my card")
    prefix, query = split_prompt(prompt)
    assert prompt.endswith("where is my card")
    assert query == "where is my card"
    block = prefix.split("Intents:\n")[1].split("\n\n")[0]
    assert block.split("\n") == NAMES  # every name, in order, one per line, before the query


def test_prompts_share_their_prefix_and_braces_in_a_query_are_safe():
    first = build_prompt(NAMES, "where is my card")
    second = build_prompt(NAMES, "what is {this}?")
    assert split_prompt(first)[0] == split_prompt(second)[0]
    assert split_prompt(second) == (split_prompt(first)[0], "what is {this}?")


# 2 -- parse_reply -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "name", "verbal"),
    [
        ("intent: card_arrival\nconfidence: 0.8", "card_arrival", 0.8),
        ("\n  intent: card_arrival\nconfidence: 0.3", "card_arrival", 0.3),
        ("intent: card_arrival", "card_arrival", None),
        ("intent: card_arrival\nconfidence: 1.5", "card_arrival", None),
        ("intent: card_arrival\nconfidence: 90%", "card_arrival", None),
        ("intent: card_arrival\nconfidence: nan", "card_arrival", None),
        ("intent: Refund_not_showing_up\nconfidence: 1", "Refund_not_showing_up", 1.0),
        ("intent: reverted_card_payment?\nconfidence: .5", "reverted_card_payment?", 0.5),
    ],
)
def test_parse_reply_usable_intent(text, name, verbal):
    label, value, span = parse_reply(text, NAMES)
    assert label == NAMES.index(name)
    assert value == verbal
    assert span == _span(text, name)


@pytest.mark.parametrize(
    "text",
    [
        "intent: refund_not_showing_up\nconfidence: 0.9",  # wrong case: no repair
        "Intent: card_arrival\nconfidence: 0.9",
        "intent: not_an_intent\nconfidence: 0.9",
        "intent: card arrival\nconfidence: 0.9",
        "",
    ],
)
def test_parse_reply_unusable_intent(text):
    assert parse_reply(text, NAMES) == (None, None, None)


# 3, 4 -- the intent log-probability confidence ------------------------------------------


def test_intent_confidence_counts_only_the_intent_tokens():
    c = fake_completion("card_arrival", 0.9, logprob=HALF)
    span = parse_reply(c.text, NAMES)[2]
    assert intent_logprob_confidence(c.text, c.logprobs, span) == pytest.approx(0.25)


def test_intent_confidence_for_a_mixed_case_name_with_more_tokens():
    c = fake_completion("Refund_not_showing_up", 0.9, logprob=-0.1)
    intent_tokens = sum(1 for _, lp in c.logprobs if lp == -0.1)
    assert intent_tokens > 2
    span = parse_reply(c.text, NAMES)[2]
    assert intent_logprob_confidence(c.text, c.logprobs, span) == pytest.approx(
        math.exp(-0.1 * intent_tokens)
    )


def test_intent_confidence_is_none_when_the_logprobs_cannot_be_trusted():
    c = fake_completion("card_arrival", 0.9)
    span = parse_reply(c.text, NAMES)[2]
    assert intent_logprob_confidence(c.text, None, span) is None
    assert intent_logprob_confidence(c.text + "x", c.logprobs, span) is None
    with_nan = (
        ("intent", 0.0),
        *c.logprobs[1:2],
        (c.logprobs[2][0], float("nan")),
        *c.logprobs[3:],
    )
    assert intent_logprob_confidence(c.text, with_nan, span) is None


def test_a_token_straddling_the_span_is_included_whole():
    text = "intent: card_arrival\nconfidence: 0.9"
    tokens = (
        ("intent", 0.0),
        (":", 0.0),
        (" card", HALF),
        ("_arrival\n", HALF),
        ("confidence", math.log(0.25)),
        (":", 0.0),
        (" 0.9", 0.0),
    )
    # "_arrival\n" counts, "confidence" does not: 0.5 * 0.5, not 0.5 * 0.5 * 0.25.
    assert intent_logprob_confidence(text, tokens, _span(text, "card_arrival")) == pytest.approx(
        0.25
    )


def test_a_slightly_positive_logprob_is_clipped_to_one():
    c = fake_completion("card_arrival", 0.9, logprob=1e-9)
    span = parse_reply(c.text, NAMES)[2]
    assert intent_logprob_confidence(c.text, c.logprobs, span) == 1.0


# 5, 6 -- the two variants and usage -----------------------------------------------------

SCRIPT = {
    "t0": fake_completion("card_arrival", 0.9, logprob=HALF, seconds=1.5, prompt_tokens=610),
    "t1": fake_completion("nope", 0.9, seconds=2.5, prompt_tokens=620, output_tokens=15),
    "t2": fake_completion("card_arrival", "90%", logprob=HALF, seconds=3.5, prompt_tokens=630),
    "t3": Completion(
        text="intent: card_arrival\nconfidence: 0.7",
        logprobs=None,
        prompt_tokens=640,
        output_tokens=16,
        seconds=4.5,
        cached_prompt_tokens=600,
    ),
}
TEXTS = ["t0", "t1", "t2", "t3"]


def _arm(variant, replies=SCRIPT):
    return LLMArm(LLMClient(FakeLLM(replies)), variant).fit(_corpus(TEXTS, [CARD] * 4))


def test_variants_abstain_independently():
    labels, confidence = _arm("logprob").predict_top(TEXTS)
    assert labels.tolist() == [CARD, ABSTAIN, CARD, ABSTAIN]
    np.testing.assert_allclose(confidence, [0.25, 0.0, 0.25, 0.0])
    labels, confidence = _arm("verbal").predict_top(TEXTS)
    assert labels.tolist() == [CARD, ABSTAIN, ABSTAIN, CARD]
    np.testing.assert_allclose(confidence, [0.9, 0.0, 0.0, 0.7])


@pytest.mark.parametrize("variant", ["logprob", "verbal"])
def test_arms_follow_the_contract(variant):
    arm = _arm(variant)
    assert isinstance(arm, Router)
    assert arm.name == f"B-{variant}"
    labels, confidence = arm.predict_top(TEXTS)
    check_output(labels, confidence, 4)
    assert np.all(confidence[labels == ABSTAIN] == 0.0)


def test_usage_follows_the_completions_in_order():
    arm = _arm("logprob")
    arm.predict_top(TEXTS)
    np.testing.assert_array_equal(arm.usage.seconds, [1.5, 2.5, 3.5, 4.5])
    np.testing.assert_array_equal(arm.usage.prompt_tokens, [610, 620, 630, 640])
    np.testing.assert_array_equal(arm.usage.cached_prompt_tokens, [0, 0, 0, 600])
    np.testing.assert_array_equal(arm.usage.output_tokens, [14, 15, 14, 16])


def test_no_texts_gives_empty_arrays_and_empty_usage():
    arm = _arm("verbal")
    labels, confidence = arm.predict_top([])
    assert len(labels) == len(confidence) == 0
    for field in ("seconds", "prompt_tokens", "cached_prompt_tokens", "output_tokens"):
        assert len(getattr(arm.usage, field)) == 0


def test_unfitted_arm_and_unknown_variant_raise():
    with pytest.raises(ValueError, match="not fitted"):
        LLMArm(LLMClient(FakeLLM(SCRIPT))).predict_top(TEXTS)
    with pytest.raises(ValueError, match="variant"):
        LLMArm(LLMClient(FakeLLM(SCRIPT)), "bogus")


# 7, 8 -- one response feeds both variants; a cache hit makes no call --------------------


def test_one_response_feeds_both_variants():
    fake = FakeLLM(SCRIPT)
    client = LLMClient(fake)
    for arm in llm_arms(client):
        arm.fit(_corpus(TEXTS, [CARD] * 4)).predict_top(TEXTS)
    assert fake.calls == 4
    assert client.calls == 4
    assert client.hits == 4


def test_a_second_pass_and_a_reopened_cache_make_no_calls(tmp_path):
    path = tmp_path / "cache.jsonl"
    fake = FakeLLM(SCRIPT)
    client = LLMClient(fake, ResponseCache(path))
    arm = LLMArm(client).fit(_corpus(TEXTS, [CARD] * 4))
    first = arm.predict_top(TEXTS)
    first_usage = arm.usage
    arm.predict_top(TEXTS)
    assert fake.calls == 4

    reopened = FakeLLM(SCRIPT)
    again = LLMArm(LLMClient(reopened, ResponseCache(path))).fit(_corpus(TEXTS, [CARD] * 4))
    second = again.predict_top(TEXTS)
    assert reopened.calls == 0
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first_usage.seconds, again.usage.seconds)
    np.testing.assert_array_equal(first_usage.prompt_tokens, again.usage.prompt_tokens)
    np.testing.assert_array_equal(
        first_usage.cached_prompt_tokens, again.usage.cached_prompt_tokens
    )
    np.testing.assert_array_equal(first_usage.output_tokens, again.usage.output_tokens)


# 9 -- cache keys ------------------------------------------------------------------------


def test_cache_key():
    assert cache_key("m", "p") == cache_key("m", "p")
    assert cache_key("m", "p") != cache_key("n", "p")
    assert cache_key("m", "p") != cache_key("m", "q")
    assert len(cache_key("m", "p")) == 64


class _Named:
    def __init__(self, identity):
        self.identity = identity
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        return fake_completion("card_arrival")


def test_providers_with_different_identities_do_not_share_entries(tmp_path):
    cache = ResponseCache(tmp_path / "c.jsonl")
    one, two = _Named("one"), _Named("two")
    LLMClient(one, cache).complete("prompt")
    LLMClient(two, cache).complete("prompt")
    assert (one.calls, two.calls) == (1, 1)
    LLMClient(one, ResponseCache(tmp_path / "c.jsonl")).complete("prompt")
    assert one.calls == 1


# 10, 11 -- persistence ------------------------------------------------------------------


class _DiesOnThird:
    identity = "dies"

    def __init__(self):
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        if self.calls == 3:
            raise ProviderError("could not reach Ollama")
        return fake_completion("card_arrival")


def test_each_response_is_written_before_the_next_call(tmp_path):
    path = tmp_path / "sub" / "cache.jsonl"
    cache = ResponseCache(path)
    assert not path.exists()  # opening creates nothing
    provider = _DiesOnThird()
    arm = LLMArm(LLMClient(provider, cache)).fit(_corpus(list("abcde"), [CARD] * 5))
    with pytest.raises(ProviderError):
        arm.predict_top(list("abcde"))
    assert len(path.read_text().splitlines()) == 2

    healthy = _Named("dies")
    resumed = LLMArm(LLMClient(healthy, ResponseCache(path))).fit(
        _corpus(list("abcde"), [CARD] * 5)
    )
    resumed.predict_top(list("abcde"))
    assert healthy.calls == 3


def test_an_in_memory_cache_never_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache = ResponseCache(None)
    cache.put("k", fake_completion("card_arrival"))
    assert cache.get("k") is not None
    assert list(tmp_path.iterdir()) == []


def test_a_damaged_cache_loads_its_good_lines(tmp_path):
    path = tmp_path / "cache.jsonl"
    cache = ResponseCache(path)
    kept = fake_completion("card_arrival", None)  # no second line, tokens still concatenate
    cache.put("a", kept)
    cache.put("b", Completion("intent: x", None, 5, 2, 0.5, 3))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"key": "no-text-field"}) + "\n")
        handle.write('{"key": "cut", "text": "inte')  # a run killed mid-write

    loaded = ResponseCache(path)
    assert loaded.skipped == 2
    assert loaded.get("a") == kept
    assert loaded.get("b") == Completion("intent: x", None, 5, 2, 0.5, 3)  # logprobs None kept
    assert loaded.get("cut") is None

    loaded.put("c", kept)  # starts on its own line, not glued to the partial one
    reloaded = ResponseCache(path)
    assert reloaded.skipped == 2
    assert reloaded.get("c") == kept


# 12 -- abstention end to end ------------------------------------------------------------


def test_abstained_rows_are_never_auto_routed_in_a_study_run():
    docs = [f"t{i}" for i in range(12)]
    replies = {
        d: fake_completion("card_arrival", 0.9, logprob=-0.05 * (i % 6)) for i, d in enumerate(docs)
    }
    bad = [0, 4, 9]
    for i in bad:
        replies[docs[i]] = fake_completion("nope", 0.9)
    corpus = _corpus(docs, [CARD] * 12)
    for variant in ("logprob", "verbal"):
        arm = LLMArm(LLMClient(FakeLLM(replies)), variant)
        for run in study.run_arm(arm, corpus, corpus):
            assert run.y_pred[bad].tolist() == [ABSTAIN] * 3
            assert run.confidence[bad].tolist() == [0.0] * 3
            assert len(run.usage.seconds) == 12
            for ratio in costs.SENSITIVITY_RATIOS:
                routed = costs.auto_route(run.confidence, RoutingCost.from_ratio(ratio))
                assert not routed[bad].any()


# the double itself ----------------------------------------------------------------------


def test_fake_llm_reports_a_missing_reply_by_query():
    fake = FakeLLM({"known": fake_completion("card_arrival")})
    with pytest.raises(KeyError, match="missing ticket"):
        fake.complete(build_prompt(NAMES, "missing ticket"))
    assert fake.calls == 1


@pytest.mark.parametrize("verbal", [0.9, "90%", None])
def test_fake_completion_tokens_concatenate_to_the_text(verbal):
    c = fake_completion("Refund_not_showing_up", verbal)
    assert "".join(token for token, _ in c.logprobs) == c.text
    assert c.text.startswith("intent: Refund_not_showing_up")
    assert ("confidence" in c.text) == (verbal is not None)
