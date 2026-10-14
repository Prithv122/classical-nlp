"""Arm C against fake providers and a fake connection only. Every number here is scripted to
exercise the code and none is model behaviour. No network."""

from __future__ import annotations

import hashlib
import json
import sys
import types

import numpy as np
import pytest

from classicalnlp import study, tickets
from classicalnlp.router import ARMS, DecisionArm, FakeDecisionProvider, Router
from classicalnlp.router import decision_arm as da
from classicalnlp.router import typesafe as ts
from classicalnlp.router.decision_arm import (
    AttemptFailed,
    AuthError,
    BudgetExceeded,
    DecisionCache,
    DecisionClient,
    Ledger,
    RawResponse,
    UsageError,
    VersionMismatch,
    build_request,
    cache_key,
    fake_decision_response,
    parse_answer,
)
from classicalnlp.router.llm_arm import PROMPT_TEMPLATE

NAMES = tickets.BANKING77_INTENTS
GOOD = fake_decision_response("card_arrival")
V2_KEYS = {
    "arm", "split", "calibration", "index", "y_true", "y_pred", "confidence",
    "seconds", "prompt_tokens", "cached_prompt_tokens", "output_tokens",
}  # fmt: skip
KEY = "test-key-not-real"


def _client(tmp_path, replies, *, price=da.DEFAULT_PRICE_PER_MTOK, run_id="r1"):
    provider = FakeDecisionProvider(replies)
    sleeps: list[float] = []
    client = DecisionClient(
        provider,
        DecisionCache(tmp_path / "c.jsonl"),
        Ledger(tmp_path / "l.jsonl", run_id),
        price_per_mtok=price,
        sleep=sleeps.append,
    )
    return client, provider, sleeps


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _request(text="where is my card"):
    return build_request(text)


# -- the request ----------------------------------------------------------------------------


def test_request_body_is_exactly_the_documented_shape():
    text = "  odd   spacing, and unicode: café कार्ड\n"
    request = build_request(text)
    assert list(request) == ["state", "model", "questions"]
    assert request["state"] == text
    assert request["model"] == "jev-1.13.0"
    question = request["questions"]["intent"]
    assert question["type"] == "choice"
    assert list(question["criteria"]) == list(NAMES) and len(NAMES) == 77
    assert set(question["criteria"].values()) == {None}
    assert json.loads(json.dumps(request)) == request


def test_instructions_are_the_first_two_sentences_of_the_b_prompt():
    assert PROMPT_TEMPLATE.startswith(da.INSTRUCTIONS + "\n")
    assert _request()["questions"]["intent"]["instructions"] == da.INSTRUCTIONS


# -- cache ----------------------------------------------------------------------------------


def test_cache_key_is_the_documented_hash():
    request = _request("hi")
    canonical = json.dumps(request, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(f"{da.ENDPOINT}\njev-1.13.0\n{canonical}".encode()).hexdigest()
    assert cache_key(da.ENDPOINT, request) == expected
    assert cache_key("other", request) != expected
    assert cache_key(da.ENDPOINT, _request("hi!")) != expected
    assert da.ENDPOINT == "typesafe:https://api.typesafe.ai/v1/systemone"


def test_a_cached_ticket_costs_nothing_and_keeps_its_original_seconds(tmp_path):
    slow = RawResponse(GOOD, seconds=0.37, cold_start=True)
    client, provider, _ = _client(tmp_path, [slow])
    first = client.decide(_request())
    assert (provider.calls, client.calls, client.hits) == (1, 1, 0)
    record = _lines(tmp_path / "c.jsonl")[0]
    assert set(record) == {
        "key", "endpoint", "requested_model", "response", "model_version",
        "input_tokens", "output_tokens", "seconds", "cold_start", "attempts",
    }  # fmt: skip
    assert record["key"] == cache_key(da.ENDPOINT, _request())
    assert len(_lines(tmp_path / "l.jsonl")) == 1

    again, provider2, _ = _client(tmp_path, [GOOD], run_id="r2")
    second = again.decide(_request())
    assert provider2.calls == 0 and again.hits == 1
    assert len(_lines(tmp_path / "l.jsonl")) == 1  # a hit writes no ledger line
    assert (second.seconds, second.cold_start, second.body) == (0.37, True, first.body)


def test_cache_tolerates_a_partial_last_line_and_never_creates_a_file_on_open(tmp_path):
    path = tmp_path / "sub" / "c.jsonl"
    assert DecisionCache(path).skipped == 0 and not path.parent.exists()
    client, _, _ = _client(tmp_path, [GOOD])
    client.decide(_request("a"))
    with (tmp_path / "c.jsonl").open("a") as handle:
        handle.write('{"key": "abc')  # a killed run
    cache = DecisionCache(tmp_path / "c.jsonl")
    assert cache.skipped == 1 and len(cache._items) == 1
    cache.put("k2", da.ENDPOINT, client.decide(_request("a")))
    assert len(DecisionCache(tmp_path / "c.jsonl")._items) == 2


# -- abstention and argmax ------------------------------------------------------------------


def _top(body, variant):
    client = types.SimpleNamespace(
        decide=lambda request: da.DecisionResponse(body, "jev-1.13.0", 1, 1, 0.1, False, 1),
        provider=types.SimpleNamespace(identity=da.ENDPOINT),
    )
    arm = DecisionArm(client, variant)
    arm.fit(types.SimpleNamespace(label_names=list(NAMES)))
    labels, confidence = arm.predict_top(["t"])
    return int(labels[0]), float(confidence[0])


def _tweak(**changes):
    body = fake_decision_response("card_arrival", 0.9, 0.8)
    answer = body["answers"]["intent"]
    for key, value in changes.items():
        if value is None:
            answer.pop(key, None)
        else:
            answer[key] = value
    return body


def test_the_two_variants_read_different_confidences_from_one_response():
    card = NAMES.index("card_arrival")
    assert _top(_tweak(), "prob") == (card, pytest.approx(0.9))
    assert _top(_tweak(), "vendor") == (card, 0.8)


@pytest.mark.parametrize("variant", ["prob", "vendor"])
@pytest.mark.parametrize(
    "bad",
    [
        {"choice": "not_an_intent"},
        {"choice": None},
        {"probabilities": {n: 1 / 76 for n in NAMES[1:]}},  # one name missing
        {"probabilities": {**dict.fromkeys(NAMES, 1 / 77), "age_limit": "x"}},
        {"probabilities": dict.fromkeys(NAMES, 0.0127)},  # sums to 0.9779
        {"probabilities": None},
    ],
)
def test_unusable_choice_or_probabilities_abstain_both_variants(variant, bad):
    assert _top(_tweak(**bad), variant) == (-1, 0.0)


def test_probabilities_are_not_renormalised_but_the_tolerance_is_1e_3():
    even = dict.fromkeys(NAMES, 1 / 77)
    inside = {**even, "age_limit": even["age_limit"] + 9e-4}
    outside = {**even, "age_limit": even["age_limit"] + 2e-3}
    assert _top(_tweak(probabilities=inside), "prob")[0] != -1
    assert _top(_tweak(probabilities=outside), "prob")[0] == -1
    assert _top(_tweak(probabilities=inside), "prob")[1] == pytest.approx(1 / 77)


@pytest.mark.parametrize("confidence", [None, 1.5, -0.1, "0.8", float("nan"), True])
def test_a_bad_confidence_field_abstains_only_the_vendor_variant(confidence):
    assert _top(_tweak(confidence=confidence), "vendor") == (-1, 0.0)
    assert _top(_tweak(confidence=confidence), "prob")[0] == NAMES.index("card_arrival")


def test_confidence_at_the_edges_is_accepted():
    for edge in (0.0, 1.0):
        assert _top(_tweak(confidence=edge), "vendor")[1] == edge


def _status(choice, probabilities):
    return parse_answer(fake_decision_response(choice, probabilities=probabilities), NAMES)[3]


def test_argmax_status_unique_tied_differs_and_unusable():
    rest = (1 - 0.9) / 76
    unique = {n: (0.9 if n == "age_limit" else rest) for n in NAMES}
    assert _status("age_limit", unique) == "unique"
    tied = {n: 0.45 if n in ("age_limit", "atm_support") else 0.1 / 75 for n in NAMES}
    assert _status("age_limit", tied) == "tied"
    assert _status("atm_support", tied) == "tied"
    assert _status("exchange_rate", unique) == "differs"
    assert _status("not_an_intent", unique) is None
    assert _status("age_limit", {**unique, "age_limit": 5.0}) is None


# -- retries, transport failures, stops -----------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 529])
def test_retryable_statuses_are_retried_after_one_second(tmp_path, status):
    client, provider, sleeps = _client(tmp_path, [AttemptFailed(f"http_{status}", status), GOOD])
    result = client.decide(_request())
    assert not result.transport_failure and result.attempts == 2
    assert (provider.calls, sleeps) == (2, [1])
    ledger = _lines(tmp_path / "l.jsonl")
    assert [(r["attempt"], r["status"]) for r in ledger] == [(1, status), (2, 200)]
    assert ledger[0]["dollars"] == 0.0 and ledger[1]["input_tokens"] == 120


def test_a_timeout_or_dropped_connection_is_retried(tmp_path):
    script = [AttemptFailed("timeout"), AttemptFailed("connection"), GOOD]
    client, _, sleeps = _client(tmp_path, script)
    assert client.decide(_request()).attempts == 3
    assert sleeps == [1, 2]
    assert [r["status"] for r in _lines(tmp_path / "l.jsonl")] == ["timeout", "connection", 200]


def test_five_attempts_then_a_transport_failure_that_is_not_cached(tmp_path):
    client, provider, sleeps = _client(tmp_path, [AttemptFailed("http_503", 503)])
    result = client.decide(_request())
    assert result.transport_failure and result.body is None and result.attempts == 5
    assert (provider.calls, sleeps) == (5, [1, 2, 4, 8])  # no sleep after the last attempt
    assert (client.calls, client.attempts, client.transport_failures) == (1, 5, 1)
    assert not (tmp_path / "c.jsonl").exists()
    assert len(_lines(tmp_path / "l.jsonl")) == 5
    again = client.decide(_request())  # the second variant must not retry it in this process
    assert again is result and provider.calls == 5


def test_a_422_is_a_transport_failure_retried_by_the_identical_command(tmp_path):
    client, provider, sleeps = _client(tmp_path, [AttemptFailed("http_422", 422)])
    assert client.decide(_request()).transport_failure
    assert (provider.calls, sleeps) == (1, [])
    assert not (tmp_path / "c.jsonl").exists()
    rerun, provider2, _ = _client(tmp_path, [GOOD], run_id="r2")
    assert not rerun.decide(_request()).transport_failure
    assert provider2.calls == 1 and (tmp_path / "c.jsonl").exists()


@pytest.mark.parametrize("status", [401, 403])
def test_unauthorised_stops_the_run_naming_the_variable(tmp_path, status):
    client, provider, sleeps = _client(tmp_path, [AttemptFailed(f"http_{status}", status)])
    with pytest.raises(AuthError, match="TYPESAFE_API_KEY") as raised:
        client.decide(_request())
    assert isinstance(raised.value, ValueError) and (provider.calls, sleeps) == (1, [])
    assert [r["status"] for r in _lines(tmp_path / "l.jsonl")] == [status]


@pytest.mark.parametrize(
    ("body", "error"),
    [
        (fake_decision_response("card_arrival", model="jev-1.14.0"), VersionMismatch),
        (fake_decision_response("card_arrival", model=None), VersionMismatch),
        (fake_decision_response("card_arrival", input_tokens=None), UsageError),
        (fake_decision_response("card_arrival", input_tokens=4001), UsageError),
    ],
)
def test_version_and_usage_violations_stop_the_run_after_the_ledger_line(tmp_path, body, error):
    client, _, _ = _client(tmp_path, [body])
    with pytest.raises(error):
        client.decide(_request())
    assert len(_lines(tmp_path / "l.jsonl")) == 1
    assert not (tmp_path / "c.jsonl").exists()


def test_the_token_guard_allows_exactly_4000(tmp_path):
    client, _, _ = _client(tmp_path, [fake_decision_response("card_arrival", input_tokens=4000)])
    assert client.decide(_request()).input_tokens == 4000


def test_dollars_are_input_tokens_times_the_price(tmp_path):
    client, _, _ = _client(tmp_path, [GOOD], price=0.042)
    client.decide(_request())
    assert _lines(tmp_path / "l.jsonl")[0]["dollars"] == pytest.approx(120 * 0.042 / 1e6)


def test_the_run_cap_aborts_before_the_next_attempt(tmp_path):
    big = fake_decision_response("card_arrival", input_tokens=4000)
    client, provider, _ = _client(tmp_path, [big], price=250)  # $1.00 per call at 4,000 tokens
    client.decide(_request("a"))
    client.decide(_request("b"))
    with pytest.raises(BudgetExceeded, match="run cap"):
        client.decide(_request("c"))
    assert provider.calls == 2 and client.ledger.run_total == pytest.approx(2.0)


def test_the_study_cap_sums_the_ledger_across_runs(tmp_path):
    big = fake_decision_response("card_arrival", input_tokens=4000)
    first, _, _ = _client(tmp_path, [big], price=250, run_id="r1")
    first.decide(_request("a"))
    first.decide(_request("b"))
    second, _, _ = _client(tmp_path, [big], price=250, run_id="r2")
    assert second.ledger.study_total == pytest.approx(2.0) and second.ledger.run_total == 0.0
    second.decide(_request("c"))
    second.decide(_request("d"))
    third, provider3, _ = _client(tmp_path, [big], price=250, run_id="r3")
    third.decide(_request("e"))  # study total 4.00 -> 5.00 after this call
    with pytest.raises(BudgetExceeded, match="study cap"):
        third.decide(_request("f"))
    assert provider3.calls == 1 and third.ledger.run_total < da.RUN_CAP_USD


def test_bypassing_the_cache_records_a_self_agreement_line_and_no_record(tmp_path):
    client, provider, _ = _client(tmp_path, [GOOD])
    client.decide(_request())
    client.decide(_request(), kind="self_agreement", use_cache=False)
    assert provider.calls == 2
    assert [r["kind"] for r in _lines(tmp_path / "l.jsonl")] == ["record", "self_agreement"]
    assert len(_lines(tmp_path / "c.jsonl")) == 1


# -- the arm through the study ----------------------------------------------------------------


def _reply_by_text(request):
    n = len(request["state"])
    if n % 5 == 0:
        return AttemptFailed("http_422", 422)
    return fake_decision_response(["card_arrival", "age_limit", "exchange_rate"][n % 3], 0.6)


def _fixture_runs(tmp_path):
    fit_all, test = tickets.validation_split(tickets.load_fixture())
    fit, val = tickets.validation_split(fit_all)
    client, provider, _ = _client(tmp_path, _reply_by_text)
    runs = []
    for arm in da.decision_arms(client):
        runs += study.run_arm(arm, fit, val, test, split_name="heldout")
    return client, provider, runs, len(set(val.documents + test.documents))


def test_arm_c_satisfies_the_contract_and_is_registered(tmp_path):
    client, _, _ = _client(tmp_path, [GOOD])
    assert ARMS["C"] is DecisionArm
    arms = da.decision_arms(client)
    assert [a.name for a in arms] == ["C-prob", "C-vendor"]
    assert all(isinstance(a, Router) for a in arms)
    with pytest.raises(ValueError, match="not fitted"):
        arms[0].predict_top(["x"])
    with pytest.raises(ValueError, match="variant"):
        DecisionArm(client, "other")


def test_saved_c_rows_carry_six_extra_keys_and_flag_transport_failures(tmp_path):
    client, provider, runs, distinct = _fixture_runs(tmp_path)
    assert {r.split for r in runs} == {"heldout"} and len(runs) == 4
    path = tmp_path / "runs.jsonl"
    study.save_runs(runs, path)
    rows = _lines(path)
    assert {frozenset(r) for r in rows} == {frozenset(V2_KEYS | set(da.EXTRA_KEYS))}
    failed = [r for r in rows if r["transport_failure"]]
    assert failed and all(r["y_pred"] == -1 and r["confidence"] == 0.0 for r in failed)
    assert all(r["model_version"] is None and r["argmax_status"] is None for r in failed)
    ok = [r for r in rows if not r["transport_failure"]]
    assert {r["model_version"] for r in ok} == {"jev-1.13.0"}
    assert {r["endpoint"] for r in rows} == {da.ENDPOINT}
    assert all(r["cached_prompt_tokens"] is None for r in rows)
    assert {r["argmax_status"] for r in ok} == {"unique"}
    # One response per distinct ticket feeds both variants: the second variant only hits.
    assert provider.calls == client.calls == client.hits == distinct
    assert client.transport_failures > 0


def test_the_prob_and_vendor_runs_share_labels_but_not_confidences(tmp_path):
    _, _, runs, _ = _fixture_runs(tmp_path)
    prob, vendor = runs[0], runs[2]
    assert (prob.arm, vendor.arm) == ("C-prob", "C-vendor")
    ok = prob.y_pred != -1
    assert np.array_equal(prob.y_pred, vendor.y_pred)
    assert np.allclose(prob.confidence[ok], 0.6) and np.allclose(vendor.confidence[ok], 0.8)


# -- the transport ----------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status=200, payload=b"{}", will_close=False):
        self.status, self._payload, self.will_close = status, payload, will_close

    def read(self):
        return self._payload


class FakeConnection:
    def __init__(self, script):
        self.script, self.sent, self.closed = script, [], False

    def request(self, method, path, body=None, headers=None):
        self.sent.append((method, path, body, headers))

    def getresponse(self):
        out = self.script.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    def close(self):
        self.closed = True


def _provider(script, key=KEY):
    made = []

    def factory(host, timeout):
        made.append((host, timeout, FakeConnection(script)))
        return made[-1][2]

    environ = {"TYPESAFE_API_KEY": key} if key else {}
    return ts.TypeSafeProvider(connection_factory=factory, environ=environ), made


def test_constructing_a_provider_opens_nothing_and_reads_no_credential():
    provider, made = _provider([], key=None)
    assert made == [] and provider.identity == da.ENDPOINT
    assert KEY not in repr(ts.TypeSafeProvider(environ={"TYPESAFE_API_KEY": KEY}))


def test_one_connection_is_reused_and_only_the_first_request_is_cold():
    body = json.dumps(GOOD).encode()
    provider, made = _provider([FakeResponse(payload=body), FakeResponse(payload=body)])
    first, second = provider.decide(_request("a")), provider.decide(_request("b"))
    assert (first.cold_start, second.cold_start) == (True, False)
    assert len(made) == 1 and made[0][:2] == ("api.typesafe.ai", 30.0)
    method, path, sent, headers = made[0][2].sent[0]
    assert (method, path) == ("POST", "/v1/systemone")
    assert json.loads(sent) == _request("a")
    assert headers["Authorization"] == f"Bearer {KEY}"
    assert first.body == GOOD and first.seconds >= 0


def test_a_dropped_connection_fails_the_attempt_and_the_next_one_reconnects():
    script = [OSError("reset"), FakeResponse(payload=json.dumps(GOOD).encode())]
    provider, made = _provider(script)
    with pytest.raises(AttemptFailed) as dropped:
        provider.decide(_request())
    assert dropped.value.error == "connection" and dropped.value.retryable
    assert made[0][2].closed
    again = provider.decide(_request())
    assert len(made) == 2 and again.cold_start


@pytest.mark.parametrize(
    ("outcome", "error", "status", "retryable"),
    [
        (TimeoutError("slow"), "timeout", None, True),
        (FakeResponse(429), "http_429", 429, True),
        (FakeResponse(401), "http_401", 401, False),
        (FakeResponse(422), "http_422", 422, False),
        (FakeResponse(200, b"<html>"), "non_json", 200, False),
        (FakeResponse(200, b"[1, 2]"), "non_json", 200, False),
    ],
)
def test_failed_attempts_are_classified(outcome, error, status, retryable):
    provider, _ = _provider([outcome])
    with pytest.raises(AttemptFailed) as failure:
        provider.decide(_request())
    assert (failure.value.error, failure.value.status) == (error, status)
    assert failure.value.retryable is retryable


def test_a_connection_the_server_closes_is_not_reused():
    provider, made = _provider([FakeResponse(payload=json.dumps(GOOD).encode(), will_close=True)])
    provider.decide(_request())
    assert made[0][2].closed and provider._connection is None


def test_ping_is_a_get_on_the_models_path():
    provider, made = _provider([FakeResponse(200)])
    status, seconds = provider.ping()
    assert status == 200 and seconds >= 0
    assert made[0][2].sent[0][:3] == ("GET", "/v1/models", None)


# -- credentials ------------------------------------------------------------------------------


class _Keyring(types.ModuleType):
    def __init__(self, value):
        super().__init__("keyring")
        self.value, self.asked = value, []

    def get_password(self, service, name):
        self.asked.append((service, name))
        return self.value


def test_the_environment_wins_over_the_keyring(monkeypatch):
    fake = _Keyring("from-keyring")
    monkeypatch.setitem(sys.modules, "keyring", fake)
    assert ts.resolve_api_key({"TYPESAFE_API_KEY": KEY}) == KEY
    assert fake.asked == []


def test_the_keyring_is_skipped_when_disabled_and_used_otherwise(monkeypatch):
    fake = _Keyring("from-keyring")
    monkeypatch.setitem(sys.modules, "keyring", fake)
    with pytest.raises(ts.MissingCredential, match="TYPESAFE_API_KEY"):
        ts.resolve_api_key({"SECRETS_NO_KEYRING": "1"})
    assert fake.asked == []
    assert ts.resolve_api_key({}) == "from-keyring"
    assert fake.asked == [("classical-nlp", "TYPESAFE_API_KEY")]
    monkeypatch.setitem(sys.modules, "keyring", None)  # package not installed
    with pytest.raises(ts.MissingCredential):
        ts.resolve_api_key({})


def test_the_key_is_looked_up_once_not_per_request(monkeypatch):
    fake = _Keyring(KEY)
    monkeypatch.setitem(sys.modules, "keyring", fake)
    body = json.dumps(GOOD).encode()
    made = []

    def factory(host, timeout):
        made.append(FakeConnection([FakeResponse(payload=body)] * 3))
        return made[-1]

    provider = ts.TypeSafeProvider(connection_factory=factory, environ={})
    for _ in range(3):
        provider.decide(_request())
    assert fake.asked == [("classical-nlp", "TYPESAFE_API_KEY")]


def test_a_missing_key_stops_before_any_connection():
    provider, made = _provider([], key=None)
    with pytest.raises(ts.MissingCredential, match="TYPESAFE_API_KEY") as raised:
        provider.decide(_request())
    assert isinstance(raised.value, ValueError) and made == []


def test_the_key_never_reaches_the_cache_the_ledger_or_an_error(tmp_path):
    body = json.dumps(GOOD).encode()
    provider, _ = _provider([FakeResponse(payload=body), FakeResponse(401)])
    client = DecisionClient(
        provider, DecisionCache(tmp_path / "c.jsonl"), Ledger(tmp_path / "l.jsonl")
    )
    client.decide(_request("a"))
    with pytest.raises(AuthError) as raised:
        client.decide(_request("b"))
    texts = [str(raised.value), repr(provider), repr(client.provider)]
    texts += [(tmp_path / name).read_text() for name in ("c.jsonl", "l.jsonl")]
    assert all(KEY not in text for text in texts)


# -- review fixes: key hygiene, probabilities over the 77 names, first cache record --------


def test_a_key_with_a_trailing_newline_is_sent_stripped():
    body = json.dumps(GOOD).encode()
    provider, made = _provider([FakeResponse(payload=body)], key=KEY + "\n")
    provider.decide(_request())
    headers = made[0][2].sent[0][3]
    assert headers["Authorization"] == f"Bearer {KEY}"


def test_a_key_with_a_control_character_inside_is_refused_without_quoting_it():
    provider, made = _provider([], key="test-key\nnot-real")
    with pytest.raises(ts.MissingCredential, match="TYPESAFE_API_KEY") as raised:
        provider.decide(_request())
    assert "not-real" not in str(raised.value) and made == []


def test_a_header_error_from_http_client_never_carries_the_key():
    class Rejecting(FakeConnection):
        def request(self, method, path, body=None, headers=None):
            raise ValueError(f"Invalid header value {headers['Authorization'].encode()!r}")

    provider = ts.TypeSafeProvider(
        connection_factory=lambda host, timeout: Rejecting([]),
        environ={"TYPESAFE_API_KEY": KEY},
    )
    with pytest.raises(da.DecisionError) as raised:
        provider.decide(_request())
    assert KEY not in str(raised.value)
    # Nothing chained at all: ``from None`` only hides a __context__, it does not drop it.
    assert raised.value.__cause__ is None and raised.value.__context__ is None


def test_only_the_77_names_are_summed_and_ranked():
    heavy = {n: 1.6 / 77 for n in NAMES}  # the names sum to 1.6
    assert _top(_tweak(probabilities={**heavy, "other": -0.6}), "prob") == (-1, 0.0)
    half = {n: 0.5 / 77 for n in NAMES}  # the names sum to 0.5
    assert _top(_tweak(probabilities={**half, "other": 0.5}), "prob") == (-1, 0.0)
    peaked = {n: (0.5 if n == "card_arrival" else 0.5 / 76) for n in NAMES}
    assert _status("card_arrival", {**peaked, "other": 0.9}) == "unique"
    assert _top(_tweak(probabilities={**peaked, "other": 0.9}), "prob")[1] == 0.5


def test_the_first_cached_response_for_a_ticket_is_its_record(tmp_path):
    cache = DecisionCache(tmp_path / "c.jsonl")
    first = da.DecisionResponse(GOOD, "jev-1.13.0", 10, 1, 0.1, False, 1)
    later = da.DecisionResponse(GOOD, "jev-1.13.0", 99, 1, 0.9, False, 1)
    cache.put("k", da.ENDPOINT, first)
    cache.put("k", da.ENDPOINT, later)
    reloaded = DecisionCache(tmp_path / "c.jsonl").get("k")
    assert (reloaded.input_tokens, reloaded.seconds) == (10, 0.1)
