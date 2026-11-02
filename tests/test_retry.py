"""call_with_retry and generate_content_with_retry: every kind of failure gets its own reaction."""
import types

import pytest
from google.genai import errors as genai_errors

from backend.core import utilities
from backend.core.errors import ErrorKind, LLMCallError

PER_MINUTE = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
PER_DAY = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"


def api_error(code, status, message="boom", details=None):
    """Builds the same error object the Gemini SDK raises for an HTTP error response."""
    body = {"error": {"code": code, "message": message, "status": status, "details": details or []}}
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, body)


def quota_details(quota_id, retry_delay=None):
    details = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": quota_id}]}]
    if retry_delay:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay})
    return details


def per_minute_429(retry_delay=None):
    return api_error(429, "RESOURCE_EXHAUSTED", details=quota_details(PER_MINUTE, retry_delay))


def daily_429():
    return api_error(429, "RESOURCE_EXHAUSTED", details=quota_details(PER_DAY))


class Flaky:
    """A callable that raises the given errors one after another, then returns `result`.
    With repeat_last=True the last error is raised for ever."""

    def __init__(self, *errors, result="ok", repeat_last=False):
        self.errors = list(errors)
        self.result = result
        self.repeat_last = repeat_last
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.errors:
            if self.repeat_last and len(self.errors) == 1:
                raise self.errors[0]
            raise self.errors.pop(0)
        return self.result


@pytest.fixture
def sleeps(monkeypatch):
    """Replaces time.sleep so retries run instantly; returns the list of requested pauses."""
    recorded = []
    monkeypatch.setattr(utilities.time, "sleep", recorded.append)
    return recorded


# --------------------------------------------------------------------------------------
# call_with_retry: one reaction per kind of error
# --------------------------------------------------------------------------------------
def test_success_returns_at_once(sleeps):
    fn = Flaky(result=42)
    assert utilities.call_with_retry(fn) == 42
    assert fn.calls == 1 and sleeps == []


def test_per_minute_limit_waits_the_pause_google_asks_for(sleeps):
    fn = Flaky(per_minute_429("34s"))
    assert utilities.call_with_retry(fn) == "ok"
    assert sleeps == [35.0]  # 34 s asked for + 1 s padding


def test_per_minute_limit_without_a_hint_backs_off_exponentially(sleeps):
    fn = Flaky(per_minute_429(), per_minute_429())
    assert utilities.call_with_retry(fn) == "ok"
    assert sleeps == [2.0, 4.0]


def test_waiting_longer_than_the_budget_gives_up_as_rate_limited(sleeps):
    fn = Flaky(per_minute_429("34s"), repeat_last=True)
    with pytest.raises(LLMCallError) as caught:
        utilities.call_with_retry(fn)
    assert caught.value.kind is ErrorKind.RATE_LIMITED
    assert caught.value.retry_after == 34.0
    assert sleeps == [35.0]  # a second 35 s wait would pass the 60 s budget
    assert fn.calls == 2


def test_transient_errors_are_retried_three_times_then_reported(sleeps):
    fn = Flaky(api_error(503, "UNAVAILABLE"), repeat_last=True)
    with pytest.raises(LLMCallError) as caught:
        utilities.call_with_retry(fn)
    assert caught.value.kind is ErrorKind.TRANSIENT
    assert fn.calls == 4  # the first try and three retries
    assert sleeps == [2.0, 4.0, 8.0]


def test_transient_error_that_clears_up_returns_the_result(sleeps):
    fn = Flaky(api_error(500, "INTERNAL"), result="recovered")
    assert utilities.call_with_retry(fn) == "recovered"
    assert sleeps == [2.0]


def test_daily_quota_is_never_retried(sleeps):
    fn = Flaky(daily_429(), repeat_last=True)
    with pytest.raises(LLMCallError) as caught:
        utilities.call_with_retry(fn)
    assert caught.value.kind is ErrorKind.QUOTA_EXHAUSTED
    assert fn.calls == 1 and sleeps == []


@pytest.mark.parametrize(
    "error, kind",
    [
        (api_error(400, "INVALID_ARGUMENT", "API key not valid.",
                   [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}]),
         ErrorKind.AUTH),
        (api_error(403, "PERMISSION_DENIED"), ErrorKind.AUTH),
        (api_error(404, "NOT_FOUND"), ErrorKind.CONFIG),
        (api_error(400, "INVALID_ARGUMENT", "Request payload size exceeds the limit"), ErrorKind.BAD_REQUEST),
    ],
)
def test_errors_that_waiting_cannot_fix_are_raised_at_once(sleeps, error, kind):
    fn = Flaky(error, repeat_last=True)
    with pytest.raises(LLMCallError) as caught:
        utilities.call_with_retry(fn)
    assert caught.value.kind is kind
    assert fn.calls == 1 and sleeps == []


def test_unknown_error_is_not_retried_and_keeps_its_traceback(sleeps):
    bug = KeyError("candidates")
    fn = Flaky(bug, repeat_last=True)
    with pytest.raises(LLMCallError) as caught:
        utilities.call_with_retry(fn)
    assert caught.value.kind is ErrorKind.UNKNOWN
    assert caught.value.__cause__ is bug
    assert fn.calls == 1 and sleeps == []


def test_the_label_names_the_failing_call(sleeps):
    with pytest.raises(LLMCallError) as caught:
        utilities.call_with_retry(Flaky(api_error(404, "NOT_FOUND")), label="embed_content")
    assert "embed_content" in str(caught.value)


def test_an_error_that_is_already_classified_passes_through_unchanged(sleeps):
    original = LLMCallError(ErrorKind.AUTH, "bad key")
    with pytest.raises(LLMCallError) as caught:
        utilities.call_with_retry(Flaky(original, repeat_last=True))
    assert caught.value is original


# --------------------------------------------------------------------------------------
# generate_content_with_retry: model fallback
# --------------------------------------------------------------------------------------
def fake_client(behaviour):
    """behaviour(model) returns a response or raises. `asked` records the models that were called."""
    asked = []

    def generate_content(*args, **kwargs):
        asked.append(kwargs["model"])
        return behaviour(kwargs["model"])

    client = types.SimpleNamespace(models=types.SimpleNamespace(generate_content=generate_content))
    return client, asked


def test_daily_quota_switches_to_the_next_model(monkeypatch, sleeps):
    def behaviour(model):
        if model == "primary":
            raise daily_429()
        return f"answer from {model}"

    client, asked = fake_client(behaviour)
    monkeypatch.setattr(utilities, "client", client)
    monkeypatch.setattr(utilities, "LLM_MODEL_FALLBACKS", ["backup"])

    assert utilities.generate_content_with_retry(model="primary", contents="x") == "answer from backup"
    assert asked == ["primary", "backup"]
    assert sleeps == []


def test_without_fallbacks_a_daily_quota_stops_the_call(monkeypatch, sleeps):
    client, asked = fake_client(lambda model: (_ for _ in ()).throw(daily_429()))
    monkeypatch.setattr(utilities, "client", client)
    monkeypatch.setattr(utilities, "LLM_MODEL_FALLBACKS", [])

    with pytest.raises(LLMCallError) as caught:
        utilities.generate_content_with_retry(model="primary", contents="x")
    assert caught.value.kind is ErrorKind.QUOTA_EXHAUSTED
    assert asked == ["primary"]


def test_when_every_model_is_out_of_quota_the_last_error_is_raised(monkeypatch, sleeps):
    client, asked = fake_client(lambda model: (_ for _ in ()).throw(daily_429()))
    monkeypatch.setattr(utilities, "client", client)
    monkeypatch.setattr(utilities, "LLM_MODEL_FALLBACKS", ["backup", "last_resort"])

    with pytest.raises(LLMCallError) as caught:
        utilities.generate_content_with_retry(model="primary", contents="x")
    assert caught.value.kind is ErrorKind.QUOTA_EXHAUSTED
    assert asked == ["primary", "backup", "last_resort"]


def test_other_fatal_errors_do_not_switch_models(monkeypatch, sleeps):
    client, asked = fake_client(lambda model: (_ for _ in ()).throw(api_error(403, "PERMISSION_DENIED")))
    monkeypatch.setattr(utilities, "client", client)
    monkeypatch.setattr(utilities, "LLM_MODEL_FALLBACKS", ["backup"])

    with pytest.raises(LLMCallError) as caught:
        utilities.generate_content_with_retry(model="primary", contents="x")
    assert caught.value.kind is ErrorKind.AUTH
    assert asked == ["primary"]


def test_the_primary_model_is_not_tried_twice(monkeypatch, sleeps):
    client, asked = fake_client(lambda model: (_ for _ in ()).throw(daily_429()))
    monkeypatch.setattr(utilities, "client", client)
    monkeypatch.setattr(utilities, "LLM_MODEL_FALLBACKS", ["primary", "backup"])

    with pytest.raises(LLMCallError):
        utilities.generate_content_with_retry(model="primary", contents="x")
    assert asked == ["primary", "backup"]


def test_a_per_minute_limit_is_waited_out_on_the_same_model(monkeypatch, sleeps):
    calls = {"n": 0}

    def behaviour(model):
        calls["n"] += 1
        if calls["n"] == 1:
            raise per_minute_429("10s")
        return "answer"

    client, asked = fake_client(behaviour)
    monkeypatch.setattr(utilities, "client", client)
    monkeypatch.setattr(utilities, "LLM_MODEL_FALLBACKS", ["backup"])

    assert utilities.generate_content_with_retry(model="primary", contents="x") == "answer"
    assert asked == ["primary", "primary"]
    assert sleeps == [11.0]


def test_a_missing_client_is_reported_as_an_auth_error(monkeypatch):
    monkeypatch.setattr(utilities, "client", None)
    with pytest.raises(LLMCallError) as caught:
        utilities.generate_content_with_retry(model="primary", contents="x")
    assert caught.value.kind is ErrorKind.AUTH
    assert "GEMINI_API_KEY" in str(caught.value)
