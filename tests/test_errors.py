"""Every kind of LLM failure must map to the right ErrorKind (built from the real SDK error classes)."""
import json

import httpx
import pytest
import requests
from google.genai import errors as genai_errors
from pydantic import BaseModel

from backend.core.errors import (
    ContentBlockedError, ErrorKind, LLMCallError, classify_error, retry_after_seconds, to_llm_error,
)


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


class TestGeminiApiErrors:
    def test_per_minute_limit_is_rate_limited_with_the_requested_pause(self):
        exc = api_error(429, "RESOURCE_EXHAUSTED",
                        details=quota_details("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "34s"))
        assert classify_error(exc) is ErrorKind.RATE_LIMITED
        assert retry_after_seconds(exc) == 34.0

    def test_daily_limit_is_quota_exhausted_not_worth_retrying(self):
        exc = api_error(429, "RESOURCE_EXHAUSTED",
                        details=quota_details("GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
        assert classify_error(exc) is ErrorKind.QUOTA_EXHAUSTED
        assert ErrorKind.QUOTA_EXHAUSTED.fatal and not ErrorKind.QUOTA_EXHAUSTED.retryable

    def test_invalid_key_arrives_as_400_but_is_an_auth_error(self):
        details = [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}]
        exc = api_error(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.", details)
        assert classify_error(exc) is ErrorKind.AUTH

    @pytest.mark.parametrize("code,status", [(401, "UNAUTHENTICATED"), (403, "PERMISSION_DENIED")])
    def test_permission_errors_are_auth(self, code, status):
        assert classify_error(api_error(code, status)) is ErrorKind.AUTH

    def test_unknown_model_is_a_config_error(self):
        assert classify_error(api_error(404, "NOT_FOUND")) is ErrorKind.CONFIG

    def test_unsupported_region_is_a_config_error(self):
        assert classify_error(api_error(400, "FAILED_PRECONDITION")) is ErrorKind.CONFIG

    def test_other_400_is_a_bad_request_that_retrying_cannot_fix(self):
        kind = classify_error(api_error(400, "INVALID_ARGUMENT", "Request payload size exceeds the limit"))
        assert kind is ErrorKind.BAD_REQUEST
        assert not kind.retryable and not kind.fatal

    @pytest.mark.parametrize("code,status", [(500, "INTERNAL"), (503, "UNAVAILABLE"), (504, "DEADLINE_EXCEEDED")])
    def test_server_side_errors_are_transient(self, code, status):
        kind = classify_error(api_error(code, status))
        assert kind is ErrorKind.TRANSIENT and kind.retryable


class TestOtherFailures:
    @pytest.mark.parametrize("exc", [
        httpx.ReadTimeout("slow"), httpx.ConnectError("down"),
        requests.exceptions.ConnectionError("down"), requests.exceptions.Timeout("slow"),
        TimeoutError(), ConnectionResetError(),
    ])
    def test_network_problems_are_transient(self, exc):
        assert classify_error(exc) is ErrorKind.TRANSIENT

    def test_invalid_json_from_the_model_is_bad_output(self):
        with pytest.raises(json.JSONDecodeError) as caught:
            json.loads("{not json")
        assert classify_error(caught.value) is ErrorKind.BAD_OUTPUT

    def test_schema_violation_is_bad_output(self):
        class Schema(BaseModel):
            needs_refinement: bool

        with pytest.raises(Exception) as caught:
            Schema(needs_refinement="definitely maybe")
        assert classify_error(caught.value) is ErrorKind.BAD_OUTPUT

    def test_blocked_content_is_its_own_kind(self):
        assert classify_error(ContentBlockedError("blocked by safety filter")) is ErrorKind.CONTENT_BLOCKED

    @pytest.mark.parametrize("exc", [KeyError("x"), TypeError("x"), AttributeError("x")])
    def test_programming_errors_are_unknown_and_never_retried(self, exc):
        kind = classify_error(exc)
        assert kind is ErrorKind.UNKNOWN and not kind.retryable


class TestWrapping:
    def test_wrapping_keeps_kind_label_and_retry_hint(self):
        exc = api_error(429, "RESOURCE_EXHAUSTED",
                        details=quota_details("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "12s"))
        wrapped = to_llm_error(exc, label="query_gen")
        assert isinstance(wrapped, LLMCallError)
        assert wrapped.kind is ErrorKind.RATE_LIMITED and wrapped.retry_after == 12.0
        assert "query_gen" in str(wrapped)

    def test_wrapping_twice_changes_nothing(self):
        once = to_llm_error(api_error(503, "UNAVAILABLE"), label="x")
        assert to_llm_error(once) is once
        assert classify_error(once) is ErrorKind.TRANSIENT

    def test_only_retryable_and_fatal_kinds_are_flagged(self):
        retryable = {k for k in ErrorKind if k.retryable}
        fatal = {k for k in ErrorKind if k.fatal}
        assert retryable == {ErrorKind.RATE_LIMITED, ErrorKind.TRANSIENT}
        assert fatal == {ErrorKind.AUTH, ErrorKind.CONFIG, ErrorKind.QUOTA_EXHAUSTED}
        assert not retryable & fatal
