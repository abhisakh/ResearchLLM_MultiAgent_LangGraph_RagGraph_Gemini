"""How a run that could not finish is reported to an API client and to the person using the UI."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from backend.core import http_errors
from backend.core.errors import ErrorKind, describe_failure
from backend.core.http_errors import STATUS_FOR_KIND, http_response_for, seconds_until_midnight_pacific
from frontend.api_errors import describe_api_error, wait_text


def failure_of(kind, error_id="abc12345", agent="synthesis_agent"):
    """A failure dict exactly as the supervisor stores it in the state."""
    return describe_failure({"id": error_id, "agent": agent, "kind": kind.value, "message": "raw sdk text"})


def pacific_database_available():
    try:
        http_errors.ZoneInfo("America/Los_Angeles")
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------------------
# Status codes, bodies and headers
# --------------------------------------------------------------------------------------
def test_every_kind_of_error_has_a_status_code():
    assert set(STATUS_FOR_KIND) == set(ErrorKind)


@pytest.mark.parametrize(
    "kind, status",
    [
        (ErrorKind.QUOTA_EXHAUSTED, 503),
        (ErrorKind.RATE_LIMITED, 503),
        (ErrorKind.TRANSIENT, 503),
        (ErrorKind.AUTH, 503),
        (ErrorKind.CONFIG, 503),
        (ErrorKind.CONTENT_BLOCKED, 422),
        (ErrorKind.BAD_REQUEST, 502),
        (ErrorKind.BAD_OUTPUT, 502),
        (ErrorKind.UNKNOWN, 500),
    ],
)
def test_status_code_per_kind(kind, status):
    assert http_response_for(failure_of(kind))[0] == status


def test_the_body_carries_the_kind_the_safe_message_and_the_error_id():
    _, detail, _ = http_response_for(failure_of(ErrorKind.AUTH))
    assert detail == {
        "error": "auth",
        "message": failure_of(ErrorKind.AUTH)["message"],
        "error_id": "abc12345",
    }
    assert "GEMINI_API_KEY" in detail["message"]


def test_an_unexpected_error_gives_the_error_id_to_quote():
    _, detail, _ = http_response_for(failure_of(ErrorKind.UNKNOWN))
    assert "abc12345" in detail["message"]


def test_the_raw_error_text_never_reaches_the_client():
    _, detail, _ = http_response_for(failure_of(ErrorKind.TRANSIENT))
    assert "raw sdk text" not in str(detail)


def test_a_failure_with_an_unrecognised_kind_is_treated_as_unknown():
    status, detail, _ = http_response_for({"kind": "something_new", "message": "m", "error_id": "x"})
    assert status == 500 and detail["error"] == "unknown"


def test_a_daily_quota_tells_the_client_when_it_resets():
    now = datetime(2026, 7, 10, 19, 0, tzinfo=timezone.utc)
    _, _, headers = http_response_for(failure_of(ErrorKind.QUOTA_EXHAUSTED), now=now)
    assert headers == {"Retry-After": str(seconds_until_midnight_pacific(now))}


@pytest.mark.parametrize("kind, seconds", [(ErrorKind.RATE_LIMITED, "60"), (ErrorKind.TRANSIENT, "30")])
def test_short_outages_ask_the_client_to_wait_a_little(kind, seconds):
    assert http_response_for(failure_of(kind))[2] == {"Retry-After": seconds}


@pytest.mark.parametrize(
    "kind",
    [ErrorKind.AUTH, ErrorKind.CONFIG, ErrorKind.CONTENT_BLOCKED, ErrorKind.BAD_REQUEST,
     ErrorKind.BAD_OUTPUT, ErrorKind.UNKNOWN],
)
def test_waiting_does_not_help_for_the_other_kinds(kind):
    assert http_response_for(failure_of(kind))[2] == {}


# --------------------------------------------------------------------------------------
# Midnight Pacific time
# --------------------------------------------------------------------------------------
@pytest.mark.skipif(not pacific_database_available(), reason="no time zone database in this environment")
@pytest.mark.parametrize(
    "now, seconds",
    [
        (datetime(2026, 7, 10, 19, 0, tzinfo=timezone.utc), 12 * 3600),   # noon PDT (UTC-7)
        (datetime(2026, 1, 15, 7, 59, tzinfo=timezone.utc), 60),          # 23:59 PST (UTC-8)
        (datetime(2026, 1, 15, 8, 0, tzinfo=timezone.utc), 24 * 3600),    # exactly midnight: a whole day, never 0
    ],
)
def test_seconds_until_the_daily_quota_resets(now, seconds):
    assert seconds_until_midnight_pacific(now) == seconds


def test_without_a_time_zone_database_standard_time_is_assumed(monkeypatch):
    def missing(name):
        raise KeyError(name)

    monkeypatch.setattr(http_errors, "ZoneInfo", missing)
    now = datetime(2026, 7, 10, 19, 0, tzinfo=timezone.utc)  # 11:00 at UTC-8
    assert seconds_until_midnight_pacific(now) == 13 * 3600


# --------------------------------------------------------------------------------------
# What the person using the Streamlit UI reads
# --------------------------------------------------------------------------------------
def api_response(status, body=None, headers=None, json_error=False):
    def json_body():
        if json_error:
            raise ValueError("not json")
        return body

    return SimpleNamespace(status_code=status, headers=headers or {}, json=json_body)


@pytest.mark.parametrize(
    "seconds, text",
    [(1, "1 second"), (45, "45 seconds"), (600, "10 minutes"), (3600, "60 minutes"), (43200, "12 hours")],
)
def test_wait_text(seconds, text):
    assert wait_text(seconds) == text


def test_the_ui_shows_the_backend_message_and_how_long_to_wait():
    response = api_response(
        503, {"detail": {"error": "quota_exhausted", "message": "Quota used up."}}, {"Retry-After": "600"}
    )
    assert describe_api_error(response) == "Quota used up. You can try again in about 10 minutes."


def test_the_ui_shows_just_the_message_when_there_is_no_retry_after():
    response = api_response(503, {"detail": {"message": "Check GEMINI_API_KEY."}})
    assert describe_api_error(response) == "Check GEMINI_API_KEY."


@pytest.mark.parametrize(
    "response",
    [
        api_response(500, {"detail": "Internal Server Error"}),                  # FastAPI's own plain error
        api_response(422, {"detail": [{"loc": ["body", "message"], "msg": "x"}]}),  # request validation error
        api_response(502, json_error=True),                                       # an HTML page from a proxy
    ],
)
def test_the_ui_falls_back_to_a_generic_sentence_with_the_status(response):
    text = describe_api_error(response)
    assert str(response.status_code) in text and "could not complete" in text
