"""POST /research-chat: a run that fails gets a clear HTTP error, a run that finishes says how far to trust it.

These tests use the real FastAPI app and the real SQLite chat log; only the research graph is scripted
(see the `api` fixture in conftest.py).
"""
import json
import re

import pytest

from backend.core.errors import ErrorKind, LLMCallError, describe_failure


def finished_state(**overrides):
    state = {
        "final_report": "The report.",
        "visited_nodes": ["supervisor_agent", "synthesis_agent"],
        "report_generated": True,
        "refinement_retries": 0,
        "errors": [],
        "evaluation_unavailable": False,
        "failure": None,
    }
    state.update(overrides)
    return state


def failed_state(kind, agent="query_gen_agent", error_id="abc12345"):
    """The final state of a run the supervisor ended: no report, and `failure` says why."""
    failure = describe_failure({"id": error_id, "agent": agent, "kind": kind.value, "message": "raw sdk text"})
    return finished_state(final_report="", report_generated=False, failure=failure)


def ask(client, message="What is a perovskite?", session_id=None):
    return client.post("/research-chat", json={"message": message, "session_id": session_id})


# --------------------------------------------------------------------------------------
# A run that finishes
# --------------------------------------------------------------------------------------
def test_a_finished_run_returns_the_report(api):
    client, script = api
    script["result"] = finished_state()

    response = ask(client)

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "The report."
    assert body["visited_path"] == ["supervisor_agent", "synthesis_agent"]
    assert body["metadata"]["evaluation_unavailable"] is False
    assert body["metadata"]["degraded_agents"] == []


def test_execution_time_is_the_duration_of_the_run_in_seconds(api):
    client, script = api
    script["result"] = finished_state()

    seconds = ask(client).json()["metadata"]["execution_time"]

    # The scripted graph returns at once. The old value was an ISO timestamp string, which fails this.
    assert isinstance(seconds, (int, float))
    assert 0 <= seconds < 60


def test_the_response_says_when_the_report_was_not_reviewed(api):
    client, script = api
    script["result"] = finished_state(
        evaluation_unavailable=True,
        errors=[{"id": "e1", "agent": "evaluation_agent", "kind": "quota_exhausted", "message": "[quota_exhausted] raw"}],
    )

    response = ask(client)

    assert response.status_code == 200
    metadata = response.json()["metadata"]
    assert metadata["evaluation_unavailable"] is True
    assert metadata["degraded_agents"] == [{"agent": "evaluation_agent", "kind": "quota_exhausted"}]
    assert "raw" not in response.text  # the error details stay in the server log


# --------------------------------------------------------------------------------------
# A run that fails
# --------------------------------------------------------------------------------------
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
def test_a_failed_run_gets_the_status_for_its_kind(api, kind, status):
    client, script = api
    script["result"] = failed_state(kind)

    response = ask(client)

    assert response.status_code == status
    detail = response.json()["detail"]
    assert detail["error"] == kind.value
    assert detail["error_id"] == "abc12345"
    assert "raw sdk text" not in response.text


def test_a_daily_quota_says_when_to_come_back(api):
    client, script = api
    script["result"] = failed_state(ErrorKind.QUOTA_EXHAUSTED)

    response = ask(client)

    assert int(response.headers["Retry-After"]) > 0
    assert "midnight Pacific" in response.json()["detail"]["message"]


def test_a_rate_limit_asks_for_a_short_wait(api):
    client, script = api
    script["result"] = failed_state(ErrorKind.RATE_LIMITED)
    assert ask(client).headers["Retry-After"] == "60"


def test_a_bad_key_names_the_setting_to_fix_and_asks_for_no_retry(api):
    client, script = api
    script["result"] = failed_state(ErrorKind.AUTH)

    response = ask(client)

    assert "GEMINI_API_KEY" in response.json()["detail"]["message"]
    assert "Retry-After" not in response.headers


def test_an_unexpected_failure_gives_the_error_id_to_quote(api):
    client, script = api
    script["result"] = failed_state(ErrorKind.UNKNOWN, error_id="3f9a1c2e")

    assert "3f9a1c2e" in ask(client).json()["detail"]["message"]


def test_a_failed_run_is_kept_in_the_chat_log_as_an_error_turn(api):
    client, script = api
    script["result"] = failed_state(ErrorKind.QUOTA_EXHAUSTED)

    ask(client, "What is a perovskite?", session_id="session-1")
    history = client.get("/chat-history/session-1").json()

    assert [turn["role"] for turn in history] == ["user", "error"]
    assert history[1]["message"] == describe_failure(
        {"id": "abc12345", "agent": "query_gen_agent", "kind": "quota_exhausted"}
    )["message"]
    assert history[1]["tool_used"] == "query_gen_agent"


# --------------------------------------------------------------------------------------
# A crash instead of a clean failure
# --------------------------------------------------------------------------------------
def test_a_crash_inside_the_graph_does_not_leak_internals(api):
    client, script = api
    script["raises"] = RuntimeError("secret internals at /Users/someone/.env")

    response = ask(client, session_id="session-2")

    assert response.status_code == 500
    assert "secret internals" not in response.text
    error_id = response.json()["detail"]["error_id"]
    assert re.fullmatch(r"[0-9a-f]{8}", error_id)
    assert error_id in response.json()["detail"]["message"]

    stored = json.dumps(client.get("/chat-history/session-2").json())
    assert "secret internals" not in stored
    assert "Traceback" not in stored


def test_an_llm_error_that_escapes_an_agent_is_still_classified(api):
    client, script = api
    script["raises"] = LLMCallError(ErrorKind.QUOTA_EXHAUSTED, "daily quota")

    response = ask(client)

    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "quota_exhausted"
    assert "Retry-After" in response.headers


def test_a_run_that_ends_without_a_report_or_a_failure_is_an_error(api):
    client, script = api
    script["result"] = finished_state(final_report="")

    response = ask(client)

    assert response.status_code == 500
    assert response.json()["detail"]["error"] == "unknown"


def test_a_message_of_only_whitespace_is_rejected(api):
    client, _ = api
    assert ask(client, "   ").status_code == 400
