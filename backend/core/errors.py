"""Classifies failures of LLM calls so that each kind of error gets the right reaction.

A 429 "slow down" needs a pause, an invalid API key needs a stop, a malformed JSON answer deserves
one more try. Treating them all the same (retry everything, or give up on everything) either wastes
quota or hides the real cause from the user.
"""
import json
import logging
import re
import uuid
from enum import Enum
from typing import Optional

import httpx
from google.genai import errors as genai_errors
from pydantic import ValidationError

try:  # requests ships with google-genai; never let an import problem here break the backend
    import requests
except ImportError:  # pragma: no cover
    requests = None

logger = logging.getLogger(__name__)


class ErrorKind(str, Enum):
    RATE_LIMITED = "rate_limited"          # 429 per-minute limit: wait, then retry
    QUOTA_EXHAUSTED = "quota_exhausted"    # 429 daily quota: retrying now is pointless
    TRANSIENT = "transient"                # 500/503/504, timeouts, dropped connections: retry a few times
    AUTH = "auth"                          # missing, invalid or forbidden API key: a person must fix it
    CONFIG = "config"                      # unknown model name, unsupported region: a person must fix it
    BAD_REQUEST = "bad_request"            # the request itself is wrong (400): retrying cannot help
    CONTENT_BLOCKED = "content_blocked"    # a safety filter blocked the prompt or the answer
    BAD_OUTPUT = "bad_output"              # the model answered, but not with valid JSON / the expected schema
    UNKNOWN = "unknown"                    # anything else: most likely a bug in our own code

    @property
    def retryable(self) -> bool:
        """Waiting and calling again can succeed."""
        return self in (ErrorKind.RATE_LIMITED, ErrorKind.TRANSIENT)

    @property
    def fatal(self) -> bool:
        """The run cannot succeed until a person fixes something (key, model) or the quota resets."""
        return self in (ErrorKind.AUTH, ErrorKind.CONFIG, ErrorKind.QUOTA_EXHAUSTED)

    @property
    def stops_run(self) -> bool:
        """The supervisor ends the run: either a person must fix something, or the cause is an
        unexpected error (most likely a bug), which retrying or carrying on would only hide."""
        return self.fatal or self is ErrorKind.UNKNOWN


class ContentBlockedError(Exception):
    """The model returned nothing usable because a safety filter blocked the prompt or the answer."""


class LLMCallError(Exception):
    """A classified failure of an LLM call. This is what the agents catch."""

    def __init__(self, kind: ErrorKind, message: str, *, retry_after: Optional[float] = None, label: str = ""):
        prefix = f"{label}: " if label else ""
        super().__init__(f"[{kind.value}] {prefix}{message}")
        self.kind = kind
        self.retry_after = retry_after
        self.label = label


_DAILY_QUOTA = re.compile(r"per[\s_-]?day|daily", re.IGNORECASE)
_RETRY_DELAY = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", re.IGNORECASE)
_RETRY_IN = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)

_NETWORK_ERRORS = (httpx.TransportError, TimeoutError, ConnectionError) + (
    (requests.exceptions.Timeout, requests.exceptions.ConnectionError) if requests else ()
)


def _api_error_text(exc: genai_errors.APIError) -> str:
    return f"{getattr(exc, 'message', '')} {getattr(exc, 'details', '')}"


def _classify_api_error(exc: genai_errors.APIError) -> ErrorKind:
    code = getattr(exc, "code", None) or 0
    status = (getattr(exc, "status", None) or "").upper()
    text = _api_error_text(exc)

    if code == 429 or status == "RESOURCE_EXHAUSTED":
        return ErrorKind.QUOTA_EXHAUSTED if _DAILY_QUOTA.search(text) else ErrorKind.RATE_LIMITED
    # An invalid key arrives as HTTP 400 (reason API_KEY_INVALID), so check it before the generic 400
    if "API_KEY_INVALID" in text or code in (401, 403) or status in ("UNAUTHENTICATED", "PERMISSION_DENIED"):
        return ErrorKind.AUTH
    if code == 404 or status in ("NOT_FOUND", "FAILED_PRECONDITION"):
        return ErrorKind.CONFIG
    if code == 400 or status == "INVALID_ARGUMENT":
        return ErrorKind.BAD_REQUEST
    if code in (408, 499) or code >= 500 or status in ("INTERNAL", "UNAVAILABLE", "DEADLINE_EXCEEDED"):
        return ErrorKind.TRANSIENT
    return ErrorKind.UNKNOWN


def classify_error(exc: BaseException) -> ErrorKind:
    """Maps any exception raised around an LLM call to an ErrorKind."""
    if isinstance(exc, LLMCallError):
        return exc.kind
    if isinstance(exc, ContentBlockedError):
        return ErrorKind.CONTENT_BLOCKED
    if isinstance(exc, genai_errors.APIError):
        return _classify_api_error(exc)
    if isinstance(exc, _NETWORK_ERRORS):
        return ErrorKind.TRANSIENT
    if isinstance(exc, (json.JSONDecodeError, ValidationError)):
        return ErrorKind.BAD_OUTPUT
    return ErrorKind.UNKNOWN


def retry_after_seconds(exc: BaseException) -> Optional[float]:
    """The pause the API asked for (retryDelay / 'retry in 34s'), if the error carries one."""
    if isinstance(exc, LLMCallError):
        return exc.retry_after
    text = _api_error_text(exc) if isinstance(exc, genai_errors.APIError) else str(exc)
    for pattern in (_RETRY_DELAY, _RETRY_IN):
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return None


def to_llm_error(exc: BaseException, label: str = "") -> LLMCallError:
    """Wraps any exception into an LLMCallError (unchanged if it already is one)."""
    if isinstance(exc, LLMCallError):
        return exc
    return LLMCallError(classify_error(exc), str(exc)[:300], retry_after=retry_after_seconds(exc), label=label)


# What a person should be told for each kind of failure (shown in reports and API responses).
USER_MESSAGES = {
    ErrorKind.RATE_LIMITED: "Gemini is limiting the request rate right now. Please try again in a minute.",
    ErrorKind.QUOTA_EXHAUSTED: (
        "The Gemini quota is used up for today (it resets at midnight Pacific time). "
        "Try again later, or add models to LLM_MODEL_FALLBACKS."
    ),
    ErrorKind.TRANSIENT: "Gemini is temporarily unavailable. Please try again shortly.",
    ErrorKind.AUTH: "The Gemini API key is missing, invalid or not allowed to use this model. Check GEMINI_API_KEY.",
    ErrorKind.CONFIG: "The configured Gemini model is not available. Check LLM_MODEL and LLM_MODEL_FALLBACKS.",
    ErrorKind.BAD_REQUEST: "Gemini rejected the request, for example because the prompt was too large.",
    ErrorKind.CONTENT_BLOCKED: "Gemini's safety filter blocked this request or its answer.",
    ErrorKind.BAD_OUTPUT: "Gemini returned an answer that could not be used. Please try again.",
    ErrorKind.UNKNOWN: "An unexpected error occurred. The details were written to the server log.",
}


def user_message(kind: ErrorKind) -> str:
    return USER_MESSAGES[kind]


def record_error(state: dict, agent: str, error: LLMCallError) -> dict:
    """Notes a failed LLM call in state["errors"]. The supervisor reads this list to decide how to go on.

    Every entry gets a short id, written to the log next to the details, so that a person who is only
    shown "error ID 3f9a1c2e" can be matched with the traceback in the server log.
    """
    entry = {"id": uuid.uuid4().hex[:8], "agent": agent, "kind": error.kind.value, "message": str(error)}
    state.setdefault("errors", []).append(entry)
    logger.warning(f"[error {entry['id']}] {agent}: {error}")
    return entry


def describe_failure(entry: dict) -> dict:
    """What the person is told when a run ends because of this error entry (see record_error)."""
    try:
        kind = ErrorKind(entry.get("kind"))
    except ValueError:
        kind = ErrorKind.UNKNOWN
    message = user_message(kind)
    if kind is ErrorKind.UNKNOWN:
        message = f"{message} (error ID {entry.get('id', 'unknown')})"
    return {"agent": entry.get("agent", ""), "kind": kind.value, "message": message, "error_id": entry.get("id", "")}
