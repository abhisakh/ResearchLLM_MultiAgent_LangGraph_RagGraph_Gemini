"""How a run that could not finish is reported to an API client.

The supervisor leaves a `failure` dict in the final state (see errors.describe_failure). This module turns
it into an HTTP status, a JSON body and response headers. It knows nothing about FastAPI, so it is easy
to test on its own.
"""
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

try:  # the time zone database can be missing in slim containers; seconds_until_midnight_pacific copes with that
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None

from backend.core.errors import ErrorKind

# 503: the service cannot answer right now and it is not the caller's fault (try again later, or an operator
#      has to fix the configuration).
# 422: the content of the request was blocked by a safety filter, so a different request may work.
# 502: Gemini rejected or garbled what we exchanged with it.
# 500: an unexpected error, most likely a bug.
STATUS_FOR_KIND = {
    ErrorKind.QUOTA_EXHAUSTED: 503,
    ErrorKind.RATE_LIMITED: 503,
    ErrorKind.TRANSIENT: 503,
    ErrorKind.AUTH: 503,
    ErrorKind.CONFIG: 503,
    ErrorKind.CONTENT_BLOCKED: 422,
    ErrorKind.BAD_REQUEST: 502,
    ErrorKind.BAD_OUTPUT: 502,
    ErrorKind.UNKNOWN: 500,
}

# How long a client should wait before trying again (the Retry-After header). A daily quota is
# handled separately: it resets at midnight Pacific time.
SECONDS_TO_RETRY = {ErrorKind.RATE_LIMITED: 60, ErrorKind.TRANSIENT: 30}


def seconds_until_midnight_pacific(now: Optional[datetime] = None) -> int:
    """Seconds until the Gemini daily quota resets (midnight Pacific time)."""
    now = now or datetime.now(timezone.utc)
    try:
        pacific = ZoneInfo("America/Los_Angeles")
    except Exception:  # no time zone database here: assume standard time (UTC-8), at worst an hour off
        pacific = timezone(timedelta(hours=-8))
    local = now.astimezone(pacific)
    midnight = (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((midnight - local).total_seconds()))


def retry_after_for(kind: ErrorKind, now: Optional[datetime] = None) -> Optional[int]:
    if kind is ErrorKind.QUOTA_EXHAUSTED:
        return seconds_until_midnight_pacific(now)
    return SECONDS_TO_RETRY.get(kind)


def http_response_for(failure: Dict[str, str], now: Optional[datetime] = None) -> Tuple[int, Dict[str, str], Dict[str, str]]:
    """(status code, JSON detail, headers) for a failure dict as produced by errors.describe_failure."""
    try:
        kind = ErrorKind(failure.get("kind"))
    except ValueError:
        kind = ErrorKind.UNKNOWN

    detail = {
        "error": kind.value,
        "message": failure.get("message", ""),
        "error_id": failure.get("error_id", ""),
    }
    headers = {}
    retry_after = retry_after_for(kind, now)
    if retry_after:
        headers["Retry-After"] = str(retry_after)
    return STATUS_FOR_KIND[kind], detail, headers
