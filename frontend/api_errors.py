"""Turns a failed /research-chat response into a sentence for the person using the UI.

Kept free of Streamlit imports so it can be unit-tested.
"""


def wait_text(seconds: int) -> str:
    """'45 seconds', '10 minutes', '12 hours': a rough, readable wait time."""
    if seconds < 90:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    if seconds < 7200:
        return f"{round(seconds / 60)} minutes"
    return f"{round(seconds / 3600)} hours"


def describe_api_error(response) -> str:
    """The backend's message for a failed run, plus how long to wait if it sent a Retry-After header."""
    try:
        body = response.json()
    except ValueError:
        body = {}
    detail = body.get("detail") if isinstance(body, dict) else None

    if isinstance(detail, dict) and detail.get("message"):
        text = str(detail["message"])
    else:
        text = f"The backend could not complete the request (status {response.status_code})."

    retry_after = str(response.headers.get("Retry-After", ""))
    if retry_after.isdigit():
        text += f" You can try again in about {wait_text(int(retry_after))}."
    return text
