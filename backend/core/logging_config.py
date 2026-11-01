"""One-time logging setup for the whole backend (plain text, to stdout).

Every module gets its own logger with `logger = logging.getLogger(__name__)`; nothing else is needed.
The level comes from LOG_LEVEL in .env (DEBUG, INFO, WARNING, ERROR; default INFO).
"""
import logging
import os
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Libraries that log every HTTP request or model download at INFO would drown the pipeline output.
_NOISY_LIBRARIES = (
    "httpx", "httpcore", "urllib3", "google_genai", "sentence_transformers",
    "huggingface_hub", "filelock", "asyncio",
)

_configured = False


class _SkipProbeRequests(logging.Filter):
    """Drops uvicorn access-log lines for the Docker healthcheck and the docs page."""

    PATHS = {"/health", "/docs"}

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        # uvicorn access records carry (client, method, path, http_version, status_code)
        if isinstance(args, tuple) and len(args) >= 3 and args[2] in self.PATHS:
            return False
        return True


def setup_logging() -> None:
    """Configure the root logger once; calling it again does nothing."""
    global _configured
    if _configured:
        return
    _configured = True

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format=_FORMAT, datefmt=_DATE_FORMAT, stream=sys.stdout)

    for name in _NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").addFilter(_SkipProbeRequests())
