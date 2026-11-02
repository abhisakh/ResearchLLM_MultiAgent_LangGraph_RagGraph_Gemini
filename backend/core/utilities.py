import json
import logging
import os
import numpy as np
import threading
import time
from collections import deque
from pathlib import Path
from google import genai
from google.genai import types
from typing import Optional
from dotenv import load_dotenv

from .errors import ErrorKind, LLMCallError, to_llm_error
from .logging_config import setup_logging

logger = logging.getLogger(__name__)

# Repository root (this file is backend/core/utilities.py, so two levels up).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load <repo root>/.env explicitly, so it does not matter which folder the app is started from.
# In Docker there is no .env file (the variables come from the container environment) and a
# missing file is simply ignored. Variables already set in the environment are never overridden.
load_dotenv(_PROJECT_ROOT / ".env")

# Configure logging right after .env is loaded (LOG_LEVEL lives there) and before anything logs.
setup_logging()

# --- Global Configuration Constants ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MP_API_KEY = os.getenv("MP_API_KEY")

EMBED_MODEL = "models/gemini-embedding-001"
# Generation model; override with LLM_MODEL in .env.
LLM_MODEL = os.getenv("LLM_MODEL") or "models/gemini-3.6-flash"

# When the daily quota of LLM_MODEL is used up, these models are tried in order (comma-separated in .env).
# Quotas are counted per model, so a lighter model still has its own allowance.
LLM_MODEL_FALLBACKS = [m.strip() for m in (os.getenv("LLM_MODEL_FALLBACKS") or "").split(",") if m.strip()]


def _requests_per_minute(raw: Optional[str]) -> int:
    """LLM_REQUESTS_PER_MINUTE from .env: a whole number; 0, empty or invalid means no pacing."""
    try:
        return max(0, int((raw or "").strip() or 0))
    except ValueError:
        logger.warning(f'LLM_REQUESTS_PER_MINUTE={raw!r} is not a whole number; request pacing is off.')
        return 0


# Most Gemini requests sent per minute and model (see RequestPacer). 0 = no pacing. Set it to the limit AI Studio
# shows for your model: a free-tier limit of 5 is lower than the number of calls one question needs.
LLM_REQUESTS_PER_MINUTE = _requests_per_minute(os.getenv("LLM_REQUESTS_PER_MINUTE"))


# RAG Utility Constants
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()

# Dynamic dimension configuration
if EMBEDDING_PROVIDER == "local":
    DIMENSION = 384  # all-MiniLM-L6-v2 fixed dimension
else:
    DIMENSION = 3072 # Gemini embedding-001 dimension

# Where the FAISS index and its pickled metadata live.
# Defaults to the repository root (same place as before, so nothing moves for existing setups),
# and no longer depends on the directory the app is started from.
# Override with VECTOR_DATA_DIR (e.g. a mounted volume in Docker).
VECTOR_DATA_DIR = Path(os.getenv("VECTOR_DATA_DIR") or _PROJECT_ROOT).expanduser().resolve()
VECTOR_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Kept as str: faiss.read_index / faiss.write_index do not accept Path objects.
VECTOR_INDEX_PATH = str(VECTOR_DATA_DIR / "vector_index.faiss")
VECTOR_DATA_PATH = str(VECTOR_DATA_DIR / "vector_data.pkl")

# Contact emails for the public APIs (NCBI asks for one; OpenAlex uses it for its "polite pool").
# Set them in .env. Empty by default: the agents log a warning and still run.
ENTREZ_EMAIL = (os.getenv("ENTREZ_EMAIL") or "").strip()
OPENALEX_EMAIL = (os.getenv("OPENALEX_EMAIL") or "").strip()

# --- Gemini Client Initialization ---
client: Optional[genai.Client] = None
if GEMINI_API_KEY:
    try:
        # Pass the key explicitly. Without api_key, google-genai reads GOOGLE_API_KEY first and GEMINI_API_KEY only
        # second, so a GOOGLE_API_KEY left in the shell would silently override the key in .env.
        client = genai.Client(api_key=GEMINI_API_KEY)
        # Only the last 4 characters: enough to tell which key is in use, not enough to use it.
        logger.info(f'>> [INIT] Gemini client initialized (key ending ...{GEMINI_API_KEY[-4:]}).')
    except Exception:
        logger.error('>> [FATAL] Failed to initialize Gemini client despite finding key.')
else:
    logger.error('>> [FATAL] GEMINI_API_KEY not found. LLM/Tool Agents will fail.')


# ----------- Retry logic (single implementation, shared by LLM and embedding calls) -----------
# How each kind of failure is handled (the kinds are defined in errors.py):
#   rate_limited     wait the pause Gemini asks for, then retry; give up once the waits would add up to
#                    more than MAX_TOTAL_WAIT_SECONDS (the pipeline must not stall for minutes)
#   transient        retry with exponential backoff (500/503, timeouts, dropped connections)
#   quota_exhausted  daily quota used up: waiting is pointless, raise at once (the caller may switch model)
#   auth / config    a person must fix the key or the model name: raise at once
#   bad_request, content_blocked, bad_output, unknown
#                    repeating the same call cannot help (an unknown error is most likely a bug): raise at once
# Every failure leaves call_with_retry as an LLMCallError that carries its kind.
MAX_TOTAL_WAIT_SECONDS = 60.0
RETRY_PADDING_SECONDS = 1.0   # added to the pause Gemini asks for, so the retry does not land a moment too early


def call_with_retry(fn, *, max_retries: int = 3, initial_delay: float = 2.0,
                    max_total_wait: float = MAX_TOTAL_WAIT_SECONDS, label: str = "Gemini call"):
    """Call fn(), retrying only the failures that waiting can fix. Raises LLMCallError otherwise."""
    waited = 0.0
    delay = initial_delay
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as exc:
            error = to_llm_error(exc, label=label)

            if not error.kind.retryable:
                if error.kind is ErrorKind.UNKNOWN:
                    # most likely a bug in our own code: keep the full traceback in the log
                    logger.error(f'[{label}] Unexpected error, not retrying: {exc}', exc_info=exc)
                else:
                    logger.warning(f'[{label}] {error.kind.value} error, not retrying: {exc}')
                raise _chained(error, exc)

            if attempt > max_retries:
                logger.warning(f'[{label}] Still failing after {max_retries} retries ({error.kind.value}). Giving up.')
                raise _chained(error, exc)

            if error.kind is ErrorKind.RATE_LIMITED and error.retry_after is not None:
                pause = error.retry_after + RETRY_PADDING_SECONDS
            else:
                pause = delay

            if waited + pause > max_total_wait:
                logger.warning(f'[{label}] Waiting {pause:.0f}s more would exceed the {max_total_wait:.0f}s budget. Giving up.')
                raise _chained(error, exc)

            logger.info(f'[{label}] {error.kind.value}: retrying in {pause:.0f}s (retry {attempt}/{max_retries})')
            time.sleep(pause)
            waited += pause
            delay *= 2


def _chained(error: LLMCallError, cause: BaseException) -> LLMCallError:
    """The classified error, with the original exception kept as __cause__ (traceback stays in the log)."""
    if error is not cause:
        error.__cause__ = cause
    return error


class RequestPacer:
    """Keeps the number of requests per minute under a limit by waiting BEFORE a request is sent.

    The free tier of Gemini allows only a handful of requests per minute and answers with HTTP 429 when they
    are exceeded, so a pipeline that makes 6 or more calls per question would fail by design. Pacing turns that
    failure into a short wait. The limit is counted per model, as Google counts it. It only knows this process:
    another program using the same key at the same time is invisible to it.
    """

    WINDOW_SECONDS = 60.0
    MARGIN_SECONDS = 1.0   # clocks and network delay differ a little from Google's, so stay slightly under

    def __init__(self, per_minute: int, clock=time.monotonic, sleep=None):
        self.per_minute = per_minute
        self._clock = clock
        self._sleep = sleep or (lambda seconds: time.sleep(seconds))
        self._lock = threading.Lock()
        # model -> send times of its last `per_minute` requests (reserved ahead, so they may lie in the future)
        self._slots = {}

    def wait_for_turn(self, model: str = "") -> float:
        """Blocks until one more request fits under the limit. Returns the seconds it waited (0 if none)."""
        if self.per_minute <= 0:
            return 0.0
        with self._lock:
            now = self._clock()
            slots = self._slots.setdefault(model or "", deque(maxlen=self.per_minute))
            if len(slots) == self.per_minute:
                earliest = slots[0] + self.WINDOW_SECONDS + self.MARGIN_SECONDS
            else:
                earliest = now
            send_at = max(now, earliest)
            slots.append(send_at)   # reserved now, so concurrent callers queue up behind each other
        delay = send_at - now
        if delay > 0:
            self._sleep(delay)
        return delay


LLM_PACER = RequestPacer(LLM_REQUESTS_PER_MINUTE)


class LLMUsage:
    """Running totals of Gemini calls and tokens, so the benchmark can report the cost of a query.

    The benchmark reads a snapshot before and after each query. The API server does not use the totals
    (with concurrent requests they would mix), they are only counted. Token counts come from the
    response's usage_metadata; a response without it (or a fake in a test) counts as one call, zero tokens.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.calls = self.prompt_tokens = self.output_tokens = self.thinking_tokens = 0

    def add(self, response) -> None:
        meta = getattr(response, "usage_metadata", None)
        with self._lock:
            self.calls += 1
            self.prompt_tokens += int(getattr(meta, "prompt_token_count", 0) or 0)
            self.output_tokens += int(getattr(meta, "candidates_token_count", 0) or 0)
            self.thinking_tokens += int(getattr(meta, "thoughts_token_count", 0) or 0)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "output_tokens": self.output_tokens,
                "thinking_tokens": self.thinking_tokens,
            }


LLM_USAGE = LLMUsage()


def generate_content_with_retry(*args, **kwargs):
    """client.models.generate_content with retry. If the daily quota of the requested model is used
    up, the models in LLM_MODEL_FALLBACKS are tried in turn (quotas are counted per model)."""
    if client is None:
        raise LLMCallError(ErrorKind.AUTH, "Gemini client is not initialised: set GEMINI_API_KEY in .env.",
                           label="generate_content")

    primary = kwargs.get("model")
    models = [primary] + [m for m in LLM_MODEL_FALLBACKS if m != primary] if primary else [None]
    for position, model in enumerate(models):
        call_kwargs = dict(kwargs, model=model) if model else kwargs
        label = f"generate_content[{model}]" if model else "generate_content"
        try:
            def send():
                # Paced before every attempt, retries included: a retry is one more request for the limit too.
                waited = LLM_PACER.wait_for_turn(model or "")
                if waited >= 1:
                    logger.info(f'[{label}] Waited {waited:.0f}s to stay under {LLM_PACER.per_minute} requests per minute.')
                return client.models.generate_content(*args, **call_kwargs)

            response = call_with_retry(send, label=label)
            LLM_USAGE.add(response)
            return response
        except LLMCallError as error:
            is_last = position == len(models) - 1
            if error.kind is ErrorKind.QUOTA_EXHAUSTED and not is_last:
                logger.warning(f'[{label}] Daily quota used up. Switching to {models[position + 1]}.')
                continue
            raise


# ----------- Reading the model's answer (typed errors instead of None / bare exceptions) -----------
# Gemini reports a blocked prompt or answer inside the response object instead of raising.
_BLOCKED_FINISH_REASONS = {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"}


def _raise_if_blocked(response, label: str) -> None:
    reason = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
    if reason:
        raise LLMCallError(ErrorKind.CONTENT_BLOCKED,
                           f"The prompt was blocked ({getattr(reason, 'name', reason)}).", label=label)
    for candidate in getattr(response, "candidates", None) or []:
        finish = getattr(candidate, "finish_reason", None)
        if finish and getattr(finish, "name", str(finish)).upper() in _BLOCKED_FINISH_REASONS:
            raise LLMCallError(ErrorKind.CONTENT_BLOCKED,
                               f"The answer was blocked ({getattr(finish, 'name', finish)}).", label=label)


def _text_parts(response) -> list:
    """The answer's text parts (thought parts excluded), or [] for a response without candidates (a test fake)."""
    try:
        parts = response.candidates[0].content.parts or []
    except (AttributeError, IndexError, TypeError):
        return []
    return [p.text for p in parts if isinstance(getattr(p, "text", None), str) and not getattr(p, "thought", False)]


def _warn_if_parts_touch(parts: list, label: str) -> None:
    """Diagnostic for words that arrive glued together ("distinctthermodynamic").

    response.text joins the answer's parts with no separator. If one part ends and the next begins with no
    whitespace between them, the joined text runs the two words together. A word that was merely split in
    the middle is flagged too, so read the printed context before drawing a conclusion.
    """
    if len(parts) > 1:
        logger.info(f'[{label}] The answer arrived in {len(parts)} text parts.')
    touching = [
        repr(left[-15:] + "|" + right[:15])
        for left, right in zip(parts, parts[1:])
        if left and right and not left[-1].isspace() and not right[0].isspace()
    ]
    if touching:
        logger.warning(
            f'[{label}] {len(touching)} part boundary(ies) have no whitespace between the parts '
            f'(response.text joins them as they are): {", ".join(touching[:5])}'
        )


def response_text(response, label: str = "Gemini call") -> str:
    """The text of a response. Raises content_blocked if a safety filter fired, bad_output if it is empty."""
    _raise_if_blocked(response, label)
    _warn_if_parts_touch(_text_parts(response), label)
    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise LLMCallError(ErrorKind.BAD_OUTPUT, "The model returned an empty answer.", label=label)
    return text


def response_json(response, label: str = "Gemini call") -> dict:
    """The JSON object in a response. Raises bad_output if it is not valid JSON or not an object."""
    text = response_text(response, label)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _chained(LLMCallError(ErrorKind.BAD_OUTPUT, f"The answer is not valid JSON: {exc}", label=label), exc)
    if not isinstance(data, dict):
        raise LLMCallError(ErrorKind.BAD_OUTPUT, "The answer is JSON but not an object.", label=label)
    return data


def retry_on_bad_output(fn, *, label: str = "Gemini call", retries: int = 1):
    """Call fn(); if the model answered but not usably (bad_output), ask again `retries` times.
    Whatever fails is raised as an LLMCallError (the traceback of an unknown error is logged here)."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:
            error = to_llm_error(exc, label=label)
            if error.kind is ErrorKind.BAD_OUTPUT and attempt < retries:
                logger.warning(f'[{label}] Unusable answer ({error}). Asking once more.')
                continue
            if error.kind is ErrorKind.UNKNOWN and not isinstance(exc, LLMCallError):
                logger.error(f'[{label}] Unexpected error: {exc}', exc_info=exc)
            raise _chained(error, exc)


def llm_text(call, *, label: str = "Gemini call") -> str:
    """call() makes the Gemini request; returns the answer text (one retry if it was unusable)."""
    return retry_on_bad_output(lambda: response_text(call(), label), label=label)


def llm_json(call, *, label: str = "Gemini call") -> dict:
    """call() makes the Gemini request; returns the answer as a dict (one retry if it was unusable)."""
    return retry_on_bad_output(lambda: response_json(call(), label), label=label)

# --- Shared Utility Functions ---

# Local embedder (sentence-transformers) is loaded lazily, so importing this module stays cheap
# and the fast CI job does not need sentence-transformers installed.
_LOCAL_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_local_model = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(_LOCAL_EMBED_MODEL_NAME)
    return _local_model


def preload_embedding_model() -> None:
    """Load the local embedder now (app startup) so the first request does not pay for it."""
    if EMBEDDING_PROVIDER == "local":
        _get_local_model()


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalise so that inner product equals cosine similarity."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def get_embedding(text: str) -> np.ndarray:
    """
    The one embedding function. Routes to the local sentence-transformers model or to the
    Gemini API depending on EMBEDDING_PROVIDER. Returns a normalised float32 vector of
    length DIMENSION, or a zero vector if embedding failed.
    """
    zero = np.zeros(DIMENSION, dtype=np.float32)

    if EMBEDDING_PROVIDER == "local":
        try:
            vec = _get_local_model().encode(text, convert_to_numpy=True).astype(np.float32)
        except Exception as e:
            logger.error(f'[LOCAL EMBEDDING ERROR] {e}')
            return zero
        return _normalize(vec)

    if client is None:
        logger.error('[EMBEDDING ERROR] Gemini client not initialized.')
        return zero

    try:
        response = call_with_retry(
            lambda: client.models.embed_content(
                model=EMBED_MODEL,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=DIMENSION),
            ),
            label="embed_content",
        )
        vec = np.array(response.embeddings[0].values, dtype=np.float32)
    except Exception as e:
        logger.error(f'[EMBEDDING ERROR] {e}')
        return zero
    return _normalize(vec)


def embedding_model(text: str) -> np.ndarray:
    return get_embedding(text)
