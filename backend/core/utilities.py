import os
import numpy as np
import faiss
import time
from pathlib import Path
from google import genai
from google.genai import types
from typing import Optional, Any, Union, List
from dotenv import load_dotenv, find_dotenv

# Load Environment Variables
load_dotenv(find_dotenv())

# --- ANSI Color Codes ---
C_RESET = "\033[0m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"  # Success/Done
C_YELLOW = "\033[93m" # Data flow/State update/DEBUG
C_BLUE = "\033[94m"  # Agent Info
C_MAGENTA = "\033[95m" # Router/Supervisor
C_CYAN = "\033[96m"  # Initialization/Setup
C_ACTION = "\033[38;5;208m" # Action/Start
C_PURPLE = "\033[95m"  # Reranking / Special Logic

# --- Global Configuration Constants ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MP_API_KEY = os.getenv("MP_API_KEY")

EMBED_MODEL = "models/gemini-embedding-001"
# Generation model; override with LLM_MODEL in .env.
LLM_MODEL = os.getenv("LLM_MODEL") or "models/gemini-3.6-flash"


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
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
        client = genai.Client()
        print(f"{C_CYAN} >> [INIT] Gemini client initialized successfully.{C_RESET}")
    except Exception:
        print(f"{C_RED} >> [FATAL] Failed to initialize Gemini client despite finding key.{C_RESET}")
else:
    print(f"{C_RED} >> [FATAL] GEMINI_API_KEY not found. LLM/Tool Agents will fail.{C_RESET}")


# ----------- Retry logic (single implementation, shared by LLM and embedding calls) -----------
def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


def call_with_retry(fn, *, max_retries: int = 5, initial_delay: float = 2.0, label: str = "Gemini call"):
    """
    Call fn() and retry with exponential backoff while Gemini answers with a rate-limit
    error (429 / RESOURCE_EXHAUSTED). Any other exception is raised immediately.
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if not _is_rate_limit_error(e):
                raise
            if attempt == max_retries:
                break
            print(f"{C_YELLOW}[RATE LIMIT] Hit 429 limit on {label}. Retrying in {delay}s... (Attempt {attempt}/{max_retries}){C_RESET}")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"Max retries exceeded for {label} due to rate limits.")


def generate_content_with_retry(*args, **kwargs):
    if client is None:
        raise RuntimeError("Gemini client not initialized.")
    return call_with_retry(
        lambda: client.models.generate_content(*args, **kwargs),
        label="generate_content",
    )

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
            print(f"{C_RED}[LOCAL EMBEDDING ERROR] {e}{C_RESET}")
            return zero
        return _normalize(vec)

    if client is None:
        print(f"{C_RED}[EMBEDDING ERROR] Gemini client not initialized.{C_RESET}")
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
        print(f"{C_RED}[EMBEDDING ERROR] {e}{C_RESET}")
        return zero
    return _normalize(vec)


def embedding_model(text: str) -> np.ndarray:
    return get_embedding(text)
