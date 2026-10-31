import os
import numpy as np
import faiss
import time
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
LLM_MODEL = "models/gemini-3.6-flash"


# RAG Utility Constants
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()

# Dynamic dimension configuration
if EMBEDDING_PROVIDER == "local":
    DIMENSION = 384  # all-MiniLM-L6-v2 fixed dimension
else:
    DIMENSION = 3072 # Gemini embedding-001 dimension
VECTOR_INDEX_PATH = "vector_index.faiss"
VECTOR_DATA_PATH = "vector_data.pkl"

# Entrez Configuration (Required by PubMedAgent)
ENTREZ_EMAIL = "your.email@example.com" # !!! REPLACE WITH REAL EMAIL !!!

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


# ----------- Retry Logic for LLM query -----------
def generate_content_with_retry(*args, **kwargs):
    if client is None:
        raise RuntimeError("Gemini client not initialized.")

    max_retries = 5
    delay = 2.0

    for attempt in range(max_retries):
        try:
            return client.models.generate_content(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"{C_YELLOW}[RATE LIMIT] Hit 429 limit. Retrying in {delay}s... (Attempt {attempt+1}/{max_retries}){C_RESET}")
                time.sleep(delay)
                delay *= 2
            else:
                raise e
    raise RuntimeError("Max retries exceeded for generate_content due to rate limits.")

# --- Shared Utility Functions ---

def get_embedding(text: str) -> np.ndarray:
    if client is None:
        print(f"{C_RED}[EMBEDDING ERROR] Gemini client not initialized.{C_RESET}")
        return np.zeros(DIMENSION, dtype=np.float32)

    max_retries = 5
    delay = 2.0  # initial delay in seconds

    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=EMBED_MODEL,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=DIMENSION)
            )
            vec = np.array(response.embeddings[0].values, dtype=np.float32)

            # Normalize vector
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"{C_YELLOW}[RATE LIMIT] Hit 429 limit. Retrying in {delay}s... (Attempt {attempt+1}/{max_retries}){C_RESET}")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"{C_RED}[EMBEDDING ERROR] {e}{C_RESET}")
                break

    return np.zeros(DIMENSION, dtype=np.float32)

def embedding_model(text: str) -> np.ndarray:
    return get_embedding(text)
