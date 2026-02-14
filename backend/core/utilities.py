import os
import numpy as np
import faiss
from openai import OpenAI
from typing import Optional, Any
from dotenv import load_dotenv

# Load Environment Variables (Ensure this runs once at the top level)
# We load them here so other modules can import the API keys/settings
load_dotenv()

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
OPENAI_API_KEY = os.getenv("GPT_5_API_KEY")
MP_API_KEY = os.getenv("MP_API_KEY")
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini" # Using the reliable GPT-4 model from the combined code

# RAG Utility Constants
DIMENSION = 1536
VECTOR_INDEX_PATH = "vector_index.faiss"
VECTOR_DATA_PATH = "vector_data.pkl"

# Entrez Configuration (Required by PubMedAgent)
ENTREZ_EMAIL = "your.email@example.com" # !!! REPLACE WITH REAL EMAIL !!!

# --- OpenAI Client Initialization ---
client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        print(f"{C_CYAN} >> [INIT] OpenAI client initialized successfully.{C_RESET}")
    except Exception:
        print(f"{C_RED} >> [FATAL] Failed to initialize OpenAI client despite finding key.{C_RESET}")
else:
    print(f"{C_RED} >> [FATAL] GPT_API_KEY not found. LLM/Tool Agents will fail.{C_RESET}")

# --- Shared Utility Function ---

def get_embedding(text: str) -> np.ndarray:
    """
    Function to get embedding (requires global 'client' and constants).
    """
    # Defensive check against API key failure
    if client is None:
        print(f"{C_RED}[EMBEDDING ERROR] OpenAI client not initialized. Returning zeros.{C_RESET}")
        return np.zeros(DIMENSION, dtype=np.float32)
    try:
        # Use the global client and model
        response = client.embeddings.create(input=text, model=EMBED_MODEL)
        return np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as e:
        print(f"{C_RED}[EMBEDDING ERROR] Failed to get embedding for text: '{text[:20]}...': {e}{C_RESET}")
        return np.zeros(DIMENSION, dtype=np.float32)