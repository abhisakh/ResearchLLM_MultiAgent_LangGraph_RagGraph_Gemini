import os
import pickle
import time
import numpy as np
import faiss
from typing import List, Tuple, Optional, Dict, Any
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables dynamically across project directories
load_dotenv(find_dotenv())

# Import utilities for constants, client, and embedding function
from .utilities import (
    get_embedding, C_RESET, C_CYAN, C_RED, C_BLUE, C_GREEN, C_MAGENTA, C_YELLOW,
    DIMENSION, VECTOR_INDEX_PATH, VECTOR_DATA_PATH, EMBED_MODEL, client
)

# ==============================================================================
# HYBRID EMBEDDING PROVIDER SETUP
# ------------------------------------------------------------------------------
# Configurable via environment variable: EMBEDDING_PROVIDER ("local" or "gemini")
# Defaulting to "local" to avoid Gemini 429 rate limit errors when free quota exhausts.
# ==============================================================================
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()

# Initialize local model if selected
_local_model = None
if EMBEDDING_PROVIDER == "local":
    # Using a fast, lightweight sentence transformer model
    _local_model = SentenceTransformer("all-MiniLM-L6-v2")


# ==============================================================================
# REASON FOR UPDATING THE EMBEDDING FUNCTION:
# ------------------------------------------------------------------------------
# OLD GEMINI-ONLY CODE (Commented out below):
# def _get_embedding(text: str) -> np.ndarray:
#     if client is None:
#         return np.zeros(DIMENSION, dtype=np.float32)
#     try:
#         response = client.models.embed_content(
#             model=EMBED_MODEL,
#             contents=text
#         )
#         emb = np.array(response.embeddings[0].values, dtype=np.float32)
#         faiss.normalize_L2(emb.reshape(1, -1))
#         return emb
#     except Exception as e:
#         print(f"{C_RED}[EMBEDDING ERROR] Failed to get embedding: {e}{C_RESET}")
#         return np.zeros(DIMENSION, dtype=np.float32)
#
# WHY IT WAS UPDATED:
# Continuous Gemini API calls trigger 429 RESOURCE_EXHAUSTED rate limits.
# This hybrid setup lets you switch to a local model instantly via config
# while retaining Gemini as an alternative option.
# ==============================================================================

def _get_embedding(text: str, max_retries: int = 3, initial_delay: float = 2.0) -> np.ndarray:
    """
    Hybrid embedding generator that routes to either Local (SentenceTransformers)
    or Gemini API based on the EMBEDDING_PROVIDER environment variable.
    """
    global EMBEDDING_PROVIDER, _local_model

    # 1. LOCAL EMBEDDER PATH
    if EMBEDDING_PROVIDER == "local":
        try:
            if _local_model is None:
                _local_model = SentenceTransformer("all-MiniLM-L6-v2")

            emb = _local_model.encode(text, convert_to_numpy=True).astype(np.float32)

            # MANDATORY FOR COSINE SIMILARITY: Normalize the vector
            faiss.normalize_L2(emb.reshape(1, -1))
            return emb
        except Exception as e:
            print(f"{C_RED}[LOCAL EMBEDDING ERROR] {e}{C_RESET}")
            return np.zeros(DIMENSION, dtype=np.float32)

    # 2. GEMINI API EMBEDDER PATH (with retry & backoff)
    if client is None:
        return np.zeros(DIMENSION, dtype=np.float32)

    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=EMBED_MODEL,
                contents=text
            )
            emb = np.array(response.embeddings[0].values, dtype=np.float32)

            # MANDATORY FOR COSINE SIMILARITY: Normalize the vector
            faiss.normalize_L2(emb.reshape(1, -1))
            return emb
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < max_retries - 1:
                    print(f"{C_YELLOW}[RATE LIMIT] Gemini quota exceeded. Retrying in {delay}s (Attempt {attempt + 1}/{max_retries})...{C_RESET}")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    continue

            print(f"{C_RED}[GEMBEDDING ERROR] Failed to get embedding: {e}{C_RESET}")
            break

    return np.zeros(DIMENSION, dtype=np.float32)


class VectorDBWrapper:
    def __init__(self, dimension: int = DIMENSION):
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.text_store: List[Dict[str, Any]] = []

        # Local model does not strictly require the Gemini client API key to run
        if EMBEDDING_PROVIDER == "local" or client is not None:
            self._initialize_db()
        else:
            print(f"{C_RED}[VectorDB] Skipping initialization due to missing API key and provider requirements.{C_RESET}")

    def _initialize_db(self):
        if os.path.exists(VECTOR_INDEX_PATH) and os.path.exists(VECTOR_DATA_PATH):
            try:
                self.index = faiss.read_index(VECTOR_INDEX_PATH)
                with open(VECTOR_DATA_PATH, "rb") as f:
                    self.text_store = pickle.load(f)
                print(f"{C_CYAN}[VectorDB] Loaded existing Cosine DB ({EMBEDDING_PROVIDER.upper()} mode). Chunks: {len(self.text_store)}{C_RESET}")
            except Exception:
                print(f"{C_RED}[VectorDB] Failed to load DB. Creating new one.{C_RESET}")
                self._create_new_db()
        else:
            self._create_new_db()

    def _create_new_db(self):
        # IndexFlatIP for Inner Product (Cosine Similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.text_store = []
        self._save_db()
        print(f"{C_CYAN}[VectorDB] Created new IndexFlatIP DB ({EMBEDDING_PROVIDER.upper()} mode).{C_RESET}")

    def reset_db(self):
        print(f"{C_RED}[VectorDB] Starting database reset...{C_RESET}")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.text_store = []

        if os.path.exists(VECTOR_INDEX_PATH):
            os.remove(VECTOR_INDEX_PATH)
        if os.path.exists(VECTOR_DATA_PATH):
            os.remove(VECTOR_DATA_PATH)

        self._save_db()
        print(f"{C_GREEN}[VectorDB] Database reset complete.{C_RESET}")

    def _save_db(self):
        if self.index is not None:
            faiss.write_index(self.index, VECTOR_INDEX_PATH)
            with open(VECTOR_DATA_PATH, "wb") as f:
                pickle.dump(self.text_store, f)

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        if self.index is None:
            return

        existing_texts = {c.get("text") for c in self.text_store}
        new_embeddings = []
        new_chunks = []

        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if text and text not in existing_texts:
                emb = _get_embedding(text)  # Uses active provider
                if not np.all(emb == 0):
                    new_embeddings.append(emb)
                    new_chunks.append(chunk)

                # Only sleep if using API to avoid burst limits; local needs no delay
                if EMBEDDING_PROVIDER == "gemini":
                    time.sleep(0.1)

        if new_embeddings:
            embeddings_array = np.array(new_embeddings, dtype=np.float32)
            self.index.add(embeddings_array)
            self.text_store.extend(new_chunks)
            self._save_db()
            print(f"{C_BLUE}[VectorDB] Added {len(new_chunks)} new chunks.{C_RESET}")

    def search(self, query: str, k: int = 20) -> List[Tuple[Dict[str, Any], float]]:
        if self.index is None or self.index.ntotal == 0:
            return []

        query_embedding = _get_embedding(query).reshape(1, -1)

        if np.all(query_embedding == 0):
            print(f"{C_RED}[VectorDB ERROR] Invalid query embedding.{C_RESET}")
            return []

        k_actual = min(k, self.index.ntotal)

        query_matrix = np.asarray(query_embedding, dtype=np.float32)
        D, I = self.index.search(query_matrix, k_actual)

        results = []
        for score, idx in zip(D[0], I[0]):
            results.append((self.text_store[idx], float(score)))

        return results

# ==================================================================================================
# INTEGRATED TEST BLOCK
# ==================================================================================================
if __name__ == "__main__":
    TEST_CHUNKS = [
        {"text": "CsSnI3 is a lead-free perovskite material being studied as a less toxic alternative to lead-halide perovskites."},
        {"text": "Computational data predicts that the cubic phase of CsSnI3 has a theoretical bandgap around 1.3 eV, which is near-optimal for solar cells."},
        {"text": "Experimental synthesis often struggles with the rapid oxidation of Sn2+ to Sn4+, leading to poor long-term stability."},
        {"text": "The instability issue is partially mitigated by using excess SnF2 during the solution processing method."},
        {"text": "An irrelevant chunk about quantum computing algorithms."}
    ]

    TEST_QUERIES = [
        "Stability challenges in CsSnI3 solar cells",
        "Bandgap of lead-free perovskite CsSnI3",
        "Irrelevant topic not in the chunks",
    ]

    print(f"{C_CYAN}*** STARTING VectorDBWrapper ISOLATED TEST ***{C_RESET}")

    # --- 1. SETUP: Initialize and Reset ---
    print(f"\n{C_MAGENTA}--- 1. Testing Initialization and Reset ---{C_RESET}")
    db = VectorDBWrapper()

    if client is None or db.index is None:
        print(f"{C_RED}[TEST FAILURE] Missing client or uninitialized index. Check GEMINI_API_KEY in .env.{C_RESET}")
        exit(1)

    db.reset_db()

    if db.index.ntotal == 0 and len(db.text_store) == 0:
        print(f"{C_GREEN}[TEST SUCCESS] DB reset and initialized correctly (0 chunks).{C_RESET}")
    else:
        print(f"{C_RED}[TEST FAILURE] DB reset failed.{C_RESET}")
        exit(1)

    # --- 2. FUNCTIONALITY: Add Chunks ---
    print(f"\n{C_MAGENTA}--- 2. Testing Chunk Addition and Persistence ---{C_RESET}")
    db.add_chunks(TEST_CHUNKS)

    if db.index.ntotal == len(TEST_CHUNKS):
        print(f"{C_GREEN}[TEST SUCCESS] Added {len(TEST_CHUNKS)} chunks successfully to the index.{C_RESET}")
    else:
        print(f"{C_RED}[TEST FAILURE] Expected {len(TEST_CHUNKS)} chunks, found {db.index.ntotal}.{C_RESET}")

    db_reloaded = VectorDBWrapper()
    if db_reloaded.index and db_reloaded.index.ntotal == len(TEST_CHUNKS):
        print(f"{C_GREEN}[TEST SUCCESS] DB persistence verified (reloaded with {len(TEST_CHUNKS)} chunks).{C_RESET}")
    else:
        print(f"{C_RED}[TEST FAILURE] DB persistence failed.{C_RESET}")

    # --- 3. FUNCTIONALITY: Search ---
    print(f"\n{C_MAGENTA}--- 3. Testing Semantic Search Accuracy ---{C_RESET}")
    for query_text in TEST_QUERIES:
        print(f"{C_BLUE}Searching for: '{query_text}'{C_RESET}")
        results = db_reloaded.search(query_text, k=3)

        if not results:
            print(f"{C_RED}[TEST FAILURE] Search returned 0 results for query: '{query_text}'.{C_RESET}")
            continue

        top_chunk_dict, top_score = results[0]
        top_text = top_chunk_dict.get("text", "")

        if "Stability" in query_text:
            expected_keyword = "oxidation"
        elif "Bandgap" in query_text:
            expected_keyword = "cubic"
        else:
            expected_keyword = None

        print(f"  Top Result Score (Cosine Similarity): {top_score:.4f}")

        if expected_keyword and expected_keyword.lower() in top_text.lower():
            print(f"{C_GREEN}[TEST SUCCESS] Semantic Match: Top result contains '{expected_keyword}'. (Score: {top_score:.4f}){C_RESET}")
        elif not expected_keyword and top_score < 0.6:
            print(f"{C_GREEN}[TEST SUCCESS] Irrelevance Check: Top result similarity score is low (<0.6).{C_RESET}")
        else:
            print(f"{C_YELLOW}[TEST WARNING] Semantic Match was weak or unexpected. Top chunk: '{top_text[:50]}...'{C_RESET}")

    # --- 4. CLEANUP ---
    print(f"\n{C_MAGENTA}--- 4. Testing Cleanup ---{C_RESET}")
    db_reloaded.reset_db()
    db_final = VectorDBWrapper()
    if db_final.index and db_final.index.ntotal == 0:
        print(f"{C_GREEN}[TEST SUCCESS] Final cleanup (reset_db) successful.{C_RESET}")
    else:
        print(f"{C_RED}[TEST FAILURE] Final reset failed.{C_RESET}")

    print(f"\n{C_CYAN}*** VectorDBWrapper ISOLATED TEST COMPLETE ***{C_RESET}")

# ------- GPT-5 Vector Database Wrapper (Cosine Similarity) -------
# import os
# import pickle
# import numpy as np
# import faiss
# from typing import List, Tuple, Optional, Dict, Any
# # Import utilities for constants, client, and embedding function
# from .utilities import (
#     get_embedding, C_RESET, C_CYAN, C_RED, C_BLUE, C_GREEN, C_MAGENTA, C_YELLOW,
#     DIMENSION, VECTOR_INDEX_PATH, VECTOR_DATA_PATH, EMBED_MODEL, client
# )


# def _get_embedding(text: str) -> np.ndarray:
#     if client is None:
#         return np.zeros(DIMENSION, dtype=np.float32)
#     try:
#         response = client.embeddings.create(
#             input=text,
#             model=EMBED_MODEL
#         )
#         emb = np.array(response.data[0].embedding, dtype=np.float32)

#         # MANDATORY FOR COSINE SIMILARITY: Normalize the vector
#         faiss.normalize_L2(emb.reshape(1, -1))
#         return emb
#     except Exception as e:
#         print(f"{C_RED}[EMBEDDING ERROR] Failed to get embedding: {e}{C_RESET}")
#         return np.zeros(DIMENSION, dtype=np.float32)

# class VectorDBWrapper:
#     def __init__(self, dimension: int = DIMENSION):
#         self.dimension = dimension
#         self.index: Optional[faiss.Index] = None
#         self.text_store: List[Dict[str, Any]] = []

#         if client is not None:
#             self._initialize_db()
#         else:
#             print(f"{C_RED}[VectorDB] Skipping initialization due to missing API key.{C_RESET}")

#     def _initialize_db(self):
#         if os.path.exists(VECTOR_INDEX_PATH) and os.path.exists(VECTOR_DATA_PATH):
#             try:
#                 self.index = faiss.read_index(VECTOR_INDEX_PATH)
#                 with open(VECTOR_DATA_PATH, "rb") as f:
#                     self.text_store = pickle.load(f)
#                 print(f"{C_CYAN}[VectorDB] Loaded existing Cosine DB. Chunks: {len(self.text_store)}{C_RESET}")
#             except Exception:
#                 print(f"{C_RED}[VectorDB] Failed to load DB. Creating new one.{C_RESET}")
#                 self._create_new_db()
#         else:
#             self._create_new_db()

#     def _create_new_db(self):
#         # UPGRADE: Using IndexFlatIP for Inner Product (Cosine Similarity)
#         self.index = faiss.IndexFlatIP(self.dimension)
#         self.text_store = []
#         self._save_db()
#         print(f"{C_CYAN}[VectorDB] Created new IndexFlatIP DB (Cosine Similarity).{C_RESET}")

#     def reset_db(self):
#         print(f"{C_RED}[VectorDB] Starting database reset...{C_RESET}")
#         # Ensure reset also uses the IP index
#         self.index = faiss.IndexFlatIP(self.dimension)
#         self.text_store = []

#         if os.path.exists(VECTOR_INDEX_PATH):
#             os.remove(VECTOR_INDEX_PATH)
#         if os.path.exists(VECTOR_DATA_PATH):
#             os.remove(VECTOR_DATA_PATH)

#         self._save_db()
#         print(f"{C_GREEN}[VectorDB] Database reset complete.{C_RESET}")

#     def _save_db(self):
#         faiss.write_index(self.index, VECTOR_INDEX_PATH)
#         with open(VECTOR_DATA_PATH, "wb") as f:
#             pickle.dump(self.text_store, f)

#     def add_chunks(self, chunks: List[Dict[str, Any]]):
#         if client is None or self.index is None:
#             return

#         existing_texts = {c.get("text") for c in self.text_store}
#         new_embeddings = []
#         new_chunks = []

#         for chunk in chunks:
#             text = chunk.get("text", "").strip()
#             if text and text not in existing_texts:
#                 emb = _get_embedding(text) # This is now normalized
#                 if not np.all(emb == 0):
#                     new_embeddings.append(emb)
#                     new_chunks.append(chunk)

#         if new_embeddings:
#             # Vectors are already normalized by _get_embedding
#             self.index.add(np.array(new_embeddings).astype("float32"))
#             self.text_store.extend(new_chunks)
#             self._save_db()
#             print(f"{C_BLUE}[VectorDB] Added {len(new_chunks)} new chunks.{C_RESET}")

#     def search(self, query: str, k: int = 20) -> List[Tuple[Dict[str, Any], float]]:
#         if client is None or self.index is None or self.index.ntotal == 0:
#             return []

#         # This will return a normalized vector
#         query_embedding = _get_embedding(query).reshape(1, -1)

#         if np.all(query_embedding == 0):
#             print(f"{C_RED}[VectorDB ERROR] Invalid query embedding.{C_RESET}")
#             return []

#         k_actual = min(k, self.index.ntotal)

#         # In IndexFlatIP, D represents Similarity Scores (higher is better)
#         D, I = self.index.search(query_embedding.astype("float32"), k_actual)

#         results = []
#         for score, idx in zip(D[0], I[0]):
#             results.append((self.text_store[idx], score))

#         # We keep them in the order FAISS provides (highest similarity first)
#         return results