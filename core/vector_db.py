import os
import pickle
import numpy as np
import faiss
from typing import List, Tuple, Optional, Dict, Any
# Import utilities for constants, client, and embedding function
from .utilities import (
    get_embedding, C_RESET, C_CYAN, C_RED, C_BLUE, C_GREEN, C_MAGENTA, C_YELLOW,
    DIMENSION, VECTOR_INDEX_PATH, VECTOR_DATA_PATH, EMBED_MODEL, client
)


def _get_embedding(text: str) -> np.ndarray:
    if client is None:
        return np.zeros(DIMENSION, dtype=np.float32)
    try:
        response = client.embeddings.create(
            input=text,
            model=EMBED_MODEL
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as e:
        print(f"{C_RED}[EMBEDDING ERROR] Failed to get embedding: {e}{C_RESET}")
        return np.zeros(DIMENSION, dtype=np.float32)

class VectorDBWrapper:
    def __init__(self, dimension: int = DIMENSION):
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.text_store: List[Dict[str, Any]] = []

        if client is not None:
            self._initialize_db()
        else:
            print(f"{C_RED}[VectorDB] Skipping initialization due to missing API key.{C_RESET}")

    def _initialize_db(self):
        if os.path.exists(VECTOR_INDEX_PATH) and os.path.exists(VECTOR_DATA_PATH):
            try:
                self.index = faiss.read_index(VECTOR_INDEX_PATH)
                with open(VECTOR_DATA_PATH, "rb") as f:
                    self.text_store = pickle.load(f)
                print(f"{C_CYAN}[VectorDB] Loaded existing DB. Chunks: {len(self.text_store)}{C_RESET}")
            except Exception:
                print(f"{C_RED}[VectorDB] Failed to load DB. Creating new one.{C_RESET}")
                self._create_new_db()
        else:
            self._create_new_db()

    def reset_db(self):
        print(f"{C_RED}[VectorDB] Starting database reset...{C_RESET}")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.text_store = []

        if os.path.exists(VECTOR_INDEX_PATH):
            os.remove(VECTOR_INDEX_PATH)
        if os.path.exists(VECTOR_DATA_PATH):
            os.remove(VECTOR_DATA_PATH)

        self._save_db()
        print(f"{C_GREEN}[VectorDB] Database reset complete.{C_RESET}")

    def _create_new_db(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.text_store = []
        self._save_db()
        print(f"{C_CYAN}[VectorDB] Created new IndexFlatL2 DB.{C_RESET}")

    def _save_db(self):
        faiss.write_index(self.index, VECTOR_INDEX_PATH)
        with open(VECTOR_DATA_PATH, "wb") as f:
            pickle.dump(self.text_store, f)

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        if client is None or self.index is None:
            return

        existing_texts = {c.get("text") for c in self.text_store}
        new_embeddings = []
        new_chunks = []

        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if text and text not in existing_texts:
                emb = _get_embedding(text)
                if not np.all(emb == 0):
                    new_embeddings.append(emb)
                    new_chunks.append(chunk)

        if new_embeddings:
            self.index.add(np.array(new_embeddings).astype("float32"))
            self.text_store.extend(new_chunks)
            self._save_db()
            print(f"{C_BLUE}[VectorDB] Added {len(new_chunks)} new chunks.{C_RESET}")

    def search(self, query: str, k: int = 20) -> List[Tuple[Dict[str, Any], float]]:
        if client is None or self.index is None or self.index.ntotal == 0:
            return []

        query_embedding = _get_embedding(query).reshape(1, -1)
        if np.all(query_embedding == 0):
            print(f"{C_RED}[VectorDB ERROR] Invalid query embedding.{C_RESET}")
            return []

        k_actual = min(k, self.index.ntotal)
        D, I = self.index.search(query_embedding.astype("float32"), k_actual)

        results = []
        for dist, idx in zip(D[0], I[0]):
            results.append((self.text_store[idx], dist))

        return results

# ==================================================================================================
# INTEGRATED TEST BLOCK
# ==================================================================================================
if __name__ == "__main__":
    # --- Mock Data for Testing ---
    # These chunks and queries are highly relevant to the main project context (CsSnI3)
    TEST_CHUNKS = [
        "CsSnI3 is a lead-free perovskite material being studied as a less toxic alternative to lead-halide perovskites.",
        "Computational data predicts that the cubic phase of CsSnI3 has a theoretical bandgap around 1.3 eV, which is near-optimal for solar cells.",
        "Experimental synthesis often struggles with the rapid oxidation of Sn2+ to Sn4+, leading to poor long-term stability.",
        "The instability issue is partially mitigated by using excess SnF2 during the solution processing method.",
        "An irrelevant chunk about quantum computing algorithms."
    ]

    # Specific queries to test semantic search accuracy
    TEST_QUERIES = [
        "Stability challenges in CsSnI3 solar cells",
        "Bandgap of lead-free perovskite CsSnI3",
        "Irrelevant topic not in the chunks",
    ]

    print(f"{C_CYAN}*** STARTING VectorDBWrapper ISOLATED TEST ***{C_RESET}")

    # --- 1. SETUP: Initialize and Reset ---
    print(f"\n{C_MAGENTA}--- 1. Testing Initialization and Reset ---{C_RESET}")
    db = VectorDBWrapper()

    # Ensure the database is clean before starting the test
    db.reset_db()

    if db.index is not None and db.index.ntotal == 0 and len(db.text_store) == 0:
        print(f"{C_GREEN}[TEST SUCCESS] DB reset and initialized correctly (0 chunks).{C_RESET}")
    else:
        print(f"{C_RED}[TEST FAILURE] DB reset failed. Check FAISS/pickle setup.{C_RESET}")
        if client is None:
            print(f"{C_YELLOW}[TEST NOTE] Cannot proceed without a valid LLM client for embeddings.{C_RESET}")
        exit()

    # --- 2. FUNCTIONALITY: Add Chunks ---
    print(f"\n{C_MAGENTA}--- 2. Testing Chunk Addition and Persistence ---{C_RESET}")

    # Add the mock chunks
    db.add_chunks(TEST_CHUNKS)

    # ASSERTION 2.1: Check if all chunks were added
    if db.index.ntotal == len(TEST_CHUNKS):
        print(f"{C_GREEN}[TEST SUCCESS] Added {len(TEST_CHUNKS)} chunks successfully to the index.{C_RESET}")
    else:
        print(f"{C_RED}[TEST FAILURE] Expected {len(TEST_CHUNKS)} chunks, found {db.index.ntotal}. Check _get_embedding() and add logic.{C_RESET}")

    # ASSERTION 2.2: Check persistence (Load a new instance and verify count)
    db_reloaded = VectorDBWrapper()
    if db_reloaded.index.ntotal == len(TEST_CHUNKS):
         print(f"{C_GREEN}[TEST SUCCESS] DB persistence verified (reloaded with {len(TEST_CHUNKS)} chunks).{C_RESET}")
    else:
         print(f"{C_RED}[TEST FAILURE] DB persistence failed. Check _save_db() and file paths.{C_RESET}")

    # --- 3. FUNCTIONALITY: Search ---
    print(f"\n{C_MAGENTA}--- 3. Testing Semantic Search Accuracy ---{C_RESET}")


    for query_text in TEST_QUERIES:
        print(f"{C_BLUE}Searching for: '{query_text}'{C_RESET}")

        # We use the reloaded instance to ensure we test the persisted index
        results = db_reloaded.search(query_text, k=3)

        if not results:
            print(f"{C_RED}[TEST FAILURE] Search returned 0 results for query: '{query_text}'.{C_RESET}")
            continue

        # ASSERTION 3.1: Check if the top result is highly relevant (semantic test)
        top_chunk, top_dist = results[0]

        if "Stability" in query_text:
            # Expect chunk 3 ("Experimental synthesis often struggles...")
            expected_keyword = "oxidation"
        elif "Bandgap" in query_text:
            # Expect chunk 2 ("Computational data predicts...")
            expected_keyword = "computational"
        else:
             # Expect an irrelevant chunk with a high distance
             expected_keyword = None

        print(f"  Top Result Dist: {top_dist:.4f}")

        if expected_keyword and expected_keyword in top_chunk:
             print(f"{C_GREEN}[TEST SUCCESS] Semantic Match: Top result contains '{expected_keyword}'. (Dist: {top_dist:.4f}){C_RESET}")
        elif not expected_keyword and top_dist > 0.5: # Irrelevant query should have high distance
             print(f"{C_GREEN}[TEST SUCCESS] Irrelevance Check: Top result distance is high (>0.5).{C_RESET}")
        else:
             print(f"{C_YELLOW}[TEST WARNING] Semantic Match was weak or unexpected. Top chunk: '{top_chunk[:50]}...'{C_RESET}")


    # --- 4. CLEANUP ---
    print(f"\n{C_MAGENTA}--- 4. Testing Cleanup ---{C_RESET}")
    db_reloaded.reset_db()

    # Final check after reset
    db_final = VectorDBWrapper()
    if db_final.index is not None and db_final.index.ntotal == 0:
        print(f"{C_GREEN}[TEST SUCCESS] Final cleanup (reset_db) successful.{C_RESET}")
    else:
        print(f"{C_RED}[TEST FAILURE] Final reset failed. Index count: {db_final.index.ntotal if db_final.index else 'N/A'}.{C_RESET}")

    print(f"\n{C_CYAN}*** VectorDBWrapper ISOLATED TEST COMPLETE ***{C_RESET}")