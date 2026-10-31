import os
import pickle
import time
import numpy as np
import faiss
from typing import List, Tuple, Optional, Dict, Any
from dotenv import load_dotenv, find_dotenv

# Load environment variables dynamically across project directories
load_dotenv(find_dotenv())

# Import utilities for constants, client, and the shared embedding function
from .utilities import (
    get_embedding, preload_embedding_model, EMBEDDING_PROVIDER,
    C_RESET, C_CYAN, C_RED, C_BLUE, C_GREEN, C_MAGENTA, C_YELLOW,
    DIMENSION, VECTOR_INDEX_PATH, VECTOR_DATA_PATH, client
)

# Embeddings (local sentence-transformers or Gemini, chosen by EMBEDDING_PROVIDER in .env)
# are produced by utilities.get_embedding, the single implementation shared project-wide.


class VectorDBWrapper:
    def __init__(self, dimension: int = DIMENSION):
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.text_store: List[Dict[str, Any]] = []

        # Local model does not strictly require the Gemini client API key to run
        if EMBEDDING_PROVIDER == "local" or client is not None:
            preload_embedding_model()  # no-op unless the provider is "local"
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
                emb = get_embedding(text)  # Uses active provider
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

        query_embedding = get_embedding(query).reshape(1, -1)

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
