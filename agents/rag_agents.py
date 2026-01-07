import time
import re
import requests
from io import BytesIO
from typing import Dict, List, Any, Optional, Tuple
from pypdf import PdfReader
import json # Added for the test block

from core.research_state import ResearchState
from core.vector_db import VectorDBWrapper
from core.utilities import (
    C_ACTION, C_RESET, C_GREEN, C_YELLOW, C_RED, C_BLUE, C_MAGENTA, C_CYAN, # Added C_CYAN for testing
    client, LLM_MODEL
)

# ==================================================================================================
# SECTION 7: RETRIEVAL AGENT (PRODUCTION-GRADE, MODEL-AGNOSTIC)
# ==================================================================================================

class RetrievalAgent:
    def __init__(self, agent_id: str = "retrieval_agent", chunk_size: int = 500, model: str = LLM_MODEL):
        self.id = agent_id
        self.chunk_size = chunk_size
        self.model = model

    def _download_pdf(self, url: str) -> Optional[BytesIO]:
        try:
            time.sleep(1)  # Polite crawling
            response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()

            # --- CRITICAL CHECK ---
            content_type = response.headers.get('Content-Type', '').lower()

            # If the response is HTML, it's a landing page, not a raw PDF
            if 'text/html' in content_type or b'<!doc' in response.content[:10].lower():
                print(f"{C_YELLOW}[{self.id} WARN] URL {url[:30]}... is an HTML page, skipping PDF parser.{C_RESET}")
                return None

            return BytesIO(response.content)
        except Exception as e:
            if 'arxiv' not in url.lower():
                print(f"{C_RED}[{self.id} ERROR] Failed to download PDF from {url[:30]}...: {e}{C_RESET}")
            return None

    def _extract_text_from_pdf(self, pdf_stream: BytesIO) -> str:
        try:
            reader = PdfReader(pdf_stream)
            return " ".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            print(f"{C_RED}[{self.id} ERROR] PDF text extraction failed: {e}{C_RESET}")
            return ""

    def _chunk_text(self, text: str) -> List[str]:
        """
        Semantic-first, model-agnostic chunking.
        Safe for OpenAI / Gemini / Claude.
        """
        if not text:
            return []

        max_chars = int(self.chunk_size * 3.5)
        overlap_chars = int(max_chars * 0.1)

        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks, current_chunk = [], ""

        for sentence in sentences:
            # HARD SPLIT oversized sentences
            if len(sentence) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                for i in range(0, len(sentence), max_chars):
                    chunks.append(sentence[i:i + max_chars].strip())
                continue

            if len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = (
                    current_chunk[-overlap_chars:] + " " + sentence
                    if overlap_chars > 0 and len(current_chunk) > overlap_chars
                    else sentence
                )

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def execute(self, state: ResearchState) -> ResearchState:
        print(f"\n{C_ACTION}[{self.id.upper()} START] Running Retrieval + Chunking...{C_RESET}")

        raw_data = state.get('raw_tool_data', [])
        pdf_entries = [d for d in raw_data if d.get('metadata', {}).get('pdf_url')]

        state.setdefault('full_text_chunks', [])
        all_new_chunks = []
        downloaded_urls = set()

        # --- PDF PROCESSING ---
        for entry in pdf_entries:
            pdf_url = entry['metadata']['pdf_url']
            pdf_stream = self._download_pdf(pdf_url)
            if not pdf_stream:
                continue

            text = self._extract_text_from_pdf(pdf_stream)
            if not text.strip():
                continue

            downloaded_urls.add(pdf_url)
            chunks = self._chunk_text(text)
            doc_hash = abs(hash(pdf_url)) % 10000

            for i, chunk in enumerate(chunks):
                all_new_chunks.append({
                    "chunk_id": f"{entry['tool_id']}_{doc_hash}_{i}",
                    "doc_id": pdf_url,
                    "chunk_index": i,
                    "text": chunk,
                    "source": entry['tool_id'],
                    "url": pdf_url
                })

        # --- ABSTRACT / SNIPPET FALLBACK ---
        for entry in raw_data:
            if entry.get('tool_id') == 'materials_agent':
                continue
            if entry.get('metadata', {}).get('pdf_url') in downloaded_urls:
                continue

            raw_text = entry.get('text', '').strip()
            if not raw_text:
                continue

            source_url = entry.get('metadata', {}).get('url', 'N/A')
            chunks = self._chunk_text(raw_text)
            doc_hash = abs(hash(source_url)) % 10000

            for i, chunk in enumerate(chunks):
                all_new_chunks.append({
                    "chunk_id": f"{entry['tool_id']}_abs_{doc_hash}_{i}",
                    "doc_id": source_url,
                    "chunk_index": i,
                    "text": chunk,
                    "source": entry['tool_id'],
                    "url": source_url
                })

        if not all_new_chunks:
            all_new_chunks = [{"text": "No content retrieved.", "source": "flow_control", "url": ""}]

        state['full_text_chunks'].extend(all_new_chunks)

        print(f"{C_YELLOW}[{self.id.upper()} STATE] Total chunks: {len(state['full_text_chunks'])}{C_RESET}")
        print(f"{C_GREEN}[{self.id.upper()} DONE] Retrieval complete.{C_RESET}")
        return state


# ==================================================================================================
# RAG Agent (Section 8) - FULLY UPGRADED
# ==================================================================================================
class RAGAgent:
    def __init__(
        self,
        agent_id: str = "rag_agent",
        max_chunks_to_keep: int = 5,
        vector_db: Optional[VectorDBWrapper] = None
    ):
        self.id = agent_id
        self.max_chunks_to_keep = max_chunks_to_keep
        self.vector_db = vector_db if vector_db is not None else VectorDBWrapper()

    def _passes_keyword_gate(self, chunk_text: str, literal_term: str) -> bool:
        chunk_lower = chunk_text.lower()
        is_academic_noise = any(
            x in chunk_lower for x in ["full text", "arxiv paper", "pubmed abstract"]
        )
        contains_literal = literal_term in chunk_lower if literal_term else True
        return not (is_academic_noise and not contains_literal)

    def execute(self, state: ResearchState) -> ResearchState:
        print(f"\n{C_ACTION}[{self.id.upper()} START] Running RAG: Vector Search + Neighbor Expansion...{C_RESET}")

        # --- SAFETY CHECKS ---
        if client is None or self.vector_db.index is None:
            state['filtered_context'] = "RAG processing skipped due to missing API key or Vector DB."
            state['rag_complete'] = True
            print(f"{C_RED}[{self.id} FAIL] RAG skipped.{C_RESET}")
            return state

        query = state.get('semantic_query', '')
        literal_term = state.get('api_search_term', '').lower()

        # --- 1. COLLECT STRUCTURED CONTEXT (NON-VECTOR) ---
        structured_context = []
        for d in state.get('raw_tool_data', []):
            if d.get('tool_id') == 'materials_agent' and d.get('text'):
                structured_context.append(
                    f"--- Structured Data (Materials Property) ---\n{d['text']}"
                )

        if structured_context:
            print(f"{C_BLUE}[{self.id} INFO] Preserved {len(structured_context)} structured context blocks.{C_RESET}")

        # --- 2. PREPARE CHUNKS FOR VECTOR DB ---
        structured_chunks = state.get('full_text_chunks', [])
        chunks_for_db = [
            c for c in structured_chunks
            if isinstance(c, dict) and c.get('text')
        ]

        if not chunks_for_db and not structured_context:
            state['filtered_context'] = "No data available for RAG."
            state['rag_complete'] = True
            print(f"{C_RED}[{self.id} FAIL] No chunks or structured data found.{C_RESET}")
            return state

        # --- 3. INDEX CHUNKS ---
        if chunks_for_db:
            self.vector_db.add_chunks(chunks_for_db)
            print(
                f"{C_BLUE}[{self.id} INFO] Vector DB contains "
                f"{self.vector_db.index.ntotal} total chunks.{C_RESET}"
            )

        # --- 4. VECTOR SEARCH ---
        top_k_results = self.vector_db.search(query, k=8)
        print(
            f"{C_BLUE}[{self.id} INFO] Vector search returned "
            f"{len(top_k_results)} candidate chunks.{C_RESET}"
        )

        if not top_k_results:
            print(f"{C_YELLOW}[{self.id} WARN] No vector matches found.{C_RESET}")

        # --- 5. BUILD DOCUMENT MAP FOR NEIGHBOR EXPANSION ---
        doc_map = {}
        for c in self.vector_db.text_store:
            doc_map.setdefault(c.get('doc_id'), []).append(c)

        for doc_id in doc_map:
            doc_map[doc_id].sort(key=lambda x: x.get('chunk_index', 0))

        # --- 6. NEIGHBOR EXPANSION ---
        expanded_chunks = []
        DISTANCE_THRESHOLD = 1.6

        for chunk_dict, distance in top_k_results:
            if distance > DISTANCE_THRESHOLD:
                print(
                    f"{C_YELLOW}[{self.id} DEBUG] Skipping chunk "
                    f"(distance={distance:.2f} > {DISTANCE_THRESHOLD}).{C_RESET}"
                )
                continue

            doc_id = chunk_dict.get("doc_id")
            idx = chunk_dict.get("chunk_index", 0)
            family = doc_map.get(doc_id, [])

            if not family:
                continue

            idx = min(idx, len(family) - 1)

            # Expand to previous, current, next
            for i in range(max(0, idx - 1), min(len(family), idx + 2)):
                expanded_chunks.append(family[i])

        print(
            f"{C_BLUE}[{self.id} INFO] Expanded to "
            f"{len(expanded_chunks)} chunks after neighbor expansion.{C_RESET}"
        )

        # --- 7. DEDUPLICATION + KEYWORD FILTER ---
        seen_ids = set()
        final_chunks = []

        for c in expanded_chunks:
            if c["chunk_id"] in seen_ids:
                continue
            if not self._passes_keyword_gate(c["text"], literal_term):
                continue

            final_chunks.append(c["text"])
            seen_ids.add(c["chunk_id"])

            if len(final_chunks) >= self.max_chunks_to_keep:
                break

        # --- 8. FALLBACK IF FILTERING REMOVED EVERYTHING ---
        if not final_chunks and chunks_for_db:
            print(
                f"{C_RED}[{self.id} FALLBACK] No chunks passed filters. "
                f"Using raw chunks as fallback.{C_RESET}"
            )
            final_chunks = [c["text"] for c in chunks_for_db[:self.max_chunks_to_keep]]

        # --- 9. ASSEMBLE FINAL CONTEXT ---
        final_context_list = structured_context + final_chunks
        state['filtered_context'] = (
            "\n---\n".join(final_context_list)
            if final_context_list
            else "No relevant context found."
        )

        # --- 10. FINAL STATE UPDATE ---
        state['rag_complete'] = True

        print(
            f"{C_YELLOW}[{self.id.upper()} STATE] Final context assembled from "
            f"{len(final_context_list)} sections "
            f"({len(structured_context)} structured, "
            f"{len(final_chunks)} retrieved).{C_RESET}"
        )
        print(f"{C_GREEN}[{self.id.upper()} DONE] RAG processing complete.{C_RESET}")

        return state


# ==================================================================================================
# INTEGRATED TEST BLOCK
# ==================================================================================================

# --- Mock State Data for Testing ---
# This data simulates the output after the 'tool_agents' (arxiv, pubmed, materials) have run.
MOCK_INITIAL_STATE = {
    "user_query": "A detailed review on the synthesis and bandgap stability of lead-free CsSnI3 perovskite solar cells using computational and experimental data.",
    "semantic_query": "A detailed review on the synthesis and bandgap stability of lead-free CsSnI3 perovskite solar cells using computational and experimental data.",
    "api_search_term": "CsSnI3",
    "active_tools": ["arxiv", "pubmed", "materials"],
    "raw_tool_data": [
        # Irrelevant ArXiv data to test vector filtering (PDF links are expected to fail, forcing fallback to abstract text)
        { "text": "Title: Selfi: Self Improving Reconstruction Engine via 3D Geometric Feature Alignment. Abstract: Novel View Synthesis (NVS) has traditionally relied on models with explicit 3D inductive biases...", "source_type": "arxiv", "tool_id": "arxiv_agent", "metadata": { "pdf_url": "https://arxiv.org/pdf/2512.08930v1" } },
        { "text": "Title: On a cross-diffusion hybrid model: Cancer Invasion Tissue with Normal Cell Involved. Abstract: In this paper, we study a well-posedness problem on a new mathematical model for cancer invasion...", "source_type": "arxiv", "tool_id": "arxiv_agent", "metadata": { "pdf_url": "https://arxiv.org/pdf/2512.08929v1" } },
        { "text": "Title: SAQ: Stabilizer-Aware Quantum Error Correction Decoder. Abstract: Quantum Error Correction (QEC) decoding faces a fundamental accuracy-efficiency tradeoff...", "source_type": "arxiv", "tool_id": "arxiv_agent", "metadata": { "pdf_url": "https://arxiv.org/pdf/2512.08914v1" } },
        # Relevant PubMed abstracts (testing abstract fallback and semantic relevance)
        { "text": "Title: Advancement on Lead-Free Organic-Inorganic Halide Perovskite Solar Cells: A Review.. Abstract: Remarkable attention has been committed to the recently discovered cost effective and solution processable lead-free organic-inorganic halide perovskite solar cells...", "source_type": "pubmed", "tool_id": "pubmed_agent", "metadata": { "external_id": "29899206" } },
        { "text": "Title: A Brief Review of Perovskite Quantum Dot Solar Cells: Synthesis, Property and Defect Passivation.. Abstract: Perovskite quantum dot solar cells (PQDSCs)... highly dependent on the properties of interfaces...", "source_type": "pubmed", "tool_id": "pubmed_agent", "metadata": { "external_id": "39289160" } },
        { "text": "Title: Development on inverted perovskite solar cells: A review.. Abstract: Recently, inverted perovskite solar cells (IPSCs) have received note-worthy consideration...", "source_type": "pubmed", "tool_id": "pubmed_agent", "metadata": { "external_id": "38298729" } },
        { "text": "Material: CsSnI3 (mp-616378). Stability: Unstable (E/hull: 0.01184 eV). Band Gap: 0.5537 eV. Energy Above Hull: 0.0118 eV.", "source_type": "materials_project", "tool_id": "materials_agent", "metadata": { "material_id": "mp-616378", "is_stable": False } },
        { "text": "Material: CsSnI3 (mp-614013). Stability: Stable. Band Gap: 0.4499 eV. Energy Above Hull: 0.0 eV.", "source_type": "materials_project", "tool_id": "materials_agent", "metadata": { "material_id": "mp-614013", "is_stable": True } },
    ],
    "full_text_chunks": [],
    "rag_complete": False,
    "filtered_context": "",
    "is_refining": False
}


def test_rag_agents_pipeline():
    """Executes the RetrievalAgent followed by the RAGAgent for testing."""
    print(f"{C_CYAN}*** STARTING RAG AGENTS PIPELINE TEST ***{C_RESET}")

    # 1. SETUP
    try:
        # NOTE: VectorDBWrapper must be successfully initialized (i.e., FAISS/embeddings work)
        vector_db_instance = VectorDBWrapper()

        # Load the mock state
        state = ResearchState(MOCK_INITIAL_STATE)

        retrieval_agent = RetrievalAgent()
        rag_agent = RAGAgent(vector_db=vector_db_instance)

    except Exception as e:
        print(f"{C_RED}SETUP FAILED: Ensure VectorDBWrapper and Agents are correctly defined/imported. Error: {e}{C_RESET}")
        return

    # --- 2. EXECUTE RETRIEVAL AGENT (Chunking & Preparation) ---
    print(f"\n{C_MAGENTA}--- 2A. EXECUTING RETRIEVAL AGENT (Chunking) ---{C_RESET}")
    try:
        state = retrieval_agent.execute(state)
        chunk_count = len(state.get('full_text_chunks', []))

        # ASSERTION 1: Check if chunks were created (8 abstract/snippet entries total)
        if chunk_count == 8:
            print(f"{C_GREEN}[TEST SUCCESS] Retrieval Agent: Created exactly {chunk_count} chunks (all abstracts/snippets).{C_RESET}")
        elif chunk_count > 0:
            print(f"{C_YELLOW}[TEST WARNING] Retrieval Agent: Expected 8 chunks, found {chunk_count}. Check chunk_size and data parsing.{C_RESET}")
        else:
            print(f"{C_RED}[TEST FAILURE] Retrieval Agent: Found 0 chunks. Check PDF download/fallback logic.{C_RESET}")
            return

    except Exception as e:
        print(f"{C_RED}[TEST FAILURE] Retrieval Agent execution failed: {type(e).__name__}: {e}{C_RESET}")
        return

    # --- 3. EXECUTE RAG AGENT (Vectorization, Search, & Filtering) ---
    print(f"\n{C_MAGENTA}--- 2B. EXECUTING RAG AGENT (Search & Filter) ---{C_RESET}")
    try:
        state = rag_agent.execute(state)
        context = state.get('filtered_context', '')

        # ASSERTION 2: Check for presence of structured data and relevant chunks
        is_structured_present = "--- Structured Data (Materials Property) ---" in context
        is_relevant_text_present = state['api_search_term'].lower() in context.lower() # CsSnI3 check

        if state['rag_complete'] and is_structured_present and is_relevant_text_present:
            print(f"{C_GREEN}[TEST SUCCESS] RAG Agent: Pipeline is complete and context is generated.{C_RESET}")
            print(f"{C_BLUE}[DEBUG] Final context length: {len(context)} characters.{C_RESET}")
        else:
            print(f"{C_RED}[TEST FAILURE] RAG Agent: Context generation failed. Complete: {state['rag_complete']}. Structured Present: {is_structured_present}. Relevant Text Present: {is_relevant_text_present}.{C_RESET}")
            return

        # ASSERTION 3: Check if irrelevant text was filtered out (Crucial RAG Test)
        if "cancer" not in context.lower() and "quantum error" not in context.lower() and "3d geometric" not in context.lower():
             print(f"{C_GREEN}[TEST SUCCESS] RAG Agent: Irrelevant abstracts were successfully filtered out by vector search/threshold.{C_RESET}")
        else:
             print(f"{C_RED}[TEST FAILURE] RAG Agent: Irrelevant topics were NOT filtered out. Check the vector search distance threshold (1.45).{C_RESET}")


    except Exception as e:
        print(f"{C_RED}[TEST FAILURE] RAG Agent execution failed: {type(e).__name__}: {e}{C_RESET}")
        return

    print(f"\n{C_CYAN}*** RAG AGENTS PIPELINE TEST COMPLETE ***{C_RESET}")


if __name__ == "__main__":
    test_rag_agents_pipeline()