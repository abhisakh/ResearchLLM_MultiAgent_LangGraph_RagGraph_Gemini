import time
import re
import requests
from io import BytesIO
from typing import Dict, List, Any, Optional, Tuple
from pypdf import PdfReader
import json # Added for the test block
from sentence_transformers import CrossEncoder
from bs4 import BeautifulSoup

from core.research_state import ResearchState
from core.vector_db import VectorDBWrapper
from core.utilities import (
    C_ACTION, C_RESET, C_GREEN, C_YELLOW, C_RED, C_BLUE, C_MAGENTA,C_PURPLE, C_CYAN, # Added C_CYAN for testing
    client, LLM_MODEL
)

# ==================================================================================================
# SECTION 7: RETRIEVAL AGENT (PRODUCTION-GRADE, MODEL-AGNOSTIC)
# ==================================================================================================

class RetrievalAgent:
    """
    Agent responsible for downloading and processing research content.
    RESTORED: Full PDF parsing, chunking, and deduplication logic.
    UPGRADED: Added BeautifulSoup HTML fallback to prevent skipping PubMed/OpenAlex results.
    """
    def __init__(self, agent_id: str = "retrieval_agent", chunk_size: int = 500, model: str = LLM_MODEL):
        self.id = agent_id
        self.chunk_size = chunk_size
        self.model = model

    def _fetch_content(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Unified fetcher that identifies PDF vs HTML content.
        """
        try:
            time.sleep(1.5)  # Slightly longer polite delay

            # --- STEALTH HEADER BLOCK ---
            stealth_headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.google.com/',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }

            response = requests.get(url, timeout=15, headers=stealth_headers, allow_redirects=True)

            # Catch 403 specifically to log the WAF encounter
            if response.status_code == 403:
                print(f"{C_YELLOW}[{self.id.upper()} WAF] 403 Forbidden on {url[:40]}. Switching to fallback. {C_RESET}")
                return None
            response.raise_for_status()

            content_type = response.headers.get('Content-Type', '').lower()

            # 1. PDF Path
            if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                return {'type': 'pdf', 'data': BytesIO(response.content)}

            # 2. HTML Path (Fallback for PubMed/OpenAlex landing pages)
            if 'text/html' in content_type or b'<!doc' in response.content[:10].lower():
                return {'type': 'html', 'data': response.text}

            return None
        except Exception as e:
            if 'arxiv' not in url.lower():
                print(f"{C_RED}[{self.id.upper()} ERROR] Failed to fetch {url[:50]}... : {e}{C_RESET}")
            return None

    def _extract_text_from_pdf(self, pdf_stream: BytesIO) -> str:
        try:
            reader = PdfReader(pdf_stream)
            return " ".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            print(f"{C_RED}[{self.id.upper()} ERROR] PDF extraction failed: {e}{C_RESET}")
            return ""

    def _extract_text_from_html(self, html_text: str) -> str:
        """
        Extracts abstracts or main content from academic landing pages.
        """
        try:
            soup = BeautifulSoup(html_text, 'html.parser')
            # Strip unwanted tags
            for script_or_style in soup(["script", "style", "header", "footer", "nav"]):
                script_or_style.decompose()

            # Target common academic abstract identifiers
            content = soup.find('div', {'id': 'abstract'}) or \
                      soup.find('div', {'class': 'abstract-content'}) or \
                      soup.find('article') or \
                      soup.find('main')

            if content:
                return content.get_text(separator=' ', strip=True)

            # Final fallback: Grab the first 8-10 meaningful paragraphs
            paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if len(p.get_text()) > 50]
            return " ".join(paragraphs[:10])
        except Exception as e:
            print(f"{C_RED}[{self.id.upper()} ERROR] HTML scrape failed: {e}{C_RESET}")
            return ""

    def _chunk_text(self, text: str) -> List[str]:
        """Original chunking logic with sentence-boundary awareness."""
        if not text: return []
        max_chars = int(self.chunk_size * 3.5)
        overlap_chars = int(max_chars * 0.1)
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current_chunk = [], ""

        for sentence in sentences:
            if len(sentence) > max_chars:
                if current_chunk: chunks.append(current_chunk.strip())
                current_chunk = ""
                for i in range(0, len(sentence), max_chars):
                    chunks.append(sentence[i:i + max_chars].strip())
                continue

            if len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = current_chunk[-overlap_chars:] + " " + sentence if overlap_chars > 0 and len(current_chunk) > overlap_chars else sentence

        if current_chunk: chunks.append(current_chunk.strip())
        return chunks

    def execute(self, state: ResearchState) -> ResearchState:
        # 1. BREADCRUMB TRACKING (Your original feature)
        if "visited_nodes" not in state or state["visited_nodes"] is None:
            state["visited_nodes"] = []
        state["visited_nodes"].append(self.id)

        print(f"\n{C_ACTION}[{self.id.upper()} START] Running Retrieval + Deduplication...{C_RESET}")

        # 2. DEDUPLICATION GATE
        existing_chunks = state.get('full_text_chunks', [])
        processed_doc_ids = {chunk['doc_id'] for chunk in existing_chunks if 'doc_id' in chunk}
        raw_data = state.get('raw_tool_data', [])

        state.setdefault('full_text_chunks', [])
        all_new_chunks = []
        downloaded_urls_this_run = set()

        # 3. UNIFIED PROCESSING (PDF & HTML)
        for entry in raw_data:
            # Check both possible URL sources
            target_url = entry.get('metadata', {}).get('pdf_url') or entry.get('metadata', {}).get('url')

            if not target_url or target_url in processed_doc_ids:
                continue

            if entry.get('tool_id') == 'materials_search':
                continue

            # FETCH PHASE
            fetch_result = self._fetch_content(target_url)
            if not fetch_result:
                continue

            # PARSE PHASE
            if fetch_result['type'] == 'pdf':
                text = self._extract_text_from_pdf(fetch_result['data'])
            else:
                text = self._extract_text_from_html(fetch_result['data'])

            if not text.strip():
                continue

            # CHUNK & STORE PHASE
            downloaded_urls_this_run.add(target_url)
            chunks = self._chunk_text(text)
            doc_hash = abs(hash(target_url)) % 10000

            for i, chunk in enumerate(chunks):
                all_new_chunks.append({
                    "chunk_id": f"{entry['tool_id']}_{doc_hash}_{i}",
                    "doc_id": target_url,
                    "chunk_index": i,
                    "text": chunk,
                    "source": entry['tool_id'],
                    "url": target_url
                })

        # 4. FINAL ABSTRACT FALLBACK (Ensuring NO data is left behind)
        for entry in raw_data:
            source_url = entry.get('metadata', {}).get('url') or entry.get('metadata', {}).get('pdf_url')
            if not source_url or source_url in processed_doc_ids or source_url in downloaded_urls_this_run:
                continue

            if entry.get('tool_id') == 'materials_search':
                continue

            raw_text = entry.get('text', '').strip()
            if not raw_text: continue

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

        # 5. COMMIT & LOGGING
        # Only add a warning chunk if the ENTIRE system (past + present) is empty.
        # This prevents overwriting valid data from Attempt 1 during Attempt 2.
        if not all_new_chunks and not state.get('full_text_chunks'):
            all_new_chunks = [{
                "chunk_id": "sys_warning_empty",
                "doc_id": "none",
                "chunk_index": 0,
                "text": "Critical: No research content could be retrieved after filtering and fallbacks.",
                "source": "system",
                "url": "N/A"
            }]

        # Use extend to preserve the 'breadcrumb' of data from previous refinement loops
        state['full_text_chunks'].extend(all_new_chunks)

        print(f"{C_YELLOW}[{self.id.upper()} STATE] Added {len(all_new_chunks)} new chunks. Total: {len(state['full_text_chunks'])}{C_RESET}")
        print(f"{C_GREEN}[{self.id.upper()} DONE] Retrieval complete.{C_RESET}")

        return state


# ==================================================================================================
# RAG Agent (Section 8) - FULLY UPGRADED
# ==================================================================================================
class RAGAgent:
    def __init__(
        self,
        agent_id: str = "rag_agent",
        max_chunks_to_keep: int = 8,
        vector_db: Optional[VectorDBWrapper] = None
    ):
        self.id = agent_id
        self.max_chunks_to_keep = max_chunks_to_keep
        self.vector_db = vector_db if vector_db is not None else VectorDBWrapper()

        # --- INITIALIZE RERANKER ---
        # We load it here once so it doesn't reload every time execute() is called
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def _passes_keyword_gate(self, chunk_text: str, literal_term: str) -> bool:
        """Filters out academic noise unless it contains the specific material term."""
        chunk_lower = chunk_text.lower()
        is_academic_noise = any(
            x in chunk_lower for x in ["full text", "arxiv paper", "pubmed abstract"]
        )
        contains_literal = literal_term in chunk_lower if literal_term else True
        return not (is_academic_noise and not contains_literal)

    def execute(self, state: ResearchState) -> ResearchState:
        # --- BREADCRUMB TRACKING ---
        if "visited_nodes" not in state or state["visited_nodes"] is None:
            state["visited_nodes"] = []
        state["visited_nodes"].append(self.id)

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
            if d.get('tool_id') == 'materials_search' and d.get('text'):
                structured_context.append(
                    f"--- Structured Data (Materials Property) ---\n{d['text']}"
                )

        if structured_context:
            print(f"{C_BLUE}[{self.id} INFO] Preserved {len(structured_context)} structured data blocks.{C_RESET}")

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
            print(f"{C_BLUE}[{self.id} INFO] Vector DB contains {self.vector_db.index.ntotal} total chunks.{C_RESET}")

        # --- 4. VECTOR SEARCH ---
        # Fetching top 15 candidates instead of 8 (Reranker needs a larger pool to work with)
        top_k_results = self.vector_db.search(query, k=30)
        print(f"{C_BLUE}[{self.id} INFO] Vector search returned {len(top_k_results)} candidate chunks.{C_RESET}")

        if not top_k_results:
            print(f"{C_YELLOW}[{self.id} WARN] No vector matches found.{C_RESET}")

        # --- 4.5 CROSS-ENCODER RERANKING BLOCK ---
        # To deactivate: Comment out this entire block and uncomment the original DISTANCE_THRESHOLD logic below
        if top_k_results and query:
            print(f"{C_PURPLE}[{self.id} RERANK] Applying Cross-Encoder scoring...{C_RESET}")
            sentence_pairs = [[query, res[0]['text']] for res in top_k_results]
            scores = self.reranker.predict(sentence_pairs)

            reranked_list = []
            for i in range(len(top_k_results)):
                reranked_list.append((top_k_results[i][0], scores[i]))

            # Sort by score descending (Higher is better)
            top_k_results = sorted(reranked_list, key=lambda x: x[1], reverse=True)

            # Reranker Threshold: Scores > 0.1 are usually relevant
            ACTIVE_THRESHOLD = - 5
            IS_RERANKED = True
            print(f"{C_PURPLE}[{self.id} RERANK] Top Score: {top_k_results[0][1]:.4f}{C_RESET}")
        else:
            # UPGRADE: Fallback for Cosine Similarity Score
            # If we aren't reranking, we want chunks with similarity > 0.35
            ACTIVE_THRESHOLD = 0.35
            IS_RERANKED = False

        # --- 5. BUILD DOCUMENT MAP FOR NEIGHBOR EXPANSION ---
        doc_map = {}
        for c in self.vector_db.text_store:
            doc_id = c.get('doc_id')
            if doc_id:
                doc_map.setdefault(doc_id, []).append(c)

        for doc_id in doc_map:
            doc_map[doc_id].sort(key=lambda x: x.get('chunk_index', 0))

        # --- 6. NEIGHBOR EXPANSION ---
        expanded_chunks = []

        for chunk_dict, score in top_k_results:
            # UNIFIED LOGIC: Higher is always better now!
            # If the score (either Reranker or Cosine) is too low, skip it.
            if score < ACTIVE_THRESHOLD:
                continue

            doc_id = chunk_dict.get("doc_id")
            family = doc_map.get(doc_id, [])

            if not family:
                continue

            # Find the actual position in the original document
            actual_idx = next((i for i, item in enumerate(family) if item["chunk_id"] == chunk_dict["chunk_id"]), 0)

            # Grab context: 1 chunk before, current chunk, and 1 chunk after
            start_idx = max(0, actual_idx - 1)
            end_idx = min(len(family), actual_idx + 2)

            for i in range(start_idx, end_idx):
                expanded_chunks.append(family[i])

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
            print(f"{C_RED}[{self.id} FALLBACK] No chunks passed filters. Using raw chunks as fallback.{C_RESET}")
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

        print(f"{C_YELLOW}[{self.id.upper()} STATE] Final context assembled: {len(final_context_list)} sections.{C_RESET}")
        print(f"{C_GREEN}[{self.id.upper()} DONE] RAG processing complete.{C_RESET}")

        return state
# class RAGAgent:
#     def __init__(
#         self,
#         agent_id: str = "rag_agent",
#         max_chunks_to_keep: int = 8,  # Increased slightly to accommodate expanded windows
#         vector_db: Optional[VectorDBWrapper] = None
#     ):
#         self.id = agent_id
#         self.max_chunks_to_keep = max_chunks_to_keep
#         self.vector_db = vector_db if vector_db is not None else VectorDBWrapper()

#     def _passes_keyword_gate(self, chunk_text: str, literal_term: str) -> bool:
#         """Filters out academic noise unless it contains the specific material term."""
#         chunk_lower = chunk_text.lower()
#         is_academic_noise = any(
#             x in chunk_lower for x in ["full text", "arxiv paper", "pubmed abstract"]
#         )
#         contains_literal = literal_term in chunk_lower if literal_term else True
#         return not (is_academic_noise and not contains_literal)

#     def execute(self, state: ResearchState) -> ResearchState:
#         # --- BREADCRUMB TRACKING ---
#         if "visited_nodes" not in state or state["visited_nodes"] is None:
#             state["visited_nodes"] = []
#         state["visited_nodes"].append(self.id)

#         print(f"\n{C_ACTION}[{self.id.upper()} START] Running RAG: Vector Search + Neighbor Expansion...{C_RESET}")

#         # --- SAFETY CHECKS ---
#         if client is None or self.vector_db.index is None:
#             state['filtered_context'] = "RAG processing skipped due to missing API key or Vector DB."
#             state['rag_complete'] = True
#             print(f"{C_RED}[{self.id} FAIL] RAG skipped.{C_RESET}")
#             return state

#         query = state.get('semantic_query', '')
#         literal_term = state.get('api_search_term', '').lower()

#         # --- 1. COLLECT STRUCTURED CONTEXT (NON-VECTOR) ---
#         structured_context = []
#         for d in state.get('raw_tool_data', []):
#             # Matches the standardized tool naming
#             if d.get('tool_id') == 'materials_search' and d.get('text'):
#                 structured_context.append(
#                     f"--- Structured Data (Materials Property) ---\n{d['text']}"
#                 )

#         if structured_context:
#             print(f"{C_BLUE}[{self.id} INFO] Preserved {len(structured_context)} structured data blocks.{C_RESET}")

#         # --- 2. PREPARE CHUNKS FOR VECTOR DB ---
#         structured_chunks = state.get('full_text_chunks', [])
#         chunks_for_db = [
#             c for c in structured_chunks
#             if isinstance(c, dict) and c.get('text')
#         ]

#         if not chunks_for_db and not structured_context:
#             state['filtered_context'] = "No data available for RAG."
#             state['rag_complete'] = True
#             print(f"{C_RED}[{self.id} FAIL] No chunks or structured data found.{C_RESET}")
#             return state

#         # --- 3. INDEX CHUNKS ---
#         if chunks_for_db:
#             self.vector_db.add_chunks(chunks_for_db)
#             print(f"{C_BLUE}[{self.id} INFO] Vector DB contains {self.vector_db.index.ntotal} total chunks.{C_RESET}")

#         # --- 4. VECTOR SEARCH ---
#         # Fetching top 8 candidates before expansion
#         top_k_results = self.vector_db.search(query, k=8)
#         print(f"{C_BLUE}[{self.id} INFO] Vector search returned {len(top_k_results)} candidate chunks.{C_RESET}")

#         if not top_k_results:
#             print(f"{C_YELLOW}[{self.id} WARN] No vector matches found.{C_RESET}")

#         # --- 5. BUILD DOCUMENT MAP FOR NEIGHBOR EXPANSION ---
#         doc_map = {}
#         for c in self.vector_db.text_store:
#             doc_id = c.get('doc_id')
#             if doc_id:
#                 doc_map.setdefault(doc_id, []).append(c)

#         # Ensure chunks are in the correct order for expansion
#         for doc_id in doc_map:
#             doc_map[doc_id].sort(key=lambda x: x.get('chunk_index', 0))

#         # --- 6. NEIGHBOR EXPANSION ---
#         expanded_chunks = []
#         DISTANCE_THRESHOLD = 1.6  # Adjust based on embedding model sensitivity

#         for chunk_dict, distance in top_k_results:
#             if distance > DISTANCE_THRESHOLD:
#                 continue

#             doc_id = chunk_dict.get("doc_id")
#             idx = chunk_dict.get("chunk_index", 0)
#             family = doc_map.get(doc_id, [])

#             if not family:
#                 continue

#             # Find actual index in family list in case IDs aren't sequential
#             actual_idx = next((i for i, item in enumerate(family) if item["chunk_id"] == chunk_dict["chunk_id"]), idx)

#             # Expansion: Grab 1 before and 1 after the hit
#             start_idx = max(0, actual_idx - 1)
#             end_idx = min(len(family), actual_idx + 2)

#             for i in range(start_idx, end_idx):
#                 expanded_chunks.append(family[i])

#         # --- 7. DEDUPLICATION + KEYWORD FILTER ---
#         seen_ids = set()
#         final_chunks = []

#         for c in expanded_chunks:
#             if c["chunk_id"] in seen_ids:
#                 continue
#             if not self._passes_keyword_gate(c["text"], literal_term):
#                 continue

#             final_chunks.append(c["text"])
#             seen_ids.add(c["chunk_id"])

#             if len(final_chunks) >= self.max_chunks_to_keep:
#                 break

#         # --- 8. FALLBACK IF FILTERING REMOVED EVERYTHING ---
#         if not final_chunks and chunks_for_db:
#             print(f"{C_RED}[{self.id} FALLBACK] No chunks passed filters. Using raw chunks as fallback.{C_RESET}")
#             final_chunks = [c["text"] for c in chunks_for_db[:self.max_chunks_to_keep]]

#         # --- 9. ASSEMBLE FINAL CONTEXT ---
#         # Combine structured data (priority) with retrieval chunks
#         final_context_list = structured_context + final_chunks
#         state['filtered_context'] = (
#             "\n---\n".join(final_context_list)
#             if final_context_list
#             else "No relevant context found."
#         )

#         # --- 10. FINAL STATE UPDATE ---
#         state['rag_complete'] = True

#         print(f"{C_YELLOW}[{self.id.upper()} STATE] Final context assembled: {len(final_context_list)} sections.{C_RESET}")
#         print(f"{C_GREEN}[{self.id.upper()} DONE] RAG processing complete.{C_RESET}")

#         return state

# ==================================================================================================
# INTEGRATED TEST BLOCK: RETRIEVAL AGENT ONLY
# ==================================================================================================
# --- Mock State Data for Testing ---
# This data simulates the output after the 'tool_agents' (arxiv, pubmed, materials) have run.
MOCK_INITIAL_STATE: ResearchState = {
    "user_query": "A detailed review on the synthesis and bandgap stability of lead-free CsSnI3 perovskite solar cells using computational and experimental data published in the last decade.",
    "semantic_query": "A detailed review on the synthesis and bandgap stability of lead-free CsSnI3 perovskite solar cells using computational and experimental data published in the last decade.",
    "primary_intent": "literature_review",
    "execution_plan": [
        "Step 1: Define the specific parameters for the literature review, focusing on synthesis and bandgap stability of CsSnI3 perovskite solar cells, and set the time frame to the last decade.",
        "Step 2: Use the 'materials' tool to gather data on the material properties and synthesis methods of CsSnI3 perovskite solar cells.",
        "Step 3: Utilize the 'chemrxiv' tool to find preprints related to experimental and computational studies on CsSnI3 solar cells.",
        "Step 4: Search 'arxiv' for relevant articles in Physics and Materials Science that discuss computational models and experimental results for CsSnI3.",
        "Step 5: Compile findings from 'semanticscholar' and 'openalex' to ensure a comprehensive review of peer-reviewed literature and citations related to the topic."
    ],
    "material_elements": [
        "topic: synthesis and bandgap stability of lead-free CsSnI3 perovskite solar cells",
        "time frame: last_decade",
        "specific requirements: using computational and experimental data",
        "CsSnI3",
        "Cs",
        "Sn",
        "I"
    ],
    "system_constraints": [
        "topic: synthesis and bandgap stability of lead-free CsSnI3 perovskite solar cells",
        "time frame: last_decade",
        "specific requirements: using computational and experimental data"
    ],
    "api_search_term": "CsSnI3",
    "tiered_queries": {
        "materials": {
            "simple": "CsSnI3 AND synthesis AND bandgap AND stability"
        },
        "arxiv": {
            "strict": "CsSnI3 AND synthesis AND bandgap AND stability AND lead-free",
            "moderate": "CsSnI3 OR lead-free AND perovskite AND solar cells AND computational AND experimental",
            "broad": "perovskite solar cells AND synthesis AND stability AND computational AND experimental"
        },
        "openalex": {
            "simple": "CsSnI3 AND lead-free AND perovskite AND solar cells"
        },
        "chemrxiv": {
            "simple": "CsSnI3 AND synthesis AND bandgap AND stability"
        },
        "pubmed": {
            "strict": "CsSnI3 AND synthesis AND bandgap AND stability AND lead-free",
            "moderate": "CsSnI3 OR lead-free AND perovskite AND solar cells AND computational AND experimental",
            "broad": "perovskite solar cells AND synthesis AND stability AND computational AND experimental"
        },
        "semanticscholar": {
            "strict": "CsSnI3 AND synthesis AND bandgap stability AND lead-free",
            "moderate": "CsSnI3 OR lead-free AND perovskite AND solar cells AND computational AND experimental"
        }
    },
    "active_tools": [
        "materials",
        "arxiv",
        "openalex",
        "chemrxiv",
        "pubmed",
        "semanticscholar"
    ],
    "raw_tool_data": [
        {
            "text": "Material: CsSnI3 (mp-616378). Stability: Unstable (E/hull: 0.011843993999996002 eV). Band Gap: $0.5537000000000001\\ \text{eV}$. Energy Above Hull: $0.011843993999996002\\ \text{eV}$.",
            "source_type": "materials_project",
            "tool_id": "materials_search",
            "metadata": {
                "material_id": "mp-616378",
                "formula": "CsSnI3",
                "is_stable": False,
                "band_gap": 0.5537000000000001,
                "energy_above_hull": 0.011843993999996002
            }
        },
        {
            "text": "Material: CsSnI3 (mp-614013). Stability: Stable. Band Gap: $0.449999999999999\\ \text{eV}$. Energy Above Hull: $0.0\\ \text{eV}$.",
            "source_type": "materials_project",
            "tool_id": "materials_search",
            "metadata": {
                "material_id": "mp-614013",
                "formula": "CsSnI3",
                "is_stable": True,
                "band_gap": 0.449999999999999,
                "energy_above_hull": 0.0
            }
        },
        {
            "text": "Material: CsSnI3 (mp-27381). Stability: Unstable (E/hull: 0.003249338499999 eV). Band Gap: $2.0632\\ \text{eV}$. Energy Above Hull: $0.003249338499999\\ \text{eV}$.",
            "source_type": "materials_project",
            "tool_id": "materials_search",
            "metadata": {
                "material_id": "mp-27381",
                "formula": "CsSnI3",
                "is_stable": False,
                "band_gap": 2.0632,
                "energy_above_hull": 0.003249338499999
            }
        },
        {
            "text": "Material: CsSnI3 (mp-568570). Stability: Unstable (E/hull: 0.011059161000002002 eV). Band Gap: $0.617499999999999\\ \text{eV}$. Energy Above Hull: $0.011059161000002002\\ \text{eV}$.",
            "source_type": "materials_project",
            "tool_id": "materials_search",
            "metadata": {
                "material_id": "mp-568570",
                "formula": "CsSnI3",
                "is_stable": False,
                "band_gap": 0.617499999999999,
                "energy_above_hull": 0.011059161000002002
            }
        },
        {
            "text": "Title: Full Optoelectronic Simulation of Lead-Free Perovskite/Organic Tandem Solar Cells.. Abstract: Organic and perovskite semiconductor materials are considered an interesting combination thanks to their similar processing technologies and band gap tunability. Here, we present the design and analysis of perovskite/organic tandem solar cells (TSCs) by using a full optoelectronic simulator (SETFOS). A wide band gap lead-free ASnI<sub>2</sub>Br perovskite top subcell is utilized in conjunction with a narrow band gap DPPEZnP-TBO:PC61BM heterojunction organic bottom subcell to form the tandem configuration. The top and bottom cells were designed according to previous experimental work keeping the same materials and physical parameters. The calibration of the two cells regarding simulation and experimental data shows very good agreement, implying the validation of the simulation process. Accordingly, the two cells are combined to develop a 2T tandem cell. Further, upon optimizing the thickness of the front and rear subcells, a current matching condition is satisfied for which the proposed perovskite/organic TSC achieves an efficiency of 13.32%, <i>J<sub>sc</sub></i> of 13.74 mA/cm<sup>2</sup>, and <i>V<sub>oc</sub></i> of 1.486 V. On the other hand, when optimizing the tandem by utilizing full optoelectronic simulation, the tandem shows a higher efficiency of about 14%, although it achieves a decreased <i>J<sub>sc</sub></i> of 12.27 mA/cm<sup>2</sup>. The study shows that the efficiency can be further improved when concurrently optimizing the various tandem layers by global optimization routines. Furthermore, the impact of defects is demonstrated to highlight other possible routes to improve efficiency. The current simulation study can provide a physical understanding and potential directions for further efficiency improvement for lead-free perovskite/organic TSC.",
            "source_type": "pubmed",
            "tool_id": "pubmed_search",
            "metadata": {
                "pmid": "36772085",
                "title": "Full Optoelectronic Simulation of Lead-Free Perovskite/Organic Tandem Solar Cells.",
                "abstract": "Organic and perovskite semiconductor materials are considered an interesting combination thanks to their similar processing technologies and band gap tunability. Here, we present the design and analysis of perovskite/organic tandem solar cells (TSCs) by using a full optoelectronic simulator (SETFOS). A wide band gap lead-free ASnI<sub>2</sub>Br perovskite top subcell is utilized in conjunction with a narrow band gap DPPEZnP-TBO:PC61BM heterojunction organic bottom subcell to form the tandem configuration. The top and bottom cells were designed according to previous experimental work keeping the same materials and physical parameters. The calibration of the two cells regarding simulation and experimental data shows very good agreement, implying the validation of the simulation process. Accordingly, the two cells are combined to develop a 2T tandem cell. Further, upon optimizing the thickness of the front and rear subcells, a current matching condition is satisfied for which the proposed perovskite/organic TSC achieves an efficiency of 13.32%, <i>J<sub>sc</sub></i> of 13.74 mA/cm<sup>2</sup>, and <i>V<sub>oc</sub></i> of 1.486 V. On the other hand, when optimizing the tandem by utilizing full optoelectronic simulation, the tandem shows a higher efficiency of about 14%, although it achieves a decreased <i>J<sub>sc</sub></i> of 12.27 mA/cm<sup>2</sup>. The study shows that the efficiency can be further improved when concurrently optimizing the various tandem layers by global optimization routines. Furthermore, the impact of defects is demonstrated to highlight other possible routes to improve efficiency. The current simulation study can provide a physical understanding and potential directions for further efficiency improvement for lead-free perovskite/organic TSC.",
                "external_id": "36772085",
                "pdf_url": "https://pubmed.ncbi.nlm.nih.gov/36772085/"
            }
        },
        {
            "text": "Title: Numerical Simulation and Experimental Study of Methyl Ammonium Bismuth Iodide Absorber Layer Based Lead Free Perovskite Solar Cells.. Abstract: In the past few years, there has been a significant increase in the development and production of perovskite or perovskite-like materials that do not contain lead (Pb) for the purpose of constructing solar cells. The development and testing of lead-free perovskite-like structures for solar cells is crucial. In this study, we used the solar cell capacitance software (SCAPS) to simulate perovskite solar cells based on methyl ammonium bismuth iodide (MA<sub>3</sub> Bi<sub>2</sub> I<sub>9</sub> ). The electron-transport layer, hole-transport layer, and absorber layer thickness were optimized using SCAPS. The simulated perovskite solar cells (FTO/TiO<sub>2</sub> /MA<sub>3</sub> Bi<sub>2</sub> I<sub>9</sub> /spiro-OMeTAD/Au) performed well with a power conversion efficiency of 14.07\u2009% and a reasonable open circuit voltage of 1.34\u2005V, using the optimized conditions determined by SCAPS. Additionally, we conducted experiments to fabricate perovskite solar cells under controlled humidity, which showed a power conversion efficiency of 1.31\u2009%.",
            "source_type": "pubmed",
            "tool_id": "pubmed_search",
            "metadata": {
                "pmid": "37029556",
                "title": "Numerical Simulation and Experimental Study of Methyl Ammonium Bismuth Iodide Absorber Layer Based Lead Free Perovskite Solar Cells.",
                "abstract": "In the past few years, there has been a significant increase in the development and production of perovskite or perovskite-like materials that do not contain lead (Pb) for the purpose of constructing solar cells. The development and testing of lead-free perovskite-like structures for solar cells is crucial. In this study, we used the solar cell capacitance software (SCAPS) to simulate perovskite solar cells based on methyl ammonium bismuth iodide (MA<sub>3</sub> Bi<sub>2</sub> I<sub>9</sub> ). The electron-transport layer, hole-transport layer, and absorber layer thickness were optimized using SCAPS. The simulated perovskite solar cells (FTO/TiO<sub>2</sub> /MA<sub>3</sub> Bi<sub>2</sub> I<sub>9</sub> /spiro-OMeTAD/Au) performed well with a power conversion efficiency of 14.07\u2009% and a reasonable open circuit voltage of 1.34\u2005V, using the optimized conditions determined by SCAPS. Additionally, we conducted experiments to fabricate perovskite solar cells under controlled humidity, which showed a power conversion efficiency of 1.31\u2009%.",
                "external_id": "37029556",
                "pdf_url": "https://pubmed.ncbi.nlm.nih.gov/37029556/"
            }
        },
        {
            "text": "Title: Lead-Free Organic-Inorganic Hybrid Perovskites for Photovoltaic Applications: Recent Advances and Perspectives.. Abstract: Organic-inorganic hybrid halide perovskites (e.g., MAPbI<sub>3</sub> ) have recently emerged as novel active materials for photovoltaic applications with power conversion efficiency over 22%. Conventional perovskite solar cells (PSCs); however, suffer the issue that lead is toxic to the environment and organisms for a long time and is hard to excrete from the body. Therefore, it is imperative to find environmentally-friendly metal ions to replace lead for the further development of PSCs. Previous work has demonstrated that Sn, Ge, Cu, Bi, and Sb ions could be used as alternative ions in perovskite configurations to form a new environmentally-friendly lead-free perovskite structure. Here, we review recent progress on lead-free PSCs in terms of the theoretical insight and experimental explorations of the crystal structure of lead-free perovskite, thin film deposition, and device performance. We also discuss the importance of obtaining further understanding of the fundamental properties of lead-free hybrid perovskites, especially those related to photophysics.",
            "source_type": "pubmed",
            "tool_id": "pubmed_search",
            "metadata": {
                "pmid": "28160346",
                "title": "Lead-Free Organic-Inorganic Hybrid Perovskites for Photovoltaic Applications: Recent Advances and Perspectives.",
                "abstract": "Organic-inorganic hybrid halide perovskites (e.g., MAPbI<sub>3</sub> ) have recently emerged as novel active materials for photovoltaic applications with power conversion efficiency over 22%. Conventional perovskite solar cells (PSCs); however, suffer the issue that lead is toxic to the environment and organisms for a long time and is hard to excrete from the body. Therefore, it is imperative to find environmentally-friendly metal ions to replace lead for the further development of PSCs. Previous work has demonstrated that Sn, Ge, Cu, Bi, and Sb ions could be used as alternative ions in perovskite configurations to form a new environmentally-friendly lead-free perovskite structure. Here, we review recent progress on lead-free PSCs in terms of the theoretical insight and experimental explorations of the crystal structure of lead-free perovskite, thin film deposition, and device performance. We also discuss the importance of obtaining further understanding of the fundamental properties of lead-free hybrid perovskites, especially those related to photophysics.",
                "external_id": "28160346",
                "pdf_url": "https://pubmed.ncbi.nlm.nih.gov/28160346/"
            }
        },
        {
            "text": "Title: Germanium-Based Halide Perovskites: Materials, Properties, and Applications.. Abstract: Perovskites are attracting an increasing interest in the wide community of photovoltaics, optoelectronic, and detection, traditionally relying on lead-based systems. This Minireview provides an overview of the current status of experimental and computational results available on Ge-containing 3D and low-dimensional halide perovskites. While stability issues analogous to those of tin-based materials are present, some strategies to afford this problem in Ge metal halide perovskites (MHPs) for photovoltaics have already been identified and successfully employed, reaching efficiencies of solar devices greater than 7\u2009% at up to 500\u2005h of illumination. Interestingly, some Ge-containing MHPs showed promising nonlinear optical responses as well as quite broad emissions, which are worthy of further investigation starting from the basic materials chemistry perspective, where a large space for properties modulation through compositions/alloying/fnanostructuring is present.",
            "source_type": "pubmed",
            "tool_id": "pubmed_search",
            "metadata": {
                "pmid": "34126001",
                "title": "Germanium-Based Halide Perovskites: Materials, Properties, and Applications.",
                "abstract": "Perovskites are attracting an increasing interest in the wide community of photovoltaics, optoelectronic, and detection, traditionally relying on lead-based systems. This Minireview provides an overview of the current status of experimental and computational results available on Ge-containing 3D and low-dimensional halide perovskites. While stability issues analogous to those of tin-based materials are present, some strategies to afford this problem in Ge metal halide perovskites (MHPs) for photovoltaics have already been identified and successfully employed, reaching efficiencies of solar devices greater than 7\u2009% at up to 500\u2005h of illumination. Interestingly, some Ge-containing MHPs showed promising nonlinear optical responses as well as quite broad emissions, which are worthy of further investigation starting from the basic materials chemistry perspective, where a large space for properties modulation through compositions/alloying/fnanostructuring is present.",
                "external_id": "34126001",
                "pdf_url": "https://pubmed.ncbi.nlm.nih.gov/34126001/"
            }
        },
        {
            "text": "Title: Numerical optimization of Rb<sub>2</sub>AuScBr<sub>6</sub> and Rb<sub>2</sub>AuScCl<sub>6</sub>-based lead-free perovskite solar cells: device engineering and performance mapping.. Abstract: Perovskite solar cells (PSCs) exhibit significant potential for next-generation photovoltaic technology, integrating high power conversion efficiency (PCE), cost-effectiveness, and tunable optoelectronic properties. This report presents a comprehensive numerical optimization of Rb<sub>2</sub>AuScBr<sub>6</sub> and Rb<sub>2</sub>AuScCl<sub>6</sub>-based PSCs, with particular emphasis on the influence of electron transport layers (ETLs) and critical device parameters. The configuration ITO/TiO<sub>2</sub>/Rb<sub>2</sub>AuScBr<sub>6</sub>/CBTS/Ni achieves a PCE of 27.49%, whereas the configuration ITO/WS<sub>2</sub>/Rb<sub>2</sub>AuScCl<sub>6</sub>/CBTS/Ni attains 22.41%, thereby underscoring the high efficiency of these lead-free materials. Device performance is markedly improved through increased perovskite layer thickness and reduced defect density. Further stabilization of performance is achieved by optimizing electron affinity, series resistance, and shunt resistance. Additionally, thermal stability is enhanced through the adjustment of operational temperature. The superior PCE observed in Rb<sub>2</sub>AuScBr<sub>6</sub> is ascribed to the selection of the ETL, an optimal band gap, absorber layer thickness, lower defect density, and appropriate contact interfaces. Overall, these Rb<sub>2</sub>AuScBr<sub>6</sub> and Rb<sub>2</sub>AuScCl<sub>6</sub> perovskites demonstrate exceptional promise for practical, efficient, and stable PSC applications, thereby encouraging further experimental validation and device engineering.",
            "source_type": "pubmed",
            "tool_id": "pubmed_search",
            "metadata": {
                "pmid": "41268489",
                "title": "Numerical optimization of Rb<sub>2</sub>AuScBr<sub>6</sub> and Rb<sub>2</sub>AuScCl<sub>6</sub>-based lead-free perovskite solar cells: device engineering and performance mapping.",
                "abstract": "Perovskite solar cells (PSCs) exhibit significant potential for next-generation photovoltaic technology, integrating high power conversion efficiency (PCE), cost-effectiveness, and tunable optoelectronic properties. This report presents a comprehensive numerical optimization of Rb<sub>2</sub>AuScBr<sub>6</sub> and Rb<sub>2</sub>AuScCl<sub>6</sub>-based PSCs, with particular emphasis on the influence of electron transport layers (ETLs) and critical device parameters. The configuration ITO/TiO<sub>2</sub>/Rb<sub>2</sub>AuScBr<sub>6</sub>/CBTS/Ni achieves a PCE of 27.49%, whereas the configuration ITO/WS<sub>2</sub>/Rb<sub>2</sub>AuScCl<sub>6</sub>/CBTS/Ni attains 22.41%, thereby underscoring the high efficiency of these lead-free materials. Device performance is markedly improved through increased perovskite layer thickness and reduced defect density. Further stabilization of performance is achieved by optimizing electron affinity, series resistance, and shunt resistance. Additionally, thermal stability is enhanced through the adjustment of operational temperature. The superior PCE observed in Rb<sub>2</sub>AuScBr<sub>6</sub> is ascribed to the selection of the ETL, an optimal band gap, absorber layer thickness, lower defect density, and appropriate contact interfaces. Overall, these Rb<sub>2</sub>AuScBr<sub>6</sub> and Rb<sub>2</sub>AuScCl<sub>6</sub> perovskites demonstrate exceptional promise for practical, efficient, and stable PSC applications, thereby encouraging further experimental validation and device engineering.",
                "external_id": "41268489",
                "pdf_url": "https://pubmed.ncbi.nlm.nih.gov/41268489/"
            }
        },
        {
            "text": "Title: Lead Free Perovskites. Abstract: One of the most viable renewable energies is solar power because of its versatility, reliability, and abundance. In the market, a majority of the solar panels are made from silicon wafers. These solar panels have an efficiency of 26.4 percent and can last more than 25 years. The perovskite solar cell is a relatively new type of solar technology that has a similar maximum efficiency and much cheaper costs, the only downside is that it is less stable and the most efficient type uses lead. The name",
            "source_type": "arxiv",
            "tool_id": "arxiv_search",
            "metadata": {
                "arxiv_id": "http://arxiv.org/abs/2407.17520v2",
                "title": "Lead Free Perovskites",
                "abstract": "One of the most viable renewable energies is solar power because of its versatility, reliability, and abundance. In the market, a majority of the solar panels are made from silicon wafers. These solar panels have an efficiency of 26.4 percent and can last more than 25 years. The perovskite solar cell is a relatively new type of solar technology that has a similar maximum efficiency and much cheaper costs, the only downside is that it is less stable and the most efficient type uses lead. The name",
                "published_year": 2024,
                "pdf_url": "https://arxiv.org/pdf/2407.17520v2"
            }
        },
        {
            "text": "Title: Unveiling architectural and optoelectronic synergies in lead-free perovskite/perovskite/kesterite triple-junction monolithic tandem solar cells. Abstract: The widespread use of lead-based materials in tandem solar cells raises critical environmental and health concerns due to their inherent toxicity and risk of contamination. To address this challenge, we focused on lead-free tandem architectures based on non-toxic, environmentally benign materials such as tin-based perovskites and kesterites, which are essential for advancing sustainable photovoltaic technologies. In this study, we present the proposition, design, and optimization of two distinct",
            "source_type": "arxiv",
            "tool_id": "arxiv_search",
            "metadata": {
                "arxiv_id": "http://arxiv.org/abs/2511.06059v2",
                "title": "Unveiling architectural and optoelectronic synergies in lead-free perovskite/perovskite/kesterite triple-junction monolithic tandem solar cells",
                "abstract": "The widespread use of lead-based materials in tandem solar cells raises critical environmental and health concerns due to their inherent toxicity and risk of contamination. To address this challenge, we focused on lead-free tandem architectures based on non-toxic, environmentally benign materials such as tin-based perovskites and kesterites, which are essential for advancing sustainable photovoltaic technologies. In this study, we present the proposition, design, and optimization of two distinct",
                "published_year": 2025,
                "pdf_url": "https://arxiv.org/pdf/2511.06059v2"
            }
        },
        {
            "text": "Title: Efficient Passivation of Surface Defects by Lewis Base in Lead-free Tin-based Perovskite Solar Cells. Abstract: Lead-free tin-based perovskites are highly appealing for the next generation of solar cells due to their intriguing optoelectronic properties. However, the tendency of Sn2+ oxidation to Sn4+ in the tin-based perovskites induces serious film degradation and performance deterioration. Herein, we demonstrate, through the density functional theory based first-principle calculations in a surface slab model, that the surface defects of the Sn-based perovskite FASnI3 (FA = NH2CHNH2+) could be effective",
            "source_type": "arxiv",
            "tool_id": "arxiv_search",
            "metadata": {
                "arxiv_id": "http://arxiv.org/abs/2206.06782v1",
                "title": "Efficient Passivation of Surface Defects by Lewis Base in Lead-free Tin-based Perovskite Solar Cells",
                "abstract": "Lead-free tin-based perovskites are highly appealing for the next generation of solar cells due to their intriguing optoelectronic properties. However, the tendency of Sn2+ oxidation to Sn4+ in the tin-based perovskites induces serious film degradation and performance deterioration. Herein, we demonstrate, through the density functional theory based first-principle calculations in a surface slab model, that the surface defects of the Sn-based perovskite FASnI3 (FA = NH2CHNH2+) could be effective",
                "published_year": 2022,
                "pdf_url": "https://arxiv.org/pdf/2206.06782v1"
            }
        },
        {
            "text": "Title: Optimization and Performance Evaluation of Cs$_2$CuBiCl$_6$ Double Perovskite Solar Cell for Lead-Free Photovoltaic Applications. Abstract: In the previous decade, there has been a significant advancement in the performance of perovskite solar cells (PSCs), characterized by a notable increase in efficiency from 3.8% to 25%. Nonetheless, PSCs face many problems when we commercialize them because of their toxicity and stability. Consequently, lead-PSCs need an alternative solar cell with high performance and low processing cost; lead-free inorganic perovskites have been explored. Recent research showcased Cs$_2$CuBiCl$_6$, a lead-free",
            "source_type": "arxiv",
            "tool_id": "arxiv_search",
            "metadata": {
                "arxiv_id": "http://arxiv.org/abs/2502.16850v1",
                "title": "Optimization and Performance Evaluation of Cs$_2$CuBiCl$_6$ Double Perovskite Solar Cell for Lead-Free Photovoltaic Applications",
                "abstract": "In the previous decade, there has been a significant advancement in the performance of perovskite solar cells (PSCs), characterized by a notable increase in efficiency from 3.8% to 25%. Nonetheless, PSCs face many problems when we commercialize them because of their toxicity and stability. Consequently, lead-PSCs need an alternative solar cell with high performance and low processing cost; lead-free inorganic perovskites have been explored. Recent research showcased Cs$_2$CuBiCl$_6$, a lead-free",
                "published_year": 2025,
                "pdf_url": "https://arxiv.org/pdf/2502.16850v1"
            }
        },
        {
            "text": "Title: Exploring Lead Free Mixed Halide Double Perovskites Solar Cell. Abstract: The significant surge in energy use and escalating environmental concerns have sparked worldwide interest towards the study and implementation of solar cell technology. Perovskite solar cells (PSCs) have garnered remarkable attention as an emerging third-generation solar cell technology. This paper presents an in-depth analysis of lead-free mixed halide double perovskites in the context of their potential uses in solar cell technology. Through the previous studies of various mixed halide double ",
            "source_type": "arxiv",
            "tool_id": "arxiv_search",
            "metadata": {
                "arxiv_id": "http://arxiv.org/abs/2401.09584v1",
                "title": "Exploring Lead Free Mixed Halide Double Perovskites Solar Cell",
                "abstract": "The significant surge in energy use and escalating environmental concerns have sparked worldwide interest towards the study and implementation of solar cell technology. Perovskite solar cells (PSCs) have garnered remarkable attention as an emerging third-generation solar cell technology. This paper presents an in-depth analysis of lead-free mixed halide double perovskites in the context of their potential uses in solar cell technology. Through the previous studies of various mixed halide double ",
                "published_year": 2024,
                "pdf_url": "https://arxiv.org/pdf/2401.09584v1"
            }
        },
        {
            "text": "Title: From unstable CsSnI3 to air-stable Cs2SnI6: A lead-free perovskite solar cell light absorber with bandgap of 1.48 eV and high absorption coefficient. Abstract: Abstract unavailable.",
            "source_type": "openalex",
            "tool_id": "openalex_search",
            "metadata": {
                "openalex_id": "https://openalex.org/W2523562186",
                "title": "From unstable CsSnI3 to air-stable Cs2SnI6: A lead-free perovskite solar cell light absorber with bandgap of 1.48 eV and high absorption coefficient",
                "pdf_url": "https://doi.org/10.1016/j.solmat.2016.09.022"
            }
        },
        {
            "text": "Title: Investigation of photovoltaic performance of lead-free CsSnI3-based perovskite solar cell with different hole transport layers: First Principle Calculations and SCAPS-1D Analysis. Abstract: Abstract unavailable.",
            "source_type": "openalex",
            "tool_id": "openalex_search",
            "metadata": {
                "openalex_id": "https://openalex.org/W4311419982",
                "title": "Investigation of photovoltaic performance of lead-free CsSnI3-based perovskite solar cell with different hole transport layers: First Principle Calculations and SCAPS-1D Analysis",
                "pdf_url": "https://doi.org/10.1016/j.solener.2022.11.025"
            }
        },
        {
            "text": "Title: SCAPS-1D Simulation for Device Optimization to Improve Efficiency in Lead-Free CsSnI3 Perovskite Solar Cells. Abstract: In this study, a novel systematic analysis was conducted to explore the impact of various parameters, including acceptor density (NA), individual layer thickness, defect density, interface defect density, and the metal electrode work function, on efficiency within the FTO/ZnO/CsSnI3/NiOx/Au perovskite solar cell structure through the SCAPS-1D (Solar Cell Capacitance Simulator in 1 Dimension) simulation. ZnO served as the electron transport layer (ETL), CsSnI3 as the perovskite absorption layer (PAL), and NiOx as the hole transport layer (HTL), all contributing to the optimization of device performance. To achieve the optimal power conversion efficiency (PCE), we determined the ideal PAL acceptor density (NA) to be 2 \u00d7 1019 cm\u22123 and the optimal thicknesses to be 20 nm for the ETL (ZnO), 700 nm for the PAL (CsSnI3), and 10 nm for the HTL (NiOx), with the metal electrode remaining as Au. As a result of the optimization process, efficiency increased from 11.89% to 23.84%. These results are expected to contribute to the performance enhancement of eco-friendly, lead-free inorganic hybrid solar cells with Sn-based perovskite as the PAL.",
            "source_type": "openalex",
            "tool_id": "openalex_search",
            "metadata": {
                "openalex_id": "https://openalex.org/W4395003215",
                "title": "SCAPS-1D Simulation for Device Optimization to Improve Efficiency in Lead-Free CsSnI3 Perovskite Solar Cells",
                "pdf_url": "https://doi.org/10.3390/inorganics12040123"
            }
        },
        {
            "text": "Title: 20.730% highly efficient lead-free CsSnI3-based perovskite solar cells with various charge transport materials: a SCAPS-1D study. Abstract: Abstract unavailable.",
            "source_type": "openalex",
            "tool_id": "openalex_search",
            "metadata": {
                "openalex_id": "https://openalex.org/W4405581837",
                "title": "20.730% highly efficient lead-free CsSnI3-based perovskite solar cells with various charge transport materials: a SCAPS-1D study",
                "pdf_url": "https://doi.org/10.1007/s41939-024-00701-2"
            }
        },
        {
            "text": "Title: Nitrogen-doped titanium dioxide as a novel eco-friendly hole transport layer in lead-free CsSnI3 based perovskite solar cells. Abstract: Abstract unavailable.",
            "source_type": "openalex",
            "tool_id": "openalex_search",
            "metadata": {
                "openalex_id": "https://openalex.org/W4389805580",
                "title": "Nitrogen-doped titanium dioxide as a novel eco-friendly hole transport layer in lead-free CsSnI3 based perovskite solar cells",
                "pdf_url": "https://doi.org/10.1016/j.materresbull.2023.112642"
            }
        },
        {
            "text": "Title: An Absorber Enrichment Study and Implications on the Performance of Lead-Free CsSnI3 Perovskite Solar Cells (PSCs) Using One-Dimensional Solar Cell Capacitance Simulator (1D-SCAPS). Abstract: Abstract unavailable.",
            "source_type": "openalex",
            "tool_id": "openalex_search",
            "metadata": {
                "openalex_id": "https://openalex.org/W4390944690",
                "title": "An Absorber Enrichment Study and Implications on the Performance of Lead-Free CsSnI3 Perovskite Solar Cells (PSCs) Using One-Dimensional Solar Cell Capacitance Simulator (1D-SCAPS)",
                "pdf_url": "https://doi.org/10.1007/s13538-023-01406-6"
            }
        },
        {
            "text": "Title: Impact of Hole Transport Layers in Inorganic Lead-Free B-\u03b3-CsSnI3 Perovskite Solar Cells: A Numerical Analysis. Abstract: Tin-based halide perovskite compounds have attracted enormous interest as effective replacements for the conventional lead halide perovskite solar cells (PCSs). However, achieving high efficiency for tin-based perovskite solar cells is still challenging. Herein, we introduced copper sulfide (CuS) as a hole transport material (HTM) in lead free tin-based B-&gamma;-CsSnI3 PSCs to enhance the photovoltaic (PV) performances. The lead free tin-based CsSnI3 perovskite solar cell structure consisting of CuS/CsSnI3/TiO2/ITO was modeled and the output characteristics were investigated by using the one dimensional solar cell capacitance simulator (SCAPS-1D). The CuS hole transport layer (HTL) with proper band arrangement may notably minimize the recombination of the charge carrier at the back side of the perovskite absorber. Density functional theory (DFT)-extracted physical parameters including the band gap and absorption spectrum of CuS were used in the SCAPS-1D program to analyze the characteristics of the proposed PV device. The PV performance parameters of the proposed device were numerically evaluated by varying the absorber thickness and doping concentration. In this work, the variation of the functional temperature on the cell outputs was also studied. Furthermore, different HTMs were employed to investigate the PV characteristics of the proposed CsSnI3 PSC. The power conversion efficiency (PCE) of ~29% was achieved with open circuit voltage (Voc) of 0.99 V, a fill factor of ~87%, and short circuit current density (Jsc) of 33.5 mA/cm2 for the optimized device. This work addressed guidelines and introduced a convenient approach to design and fabricate highly efficient, inexpensive, and stable lead free tin-based perovskite solar cells.",
            "source_type": "openalex",
            "tool_id": "openalex_search",
            "metadata": {
                "openalex_id": "https://openalex.org/W4286726627",
                "title": "Impact of Hole Transport Layers in Inorganic Lead-Free B-\u03b3-CsSnI3 Perovskite Solar Cells: A Numerical Analysis",
                "pdf_url": "https://doi.org/10.3390/ecp2022-12611"
            }
        },
        {
            "text": "Title: Device Engineering of a Novel Lead-Free Solar Cell Architecture Utilizing Inorganic CsSnCl3 and CsSnI3 Perovskite-Based Dual Absorbers for Sustainable Powering of Wireless Networks. Abstract: Abstract unavailable.",
            "source_type": "openalex",
            "tool_id": "openalex_search",
            "metadata": {
                "openalex_id": "https://openalex.org/W4404740769",
                "title": "Device Engineering of a Novel Lead-Free Solar Cell Architecture Utilizing Inorganic CsSnCl3 and CsSnI3 Perovskite-Based Dual Absorbers for Sustainable Powering of Wireless Networks",
                "pdf_url": "https://doi.org/10.1007/s11664-024-11605-9"
            }
        },
        {
            "text": "Title: Nitrogen-doped Titanium Dioxide as a novel eco-friendly Hole Transport Layer in Lead-Free CsSnI3 based Perovskite Solar Cells. Abstract: Abstract Despite recent abrupt rise in the efficiency of perovskite solar cells (PSCs), the contact layers maybe limit the efficiency of PSCs. The hole transporting layer (HTL) is an essential layer for reducing the recombination and loosing charges in fabricated devices by avoiding direct contact of gold to perovskite absorber layer in an efficient PSC device. The pristine spiro-OMeTAD, as most widely used HTL, still suffers from poor electrical conductivity, low hole mobility, and low oxidation rate. In this research, the nitrogen doped TiO 2 (N-TiO 2 ) proposed as a low-cost, efficient, safe replacement for spiro-OMeTAD HTL in PSCs. The variation in the device design key parameters such as the thickness and bulk defect density of perovskite layer, simultaneous modifications of defect density and defect energy level, and acceptor doping concentration in absorber layer are examined with their impact on the photovoltaic characteristic parameters. The effect of an increase in operating temperature from 280 K to 460 K on the performance of CsSnI 3 -based perovskite devices is also investigated. The standard simulated lead-free CsSnI 3 \u2013based PSCs with spiro-OMeTAD HTL by SCAPS-1D software revealed the highest power conservation efficiency (PCE) of 23.63%. The CsSnI 3 -based solar cell with N-TiO 2 as HTL showed FF (79.65%), V OC (0.98 V), J sc (34.69 mA/cm 2 ), and efficiency (27.03%) higher than the standard device with conventional spiro-OMeTAD HTL. The outcomes of N-TiO 2 presence as an HTL signify a critical avenue for the possibility of fabricating high PCE CsSnI 3 -based perovskite devices made of stable, low-cost, efficient, safe, and eco-friendly materials.",
            "source_type": "openalex",
            "tool_id": "openalex_search",
            "metadata": {
                "openalex_id": "https://openalex.org/W4385704474",
                "title": "Nitrogen-doped Titanium Dioxide as a novel eco-friendly Hole Transport Layer in Lead-Free CsSnI3 based Perovskite Solar Cells",
                "pdf_url": "http://dx.doi.org/10.21203/rs.3.rs-3185005/v1"
            }
        },
        {
            "text": "Title: Performance enhancement of lead-free CsSnI3 and CsSnCl3 perovskite solar cells by tuning layer interfaces. Abstract: Abstract unavailable.",
            "source_type": "openalex",
            "tool_id": "openalex_search",
            "metadata": {
                "openalex_id": "https://openalex.org/W4414558869",
                "title": "Performance enhancement of lead-free CsSnI3 and CsSnCl3 perovskite solar cells by tuning layer interfaces",
                "pdf_url": "https://doi.org/10.1007/s11082-025-08461-0"
            }
        }
    ],
    "full_text_chunks": [],
    "rag_complete": False,
    "filtered_context": "",
    "references": [
        "\u269b\ufe0f Materials Project: mp-616378 (CsSnI3)",
        "\u269b\ufe0f Materials Project: mp-614013 (CsSnI3)",
        "\u269b\ufe0f Materials Project: mp-27381 (CsSnI3)",
        "\u269b\ufe0f Materials Project: mp-568570 (CsSnI3)",
        "\ud83d\udcc4 Journal Article: Full Optoelectronic Simulation of Lead-Free Perovskite/Organic Tandem Solar Cells.",
        "\ud83d\udcc4 Journal Article: Numerical Simulation and Experimental Study of Methyl Ammonium Bismuth Iodide Absorber Layer Based Lead Free Perovskite Solar Cells.",
        "\ud83d\udcc4 Journal Article: Lead-Free Organic-Inorganic Hybrid Perovskites for Photovoltaic Applications: Recent Advances and Perspectives.",
        "\ud83d\udcc4 Journal Article: Germanium-Based Halide Perovskites: Materials, Properties, and Applications.",
        "\ud83d\udcc4 Journal Article: Numerical optimization of Rb<sub>2</sub>AuScBr<sub>6</sub> and Rb<sub>2</sub>AuScCl<sub>6</sub>-based lead-free perovskite solar cells: device engineering and performance mapping.",
        "\ud83d\udd17 Arxiv: Lead Free Perovskites",
        "\ud83d\udd17 Arxiv: Unveiling architectural and optoelectronic synergies in lead-free perovskite/perovskite/kesterite triple-junction monolithic tandem solar cells",
        "\ud83d\udd17 Arxiv: Efficient Passivation of Surface Defects by Lewis Base in Lead-free Tin-based Perovskite Solar Cells",
        "\ud83d\udd17 Arxiv: Optimization and Performance Evaluation of Cs$_2$CuBiCl$_6$ Double Perovskite Solar Cell for Lead-Free Photovoltaic Applications",
        "\ud83d\udd17 Arxiv: Exploring Lead Free Mixed Halide Double Perovskites Solar Cell",
        "\ud83d\udd17 OpenAlex: From unstable CsSnI3 to air-stable Cs2SnI6: A lead-free perovskite solar cell light absorber with bandgap of 1.48 eV and high absorption coefficient",
        "\ud83d\udd17 OpenAlex: Investigation of photovoltaic performance of lead-free CsSnI3-based perovskite solar cell with different hole transport layers: First Principle Calculations and SCAPS-1D Analysis",
        "\ud83d\udd17 OpenAlex: SCAPS-1D Simulation for Device Optimization to Improve Efficiency in Lead-Free CsSnI3 Perovskite Solar Cells",
        "\ud83d\udd17 OpenAlex: 20.730% highly efficient lead-free CsSnI3-based perovskite solar cells with various charge transport materials: a SCAPS-1D study",
        "\ud83d\udd17 OpenAlex: Nitrogen-doped titanium dioxide as a novel eco-friendly hole transport layer in lead-free CsSnI3 based perovskite solar cells",
        "\ud83d\udd17 OpenAlex: An Absorber Enrichment Study and Implications on the Performance of Lead-Free CsSnI3 Perovskite Solar Cells (PSCs) Using One-Dimensional Solar Cell Capacitance Simulator (1D-SCAPS)",
        "\ud83d\udd17 OpenAlex: Impact of Hole Transport Layers in Inorganic Lead-Free B-\u03b3-CsSnI3 Perovskite Solar Cells: A Numerical Analysis",
        "\ud83d\udd17 OpenAlex: Device Engineering of a Novel Lead-Free Solar Cell Architecture Utilizing Inorganic CsSnCl3 and CsSnI3 Perovskite-Based Dual Absorbers for Sustainable Powering of Wireless Networks",
        "\ud83d\udd17 OpenAlex: Nitrogen-doped Titanium Dioxide as a novel eco-friendly Hole Transport Layer in Lead-Free CsSnI3 based Perovskite Solar Cells",
        "\ud83d\udd17 OpenAlex: Performance enhancement of lead-free CsSnI3 and CsSnCl3 perovskite solar cells by tuning layer interfaces"
    ],
    "final_report": "",
    "report_generated": False,
    "needs_refinement": False,
    "refinement_reason": "",
    "is_refining": False,
    "refinement_retries": 0,
    "next": "",
    "visited_nodes": [
        "intent_agent",
        "planning_agent",
        "query_gen_agent",
        "materials_search",
        "pubmed_search",
        "arxiv_search",
        "openalex_search"
    ]
}

def test_retrieval_agent():
    """
    Isolated test for RetrievalAgent.
    Verifies:
    - PDF / HTML / abstract fallback
    - Chunk creation
    - Deduplication
    - State updates
    """

    print(f"{C_CYAN}*** STARTING RETRIEVAL AGENT TEST ***{C_RESET}")

    # --- 1. SETUP ---
    try:
        state = MOCK_INITIAL_STATE
        retrieval_agent = RetrievalAgent(chunk_size=500)

    except Exception as e:
        print(f"{C_RED}[SETUP FAILED] {type(e).__name__}: {e}{C_RESET}")
        return

    # --- 2. EXECUTE RETRIEVAL AGENT ---
    print(f"\n{C_MAGENTA}--- EXECUTING RETRIEVAL AGENT ---{C_RESET}")
    try:
        state = retrieval_agent.execute(state)

    except Exception as e:
        print(f"{C_RED}[TEST FAILURE] RetrievalAgent crashed: {type(e).__name__}: {e}{C_RESET}")
        return

    # --- 3. ASSERTIONS ---
    chunks = state.get("full_text_chunks", [])

    # ASSERTION 1: Chunks exist
    if not chunks:
        print(f"{C_RED}[TEST FAILURE] No chunks were generated.{C_RESET}")
        return
    else:
        print(f"{C_GREEN}[PASS] Generated {len(chunks)} total chunks.{C_RESET}")

    # ASSERTION 2: Chunks are well-formed
    required_keys = {"chunk_id", "doc_id", "chunk_index", "text", "source", "url"}
    malformed = [
        c for c in chunks
        if not isinstance(c, dict) or not required_keys.issubset(c.keys())
    ]

    if malformed:
        print(f"{C_RED}[TEST FAILURE] Found malformed chunks: {len(malformed)}{C_RESET}")
        return
    else:
        print(f"{C_GREEN}[PASS] All chunks have required schema.{C_RESET}")

    # ASSERTION 3: Deduplication by doc_id
    doc_ids = [c["doc_id"] for c in chunks if c.get("doc_id")]
    if len(doc_ids) != len(set(doc_ids)):
        print(f"{C_YELLOW}[WARN] Duplicate doc_ids detected (expected if multiple chunks per doc).{C_RESET}")
    else:
        print(f"{C_GREEN}[PASS] No duplicate documents processed unexpectedly.{C_RESET}")

    # ASSERTION 4: Abstract fallback worked (non-PDF sources)
    abstract_chunks = [
        c for c in chunks
        if "abstract" in c["chunk_id"].lower() or "_abs_" in c["chunk_id"]
    ]

    if abstract_chunks:
        print(f"{C_GREEN}[PASS] Abstract fallback activated ({len(abstract_chunks)} chunks).{C_RESET}")
    else:
        print(f"{C_YELLOW}[WARN] No abstract fallback chunks detected. PDFs may have succeeded.{C_RESET}")

    # ASSERTION 5: Irrelevant PDFs didn’t crash pipeline
    irrelevant_markers = ["cancer", "quantum error", "3d geometric"]
    crash_markers = [
        c for c in chunks
        if any(m in c["text"].lower() for m in irrelevant_markers)
    ]

    if crash_markers:
        print(f"{C_BLUE}[INFO] Irrelevant content present (expected before RAG filtering).{C_RESET}")
    else:
        print(f"{C_GREEN}[PASS] No irrelevant text surfaced aggressively.{C_RESET}")

    # --- 4. VISITED NODE TRACE ---
    visited = state.get("visited_nodes", [])
    if "retrieval_agent" in visited:
        print(f"{C_GREEN}[PASS] Breadcrumb tracking verified.{C_RESET}")
    else:
        print(f"{C_RED}[FAIL] Breadcrumb tracking missing retrieval_agent.{C_RESET}")

    print(f"\n{C_CYAN}*** RETRIEVAL AGENT TEST COMPLETE ***{C_RESET}")
    #print(json.dumps(state, indent=4)) # Uncomment to see full final state
    with open("./rag_tool.txt", 'w', encoding='utf-8') as f:
        f.write(json.dumps(state, indent=4, default=lambda x: str(x)))


if __name__ == "__main__":
    test_retrieval_agent()

# ==================================================================================================
# INTEGRATED TEST BLOCK: RAG AGENT ONLY
# ==================================================================================================

# def test_rag_agent_only():
#     """
#     Isolated test for RAGAgent.
#     Verifies:
#     - Vector indexing
#     - Semantic filtering
#     - Reranking
#     - Neighbor expansion
#     - Structured data preservation
#     """

#     print(f"{C_CYAN}*** STARTING RAG AGENT ONLY TEST ***{C_RESET}")

#     # --- 1. SETUP ---
#     try:
#         vector_db = VectorDBWrapper()
#         rag_agent = RAGAgent(vector_db=vector_db)

#         # --- MOCK STATE (NO RETRIEVAL STEP) ---
#         state = ResearchState({
#     "user_query": "A detailed review on the synthesis and bandgap stability of lead-free CsSnI3 perovskite solar cells using computational and experimental data published in the last decade.",
#     "semantic_query": "A detailed review on the synthesis and bandgap stability of lead-free CsSnI3 perovskite solar cells using computational and experimental data published in the last decade.",
#     "primary_intent": "literature_review",
#     "execution_plan": [
#         "Step 1: Define the specific parameters for the literature review, focusing on synthesis and bandgap stability of CsSnI3 perovskite solar cells, and set the time frame to the last decade.",
#         "Step 2: Use the 'materials' tool to gather data on the material properties and synthesis methods of CsSnI3 perovskite solar cells.",
#         "Step 3: Utilize the 'chemrxiv' tool to find preprints related to experimental and computational studies on CsSnI3 solar cells.",
#         "Step 4: Search 'arxiv' for relevant articles in Physics and Materials Science that discuss computational models and experimental results for CsSnI3.",
#         "Step 5: Compile findings from 'semanticscholar' and 'openalex' to ensure a comprehensive review of peer-reviewed literature and citations related to the topic."
#     ],
#     "material_elements": [
#         "topic: synthesis and bandgap stability of lead-free CsSnI3 perovskite solar cells",
#         "time frame: last_decade",
#         "specific requirements: using computational and experimental data",
#         "CsSnI3",
#         "Cs",
#         "Sn",
#         "I"
#     ],
#     "system_constraints": [
#         "topic: synthesis and bandgap stability of lead-free CsSnI3 perovskite solar cells",
#         "time frame: last_decade",
#         "specific requirements: using computational and experimental data"
#     ],
#     "api_search_term": "CsSnI3",
#     "tiered_queries": {
#         "materials": {
#             "simple": "CsSnI3 AND synthesis AND bandgap AND stability"
#         },
#         "arxiv": {
#             "strict": "CsSnI3 AND synthesis AND bandgap AND stability AND lead-free",
#             "moderate": "CsSnI3 OR lead-free AND perovskite AND solar cells AND computational AND experimental",
#             "broad": "perovskite solar cells AND synthesis AND stability AND computational AND experimental"
#         },
#         "openalex": {
#             "simple": "CsSnI3 AND lead-free AND perovskite AND solar cells"
#         },
#         "chemrxiv": {
#             "simple": "CsSnI3 AND synthesis AND bandgap AND stability"
#         },
#         "pubmed": {
#             "strict": "CsSnI3 AND synthesis AND bandgap AND stability AND lead-free",
#             "moderate": "CsSnI3 OR lead-free AND perovskite AND solar cells AND computational AND experimental",
#             "broad": "perovskite solar cells AND synthesis AND stability AND computational AND experimental"
#         },
#         "semanticscholar": {
#             "strict": "CsSnI3 AND synthesis AND bandgap stability AND lead-free",
#             "moderate": "CsSnI3 OR lead-free AND perovskite AND solar cells AND computational AND experimental"
#         }
#     },
#     "active_tools": [
#         "materials",
#         "arxiv",
#         "openalex",
#         "chemrxiv",
#         "pubmed",
#         "semanticscholar"
#     ],
#     "raw_tool_data": [
#         {
#             "text": "Material: CsSnI3 (mp-616378). Stability: Unstable (E/hull: 0.011843993999996002 eV). Band Gap: $0.5537000000000001\\ \text{eV}$. Energy Above Hull: $0.011843993999996002\\ \text{eV}$.",
#             "source_type": "materials_project",
#             "tool_id": "materials_search",
#             "metadata": {
#                 "material_id": "mp-616378",
#                 "formula": "CsSnI3",
#                 "is_stable": False,
#                 "band_gap": 0.5537000000000001,
#                 "energy_above_hull": 0.011843993999996002
#             }
#         },
#         {
#             "text": "Material: CsSnI3 (mp-614013). Stability: Stable. Band Gap: $0.449999999999999\\ \text{eV}$. Energy Above Hull: $0.0\\ \text{eV}$.",
#             "source_type": "materials_project",
#             "tool_id": "materials_search",
#             "metadata": {
#                 "material_id": "mp-614013",
#                 "formula": "CsSnI3",
#                 "is_stable": True,
#                 "band_gap": 0.449999999999999,
#                 "energy_above_hull": 0.0
#             }
#         },
#         {
#             "text": "Material: CsSnI3 (mp-27381). Stability: Unstable (E/hull: 0.003249338499999 eV). Band Gap: $2.0632\\ \text{eV}$. Energy Above Hull: $0.003249338499999\\ \text{eV}$.",
#             "source_type": "materials_project",
#             "tool_id": "materials_search",
#             "metadata": {
#                 "material_id": "mp-27381",
#                 "formula": "CsSnI3",
#                 "is_stable": False,
#                 "band_gap": 2.0632,
#                 "energy_above_hull": 0.003249338499999
#             }
#         },
#         {
#             "text": "Material: CsSnI3 (mp-568570). Stability: Unstable (E/hull: 0.011059161000002002 eV). Band Gap: $0.617499999999999\\ \text{eV}$. Energy Above Hull: $0.011059161000002002\\ \text{eV}$.",
#             "source_type": "materials_project",
#             "tool_id": "materials_search",
#             "metadata": {
#                 "material_id": "mp-568570",
#                 "formula": "CsSnI3",
#                 "is_stable": False,
#                 "band_gap": 0.617499999999999,
#                 "energy_above_hull": 0.011059161000002002
#             }
#         },
#         {
#             "text": "Title: Full Optoelectronic Simulation of Lead-Free Perovskite/Organic Tandem Solar Cells.. Abstract: Organic and perovskite semiconductor materials are considered an interesting combination thanks to their similar processing technologies and band gap tunability. Here, we present the design and analysis of perovskite/organic tandem solar cells (TSCs) by using a full optoelectronic simulator (SETFOS). A wide band gap lead-free ASnI<sub>2</sub>Br perovskite top subcell is utilized in conjunction with a narrow band gap DPPEZnP-TBO:PC61BM heterojunction organic bottom subcell to form the tandem configuration. The top and bottom cells were designed according to previous experimental work keeping the same materials and physical parameters. The calibration of the two cells regarding simulation and experimental data shows very good agreement, implying the validation of the simulation process. Accordingly, the two cells are combined to develop a 2T tandem cell. Further, upon optimizing the thickness of the front and rear subcells, a current matching condition is satisfied for which the proposed perovskite/organic TSC achieves an efficiency of 13.32%, <i>J<sub>sc</sub></i> of 13.74 mA/cm<sup>2</sup>, and <i>V<sub>oc</sub></i> of 1.486 V. On the other hand, when optimizing the tandem by utilizing full optoelectronic simulation, the tandem shows a higher efficiency of about 14%, although it achieves a decreased <i>J<sub>sc</sub></i> of 12.27 mA/cm<sup>2</sup>. The study shows that the efficiency can be further improved when concurrently optimizing the various tandem layers by global optimization routines. Furthermore, the impact of defects is demonstrated to highlight other possible routes to improve efficiency. The current simulation study can provide a physical understanding and potential directions for further efficiency improvement for lead-free perovskite/organic TSC.",
#             "source_type": "pubmed",
#             "tool_id": "pubmed_search",
#             "metadata": {
#                 "pmid": "36772085",
#                 "title": "Full Optoelectronic Simulation of Lead-Free Perovskite/Organic Tandem Solar Cells.",
#                 "abstract": "Organic and perovskite semiconductor materials are considered an interesting combination thanks to their similar processing technologies and band gap tunability. Here, we present the design and analysis of perovskite/organic tandem solar cells (TSCs) by using a full optoelectronic simulator (SETFOS). A wide band gap lead-free ASnI<sub>2</sub>Br perovskite top subcell is utilized in conjunction with a narrow band gap DPPEZnP-TBO:PC61BM heterojunction organic bottom subcell to form the tandem configuration. The top and bottom cells were designed according to previous experimental work keeping the same materials and physical parameters. The calibration of the two cells regarding simulation and experimental data shows very good agreement, implying the validation of the simulation process. Accordingly, the two cells are combined to develop a 2T tandem cell. Further, upon optimizing the thickness of the front and rear subcells, a current matching condition is satisfied for which the proposed perovskite/organic TSC achieves an efficiency of 13.32%, <i>J<sub>sc</sub></i> of 13.74 mA/cm<sup>2</sup>, and <i>V<sub>oc</sub></i> of 1.486 V. On the other hand, when optimizing the tandem by utilizing full optoelectronic simulation, the tandem shows a higher efficiency of about 14%, although it achieves a decreased <i>J<sub>sc</sub></i> of 12.27 mA/cm<sup>2</sup>. The study shows that the efficiency can be further improved when concurrently optimizing the various tandem layers by global optimization routines. Furthermore, the impact of defects is demonstrated to highlight other possible routes to improve efficiency. The current simulation study can provide a physical understanding and potential directions for further efficiency improvement for lead-free perovskite/organic TSC.",
#                 "external_id": "36772085",
#                 "pdf_url": "https://pubmed.ncbi.nlm.nih.gov/36772085/"
#             }
#         },
#         {
#             "text": "Title: Numerical Simulation and Experimental Study of Methyl Ammonium Bismuth Iodide Absorber Layer Based Lead Free Perovskite Solar Cells.. Abstract: In the past few years, there has been a significant increase in the development and production of perovskite or perovskite-like materials that do not contain lead (Pb) for the purpose of constructing solar cells. The development and testing of lead-free perovskite-like structures for solar cells is crucial. In this study, we used the solar cell capacitance software (SCAPS) to simulate perovskite solar cells based on methyl ammonium bismuth iodide (MA<sub>3</sub> Bi<sub>2</sub> I<sub>9</sub> ). The electron-transport layer, hole-transport layer, and absorber layer thickness were optimized using SCAPS. The simulated perovskite solar cells (FTO/TiO<sub>2</sub> /MA<sub>3</sub> Bi<sub>2</sub> I<sub>9</sub> /spiro-OMeTAD/Au) performed well with a power conversion efficiency of 14.07\u2009% and a reasonable open circuit voltage of 1.34\u2005V, using the optimized conditions determined by SCAPS. Additionally, we conducted experiments to fabricate perovskite solar cells under controlled humidity, which showed a power conversion efficiency of 1.31\u2009%.",
#             "source_type": "pubmed",
#             "tool_id": "pubmed_search",
#             "metadata": {
#                 "pmid": "37029556",
#                 "title": "Numerical Simulation and Experimental Study of Methyl Ammonium Bismuth Iodide Absorber Layer Based Lead Free Perovskite Solar Cells.",
#                 "abstract": "In the past few years, there has been a significant increase in the development and production of perovskite or perovskite-like materials that do not contain lead (Pb) for the purpose of constructing solar cells. The development and testing of lead-free perovskite-like structures for solar cells is crucial. In this study, we used the solar cell capacitance software (SCAPS) to simulate perovskite solar cells based on methyl ammonium bismuth iodide (MA<sub>3</sub> Bi<sub>2</sub> I<sub>9</sub> ). The electron-transport layer, hole-transport layer, and absorber layer thickness were optimized using SCAPS. The simulated perovskite solar cells (FTO/TiO<sub>2</sub> /MA<sub>3</sub> Bi<sub>2</sub> I<sub>9</sub> /spiro-OMeTAD/Au) performed well with a power conversion efficiency of 14.07\u2009% and a reasonable open circuit voltage of 1.34\u2005V, using the optimized conditions determined by SCAPS. Additionally, we conducted experiments to fabricate perovskite solar cells under controlled humidity, which showed a power conversion efficiency of 1.31\u2009%.",
#                 "external_id": "37029556",
#                 "pdf_url": "https://pubmed.ncbi.nlm.nih.gov/37029556/"
#             }
#         },
#         {
#             "text": "Title: Lead-Free Organic-Inorganic Hybrid Perovskites for Photovoltaic Applications: Recent Advances and Perspectives.. Abstract: Organic-inorganic hybrid halide perovskites (e.g., MAPbI<sub>3</sub> ) have recently emerged as novel active materials for photovoltaic applications with power conversion efficiency over 22%. Conventional perovskite solar cells (PSCs); however, suffer the issue that lead is toxic to the environment and organisms for a long time and is hard to excrete from the body. Therefore, it is imperative to find environmentally-friendly metal ions to replace lead for the further development of PSCs. Previous work has demonstrated that Sn, Ge, Cu, Bi, and Sb ions could be used as alternative ions in perovskite configurations to form a new environmentally-friendly lead-free perovskite structure. Here, we review recent progress on lead-free PSCs in terms of the theoretical insight and experimental explorations of the crystal structure of lead-free perovskite, thin film deposition, and device performance. We also discuss the importance of obtaining further understanding of the fundamental properties of lead-free hybrid perovskites, especially those related to photophysics.",
#             "source_type": "pubmed",
#             "tool_id": "pubmed_search",
#             "metadata": {
#                 "pmid": "28160346",
#                 "title": "Lead-Free Organic-Inorganic Hybrid Perovskites for Photovoltaic Applications: Recent Advances and Perspectives.",
#                 "abstract": "Organic-inorganic hybrid halide perovskites (e.g., MAPbI<sub>3</sub> ) have recently emerged as novel active materials for photovoltaic applications with power conversion efficiency over 22%. Conventional perovskite solar cells (PSCs); however, suffer the issue that lead is toxic to the environment and organisms for a long time and is hard to excrete from the body. Therefore, it is imperative to find environmentally-friendly metal ions to replace lead for the further development of PSCs. Previous work has demonstrated that Sn, Ge, Cu, Bi, and Sb ions could be used as alternative ions in perovskite configurations to form a new environmentally-friendly lead-free perovskite structure. Here, we review recent progress on lead-free PSCs in terms of the theoretical insight and experimental explorations of the crystal structure of lead-free perovskite, thin film deposition, and device performance. We also discuss the importance of obtaining further understanding of the fundamental properties of lead-free hybrid perovskites, especially those related to photophysics.",
#                 "external_id": "28160346",
#                 "pdf_url": "https://pubmed.ncbi.nlm.nih.gov/28160346/"
#             }
#         },
#         {
#             "text": "Title: Germanium-Based Halide Perovskites: Materials, Properties, and Applications.. Abstract: Perovskites are attracting an increasing interest in the wide community of photovoltaics, optoelectronic, and detection, traditionally relying on lead-based systems. This Minireview provides an overview of the current status of experimental and computational results available on Ge-containing 3D and low-dimensional halide perovskites. While stability issues analogous to those of tin-based materials are present, some strategies to afford this problem in Ge metal halide perovskites (MHPs) for photovoltaics have already been identified and successfully employed, reaching efficiencies of solar devices greater than 7\u2009% at up to 500\u2005h of illumination. Interestingly, some Ge-containing MHPs showed promising nonlinear optical responses as well as quite broad emissions, which are worthy of further investigation starting from the basic materials chemistry perspective, where a large space for properties modulation through compositions/alloying/fnanostructuring is present.",
#             "source_type": "pubmed",
#             "tool_id": "pubmed_search",
#             "metadata": {
#                 "pmid": "34126001",
#                 "title": "Germanium-Based Halide Perovskites: Materials, Properties, and Applications.",
#                 "abstract": "Perovskites are attracting an increasing interest in the wide community of photovoltaics, optoelectronic, and detection, traditionally relying on lead-based systems. This Minireview provides an overview of the current status of experimental and computational results available on Ge-containing 3D and low-dimensional halide perovskites. While stability issues analogous to those of tin-based materials are present, some strategies to afford this problem in Ge metal halide perovskites (MHPs) for photovoltaics have already been identified and successfully employed, reaching efficiencies of solar devices greater than 7\u2009% at up to 500\u2005h of illumination. Interestingly, some Ge-containing MHPs showed promising nonlinear optical responses as well as quite broad emissions, which are worthy of further investigation starting from the basic materials chemistry perspective, where a large space for properties modulation through compositions/alloying/fnanostructuring is present.",
#                 "external_id": "34126001",
#                 "pdf_url": "https://pubmed.ncbi.nlm.nih.gov/34126001/"
#             }
#         },
#         {
#             "text": "Title: Numerical optimization of Rb<sub>2</sub>AuScBr<sub>6</sub> and Rb<sub>2</sub>AuScCl<sub>6</sub>-based lead-free perovskite solar cells: device engineering and performance mapping.. Abstract: Perovskite solar cells (PSCs) exhibit significant potential for next-generation photovoltaic technology, integrating high power conversion efficiency (PCE), cost-effectiveness, and tunable optoelectronic properties. This report presents a comprehensive numerical optimization of Rb<sub>2</sub>AuScBr<sub>6</sub> and Rb<sub>2</sub>AuScCl<sub>6</sub>-based PSCs, with particular emphasis on the influence of electron transport layers (ETLs) and critical device parameters. The configuration ITO/TiO<sub>2</sub>/Rb<sub>2</sub>AuScBr<sub>6</sub>/CBTS/Ni achieves a PCE of 27.49%, whereas the configuration ITO/WS<sub>2</sub>/Rb<sub>2</sub>AuScCl<sub>6</sub>/CBTS/Ni attains 22.41%, thereby underscoring the high efficiency of these lead-free materials. Device performance is markedly improved through increased perovskite layer thickness and reduced defect density. Further stabilization of performance is achieved by optimizing electron affinity, series resistance, and shunt resistance. Additionally, thermal stability is enhanced through the adjustment of operational temperature. The superior PCE observed in Rb<sub>2</sub>AuScBr<sub>6</sub> is ascribed to the selection of the ETL, an optimal band gap, absorber layer thickness, lower defect density, and appropriate contact interfaces. Overall, these Rb<sub>2</sub>AuScBr<sub>6</sub> and Rb<sub>2</sub>AuScCl<sub>6</sub> perovskites demonstrate exceptional promise for practical, efficient, and stable PSC applications, thereby encouraging further experimental validation and device engineering.",
#             "source_type": "pubmed",
#             "tool_id": "pubmed_search",
#             "metadata": {
#                 "pmid": "41268489",
#                 "title": "Numerical optimization of Rb<sub>2</sub>AuScBr<sub>6</sub> and Rb<sub>2</sub>AuScCl<sub>6</sub>-based lead-free perovskite solar cells: device engineering and performance mapping.",
#                 "abstract": "Perovskite solar cells (PSCs) exhibit significant potential for next-generation photovoltaic technology, integrating high power conversion efficiency (PCE), cost-effectiveness, and tunable optoelectronic properties. This report presents a comprehensive numerical optimization of Rb<sub>2</sub>AuScBr<sub>6</sub> and Rb<sub>2</sub>AuScCl<sub>6</sub>-based PSCs, with particular emphasis on the influence of electron transport layers (ETLs) and critical device parameters. The configuration ITO/TiO<sub>2</sub>/Rb<sub>2</sub>AuScBr<sub>6</sub>/CBTS/Ni achieves a PCE of 27.49%, whereas the configuration ITO/WS<sub>2</sub>/Rb<sub>2</sub>AuScCl<sub>6</sub>/CBTS/Ni attains 22.41%, thereby underscoring the high efficiency of these lead-free materials. Device performance is markedly improved through increased perovskite layer thickness and reduced defect density. Further stabilization of performance is achieved by optimizing electron affinity, series resistance, and shunt resistance. Additionally, thermal stability is enhanced through the adjustment of operational temperature. The superior PCE observed in Rb<sub>2</sub>AuScBr<sub>6</sub> is ascribed to the selection of the ETL, an optimal band gap, absorber layer thickness, lower defect density, and appropriate contact interfaces. Overall, these Rb<sub>2</sub>AuScBr<sub>6</sub> and Rb<sub>2</sub>AuScCl<sub>6</sub> perovskites demonstrate exceptional promise for practical, efficient, and stable PSC applications, thereby encouraging further experimental validation and device engineering.",
#                 "external_id": "41268489",
#                 "pdf_url": "https://pubmed.ncbi.nlm.nih.gov/41268489/"
#             }
#         },
#         {
#             "text": "Title: Lead Free Perovskites. Abstract: One of the most viable renewable energies is solar power because of its versatility, reliability, and abundance. In the market, a majority of the solar panels are made from silicon wafers. These solar panels have an efficiency of 26.4 percent and can last more than 25 years. The perovskite solar cell is a relatively new type of solar technology that has a similar maximum efficiency and much cheaper costs, the only downside is that it is less stable and the most efficient type uses lead. The name",
#             "source_type": "arxiv",
#             "tool_id": "arxiv_search",
#             "metadata": {
#                 "arxiv_id": "http://arxiv.org/abs/2407.17520v2",
#                 "title": "Lead Free Perovskites",
#                 "abstract": "One of the most viable renewable energies is solar power because of its versatility, reliability, and abundance. In the market, a majority of the solar panels are made from silicon wafers. These solar panels have an efficiency of 26.4 percent and can last more than 25 years. The perovskite solar cell is a relatively new type of solar technology that has a similar maximum efficiency and much cheaper costs, the only downside is that it is less stable and the most efficient type uses lead. The name",
#                 "published_year": 2024,
#                 "pdf_url": "https://arxiv.org/pdf/2407.17520v2"
#             }
#         },
#         {
#             "text": "Title: Unveiling architectural and optoelectronic synergies in lead-free perovskite/perovskite/kesterite triple-junction monolithic tandem solar cells. Abstract: The widespread use of lead-based materials in tandem solar cells raises critical environmental and health concerns due to their inherent toxicity and risk of contamination. To address this challenge, we focused on lead-free tandem architectures based on non-toxic, environmentally benign materials such as tin-based perovskites and kesterites, which are essential for advancing sustainable photovoltaic technologies. In this study, we present the proposition, design, and optimization of two distinct",
#             "source_type": "arxiv",
#             "tool_id": "arxiv_search",
#             "metadata": {
#                 "arxiv_id": "http://arxiv.org/abs/2511.06059v2",
#                 "title": "Unveiling architectural and optoelectronic synergies in lead-free perovskite/perovskite/kesterite triple-junction monolithic tandem solar cells",
#                 "abstract": "The widespread use of lead-based materials in tandem solar cells raises critical environmental and health concerns due to their inherent toxicity and risk of contamination. To address this challenge, we focused on lead-free tandem architectures based on non-toxic, environmentally benign materials such as tin-based perovskites and kesterites, which are essential for advancing sustainable photovoltaic technologies. In this study, we present the proposition, design, and optimization of two distinct",
#                 "published_year": 2025,
#                 "pdf_url": "https://arxiv.org/pdf/2511.06059v2"
#             }
#         },
#         {
#             "text": "Title: Efficient Passivation of Surface Defects by Lewis Base in Lead-free Tin-based Perovskite Solar Cells. Abstract: Lead-free tin-based perovskites are highly appealing for the next generation of solar cells due to their intriguing optoelectronic properties. However, the tendency of Sn2+ oxidation to Sn4+ in the tin-based perovskites induces serious film degradation and performance deterioration. Herein, we demonstrate, through the density functional theory based first-principle calculations in a surface slab model, that the surface defects of the Sn-based perovskite FASnI3 (FA = NH2CHNH2+) could be effective",
#             "source_type": "arxiv",
#             "tool_id": "arxiv_search",
#             "metadata": {
#                 "arxiv_id": "http://arxiv.org/abs/2206.06782v1",
#                 "title": "Efficient Passivation of Surface Defects by Lewis Base in Lead-free Tin-based Perovskite Solar Cells",
#                 "abstract": "Lead-free tin-based perovskites are highly appealing for the next generation of solar cells due to their intriguing optoelectronic properties. However, the tendency of Sn2+ oxidation to Sn4+ in the tin-based perovskites induces serious film degradation and performance deterioration. Herein, we demonstrate, through the density functional theory based first-principle calculations in a surface slab model, that the surface defects of the Sn-based perovskite FASnI3 (FA = NH2CHNH2+) could be effective",
#                 "published_year": 2022,
#                 "pdf_url": "https://arxiv.org/pdf/2206.06782v1"
#             }
#         },
#         {
#             "text": "Title: Optimization and Performance Evaluation of Cs$_2$CuBiCl$_6$ Double Perovskite Solar Cell for Lead-Free Photovoltaic Applications. Abstract: In the previous decade, there has been a significant advancement in the performance of perovskite solar cells (PSCs), characterized by a notable increase in efficiency from 3.8% to 25%. Nonetheless, PSCs face many problems when we commercialize them because of their toxicity and stability. Consequently, lead-PSCs need an alternative solar cell with high performance and low processing cost; lead-free inorganic perovskites have been explored. Recent research showcased Cs$_2$CuBiCl$_6$, a lead-free",
#             "source_type": "arxiv",
#             "tool_id": "arxiv_search",
#             "metadata": {
#                 "arxiv_id": "http://arxiv.org/abs/2502.16850v1",
#                 "title": "Optimization and Performance Evaluation of Cs$_2$CuBiCl$_6$ Double Perovskite Solar Cell for Lead-Free Photovoltaic Applications",
#                 "abstract": "In the previous decade, there has been a significant advancement in the performance of perovskite solar cells (PSCs), characterized by a notable increase in efficiency from 3.8% to 25%. Nonetheless, PSCs face many problems when we commercialize them because of their toxicity and stability. Consequently, lead-PSCs need an alternative solar cell with high performance and low processing cost; lead-free inorganic perovskites have been explored. Recent research showcased Cs$_2$CuBiCl$_6$, a lead-free",
#                 "published_year": 2025,
#                 "pdf_url": "https://arxiv.org/pdf/2502.16850v1"
#             }
#         },
#         {
#             "text": "Title: Exploring Lead Free Mixed Halide Double Perovskites Solar Cell. Abstract: The significant surge in energy use and escalating environmental concerns have sparked worldwide interest towards the study and implementation of solar cell technology. Perovskite solar cells (PSCs) have garnered remarkable attention as an emerging third-generation solar cell technology. This paper presents an in-depth analysis of lead-free mixed halide double perovskites in the context of their potential uses in solar cell technology. Through the previous studies of various mixed halide double ",
#             "source_type": "arxiv",
#             "tool_id": "arxiv_search",
#             "metadata": {
#                 "arxiv_id": "http://arxiv.org/abs/2401.09584v1",
#                 "title": "Exploring Lead Free Mixed Halide Double Perovskites Solar Cell",
#                 "abstract": "The significant surge in energy use and escalating environmental concerns have sparked worldwide interest towards the study and implementation of solar cell technology. Perovskite solar cells (PSCs) have garnered remarkable attention as an emerging third-generation solar cell technology. This paper presents an in-depth analysis of lead-free mixed halide double perovskites in the context of their potential uses in solar cell technology. Through the previous studies of various mixed halide double ",
#                 "published_year": 2024,
#                 "pdf_url": "https://arxiv.org/pdf/2401.09584v1"
#             }
#         },
#         {
#             "text": "Title: From unstable CsSnI3 to air-stable Cs2SnI6: A lead-free perovskite solar cell light absorber with bandgap of 1.48 eV and high absorption coefficient. Abstract: Abstract unavailable.",
#             "source_type": "openalex",
#             "tool_id": "openalex_search",
#             "metadata": {
#                 "openalex_id": "https://openalex.org/W2523562186",
#                 "title": "From unstable CsSnI3 to air-stable Cs2SnI6: A lead-free perovskite solar cell light absorber with bandgap of 1.48 eV and high absorption coefficient",
#                 "pdf_url": "https://doi.org/10.1016/j.solmat.2016.09.022"
#             }
#         },
#         {
#             "text": "Title: Investigation of photovoltaic performance of lead-free CsSnI3-based perovskite solar cell with different hole transport layers: First Principle Calculations and SCAPS-1D Analysis. Abstract: Abstract unavailable.",
#             "source_type": "openalex",
#             "tool_id": "openalex_search",
#             "metadata": {
#                 "openalex_id": "https://openalex.org/W4311419982",
#                 "title": "Investigation of photovoltaic performance of lead-free CsSnI3-based perovskite solar cell with different hole transport layers: First Principle Calculations and SCAPS-1D Analysis",
#                 "pdf_url": "https://doi.org/10.1016/j.solener.2022.11.025"
#             }
#         },
#         {
#             "text": "Title: SCAPS-1D Simulation for Device Optimization to Improve Efficiency in Lead-Free CsSnI3 Perovskite Solar Cells. Abstract: In this study, a novel systematic analysis was conducted to explore the impact of various parameters, including acceptor density (NA), individual layer thickness, defect density, interface defect density, and the metal electrode work function, on efficiency within the FTO/ZnO/CsSnI3/NiOx/Au perovskite solar cell structure through the SCAPS-1D (Solar Cell Capacitance Simulator in 1 Dimension) simulation. ZnO served as the electron transport layer (ETL), CsSnI3 as the perovskite absorption layer (PAL), and NiOx as the hole transport layer (HTL), all contributing to the optimization of device performance. To achieve the optimal power conversion efficiency (PCE), we determined the ideal PAL acceptor density (NA) to be 2 \u00d7 1019 cm\u22123 and the optimal thicknesses to be 20 nm for the ETL (ZnO), 700 nm for the PAL (CsSnI3), and 10 nm for the HTL (NiOx), with the metal electrode remaining as Au. As a result of the optimization process, efficiency increased from 11.89% to 23.84%. These results are expected to contribute to the performance enhancement of eco-friendly, lead-free inorganic hybrid solar cells with Sn-based perovskite as the PAL.",
#             "source_type": "openalex",
#             "tool_id": "openalex_search",
#             "metadata": {
#                 "openalex_id": "https://openalex.org/W4395003215",
#                 "title": "SCAPS-1D Simulation for Device Optimization to Improve Efficiency in Lead-Free CsSnI3 Perovskite Solar Cells",
#                 "pdf_url": "https://doi.org/10.3390/inorganics12040123"
#             }
#         },
#         {
#             "text": "Title: 20.730% highly efficient lead-free CsSnI3-based perovskite solar cells with various charge transport materials: a SCAPS-1D study. Abstract: Abstract unavailable.",
#             "source_type": "openalex",
#             "tool_id": "openalex_search",
#             "metadata": {
#                 "openalex_id": "https://openalex.org/W4405581837",
#                 "title": "20.730% highly efficient lead-free CsSnI3-based perovskite solar cells with various charge transport materials: a SCAPS-1D study",
#                 "pdf_url": "https://doi.org/10.1007/s41939-024-00701-2"
#             }
#         },
#         {
#             "text": "Title: Nitrogen-doped titanium dioxide as a novel eco-friendly hole transport layer in lead-free CsSnI3 based perovskite solar cells. Abstract: Abstract unavailable.",
#             "source_type": "openalex",
#             "tool_id": "openalex_search",
#             "metadata": {
#                 "openalex_id": "https://openalex.org/W4389805580",
#                 "title": "Nitrogen-doped titanium dioxide as a novel eco-friendly hole transport layer in lead-free CsSnI3 based perovskite solar cells",
#                 "pdf_url": "https://doi.org/10.1016/j.materresbull.2023.112642"
#             }
#         },
#         {
#             "text": "Title: An Absorber Enrichment Study and Implications on the Performance of Lead-Free CsSnI3 Perovskite Solar Cells (PSCs) Using One-Dimensional Solar Cell Capacitance Simulator (1D-SCAPS). Abstract: Abstract unavailable.",
#             "source_type": "openalex",
#             "tool_id": "openalex_search",
#             "metadata": {
#                 "openalex_id": "https://openalex.org/W4390944690",
#                 "title": "An Absorber Enrichment Study and Implications on the Performance of Lead-Free CsSnI3 Perovskite Solar Cells (PSCs) Using One-Dimensional Solar Cell Capacitance Simulator (1D-SCAPS)",
#                 "pdf_url": "https://doi.org/10.1007/s13538-023-01406-6"
#             }
#         },
#         {
#             "text": "Title: Impact of Hole Transport Layers in Inorganic Lead-Free B-\u03b3-CsSnI3 Perovskite Solar Cells: A Numerical Analysis. Abstract: Tin-based halide perovskite compounds have attracted enormous interest as effective replacements for the conventional lead halide perovskite solar cells (PCSs). However, achieving high efficiency for tin-based perovskite solar cells is still challenging. Herein, we introduced copper sulfide (CuS) as a hole transport material (HTM) in lead free tin-based B-&gamma;-CsSnI3 PSCs to enhance the photovoltaic (PV) performances. The lead free tin-based CsSnI3 perovskite solar cell structure consisting of CuS/CsSnI3/TiO2/ITO was modeled and the output characteristics were investigated by using the one dimensional solar cell capacitance simulator (SCAPS-1D). The CuS hole transport layer (HTL) with proper band arrangement may notably minimize the recombination of the charge carrier at the back side of the perovskite absorber. Density functional theory (DFT)-extracted physical parameters including the band gap and absorption spectrum of CuS were used in the SCAPS-1D program to analyze the characteristics of the proposed PV device. The PV performance parameters of the proposed device were numerically evaluated by varying the absorber thickness and doping concentration. In this work, the variation of the functional temperature on the cell outputs was also studied. Furthermore, different HTMs were employed to investigate the PV characteristics of the proposed CsSnI3 PSC. The power conversion efficiency (PCE) of ~29% was achieved with open circuit voltage (Voc) of 0.99 V, a fill factor of ~87%, and short circuit current density (Jsc) of 33.5 mA/cm2 for the optimized device. This work addressed guidelines and introduced a convenient approach to design and fabricate highly efficient, inexpensive, and stable lead free tin-based perovskite solar cells.",
#             "source_type": "openalex",
#             "tool_id": "openalex_search",
#             "metadata": {
#                 "openalex_id": "https://openalex.org/W4286726627",
#                 "title": "Impact of Hole Transport Layers in Inorganic Lead-Free B-\u03b3-CsSnI3 Perovskite Solar Cells: A Numerical Analysis",
#                 "pdf_url": "https://doi.org/10.3390/ecp2022-12611"
#             }
#         },
#         {
#             "text": "Title: Device Engineering of a Novel Lead-Free Solar Cell Architecture Utilizing Inorganic CsSnCl3 and CsSnI3 Perovskite-Based Dual Absorbers for Sustainable Powering of Wireless Networks. Abstract: Abstract unavailable.",
#             "source_type": "openalex",
#             "tool_id": "openalex_search",
#             "metadata": {
#                 "openalex_id": "https://openalex.org/W4404740769",
#                 "title": "Device Engineering of a Novel Lead-Free Solar Cell Architecture Utilizing Inorganic CsSnCl3 and CsSnI3 Perovskite-Based Dual Absorbers for Sustainable Powering of Wireless Networks",
#                 "pdf_url": "https://doi.org/10.1007/s11664-024-11605-9"
#             }
#         },
#         {
#             "text": "Title: Nitrogen-doped Titanium Dioxide as a novel eco-friendly Hole Transport Layer in Lead-Free CsSnI3 based Perovskite Solar Cells. Abstract: Abstract Despite recent abrupt rise in the efficiency of perovskite solar cells (PSCs), the contact layers maybe limit the efficiency of PSCs. The hole transporting layer (HTL) is an essential layer for reducing the recombination and loosing charges in fabricated devices by avoiding direct contact of gold to perovskite absorber layer in an efficient PSC device. The pristine spiro-OMeTAD, as most widely used HTL, still suffers from poor electrical conductivity, low hole mobility, and low oxidation rate. In this research, the nitrogen doped TiO 2 (N-TiO 2 ) proposed as a low-cost, efficient, safe replacement for spiro-OMeTAD HTL in PSCs. The variation in the device design key parameters such as the thickness and bulk defect density of perovskite layer, simultaneous modifications of defect density and defect energy level, and acceptor doping concentration in absorber layer are examined with their impact on the photovoltaic characteristic parameters. The effect of an increase in operating temperature from 280 K to 460 K on the performance of CsSnI 3 -based perovskite devices is also investigated. The standard simulated lead-free CsSnI 3 \u2013based PSCs with spiro-OMeTAD HTL by SCAPS-1D software revealed the highest power conservation efficiency (PCE) of 23.63%. The CsSnI 3 -based solar cell with N-TiO 2 as HTL showed FF (79.65%), V OC (0.98 V), J sc (34.69 mA/cm 2 ), and efficiency (27.03%) higher than the standard device with conventional spiro-OMeTAD HTL. The outcomes of N-TiO 2 presence as an HTL signify a critical avenue for the possibility of fabricating high PCE CsSnI 3 -based perovskite devices made of stable, low-cost, efficient, safe, and eco-friendly materials.",
#             "source_type": "openalex",
#             "tool_id": "openalex_search",
#             "metadata": {
#                 "openalex_id": "https://openalex.org/W4385704474",
#                 "title": "Nitrogen-doped Titanium Dioxide as a novel eco-friendly Hole Transport Layer in Lead-Free CsSnI3 based Perovskite Solar Cells",
#                 "pdf_url": "http://dx.doi.org/10.21203/rs.3.rs-3185005/v1"
#             }
#         },
#         {
#             "text": "Title: Performance enhancement of lead-free CsSnI3 and CsSnCl3 perovskite solar cells by tuning layer interfaces. Abstract: Abstract unavailable.",
#             "source_type": "openalex",
#             "tool_id": "openalex_search",
#             "metadata": {
#                 "openalex_id": "https://openalex.org/W4414558869",
#                 "title": "Performance enhancement of lead-free CsSnI3 and CsSnCl3 perovskite solar cells by tuning layer interfaces",
#                 "pdf_url": "https://doi.org/10.1007/s11082-025-08461-0"
#             }
#         }
#     ],
#     "full_text_chunks": [],
#     "rag_complete": False,
#     "filtered_context": "",
#     "references": [
#         "\u269b\ufe0f Materials Project: mp-616378 (CsSnI3)",
#         "\u269b\ufe0f Materials Project: mp-614013 (CsSnI3)",
#         "\u269b\ufe0f Materials Project: mp-27381 (CsSnI3)",
#         "\u269b\ufe0f Materials Project: mp-568570 (CsSnI3)",
#         "\ud83d\udcc4 Journal Article: Full Optoelectronic Simulation of Lead-Free Perovskite/Organic Tandem Solar Cells.",
#         "\ud83d\udcc4 Journal Article: Numerical Simulation and Experimental Study of Methyl Ammonium Bismuth Iodide Absorber Layer Based Lead Free Perovskite Solar Cells.",
#         "\ud83d\udcc4 Journal Article: Lead-Free Organic-Inorganic Hybrid Perovskites for Photovoltaic Applications: Recent Advances and Perspectives.",
#         "\ud83d\udcc4 Journal Article: Germanium-Based Halide Perovskites: Materials, Properties, and Applications.",
#         "\ud83d\udcc4 Journal Article: Numerical optimization of Rb<sub>2</sub>AuScBr<sub>6</sub> and Rb<sub>2</sub>AuScCl<sub>6</sub>-based lead-free perovskite solar cells: device engineering and performance mapping.",
#         "\ud83d\udd17 Arxiv: Lead Free Perovskites",
#         "\ud83d\udd17 Arxiv: Unveiling architectural and optoelectronic synergies in lead-free perovskite/perovskite/kesterite triple-junction monolithic tandem solar cells",
#         "\ud83d\udd17 Arxiv: Efficient Passivation of Surface Defects by Lewis Base in Lead-free Tin-based Perovskite Solar Cells",
#         "\ud83d\udd17 Arxiv: Optimization and Performance Evaluation of Cs$_2$CuBiCl$_6$ Double Perovskite Solar Cell for Lead-Free Photovoltaic Applications",
#         "\ud83d\udd17 Arxiv: Exploring Lead Free Mixed Halide Double Perovskites Solar Cell",
#         "\ud83d\udd17 OpenAlex: From unstable CsSnI3 to air-stable Cs2SnI6: A lead-free perovskite solar cell light absorber with bandgap of 1.48 eV and high absorption coefficient",
#         "\ud83d\udd17 OpenAlex: Investigation of photovoltaic performance of lead-free CsSnI3-based perovskite solar cell with different hole transport layers: First Principle Calculations and SCAPS-1D Analysis",
#         "\ud83d\udd17 OpenAlex: SCAPS-1D Simulation for Device Optimization to Improve Efficiency in Lead-Free CsSnI3 Perovskite Solar Cells",
#         "\ud83d\udd17 OpenAlex: 20.730% highly efficient lead-free CsSnI3-based perovskite solar cells with various charge transport materials: a SCAPS-1D study",
#         "\ud83d\udd17 OpenAlex: Nitrogen-doped titanium dioxide as a novel eco-friendly hole transport layer in lead-free CsSnI3 based perovskite solar cells",
#         "\ud83d\udd17 OpenAlex: An Absorber Enrichment Study and Implications on the Performance of Lead-Free CsSnI3 Perovskite Solar Cells (PSCs) Using One-Dimensional Solar Cell Capacitance Simulator (1D-SCAPS)",
#         "\ud83d\udd17 OpenAlex: Impact of Hole Transport Layers in Inorganic Lead-Free B-\u03b3-CsSnI3 Perovskite Solar Cells: A Numerical Analysis",
#         "\ud83d\udd17 OpenAlex: Device Engineering of a Novel Lead-Free Solar Cell Architecture Utilizing Inorganic CsSnCl3 and CsSnI3 Perovskite-Based Dual Absorbers for Sustainable Powering of Wireless Networks",
#         "\ud83d\udd17 OpenAlex: Nitrogen-doped Titanium Dioxide as a novel eco-friendly Hole Transport Layer in Lead-Free CsSnI3 based Perovskite Solar Cells",
#         "\ud83d\udd17 OpenAlex: Performance enhancement of lead-free CsSnI3 and CsSnCl3 perovskite solar cells by tuning layer interfaces"
#     ],
#     "final_report": "",
#     "report_generated": False,
#     "needs_refinement": False,
#     "refinement_reason": "",
#     "is_refining": False,
#     "refinement_retries": 0,
#     "next": "",
#     "visited_nodes": [
#         "intent_agent",
#         "planning_agent",
#         "query_gen_agent",
#         "materials_search",
#         "pubmed_search",
#         "arxiv_search",
#         "openalex_search"
#     ]
# })

#     except Exception as e:
#         print(f"{C_RED}[SETUP FAILED] {type(e).__name__}: {e}{C_RESET}")
#         return

#     # --- 2. EXECUTE RAG AGENT ---
#     print(f"\n{C_MAGENTA}--- EXECUTING RAG AGENT ---{C_RESET}")
#     try:
#         state = rag_agent.execute(state)

#     except Exception as e:
#         print(f"{C_RED}[TEST FAILURE] RAGAgent crashed: {type(e).__name__}: {e}{C_RESET}")
#         return

#     # --- 3. ASSERTIONS ---
#     context = state.get("filtered_context", "")

#     # ASSERTION 1: Pipeline completed
#     if state.get("rag_complete") is True:
#         print(f"{C_GREEN}[PASS] RAG pipeline completed.{C_RESET}")
#     else:
#         print(f"{C_RED}[FAIL] rag_complete flag not set.{C_RESET}")
#         return

#     # ASSERTION 2: Structured data preserved
#     if "Structured Data" in context and "Band Gap" in context:
#         print(f"{C_GREEN}[PASS] Structured materials data preserved.{C_RESET}")
#     else:
#         print(f"{C_RED}[FAIL] Structured data missing from context.{C_RESET}")

#     # ASSERTION 3: Relevant semantic content included
#     if "csnni3" in context.lower() or "csnsi3" in context.lower():
#         print(f"{C_GREEN}[PASS] Relevant CsSnI3 content retrieved.{C_RESET}")
#     else:
#         print(f"{C_RED}[FAIL] Relevant CsSnI3 content missing.{C_RESET}")

#     # ASSERTION 4: Irrelevant topic filtered out
#     if "quantum error correction" not in context.lower():
#         print(f"{C_GREEN}[PASS] Irrelevant content filtered out.{C_RESET}")
#     else:
#         print(f"{C_RED}[FAIL] Irrelevant content leaked into context.{C_RESET}")

#     # ASSERTION 5: Neighbor expansion worked
#     if "oxidation" in context.lower():
#         print(f"{C_GREEN}[PASS] Neighbor expansion preserved contextual continuity.{C_RESET}")
#     else:
#         print(f"{C_YELLOW}[WARN] Neighbor expansion may not have triggered (check thresholds).{C_RESET}")

#     # ASSERTION 6: Breadcrumb tracking
#     if "rag_agent" in state.get("visited_nodes", []):
#         print(f"{C_GREEN}[PASS] Breadcrumb tracking verified.{C_RESET}")
#     else:
#         print(f"{C_RED}[FAIL] Breadcrumb tracking missing rag_agent.{C_RESET}")

#     print(f"\n{C_CYAN}*** RAG AGENT ONLY TEST COMPLETE ***{C_RESET}")


# if __name__ == "__main__":
#     test_rag_agent_only()


# # ==================================================================================================
# # INTEGRATED TEST BLOCK :::: RETRIEVAL AGENT + RAG AGENT
# # ==================================================================================================

# # --- Mock State Data for Testing ---
# # This data simulates the output after the 'tool_agents' (arxiv, pubmed, materials) have run.
# MOCK_INITIAL_STATE = {
#     "user_query": "A detailed review on the synthesis and bandgap stability of lead-free CsSnI3 perovskite solar cells using computational and experimental data.",
#     "semantic_query": "A detailed review on the synthesis and bandgap stability of lead-free CsSnI3 perovskite solar cells using computational and experimental data.",
#     "api_search_term": "CsSnI3",
#     "active_tools": ["arxiv", "pubmed", "materials"],
#     "raw_tool_data": [
#         # Irrelevant ArXiv data to test vector filtering (PDF links are expected to fail, forcing fallback to abstract text)
#         { "text": "Title: Selfi: Self Improving Reconstruction Engine via 3D Geometric Feature Alignment. Abstract: Novel View Synthesis (NVS) has traditionally relied on models with explicit 3D inductive biases...", "source_type": "arxiv", "tool_id": "arxiv_agent", "metadata": { "pdf_url": "https://arxiv.org/pdf/2512.08930v1" } },
#         { "text": "Title: On a cross-diffusion hybrid model: Cancer Invasion Tissue with Normal Cell Involved. Abstract: In this paper, we study a well-posedness problem on a new mathematical model for cancer invasion...", "source_type": "arxiv", "tool_id": "arxiv_agent", "metadata": { "pdf_url": "https://arxiv.org/pdf/2512.08929v1" } },
#         { "text": "Title: SAQ: Stabilizer-Aware Quantum Error Correction Decoder. Abstract: Quantum Error Correction (QEC) decoding faces a fundamental accuracy-efficiency tradeoff...", "source_type": "arxiv", "tool_id": "arxiv_agent", "metadata": { "pdf_url": "https://arxiv.org/pdf/2512.08914v1" } },
#         # Relevant PubMed abstracts (testing abstract fallback and semantic relevance)
#         { "text": "Title: Advancement on Lead-Free Organic-Inorganic Halide Perovskite Solar Cells: A Review.. Abstract: Remarkable attention has been committed to the recently discovered cost effective and solution processable lead-free organic-inorganic halide perovskite solar cells...", "source_type": "pubmed", "tool_id": "pubmed_agent", "metadata": { "external_id": "29899206" } },
#         { "text": "Title: A Brief Review of Perovskite Quantum Dot Solar Cells: Synthesis, Property and Defect Passivation.. Abstract: Perovskite quantum dot solar cells (PQDSCs)... highly dependent on the properties of interfaces...", "source_type": "pubmed", "tool_id": "pubmed_agent", "metadata": { "external_id": "39289160" } },
#         { "text": "Title: Development on inverted perovskite solar cells: A review.. Abstract: Recently, inverted perovskite solar cells (IPSCs) have received note-worthy consideration...", "source_type": "pubmed", "tool_id": "pubmed_agent", "metadata": { "external_id": "38298729" } },
#         { "text": "Material: CsSnI3 (mp-616378). Stability: Unstable (E/hull: 0.01184 eV). Band Gap: 0.5537 eV. Energy Above Hull: 0.0118 eV.", "source_type": "materials_project", "tool_id": "materials_agent", "metadata": { "material_id": "mp-616378", "is_stable": False } },
#         { "text": "Material: CsSnI3 (mp-614013). Stability: Stable. Band Gap: 0.4499 eV. Energy Above Hull: 0.0 eV.", "source_type": "materials_project", "tool_id": "materials_agent", "metadata": { "material_id": "mp-614013", "is_stable": True } },
#     ],
#     "full_text_chunks": [],
#     "rag_complete": False,
#     "filtered_context": "",
#     "is_refining": False
# }


# def test_rag_agents_pipeline():
#     """Executes the RetrievalAgent followed by the RAGAgent for testing."""
#     print(f"{C_CYAN}*** STARTING RAG AGENTS PIPELINE TEST ***{C_RESET}")

#     # 1. SETUP
#     try:
#         # NOTE: VectorDBWrapper must be successfully initialized (i.e., FAISS/embeddings work)
#         vector_db_instance = VectorDBWrapper()

#         # Load the mock state
#         state = ResearchState(MOCK_INITIAL_STATE)

#         retrieval_agent = RetrievalAgent()
#         rag_agent = RAGAgent(vector_db=vector_db_instance)

#     except Exception as e:
#         print(f"{C_RED}SETUP FAILED: Ensure VectorDBWrapper and Agents are correctly defined/imported. Error: {e}{C_RESET}")
#         return

#     # --- 2. EXECUTE RETRIEVAL AGENT (Chunking & Preparation) ---
#     print(f"\n{C_MAGENTA}--- 2A. EXECUTING RETRIEVAL AGENT (Chunking) ---{C_RESET}")
#     try:
#         state = retrieval_agent.execute(state)
#         chunk_count = len(state.get('full_text_chunks', []))

#         # ASSERTION 1: Check if chunks were created (8 abstract/snippet entries total)
#         if chunk_count == 8:
#             print(f"{C_GREEN}[TEST SUCCESS] Retrieval Agent: Created exactly {chunk_count} chunks (all abstracts/snippets).{C_RESET}")
#         elif chunk_count > 0:
#             print(f"{C_YELLOW}[TEST WARNING] Retrieval Agent: Expected 8 chunks, found {chunk_count}. Check chunk_size and data parsing.{C_RESET}")
#         else:
#             print(f"{C_RED}[TEST FAILURE] Retrieval Agent: Found 0 chunks. Check PDF download/fallback logic.{C_RESET}")
#             return

#     except Exception as e:
#         print(f"{C_RED}[TEST FAILURE] Retrieval Agent execution failed: {type(e).__name__}: {e}{C_RESET}")
#         return

#     # --- 3. EXECUTE RAG AGENT (Vectorization, Search, & Filtering) ---
#     print(f"\n{C_MAGENTA}--- 2B. EXECUTING RAG AGENT (Search & Filter) ---{C_RESET}")
#     try:
#         state = rag_agent.execute(state)
#         context = state.get('filtered_context', '')

#         # ASSERTION 2: Check for presence of structured data and relevant chunks
#         is_structured_present = "--- Structured Data (Materials Property) ---" in context
#         is_relevant_text_present = state['api_search_term'].lower() in context.lower() # CsSnI3 check

#         if state['rag_complete'] and is_structured_present and is_relevant_text_present:
#             print(f"{C_GREEN}[TEST SUCCESS] RAG Agent: Pipeline is complete and context is generated.{C_RESET}")
#             print(f"{C_BLUE}[DEBUG] Final context length: {len(context)} characters.{C_RESET}")
#         else:
#             print(f"{C_RED}[TEST FAILURE] RAG Agent: Context generation failed. Complete: {state['rag_complete']}. Structured Present: {is_structured_present}. Relevant Text Present: {is_relevant_text_present}.{C_RESET}")
#             return

#         # ASSERTION 3: Check if irrelevant text was filtered out (Crucial RAG Test)
#         if "cancer" not in context.lower() and "quantum error" not in context.lower() and "3d geometric" not in context.lower():
#              print(f"{C_GREEN}[TEST SUCCESS] RAG Agent: Irrelevant abstracts were successfully filtered out by vector search/threshold.{C_RESET}")
#         else:
#              print(f"{C_RED}[TEST FAILURE] RAG Agent: Irrelevant topics were NOT filtered out. Check the vector search distance threshold (1.45).{C_RESET}")


#     except Exception as e:
#         print(f"{C_RED}[TEST FAILURE] RAG Agent execution failed: {type(e).__name__}: {e}{C_RESET}")
#         return

#     print(f"\n{C_CYAN}*** RAG AGENTS PIPELINE TEST COMPLETE ***{C_RESET}")


# if __name__ == "__main__":
#     test_rag_agents_pipeline()

