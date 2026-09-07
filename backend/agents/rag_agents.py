import time
import re
import requests
from io import BytesIO
from typing import Dict, List, Any, Optional, Tuple
from pypdf import PdfReader
import json
from sentence_transformers import CrossEncoder
from bs4 import BeautifulSoup
from pathlib import Path

from backend.core.research_state import ResearchState
from backend.core.vector_db import VectorDBWrapper
from backend.core.utilities import (
    C_ACTION, C_RESET, C_GREEN, C_YELLOW, C_RED, C_BLUE, C_MAGENTA, C_PURPLE, C_CYAN,
    client, LLM_MODEL, embedding_model
)

# ==================================================================================================
# SECTION 7: RETRIEVAL AGENT (PRODUCTION-GRADE, MODEL-AGNOSTIC)
# ==================================================================================================
class RetrievalAgent:
    """
    Agent responsible for downloading and processing research content.
    ALIGNED: Reports back to Supervisor Hub; uses processed_doc_ids to avoid redundant work.
    """
    def __init__(self, agent_id: str = "retrieval_agent", chunk_size: int = 500, model: str = LLM_MODEL):
        self.id = agent_id
        self.chunk_size = chunk_size
        self.model = model

    def _fetch_content(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            time.sleep(1.5)
            stealth_headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Referer': 'https://www.google.com/',
                'Connection': 'keep-alive'
            }
            response = requests.get(url, timeout=15, headers=stealth_headers, allow_redirects=True)

            if response.status_code == 403:
                print(f"{C_YELLOW}[{self.id.upper()} WAF] 403 Forbidden on {url[:40]}.{C_RESET}")
                return None

            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '').lower()

            if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                return {'type': 'pdf', 'data': BytesIO(response.content)}
            if 'text/html' in content_type or b'<!doc' in response.content[:10].lower():
                return {'type': 'html', 'data': response.text}
            return None
        except Exception:
            return None

    def _extract_text_from_pdf(self, pdf_stream: BytesIO) -> str:
        try:
            reader = PdfReader(pdf_stream)
            return " ".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""

    def _extract_text_from_html(self, html_text: str) -> str:
        try:
            soup = BeautifulSoup(html_text, 'html.parser')
            for s in soup(["script", "style", "header", "footer", "nav"]):
                s.decompose()
            content = soup.find('div', {'id': 'abstract'}) or soup.find('div', {'class': 'abstract-content'}) or soup.find('article')
            if content:
                return content.get_text(separator=' ', strip=True)
            paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if len(p.get_text()) > 50]
            return " ".join(paragraphs[:10])
        except Exception:
            return ""

    def _chunk_text(self, text: str) -> List[str]:
        if not text:
            return []
        max_chars = int(self.chunk_size * 3.5)
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current_chunk = [], ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def execute(self, state: ResearchState) -> ResearchState:
        state.setdefault("visited_nodes", []).append(self.id)
        print(f"\n{C_ACTION}[{self.id.upper()} START] Fetching and Processing Content...{C_RESET}")

        existing_chunks = state.get('full_text_chunks', [])
        processed_doc_ids = {chunk['doc_id'] for chunk in existing_chunks if 'doc_id' in chunk}
        raw_data = state.get('raw_tool_data', [])
        all_new_chunks = []
        downloaded_urls_this_run = set()

        for entry in raw_data:
            raw_url = entry.get('metadata', {}).get('pdf_url') or entry.get('metadata', {}).get('url')
            target_url = raw_url.rstrip('.') if raw_url else None

            if not target_url or target_url in processed_doc_ids or entry.get('tool_id') == 'materials_search':
                continue

            fetch_result = self._fetch_content(target_url)
            if not fetch_result:
                continue

            print(f"{C_GREEN}[{self.id.upper()} SUCCESS] Downloaded {target_url[:40]}...{C_RESET}")

            text = self._extract_text_from_pdf(fetch_result['data']) if fetch_result['type'] == 'pdf' else self._extract_text_from_html(fetch_result['data'])
            if not text.strip():
                continue

            downloaded_urls_this_run.add(target_url)
            chunks = self._chunk_text(text)
            doc_hash = abs(hash(target_url)) % 10000
            for i, chunk in enumerate(chunks):
                all_new_chunks.append({"chunk_id": f"{entry['tool_id']}_{doc_hash}_{i}", "doc_id": target_url, "text": chunk, "source": entry['tool_id']})

        # Abstract Fallback logic
        for entry in raw_data:
            raw_url = entry.get('metadata', {}).get('url') or entry.get('metadata', {}).get('pdf_url')
            url = raw_url.rstrip('.') if raw_url else None

            if not url or url in processed_doc_ids or url in downloaded_urls_this_run or entry.get('tool_id') == 'materials_search':
                continue
            if not entry.get('text'):
                continue
            chunks = self._chunk_text(entry['text'])
            for i, chunk in enumerate(chunks):
                all_new_chunks.append({"chunk_id": f"{entry['tool_id']}_abs_{abs(hash(url))%1000}_{i}", "doc_id": url, "text": chunk, "source": entry['tool_id']})

        state.setdefault('full_text_chunks', []).extend(all_new_chunks)
        state["next"] = "supervisor_agent" # HUB-AND-SPOKE ROUTE
        print(f"{C_GREEN}[{self.id.upper()} DONE] Retrieval complete.{C_RESET}")
        return state


# ==================================================================================================
# SECTION 8: RAG AGENT (FULLY UPGRADED WITH EMBEDDING MODEL)
# ==================================================================================================
class RAGAgent:
    """
    Agent responsible for Vector Search, Reranking, and Neighbor Expansion.
    ALIGNED: Uses explicit embedding_model for indexing/searching, Cross-Encoders for precision, and routes to Hub.
    """
    def __init__(
        self,
        agent_id: str = "rag_agent",
        max_chunks_to_keep: int = 8,
        vector_db: Optional[VectorDBWrapper] = None,
        embed_model = embedding_model
    ):
        self.id = agent_id
        self.max_chunks_to_keep = max_chunks_to_keep
        self.embed_model = embed_model

        # Initialize VectorDB Wrapper using positional argument or default
        if vector_db is not None:
            self.vector_db = vector_db
        else:
            self.vector_db = VectorDBWrapper(self.embed_model)

        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def _passes_keyword_gate(self, chunk_text: str, literal_term: str, source: str = "") -> bool:
        """
        Optimized keyword gate: Bypasses trusted repository sources and targets
        actual web scraping/bot protection garbage instead of academic terms.
        """
        chunk_lower = chunk_text.lower()

        # 1. Trusted Source Bypass: Never filter out chunks originating from verified preprints or databases
        if source in ['arxiv', 'chemrxiv', 'materials_search']:
            return True

        # 2. True Web Scraping/WAF Noise Detection
        is_web_noise = any(x in chunk_lower for x in [
            "access denied",
            "403 forbidden",
            "enable javascript",
            "cloudflare",
            "captcha",
            "cookie policy",
            "sign in to view full text"
        ])

        contains_literal = literal_term in chunk_lower if literal_term else True
        return not (is_web_noise and not contains_literal)

    def _get_chunk_idx(self, chunk_dict: Dict[str, Any]) -> int:
        """Helper to extract integer index from chunk_id (e.g. 'arxiv_1234_2' -> 2)."""
        try:
            return int(str(chunk_dict.get('chunk_id', '')).rsplit('_', 1)[-1])
        except (ValueError, IndexError):
            return 0

    def execute(self, state: ResearchState) -> ResearchState:
        state.setdefault("visited_nodes", []).append(self.id)
        print(f"\n{C_ACTION}[{self.id.upper()} START] Reranking & Neighbor Expansion...{C_RESET}")

        chunks_for_db = [c for c in state.get('full_text_chunks', []) if isinstance(c, dict) and c.get('text')]

        if client is None or not chunks_for_db:
            state.update({'filtered_context': "No relevant context found.", 'rag_complete': True, 'next': 'supervisor_agent'})
            print(f"{C_YELLOW}[{self.id.upper()} SKIPPED] No text chunks to process.{C_RESET}")
            return state

        query = state.get('semantic_query', '')
        literal_term = state.get('api_search_term', '').lower()

        # 1. Vector DB Indexing with embedding model
        print(f"{C_BLUE}[{self.id.upper()} INDEX] Adding {len(chunks_for_db)} chunks to vector database...{C_RESET}")
        self.vector_db.add_chunks(chunks_for_db)

        # 2. Vector Search using embedding model (Top 30 candidates)
        top_k_results = self.vector_db.search(query, k=30)

        # 3. Cross-Encoder Reranking
        if top_k_results and query:
            print(f"{C_PURPLE}[{self.id.upper()} RERANK] Scoring top {len(top_k_results)} candidates...{C_RESET}")
            sentence_pairs = [[query, res[0]['text']] for res in top_k_results]
            scores = self.reranker.predict(sentence_pairs)
            reranked_list = sorted([(top_k_results[i][0], scores[i]) for i in range(len(top_k_results))], key=lambda x: x[1], reverse=True)
            top_k_results = reranked_list
            ACTIVE_THRESHOLD = -5.0
        else:
            ACTIVE_THRESHOLD = 0.35

        # 4. Neighbor Expansion & Keyword Filtering
        doc_map: Dict[str, List[Dict[str, Any]]] = {}
        for c in self.vector_db.text_store:
            doc_id = c.get('doc_id')
            if doc_id:
                doc_map.setdefault(doc_id, []).append(c)

        # Safely sort document chunks sequentially by their chunk index
        for d in doc_map:
            doc_map[d].sort(key=self._get_chunk_idx)

        final_chunks, seen_ids = [], set()
        for chunk_dict, score in top_k_results:
            if score < ACTIVE_THRESHOLD:
                continue

            doc_id = chunk_dict.get("doc_id")
            family = doc_map.get(doc_id, [])
            actual_idx = next((i for i, item in enumerate(family) if item.get("chunk_id") == chunk_dict.get("chunk_id")), 0)

            # Neighbor Expansion logic (1 chunk before, current, 1 chunk after)
            for i in range(max(0, actual_idx - 1), min(len(family), actual_idx + 2)):
                c = family[i]
                chunk_id = c.get("chunk_id")

                # Pass source metadata into keyword gate
                if chunk_id not in seen_ids and self._passes_keyword_gate(c["text"], literal_term, c.get("source")):
                    final_chunks.append(c["text"])
                    seen_ids.add(chunk_id)

            if len(final_chunks) >= self.max_chunks_to_keep:
                break

        # 5. Assemble Context
        structured_context = [f"--- Structured Data ---\n{d['text']}" for d in state.get('raw_tool_data', []) if d.get('tool_id') == 'materials_search']

        if structured_context or final_chunks:
            state['filtered_context'] = "\n---\n".join(structured_context + final_chunks)
        else:
            state['filtered_context'] = "No relevant context found."

        state['rag_complete'] = True
        state["next"] = "supervisor_agent" # HUB-AND-SPOKE ROUTE

        print(f"{C_GREEN}[{self.id.upper()} DONE] RAG processing complete.{C_RESET}")
        return state


# ==================================================================================================
# INTEGRATED CLI & TEST RUNNER
# ==================================================================================================

def load_json_state(filepath: Path) -> ResearchState:
    """Helper to safely read state JSON files."""
    if not filepath.exists():
        raise FileNotFoundError(f"{C_RED}[FILE NOT FOUND] {filepath}{C_RESET}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"{C_GREEN}[LOAD SUCCESS] Loaded {filepath.name}{C_RESET}")
    return data

def save_json_state(state: ResearchState, filepath: Path):
    """Helper to write updated state back out to JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    print(f"{C_GREEN}[SAVE SUCCESS] State output written to {filepath.name}{C_RESET}")

def run_retrieval_test(root_dir: Path) -> ResearchState:
    """Runs RetrievalAgent using level_2_test_input.json."""
    print(f"\n{C_CYAN}=== RUNNING TEST 1: RETRIEVAL AGENT ==={C_RESET}")
    input_file = root_dir / "level_2_test_input.json"
    output_file = root_dir / "level_3_retrieval_output.json"

    state = load_json_state(input_file)
    agent = RetrievalAgent()
    updated_state = agent.execute(state)

    chunks_count = len(updated_state.get("full_text_chunks", []))
    print(f"{C_CYAN}[RETRIEVAL RESULT] Extracted {chunks_count} total text chunks.{C_RESET}")

    save_json_state(updated_state, output_file)
    return updated_state

def run_rag_test(root_dir: Path, state_input: Optional[ResearchState] = None) -> ResearchState:
    """Runs RAGAgent using loaded state or reading from saved output JSON."""
    print(f"\n{C_CYAN}=== RUNNING TEST 2: RAG AGENT ==={C_RESET}")
    output_file = root_dir / "level_4_rag_output.json"

    if state_input is None:
        primary = root_dir / "level_3_retrieval_output.json"
        fallback = root_dir / "level_2_test_input.json"
        target = primary if primary.exists() else fallback
        state_input = load_json_state(target)

    agent = RAGAgent()
    updated_state = agent.execute(state_input)

    context = updated_state.get("filtered_context", "")
    print(f"{C_CYAN}[RAG RESULT] Filtered Context Length: {len(context)} characters.{C_RESET}")

    save_json_state(updated_state, output_file)
    return updated_state


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    print(f"\n{C_MAGENTA}===================================================={C_RESET}")
    print(f"{C_MAGENTA}      AGENT PIPELINE CLI TEST RUNNER                {C_RESET}")
    print(f"{C_MAGENTA}===================================================={C_RESET}")
    print("1: Run Retrieval Agent Test  (Input: level_2_test_input.json)")
    print("2: Run RAG Agent Test        (Input: level_3_retrieval_output.json)")
    print("3: Run BOTH sequentially    (Retrieval -> RAG)")
    print("4: Exit")
    print(f"{C_MAGENTA}----------------------------------------------------{C_RESET}")

    choice = input("Enter your choice (1-4): ").strip()

    try:
        if choice == "1":
            run_retrieval_test(project_root)
        elif choice == "2":
            run_rag_test(project_root)
        elif choice == "3":
            retrieval_state = run_retrieval_test(project_root)
            run_rag_test(project_root, state_input=retrieval_state)
        elif choice == "4":
            print(f"{C_YELLOW}Exiting without running tests.{C_RESET}")
        else:
            print(f"{C_RED}Invalid option selected. Exiting.{C_RESET}")
    except Exception as e:
        print(f"\n{C_RED}[TEST FAILED] Error: {e}{C_RESET}")