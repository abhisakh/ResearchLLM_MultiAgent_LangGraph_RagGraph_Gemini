import hashlib
import logging
import os
import time
import re
from io import BytesIO
from urllib.parse import urljoin
from typing import Dict, List, Any, Optional
from pypdf import PdfReader
from sentence_transformers import CrossEncoder
from bs4 import BeautifulSoup

from backend.core.research_state import ResearchState
from backend.core.text_repair import repair_pdf_text
from backend.core.url_safety import check_public_url, public_only_session
from backend.core.vector_db import VectorDBWrapper
from backend.core.utilities import (
    client, LLM_MODEL
)

logger = logging.getLogger(__name__)

# Honest, identifiable User-Agent: site operators can see who is fetching and how to reach the maintainer.
_CONTACT = os.getenv("ENTREZ_EMAIL") or os.getenv("OPENALEX_EMAIL") or "no-contact-configured"
USER_AGENT = f"ResearchLLM/1.0 (academic research assistant; mailto:{_CONTACT})"

# Limits for downloading a page or PDF chosen by a search result (see docs/security.md).
MAX_REDIRECTS = 5
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024  # a research PDF is a few MB; refuse anything above this
MAX_DOWNLOAD_SECONDS = 60  # one download in total (redirects and body); a read timeout alone cannot stop a slow drip
_REDIRECT_CODES = (301, 302, 303, 307, 308)


class _DownloadRefused(Exception):
    """A download was stopped on purpose (too large, too slow); the message says why."""


def _now() -> float:
    return time.monotonic()


def doc_digest(url: str) -> str:
    """A short identifier of a document URL that is the same in every run. Python's built-in hash() is
    randomised per process (so IDs changed with every restart) and, cut down to a few digits, collided."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def make_chunk_id(tool_id: str, url: str, index: int, abstract_only: bool = False) -> str:
    """'<tool>_<digest>_<n>' for full text, '<tool>_abs_<digest>_<n>' for the abstract fallback.

    The digest keeps IDs unique across documents; the trailing number is the chunk's position in its
    document, which RAGAgent uses to order neighbours (see RAGAgent._get_chunk_idx).
    """
    marker = "abs_" if abstract_only else ""
    return f"{tool_id}_{marker}{doc_digest(url)}_{index}"

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
        http = None
        try:
            time.sleep(1.5)
            headers = {
                'User-Agent': USER_AGENT,
                'Accept': 'application/pdf,text/html;q=0.9,*/*;q=0.5',
            }
            # One time budget for the whole download, redirects and body included: the per-read timeout alone
            # cannot stop a server that sends a few bytes at a time for hours.
            deadline = _now() + MAX_DOWNLOAD_SECONDS
            # A session that also verifies the address each connection really goes to (DNS rebinding).
            http = public_only_session()
            current = url
            for _ in range(MAX_REDIRECTS + 1):
                if _now() > deadline:
                    raise _DownloadRefused(f"took longer than {MAX_DOWNLOAD_SECONDS} s")
                # Checked before every request, redirects included: a harmless page may redirect to an internal address.
                reason = check_public_url(current)
                if reason:
                    logger.warning(f'[{self.id.upper()} BLOCKED] Not fetching {current[:80]}: {reason}.')
                    return None
                response = http.get(current, timeout=15, headers=headers, allow_redirects=False, stream=True)
                if response.status_code in _REDIRECT_CODES and response.headers.get('Location'):
                    current = urljoin(current, response.headers['Location'])
                    response.close()
                    continue
                break
            else:
                logger.warning(f'[{self.id.upper()} BLOCKED] More than {MAX_REDIRECTS} redirects from {url[:80]}.')
                return None

            if response.status_code == 403:
                response.close()
                logger.info(f'[{self.id.upper()} WAF] 403 Forbidden on {url[:40]}.')
                return None

            response.raise_for_status()
            body = self._read_capped(response, deadline)
            content_type = response.headers.get('Content-Type', '').lower()

            if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                return {'type': 'pdf', 'data': BytesIO(body)}
            if 'text/html' in content_type or b'<!doc' in body[:10].lower():
                return {'type': 'html', 'data': body.decode(response.encoding or 'utf-8', errors='replace')}
            return None
        except _DownloadRefused as refusal:
            logger.warning(f'[{self.id.upper()} BLOCKED] Stopped downloading {url[:80]}: {refusal}.')
            return None
        except Exception:
            return None
        finally:
            if http is not None:
                http.close()

    @staticmethod
    def _read_capped(response, deadline: Optional[float] = None) -> bytes:
        """The response body. Raises _DownloadRefused if it is larger than MAX_DOWNLOAD_BYTES (a hostile server
        could stream forever) or still arriving after the deadline (one that sends a few bytes at a time)."""
        declared = response.headers.get('Content-Length', '')
        if declared.isdigit() and int(declared) > MAX_DOWNLOAD_BYTES:
            response.close()
            raise _DownloadRefused(f"larger than {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB")
        body = bytearray()
        for piece in response.iter_content(chunk_size=65536):
            body.extend(piece)
            if len(body) > MAX_DOWNLOAD_BYTES:
                response.close()
                raise _DownloadRefused(f"larger than {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB")
            if deadline is not None and _now() > deadline:
                response.close()
                raise _DownloadRefused(f"took longer than {MAX_DOWNLOAD_SECONDS} s")
        return bytes(body)

    def _extract_text_from_pdf(self, pdf_stream: BytesIO) -> str:
        try:
            reader = PdfReader(pdf_stream)
            raw_text = " ".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""
        text, repaired = repair_pdf_text(raw_text)
        if repaired:
            logger.info(f'[{self.id.upper()}] Repaired {repaired} ligature-damaged word(s) in a PDF.')
        return text

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
                if current_chunk:  # a first sentence longer than the limit must not produce an empty chunk
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def execute(self, state: ResearchState) -> ResearchState:
        state.setdefault("visited_nodes", []).append(self.id)
        logger.info(f'[{self.id.upper()} START] Fetching and Processing Content...')

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

            logger.info(f'[{self.id.upper()} SUCCESS] Downloaded {target_url[:40]}...')

            text = self._extract_text_from_pdf(fetch_result['data']) if fetch_result['type'] == 'pdf' else self._extract_text_from_html(fetch_result['data'])
            if not text.strip():
                continue

            downloaded_urls_this_run.add(target_url)
            chunks = self._chunk_text(text)
            for i, chunk in enumerate(chunks):
                all_new_chunks.append({"chunk_id": make_chunk_id(entry['tool_id'], target_url, i), "doc_id": target_url, "text": chunk, "source": entry['tool_id']})

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
                all_new_chunks.append({"chunk_id": make_chunk_id(entry['tool_id'], url, i, abstract_only=True), "doc_id": url, "text": chunk, "source": entry['tool_id']})

        state.setdefault('full_text_chunks', []).extend(all_new_chunks)
        state["next"] = "supervisor_agent" # HUB-AND-SPOKE ROUTE
        logger.info(f'[{self.id.upper()} DONE] Retrieval complete.')
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
    ):
        self.id = agent_id
        self.max_chunks_to_keep = max_chunks_to_keep

        # Use the shared vector DB if one is passed in, otherwise open the default one (the embedding
        # function and its dimension come from backend/core/utilities.py).
        self.vector_db = vector_db if vector_db is not None else VectorDBWrapper()

        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def _passes_keyword_gate(self, chunk_text: str, literal_term: str, source: str = "") -> bool:
        """
        Optimized keyword gate: Bypasses trusted repository sources and targets
        actual web scraping/bot protection garbage instead of academic terms.
        """
        chunk_lower = chunk_text.lower()

        # 1. Trusted Source Bypass: Never filter out chunks originating from verified preprints or databases.
        # Chunks carry the tool's id ("arxiv_search"), so compare without the "_search" suffix.
        if (source or "").replace("_search", "") in ('arxiv', 'chemrxiv', 'materials'):
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
        logger.info(f'[{self.id.upper()} START] Reranking & Neighbor Expansion...')

        chunks_for_db = [c for c in state.get('full_text_chunks', []) if isinstance(c, dict) and c.get('text')]

        if client is None or not chunks_for_db:
            state.update({'filtered_context': "No relevant context found.", 'context_chunks': [], 'rag_complete': True, 'next': 'supervisor_agent'})
            logger.info(f'[{self.id.upper()} SKIPPED] No text chunks to process.')
            return state

        query = state.get('semantic_query', '')
        literal_term = state.get('api_search_term', '').lower()

        # 1. Vector DB Indexing with embedding model
        logger.info(f'[{self.id.upper()} INDEX] Adding {len(chunks_for_db)} chunks to vector database...')
        self.vector_db.add_chunks(chunks_for_db)

        # 2. Vector Search using embedding model (Top 30 candidates)
        top_k_results = self.vector_db.search(query, k=30)

        # 3. Cross-Encoder Reranking
        if top_k_results and query:
            logger.info(f'[{self.id.upper()} RERANK] Scoring top {len(top_k_results)} candidates...')
            sentence_pairs = [[query, res[0]['text']] for res in top_k_results]
            scores = self.reranker.predict(sentence_pairs)
            reranked_list = sorted([(top_k_results[i][0], scores[i]) for i in range(len(top_k_results))], key=lambda x: x[1], reverse=True)
            top_k_results = reranked_list
            ACTIVE_THRESHOLD = -5.0
        else:
            ACTIVE_THRESHOLD = 0.35

        # 4. Neighbor Expansion & Keyword Filtering
        # Group the chunks by document in reading order, and remember each chunk's position by its ID.
        doc_map: Dict[str, List[Dict[str, Any]]] = {}
        for c in self.vector_db.text_store:
            doc_id = c.get('doc_id')
            if doc_id:
                doc_map.setdefault(doc_id, []).append(c)

        position_of: Dict[str, int] = {}
        for family in doc_map.values():
            family.sort(key=self._get_chunk_idx)  # sequential by chunk index
            for pos, item in enumerate(family):
                if item.get("chunk_id"):
                    position_of[item["chunk_id"]] = pos

        final_chunks, selected_chunks, seen_ids = [], [], set()
        for chunk_dict, score in top_k_results:
            if score < ACTIVE_THRESHOLD:
                continue

            family = doc_map.get(chunk_dict.get("doc_id"), [])
            actual_idx = position_of.get(chunk_dict.get("chunk_id"))
            if actual_idx is None or family[actual_idx].get("chunk_id") != chunk_dict.get("chunk_id"):
                # Unknown or ambiguous ID: use this chunk on its own rather than another chunk's neighbours
                family, actual_idx = [chunk_dict], 0

            # Neighbor Expansion logic (1 chunk before, current, 1 chunk after)
            for i in range(max(0, actual_idx - 1), min(len(family), actual_idx + 2)):
                c = family[i]
                chunk_key = c.get("chunk_id") or id(c)

                # Pass source metadata into keyword gate
                if chunk_key not in seen_ids and self._passes_keyword_gate(c["text"], literal_term, c.get("source")):
                    final_chunks.append(c["text"])
                    # Keep where the text came from: synthesis can only cite what it can trace to a source
                    selected_chunks.append({key: c.get(key) for key in ("chunk_id", "doc_id", "source", "text")})
                    seen_ids.add(chunk_key)

            if len(final_chunks) >= self.max_chunks_to_keep:
                break

        # 5. Assemble Context
        structured_context = [f"--- Structured Data ---\n{d['text']}" for d in state.get('raw_tool_data', []) if d.get('tool_id') == 'materials_search']

        if structured_context or final_chunks:
            state['filtered_context'] = "\n---\n".join(structured_context + final_chunks)
        else:
            state['filtered_context'] = "No relevant context found."

        state['context_chunks'] = selected_chunks
        state['rag_complete'] = True
        state["next"] = "supervisor_agent" # HUB-AND-SPOKE ROUTE

        logger.info(f'[{self.id.upper()} DONE] RAG processing complete.')
        return state
