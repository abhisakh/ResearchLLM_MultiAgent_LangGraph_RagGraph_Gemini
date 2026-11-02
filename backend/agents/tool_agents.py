import logging
import requests
import re
import datetime
import os
from typing import Dict, Any, List, Optional
# --- Actual External Library Imports ---
from Bio import Entrez
from arxiv import Search, SortCriterion, SortOrder, Client as ArxivClient
from ddgs import DDGS
from mp_api.client import MPRester
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


# Relative imports from the modular structure
from backend.core.research_state import ResearchState
from backend.core.utilities import (
    MP_API_KEY, ENTREZ_EMAIL, OPENALEX_EMAIL
)

logger = logging.getLogger(__name__)

# ENTREZ_EMAIL and OPENALEX_EMAIL come from .env (see backend/core/utilities.py).


# ==================================================================================
# 1. BASE TOOL AGENT - FIX APPLIED
# ==================================================================================

class BaseToolAgent:
    """Base class for all tool agents to implement common logic and guardrails."""
    def __init__(self, agent_id: str):
        self.id = agent_id

    def _get_tool_key(self) -> str:
        """Standardizes IDs like 'pubmed_search' or 'pubmed_agent' into 'pubmed'."""
        return self.id.replace("_search", "").replace("_agent", "")

    def _should_run(self, state: ResearchState) -> bool:
        """Dynamic Guardrail: Check if the tool is in the active_tools list AND has valid input."""
        tool_key = self._get_tool_key()

        if tool_key not in state.get('active_tools', []):
            logger.info(f'[{self.id.upper()}] Skipping (Not in active_tools).')
            return False

        if tool_key not in ['materials']:
            if not state.get('tiered_queries', {}).get(tool_key):
                 logger.info(f'[{self.id.upper()}] Skipping (No queries available).')
                 return False

        if tool_key == 'materials':
            target_formula = state.get('api_search_term')
            if not target_formula or len(target_formula) < 2 or ":" in target_formula:
                 logger.info(f'[{self.id.upper()}] Skipping (Invalid/Missing api_search_term).')
                 return False

        return True

    def _get_query_data(self, state: ResearchState) -> Dict[str, str]:
        return state.get('tiered_queries', {}).get(self._get_tool_key(), {})

    def execute(self, state: ResearchState) -> ResearchState:
        """Template Method: Bookkeeping -> Guardrail -> run_tool_logic."""
        # 1. ALWAYS track visit for the Mermaid UI (Refinement-safe)
        if "visited_nodes" not in state or state["visited_nodes"] is None:
            state["visited_nodes"] = []
        state["visited_nodes"].append(self.id)

        # 2. Safety Check
        if not self._should_run(state):
            return state

        # 3. Call child logic (PubMed, Arxiv, etc.). One tool failing must not kill the run: the other
        # sources can still produce a report, so log the traceback and carry on without this tool.
        try:
            state = self.run_tool_logic(state)
        except Exception:
            logger.exception(f'[{self.id.upper()} ERROR] Tool failed; continuing without its results.')

        # 4. CRITICAL: Hand control back to the Central Hub
        state["next"] = "supervisor_agent"
        return state

    def run_tool_logic(self, state: ResearchState) -> ResearchState:
        raise NotImplementedError("Subclasses must implement run_tool_logic.")

# ==================================================================================
# 2. CONCRETE TOOL AGENTS
# ==================================================================================

class PubMedAgent(BaseToolAgent):
    """Queries PubMed using tiered searches via Entrez and ensures URLs are captured."""

    # Updated default agent_id to match ResearchGraph naming
    def __init__(self, agent_id: str = "pubmed_search", min_results: int = 5):
        super().__init__(agent_id)
        self.min_results = min_results
        self.query_order = ["strict", "moderate", "broad"]
        if ENTREZ_EMAIL:
            Entrez.email = ENTREZ_EMAIL
        else:
            logger.warning(f'[{self.id} WARNING] ENTREZ_EMAIL is not set in .env. NCBI may throttle or block requests.')

    def _execute_tiered_search(self, tiered_queries: Dict[str, str]) -> List[str]:
        pmids: List[str] = []
        for tier in self.query_order:
            current_query = tiered_queries.get(tier)
            if not current_query or not current_query.strip():
                continue
            logger.info(f"[{self.id} SEARCH] Trying '{tier}' query: '{current_query[:50]}...'")
            try:
                handle = Entrez.esearch(db="pubmed", term=current_query, retmax=self.min_results, sort="relevance")
                record = Entrez.read(handle)
                handle.close()
                ids = record.get("IdList", [])
                if ids:
                    return ids
            except Exception as e:
                logger.error(f"[{self.id} FAIL] Tier '{tier}' failed: {e}.")
        return []

    def _fetch_metadata_for_pmids(self, pmids: List[str]) -> List[Dict[str, Any]]:
        if not pmids:
            return []
        metadata_list = []
        try:
            handle = Entrez.efetch(db="pubmed", id=pmids, rettype="medline", retmode="xml")
            records = Entrez.read(handle)
            handle.close()

            for record in records.get('PubmedArticle', []):
                citation = record.get('MedlineCitation', {})
                article = citation.get('Article', {})
                pmid = str(citation.get('PMID', 'N/A'))
                title = str(article.get('ArticleTitle', 'No Title')).strip()
                abstract_data = article.get('Abstract', {}).get('AbstractText', [])
                abstract = " ".join([str(s) for s in abstract_data]).strip()

                # Predictable URL format for PubMed
                pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                text_content = f"Title: {title}. Abstract: {abstract}"

                metadata_list.append({
                    'text': text_content,
                    'source_type': 'pubmed',
                    'tool_id': self.id,
                    'metadata': {
                        'pmid': pmid,
                        'title': title,
                        'abstract': abstract,
                        'external_id': pmid,
                        'pdf_url': pubmed_url
                    }
                })
        except Exception as e:
            logger.error(f'[{self.id} ERROR] Metadata fetch failed: {e}')
            return []
        return metadata_list

    # --- CORE REFACTOR: Replacing execute with run_tool_logic ---
    def run_tool_logic(self, state: ResearchState) -> ResearchState:
        """
        The actual API interaction logic.
        Bookkeeping and Guardrails are now handled by BaseToolAgent.execute()
        """
        logger.info(f'[{self.id.upper()} START] Executing tiered PubMed search...')

        # Use helper from BaseToolAgent to get queries
        pubmed_queries = self._get_query_data(state)

        pmids = self._execute_tiered_search(pubmed_queries)
        if pmids:
            standardized_data = self._fetch_metadata_for_pmids(pmids)
            state.setdefault('raw_tool_data', []).extend(standardized_data)

            # Update references for the final report
            for r in standardized_data:
                ref_str = f"📄 Journal Article: {r['metadata']['title']}"
                state.setdefault('references', []).append(ref_str)

            logger.info(f'[{self.id.upper()} DONE] Added {len(standardized_data)} PubMed results.')
        else:
            logger.warning(f'[{self.id.upper()} WARNING] No relevant results retrieved.')

        return state


class ArxivAgent(BaseToolAgent):
    """Queries ArXiv using tiered searches with specific date filtering logic."""

    # Updated default agent_id to match ResearchGraph naming "arxiv_search"
    def __init__(self, agent_id: str = "arxiv_search", min_results: int = 5):
        super().__init__(agent_id)
        self.min_results = min_results
        self.query_order = ["strict", "moderate", "broad"]
        self.client_arxiv = ArxivClient()

    def _parse_time_constraint(self, state: ResearchState) -> Optional[str]:
        """Extracts the time period from the stable system_constraints."""
        for item in state.get("system_constraints", []):
            if item.startswith("TIME_PERIOD:"):
                return item.split(":", 1)[1].strip().lower()
        return None

    def _calculate_date_filter(self, time_period: str) -> str:
        """Constructs the ArXiv-specific Lucene query for date ranges."""
        if time_period == "last_decade":
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days=3652)
        else:
            # Only "last_decade" is supported. Say so instead of silently searching all years.
            logger.warning(f'[{self.id} WARN] Unsupported TIME_PERIOD "{time_period}": searching without a date filter.')
            return ""

        start_date_str = start_date.strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")
        return f" AND submitted:[{start_date_str} TO {end_date_str}]"

    def _call_arxiv_search(self, term: str, date_filter: str = "") -> List[Any]:
        """Executes the raw API call to ArXiv."""
        results = []
        full_term = f"{term}{date_filter}"
        try:
            search = Search(
                query=full_term,
                max_results=self.min_results,
                sort_by=SortCriterion.Relevance,
                sort_order=SortOrder.Descending
            )
            for r in self.client_arxiv.results(search):
                results.append(r)
        except Exception as e:
            logger.error(f'[{self.id} ERROR] ArXiv API failed: {str(e)}')
            return []
        return results

    def _standardize_arxiv_results(self, raw_results: List[Any]) -> List[Dict[str, Any]]:
        """Standardizes ArXiv data into the unified raw_tool_data format."""
        standardized_list = []
        for r in raw_results:
            summary_snippet = getattr(r, 'summary', 'No summary')[:500].replace('\n', ' ')
            text_content = f"Title: {r.title}. Abstract: {summary_snippet}"
            standardized_list.append({
                'text': text_content,
                'source_type': 'arxiv',
                'tool_id': self.id,
                'metadata': {
                    'arxiv_id': r.entry_id,
                    'title': r.title,
                    'abstract': summary_snippet,
                    'published_year': r.published.year,
                    'pdf_url': r.pdf_url
                }
            })
        return standardized_list

    # --- CORE REFACTOR: execute renamed to run_tool_logic ---
    def run_tool_logic(self, state: ResearchState) -> ResearchState:
        """
        Specific ArXiv retrieval logic.
        Note: visited_nodes and _should_run are handled by BaseToolAgent.
        """
        logger.info(f'[{self.id.upper()} START] Executing tiered ArXiv search...')

        # 1. Prepare Inputs
        arxiv_queries = self._get_query_data(state)
        time_period = self._parse_time_constraint(state)
        date_filter = self._calculate_date_filter(time_period) if time_period else ""

        if date_filter:
            logger.info(f'[{self.id} INFO] Applying ArXiv date filter: {date_filter.strip()}')

        # 2. Tiered Search Loop
        raw_results = []
        for tier in self.query_order:
            current_query = arxiv_queries.get(tier)
            if current_query and current_query.strip():
                raw_results = self._call_arxiv_search(current_query, date_filter)
                if raw_results:
                    logger.info(f"[{self.id} SUCCESS] Found results in '{tier}' tier.")
                    break

        # 3. State Update
        if raw_results:
            standardized_data = self._standardize_arxiv_results(raw_results)
            state.setdefault('raw_tool_data', []).extend(standardized_data)

            # Add to readable references for the final report
            new_refs = [f"🔗 Arxiv: {r['metadata']['title']}" for r in standardized_data]
            state.setdefault('references', []).extend(new_refs)

            logger.info(f'[{self.id.upper()} DONE] Added {len(standardized_data)} ArXiv papers.')

        return state

class OpenAlexAgent(BaseToolAgent):
    """Queries OpenAlex for works related to the research query."""

    # Updated default agent_id to match ResearchGraph naming "openalex_search"
    def __init__(self, agent_id: str = "openalex_search", max_results: int = 10):
        super().__init__(agent_id)
        self.max_results = max_results
        self.base_url = "https://api.openalex.org/works"
        self.email_for_polite_pool = OPENALEX_EMAIL
        if not self.email_for_polite_pool:
            logger.warning(f'[{self.id} WARNING] OPENALEX_EMAIL is not set in .env. Requests use the slower common pool.')

    def _call_openalex_api(self, query: str) -> List[Dict[str, Any]]:
        """Handles the HTTP request to the OpenAlex API."""
        # OpenAlex treats "," (AND) and "|" (OR) inside a filter as separators, so a query containing
        # them would be cut into several filters. Replace them with spaces.
        clean_query = re.sub(r"[,|]", " ", query).strip()

        # params= lets requests URL-encode the value instead of splicing the raw query into the URL.
        params = {
            # title.search filter provides high-relevance matches for specific research queries
            "filter": f"title.search:{clean_query}",
            "per-page": self.max_results,
        }
        if self.email_for_polite_pool:
            params["mailto"] = self.email_for_polite_pool

        try:
            r = requests.get(self.base_url, params=params, timeout=10)
            r.raise_for_status()
            return r.json().get("results", [])
        except requests.exceptions.RequestException as e:
            logger.error(f'[{self.id} ERROR] OpenAlex request failed: {e}')
            return []

    def _reconstruct_openalex_abstract(self, inverted_index: Dict[str, List[int]]) -> str:
        """
        OpenAlex provides abstracts in an 'inverted index' format for legal/copyright reasons.
        This method maps words back to their original positions to create a readable string.
        """
        if not inverted_index: return "Abstract unavailable."

        try:
            all_indices = [idx for indices in inverted_index.values() for idx in indices]
            if not all_indices: return "Abstract unavailable."

            max_index = max(all_indices)
            words = [None] * (max_index + 1)

            for word, indices in inverted_index.items():
                for index in indices:
                    if 0 <= index < len(words):
                        words[index] = word

            return " ".join(word for word in words if word is not None).strip()

        except Exception as e:
            logger.warning(f'[{self.id} WARN] Abstract reconstruction failed: {e}')
            return "Abstract reconstruction failed."

    def _standardize_openalex_results(self, raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Converts raw API response into standardized internal research format."""
        standardized_list = []
        for r in raw_results:
            # The API can return null (not just a missing key) for the title and the abstract
            title = r.get('title') or 'No Title Available'
            abstract = self._reconstruct_openalex_abstract(r.get('abstract_inverted_index') or {})
            text_content = f"Title: {title}. Abstract: {abstract}"

            standardized_list.append({
                'text': text_content,
                'source_type': 'openalex',
                'tool_id': self.id,
                'metadata': {
                    'openalex_id': r.get('id', 'N/A'),
                    'title': title,
                    # OpenAlex often provides direct landing page or PDF links
                    'pdf_url': (r.get('primary_location') or {}).get('landing_page_url')
                }
            })
        return standardized_list

    # --- CORE REFACTOR: execute renamed to run_tool_logic ---
    def run_tool_logic(self, state: ResearchState) -> ResearchState:
        """
        Specific OpenAlex retrieval logic.
        Breadcrumbs and Guardrails are inherited from BaseToolAgent.
        """
        logger.info(f'[{self.id.upper()} START] Retrieving open-access data...')

        # Use helper from BaseToolAgent to get tool-specific queries
        search_query = self._get_query_data(state).get('simple')

        if not search_query or not search_query.strip():
             logger.error(f'[{self.id} FAIL] No search query found in state. Skipping.')
             return state

        raw_results = self._call_openalex_api(search_query)

        if raw_results:
            standardized_data = self._standardize_openalex_results(raw_results)

            # Store data for RAG and Synthesis
            state.setdefault('raw_tool_data', []).extend(standardized_data)

            # Record references
            new_refs = [f"🔗 OpenAlex: {r['metadata']['title']}" for r in standardized_data]
            state.setdefault('references', []).extend(new_refs)

            logger.info(f'[{self.id.upper()} DONE] Added {len(standardized_data)} works.')
        else:
            logger.info(f"[{self.id.upper()} INFO] No results for query: '{search_query[:80]}'")

        return state


class MaterialsAgent(BaseToolAgent):
    """Queries Materials Project for material properties."""

    # Updated default agent_id to match ResearchGraph naming "materials_search"
    def __init__(self, agent_id: str = "materials_search", max_results: int = 5):
        super().__init__(agent_id)
        self.max_results = max_results

    def _call_materials_project_api(self, formula: str, max_results: int) -> List[Dict[str, Any]]:
        """Handles the specialized MPRester client connection and search."""
        if not MP_API_KEY:
            logger.error(f'[{self.id} ERROR] MP_API_KEY is not set.')
            return []

        try:
            with MPRester(MP_API_KEY, use_document_model=False) as mpr:
                docs = mpr.materials.summary.search(
                    formula=formula,
                    fields=[
                        "material_id",
                        "formula_pretty",
                        "is_stable",
                        "band_gap",
                        "energy_above_hull"
                    ],
                    num_chunks=1,
                    chunk_size=max_results
                )

                results = []
                for d in docs:
                    # mpr.summary.search returns a list of dictionaries or Document objects
                    results.append({
                        "material_id": d.get("material_id", "N/A"),
                        "formula": d.get("formula_pretty", "N/A"),
                        "is_stable": d.get("is_stable", False),
                        "band_gap": d.get("band_gap", None),
                        "energy_above_hull": d.get("energy_above_hull", None)
                    })
                return results

        except Exception as e:
            logger.error(f'[{self.id} ERROR] MP API call failed: {type(e).__name__}: {e}')
            return []

    def _standardize_mp_results(self, raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Standardizes Materials Project data with LaTeX formatting for scientific values."""
        standardized_list = []
        for result in raw_results:
            stability_status = "Stable" if result['is_stable'] else f"Unstable (E/hull: {result['energy_above_hull']} eV)"

            band_gap_val = result['band_gap']
            band_gap_str = fr"${band_gap_val}\ \text{{eV}}$" if band_gap_val is not None else "Metallic/Unknown"

            energy_above_hull_val = result['energy_above_hull']
            energy_above_hull_str = rf"${energy_above_hull_val}\ \text{{eV}}$" if energy_above_hull_val is not None else "N/A"

            text_content = (
                f"Material: {result['formula']} ({result['material_id']}). "
                f"Stability: {stability_status}. "
                f"Band Gap: {band_gap_str}. "
                f"Energy Above Hull: {energy_above_hull_str}."
            )

            standardized_list.append({
                'text': text_content,
                'source_type': 'materials_project',
                'tool_id': self.id,
                'metadata': {
                    'material_id': result['material_id'],
                    'formula': result['formula'],
                    'is_stable': result['is_stable'],
                    'band_gap': result['band_gap'],
                    'energy_above_hull': result['energy_above_hull'],
                }
            })
        return standardized_list

    # --- CORE REFACTOR: execute renamed to run_tool_logic ---
    def run_tool_logic(self, state: ResearchState) -> ResearchState:
        """
        Specific Materials Project retrieval logic.
        Validates target_formula and updates the state with thermodynamic data.
        """
        logger.info(f'[{self.id.upper()} START] Retrieving material properties...')

        # Retrieve target formula (guaranteed valid by BaseToolAgent._should_run)
        target_formula = state.get('api_search_term')

        # Execute API Call
        raw_results = self._call_materials_project_api(target_formula, self.max_results)

        if raw_results:
            standardized_data = self._standardize_mp_results(raw_results)
            state.setdefault('raw_tool_data', []).extend(standardized_data)

            # Use formula_pretty for the reference if available
            formula_for_ref = raw_results[0].get('formula', target_formula)

            new_refs = [
                f"⚛️ Materials Project: {r['metadata']['material_id']} ({formula_for_ref})"
                for r in standardized_data
            ]
            state.setdefault('references', []).extend(new_refs)

            logger.info(f'[{self.id.upper()} DONE] Added {len(standardized_data)} entries for {target_formula}.')
        else:
             logger.warning(f"[{self.id.upper()} WARNING] No MP data found for '{target_formula}'.")

        return state


class WebAgent(BaseToolAgent):
    """Performs web search via DuckDuckGo with automated noise filtering."""

    # Updated default agent_id to match ResearchGraph naming "web_search"
    def __init__(self, agent_id: str = "web_search"):
        super().__init__(agent_id)
        # Dynamic list of patterns to exclude from research data
        self.noise_patterns = [
            'google.com/help', 'support.google', 'whatsapp.com',
            'stackoverflow.com', 'facebook.com', 'instagram.com',
            'login', 'signin', 'signup', 'gmail help', 'youtube.com/watch'
        ]

    def _call_ddg_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Executes the raw DuckDuckGo search."""
        try:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=limit))
        except Exception as e:
            logger.error(f'[{self.id} ERROR] DDGS search failed: {e}')
            return []

    def _standardize_web_results(self, raw_results: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Standardizes web results while applying academic noise filters."""
        standardized_list = []
        for result in raw_results:
            url = (result.get('href') or '').lower()
            title = (result.get('title') or '').lower()

            # A result without a link cannot be cited or retrieved
            if not url:
                continue

            # --- DYNAMIC FILTERING ---
            # 1. Block known noise domains and social media
            if any(pattern in url for pattern in self.noise_patterns):
                continue

            # 2. Block generic 'how-to' or support titles that aren't scientific
            if any(p in title for p in ['how to', 'transfer', 'account help', 'set up']):
                continue

            text_content = f"Title: {result.get('title')}. Snippet: {result.get('body')}"
            standardized_list.append({
                'text': text_content,
                'source_type': 'web_search',
                'tool_id': self.id,
                'metadata': {
                    'title': result.get('title'),
                    'url': result.get('href'),
                    'source_name': 'DuckDuckGo Search'
                }
            })
        return standardized_list

    # --- CORE REFACTOR: execute renamed to run_tool_logic ---
    def run_tool_logic(self, state: ResearchState) -> ResearchState:
        """
        Specific Web Search retrieval logic.
        Uses specialized academic-intent queries from the state.
        """
        logger.info(f'[{self.id.upper()} START] Retrieving context (Filtered Web Search)...')

        # 1. Get query (Priority: simple -> broad -> semantic fallback)
        queries = self._get_query_data(state)
        focused_query = queries.get('simple') or queries.get('broad') or state.get('semantic_query', 'research')

        # 2. Run search with overhead for filtering
        raw_results = self._call_ddg_search(query=focused_query, limit=12)

        if raw_results:
            standardized_results = self._standardize_web_results(raw_results)

            if standardized_results:
                # Add to raw_tool_data for RAG processing
                state.setdefault('raw_tool_data', []).extend(standardized_results)

                # Add clean URLs to references
                new_refs = [
                    f"🔗 Web Source: {r['metadata']['title']} ({r['metadata']['url']})"
                    for r in standardized_results
                ]
                state.setdefault('references', []).extend(new_refs)

                logger.info(f'[{self.id.upper()} DONE] Added {len(standardized_results)} relevant web sources.')
            else:
                logger.warning(f'[{self.id} WARN] All results filtered out as non-academic noise.')
        else:
            logger.warning(f"[{self.id} WARN] No web results found for: '{focused_query[:40]}...'")

        return state

##========================= Additional RESEARCH Tool ===============================

class ChemRxivAgent(BaseToolAgent):
    """Queries ChemRxiv (via Figshare API) for chemistry and material science preprints."""

    def __init__(self, agent_id: str = "chemrxiv_search", max_results: int = 5):
        super().__init__(agent_id)
        self.max_results = max_results
        # Use the specific search endpoint
        self.base_url = "https://api.figshare.com/v2/articles/search"

    def _call_chemrxiv_api(self, query: str) -> List[Dict[str, Any]]:
        """Handles POST request with sanitized query logic."""

        # 1. SANITIZE QUERY: Figshare hates AND/OR/quotes in their basic search_for key
        # We strip boolean logic to prevent 400 Bad Request
        clean_query = query.replace(" AND ", " ").replace(" OR ", " ").replace('"', '').strip()

        # 2. UPDATED PAYLOAD: Using the documented search schema
        payload = {
            "search_for": clean_query,
            "institution": 259,  # ChemRxiv ID
            "page_size": self.max_results,
            "order": "published_date",
            "item_type": 1 # 1 is for 'article' (includes preprints)
        }

        try:
            # Use a User-Agent header; some Figshare nodes throttle generic Python-requests agents
            headers = {"User-Agent": "ResearchAssistant/1.0 (Scientific Research Bot)"}

            r = requests.post(self.base_url, json=payload, headers=headers, timeout=15)

            # Print specific error message from the server if it fails
            if r.status_code != 200:
                logger.error(f'[{self.id.upper()} ERROR] API Status {r.status_code}: {r.text}')
                return []

            data = r.json()
            # An error body is JSON too, but a dict: only a list holds preprints
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f'[{self.id.upper()} ERROR] API request failed: {e}')
            return []

    def _standardize_results(self, raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        standardized = []
        for r in raw_results:
            title = r.get('title', 'No Title')
            # Extract description and remove HTML tags
            description = r.get('description', '') or r.get('snippet', '')
            clean_text = re.sub('<[^<]+?>', '', description).strip()

            # Use DOI link as primary if public_html is missing
            url = r.get('url_public_html') or f"https://doi.org/{r.get('doi')}"

            standardized.append({
                'text': f"Title: {title}. Abstract: {clean_text[:1200]}",
                'source_type': 'chemrxiv',
                'tool_id': self.id,
                'url': url, # Added at top level for RetrievalAgent
                'metadata': {
                    'title': title,
                    'pdf_url': url,
                    'doi': r.get('doi'),
                    'published_date': r.get('published_date')
                }
            })
        return standardized

    def run_tool_logic(self, state: ResearchState) -> ResearchState:
        logger.info(f'[{self.id.upper()} START] Querying ChemRxiv...')

        # Fallback logic for queries
        queries = state.get('tiered_queries', {}).get('chemrxiv', {})
        search_query = queries.get('simple') or state.get('semantic_query')

        if not search_query:
            logger.error(f'[{self.id.upper()} FAIL] No search query available.')
            return state

        raw_data = self._call_chemrxiv_api(search_query)

        if raw_data:
            standardized = self._standardize_results(raw_data)
            # Ensure raw_tool_data exists
            if 'raw_tool_data' not in state:
                state['raw_tool_data'] = []

            state['raw_tool_data'].extend(standardized)

            new_refs = [f"🧪 ChemRxiv: {r['metadata']['title']}" for r in standardized]
            if 'references' not in state:
                state['references'] = []
            state['references'].extend(new_refs)

            logger.info(f'[{self.id.upper()} DONE] Successfully added {len(standardized)} preprints.')
        else:
            logger.info(f'[{self.id.upper()} INFO] No results found for query: {search_query}')

        return state

class SemanticScholarAgent(BaseToolAgent):
    """
    Semantic Scholar agent. Works without an API key (public tier, heavily rate limited, so it fails
    fast) and uses S2_API_KEY from .env when present (much higher limits, more patient retries).
    """
    def __init__(self, agent_id: str = "semanticscholar_search", max_results: int = 5):
        super().__init__(agent_id)
        self.max_results = max_results
        self.search_url = "https://api.semanticscholar.org/graph/v1/paper/search"

        # Safe check for API Key
        self.api_key = os.getenv("S2_API_KEY")

        self.session = requests.Session()
        # Without an API key the public tier answers most requests with HTTP 429, and waiting minutes
        # for retries just stalls the whole pipeline: fail fast and let the other tools carry on.
        # With a key the limits are far looser, so retry more patiently.
        if self.api_key:
            retry_strategy = Retry(
                total=4,
                backoff_factor=2,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET"]
            )
        else:
            retry_strategy = Retry(
                total=2,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET"],
                respect_retry_after_header=False
            )
        self.session.mount("https://", HTTPAdapter(max_retries=retry_strategy))

    def _call_s2_api(self, query: str):
        # Clean query for public tier reliability
        clean_query = re.sub(r'[^\w\s-]', '', query).strip()

        params = {
            "query": clean_query,
            "limit": self.max_results,
            "fields": "paperId,title,abstract,url,openAccessPdf,venue,year,citationCount"
        }

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            # The session handles the 429 retries automatically
            r = self.session.get(self.search_url, params=params, headers=headers, timeout=25)
            r.raise_for_status()
            return r.json().get('data', [])
        except Exception as e:
            # This will now catch and report errors without crashing the whole backend
            logger.error(f'[S2 ERROR] Public Tier limit or error: {e}')
            return []

    def _standardize_s2_results(self, raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Converts Semantic Scholar papers into the unified raw_tool_data format."""
        standardized_list = []
        for r in raw_results:
            title = (r.get('title') or '').strip()
            if not title:
                continue
            # S2 often returns abstract=null, and openAccessPdf=null for closed-access papers
            abstract = (r.get('abstract') or '').strip() or "Abstract unavailable."
            open_access = r.get('openAccessPdf') or {}
            page_url = r.get('url')
            pdf_url = open_access.get('url') or page_url

            standardized_list.append({
                'text': f"Title: {title}. Abstract: {abstract[:1200]}",
                'source_type': 'semanticscholar',
                'tool_id': self.id,
                'metadata': {
                    's2_id': r.get('paperId'),
                    'title': title,
                    'abstract': abstract,
                    'venue': r.get('venue'),
                    'published_year': r.get('year'),
                    'citation_count': r.get('citationCount'),
                    'pdf_url': pdf_url,
                }
            })
        return standardized_list

    def run_tool_logic(self, state: ResearchState) -> ResearchState:
        logger.info(f'[{self.id.upper()} START] Searching S2 (Public Tier Mode)...')

        # Query priority: moderate tier -> strict tier -> the cleaned user query
        queries = state.get('tiered_queries', {}).get('semanticscholar', {})
        search_query = queries.get('moderate') or queries.get('strict') or state.get('semantic_query')

        if not search_query or not search_query.strip():
            logger.error(f'[{self.id.upper()} FAIL] No search query available.')
            return state

        raw_data = self._call_s2_api(search_query)
        standardized_data = self._standardize_s2_results(raw_data)

        if standardized_data:
            state.setdefault('raw_tool_data', []).extend(standardized_data)
            state.setdefault('references', []).extend(
                f"🔗 Semantic Scholar: {r['metadata']['title']}" for r in standardized_data
            )
            logger.info(f'[{self.id.upper()} DONE] Added {len(standardized_data)} papers.')
        else:
            logger.info(f'[{self.id.upper()} INFO] No results. This is common on Public Tier due to rate limits.')

        return state
