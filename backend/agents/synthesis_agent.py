import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from google.genai import types

# Relative imports from the modular structure
from backend.core.errors import record_error, to_llm_error, user_message
from backend.core.utilities import (
    LLM_MODEL, generate_content_with_retry, llm_text
)

logger = logging.getLogger(__name__)

# Shown after a reference, so the reader can see which database a source came from
TOOL_LABELS = {
    "arxiv_search": "arXiv",
    "pubmed_search": "PubMed",
    "openalex_search": "OpenAlex",
    "chemrxiv_search": "ChemRxiv",
    "semanticscholar_search": "Semantic Scholar",
    "web_search": "Web",
    "materials_search": "Materials Project",
}
MATERIALS_URL = "https://next-gen.materialsproject.org/materials/{}"
NOISE_URL_PARTS = ("google.com/help", "support.google", "login", "signin", "signup")  # shown as plain text, not linked

MAX_SNIPPET_SOURCES = 5   # sources offered to the model when RAG found nothing to quote
SNIPPET_CHARS = 300

_CITATION = re.compile(r"\[(\d+)\]")
_CITATION_LIST = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)+)\]")  # "[1, 2]" is written as "[1][2]"
_CITATION_RUN = re.compile(r"(?:\[\d+\]){2,}")  # adjacent citations such as "[4][3]"
_REFERENCES_HEADING = re.compile(r"(?im)^[ \t]{0,3}(?:#{1,6}[ \t]*)?(?:\*\*)?references(?:\*\*)?[ \t]*:?[ \t]*$")

# Markdown the model may have put in its prose. The report's real links are built by code (the reference list), so any
# other link or image is something the model was talked into writing: an image makes the reader's browser contact
# whatever address it names, and a link can send the reader to a page of an attacker's choosing.
_MD_IMAGE = re.compile(r"[ \t]*!\[[^\]\n]*\]\([^)\n]*\)")
_MD_LINK = re.compile(r"\[([^\]\n]+)\]\([^)\n]*\)")


def _normalise_url(url: Optional[str]) -> Optional[str]:
    """The form in which RetrievalAgent stores a document's URL as the chunks' doc_id."""
    return url.strip().rstrip(".") if isinstance(url, str) and url.strip() else None


GENERIC_TITLES = {"untitled source", "unidentified source", "no title available"}
MIN_TITLE_KEY_CHARS = 20  # shorter titles ("Home", "CsPbI3") do not identify a paper


def _title_key(title: Optional[str]) -> Optional[str]:
    """A key by which two listings of the same paper are recognised, or None if the title is too generic.

    Case, punctuation and a trailing "..." (search engines cut long titles) do not matter. Only the same title
    is merged, never a similar one: wrongly merging two different papers would hide a source.
    """
    if not title or title.strip().lower() in GENERIC_TITLES:
        return None
    key = re.sub(r"[\W_]+", " ", title.lower()).strip()
    return key if len(key) >= MIN_TITLE_KEY_CHARS else None


def _source_key(described: Dict[str, Any], fallback: str) -> str:
    """One key per paper: its title when that identifies it, else the given fallback (a URL or an index)."""
    title_key = _title_key(described.get("title"))
    return f"title:{title_key}" if title_key else fallback


def _describe(entry: Dict[str, Any]) -> Dict[str, Any]:
    """The title and link by which a raw_tool_data entry is cited."""
    meta = entry.get("metadata") or {}
    tool_id = entry.get("tool_id", "")

    if tool_id == "materials_search":
        material_id, formula = meta.get("material_id"), meta.get("formula")
        title = f"{formula} ({material_id})" if formula and material_id else (material_id or "Materials Project entry")
        url = MATERIALS_URL.format(material_id) if material_id else None
    else:
        title = meta.get("title") or "Untitled source"
        url = _normalise_url(meta.get("pdf_url") or meta.get("url") or entry.get("url") or meta.get("openalex_id"))

    return {"title": str(title).strip(), "url": url, "tool_id": tool_id}


class SynthesisAgent:
    """
    Writes the final report from numbered evidence.

    Citation provenance: the model only sees sources that carry a number, and it may only cite those numbers.
    After it answers, code (not the model) drops citations of numbers that were never shown, renumbers the
    rest in order of first appearance, and builds the reference list from the sources actually cited. Whatever
    the model wrote as its own "References" section is discarded, so an invented source cannot reach the report.
    """
    def __init__(self, agent_id: str = "synthesis_agent", model: str = LLM_MODEL):
        self.id = agent_id
        self.model = model

    # =====================================================
    # 1. MATERIAL DATA EXTRACTION (used for the report's heading)
    # =====================================================
    def _extract_material_data(self, state: Dict) -> Tuple[str, str, bool]:
        target_formula = state.get('material_formula', state.get('api_search_term', 'N/A'))
        materials_results = [d for d in state.get("raw_tool_data", []) if d.get("tool_id") == "materials_search"]
        material_data = [result.get('text', 'N/A') for result in materials_results]

        data_is_present = bool(material_data)
        summary = "\n".join(material_data) if data_is_present else f"No material property data was retrieved for {target_formula}."
        return summary, target_formula, data_is_present

    # =====================================================
    # 2. NUMBERED EVIDENCE: what the model may cite
    # =====================================================
    def _collect_sources(self, state: Dict) -> Tuple[List[Dict[str, Any]], bool]:
        """The sources the model is shown, numbered from 1, each with the passages it may quote.

        Returns (sources, snippets_only). Order: Materials Project entries, then the documents behind the
        chunks RAG selected (in RAG order). If RAG selected nothing, the first few tool results are offered
        as short snippets instead (snippets_only=True), because the model must not cite text it never saw.
        """
        raw = [e for e in state.get("raw_tool_data", []) if isinstance(e, dict)]

        by_url: Dict[str, Dict[str, Any]] = {}
        for entry in raw:
            meta = entry.get("metadata") or {}
            for url in (meta.get("pdf_url"), meta.get("url"), entry.get("url")):
                key = _normalise_url(url)
                if key:
                    by_url.setdefault(key, entry)

        sources: List[Dict[str, Any]] = []
        position: Dict[str, int] = {}

        def add(key: str, described: Dict[str, Any], passage: str) -> None:
            if key not in position:
                position[key] = len(sources)
                sources.append({"number": len(sources) + 1, **described, "passages": []})
            if passage:
                sources[position[key]]["passages"].append(passage)

        for index, entry in enumerate(raw):
            if entry.get("tool_id") == "materials_search":
                add(f"materials:{index}", _describe(entry), str(entry.get("text", "")).strip())

        chunks = [c for c in state.get("context_chunks", []) if isinstance(c, dict) and (c.get("text") or "").strip()]
        for chunk in chunks:
            doc = _normalise_url(chunk.get("doc_id"))
            entry = by_url.get(doc) if doc else None
            described = _describe(entry) if entry else {
                "title": doc or "Unidentified source", "url": doc, "tool_id": chunk.get("source") or "",
            }
            add(_source_key(described, f"doc:{doc or chunk.get('chunk_id')}"), described, chunk["text"].strip())

        snippets_only = not chunks
        if snippets_only:
            taken = 0
            for index, entry in enumerate(raw):
                if taken >= MAX_SNIPPET_SOURCES:
                    break
                if entry.get("tool_id") == "materials_search" or not entry.get("text"):
                    continue
                described = _describe(entry)
                add(_source_key(described, f"snippet:{index}"), described, str(entry["text"])[:SNIPPET_CHARS])
                taken += 1

        return sources, snippets_only

    def _format_evidence(self, sources: List[Dict[str, Any]], snippets_only: bool) -> str:
        if not sources:
            return "(No evidence was retrieved. Say that the question cannot be answered from retrieved sources.)"
        blocks = []
        for source in sources:
            link = f" ({source['url']})" if source.get("url") else ""
            blocks.append(f"[{source['number']}] {source['title']}{link}\n" + "\n\n".join(source["passages"]))
        note = ""
        if snippets_only:
            note = "Only short snippets are available, so keep every claim modest.\n\n"
        return note + "\n\n".join(blocks)

    # =====================================================
    # 3. PROMPT
    # =====================================================
    def _format_prompt(self, state: Dict, sources: List[Dict[str, Any]], snippets_only: bool) -> str:
        query = state.get("semantic_query", "No query provided")
        _, target_formula, data_is_present = self._extract_material_data(state)
        heading = f"## Stability and Bandgap of {target_formula}" if data_is_present else "## Introduction and Scope of Review"
        evidence = self._format_evidence(sources, snippets_only)

        return f"""
        [EVIDENCE]
        Every source is numbered. These are the only sources you may use.

        {evidence}

        [OBJECTIVE]
        Generate a scientific report for: "{query}".

        [MANDATORY RULES]
        1. Support every factual claim with a citation such as [1] or [2][3], using ONLY the numbers shown in EVIDENCE.
        2. Never cite a number that is not in EVIDENCE, and never invent a source, title, author or link.
        3. Do NOT write a References section: it is generated automatically from the sources you cite.
        4. If the evidence does not cover something, say so instead of filling the gap from memory.
        5. STRUCTURE: {heading} | Key Findings | Conclusion
        """

    # =====================================================
    # 4. CITATIONS: validated, renumbered and listed by code
    # =====================================================
    def _strip_references(self, report: str) -> str:
        """The report without any References section the model wrote on its own."""
        match = _REFERENCES_HEADING.search(report)
        return report[:match.start()] if match else report

    def _strip_links_and_images(self, body: str) -> str:
        """The prose without Markdown images and links: images are dropped, links keep their visible text.
        A numbered link such as "[1](https://...)" keeps its "[1]", so the citation itself survives."""
        body, images = _MD_IMAGE.subn("", body)

        def keep_text(match: "re.Match[str]") -> str:
            text = match.group(1)
            return f"[{text}]" if text.strip().isdigit() else text

        body, links = _MD_LINK.subn(keep_text, body)
        if images or links:
            logger.warning(f'[SYNTHESIS] Removed {images} image(s) and {links} link(s) the model wrote in the report text.')
        return body

    def _finalize_citations(self, report: str, sources: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]], int]:
        """(report with its reference list, the sources it cites, number of invalid citations removed).

        A citation counts only if its number belongs to a source the model was shown. The rest are removed.
        Valid citations are renumbered in order of first appearance, and the reference list contains exactly
        the sources cited.
        """
        body = self._strip_references(report).strip()
        body = self._strip_links_and_images(body)
        body = _CITATION_LIST.sub(lambda m: "".join(f"[{n}]" for n in re.split(r"\s*[,;]\s*", m.group(1))), body)

        shown = {source["number"]: source for source in sources}
        new_number: Dict[int, int] = {}
        removed = 0

        def renumber(match: "re.Match[str]") -> str:
            nonlocal removed
            number = int(match.group(1))
            if number not in shown:
                removed += 1
                return ""
            if number not in new_number:
                new_number[number] = len(new_number) + 1
            return f"[{new_number[number]}]"

        body = _CITATION.sub(renumber, body)
        if removed:
            body = re.sub(r"[ \t]+([.,;:])", r"\1", body)  # a removed citation must not leave a gap before punctuation
            logger.warning(f'[SYNTHESIS] Removed {removed} citation(s) of sources that were never shown to the model.')

        # Adjacent citations read best in ascending order ("[4][3]" -> "[3][4]") and never repeat ("[2][2]" -> "[2]").
        # Only the order inside a run changes: the numbering by first appearance above, and so the list of cited
        # sources, stays as it is.
        body = _CITATION_RUN.sub(
            lambda run: "".join(f"[{n}]" for n in sorted({int(d) for d in re.findall(r"\d+", run.group(0))})), body
        )

        cited = []
        for old, new in sorted(new_number.items(), key=lambda pair: pair[1]):
            source = shown[old]
            cited.append({"number": new, "title": source["title"], "url": source.get("url"), "tool_id": source.get("tool_id", "")})

        return f"{body}\n\n## References\n\n{self._format_references(cited)}", cited, removed

    def _format_references(self, cited: List[Dict[str, Any]]) -> str:
        if not cited:
            return "_No sources were cited._"
        lines = []
        for source in cited:
            title = source["title"].replace("[", "(").replace("]", ")")
            url = source.get("url")
            linked = f"[{title}]({url})" if url and not any(part in url.lower() for part in NOISE_URL_PARTS) else title
            label = TOOL_LABELS.get(source.get("tool_id", ""))
            lines.append(f"[{source['number']}] {linked}" + (f" ({label})" if label else ""))
        return "\n\n".join(lines)

    # =====================================================
    # 5. EXECUTION
    # =====================================================
    def execute(self, state: Dict) -> Dict:
        state.setdefault("visited_nodes", []).append(self.id)

        # Intent Guardrail
        if state.get("primary_intent") == "irrelevant":
            logger.warning('[SYNTHESIS] Rejecting irrelevant query.')
            state['final_report'] = "Query rejected based on scope."
            state['report_generated'] = True
            state['next'] = 'supervisor_agent'
            return state

        logger.info('[SYNTHESIS START] Writing report from numbered evidence via Gemini...')
        sources, snippets_only = self._collect_sources(state)
        prompt = self._format_prompt(state, sources, snippets_only)

        try:
            raw_report = llm_text(
                lambda: generate_content_with_retry(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction="You are a scientific reporting assistant. Use Markdown for all formatting.",
                        temperature=0.1
                    )
                ),
                label=self.id,
            )

            report, cited, removed = self._finalize_citations(raw_report, sources)
            state['final_report'] = report
            state['sources'] = cited
            state['invalid_citations_removed'] = removed
            state['report_generated'] = True
            # Earlier failures of this agent were recovered from. Dropping them keeps the supervisor's
            # "too many failed syntheses" count to consecutive failures, also across refinement cycles.
            state['errors'] = [e for e in state.get('errors', []) if e.get('agent') != self.id]
            logger.info(f'[SYNTHESIS DONE] Report generated with {len(cited)} cited source(s).')
        except Exception as exc:
            # No report could be written. report_generated stays False and the failure is recorded, so the
            # supervisor can tell a failed synthesis from a finished one and stop instead of looping.
            error = to_llm_error(exc, label=self.id)
            record_error(state, self.id, error)
            logger.error(f'[SYNTHESIS ERROR] {error}')
            state['final_report'] = f"Error generating report. {user_message(error.kind)}"

        state['next'] = 'supervisor_agent'
        return state
