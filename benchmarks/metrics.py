"""Metrics for the benchmark: pure functions of a finished run's state, so they are tested without any API.

What they measure, and what they do not (details in benchmarks/README.md):
- cost and speed (seconds, requests, tokens, refinement cycles) are exact;
- citation numbers are checked by code: how many were invalid is counted, and the final report cannot contain one;
- "citation coverage" and the "grounding score" are HEURISTICS on the report text. They flag reports worth reading
  closely. They cannot prove that a claim is true, and a low score is a prompt to check, not a verdict.
"""
import re
import statistics
from typing import Any, Dict, List, Optional

REJECTION_TEXT = "Query rejected based on scope."   # what SynthesisAgent returns for an out-of-scope question
MATERIALS_URL = "https://next-gen.materialsproject.org/materials/{}"
MIN_CLAIM_WORDS = 8   # shorter sentences ("Key Findings", "See below.") are not counted as claims

_REFERENCES_HEADING = re.compile(r"(?im)^[ \t]{0,3}#{1,6}[ \t]*references[ \t]*$")
_CITATION = re.compile(r"\[(\d+)\]")
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")   # keeps numbers (2.5181) and formulas (cspbi3) in one piece
_STOPWORDS = frozenset(
    "about above after also among because been before being between both could does each from have however into "
    "more most other over same should since some such than that their them then there these they this those through "
    "under until upon using very were what when where which while will with within without would text".split()
)


# --------------------------------------------------------------------------------------
# Reading the report
# --------------------------------------------------------------------------------------
def report_body(report: str) -> str:
    """The report without its reference list."""
    report = report or ""
    match = _REFERENCES_HEADING.search(report)
    return (report[:match.start()] if match else report).strip()


def citations_in(text: str) -> List[int]:
    """The citation numbers in a text, in order, one entry per marker."""
    return [int(number) for number in _CITATION.findall(text or "")]


def claim_sentences(body: str) -> List[str]:
    """The substantive sentences of a report body: not headings, notes or very short lines."""
    sentences: List[str] = []
    for line in (body or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("---"):
            continue
        line = re.sub(r"^[*\-+]\s+", "", line)             # a bullet marker
        line = re.sub(r"^\*\*[^*]+\*\*:?\s*", "", line)     # a bold label such as "**Key finding:**"
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z\[$*(])", line):
            part = part.strip()
            if len(part.split()) >= MIN_CLAIM_WORDS and not part.lower().startswith(("note:", "*note")):
                sentences.append(part)
    return sentences


# --------------------------------------------------------------------------------------
# Grounding: do the specific words and numbers of a cited sentence occur in the cited source?
# --------------------------------------------------------------------------------------
def _tokens(text: str) -> set:
    words = _TOKEN.findall((text or "").lower())
    return {w for w in words if w not in _STOPWORDS and (len(w) >= 4 or any(c.isdigit() for c in w))}


def _url_key(url: Any) -> Optional[str]:
    return url.strip().rstrip(".").rstrip("/").lower() if isinstance(url, str) and url.strip() else None


def evidence_by_url(state: Dict[str, Any]) -> Dict[str, str]:
    """All the text the run retrieved, per source address: the passages RAG selected plus the tool results."""
    evidence: Dict[str, str] = {}

    def add(url: Any, text: Any) -> None:
        key = _url_key(url)
        if key:
            evidence[key] = f"{evidence.get(key, '')} {text}"

    for chunk in state.get("context_chunks") or []:
        if isinstance(chunk, dict):
            add(chunk.get("doc_id"), chunk.get("text", ""))
    for entry in state.get("raw_tool_data") or []:
        if not isinstance(entry, dict):
            continue
        meta = entry.get("metadata") or {}
        urls = [meta.get("pdf_url"), meta.get("url"), entry.get("url")]
        if entry.get("tool_id") == "materials_search" and meta.get("material_id"):
            urls.append(MATERIALS_URL.format(meta["material_id"]))
        for url in urls:
            add(url, entry.get("text", ""))
    return evidence


def grounding_scores(body: str, sources: List[Dict[str, Any]], evidence: Dict[str, str]) -> List[float]:
    """For every cited claim sentence: the share of its content words and numbers found in the cited sources' text.

    1.0 means every specific term of the sentence occurs in what was retrieved; 0.0 means none does. A paraphrase
    scores below 1.0 although it can be perfectly faithful, so only low scores are informative.
    """
    address = {source.get("number"): _url_key(source.get("url")) for source in sources}
    scores: List[float] = []
    for sentence in claim_sentences(body):
        cited = [n for n in citations_in(sentence) if address.get(n)]
        if not cited:
            continue
        source_text = " ".join(evidence.get(address[n], "") for n in cited)
        words = _tokens(_CITATION.sub(" ", sentence))
        if not source_text.strip() or not words:
            continue
        scores.append(len(words & _tokens(source_text)) / len(words))
    return scores


# --------------------------------------------------------------------------------------
# One run, and a set of runs
# --------------------------------------------------------------------------------------
def _passed(question: Dict[str, Any], record: Dict[str, Any], report: str, visited: List[str]) -> bool:
    if record["status"] == "failed":
        return False
    if question.get("expect") == "reject":
        # An out-of-scope question must be refused without any search.
        return report.strip() == REJECTION_TEXT and not any(name.endswith("_search") for name in visited)
    return record["sources_cited"] > 0


def measure_run(question: Dict[str, Any], state: Dict[str, Any], seconds: float, usage: Dict[str, int]) -> Dict[str, Any]:
    """The record of one question: cost, speed and the quality indicators of its report."""
    failure = state.get("failure") or None
    report = str(state.get("final_report") or "")
    visited = list(state.get("visited_nodes") or [])
    sources = [s for s in (state.get("sources") or []) if isinstance(s, dict)]

    record: Dict[str, Any] = {
        "id": question.get("id", ""),
        "question": question.get("question", ""),
        "expect": question.get("expect", "report"),
        "status": "failed" if failure else "ok",
        "failure_kind": failure.get("kind") if isinstance(failure, dict) else None,
        "seconds": round(seconds, 1),
        "requests": int(usage.get("calls", 0)),
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
        "thinking_tokens": int(usage.get("thinking_tokens", 0)),
        "refinement_cycles": int(state.get("refinement_retries") or 0),
        "evaluation_unavailable": bool(state.get("evaluation_unavailable")),
        "documents_retrieved": len([e for e in (state.get("raw_tool_data") or []) if isinstance(e, dict)]),
        "sources_cited": len(sources),
        "citation_markers": None,
        "invalid_citations_removed": None,
        "invalid_citation_rate": None,
        "claim_sentences": None,
        "citation_coverage": None,
        "grounding_score": None,
    }

    if not failure:
        body = report_body(report)
        markers = citations_in(body)
        removed = int(state.get("invalid_citations_removed") or 0)
        sentences = claim_sentences(body)
        cited_sentences = [s for s in sentences if citations_in(s)]
        scores = grounding_scores(body, sources, evidence_by_url(state))
        record.update({
            "citation_markers": len(markers),
            "invalid_citations_removed": removed,
            # How often the model tried to cite a source it was never shown (code removes those before the report is final)
            "invalid_citation_rate": round(removed / (removed + len(markers)), 3) if removed + len(markers) else None,
            "claim_sentences": len(sentences),
            "citation_coverage": round(len(cited_sentences) / len(sentences), 3) if sentences else None,
            "grounding_score": round(statistics.mean(scores), 3) if scores else None,
        })

    record["passed"] = _passed(question, record, report, visited)
    return record


def _mean(values: List[Optional[float]], digits: int = 3) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), digits) if values else None


def summarise(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Totals and averages. Averages cover completed report questions only, so that a refused out-of-scope question
    (cheap and fast by design) and a run that stopped on an error do not distort them."""
    completed = [r for r in records if r["status"] == "ok"]
    reports = [r for r in completed if r["expect"] == "report"]

    def tokens(r: Dict[str, Any]) -> int:
        return r["prompt_tokens"] + r["output_tokens"] + r["thinking_tokens"]

    return {
        "questions": len(records),
        "completed": len(completed),
        "failed": len(records) - len(completed),
        "passed": sum(1 for r in completed if r["passed"]),
        "total_requests": sum(r["requests"] for r in records),
        "mean_seconds": _mean([r["seconds"] for r in reports], 1),
        "median_seconds": round(statistics.median([r["seconds"] for r in reports]), 1) if reports else None,
        "mean_requests": _mean([r["requests"] for r in reports], 1),
        "mean_tokens": _mean([tokens(r) for r in reports], 0),
        "mean_refinement_cycles": _mean([r["refinement_cycles"] for r in reports], 2),
        "mean_sources_cited": _mean([r["sources_cited"] for r in reports], 1),
        "mean_citation_coverage": _mean([r["citation_coverage"] for r in reports]),
        "mean_grounding_score": _mean([r["grounding_score"] for r in reports]),
        "mean_invalid_citation_rate": _mean([r["invalid_citation_rate"] for r in reports]),
    }


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------
def _cell(value: Any, digits: Optional[int] = None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and digits is not None:
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(records: List[Dict[str, Any]], summary: Dict[str, Any], meta: Dict[str, Any]) -> str:
    """A results page: the run's settings, one row per question, and the averages."""
    lines = [
        "# Benchmark results",
        "",
        f"Date: {meta.get('date', '?')} | Model: `{meta.get('model', '?')}` | "
        f"Requests per minute limit: {meta.get('requests_per_minute') or 'none'} | "
        f"Questions run: {summary['questions']}",
        "",
        "| Question | Result | Time (s) | Requests | Tokens | Cycles | Sources cited / retrieved | "
        "Cited claim sentences | Grounding | Invalid citations removed |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        if r["status"] == "failed":
            result = f"stopped: {r['failure_kind']}"
        elif r["passed"]:
            result = "ok" + (" (unreviewed)" if r["evaluation_unavailable"] else "")
        else:
            result = "FAILED CHECK"
        tokens = r["prompt_tokens"] + r["output_tokens"] + r["thinking_tokens"]
        coverage = _cell(None if r["citation_coverage"] is None else round(r["citation_coverage"] * 100), 0)
        lines.append(
            f"| {r['id']} | {result} | {_cell(r['seconds'], 1)} | {r['requests']} | {tokens} | {r['refinement_cycles']} | "
            f"{r['sources_cited']} / {r['documents_retrieved']} | {coverage + '%' if coverage != '-' else '-'} | "
            f"{_cell(r['grounding_score'], 2)} | {_cell(r['invalid_citations_removed'])} |"
        )
    lines += [
        "",
        f"Completed {summary['completed']} of {summary['questions']}; passed their check: {summary['passed']}. "
        f"Total Gemini requests: {summary['total_requests']}.",
        "",
        "Averages over the completed report questions:",
        "",
        f"- time: {_cell(summary['mean_seconds'])} s (median {_cell(summary['median_seconds'])} s)",
        f"- Gemini requests: {_cell(summary['mean_requests'])}, tokens: {_cell(summary['mean_tokens'])}",
        f"- refinement cycles: {_cell(summary['mean_refinement_cycles'])}",
        f"- sources cited: {_cell(summary['mean_sources_cited'])}",
        f"- cited claim sentences (heuristic): {_cell(summary['mean_citation_coverage'])}",
        f"- grounding score (heuristic): {_cell(summary['mean_grounding_score'])}",
        f"- invalid-citation rate before cleanup: {_cell(summary['mean_invalid_citation_rate'])}",
        "",
        "How to read these numbers, and their limits: see `benchmarks/README.md`.",
        "",
    ]
    return "\n".join(lines)
