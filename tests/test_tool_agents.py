"""Tool agents (level 2): guardrails, result standardisation and state updates.

Each agent's network call is replaced by a canned response; everything else (the guardrail, the
tiered-query lookup, standardisation, references, hand-back to the supervisor) runs for real.
These replace the live-API harness that used to live in tool_agents.py.
"""
import datetime
from types import SimpleNamespace

import pytest


@pytest.fixture
def planned(sample_state):
    """The level-1 planning output: active tools and tiered queries for arxiv, openalex, web, materials, S2."""
    return sample_state("level_1_planning_output")


def fail_if_called(*args, **kwargs):
    raise AssertionError("the network call must not run")


# --------------------------------------------------------------------------------------
# Guardrails shared by every tool (BaseToolAgent)
# --------------------------------------------------------------------------------------
def test_tool_that_is_not_active_is_skipped_but_still_visited(tool_agents_module, planned, monkeypatch):
    agent = tool_agents_module.ArxivAgent()
    monkeypatch.setattr(agent, "_call_arxiv_search", fail_if_called)
    planned["active_tools"] = ["web"]
    before = len(planned["raw_tool_data"])

    state = agent.execute(planned)

    assert "arxiv_search" in state["visited_nodes"]
    assert len(state["raw_tool_data"]) == before


def test_tool_without_queries_is_skipped(tool_agents_module, planned, monkeypatch):
    agent = tool_agents_module.OpenAlexAgent()
    monkeypatch.setattr(agent, "_call_openalex_api", fail_if_called)
    planned["tiered_queries"].pop("openalex")

    state = agent.execute(planned)

    assert len(state["raw_tool_data"]) == 0


# --------------------------------------------------------------------------------------
# ArXiv
# --------------------------------------------------------------------------------------
def fake_arxiv_paper():
    return SimpleNamespace(
        title="Lead Free Perovskites",
        summary="A review of lead-free perovskite solar cells.\nSecond line.",
        entry_id="http://arxiv.org/abs/2407.17520v2",
        published=datetime.datetime(2024, 7, 1),
        pdf_url="https://arxiv.org/pdf/2407.17520v2",
    )


def test_arxiv_results_are_standardised_and_handed_back(tool_agents_module, planned, monkeypatch):
    agent = tool_agents_module.ArxivAgent()
    monkeypatch.setattr(agent, "_call_arxiv_search", lambda term, date_filter="": [fake_arxiv_paper()])

    state = agent.execute(planned)

    entry = state["raw_tool_data"][-1]
    assert entry["source_type"] == "arxiv"
    assert entry["tool_id"] == "arxiv_search"
    assert entry["metadata"]["published_year"] == 2024
    assert entry["metadata"]["pdf_url"] == "https://arxiv.org/pdf/2407.17520v2"
    assert "\n" not in entry["text"]
    assert state["references"][-1].endswith("Arxiv: Lead Free Perovskites")
    assert state["next"] == "supervisor_agent"


def test_arxiv_tries_the_next_tier_when_the_first_is_empty(tool_agents_module, planned, monkeypatch):
    agent = tool_agents_module.ArxivAgent()
    queries_tried = []

    def search(term, date_filter=""):
        queries_tried.append(term)
        return [] if len(queries_tried) == 1 else [fake_arxiv_paper()]

    monkeypatch.setattr(agent, "_call_arxiv_search", search)
    state = agent.execute(planned)

    assert queries_tried == [planned["tiered_queries"]["arxiv"]["strict"], planned["tiered_queries"]["arxiv"]["moderate"]]
    assert len(state["raw_tool_data"]) == 1


@pytest.mark.parametrize("constraint, expects_filter", [("TIME_PERIOD: last_decade", True), ("TOPIC: perovskites", False)])
def test_arxiv_date_filter_only_applies_to_a_time_period(tool_agents_module, planned, monkeypatch, constraint, expects_filter):
    agent = tool_agents_module.ArxivAgent()
    filters = []

    def search(term, date_filter=""):
        filters.append(date_filter)
        return [fake_arxiv_paper()]

    monkeypatch.setattr(agent, "_call_arxiv_search", search)
    planned["system_constraints"] = [constraint]
    agent.execute(planned)

    assert filters[0].startswith(" AND submitted:[") is expects_filter


# --------------------------------------------------------------------------------------
# OpenAlex
# --------------------------------------------------------------------------------------
def test_openalex_results_get_a_readable_abstract(tool_agents_module, planned, monkeypatch):
    agent = tool_agents_module.OpenAlexAgent()
    work = {
        "id": "https://openalex.org/W1",
        "title": "A paper",
        "abstract_inverted_index": {"world": [1], "Hello": [0]},
        "primary_location": {"landing_page_url": "https://example.org/paper"},
    }
    monkeypatch.setattr(agent, "_call_openalex_api", lambda query: [work])

    state = agent.execute(planned)

    entry = state["raw_tool_data"][-1]
    assert entry["text"] == "Title: A paper. Abstract: Hello world"
    assert entry["metadata"]["openalex_id"] == "https://openalex.org/W1"
    assert state["references"][-1].endswith("OpenAlex: A paper")


# --------------------------------------------------------------------------------------
# Web search
# --------------------------------------------------------------------------------------
def test_web_results_are_filtered_for_noise(tool_agents_module, planned, monkeypatch):
    agent = tool_agents_module.WebAgent()
    results = [
        {"href": "https://example.org/paper", "title": "Lead-free perovskites review", "body": "snippet"},
        {"href": "https://www.facebook.com/post", "title": "A post", "body": "noise domain"},
        {"href": "https://example.org/help", "title": "How to reset your password", "body": "noise title"},
    ]
    monkeypatch.setattr(agent, "_call_ddg_search", lambda query, limit=10: results)

    state = agent.execute(planned)

    assert [d["metadata"]["url"] for d in state["raw_tool_data"]] == ["https://example.org/paper"]
    assert state["raw_tool_data"][0]["source_type"] == "web_search"


# --------------------------------------------------------------------------------------
# Materials Project
# --------------------------------------------------------------------------------------
def test_materials_results_are_formatted(tool_agents_module, planned, monkeypatch):
    agent = tool_agents_module.MaterialsAgent()
    material = {"material_id": "mp-1", "formula": "ZnO", "is_stable": True, "band_gap": 3.4, "energy_above_hull": 0.0}
    monkeypatch.setattr(agent, "_call_materials_project_api", lambda formula, max_results: [material])

    state = agent.execute(planned)

    entry = state["raw_tool_data"][-1]
    assert entry["source_type"] == "materials_project"
    assert "Material: ZnO (mp-1). Stability: Stable." in entry["text"]
    assert "mp-1" in state["references"][-1]


@pytest.mark.parametrize("bad_term", ["", "x", "TOPIC: graphene"])
def test_materials_lookup_needs_a_plausible_formula(tool_agents_module, planned, monkeypatch, bad_term):
    agent = tool_agents_module.MaterialsAgent()
    monkeypatch.setattr(agent, "_call_materials_project_api", fail_if_called)
    planned["api_search_term"] = bad_term

    state = agent.execute(planned)

    assert len(state["raw_tool_data"]) == 0


# --------------------------------------------------------------------------------------
# PubMed
# --------------------------------------------------------------------------------------
def test_pubmed_results_reach_the_state(tool_agents_module, planned, monkeypatch):
    agent = tool_agents_module.PubMedAgent()
    record = {
        "text": "Title: T. Abstract: A",
        "source_type": "pubmed",
        "tool_id": "pubmed_search",
        "metadata": {"pmid": "123", "title": "T", "pdf_url": "https://pubmed.ncbi.nlm.nih.gov/123/"},
    }
    monkeypatch.setattr(agent, "_execute_tiered_search", lambda queries: ["123"])
    monkeypatch.setattr(agent, "_fetch_metadata_for_pmids", lambda pmids: [record])
    planned["active_tools"] = ["pubmed"]
    planned["tiered_queries"] = {"pubmed": {"strict": "perovskite"}}

    state = agent.execute(planned)

    assert state["raw_tool_data"][-1] == record
    assert state["references"][-1].endswith("Journal Article: T")


# --------------------------------------------------------------------------------------
# Semantic Scholar
# --------------------------------------------------------------------------------------
S2_PAPERS = [
    {"paperId": "p1", "title": "No abstract, closed access", "abstract": None, "openAccessPdf": None,
     "url": "https://www.semanticscholar.org/paper/p1", "venue": "Journal", "year": 2020, "citationCount": 3},
    {"paperId": "p2", "title": "Open access paper", "abstract": "Real abstract.",
     "openAccessPdf": {"url": "https://example.org/p2.pdf"}, "url": "https://www.semanticscholar.org/paper/p2"},
    {"paperId": "p3", "title": None, "abstract": "A paper without a title is dropped."},
]


def test_semantic_scholar_results_are_standardised(tool_agents_module, planned, monkeypatch):
    agent = tool_agents_module.SemanticScholarAgent()
    monkeypatch.setattr(agent, "_call_s2_api", lambda query: S2_PAPERS)

    state = agent.execute(planned)

    first, second = state["raw_tool_data"][-2:]
    assert first["metadata"]["pdf_url"] == "https://www.semanticscholar.org/paper/p1"  # no open-access PDF: landing page
    assert "Abstract unavailable." in first["text"]
    assert second["metadata"]["pdf_url"] == "https://example.org/p2.pdf"
    assert second["metadata"]["s2_id"] == "p2"
    assert len(state["raw_tool_data"]) == 2  # the untitled paper is dropped
    assert state["references"][-1].endswith("Semantic Scholar: Open access paper")
    assert state["next"] == "supervisor_agent"


def test_semantic_scholar_falls_back_to_the_semantic_query(tool_agents_module, planned, monkeypatch):
    agent = tool_agents_module.SemanticScholarAgent()
    seen = []
    monkeypatch.setattr(agent, "_call_s2_api", lambda query: seen.append(query) or [])
    planned["tiered_queries"]["semanticscholar"] = {"other_tier": "ignored"}
    planned["semantic_query"] = "bandgap of Ni-doped ZnO"

    agent.execute(planned)

    assert seen == ["bandgap of Ni-doped ZnO"]


def test_semantic_scholar_without_any_query_makes_no_request(tool_agents_module, planned, monkeypatch):
    agent = tool_agents_module.SemanticScholarAgent()
    monkeypatch.setattr(agent, "_call_s2_api", fail_if_called)
    planned["tiered_queries"]["semanticscholar"] = {"other_tier": "ignored"}
    planned["semantic_query"] = ""

    state = agent.execute(planned)

    assert len(state["raw_tool_data"]) == 0


# --------------------------------------------------------------------------------------
# ChemRxiv
# --------------------------------------------------------------------------------------
def test_chemrxiv_results_are_cleaned_of_html(tool_agents_module, planned, monkeypatch):
    agent = tool_agents_module.ChemRxivAgent()
    preprint = {
        "title": "A preprint", "description": "<p>Tin <b>perovskite</b> stability.</p>",
        "url_public_html": "https://chemrxiv.org/x", "doi": "10.26434/x", "published_date": "2025-01-01",
    }
    monkeypatch.setattr(agent, "_call_chemrxiv_api", lambda query: [preprint])
    planned["active_tools"] = ["chemrxiv"]
    planned["tiered_queries"] = {"chemrxiv": {"simple": "tin perovskite"}}

    state = agent.execute(planned)

    entry = state["raw_tool_data"][-1]
    assert entry["text"] == "Title: A preprint. Abstract: Tin perovskite stability."
    assert entry["url"] == "https://chemrxiv.org/x"
