"""Stage-by-stage tests for the LLM-backed agents, with Gemini replaced by canned responses.

These replace the `if __name__ == "__main__"` harnesses that used to live at the bottom of each
agent module. They use the JSON files in examples/sample_outputs as realistic input states.
"""
import json

from backend.agents import evaluation_agent as evaluation_module
from backend.agents import planning_agents as planning_module
from backend.agents import procedural_agents as procedural_module
from backend.agents import supervisor_agent as supervisor_module
from backend.agents import synthesis_agent as synthesis_module
from backend.agents.evaluation_agent import EvaluationAgent
from backend.agents.planning_agents import IntentAgent, PlanningAgent, QueryGenerationAgent
from backend.agents.procedural_agents import CleanQueryAgent
from backend.agents.supervisor_agent import SupervisorAgent
from backend.agents.synthesis_agent import SynthesisAgent
from backend.core.research_state import ResearchState, new_research_state


# --------------------------------------------------------------------------------------
# The shared starting state
# --------------------------------------------------------------------------------------
def test_new_research_state_covers_every_state_field():
    assert set(new_research_state("q")) == set(ResearchState.__annotations__)


def test_new_research_state_returns_independent_objects():
    first, second = new_research_state("a"), new_research_state("b")
    first["references"].append("x")
    first["visited_nodes"].append("clean_query_agent")
    assert second["references"] == []
    assert second["visited_nodes"] == []
    assert second["user_query"] == "b"


# --------------------------------------------------------------------------------------
# CleanQueryAgent (level 0)
# --------------------------------------------------------------------------------------
RAW_QUERY = "  Can we use g-raphene?! 10.00% doping of Ni in ZnO--what's the bandgap?  "


def test_clean_query_removes_noise_but_keeps_chemistry(monkeypatch):
    monkeypatch.setattr(procedural_module, "client", None)  # no LLM: the semantic query is the cleaned query
    state = CleanQueryAgent().execute(new_research_state(RAW_QUERY))
    query = state["semantic_query"]
    assert query == query.strip()
    assert "?" not in query and "!" not in query
    assert "g-raphene" in query and "10.00%" in query
    assert state["next"] == "supervisor_agent"
    assert state["visited_nodes"] == ["clean_query_agent"]


def test_clean_query_uses_the_llm_rewrite(monkeypatch, fake_response):
    monkeypatch.setattr(procedural_module, "client", object())
    monkeypatch.setattr(
        procedural_module, "generate_content_with_retry", lambda **kwargs: fake_response("  optimised query \n")
    )
    state = CleanQueryAgent().execute(new_research_state(RAW_QUERY))
    assert state["semantic_query"] == "optimised query"


def test_clean_query_falls_back_to_the_cleaned_text_when_the_llm_fails(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("quota")

    monkeypatch.setattr(procedural_module, "client", object())
    monkeypatch.setattr(procedural_module, "generate_content_with_retry", boom)
    state = CleanQueryAgent().execute(new_research_state("What is  CsPbI3?"))
    assert state["semantic_query"] == "What is CsPbI3"


def test_empty_query_gets_a_placeholder(monkeypatch):
    monkeypatch.setattr(procedural_module, "client", None)
    assert CleanQueryAgent().execute(new_research_state("   "))["semantic_query"] == "empty_query"


# --------------------------------------------------------------------------------------
# Intent -> Planning -> Query generation (level 1)
# --------------------------------------------------------------------------------------
def test_planning_chain_fills_the_state(monkeypatch, sample_state):
    state = sample_state("level_0_procedural_output")
    intent, planning, query_gen = IntentAgent(), PlanningAgent(), QueryGenerationAgent()

    monkeypatch.setattr(intent, "_call_llm_and_parse", lambda prompt: {
        "primary_intent": "materials_research",
        "reasoning": "needs data",
        "extracted_constraints": ["TIME_PERIOD: last_decade"],
    })
    monkeypatch.setattr(planning, "_call_llm_and_parse", lambda prompt: {
        "execution_plan": ["search", "summarise"],
        "active_tools": ["arxiv", "openalex", "not_a_tool"],
    })
    monkeypatch.setattr(query_gen, "_call_llm_and_parse", lambda prompt, tools: {
        "tiered_queries": {tool: {"simple": f"{tool} query"} for tool in tools},
        "material_elements": ["ZnO", "Ni"],
    })

    for agent in (intent, planning, query_gen):
        state = agent.execute(state)

    assert state["primary_intent"] == "materials_research"
    assert set(state["active_tools"]) == {"arxiv", "openalex"}  # unknown tool dropped
    assert set(state["tiered_queries"]) == {"arxiv", "openalex"}
    assert state["api_search_term"] == "ZnO"
    assert "TIME_PERIOD: last_decade" in state["material_elements"] and "Ni" in state["material_elements"]
    assert state["next"] == "supervisor_agent"
    assert state["visited_nodes"][-3:] == ["intent_agent", "planning_agent", "query_gen_agent"]


def test_unknown_intent_falls_back_to_general_research(monkeypatch):
    agent = IntentAgent()
    monkeypatch.setattr(agent, "_call_llm_and_parse", lambda prompt: {"primary_intent": "astrology"})
    assert agent.execute(new_research_state("q"))["primary_intent"] == "general_research"


def test_intent_agent_defaults_to_research_when_the_llm_fails(monkeypatch):
    agent = IntentAgent()
    monkeypatch.setattr(agent, "_call_llm_and_parse", lambda prompt: None)
    state = agent.execute(new_research_state("q"))
    assert state["primary_intent"] == "general_research"
    assert state["system_constraints"] == []


def test_literature_reviews_always_get_pubmed_and_arxiv(monkeypatch):
    agent = PlanningAgent()
    monkeypatch.setattr(agent, "_call_llm_and_parse", lambda prompt: {"execution_plan": ["p"], "active_tools": ["web"]})
    state = new_research_state("q")
    state["primary_intent"] = "literature_review"
    assert set(agent.execute(state)["active_tools"]) == {"web", "pubmed", "arxiv"}


def test_planning_falls_back_to_default_tools_when_the_llm_fails(monkeypatch):
    agent = PlanningAgent()
    monkeypatch.setattr(agent, "_call_llm_and_parse", lambda prompt: None)
    state = agent.execute(new_research_state("q"))
    assert set(state["active_tools"]) == {"arxiv", "openalex", "web"}
    assert state["execution_plan"]


def test_refinement_planning_keeps_the_previous_tools(monkeypatch):
    agent = PlanningAgent()
    monkeypatch.setattr(agent, "_call_llm_and_parse", lambda prompt: {"execution_plan": ["p"], "active_tools": ["openalex"]})
    state = new_research_state("q")
    state.update(is_refining=True, refinement_reason="missing papers", active_tools=["arxiv"])
    assert set(agent.execute(state)["active_tools"]) == {"arxiv", "openalex"}


def test_query_generation_keeps_only_queries_for_active_tools(monkeypatch, fake_response):
    payload = {
        "tiered_queries": {"arxiv": {"strict": "a"}, "ghost_tool": {"strict": "g"}},
        "material_elements": ["ZnO", "", "Ni"],
    }
    monkeypatch.setattr(planning_module, "client", object())
    monkeypatch.setattr(
        planning_module, "generate_content_with_retry", lambda **kwargs: fake_response(json.dumps(payload))
    )
    result = QueryGenerationAgent()._call_llm_and_parse("prompt", ["arxiv"])
    assert result == {"tiered_queries": {"arxiv": {"strict": "a"}}, "material_elements": ["ZnO", "Ni"]}


def test_query_generation_without_active_tools_returns_to_the_hub():
    state = QueryGenerationAgent().execute(new_research_state("q"))
    assert state["next"] == "supervisor_agent"
    assert state["tiered_queries"] == {}


# --------------------------------------------------------------------------------------
# SynthesisAgent (level 5)
# --------------------------------------------------------------------------------------
LLM_REPORT = (
    "## Introduction\nClaim B [2]. Claim A [1].\n\n"
    "## References\n\n[1] Paper A\n\n[2] Paper B"
)


def test_synthesis_writes_the_report_and_renumbers_citations(monkeypatch, sample_state, fake_gemini):
    client = fake_gemini(text=LLM_REPORT)
    monkeypatch.setattr(synthesis_module, "client", client)

    state = SynthesisAgent().execute(sample_state("level_4_rag_output"))

    assert state["report_generated"] is True
    assert state["next"] == "supervisor_agent"
    body, references = state["final_report"].split("## References")
    assert "Claim B [1]. Claim A [2]." in body  # numbered in order of first appearance
    assert "[1] Paper B" in references and "[2] Paper A" in references
    assert "SOURCE C" in client.calls[0]["contents"]  # the prompt carried the verified-links block


def test_synthesis_rejects_irrelevant_queries_without_calling_the_llm(monkeypatch, sample_state, fake_gemini):
    client = fake_gemini(text=LLM_REPORT)
    monkeypatch.setattr(synthesis_module, "client", client)
    state = sample_state("level_4_rag_output")
    state["primary_intent"] = "irrelevant"

    state = SynthesisAgent().execute(state)

    assert state["final_report"] == "Query rejected based on scope."
    assert state["report_generated"] is True
    assert client.calls == []


def test_synthesis_reports_an_error_when_the_llm_fails(monkeypatch, sample_state, fake_gemini):
    monkeypatch.setattr(synthesis_module, "client", fake_gemini(error=RuntimeError("quota")))
    state = SynthesisAgent().execute(sample_state("level_4_rag_output"))
    assert "Error" in state["final_report"]
    assert state["next"] == "supervisor_agent"


# --------------------------------------------------------------------------------------
# EvaluationAgent (level 6)
# --------------------------------------------------------------------------------------
LONG_REPORT = "A sufficiently long report sentence. " * 20


def evaluation_state(sample_state):
    state = sample_state("level_5_synthesis_output")
    state["final_report"] = LONG_REPORT
    return state


def test_evaluation_accepts_a_good_report(monkeypatch, sample_state, fake_gemini):
    verdict = {"needs_refinement": False, "refinement_reason": "Report is satisfactory"}
    monkeypatch.setattr(evaluation_module, "client", fake_gemini(text=json.dumps(verdict)))

    state = EvaluationAgent().execute(evaluation_state(sample_state))

    assert state["needs_refinement"] is False
    assert state["refinement_reason"] == "Report is satisfactory"
    assert state["next"] == "supervisor_agent"


def test_evaluation_can_request_refinement(monkeypatch, sample_state, fake_gemini):
    verdict = {"needs_refinement": True, "refinement_reason": "Missing papers on stability"}
    monkeypatch.setattr(evaluation_module, "client", fake_gemini(text=json.dumps(verdict)))

    state = EvaluationAgent().execute(evaluation_state(sample_state))

    assert state["needs_refinement"] is True
    assert state["refinement_reason"] == "Missing papers on stability"


def test_evaluation_sends_a_too_short_report_back_without_calling_the_llm(monkeypatch, sample_state, fake_gemini):
    client = fake_gemini(text="{}")
    monkeypatch.setattr(evaluation_module, "client", client)
    state = evaluation_state(sample_state)
    state["final_report"] = "Too short."

    state = EvaluationAgent().execute(state)

    assert state["needs_refinement"] is True
    assert "insufficient" in state["refinement_reason"].lower()
    assert client.calls == []


def test_evaluation_is_skipped_without_a_client(monkeypatch, sample_state):
    monkeypatch.setattr(evaluation_module, "client", None)
    state = EvaluationAgent().execute(evaluation_state(sample_state))
    assert state["needs_refinement"] is False
    assert state["next"] == "supervisor_agent"


# --------------------------------------------------------------------------------------
# SupervisorAgent on a recorded state (level 6 -> 7)
# --------------------------------------------------------------------------------------
GRAPH_NODES = {
    "clean_query_agent", "intent_agent", "planning_agent", "query_gen_agent",
    "semanticscholar_search", "chemrxiv_search", "pubmed_search", "arxiv_search",
    "openalex_search", "materials_search", "web_search", "retrieval_agent",
    "rag_agent", "synthesis_agent", "evaluation_agent", "END",
}


def test_supervisor_routes_a_recorded_state_to_a_node_the_graph_knows(monkeypatch, sample_state):
    monkeypatch.setattr(supervisor_module.time, "sleep", lambda seconds: None)
    state = SupervisorAgent().execute(sample_state("level_6_evaluation_output"))
    assert state["next"] in GRAPH_NODES
    assert state["visited_nodes"][-1] == "supervisor_agent"
