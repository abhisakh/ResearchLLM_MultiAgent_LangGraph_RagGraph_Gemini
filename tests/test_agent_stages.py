"""Stage-by-stage tests for the LLM-backed agents, with Gemini replaced by scripted answers.

These replace the `if __name__ == "__main__"` harnesses that used to live at the bottom of each
agent module, and use the JSON files in examples/sample_outputs as realistic input states.

Error handling (typed errors, step 2): when an LLM call fails, the agent degrades to its own
fallback AND records the failure in state["errors"], so the supervisor can tell a silent
fallback from a run that can no longer succeed.
"""
import json
from types import SimpleNamespace

import pytest

from backend.agents import evaluation_agent as evaluation_module
from backend.agents import planning_agents as planning_module
from backend.agents import procedural_agents as procedural_module
from backend.agents import synthesis_agent as synthesis_module
from backend.agents.evaluation_agent import EvaluationAgent
from backend.agents.planning_agents import IntentAgent, PlanningAgent, QueryGenerationAgent
from backend.agents.procedural_agents import CleanQueryAgent
from backend.agents.supervisor_agent import SupervisorAgent
from backend.agents.synthesis_agent import SynthesisAgent
from backend.core.errors import ErrorKind, LLMCallError, user_message
from backend.core.research_state import ResearchState, new_research_state


def use_llm(monkeypatch, module, llm):
    """Make `module` call `llm` instead of Gemini; returns llm so the test can inspect its calls."""
    monkeypatch.setattr(module, "generate_content_with_retry", llm)
    return llm


def error_kinds(state):
    return [e["kind"] for e in state["errors"]]


# --------------------------------------------------------------------------------------
# The shared starting state
# --------------------------------------------------------------------------------------
def test_new_research_state_covers_every_state_field():
    assert set(new_research_state("q")) == set(ResearchState.__annotations__)


def test_new_research_state_returns_independent_objects():
    first, second = new_research_state("a"), new_research_state("b")
    first["references"].append("x")
    first["visited_nodes"].append("clean_query_agent")
    first["errors"].append({"agent": "x", "kind": "auth", "message": "m"})
    assert second["references"] == []
    assert second["visited_nodes"] == []
    assert second["errors"] == []
    assert second["user_query"] == "b"


# --------------------------------------------------------------------------------------
# CleanQueryAgent (level 0)
# --------------------------------------------------------------------------------------
RAW_QUERY = "  Can we use g-raphene?! 10.00% doping of Ni in ZnO--what's the bandgap?  "


def test_clean_query_sends_the_llm_a_cleaned_query_and_uses_its_answer(monkeypatch, fake_llm):
    llm = use_llm(monkeypatch, procedural_module, fake_llm("  Bandgap of Ni-doped ZnO \n"))

    state = CleanQueryAgent().execute(new_research_state(RAW_QUERY))

    raw_input_line = llm.calls[0]["contents"].split("RAW INPUT:")[1].splitlines()[0].strip()
    assert "?" not in raw_input_line and "!" not in raw_input_line
    assert "g-raphene" in raw_input_line and "10.00%" in raw_input_line  # chemistry survives the cleaning
    assert state["semantic_query"] == "Bandgap of Ni-doped ZnO"
    assert state["errors"] == []
    assert state["next"] == "supervisor_agent"
    assert state["visited_nodes"] == ["clean_query_agent"]


@pytest.mark.parametrize("kind", [ErrorKind.AUTH, ErrorKind.QUOTA_EXHAUSTED, ErrorKind.TRANSIENT])
def test_clean_query_keeps_the_cleaned_query_and_records_the_error(monkeypatch, fake_llm, kind):
    use_llm(monkeypatch, procedural_module, fake_llm(LLMCallError(kind, "failed")))

    state = CleanQueryAgent().execute(new_research_state("What is  CsPbI3?"))

    assert state["semantic_query"] == "What is CsPbI3"
    assert error_kinds(state) == [kind.value]
    assert state["errors"][0]["agent"] == "clean_query_agent"


def test_clean_query_asks_again_once_when_the_answer_is_empty(monkeypatch, fake_llm):
    llm = use_llm(monkeypatch, procedural_module, fake_llm("", "A good query"))
    state = CleanQueryAgent().execute(new_research_state("q"))
    assert len(llm.calls) == 2
    assert state["semantic_query"] == "A good query"
    assert state["errors"] == []


def test_clean_query_gives_up_after_one_more_unusable_answer(monkeypatch, fake_llm):
    llm = use_llm(monkeypatch, procedural_module, fake_llm(""))
    state = CleanQueryAgent().execute(new_research_state("q"))
    assert len(llm.calls) == 2
    assert state["semantic_query"] == "q"
    assert error_kinds(state) == ["bad_output"]


def test_a_bug_is_recorded_as_unknown_not_hidden(monkeypatch, fake_llm):
    use_llm(monkeypatch, procedural_module, fake_llm(KeyError("oops")))
    state = CleanQueryAgent().execute(new_research_state("q"))
    assert error_kinds(state) == ["unknown"]


def test_empty_query_gets_a_placeholder_without_calling_the_llm(monkeypatch, fake_llm):
    llm = use_llm(monkeypatch, procedural_module, fake_llm("unused"))
    state = CleanQueryAgent().execute(new_research_state("   "))
    assert state["semantic_query"] == "empty_query"
    assert llm.calls == []


# --------------------------------------------------------------------------------------
# IntentAgent
# --------------------------------------------------------------------------------------
INTENT_ANSWER = {
    "primary_intent": "materials_research",
    "reasoning": "needs data",
    "extracted_constraints": ["TIME_PERIOD: last_decade"],
}


def test_intent_agent_reads_the_llm_answer(monkeypatch, fake_llm):
    use_llm(monkeypatch, planning_module, fake_llm(json.dumps(INTENT_ANSWER)))
    state = IntentAgent().execute(new_research_state("q"))
    assert state["primary_intent"] == "materials_research"
    assert state["system_constraints"] == ["TIME_PERIOD: last_decade"]
    assert state["errors"] == []


def test_intent_agent_asks_again_after_invalid_json(monkeypatch, fake_llm):
    llm = use_llm(monkeypatch, planning_module, fake_llm("{not json", json.dumps(INTENT_ANSWER)))
    state = IntentAgent().execute(new_research_state("q"))
    assert len(llm.calls) == 2
    assert state["primary_intent"] == "materials_research"
    assert state["errors"] == []


def test_intent_agent_degrades_to_general_research_and_records_the_error(monkeypatch, fake_llm):
    use_llm(monkeypatch, planning_module, fake_llm(LLMCallError(ErrorKind.QUOTA_EXHAUSTED, "daily limit")))
    state = IntentAgent().execute(new_research_state("q"))
    assert state["primary_intent"] == "general_research"
    assert state["system_constraints"] == []
    assert error_kinds(state) == ["quota_exhausted"]


def test_unknown_intent_falls_back_to_general_research(monkeypatch, fake_llm):
    use_llm(monkeypatch, planning_module, fake_llm(json.dumps({"primary_intent": "astrology"})))
    assert IntentAgent().execute(new_research_state("q"))["primary_intent"] == "general_research"


# --------------------------------------------------------------------------------------
# Intent -> Planning -> Query generation (level 1)
# --------------------------------------------------------------------------------------
def test_planning_chain_fills_the_state(monkeypatch, sample_state):
    state = sample_state("level_0_procedural_output")
    intent, planning, query_gen = IntentAgent(), PlanningAgent(), QueryGenerationAgent()

    monkeypatch.setattr(intent, "_call_llm_and_parse", lambda prompt: INTENT_ANSWER)
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


# --------------------------------------------------------------------------------------
# PlanningAgent
# --------------------------------------------------------------------------------------
def test_literature_reviews_always_get_pubmed_and_arxiv(monkeypatch, fake_llm):
    answer = json.dumps({"execution_plan": ["p"], "active_tools": ["web"]})
    use_llm(monkeypatch, planning_module, fake_llm(answer))
    state = new_research_state("q")
    state["primary_intent"] = "literature_review"
    assert set(PlanningAgent().execute(state)["active_tools"]) == {"web", "pubmed", "arxiv"}


@pytest.mark.parametrize("kind", [ErrorKind.RATE_LIMITED, ErrorKind.TRANSIENT, ErrorKind.AUTH])
def test_planning_falls_back_to_default_tools_and_records_the_error(monkeypatch, fake_llm, kind):
    use_llm(monkeypatch, planning_module, fake_llm(LLMCallError(kind, "failed")))
    state = PlanningAgent().execute(new_research_state("q"))
    assert set(state["active_tools"]) == {"arxiv", "openalex", "web"}
    assert state["execution_plan"]
    assert error_kinds(state) == [kind.value]


def test_planning_never_ends_with_an_empty_tool_set(monkeypatch, fake_llm):
    answer = json.dumps({"execution_plan": ["p"], "active_tools": ["not_a_tool", "also_not"]})
    use_llm(monkeypatch, planning_module, fake_llm(answer))
    state = PlanningAgent().execute(new_research_state("q"))
    assert state["active_tools"] == PlanningAgent.DEFAULT_TOOLS


def test_planning_keeps_the_llm_tool_order_and_drops_duplicates(monkeypatch, fake_llm):
    answer = json.dumps({"execution_plan": ["p"], "active_tools": ["web", "arxiv", "web", "openalex"]})
    use_llm(monkeypatch, planning_module, fake_llm(answer))
    state = PlanningAgent().execute(new_research_state("q"))
    assert state["active_tools"] == ["web", "arxiv", "openalex"]


def test_refinement_planning_keeps_the_previous_tools(monkeypatch, fake_llm):
    answer = json.dumps({"execution_plan": ["p"], "active_tools": ["openalex"]})
    use_llm(monkeypatch, planning_module, fake_llm(answer))
    state = new_research_state("q")
    state.update(is_refining=True, refinement_reason="missing papers", active_tools=["arxiv"])
    assert set(PlanningAgent().execute(state)["active_tools"]) == {"arxiv", "openalex"}


# --------------------------------------------------------------------------------------
# QueryGenerationAgent
# --------------------------------------------------------------------------------------
def test_query_generation_keeps_only_queries_for_active_tools(monkeypatch, fake_llm):
    payload = {
        "tiered_queries": {"arxiv": {"strict": "a"}, "ghost_tool": {"strict": "g"}},
        "material_elements": ["ZnO", "", "Ni"],
    }
    use_llm(monkeypatch, planning_module, fake_llm(json.dumps(payload)))
    result = QueryGenerationAgent()._call_llm_and_parse("prompt", ["arxiv"])
    assert result == {"tiered_queries": {"arxiv": {"strict": "a"}}, "material_elements": ["ZnO", "Ni"]}


def test_query_generation_without_active_tools_returns_to_the_hub():
    state = QueryGenerationAgent().execute(new_research_state("q"))
    assert state["next"] == "supervisor_agent"
    assert state["tiered_queries"] == {}


@pytest.mark.parametrize("failure", [LLMCallError(ErrorKind.RATE_LIMITED, "slow down"), RuntimeError("bug")])
def test_query_generation_falls_back_to_the_semantic_query_for_every_tool(monkeypatch, fake_llm, failure):
    use_llm(monkeypatch, planning_module, fake_llm(failure))
    state = new_research_state("q")
    state.update(semantic_query="perovskite stability", active_tools=["arxiv", "web"])

    state = QueryGenerationAgent().execute(state)

    assert state["tiered_queries"] == {
        "arxiv": {"strict": "perovskite stability", "moderate": "perovskite stability", "broad": "perovskite stability"},
        "web": {"simple": "perovskite stability"},
    }
    assert state["api_search_term"] == "perovskite stability"
    assert len(state["errors"]) == 1 and state["errors"][0]["agent"] == "query_gen_agent"


def test_query_generation_falls_back_when_the_llm_returns_no_queries(monkeypatch, fake_llm):
    use_llm(monkeypatch, planning_module, fake_llm(json.dumps({"tiered_queries": {}, "material_elements": []})))
    state = new_research_state("q")
    state.update(semantic_query="perovskite stability", active_tools=["web"])

    state = QueryGenerationAgent().execute(state)

    assert state["tiered_queries"] == {"web": {"simple": "perovskite stability"}}
    assert state["errors"] == []  # the call worked, the answer was just empty


# --------------------------------------------------------------------------------------
# SynthesisAgent (level 5)
# --------------------------------------------------------------------------------------
LLM_REPORT = (
    "## Introduction\nClaim B [2]. Claim A [1].\n\n"
    "## References\n\n[1] Paper A\n\n[2] Paper B"
)


def synthesis_state(sample_state):
    state = sample_state("level_4_rag_output")
    state["report_generated"] = False
    state["errors"] = []  # the recorded samples predate the errors field
    return state


def evidence_state(with_chunks=True):
    """A state with two retrieved documents: what RAG hands to synthesis (source [1] = Paper A, [2] = Paper B)."""
    state = new_research_state("CsSnI3 stability")
    state["semantic_query"] = "CsSnI3 stability"
    state["raw_tool_data"] = [
        {"tool_id": "arxiv_search", "text": "Title: Paper A. Abstract: about A.",
         "metadata": {"title": "Paper A", "pdf_url": "https://example.org/a"}},
        {"tool_id": "web_search", "text": "Title: Paper B. Snippet: about B.",
         "metadata": {"title": "Paper B", "url": "https://example.org/b"}},
    ]
    if with_chunks:
        state["context_chunks"] = [
            {"chunk_id": "arxiv_search_a_0", "doc_id": "https://example.org/a", "source": "arxiv_search", "text": "Passage from A."},
            {"chunk_id": "web_search_b_0", "doc_id": "https://example.org/b", "source": "web_search", "text": "Passage from B."},
        ]
    return state


def test_synthesis_writes_a_report_and_hands_back_to_the_supervisor(monkeypatch, sample_state, fake_llm):
    use_llm(monkeypatch, synthesis_module, fake_llm(LLM_REPORT))

    state = SynthesisAgent().execute(synthesis_state(sample_state))

    assert state["report_generated"] is True
    assert state["next"] == "supervisor_agent"
    assert state["errors"] == []
    assert state["final_report"].rstrip().split("## References")[-1].strip()  # a reference list was added


def test_the_model_is_shown_numbered_sources_and_their_passages(monkeypatch, fake_llm):
    llm = use_llm(monkeypatch, synthesis_module, fake_llm("Claim [1]."))

    SynthesisAgent().execute(evidence_state())

    prompt = llm.calls[0]["contents"]
    assert "[1] Paper A (https://example.org/a)" in prompt and "Passage from A." in prompt
    assert "[2] Paper B (https://example.org/b)" in prompt and "Passage from B." in prompt
    assert "Do NOT write a References section" in prompt


def test_citations_are_renumbered_and_the_reference_list_is_built_by_code(monkeypatch, fake_llm):
    model_report = (
        "## Introduction\nClaim B [2]. Claim A [1].\n\n"
        "## References\n\n[1] An invented source\n\n[2] Another invented one"
    )
    use_llm(monkeypatch, synthesis_module, fake_llm(model_report))

    state = SynthesisAgent().execute(evidence_state())

    body, references = state["final_report"].split("## References")
    assert "Claim B [1]. Claim A [2]." in body  # numbered in order of first appearance
    assert "[1] [Paper B](https://example.org/b) (Web)" in references
    assert "[2] [Paper A](https://example.org/a) (arXiv)" in references
    assert "invented" not in state["final_report"]  # whatever the model wrote as References is discarded
    assert state["sources"] == [
        {"number": 1, "title": "Paper B", "url": "https://example.org/b", "tool_id": "web_search"},
        {"number": 2, "title": "Paper A", "url": "https://example.org/a", "tool_id": "arxiv_search"},
    ]
    assert state["invalid_citations_removed"] == 0


def test_a_citation_of_a_source_the_model_never_saw_is_removed(monkeypatch, fake_llm):
    use_llm(monkeypatch, synthesis_module, fake_llm("Real claim [1]. Invented claim [7]."))

    state = SynthesisAgent().execute(evidence_state())

    body, references = state["final_report"].split("## References")
    assert "[7]" not in state["final_report"]
    assert "Invented claim." in body  # no stray space left before the full stop
    assert state["invalid_citations_removed"] == 1
    assert [s["number"] for s in state["sources"]] == [1]  # only the source that was actually cited is listed
    assert "Paper B" not in references


def test_a_list_citation_is_written_as_separate_brackets(monkeypatch, fake_llm):
    use_llm(monkeypatch, synthesis_module, fake_llm("Both agree [1, 2]."))

    state = SynthesisAgent().execute(evidence_state())

    assert "Both agree [1][2]." in state["final_report"]
    assert [s["number"] for s in state["sources"]] == [1, 2]


def test_a_report_without_citations_lists_no_sources(monkeypatch, fake_llm):
    use_llm(monkeypatch, synthesis_module, fake_llm("The evidence does not cover this."))

    state = SynthesisAgent().execute(evidence_state())

    assert state["sources"] == []
    assert "_No sources were cited._" in state["final_report"]


def test_without_selected_chunks_the_model_only_gets_short_snippets(monkeypatch, fake_llm):
    llm = use_llm(monkeypatch, synthesis_module, fake_llm("Claim [1]."))
    state = evidence_state(with_chunks=False)
    state["raw_tool_data"][0]["text"] = "Title: Paper A. " + "word " * 100

    SynthesisAgent().execute(state)

    prompt = llm.calls[0]["contents"]
    assert "Only short snippets are available" in prompt
    assert "word " * 100 not in prompt  # cut to a short snippet
    assert "[1] Paper A" in prompt


def test_a_good_report_clears_the_failures_it_recovered_from(monkeypatch, sample_state, fake_llm):
    use_llm(monkeypatch, synthesis_module, fake_llm(LLM_REPORT))
    state = synthesis_state(sample_state)
    state["errors"] = [
        {"id": "a1", "agent": "synthesis_agent", "kind": "transient", "message": "earlier failure"},
        {"id": "b2", "agent": "clean_query_agent", "kind": "rate_limited", "message": "degraded earlier"},
    ]

    state = SynthesisAgent().execute(state)

    # only the synthesis failure is forgotten, so the supervisor counts consecutive failures only
    assert [e["id"] for e in state["errors"]] == ["b2"]


def test_synthesis_rejects_irrelevant_queries_without_calling_the_llm(monkeypatch, sample_state, fake_llm):
    llm = use_llm(monkeypatch, synthesis_module, fake_llm(LLM_REPORT))
    state = synthesis_state(sample_state)
    state["primary_intent"] = "irrelevant"

    state = SynthesisAgent().execute(state)

    assert state["final_report"] == "Query rejected based on scope."
    assert state["report_generated"] is True
    assert llm.calls == []


@pytest.mark.parametrize("kind", [ErrorKind.QUOTA_EXHAUSTED, ErrorKind.AUTH, ErrorKind.TRANSIENT, ErrorKind.RATE_LIMITED])
def test_failed_synthesis_is_recorded_and_not_marked_as_generated(monkeypatch, sample_state, fake_llm, kind):
    use_llm(monkeypatch, synthesis_module, fake_llm(LLMCallError(kind, "failed")))

    state = SynthesisAgent().execute(synthesis_state(sample_state))

    assert state["report_generated"] is False
    assert state["final_report"].startswith("Error generating report.")
    assert user_message(kind) in state["final_report"]
    assert error_kinds(state) == [kind.value]
    assert state["errors"][0]["agent"] == "synthesis_agent"
    assert state["next"] == "supervisor_agent"


def test_blocked_content_is_not_retried(monkeypatch, sample_state, fake_llm, fake_response):
    blocked = fake_response(None, prompt_feedback=SimpleNamespace(block_reason="SAFETY"))
    llm = use_llm(monkeypatch, synthesis_module, fake_llm(blocked))

    state = SynthesisAgent().execute(synthesis_state(sample_state))

    assert len(llm.calls) == 1
    assert error_kinds(state) == ["content_blocked"]


def test_empty_report_is_requested_once_more_then_recorded(monkeypatch, sample_state, fake_llm):
    llm = use_llm(monkeypatch, synthesis_module, fake_llm(""))

    state = SynthesisAgent().execute(synthesis_state(sample_state))

    assert len(llm.calls) == 2
    assert error_kinds(state) == ["bad_output"]
    assert state["report_generated"] is False


# --------------------------------------------------------------------------------------
# EvaluationAgent (level 6)
# --------------------------------------------------------------------------------------
LONG_REPORT = "A sufficiently long report sentence. " * 20


def evaluation_state(sample_state):
    state = sample_state("level_5_synthesis_output")
    state["final_report"] = LONG_REPORT
    state["errors"] = []  # the recorded samples predate the errors field
    return state


def verdict(needs_refinement, reason):
    return json.dumps({"needs_refinement": needs_refinement, "refinement_reason": reason})


def test_evaluation_accepts_a_good_report(monkeypatch, sample_state, fake_llm):
    use_llm(monkeypatch, evaluation_module, fake_llm(verdict(False, "Report is satisfactory")))

    state = EvaluationAgent().execute(evaluation_state(sample_state))

    assert state["needs_refinement"] is False
    assert state["refinement_reason"] == "Report is satisfactory"
    assert state["evaluation_unavailable"] is False
    assert state["errors"] == []
    assert state["next"] == "supervisor_agent"


def test_evaluation_can_request_refinement(monkeypatch, sample_state, fake_llm):
    use_llm(monkeypatch, evaluation_module, fake_llm(verdict(True, "Missing papers on stability")))

    state = EvaluationAgent().execute(evaluation_state(sample_state))

    assert state["needs_refinement"] is True
    assert state["refinement_reason"] == "Missing papers on stability"


def verdict_with_stage(needs_refinement, reason, stage):
    return json.dumps({"needs_refinement": needs_refinement, "refinement_reason": reason, "refine_stage": stage})


@pytest.mark.parametrize("stage", ["search", "retrieval", "context", "synthesis"])
def test_evaluation_passes_on_the_stage_it_blames(monkeypatch, sample_state, fake_llm, stage):
    use_llm(monkeypatch, evaluation_module, fake_llm(verdict_with_stage(True, "Something is off", stage)))

    state = EvaluationAgent().execute(evaluation_state(sample_state))

    assert state["needs_refinement"] is True
    assert state["refinement_stage"] == stage


def test_evaluation_without_a_stage_leaves_it_empty_for_the_supervisor_to_guess(monkeypatch, sample_state, fake_llm):
    use_llm(monkeypatch, evaluation_module, fake_llm(verdict(True, "Missing papers")))

    state = EvaluationAgent().execute(evaluation_state(sample_state))

    assert state["needs_refinement"] is True
    assert state["refinement_stage"] is None


def test_an_approved_report_carries_no_stage_even_if_the_model_names_one(monkeypatch, sample_state, fake_llm):
    use_llm(monkeypatch, evaluation_module, fake_llm(verdict_with_stage(False, "Report is satisfactory", "search")))

    state = EvaluationAgent().execute(evaluation_state(sample_state))

    assert state["needs_refinement"] is False
    assert state["refinement_stage"] is None


def test_an_unknown_stage_is_retried_once_then_flagged(monkeypatch, sample_state, fake_llm):
    llm = use_llm(monkeypatch, evaluation_module, fake_llm(verdict_with_stage(True, "x", "banana")))

    state = EvaluationAgent().execute(evaluation_state(sample_state))

    assert len(llm.calls) == 2
    assert state["evaluation_unavailable"] is True
    assert error_kinds(state) == ["bad_output"]


def test_a_too_short_report_is_sent_back_to_synthesis(monkeypatch, sample_state, fake_llm):
    use_llm(monkeypatch, evaluation_module, fake_llm(verdict(False, "ok")))
    state = evaluation_state(sample_state)
    state["final_report"] = "Too short."

    state = EvaluationAgent().execute(state)

    assert state["refinement_stage"] == "synthesis"


def test_a_failed_evaluation_names_no_stage(monkeypatch, sample_state, fake_llm):
    use_llm(monkeypatch, evaluation_module, fake_llm(LLMCallError(ErrorKind.TRANSIENT, "failed")))

    state = EvaluationAgent().execute(evaluation_state(sample_state))

    assert state["refinement_stage"] is None


def test_evaluation_sends_a_too_short_report_back_without_calling_the_llm(monkeypatch, sample_state, fake_llm):
    llm = use_llm(monkeypatch, evaluation_module, fake_llm(verdict(False, "ok")))
    state = evaluation_state(sample_state)
    state["final_report"] = "Too short."

    state = EvaluationAgent().execute(state)

    assert state["needs_refinement"] is True
    assert "insufficient" in state["refinement_reason"].lower()
    assert llm.calls == []


def test_evaluation_asks_again_after_invalid_json(monkeypatch, sample_state, fake_llm):
    llm = use_llm(monkeypatch, evaluation_module, fake_llm("{oops", verdict(False, "fine")))
    state = EvaluationAgent().execute(evaluation_state(sample_state))
    assert len(llm.calls) == 2
    assert state["evaluation_unavailable"] is False


@pytest.mark.parametrize("kind", [ErrorKind.RATE_LIMITED, ErrorKind.TRANSIENT, ErrorKind.QUOTA_EXHAUSTED, ErrorKind.AUTH])
def test_a_failed_evaluation_is_flagged_not_silently_approved(monkeypatch, sample_state, fake_llm, kind):
    use_llm(monkeypatch, evaluation_module, fake_llm(LLMCallError(kind, "failed")))

    state = EvaluationAgent().execute(evaluation_state(sample_state))

    assert state["evaluation_unavailable"] is True
    assert state["refinement_reason"].startswith("Evaluation unavailable")
    assert user_message(kind) in state["refinement_reason"]
    assert state["needs_refinement"] is False  # the report is delivered, flagged, not re-written
    assert error_kinds(state) == [kind.value]
    assert state["next"] == "supervisor_agent"


def test_an_answer_that_breaks_the_schema_is_retried_then_flagged(monkeypatch, sample_state, fake_llm):
    llm = use_llm(monkeypatch, evaluation_module, fake_llm(json.dumps({"wrong": "shape"})))

    state = EvaluationAgent().execute(evaluation_state(sample_state))

    assert len(llm.calls) == 2
    assert state["evaluation_unavailable"] is True
    assert error_kinds(state) == ["bad_output"]


# --------------------------------------------------------------------------------------
# SupervisorAgent on a recorded state (level 6 -> 7)
# --------------------------------------------------------------------------------------
GRAPH_NODES = {
    "clean_query_agent", "intent_agent", "planning_agent", "query_gen_agent",
    "semanticscholar_search", "chemrxiv_search", "pubmed_search", "arxiv_search",
    "openalex_search", "materials_search", "web_search", "retrieval_agent",
    "rag_agent", "synthesis_agent", "evaluation_agent", "END",
}


def test_supervisor_routes_a_recorded_state_to_a_node_the_graph_knows(sample_state):
    state = SupervisorAgent().execute(sample_state("level_6_evaluation_output"))
    assert state["next"] in GRAPH_NODES
    assert state["visited_nodes"][-1] == "supervisor_agent"
