"""Unit tests for the SupervisorAgent routing logic.

The supervisor is a deterministic (non-LLM) router, so its decisions can be tested
without API keys, network access or the heavy ML dependencies.

Run with:  pytest            (integration tests are excluded by default)
"""
import pytest

from backend.agents import supervisor_agent as supervisor_module
from backend.agents.supervisor_agent import SupervisorAgent

TOOL_NODES = ["pubmed_search", "arxiv_search"]


@pytest.fixture
def sleeps(monkeypatch):
    """Replace time.sleep so the rate-limit back-off path runs instantly; returns the recorded delays."""
    recorded = []
    monkeypatch.setattr(supervisor_module.time, "sleep", recorded.append)
    return recorded


@pytest.fixture
def supervisor(sleeps):
    return SupervisorAgent()


def planned_state(**overrides):
    """A state that has finished the setup phase (cleaned, classified, planned, queries generated)."""
    state = {
        "user_query": "thermal stability of CsPbI3",
        "semantic_query": "thermal stability of CsPbI3 perovskites",
        "primary_intent": "general_research",
        "execution_plan": ["search", "synthesize"],
        "tiered_queries": {"pubmed": {"strict": "q1"}, "arxiv": {"strict": "q2"}},
        "active_tools": ["pubmed", "arxiv"],
        "visited_nodes": [],
        "raw_tool_data": [],
        "full_text_chunks": [],
        "rag_complete": False,
        "report_generated": False,
        "needs_refinement": False,
        "refinement_retries": 0,
    }
    state.update(overrides)
    return state


# --------------------------------------------------------------------------------------
# Setup phase: the supervisor walks the checklist in a fixed order
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "state, expected",
    [
        ({}, "clean_query_agent"),
        ({"semantic_query": "q"}, "intent_agent"),
        ({"semantic_query": "q", "primary_intent": "general_research"}, "planning_agent"),
        (
            {"semantic_query": "q", "primary_intent": "general_research", "execution_plan": ["p"]},
            "query_gen_agent",
        ),
    ],
)
def test_setup_phase_order(supervisor, state, expected):
    assert supervisor.select_next_agent(state) == expected


# --------------------------------------------------------------------------------------
# Tool phase
# --------------------------------------------------------------------------------------
def test_tools_run_in_plan_order_and_only_once(supervisor):
    assert supervisor.select_next_agent(planned_state()) == "pubmed_search"
    assert supervisor.select_next_agent(planned_state(visited_nodes=["pubmed_search"])) == "arxiv_search"


def test_tool_without_generated_queries_is_skipped(supervisor):
    state = planned_state(active_tools=["web", "arxiv"])  # no "web" entry in tiered_queries
    assert supervisor.select_next_agent(state) == "arxiv_search"


def test_unknown_tool_is_ignored(supervisor):
    state = planned_state(active_tools=["not_a_tool"])
    assert supervisor.select_next_agent(state) == "synthesis_agent"


# --------------------------------------------------------------------------------------
# Processing and finalisation phase
# --------------------------------------------------------------------------------------
def test_raw_data_goes_to_retrieval(supervisor):
    state = planned_state(visited_nodes=TOOL_NODES, raw_tool_data=[{"title": "paper"}])
    assert supervisor.select_next_agent(state) == "retrieval_agent"


def test_chunks_go_to_rag_until_rag_is_complete(supervisor):
    state = planned_state(
        visited_nodes=TOOL_NODES, raw_tool_data=[{}], full_text_chunks=[{"text": "chunk"}]
    )
    assert supervisor.select_next_agent(state) == "rag_agent"
    state["rag_complete"] = True
    assert supervisor.select_next_agent(state) == "synthesis_agent"


def test_no_retrieved_data_goes_straight_to_synthesis(supervisor):
    assert supervisor.select_next_agent(planned_state(visited_nodes=TOOL_NODES)) == "synthesis_agent"


def test_report_is_evaluated_once_then_workflow_ends(supervisor):
    state = planned_state(visited_nodes=TOOL_NODES, report_generated=True)
    assert supervisor.select_next_agent(state) == "evaluation_agent"
    state["visited_nodes"] = TOOL_NODES + ["evaluation_agent"]
    assert supervisor.select_next_agent(state) == "END"


# --------------------------------------------------------------------------------------
# execute(): breadcrumbs and the out-of-scope guardrail
# --------------------------------------------------------------------------------------
def test_execute_records_its_own_visit(supervisor):
    state = supervisor.execute(planned_state())
    assert state["visited_nodes"] == ["supervisor_agent"]
    assert state["next"] == "pubmed_search"


def test_irrelevant_query_skips_research_and_goes_to_synthesis(supervisor):
    state = supervisor.execute(planned_state(primary_intent="irrelevant"))
    assert state["next"] == "synthesis_agent"


def test_irrelevant_query_ends_after_the_rejection_is_written(supervisor):
    state = supervisor.execute(planned_state(primary_intent="irrelevant", report_generated=True))
    assert state["next"] == "END"


# --------------------------------------------------------------------------------------
# Refinement loop
# --------------------------------------------------------------------------------------
def refining_state(reason, **overrides):
    defaults = dict(
        needs_refinement=True,
        refinement_reason=reason,
        active_tools=["pubmed"],
        report_generated=True,
        rag_complete=True,
        filtered_context="old context",
    )
    defaults.update(overrides)
    return planned_state(**defaults)


def test_refinement_stops_after_max_attempts(supervisor):
    state = supervisor.execute(
        refining_state("missing papers", refinement_retries=SupervisorAgent.MAX_REFINEMENT_ATTEMPTS)
    )
    assert state["next"] == "END"
    assert state["refinement_retries"] == SupervisorAgent.MAX_REFINEMENT_ATTEMPTS


def test_data_issue_widens_the_tool_set_and_replans(supervisor):
    state = supervisor.execute(refining_state("Missing papers on perovskites"))
    assert state["next"] == "planning_agent"
    assert state["active_tools"] == ["pubmed", "openalex", "semanticscholar"]
    assert state["refinement_retries"] == 1
    assert state["tiered_queries"] == {}  # cleared so query generation runs again


def test_refinement_resets_the_control_flags(supervisor):
    state = supervisor.execute(refining_state("Missing papers"))
    assert state["is_refining"] is True
    assert state["needs_refinement"] is False
    assert state["report_generated"] is False
    assert state["rag_complete"] is False
    assert state["filtered_context"] == ""


@pytest.mark.parametrize(
    "reason, expected_next",
    [
        ("PDF parsing failed", "retrieval_agent"),
        ("Low relevance of the retrieved snippets", "rag_agent"),
        ("The tone of the report is off", "synthesis_agent"),
    ],
)
def test_refinement_jumps_back_to_the_stage_named_in_the_feedback(supervisor, reason, expected_next):
    assert supervisor.execute(refining_state(reason))["next"] == expected_next


def test_data_keywords_take_precedence_over_pdf_keywords(supervisor):
    """Documents current behaviour: 'missing' (data) is checked before 'pdf'."""
    assert supervisor.execute(refining_state("missing pdf text"))["next"] == "planning_agent"


@pytest.mark.parametrize(
    "reason, extra_tool",
    [("no chemrxiv hits found", "chemrxiv"), ("needs more arxiv preprint coverage", "arxiv")],
)
def test_feedback_can_add_specific_preprint_tools(supervisor, reason, extra_tool):
    assert extra_tool in supervisor.execute(refining_state(reason))["active_tools"]


def test_refinement_does_not_duplicate_tools(supervisor):
    state = supervisor.execute(refining_state("missing", active_tools=["openalex"]))
    assert state["active_tools"] == ["openalex", "semanticscholar"]


# --------------------------------------------------------------------------------------
# Re-querying, re-retrieval and re-evaluation after a refinement cycle
# --------------------------------------------------------------------------------------
def test_data_refinement_sends_the_workflow_back_through_query_generation(supervisor):
    state = supervisor.execute(refining_state("Missing papers"))
    assert supervisor.select_next_agent(state) == "query_gen_agent"


def test_revised_report_is_evaluated_again_after_a_refinement_cycle(supervisor):
    state = planned_state(
        visited_nodes=TOOL_NODES + ["evaluation_agent"], report_generated=True, refinement_retries=1
    )
    assert supervisor.select_next_agent(state) == "evaluation_agent"


def test_each_report_is_evaluated_only_once(supervisor):
    state = planned_state(
        visited_nodes=TOOL_NODES + ["evaluation_agent"] * 2, report_generated=True, refinement_retries=1
    )
    assert supervisor.select_next_agent(state) == "END"


def retrieval_pending_state(**overrides):
    """Refinement state: arxiv documents are processed, openalex documents are not."""
    state = planned_state(
        visited_nodes=TOOL_NODES + ["retrieval_agent", "openalex_search"],
        is_refining=True,
        refinement_retries=1,
        raw_tool_data=[
            {"tool_id": "arxiv_search", "metadata": {"pdf_url": "https://a.org/1"}},
            {"tool_id": "openalex_search", "metadata": {"pdf_url": "https://o.org/2"}},
        ],
        full_text_chunks=[{"doc_id": "https://a.org/1", "text": "chunk"}],
    )
    state.update(overrides)
    return state


def test_documents_from_new_tools_are_retrieved_after_refinement(supervisor):
    assert supervisor.select_next_agent(retrieval_pending_state()) == "retrieval_agent"


def test_retrieval_is_skipped_when_every_document_is_already_processed(supervisor):
    chunks = [
        {"doc_id": "https://a.org/1", "text": "chunk"},
        {"doc_id": "https://o.org/2", "text": "chunk"},
    ]
    state = retrieval_pending_state(full_text_chunks=chunks)
    assert supervisor.select_next_agent(state) == "rag_agent"


def test_extra_retrieval_is_limited_to_one_pass_per_refinement_cycle(supervisor):
    state = retrieval_pending_state(
        visited_nodes=TOOL_NODES + ["retrieval_agent"] * 2 + ["openalex_search"]
    )
    assert supervisor.select_next_agent(state) == "rag_agent"


def test_extra_retrieval_only_happens_while_refining(supervisor):
    assert supervisor.select_next_agent(retrieval_pending_state(is_refining=False)) == "rag_agent"


def test_materials_entries_never_trigger_retrieval(supervisor):
    state = retrieval_pending_state(
        raw_tool_data=[{"tool_id": "materials_search", "metadata": {"pdf_url": "https://mp.org/3"}}],
        full_text_chunks=[{"doc_id": "https://a.org/1", "text": "chunk"}],
    )
    assert supervisor.select_next_agent(state) == "rag_agent"


# --------------------------------------------------------------------------------------
# Rate-limit throttle
# --------------------------------------------------------------------------------------
def test_repeated_dispatch_to_the_same_node_backs_off(supervisor, sleeps):
    state = {"visited_nodes": []}
    supervisor.execute(state)  # first dispatch to clean_query_agent: no delay
    assert sleeps == []
    supervisor.execute(state)  # same node again: exponential back-off (2 ** 2 seconds)
    assert sleeps == [4]
    assert state["next"] == "clean_query_agent"


def test_query_gen_loop_is_broken_with_fallback_queries(supervisor, sleeps):
    state = planned_state(
        tiered_queries={}, active_tools=[], consecutive_node_counts={"query_gen_agent": 4}
    )
    state = supervisor.execute(state)
    assert sleeps == [32]
    assert state["tiered_queries"] == {"fallback": ["General system search overview"]}
    assert state["next"] == "synthesis_agent"


# --------------------------------------------------------------------------------------
# Whole-graph behaviour with stub agents
# --------------------------------------------------------------------------------------
def run_with_stub_agents(supervisor, first_evaluation_feedback, always_reject=False, max_steps=80):
    """Drive the supervisor like the LangGraph hub does, with trivial stand-ins for every agent.

    Each stub applies the state change its real counterpart makes (including appending its own
    name to visited_nodes). The first evaluation asks for refinement if feedback is given.
    Returns (route taken, number of evaluations, tool nodes that ran).
    """
    state = {
        "user_query": "q",
        "visited_nodes": [],
        "raw_tool_data": [],
        "full_text_chunks": [],
        "rag_complete": False,
        "report_generated": False,
        "needs_refinement": False,
        "refinement_retries": 0,
    }
    route, tool_runs, evaluations = [], [], 0

    for _ in range(max_steps):
        state = supervisor.execute(state)
        node = state["next"]
        route.append(node)
        if node == "END":
            break
        state["visited_nodes"].append(node)

        if node == "clean_query_agent":
            state["semantic_query"] = "cleaned"
        elif node == "intent_agent":
            state["primary_intent"] = "general_research"
        elif node == "planning_agent":  # like the real one: keeps tiered_queries, merges tools
            state["execution_plan"] = ["plan"]
            state["active_tools"] = list(dict.fromkeys(state.get("active_tools", []) + ["arxiv"]))
        elif node == "query_gen_agent":
            state["tiered_queries"] = {t: {"strict": "q"} for t in state["active_tools"]}
        elif node.endswith("_search"):
            tool_runs.append(node)
            state["raw_tool_data"].append(
                {"tool_id": node, "metadata": {"pdf_url": f"https://example.org/{node}"}}
            )
        elif node == "retrieval_agent":  # incremental, like the real agent: skips processed documents
            done = {c["doc_id"] for c in state["full_text_chunks"]}
            state["full_text_chunks"] += [
                {"doc_id": e["metadata"]["pdf_url"], "text": "chunk"}
                for e in state["raw_tool_data"]
                if e["metadata"]["pdf_url"] not in done
            ]
        elif node == "rag_agent":
            state["rag_complete"] = True
        elif node == "synthesis_agent":
            state["report_generated"] = True
        elif node == "evaluation_agent":
            evaluations += 1
            if first_evaluation_feedback and (always_reject or evaluations == 1):
                state["needs_refinement"] = True
                state["refinement_reason"] = first_evaluation_feedback

    return route, evaluations, tool_runs


def test_happy_path_runs_every_stage_once_and_ends(supervisor):
    route, evaluations, tool_runs = run_with_stub_agents(supervisor, first_evaluation_feedback="")
    assert route[-1] == "END"
    assert route[:4] == ["clean_query_agent", "intent_agent", "planning_agent", "query_gen_agent"]
    assert evaluations == 1
    assert tool_runs == ["arxiv_search"]


def test_refinement_cycle_reruns_new_tools_and_reevaluates(supervisor):
    route, evaluations, tool_runs = run_with_stub_agents(
        supervisor, first_evaluation_feedback="Missing papers on perovskites"
    )
    assert route[-1] == "END"
    assert tool_runs == ["arxiv_search", "openalex_search", "semanticscholar_search"]
    assert route.count("query_gen_agent") == 2
    assert route.count("retrieval_agent") == 2
    assert route.count("rag_agent") == 2
    assert evaluations == 2


def test_refinement_loop_is_bounded_when_the_evaluator_never_approves(supervisor):
    route, evaluations, _ = run_with_stub_agents(
        supervisor, first_evaluation_feedback="Missing papers", always_reject=True
    )
    assert route[-1] == "END"
    assert evaluations == SupervisorAgent.MAX_REFINEMENT_ATTEMPTS + 1
