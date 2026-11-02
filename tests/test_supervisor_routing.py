"""Unit tests for the SupervisorAgent routing logic.

The supervisor is a deterministic (non-LLM) router, so its decisions can be tested
without API keys, network access or the heavy ML dependencies.

Run with:  pytest            (integration tests are excluded by default)
"""
import pytest

from backend.agents.supervisor_agent import SupervisorAgent
from backend.core.errors import ErrorKind

TOOL_NODES = ["pubmed_search", "arxiv_search"]


@pytest.fixture
def supervisor():
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
# The evaluator names the stage to restart from; keywords are only a fallback
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "stage, expected_next",
    [
        ("search", "planning_agent"),
        ("retrieval", "retrieval_agent"),
        ("context", "rag_agent"),
        ("synthesis", "synthesis_agent"),
    ],
)
def test_the_stage_named_by_the_evaluator_decides_where_the_cycle_restarts(supervisor, stage, expected_next):
    state = supervisor.execute(refining_state("Something is off", refinement_stage=stage))
    assert state["next"] == expected_next


def test_the_named_stage_beats_misleading_keywords(supervisor):
    # "missing" and "papers" would send a keyword guess to the search stage; the evaluator blamed the writing
    state = supervisor.execute(refining_state("Missing papers are cited loosely", refinement_stage="synthesis"))
    assert state["next"] == "synthesis_agent"
    assert state["active_tools"] == ["pubmed"]  # the tool set was not widened


def test_an_unrecognised_stage_falls_back_to_the_keywords(supervisor):
    state = supervisor.execute(refining_state("PDF parsing failed", refinement_stage="banana"))
    assert state["next"] == "retrieval_agent"


def test_the_stage_is_cleared_once_it_has_been_used(supervisor):
    state = supervisor.execute(refining_state("Missing papers", refinement_stage="search"))
    assert state["refinement_stage"] is None


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
# Recorded errors decide whether the run can go on
# --------------------------------------------------------------------------------------
def error_entry(agent, kind, error_id="abc12345"):
    return {"id": error_id, "agent": agent, "kind": kind.value, "message": f"[{kind.value}] {agent}: failed"}


@pytest.mark.parametrize(
    "kind", [ErrorKind.AUTH, ErrorKind.CONFIG, ErrorKind.QUOTA_EXHAUSTED, ErrorKind.UNKNOWN]
)
def test_errors_that_a_retry_cannot_fix_end_the_run_with_a_clear_message(supervisor, kind):
    state = supervisor.execute(planned_state(errors=[error_entry("planning_agent", kind)]))
    assert state["next"] == "END"
    assert state["failure"]["kind"] == kind.value
    assert state["failure"]["agent"] == "planning_agent"
    assert state["failure"]["message"]


def test_an_unknown_error_tells_the_person_its_error_id(supervisor):
    state = supervisor.execute(planned_state(errors=[error_entry("planning_agent", ErrorKind.UNKNOWN)]))
    assert "abc12345" in state["failure"]["message"]
    assert state["failure"]["error_id"] == "abc12345"


@pytest.mark.parametrize(
    "kind",
    [ErrorKind.RATE_LIMITED, ErrorKind.TRANSIENT, ErrorKind.BAD_OUTPUT, ErrorKind.BAD_REQUEST, ErrorKind.CONTENT_BLOCKED],
)
def test_errors_the_agent_already_degraded_around_do_not_stop_the_run(supervisor, kind):
    state = supervisor.execute(planned_state(errors=[error_entry("query_gen_agent", kind)]))
    assert state["next"] == "pubmed_search"
    assert state.get("failure") is None


def test_a_failed_audit_does_not_throw_away_a_finished_report(supervisor):
    state = planned_state(
        visited_nodes=TOOL_NODES + ["synthesis_agent", "evaluation_agent"],
        report_generated=True,
        final_report="A finished report.",
        errors=[error_entry("evaluation_agent", ErrorKind.QUOTA_EXHAUSTED)],
    )
    state = supervisor.execute(state)
    assert state["next"] == "END"
    assert state.get("failure") is None


def test_a_failed_audit_without_a_report_ends_the_run(supervisor):
    state = supervisor.execute(planned_state(errors=[error_entry("evaluation_agent", ErrorKind.AUTH)]))
    assert state["next"] == "END"
    assert state["failure"]["kind"] == "auth"


def test_a_failed_synthesis_is_tried_once_more(supervisor):
    state = supervisor.execute(
        planned_state(visited_nodes=TOOL_NODES, errors=[error_entry("synthesis_agent", ErrorKind.TRANSIENT)])
    )
    assert state["next"] == "synthesis_agent"
    assert state.get("failure") is None


def test_a_second_failed_synthesis_ends_the_run(supervisor):
    errors = [
        error_entry("synthesis_agent", ErrorKind.TRANSIENT, "e1"),
        error_entry("synthesis_agent", ErrorKind.RATE_LIMITED, "e2"),
    ]
    state = supervisor.execute(planned_state(visited_nodes=TOOL_NODES, errors=errors))
    assert state["next"] == "END"
    assert state["failure"]["agent"] == "synthesis_agent"
    assert state["failure"]["kind"] == "rate_limited"
    assert state["failure"]["error_id"] == "e2"


@pytest.mark.parametrize("kind", [ErrorKind.CONTENT_BLOCKED, ErrorKind.BAD_REQUEST])
def test_a_synthesis_failure_that_a_retry_cannot_fix_ends_at_once(supervisor, kind):
    state = supervisor.execute(
        planned_state(visited_nodes=TOOL_NODES, errors=[error_entry("synthesis_agent", kind)])
    )
    assert state["next"] == "END"
    assert state["failure"]["kind"] == kind.value


def test_a_synthesis_that_succeeded_after_a_failure_is_not_held_against_the_run(supervisor):
    state = supervisor.execute(
        planned_state(
            visited_nodes=TOOL_NODES + ["synthesis_agent"] * 2,
            report_generated=True,
            final_report="A report.",
            errors=[error_entry("synthesis_agent", ErrorKind.TRANSIENT)],
        )
    )
    assert state["next"] == "evaluation_agent"
    assert state.get("failure") is None


# --------------------------------------------------------------------------------------
# Loop guard (replaces the sleeping rate-limit throttle)
# --------------------------------------------------------------------------------------
def test_an_agent_dispatched_over_and_over_ends_the_run_instead_of_looping(supervisor):
    visited = ["clean_query_agent"] * SupervisorAgent.MAX_DISPATCHES_PER_NODE
    state = supervisor.execute({"visited_nodes": visited})
    assert state["next"] == "END"
    assert state["failure"]["kind"] == "unknown"
    assert state["failure"]["error_id"]
    assert state["errors"][-1]["agent"] == "supervisor_agent"


def test_the_dispatch_cap_leaves_room_for_every_legitimate_repeat(supervisor):
    # busiest healthy node: synthesis = first report + one per refinement cycle + one retry after a failure
    busiest = 1 + SupervisorAgent.MAX_REFINEMENT_ATTEMPTS + 1
    assert busiest < SupervisorAgent.MAX_DISPATCHES_PER_NODE


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
