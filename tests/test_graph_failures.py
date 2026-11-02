"""The production graph wiring and the real SupervisorAgent, run with stub agents.

The supervisor's unit tests feed it hand-made states. These tests run it inside LangGraph, where the
state really travels from node to node: a key missing from ResearchState is dropped on the way, which is
exactly how the old rate-limit throttle silently never worked.
"""
import copy

from backend.agents.supervisor_agent import SupervisorAgent
from backend.core.errors import ErrorKind, LLMCallError, record_error
from backend.core.research_state import RECURSION_LIMIT, new_research_state
from backend.graph.research_graph import ResearchGraph


class StubAgent:
    """Stands in for a real agent: counts its calls, leaves its breadcrumb, applies a scripted state change."""

    def __init__(self, name, work):
        self.name, self.work, self.calls = name, work, 0

    def execute(self, state):
        self.calls += 1
        state.setdefault("visited_nodes", []).append(self.name)
        self.work(state)
        state["next"] = "supervisor_agent"
        return state


def sets(**fields):
    def work(state):
        state.update(copy.deepcopy(fields))
    return work


def fails(agent, kind):
    """The agent's LLM call failed: nothing is produced, the failure is recorded (like record_error in the agents)."""
    def work(state):
        record_error(state, agent, LLMCallError(kind, "boom", label=agent))
    return work


def tool(name):
    def work(state):
        state["raw_tool_data"].append({"tool_id": name, "metadata": {"pdf_url": f"https://example.org/{name}"}})
    return work


def retrieval(state):
    state["full_text_chunks"] = [
        {"doc_id": e["metadata"]["pdf_url"], "text": "chunk"} for e in state["raw_tool_data"]
    ]


HEALTHY = {
    "clean_query_agent": sets(semantic_query="cleaned"),
    "intent_agent": sets(primary_intent="general_research"),
    "planning_agent": sets(execution_plan=["plan"], active_tools=["arxiv"]),
    "query_gen_agent": sets(tiered_queries={"arxiv": {"strict": "q"}}),
    "arxiv_search": tool("arxiv_search"),
    "retrieval_agent": retrieval,
    "rag_agent": sets(rag_complete=True, filtered_context="context"),
    "synthesis_agent": sets(final_report="A report.", report_generated=True),
    "evaluation_agent": sets(needs_refinement=False),
}
IDLE_NODES = ["semanticscholar_search", "chemrxiv_search", "pubmed_search", "openalex_search",
              "materials_search", "web_search"]


def run(overrides=None):
    """Runs the whole graph with healthy stubs, except where `overrides` scripts something else."""
    behaviours = {**HEALTHY, **(overrides or {})}
    agents = {"supervisor_agent": SupervisorAgent()}
    for name in [*HEALTHY, *IDLE_NODES]:
        agents[name] = StubAgent(name, behaviours.get(name, lambda state: None))
    final = ResearchGraph(agents=agents).graph.invoke(
        new_research_state("q"), config={"recursion_limit": RECURSION_LIMIT}
    )
    return final, agents


def test_a_healthy_run_ends_with_a_report_and_no_failure():
    final, agents = run()
    assert final["report_generated"] is True
    assert final["failure"] is None
    assert final["errors"] == []
    assert agents["synthesis_agent"].calls == 1


def test_a_bad_key_ends_the_run_at_the_first_agent():
    final, agents = run({"clean_query_agent": fails("clean_query_agent", ErrorKind.AUTH)})
    assert final["failure"]["kind"] == "auth"
    assert final["failure"]["agent"] == "clean_query_agent"
    assert agents["clean_query_agent"].calls == 1
    assert agents["intent_agent"].calls == 0


def test_an_exhausted_daily_quota_ends_the_run_before_any_tool_runs():
    final, agents = run({"query_gen_agent": fails("query_gen_agent", ErrorKind.QUOTA_EXHAUSTED)})
    assert final["failure"]["kind"] == "quota_exhausted"
    assert agents["arxiv_search"].calls == 0
    assert final["report_generated"] is False


def test_a_synthesis_that_keeps_failing_is_tried_twice_then_ends_the_run():
    final, agents = run({"synthesis_agent": fails("synthesis_agent", ErrorKind.TRANSIENT)})
    assert agents["synthesis_agent"].calls == SupervisorAgent.MAX_SYNTHESIS_FAILURES
    assert final["failure"]["agent"] == "synthesis_agent"
    assert final["report_generated"] is False


def test_recorded_errors_survive_the_trip_through_the_graph_and_a_retry_can_succeed():
    attempts = []

    def flaky_synthesis(state):
        attempts.append(1)
        if len(attempts) == 1:
            fails("synthesis_agent", ErrorKind.RATE_LIMITED)(state)
        else:
            state.update(final_report="A report.", report_generated=True)

    final, agents = run({"synthesis_agent": flaky_synthesis})
    assert agents["synthesis_agent"].calls == 2
    assert final["report_generated"] is True
    assert final["failure"] is None
    assert [e["kind"] for e in final["errors"]] == ["rate_limited"]  # still there after two supervisor hops


def test_a_failed_audit_still_delivers_the_report():
    def audit_fails(state):
        fails("evaluation_agent", ErrorKind.QUOTA_EXHAUSTED)(state)
        state["evaluation_unavailable"] = True

    final, _ = run({"evaluation_agent": audit_fails})
    assert final["failure"] is None
    assert final["report_generated"] is True and final["final_report"]
    assert final["evaluation_unavailable"] is True


def test_an_agent_that_silently_makes_no_progress_cannot_loop_forever():
    final, agents = run({"clean_query_agent": lambda state: None})
    assert agents["clean_query_agent"].calls == SupervisorAgent.MAX_DISPATCHES_PER_NODE
    assert final["failure"]["kind"] == "unknown"
    assert final["failure"]["error_id"]


ALL_TOOLS = ["semanticscholar", "chemrxiv", "pubmed", "arxiv", "openalex", "materials", "web"]


def test_the_longest_possible_run_fits_inside_the_recursion_limit():
    """Every tool active, and an evaluator that sends the run back to search until the supervisor stops it.

    That is the busiest healthy run there is: three evaluations, two refinement cycles. It takes about 51 graph
    steps, so the old CLI limit of 50 cut such a run off at the very end, after the report already existed.
    """
    def always_refine(state):
        state.update(needs_refinement=True, refinement_reason="missing papers", refinement_stage="search")

    final, agents = run({
        "planning_agent": sets(execution_plan=["plan"], active_tools=list(ALL_TOOLS)),
        "query_gen_agent": sets(tiered_queries={tool_name: {"strict": "q"} for tool_name in ALL_TOOLS}),
        "evaluation_agent": always_refine,
        **{f"{tool_name}_search": tool(f"{tool_name}_search") for tool_name in ALL_TOOLS},
    })

    assert final["failure"] is None
    assert final["refinement_retries"] == SupervisorAgent.MAX_REFINEMENT_ATTEMPTS
    assert agents["evaluation_agent"].calls == 1 + SupervisorAgent.MAX_REFINEMENT_ATTEMPTS
    assert final["report_generated"] is True
