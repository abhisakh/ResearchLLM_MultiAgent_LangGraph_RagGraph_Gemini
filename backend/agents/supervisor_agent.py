import logging

from backend.core.errors import ErrorKind, LLMCallError, describe_failure, record_error
from backend.core.research_state import ResearchState

logger = logging.getLogger(__name__)

# ==================================================================================================
# SECTION 1: SUPERVISOR AGENT (PROCEDURAL ROUTER)
# ==================================================================================================
class SupervisorAgent:
    """
    Finalized Hub-and-Spoke Orchestrator.
    Controls the flow between specialized agents, handles guardrails for
    irrelevant queries, manages the iterative refinement loop, and ends the run
    when the errors recorded by the agents show it cannot succeed.
    """

    MAX_REFINEMENT_ATTEMPTS = 2

    # A failed synthesis is tried once more (the LLM layer has already retried transient errors
    # and switched models); a second failure ends the run.
    MAX_SYNTHESIS_FAILURES = 2
    # Retrying a synthesis cannot help when the request itself is the problem.
    NOT_WORTH_RETRYING = {ErrorKind.CONTENT_BLOCKED.value, ErrorKind.BAD_REQUEST.value}

    # Safety net against any agent being dispatched over and over without making progress.
    # In a healthy run the busiest node (synthesis) runs at most 1 + MAX_REFINEMENT_ATTEMPTS times
    # plus one retry, so this is never reached. Hitting it means a bug: the run ends with an error ID
    # instead of looping until LangGraph's recursion limit.
    MAX_DISPATCHES_PER_NODE = 8

    # Stages a refinement cycle can restart from. The evaluator names one (state["refinement_stage"]).
    REFINEMENT_STAGES = ("search", "retrieval", "context", "synthesis")

    # Keyword sets: only a fallback to guess the stage when the evaluator did not name one
    # (an older recorded state, or a model that left the field empty).
    DATA_KEYWORDS = ["missing", "data", "search", "papers", "pubmed", "arxiv", "found", "sources", "literature"]
    PDF_KEYWORDS = ["pdf", "extraction", "parsing", "read"]
    CONTEXT_KEYWORDS = ["relevance", "context", "snippets", "rag"]

    def __init__(self, agent_id: str = "supervisor_agent"):
        self.id = agent_id
        # Mapping active_tools to their specific graph node IDs
        self.tool_node_map = {
            "semanticscholar": "semanticscholar_search",
            "chemrxiv": "chemrxiv_search",
            "pubmed": "pubmed_search",
            "arxiv": "arxiv_search",
            "openalex": "openalex_search",
            "materials": "materials_search",
            "web": "web_search"
        }

    def execute(self, state: ResearchState) -> ResearchState:
        # 1. Breadcrumb Tracking
        state.setdefault("visited_nodes", []).append(self.id)

        logger.info(f'[{self.id.upper()} HUB] Analyzing state for next dispatch...')

        # 2. Recorded errors decide first whether the run can go on at all
        failing = self._find_run_ending_error(state)
        if failing:
            return self._end_run(state, failing)

        # 3. Guardrail: Out-of-Scope Handling
        # If IntentAgent flagged 'irrelevant', bypass research and go to Synthesis for rejection
        if state.get("primary_intent") == "irrelevant":
            if state.get("report_generated"):
                state["next"] = "END"
                return state
            logger.warning(f'[{self.id.upper()} GUARDRAIL] Irrelevant query detected. Routing to Synthesis Agent.')
            state["next"] = "synthesis_agent"
            return state

        # 4. Refinement Gate: Handle loops from EvaluationAgent
        if state.get("needs_refinement", False):
            return self._handle_refinement(state)

        # 5. Standard Orchestration: Sequential Flow Checklist
        next_destination = self.select_next_agent(state)

        # 6. Loop guard: no agent may be dispatched over and over
        dispatched = state["visited_nodes"].count(next_destination)
        if next_destination != "END" and dispatched >= self.MAX_DISPATCHES_PER_NODE:
            error = LLMCallError(
                ErrorKind.UNKNOWN,
                f"{next_destination} was dispatched {dispatched} times without finishing its work",
                label=self.id,
            )
            return self._end_run(state, record_error(state, self.id, error))

        state["next"] = next_destination
        logger.info(f"[{self.id.upper()} HUB] Next Destination: **{state['next']}**")
        return state

    # ----------------------------------------------------------------------------------------------
    # Error handling
    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _kind_of(entry: dict) -> ErrorKind:
        try:
            return ErrorKind(entry.get("kind"))
        except ValueError:
            return ErrorKind.UNKNOWN

    def _find_run_ending_error(self, state: ResearchState):
        """The recorded error that makes further work pointless, or None if the run can go on.

        Agents that failed but degraded to a fallback (a rate limit, a transient error, bad output) do
        not stop the run: the fallback already moved the state forward. A run ends when
        - the kind cannot be fixed by waiting or retrying (bad key, wrong model, daily quota, a bug), or
        - synthesis keeps failing, so there is no report to deliver.
        """
        errors = state.get("errors", [])
        report_exists = bool(state.get("report_generated") and state.get("final_report"))

        for entry in errors:
            if self._kind_of(entry).stops_run:
                # A failed audit must not throw away a finished report: it is delivered flagged
                # evaluation_unavailable (see EvaluationAgent) instead of being dropped.
                if entry.get("agent") == "evaluation_agent" and report_exists:
                    continue
                return entry

        failed_syntheses = [e for e in errors if e.get("agent") == "synthesis_agent"]
        if failed_syntheses and not state.get("report_generated"):
            last = failed_syntheses[-1]
            if len(failed_syntheses) >= self.MAX_SYNTHESIS_FAILURES or last.get("kind") in self.NOT_WORTH_RETRYING:
                return last
        return None

    def _end_run(self, state: ResearchState, entry: dict) -> ResearchState:
        """Stops the workflow and leaves a person-readable reason in state["failure"]."""
        failure = describe_failure(entry)
        state["failure"] = failure
        state["next"] = "END"
        logger.error(
            f"[{self.id.upper()} ABORT] {failure['agent']} ended the run ({failure['kind']}): {failure['message']}"
        )
        return state

    # ----------------------------------------------------------------------------------------------
    # Routing
    # ----------------------------------------------------------------------------------------------
    def select_next_agent(self, state: ResearchState) -> str:
        """Determines the next logical node based on state completeness."""
        visited = state.get("visited_nodes", [])
        retries = state.get("refinement_retries", 0)

        # --- A. Setup Phase ---
        if not state.get("semantic_query"): return "clean_query_agent"
        if not state.get("primary_intent"): return "intent_agent"
        if not state.get("execution_plan"): return "planning_agent"
        if not state.get("tiered_queries"): return "query_gen_agent"

        # --- B. Tool Execution Phase (The Star Spokes) ---
        active_tools = state.get("active_tools", [])
        for tool in active_tools:
            node_name = self.tool_node_map.get(tool)
            if node_name and node_name not in visited:
                if tool in state.get("tiered_queries", {}):
                    return node_name

        # --- C. Processing & Finalization Phase ---
        if state.get("raw_tool_data") and not state.get("full_text_chunks"):
            return "retrieval_agent"
        # After a refinement cycle: fetch the documents added by newly run tools.
        # Bounded: at most one extra retrieval pass per refinement cycle.
        if (
            state.get("is_refining")
            and visited.count("retrieval_agent") <= retries
            and self._has_unretrieved_data(state)
        ):
            return "retrieval_agent"
        if state.get("full_text_chunks") and not state.get("rag_complete"):
            return "rag_agent"

        if not state.get("report_generated"): return "synthesis_agent"

        if state.get("report_generated") and not state.get("needs_refinement"):
            # Every synthesis gets evaluated: the first report, and the revised report
            # produced by each refinement cycle (MAX_REFINEMENT_ATTEMPTS bounds the loop).
            if visited.count("evaluation_agent") <= retries:
                return "evaluation_agent"
            return "END"

        return "END"

    @staticmethod
    def _normalize_url(url):
        return url.rstrip(".") if url else None

    def _has_unretrieved_data(self, state: ResearchState) -> bool:
        """True if raw_tool_data holds a document whose content retrieval has not processed yet.

        Mirrors RetrievalAgent: it keys chunks by doc_id (the document URL) and skips
        Materials Project entries, which have no document to download.
        """
        processed = {
            c.get("doc_id")
            for c in state.get("full_text_chunks", [])
            if isinstance(c, dict) and c.get("doc_id")
        }
        for entry in state.get("raw_tool_data", []):
            if entry.get("tool_id") == "materials_search":
                continue
            meta = entry.get("metadata") or {}
            urls = {self._normalize_url(meta.get("pdf_url")), self._normalize_url(meta.get("url"))} - {None}
            if urls and not (urls & processed):
                return True
        return False

    def _refinement_stage(self, state: ResearchState, reason: str) -> str:
        """Where a refinement cycle restarts: search, retrieval, context or synthesis.

        The evaluator names the stage itself (state["refinement_stage"]). Only when it did not is the stage
        guessed from keywords in the reason, in the old precedence order: search, then retrieval, then context.
        """
        stage = state.get("refinement_stage")
        if stage in self.REFINEMENT_STAGES:
            return stage
        if any(k in reason for k in self.DATA_KEYWORDS):
            return "search"
        if any(k in reason for k in self.PDF_KEYWORDS):
            return "retrieval"
        if any(k in reason for k in self.CONTEXT_KEYWORDS):
            return "context"
        return "synthesis"

    def _handle_refinement(self, state: ResearchState) -> ResearchState:
        """Logic to reset state and pivot strategy based on evaluator feedback."""
        retries = state.get("refinement_retries", 0)

        if retries >= self.MAX_REFINEMENT_ATTEMPTS:
            logger.warning(f'[{self.id.upper()}] Max refinement attempts reached. Terminating.')
            state["next"] = "END"
            return state

        state["refinement_retries"] = retries + 1
        reason = state.get("refinement_reason", "").lower()

        logger.info(f"[{self.id.upper()} REFINEMENT] Cycle {state['refinement_retries']}: {reason[:100]}...")

        stage = self._refinement_stage(state, reason)

        state.update({
            "is_refining": True,
            "needs_refinement": False,
            "refinement_stage": None,  # consumed: the next evaluation names its own
            "report_generated": False,
            "rag_complete": False,
            "filtered_context": ""
        })

        if stage == "search":
            active_tools = state.get("active_tools", [])

            # Dynamically ensure missing preprint/database tools mentioned in reason are added
            target_tools = ["openalex", "semanticscholar"]
            if "arxiv" in reason or "preprint" in reason:
                target_tools.append("arxiv")
            if "chemrxiv" in reason:
                target_tools.append("chemrxiv")

            for t in target_tools:
                if t not in active_tools:
                    active_tools.append(t)

            state["active_tools"] = active_tools
            # Drop the old queries so query_gen_agent runs again and writes queries for the
            # newly added tools; otherwise the tool phase would skip them.
            state["tiered_queries"] = {}
            state["next"] = "planning_agent"
        elif stage == "retrieval":
            state["next"] = "retrieval_agent"
        elif stage == "context":
            state["next"] = "rag_agent"
        else:
            state["next"] = "synthesis_agent"

        return state
