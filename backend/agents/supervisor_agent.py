import logging
import time  # <--- Added for rate-limit throttling

from backend.core.research_state import ResearchState

logger = logging.getLogger(__name__)

# ==================================================================================================
# SECTION 1: SUPERVISOR AGENT (PROCEDURAL ROUTER)
# ==================================================================================================
class SupervisorAgent:
    """
    Finalized Hub-and-Spoke Orchestrator.
    Controls the flow between specialized agents, handles guardrails for
    irrelevant queries, and manages the iterative refinement loop.
    """

    MAX_REFINEMENT_ATTEMPTS = 2
    MAX_CONSECUTIVE_RETRIES = 5  # <--- Guardrail threshold to step down aggressiveness

    # Keyword sets for automated refinement triage
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

        # 2. Guardrail: Out-of-Scope Handling
        # If IntentAgent flagged 'irrelevant', bypass research and go to Synthesis for rejection
        if state.get("primary_intent") == "irrelevant":
            if state.get("report_generated"):
                state["next"] = "END"
                return state
            logger.warning(f'[{self.id.upper()} GUARDRAIL] Irrelevant query detected. Routing to Synthesis Agent.')
            state["next"] = "synthesis_agent"
            return state

        # 3. Refinement Gate: Handle loops from EvaluationAgent
        if state.get("needs_refinement", False):
            return self._handle_refinement(state)

        # 4. Standard Orchestration: Sequential Flow Checklist
        next_destination = self.select_next_agent(state)

        # --- NEW CONCURRENCY RATE LIMIT MITIGATION ---
        # Track if the workflow is stuck trying to invoke the same agent repeatedly
        consecutive_tracker = state.get("consecutive_node_counts", {})
        current_count = consecutive_tracker.get(next_destination, 0) + 1
        consecutive_tracker[next_destination] = current_count
        state["consecutive_node_counts"] = consecutive_tracker

        # If an agent loop is detected (like query_gen_agent hitting 429 over and over)
        if current_count > 1:
            # Exponentially back off: 2s, 4s, 8s, 16s, etc.
            backoff_sleep = min(2 ** current_count, 45)
            logger.info(f"[{self.id.upper()} THROTTLE] Detected loop on '{next_destination}' (Hit count: {current_count}). Free quota cooling down. Sleeping for {backoff_sleep}s...")
            time.sleep(backoff_sleep)

            # If it's hammered the free tier too many times consecutively, gracefully pivot
            if current_count >= self.MAX_CONSECUTIVE_RETRIES and next_destination == "query_gen_agent":
                logger.warning(f'[{self.id.upper()} BREAK] Rate limit loop unbroken. Injecting blank queries object to escape cycle.')
                state["tiered_queries"] = {"fallback": ["General system search overview"]}
                next_destination = self.select_next_agent(state)
        else:
            # Clear historical counters for other nodes if we successfully progressed elsewhere
            state["consecutive_node_counts"] = {next_destination: 1}
        # ----------------------------------------------

        state["next"] = next_destination
        logger.info(f'[{self.id.upper()} HUB] Next Destination: **{state['next']}**')
        return state

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

    def _handle_refinement(self, state: ResearchState) -> ResearchState:
        """Logic to reset state and pivot strategy based on evaluator feedback."""
        retries = state.get("refinement_retries", 0)

        if retries >= self.MAX_REFINEMENT_ATTEMPTS:
            logger.warning(f'[{self.id.upper()}] Max refinement attempts reached. Terminating.')
            state["next"] = "END"
            return state

        state["refinement_retries"] = retries + 1
        reason = state.get("refinement_reason", "").lower()

        logger.info(f'[{self.id.upper()} REFINEMENT] Cycle {state['refinement_retries']}: {reason[:100]}...')

        is_data_issue = any(k in reason for k in self.DATA_KEYWORDS)
        is_pdf_issue = any(k in reason for k in self.PDF_KEYWORDS)
        is_context_issue = any(k in reason for k in self.CONTEXT_KEYWORDS)

        state.update({
            "is_refining": True,
            "needs_refinement": False,
            "report_generated": False,
            "rag_complete": False,
            "filtered_context": ""
        })

        if is_data_issue:
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
        elif is_pdf_issue:
            state["next"] = "retrieval_agent"
        elif is_context_issue:
            state["next"] = "rag_agent"
        else:
            state["next"] = "synthesis_agent"

        return state
