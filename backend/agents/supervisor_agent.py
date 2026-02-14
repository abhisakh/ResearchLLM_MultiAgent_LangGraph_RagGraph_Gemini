from typing import Dict, Any, List
from core.research_state import ResearchState
from core.utilities import C_ACTION, C_RESET, C_BLUE, C_YELLOW, C_MAGENTA, C_RED, C_CYAN

# ==================================================================================================
# SECTION 1: SUPERVISOR AGENT (PROCEDURAL ROUTER)
# ==================================================================================================
class SupervisorAgent:
    """
    Procedural Router for LangGraph-based multi-agent research workflow.
    Fully aligned with ResearchState TypedDict.
    """

    MAX_REFINEMENT_ATTEMPTS = 2

    DATA_KEYWORDS = [
        "missing", "data", "search", "papers",
        "pubmed", "arxiv", "found", "sources", "literature"
    ]
    PDF_KEYWORDS = ["pdf", "extraction", "parsing", "read"]
    CONTEXT_KEYWORDS = ["relevance", "context", "snippets", "rag"]

    def __init__(self, agent_id: str = "supervisor_agent"):
        self.id = agent_id
        print(f"{C_CYAN}[{self.id.upper()} INIT] Supervisor logic synchronized.{C_RESET}")

    def execute(self, state: ResearchState) -> ResearchState:
        # --- Breadcrumb Tracking ---
        state.setdefault("visited_nodes", []).append(self.id)

        # --- Guardrail: Irrelevant Intent ---
        if state.get("primary_intent") == "irrelevant":
            state["next"] = "END"
            return state

        # --- Refinement Gate ---
        if state.get("needs_refinement", False):
            return self._handle_refinement(state)

        # --- Normal Sequential Routing ---
        state["next"] = self.select_next_agent(state)
        return state

    # =====================================================
    # Refinement Handler (State-Accurate)
    # =====================================================
    def _handle_refinement(self, state: ResearchState) -> ResearchState:
        retries = state.get("refinement_retries", 0)

        if retries >= self.MAX_REFINEMENT_ATTEMPTS:
            state["next"] = "END"
            return state

        retries += 1
        state["refinement_retries"] = retries

        reason = state.get("refinement_reason", "").lower()

        print(f"\n{C_MAGENTA}[SUPERVISOR] Refinement {retries}/{self.MAX_REFINEMENT_ATTEMPTS}{C_RESET}")
        print(f"{C_MAGENTA}Reason: {reason}{C_RESET}")

        # --- Classify refinement ---
        is_data_issue = any(k in reason for k in self.DATA_KEYWORDS)
        is_pdf_issue = any(k in reason for k in self.PDF_KEYWORDS)
        is_context_issue = any(k in reason for k in self.CONTEXT_KEYWORDS)

        injected = False

        # --- Tool Injection ONLY for data acquisition ---
        if is_data_issue:
            tools_to_inject = ["openalex", "semanticscholar", "chemrxiv"]
            active_tools = state.get("active_tools", [])

            for tool in tools_to_inject:
                if tool not in active_tools:
                    active_tools.append(tool)
                    injected = True

            state["active_tools"] = active_tools

        # --- Reset ONLY downstream regenerable artifacts ---
        state.update({
            "is_refining": True,
            "needs_refinement": False,
            "report_generated": False,
            "final_report": "",
            "filtered_context": "",
            "rag_complete": False
        })

        # =================================================
        # ENFORCED ROUTING (CRITICAL FIX)
        # =================================================
        if injected:
            # MUST regenerate queries so tools actually execute
            state["next"] = "planning_agent"
            return state

        if is_data_issue:
            # Even if 'injected' is False, if the Evaluator is unhappy
            # with data coverage, we should re-plan or re-query.
            state["next"] = "query_gen_agent" # Go here to regenerate better queries
            return state

        if is_pdf_issue:
            state["next"] = "retrieval_agent"

        elif is_context_issue:
            state["next"] = "rag_agent"

        else:
            state["next"] = "synthesis_agent"

        return state

    # =====================================================
    # Initial Sequential Flow (Exact to ResearchState)
    # =====================================================
    def select_next_agent(self, state: ResearchState) -> str:
        if state.get("report_generated") and not state.get("needs_refinement"):
            return "END"

        if not state.get("semantic_query"):
            return "clean_query_agent"

        if not state.get("primary_intent"):
            return "intent_agent"

        if not state.get("execution_plan"):
            return "planning_agent"

        if not state.get("tiered_queries"):
            return "query_gen_agent"

        if not state.get("raw_tool_data"):
            return "retrieval_agent"

        if not state.get("full_text_chunks"):
            return "retrieval_agent"

        if not state.get("filtered_context") or not state.get("rag_complete"):
            return "rag_agent"

        if not state.get("report_generated"):
            return "synthesis_agent"

        return "evaluation_agent"


# class SupervisorAgent:
#     """
#     Procedural Router: Manages the lifecycle of the research,
#     handling the transition from initial planning to iterative refinement.
#     RESTORED: Full deterministic routing and phase-based logic.
#     UPGRADED: Triple-Tool Injection (OpenAlex, SemanticScholar, ChemRxiv) for refinement.
#     """
#     MAX_REFINEMENT_ATTEMPTS = 2

#     def __init__(self, agent_id: str = "supervisor_agent"):
#         self.id = agent_id
#         print(f"{C_CYAN}[{self.id.upper()} INIT] Supervisor logic synchronized.{C_RESET}")

#     def execute(self, state: ResearchState) -> ResearchState:
#         # 1. BREADCRUMB TRACKING
#         if "visited_nodes" not in state or state["visited_nodes"] is None:
#             state["visited_nodes"] = []
#         state["visited_nodes"].append(self.id)

#         # --- GUARDRAIL SHORT-CIRCUIT ---
#         if state.get("primary_intent") == "irrelevant":
#             print(f"{C_RED}[{self.id.upper()}] Irrelevant query detected. Terminating flow.{C_RESET}")
#             state['next'] = 'END'
#             return state

#         # 2. EVALUATION & REFINEMENT GATE
#         needs_refinement = state.get('needs_refinement', False)
#         current_retries = state.get('refinement_retries', 0)

#         if needs_refinement:
#             if current_retries < self.MAX_REFINEMENT_ATTEMPTS:
#                 current_retries += 1
#                 reason = state.get('refinement_reason', 'Improving report quality.').lower()

#                 print(f"\n{C_ACTION}[{self.id.upper()}] *** REFINEMENT LOOP {current_retries}/{self.MAX_REFINEMENT_ATTEMPTS} ***{C_RESET}")
#                 print(f"{C_YELLOW}Reason: {reason}{C_RESET}")

#                 # --- UPGRADED: MULTI-TOOL REFINEMENT INJECTION ---
#                 active_tools = state.get("active_tools", [])

#                 # Pivot to broader academic databases if the primary ones (Pubmed/Arxiv) failed
#                 tools_to_inject = ["openalex", "semanticscholar", "chemrxiv"]
#                 injected_any = False
#                 for tool in tools_to_inject:
#                     if tool not in active_tools:
#                         active_tools.append(tool)
#                         injected_any = True

#                 if injected_any:
#                     print(f"{C_CYAN}[{self.id.upper()}] Broadening scope: Injected {tools_to_inject} into active tools.{C_RESET}")
#                     state["active_tools"] = active_tools

#                 # Update State for Refinement
#                 state.update({
#                     'is_refining': True,
#                     'refinement_retries': current_retries,
#                     'report_generated': False,
#                     'needs_refinement': False  # Reset for next evaluation pass
#                 })

#                 # DETERMINISTIC ROUTING BASED ON REASON (RESTORED ORIGINAL LOGIC)
#                 # If data is missing, we MUST go back to Planning to integrate the NEW tools
#                 if any(k in reason for k in ["missing", "data", "search", "papers", "pubmed", "arxiv", "found"]):
#                     state['next'] = "planning_agent"
#                 elif any(k in reason for k in ["pdf", "extraction", "parsing", "read"]):
#                     state['next'] = "retrieval_agent"
#                 elif any(k in reason for k in ["relevance", "context", "snippets"]):
#                     state['next'] = "rag_agent"
#                 else:
#                     state['next'] = "synthesis_agent"

#                 return state
#             else:
#                 print(f"{C_RED}[{self.id.upper()}] Max refinements reached. Ending.{C_RESET}")
#                 state['next'] = 'END'
#                 return state

#         # 3. INITIAL SEQUENTIAL FLOW (Phase-based routing)
#         state['next'] = self.select_next_agent(state)

#         print(f"{C_MAGENTA} >> [ROUTER] Supervisor directs flow to: {state['next']}{C_RESET}")
#         return state

#     def select_next_agent(self, state: ResearchState) -> str:
#         """
#         Logic for the first-pass sequential execution.
#         Ensures all nodes are visited in the correct order.
#         """
#         # --- GUARDRAIL EXIT ---
#         if state.get("primary_intent") == "irrelevant":
#             return "END"

#         # --- SUCCESS EXIT ---
#         if state.get("report_generated") and not state.get("needs_refinement"):
#             return "END"

#         # --- INITIAL FLOW ---
#         if not state.get("semantic_query"):
#             return "clean_query_agent"

#         if not state.get("primary_intent"):
#             return "intent_agent"

#         if not state.get("execution_plan"):
#             return "planning_agent"

#         if not state.get("tiered_queries"):
#             return "query_gen_agent"

#         # --- TOOL DATA RETRIEVAL ---
#         # If tools are selected but no data has been fetched yet
#         if not state.get("raw_tool_data"):
#             # This represents the tool execution node (usually a tool executor in the graph)
#             return "retrieval_agent"

#         # --- POST-TOOL PROCESSING ---
#         if not state.get("full_text_chunks"):
#             return "retrieval_agent"

#         # RAG / Filtering Phase
#         if not state.get("filtered_context") or not state.get("rag_complete"):
#             return "rag_agent"

#         # Synthesis Phase
#         if not state.get("report_generated"):
#             return "synthesis_agent"

#         # Evaluation Phase (if report is generated, it goes to evaluation before END)
#         if not state.get("evaluation_complete") and state.get("report_generated"):
#             return "evaluation_agent"

#         return "END"


# from typing import Dict, Any, Optional
# from core.research_state import ResearchState
# from core.utilities import C_ACTION, C_RESET, C_BLUE, C_YELLOW, C_MAGENTA, C_RED

# # ==================================================================================================
# # SECTION 1: SUPERVISOR AGENT (PROCEDURAL ROUTER)
# # ==================================================================================================
# class SupervisorAgent:
#     """
#     Central coordinator of the fully dynamic multi-agent research workflow,
#     managing the Synthesis-Evaluation-Refinement loop and dynamic tool sequencing.

#     The complex routing logic resides in select_next_agent, which is called by
#     the LangGraph router function in research_graph.py.
#     """
#     ALL_TOOL_KEYS = ["pubmed", "arxiv", "openalex", "web", "materials"] # For global initialization
#     MAX_REFINEMENT_ATTEMPTS = 2
#     MAX_TOOL_RETRIES = 2
#     # Retrieval retries are managed implicitly by routing to 'rag_filter' vs 'END'

#     def __init__(self):
#         # Initialize internal counters, though the primary state should be held in ResearchState
#         self.tool_retries = {tool: 0 for tool in self.ALL_TOOL_KEYS}
#         self.retrieval_retries = 0
#         self.refinement_retries = 0
#         print(f"[{self.__class__.__name__.upper()} INIT] Supervisor initialized.{C_RESET}")


#     def execute(self, state: ResearchState) -> ResearchState:
#         """
#         UPDATED: Now populates the 'next' key to prevent routing errors.
#         """
#         is_refining = state.get('is_refining', False)

#         if is_refining:
#             reason = state.get('refinement_reason', 'Addressing previous shortcomings.')
#             print(f"\n{C_ACTION}[{self.__class__.__name__.upper()} ACTION] Entering **REFINEMENT** phase. Reason: {reason[:60]}...{C_RESET}")

#             self.refinement_retries = state.get('refinement_retries', 0)

#             # Clear transient data
#             state['raw_tool_data'] = []
#             state['full_text_chunks'] = []
#             state['references'] = []
#             state['report_generated'] = False
#             state['needs_refinement'] = False
#             state['is_refining'] = True
#         else:
#             print(f"\n{C_ACTION}[{self.__class__.__name__.upper()} ACTION] Starting **INITIAL SEARCH** phase.{C_RESET}")
#             self.tool_retries = {tool: 0 for tool in self.ALL_TOOL_KEYS}
#             self.retrieval_retries = 0
#             self.refinement_retries = 0
#             state['is_refining'] = False

#         # --- THE CRITICAL FIX START ---
#         # 1. Determine the next destination using your logic
#         next_destination = self.select_next_agent(state) or "clean_query_agent"

#         # 2. Update the state's 'next' key so the graph router can see it
#         state['next'] = next_destination

#         # 3. Print the decision for clear console logging
#         print(f"{C_MAGENTA} >> [ROUTER] Supervisor directs flow to: {next_destination}{C_RESET}")
#         # --- THE CRITICAL FIX END ---

#         return state


#     def select_next_agent(self, state: ResearchState) -> Optional[str]:
#         """
#         The core sequencing logic, called by the graph's conditional router.
#         It uses the state to decide the next mandatory step.
#         """
#         print(f"\n{C_MAGENTA}[{self.__class__.__name__.upper()} ROUTER] Assessing State for Next Action...{C_RESET}")

#         # 1. Check for immediate termination (Post-Evaluation Check)
#         if state.get("report_generated", False) and not state.get("needs_refinement", True):
#             print(f"{C_MAGENTA}[ROUTING] State: Evaluation passed. -> END{C_RESET}")
#             return "END"

#         # 2. Refinement Loop Maxed Out Check
#         current_retries = state.get('refinement_retries', 0)
#         if current_retries > self.MAX_REFINEMENT_ATTEMPTS:
#             print(f"{C_RED}[ROUTING FAIL] Refinement loop maxed out ({current_retries}/{self.MAX_REFINEMENT_ATTEMPTS}). Forcing END.{C_RESET}")
#             return "END"

#         # 3. Planning Phase Checks
#         # The planning phase must be rerun during refinement to potentially adjust search queries
#         if not state.get("semantic_query"):
#             print(f"{C_MAGENTA}[ROUTING] State: No semantic query. -> clean_query_agent{C_RESET}")
#             return "clean_query_agent"

#         if not state.get("primary_intent"):
#             print(f"{C_MAGENTA}[ROUTING] State: No intent. -> intent_agent{C_RESET}")
#             return "intent_agent"

#         if not state.get("execution_plan") or not state.get("active_tools"):
#             print(f"{C_MAGENTA}[ROUTING] State: No plan/tools. -> planning_agent{C_RESET}")
#             return "planning_agent"

#         # If planning is complete, move to query generation
#         if not state.get("tiered_queries"):
#             print(f"{C_MAGENTA}[ROUTING] State: Plan complete, needs queries. -> query_gen_agent{C_RESET}")
#             return "query_gen_agent"

#         # 4. Tool/Data Acquisition Check (The most complex part of the router)

#         # We assume that if tiered_queries exist, we must route to the tool execution phase.
#         # This will be handled by the 'route_to_tools' router function attached to query_gen_agent.

#         # If the state has reached this point, it means planning is done, and it should proceed
#         # to the tool sequence starting at the query_gen_agent node (which has the conditional edge).

#         # To handle the refinement loop's restart, we look at the RAG completion flag:
#         if not state.get("rag_complete", False):
#             # If RAG isn't complete, we need to restart the data flow from the start of execution
#             # (which is after query_generation). We route back to trigger the tool sequence.
#             print(f"{C_MAGENTA}[ROUTING] State: Planning done, data acquisition incomplete. -> query_gen_agent (to trigger tool routing){C_RESET}")
#             return "query_gen_agent"


#         # 5. Synthesis/Evaluation Check (Post-RAG)
#         if state.get("rag_complete", False) and not state.get("report_generated", False):
#             print(f"{C_MAGENTA}[ROUTING] State: Context ready, needs final report. -> synthesis_agent{C_RESET}")
#             return "synthesis"

#         # 6. Fallback (Should not be hit if graph is structured correctly)
#         print(f"{C_RED}[ROUTING] FALLBACK: Routing to END.{C_RESET}")
#         return "END"