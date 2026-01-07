from typing import Dict, Any, Optional
from core.research_state import ResearchState
from core.utilities import C_ACTION, C_RESET, C_BLUE, C_YELLOW, C_MAGENTA, C_RED

# ==================================================================================================
# SECTION 1: SUPERVISOR AGENT (PROCEDURAL ROUTER)
# ==================================================================================================
class SupervisorAgent:
    """
    Central coordinator of the research workflow.
    Fixed to handle the Refinement Loop and avoid the 'END' Fallback.
    """
    ALL_TOOL_KEYS = ["pubmed", "arxiv", "openalex", "web", "materials"]
    MAX_REFINEMENT_ATTEMPTS = 2

    def __init__(self, agent_id: str = "supervisor_agent"):
        self.id = agent_id
        print(f"[{self.id.upper()} INIT] Supervisor initialized.{C_RESET}")

    def execute(self, state: ResearchState) -> ResearchState:
        """
        Logic to prepare the state for the next node.
        """
        # Detect if we are entering a refinement loop from Evaluation
        needs_refinement = state.get('needs_refinement', False)

        if needs_refinement:
            current_retries = state.get('refinement_retries', 0) + 1
            reason = state.get('refinement_reason', 'Addressing feedback.')

            print(f"\n{C_ACTION}[{self.id.upper()} ACTION] *** REFINEMENT LOOP {current_retries}/{self.MAX_REFINEMENT_ATTEMPTS} ***{C_RESET}")
            print(f"{C_YELLOW}Reason: {reason}{C_RESET}")

            # RESET STATE FOR REFRESHED SEARCH/SYNTHESIS
            state.update({
                'is_refining': True,
                'refinement_retries': current_retries,
                'report_generated': False,
                'rag_complete': False, # Force re-filtering
                'raw_tool_data': [],    # Clear to avoid duplicate/old data
                'references': [],
                'needs_refinement': False # Reset flag so we don't loop forever
            })
        else:
            # Standard path
            if not state.get('is_refining'):
                print(f"\n{C_ACTION}[{self.id.upper()} ACTION] Starting **INITIAL SEARCH** phase.{C_RESET}")
                state['refinement_retries'] = 0

        # Determine the next destination
        next_destination = self.select_next_agent(state)
        state['next'] = next_destination

        print(f"{C_MAGENTA} >> [ROUTER] Supervisor directs flow to: {next_destination}{C_RESET}")
        return state

    def select_next_agent(self, state: ResearchState) -> str:
        """
        Pure routing logic. Ensure strings match your graph.add_node() keys.
        """
        # 1. SUCCESS EXIT
        if state.get("report_generated") and not state.get("needs_refinement"):
            return "END"

        # 2. MAX RETRY EXIT
        if state.get('refinement_retries', 0) > self.MAX_REFINEMENT_ATTEMPTS:
            print(f"{C_RED}[ROUTING] Max refinement limit reached. Terminating.{C_RESET}")
            return "END"

        # 3. REFINEMENT OVERRIDE
        # If we are refining, we usually want to re-run queries or re-synthesize.
        if state.get("is_refining"):
            # If we already have the data but synthesis was poor, go to synthesis.
            # But usually, refinement means we need BETTER data.
            if not state.get("raw_tool_data"):
                return "query_gen_agent"
            return "synthesis_agent"

        # 4. INITIAL SEQUENTIAL FLOW
        if not state.get("semantic_query"):
            return "clean_query_agent"

        if not state.get("primary_intent"):
            return "intent_agent"

        if not state.get("execution_plan"):
            return "planning_agent"

        if not state.get("tiered_queries"):
            return "query_gen_agent"

        # 5. DATA ACQUISITION TO SYNTHESIS BRIDGE
        if not state.get("rag_complete"):
            # This represents the transition after tools have run
            return "synthesis_agent"

        if not state.get("report_generated"):
            return "synthesis_agent"

        # 6. FINAL FALLBACK
        return "END"


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