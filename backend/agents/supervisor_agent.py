import os
import json
from pathlib import Path
from typing import Dict, Any, List

from backend.core.research_state import ResearchState
from backend.core.utilities import C_ACTION, C_RESET, C_BLUE, C_YELLOW, C_MAGENTA, C_RED, C_CYAN, C_GREEN

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

        print(f"\n{C_CYAN}[{self.id.upper()} HUB] Analyzing state for next dispatch...{C_RESET}")

        # 2. Guardrail: Out-of-Scope Handling
        # If IntentAgent flagged 'irrelevant', bypass research and go to Synthesis for rejection
        if state.get("primary_intent") == "irrelevant":
            if state.get("report_generated"):
                state["next"] = "END"
                return state
            print(f"{C_RED}[{self.id.upper()} GUARDRAIL] Irrelevant query detected. Routing to Synthesis Agent.{C_RESET}")
            state["next"] = "synthesis_agent"
            return state

        # 3. Refinement Gate: Handle loops from EvaluationAgent
        if state.get("needs_refinement", False):
            return self._handle_refinement(state)

        # 4. Standard Orchestration: Sequential Flow Checklist
        state["next"] = self.select_next_agent(state)

        print(f"{C_CYAN}[{self.id.upper()} HUB] Next Destination: **{state['next']}**{C_RESET}")
        return state

    def select_next_agent(self, state: ResearchState) -> str:
        """Determines the next logical node based on state completeness."""
        visited = state.get("visited_nodes", [])

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
        if state.get("full_text_chunks") and not state.get("rag_complete"):
            return "rag_agent"

        if not state.get("report_generated"): return "synthesis_agent"

        if state.get("report_generated") and not state.get("needs_refinement"):
            if "evaluation_agent" not in visited:
                return "evaluation_agent"
            else:
                return "END"

        return "END"

    def _handle_refinement(self, state: ResearchState) -> ResearchState:
        """Logic to reset state and pivot strategy based on evaluator feedback."""
        retries = state.get("refinement_retries", 0)

        if retries >= self.MAX_REFINEMENT_ATTEMPTS:
            print(f"{C_RED}[{self.id.upper()}] Max refinement attempts reached. Terminating.{C_RESET}")
            state["next"] = "END"
            return state

        state["refinement_retries"] = retries + 1
        reason = state.get("refinement_reason", "").lower()

        print(f"{C_MAGENTA}[{self.id.upper()} REFINEMENT] Cycle {state['refinement_retries']}: {reason[:100]}...{C_RESET}")

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
            state["next"] = "planning_agent"
        elif is_pdf_issue:
            state["next"] = "retrieval_agent"
        elif is_context_issue:
            state["next"] = "rag_agent"
        else:
            state["next"] = "synthesis_agent"

        return state

# =====================================================
# TESTING BLOCK USING INPUT FILE: level_6_evaluation_output.json
# =====================================================
if __name__ == "__main__":
    print(f"{C_BLUE}==================================================")
    print("      RUNNING SUPERVISOR AGENT TEST SUITE         ")
    print(f"=================================================={C_RESET}")

    project_root = Path(__file__).resolve().parent.parent

    # Look for level_6_evaluation_output.json (with fallback for alternative spelling)
    input_file = project_root / "level_6_evaluation_output.json"
    if not input_file.exists():
        fallback_match = project_root / "level_6_evaluation_ouput.json"
        if fallback_match.exists():
            input_file = fallback_match
        else:
            matches = list(project_root.glob("**/level_6_evaluation_output.json")) + list(project_root.glob("**/level_6_evaluation_ouput.json"))
            if matches:
                input_file = matches[0]

    if not input_file.exists():
        print(f"{C_RED}[TEST ERROR] Input file 'level_6_evaluation_output.json' not found in project root or subdirectories.{C_RESET}")
    else:
        print(f"{C_YELLOW}[TEST SETUP] Loading state from input file: {input_file}{C_RESET}")

        try:
            with open(input_file, "r", encoding="utf-8") as f:
                test_state = json.load(f)

            agent = SupervisorAgent()
            updated_state = agent.execute(test_state)

            print(f"\n{C_GREEN}================ SUPERVISOR RESULTS ================{C_RESET}\n")
            print(f"Next Node Assigned: {updated_state.get('next')}")
            print(f"\n{C_GREEN}===================================================={C_RESET}")

            output_json_file = project_root / "level_7_supervisor_output.json"
            output_md_file = project_root / "level_7_supervisor_output.md"

            with open(output_json_file, "w", encoding="utf-8") as f:
                json.dump(updated_state, f, indent=2, ensure_ascii=False)
            print(f"{C_GREEN}[SAVED] Updated state written to: {output_json_file}{C_RESET}")

            with open(output_md_file, "w", encoding="utf-8") as f:
                f.write(f"# Supervisor Agent Routing Report\n\n")
                f.write(f"- **Next Node**: `{updated_state.get('next')}`\n")
                f.write(f"- **Visited Nodes**: `{updated_state.get('visited_nodes', [])}`\n")
            print(f"{C_GREEN}[SAVED] Supervisor summary written to: {output_md_file}{C_RESET}")

            print(f"{C_BLUE}[TEST SUCCESS] Routing completed successfully.{C_RESET}")

        except Exception as err:
            print(f"{C_RED}[TEST FAILED] Execution raised an exception: {err}{C_RESET}")

# ------------- GPT-5 MULTI-AGENT RESEARCH PIPELINE ----------------
# from typing import Dict, Any, List
# from core.research_state import ResearchState
# from core.utilities import C_ACTION, C_RESET, C_BLUE, C_YELLOW, C_MAGENTA, C_RED, C_CYAN

# # ==================================================================================================
# # SECTION 1: SUPERVISOR AGENT (PROCEDURAL ROUTER)
# # ==================================================================================================
# class SupervisorAgent:
#     """
#     Finalized Hub-and-Spoke Orchestrator.
#     Controls the flow between specialized agents, handles guardrails for
#     irrelevant queries, and manages the iterative refinement loop.
#     """

#     MAX_REFINEMENT_ATTEMPTS = 2

#     # Keyword sets for automated refinement triage
#     DATA_KEYWORDS = ["missing", "data", "search", "papers", "pubmed", "arxiv", "found", "sources", "literature"]
#     PDF_KEYWORDS = ["pdf", "extraction", "parsing", "read"]
#     CONTEXT_KEYWORDS = ["relevance", "context", "snippets", "rag"]

#     def __init__(self, agent_id: str = "supervisor_agent"):
#         self.id = agent_id
#         # Mapping active_tools to their specific graph node IDs
#         self.tool_node_map = {
#             "semanticscholar": "semanticscholar_search",
#             "chemrxiv": "chemrxiv_search",
#             "pubmed": "pubmed_search",
#             "arxiv": "arxiv_search",
#             "openalex": "openalex_search",
#             "materials": "materials_search",
#             "web": "web_search"
#         }

#     def execute(self, state: ResearchState) -> ResearchState:
#         # 1. Breadcrumb Tracking
#         state.setdefault("visited_nodes", []).append(self.id)

#         print(f"\n{C_CYAN}[{self.id.upper()} HUB] Analyzing state for next dispatch...{C_RESET}")

#         # 2. Guardrail: Out-of-Scope Handling
#         # If IntentAgent flagged 'irrelevant', bypass research and go to Synthesis for rejection
#         if state.get("primary_intent") == "irrelevant":
#             if state.get("report_generated"):
#                 state["next"] = "END" # Or however your LangGraph handles termination
#                 return state
#             print(f"{C_RED}[{self.id.upper()} GUARDRAIL] Irrelevant query detected. Routing to Synthesis Agent.{C_RESET}")
#             state["next"] = "synthesis_agent"
#             return state

#         # 3. Refinement Gate: Handle loops from EvaluationAgent
#         if state.get("needs_refinement", False):
#             return self._handle_refinement(state)

#         # 4. Standard Orchestration: Sequential Flow Checklist
#         state["next"] = self.select_next_agent(state)

#         print(f"{C_CYAN}[{self.id.upper()} HUB] Next Destination: **{state['next']}**{C_RESET}")
#         return state

#     def select_next_agent(self, state: ResearchState) -> str:
#         """Determines the next logical node based on state completeness."""
#         visited = state.get("visited_nodes", [])

#         # --- A. Setup Phase ---
#         if not state.get("semantic_query"): return "clean_query_agent"
#         if not state.get("primary_intent"): return "intent_agent"
#         if not state.get("execution_plan"): return "planning_agent"
#         if not state.get("tiered_queries"): return "query_gen_agent"

#         # --- B. Tool Execution Phase (The Star Spokes) ---
#         active_tools = state.get("active_tools", [])
#         for tool in active_tools:
#             node_name = self.tool_node_map.get(tool)
#             # Visit tool node only if it's in the plan AND hasn't been visited yet
#             if node_name and node_name not in visited:
#                 # Extra safety: Ensure QueryGen actually produced strings for this tool
#                 if tool in state.get("tiered_queries", {}):
#                     return node_name

#         # --- C. Processing & Finalization Phase ---
#         # If tools are done, but we haven't processed full texts yet
#         if state.get("raw_tool_data") and not state.get("full_text_chunks"):
#             return "retrieval_agent"
#         # Only after retrieval is done should we go to RAG
#         if state.get("full_text_chunks") and not state.get("rag_complete"):
#             return "rag_agent"

#         # If synthesis hasn't run yet, or we just finished RAG
#         if not state.get("report_generated"): return "synthesis_agent"

#         # Final check: If report exists, send to Evaluation
#         if state.get("report_generated") and not state.get("needs_refinement"):
#             # Check if we already evaluated this specific version
#             # (Evaluation usually happens once per synthesis)
#             if "evaluation_agent" not in visited:
#                 return "evaluation_agent"
#             else:
#                 return "END"

#         return "END"

#     def _handle_refinement(self, state: ResearchState) -> ResearchState:
#         """Logic to reset state and pivot strategy based on evaluator feedback."""
#         retries = state.get("refinement_retries", 0)

#         if retries >= self.MAX_REFINEMENT_ATTEMPTS:
#             print(f"{C_RED}[{self.id.upper()}] Max refinement attempts reached. Terminating.{C_RESET}")
#             state["next"] = "END"
#             return state

#         state["refinement_retries"] = retries + 1
#         reason = state.get("refinement_reason", "").lower()

#         print(f"{C_MAGENTA}[{self.id.upper()} REFINEMENT] Cycle {state['refinement_retries']}: {reason[:100]}...{C_RESET}")

#         # Determine which agent needs to re-run based on feedback keywords
#         is_data_issue = any(k in reason for k in self.DATA_KEYWORDS)
#         is_pdf_issue = any(k in reason for k in self.PDF_KEYWORDS)
#         is_context_issue = any(k in reason for k in self.CONTEXT_KEYWORDS)

#         # Reset core flags to allow re-execution
#         state.update({
#             "is_refining": True,
#             "needs_refinement": False,
#             "report_generated": False,
#             "rag_complete": False,
#             "filtered_context": ""
#         })

#         # Logic for where to jump back to
#         if is_data_issue:
#             # Inject broad tools if we are short on data
#             active_tools = state.get("active_tools", [])
#             for t in ["openalex", "semanticscholar"]:
#                 if t not in active_tools: active_tools.append(t)
#             state["active_tools"] = active_tools
#             state["next"] = "planning_agent" # Recalculate strategy
#         elif is_pdf_issue:
#             state["next"] = "retrieval_agent"
#         elif is_context_issue:
#             state["next"] = "rag_agent"
#         else:
#             state["next"] = "synthesis_agent" # Re-draft with better instructions

#         return state
