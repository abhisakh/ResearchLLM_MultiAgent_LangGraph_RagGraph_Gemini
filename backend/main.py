import time
from datetime import datetime
from typing import Optional
from core.research_state import ResearchState
from core.vector_db import VectorDBWrapper
from graph.research_graph import ResearchGraph
from core.utilities import (
    C_CYAN, C_RESET, C_ACTION, C_GREEN,
    C_RED, C_MAGENTA, C_YELLOW
)

# ==================================================================================================
# SECTION: UTILITIES & INITIALIZATION
# ==================================================================================================

def initialize_research_session() -> ResearchGraph:
    """Initializes the VectorDB and the LangGraph workflow."""
    print(f"{C_CYAN}*** RESEARCH AGENT SYSTEM INITIALIZATION ***{C_RESET}")

    # Initialize the database and ensure it's clean for a new session
    vector_db = VectorDBWrapper()
    vector_db.reset_db()

    # Build the graph, passing the DB instance
    research_graph = ResearchGraph(vector_db=vector_db)

    print(f"{C_GREEN}*** INITIALIZATION COMPLETE ***{C_RESET}")
    return research_graph

def print_mermaid_code(compiled_graph, final_state: Optional[ResearchState] = None):
    """
    Prints Mermaid code with 'Visited Path' highlighting.
    Allows for visual debugging of loops and agent transitions.
    """
    try:
        # Generate the base blueprint of the graph
        mermaid_code = compiled_graph.get_graph().draw_mermaid()

        # --- ACTIVE PATH HIGHLIGHTING ---
        # If a final state is provided, we highlight the nodes actually visited in green
        if final_state and final_state.get("visited_nodes"):
            visited = list(set(final_state["visited_nodes"]))
            # CSS styling for Mermaid nodes
            style_lines = [
                f"    style {node} fill:#2ecc71,stroke:#27ae60,stroke-width:2px,color:#fff"
                for node in visited if node != "END"
            ]
            mermaid_code += "\n" + "\n".join(style_lines)

        print("\n" + "="*60)
        print("📊 PROFESSIONAL MERMAID GRAPH (Paste into mermaid.live)")
        print("="*60 + "\n")
        print(mermaid_code)
        print("\n" + "="*60)
    except Exception as e:
        print(f"{C_RED}Error generating Mermaid code: {e}{C_RESET}")

# ==================================================================================================
# SECTION: CORE EXECUTION
# ==================================================================================================

def run_research_query(research_graph: ResearchGraph, query: str):
    """Executes the research query through the LangGraph state machine."""

    # 1. Initialize the starting state (The Graph's Memory)
    initial_state: ResearchState = {
        "user_query": query,
        "semantic_query": "",
        "primary_intent": "",
        "reasoning": "",
        "execution_plan": [],
        "material_elements": [],
        "system_constraints": [],
        "api_search_term": "",
        "tiered_queries": {},
        "active_tools": [],
        "raw_tool_data": [],
        "full_text_chunks": [],
        "rag_complete": False,
        "filtered_context": "",
        "references": [],
        "final_report": "",
        "report_generated": False,
        "needs_refinement": False,
        "refinement_reason": "",
        "is_refining": False,
        "refinement_retries": 0,
        "next": "",
        "visited_nodes": [],
    }

    print(f"\n{C_ACTION}--- STARTING RESEARCH FOR QUERY: '{query}' ---{C_RESET}")
    start_time = time.time()
    final_state = initial_state

    # 2. RUN THE GRAPH STREAM
    # The .stream() method handles the iterative logic and the supervisor's routing.
    # We use a recursion_limit to prevent infinite loops.
    try:
        # recursion_limit=50 allows for multiple refinement cycles
        for output in research_graph.graph.stream(initial_state, config={"recursion_limit": 50}):
            for node_name, node_state in output.items():
                # Update our local state tracker with updates from the agent nodes
                final_state.update(node_state)

                if node_name != "__end__":
                    # Visual feedback for the refinement loop
                    if node_name == "supervisor_agent" and final_state.get('is_refining'):
                         print(f"{C_YELLOW}>> [SYSTEM] Supervisor triggering refinement cycle...{C_RESET}")

                    print(f"{C_MAGENTA}>> [NODE] Executed {node_name} successfully.{C_RESET}")

    except Exception as e:
        print(f"{C_RED}>> [CRITICAL ERROR] Graph execution failed: {e}{C_RESET}")

    duration = time.time() - start_time

    # 3. OUTPUT GENERATION
    is_refined = final_state.get('refinement_retries', 0) > 0

    print(f"\n{C_GREEN}==================================================================={C_RESET}")
    print(f"{C_GREEN}*** FINAL RESEARCH REPORT ({'REFINED' if is_refined else 'INITIAL'}) ***{C_RESET}")
    print(f"{C_GREEN}Execution Time: {duration:.2f} seconds{C_RESET}")
    print(f"{C_GREEN}==================================================================={C_RESET}")

    # Display the report or the error message
    report = final_state.get("final_report")
    if report:
        print(report)
    else:
        print(f"{C_RED}ERROR: The system failed to generate a final report.{C_RESET}")

    print(f"\n{C_GREEN}==================================================================={C_RESET}")

    # 4. VISUAL AUDIT TRAIL
    # Crucially pass the final_state here to enable green highlighting of the path taken
    print_mermaid_code(research_graph.graph, final_state=final_state)

    return final_state

# ==================================================================================================
# SECTION: ENTRY POINT
# ==================================================================================================

if __name__ == "__main__":
    # Initialize the graph
    research_system = initialize_research_session()

    # Example Query
    user_input = "Identify the thermal stability of CsPbI3 perovskites and any recent doping strategies."

    # Execute
    final_results = run_research_query(research_system, user_input)