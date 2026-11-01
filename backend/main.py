import logging
import time
from typing import Optional
from backend.core.research_state import ResearchState, new_research_state
from backend.core.vector_db import VectorDBWrapper
from backend.graph.research_graph import ResearchGraph

logger = logging.getLogger(__name__)

# ==================================================================================================
# SECTION: UTILITIES & INITIALIZATION
# ==================================================================================================

def initialize_research_session() -> ResearchGraph:
    """Initializes the VectorDB and the LangGraph workflow."""
    logger.info("*** RESEARCH AGENT SYSTEM INITIALIZATION ***")

    # Initialize the database and ensure it's clean for a new session
    vector_db = VectorDBWrapper()
    vector_db.reset_db()

    # Build the graph, passing the DB instance
    research_graph = ResearchGraph(vector_db=vector_db)

    logger.info("*** INITIALIZATION COMPLETE ***")
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
        logger.error(f"Error generating Mermaid code: {e}")

# ==================================================================================================
# SECTION: CORE EXECUTION
# ==================================================================================================

def run_research_query(research_graph: ResearchGraph, query: str):
    """Executes the research query through the LangGraph state machine."""

    # 1. Initialize the starting state (The Graph's Memory)
    initial_state: ResearchState = new_research_state(query)

    logger.info(f"--- STARTING RESEARCH FOR QUERY: '{query}' ---")
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
                        logger.info(">> [SYSTEM] Supervisor triggering refinement cycle...")

                    logger.info(f">> [NODE] Executed {node_name} successfully.")

    except Exception as e:
        logger.error(f">> [CRITICAL ERROR] Graph execution failed: {e}")

    duration = time.time() - start_time

    # 3. OUTPUT GENERATION (this is the program's result, so it is printed, not logged)
    is_refined = final_state.get('refinement_retries', 0) > 0

    print("\n" + "=" * 67)
    print(f"*** FINAL RESEARCH REPORT ({'REFINED' if is_refined else 'INITIAL'}) ***")
    print(f"Execution Time: {duration:.2f} seconds")
    print("=" * 67)

    # Display the report or the error message
    report = final_state.get("final_report")
    if report:
        print(report)
    else:
        logger.error("The system failed to generate a final report.")

    print("\n" + "=" * 67)

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
    user_input = "What are the latest advancements in Machine learning?"

    # Execute
    final_results = run_research_query(research_system, user_input)
