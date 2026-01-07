import time
from datetime import datetime
from core.research_state import ResearchState
from core.vector_db import VectorDBWrapper
from graph.research_graph import ResearchGraph
from core.utilities import C_CYAN, C_RESET, C_ACTION, C_GREEN, C_RED, C_MAGENTA

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

def print_mermaid_code(compiled_graph):
    """Prints the Mermaid string for copy-pasting into mermaid.live"""
    try:
        # Get the mermaid syntax string
        mermaid_code = compiled_graph.get_graph().draw_mermaid()
        print("\n" + "="*50)
        print("📊 MERMAID GRAPH CODE (Copy the lines below):")
        print("="*50 + "\n")
        print(mermaid_code)
        print("\n" + "="*50)
    except Exception as e:
        print(f"Error generating Mermaid code: {e}")

def run_research_query(research_graph: ResearchGraph, query: str):
    """Executes the research query through the LangGraph state machine."""

    # 1. Initialize the starting state
    initial_state: ResearchState = {
        "user_query": query,
        "semantic_query": "",
        "primary_intent": "",
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
        "next": "",
    }

    print(f"\n{C_ACTION}--- STARTING RESEARCH FOR QUERY: '{query}' ---{C_RESET}")
    start_time = time.time()
    start_time_clock = datetime.fromtimestamp(start_time).strftime('%H:%M:%S')

    # 2. Run the graph with iteration for the refinement loop
    # We will limit the loop to 2 passes (initial + 1 refinement)
    max_iterations = 2
    current_iteration = 0
    final_state: ResearchState = initial_state.copy()

    # The graph.stream() handles the flow based on conditional edges
    for s in research_graph.graph.stream(initial_state):
        if not s: continue # Skip empty updates

        # Check for iteration completion and update final state
        if 'evaluation' in s:
             final_state = s['evaluation']
             current_iteration += 1
             needs_refinement = final_state.get('needs_refinement', False)

             # If refinement is needed AND we haven't hit the max limit
             if needs_refinement and current_iteration < max_iterations:
                 print(f"{C_RED}*** REFINEMENT LOOP {current_iteration}/{max_iterations-1} TRIGGERED ***{C_RESET}")
                 # Prepare the state for the refinement run
                 initial_state = final_state.copy()
                 # Reset transient data (we want new data from RAG agent)
                 initial_state['raw_tool_data'] = []
                 initial_state['full_text_chunks'] = []
                 initial_state['references'] = []
                 initial_state['report_generated'] = False
                 initial_state['needs_refinement'] = False
                 initial_state['is_refining'] = True # Set flag for RAG/Synthesis agent prompts
                 initial_state['next'] = 'rag_filter' # Force start at rag_filter (router handles this path)
             elif needs_refinement and current_iteration >= max_iterations:
                 print(f"{C_RED}*** MAX REFINEMENT ITERATIONS REACHED. TERMINATING ***{C_RESET}")
                 break # Exit the loop and stop stream
             elif not needs_refinement:
                 break # Report accepted, exit loop

        # Print out the current node execution
        for key, value in s.items():
            if key != "__end__":
                print(f"{C_MAGENTA}>> [NODE] Executed {key} successfully.{C_RESET}")

    end_time = time.time()
    end_time_clock = datetime.fromtimestamp(end_time).strftime('%H:%M:%S')
    duration = end_time - start_time

    # 3. Output the Final Report
    print(f"\n{C_GREEN}==================================================================={C_RESET}")
    print(f"{C_GREEN}*** FINAL RESEARCH REPORT ({'REFINED' if current_iteration > 1 else 'INITIAL'}) ***{C_RESET}")
    print(f"""{C_GREEN}Total Time: {duration:.2f} seconds,
          start_time: {start_time_clock}, end_time: {end_time_clock}{C_RESET}""")
    print(f"{C_GREEN}==================================================================={C_RESET}")
    print(final_state.get("final_report", "ERROR: Report not generated."))
    print(f"\n{C_GREEN}==================================================================={C_RESET}")
    print_mermaid_code(research_graph.graph)
    return final_state


