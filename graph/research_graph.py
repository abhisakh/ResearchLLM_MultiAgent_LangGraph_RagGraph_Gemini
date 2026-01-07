from langgraph.graph import StateGraph, END
from typing import Optional

from core.research_state import ResearchState
from core.vector_db import VectorDBWrapper
from core.utilities import C_CYAN, C_RESET, C_MAGENTA

# --- 1. Import Agents ---
from agents.procedural_agents import CleanQueryAgent
from agents.planning_agents import IntentAgent, PlanningAgent, QueryGenerationAgent
from agents.tool_agents import PubMedAgent, ArxivAgent, OpenAlexAgent, MaterialsAgent, WebAgent
from agents.rag_agents import RetrievalAgent, RAGAgent
from agents.synthesis_agent import SynthesisAgent
from agents.evaluation_agent import EvaluationAgent
from agents.supervisor_agent import SupervisorAgent # <-- NEW

# --- 2. Define Conditional Edges (Routers) ---
# (Routings defined previously remain the same)

def route_from_supervisor(state: ResearchState) -> str:
    """Determines where execution begins or resumes."""
    next_node = state.get('next', 'clean_query_agent')
    print(f"\n{C_MAGENTA}>> [ROUTER] Supervisor directs flow to: {next_node}{C_RESET}")
    return next_node

def route_to_tools(state: ResearchState) -> str:
    """Determines which tool to start with after query generation from QueryGenerationAgent"""
    active_tools = state.get("active_tools", [])
    print(f"\n{C_MAGENTA}>> [ROUTER] Routing to Tool Execution based on selection: {active_tools}{C_RESET}")

    if "pubmed" in active_tools:
        return "pubmed_search"
    if "arxiv" in active_tools:
        return "arxiv_search"
    if "openalex" in active_tools:
        return "openalex_search"
    if "materials" in active_tools:
        return "materials_search"
    if "web" in active_tools:
        return "web_search"
    return "retrieve_data"

def route_next_tool(current_tool: str, state: ResearchState) -> str:
    """Controls sequential tool execution"""
    active_tools = state.get("active_tools", [])
    tool_map = {
        "pubmed": "pubmed_search", "arxiv": "arxiv_search",
        "openalex": "openalex_search", "materials": "materials_search",
        "web": "web_search"
    }
    tool_keys = list(tool_map.keys())

    try:
        current_index = tool_keys.index(current_tool)
    except ValueError:
        return "retrieve_data"

    for next_tool_key in tool_keys[current_index + 1:]:
        if next_tool_key in active_tools:
            return tool_map[next_tool_key]

    return "retrieve_data"

def route_after_evaluation(state: ResearchState) -> str:
    """Conditional routing based on the Evaluation Agent's decision."""
    if state.get("needs_refinement", False):
        print(f"{C_MAGENTA}>> [ROUTER] Evaluation: **REFINE** needed. Looping back to SUPERVISOR.{C_RESET}")
        return "supervisor" # Loop back to supervisor to manage state transition
    else:
        print(f"{C_MAGENTA}>> [ROUTER] Evaluation: **ACCEPTABLE**. TERMINATING.{C_RESET}")
        return END

# --- 3. Graph Builder Class ---

class ResearchGraph:
    def __init__(self, vector_db: Optional[VectorDBWrapper] = None):
        self.vector_db = vector_db if vector_db is not None else VectorDBWrapper()
        print(f"{C_CYAN}>> [INIT] Initializing Graph with VectorDB.{C_RESET}")
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(ResearchState)

        # 1. Initialize Agents (STANDARD NAMES)
        agents = {
            "supervisor": SupervisorAgent(),
            "clean_query_agent": CleanQueryAgent(),
            "intent_agent": IntentAgent(),
            "planning_agent": PlanningAgent(),
            "query_gen_agent": QueryGenerationAgent(),
            "pubmed_search": PubMedAgent(),
            "arxiv_search": ArxivAgent(),
            "openalex_search": OpenAlexAgent(),
            "materials_search": MaterialsAgent(),
            "web_search": WebAgent(),
            "retrieve_data": RetrievalAgent(),
            "rag_filter": RAGAgent(vector_db=self.vector_db),
            "synthesis_agent": SynthesisAgent(), # Standardized Name
            "evaluation_agent": EvaluationAgent(), # Standardized Name
        }

        # 2. Add Nodes
        for name, agent in agents.items():
            workflow.add_node(name, agent.execute)

        # 3. Entry Point
        workflow.set_entry_point("supervisor")

        # 4. Supervisor Routing (FIXED: Expanded mapping to prevent 'END' errors)
        workflow.add_conditional_edges(
            "supervisor",
            lambda state: state.get("next", "clean_query_agent"),
            {
                "clean_query_agent": "clean_query_agent",
                "intent_agent": "intent_agent",
                "planning_agent": "planning_agent",
                "query_gen_agent": "query_gen_agent",
                "synthesis_agent": "synthesis_agent",
                "rag_filter": "rag_filter",
                "END": END
            }
        )

        # 5. Planning Edges
        workflow.add_edge("clean_query_agent", "intent_agent")
        workflow.add_edge("intent_agent", "planning_agent")
        workflow.add_edge("planning_agent", "query_gen_agent")

        # 6. Tool Routing Logic
        workflow.add_conditional_edges(
            "query_gen_agent",
            route_to_tools,
            {
                "pubmed_search": "pubmed_search",
                "arxiv_search": "arxiv_search",
                "openalex_search": "openalex_search",
                "materials_search": "materials_search",
                "web_search": "web_search",
                "retrieve_data": "retrieve_data",
            }
        )

        # 7. Tool-to-Tool Sequence
        for tool in ["pubmed", "arxiv", "openalex", "materials", "web"]:
            workflow.add_conditional_edges(
                f"{tool}_search",
                lambda state, k=tool: route_next_tool(k, state),
                {
                    "pubmed_search": "pubmed_search",
                    "arxiv_search": "arxiv_search",
                    "openalex_search": "openalex_search",
                    "materials_search": "materials_search",
                    "web_search": "web_search",
                    "retrieve_data": "retrieve_data",
                }
            )

        # 8. Data Pipeline
        workflow.add_edge("retrieve_data", "rag_filter")
        workflow.add_edge("rag_filter", "synthesis_agent")
        workflow.add_edge("synthesis_agent", "evaluation_agent")

        # 9. Loop-back to Supervisor (Refinement)
        workflow.add_conditional_edges(
            "evaluation_agent",
            route_after_evaluation,
            {
                "supervisor": "supervisor",
                END: END
            }
        )

        return workflow.compile()
