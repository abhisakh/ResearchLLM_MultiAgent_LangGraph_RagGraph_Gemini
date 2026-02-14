from langgraph.graph import StateGraph, END
from typing import Optional

from core.research_state import ResearchState
from core.vector_db import VectorDBWrapper
from core.utilities import C_CYAN, C_RESET, C_MAGENTA

# --- 1. Import Agents ---
from agents.procedural_agents import CleanQueryAgent
from agents.planning_agents import IntentAgent, PlanningAgent, QueryGenerationAgent
from agents.tool_agents import PubMedAgent, ArxivAgent, OpenAlexAgent, MaterialsAgent, WebAgent, SemanticScholarAgent, ChemRxivAgent
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

    if "semanticscholar" in active_tools:
        return "semanticscholar_search"
    if "chemrxiv" in active_tools:
        return "chemrxiv_search"
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
    return "retrieval_agent"

def route_next_tool(current_tool: str, state: ResearchState) -> str:
    """Controls sequential tool execution"""
    active_tools = state.get("active_tools", [])
    tool_map = {"semanticscholar": "semanticscholar_search",
        "chemrxiv": "chemrxiv_search",
        "pubmed": "pubmed_search", "arxiv": "arxiv_search",
        "openalex": "openalex_search", "materials": "materials_search",
        "web": "web_search"
    }
    tool_keys = list(tool_map.keys())

    try:
        current_index = tool_keys.index(current_tool)
    except ValueError:
        return "retrieval_agent"

    for next_tool_key in tool_keys[current_index + 1:]:
        if next_tool_key in active_tools:
            return tool_map[next_tool_key]

    return "retrieval_agent"

def route_after_evaluation(state: ResearchState) -> str:
    """Conditional routing based on the Evaluation Agent's decision."""
    if state.get("needs_refinement", False):
        print(f"{C_MAGENTA}>> [ROUTER] Evaluation: **REFINE** needed. Looping back to SUPERVISOR.{C_RESET}")
        return "supervisor_agent" # Loop back to supervisor to manage state transition
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

        # 1. Initialize Agents
        agents = {
            "supervisor_agent": SupervisorAgent(),
            "clean_query_agent": CleanQueryAgent(),
            "intent_agent": IntentAgent(),
            "planning_agent": PlanningAgent(),
            "query_gen_agent": QueryGenerationAgent(),
            "semanticscholar_search": SemanticScholarAgent(),
            "chemrxiv_search": ChemRxivAgent(),
            "pubmed_search": PubMedAgent(),
            "arxiv_search": ArxivAgent(),
            "openalex_search": OpenAlexAgent(),
            "materials_search": MaterialsAgent(),
            "web_search": WebAgent(),
            "retrieval_agent": RetrievalAgent(),
            "rag_agent": RAGAgent(vector_db=self.vector_db), # Verified name
            "synthesis_agent": SynthesisAgent(),
            "evaluation_agent": EvaluationAgent(),
        }

        # 2. Add Nodes
        for name, agent in agents.items():
            workflow.add_node(name, agent.execute)

        # 3. Entry Point
        workflow.set_entry_point("supervisor_agent")

        # 4. Supervisor Routing (UPDATED: Added retrieval_agent and intent_agent)
        workflow.add_conditional_edges(
            "supervisor_agent",
            lambda state: state.get("next", "clean_query_agent"),
            {
                "clean_query_agent": "clean_query_agent",
                "intent_agent": "intent_agent",
                "planning_agent": "planning_agent",
                "query_gen_agent": "query_gen_agent",
                "retrieval_agent": "retrieval_agent", # Necessary for PDF-refinement
                "rag_agent": "rag_agent",
                "synthesis_agent": "synthesis_agent",
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
                "semanticscholar_search": "semanticscholar_search",
                "chemrxiv_search": "chemrxiv_search",
                "pubmed_search": "pubmed_search",
                "arxiv_search": "arxiv_search",
                "openalex_search": "openalex_search",
                "materials_search": "materials_search",
                "web_search": "web_search",
                "retrieval_agent": "retrieval_agent",
            }
        )

        # 7. Tool-to-Tool Sequence (Sequential chaining)
        for tool in ["semanticscholar", "chemrxiv", "pubmed", "arxiv", "openalex", "materials", "web"]:
            workflow.add_conditional_edges(
                f"{tool}_search",
                lambda state, k=tool: route_next_tool(k, state),
                {
                    "semanticscholar_search": "semanticscholar_search",
                    "chemrxiv_search": "chemrxiv_search",
                    "pubmed_search": "pubmed_search",
                    "arxiv_search": "arxiv_search",
                    "openalex_search": "openalex_search",
                    "materials_search": "materials_search",
                    "web_search": "web_search",
                    "retrieval_agent": "retrieval_agent",
                }
            )

        # 8. Data Pipeline
        workflow.add_edge("retrieval_agent", "rag_agent")
        workflow.add_edge("rag_agent", "synthesis_agent")
        workflow.add_edge("synthesis_agent", "evaluation_agent")

        # 9. Loop-back to Supervisor (Refinement)
        workflow.add_conditional_edges(
            "evaluation_agent",
            route_after_evaluation,
            {
                "supervisor_agent": "supervisor_agent",
                END: END
            }
        )

        return workflow.compile()
