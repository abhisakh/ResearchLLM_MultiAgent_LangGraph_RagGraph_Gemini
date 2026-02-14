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


class ResearchGraph:
    def __init__(self, vector_db: Optional[VectorDBWrapper] = None):
        self.vector_db = vector_db if vector_db is not None else VectorDBWrapper()
        print(f"{C_CYAN}>> [INIT] Initializing Central Hub Graph with VectorDB.{C_RESET}")
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
            "rag_agent": RAGAgent(vector_db=self.vector_db),
            "synthesis_agent": SynthesisAgent(),
            "evaluation_agent": EvaluationAgent(),
        }

        # 2. Add Nodes
        for name, agent in agents.items():
            workflow.add_node(name, agent.execute)

        # 3. SET ENTRY POINT: The Hub always starts the process
        workflow.set_entry_point("supervisor_agent")

        # =================================================================
        # 4. THE CENTRAL HUB DYNAMICS (All roads lead to Supervisor)
        # =================================================================

        # Every processing node returns to the Supervisor for state validation
        workflow.add_edge("clean_query_agent", "supervisor_agent")
        workflow.add_edge("intent_agent", "supervisor_agent")
        workflow.add_edge("planning_agent", "supervisor_agent")
        workflow.add_edge("query_gen_agent", "supervisor_agent")
        workflow.add_edge("retrieval_agent", "supervisor_agent")
        workflow.add_edge("rag_agent", "supervisor_agent")
        workflow.add_edge("synthesis_agent", "supervisor_agent")
        workflow.add_edge("evaluation_agent", "supervisor_agent")

        # Tool nodes also report back to the Hub (This allows for tool-level error handling)
        workflow.add_edge("semanticscholar_search", "supervisor_agent")
        workflow.add_edge("chemrxiv_search", "supervisor_agent")
        workflow.add_edge("pubmed_search", "supervisor_agent")
        workflow.add_edge("arxiv_search", "supervisor_agent")
        workflow.add_edge("openalex_search", "supervisor_agent")
        workflow.add_edge("materials_search", "supervisor_agent")
        workflow.add_edge("web_search", "supervisor_agent")

        # =================================================================
        # 5. THE SUPERVISOR ROUTING (Decision Matrix)
        # =================================================================

        # We define a mapping for all possible transitions the Supervisor might command
        routing_map = {
            "clean_query_agent": "clean_query_agent",
            "intent_agent": "intent_agent",
            "planning_agent": "planning_agent",
            "query_gen_agent": "query_gen_agent",
            "semanticscholar_search": "semanticscholar_search",
            "chemrxiv_search": "chemrxiv_search",
            "pubmed_search": "pubmed_search",
            "arxiv_search": "arxiv_search",
            "openalex_search": "openalex_search",
            "materials_search": "materials_search",
            "web_search": "web_search",
            "retrieval_agent": "retrieval_agent",
            "rag_agent": "rag_agent",
            "synthesis_agent": "synthesis_agent",
            "evaluation_agent": "evaluation_agent",
            "END": END
        }

        workflow.add_conditional_edges(
            "supervisor_agent",
            lambda state: state.get("next", "END"),
            routing_map
        )

        return workflow.compile()
