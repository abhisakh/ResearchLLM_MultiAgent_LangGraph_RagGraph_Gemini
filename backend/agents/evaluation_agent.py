import os
import json
from pydantic import BaseModel, Field
from typing import Dict, Any

# --- Synchronized Imports from Synthesis logic ---
from core.research_state import ResearchState
from core.utilities import (
    C_ACTION, C_RESET, C_RED, C_BLUE, C_MAGENTA,
    LLM_MODEL, client  # Use the working, authenticated global client
)

# ==================================================================================================
# SECTION 9.A.: EVALUATION AGENT
# ==================================================================================================

class EvaluationSchema(BaseModel):
    """Schema for Evaluation Agent output to ensure reliable boolean routing."""
    needs_refinement: bool = Field(description="TRUE if the report fails to address the plan. Otherwise FALSE.")
    refinement_reason: str = Field(description="Specific reason for refinement or 'Report is satisfactory'.")

class EvaluationAgent:
    """
    Evaluates SynthesisAgent's report using the global OpenAI client.
    Synchronized with SynthesisAgent logic to ensure 100% authentication success.
    """

    def __init__(self, agent_id: str = "evaluation_agent", model: str = LLM_MODEL):
        self.id = agent_id
        self.model = model

    def execute(self, state: ResearchState) -> ResearchState:
        # --- MODIFICATION 1: BREADCRUMB TRACKING ---
        if "visited_nodes" not in state or state["visited_nodes"] is None:
            state["visited_nodes"] = []
        state["visited_nodes"].append(self.id)

        print(f"\n{C_ACTION}[{self.id.upper()} START] Evaluating report...{C_RESET}")

        # 1. Verification: Ensure client is available
        if client is None:
            print(f"{C_RED}[{self.id.upper()} ERROR] Global client is None.{C_RESET}")
            state.update({'needs_refinement': False, 'report_generated': True})
            return state

        user_query = state.get('user_query', '')
        execution_plan = state.get('execution_plan', [])
        final_report = state.get('final_report', '')

        # --- Handle empty report ---
        if not final_report or len(final_report) < 200:
            print(f"{C_RED}[{self.id.upper()} ERROR] Final report is insufficient. Forcing refinement.{C_RESET}")
            state.update({
                'needs_refinement': True,
                'refinement_reason': "Synthesis produced insufficient or empty content.",
                'report_generated': True
            })
            return state

        # --- Construct evaluation prompt ---
        eval_prompt = f"""
        Analyze if the 'Final Report' successfully addresses the 'Execution Plan'.

        USER INTENT: {user_query}
        PLAN: {execution_plan}
        REPORT: {final_report}

        Respond ONLY with a JSON object matching the EvaluationSchema:
        - needs_refinement: boolean (True if missing data or logic errors exist)
        - refinement_reason: string (be specific about what is missing)
        """

        try:
            # 2. Use the Global Client (SDK Style) with Beta Parsed output
            response = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a critical Research Evaluator. Use structured output."},
                    {"role": "user", "content": eval_prompt}
                ],
                response_format=EvaluationSchema,
                temperature=0.0
            )

            result = response.choices[0].message.parsed

            # --- Update shared state ---
            state.update({
                'needs_refinement': result.needs_refinement,
                'refinement_reason': result.refinement_reason,
                'report_generated': True
            })

            print(f"{C_MAGENTA}[{self.id.upper()} RESULT] Needs Refinement: {result.needs_refinement}{C_RESET}")
            print(f"{C_MAGENTA}[{self.id.upper()} REASON] {result.refinement_reason}{C_RESET}")

        except Exception as e:
            print(f"{C_RED}[{self.id.upper()} ERROR] Global Client call failed: {e}{C_RESET}")
            state.update({'needs_refinement': False, 'report_generated': True})

        return state