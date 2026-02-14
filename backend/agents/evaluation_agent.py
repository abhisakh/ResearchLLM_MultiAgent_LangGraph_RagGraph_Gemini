import os
import json
from pydantic import BaseModel, Field
from typing import Dict, Any

# --- Synchronized Imports from Synthesis logic ---
from core.research_state import ResearchState
from core.utilities import (
    C_ACTION, C_RESET, C_RED, C_BLUE, C_MAGENTA,C_GREEN,
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
    Evaluates SynthesisAgent's report.
    ALIGNED: Returns control to the Supervisor Hub to decide on termination or refinement.
    """

    def __init__(self, agent_id: str = "evaluation_agent", model: str = LLM_MODEL):
        self.id = agent_id
        self.model = model

    def execute(self, state: ResearchState) -> ResearchState:
        # 1. BREADCRUMB TRACKING
        state.setdefault("visited_nodes", []).append(self.id)

        print(f"\n{C_ACTION}[{self.id.upper()} START] Performing quality audit...{C_RESET}")

        if client is None:
            state.update({'needs_refinement': False, 'next': 'supervisor_agent'})
            return state

        user_query = state.get('user_query', '')
        execution_plan = state.get('execution_plan', [])
        final_report = state.get('final_report', '')

        # 2. Guardrail: Empty/Short Report
        if not final_report or len(final_report) < 200:
            print(f"{C_RED}[{self.id.upper()} ERROR] Content insufficient. Triggering refinement cycle.{C_RESET}")
            state.update({
                'needs_refinement': True,
                'refinement_reason': "Synthesis produced insufficient or empty content.",
                'next': 'supervisor_agent'
            })
            return state

        # 3. Evaluation Logic
        eval_prompt = f"""
        Analyze if the 'Final Report' successfully addresses the 'Execution Plan'.

        USER INTENT: {user_query}
        PLAN: {execution_plan}
        REPORT: {final_report}

        Respond ONLY with a JSON object matching the EvaluationSchema.
        """

        try:
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
                # Crucial: next is always the Hub
                'next': 'supervisor_agent'
            })

            color = C_RED if result.needs_refinement else C_GREEN
            print(f"{color}[{self.id.upper()} RESULT] Needs Refinement: {result.needs_refinement}{C_RESET}")
            print(f"{color}[{self.id.upper()} REASON] {result.refinement_reason}{C_RESET}")

        except Exception as e:
            print(f"{C_RED}[{self.id.upper()} ERROR] Evaluation failed: {e}{C_RESET}")
            # Fallback: Don't loop infinitely on error
            state.update({'needs_refinement': False, 'next': 'supervisor_agent'})

        return state