import logging
import json
from pydantic import BaseModel, Field
from google.genai import types

# --- Synchronized Imports from Synthesis logic ---
from backend.core.research_state import ResearchState
from backend.core.utilities import (
    LLM_MODEL, client
)

logger = logging.getLogger(__name__)

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

        logger.info(f'[{self.id.upper()} START] Performing quality audit...')

        if client is None:
            state.update({'needs_refinement': False, 'next': 'supervisor_agent'})
            return state

        user_query = state.get('user_query', '')
        execution_plan = state.get('execution_plan', [])
        final_report = state.get('final_report', '')

        # 2. Guardrail: Empty/Short Report
        if not final_report or len(final_report) < 200:
            logger.error(f'[{self.id.upper()} ERROR] Content insufficient. Triggering refinement cycle.')
            state.update({
                'needs_refinement': True,
                'refinement_reason': "Synthesis produced insufficient or empty content.",
                'next': 'supervisor_agent'
            })
            return state

        # 3. Evaluation Logic via Google GenAI SDK
        eval_prompt = f"""
        Analyze if the 'Final Report' successfully addresses the 'Execution Plan'.

        USER INTENT: {user_query}
        PLAN: {execution_plan}
        REPORT: {final_report}

        Provide a structured evaluation determining whether refinement is needed.
        """

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=eval_prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are a critical Research Evaluator. Use structured output matching the requested schema.",
                    response_mime_type="application/json",
                    response_schema=EvaluationSchema,
                    temperature=0.0
                )
            )

            raw_text = response.text.strip()
            result_dict = json.loads(raw_text)
            result = EvaluationSchema(**result_dict)

            # --- Update shared state ---
            state.update({
                'needs_refinement': result.needs_refinement,
                'refinement_reason': result.refinement_reason,
                'next': 'supervisor_agent'
            })

            log = logger.warning if result.needs_refinement else logger.info
            log(f"[{self.id.upper()} RESULT] Needs Refinement: {result.needs_refinement}")
            log(f"[{self.id.upper()} REASON] {result.refinement_reason}")

        except Exception as e:
            logger.error(f'[{self.id.upper()} ERROR] Evaluation failed: {e}')
            state.update({'needs_refinement': False, 'next': 'supervisor_agent'})

        return state
