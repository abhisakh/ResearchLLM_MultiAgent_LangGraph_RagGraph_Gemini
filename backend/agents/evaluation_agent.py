import logging
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from google.genai import types

# --- Synchronized Imports from Synthesis logic ---
from backend.core.research_state import ResearchState
from backend.core.errors import record_error, to_llm_error, user_message
from backend.core.utilities import (
    LLM_MODEL, generate_content_with_retry, response_json, retry_on_bad_output
)

logger = logging.getLogger(__name__)

# ==================================================================================================
# SECTION 9.A.: EVALUATION AGENT
# ==================================================================================================

class RefineStage(str, Enum):
    """Where a refinement cycle should restart. The supervisor routes on this (see SupervisorAgent)."""
    SEARCH = "search"          # too few, off-topic or missing sources: search again
    RETRIEVAL = "retrieval"    # documents could not be downloaded or their text not extracted
    CONTEXT = "context"        # the sources are fine but the passages chosen for the report are not relevant
    SYNTHESIS = "synthesis"    # the sources and passages are fine: the writing, structure or citations are not


class EvaluationSchema(BaseModel):
    """Schema for Evaluation Agent output to ensure reliable boolean routing."""
    needs_refinement: bool = Field(description="TRUE if the report fails to address the plan. Otherwise FALSE.")
    refinement_reason: str = Field(description="Specific reason for refinement or 'Report is satisfactory'.")
    refine_stage: Optional[RefineStage] = Field(
        default=None,
        description="When refinement is needed: the stage that caused the problem. Empty when no refinement is needed.",
    )

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

        user_query = state.get('user_query', '')
        execution_plan = state.get('execution_plan', [])
        final_report = state.get('final_report', '')

        # 2. Guardrail: Empty/Short Report
        if not final_report or len(final_report) < 200:
            logger.error(f'[{self.id.upper()} ERROR] Content insufficient. Triggering refinement cycle.')
            state.update({
                'needs_refinement': True,
                'refinement_reason': "Synthesis produced insufficient or empty content.",
                'refinement_stage': RefineStage.SYNTHESIS.value,
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

        If refinement is needed, also set refine_stage to the stage that caused the problem:
        - "search": too few, off-topic or missing sources, so more searching is needed.
        - "retrieval": documents could not be downloaded or their text could not be extracted.
        - "context": the sources exist, but the passages chosen for the report are not relevant to the question.
        - "synthesis": the sources and passages are fine; the writing, structure or citations are the problem.
        If no refinement is needed, leave refine_stage empty.
        """

        try:
            result = retry_on_bad_output(
                lambda: EvaluationSchema.model_validate(response_json(
                    generate_content_with_retry(
                        model=self.model,
                        contents=eval_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction="You are a critical Research Evaluator. Use structured output matching the requested schema.",
                            response_mime_type="application/json",
                            response_schema=EvaluationSchema,
                            temperature=0.0
                        )
                    ),
                    self.id,
                )),
                label=self.id,
            )

            # --- Update shared state ---
            state.update({
                'needs_refinement': result.needs_refinement,
                'refinement_reason': result.refinement_reason,
                'refinement_stage': result.refine_stage.value if (result.needs_refinement and result.refine_stage) else None,
                'evaluation_unavailable': False,
                'next': 'supervisor_agent'
            })

            log = logger.warning if result.needs_refinement else logger.info
            log(f"[{self.id.upper()} RESULT] Needs Refinement: {result.needs_refinement}")
            log(f"[{self.id.upper()} REASON] {result.refinement_reason}")

        except Exception as exc:
            # Never approve silently. The report is delivered, flagged as not audited, and the failure
            # is recorded so the supervisor and the API can say so.
            error = to_llm_error(exc, label=self.id)
            record_error(state, self.id, error)
            logger.error(f'[{self.id.upper()} ERROR] Evaluation unavailable: {error}')
            state.update({
                'needs_refinement': False,
                'refinement_reason': f"Evaluation unavailable: {user_message(error.kind)}",
                'refinement_stage': None,
                'evaluation_unavailable': True,
                'next': 'supervisor_agent'
            })

        return state
