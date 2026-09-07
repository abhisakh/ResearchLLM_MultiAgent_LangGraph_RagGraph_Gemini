import os
import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, Any
from google.genai import types

# --- Synchronized Imports from Synthesis logic ---
from backend.core.research_state import ResearchState
from backend.core.utilities import (
    C_ACTION, C_RESET, C_RED, C_BLUE, C_MAGENTA, C_GREEN,C_YELLOW,
    LLM_MODEL, client
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

            color = C_RED if result.needs_refinement else C_GREEN
            print(f"{color}[{self.id.upper()} RESULT] Needs Refinement: {result.needs_refinement}{C_RESET}")
            print(f"{color}[{self.id.upper()} REASON] {result.refinement_reason}{C_RESET}")

        except Exception as e:
            print(f"{C_RED}[{self.id.upper()} ERROR] Evaluation failed: {e}{C_RESET}")
            state.update({'needs_refinement': False, 'next': 'supervisor_agent'})

        return state


# =====================================================
# TESTING BLOCK USING INPUT FILE: level_5_synthesis_ouput.json
# =====================================================
if __name__ == "__main__":
    print(f"{C_BLUE}==================================================")
    print("      RUNNING EVALUATION AGENT TEST SUITE         ")
    print(f"=================================================={C_RESET}")

    project_root = Path(__file__).resolve().parent.parent

    input_file = project_root / "level_5_synthesis_ouput.json"
    if not input_file.exists():
        matches = list(project_root.glob("**/level_5_synthesis_ouput.json"))
        if matches:
            input_file = matches[0]

    if not input_file.exists():
        print(f"{C_RED}[TEST ERROR] Input file 'level_5_synthesis_ouput.json' not found in project root or subdirectories.{C_RESET}")
    else:
        print(f"{C_YELLOW}[TEST SETUP] Loading state from input file: {input_file}{C_RESET}")

        try:
            with open(input_file, "r", encoding="utf-8") as f:
                test_state = json.load(f)

            agent = EvaluationAgent()
            updated_state = agent.execute(test_state)

            print(f"\n{C_GREEN}================ EVALUATION RESULTS ================{C_RESET}\n")
            print(f"Needs Refinement: {updated_state.get('needs_refinement')}")
            print(f"Refinement Reason: {updated_state.get('refinement_reason')}")
            print(f"\n{C_GREEN}===================================================={C_RESET}")

            output_json_file = project_root / "level_6_evaluation_ouput.json"
            output_md_file = project_root / "level_6_evaluation_ouput.md"

            with open(output_json_file, "w", encoding="utf-8") as f:
                json.dump(updated_state, f, indent=2, ensure_ascii=False)
            print(f"{C_GREEN}[SAVED] Updated state written to: {output_json_file}{C_RESET}")

            with open(output_md_file, "w", encoding="utf-8") as f:
                f.write(f"# Evaluation Agent Report\n\n")
                f.write(f"- **Needs Refinement**: `{updated_state.get('needs_refinement')}`\n")
                f.write(f"- **Refinement Reason**: {updated_state.get('refinement_reason')}\n")
                f.write(f"- **Next Node**: `{updated_state.get('next')}`\n")
            print(f"{C_GREEN}[SAVED] Evaluation summary written to: {output_md_file}{C_RESET}")

            print(f"{C_BLUE}[TEST SUCCESS] State key 'next': {updated_state.get('next')}{C_RESET}")

        except Exception as err:
            print(f"{C_RED}[TEST FAILED] Execution raised an exception: {err}{C_RESET}")
# ---------- GPT-5 EVALUATION AGENT ----------
# import os
# import json
# from pydantic import BaseModel, Field
# from typing import Dict, Any

# # --- Synchronized Imports from Synthesis logic ---
# from core.research_state import ResearchState
# from core.utilities import (
#     C_ACTION, C_RESET, C_RED, C_BLUE, C_MAGENTA,C_GREEN,
#     LLM_MODEL, client  # Use the working, authenticated global client
# )

# # ==================================================================================================
# # SECTION 9.A.: EVALUATION AGENT
# # ==================================================================================================

# class EvaluationSchema(BaseModel):
#     """Schema for Evaluation Agent output to ensure reliable boolean routing."""
#     needs_refinement: bool = Field(description="TRUE if the report fails to address the plan. Otherwise FALSE.")
#     refinement_reason: str = Field(description="Specific reason for refinement or 'Report is satisfactory'.")

# class EvaluationAgent:
#     """
#     Evaluates SynthesisAgent's report.
#     ALIGNED: Returns control to the Supervisor Hub to decide on termination or refinement.
#     """

#     def __init__(self, agent_id: str = "evaluation_agent", model: str = LLM_MODEL):
#         self.id = agent_id
#         self.model = model

#     def execute(self, state: ResearchState) -> ResearchState:
#         # 1. BREADCRUMB TRACKING
#         state.setdefault("visited_nodes", []).append(self.id)

#         print(f"\n{C_ACTION}[{self.id.upper()} START] Performing quality audit...{C_RESET}")

#         if client is None:
#             state.update({'needs_refinement': False, 'next': 'supervisor_agent'})
#             return state

#         user_query = state.get('user_query', '')
#         execution_plan = state.get('execution_plan', [])
#         final_report = state.get('final_report', '')

#         # 2. Guardrail: Empty/Short Report
#         if not final_report or len(final_report) < 200:
#             print(f"{C_RED}[{self.id.upper()} ERROR] Content insufficient. Triggering refinement cycle.{C_RESET}")
#             state.update({
#                 'needs_refinement': True,
#                 'refinement_reason': "Synthesis produced insufficient or empty content.",
#                 'next': 'supervisor_agent'
#             })
#             return state

#         # 3. Evaluation Logic
#         eval_prompt = f"""
#         Analyze if the 'Final Report' successfully addresses the 'Execution Plan'.

#         USER INTENT: {user_query}
#         PLAN: {execution_plan}
#         REPORT: {final_report}

#         Respond ONLY with a JSON object matching the EvaluationSchema.
#         """

#         try:
#             response = client.beta.chat.completions.parse(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": "You are a critical Research Evaluator. Use structured output."},
#                     {"role": "user", "content": eval_prompt}
#                 ],
#                 response_format=EvaluationSchema,
#                 temperature=0.0
#             )

#             result = response.choices[0].message.parsed

#             # --- Update shared state ---
#             state.update({
#                 'needs_refinement': result.needs_refinement,
#                 'refinement_reason': result.refinement_reason,
#                 # Crucial: next is always the Hub
#                 'next': 'supervisor_agent'
#             })

#             color = C_RED if result.needs_refinement else C_GREEN
#             print(f"{color}[{self.id.upper()} RESULT] Needs Refinement: {result.needs_refinement}{C_RESET}")
#             print(f"{color}[{self.id.upper()} REASON] {result.refinement_reason}{C_RESET}")

#         except Exception as e:
#             print(f"{C_RED}[{self.id.upper()} ERROR] Evaluation failed: {e}{C_RESET}")
#             # Fallback: Don't loop infinitely on error
#             state.update({'needs_refinement': False, 'next': 'supervisor_agent'})

#         return state