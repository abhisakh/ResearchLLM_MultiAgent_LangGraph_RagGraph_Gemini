import json
import re
from typing import Dict, Any, List, Optional
from google.genai import types

# Keep package import paths identical to your project setup
from backend.core.research_state import ResearchState
from backend.core.utilities import (
    C_ACTION, C_RESET, C_GREEN, C_YELLOW, C_RED, C_BLUE,
    client, LLM_MODEL, C_MAGENTA, C_CYAN, generate_content_with_retry
)

# ==================================================================================================
# Intent Agent (Section 3)
# ==================================================================================================
class IntentAgent:
    """
    Agent responsible for determining the primary intent and extracting constraints.
    Persists 'reasoning' to the State and routes back to the Supervisor Hub.
    """
    def __init__(self, model: str = LLM_MODEL):
        self.id = "intent_agent"
        self.model = model
        self.valid_intents = [
            "literature_review", "materials_research", "comparative_analysis",
            "medical_diagnosis", "general_research", "data_extraction", "irrelevant"
        ]

    def _format_prompt(self, semantic_query: str) -> str:
        intent_list_str = ", ".join([f"'{i}'" for i in self.valid_intents])
        return f"""
        You are an elite Research Librarian. Your goal is to determine if a query requires
        **Systematic Investigation** (Academic/Technical) or if it is a **Real-time/Casual Interaction**.

        **Query:** "{semantic_query}"

        **Reasoning Framework:**
        - **Research Intent:** The query seeks to understand 'why', 'how', or 'what are the trends'. Requires multi-source synthesis.
        - **Irrelevant Intent:** The query is a 'right now' request (weather, stocks), creative (poem), or casual chat.

        **Output JSON Format:**
        {{
          "primary_intent": "{intent_list_str}",
          "scientific_depth_score": 0.0 to 1.0,
          "reasoning": "Explain WHY this is or isn't a research query.",
          "extracted_constraints": ["KEY: VALUE"]
        }}
        """

    def _call_llm_and_parse(self, prompt: str) -> Optional[Dict[str, Any]]:
        if client is None:
            return None
        try:
            print(f"{C_BLUE}[{self.id.upper()} ACTION] Classifying intent...{C_RESET}")
            response = generate_content_with_retry(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are an expert intent extractor. Output ONLY JSON.",
                    response_mime_type="application/json",
                    temperature=0.0,
                    tools=[]  # Silences AFC SDK warning
                )
            )
            return json.loads(response.text.strip()) if response.text else None
        except Exception as e:
            print(f"{C_RED}[{self.id} ERROR] {e}{C_RESET}")
            return None

    def execute(self, state: ResearchState) -> ResearchState:
        # 1. TRACK VISIT
        state.setdefault("visited_nodes", []).append(self.id)

        print(f"\n{C_ACTION}[{self.id.upper()} START] Analyzing query intent...{C_RESET}")
        semantic_query = state.get("semantic_query", "")

        llm_output = self._call_llm_and_parse(self._format_prompt(semantic_query))

        if llm_output and llm_output.get("primary_intent"):
            state["reasoning"] = llm_output.get("reasoning", "Scientific inquiry detected.")

            primary_intent = llm_output["primary_intent"]
            if primary_intent not in self.valid_intents:
                primary_intent = "general_research"

            state["primary_intent"] = primary_intent
            state["system_constraints"] = llm_output.get("extracted_constraints", [])
            state["material_elements"] = []

            if primary_intent == "irrelevant":
                print(f"{C_RED}[{self.id.upper()} REJECT] Out-of-scope query flagged.{C_RESET}")
            else:
                print(f"{C_YELLOW}[{self.id.upper()} STATE] Intent: **{primary_intent}**{C_RESET}")
        else:
            state["primary_intent"] = "general_research"
            state["reasoning"] = "LLM parsing failure; defaulting to research."
            state["system_constraints"] = []

        state["next"] = "supervisor_agent"

        print(f"{C_GREEN}[{self.id.upper()} DONE] Returning control to Supervisor.{C_RESET}")
        return state

# ==================================================================================================
# Planning Agent (Section 4)
# ==================================================================================================
class PlanningAgent:
    """
    Agent responsible for creating an execution plan AND dynamically selecting
    the necessary tools. Logic updated for Hub-and-Spoke orchestration.
    """
    def __init__(self, model: str = LLM_MODEL):
        self.id = "planning_agent"
        self.model = model
        self.available_tools = [
            "pubmed", "arxiv", "openalex", "web",
            "materials", "semanticscholar", "chemrxiv"
        ]

    def _format_prompt(self, intent: str, query: str, constraints: List[str], reasoning: str) -> str:
        tool_list_str = ", ".join([f"'{t}'" for t in self.available_tools])
        constraints_str = "\n".join(constraints) if constraints else "None"

        return f"""
        You are an expert project manager for a multi-agent research system.

        **Context from Intent Analysis:** {reasoning}
        **Primary Intent:** {intent}
        **Semantic Query:** "{query}"
        **Extracted Constraints:**
        ---
        {constraints_str}
        ---

        **Available Tools:** {tool_list_str}

        **Instructions:**
        1. Generate a detailed, step-by-step execution plan (minimum 5 steps).
        2. Select all necessary tools based on the guidelines (Materials for properties, Arxiv for Physics, etc.).
        3. Output ONLY a single JSON object:
           {{
             "execution_plan": ["step 1", "step 2", ...],
             "active_tools": ["tool1", "tool2"]
           }}
        """

    def _call_llm_and_parse(self, prompt: str) -> Optional[Dict[str, Any]]:
        if client is None: return None
        try:
            print(f"{C_BLUE}[{self.id.upper()} ACTION] Planning and Tool Selection...{C_RESET}")
            response = generate_content_with_retry(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are an expert research planner. Output ONLY JSON.",
                    response_mime_type="application/json",
                    temperature=0.1,
                    tools=[]
                )
            )
            return json.loads(response.text.strip()) if response.text else None
        except Exception as e:
            print(f"{C_RED}[{self.id.upper()} ERROR] {e}{C_RESET}")
            return None

    def execute(self, state: ResearchState) -> ResearchState:
        # 1. TRACK VISIT
        state.setdefault("visited_nodes", []).append(self.id)

        # 2. EXTRACT STATE DATA
        primary_intent = state.get("primary_intent", "general_research")
        semantic_query = state.get("semantic_query", "")
        reasoning = state.get("reasoning", "")
        system_constraints_original = state.get("system_constraints", [])

        is_refining = state.get('is_refining', False)
        refinement_reason = state.get('refinement_reason', '')

        system_constraints_for_prompt = list(system_constraints_original)
        prompt_modifier = ""

        if is_refining and refinement_reason:
            print(f"{C_MAGENTA}[{self.id.upper()} REFINEMENT MODE] Strategy guided by Evaluation feedback.{C_RESET}")
            prompt_modifier = f"""
            **REFINEMENT INSTRUCTION:** The previous attempt failed because: "{refinement_reason}".
            You MUST expand the Tool Selection. Add alternatives to fill the data gaps.
            """
            system_constraints_for_prompt.append(f"PREVIOUS_FAILURE_REASON: {refinement_reason}")

        # 3. GENERATE PROMPT AND CALL LLM
        full_prompt = self._format_prompt(primary_intent, semantic_query, system_constraints_for_prompt, reasoning)
        if prompt_modifier:
            full_prompt = prompt_modifier + full_prompt

        llm_output = self._call_llm_and_parse(full_prompt)

        # 4. PROCESS OUTPUT
        if llm_output and isinstance(llm_output, dict):
            execution_plan = llm_output.get("execution_plan", [])
            active_tools = llm_output.get("active_tools", [])
            validated_tools = [t for t in active_tools if t in self.available_tools]

            if primary_intent == 'literature_review':
                for tool in ['pubmed', 'arxiv']:
                    if tool not in validated_tools: validated_tools.append(tool)

            if is_refining:
                previous_tools = state.get("active_tools", [])
                validated_tools = list(set(validated_tools + previous_tools))

            state["execution_plan"] = execution_plan if execution_plan else ["Execute search.", "Synthesize."]
            state["active_tools"] = list(set(validated_tools))
        else:
            state["execution_plan"] = ["Execute multi-tool search.", "Generate synthesis report."]
            state["active_tools"] = ["arxiv", "openalex", "web"]

        state["next"] = "supervisor_agent"

        print(f"{C_GREEN}[{self.id.upper()} DONE] Returning to Supervisor.{C_RESET}")
        return state


# ==================================================================================================
# Query Generation Agent (Section 6)
# ==================================================================================================
class QueryGenerationAgent:
    """
    Agent responsible for generating tiered, tool-specific search queries.
    Operates as a 'Spoke' in the Hub-and-Spoke architecture.
    """

    def __init__(self, agent_id: str = "query_gen_agent", model: str = LLM_MODEL):
        self.id = agent_id
        self.model = model

        self.search_tiers = {
            "pubmed": ["strict", "moderate", "broad"],
            "arxiv": ["strict", "moderate", "broad"],
            "semanticscholar": ["strict", "moderate"],
            "chemrxiv": ["simple"],
            "openalex": ["simple"],
            "web": ["simple"],
            "materials": ["simple"]
        }

    def _get_constraints_from_list(self, constraint_list: List[str]) -> Dict[str, Any]:
        """Parses 'KEY: VALUE' strings into a dictionary."""
        constraints = {}
        for item in constraint_list:
            if ":" in item:
                key, value = item.split(":", 1)
                constraints[key.strip()] = value.strip()
        return constraints

    def _format_prompt(
        self,
        semantic_query: str,
        constraints: Dict[str, Any],
        active_tools: List[str],
        reasoning: str = "",
        refinement_reason: str = ""
    ) -> str:
        required_tools = ", ".join([f"'{tool}'" for tool in active_tools])

        tier_instructions = "\n".join([
            f"- **{tool.capitalize()}**: requires tiers {', '.join([f'`{tier}`' for tier in self.search_tiers.get(tool, [])])}"
            for tool in active_tools if tool in self.search_tiers
        ])

        constraints_str = "\n".join([f"- {k}: {v}" for k, v in constraints.items()]) if constraints else "None"

        context_block = f"**INTENT ANALYSIS:** {reasoning}" if reasoning else ""
        refinement_block = f"**REFINEMENT FEEDBACK:** {refinement_reason}" if refinement_reason else ""

        example_output = {
            "tiered_queries": {
                tool: {tier: f"<{tool}_{tier}_query>" for tier in self.search_tiers.get(tool, [])}
                for tool in active_tools
            },
            "material_elements": ["Primary_Formula", "Element_1", "Element_2"]
        }

        return f"""
        You are an expert Research Query Engineer. Your task is to translate a research objective
        into highly optimized, tool-specific search strings.

        **CONTEXT:**
        {context_block}
        {refinement_block}

        **RESEARCH PARAMETERS:**
        - Semantic Query: "{semantic_query}"
        - Active Tools: {required_tools}
        - Constraints: {constraints_str}

        **GUIDELINES:**
        1. **Precision**: 'Strict' tiers should use exact terminology.
        2. **Recall**: 'Broad' tiers should use synonyms and related phenomena.
        3. **Logic**: For ChemRxiv/OpenAlex ('simple' tier), use keyword strings joined by 'AND'.
        4. **Exclusions**: For Web, add negative operators (e.g., -buy, -stock, -pinterest).
        5. **Formulas**: Extract chemical formulas for the 'material_elements' key.

        **OUTPUT FORMAT (JSON ONLY):**
        {json.dumps(example_output, indent=4)}

        **TIER SPECIFICATIONS:**
        {tier_instructions}
        """

    def _call_llm_and_parse(self, prompt: str, active_tools: List[str]) -> Optional[Dict[str, Any]]:
        if client is None: return None
        try:
            response = generate_content_with_retry(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are a research planning assistant. Output ONLY JSON.",
                    response_mime_type="application/json",
                    temperature=0.1,
                    tools=[]
                )
            )
            data = json.loads(response.text.strip()) if response.text else {}

            return {
                "tiered_queries": {k: v for k, v in data.get('tiered_queries', {}).items() if k in active_tools},
                "material_elements": [str(e).strip() for e in data.get('material_elements', []) if e]
            }
        except Exception as e:
            print(f"{C_RED}[{self.id.upper()} ERROR] LLM Call Failed: {e}{C_RESET}")
            return None

    def execute(self, state: ResearchState) -> ResearchState:
        state.setdefault("visited_nodes", []).append(self.id)
        print(f"\n{C_ACTION}[{self.id.upper()} START] Generating tool-specific queries...{C_RESET}")

        semantic_query = state.get("semantic_query", "")
        active_tools = state.get("active_tools", [])
        reasoning = state.get("reasoning", "")
        refinement_reason = state.get("refinement_reason", "")
        system_constraints_list = state.get("system_constraints", [])

        if not active_tools:
            print(f"{C_RED}[{self.id.upper()} FAIL] No active tools provided. Returning to Hub.{C_RESET}")
            state["next"] = "supervisor_agent"
            return state

        constraints_dict = self._get_constraints_from_list(system_constraints_list)
        prompt = self._format_prompt(semantic_query, constraints_dict, active_tools, reasoning, refinement_reason)
        llm_output = self._call_llm_and_parse(prompt, active_tools)

        if llm_output and llm_output.get("tiered_queries"):
            state["tiered_queries"] = llm_output["tiered_queries"]

            merged_elements = list(system_constraints_list)
            merged_elements.extend(llm_output.get("material_elements", []))
            state["material_elements"] = list(set(merged_elements))

            state["api_search_term"] = llm_output["material_elements"][0] if llm_output["material_elements"] else semantic_query

            total_q = sum(len(v) for v in state["tiered_queries"].values())
            print(f"{C_YELLOW}[{self.id.upper()} STATE] {total_q} queries generated for {len(active_tools)} tools.{C_RESET}")
        else:
            print(f"{C_RED}[{self.id.upper()} FAIL] Using fallback. Queries empty.{C_RESET}")
            state["tiered_queries"] = {}

        state["next"] = "supervisor_agent"

        print(f"{C_GREEN}[{self.id.upper()} DONE] Handing state back to Supervisor Hub.{C_RESET}")
        return state

# ==================================================================================================
# CODE DEBUG BLOCK
# ==================================================================================================
if __name__ == "__main__":
    from pathlib import Path
    from backend.core.research_state import ResearchState

    print(f"\n{C_CYAN}*** STARTING AGENT CHAIN ISOLATED TESTS ***{C_RESET}")

    project_root = Path(__file__).resolve().parent.parent.parent / "examples" / "sample_outputs"
    input_file = project_root / "level_0_procedural_output.json"

    if not input_file.exists():
        print(f"{C_RED}[TEST ERROR] Input file 'level_0_procedural_output.json' not found in project root.{C_RESET}")
    else:
        print(f"{C_YELLOW}[TEST SETUP] Loading state from input file: {input_file}{C_RESET}")
        with open(input_file, "r", encoding="utf-8") as f:
            initial_state = json.load(f)

        # --- 1. Test IntentAgent ---
        print(f"\n{C_MAGENTA}--- 1. Testing IntentAgent ---{C_RESET}")
        intent_agent = IntentAgent()
        state_after_intent = intent_agent.execute(initial_state)

        # --- 2. Test PlanningAgent ---
        print(f"\n{C_MAGENTA}--- 2. Testing PlanningAgent ---{C_RESET}")
        planning_agent = PlanningAgent()
        state_after_planning = planning_agent.execute(state_after_intent)

        # --- 3. Test QueryGenerationAgent ---
        print(f"\n{C_MAGENTA}--- 3. Testing QueryGenerationAgent ---{C_RESET}")
        query_gen_agent = QueryGenerationAgent()
        state_after_query_gen = query_gen_agent.execute(state_after_planning)

        # Save output to level_1_planning_output.json in project root
        output_json_file = project_root / "level_1_planning_output.json"
        with open(output_json_file, "w", encoding="utf-8") as f:
            json.dump(state_after_query_gen, f, indent=4, ensure_ascii=False)
        print(f"{C_GREEN}[SAVED] Updated state written to: {output_json_file}{C_RESET}")

        print(f"\n{C_CYAN}*** AGENT CHAIN TESTS COMPLETE ***{C_RESET}")
