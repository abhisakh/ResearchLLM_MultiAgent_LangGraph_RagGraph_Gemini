import json
import re
from typing import Dict, Any, List, Optional
from core.research_state import ResearchState
from core.utilities import (
    C_ACTION, C_RESET, C_GREEN, C_YELLOW, C_RED, C_BLUE,
    client, LLM_MODEL, C_MAGENTA, C_CYAN
)

# ==================================================================================================
# Intent Agent (Section 3) - CODE IS CORRECT
# ==================================================================================================
class IntentAgent:
    """
    Agent responsible for determining the primary intent and extracting constraints.
    Now correctly persists 'reasoning' to the State and routes back to the Supervisor Hub.
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
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert intent extractor. Output ONLY JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content.strip())
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
            # --- FIX 1: USE THE NEW 'reasoning' KEY ---
            state["reasoning"] = llm_output.get("reasoning", "Scientific inquiry detected.")

            primary_intent = llm_output["primary_intent"]
            if primary_intent not in self.valid_intents:
                primary_intent = "general_research"

            state["primary_intent"] = primary_intent
            state["system_constraints"] = llm_output.get("extracted_constraints", [])
            state["material_elements"] = []

            # --- HUB-AND-SPOKE LOGIC ---
            # We don't set next="END" anymore.
            # We return to the Supervisor, and the Supervisor handles the routing.
            if primary_intent == "irrelevant":
                print(f"{C_RED}[{self.id.upper()} REJECT] Out-of-scope query flagged.{C_RESET}")
            else:
                print(f"{C_YELLOW}[{self.id.upper()} STATE] Intent: **{primary_intent}**{C_RESET}")
        else:
            # Fallback
            state["primary_intent"] = "general_research"
            state["reasoning"] = "LLM parsing failure; defaulting to research."
            state["system_constraints"] = []

        # --- FIX 2: ROUTE BACK TO HUB ---
        state["next"] = "supervisor_agent"

        print(f"{C_GREEN}[{self.id.upper()} DONE] Returning control to Supervisor.{C_RESET}")
        return state

# ==================================================================================================
# Planning Agent (Section 4) - FIXES APPLIED (Model and State Corruption)
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
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert research planner. Output ONLY JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"{C_RED}[{self.id.upper()} ERROR] {e}{C_RESET}")
            return None

    def execute(self, state: ResearchState) -> ResearchState:
        # 1. TRACK VISIT
        state.setdefault("visited_nodes", []).append(self.id)

        # 2. EXTRACT STATE DATA
        primary_intent = state.get("primary_intent", "general_research")
        semantic_query = state.get("semantic_query", "")
        reasoning = state.get("reasoning", "") # Reading the reasoning key we added
        system_constraints_original = state.get("system_constraints", [])

        # --- FIX: Refinement logic now uses 'refinement_reason' strictly ---
        is_refining = state.get('is_refining', False)
        refinement_reason = state.get('refinement_reason', '') # This is the key we protected!

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

        # 4. PROCESS OUTPUT (Applying Fallbacks)
        if llm_output and isinstance(llm_output, dict):
            execution_plan = llm_output.get("execution_plan", [])
            active_tools = llm_output.get("active_tools", [])
            validated_tools = [t for t in active_tools if t in self.available_tools]

            # Baseline Enforcement
            if primary_intent == 'literature_review':
                for tool in ['pubmed', 'arxiv']:
                    if tool not in validated_tools: validated_tools.append(tool)

            # Additive Refinement
            if is_refining:
                previous_tools = state.get("active_tools", [])
                validated_tools = list(set(validated_tools + previous_tools))

            state["execution_plan"] = execution_plan if execution_plan else ["Execute search.", "Synthesize."]
            state["active_tools"] = list(set(validated_tools))
        else:
            state["execution_plan"] = ["Execute multi-tool search.", "Generate synthesis report."]
            state["active_tools"] = ["arxiv", "openalex", "web"]

        # --- HUB-AND-SPOKE FIX ---
        # We NO LONGER set next based on downstream agents.
        # Every node returns to the Supervisor.
        state["next"] = "supervisor_agent"

        print(f"{C_GREEN}[{self.id.upper()} DONE] Returning to Supervisor.{C_RESET}")
        return state


# ==================================================================================================
# Query Generation Agent (Section 6) - COIE-Enhanced & Fully Rewritten
# ==================================================================================================
class QueryGenerationAgent:
    """
    Agent responsible for generating tiered, tool-specific search queries.
    ALIGNED: Operates as a 'Spoke' in the Hub-and-Spoke architecture.
    """

    def __init__(self, agent_id: str = "query_gen_agent", model: str = LLM_MODEL):
        self.id = agent_id
        self.model = model

        # Tool-specific search tiers optimized for individual API capabilities
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
        """Constructs a tool-aware prompt for the LLM."""
        required_tools = ", ".join([f"'{tool}'" for tool in active_tools])

        tier_instructions = "\n".join([
            f"- **{tool.capitalize()}**: requires tiers {', '.join([f'`{tier}`' for tier in self.search_tiers.get(tool, [])])}"
            for tool in active_tools if tool in self.search_tiers
        ])

        constraints_str = "\n".join([f"- {k}: {v}" for k, v in constraints.items()]) if constraints else "None"

        # Contextual logic for refinement cycles
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
        """Handles LLM communication and validates output structure."""
        if client is None: return None
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research planning assistant. Output ONLY JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content.strip())

            # Clean elements and filter queries to only active tools
            return {
                "tiered_queries": {k: v for k, v in data.get('tiered_queries', {}).items() if k in active_tools},
                "material_elements": [str(e).strip() for e in data.get('material_elements', []) if e]
            }
        except Exception as e:
            print(f"{C_RED}[{self.id.upper()} ERROR] LLM Call Failed: {e}{C_RESET}")
            return None

    def execute(self, state: ResearchState) -> ResearchState:
        """Orchestrates query generation and returns control to the Supervisor Hub."""
        # 1. Track visit
        state.setdefault("visited_nodes", []).append(self.id)
        print(f"\n{C_ACTION}[{self.id.upper()} START] Generating tool-specific queries...{C_RESET}")

        # 2. Extract Data
        semantic_query = state.get("semantic_query", "")
        active_tools = state.get("active_tools", [])
        reasoning = state.get("reasoning", "")
        refinement_reason = state.get("refinement_reason", "")
        system_constraints_list = state.get("system_constraints", [])

        if not active_tools:
            print(f"{C_RED}[{self.id.upper()} FAIL] No active tools provided. Returning to Hub.{C_RESET}")
            state["next"] = "supervisor_agent"
            return state

        # 3. Process Logic
        constraints_dict = self._get_constraints_from_list(system_constraints_list)
        prompt = self._format_prompt(semantic_query, constraints_dict, active_tools, reasoning, refinement_reason)
        llm_output = self._call_llm_and_parse(prompt, active_tools)

        # 4. Update State
        if llm_output and llm_output.get("tiered_queries"):
            state["tiered_queries"] = llm_output["tiered_queries"]

            # Merge existing constraints with newly extracted elements
            merged_elements = list(system_constraints_list)
            merged_elements.extend(llm_output.get("material_elements", []))
            state["material_elements"] = list(set(merged_elements)) # De-duplicate

            # Set primary search term for Materials Project
            state["api_search_term"] = llm_output["material_elements"][0] if llm_output["material_elements"] else semantic_query

            total_q = sum(len(v) for v in state["tiered_queries"].values())
            print(f"{C_YELLOW}[{self.id.upper()} STATE] {total_q} queries generated for {len(active_tools)} tools.{C_RESET}")
        else:
            print(f"{C_RED}[{self.id.upper()} FAIL] Using fallback. Queries empty.{C_RESET}")
            state["tiered_queries"] = {}

        # 5. ENFORCE HUB-AND-SPOKE ROUTING
        # Every node yields control back to the Supervisor to determine the next destination.
        state["next"] = "supervisor_agent"

        print(f"{C_GREEN}[{self.id.upper()} DONE] Handing state back to Supervisor Hub.{C_RESET}")
        return state

#================ CODE DEBUG BLOCK ===============================
if __name__ == "__main__":
    # Ensure ResearchState has the 'system_constraints' key for proper testing
    from core.research_state import ResearchState # Assume ResearchState is updated

    # --- Setup ---
    print(f"\n{C_CYAN}*** STARTING AGENT CHAIN ISOLATED TESTS ***{C_RESET}")

    # TEST CASE 1: Complex Material Science Query with Time Constraint
    test_query_1 = "A detailed review on the synthesis and bandgap stability of lead-free CsSnI3 perovskite solar cells using computational and experimental data published in the last decade."
    #test_query_1 = "A detailed review on the synthesis and bandgap stability of lead-free cesium-tin halide perovskite solar cells using computational and experimental data published in the last decade."
    #test_query_1 = "Provide me a brief review on the advance materials which we use for building quantum computer."
    # TEST CASE 2: Simple Literature Review
    test_query_2 = "Recent advancements in non-invasive blood sugar monitoring for type 2 diabetes."

    current_test_query = test_query_1 # <--- CHANGE THIS TO test_query_2 TO TEST CASE 2

    # Initialize Test State (Ensure 'system_constraints' is present in initial state)
    initial_state: ResearchState = {
        "user_query": current_test_query,
        "semantic_query": current_test_query,
        "primary_intent": "",
        "reasoning": "",
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
        "refinement_retries": 0,
        "next": "",
        "visited_nodes": [],
    }
    print(f"\n{C_GREEN}============= INITIALIZED RESEARCH STATE ==========================={C_RESET}")
    #print(json.dumps(initial_state, indent=4))
    for i, (key, value) in enumerate(initial_state.items()):
        print(f"{C_CYAN}{i}: {key}{C_RESET}")
        print(value)
        print(f"{C_YELLOW}{'-' * 40}{C_RESET}")
    # --- 1. Test IntentAgent ---
    print(f"\n{C_MAGENTA}--- 1. Testing IntentAgent ---{C_RESET}")
    intent_agent = IntentAgent()
    state_after_intent = intent_agent.execute(initial_state)

    print(f"{C_GREEN}[RESULT] Primary Intent: {state_after_intent.get('primary_intent')}{C_RESET}")
    # CHECK 1: Constraints should be in 'system_constraints'
    print(f"{C_GREEN}[CHECK 1] System Constraints: {state_after_intent.get('system_constraints')}{C_RESET}")
    # CHECK 2: 'material_elements' should be empty
    print(f"{C_GREEN}[CHECK 2] Material Elements (Expected Empty): {state_after_intent.get('material_elements')}{C_RESET}")
    print("-" * 40)

    print(f"\n{C_GREEN}============= RESEARCH STATE after IntentAgent ==========================={C_RESET}")
    #print(json.dumps(state_after_intent, indent=4))
    for i, (key, value) in enumerate(state_after_intent.items()):
        print(f"{C_CYAN}{i}: {key}{C_RESET}")
        print(value)
        print(f"{C_YELLOW}{'-' * 40}{C_RESET}")

    # --- 2. Test PlanningAgent ---
    print(f"\n{C_MAGENTA}--- 2. Testing PlanningAgent ---{C_RESET}")
    planning_agent = PlanningAgent()
    state_after_planning = planning_agent.execute(state_after_intent)

    print(f"{C_GREEN}[RESULT] Active Tools: {state_after_planning.get('active_tools')}{C_RESET}")
    print(f"{C_GREEN}[RESULT] Execution Plan (Steps):{C_RESET}")
    for i, step in enumerate(state_after_planning.get('execution_plan', [])):
        print(f"  {i+1}. {step}")

    print("-" * 40)

    print(f"\n{C_GREEN}============= RESEARCH STATE after PlanningAgent ==========================={C_RESET}")
    #print(json.dumps(state_after_planning, indent=4))
    for i, (key, value) in enumerate(state_after_planning.items()):
        print(f"{C_CYAN}{i}: {key}{C_RESET}")
        print(value)
        print(f"{C_YELLOW}{'-' * 40}{C_RESET}")

    # --- 3. Test QueryGenerationAgent ---±
    print(f"\n{C_MAGENTA}--- 3. Testing QueryGenerationAgent ---{C_RESET}")
    query_gen_agent = QueryGenerationAgent()
    state_after_query_gen = query_gen_agent.execute(state_after_planning)

    # CHECK 3: Verify 'material_elements' is MERGED (constraints + compounds)
    extracted_elements = state_after_query_gen.get('material_elements', [])
    print(f"{C_BLUE}[CHECK 3] Merged Elements (material_elements): {extracted_elements}{C_RESET}")
    print(f"{C_BLUE}[CHECK 4] API Search Term: {state_after_query_gen.get('api_search_term')}{C_RESET}")

    # CHECK 5: Verify constraints were preserved in 'system_constraints'
    print(f"{C_BLUE}[CHECK 5] System Constraints (Must be identical to CHECK 1): {state_after_query_gen.get('system_constraints')}{C_RESET}")

    print(f"{C_GREEN}[RESULT] Tiered Queries (Sample):{C_RESET}")
    for tool, queries in state_after_query_gen.get('tiered_queries', {}).items():
        if isinstance(queries, dict) and queries:
            for i, (tier, q) in enumerate(queries.items()):
                print(f"  - {tool.upper()}/{tier.upper()}: {q[:80]}...")

    print(f"\n{C_CYAN}*** AGENT CHAIN TESTS COMPLETE ***{C_RESET}")

    print(f"\n{C_GREEN}============= UPDATED RESEARCH STATE ==========================={C_RESET}")

    #print(json.dumps(state_after_query_gen, indent=4)) # Uncomment to see full final state
    print(type(state_after_query_gen))
    for i, (key, value) in enumerate(state_after_query_gen.items()):
        print(f"{C_CYAN}{i}: {key}{C_RESET}")
        print(value)
        print(f"{C_YELLOW}{'-' * 40}{C_RESET}")
    print(json.dumps(state_after_query_gen, indent=4)) # Uncomment to see full final state
