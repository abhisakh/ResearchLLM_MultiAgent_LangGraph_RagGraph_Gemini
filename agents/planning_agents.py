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
    Agent responsible for determining the primary intent of the user query and extracting constraints.
    It writes stable constraints to the 'system_constraints' key.
    """
    def __init__(self, model: str = LLM_MODEL): # Correctly uses LLM_MODEL variable
        self.id = "intent_agent"
        self.model = model
        self.valid_intents = [
            "literature_review", "materials_research", "comparative_analysis",
            "medical_diagnosis", "general_research", "data_extraction"
        ]

    def _format_prompt(self, semantic_query: str) -> str:
        intent_list_str = ", ".join([f"'{i}'" for i in self.valid_intents])
        return f"""
        Analyze the following research query and classify its primary intent.

        **Semantic Query:** "{semantic_query}"

        **Instructions:**
        1. Choose the single best classification from this list: [{intent_list_str}].
        2. Provide a brief, one-sentence description of why you chose that intent.
        3. **CRITICAL:** Extract all key constraints (topic, time frame, specific requirements)
        and format them into a list of strings, using the KEY: VALUE format. Use 'last_decade' for the time period if specified.

        **Output MUST be a single JSON object (Dict) with three keys:**
        - 'primary_intent': (string) The chosen intent.
        - 'reasoning': (string) The justification.
        - **'extracted_constraints': (list of strings) The structured constraints (e.g., ["TOPIC: Quantum Computation", "TIME_PERIOD: last_decade"]).**
        """

    def _call_llm_and_parse(self, prompt: str) -> Optional[Dict[str, Any]]:
        if client is None:
            print(f"{C_RED}[{self.id} ERROR] LLM client is not initialized.{C_RESET}")
            return None
        try:
            print(f"{C_BLUE}[{self.id} ACTION] Calling LLM ({self.model}) to classify intent and extract constraints...{C_RESET}")
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert research intent and constraint extractor. Output ONLY the requested JSON object."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            json_str = response.choices[0].message.content.strip()
            llm_output = json.loads(json_str)
            return llm_output

        except Exception as e:
            print(f"{C_RED}[{self.id} ERROR] LLM call or JSON parsing failed: {e}{C_RESET}")
            return None

    def execute(self, state: ResearchState) -> ResearchState:
        print(f"\n{C_ACTION}[{self.id.upper()} START] Determining intent and extracting constraints...{C_RESET}")
        semantic_query = state.get("semantic_query", "")

        llm_output = self._call_llm_and_parse(self._format_prompt(semantic_query))

        if llm_output and llm_output.get("primary_intent"):
            primary_intent = llm_output["primary_intent"]
            if primary_intent not in self.valid_intents:
                print(f"{C_YELLOW}[{self.id} WARN] LLM returned invalid intent: '{primary_intent}'. Defaulting to 'general_research'.{C_RESET}")
                primary_intent = "general_research"

            state["primary_intent"] = primary_intent

            extracted_constraints = llm_output.get("extracted_constraints", [])
            state["system_constraints"] = extracted_constraints

            # Initialize material_elements as an empty list (will be populated by QueryGenerationAgent)
            state["material_elements"] = []

            print(f"{C_YELLOW}[{self.id.upper()} STATE] Determined primary_intent: **{primary_intent}**{C_RESET}")
            print(f"{C_YELLOW}[{self.id.upper()} STATE] Extracted Constraints: **{extracted_constraints}**{C_RESET}")
        else:
            state["primary_intent"] = "general_research"
            state["system_constraints"] = []
            state["material_elements"] = []
            print(f"{C_RED}[{self.id.upper()} FAIL] LLM classification/extraction failed. Defaulting.{C_RESET}")

        print(f"{C_GREEN}[{self.id.upper()} DONE] Intent and constraints determined.{C_RESET}")
        return state

# ==================================================================================================
# Planning Agent (Section 4) - FIXES APPLIED (Model and State Corruption)
# ==================================================================================================
class PlanningAgent:
    """
    Agent responsible for creating an execution plan AND dynamically selecting
    the necessary tools. Logic is updated to handle REFINEMENT by deactivating
    failed tools (like pubmed) and activating alternatives (openalex, web).
    """
    # FIX 1: Corrected constructor to use LLM_MODEL variable
    def __init__(self, model: str = LLM_MODEL):
        self.id = "planning_agent"
        self.model = model
        self.available_tools = [
            "pubmed", "arxiv", "openalex", "web", "materials"
        ]

    def _format_prompt(self, intent: str, query: str, constraints: List[str]) -> str:
        tool_list_str = ", ".join([f"'{t}'" for t in self.available_tools])
        constraints_str = "\n".join(constraints) if constraints else "None"

        return f"""
        You are an expert project manager for a multi-agent research system. Based on the user's intent, query, and structured constraints,
        generate a detailed, step-by-step execution plan (minimum 5 steps) AND dynamically select the minimal set of tools required.

        **Primary Intent:** {intent}
        **Semantic Query:** "{query}"
        **Extracted Constraints:**
        ---
        {constraints_str}
        ---

        **Available Tools (Choose a subset):** {tool_list_str}

        **Tool Selection Guidelines:**
        1. If the query involves specific material properties (bandgap or stability) OR a compound formula/specific element is present in constraints, include **'materials'**.
        2. If the query is a literature review, include **'pubmed'** and **'arxiv'** (unless instructed otherwise).
        3. For general background or current events, include **'web'**.
        4. Select only the tools absolutely necessary for the query.

        **Instructions:**
        1. **Plan Detail:** Each step must be concise but actionable.
        2. **Output Format:** Output ONLY a single JSON object with two required keys.
           - 'execution_plan': (list of strings) The step-by-step plan.
           - 'active_tools': (list of strings) The selected tools.
        """

    def _call_llm_and_parse(self, prompt: str) -> Optional[Dict[str, Any]]:
        if client is None:
            print(f"{C_RED}[{self.id} ERROR] LLM client is not initialized.{C_RESET}")
            return None
        try:
            print(f"{C_BLUE}[{self.id} ACTION] Calling LLM ({self.model}) to generate plan and select tools...{C_RESET}")
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert research planner. Output ONLY the requested JSON object."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            json_str = response.choices[0].message.content.strip()
            return json.loads(json_str)

        except Exception as e:
            print(f"{C_RED}[{self.id} ERROR] LLM call or JSON parsing failed: {e}{C_RESET}")
            return None

    def execute(self, state: ResearchState) -> ResearchState:
        print(f"\n{C_ACTION}[{self.id.upper()} START] Generating plan and selecting tools (LLM Planning)...{C_RESET}")
        primary_intent = state.get("primary_intent", "general_research")
        semantic_query = state.get("semantic_query", "")

        # Read the stable constraints
        system_constraints_original = state.get("system_constraints", [])

        # --- REFINEMENT AWARENESS LOGIC ---
        is_refining = state.get('is_refining', False)
        refinement_reason = state.get('refinement_reason', '')

        prompt_modifier = ""
        # FIX 2: Create a local copy for prompt modification to avoid state corruption
        system_constraints_for_prompt = list(system_constraints_original)

        if is_refining and refinement_reason:
            print(f"{C_MAGENTA}[{self.id.upper()} REFINEMENT MODE] Strategy guided by: {refinement_reason[:80]}...{C_RESET}")

            prompt_modifier = f"""
            **REFINEMENT INSTRUCTION:** The previous attempt failed because: "{refinement_reason}".
            You MUST revise the Execution Plan and Tool Selection to specifically address this failure.

            - **CRITICAL:** The 'pubmed' tool yielded no results. You must **DEACTIVATE 'pubmed'** and **ACTIVATE 'openalex'** and **'web'** to find the necessary peer-reviewed and experimental data.
            - **Goal:** Focus the new plan on fulfilling the missing data requirements identified above.
            """
            # Append the refinement reason to the local copy, used ONLY for the prompt
            system_constraints_for_prompt.append(f"PREVIOUS_FAILURE_REASON: {refinement_reason}")

        # Pass the potentially augmented list to the prompt formatter
        full_prompt = self._format_prompt(primary_intent, semantic_query, system_constraints_for_prompt)

        if prompt_modifier:
            full_prompt = prompt_modifier + full_prompt

        llm_output = self._call_llm_and_parse(full_prompt)

        if llm_output and isinstance(llm_output, dict):
            execution_plan = llm_output.get("execution_plan", [])
            active_tools = llm_output.get("active_tools", [])

            validated_tools = [t for t in active_tools if t in self.available_tools]

            # --- POST-PROCESSING FOR REFINEMENT GUARANTEE ---
            if is_refining:
                validated_tools = [t for t in validated_tools if t != 'pubmed']
                if 'openalex' not in validated_tools: validated_tools.append('openalex')
                if 'web' not in validated_tools: validated_tools.append('web')
            # ------------------------------------------------

            if not validated_tools:
                 print(f"{C_RED}[{self.id.upper()} FAIL] LLM failed to select valid tools. Using procedural fallback.{C_RESET}")
                 validated_tools = ["arxiv", "openalex", "web", "materials"]

            if not execution_plan:
                 execution_plan = ["Execute selected tool searches.", "Generate a final synthesis report."]

            state["execution_plan"] = execution_plan
            state["active_tools"] = validated_tools

            print(f"{C_YELLOW}[{self.id.upper()} STATE] Plan generated. Active Tools selected: **{validated_tools}**{C_RESET}")

        else:
            state["execution_plan"] = ["Execute multi-tool searches.", "Generate final synthesis report."]
            state["active_tools"] = ["arxiv", "openalex", "web", "materials"]
            print(f"{C_RED}[{self.id.upper()} FAIL] LLM planning/selection failed. Using fallback plan/tools.{C_RESET}")

        print(f"{C_GREEN}[{self.id.upper()} DONE] Plan and Tool selection complete.{C_RESET}")
        return state

# ==================================================================================================
# Query Generation Agent (Section 6) - FIX APPLIED (Model Name)
# ==================================================================================================
class QueryGenerationAgent:
    """
    Agent responsible for generating tiered, tool-specific queries.
    Updated instructions ensure the problematic ArXiv category filter is removed
    during refinement and queries for deactivated tools are skipped.
    """
    # FIX 1: Corrected constructor to use LLM_MODEL variable
    def __init__(self, agent_id: str = "query_gen_agent", model: str = LLM_MODEL):
        self.id = agent_id
        self.model = model

        self.search_tiers = {
            "pubmed": ["strict", "moderate", "broad"],
            "arxiv": ["strict", "moderate", "broad"],
            "openalex": ["simple"],
            "web": ["simple"],
        }

    def _get_constraints_from_list(self, constraint_list: List[str]) -> Dict[str, Any]:
        """
        Parses the structured constraints from the list (e.g., "KEY: VALUE").
        """
        constraints = {}
        for item in constraint_list:
            try:
                key, value = item.split(":", 1)
                constraints[key.strip()] = value.strip()
            except ValueError:
                pass
        return constraints

    def _format_prompt(self, semantic_query: str, constraints: Dict[str, Any], active_tools: List[str]) -> str:
        """Constructs the prompt, dynamically incorporating the required tool structure and constraints."""

        required_tools = ", ".join([f"'{tool}'" for tool in active_tools])

        tier_instructions = "\n".join([
            f"-   **{tool.capitalize()}**: Requires keys: {', '.join([f'`{tier}`' for tier in self.search_tiers.get(tool, [])])}"
            for tool in active_tools if tool in self.search_tiers
        ])

        constraints_str = "\n".join([f"- {k}: {v}" for k, v in constraints.items()]) if constraints else "None"

        example_queries = {
            tool: {tier: f"<{tool}_{tier}_query>" for tier in self.search_tiers.get(tool, [])}
            for tool in active_tools if tool in self.search_tiers
        }
        example_output = {
            "tiered_queries": example_queries,
            "material_elements": ["CsSnI3", "Cs", "Sn", "I"]
        }

        return f"""
        You are an expert research planner. Your task is to analyze the user's research query and structured constraints,
        and generate mandatory, non-empty search queries ONLY for the currently active tools.

        **Semantic Query:** "{semantic_query}"

        **Structured Constraints (from Intent Agent):**
        ---
        {constraints_str}
        ---

        **CRITICAL: WEB SEARCH SANITIZATION (Dynamic Filtering):**
        1. **Eliminate Noise**: For 'web' queries, you must append negative search operators to exclude non-academic 'help' content.
           * Example: If the query is about 'new trends', your web search should look like: `"emerging trends in quantum computing" -inurl:help -inurl:support -inurl:login -inurl:whatsapp`
        2. **Transform Temporal Terms**: Change conversational words like 'new', 'latest', or 'recent' into high-intent academic phrases like 'state-of-the-art', 'emerging paradigms', or 'breakthroughs 2024..2025'.
        3. **Avoid Consumer Intent**: Do not generate queries that look like 'how to' or 'sign in'. Focus on 'theoretical foundations', 'experimental verification', or 'industrial applications'.

        **CRITICAL ARXİV INSTRUCTION (Read Carefully):**
        1. **DO NOT** include date keywords or ranges (e.g., '2015-2025', 'last decade', 'recent') in the ArXiv query strings. The date filter will be applied externally.
        2. **STRICT ARXİV QUERY ADJUSTMENT:** The previous ArXiv search was too narrow. For the 'strict' query, you **MUST NOT use** the previous specific category filter (`AND cat:cond-mat.mes-hall`). Focus only on precise keywords to maximize relevance, or use a broader materials science category (`cond-mat.mtrl-sci`) if needed.

        **Mandatory Query Requirements:**
        - You MUST ONLY generate queries for the following active tools: {required_tools}.
        - Ensure every query is specialized for the target tool (e.g., PubMed should look like 'MeSH' terms, Web should look like advanced Google dorks).

        **CRITICAL MATERIAL ELEMENT INSTRUCTION:**
        1.  **If the query mentions a specific chemical material or compound:**
            * Identify the most specific chemical formula (e.g., CsSnI3).
            * Your `material_elements` list MUST contain this **most specific formula** as the **FIRST** element.
            * Follow the formula with the individual constituent elements (e.g., 'Cs', 'Sn', 'I').

        **Required Output Structure:**
        The output MUST be a single JSON object (Dict) with two primary keys: **'tiered_queries'** and **'material_elements'**. The 'tiered_queries' key MUST ONLY contain the active tools.

        **Tool and Tier Specification (Only for Active Tools):**
        {tier_instructions}

        **Example Output Structure (MUST be strictly followed, replace <...> with actual queries/elements):**
        {json.dumps(example_output, indent=4)}
        """

    def _call_llm_and_parse(self, prompt: str) -> Optional[Dict[str, Any]]:
        if client is None:
            print(f"{C_RED}[{self.id} ERROR] LLM client is not initialized.{C_RESET}")
            return None
        try:
            print(f"{C_BLUE}[{self.id} ACTION] Calling LLM ({self.model}) to generate structured queries...{C_RESET}")
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research planning expert. Output ONLY the requested JSON object."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            json_str = response.choices[0].message.content.strip()
            llm_output = json.loads(json_str)

            if 'tiered_queries' not in llm_output or 'material_elements' not in llm_output:
                 raise ValueError("LLM output is missing the mandatory 'tiered_queries' or 'material_elements' key.")

            return llm_output

        except Exception as e:
            print(f"{C_RED}[{self.id} ERROR] LLM call or JSON parsing failed: {e}{C_RESET}")
            return None

    def execute(self, state: ResearchState) -> ResearchState:
        print(f"\n{C_ACTION}[{self.id.upper()} START] Generating tiered search queries and extracting elements (LLM)...{C_RESET}")

        semantic_query = state.get("semantic_query", "")
        active_tools = state.get("active_tools", [])
        if not semantic_query or not active_tools:
            print(f"{C_RED}[{self.id.upper()} FAIL] Missing semantic query or active tools. Cannot generate queries.{C_RESET}")
            return state

        system_constraints_list = state.get("system_constraints", [])
        constraints_dict = self._get_constraints_from_list(system_constraints_list)

        prompt = self._format_prompt(semantic_query, constraints_dict, active_tools)
        llm_output = self._call_llm_and_parse(prompt)

        if llm_output and llm_output.get("tiered_queries"):
            llm_queries = llm_output["tiered_queries"]
            filtered_queries = {k: v for k, v in llm_queries.items() if k in active_tools}
            state["tiered_queries"] = filtered_queries

            chemical_elements = llm_output.get("material_elements", [])
            cleaned_chemical_elements = [
                str(e).strip() for e in chemical_elements
                if e is not None and str(e).strip()
            ]

            # Merge stable constraints with new chemical elements
            final_elements_list = list(system_constraints_list)
            final_elements_list.extend(cleaned_chemical_elements)

            state["material_elements"] = final_elements_list

            if cleaned_chemical_elements:
                 state["api_search_term"] = cleaned_chemical_elements[0]
            else:
                 state["api_search_term"] = semantic_query

            query_dict = state["tiered_queries"]
            num_queries = sum(
                1 for tool_queries in query_dict.values()
                for query in tool_queries.values() if query.strip()
            )

            print(f"{C_YELLOW}[{self.id.upper()} STATE] Generated {num_queries} non-empty queries for active tools: {list(filtered_queries.keys())}{C_RESET}")
            print(f"{C_BLUE}[{self.id.upper()} STATE] Merged Elements (material_elements): {state['material_elements']}{C_RESET}")
        else:
            state["tiered_queries"] = {}
            state["material_elements"] = list(state.get("system_constraints", []))
            state["api_search_term"] = semantic_query

        print(f"{C_GREEN}[{self.id.upper()} DONE] Queries generated. Ready for tool execution.{C_RESET}")
        return state

#================ CODE DEBUG BLOCK ===============================
if __name__ == "__main__":
    # Ensure ResearchState has the 'system_constraints' key for proper testing
    from core.research_state import ResearchState # Assume ResearchState is updated

    # --- Setup ---
    print(f"\n{C_CYAN}*** STARTING AGENT CHAIN ISOLATED TESTS ***{C_RESET}")

    # TEST CASE 1: Complex Material Science Query with Time Constraint
    #test_query_1 = "A detailed review on the synthesis and bandgap stability of lead-free CsSnI3 perovskite solar cells using computational and experimental data published in the last decade."
    #test_query_1 = "A detailed review on the synthesis and bandgap stability of lead-free cesium-tin halide perovskite solar cells using computational and experimental data published in the last decade."
    test_query_1 = "Provide me a brief review on the advance materials which we use for building quantum computer."
    # TEST CASE 2: Simple Literature Review
    test_query_2 = "Recent advancements in non-invasive blood sugar monitoring for type 2 diabetes."

    current_test_query = test_query_1 # <--- CHANGE THIS TO test_query_2 TO TEST CASE 2

    # Initialize Test State (Ensure 'system_constraints' is present in initial state)
    initial_state: ResearchState = {
        "user_query": current_test_query,
        "semantic_query": current_test_query,
        "primary_intent": "",
        "system_constraints": [], # NEW KEY
        "execution_plan": [],
        "material_elements": [],
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
        "next": None,
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