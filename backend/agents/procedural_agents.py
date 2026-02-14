import re
import json
from typing import Any
from core.research_state import ResearchState
from core.utilities import C_ACTION, C_RESET, C_GREEN, C_YELLOW, C_RED, client

class CleanQueryAgent:
    """
    Agent responsible for cleaning the user's query and generating a semantic query.
    UPGRADED: Preserves technical terminology and routes back to Supervisor Hub.
    """
    def __init__(self, agent_id: str = "clean_query_agent"):
        self.id = agent_id

    def execute(self, state: ResearchState) -> ResearchState:
        # 1. TRACK VISIT
        state.setdefault("visited_nodes", []).append(self.id)

        user_query = state.get("user_query", "").strip()
        if not user_query:
            state["semantic_query"] = "empty_query"
            state["next"] = "supervisor_agent"
            return state

        print(f"\n{C_ACTION}[{self.id.upper()} START] Cleaning initial query: '{user_query[:60]}...'{C_RESET}")

        # 2. ROBUST CLEANING (Preserve chemical dashes/dots)
        cleaned_query = " ".join(user_query.split())
        cleaned_query = re.sub(r'[?!()\[\]\"\'*]', '', cleaned_query)
        cleaned_query = cleaned_query.strip()

        # 3. SEMANTIC GENERATION
        semantic_query = self.generate_semantic_query(cleaned_query)

        # 4. UPDATE STATE & ROUTE
        state["semantic_query"] = semantic_query
        state["next"] = "supervisor_agent" # HUB ROUTE

        print(f"{C_YELLOW}[{self.id.upper()} STATE] Updated semantic_query: **'{semantic_query}'** {C_RESET}")
        return state

    def generate_semantic_query(self, query: str) -> str:
        if client is None: return query

        prompt = f"""
        You are a Research Query Optimizer.
        TASK: Transform raw user input into a formal, concise, and semantically rich research query.

        CRITICAL RESTRAINT:
        - Preserve unique or unrecognized terminology.
        - Do NOT map unknown terms to common concepts (e.g., if you see a niche term, keep it).
        - Only fix obvious spelling errors in non-technical words.
        - Format chemical formulas correctly (e.g., CH3NH3PbI3).

        RAW INPUT: {query}

        OUTPUT: Provide ONLY the optimized string. No preamble.
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content.strip() or query
        except Exception:
            return query

    # def generate_semantic_query(self, query: str) -> str:
    #     """
    #     Placeholder for semantic enrichment. In a full implementation,
    #     this would call a 'CleanQuery' LLM chain.
    #     """
    #     # For now, we return the cleaned query to satisfy the Supervisor's routing logic
    #     return query


#================ CODE DEBUG BLOCK (No changes needed, as it now calls execute()) ===============================
if __name__ == "__main__":
    from core.research_state import ResearchState
    from core.utilities import C_CYAN, C_RESET, C_GREEN, C_YELLOW, C_RED

    print(f"\n{C_CYAN}*** STARTING PROCEDURAL AGENT ISOLATED TESTS ***{C_RESET}")

    # --- Test Case 1: Simple cleanup ---
    test_query_1 = "  What's the synthesis route for LiFePO4? (Please ignore the brackets)   "

    # --- Test Case 2: Query with unwanted characters/spacing ---
    test_query_2 = "Can we use g-raphene?! 10.00% doping of Ni in ZnO--what's the bandgap?"

    current_test_query = test_query_2

    # 1. Initialize Test State
    initial_state: ResearchState = {
        "user_query": current_test_query,
        "semantic_query": "",
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

    # --- Test CleanQueryAgent ---
    print(f"\n{C_YELLOW}--- Testing CleanQueryAgent ---{C_RESET}")

    try:
        clean_agent = CleanQueryAgent()
        # This call will now succeed
        state_after_clean = clean_agent.execute(initial_state)

        original_query = initial_state.get("user_query")
        cleaned_query = state_after_clean.get("semantic_query")

        print(f"{C_GREEN}[RESULT] Original Query: '{original_query.strip()}'{C_RESET}")
        print(f"{C_GREEN}[RESULT] Cleaned Query:  '{cleaned_query}'{C_RESET}")

        # Verification Checks (Adjusted to match the added cleaning logic)
        assert cleaned_query.strip() == cleaned_query, "Cleaned query has leading/trailing whitespace."
        assert "!" not in cleaned_query, "Cleaned query still contains '!'."
        assert "?" not in cleaned_query, "Cleaned query still contains '?'."


        print(f"{C_GREEN}[VERIFY] CleanQueryAgent passed all checks.{C_RESET}")

    except NameError:
        print(f"{C_RED}[FAIL] NameError: Ensure 'CleanQueryAgent' class is defined in this file.{C_RESET}")
    except AttributeError as e:
        # This error should now be fixed
        print(f"{C_RED}[FAIL] AttributeError: Still encountering error: {e}{C_RESET}")
    except AssertionError as e:
        print(f"{C_RED}[FAIL] Assertion Failed: {e}{C_RESET}")
    except Exception as e:
        print(f"{C_RED}[FAIL] An unexpected error occurred: {e}{C_RESET}")

    print(f"\n{C_CYAN}*** PROCEDURAL AGENT TESTS COMPLETE ***{C_RESET}")
    print(json.dumps(state_after_clean, indent=4))
