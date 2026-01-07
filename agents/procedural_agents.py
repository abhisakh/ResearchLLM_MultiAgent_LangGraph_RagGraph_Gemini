import re
from core.research_state import ResearchState
from core.utilities import C_ACTION, C_RESET, C_GREEN, C_YELLOW, C_RED

class CleanQueryAgent:
    """
    Agent responsible for cleaning the user's query and generating a semantic query.
    """
    def __init__(self):
        self.id = "clean_query_agent"

    def execute(self, state: ResearchState) -> ResearchState:
        """
        Processes the user query and fills the 'semantic_query' in the state.
        """
        user_query = state.get("user_query", "").strip()
        print(f"\n{C_ACTION}[{self.id.upper()} START] Cleaning initial query: '{user_query[:100]}...'{C_RESET}")

        # Basic cleaning: collapse multiple whitespaces
        cleaned_query = " ".join(user_query.split())

        # --- Add more robust cleaning here ---
        # Example: Remove common non-semantic punctuation for cleaner LLM input
        cleaned_query = re.sub(r'[?!()\[\]\"\'*]', '', cleaned_query)
        cleaned_query = re.sub(r'--', ' ', cleaned_query)
        cleaned_query = cleaned_query.strip()
        # ------------------------------------

        semantic_query = self.generate_semantic_query(cleaned_query)

        # Update the state with the cleaned versions
        state["user_query"] = cleaned_query
        state["semantic_query"] = semantic_query

        print(f"{C_YELLOW}[{self.id.upper()} STATE] Updated semantic_query: **'{semantic_query[:50]}...'** {C_RESET}")
        print(f"{C_GREEN}[{self.id.upper()} DONE] Query cleaning complete.{C_RESET}")
        return state

    def generate_semantic_query(self, query: str) -> str:
        # Placeholder for future NLP enhancements.
        return query


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