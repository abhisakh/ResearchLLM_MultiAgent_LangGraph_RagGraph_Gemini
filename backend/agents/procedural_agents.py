import re
import json
from typing import Any
from core.research_state import ResearchState
from core.utilities import C_ACTION, C_RESET, C_GREEN, C_YELLOW, C_RED, client

class CleanQueryAgent:
    """
    Agent responsible for cleaning the user's query and generating a semantic query.
    """
    def __init__(self):
        # Match this exactly to the key in ResearchGraph.add_node()
        self.id = "clean_query_agent"

    def execute(self, state: Any) -> Any:
        # 1. TRACK VISIT (The Breadcrumb Trail for Mermaid)
        if "visited_nodes" not in state or state["visited_nodes"] is None:
            state["visited_nodes"] = []

        state["visited_nodes"].append(self.id)

        user_query = state.get("user_query", "").strip()

        # Guard clause for empty queries
        if not user_query:
            print(f"{C_RED}[{self.id.upper()} ERROR] No user query found in state!{C_RESET}")
            state["semantic_query"] = "empty_query"
            return state

        print(f"\n{C_ACTION}[{self.id.upper()} START] Cleaning initial query: '{user_query[:100]}...'{C_RESET}")

        # 2. ROBUST CLEANING
        # Collapse whitespaces and remove special characters that might confuse downstream LLMs
        cleaned_query = " ".join(user_query.split())
        cleaned_query = re.sub(r'[?!()\[\]\"\'*]', '', cleaned_query)
        cleaned_query = re.sub(r'--', ' ', cleaned_query)
        cleaned_query = cleaned_query.strip()

        # 3. SEMANTIC GENERATION
        # This fills the key the Supervisor is looking for to move to 'intent_agent'
        semantic_query = self.generate_semantic_query(cleaned_query)

        # Update the state
        state["user_query"] = user_query
        state["semantic_query"] = semantic_query

        print(f"{C_YELLOW}[{self.id.upper()} STATE] Updated semantic_query: **'{semantic_query[:50]}...'** {C_RESET}")
        print(f"{C_GREEN}[{self.id.upper()} DONE] Query cleaning complete.{C_RESET}")

        return state


    def generate_semantic_query(self, query: str) -> str:
        """
        Calls the LLM to transform a raw, messy user prompt into a
        precise, scientifically-framed research query.
        """
        if client is None:
            return query  # Fallback to raw query if API is down

        prompt = f"""
        You are a Research Query Optimizer.
        TASK: Transform the following raw user input into a formal, concise, and semantically rich research query.

        GOALS:
        - Remove conversational filler ("Tell me about", "I'm looking for").
        - Use precise scientific terminology (e.g., instead of "cleaning water", use "wastewater remediation").
        - Ensure material formulas are correctly formatted.

        RAW INPUT: {query}

        OUTPUT: Provide ONLY the optimized string. No preamble.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini", # Use a smaller, faster model for cleaning
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100
            )
            optimized_query = response.choices[0].message.content.strip()
            # If the LLM returns nothing, fallback to the cleaned input
            return optimized_query if optimized_query else query
        except Exception as e:
            print(f"{C_RED}[CLEAN_QUERY_AGENT ERROR] LLM failed: {e}{C_RESET}")
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
