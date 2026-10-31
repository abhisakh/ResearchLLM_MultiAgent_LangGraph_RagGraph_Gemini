import re
import json
from pathlib import Path
from typing import Any
from google.genai import types
from backend.core.research_state import ResearchState
from backend.core.utilities import (
    C_ACTION, C_RESET, C_GREEN, C_YELLOW, C_RED, client, LLM_MODEL, generate_content_with_retry
)

class CleanQueryAgent:
    """
    Agent responsible for cleaning the user's query and generating a semantic query.
    UPGRADED: Preserves technical terminology and routes back to Supervisor Hub.
    """
    def __init__(self, agent_id: str = "clean_query_agent", model: str = LLM_MODEL):
        self.id = agent_id
        self.model = model

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
        if client is None:
            return query

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
            response = generate_content_with_retry(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0
                )
            )
            return response.text.strip() if response.text else query
        except Exception as e:
            print(f"{C_RED}[{self.id.upper()} ERROR] Gemini generation failed: {e}{C_RESET}")
            return query
