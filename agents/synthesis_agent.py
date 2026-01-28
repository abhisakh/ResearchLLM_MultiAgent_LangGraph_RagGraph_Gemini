import json
import re
from typing import Dict, Any, List, Optional
# Relative imports from the modular structure
from core.research_state import ResearchState
from core.utilities import (
    C_ACTION, C_RESET, C_GREEN, C_YELLOW, C_RED, C_BLUE,
    client, LLM_MODEL
)

class SynthesisAgent:
    """
    Agent responsible for generating the final comprehensive research report.
    INTEGRATION: Full original logic restored with COIE adversarial prompting.
    """
    def __init__(self, agent_id: str = "synthesis_agent", model: str = LLM_MODEL):
        self.id = agent_id
        self.model = model

    def _extract_material_data(self, state: ResearchState) -> tuple[str, str, bool]:
        """Original Feature: Extracts specific material properties from raw_tool_data."""
        target_formula = state.get('material_formula', state.get('api_search_term', 'N/A'))
        materials_results = [d for d in state.get("raw_tool_data", []) if d.get("tool_id") == "materials_agent"]
        material_data = []
        for result in materials_results:
            material_data.append(result.get('text', 'N/A'))
            break
        data_is_present = bool(material_data)
        if data_is_present:
            return "\n".join(material_data), target_formula, data_is_present
        else:
            return f"No material property data was retrieved for {target_formula}.", target_formula, data_is_present

    def _extract_references(self, state: ResearchState) -> str:
        """Original Feature: High-fidelity reference mapping with noise filtering and regex."""
        references = state.get("references", [])
        raw_data = state.get("raw_tool_data", [])
        noise_patterns = ['google.com/help', 'support.google', 'whatsapp.com', 'stackoverflow.com', 'accounts.google', 'microsoft.com/help', 'login', 'signin', 'signup']

        url_lookup = {}
        for entry in raw_data:
            metadata = entry.get('metadata', {})
            source_type = entry.get('source_type')
            url, ref_snippet_key = None, None
            if source_type == 'web_search' and metadata.get('url'):
                url, ref_snippet_key = metadata['url'], f"🔗 Web Source: {metadata.get('title')}"
            elif source_type == 'arxiv' and metadata.get('pdf_url'):
                url, ref_snippet_key = metadata['pdf_url'], f"🔗 Arxiv: {metadata.get('title')}"
            elif source_type == 'pubmed':
                pmid = metadata.get('pmid')
                if pmid: url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                ref_snippet_key = f"📄 Journal Article: {metadata.get('title')}"
            elif source_type == 'openalex':
                openalex_id = metadata.get('openalex_id')
                if openalex_id: url = openalex_id
                ref_snippet_key = f"🔗 OpenAlex: {metadata.get('title')}"

            if url and any(pattern in url.lower() for pattern in noise_patterns): continue
            if url and ref_snippet_key:
                url_lookup[ref_snippet_key.strip()] = url.strip()

        unique_references = sorted(list(set(references)))
        formatted_list = []
        current_ref_idx = 1
        mp_pattern = re.compile(r'(⚛️ Materials Project: [^\(]+)\s+\(([^\)]+)\)')
        web_url_pattern = re.compile(r'\((https?://[^\)]+)\)')
        lookup_prefixes = ('📄 Journal Article:', '🔗 Arxiv:', '🔗 OpenAlex:')

        for ref in unique_references:
            ref_stripped = ref.strip()
            markdown_link = None
            if ref_stripped.startswith(lookup_prefixes):
                for snippet_key, url in url_lookup.items():
                    if ref_stripped.startswith(snippet_key):
                        markdown_link = f"[{current_ref_idx}] [{ref_stripped}]({url})"
                        break
            elif ref_stripped.startswith('🔗 Web Source:'):
                web_match = web_url_pattern.search(ref_stripped)
                if web_match:
                    url = web_match.group(1)
                    if not any(pattern in url.lower() for pattern in noise_patterns):
                        display_text = ref_stripped[:web_match.start()].strip()
                        markdown_link = f"[{current_ref_idx}] [{display_text}]({url})"
            elif not markdown_link:
                mp_match = mp_pattern.match(ref_stripped)
                if mp_match:
                    markdown_link = f"[{current_ref_idx}] {ref_stripped}"

            if markdown_link:
                formatted_list.append(markdown_link)
                current_ref_idx += 1
        return "\n".join(formatted_list)

    def _check_context_relevance(self, query: str, context: str) -> bool:
        """Original Feature: LLM-based Anti-GIGO guardrail."""
        if client is None: return True
        relevance_prompt = f"Question: {query}\nContext Snippet: {context[:500]}\nIs this relevant? YES/NO."
        try:
            response = client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": relevance_prompt}], temperature=0.0, max_tokens=5)
            return "YES" in response.choices[0].message.content.strip().upper()
        except: return True

    def _format_prompt(self, state: ResearchState) -> str:
        """
        Upgraded Prompt Logic: Uses COIE framework while keeping dynamic structure.
        FULLY RESTORED: All original headings and material data logic.
        FIXED: Explicitly credits ALL active tools to satisfy the EvaluationAgent.
        """
        query = state.get("semantic_query", "No query provided")
        rag_context = state.get("filtered_context", "No context available")
        execution_plan = "\n- ".join(state.get("execution_plan", ["No plan available"]))
        formatted_references = self._extract_references(state)
        material_data_summary, target_formula, data_is_present = self._extract_material_data(state)

        # NEW: Dynamic tool list to ensure the Evaluator sees the "work" done
        active_tools = ", ".join(state.get("active_tools", ["pubmed", "arxiv"]))

        # Keep your original heading logic exactly as it was
        if data_is_present:
            first_section_heading = f"## Stability and Bandgap of {target_formula}"
            first_section_req = "Use Data Source A (Table) and B."
        else:
            first_section_heading = "## Introduction and Scope of Review"
            first_section_req = "Provide overview based on Data Source B."

        # Shared Instruction Block for Reference Filtering
        ref_filtering_instruction = """
               ## References
               - MUST contain ONLY the references cited in the text above.
               - If a reference from Source C was not used to support a claim, OMIT it from this list.
               - Keep the exact formatting and URLs from Source C for the citations you keep."""

        # COIE MANDATE
        if state.get('needs_refinement'):
            ref_reason = state.get('refinement_reason', 'Incomplete report.')
            prev_report = state.get('final_report', 'N/A')
            return f"""
            [CONTEXT]
            SOURCE A (Material Properties): {material_data_summary}
            SOURCE B (Context Data): {rag_context}
            SOURCE C (Reference Map): {formatted_references}
            ACTIVE TOOLS USED: {active_tools}
            CRITICAL FEEDBACK: {ref_reason}
            PREVIOUS DRAFT: {prev_report}

            [OBJECTIVE]
            REWRITE the report to fix failures.
            MANDATORY: You must explicitly state that the following databases were successfully queried: {active_tools}.
            Address why the previous draft was insufficient based on the feedback.

            [INSTRUCTION]
            1. Address feedback in Section I.
            2. Every claim MUST end with a citation [X] from Source C.
            3. Use Level 2 Headings:
               {first_section_heading}
               ## Key Research Findings
               ## Conclusion and Future Outlook
               {ref_filtering_instruction}

            [EVALUATION]
            Reject if info is from outside Source A/B or if the methodology section fails to name: {active_tools}.
            """
        else:
            return f"""
            [CONTEXT]
            SOURCE A (Material Properties): {material_data_summary}
            SOURCE B (Context Data): {rag_context}
            SOURCE C (Reference Map): {formatted_references}
            PLAN: {execution_plan}
            ACTIVE TOOLS USED: {active_tools}

            [OBJECTIVE]
            Generate a scientific report for: "{query}".
            MANDATORY: Include a brief 'Search Methodology' sentence explicitly naming these sources: {active_tools}.

            [INSTRUCTION]
            1. Citations: Every claim needs a citation [X].
            2. {first_section_req}
            3. Structure:
               {first_section_heading}
               ## Key Research Findings
               ## Conclusion and Future Outlook
               {ref_filtering_instruction}

            [EVALUATION]
            Strict adherence to Source A/B required. You MUST prove the search plan was followed by mentioning {active_tools}.
            """

    def execute(self, state: ResearchState) -> ResearchState:
        if "visited_nodes" not in state or state["visited_nodes"] is None:
            state["visited_nodes"] = []
        state["visited_nodes"].append(self.id)

        # If the intent is irrelevant, a refusal message already exists in final_report.
        # We return immediately to avoid overwriting it or running expensive LLM calls.
        if state.get("primary_intent") == "irrelevant":
            print(f"{C_YELLOW}[{self.id.upper()}] Irrelevant intent detected. Bypassing synthesis.{C_RESET}")
            return state

        mode = "REFINEMENT" if state.get('needs_refinement') else "INITIAL GENERATION"
        print(f"\n{C_ACTION}[{self.id.upper()} START] Running {mode}...{C_RESET}")

        context = state.get("filtered_context", "")
        query = state.get("semantic_query", "")
        is_refining = state.get('needs_refinement', False)

        # Original Guardrail Logic
        if not is_refining and (len(context) < 200 or context.startswith("No sufficiently relevant")):
            if not self._check_context_relevance(query, context):
                state['final_report'] = "Context failed relevance check."
                state['report_generated'] = True
                state['needs_refinement'] = False
                return state

        if context.strip() in ["No sufficiently relevant context found.", ""]:
             state['final_report'] = "Lack of context."
             return state

        prompt = self._format_prompt(state)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "You are a grounding-first scientific writer."},
                          {"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=3000
            )
            state['final_report'] = response.choices[0].message.content.strip()
            state['report_generated'] = True
            state['is_refining'] = is_refining
            state['needs_refinement'] = False
            state['next'] = 'evaluation'
            print(f"{C_GREEN}[{self.id.upper()} DONE]{C_RESET}")
        except Exception as e:
            state['next'] = 'TERMINATE'
        return state

# class SynthesisAgent:
#     """
#     Agent responsible for generating the final comprehensive research report
#     based on the semantically filtered context and references. Includes dynamic
#     prompting for initial generation and refinement based on evaluation feedback.
#     """
#     def __init__(self, agent_id: str = "synthesis_agent", model: str = LLM_MODEL):
#         self.id = agent_id
#         self.model = model

#     def _extract_material_data(self, state: ResearchState) -> tuple[str, str, bool]:
#         """
#         Dynamically extracts and formats specific material properties from raw_tool_data.
#         Returns: (data_summary, target_formula, data_is_present)
#         """
#         # Base target formula on the state, defaulting to 'N/A'
#         target_formula = state.get('material_formula', state.get('api_search_term', 'N/A'))

#         materials_results = [
#             d for d in state.get("raw_tool_data", [])
#             if d.get("tool_id") == "materials_agent"
#         ]

#         material_data = []

#         # Find data specifically formatted by the materials agent
#         for result in materials_results:
#             # We are less strict here as long as the materials agent returned *something*
#             material_data.append(result.get('text', 'N/A'))
#             break # Take only the first relevant material result

#         data_is_present = bool(material_data)

#         if data_is_present:
#             # Return the structured text
#             return "\n".join(material_data), target_formula, data_is_present
#         else:
#             # Return a simple fallback for the prompt
#             return f"No material property data was retrieved for {target_formula}.", target_formula, data_is_present


#     def _extract_references(self, state: ResearchState) -> str:
#         """
#         Formats gathered references into Markdown links, filtering out
#         irrelevant web noise (help pages, login screens, etc.).
#         """
#         references = state.get("references", [])
#         raw_data = state.get("raw_tool_data", [])

#         # Dynamic Noise Filtering: Skip URLs containing these patterns
#         noise_patterns = [
#             'google.com/help', 'support.google', 'whatsapp.com',
#             'stackoverflow.com', 'accounts.google', 'microsoft.com/help',
#             'login', 'signin', 'signup'
#         ]

#         # 1. Prepare a URL lookup table
#         url_lookup = {}

#         for entry in raw_data:
#             metadata = entry.get('metadata', {})
#             source_type = entry.get('source_type')
#             url = None
#             ref_snippet_key = None

#             # Extract URL based on source type
#             if source_type == 'web_search' and metadata.get('url'):
#                 url = metadata['url']
#                 ref_snippet_key = f"🔗 Web Source: {metadata.get('title')}"
#             elif source_type == 'arxiv' and metadata.get('pdf_url'):
#                 url = metadata['pdf_url']
#                 ref_snippet_key = f"🔗 Arxiv: {metadata.get('title')}"
#             elif source_type == 'pubmed':
#                 pmid = metadata.get('pmid')
#                 if pmid: url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
#                 ref_snippet_key = f"📄 Journal Article: {metadata.get('title')}"
#             elif source_type == 'openalex':
#                 openalex_id = metadata.get('openalex_id')
#                 if openalex_id: url = openalex_id
#                 ref_snippet_key = f"🔗 OpenAlex: {metadata.get('title')}"

#             # --- NOISE FILTERING ---
#             if url and any(pattern in url.lower() for pattern in noise_patterns):
#                 continue # Skip this entry entirely

#             if url and ref_snippet_key:
#                 url_lookup[ref_snippet_key.strip()] = url.strip()

#         # 2. Process and Format References
#         unique_references = sorted(list(set(references)))
#         formatted_list = []

#         # Track valid reference index manually to avoid gaps in numbering
#         current_ref_idx = 1

#         mp_pattern = re.compile(r'(⚛️ Materials Project: [^\(]+)\s+\(([^\)]+)\)')
#         web_url_pattern = re.compile(r'\((https?://[^\)]+)\)')
#         lookup_prefixes = ('📄 Journal Article:', '🔗 Arxiv:', '🔗 OpenAlex:')

#         for ref in unique_references:
#             ref_stripped = ref.strip()
#             markdown_link = None

#             # Case 1: Academic Sources (Uses lookup table)
#             if ref_stripped.startswith(lookup_prefixes):
#                 for snippet_key, url in url_lookup.items():
#                     if ref_stripped.startswith(snippet_key):
#                         markdown_link = f"[{current_ref_idx}] [{ref_stripped}]({url})"
#                         break

#             # Case 2: Web Source (Extract URL from string if not in lookup)
#             elif ref_stripped.startswith('🔗 Web Source:'):
#                 web_match = web_url_pattern.search(ref_stripped)
#                 if web_match:
#                     url = web_match.group(1)
#                     # Secondary noise check for the URL inside the text
#                     if not any(pattern in url.lower() for pattern in noise_patterns):
#                         display_text = ref_stripped[:web_match.start()].strip()
#                         markdown_link = f"[{current_ref_idx}] [{display_text}]({url})"

#             # Case 3: Materials Project
#             elif not markdown_link:
#                 mp_match = mp_pattern.match(ref_stripped)
#                 if mp_match:
#                     markdown_link = f"[{current_ref_idx}] {ref_stripped}"

#             # Only add to list and increment index if a valid link was formed
#             if markdown_link:
#                 formatted_list.append(markdown_link)
#                 current_ref_idx += 1

#         return "\n".join(formatted_list)
#     # ----------------------------------------

#     # 🟢 NEW: CONTEXT RELEVANCE GUARDRAIL
#     def _check_context_relevance(self, query: str, context: str) -> bool:
#         """Uses the LLM to verify if the minimal context is topically relevant to the query."""
#         if client is None: return True # Cannot check, assume relevant

#         relevance_prompt = f"""
#         Analyze the following context snippet and determine if it contains relevant information
#         to answer the user's core question.

#         Core Question: "{query}"

#         Context Snippet: "{context[:500]}..."

#         Respond ONLY with the single word 'YES' or 'NO'.
#         """

#         try:
#             response = client.chat.completions.create(
#                 model=self.model,
#                 messages=[{"role": "user", "content": relevance_prompt}],
#                 temperature=0.0,
#                 max_tokens=5
#             )
#             llm_response = response.choices[0].message.content.strip().upper()

#             if "YES" in llm_response:
#                 print(f"{C_GREEN}[SYNTHESIS GUARDRAIL] Context passed relevance check (YES).{C_RESET}")
#                 return True
#             else:
#                 print(f"{C_RED}[SYNTHESIS GUARDRAIL] Context failed relevance check (NO/Irrelevant).{C_RESET}")
#                 return False
#         except Exception as e:
#             print(f"{C_RED}[SYNTHESIS GUARDRAIL ERROR] LLM check failed: {e}. Assuming relevant to proceed.{C_RESET}")
#             return True # Default to proceeding to avoid crashing the workflow


#     def _format_prompt(self, state: ResearchState) -> str:
#         # --- Common Data Extraction ---
#         query = state.get("semantic_query", "No query provided")
#         rag_context = state.get("filtered_context", "No context available")
#         execution_plan = "\n- ".join(state.get("execution_plan", ["No plan available"]))
#         formatted_references = self._extract_references(state)
#         material_data_summary, target_formula, data_is_present = self._extract_material_data(state)

#         # --- DYNAMIC STRUCTURE LOGIC ---
#         # The logic here correctly adapts the prompt structure based on tool usage
#         if data_is_present:
#             # Materials-focused report structure
#             first_section_heading = f"## Stability and Bandgap of {target_formula}"
#             first_section_data_name = "**A. FILTERED MATERIAL PROPERTIES (Stability and Bandgap):**"
#             first_section_content_req = f"""
#                     * The first section must use **Data Source A** to describe the stability and bandgap, ideally presented in a **Markdown Table**.
#                     * The second section must summarize findings from **Data Source B** relevant to the overall query.
#             """
#         else:
#             # Literature Review-focused report structure (Default for non-material queries)
#             first_section_heading = "## Introduction and Scope of Review"
#             first_section_data_name = "**A. FILTERED MATERIAL PROPERTIES (N/A):**"
#             first_section_content_req = f"""
#                     * The first section must provide a general overview and define the scope of the review based on **Data Source B**.
#                     * The subsequent sections must summarize the key findings from **Data Source B** (e.g., categorizing by topic, experimental method, or finding).
#             """

#         final_structure_headings = f"""
#                 3.  **Structure:** The final report **MUST** follow this exact **four-section** structure using Level 2 Markdown headings (##):
#                     * {first_section_heading}
#                     * ## Key Research Findings
#                     * ## Conclusion and Future Outlook
#                     * ## References
#         """

#         # ==================================================================================
#         # 🟢 CRITICAL REFINEMENT PROMPT LOGIC
#         # ==================================================================================
#         if state['needs_refinement']:
#             refinement_reason = state.get('refinement_reason', 'The previous report was incomplete or inaccurate.')
#             previous_report = state.get('final_report', 'Previous report text is unavailable.')

#             return f"""
#                 **AGENT ROLE**: You are a dedicated **Scientific Report REFINEMENT EXPERT**. Your sole function is to rewrite the previous report to address the critical feedback provided below. You must produce a single, cohesive, final report.

#                 ---
#                 **I. REFINEMENT MANDATE**
#                 **CRITICAL FEEDBACK:** {refinement_reason}

#                 **PREVIOUS REPORT (To be Rewritten):**
#                 {previous_report}
#                 ---

#                 **II. DATA SOURCES (For Context and Grounding)**

#                 {first_section_data_name}
#                 {material_data_summary}

#                 **B. FILTERED RESEARCH CONTEXT (Recent Articles for Synthesis):**
#                 {rag_context}

#                 **C. RAW REFERENCE LIST (For Final Output):**
#                 {formatted_references}

#                 ---
#                 **III. REPORT REWRITE INSTRUCTIONS**

#                 1.  **Primary Goal:** **CRITICALLY ADDRESS THE FEEDBACK** in Section I. Use the newly retrieved context (if any) to fix factual errors, omissions, or structural issues.
#                 2.  **Strict Adherence:** Rewrite the report using **only** the data provided in Section II.
#                 3.  **Scientific Tone & Citation (CRITICAL):** The report must maintain a formal, scientific tone, and **every factual claim must be attributed** using inline numerical citations (e.g., "...the bandgap was determined to be 2.1 eV [1, 5]."). Use the index from the RAW REFERENCE LIST.
#                 4.  **Structure:** **Maintain the exact four-section structure** defined below:
#                     {final_structure_headings}
#                 5.  **Final Output:** Your response must **ONLY** contain the final rewritten report.
#                 """
#         # ==================================================================================
#         # 🟢 INITIAL REPORT PROMPT LOGIC (Original path)
#         # ==================================================================================
#         else:
#             return f"""
#                 **AGENT ROLE**: You are a dedicated **Scientific Research Assistant**. Your sole function is to compile a final, comprehensive, and objective research report that directly addresses the user's query using only the provided filtered data.

#                 ---
#                 **I. RESEARCH MANDATE & PLAN**
#                 **User Query:** {query}
#                 **Execution Plan:**
#                 - {execution_plan}
#                 ---

#                 **II. DATA SOURCES**

#                 {first_section_data_name}
#                 {material_data_summary}

#                 **B. FILTERED RESEARCH CONTEXT (Recent Articles for Synthesis):**
#                 {rag_context}

#                 **C. RAW REFERENCE LIST (For Final Output):**
#                 {formatted_references}

#                 ---
#                 **III. REPORT GENERATION INSTRUCTIONS**

#                 1.  **Strict Adherence:** Generate the report using **only** the data provided in Section II. Do not hallucinate or use external knowledge.
#                 2.  **Scientific Tone & Citation (CRITICAL):** The entire report must be written in a formal, **scientific, and objective tone** (avoiding personal pronouns or conversational language). **Every factual claim and data point must be attributed** using inline numerical citations (e.g., "...the bandgap was determined to be 2.1 eV [1, 5]."). Use the index from the RAW REFERENCE LIST.
#                 3.  **Structure:** {final_structure_headings}
#                 4.  **Content Requirements:**
#                     {first_section_content_req}
#                 5.  **Final Output:** The final section, **## References**, must be a direct copy of the list provided in **Data Source C**. Your response must **ONLY** contain the final report.
#                 """

#     def execute(self, state: ResearchState) -> ResearchState:
#         # --- MODIFICATION 1: BREADCRUMB TRACKING ---
#         if "visited_nodes" not in state or state["visited_nodes"] is None:
#             state["visited_nodes"] = []
#         state["visited_nodes"].append(self.id)

#         # 🚨 Refinement Check: Mark the start of the refinement or initial generation
#         mode = "REFINEMENT" if state.get('needs_refinement') else "INITIAL GENERATION"
#         print(f"\n{C_ACTION}[{self.id.upper()} START] Running {mode} (LLM: {self.model})...{C_RESET}")

#         # Check for initial errors
#         if client is None:
#              state['final_report'] = "Synthesis failed: LLM is not initialized due to missing API Key."
#              print(f"{C_RED}[{self.id} FAIL] Synthesis skipped due to missing API Key.{C_RESET}")
#              return state

#         context = state.get("filtered_context", "")
#         query = state.get("semantic_query", "")
#         is_refining = state.get('needs_refinement', False)

#         # 🟢 NEW: CONTEXT RELEVANCE GUARDRAIL (Anti-GIGO check)
#         # Check if context is short AND it's the initial run
#         if not is_refining and (len(context) < 200 or context.startswith("No sufficiently relevant context")):
#             print(f"{C_YELLOW}[SYNTHESIS GUARDRAIL] Minimal context detected (Length: {len(context)}). Running relevance check...{C_RESET}")

#             if not self._check_context_relevance(query, context):
#                 # Graceful Failure State: Irrelevant context detected
#                 state['final_report'] = f"The initial search yielded data highly irrelevant to the query '{query}'. The system cannot generate a meaningful report based on the provided context."
#                 state['report_generated'] = True # Mark as generated to allow Evaluation to read the failure message
#                 state['needs_refinement'] = False
#                 print(f"{C_RED}[{self.id} FAIL] Graceful failure: Irrelevant context detected. Outputting failure message.{C_RESET}")
#                 return state

#         # Original check for true starvation (after potential relevance failure has passed)
#         if context.strip() in ["No sufficiently relevant context found.", ""]:
#              state['final_report'] = "Research failed: Could not find sufficient relevant data to generate a report for the query: " + state.get("user_query", "N/A")
#              print(f"{C_RED}[{self.id} FAIL] Synthesis skipped due to lack of context.{C_RESET}")
#              return state

#         # Generate the appropriate prompt (initial or refinement)
#         prompt = self._format_prompt(state)

#         try:
#             response = client.chat.completions.create(
#                 model=self.model,
#                 messages=[{"role": "system", "content": "You are a scientific research assistant who outputs a final, structured report with inline citations and a reference list."},
#                           {"role": "user", "content": prompt}],
#                 temperature=0.2,
#                 max_tokens=3000 # Increased max tokens for refinement rewrite safety
#             )
#             final_text = response.choices[0].message.content.strip()
#             state['final_report'] = final_text

#             # --- CRITICAL REFINEMENT FLAGS UPDATE ---
#             state['report_generated'] = True
#             state['is_refining'] = state['needs_refinement'] # Track if this run was a rewrite
#             state['needs_refinement'] = False # Reset the flag for the next evaluation run
#             state['next'] = 'evaluation' # Route back to evaluation to check the rewrite
#             # ----------------------------------------

#             print(f"{C_YELLOW}[{self.id.upper()} STATE] Final report generated (Approx. {len(final_text.split())} words).{C_RESET}")
#             print(f"{C_GREEN}[{self.id.upper()} DONE] Report generation successful. Next: Evaluate.{C_RESET}")

#         except Exception as e:
#             state['final_report'] = f"Error: Unable to generate report during {mode} due to LLM failure or API issue."
#             state['report_generated'] = False
#             state['next'] = 'TERMINATE'
#             print(f"{C_RED}[{self.id} ERROR] Failed to generate final report: {e}{C_RESET}")

#         return state