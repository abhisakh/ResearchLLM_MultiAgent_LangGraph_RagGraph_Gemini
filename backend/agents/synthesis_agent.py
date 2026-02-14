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
    UPGRADED: Integrated Fallback Logic to prevent 'Insufficient Content' loops.
    """
    def __init__(self, agent_id: str = "synthesis_agent", model: str = LLM_MODEL):
        self.id = agent_id
        self.model = model

    def _extract_material_data(self, state: ResearchState) -> tuple[str, str, bool]:
        target_formula = state.get('material_formula', state.get('api_search_term', 'N/A'))
        materials_results = [d for d in state.get("raw_tool_data", []) if d.get("tool_id") == "materials_agent"]
        material_data = [result.get('text', 'N/A') for result in materials_results]

        data_is_present = bool(material_data)
        summary = "\n".join(material_data) if data_is_present else f"No material property data was retrieved for {target_formula}."
        return summary, target_formula, data_is_present

    def _extract_references(self, state: ResearchState) -> str:
        # Original Feature: High-fidelity reference mapping with noise filtering
        references = state.get("references", [])
        raw_data = state.get("raw_tool_data", [])
        noise_patterns = ['google.com/help', 'support.google', 'login', 'signin', 'signup']

        url_lookup = {}
        for entry in raw_data:
            metadata = entry.get('metadata', {})
            source_type = entry.get('source_type')
            url, ref_key = None, None

            if source_type == 'web_search' and metadata.get('url'):
                url, ref_key = metadata['url'], f"🔗 Web Source: {metadata.get('title')}"
            elif source_type == 'arxiv' and metadata.get('pdf_url'):
                url, ref_key = metadata['pdf_url'], f"🔗 Arxiv: {metadata.get('title')}"
            elif source_type == 'pubmed':
                pmid = metadata.get('pmid')
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
                ref_key = f"📄 Journal Article: {metadata.get('title')}"
            elif source_type == 'openalex' and metadata.get('openalex_id'):
                url, ref_key = metadata['openalex_id'], f"🔗 OpenAlex: {metadata.get('title')}"

            if url and not any(p in url.lower() for p in noise_patterns) and ref_key:
                url_lookup[ref_key.strip()] = url.strip()

        unique_references = sorted(list(set(references)))
        formatted_list = []
        for i, ref in enumerate(unique_references, 1):
            ref_s = ref.strip()
            link = None
            for key, url in url_lookup.items():
                if ref_s.startswith(key):
                    link = f"[{i}] [{ref_s}]({url})"
                    break
            if link: formatted_list.append(link)
            else: formatted_list.append(f"[{i}] {ref_s}")

        return "\n".join(formatted_list)

    def _check_context_relevance(self, query: str, context: str) -> bool:
        if client is None: return True
        # If context is explicitly the 'No relevant context' fallback, fail relevance.
        if "No relevant context" in context: return False

        relevance_prompt = f"Question: {query}\nContext Snippet: {context[:500]}\nIs this relevant? YES/NO."
        try:
            response = client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": relevance_prompt}], temperature=0.0, max_tokens=5)
            return "YES" in response.choices[0].message.content.strip().upper()
        except: return True

    def _reorder_citations(self, report_text: str) -> str:
        """
        Forces a strictly sequential [1], [2], [3]... numbering
        based on the order citations first appear in the text.
        """
        # 1. Split body and references
        if "## References" in report_text:
            body, ref_section = report_text.split("## References", 1)
        else:
            return report_text

        # 2. Find all occurrences of [N] in the body
        # This maintains the order they appear to the reader
        found_citations = re.findall(r'\[(\d+)\]', body)

        # 3. Create a strict mapping based on first appearance
        old_to_new = {}
        new_counter = 1
        for old_id in found_citations:
            if old_id not in old_to_new:
                old_to_new[old_id] = str(new_counter)
                new_counter += 1

        # 4. Replace citations in the body text
        def replace_body_cite(match):
            old_num = match.group(1)
            return f"[{old_to_new.get(old_num, old_num)}]"

        new_body = re.sub(r'\[(\d+)\]', replace_body_cite, body)

        # 5. Rebuild the Reference section based ONLY on used citations
        # Map the content of the old references
        raw_refs = re.findall(r'\[(\d+)\]\s*(.+?)(?=\n\[\d+\]|\Z)', ref_section, re.DOTALL)
        ref_content_map = {item[0]: item[1] for item in raw_refs}

        new_ref_list = []
        # Sort by the NEW sequence [1, 2, 3...]
        sorted_mapping = sorted(old_to_new.items(), key=lambda x: int(x[1]))

        for old_id, new_id in sorted_mapping:
            content = ref_content_map.get(old_id, "Source content missing.")
            # Ensure we don't have nested old brackets in the content
            clean_content = re.sub(r'\[\d+\]$', '', content).strip()
            new_ref_list.append(f"[{new_id}] {clean_content}")

        return f"{new_body.strip()}\n\n## References\n\n" + "\n\n".join(new_ref_list)

    def _format_prompt(self, state: ResearchState) -> str:
        query = state.get("semantic_query", "No query provided")
        rag_context = state.get("filtered_context", "")

        # FIX: If RAG failed, pull the raw snippets as a backup to avoid empty reports
        if not rag_context or "No relevant context" in rag_context:
            raw_snippets = [f"{d.get('tool_id')}: {d.get('text')[:300]}" for d in state.get("raw_tool_data", [])[:5]]
            rag_context = "CRITICAL: RAG returned low relevance. Summarize based on these raw snippets:\n" + "\n".join(raw_snippets)

        execution_plan = "\n- ".join(state.get("execution_plan", ["No plan available"]))
        formatted_references = self._extract_references(state)
        material_data_summary, target_formula, data_is_present = self._extract_material_data(state)
        active_tools = ", ".join(state.get("active_tools", ["pubmed", "arxiv"]))

        heading = f"## Stability and Bandgap of {target_formula}" if data_is_present else "## Introduction and Scope of Review"

        ref_instruction = "## References\n- OMIT references not cited in the text.\n- Keep exact formatting."

        # Define the COMMON MANDATORY INSTRUCTIONS to ensure consistency
        mandatory_instructions = f"""
        [MANDATORY RULES]:
        1. **CITATIONS:** Every scientific claim must be supported by a citation in the format [X].
        2. **RE-INDEXING:** Use the reference numbers exactly as provided in 'SOURCE C'. Do not invent new numbers.
        3. **SELECTIVE BIBLIOGRAPHY:** Under the 'References' heading, ONLY list the sources you actually cited in the text.
        4. **SEQUENTIAL FLOW:** Aim for a professional academic flow. Ensure [1] is the first citation the reader sees.
        5. **STRUCTURE:** {heading} | Key Findings | Conclusion | References
        """

        base_prompt = f"""
        [CONTEXT]
        SOURCE A (Materials): {material_data_summary}
        SOURCE B (Research Chunks): {rag_context}
        SOURCE C (Verified Refs): {formatted_references}
        ACTIVE TOOLS: {active_tools}
        """

        if state.get('needs_refinement'):
            return base_prompt + f"""
            [FEEDBACK]: {state.get('refinement_reason')}
            [PREVIOUS DRAFT]: {state.get('final_report', 'N/A')}

            [OBJECTIVE]: REWRITE the report to address the feedback.
            {mandatory_instructions}
            """
        else:
            return base_prompt + f"""
            [OBJECTIVE]: Generate a new scientific report for: "{query}".
            [METHODOLOGY]: Explicitly mention that data was gathered using {active_tools}.
            {mandatory_instructions}
            """

    def execute(self, state: ResearchState) -> ResearchState:
        # 1. Track visit
        if "visited_nodes" not in state: state["visited_nodes"] = []
        state["visited_nodes"].append(self.id)

        if state.get("primary_intent") == "irrelevant": return state

        context = state.get("filtered_context", "")
        query = state.get("semantic_query", "")
        is_refining = state.get('needs_refinement', False)

        # 2. Fail-safe check for context quality
        if not is_refining:
            if len(context) < 200 or "No relevant context" in context:
                if not self._check_context_relevance(query, context):
                    print(f"{C_YELLOW}[SYNTHESIS] Low context relevance. Attempting fallback synthesis.{C_RESET}")

        # 3. Generate the report
        prompt = self._format_prompt(state)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Scientific writer. Ground reports in sources."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )

            # --- THE CRITICAL FIX FOR STRICT SEQUENCING ---
            raw_report = response.choices[0].message.content.strip()

            # We pass the raw LLM output through our re-indexer
            # This transforms [5], [1], [8] into [1], [2], [3]
            state['final_report'] = self._reorder_citations(raw_report)
            # ----------------------------------------------

            state['report_generated'] = True
            state['next'] = 'evaluation'

        except Exception as e:
            print(f"{C_RED}[SYNTHESIS ERROR] {e}{C_RESET}")
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