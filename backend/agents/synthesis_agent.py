import json
import os
import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from google.genai import types

# Relative imports from the modular structure
from backend.core.research_state import ResearchState
from backend.core.utilities import (
    C_ACTION, C_RESET, C_GREEN, C_YELLOW, C_RED, C_BLUE,
    client, LLM_MODEL
)


class SynthesisAgent:
    """
    Finalized Synthesis Agent (Star Topology & Markdown Link Optimized).
    Updated for Google GenAI SDK (Gemini models).
    """
    def __init__(self, agent_id: str = "synthesis_agent", model: str = LLM_MODEL):
        self.id = agent_id
        self.model = model

    # =====================================================
    # 1. MATERIAL DATA EXTRACTION
    # =====================================================
    def _extract_material_data(self, state: Dict) -> Tuple[str, str, bool]:
        target_formula = state.get('material_formula', state.get('api_search_term', 'N/A'))
        materials_results = [d for d in state.get("raw_tool_data", []) if d.get("tool_id") == "materials_search"]
        material_data = [result.get('text', 'N/A') for result in materials_results]

        data_is_present = bool(material_data)
        summary = "\n".join(material_data) if data_is_present else f"No material property data was retrieved for {target_formula}."
        return summary, target_formula, data_is_present

    # =====================================================
    # 2. HIGH-FIDELITY REFERENCE MAPPING (FIXED FOR LINKS)
    # =====================================================
    def _extract_references(self, state: Dict) -> str:
        references = state.get("references", [])
        raw_data = state.get("raw_tool_data", [])
        noise_patterns = ['google.com/help', 'support.google', 'login', 'signin', 'signup']

        url_lookup = {}
        for entry in raw_data:
            metadata = entry.get('metadata', {})
            source_type = entry.get('source_type')
            url, ref_key = None, None

            # Standardize URL extraction based on tool type
            if source_type == 'web_search' and metadata.get('url'):
                url, ref_key = metadata['url'], f"🔗 Web Source: {metadata.get('title')}"
            elif source_type == 'arxiv' and metadata.get('pdf_url'):
                url, ref_key = metadata['pdf_url'], f"🔗 Arxiv: {metadata.get('title')}"
            elif source_type == 'pubmed':
                pmid = metadata.get('pmid')
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
                ref_key = f"📄 Journal Article: {metadata.get('title')}"
            elif source_type == 'openalex' and (metadata.get('pdf_url') or metadata.get('openalex_id')):
                url = metadata.get('pdf_url') or metadata.get('openalex_id')
                ref_key = f"🔗 OpenAlex: {metadata.get('title')}"

            if url and not any(p in url.lower() for p in noise_patterns) and ref_key:
                url_lookup[ref_key.strip()] = url.strip()

        # Build list of Markdown-formatted links
        unique_references = sorted(list(set(references)))
        formatted_list = []
        for i, ref in enumerate(unique_references, 1):
            ref_s = ref.strip()
            link = None
            for key, url in url_lookup.items():
                if ref_s.startswith(key):
                    # Format as clickable Markdown
                    link = f"[{i}] [{ref_s}]({url})"
                    break
            if not link:
                link = f"[{i}] {ref_s}"
            formatted_list.append(link)

        return "\n".join(formatted_list)

    # =====================================================
    # 3. CITATION SEQUENCING ENGINE (STRICT FIRST-APPEARANCE ORDER)
    # =====================================================
    def _reorder_citations(self, report_text: str) -> str:
        if "## References" not in report_text:
            return report_text

        body, ref_section = report_text.split("## References", 1)

        # 1. Extract citations in the exact order they first appear in the body
        found_citations = re.findall(r'\[(\d+)\]', body)
        old_to_new = {}
        new_counter = 1
        for old_id in found_citations:
            if old_id not in old_to_new:
                old_to_new[old_id] = str(new_counter)
                new_counter += 1

        # 2. Update body text with the new sequential numbers
        new_body = re.sub(r'\[(\d+)\]', lambda m: f"[{old_to_new.get(m.group(1), m.group(1))}]", body)

        # 3. Extract all reference text lines from the generated reference section
        raw_lines = [line.strip() for line in ref_section.split('\n') if line.strip() and not line.startswith('#')]
        ref_content_map = {}

        for line in raw_lines:
            match = re.match(r'^\[(\d+)\]\s*(.+)$', line)
            if match:
                ref_id, content = match.groups()
                ref_content_map[ref_id] = content.strip()
            else:
                # Fallback if the bracket was missed by the model
                if raw_lines.index(line) + 1 not in ref_content_map:
                    ref_content_map[str(raw_lines.index(line) + 1)] = line.strip()

        # 4. Rebuild the references section strictly ordered by the body's first appearance
        new_ref_list = []
        sorted_mapping = sorted(old_to_new.items(), key=lambda x: int(x[1]))

        for old_id, new_id in sorted_mapping:
            content = ref_content_map.get(old_id, f"Source content for reference {old_id}")
            clean_content = re.sub(r'\[\d+\]$', '', content).strip()
            new_ref_list.append(f"[{new_id}] {clean_content}")

        return f"{new_body.strip()}\n\n## References\n\n" + "\n\n".join(new_ref_list)

    # =====================================================
    # 4. PROMPT FORMATTING (ENHANCED LINK ENFORCEMENT)
    # =====================================================
    def _format_prompt(self, state: Dict) -> str:
        query = state.get("semantic_query", "No query provided")
        rag_context = state.get("filtered_context", "")

        # RAG Fallback
        if not rag_context or "No relevant context" in rag_context:
            raw_snippets = [f"{d.get('tool_id')}: {d.get('text')[:300]}" for d in state.get("raw_tool_data", [])[:5]]
            rag_context = "CRITICAL: Using raw snippets due to low RAG relevance:\n" + "\n".join(raw_snippets)

        formatted_references = self._extract_references(state)
        material_data_summary, target_formula, data_is_present = self._extract_material_data(state)

        heading = f"## Stability and Bandgap of {target_formula}" if data_is_present else "## Introduction and Scope of Review"

        return f"""
        [CONTEXT]
        SOURCE A (Materials API): {material_data_summary}
        SOURCE B (Literature Chunks): {rag_context}
        SOURCE C (Verified Links):
        {formatted_references}

        [OBJECTIVE]
        Generate a scientific report for: "{query}".

        [MANDATORY RULES]
        1. Support every claim with a citation like [1], [2].
        2. In the 'References' section, you MUST copy the strings from 'SOURCE C' exactly as written, including the [Title](URL) markdown.
        3. Only list sources you actually cited in the body.
        4. STRUCTURE: {heading} | Key Findings | Conclusion | References
        """

    def execute(self, state: Dict) -> Dict:
        state.setdefault("visited_nodes", []).append(self.id)

        # Intent Guardrail
        if state.get("primary_intent") == "irrelevant":
            print(f"{C_RED}[SYNTHESIS] Rejecting irrelevant query.{C_RESET}")
            state['final_report'] = "Query rejected based on scope."
            state['report_generated'] = True
            state['next'] = 'supervisor_agent'
            return state

        print(f"\n{C_ACTION}[SYNTHESIS START] Writing report with clickable links via Gemini...{C_RESET}")
        prompt = self._format_prompt(state)

        try:
            if client is None:
                raise ValueError("Gemini client is not initialized.")

            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are a scientific reporting assistant. Use Markdown for all formatting.",
                    temperature=0.1
                )
            )
            raw_report = response.text.strip() if response.text else ""

            # Post-process to fix citation order and verify links
            state['final_report'] = self._reorder_citations(raw_report)
            state['report_generated'] = True
            print(f"{C_GREEN}[SYNTHESIS DONE] Report generated successfully.{C_RESET}")
        except Exception as e:
            print(f"{C_RED}[SYNTHESIS ERROR] {e}{C_RESET}")
            state['final_report'] = "Error generating report."

        state['next'] = 'supervisor_agent'
        return state


# =====================================================
# TESTING BLOCK USING INPUT FILE: level_4_rag_output.json
# =====================================================
if __name__ == "__main__":
    print(f"{C_BLUE}==================================================")
    print("      RUNNING SYNTHESIS AGENT TEST SUITE          ")
    print(f"=================================================={C_RESET}")

    project_root = Path(__file__).resolve().parent.parent

    # Find level_4_rag_output.json in project root or subdirectories
    input_file = project_root / "level_4_rag_output.json"
    if not input_file.exists():
        matches = list(project_root.glob("**/level_4_rag_output.json"))
        if matches:
            input_file = matches[0]

    if not input_file.exists():
        print(f"{C_RED}[TEST ERROR] Input file 'level_4_rag_output.json' not found in project root or subdirectories.{C_RESET}")
    else:
        print(f"{C_YELLOW}[TEST SETUP] Loading state from input file: {input_file}{C_RESET}")

        try:
            with open(input_file, "r", encoding="utf-8") as f:
                test_state = json.load(f)

            agent = SynthesisAgent()
            updated_state = agent.execute(test_state)

            print(f"\n{C_GREEN}================ GENERATED REPORT ================{C_RESET}\n")
            print(updated_state.get("final_report", "No report generated."))
            print(f"\n{C_GREEN}=================================================={C_RESET}")

            # Define output file paths
            output_json_file = project_root / "level_5_synthesis_ouput.json"
            output_md_file = project_root / "level_5_synthesis_ouput.md"

            # Save full state to JSON
            with open(output_json_file, "w", encoding="utf-8") as f:
                json.dump(updated_state, f, indent=2, ensure_ascii=False)
            print(f"{C_GREEN}[SAVED] Updated state written to: {output_json_file}{C_RESET}")

            # Save final report text to Markdown file
            if "final_report" in updated_state:
                with open(output_md_file, "w", encoding="utf-8") as f:
                    f.write(updated_state["final_report"])
                print(f"{C_GREEN}[SAVED] Final report written to: {output_md_file}{C_RESET}")

            print(f"{C_BLUE}[TEST SUCCESS] State key 'next': {updated_state.get('next')}{C_RESET}")

        except Exception as err:
            print(f"{C_RED}[TEST FAILED] Execution raised an exception: {err}{C_RESET}")
# -------------- GPT-4o Synthesis Agent (Finalized) --------------
# import json
# import re
# from typing import Dict, List, Tuple, Any, Optional
# # Relative imports from the modular structure
# from core.research_state import ResearchState
# from core.utilities import (
#     C_ACTION, C_RESET, C_GREEN, C_YELLOW, C_RED, C_BLUE,
#     client, LLM_MODEL
# )


# class SynthesisAgent:
#     """
#     Finalized Synthesis Agent (Star Topology & Markdown Link Optimized).
#     Fixes: Link attachment in references and greedy regex for citation re-ordering.
#     """
#     def __init__(self, agent_id: str = "synthesis_agent", model: str = "gpt-4o-mini"):
#         self.id = agent_id
#         self.model = model

#     # =====================================================
#     # 1. MATERIAL DATA EXTRACTION
#     # =====================================================
#     def _extract_material_data(self, state: Dict) -> Tuple[str, str, bool]:
#         target_formula = state.get('material_formula', state.get('api_search_term', 'N/A'))
#         materials_results = [d for d in state.get("raw_tool_data", []) if d.get("tool_id") == "materials_search"]
#         material_data = [result.get('text', 'N/A') for result in materials_results]

#         data_is_present = bool(material_data)
#         summary = "\n".join(material_data) if data_is_present else f"No material property data was retrieved for {target_formula}."
#         return summary, target_formula, data_is_present

#     # =====================================================
#     # 2. HIGH-FIDELITY REFERENCE MAPPING (FIXED FOR LINKS)
#     # =====================================================
#     def _extract_references(self, state: Dict) -> str:
#         references = state.get("references", [])
#         raw_data = state.get("raw_tool_data", [])
#         noise_patterns = ['google.com/help', 'support.google', 'login', 'signin', 'signup']

#         url_lookup = {}
#         for entry in raw_data:
#             metadata = entry.get('metadata', {})
#             source_type = entry.get('source_type')
#             url, ref_key = None, None

#             # Standardize URL extraction based on tool type
#             if source_type == 'web_search' and metadata.get('url'):
#                 url, ref_key = metadata['url'], f"🔗 Web Source: {metadata.get('title')}"
#             elif source_type == 'arxiv' and metadata.get('pdf_url'):
#                 url, ref_key = metadata['pdf_url'], f"🔗 Arxiv: {metadata.get('title')}"
#             elif source_type == 'pubmed':
#                 pmid = metadata.get('pmid')
#                 url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
#                 ref_key = f"📄 Journal Article: {metadata.get('title')}"
#             elif source_type == 'openalex' and (metadata.get('pdf_url') or metadata.get('openalex_id')):
#                 url = metadata.get('pdf_url') or metadata.get('openalex_id')
#                 ref_key = f"🔗 OpenAlex: {metadata.get('title')}"

#             if url and not any(p in url.lower() for p in noise_patterns) and ref_key:
#                 url_lookup[ref_key.strip()] = url.strip()

#         # Build list of Markdown-formatted links
#         unique_references = sorted(list(set(references)))
#         formatted_list = []
#         for i, ref in enumerate(unique_references, 1):
#             ref_s = ref.strip()
#             link = None
#             for key, url in url_lookup.items():
#                 if ref_s.startswith(key):
#                     # Format as clickable Markdown
#                     link = f"[{i}] [{ref_s}]({url})"
#                     break
#             if not link:
#                 link = f"[{i}] {ref_s}"
#             formatted_list.append(link)

#         return "\n".join(formatted_list)

#     # =====================================================
#     # 3. CITATION SEQUENCING ENGINE (GREEDY REGEX FIX)
#     # =====================================================
#     def _reorder_citations(self, report_text: str) -> str:
#         if "## References" not in report_text:
#             return report_text

#         body, ref_section = report_text.split("## References", 1)

#         # 1. Identify order of citations in the text
#         found_citations = re.findall(r'\[(\d+)\]', body)
#         old_to_new, new_counter = {}, 1
#         for old_id in found_citations:
#             if old_id not in old_to_new:
#                 old_to_new[old_id] = str(new_counter)
#                 new_counter += 1

#         # 2. Update body text with new IDs
#         new_body = re.sub(r'\[(\d+)\]', lambda m: f"[{old_to_new.get(m.group(1), m.group(1))}]", body)

#         # 3. Parse references (Updated Regex to capture Markdown links properly)
#         raw_refs = re.findall(r'\[(\d+)\]\s+(.+?)(?=\n\s*\[\d+\]|\Z)', ref_section, re.DOTALL)
#         ref_content_map = {item[0]: item[1].strip() for item in raw_refs}

#         # 4. Rebuild the Reference section based on new sequence
#         new_ref_list = []
#         sorted_mapping = sorted(old_to_new.items(), key=lambda x: int(x[1]))
#         for old_id, new_id in sorted_mapping:
#             content = ref_content_map.get(old_id, "Source content missing.")
#             # Remove trailing brackets if LLM added them
#             clean_content = re.sub(r'\[\d+\]$', '', content).strip()
#             new_ref_list.append(f"[{new_id}] {clean_content}")

#         return f"{new_body.strip()}\n\n## References\n\n" + "\n\n".join(new_ref_list)

#     # =====================================================
#     # 4. PROMPT FORMATTING (ENHANCED LINK ENFORCEMENT)
#     # =====================================================
#     def _format_prompt(self, state: Dict) -> str:
#         query = state.get("semantic_query", "No query provided")
#         rag_context = state.get("filtered_context", "")

#         # RAG Fallback
#         if not rag_context or "No relevant context" in rag_context:
#             raw_snippets = [f"{d.get('tool_id')}: {d.get('text')[:300]}" for d in state.get("raw_tool_data", [])[:5]]
#             rag_context = "CRITICAL: Using raw snippets due to low RAG relevance:\n" + "\n".join(raw_snippets)

#         formatted_references = self._extract_references(state)
#         material_data_summary, target_formula, data_is_present = self._extract_material_data(state)

#         heading = f"## Stability and Bandgap of {target_formula}" if data_is_present else "## Introduction and Scope of Review"

#         return f"""
#         [CONTEXT]
#         SOURCE A (Materials API): {material_data_summary}
#         SOURCE B (Literature Chunks): {rag_context}
#         SOURCE C (Verified Links):
#         {formatted_references}

#         [OBJECTIVE]
#         Generate a scientific report for: "{query}".

#         [MANDATORY RULES]
#         1. Support every claim with a citation like [1], [2].
#         2. In the 'References' section, you MUST copy the strings from 'SOURCE C' exactly as written, including the [Title](URL) markdown.
#         3. Only list sources you actually cited in the body.
#         4. STRUCTURE: {heading} | Key Findings | Conclusion | References
#         """

#     def execute(self, state: Dict) -> Dict:
#         state.setdefault("visited_nodes", []).append(self.id)

#         # Intent Guardrail
#         if state.get("primary_intent") == "irrelevant":
#             print(f"{C_RED}[SYNTHESIS] Rejecting irrelevant query.{C_RESET}")
#             state['final_report'] = "Query rejected based on scope."
#             state['report_generated'] = True
#             state['next'] = 'supervisor_agent'
#             return state

#         print(f"\n{C_ACTION}[SYNTHESIS START] Writing report with clickable links...{C_RESET}")
#         prompt = self._format_prompt(state)

#         try:
#             response = client.chat.completions.create(
#                 model=self.model,
#                 messages=[{"role": "system", "content": "You are a scientific reporting assistant. Use Markdown for all formatting."},
#                           {"role": "user", "content": prompt}],
#                 temperature=0.1
#             )
#             raw_report = response.choices[0].message.content.strip()
#             # Post-process to fix citation order and verify links
#             state['final_report'] = self._reorder_citations(raw_report)
#             state['report_generated'] = True
#         except Exception as e:
#             print(f"{C_RED}[SYNTHESIS ERROR] {e}{C_RESET}")
#             state['final_report'] = "Error generating report."

#         state['next'] = 'supervisor_agent'
#         return state
