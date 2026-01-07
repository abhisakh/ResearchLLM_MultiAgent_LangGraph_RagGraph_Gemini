################################################################################
# FILE: main_ui.py
# UPGRADE: Iron-Clad Sticky Sides | Large Typography | Sanitized Logic Engine
################################################################################

import streamlit as st
import requests
import json
import uuid
import base64
import os

# -----------------------------------------------------------------------------
# 1. IMAGE HANDLER (Base64 for Local ai.jpg)
# -----------------------------------------------------------------------------
def get_base64_of_bin_file(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

img_base64 = get_base64_of_bin_file("ai.jpg")
logo_html = f"data:image/jpg;base64,{img_base64}" if img_base64 else ""

# -----------------------------------------------------------------------------
# 2. MASTER CSS (Headers, Logic, and Sticky Positioning)
# -----------------------------------------------------------------------------
API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Research Assistant LLM", page_icon="🔬", layout="wide")

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');

    /* 1. APP RESET & SCROLL LOCK */
    header[data-testid="stHeader"], [data-testid="stDecoration"] {{ display: none !important; }}
    .stApp {{ background-color: #0B0E14 !important; overflow: hidden !important; height: 100vh; }}

    /* 2. THE 25% FIXED HEADER */
    .fixed-header {{
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 25vh;
        background-color: #0D1117;
        z-index: 9999;
        border-bottom: 2px solid #30363d;
        display: flex; justify-content: center; align-items: center; gap: 40px;
    }}
    .header-logo {{ width: 450px; height: 200px; border-radius: 15px; border: 2px solid #98D8C8; object-fit: contain; }}
    .header-title {{ font-size: 48px; font-weight: 800; color: #98D8C8; margin: 0; }}

    /* 3. IRON-CLAD STICKY COLUMN SYSTEM */
    [data-testid="stHorizontalBlock"] {{ margin-top: 25vh !important; }}

    /* LEFT COLUMN LOCK - Directly targeting inner div for stability */
    [data-testid="column"]:nth-child(1) > div {{
        position: fixed !important;
        width: 22vw !important;
        top: 30vh !important; /* Positioned below header */
        left: 2vw;
        z-index: 1000;
    }}

    /* RIGHT COLUMN LOCK - Directly targeting inner div for stability */
    [data-testid="column"]:nth-child(3) > div {{
        position: fixed !important;
        width: 22vw !important;
        top: 30vh !important;
        right: 2vw;
        z-index: 1000;
    }}

    /* MIDDLE COLUMN SCROLLING */
    [data-testid="column"]:nth-child(2) {{
        height: 75vh !important;
        overflow-y: auto !important;
        margin-left: 25vw !important;
        margin-right: 25vw !important;
        border-left: 1px solid #30363d;
        border-right: 1px solid #30363d;
        padding: 0 25px 100px 25px !important; /* Bottom padding for input clearance */
    }}

    /* 4. UPGRADED TYPOGRAPHY (Large Chat Text) */
    [data-testid="stChatMessage"] p {{
        font-size: 1.3rem !important;
        line-height: 1.7 !important;
        font-family: 'Inter', sans-serif !important;
        color: #E6EDF3 !important;
    }}

    .stChatInput textarea {{
        font-size: 1.1rem !important;
    }}

    [data-testid="stMarkdownContainer"] h3 {{
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #98D8C8 !important;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 25px !important;
        border-left: 6px solid #98D8C8;
        padding-left: 15px;
    }}

    /* 5. UI ELEMENTS */
    [data-testid="stChatMessage"] {{
        background-color: #161B22 !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
    }}

    .stButton>button {{
        background-color: #98D8C8 !important;
        color: #0B0E14 !important;
        font-weight: 700 !important;
        height: 3.5em;
        font-size: 1.1rem !important;
        border-radius: 8px !important;
        transition: 0.3s;
    }}

    .stButton>button:hover {{
        background-color: #7abdaf !important;
        box-shadow: 0 0 15px rgba(152, 216, 200, 0.4);
    }}

    /* Custom Scrollbar */
    [data-testid="column"]:nth-child(2)::-webkit-scrollbar {{ width: 6px; }}
    [data-testid="column"]:nth-child(2)::-webkit-scrollbar-thumb {{ background: #30363d; border-radius: 10px; }}
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def safe_json_format(data):
    try:
        if isinstance(data, str): data = json.loads(data)
        return json.dumps(data, indent=2)
    except: return str(data)

def fetch_history(session_id: str):
    try:
        res = requests.get(f"{API_BASE_URL}/chat-history/{session_id}")
        if res.status_code == 200:
            history = res.json()
            st.session_state['messages'] = [
                {"role": ("user" if e['role'] == 'user' else "assistant"), "content": e.get('message', '')}
                for e in history
            ]
            return True
    except: return False

def fetch_session_list():
    try:
        res = requests.get(f"{API_BASE_URL}/list-sessions")
        return [s['session_id'] for s in res.json()] if res.status_code == 200 else []
    except: return []

# -----------------------------------------------------------------------------
# 4. PAGE HEADER & INITIALIZATION
# -----------------------------------------------------------------------------
if 'session_id' not in st.session_state: st.session_state['session_id'] = str(uuid.uuid4())
if 'messages' not in st.session_state: st.session_state['messages'] = []

st.markdown(f"""
    <div class="fixed-header">
        <img class="header-logo" src="{logo_html}">
        <div style="text-align: left;">
            <h1 class="header-title">Research Assistant LLM</h1>
            <p style="color: #98D8C8; opacity: 0.6; margin-top: 5px; font-size: 18px; font-family: 'JetBrains Mono';">
                SYSTEM PROTOCOL: ACTIVE // ARCHITECTURE: MULTI-AGENT SYNTHESIS
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. LAYOUT EXECUTION
# -----------------------------------------------------------------------------
col_left, col_middle, col_right = st.columns([1, 2, 1])

with col_left:
    st.markdown("### 🛰️ Mission Ops")
    st.code(f"PROTOCOL ID: {st.session_state['session_id'][:12]}", language="bash")

    if st.button("🚀 INITIATE NEW MISSION", use_container_width=True):
        st.session_state['session_id'] = str(uuid.uuid4())
        st.session_state['messages'] = []
        st.rerun()

    st.divider()
    st.markdown("### 📥 Archive")
    sessions = fetch_session_list()
    if sessions:
        selected = st.selectbox("Historical Streams", options=sessions, label_visibility="collapsed")
        if st.button("RESTORE ARCHIVE DATA", use_container_width=True):
            st.session_state['session_id'] = selected
            fetch_history(selected)
            st.rerun()

with col_middle:
    st.markdown("### 📑 Research Log")

    # Display Chat History
    for msg in st.session_state['messages']:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input Interface
    if prompt := st.chat_input("Input research coordinates..."):
        st.chat_message("user").markdown(prompt)
        st.session_state['messages'].append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            status_box = st.empty()
            status_box.markdown("📡 *Synthesizing Multi-Agent Response...*")
            try:
                res = requests.post(
                    f"{API_BASE_URL}/research-chat",
                    json={"session_id": st.session_state['session_id'], "message": prompt}
                )
                if res.status_code == 200:
                    data = res.json()
                    response_msg = data.get('response', 'Error.')

                    # Formatting JSON data for research log
                    if data.get('raw_data') and data.get('raw_data') != "No tool data available":
                        response_msg += f"\n\n---\n**📊 Telemetry Data:**\n```json\n{safe_json_format(data.get('raw_data'))}\n```"

                    status_box.markdown(response_msg)
                    st.session_state['messages'].append({"role": "assistant", "content": response_msg})
                else:
                    status_box.error("Neural Link Failure: Code 500")
            except Exception as e:
                status_box.error(f"System Link Offline: {e}")

with col_right:
    st.markdown("### 🧠 Logic Engine")

    try:
        viz_res = requests.get(f"{API_BASE_URL}/graph-visualization")
        if viz_res.status_code == 200:
            viz_data = viz_res.json()
            raw_code = viz_data['mermaid_syntax']

            # 1. CUMULATIVE HISTORY SCAN
            # We join all messages to ensure we don't just see the 'last' action
            full_conversation_history = " ".join([m['content'] for m in st.session_state.get('messages', [])]).lower()

            # Mapping Log keywords -> Mermaid Node IDs
            path_mapping = {
                "clean_query_agent": ["cleaning", "clean_query", "cleaned"],
                "intent_agent": ["intent", "determining intent", "constraints"],
                "planning_agent": ["planning", "plan generated", "steps"],
                "query_gen_agent": ["query_gen", "generating queries", "tiered"],
                "pubmed_search": ["pubmed", "ncbi"],
                "arxiv_search": ["arxiv", "cornell"],
                "web_search": ["web", "duckduckgo", "ddg"],
                "retrieve_data": ["retrieval", "chunking", "chunks"],
                "rag_filter": ["rag", "vector", "embedding"],
                "synthesis_agent": ["synthesis", "final report", "writing"],
                "evaluation_agent": ["evaluation", "evaluating", "refine"]
            }

            # Identify all "Active" nodes from the entire session
            active_nodes = ["supervisor"]
            for node_id, keywords in path_mapping.items():
                if any(k in full_conversation_history for k in keywords):
                    active_nodes.append(node_id)

            # 2. REBUILD GRAPH WITH CUMULATIVE HIGHLIGHTS
            import re
            connections = re.findall(r'([\w\-]+)\s*[-.]+>\s*([\w\-]+)', raw_code)
            clean_lines = []
            seen = set()

            for start, end in connections:
                s = start.replace("__start__", "START").replace("__end__", "END")
                e = end.replace("__start__", "START").replace("__end__", "END")

                # Cleanup horizontal clutter
                search_tools = ["arxiv_search", "pubmed_search", "web_search", "openalex_search", "materials_search"]
                if s in search_tools and e in search_tools: continue

                if (s, e) not in seen:
                    # HIGHLIGHT: If both nodes have been touched during the session, make it a solid path
                    if s.lower() in active_nodes and e.lower() in active_nodes:
                        arrow = "==>"
                    else:
                        arrow = "-.->"

                    clean_lines.append(f"    {s}{arrow}{e}")
                    seen.add((s, e))

            final_mermaid = "graph TD\n" + "\n".join(clean_lines)

            # Style classes
            for node in active_nodes:
                final_mermaid += f"\n    class {node} activeNode"

            # Add specific pulse if Refinement is needed
            if "refine" in full_conversation_history:
                final_mermaid += f"\n    class evaluation_agent pulseNode"

            # 3. ADVANCED STYLING & RENDERER
            html_code = f"""
            <style>
                @keyframes pulse {{
                    0% {{ stroke: #98D8C8; stroke-width: 2; }}
                    50% {{ stroke: #ff4b4b; stroke-width: 6; }}
                    100% {{ stroke: #98D8C8; stroke-width: 2; }}
                }}
                .pulseNode rect {{ animation: pulse 2s infinite; }}
            </style>
            <div id="wrapper" style="height: 550px; width: 100%; background: #0B0E14; border: 2px solid #30363d; border-radius: 12px; display: flex;">
                <div id="mermaid-holder" style="flex-grow: 1; width: 100%; display: flex; justify-content: center; align-items: center; overflow: hidden;">
                    <pre class="mermaid">
                        {final_mermaid}
                    </pre>
                </div>
            </div>
            <script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{
                    startOnLoad: true,
                    theme: 'dark',
                    themeVariables: {{
                        'primaryColor': '#1c2128',
                        'primaryTextColor': '#E6EDF3',
                        'primaryBorderColor': '#98D8C8',
                        'lineColor': '#98D8C8',
                        'fontSize': '14px'
                    }},
                    flowchart: {{ useMaxWidth: false, curve: 'basis' }}
                }});
                const setupZoom = () => {{
                    const svg = document.querySelector("#mermaid-holder svg");
                    if (svg) {{
                        svg.style.width = "100%"; svg.style.height = "100%";
                        svgPanZoom(svg, {{ zoomEnabled: true, controlIconsEnabled: true, fit: true, center: true }});
                        return true;
                    }}
                    return false;
                }};
                let t = 0; const check = setInterval(() => {{ if (setupZoom() || t > 20) clearInterval(check); t++; }}, 300);
            </script>
            """
            st.components.v1.html(html_code, height=570)

            with st.expander("🔬 View Routing Protocol Code"):
                st.code(final_mermaid, language="mermaid")

    except Exception as e:
        st.info("System initializing...")



