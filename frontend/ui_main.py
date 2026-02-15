# ################################################################################
# # FILE: main_ui.py | FULLY OPTIMIZED & PATH-SYNCED VERSION
# ################################################################################

# import streamlit as st
# import requests
# import json
# import uuid
# import base64
# import os

# # 1. INITIALIZATION & STATE
# if "session_id" not in st.session_state:
#     st.session_state.session_id = str(uuid.uuid4())
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "last_visited_path" not in st.session_state:
#     st.session_state.last_visited_path = []
# if "processing" not in st.session_state:
#     st.session_state.processing = False

# API_BASE_URL = "http://localhost:8000"

# # 2. IMAGE HANDLER (Base64)
# def get_base64_of_bin_file(bin_file):
#     if os.path.exists(bin_file):
#         with open(bin_file, "rb") as f:
#             return base64.b64encode(f.read()).decode()
#     return ""

# logo_left_html = f"data:image/jpg;base64,{get_base64_of_bin_file('ai.jpg')}"
# logo_right_html = f"data:image/jpg;base64,{get_base64_of_bin_file('genai.jpg')}"

# # 3. PAGE CONFIG + GLOBAL STYLING
# st.set_page_config(page_title="Research Assistant LLM", page_icon="🔬", layout="wide")

# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');
# header[data-testid="stHeader"], [data-testid="stDecoration"] { display: none !important; }
# .stApp { background-color: #0B0E14 !important; }

# /* HEADER */
# .fixed-header {
#     position: fixed; top: 0; left: 0; width: 100vw; height: 25vh;
#     background: #0D1117; border-bottom: 2px solid #30363d;
#     display: flex; align-items: center; justify-content: center; gap: 4vw; z-index: 9999;
# }
# .header-logo { width: 500px; height: 200px; object-fit: cover; border-radius: 12px; border: 2px solid #FB8500; }
# .header-title { font-size: 42px; font-weight: 800; color: #FB8500; }

# /* LAYOUT SPACING */
# [data-testid="stHorizontalBlock"] { margin-top: 25vh !important; }
# [data-testid="column"]:nth-child(2) {
#     height: 72vh !important; overflow-y: auto !important;
#     border-left: 1px solid #30363d; border-right: 1px solid #30363d;
#     padding: 0 20px 120px 20px !important;
# }

# .status-pulse {
#     width: 12px; height: 12px; background: #FB8500; border-radius: 50%;
#     animation: pulse-ring 1.5s infinite;
# }
# @keyframes pulse-ring {
#     0%   { box-shadow: 0 0 0 0 rgba(251,133,0,0.7); }
#     70%  { box-shadow: 0 0 0 12px rgba(251,133,0,0); }
#     100% { box-shadow: 0 0 0 0 rgba(251,133,0,0); }
# }
# </style>
# """, unsafe_allow_html=True)

# # 4. HELPERS
# def safe_json_format(data):
#     try:
#         if isinstance(data, str): data = json.loads(data)
#         return json.dumps(data, indent=2)
#     except: return str(data)

# def fetch_history(session_id):
#     try:
#         r = requests.get(f"{API_BASE_URL}/chat-history/{session_id}")
#         if r.status_code == 200:
#             st.session_state.messages = [
#                 {"role": "user" if m["role"] == "user" else "assistant",
#                  "content": m.get("message", m.get("content", ""))}
#                 for m in r.json()]
#             return True
#     except: return False

# def fetch_session_list():
#     try:
#         r = requests.get(f"{API_BASE_URL}/list-sessions")
#         return [s["session_id"] for s in r.json()] if r.status_code == 200 else []
#     except: return []

# # 5. UI HEADER
# st.markdown(f"""
# <div class="fixed-header">
#     <img class="header-logo" src="{logo_left_html}">
#     <div><h1 class="header-title">Research Assistant LLM</h1><p style="color:#FB8500; opacity:.6; font-family:'JetBrains Mono';">SYSTEM PROTOCOL: ACTIVE</p></div>
#     <img class="header-logo" src="{logo_right_html}">
# </div>
# """, unsafe_allow_html=True)

# col_left, col_middle, col_right = st.columns([1, 2, 1])

# # --- LEFT COLUMN: MISSION OPS ---
# with col_left:
#     st.markdown("### 🛰️ Mission Ops")
#     if st.session_state.processing:
#         path = st.session_state.last_visited_path
#         current = path[-1].replace("_", " ").upper() if path else "THINKING..."
#         st.markdown(f"<div style='padding:15px; background:#2a1a00; border:1px solid #FB8500; border-radius:10px;'><div style='display:flex; gap:10px; align-items:center;'><div class='status-pulse'></div><span style='color:#FB8500;'>AGENT ACTIVE</span></div><div style='font-size:20px; font-weight:800; color:white;'>{current}</div></div>", unsafe_allow_html=True)

#     st.code(f"PROTOCOL ID: {st.session_state.session_id[:12]}", language="bash")
#     if st.button("🚀 NEW MISSION", use_container_width=True):
#         st.session_state.session_id, st.session_state.messages, st.session_state.last_visited_path = str(uuid.uuid4()), [], []
#         st.rerun()

#     st.divider()
#     sessions = fetch_session_list()
#     if sessions:
#         selected = st.selectbox("Sessions", sessions, label_visibility="collapsed")
#         if st.button("RESTORE ARCHIVE DATA", use_container_width=True):
#             if fetch_history(selected):
#                 st.session_state.session_id, st.session_state.last_visited_path = selected, []
#                 st.rerun()

# # --- MIDDLE COLUMN: RESEARCH LOG ---
# with col_middle:
#     st.markdown("### 📑 Research Log")
#     for msg in st.session_state.messages:
#         with st.chat_message(msg["role"]): st.markdown(msg["content"])

#     if prompt := st.chat_input("Input research coordinates..."):
#         st.session_state.processing = True
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("assistant"):
#             status = st.empty()
#             status.markdown("📡 *Synthesizing...*")
#             try:
#                 r = requests.post(f"{API_BASE_URL}/research-chat", json={"session_id": st.session_state.session_id, "message": prompt}, timeout=180)
#                 if r.status_code == 200:
#                     data = r.json()
#                     # SYNC FIX: Ensure naming alignment between backend JSON and Mermaid dictionary keys
#                     raw_path = data.get("visited_path", [])
#                     st.session_state.last_visited_path = [n if n != "retrieval_agent" else "retrieve_data" for n in raw_path]

#                     response = data.get("response", "")
#                     if data.get("raw_data"): response += f"\n\n```json\n{safe_json_format(data.get('raw_data'))}\n```"
#                     st.session_state.messages.append({"role": "assistant", "content": response})
#                 st.session_state.processing = False
#                 st.rerun()
#             except Exception as e:
#                 st.session_state.processing = False
#                 status.error(str(e))

# # --- RIGHT COLUMN: LOGIC ENGINE ---
# with col_right:
#     st.markdown("### 🧠 Logic Engine")
#     try:
#         r = requests.get(f"{API_BASE_URL}/graph-visualization")
#         if r.status_code == 200:
#             viz_data = r.json()
#             raw_mermaid = viz_data["mermaid_syntax"]
#             active_nodes = st.session_state.get("last_visited_path", [])

#             # 1. Clean and Inject Active Nodes
#             sanitized = raw_mermaid.split("classDef")[0].strip()
#             class_block = "\n".join(f"class {n} activeNode" for n in set(active_nodes))
#             final_mermaid = f"{sanitized}\n{class_block}"

#             # 2. Encode Path for JS Highlighting
#             js_path_array = json.dumps(active_nodes)
#             encoded_mermaid = base64.b64encode(final_mermaid.encode()).decode()

#             st.components.v1.html(f"""
#                 <script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
#                 <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;">
#                     <button id="expandBtn" style="padding:5px 10px; border:none; border-radius:6px; background:#FB8500; color:#000; cursor:pointer; font-weight:700;">🔍 Fullscreen Mode</button>
#                 </div>
#                 <div id="logic-wrapper" style="position:relative; height:620px; width:100%; border:2px solid #FB8500; border-radius:12px; overflow:hidden; background: #001219;">
#                     <pre class="mermaid" id="main-m" style="visibility: hidden;">{final_mermaid}</pre>
#                 </div>

#                 <script type="module">
#                     import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs';

#                     mermaid.initialize({{
#                         startOnLoad: true, securityLevel: 'loose', theme: 'base',
#                         flowchart: {{ nodeSpacing:140, rankSpacing:100, padding:20, curve:'basis' }},
#                         themeVariables: {{ primaryColor:'#003566', primaryTextColor:'#FFFFFF', primaryBorderColor:'#FB8500', lineColor:'#FB8500' }}
#                     }});

#                     function setupStyles(selector) {{
#                         const svg = document.querySelector(selector + " svg");
#                         if(!svg) return;

#                         const pathSequence = {js_path_array};
#                         const activeEdges = [];
#                         for(let i=0; i < pathSequence.length - 1; i++) {{
#                             activeEdges.push({{ from: pathSequence[i], to: pathSequence[i+1] }});
#                         }}

#                         svg.querySelectorAll(".edgePath").forEach(edge => {{
#                             const path = edge.querySelector("path");
#                             if(!path) return;

#                             // Check if this edge is a link in the active sequential path
#                             const isMainPath = activeEdges.some(pair =>
#                                 (edge.classList.contains("LS-"+pair.from) && edge.classList.contains("LE-"+pair.to)) ||
#                                 (edge.classList.contains("LS-"+pair.to) && edge.classList.contains("LE-"+pair.from))
#                             );

#                             const isDashed = window.getComputedStyle(path).strokeDasharray !== "none" || path.getAttribute("stroke-dasharray");

#                             if (isMainPath) {{
#                                 path.style.stroke = "#FB8500";
#                                 path.style.strokeWidth = "5px";
#                                 path.style.strokeOpacity = "1";
#                                 path.style.filter = "drop-shadow(0 0 5px #FB8500)";
#                                 // Sync Arrowhead
#                                 const markerId = path.getAttribute("marker-end")?.replace(/url\(|#|\)/g, "");
#                                 if(markerId) {{
#                                     const marker = svg.querySelector("#" + markerId + " path");
#                                     if(marker) marker.style.fill = "#FB8500";
#                                 }}
#                             }} else {{
#                                 // Ghost the rest
#                                 path.style.stroke = isDashed ? "#00CCFF" : "#003566";
#                                 path.style.strokeWidth = "1px";
#                                 path.style.strokeOpacity = "0.08";
#                             }}
#                         }});
#                     }}

#                     function fitDiagram(selector) {{
#                         const svg = document.querySelector(selector + " svg");
#                         if(svg) {{
#                             svg.style.width="100%"; svg.style.height="100%";
#                             svg.style.maxWidth="none";
#                             const pz = svgPanZoom(svg, {{ fit: true, center: true, zoomScaleSensitivity: 0.4 }});
#                             setupStyles(selector);
#                             setTimeout(() => {{ pz.fit(); pz.center(); }}, 100);
#                             if(selector === "#logic-wrapper") document.getElementById("main-m").style.visibility = "visible";
#                         }}
#                     }}

#                     setTimeout(() => {{ fitDiagram("#logic-wrapper"); }}, 1000);
#                 </script>

#                 <style>
#                     svg {{ background: #001219 !important; overflow: visible !important; }}
#                     .node rect, .node circle {{ fill: #003566 !important; stroke: #FB8500 !important; stroke-width: 2.5px !important; rx: 10; ry: 10; }}
#                     .nodeLabel {{ color: #FFFFFF !important; font-weight: 700 !important; font-size: 16px !important; }}
#                     .activeNode rect, .activeNode circle {{ fill: #FB8500 !important; stroke: #FFFFFF !important; stroke-width: 4.5px !important; }}
#                     .activeNode .nodeLabel {{ color: #001219 !important; font-weight: 900 !important; }}
#                     .activeNode {{ filter: drop-shadow(0 0 15px #FB8500); }}
#                 </style>
#             """, height=640)

#     except Exception as e:
#         st.error(f"Logic Engine Interface Error: {e}")



################################################################################
# FILE: main_ui.py
# FINAL RESTORED VERSION: Fixed Archive Sync | Sticky Layout | Path Highlights
################################################################################

import streamlit as st
import requests
import json
import uuid
import base64
import os
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. INITIALIZATION & STATE MANAGEMENT
# -----------------------------------------------------------------------------
if 'session_id' not in st.session_state: st.session_state['session_id'] = str(uuid.uuid4())
if 'messages' not in st.session_state: st.session_state['messages'] = []
if 'turn_paths' not in st.session_state: st.session_state['turn_paths'] = {}
if 'active_view_path' not in st.session_state: st.session_state['active_view_path'] = []
if 'processing' not in st.session_state: st.session_state['processing'] = False

API_BASE_URL = "http://localhost:8000"

# -----------------------------------------------------------------------------
# 2. IMAGE HANDLER (Base64)
# -----------------------------------------------------------------------------
def get_base64_of_bin_file(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

logo_left_html = f"data:image/jpg;base64,{get_base64_of_bin_file('ai.jpg')}"
logo_right_html = f"data:image/jpg;base64,{get_base64_of_bin_file('genai.jpg')}"

# -----------------------------------------------------------------------------
# 3. MASTER CSS (Headers, Logic, and Sticky Positioning)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Research Assistant LLM", page_icon="🔬", layout="wide")

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');

    header[data-testid="stHeader"], [data-testid="stDecoration"] {{ display: none !important; }}
    .stApp {{ background-color: #0B0E14 !important; overflow: hidden; }}

    /* HEADER */
    .fixed-header {{
        position: fixed; top: 0; left: 0; width: 100vw; height: 25vh;
        background-color: #0D1117; z-index: 9999; border-bottom: 2px solid #30363d;
        display: flex; justify-content:center; align-items: center; gap: 4vw;
    }}
    .header-logo {{ width: 500px; height: 200px; border-radius: 12px; border: 2px solid #98D8C8; object-fit: cover; }}
    .header-title {{ font-size: 42px; font-weight: 800; color: #98D8C8; margin: 0; }}

    /* STICKY COLUMN SYSTEM */
    [data-testid="stHorizontalBlock"] {{ margin-top: 25vh !important; }}

    [data-testid="column"]:nth-child(1) > div,
    [data-testid="column"]:nth-child(3) > div {{
        position: sticky !important;
        top: 28vh !important;
    }}

    /* MIDDLE COLUMN SCROLLING */
    [data-testid="column"]:nth-child(2) {{
        height: 72vh !important;
        overflow-y: auto !important;
        border-left: 1px solid #30363d;
        border-right: 1px solid #30363d;
        padding: 0 20px 100px 20px !important;
    }}

    /* TYPOGRAPHY & CHAT */
    [data-testid="stChatMessage"] p {{ font-size: 1.2rem !important; line-height: 1.6; font-family: 'Inter'; color: #E6EDF3; }}
    [data-testid="stMarkdownContainer"] h3 {{ font-size: 1.8rem !important; color: #98D8C8; letter-spacing: 2px; border-left: 5px solid #98D8C8; padding-left: 15px; text-transform: uppercase; }}

    /* STATUS INDICATORS */
    .status-pulse {{ width: 12px; height: 12px; background: #98D8C8; border-radius: 50%; animation: pulse-ring 1.5s infinite; }}
    @keyframes pulse-ring {{ 0% {{ box-shadow: 0 0 0 0 rgba(152, 216, 200, 0.7); }} 70% {{ box-shadow: 0 0 0 10px rgba(152, 216, 200, 0); }} 100% {{ box-shadow: 0 0 0 0 rgba(152, 216, 200, 0); }} }}

    /* CUSTOM BUTTON FOR AGENT PATH */
    .stButton > button {{
        border-radius: 20px;
        background: #1a2a24;
        border: 1px solid #98D8C8;
        color: #98D8C8;
        font-family: 'JetBrains Mono';
        padding: 5px 15px;
        transition: all 0.3s;
    }}
    .stButton > button:hover {{
        background: #98D8C8;
        color: #0B0E14;
        box-shadow: 0 0 15px #98D8C8;
    }}
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. HELPER FUNCTIONS
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
            st.session_state['messages'] = []
            st.session_state['turn_paths'] = {}
            for e in history:
                role = "user" if e.get('role') == 'user' else "assistant"
                msg_id = e.get('id')
                content = e.get('message', '')

                # --- UPDATED TELEMETRY DISPLAY ---
                visited = e.get('visited_nodes', [])
                if role == "assistant" and visited:
                    # Create a clean, joined string of agents
                    path_breadcrumb = " → ".join([f"`{n}`" for n in visited])
                    content += f"\n\n---\n**🛤️ Agent Execution Path:**\n{path_breadcrumb}"

                st.session_state['messages'].append({"id": msg_id, "role": role, "content": content})
                if visited:
                    st.session_state['turn_paths'][msg_id] = visited
            return True
    except: return False

def fetch_session_list():
    try:
        res = requests.get(f"{API_BASE_URL}/list-sessions")
        return [s['session_id'] for s in res.json()] if res.status_code == 200 else []
    except: return []

# -----------------------------------------------------------------------------
# 5. UI LAYOUT
# -----------------------------------------------------------------------------
st.markdown(f"""
    <div class="fixed-header">
        <img class="header-logo" src="{logo_left_html}">
        <div><h1 class="header-title">Research Assistant LLM</h1><p style="color:#98D8C8; opacity:0.6; font-family:'JetBrains Mono';">SYSTEM PROTOCOL: ACTIVE // ARCHITECTURE: MULTI-AGENT</p></div>
        <img class="header-logo" src="{logo_right_html}">
    </div>""", unsafe_allow_html=True)

col_left, col_middle, col_right = st.columns([1, 2, 1])

# --- LEFT COLUMN: MISSION OPS ---
with col_left:
    st.markdown("### 🛰️ Mission Ops")
    if st.session_state['processing']:
        st.markdown(f"""
            <div style='padding:15px; background:#1a2a24; border:1px solid #98D8C8; border-radius:10px; margin-bottom:20px;'>
                <div style='display:flex; align-items:center; gap:10px;'><div class="status-pulse"></div><span style='color:#98D8C8; font-family:"JetBrains Mono"; font-size:12px;'>AGENT ACTIVE</span></div>
                <div style='color:white; font-size:18px; font-weight:800; margin-top:8px;'>EXECUTING LOGIC...</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("<div style='padding:15px; background:#0D1117; border:1px solid #30363d; border-radius:10px; margin-bottom:20px; color:#666; font-family:\"JetBrains Mono\";'>SYSTEM STANDBY</div>", unsafe_allow_html=True)

    st.code(f"PROTOCOL ID: {st.session_state['session_id'][:12]}", language="bash")
    if st.button("🚀 INITIATE NEW MISSION", use_container_width=True):
        st.session_state['session_id'] = str(uuid.uuid4())
        st.session_state['messages'] = []
        st.session_state['turn_paths'] = {}
        st.session_state['active_view_path'] = []
        st.rerun()

    st.divider()
    st.markdown("### 📥 Archive")
    sessions = fetch_session_list()
    if sessions:
        selected = st.selectbox("Historical Streams", options=sessions, label_visibility="collapsed")
        if st.button("RESTORE ARCHIVE DATA", use_container_width=True):
            st.session_state['session_id'] = selected
            if fetch_history(selected): st.rerun()

# --- MIDDLE COLUMN: RESEARCH LOG ---
# --- MIDDLE COLUMN: RESEARCH LOG ---
with col_middle:
    st.markdown("### 📑 Research Log")

    for msg in st.session_state['messages']:

        # ---- CHAT MESSAGE ----
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

        # ---- ACTION BUTTONS (OUTSIDE CHAT MESSAGE) ----
        if msg["id"] in st.session_state['turn_paths']:
            colA, colB = st.columns([1, 1])

            # -------- GRAPH VIEW BUTTON --------
            with colA:
                if st.button(
                    "🕸 View Graph",
                    key=f"graph_{msg['id']}",
                    use_container_width=True
                ):
                    st.session_state['active_view_path'] = st.session_state['turn_paths'][msg["id"]]
                    st.session_state['debug_message_id'] = msg["id"]
                    st.switch_page("pages/03_Graph_View.py")

            # -------- FULL DEBUG STATE BUTTON --------
            with colB:
                if st.button(
                    "🧠 Debug State",
                    key=f"debug_{msg['id']}",
                    use_container_width=True
                ):
                    st.session_state['active_view_path'] = st.session_state['turn_paths'][msg["id"]]
                    st.session_state['debug_message_id'] = msg["id"]
                    st.switch_page("pages/02_Debug_State.py")

            # Optional tiny spacer (very small)
            st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------
    # CHAT INPUT
    # ---------------------------------------------------
    if prompt := st.chat_input("Input research coordinates..."):
        st.session_state['processing'] = True
        u_id = str(uuid.uuid4())
        st.session_state['messages'].append({
            "id": u_id,
            "role": "user",
            "content": prompt
        })

        with st.chat_message("assistant"):
            status_box = st.empty()
            status_box.markdown("📡 *Synthesizing Multi-Agent Response...*")

            try:
                res = requests.post(
                    f"{API_BASE_URL}/research-chat",
                    json={
                        "session_id": st.session_state['session_id'],
                        "message": prompt
                    },
                    timeout=600
                )

                if res.status_code == 200:
                    data = res.json()
                    msg_id = data.get('id')

                    path = [
                        str(n).strip().replace('"', '')
                        for n in data.get('visited_path', [])
                    ]

                    st.session_state['turn_paths'][msg_id] = path
                    st.session_state['active_view_path'] = path

                    response_msg = data.get('response', '')

                    if path:
                        path_breadcrumb = " → ".join(
                            [f"`{n}`" for n in path]
                        )
                        response_msg += (
                            f"\n\n---\n**Tracked Path:** {path_breadcrumb}"
                        )

                    st.session_state['messages'].append({
                        "id": msg_id,
                        "role": "assistant",
                        "content": response_msg
                    })

                st.session_state['processing'] = False
                st.rerun()

            except Exception as e:
                st.session_state['processing'] = False
                status_box.error(f"Link Error: {e}")

# --- RIGHT COLUMN: LOGIC ENGINE (Fixed Path Highlighting) ---
with col_right:
    st.markdown("### 🧠 Logic Engine")

    if st.session_state['active_view_path']:
        # Clean the path nodes to match mermaid IDs
        active_path = [str(n).strip().replace('"', '') for n in st.session_state['active_view_path']]
        st.caption(f"Active Path: {' → '.join(active_path)}")

        try:
            viz_res = requests.get(f"{API_BASE_URL}/graph-visualization")
            if viz_res.status_code == 200:
                viz_data = viz_res.json()
                raw_mermaid = viz_data["mermaid_syntax"]

                # --- NEW LOGIC: CALCULATE EDGE INDICES ---
                # Mermaid indexes links (arrows) starting from 0 in the order they appear.
                lines = raw_mermaid.split('\n')
                link_indices = []
                current_link_count = 0

                for line in lines:
                    if "-->" in line or "->" in line:
                        # Check if this specific link is part of our active path
                        for i in range(len(active_path) - 1):
                            src, dst = active_path[i], active_path[i+1]
                            if src in line and dst in line:
                                link_indices.append(current_link_count)
                        current_link_count += 1

                # Dynamic Style Injection
                active_nodes_css = "\n".join([f"class {node} activeNode" for node in active_path])
                # This makes the lines solid and bright
                active_links_css = "\n".join([f"linkStyle {idx} stroke:#98D8C8,stroke-width:4px,opacity:1.0" for idx in link_indices])

                # Assemble final syntax
                sanitized_mermaid = raw_mermaid.split("classDef")[0].strip()
                final_mermaid = f"{sanitized_mermaid}\n{active_nodes_css}\n{active_links_css}"

                html_code = f"""
<div id="logic-wrapper">
    <div id="mermaid-holder">
        <pre class="mermaid">{final_mermaid}</pre>
    </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
<script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs';

    mermaid.initialize({{
        startOnLoad: true, theme: 'base', securityLevel: 'loose',
        flowchart: {{ useMaxWidth: false, htmlLabels: true, curve: 'basis' }},
        themeVariables: {{
            primaryColor: '#003566',
            lineColor: '#30363d'
        }}
    }});

    const highlightActiveArrows = () => {{
        const svg = document.querySelector("#mermaid-holder svg");
        if (!svg) return;

        // 1. Find all active edge groups (the ones linkStyle touched)
        const activeEdges = svg.querySelectorAll('g.edgePath');

        activeEdges.forEach(edge => {{
            const style = edge.getAttribute('style') || "";
            // If the group has our active green color
            if (style.includes('rgb(152, 216, 200)') || style.includes('#98D8C8')) {{

                // Set the line to solid
                const path = edge.querySelector('.path');
                if (path) {{
                    path.style.strokeWidth = '6px';
                    path.style.opacity = '1';
                }}

                // 2. Find the Marker ID from the path's marker-end attribute
                const markerValue = window.getComputedStyle(path).markerEnd;
                const markerId = markerValue.replace('url("', '').replace('")', '').replace('#', '');

                // 3. Find that marker in the DEFS section
                const marker = svg.querySelector(`[id="${{markerId}}"]`);
                if (marker) {{
                    marker.style.overflow = 'visible'; // Ensure it isn't clipped
                    const markerPath = marker.querySelector('path');
                    if (markerPath) {{
                        markerPath.style.fill = '#FFA500'; // BRIGHT ORANGE
                        markerPath.style.stroke = '#FFA500';
                        markerPath.style.opacity = '1';
                    }}
                }}
            }}
        }});
    }};

    setTimeout(() => {{
        const svg = document.querySelector("#mermaid-holder svg");
        if (svg) {{
            svg.style.width = "100%"; svg.style.height = "100%";
            window.pz = svgPanZoom(svg, {{ zoomEnabled: true, fit: true, center: true }});

            // Run highlight logic multiple times to catch Mermaid's delayed rendering
            highlightActiveArrows();
            setTimeout(highlightActiveArrows, 100);
            setTimeout(highlightActiveArrows, 500);
        }}
    }}, 800);
</script>
<style>
    #logic-wrapper {{ height: 620px; width: 100%; background: #040926; border: 2px solid #30363d; border-radius: 16px; overflow: hidden; }}

    /* NODES */
    .activeNode rect {{ fill: #98D8C8 !important; stroke: #fff !important; stroke-width: 3px !important; filter: drop-shadow(0 0 5px #98D8C8); }}
    .activeNode .nodeLabel {{ color: #040926 !important; font-weight: 800 !important; }}

    /* DEFAULT FAINT STATE */
    .edgePath .path {{ opacity: 0.1; stroke: #30363d !important; }}
    .marker path {{ fill: #30363d !important; opacity: 0.1 !important; }}

    /* ACTIVE PATHS (FORCE GREEN) */
    g.edgePath[style*="stroke: rgb(152, 216, 200)"] .path,
    g.edgePath[style*="stroke:#98D8C8"] .path {{
        opacity: 1.0 !important;
        stroke: #98D8C8 !important;
        stroke-width: 6px !important;
    }}
</style>
"""
            st.components.v1.html(html_code, height=640)
        except Exception as e:
            st.error(f"Logic Engine Error: {e}")