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

# Production-ready API URL with Environment Variable override
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# -----------------------------------------------------------------------------
# 2. IMAGE HANDLER (Docker-Safe Paths)
# -----------------------------------------------------------------------------
def get_base64_of_bin_file(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

# Dynamically find the directory of THIS script to locate images in the same folder
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_left_path = os.path.join(current_dir, 'ai.jpg')
logo_right_path = os.path.join(current_dir, 'genai.jpg')

logo_left_html = f"data:image/jpg;base64,{get_base64_of_bin_file(logo_left_path)}"
logo_right_html = f"data:image/jpg;base64,{get_base64_of_bin_file(logo_right_path)}"

#------------------------------------------------------------------------------
# STYLE FOR H1, H2, H3 WHICH are parsed directly as chat messege by Agent
#------------------------------------------------------------------------------
st.markdown("""
<style>
[data-testid="stChatMessage"] h2 {
    font-size: 26px !important;
    color: #4ECDC4 !important;
    }
    [data-testid="stChatMessage"] ol {
    padding-left: 10px;
    }

    [data-testid="stChatMessage"] li {
        font-size: 1.05rem;
        color: #E6EDF3;
    }

    [data-testid="stChatMessage"] ol li::marker {
        color: #4ECDC4;
        font-size: 1.1rem;
    }
    [data-testid="stChatMessage"] p {
        text-align: justify;
    }

    [data-testid="stChatMessage"] li {
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. MASTER CSS (Responsive Grid & Sticky Positions)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Research Assistant LLM", page_icon="🔬", layout="wide")

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');

    header[data-testid="stHeader"], [data-testid="stDecoration"] {{ display: none !important; }}
    .stApp {{ background-color: #0B0E14 !important; overflow: hidden; }}

    /* HEADER - Increased height for larger logos */
    .fixed-header {{
        position: fixed; top: 0; left: 0; width: 100vw; height: 200px;
        background-color: #0D1117; z-index: 9999; border-bottom: 2px solid #30363d;
        display: flex; justify-content: space-around; align-items: center;
        padding: 0 5vw;
    }}
    .header-logo {{
        width: 400px; /* Doubled size */
        height: 160px; /* Doubled size */
        border-radius: 12px;
        border: 2px solid #98D8C8;
        object-fit: cover;
        box-shadow: 0 0 20px rgba(152, 216, 200, 0.2);
    }}
    .header-title {{ font-size: 42px;
                    font-weight: 800;
                    color: #FFFFFF !important;
                    margin: 0; }}

        /* STICKY COLUMN SYSTEM - MATCH HEADER HEIGHT */
        [data-testid="stHorizontalBlock"] {{
            margin-top: 200px !important;
        }}

        [data-testid="column"]:nth-child(1) > div,
        [data-testid="column"]:nth-child(3) > div {{
            position: sticky !important;
            top: 200px !important;
            max-height: calc(100vh - 220px);
            overflow-y: auto;
        }}

    /* MIDDLE COLUMN SCROLLING */
    [data-testid="column"]:nth-child(2) {{
        height: calc(100vh - 210px) !important;
        overflow-y: auto !important;
        border-left: 1px solid #30363d;
        border-right: 1px solid #30363d;
        padding: 0 20px 10px 20px !important;
    }}

    /* TYPOGRAPHY & CHAT */
    [data-testid="stChatMessage"] p {{ font-size: 1.1rem !important; line-height: 1.6; font-family: 'Inter'; color: #E6EDF3; }}
    [data-testid="stMarkdownContainer"] h3 {{ font-size:1.5rem !important; color: #98D8C8; letter-spacing: 1.5px; border-left: 5px solid #98D8C8; padding-left: 15px; text-transform: uppercase; }}

    /* STATUS INDICATORS */
    .status-pulse {{ width: 10px; height: 10px; background: #98D8C8; border-radius: 50%; animation: pulse-ring 1.5s infinite; }}
    @keyframes pulse-ring {{ 0% {{ box-shadow: 0 0 0 0 rgba(152, 216, 200, 0.7); }} 70% {{ box-shadow: 0 0 0 10px rgba(152, 216, 200, 0); }} 100% {{ box-shadow: 0 0 0 0 rgba(152, 216, 200, 0); }} }}

    .stButton > button {{
        border-radius: 12px;
        background: #1a2a24;
        border: 1px solid #98D8C8;
        color: #98D8C8;
        font-family: 'JetBrains Mono';
        transition: all 0.3s;
    }}
    .stButton > button:hover {{ background: #98D8C8; color: #0B0E14; box-shadow: 0 0 10px #98D8C8; }}
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
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
                visited = e.get('visited_nodes', [])
                if role == "assistant" and visited:
                    path_breadcrumb = " → ".join([f"`{n}`" for n in visited])
                    content += f"\n\n---\n**🛤️ Agent Execution Path:**\n{path_breadcrumb}"
                st.session_state['messages'].append({"id": msg_id, "role": role, "content": content})
                if visited: st.session_state['turn_paths'][msg_id] = visited
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
        <div style="text-align:center;">
            <h1 class="header-title">Research Assistant LLM</h1>
            <p style="color:#98D8C8; opacity:0.6; font-family:'JetBrains Mono'; font-size:12px; margin:0;">SYSTEM: ACTIVE // ARCHITECTURE: MULTI-AGENT</p>
        </div>
        <img class="header-logo" src="{logo_right_html}">
    </div>""", unsafe_allow_html=True)

col_left, col_middle, col_right = st.columns([1, 2, 1.2], gap="medium")

with col_left:
    st.markdown("### 🛰️ Mission Ops")
    if st.session_state['processing']:
        st.markdown(f"""<div style='padding:15px; background:#1a2a24; border:1px solid #98D8C8; border-radius:10px; margin-bottom:20px;'>
            <div style='display:flex; align-items:center; gap:10px;'><div class="status-pulse"></div><span style='color:#98D8C8; font-family:"JetBrains Mono"; font-size:12px;'>AGENT ACTIVE</span></div>
            <div style='color:white; font-size:18px; font-weight:800; margin-top:8px;'>EXECUTING LOGIC...</div></div>""", unsafe_allow_html=True)
    else:
        st.markdown("<div style='padding:15px; background:#0D1117; border:1px solid #30363d; border-radius:10px; margin-bottom:20px; color:#666; font-family:\"JetBrains Mono\";'>SYSTEM STANDBY</div>", unsafe_allow_html=True)

    st.code(f"ID: {st.session_state['session_id'][:12]}", language="bash")
    if st.button("🚀 NEW MISSION", use_container_width=True):
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
        if st.button("RESTORE ARCHIVE", use_container_width=True):
            st.session_state['session_id'] = selected
            if fetch_history(selected): st.rerun()

with col_middle:
    st.markdown("### 📑 Research Log")
    for msg in st.session_state['messages']:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

        if msg["id"] in st.session_state['turn_paths']:
            cA, cB = st.columns(2)
            with cA:
                if st.button("🕸 Graph", key=f"g_{msg['id']}", use_container_width=True):
                    st.session_state['active_view_path'] = st.session_state['turn_paths'][msg["id"]]
                    st.session_state['debug_message_id'] = msg["id"] # Added fix
                    st.switch_page("pages/03_Graph_View.py")
            with cB:
                if st.button("🧠 Debug", key=f"d_{msg['id']}", use_container_width=True):
                    st.session_state['active_view_path'] = st.session_state['turn_paths'][msg["id"]]
                    st.session_state['debug_message_id'] = msg["id"] # Added fix
                    st.switch_page("pages/02_Debug_State.py")

    if prompt := st.chat_input("Input coordinates..."):
        st.session_state['processing'] = True
        u_id = str(uuid.uuid4())
        st.session_state['messages'].append({"id": u_id, "role": "user", "content": prompt})
        st.rerun()  # 1. This cleanly forces Streamlit to refresh and draw the user's new message bubble.

    # 2. PLACE THIS BLOCK OUTSIDE THE CHAT_INPUT BLOCK (Right below your message rendering loops)
    if st.session_state['processing'] and len(st.session_state['messages']) > 0:
        # Only act if the last message came from the user
        if st.session_state['messages'][-1]["role"] == "user":
            with st.chat_message("assistant"):
                # Use 'with st.spinner' as a context manager so it spins while the code inside it runs
                with st.spinner("📡 *Synthesizing...*"):
                    try:
                        res = requests.post(f"{API_BASE_URL}/research-chat",
                                            json={
                                                "session_id": st.session_state['session_id'],
                                                "message": st.session_state['messages'][-1]["content"]
                                            },
                                            timeout=600)

                        if res.status_code == 200:
                            data = res.json()
                            m_id, path = data.get('id'), data.get('visited_path', [])
                            st.session_state['turn_paths'][m_id] = path
                            st.session_state['active_view_path'] = path
                            resp = data.get('response', '')
                            if path:
                                resp += f"\n\n---\n**Tracked Path:** {' → '.join([f'`{n}`' for n in path])}"
                            st.session_state['messages'].append({"id": m_id, "role": "assistant", "content": resp})
                        else:
                            st.error(f"Backend API Error: Received Status Code {res.status_code}")

                        # Turn off processing and rerun to display the newly added assistant response bubble
                        st.session_state['processing'] = False
                        st.rerun()

                    except Exception as e:
                        st.session_state['processing'] = False
                        st.error(f"Network Connection Failed: {e}")


with col_right:
    st.markdown("### 🧠 Logic Engine")
    if st.session_state['active_view_path']:
        active_path = [str(n).strip().replace('"', '') for n in st.session_state['active_view_path']]
        try:
            viz_res = requests.get(f"{API_BASE_URL}/graph-visualization")
            if viz_res.status_code == 200:
                raw_mermaid = viz_res.json()["mermaid_syntax"]
                # Logic for style injection (same as previous working version)
                active_nodes_css = "\n".join([f"class {node} activeNode" for node in active_path])
                final_mermaid = f"{raw_mermaid.split('classDef')[0].strip()}\n{active_nodes_css}"

                # HTML component remains the same for Mermaid rendering
                st.components.v1.html(f"""
                <div id="logic-wrapper" style="height:600px; background:#040926; border:1px solid #30363d; border-radius:16px;">
                    <pre class="mermaid">{final_mermaid}</pre>
                </div>
                <script type="module">
                    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs';
                    mermaid.initialize({{ startOnLoad: true, theme: 'base' }});
                </script>
                """, height=620)
        except: pass
