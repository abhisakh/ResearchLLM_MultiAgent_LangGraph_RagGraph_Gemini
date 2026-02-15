import streamlit as st
import requests

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Agent Graph View",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Agent Execution Graph")

# -------------------------------------------------
# Validate Session
# -------------------------------------------------
path = st.session_state.get("active_view_path")

if not path:
    st.warning("No execution path available.")
    st.stop()

st.caption(f"Active Path: {' → '.join(path)}")

# -------------------------------------------------
# Fetch Mermaid
# -------------------------------------------------
try:
    viz_res = requests.get(f"{API_BASE_URL}/graph-visualization")
    if viz_res.status_code != 200:
        st.error("Failed to load graph.")
        st.stop()

    viz_data = viz_res.json()
    raw_mermaid = viz_data["mermaid_syntax"]

except Exception as e:
    st.error(f"Graph error: {e}")
    st.stop()

# -------------------------------------------------
# Highlight Active Path
# -------------------------------------------------
lines = raw_mermaid.split('\n')
link_indices = []
current_link_count = 0

for line in lines:
    if "-->" in line or "->" in line:
        for i in range(len(path) - 1):
            src, dst = path[i], path[i+1]
            if src in line and dst in line:
                link_indices.append(current_link_count)
        current_link_count += 1

active_nodes_css = "\n".join([f"class {node} activeNode" for node in path])
active_links_css = "\n".join([
    f"linkStyle {idx} stroke:#98D8C8,stroke-width:4px"
    for idx in link_indices
])

sanitized_mermaid = raw_mermaid.split("classDef")[0].strip()
final_mermaid = f"{sanitized_mermaid}\n{active_nodes_css}\n{active_links_css}"

# -------------------------------------------------
# Render
# -------------------------------------------------
html_code = f"""
<div id="graph-container">
    <pre class="mermaid">{final_mermaid}</pre>
</div>

<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
mermaid.initialize({{
    startOnLoad: true,
    theme: "dark"
}});
</script>

<style>
#graph-container {{
    background:#0B0E14;
    border:1px solid #30363d;
    padding:20px;
    border-radius:12px;
}}

/* Active nodes */
.activeNode rect {{
    fill:#98D8C8 !important;
    stroke:#fff !important;
    stroke-width:3px !important;
}}

/* Active node labels (text only) */
#graph-container .activeNode .nodeLabel {{
    color: #00008B !important;  /* Only node text turns blue */
    font-weight: 800 !important;
}}
</style>

"""

st.components.v1.html(html_code, height=800)

st.divider()

if st.button("⬅ Back to Main UI"):
    st.switch_page("ui_main.py")
