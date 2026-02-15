import streamlit as st
import requests
import json
from datetime import datetime

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Agent Debug State",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================
st.title("🧠 Agent Execution Debug Console")
st.caption("Full ResearchState Inspection • Observability • Trace Analysis")

# =====================================================
# Helpers
# =====================================================
@st.cache_data(show_spinner=False)
def fetch_raw_state(message_id: str):
    try:
        res = requests.get(f"{API_BASE_URL}/debug/raw-state/{message_id}", timeout=30)
        if res.status_code == 200:
            return res.json()
    except:
        return None


def flatten_dict(d, parent_key='', sep='.'):
    """Flatten nested dict for search."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# =====================================================
# Session Validation
# =====================================================
message_id = st.session_state.get("debug_message_id")

if not message_id:
    st.warning("⚠ No message selected from main page.")
    st.stop()

st.markdown(f"**Message ID:** `{message_id}`")

data = fetch_raw_state(message_id)

if not data:
    st.error("Failed to retrieve raw state.")
    st.stop()

raw_state = data.get("raw_state")

if not raw_state:
    st.info("No raw state available.")
    st.stop()

# =====================================================
# EXECUTION SUMMARY METRICS
# =====================================================
st.divider()
st.subheader("📊 Execution Summary")

col1, col2, col3, col4 = st.columns(4)

visited = raw_state.get("visited_nodes", [])
refine_count = raw_state.get("refinement_retries", 0)
rag_complete = raw_state.get("rag_complete", False)
tools_used = len(raw_state.get("active_tools", []))

col1.metric("Nodes Visited", len(visited))
col2.metric("Refinement Retries", refine_count)
col3.metric("Tools Activated", tools_used)
col4.metric("RAG Complete", "Yes" if rag_complete else "No")

# =====================================================
# SEARCH INSIDE STATE
# =====================================================
st.divider()
st.subheader("🔎 Search Inside State")

search_term = st.text_input("Search key or value")

if search_term:
    flat = flatten_dict(raw_state)
    matches = {k: v for k, v in flat.items() if search_term.lower() in str(k).lower() or search_term.lower() in str(v).lower()}

    if matches:
        st.success(f"Found {len(matches)} matches")
        st.json(matches)
    else:
        st.warning("No matches found")

# =====================================================
# STRUCTURED TABS
# =====================================================
st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🧭 Execution Path",
    "🧠 Reasoning",
    "📚 Retrieval",
    "🛠 Tools",
    "📄 Final Output",
    "📦 Full State"
])

# ---------------- PATH ----------------
with tab1:
    st.subheader("Visited Nodes")
    if visited:
        st.code(" → ".join(visited))
        st.json(visited)
    else:
        st.info("No nodes recorded.")

# ---------------- REASONING ----------------
with tab2:
    st.subheader("Reasoning Layer")
    st.json({
        "primary_intent": raw_state.get("primary_intent"),
        "semantic_query": raw_state.get("semantic_query"),
        "execution_plan": raw_state.get("execution_plan"),
        "needs_refinement": raw_state.get("needs_refinement"),
        "refinement_reason": raw_state.get("refinement_reason"),
        "refinement_retries": raw_state.get("refinement_retries")
    })

# ---------------- RETRIEVAL ----------------
with tab3:
    st.subheader("Retrieval Pipeline")
    st.json({
        "tiered_queries": raw_state.get("tiered_queries"),
        "material_elements": raw_state.get("material_elements"),
        "filtered_context": raw_state.get("filtered_context"),
        "references": raw_state.get("references")
    })

# ---------------- TOOLS ----------------
with tab4:
    st.subheader("Tool Layer")
    st.json({
        "active_tools": raw_state.get("active_tools"),
        "raw_tool_data": raw_state.get("raw_tool_data")
    })

# ---------------- FINAL ----------------
with tab5:
    st.subheader("Final Report Output")
    st.markdown(raw_state.get("final_report", "No report generated."))

# ---------------- FULL STATE ----------------
with tab6:
    view_mode = st.radio("View Mode", ["Structured JSON", "Raw Text"], horizontal=True)

    if view_mode == "Structured JSON":
        st.json(raw_state, expanded=False)
    else:
        st.code(json.dumps(raw_state, indent=2), language="json")

# =====================================================
# FOOTER CONTROLS
# =====================================================
st.divider()

colA, colB = st.columns(2)

with colA:
    if st.button("🔄 Refresh State"):
        st.cache_data.clear()
        st.rerun()

with colB:
    if st.button("⬅ Back to Main UI"):
        st.switch_page("ui_main.py")
