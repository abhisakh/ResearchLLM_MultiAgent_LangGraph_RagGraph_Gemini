################################################################################
# PROGRAM NAME: Research Agent Workflow (Backend.py)
# DESCRIPTION:  FastAPI backend for Research Agent.
# CRITICAL:     Ensure 'graph.research_graph' provides the compiled app.
################################################################################

import os
import datetime
import re
import uuid
import traceback
import json
from typing import Optional, List, Dict, Any, Union
import time
import base64

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from dotenv import load_dotenv

import asyncio
from concurrent.futures import ThreadPoolExecutor

# --- CRITICAL CORRECTIONS TO IMPORTS ---
# Import the ResearchState structure for correct initial state creation
from core.research_state import ResearchState
# Import the Graph Builder class and the actual visualization logic
from graph.research_graph import ResearchGraph, route_after_evaluation # Importing ResearchGraph class
from core.vector_db import VectorDBWrapper # Explicitly import DB wrapper for initialization
# ---------------------------------------

executor = ThreadPoolExecutor(max_workers=2)

# Global variables for the initialized system
research_workflow_instance: Optional[ResearchGraph] = None
db_wrapper: Optional[VectorDBWrapper] = None # Global access to the vector DB wrapper
research_agent_app = None # Placeholder for the compiled LangGraph

# ------------------------------------------------------------------------------
# SECTION 1: MODULE IMPORTS AND CONFIGURATION
# ------------------------------------------------------------------------------
print(" >> [INIT] Loading necessary modules and configuration.")
load_dotenv()
# Using the GPT_API_KEY from the environment for initialization
OPENAI_API_KEY = os.getenv("GPT_API_KEY")
if not OPENAI_API_KEY:
    # Use the more general key as a fallback for the system check
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        # NOTE: Using ValueError here, although HTTPException is preferred in FastAPI endpoints
        raise ValueError(" >> [FATAL] Missing OPENAI_API_KEY or GPT_API_KEY in environment")
print(" >> [INIT] Environment variables loaded successfully.")

# ------------------------------------------------------------------------------
# SECTION 2: DATABASE SETUP (SQLite)
# ------------------------------------------------------------------------------
DATABASE_URL = "sqlite:///./chat_history.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ChatLog(Base):
    __tablename__ = "chat_logs"
    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, index=True)
    timestamp = Column(DateTime)
    role = Column(String)
    message = Column(Text)
    tool_used = Column(String)
    raw_data = Column(Text)


Base.metadata.create_all(bind=engine)
print(" >> [INIT] Database structure verified/created.")


# ------------------------------------------------------------------------------
# SECTION 3: API SETUP AND MODELS
# ------------------------------------------------------------------------------
app = FastAPI(title="Research Agent API with SQLite Logging")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatEntry(BaseModel):
    timestamp: datetime.datetime
    role: str
    message: str
    tool_used: Optional[str] = None
    # CHANGE: Allow both strings and dictionaries
    raw_data: Optional[Union[str, Dict[str, Any]]] = None

# ------------------------------------------------------------------------------
# SECTION 3.A: CRITICAL STARTUP INITIALIZATION
# ------------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Initializes the VectorDB and the ResearchGraph on startup."""
    global research_workflow_instance, db_wrapper, research_agent_app
    print(" >> [API STARTUP] Initializing Research System...")

    try:
        # 1. Initialize VectorDB
        db_wrapper = VectorDBWrapper()
        db_wrapper.reset_db()

        # 2. Initialize and Compile LangGraph
        research_workflow_instance = ResearchGraph(vector_db=db_wrapper)
        research_agent_app = research_workflow_instance.graph

        print(" >> [API STARTUP] Initialization successful: Graph compiled and DB ready.")
    except Exception as e:
        print(f" >> [FATAL STARTUP ERROR] {traceback.format_exc()}")
        # NOTE: In production, you might want to raise an exception to halt startup
        # For development, we log and continue.
        pass

# ------------------------------------------------------------------------------
# SECTION 4: HELPER FUNCTIONS (No changes needed, indentation is assumed correct)
# ------------------------------------------------------------------------------
def log_to_db(session_id: str, role: str, message: str, tool_used: Optional[str] = None, raw_data: Any = None):
    # ... (function body remains the same)
    db = SessionLocal()
    try:
        raw_str = json.dumps(raw_data) if isinstance(raw_data, (dict, list)) else str(raw_data)

        db.add(
            ChatLog(
                id=str(uuid.uuid4()),
                session_id=session_id,
                timestamp=datetime.datetime.utcnow(),
                role=role,
                message=message,
                tool_used=tool_used,
                raw_data=raw_str,
            )
        )
        db.commit()
        # print(f" >> [DB LOG] Successfully logged message. Role: {role}, Tool: {tool_used}")
    except Exception as e:
        print(f" >> [DB ERROR] Failed to log message: {e}")
    finally:
        db.close()

def _cleanse_text_data_ultimate(text: str) -> str:
    """The safety function to remove problematic surrogate characters."""
    if not isinstance(text, str):
        return ""
    surrogate_pattern = re.compile(r'[\ud800-\udfff]')
    safe_text = surrogate_pattern.sub('', text)
    try:
        return safe_text.encode('utf-8', 'ignore').decode('utf-8').strip()
    except Exception:
        return safe_text.strip()

def _cleanse_recursive_state(data: Any) -> Any:
    """Recursively cleanses all strings within a dictionary or list."""
    if isinstance(data, str):
        return _cleanse_text_data_ultimate(data)
    elif isinstance(data, list):
        return [_cleanse_recursive_state(item) for item in data]
    elif isinstance(data, dict):
        return {k: _cleanse_recursive_state(v) for k, v in data.items()}
    else:
        return data

# ------------------------------------------------------------------------------
# SECTION 5: API ENDPOINTS
# ------------------------------------------------------------------------------

@app.get("/")
async def home():
    print(" >> [HOME] Health check called.")
    return {
        "status": "running",
        "agent_status": "initialized" if research_agent_app else "failed_initialization",
        "storage": "SQLite",
    }


# @app.get("/graph-visualization")
# async def get_graph_visualization():
#     # Use a dummy function or the actual visualization if you've defined it.
#     # NOTE: Since visualize_graph was not defined in our components, we assume
#     # the visualization needs the compiled LangGraph object and a router function.

#     if not research_agent_app:
#          raise HTTPException(status_code=503, detail="Agent graph not initialized.")

#     print(" >> [GRAPH] Request received. Generating visualization...")
#     loop = asyncio.get_running_loop()
#     try:
#         # NOTE: LangGraph's .get_graph().draw_png is the standard way.
#         # This requires the router function (e.g., route_after_evaluation) for context.

#         # We'll use a placeholder for visualize_graph since its definition is missing
#         # but the intent is clear: draw the graph.

#         # Placeholder for drawing the graph:
#         def draw_graph_to_png(app_graph, router_func):
#              # This function is missing but would typically use app_graph.get_graph().draw_png()
#              # For now, we return a small, dummy PNG byte array to avoid failure,
#              # but the user must implement the graphviz visualization.
#              # Returning a 1x1 black pixel PNG for placeholder
#              return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90\x77\x53\xde\x00\x00\x00\x0cIDATx\xda`\x01\x00\x00\x00\x00\x00\x00\xfc\x03\x00\x01\x00\x01\x05\xfe\xf1\x18\x00\x00\x00\x00IEND\xaeB`\x82'


#         png_bytes = await loop.run_in_executor(
#             executor,
#             draw_graph_to_png,
#             research_agent_app,
#             route_after_evaluation # Use a router function for context
#         )

#         if not png_bytes or len(png_bytes) < 100:
#             raise RuntimeError("Graphviz visualization failed (dummy content used).")

#         print(" >> [GRAPH] Visualization generation intended.")
#         return Response(content=png_bytes, media_type="image/png")
#     except Exception:
#         print(f" >> [GRAPH ERROR] {traceback.format_exc()}")
#         raise HTTPException(status_code=500, detail="Error generating graph visualization.")


@app.get("/graph-visualization")
async def get_graph_visualization():
    if not research_agent_app:
         raise HTTPException(status_code=503, detail="Agent graph not initialized.")

    try:
        # 1. Generate the raw Mermaid syntax from LangGraph
        mermaid_code = research_agent_app.get_graph().draw_mermaid()

        # 2. SANITIZE: Clean messy HTML tags and redundant self-loops
        lines = mermaid_code.split("\n")
        clean_lines = []
        for line in lines:
            # Strip <p> tags that break the Logic Engine view
            line = line.replace("<p>", "").replace("</p>", "")

            # Filter out "Spiderweb" self-loops (Agent -.-> Agent)
            if "-.->" in line or "-->" in line:
                parts = line.strip().split()
                if len(parts) >= 3:
                    node_a = parts[0].strip()
                    node_b = parts[2].strip().replace(";", "")
                    if node_a == node_b: continue
            clean_lines.append(line)

        sanitized_mermaid = "\n".join(clean_lines)

        # 3. Generate the PNG link for fallback
        encoded_string = base64.b64encode(sanitized_mermaid.encode('ascii')).decode('ascii')
        image_url = f"https://mermaid.ink/img/{encoded_string}"

        return {
            "mermaid_syntax": sanitized_mermaid,
            "image_url": image_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research-chat")
async def research_chat(q: Query):
    if not q.message:
        return {"error": "Message cannot be empty"}

    if not research_agent_app or not db_wrapper:
        raise HTTPException(status_code=503, detail="Research system not initialized.")

    session_id = q.session_id or str(uuid.uuid4())
    print(f" >> [CHAT] Start: Session ID {session_id[:8]}..., Query: '{q.message[:40]}...'")

    log_to_db(session_id=session_id, role="user", message=q.message)

    try:
        # --- CRITICAL FIX: RESET THE VECTOR DATABASE ---
        db_wrapper.reset_db()
        print(" >> [SYSTEM RESET] Vector Database successfully cleared.")
        # ----------------------------------------------

        # NOTE: Initialize ALL keys using the imported ResearchState structure
        initial_state: ResearchState = {
            "user_query": q.message, "semantic_query": "", "primary_intent": "", "execution_plan": [],
            "material_elements": [], "api_search_term": "", "tiered_queries": {}, "active_tools": [],
            "raw_tool_data": [], "full_text_chunks": [], "rag_complete": False, "filtered_context": "",
            "references": [], "final_report": "", "report_generated": False, "needs_refinement": False,
            "refinement_reason": "", "is_refining": False, "refinement_retries": 0, "next": "supervisor" # Start at supervisor
        }

        # EXECUTION: Use .invoke() to get the final state synchronously
        print(" >> [AGENT EXEC] Invoking synchronous workflow...")

        result = await asyncio.get_running_loop().run_in_executor(
            executor,
            research_agent_app.invoke,
            initial_state,
            {"recursion_limit": 100}
        )

        print(" >> [AGENT EXEC] Workflow finished.")

        # --- RECURSIVELY CLEANSE THE ENTIRE RESULT PAYLOAD ---
        cleansed_result = _cleanse_recursive_state(result)
        # ------------------------------------------------------------------

        # EXTRACT RESULTS from the CLEANNSED object
        final_report_text = cleansed_result.get("final_report", "Agent failed to produce a final report.")
        final_tool_used_for_log = "SynthesisAgent"

        # Logging Data Preparation
        log_raw_data_dict = {
            "original_query": q.message,
            "report_preview": final_report_text[:200] + "...",
            "status": "Report Finalized",
        }

        # Log agent response
        log_to_db(
            session_id=session_id,
            role="agent",
            message=final_report_text,
            tool_used=final_tool_used_for_log,
            raw_data=log_raw_data_dict,
        )
        print(f" >> [CHAT] Success. Final report generated ({len(final_report_text)} chars).")

        # Return the cleansed data
        return {
            "session_id": session_id,
            "response": final_report_text,
            "tool_used": final_tool_used_for_log,
            "raw_data": log_raw_data_dict,
            "aggregated_subtasks": cleansed_result, # Return the full cleansed final state for debugging
        }

    except Exception as e:
        print(f" >> [CHAT ERROR] Critical error during workflow execution: {str(e)}")
        error_trace = traceback.format_exc()

        clean_error_message = _cleanse_text_data_ultimate(f"Agent crashed during execution: {str(e)}")

        log_to_db(
            session_id=session_id,
            role="agent_error",
            message=clean_error_message,
            tool_used="error_handler",
            raw_data=_cleanse_text_data_ultimate(error_trace),
        )

        raise HTTPException(
            status_code=500,
            detail={"error": "An internal agent error occurred.", "traceback": error_trace}
        )


@app.get("/chat-history/{session_id}", response_model=List[ChatEntry])
async def get_chat_history(session_id: str):
    # ... (function body remains the same, assumes correct implementation)
    db = SessionLocal()
    try:
        logs = (
            db.query(ChatLog)
            .filter(ChatLog.session_id == session_id)
            .order_by(ChatLog.timestamp)
            .all()
        )
        print(f" >> [HISTORY] Retrieved {len(logs)} messages for session {session_id[:8]}...")

        def _safe_deserialize_raw_data(data_str: str) -> Any:
            if not data_str or not isinstance(data_str, str): return data_str
            try:
                if data_str.strip().startswith(('{', '[')):
                    return json.loads(data_str)
            except json.JSONDecodeError:
                pass
            return data_str

        return [
            ChatEntry(
                timestamp=log.timestamp,
                role=log.role,
                message=log.message,
                tool_used=log.tool_used,
                raw_data=_safe_deserialize_raw_data(log.raw_data),
            ) for log in logs
        ]
    except Exception as e:
        print(f" >> [HISTORY ERROR] Failed to fetch history: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error fetching chat history")
    finally:
        db.close()


@app.get("/list-sessions")
async def list_sessions():
    # ... (function body remains the same, assumes correct implementation)
    db = SessionLocal()
    try:
        session_ids = db.query(ChatLog.session_id).distinct().all()
        session_list = []

        for (sid,) in session_ids:
            last_log = (
                db.query(ChatLog)
                .filter(ChatLog.session_id == sid)
                .order_by(ChatLog.timestamp.desc())
                .first()
            )
            if last_log:
                session_list.append({
                    "session_id": sid,
                    "last_msg": last_log.message[:100],
                    "last_ts": last_log.timestamp.isoformat()
                })

        print(f" >> [LIST SESSIONS] Retrieved {len(session_list)} sessions.")
        return session_list

    except Exception as e:
        print(f" >> [LIST SESSIONS ERROR] Failed to fetch session list: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error fetching session list")
    finally:
        db.close()