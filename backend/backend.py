import os
import datetime
import re
import uuid
import traceback
import json
import base64
import asyncio
from typing import Optional, List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from dotenv import load_dotenv

# --- PROJECT SPECIFIC IMPORTS ---
from core.research_state import ResearchState
from graph.research_graph import ResearchGraph
from core.vector_db import VectorDBWrapper
from core.utilities import (
    C_CYAN, C_RESET, C_ACTION, C_GREEN,
    C_RED, C_MAGENTA, C_YELLOW, C_BLUE
)

# --- EXECUTOR CONFIGURATION ---
executor = ThreadPoolExecutor(max_workers=5)

app = FastAPI(title="Research Agent API with SQLite Logging")

# ------------------------------------------------------------------------------
# SECTION 1: MODULE IMPORTS AND CONFIGURATION
# ------------------------------------------------------------------------------
print(f" {C_ACTION}>> [INIT] Loading necessary modules and configuration.{C_RESET}")
load_dotenv()
OPENAI_API_KEY = os.getenv("GPT_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(f"{C_RED} >> [FATAL] Missing OPENAI_API_KEY or GPT_API_KEY in environment{C_RESET}")
print(f" {C_GREEN}>> [INIT] Environment variables loaded successfully.{C_RESET}")

# ------------------------------------------------------------------------------
# SECTION 2: DATABASE SETUP (SQLite)
# ------------------------------------------------------------------------------
DATABASE_URL = "sqlite:///./chat_history.db?check_same_thread=False&timeout=20"
engine = create_engine(DATABASE_URL)
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
    visited_nodes = Column(Text)

Base.metadata.create_all(bind=engine)
print(f" {C_GREEN}>> [INIT] Database structure verified/created.{C_RESET}")

# ------------------------------------------------------------------------------
# SECTION 3: API SETUP AND MODELS
# ------------------------------------------------------------------------------
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
    id: str  # PER-TURN TRACKING
    timestamp: datetime
    role: str
    message: str
    tool_used: Optional[str] = None
    raw_data: Optional[Union[str, Dict[str, Any], List[Any]]] = None
    visited_nodes: Optional[List[str]] = None

# ------------------------------------------------------------------------------
# SECTION 3.A: CRITICAL STARTUP INITIALIZATION
# ------------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global research_workflow_instance, db_wrapper, research_agent_app
    print(f" {C_CYAN}>> [API STARTUP] Initializing Research System...{C_RESET}")
    try:
        db_wrapper = VectorDBWrapper()
        db_wrapper.reset_db()
        research_workflow_instance = ResearchGraph(vector_db=db_wrapper)
        research_agent_app = research_workflow_instance.graph
        print(f" {C_GREEN}>> [API STARTUP] Initialization successful: Graph compiled and DB ready.{C_RESET}")
    except Exception as e:
        print(f" {C_RED}>> [FATAL STARTUP ERROR] {traceback.format_exc()}{C_RESET}")
        pass

# ------------------------------------------------------------------------------
# SECTION 4: HELPER FUNCTIONS (UTF-8 FIREWALL & NORMALIZATION)
# ------------------------------------------------------------------------------

def log_to_db(msg_id, session_id, role, message, tool_used=None, raw_data=None, visited_nodes=None):
    db = SessionLocal()
    try:
        # POINT 3: Translate Agent Names to Mermaid IDs
        if visited_nodes:
            visited_nodes = [n for n in visited_nodes]
            #visited_nodes = [n if n != "retrieval_agent" else "retrieve_data" for n in visited_nodes]

        # POINT 2: Schema Enforcement
        visited_str = json.dumps(visited_nodes) if visited_nodes else "[]"

        # Safe serialization for raw_data with Cleansing
        if raw_data is not None:
            clean_raw = _cleanse_recursive_state(raw_data)
            raw_str = json.dumps(clean_raw)
        else:
            raw_str = ""

        db.add(
            ChatLog(
                id=msg_id,
                session_id=session_id,
                timestamp=datetime.now(timezone.utc),
                role=role,
                message=_cleanse_text_data_ultimate(message),
                tool_used=tool_used,
                raw_data=raw_str,
                visited_nodes=visited_str
            )
        )
        db.commit()
    except Exception as e:
        print(f"{C_RED} >> [DB ERROR] Stabilization Failed: {e}{C_RESET}")
    finally:
        db.close()

def _cleanse_text_data_ultimate(text: str) -> str:
    if not isinstance(text, str):
        return ""
    surrogate_pattern = re.compile(r'[\ud800-\udfff]')
    safe_text = surrogate_pattern.sub('', text)
    try:
        return safe_text.encode('utf-8', 'ignore').decode('utf-8').strip()
    except Exception:
        return safe_text.strip()

def _cleanse_recursive_state(data: Any) -> Any:
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
    print(f" {C_ACTION}>> [HOME] Health check called.{C_RESET}")
    return {
        "status": "running",
        "agent_status": "initialized" if research_agent_app else "failed_initialization",
        "storage": "SQLite",
    }

@app.get("/graph-visualization")
async def get_graph_visualization():
    if not research_agent_app:
         raise HTTPException(status_code=503, detail="Agent graph not initialized.")
    try:
        mermaid_code = research_agent_app.get_graph().draw_mermaid()
        lines = mermaid_code.split("\n")
        clean_lines = []
        for line in lines:
            line = line.replace("<p>", "").replace("</p>", "")
            if "-.->" in line or "-->" in line:
                parts = line.strip().split()
                if len(parts) >= 3:
                    node_a = parts[0].strip()
                    node_b = parts[2].strip().replace(";", "")
                    if node_a == node_b: continue
            clean_lines.append(line)
        sanitized_mermaid = "\n".join(clean_lines)
        encoded_string = base64.b64encode(sanitized_mermaid.encode('utf-8')).decode('utf-8')
        image_url = f"https://mermaid.ink/img/{encoded_string}"
        return {"mermaid_syntax": sanitized_mermaid, "image_url": image_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/research-chat")
async def research_chat(q: Query):
    if not q.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if not research_agent_app or not db_wrapper:
        raise HTTPException(status_code=503, detail="Research system not initialized.")

    session_id = q.session_id or str(uuid.uuid4())
    user_msg_id = str(uuid.uuid4())
    print(f"\n{C_BLUE}>> [CHAT START] Session: {session_id[:8]} | Query: {q.message[:50]}...{C_RESET}")

    log_to_db(msg_id=user_msg_id, session_id=session_id, role="user", message=q.message)

    try:
        db_wrapper.reset_db()
        print(f"{C_CYAN} >> [SYSTEM] Vector Database reset.{C_RESET}")

        initial_state: ResearchState = {
            "user_query": q.message,
            "semantic_query": "",
            "primary_intent": "",
            "reasoning": "",
            "execution_plan": [],
            "material_elements": [],
            "system_constraints": [],
            "api_search_term": "",
            "tiered_queries": {},
            "active_tools": [],
            "raw_tool_data": [],
            "full_text_chunks": [],
            "rag_complete": False,
            "filtered_context": "",
            "references": [],
            "final_report": "",
            "report_generated": False,
            "needs_refinement": False,
            "refinement_reason": "",
            "is_refining": False,
            "refinement_retries": 0,
            "next": "supervisor_agent",
            "visited_nodes": []
        }

        print(f"{C_MAGENTA} >> [AGENT] Invoking Research Workflow...{C_RESET}")
        result = await asyncio.get_running_loop().run_in_executor(
            executor,
            lambda: research_agent_app.invoke(initial_state, config={"recursion_limit": 60})
        )

        cleansed_result = _cleanse_recursive_state(result)
        final_report = cleansed_result.get("final_report", "Error: No report generated.")
        raw_path = cleansed_result.get("visited_nodes", [])
        visited_path = [n for n in raw_path]
        #visited_path = [n if n != "retrieval_agent" else "retrieve_data" for n in raw_path]
        agent_msg_id = str(uuid.uuid4())

        log_to_db(
            msg_id=agent_msg_id,
            session_id=session_id,
            role="agent",
            message=final_report,
            tool_used="SynthesisAgent",
            visited_nodes=visited_path,
            raw_data=cleansed_result
        )

        print(f"{C_GREEN} >> [CHAT SUCCESS] Report generated.{C_RESET}")

        return {
            "id": agent_msg_id,
            "session_id": session_id,
            "response": final_report,
            "visited_path": visited_path,
            "metadata": {
                "refinement_retries": cleansed_result.get("refinement_retries", 0),
                "execution_time": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        err_id = str(uuid.uuid4())
        print(f"{C_RED} >> [AGENT ERROR] {error_trace}{C_RESET}")
        log_to_db(msg_id=err_id, session_id=session_id, role="error", message=str(e), raw_data={"traceback": error_trace})
        raise HTTPException(status_code=500, detail={"error": "Agent execution failed", "message": str(e)})

@app.get("/chat-history/{session_id}", response_model=List[ChatEntry])
async def get_chat_history(session_id: str):
    db = SessionLocal()
    try:
        logs = db.query(ChatLog).filter(ChatLog.session_id == session_id).order_by(ChatLog.timestamp.asc()).all()

        def _safe_json_parse(data_str, default):
            if not data_str: return default
            try: return json.loads(data_str)
            except: return data_str

        return [
            ChatEntry(
                id=log.id,
                timestamp=log.timestamp,
                role=log.role,
                message=log.message,
                tool_used=log.tool_used,
                raw_data=_safe_json_parse(log.raw_data, log.raw_data) if log.raw_data and log.raw_data.startswith(('{','[')) else log.raw_data,
                visited_nodes=_safe_json_parse(log.visited_nodes, [])
            ) for log in logs
        ]
    except Exception as e:
        print(f" >> [HISTORY ERROR] {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Database retrieval failed")
    finally:
        db.close()

@app.get("/list-sessions")
async def list_sessions():
    db = SessionLocal()
    try:
        session_ids = db.query(ChatLog.session_id).distinct().all()
        session_list = []
        for (sid,) in session_ids:
            last_log = db.query(ChatLog).filter(ChatLog.session_id == sid).order_by(ChatLog.timestamp.desc()).first()
            if last_log:
                session_list.append({
                    "session_id": sid,
                    "last_msg": last_log.message[:100],
                    "last_ts": last_log.timestamp.isoformat()
                })
        return session_list
    except Exception as e:
        print(f" >> [LIST SESSIONS ERROR] {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error fetching session list")
    finally:
        db.close()

#--------------------------------------------------------------------
#---------------------- FOR AI TRANSPERANCY--------------------------
#--------------------------------------------------------------------
@app.get("/debug/raw-state/{message_id}")
async def get_raw_state(message_id: str):
    db = SessionLocal()
    try:
        log = db.query(ChatLog).filter(ChatLog.id == message_id).first()

        if not log:
            raise HTTPException(status_code=404, detail="Message not found")

        if not log.raw_data:
            return {"raw_data": None}

        try:
            parsed = json.loads(log.raw_data)
        except:
            parsed = log.raw_data

        return {
            "id": log.id,
            "session_id": log.session_id,
            "raw_state": parsed
        }

    finally:
        db.close()

# import os
# import datetime
# import re
# import uuid
# import traceback
# import json
# import base64
# import asyncio
# from typing import Optional, List, Dict, Any, Union
# from concurrent.futures import ThreadPoolExecutor
# from datetime import datetime, timezone

# from fastapi import FastAPI, HTTPException, Response
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# from sqlalchemy import create_engine, Column, String, Text, DateTime
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker

# from dotenv import load_dotenv

# # --- PROJECT SPECIFIC IMPORTS ---
# from core.research_state import ResearchState
# from graph.research_graph import ResearchGraph
# from core.vector_db import VectorDBWrapper
# from core.utilities import (
#     C_CYAN, C_RESET, C_ACTION, C_GREEN,
#     C_RED, C_MAGENTA, C_YELLOW, C_BLUE
# )

# # --- EXECUTOR CONFIGURATION ---
# # We define this globally so the API doesn't create a new thread pool for every request.
# # max_workers=2-5 is usually sufficient for an agentic research workflow.
# executor = ThreadPoolExecutor(max_workers=5)
# # ------------------------------

# app = FastAPI(title="Research Agent API with SQLite Logging")
# # ------------------------------------------------------------------------------
# # SECTION 1: MODULE IMPORTS AND CONFIGURATION
# # ------------------------------------------------------------------------------
# print(" >> [INIT] Loading necessary modules and configuration.")
# load_dotenv()
# # Using the GPT_API_KEY from the environment for initialization
# OPENAI_API_KEY = os.getenv("GPT_API_KEY")
# if not OPENAI_API_KEY:
#     # Use the more general key as a fallback for the system check
#     OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#     if not OPENAI_API_KEY:
#         # NOTE: Using ValueError here, although HTTPException is preferred in FastAPI endpoints
#         raise ValueError(" >> [FATAL] Missing OPENAI_API_KEY or GPT_API_KEY in environment")
# print(" >> [INIT] Environment variables loaded successfully.")

# # ------------------------------------------------------------------------------
# # SECTION 2: DATABASE SETUP (SQLite)
# # ------------------------------------------------------------------------------
# DATABASE_URL = "sqlite:///./chat_history.db"
# engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# # visited_nodes and system_constraints to the schema
# class ChatLog(Base):
#     __tablename__ = "chat_logs"
#     id = Column(String, primary_key=True, index=True)
#     session_id = Column(String, index=True)
#     timestamp = Column(DateTime)
#     role = Column(String)
#     message = Column(Text)
#     tool_used = Column(String)
#     raw_data = Column(Text)
#     visited_nodes = Column(Text) # New Column for path tracking


# Base.metadata.create_all(bind=engine)
# print(" >> [INIT] Database structure verified/created.")


# # ------------------------------------------------------------------------------
# # SECTION 3: API SETUP AND MODELS
# # ------------------------------------------------------------------------------

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# class Query(BaseModel):
#     session_id: Optional[str] = None
#     message: str

# # --- SECTION 3: API MODELS ---
# class ChatEntry(BaseModel):
#     timestamp: datetime  # No more .datetime needed
#     role: str
#     message: str
#     tool_used: Optional[str] = None
#     raw_data: Optional[Union[str, Dict[str, Any]]] = None
#     visited_nodes: Optional[List[str]] = None
# # ------------------------------------------------------------------------------
# # SECTION 3.A: CRITICAL STARTUP INITIALIZATION
# # ------------------------------------------------------------------------------
# @app.on_event("startup")
# async def startup_event():
#     """Initializes the VectorDB and the ResearchGraph on startup."""
#     global research_workflow_instance, db_wrapper, research_agent_app
#     print(" >> [API STARTUP] Initializing Research System...")

#     try:
#         # 1. Initialize VectorDB
#         db_wrapper = VectorDBWrapper()
#         db_wrapper.reset_db()

#         # 2. Initialize and Compile LangGraph
#         research_workflow_instance = ResearchGraph(vector_db=db_wrapper)
#         research_agent_app = research_workflow_instance.graph

#         print(" >> [API STARTUP] Initialization successful: Graph compiled and DB ready.")
#     except Exception as e:
#         print(f" >> [FATAL STARTUP ERROR] {traceback.format_exc()}")
#         # NOTE: In production, you might want to raise an exception to halt startup
#         # For development, we log and continue.
#         pass

# # ------------------------------------------------------------------------------
# # SECTION 4: HELPER FUNCTIONS (No changes needed, indentation is assumed correct)
# # ------------------------------------------------------------------------------

# def log_to_db(
#     session_id: str,
#     role: str,
#     message: str,
#     tool_used: Optional[str] = None,
#     raw_data: Any = None,
#     visited_nodes: Optional[List[str]] = None
# ):
#     """
#     STABILIZED LOGGER:
#     - Normalizes Node IDs for Mermaid Compatibility.
#     - Enforces JSON list structure for 'visited_nodes' column.
#     """
#     db = SessionLocal()
#     try:
#         # POINT 3: Translate Agent Names to Mermaid IDs
#         # This ensures the 'Logic Engine' in the UI always finds the right node to highlight.
#         if visited_nodes:
#             # We convert agent-specific name to the ID used in the Mermaid graph
#             visited_nodes = [n if n != "retrieval_agent" else "retrieve_data" for n in visited_nodes]

#         # POINT 2: Schema Enforcement
#         # We ensure it's a stringified list '[]' so the History API doesn't return None/Null.
#         visited_str = json.dumps(visited_nodes) if visited_nodes else "[]"

#         # Serialization for raw_data
#         raw_str = json.dumps(raw_data) if isinstance(raw_data, (dict, list)) else str(raw_data) if raw_data else ""

#         db.add(
#             ChatLog(
#                 id=str(uuid.uuid4()),
#                 session_id=session_id,
#                 timestamp=datetime.now(timezone.utc),
#                 role=role,
#                 message=_cleanse_text_data_ultimate(message),
#                 tool_used=tool_used,
#                 raw_data=raw_str,
#                 visited_nodes=visited_str
#             )
#         )
#         db.commit()
#     except Exception as e:
#         print(f"{C_RED} >> [DB ERROR] Stabilization Failed: {e}{C_RESET}")
#     finally:
#         db.close()

# def _cleanse_text_data_ultimate(text: str) -> str:
#     """The safety function to remove problematic surrogate characters."""
#     if not isinstance(text, str):
#         return ""
#     surrogate_pattern = re.compile(r'[\ud800-\udfff]')
#     safe_text = surrogate_pattern.sub('', text)
#     try:
#         return safe_text.encode('utf-8', 'ignore').decode('utf-8').strip()
#     except Exception:
#         return safe_text.strip()

# def _cleanse_recursive_state(data: Any) -> Any:
#     """Recursively cleanses all strings within a dictionary or list."""
#     if isinstance(data, str):
#         return _cleanse_text_data_ultimate(data)
#     elif isinstance(data, list):
#         return [_cleanse_recursive_state(item) for item in data]
#     elif isinstance(data, dict):
#         return {k: _cleanse_recursive_state(v) for k, v in data.items()}
#     else:
#         return data

# # ------------------------------------------------------------------------------
# # SECTION 5: API ENDPOINTS
# # ------------------------------------------------------------------------------

# @app.get("/")
# async def home():
#     print(" >> [HOME] Health check called.")
#     return {
#         "status": "running",
#         "agent_status": "initialized" if research_agent_app else "failed_initialization",
#         "storage": "SQLite",
#     }

# @app.get("/graph-visualization")
# async def get_graph_visualization():
#     if not research_agent_app:
#          raise HTTPException(status_code=503, detail="Agent graph not initialized.")

#     try:
#         # 1. Generate the raw Mermaid syntax from LangGraph
#         mermaid_code = research_agent_app.get_graph().draw_mermaid()

#         # 2. SANITIZE: Clean messy HTML tags and redundant self-loops
#         lines = mermaid_code.split("\n")
#         clean_lines = []
#         for line in lines:
#             # Strip <p> tags that break the Logic Engine view
#             line = line.replace("<p>", "").replace("</p>", "")

#             # Filter out "Spiderweb" self-loops (Agent -.-> Agent)
#             if "-.->" in line or "-->" in line:
#                 parts = line.strip().split()
#                 if len(parts) >= 3:
#                     node_a = parts[0].strip()
#                     node_b = parts[2].strip().replace(";", "")
#                     if node_a == node_b: continue
#             clean_lines.append(line)

#         sanitized_mermaid = "\n".join(clean_lines)

#         # 3. Generate the PNG link for fallback
#         encoded_string = base64.b64encode(sanitized_mermaid.encode('ascii')).decode('ascii')
#         image_url = f"https://mermaid.ink/img/{encoded_string}"

#         return {
#             "mermaid_syntax": sanitized_mermaid,
#             "image_url": image_url
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/research-chat")
# async def research_chat(q: Query):
#     """
#     Main entry point for research queries.
#     FULLY STABILIZED: Handles ResearchState initialization and node normalization.
#     """
#     if not q.message:
#         raise HTTPException(status_code=400, detail="Message cannot be empty")

#     if not research_agent_app or not db_wrapper:
#         raise HTTPException(status_code=503, detail="Research system not initialized.")

#     session_id = q.session_id or str(uuid.uuid4())
#     print(f"\n{C_BLUE}>> [CHAT START] Session: {session_id[:8]} | Query: {q.message[:50]}...{C_RESET}")

#     # Log user entry immediately
#     log_to_db(session_id=session_id, role="user", message=q.message)

#     try:
#         # 1. CLEAN START: Prepare the vector database for a fresh search
#         db_wrapper.reset_db()
#         print(f"{C_CYAN} >> [SYSTEM] Vector Database reset.{C_RESET}")

#         # 2. INITIAL STATE: Full compliance with ResearchState TypedDict
#         initial_state: ResearchState = {
#             "user_query": q.message,
#             "semantic_query": "",
#             "primary_intent": "",
#             "execution_plan": [],
#             "material_elements": [],
#             "system_constraints": [],
#             "api_search_term": "",
#             "tiered_queries": {},
#             "active_tools": [],
#             "raw_tool_data": [],
#             "full_text_chunks": [],
#             "rag_complete": False,
#             "filtered_context": "",
#             "references": [],
#             "final_report": "",
#             "report_generated": False,
#             "needs_refinement": False,
#             "refinement_reason": "",
#             "is_refining": False,
#             "refinement_retries": 0,
#             "next": "supervisor_agent",  # Starting node ID
#             "visited_nodes": []          # Path collector list
#         }

#         # 3. EXECUTION: Thread-safe invocation of the LangGraph
#         print(f"{C_MAGENTA} >> [AGENT] Invoking Research Workflow...{C_RESET}")
#         result = await asyncio.get_running_loop().run_in_executor(
#             executor,
#             lambda: research_agent_app.invoke(
#                 initial_state,
#                 config={"recursion_limit": 60}
#             )
#         )

#         # 4. DATA PROCESSING & CLEANSING: Surrogate character removal
#         cleansed_result = _cleanse_recursive_state(result)

#         final_report = cleansed_result.get("final_report", "Error: No report generated.")
#         raw_path = cleansed_result.get("visited_nodes", [])

#         # --- SYNC FIX: Normalize naming for Mermaid UI ---
#         # Converts Agent-node-name to Logic-Engine-ID
#         visited_path = [n if n != "retrieval_agent" else "retrieve_data" for n in raw_path]

#         retries = cleansed_result.get("refinement_retries", 0)

#         # 5. FINAL LOGGING: Ensure visited_nodes column is populated for History Restore
#         log_to_db(
#             session_id=session_id,
#             role="agent",
#             message=final_report,
#             tool_used="SynthesisAgent",
#             visited_nodes=visited_path,  # Normalization applied here
#             raw_data={
#                 "visited_path": visited_path,
#                 "refinement_count": retries,
#                 "is_refined": cleansed_result.get("is_refining", False)
#             }
#         )

#         print(f"{C_GREEN} >> [CHAT SUCCESS] Report generated. Path length: {len(visited_path)}{C_RESET}")

#         return {
#             "session_id": session_id,
#             "response": final_report,
#             "visited_path": visited_path,
#             "metadata": {
#                 "refinement_retries": retries,
#                 "execution_time": datetime.now(timezone.utc).isoformat()
#             }
#         }

#     except Exception as e:
#         error_trace = traceback.format_exc()
#         print(f"{C_RED} >> [AGENT ERROR] {error_trace}{C_RESET}")
#         # Log the error to DB so the UI can show the failure in history
#         log_to_db(session_id=session_id, role="error", message=str(e), raw_data={"traceback": error_trace})
#         raise HTTPException(status_code=500, detail={"error": "Agent execution failed", "message": str(e)})

# @app.get("/chat-history/{session_id}", response_model=List[ChatEntry])
# async def get_chat_history(session_id: str):
#     db = SessionLocal()
#     try:
#         logs = (
#             db.query(ChatLog)
#             .filter(ChatLog.session_id == session_id)
#             .order_by(ChatLog.timestamp)
#             .all()
#         )

#         def _safe_json_parse(data_str: str, default: Any) -> Any:
#             """Helper to prevent JSON decode crashes on historical data."""
#             if not data_str: return default
#             try:
#                 return json.loads(data_str)
#             except (json.JSONDecodeError, TypeError):
#                 return default

#         return [
#             ChatEntry(
#                 timestamp=log.timestamp,
#                 role=log.role,
#                 message=log.message,
#                 tool_used=log.tool_used,
#                 # If it's a JSON string, parse it; otherwise, return the raw string
#                 raw_data=_safe_json_parse(log.raw_data, log.raw_data) if log.raw_data and log.raw_data.startswith(('{','[')) else log.raw_data,
#                 # Always ensure this returns a List[str]
#                 visited_nodes=_safe_json_parse(log.visited_nodes, [])
#             ) for log in logs
#         ]
#     except Exception as e:
#         print(f" >> [HISTORY ERROR] {traceback.format_exc()}")
#         raise HTTPException(status_code=500, detail="Database retrieval failed")
#     finally:
#         db.close()


# @app.get("/list-sessions")
# async def list_sessions():
#     # ... (function body remains the same, assumes correct implementation)
#     db = SessionLocal()
#     try:
#         session_ids = db.query(ChatLog.session_id).distinct().all()
#         session_list = []

#         for (sid,) in session_ids:
#             last_log = (
#                 db.query(ChatLog)
#                 .filter(ChatLog.session_id == sid)
#                 .order_by(ChatLog.timestamp.desc())
#                 .first()
#             )
#             if last_log:
#                 session_list.append({
#                     "session_id": sid,
#                     "last_msg": last_log.message[:100],
#                     "last_ts": last_log.timestamp.isoformat()
#                 })

#         print(f" >> [LIST SESSIONS] Retrieved {len(session_list)} sessions.")
#         return session_list

#     except Exception as e:
#         print(f" >> [LIST SESSIONS ERROR] Failed to fetch session list: {traceback.format_exc()}")
#         raise HTTPException(status_code=500, detail="Error fetching session list")
#     finally:
#         db.close()
