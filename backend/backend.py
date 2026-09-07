import os
import re
import uuid
import traceback
import json
import base64
import asyncio
from typing import Optional, List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy import create_engine, String, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column, declarative_base, sessionmaker

from dotenv import load_dotenv, find_dotenv

# --- PROJECT SPECIFIC IMPORTS ---
from backend.core.research_state import ResearchState
from backend.graph.research_graph import ResearchGraph
from backend.core.vector_db import VectorDBWrapper
from backend.core.utilities import (
    C_CYAN, C_RESET, C_ACTION, C_GREEN,
    C_RED, C_MAGENTA, C_BLUE
)

# --- GLOBAL VARIABLES & EXECUTOR CONFIGURATION ---
executor = ThreadPoolExecutor(max_workers=5)
app = FastAPI(title="Research Agent API with SQLite Logging")

research_workflow_instance: Optional[ResearchGraph] = None
db_wrapper: Optional[VectorDBWrapper] = None
research_agent_app: Any = None

# ------------------------------------------------------------------------------
# SECTION 1: MODULE IMPORTS AND CONFIGURATION
# ------------------------------------------------------------------------------
print(f" {C_ACTION}>> [INIT] Loading necessary modules and configuration.{C_RESET}")
load_dotenv(find_dotenv())
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(" >> [FATAL] Missing GEMINI_API_KEY in environment")
print(f" {C_GREEN}>> [INIT] Environment variables loaded successfully.{C_RESET}")
client = genai.Client()

# ------------------------------------------------------------------------------
# SECTION 2: DATABASE SETUP (SQLite)
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATABASE_URL = f"sqlite:///{BASE_DIR / 'chat_history.db'}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ChatLog(Base):
    __tablename__ = "chat_logs"

    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    role: Mapped[str] = mapped_column(String)
    message: Mapped[str] = mapped_column(Text)
    tool_used: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    raw_data: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    visited_nodes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


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
    except Exception as exc:
        print(f" {C_RED}>> [FATAL STARTUP ERROR] {traceback.format_exc()}{C_RESET}")
        raise exc


# ------------------------------------------------------------------------------
# SECTION 4: HELPER FUNCTIONS (UTF-8 FIREWALL & NORMALIZATION)
# ------------------------------------------------------------------------------

def log_to_db(msg_id, session_id, role, message, tool_used=None, raw_data=None, visited_nodes=None):
    db = SessionLocal()
    try:
        visited_nodes_list = list(visited_nodes) if visited_nodes else []
        visited_str = json.dumps(visited_nodes_list)

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
    if isinstance(data, list):
        return [_cleanse_recursive_state(item) for item in data]
    if isinstance(data, dict):
        return {k: _cleanse_recursive_state(v) for k, v in data.items()}
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
                    if node_a == node_b:
                        continue
            clean_lines.append(line)
        sanitized_mermaid = "\n".join(clean_lines)
        encoded_string = base64.b64encode(sanitized_mermaid.encode('utf-8')).decode('utf-8')
        image_url = f"https://mermaid.ink/img/{encoded_string}"
        return {"mermaid_syntax": sanitized_mermaid, "image_url": image_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


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
        visited_path = list(raw_path)
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
        log_to_db(
            msg_id=err_id,
            session_id=session_id,
            role="error",
            message=str(e),
            raw_data={"traceback": error_trace}
        )
        raise HTTPException(
            status_code=500,
            detail={"error": "Agent execution failed", "message": str(e)}
        ) from e


@app.get("/chat-history/{session_id}", response_model=List[ChatEntry])
async def get_chat_history(session_id: str):
    db = SessionLocal()
    try:
        logs = db.query(ChatLog).filter(ChatLog.session_id == session_id).order_by(ChatLog.timestamp.asc()).all()

        def _safe_json_parse(data_str: Optional[str], default: Any) -> Any:
            if not data_str:
                return default
            try:
                return json.loads(data_str)
            except (json.JSONDecodeError, TypeError):
                return data_str

        entries: List[ChatEntry] = []
        for log in logs:
            raw_str = str(log.raw_data) if log.raw_data is not None else None
            parsed_raw = _safe_json_parse(raw_str, raw_str) if raw_str and raw_str.startswith(('{', '[')) else raw_str

            nodes_str = str(log.visited_nodes) if log.visited_nodes is not None else ""
            parsed_nodes = _safe_json_parse(nodes_str, [])
            visited_list = [str(n) for n in parsed_nodes] if isinstance(parsed_nodes, list) else []

            entries.append(
                ChatEntry(
                    id=str(log.id),
                    timestamp=log.timestamp,
                    role=str(log.role),
                    message=str(log.message),
                    tool_used=log.tool_used,
                    raw_data=parsed_raw,
                    visited_nodes=visited_list
                )
            )
        return entries
    except Exception as e:
        print(f" >> [HISTORY ERROR] {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Database retrieval failed") from e
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
        raise HTTPException(status_code=500, detail="Error fetching session list") from e
    finally:
        db.close()


# --------------------------------------------------------------------
# ---------------------- FOR AI TRANSPARENCY -------------------------
# --------------------------------------------------------------------
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
        except (json.JSONDecodeError, TypeError):
            parsed = log.raw_data

        return {
            "id": log.id,
            "session_id": log.session_id,
            "raw_state": parsed
        }

    finally:
        db.close()

# ========================== GPT-5 API Integration ==========================
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
# executor = ThreadPoolExecutor(max_workers=5)

# app = FastAPI(title="Research Agent API with SQLite Logging")

# # ------------------------------------------------------------------------------
# # SECTION 1: MODULE IMPORTS AND CONFIGURATION
# # ------------------------------------------------------------------------------
# print(f" {C_ACTION}>> [INIT] Loading necessary modules and configuration.{C_RESET}")
# load_dotenv()
# OPENAI_API_KEY = os.getenv("GPT_5_API_KEY") or os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     raise ValueError(f"{C_RED} >> [FATAL] Missing OPENAI_API_KEY or GPT_API_KEY in environment{C_RESET}")
# print(f" {C_GREEN}>> [INIT] Environment variables loaded successfully.{C_RESET}")

# # ------------------------------------------------------------------------------
# # SECTION 2: DATABASE SETUP (SQLite)
# # ------------------------------------------------------------------------------
# from pathlib import Path

# BASE_DIR = Path(__file__).resolve().parent
# DATABASE_URL = f"sqlite:///{BASE_DIR / 'chat_history.db'}"

# #DATABASE_URL = "sqlite:///./chat_history.db?check_same_thread=False&timeout=20"
# #DATABASE_URL = "sqlite:////app/backend/chat_history.db?check_same_thread=False&timeout=20"
# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# class ChatLog(Base):
#     __tablename__ = "chat_logs"
#     id = Column(String, primary_key=True, index=True)
#     session_id = Column(String, index=True)
#     timestamp = Column(DateTime)
#     role = Column(String)
#     message = Column(Text)
#     tool_used = Column(String)
#     raw_data = Column(Text)
#     visited_nodes = Column(Text)

# Base.metadata.create_all(bind=engine)
# print(f" {C_GREEN}>> [INIT] Database structure verified/created.{C_RESET}")

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

# class ChatEntry(BaseModel):
#     id: str  # PER-TURN TRACKING
#     timestamp: datetime
#     role: str
#     message: str
#     tool_used: Optional[str] = None
#     raw_data: Optional[Union[str, Dict[str, Any], List[Any]]] = None
#     visited_nodes: Optional[List[str]] = None

# # ------------------------------------------------------------------------------
# # SECTION 3.A: CRITICAL STARTUP INITIALIZATION
# # ------------------------------------------------------------------------------
# @app.on_event("startup")
# async def startup_event():
#     global research_workflow_instance, db_wrapper, research_agent_app
#     print(f" {C_CYAN}>> [API STARTUP] Initializing Research System...{C_RESET}")
#     try:
#         db_wrapper = VectorDBWrapper()
#         db_wrapper.reset_db()
#         research_workflow_instance = ResearchGraph(vector_db=db_wrapper)
#         research_agent_app = research_workflow_instance.graph
#         print(f" {C_GREEN}>> [API STARTUP] Initialization successful: Graph compiled and DB ready.{C_RESET}")
#     except Exception as e:
#         print(f" {C_RED}>> [FATAL STARTUP ERROR] {traceback.format_exc()}{C_RESET}")
#         pass

# # ------------------------------------------------------------------------------
# # SECTION 4: HELPER FUNCTIONS (UTF-8 FIREWALL & NORMALIZATION)
# # ------------------------------------------------------------------------------

# def log_to_db(msg_id, session_id, role, message, tool_used=None, raw_data=None, visited_nodes=None):
#     db = SessionLocal()
#     try:
#         # POINT 3: Translate Agent Names to Mermaid IDs
#         if visited_nodes:
#             visited_nodes = [n for n in visited_nodes]
#             #visited_nodes = [n if n != "retrieval_agent" else "retrieve_data" for n in visited_nodes]

#         # POINT 2: Schema Enforcement
#         visited_str = json.dumps(visited_nodes) if visited_nodes else "[]"

#         # Safe serialization for raw_data with Cleansing
#         if raw_data is not None:
#             clean_raw = _cleanse_recursive_state(raw_data)
#             raw_str = json.dumps(clean_raw)
#         else:
#             raw_str = ""

#         db.add(
#             ChatLog(
#                 id=msg_id,
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
#     if not isinstance(text, str):
#         return ""
#     surrogate_pattern = re.compile(r'[\ud800-\udfff]')
#     safe_text = surrogate_pattern.sub('', text)
#     try:
#         return safe_text.encode('utf-8', 'ignore').decode('utf-8').strip()
#     except Exception:
#         return safe_text.strip()

# def _cleanse_recursive_state(data: Any) -> Any:
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
#     print(f" {C_ACTION}>> [HOME] Health check called.{C_RESET}")
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
#         mermaid_code = research_agent_app.get_graph().draw_mermaid()
#         lines = mermaid_code.split("\n")
#         clean_lines = []
#         for line in lines:
#             line = line.replace("<p>", "").replace("</p>", "")
#             if "-.->" in line or "-->" in line:
#                 parts = line.strip().split()
#                 if len(parts) >= 3:
#                     node_a = parts[0].strip()
#                     node_b = parts[2].strip().replace(";", "")
#                     if node_a == node_b: continue
#             clean_lines.append(line)
#         sanitized_mermaid = "\n".join(clean_lines)
#         encoded_string = base64.b64encode(sanitized_mermaid.encode('utf-8')).decode('utf-8')
#         image_url = f"https://mermaid.ink/img/{encoded_string}"
#         return {"mermaid_syntax": sanitized_mermaid, "image_url": image_url}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/research-chat")
# async def research_chat(q: Query):
#     if not q.message:
#         raise HTTPException(status_code=400, detail="Message cannot be empty")
#     if not research_agent_app or not db_wrapper:
#         raise HTTPException(status_code=503, detail="Research system not initialized.")

#     session_id = q.session_id or str(uuid.uuid4())
#     user_msg_id = str(uuid.uuid4())
#     print(f"\n{C_BLUE}>> [CHAT START] Session: {session_id[:8]} | Query: {q.message[:50]}...{C_RESET}")

#     log_to_db(msg_id=user_msg_id, session_id=session_id, role="user", message=q.message)

#     try:
#         db_wrapper.reset_db()
#         print(f"{C_CYAN} >> [SYSTEM] Vector Database reset.{C_RESET}")

#         initial_state: ResearchState = {
#             "user_query": q.message,
#             "semantic_query": "",
#             "primary_intent": "",
#             "reasoning": "",
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
#             "next": "supervisor_agent",
#             "visited_nodes": []
#         }

#         print(f"{C_MAGENTA} >> [AGENT] Invoking Research Workflow...{C_RESET}")
#         result = await asyncio.get_running_loop().run_in_executor(
#             executor,
#             lambda: research_agent_app.invoke(initial_state, config={"recursion_limit": 60})
#         )

#         cleansed_result = _cleanse_recursive_state(result)
#         final_report = cleansed_result.get("final_report", "Error: No report generated.")
#         raw_path = cleansed_result.get("visited_nodes", [])
#         visited_path = [n for n in raw_path]
#         #visited_path = [n if n != "retrieval_agent" else "retrieve_data" for n in raw_path]
#         agent_msg_id = str(uuid.uuid4())

#         log_to_db(
#             msg_id=agent_msg_id,
#             session_id=session_id,
#             role="agent",
#             message=final_report,
#             tool_used="SynthesisAgent",
#             visited_nodes=visited_path,
#             raw_data=cleansed_result
#         )

#         print(f"{C_GREEN} >> [CHAT SUCCESS] Report generated.{C_RESET}")

#         return {
#             "id": agent_msg_id,
#             "session_id": session_id,
#             "response": final_report,
#             "visited_path": visited_path,
#             "metadata": {
#                 "refinement_retries": cleansed_result.get("refinement_retries", 0),
#                 "execution_time": datetime.now(timezone.utc).isoformat()
#             }
#         }

#     except Exception as e:
#         error_trace = traceback.format_exc()
#         err_id = str(uuid.uuid4())
#         print(f"{C_RED} >> [AGENT ERROR] {error_trace}{C_RESET}")
#         log_to_db(msg_id=err_id, session_id=session_id, role="error", message=str(e), raw_data={"traceback": error_trace})
#         raise HTTPException(status_code=500, detail={"error": "Agent execution failed", "message": str(e)})

# @app.get("/chat-history/{session_id}", response_model=List[ChatEntry])
# async def get_chat_history(session_id: str):
#     db = SessionLocal()
#     try:
#         logs = db.query(ChatLog).filter(ChatLog.session_id == session_id).order_by(ChatLog.timestamp.asc()).all()

#         def _safe_json_parse(data_str, default):
#             if not data_str: return default
#             try: return json.loads(data_str)
#             except: return data_str

#         return [
#             ChatEntry(
#                 id=log.id,
#                 timestamp=log.timestamp,
#                 role=log.role,
#                 message=log.message,
#                 tool_used=log.tool_used,
#                 raw_data=_safe_json_parse(log.raw_data, log.raw_data) if log.raw_data and log.raw_data.startswith(('{','[')) else log.raw_data,
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
#     db = SessionLocal()
#     try:
#         session_ids = db.query(ChatLog.session_id).distinct().all()
#         session_list = []
#         for (sid,) in session_ids:
#             last_log = db.query(ChatLog).filter(ChatLog.session_id == sid).order_by(ChatLog.timestamp.desc()).first()
#             if last_log:
#                 session_list.append({
#                     "session_id": sid,
#                     "last_msg": last_log.message[:100],
#                     "last_ts": last_log.timestamp.isoformat()
#                 })
#         return session_list
#     except Exception as e:
#         print(f" >> [LIST SESSIONS ERROR] {traceback.format_exc()}")
#         raise HTTPException(status_code=500, detail="Error fetching session list")
#     finally:
#         db.close()

# #--------------------------------------------------------------------
# #---------------------- FOR AI TRANSPERANCY--------------------------
# #--------------------------------------------------------------------
# @app.get("/debug/raw-state/{message_id}")
# async def get_raw_state(message_id: str):
#     db = SessionLocal()
#     try:
#         log = db.query(ChatLog).filter(ChatLog.id == message_id).first()

#         if not log:
#             raise HTTPException(status_code=404, detail="Message not found")

#         if not log.raw_data:
#             return {"raw_data": None}

#         try:
#             parsed = json.loads(log.raw_data)
#         except:
#             parsed = log.raw_data

#         return {
#             "id": log.id,
#             "session_id": log.session_id,
#             "raw_state": parsed
#         }

#     finally:
#         db.close()
