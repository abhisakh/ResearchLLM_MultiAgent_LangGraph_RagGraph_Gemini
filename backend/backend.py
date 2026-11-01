import logging
import os
import re
import uuid
import traceback
import json
import base64
import asyncio
from typing import Optional, List, Dict, Any, Union
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy import create_engine, String, Text, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, declarative_base, sessionmaker

# --- PROJECT SPECIFIC IMPORTS ---
from backend.core.research_state import new_research_state
from backend.graph.research_graph import ResearchGraph
from backend.core.vector_db import VectorDBWrapper

logger = logging.getLogger(__name__)

# --- GLOBAL VARIABLES & EXECUTOR CONFIGURATION ---
executor = ThreadPoolExecutor(max_workers=5)

research_workflow_instance: Optional[ResearchGraph] = None
db_wrapper: Optional[VectorDBWrapper] = None
research_agent_app: Any = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Builds the vector DB and research graph before the first request; frees the worker threads on shutdown."""
    global research_workflow_instance, db_wrapper, research_agent_app
    logger.info('>> [API STARTUP] Initializing Research System...')
    try:
        db_wrapper = VectorDBWrapper()
        db_wrapper.reset_db()
        research_workflow_instance = ResearchGraph(vector_db=db_wrapper)
        research_agent_app = research_workflow_instance.graph
        logger.info('>> [API STARTUP] Initialization successful: Graph compiled and DB ready.')
    except Exception:
        logger.exception('>> [FATAL STARTUP ERROR]')
        raise
    yield
    executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Research Agent API with SQLite Logging", lifespan=lifespan)

# ------------------------------------------------------------------------------
# SECTION 1: MODULE IMPORTS AND CONFIGURATION
# ------------------------------------------------------------------------------
logger.info('>> [INIT] Loading necessary modules and configuration.')
# .env is loaded in backend/core/utilities.py (imported above), before this line runs.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(" >> [FATAL] Missing GEMINI_API_KEY in environment")
logger.info('>> [INIT] Environment variables loaded successfully.')

# ------------------------------------------------------------------------------
# SECTION 2: DATABASE SETUP (SQLite)

# 1. Locate the local fallback path
BASE_DIR = Path(__file__).resolve().parent

# 2. Smart Environment Routing Logic
# Google Cloud Run always automatically injects the 'K_SERVICE' environment variable.
CHAT_DB_PATH = os.getenv("CHAT_DB_PATH")  # optional explicit override (used by docker-compose)

if CHAT_DB_PATH:
    logger.info(f'[DB CONFIG] CHAT_DB_PATH set. Using {CHAT_DB_PATH}')
    os.makedirs(Path(CHAT_DB_PATH).parent, exist_ok=True)
    DATABASE_URL = f"sqlite:///{CHAT_DB_PATH}"
elif os.getenv("K_SERVICE"):
    logger.info('[DB CONFIG] Cloud Run environment detected. Routing to Persistent Storage...')

    # Define the persistent directory we will mount inside the Cloud Run container
    MOUNT_DIR = Path("/mnt/db")

    # Create the directory safely if the system container initializes it slowly
    os.makedirs(MOUNT_DIR, exist_ok=True)
    DATABASE_URL = f"sqlite:///{MOUNT_DIR / 'chat_history.db'}"
else:
    logger.info('[DB CONFIG] Local Laptop or GitHub Actions pipeline detected. Routing to Local SQLite...')
    DATABASE_URL = f"sqlite:///{BASE_DIR / 'chat_history.db'}"

# 3. Standard SQLAlchemy Initialization
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
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
logger.info('>> [INIT] Database structure verified/created.')

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
        logger.error(f'>> [DB ERROR] Stabilization Failed: {e}')
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
    return {
        "status": "running",
        "agent_status": "initialized" if research_agent_app else "failed_initialization",
        "storage": "SQLite",
    }


@app.get("/health")
async def health():
    """Lightweight liveness probe for the Docker healthcheck: 200 only once the research graph is ready."""
    if not research_agent_app:
        raise HTTPException(status_code=503, detail="Research system not initialized.")
    return {"status": "ok"}


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
    if not q.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if not research_agent_app or not db_wrapper:
        raise HTTPException(status_code=503, detail="Research system not initialized.")

    session_id = q.session_id or str(uuid.uuid4())
    user_msg_id = str(uuid.uuid4())
    logger.info(f'>> [CHAT START] Session: {session_id[:8]} | Query: {q.message[:50]}...')

    log_to_db(msg_id=user_msg_id, session_id=session_id, role="user", message=q.message)

    try:
        db_wrapper.reset_db()
        logger.info('>> [SYSTEM] Vector Database reset.')

        initial_state = new_research_state(q.message)

        logger.info('>> [AGENT] Invoking Research Workflow...')
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

        logger.info('>> [CHAT SUCCESS] Report generated.')

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
        logger.exception('>> [AGENT ERROR]')
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
        logger.exception('>> [HISTORY ERROR]')
        raise HTTPException(status_code=500, detail="Database retrieval failed") from e
    finally:
        db.close()


@app.get("/list-sessions")
async def list_sessions():
    db = SessionLocal()
    try:
        # One query instead of one per session: join every log row with the newest timestamp of its session.
        newest = (
            db.query(ChatLog.session_id, func.max(ChatLog.timestamp).label("last_ts"))
            .group_by(ChatLog.session_id)
            .subquery()
        )
        rows = (
            db.query(ChatLog)
            .join(newest, (ChatLog.session_id == newest.c.session_id) & (ChatLog.timestamp == newest.c.last_ts))
            .order_by(ChatLog.timestamp.desc())
            .all()
        )
        session_list = []
        seen = set()
        for last_log in rows:
            if last_log.session_id in seen:  # two rows can share the exact same timestamp
                continue
            seen.add(last_log.session_id)
            session_list.append({
                "session_id": last_log.session_id,
                "last_msg": last_log.message[:100],
                "last_ts": last_log.timestamp.isoformat()
            })
        return session_list
    except Exception as e:
        logger.exception('>> [LIST SESSIONS ERROR]')
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
            return {"id": log.id, "session_id": log.session_id, "raw_state": None}

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
