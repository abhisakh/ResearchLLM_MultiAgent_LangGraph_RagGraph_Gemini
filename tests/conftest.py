"""Shared test fixtures.

Design rules for the unit tests:
  * No network, no API keys, no torch. Gemini is replaced by FakeGeminiClient, embeddings by a
    deterministic bag-of-words hash, and the two heavy third-party packages (sentence-transformers,
    mp-api) by stubs at import time.
  * The JSON files in examples/sample_outputs are used as realistic input states. Each call to
    `sample_state(name)` loads a fresh copy, so a test can mutate it freely.
"""
import importlib
import json
import re
import sys
import types
import zlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "sample_outputs"


# --------------------------------------------------------------------------------------
# Request pacing
# --------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def no_request_pacing(monkeypatch):
    """Unit tests never wait for the Gemini requests-per-minute pacing, whatever LLM_REQUESTS_PER_MINUTE says in .env."""
    from backend.core import utilities
    monkeypatch.setattr(utilities, "LLM_PACER", utilities.RequestPacer(0))


# --------------------------------------------------------------------------------------
# Sample states
# --------------------------------------------------------------------------------------
@pytest.fixture
def sample_state():
    """sample_state("level_2_test_input") -> a fresh dict loaded from examples/sample_outputs."""
    def load(name: str) -> dict:
        with open(SAMPLES_DIR / f"{name}.json", encoding="utf-8") as f:
            return json.load(f)
    return load


# --------------------------------------------------------------------------------------
# Fake Gemini
# --------------------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, text, **attrs):
        self.text = text
        self.__dict__.update(attrs)  # e.g. prompt_feedback=..., candidates=[...] to imitate a blocked answer


class FakeLLM:
    """Replaces generate_content_with_retry in an agent module.

    Each output is the answer text (str), a FakeResponse, or an exception to raise. The n-th call
    gets the n-th output; calls beyond the last output repeat it. Keyword arguments of every call
    are recorded in `.calls` (so tests can look at the prompt and count the retries).
    """

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        output = self.outputs[min(len(self.calls), len(self.outputs)) - 1]
        if isinstance(output, BaseException):
            raise output
        return output if isinstance(output, FakeResponse) else FakeResponse(output)


@pytest.fixture
def fake_llm():
    """The FakeLLM class: fake_llm("answer"), fake_llm("bad", "good"), fake_llm(LLMCallError(...))."""
    return FakeLLM


class FakeGeminiClient:
    """Stands in for google.genai.Client. Returns canned text, or raises `error` if given.
    Every call's keyword arguments are recorded in `.calls`."""

    def __init__(self, text="", error=None):
        self.calls = []
        self._text = text
        self._error = error
        self.models = types.SimpleNamespace(generate_content=self._generate)

    def _generate(self, *args, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return FakeResponse(self._text)


@pytest.fixture
def fake_gemini():
    """The FakeGeminiClient class: fake_gemini(text="...") or fake_gemini(error=RuntimeError())."""
    return FakeGeminiClient


@pytest.fixture
def fake_response():
    return FakeResponse


# --------------------------------------------------------------------------------------
# Deterministic embeddings and reranker
# --------------------------------------------------------------------------------------
def _fake_embedding(text: str, dim: int) -> np.ndarray:
    """Hashed bag of words, L2-normalised. Texts that share words get a higher cosine similarity."""
    vec = np.zeros(dim, dtype=np.float32)
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        vec[zlib.crc32(token.encode()) % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


class FakeCrossEncoder:
    """Score = number of words the query and the passage share."""

    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):
        return [float(len(set(q.lower().split()) & set(p.lower().split()))) for q, p in pairs]


@pytest.fixture
def vector_db_module(monkeypatch, tmp_path):
    """backend.core.vector_db wired to a temp folder and fake embeddings (no model download)."""
    import backend.core.vector_db as vdb

    monkeypatch.setattr(vdb, "VECTOR_INDEX_PATH", str(tmp_path / "vector_index.faiss"))
    monkeypatch.setattr(vdb, "VECTOR_DATA_PATH", str(tmp_path / "vector_data.pkl"))
    monkeypatch.setattr(vdb, "EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(vdb, "preload_embedding_model", lambda: None)
    monkeypatch.setattr(vdb, "get_embedding", lambda text: _fake_embedding(text, vdb.DIMENSION))
    return vdb


# --------------------------------------------------------------------------------------
# Importing modules whose dependencies are too heavy for the fast test job
# --------------------------------------------------------------------------------------
def _stub_module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


@contextmanager
def isolated_import(module_name: str, stubs: dict):
    """Import `module_name` with `stubs` (name -> module) standing in for heavy packages.
    Everything is restored on exit, so other tests see the real modules."""
    with pytest.MonkeyPatch.context() as mp:
        for name, stub in stubs.items():
            mp.setitem(sys.modules, name, stub)
        mp.delitem(sys.modules, module_name, raising=False)
        module = importlib.import_module(module_name)
        try:
            yield module
        finally:
            sys.modules.pop(module_name, None)


@pytest.fixture(scope="module")
def rag_module():
    """backend.agents.rag_agents with sentence-transformers replaced by FakeCrossEncoder."""
    stubs = {
        "sentence_transformers": _stub_module(
            "sentence_transformers", CrossEncoder=FakeCrossEncoder, SentenceTransformer=FakeCrossEncoder
        )
    }
    with isolated_import("backend.agents.rag_agents", stubs) as module:
        yield module


@pytest.fixture(scope="module")
def tool_agents_module():
    """backend.agents.tool_agents with the Materials Project client replaced by a stub."""
    stubs = {
        "mp_api": _stub_module("mp_api"),
        "mp_api.client": _stub_module("mp_api.client", MPRester=object),
    }
    with isolated_import("backend.agents.tool_agents", stubs) as module:
        yield module


@pytest.fixture(autouse=True)
def no_request_pacing(request, monkeypatch):
    """Tests must not depend on the developer's .env: a LLM_REQUESTS_PER_MINUTE set there would make every
    test that calls Gemini (through its fakes) wait or fail. Tests of the pacing itself build their own pacer.
    The live integration test keeps the real setting: it talks to the real API and needs the pacing."""
    if request.node.get_closest_marker("integration"):
        return
    from backend.core import utilities
    monkeypatch.setattr(utilities, "LLM_PACER", utilities.RequestPacer(0))


@pytest.fixture
def api(monkeypatch, tmp_path):
    """The real FastAPI app (backend.backend) with a scripted graph in place of the research graph.

    Yields (client, script). Set script["result"] to the final state the graph returns, or script["raises"]
    to an exception it raises. The chat log is written to a temporary SQLite file.
    """
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlalchemy")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("CHAT_DB_PATH", str(tmp_path / "chat_history.db"))

    script = {"result": None, "raises": None}

    class ScriptedApp:
        def invoke(self, state, config=None):
            if script["raises"] is not None:
                raise script["raises"]
            return script["result"]

    class ScriptedGraph:
        def __init__(self, vector_db=None, agents=None):
            self.graph = ScriptedApp()

    class FakeVectorDB:
        def reset_db(self):
            pass

    stubs = {
        "backend.graph.research_graph": _stub_module("backend.graph.research_graph", ResearchGraph=ScriptedGraph),
        "backend.core.vector_db": _stub_module("backend.core.vector_db", VectorDBWrapper=FakeVectorDB),
    }
    with isolated_import("backend.backend", stubs) as module:
        with TestClient(module.app) as client:  # entering the client runs the app's lifespan (startup)
            yield client, script
