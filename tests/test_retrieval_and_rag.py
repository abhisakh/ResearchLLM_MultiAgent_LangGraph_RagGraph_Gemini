"""RetrievalAgent (level 3) and RAGAgent (level 4), without network, torch or model downloads.

Retrieval's page fetch is replaced by canned pages; RAG uses fake embeddings and a fake
cross-encoder. These replace the CLI harness that used to live in rag_agents.py.
"""
import pytest

from backend.core.research_state import new_research_state

PAGE_TEXT = "Sentence about CsSnI3 stability. " * 40
ARTICLE_HTML = f"<html><body><nav>menu</nav><article>{PAGE_TEXT}</article></body></html>"


# --------------------------------------------------------------------------------------
# RetrievalAgent
# --------------------------------------------------------------------------------------
@pytest.fixture
def retrieval_state(sample_state):
    return sample_state("level_2_test_input")


def first_url(state, tool_id):
    return next(e["metadata"]["pdf_url"] for e in state["raw_tool_data"] if e["tool_id"] == tool_id)


def make_agent(rag_module, monkeypatch, pages):
    agent = rag_module.RetrievalAgent(chunk_size=60)  # ~210 characters per chunk
    monkeypatch.setattr(agent, "_fetch_content", lambda url: pages.get(url))
    return agent


def test_downloaded_page_is_chunked_and_the_rest_fall_back_to_abstracts(rag_module, retrieval_state, monkeypatch):
    page_url = first_url(retrieval_state, "pubmed_search")
    agent = make_agent(rag_module, monkeypatch, {page_url: {"type": "html", "data": ARTICLE_HTML}})

    state = agent.execute(retrieval_state)

    chunks = state["full_text_chunks"]
    assert chunks
    assert all({"chunk_id", "doc_id", "text", "source"} <= set(c) for c in chunks)

    page_chunks = [c for c in chunks if c["doc_id"] == page_url]
    assert len(page_chunks) > 1
    assert all(len(c["text"]) <= 210 for c in page_chunks)
    assert all("menu" not in c["text"] for c in page_chunks)  # <nav> was stripped

    # every other document with a URL is represented by its abstract
    expected_urls = {
        e["metadata"].get("pdf_url") or e["metadata"].get("url")
        for e in retrieval_state["raw_tool_data"]
        if e["tool_id"] != "materials_search"
    }
    assert {c["doc_id"] for c in chunks} == expected_urls
    assert state["next"] == "supervisor_agent"
    assert state["visited_nodes"][-1] == "retrieval_agent"


def test_materials_project_entries_are_never_retrieved(rag_module, retrieval_state, monkeypatch):
    agent = make_agent(rag_module, monkeypatch, {})
    state = agent.execute(retrieval_state)
    assert all(c["source"] != "materials_search" for c in state["full_text_chunks"])


def test_running_retrieval_again_adds_nothing(rag_module, retrieval_state, monkeypatch):
    agent = make_agent(rag_module, monkeypatch, {first_url(retrieval_state, "arxiv_search"): {"type": "html", "data": ARTICLE_HTML}})
    state = agent.execute(retrieval_state)
    count = len(state["full_text_chunks"])

    state = agent.execute(state)

    assert len(state["full_text_chunks"]) == count


def test_pages_without_extractable_text_fall_back_to_the_abstract(rag_module, retrieval_state, monkeypatch):
    url = first_url(retrieval_state, "pubmed_search")
    agent = make_agent(rag_module, monkeypatch, {url: {"type": "html", "data": "<html><body></body></html>"}})

    state = agent.execute(retrieval_state)

    assert any(c["doc_id"] == url and "Title:" in c["text"] for c in state["full_text_chunks"])


# --------------------------------------------------------------------------------------
# RAGAgent
# --------------------------------------------------------------------------------------
DOC = "https://example.org/doc"
CHUNKS = [
    {"chunk_id": "arxiv_search_1_0", "doc_id": DOC, "source": "arxiv_search",
     "text": "Introduction to lead-free perovskite solar cells."},
    {"chunk_id": "arxiv_search_1_1", "doc_id": DOC, "source": "arxiv_search",
     "text": "CsSnI3 bandgap stability improves with an SnF2 additive."},
    {"chunk_id": "arxiv_search_1_2", "doc_id": DOC, "source": "arxiv_search",
     "text": "Conclusions and outlook for tin perovskites."},
    {"chunk_id": "web_search_2_0", "doc_id": "https://example.org/other", "source": "web_search",
     "text": "Unrelated recipe with garlic and basil."},
]


@pytest.fixture
def rag_agent(rag_module, vector_db_module, monkeypatch):
    monkeypatch.setattr(rag_module, "client", object())  # RAG is skipped when no LLM client exists
    return rag_module.RAGAgent(vector_db=vector_db_module.VectorDBWrapper())


def rag_state(chunks):
    state = new_research_state("CsSnI3 bandgap stability")
    state.update(
        semantic_query="CsSnI3 bandgap stability",
        api_search_term="CsSnI3",
        full_text_chunks=[dict(c) for c in chunks],
        raw_tool_data=[{"tool_id": "materials_search", "text": "Material: CsSnI3 (mp-1). Band Gap: 1.3 eV."}],
    )
    return state


def test_rag_selects_the_relevant_chunk_and_its_neighbours(rag_agent):
    state = rag_agent.execute(rag_state(CHUNKS))

    context = state["filtered_context"]
    assert state["rag_complete"] is True
    assert state["next"] == "supervisor_agent"
    assert context.startswith("--- Structured Data ---\nMaterial: CsSnI3 (mp-1)")
    # the matching chunk plus the chunk before and after it
    for text in (CHUNKS[0]["text"], CHUNKS[1]["text"], CHUNKS[2]["text"]):
        assert text in context
    # the document that matched is listed before the unrelated one
    assert context.index(CHUNKS[1]["text"]) < context.index(CHUNKS[3]["text"])


def test_rag_indexes_the_chunks_it_receives(rag_agent):
    rag_agent.execute(rag_state(CHUNKS))
    assert rag_agent.vector_db.index.ntotal == len(CHUNKS)


def test_rag_with_no_chunks_reports_no_context(rag_agent):
    state = rag_agent.execute(rag_state([]))
    assert state["filtered_context"] == "No relevant context found."
    assert state["rag_complete"] is True
    assert state["next"] == "supervisor_agent"


def test_rag_is_skipped_without_an_llm_client(rag_module, vector_db_module, monkeypatch):
    monkeypatch.setattr(rag_module, "client", None)
    agent = rag_module.RAGAgent(vector_db=vector_db_module.VectorDBWrapper())

    state = agent.execute(rag_state(CHUNKS))

    assert state["filtered_context"] == "No relevant context found."
    assert agent.vector_db.index.ntotal == 0
