"""RetrievalAgent (level 3) and RAGAgent (level 4), without network, torch or model downloads.

Retrieval's page fetch is replaced by canned pages; RAG uses fake embeddings and a fake
cross-encoder. These replace the CLI harness that used to live in rag_agents.py.
"""
import hashlib

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


def test_chunking_keeps_a_long_sentence_whole_and_never_yields_an_empty_chunk(rag_module):
    agent = rag_module.RetrievalAgent(chunk_size=10)  # 35 characters per chunk
    long_sentence = "A" * 100 + "."

    chunks = agent._chunk_text(long_sentence + " Short one. Another short one.")

    assert all(chunks)
    assert chunks[0] == long_sentence  # a sentence over the limit is kept whole, not cut or dropped
    assert all(len(c) <= 35 for c in chunks[1:])
    assert agent._chunk_text("") == []


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


# --------------------------------------------------------------------------------------
# Chunk IDs: stable, unique, and the way neighbours are found
# --------------------------------------------------------------------------------------
def test_chunk_ids_follow_the_documented_format(rag_module):
    url = "https://example.org/a"
    digest = hashlib.sha1(url.encode()).hexdigest()[:12]
    assert rag_module.make_chunk_id("arxiv_search", url, 3) == f"arxiv_search_{digest}_3"
    assert rag_module.make_chunk_id("arxiv_search", url, 3, abstract_only=True) == f"arxiv_search_abs_{digest}_3"


def test_chunk_ids_do_not_collide_across_documents(rag_module):
    # The old scheme kept 4 digits of Python's hash(): 2000 documents of one tool collided dozens of times.
    ids = {rag_module.make_chunk_id("web_search", f"https://example.org/paper/{n}", 0) for n in range(2000)}
    assert len(ids) == 2000


def test_the_trailing_number_of_an_id_is_the_chunk_position(rag_module, rag_agent):
    for abstract_only in (False, True):
        chunk = {"chunk_id": rag_module.make_chunk_id("pubmed_search", "https://example.org/x", 7, abstract_only)}
        assert rag_agent._get_chunk_idx(chunk) == 7


def test_retrieval_gives_the_same_chunk_ids_on_every_run(rag_module, sample_state, monkeypatch):
    runs = []
    for _ in range(2):
        state = sample_state("level_2_test_input")
        page = {first_url(state, "pubmed_search"): {"type": "html", "data": ARTICLE_HTML}}
        agent = make_agent(rag_module, monkeypatch, page)
        runs.append([c["chunk_id"] for c in agent.execute(state)["full_text_chunks"]])

    assert runs[0] == runs[1]
    assert len(set(runs[0])) == len(runs[0])  # no two chunks share an ID


def test_two_matching_documents_both_keep_their_neighbours(rag_module, rag_agent):
    documents = {
        "https://a.org/1": ["Intro about perovskite solar cells.", "CsSnI3 bandgap stability result.", "Outlook for tin devices."],
        "https://b.org/2": ["Background on perovskite films.", "CsSnI3 bandgap stability measured.", "Conclusions for tin films."],
    }
    chunks = [
        {"chunk_id": rag_module.make_chunk_id("arxiv_search", url, i), "doc_id": url, "source": "arxiv_search", "text": text}
        for url, texts in documents.items() for i, text in enumerate(texts)
    ]

    context = rag_agent.execute(rag_state(chunks))["filtered_context"]

    for texts in documents.values():
        for text in texts:
            assert text in context


def test_a_chunk_without_an_id_is_used_on_its_own(rag_agent):
    chunks = [
        {"doc_id": DOC, "source": "web_search", "text": "CsSnI3 bandgap stability without any id."},
        {"chunk_id": "arxiv_search_1_0", "doc_id": DOC, "source": "arxiv_search", "text": "Introduction to lead-free perovskite solar cells."},
    ]

    state = rag_agent.execute(rag_state(chunks))

    assert "CsSnI3 bandgap stability without any id." in state["filtered_context"]
    assert state["rag_complete"] is True


# --------------------------------------------------------------------------------------
# _chunk_text
# --------------------------------------------------------------------------------------
@pytest.fixture
def chunker(rag_module):
    return rag_module.RetrievalAgent(chunk_size=10)  # 35 characters per chunk


def test_chunk_text_of_nothing_is_nothing(chunker):
    assert chunker._chunk_text("") == []
    assert chunker._chunk_text("   \n ") == []


def test_chunk_text_packs_whole_sentences_up_to_the_limit(chunker):
    text = "One sentence here. Another sentence there. Third one."
    assert chunker._chunk_text(text) == ["One sentence here.", "Another sentence there. Third one."]


def test_chunk_text_normalises_whitespace(chunker):
    assert chunker._chunk_text("a   b\n\nc.") == ["a b c."]


def test_a_first_sentence_longer_than_the_limit_does_not_produce_an_empty_chunk(chunker):
    long_sentence = "x" * 100 + "."
    assert chunker._chunk_text(f"{long_sentence} Next one.") == [long_sentence, "Next one."]


# --------------------------------------------------------------------------------------
# PDF extraction applies the text repair
# --------------------------------------------------------------------------------------
class _FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class _FakeReader:
    def __init__(self, stream):
        self.pages = [_FakePage("The reHlecting layer."), _FakePage("It improves efHiciency of BaTiO3."), _FakePage(None)]


def test_pdf_text_is_repaired_when_extracted(rag_module, monkeypatch):
    from io import BytesIO

    monkeypatch.setattr(rag_module, "PdfReader", _FakeReader)
    text = rag_module.RetrievalAgent()._extract_text_from_pdf(BytesIO(b"not a real pdf"))
    assert text == "The reflecting layer. It improves efficiency of BaTiO3. "  # the empty page adds a trailing space


def test_an_unreadable_pdf_gives_no_text(rag_module, monkeypatch):
    from io import BytesIO

    def broken(stream):
        raise ValueError("not a pdf")

    monkeypatch.setattr(rag_module, "PdfReader", broken)
    assert rag_module.RetrievalAgent()._extract_text_from_pdf(BytesIO(b"")) == ""


def test_rag_keeps_where_each_selected_chunk_came_from(rag_agent):
    state = rag_agent.execute(rag_state(CHUNKS))

    selected = state["context_chunks"]
    assert selected
    assert all(set(c) == {"chunk_id", "doc_id", "source", "text"} for c in selected)
    assert all(c["text"] in state["filtered_context"] for c in selected)  # what synthesis may cite is what it is shown


def test_rag_with_no_chunks_selects_no_context_chunks(rag_agent):
    assert rag_agent.execute(rag_state([]))["context_chunks"] == []


@pytest.mark.parametrize(
    "text, source, passes",
    [
        ("Access denied. Enable JavaScript to continue.", "web_search", False),  # web noise without the searched term
        ("Access denied. CsSnI3 is stable.", "web_search", True),  # noise, but it does contain the searched term
        ("Lead-free perovskites are promising.", "web_search", True),  # no noise at all
        ("Please accept our cookie policy.", "arxiv_search", True),  # trusted source, by its real tool id
        ("Please accept our cookie policy.", "chemrxiv_search", True),
        ("Please accept our cookie policy.", None, False),  # a chunk without a source must not crash the gate
    ],
)
def test_keyword_gate_drops_web_noise_but_trusts_preprint_servers(rag_agent, text, source, passes):
    assert rag_agent._passes_keyword_gate(text, "cssni3", source) is passes
