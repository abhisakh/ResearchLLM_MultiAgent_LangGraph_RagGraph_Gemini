"""One paper listed under two URLs must be one numbered source (one reference), not two.

Seen in a live run: the same paper came back from web search twice, once from a figshare mirror and once from
the publisher's DOI page, and the report cited both as separate references.
"""
import pytest

from backend.agents.synthesis_agent import SynthesisAgent, _title_key

TITLE = "Improving Phase Stability of α-CsPbI3 through Combined Strain-Doping ..."
FIGSHARE = "https://acs.figshare.com/articles/journal_contribution/Improving_Phase_Stability/29829430"
DOI = "https://pubs.acs.org/doi/10.1021/acsami.5c11492"
OTHER = "https://example.org/another-paper"


def web_entry(url, title, tool_id="web_search"):
    return {"tool_id": tool_id, "text": f"Title: {title}", "metadata": {"title": title, "url": url, "pdf_url": url}}


def chunk(url, text, n=0):
    return {"doc_id": url, "chunk_id": f"web_search_{n}", "text": text, "source": "web_search"}


def collect(raw, chunks):
    return SynthesisAgent()._collect_sources({"raw_tool_data": raw, "context_chunks": chunks})


# --------------------------------------------------------------------------------------
# What is recognised as the same paper
# --------------------------------------------------------------------------------------
def test_the_same_title_under_two_urls_is_one_source():
    sources, snippets_only = collect(
        [web_entry(FIGSHARE, TITLE), web_entry(DOI, TITLE)],
        [chunk(FIGSHARE, "Passage from figshare."), chunk(DOI, "Passage from the DOI page.", 1)],
    )

    assert snippets_only is False
    assert len(sources) == 1
    assert sources[0]["number"] == 1
    assert sources[0]["url"] == FIGSHARE  # the first listing in RAG order is kept
    assert sources[0]["passages"] == ["Passage from figshare.", "Passage from the DOI page."]  # nothing is lost


def test_case_punctuation_and_a_cut_off_title_do_not_matter():
    sources, _ = collect(
        [web_entry(FIGSHARE, TITLE), web_entry(DOI, "Improving phase stability of α-CsPbI3 through combined strain-doping")],
        [chunk(FIGSHARE, "One."), chunk(DOI, "Two.", 1)],
    )
    assert len(sources) == 1


def test_a_second_chunk_of_the_same_document_joins_its_source():
    sources, _ = collect(
        [web_entry(FIGSHARE, TITLE)],
        [chunk(FIGSHARE, "First.", 0), chunk(FIGSHARE, "Second.", 1)],
    )
    assert len(sources) == 1 and sources[0]["passages"] == ["First.", "Second."]


def test_snippets_are_merged_the_same_way():
    sources, snippets_only = collect([web_entry(FIGSHARE, TITLE), web_entry(DOI, TITLE)], [])
    assert snippets_only is True
    assert len(sources) == 1


# --------------------------------------------------------------------------------------
# What must NOT be merged
# --------------------------------------------------------------------------------------
def test_different_papers_stay_separate_and_keep_their_order():
    sources, _ = collect(
        [web_entry(FIGSHARE, TITLE), web_entry(OTHER, "Thermal decomposition of hybrid perovskite films")],
        [chunk(FIGSHARE, "A."), chunk(OTHER, "B.")],
    )
    assert [s["number"] for s in sources] == [1, 2]
    assert [s["url"] for s in sources] == [FIGSHARE, OTHER]


@pytest.mark.parametrize("title", ["Home", "CsPbI3", "Untitled source", ""])
def test_short_or_generic_titles_are_never_merged(title):
    first, second = "https://example.org/a", "https://example.org/b"
    entries = [
        {"tool_id": "web_search", "text": "t", "metadata": {"title": title, "url": first}},
        {"tool_id": "web_search", "text": "t", "metadata": {"title": title, "url": second}},
    ]
    sources, _ = collect(entries, [chunk(first, "A."), chunk(second, "B.")])
    assert len(sources) == 2


def test_materials_project_entries_stay_separate():
    materials = [
        {"tool_id": "materials_search", "text": "Material: CsPbI3 (mp-aaa).", "metadata": {"material_id": "mp-aaa", "formula": "CsPbI3"}},
        {"tool_id": "materials_search", "text": "Material: CsPbI3 (mp-bbb).", "metadata": {"material_id": "mp-bbb", "formula": "CsPbI3"}},
    ]
    sources, _ = collect(materials, [])
    assert [s["title"] for s in sources] == ["CsPbI3 (mp-aaa)", "CsPbI3 (mp-bbb)"]


# --------------------------------------------------------------------------------------
# The effect on the report
# --------------------------------------------------------------------------------------
def test_the_reference_list_names_a_merged_paper_once():
    agent = SynthesisAgent()
    sources, _ = collect(
        [web_entry(FIGSHARE, TITLE), web_entry(DOI, TITLE)],
        [chunk(FIGSHARE, "A."), chunk(DOI, "B.", 1)],
    )

    report, cited, removed = agent._finalize_citations("Strain and doping help [1]. This is confirmed again [1].", sources)

    assert len(cited) == 1 and removed == 0
    assert report.count("Improving Phase Stability") == 1
    assert report.count("(Web)") == 1


# --------------------------------------------------------------------------------------
# _title_key
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("title", [None, "", "   ", "Home", "Untitled source", "Unidentified source", "No Title Available"])
def test_title_key_refuses_titles_that_do_not_identify_a_paper(title):
    assert _title_key(title) is None


def test_title_key_ignores_case_punctuation_and_trailing_dots():
    assert _title_key(TITLE) == _title_key("improving phase stability of α-cspbi3 through combined strain-doping")
    assert _title_key(TITLE) is not None
