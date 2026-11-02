"""Adjacent citations are listed in ascending order and never repeat (SynthesisAgent._finalize_citations)."""
from backend.agents.synthesis_agent import SynthesisAgent


def shown_sources(count):
    """The numbered sources the model was shown."""
    return [
        {"number": n, "title": f"Source number {n}", "url": f"https://example.org/{n}",
         "tool_id": "arxiv_search", "passages": ["text"]}
        for n in range(1, count + 1)
    ]


def finalize(report, count=5):
    return SynthesisAgent()._finalize_citations(report, shown_sources(count))


def test_an_out_of_order_run_is_sorted():
    # The model's Conclusion cited "[4][3]" after both had been introduced earlier in the report.
    report, cited, removed = finalize("A [1]. B [2][3]. C [4][3].")

    assert "C [3][4]." in report
    assert [source["number"] for source in cited] == [1, 2, 3, 4]
    assert removed == 0


def test_a_repeated_citation_in_a_run_is_written_once():
    report, cited, _ = finalize("A [2][2].")

    assert "A [1]." in report
    assert [source["title"] for source in cited] == ["Source number 2"]


def test_runs_are_sorted_by_number_not_as_text():
    body = " ".join(f"Claim {n} [{n}]." for n in range(1, 11)) + " Last [10][9]."
    report, _, _ = finalize(body, count=10)

    assert "Last [9][10]." in report


def test_citations_that_are_not_adjacent_stay_where_they_are():
    report, _, _ = finalize("A [1], [2]. B [2] and [1].")

    assert "A [1], [2]." in report
    assert "B [2] and [1]." in report


def test_sorting_does_not_change_which_sources_are_listed_or_how_they_are_numbered():
    report, cited, _ = finalize("A [3]. B [5][1]. C [1][5].")

    # first appearance: 3 -> 1, 5 -> 2, 1 -> 3
    assert [source["title"] for source in cited] == ["Source number 3", "Source number 5", "Source number 1"]
    assert "B [2][3]." in report   # [5][1] became [2][3]: already ascending
    assert "C [3][2]." not in report and "C [2][3]." in report   # [1][5] became [3][2], then sorted
    assert "[1] [Source number 3]" in report.split("## References")[1]
