"""The benchmark's metrics and runner, tested with fake runs: no API, no network, no heavy imports."""
import json
from pathlib import Path

import pytest

from benchmarks.metrics import (
    REJECTION_TEXT, citations_in, claim_sentences, evidence_by_url, grounding_scores,
    measure_run, render_markdown, report_body, summarise,
)
from benchmarks.run import STOP_KINDS, load_questions, run_benchmark, write_results

QUESTIONS_FILE = Path(__file__).resolve().parents[1] / "benchmarks" / "questions.json"

GROUNDED = "The ground state phase shows a bandgap of 2.5181 eV with zero energy above hull [1]."
UNGROUNDED = "Quantum gravity resolves the black hole information paradox completely [1]."
EVIDENCE = "The ground state phase shows a bandgap of 2.5181 eV with zero energy above hull in this computed entry."
REPORT = (
    f"## Findings\n\n* **Phase:** {GROUNDED}\n* {UNGROUNDED}\n\n"
    "Note: this short remark is ignored entirely by the count.\n\n"
    "## References\n\n[1] [Entry](https://example.org/a) (Web)"
)
QUESTION = {"id": "q", "question": "Q?", "expect": "report"}
USAGE = {"calls": 9, "prompt_tokens": 1000, "output_tokens": 500, "thinking_tokens": 200}


def finished_state(**overrides):
    state = {
        "final_report": REPORT,
        "sources": [{"number": 1, "title": "Entry", "url": "https://example.org/a", "tool_id": "web_search"}],
        # the trailing slash must not stop the chunk from being matched to its source
        "context_chunks": [{"chunk_id": "web_search_a_0", "doc_id": "https://example.org/a/", "source": "web_search", "text": EVIDENCE}],
        "raw_tool_data": [
            {"tool_id": "web_search", "text": "snippet", "metadata": {"url": "https://example.org/a"}},
            {"tool_id": "arxiv_search", "text": "other", "metadata": {"pdf_url": "https://example.org/b"}},
        ],
        "invalid_citations_removed": 1,
        "refinement_retries": 1,
        "visited_nodes": ["clean_query_agent", "web_search", "synthesis_agent"],
        "failure": None,
    }
    state.update(overrides)
    return state


# --------------------------------------------------------------------------------------
# Reading the report
# --------------------------------------------------------------------------------------
def test_the_report_body_stops_before_the_reference_list():
    body = report_body(REPORT)
    assert GROUNDED in body and "Entry" not in body
    assert report_body("No heading here [1].") == "No heading here [1]."
    assert report_body("") == ""


def test_citations_are_listed_one_per_marker():
    assert citations_in("a [1] b [2][2] c") == [1, 2, 2]
    assert citations_in("") == []


def test_claim_sentences_skip_headings_notes_and_short_lines():
    assert claim_sentences(report_body(REPORT)) == [GROUNDED, UNGROUNDED]


# --------------------------------------------------------------------------------------
# Grounding
# --------------------------------------------------------------------------------------
def test_grounding_is_high_when_the_cited_text_has_the_terms_and_zero_when_it_has_none():
    state = finished_state()
    evidence = evidence_by_url(state)

    scores = grounding_scores(report_body(REPORT), state["sources"], evidence)

    assert scores == [1.0, 0.0]


def test_a_sentence_citing_a_source_without_an_address_is_not_scored():
    state = finished_state(sources=[{"number": 1, "title": "No link", "url": None, "tool_id": ""}])
    assert grounding_scores(report_body(REPORT), state["sources"], evidence_by_url(state)) == []


def test_materials_project_entries_are_matched_by_their_generated_address():
    state = {"raw_tool_data": [{
        "tool_id": "materials_search", "text": "CsPbI3 energy above hull 0.0 eV",
        "metadata": {"material_id": "mp-1", "formula": "CsPbI3"},
    }]}
    assert "cspbi3 energy above hull" in evidence_by_url(state)["https://next-gen.materialsproject.org/materials/mp-1"].lower()


# --------------------------------------------------------------------------------------
# One run and many runs
# --------------------------------------------------------------------------------------
def test_a_finished_run_is_measured():
    record = measure_run(QUESTION, finished_state(), 123.456, USAGE)

    assert record["status"] == "ok" and record["failure_kind"] is None
    assert record["seconds"] == 123.5
    assert (record["requests"], record["prompt_tokens"], record["output_tokens"], record["thinking_tokens"]) == (9, 1000, 500, 200)
    assert record["refinement_cycles"] == 1
    assert record["documents_retrieved"] == 2 and record["sources_cited"] == 1
    assert record["citation_markers"] == 2
    assert record["invalid_citations_removed"] == 1
    assert record["invalid_citation_rate"] == round(1 / 3, 3)   # one invented citation against two valid ones
    assert record["claim_sentences"] == 2 and record["citation_coverage"] == 1.0
    assert record["grounding_score"] == 0.5
    assert record["passed"] is True


def test_a_run_that_stopped_on_an_error_has_no_quality_numbers_and_does_not_pass():
    record = measure_run(QUESTION, finished_state(failure={"kind": "quota_exhausted", "agent": "planning_agent"}), 5, USAGE)

    assert record["status"] == "failed" and record["failure_kind"] == "quota_exhausted"
    assert record["citation_coverage"] is None and record["grounding_score"] is None
    assert record["report"] is None and record["weakest_cited_sentences"] is None
    assert record["passed"] is False


def test_the_record_keeps_the_report_and_its_weakest_cited_sentences_for_auditing():
    record = measure_run(QUESTION, finished_state(), 5.0, USAGE)

    assert record["report"] == REPORT
    # the claim whose words are not in the source it cites comes first, so a reader can check the lowest scores by eye
    assert record["weakest_cited_sentences"] == [
        {"score": 0.0, "sentence": UNGROUNDED},
        {"score": 1.0, "sentence": GROUNDED},
    ]


def test_a_report_question_with_no_cited_source_does_not_pass():
    assert measure_run(QUESTION, finished_state(sources=[]), 5, USAGE)["passed"] is False


def test_an_out_of_scope_question_passes_only_when_refused_without_any_search():
    question = {"id": "r", "question": "Weather?", "expect": "reject"}
    refused = {"final_report": REJECTION_TEXT, "visited_nodes": ["clean_query_agent", "intent_agent", "synthesis_agent"]}

    assert measure_run(question, refused, 3, {"calls": 2})["passed"] is True
    assert measure_run(question, dict(refused, visited_nodes=refused["visited_nodes"] + ["arxiv_search"]), 3, {"calls": 2})["passed"] is False
    assert measure_run(question, dict(refused, final_report="A long report."), 3, {"calls": 2})["passed"] is False


def test_the_summary_averages_only_completed_report_questions():
    report = measure_run(QUESTION, finished_state(), 100.0, USAGE)
    refused = measure_run({"id": "r", "question": "W?", "expect": "reject"},
                          {"final_report": REJECTION_TEXT, "visited_nodes": ["synthesis_agent"]}, 4.0, {"calls": 2})
    failed = measure_run(QUESTION, finished_state(failure={"kind": "crash"}), 1.0, {"calls": 1})

    summary = summarise([report, refused, failed])

    assert (summary["questions"], summary["completed"], summary["failed"], summary["passed"]) == (3, 2, 1, 2)
    assert summary["total_requests"] == 12
    assert summary["mean_seconds"] == 100.0 and summary["mean_requests"] == 9.0   # the refusal and the failure are excluded
    assert summary["mean_tokens"] == 1700 and summary["mean_grounding_score"] == 0.5


def test_a_summary_of_nothing_does_not_crash():
    summary = summarise([])
    assert summary["questions"] == 0 and summary["mean_seconds"] is None and summary["median_seconds"] is None


def test_the_results_table_names_every_question_and_flags_problems():
    ok = measure_run(dict(QUESTION, id="alpha"), finished_state(evaluation_unavailable=True), 10.0, USAGE)
    stopped = measure_run(dict(QUESTION, id="beta"), finished_state(failure={"kind": "quota_exhausted"}), 1.0, USAGE)
    records = [ok, stopped]

    page = render_markdown(records, summarise(records), {"date": "2026-11-04 09:00 UTC", "model": "m", "requests_per_minute": 5})

    assert "| alpha | ok (unreviewed) |" in page
    assert "| beta | stopped: quota_exhausted |" in page
    assert "100%" in page and "Completed 1 of 2" in page and "2026-11-04 m" in page


# --------------------------------------------------------------------------------------
# The question file and the runner
# --------------------------------------------------------------------------------------
def test_the_shipped_question_file_is_valid():
    questions = load_questions(QUESTIONS_FILE)

    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)) and len(ids) >= 3
    assert {q["expect"] for q in questions} == {"report", "reject"}


def test_questions_can_be_picked_by_id_and_limited():
    assert [q["id"] for q in load_questions(QUESTIONS_FILE, only=["weather-hamburg"])] == ["weather-hamburg"]
    assert len(load_questions(QUESTIONS_FILE, limit=2)) == 2
    with pytest.raises(ValueError, match="Unknown question id"):
        load_questions(QUESTIONS_FILE, only=["nope"])


def test_a_malformed_question_file_is_refused(tmp_path):
    bad = tmp_path / "q.json"
    bad.write_text(json.dumps([{"id": "x", "question": "Q?", "expect": "maybe"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="Bad benchmark question"):
        load_questions(bad)


class Harness:
    """A fake pipeline: every question costs `cost` requests and answers with the state the script provides."""

    def __init__(self, states):
        self.states, self.asked = list(states), []
        self.usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}
        self.ticks = 0.0
        self.log = []

    def run_query(self, text):
        self.asked.append(text)
        self.usage["calls"] += 7
        state = self.states.pop(0)
        if isinstance(state, Exception):
            raise state
        return state

    def snapshot(self):
        return dict(self.usage)

    def clock(self):
        self.ticks += 5.0
        return self.ticks


def run_with(harness, questions):
    return run_benchmark(questions, harness.run_query, harness.snapshot, clock=harness.clock, log=harness.log.append)


def questions_named(*ids):
    return [{"id": i, "question": f"{i}?", "expect": "report"} for i in ids]


def test_the_runner_measures_each_question_with_its_own_cost():
    harness = Harness([finished_state(), finished_state()])

    records = run_with(harness, questions_named("a", "b"))

    assert [r["id"] for r in records] == ["a", "b"]
    assert all(r["requests"] == 7 and r["seconds"] == 5.0 for r in records)   # each question's own share, not the running total
    assert harness.asked == ["a?", "b?"]


@pytest.mark.parametrize("kind", sorted(STOP_KINDS))
def test_the_runner_stops_when_the_cause_would_stop_every_later_question(kind):
    harness = Harness([finished_state(), finished_state(failure={"kind": kind}), finished_state()])

    records = run_with(harness, questions_named("a", "b", "c"))

    assert [r["id"] for r in records] == ["a", "b"]
    assert harness.asked == ["a?", "b?"]   # the third question was never asked
    assert any("Stopping" in line for line in harness.log)


def test_a_crash_in_one_question_is_recorded_and_the_next_one_still_runs():
    harness = Harness([RuntimeError("boom"), finished_state()])

    records = run_with(harness, questions_named("a", "b"))

    assert records[0]["status"] == "failed" and records[0]["failure_kind"] == "crash"
    assert records[1]["status"] == "ok"


def test_results_are_written_as_json_and_a_markdown_table(tmp_path):
    records = [measure_run(QUESTION, finished_state(), 10.0, USAGE)]
    meta = {"date": "2026-11-04 09:00 UTC", "model": "m", "requests_per_minute": 5}

    json_path, markdown_path = write_results(records, meta, tmp_path / "results")

    assert json_path.name == "20261104_0900_UTC.json" and markdown_path.name == "latest.md"
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(saved) == {"meta", "summary", "records"} and saved["records"][0]["id"] == "q"
    assert markdown_path.read_text(encoding="utf-8").startswith("# Benchmark results")


def test_a_page_with_no_completed_report_says_there_are_no_averages():
    records = [measure_run(QUESTION, finished_state(failure={"kind": "quota_exhausted"}), 1.0, USAGE)]

    page = render_markdown(records, summarise(records), {"date": "2026-11-04 09:00 UTC", "model": "m"})

    assert "No completed report question yet" in page


def run_file(tmp_path, date, records, order=None):
    """Writes the results of one run into tmp_path and returns the text of latest.md."""
    write_results(records, {"date": date, "model": "m", "requests_per_minute": 5}, tmp_path, order=order)
    return (tmp_path / "latest.md").read_text(encoding="utf-8")


def test_the_page_keeps_the_newest_completed_result_of_every_question_across_runs(tmp_path):
    first = [measure_run(dict(QUESTION, id="a"), finished_state(), 10.0, USAGE),
             measure_run(dict(QUESTION, id="b"), finished_state(), 21.0, USAGE)]
    second = [measure_run(dict(QUESTION, id="b"), finished_state(), 33.0, USAGE),
              measure_run(dict(QUESTION, id="c"), finished_state(failure={"kind": "quota_exhausted"}), 1.0, USAGE)]

    run_file(tmp_path, "2026-11-04 09:00 UTC", first)
    page = run_file(tmp_path, "2026-11-05 09:00 UTC", second, order=["a", "b", "c"])

    assert "| a | ok |" in page and "2026-11-04 m" in page   # a is still there, from the first day
    assert "| b | ok |" in page and "33.0" in page and "21.0" not in page   # b was replaced by the newer run
    assert "| c |" not in page   # a run that stopped on an error does not appear, and replaces nothing


def test_a_broken_run_file_does_not_stop_the_page_from_being_written(tmp_path):
    (tmp_path / "20261101_0900_UTC.json").write_text("not json", encoding="utf-8")

    page = run_file(tmp_path, "2026-11-05 09:00 UTC", [measure_run(QUESTION, finished_state(), 10.0, USAGE)])

    assert "| q | ok |" in page


def test_when_no_run_has_completed_the_page_shows_this_runs_failure(tmp_path):
    page = run_file(tmp_path, "2026-11-05 09:00 UTC",
                    [measure_run(QUESTION, finished_state(failure={"kind": "auth"}), 1.0, USAGE)])

    assert "| q | stopped: auth |" in page
