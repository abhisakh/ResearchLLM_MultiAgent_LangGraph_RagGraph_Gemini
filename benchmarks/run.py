"""Runs the benchmark questions through the real pipeline and writes a results table.

    python -m benchmarks.run --limit 1                  # the first question only (about 6-11 Gemini requests)
    python -m benchmarks.run --only weather-hamburg     # the cheap out-of-scope check (about 2 requests)
    python -m benchmarks.run --yes                      # every question, without asking first

It uses the key in .env and spends real requests: on the free tier (20 per day per model) run one or two questions
a day, or set LLM_MODEL_FALLBACKS, or use a billed key. See benchmarks/README.md.
"""
import argparse
import contextlib
import datetime
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from benchmarks.metrics import measure_run, render_markdown, summarise

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = BENCH_DIR / "questions.json"
DEFAULT_OUT_DIR = BENCH_DIR / "results"
VALID_EXPECTATIONS = ("report", "reject")

# A run that stops for one of these reasons would stop every later question in the same way, so the benchmark
# stops too and keeps what it has measured.
STOP_KINDS = frozenset({"quota_exhausted", "auth", "config"})


def load_questions(path: Any, only: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """The benchmark questions, optionally only some ids and/or only the first `limit` of them."""
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    for question in questions:
        if not question.get("id") or not question.get("question") or question.get("expect") not in VALID_EXPECTATIONS:
            raise ValueError(f"Bad benchmark question (needs id, question and expect in {VALID_EXPECTATIONS}): {question}")
    if only:
        wanted = {name.strip() for name in only if name.strip()}
        unknown = wanted - {q["id"] for q in questions}
        if unknown:
            raise ValueError(f"Unknown question id(s): {sorted(unknown)}")
        questions = [q for q in questions if q["id"] in wanted]
    if limit is not None:
        questions = questions[:max(0, limit)]
    return questions


def run_benchmark(
    questions: List[Dict[str, Any]],
    run_query: Callable[[str], Dict[str, Any]],
    usage_snapshot: Callable[[], Dict[str, int]],
    clock: Callable[[], float] = time.monotonic,
    log: Callable[[str], None] = print,
) -> List[Dict[str, Any]]:
    """Runs each question and returns one record per question. `run_query` and `usage_snapshot` are passed in so the
    tests can use fakes; main() passes the real pipeline and the real request counter."""
    records: List[Dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        log(f"[{index}/{len(questions)}] {question['id']}: {question['question']}")
        before = usage_snapshot()
        started = clock()
        try:
            state = run_query(question["question"])
        except Exception as exc:   # the pipeline reports its own failures in state["failure"], so this is a crash
            state = {"failure": {"kind": "crash", "message": str(exc)}}
        seconds = clock() - started
        after = usage_snapshot()
        usage = {key: after.get(key, 0) - before.get(key, 0) for key in after}

        record = measure_run(question, state if isinstance(state, dict) else {}, seconds, usage)
        records.append(record)
        outcome = f"stopped: {record['failure_kind']}" if record["status"] == "failed" else "completed"
        log(f"    {outcome} in {record['seconds']} s, {record['requests']} request(s)")

        if record["failure_kind"] in STOP_KINDS:
            log("    Stopping here: the same cause would stop every later question.")
            break
    return records


def write_results(records: List[Dict[str, Any]], meta: Dict[str, Any], out_dir: Any) -> List[Path]:
    """Writes <date>.json (everything measured) and latest.md (the table to read). Returns the two paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarise(records)
    stamp = str(meta.get("date", "run")).replace("-", "").replace(":", "").replace(" ", "_")
    json_path = out_dir / f"{stamp}.json"
    markdown_path = out_dir / "latest.md"
    json_path.write_text(json.dumps({"meta": meta, "summary": summary, "records": records}, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(records, summary, meta), encoding="utf-8")
    return [json_path, markdown_path]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ResearchLLM benchmark questions and write a results table.")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS), help="JSON file with the questions")
    parser.add_argument("--only", help="comma-separated question ids to run")
    parser.add_argument("--limit", type=int, help="run only the first N questions")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="where to write the results")
    parser.add_argument("--yes", action="store_true", help="do not ask before using the Gemini API")
    args = parser.parse_args(argv)

    questions = load_questions(args.questions, only=args.only.split(",") if args.only else None, limit=args.limit)
    if not questions:
        print("No questions selected.")
        return 1

    print(f"{len(questions)} question(s) selected:")
    for question in questions:
        print(f"  - {question['id']} ({question['expect']})")
    print("A report question uses about 6 to 11 Gemini requests, an out-of-scope question about 2. "
          "The free tier allows 20 requests per day per model.")
    if not args.yes and input("Run now? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Cancelled.")
        return 0

    # Heavy imports only now: --help and the unit tests need neither torch nor a key.
    from backend.core.utilities import LLM_MODEL, LLM_REQUESTS_PER_MINUTE, LLM_USAGE
    from backend.main import initialize_research_session, run_research_query

    def run_query(text: str) -> Dict[str, Any]:
        # A fresh session per question gives each one its own empty vector index. The CLI runner prints the whole
        # report and a graph, which would bury the benchmark's own output, so its stdout is discarded.
        with contextlib.redirect_stdout(io.StringIO()):
            return run_research_query(initialize_research_session(), text)

    records = run_benchmark(questions, run_query, LLM_USAGE.snapshot)
    meta = {
        "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "model": LLM_MODEL,
        "requests_per_minute": LLM_REQUESTS_PER_MINUTE,
    }
    for path in write_results(records, meta, args.out_dir):
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
