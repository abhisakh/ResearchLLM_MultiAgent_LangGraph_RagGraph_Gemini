# Benchmark

A small, repeatable check of ResearchLLM: the same fixed questions (`questions.json`), run through the real pipeline and
measured the same way every time. It answers four questions: how long does an answer take, what does it cost, how well
are the claims tied to sources, and does the system refuse what it should refuse.

**Status: the harness is built and unit-tested, but no results are published yet.** Run it once with your key and
commit `results/latest.md`. Until then, do not quote any numbers from it.

## Run it

```bash
python -m benchmarks.run --limit 1                # the first question only
python -m benchmarks.run --only weather-hamburg   # the cheap out-of-scope check
python -m benchmarks.run --yes                    # every question, without asking first
```

It asks before spending requests. A report question uses about 6 to 11 Gemini requests (more with refinement cycles), an
out-of-scope question about 2. The free tier allows 20 requests per day per model, so run one or two questions a day, set
`LLM_MODEL_FALLBACKS`, or use a billed key. If the quota, the key or the model setting stops a run, the benchmark stops
too (the same cause would stop every later question) and keeps what it measured. Set `LLM_REQUESTS_PER_MINUTE` in `.env`
as for any live run.

It writes `results/<date>.json` (everything measured) and `results/latest.md` (the table to read).

## What the columns mean

| Column | Meaning | How reliable |
|---|---|---|
| Time, Requests, Tokens, Cycles | Wall-clock seconds, Gemini requests, tokens (prompt + output + thinking), refinement cycles | Exact |
| Sources cited / retrieved | Sources in the final reference list, against documents the tools returned | Exact |
| Invalid citations removed | Citations the model wrote for sources it was never shown; code removes them before the report is final | Exact. The final report never contains one |
| Cited claim sentences | Share of substantive sentences (8 or more words) that carry a citation | Heuristic: a sentence split by punctuation rules |
| Grounding | Share of a cited sentence's content words and numbers that appear in the cited source's retrieved text, averaged | Heuristic, see below |
| Result | `ok`, `ok (unreviewed)` (the evaluation agent could not run), `stopped: <reason>`, or `FAILED CHECK` | A question passes when it completed and cites at least one source; an out-of-scope question passes when it was refused without any search |

## Limits you should know

- **Grounding is a lexical heuristic, not a fact check.** It asks whether the specific words and numbers of a sentence
  occur in the text of the source it cites. A faithful paraphrase can score below 1.0, and a sentence that copies the right
  words but draws the wrong conclusion can score 1.0. Use it to find reports worth reading closely, not to certify them.
- Averages cover completed report questions only. A refused out-of-scope question (fast and cheap by design) and a run that
  stopped on an error would distort them, so they are listed in the table but left out of the averages.
- Five questions on one model are a smoke test, not a leaderboard. Results vary between runs because the sources the
  search tools return change, and the model is not deterministic.
- Time depends on `LLM_REQUESTS_PER_MINUTE`: pacing adds about a minute for every five requests, so compare runs only with
  the same setting (it is printed at the top of the results).

## Adding a question

Add an object to `questions.json` with a unique `id`, the `question`, and `expect`: `"report"` (a cited report is
expected) or `"reject"` (the question is out of scope and must be refused without any search).
