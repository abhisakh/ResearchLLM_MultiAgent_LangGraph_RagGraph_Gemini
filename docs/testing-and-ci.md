# Testing and CI

[← Back to the README](../README.md)

Install the test tools first (`pip install -r requirements-dev.txt`). Tests come in two tiers:

```bash
python -m pytest                        # fast unit tests: no API keys, no network (default)
python -m pytest -m integration -s      # live end-to-end test: needs GEMINI_API_KEY
```

## Unit tests

They cover:

- the supervisor's routing, refinement loop and failure handling,
- the classification of Gemini errors and the retry and model-fallback logic,
- each agent with a scripted LLM, including every failure kind,
- the production graph wired with stub agents,
- the tool, retrieval and vector-store code against canned data,
- the API, including the HTTP status for every kind of failure.

## Integration test

`tests/test_ai_rigor.py` runs a real query end to end and asserts:

1. a non-trivial report is produced,
2. references are returned,
3. the graph visited more than a minimal set of nodes,
4. every citation number in the report maps to an existing reference,
5. a RAG utilisation ratio is reported.

## GitHub Actions

Two workflows run these tiers:

- `llmops_ci.yml` runs on every push and pull request: unit tests, then a Docker build. The image is pushed to Docker Hub only from the default branch.
- `llm_audit.yml` runs the live integration test weekly and on demand, and uploads the reasoning log as an artifact. It needs `GEMINI_API_KEY` (and optionally `MP_API_KEY`) as repository secrets.
