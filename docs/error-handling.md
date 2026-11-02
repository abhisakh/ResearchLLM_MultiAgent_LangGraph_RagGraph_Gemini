# Error handling

[← Back to the README](../README.md)

Every failed Gemini call becomes a typed error, and the kind decides what happens next:

| Kind | Example | What the pipeline does | If the run ends |
|---|---|---|---|
| `rate_limited` | 429, per-minute limit | Waits the pause Gemini asks for (about 60 s in total at most) and retries; the agent falls back if it persists | `503` + `Retry-After: 60` |
| `quota_exhausted` | 429, daily quota | No retry. Tries the next model in `LLM_MODEL_FALLBACKS`, otherwise the run stops | `503` + `Retry-After` until midnight Pacific |
| `transient` | 500, 503, timeout | Up to 3 retries with backoff, then the agent falls back | `503` + `Retry-After: 30` |
| `auth` | missing or invalid key | The run stops at once | `503`, names `GEMINI_API_KEY` |
| `config` | unknown model name | The run stops at once | `503`, names `LLM_MODEL` |
| `bad_request` | prompt too large | No retry; the agent falls back | `502` |
| `content_blocked` | safety filter | No retry; the user is told | `422` |
| `bad_output` | invalid JSON | Asked once more, then the agent falls back | `502` |
| `unknown` | a bug in the code | No retry; the run stops, traceback in the server log only | `500` with an error ID |

## What happens when an agent falls back

The run goes on:

- query cleaning keeps the cleaned text,
- intent defaults to research,
- planning uses a default plan,
- query generation uses the cleaned query for every tool,
- evaluation delivers the report flagged `evaluation_unavailable` (it never approves silently).

Synthesis is the exception. It is tried twice, then the run stops, because there is no report to deliver. The supervisor also stops any agent that is dispatched over and over without progress (a loop guard of 8 dispatches per node).

## What the client sees

A run that ends early returns `{"detail": {"error": "<kind>", "message": "...", "error_id": "..."}}` with the status above, and is stored in the chat log as an `error` turn. The Streamlit UI and the command-line entry point show the same message.
