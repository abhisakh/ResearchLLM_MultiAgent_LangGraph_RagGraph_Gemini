# Benchmark results

The latest completed run of each question. The free tier allows only a few questions a day, so rows can come from different days and models; the last column says which.

| Question | Result | Time (s) | Requests | Tokens | Cycles | Sources cited / retrieved | Cited claim sentences | Grounding | Invalid citations removed | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| cspbi3-stability | ok | 246.5 | 14 | 41259 | 2 | 5 / 8 | 86% | 0.69 | 0 | 2026-11-06 models/gemini-3.6-flash |
| cathode-degradation | ok | 290.7 | 12 | 41715 | 2 | 7 / 15 | 88% | 0.69 | 0 | 2026-11-07 models/gemini-3.6-flash |
| weather-hamburg | ok | 15.9 | 2 | 1118 | 0 | 0 / 0 | - | - | 0 | 2026-11-05 models/gemini-3.6-flash |

Completed 3 of 3; passed their check: 3. Gemini requests used by these runs: 28.

Averages over the completed report questions:

- time: 268.6 s (median 268.6 s)
- Gemini requests: 13, tokens: 41487
- refinement cycles: 2
- sources cited: 6
- cited claim sentences (heuristic): 0.869
- grounding score (heuristic): 0.69
- invalid-citation rate before cleanup: 0.0

How to read these numbers, and their limits: see `benchmarks/README.md`.
