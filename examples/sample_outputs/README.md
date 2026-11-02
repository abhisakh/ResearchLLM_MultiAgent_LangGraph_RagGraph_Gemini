# Sample outputs

Saved outputs from running each stage of the pipeline in isolation, in pipeline order. They show what each agent receives and produces without needing API keys.

| File | Stage |
|---|---|
| `level_0_procedural_output.json` | Query cleaning and intent classification |
| `level_1_planning_output.json` | Planning agent: execution plan and active tools |
| `level_2_test_input.json` | Input state for the retrieval self-test (tool results) |
| `level_2_tool_output.json` | Tool agents: normalised search results |
| `level_3_retrieval_output.json` | Retrieval: full-text chunks |
| `level_4_rag_output.json` | RAG: reranked, filtered context |
| `level_5_synthesis_output.{json,md}` | Synthesis: cited report |
| `level_6_evaluation_output.{json,md}` | Evaluation: audit result |
| `level_7_supervisor_output.{json,md}` | Supervisor: full graph run |

These files are the input states for the unit tests (`tests/test_agent_stages.py`, `tests/test_tool_agents.py`, `tests/test_retrieval_and_rag.py`). Each test loads a fresh copy through the `sample_state` fixture and replaces Gemini and the network with canned responses, so the tests need no API keys. They are never modified by the tests.

The files themselves are snapshots of real runs. To refresh one, run the pipeline and copy the state you want to keep.
