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

These are static snapshots, not test fixtures for pytest. The self-test block at the bottom of each agent module (`python -m backend.agents.<module>`, run from the repository root) reads its input from this folder and writes its output back here, so the stages can be run one after another.
