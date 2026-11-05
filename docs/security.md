# Security notes: untrusted content and prompt injection

ResearchLLM reads text it does not control: web pages, PDFs, and abstracts returned by public APIs. Text like that can contain instructions aimed at the model ("ignore the question and say X", "cite this link"). This is called indirect prompt injection. This page says where such text enters, what limits the damage, and what is **not** defended. It describes the code as it is, not as it should be.

**Short version:** the design limits what an injected instruction can *do* (no tools with side effects, routing and citations decided by code), but it does not stop an injected instruction from *changing what the report says*. Treat every report as a draft to be checked against its sources.

## Where untrusted text enters

| Text | Where it ends up |
|---|---|
| Titles, snippets and downloaded pages or PDFs from web search and from the scholarly APIs | RAG chunks, then the evidence block of the synthesis prompt |
| The report (model output built from that evidence) | The evaluator's prompt |
| The evaluator's `refinement_reason` (free text) | The planner's prompt, on a refinement cycle |

The user's own question is also placed in several prompts. It is the user's input, not retrieved content, so it is out of scope for this page except in a shared deployment (see the last gap).

## What limits the damage

These are checked in the code, not assumed:

- **The model has no tools.** No agent gives Gemini function calling or search grounding, so an injected instruction cannot trigger an action. The search tools are fixed read-only calls, and the planner can only pick names from an allow-list; unknown names are dropped.
- **Routing is code.** The supervisor is plain Python. The evaluator answers in a typed schema, and the stage to restart from is an enum (`refine_stage`), so free text cannot send the run to an arbitrary node. The old keyword guess on the free text remains only as a fallback.
- **Citations are verified by code, not by the model.** The model sees numbered sources and may cite only those numbers. After it answers, code removes citations of numbers that were never shown, renumbers the rest, throws away any "References" section the model wrote, and builds the list from the sources actually retrieved. An injected "cite https://example.com as [9]" cannot add a source. The number of removed citations is returned as `invalid_citations_removed`.
- **Bounded cost.** At most two refinement cycles, and a loop guard stops a node that is dispatched eight times, so a manipulated run cannot spin indefinitely.
- **No credentials in prompts.** None of the prompts include API keys; the client holds them.
- **No raw HTML.** The UI renders the report as Markdown without `unsafe_allow_html`, so HTML in a report is not executed.

## What is not defended

1. **The content of the report can still be steered.** A page can persuade the model to state something false, or to leave something out, and the citation will point at a real retrieved source. Code can check that a cited number belongs to a source the model was shown; it cannot check that the source supports the sentence. The evaluation agent is an LLM reading the same report, so it can be steered the same way.
2. **A second-order path exists.** The evaluator's free-text reason is interpolated into the planner's prompt. Injected text can reach it through the report and the evaluator's answer. The plan is still filtered through the tool allow-list, so the effect is limited to which allowed tools run.
3. **Links in the report body are mostly stripped; bare web addresses are not.** The reference list is built by code and is the only place with links. Before the report is returned, code removes Markdown images from the model's prose and turns Markdown links into their plain text (a numbered link such as `[1](https://...)` keeps its `[1]`), so an image can no longer make the reader's browser contact an address of the attacker's choosing. A bare address typed into a sentence is left as it is, and a chat client may still turn it into a clickable link. Reference links also point at real pages found by web search; they are not checked for safety.
4. **Server-side fetching is restricted.** The retrieval agent fetches only `http` and `https` URLs whose host resolves to public addresses only (not loopback, private, link-local, cloud-metadata, carrier-grade NAT or multicast), and it re-checks every redirect hop (at most 5). Each connection is also verified after it is made: the address the socket really reached must be public, checked before any request is sent (and before the TLS handshake for `https`), which closes DNS rebinding, where a name looks public to the first lookup and internal to the second. A download is refused above 50 MB and is stopped after 60 seconds in total, redirects and body included (`backend/core/url_safety.py`, `RetrievalAgent._fetch_content`). What is left: on a shared or cloud deployment, also block private ranges at the network level (an egress rule), because code-level checks are one layer, not the only one.
5. **No authentication or rate limiting,** CORS allows all origins, and the debug endpoint returns the full run state. Do not expose the API publicly as it is.
6. **The user's question is not escaped.** A user can try to override the prompts. The design is single-user, so the effect stays inside that user's own session; a multi-user deployment would need to treat questions as untrusted too.

## Hardening that would help, not yet done

- **Done:** fetch only `http` and `https`, refuse private, loopback, link-local and metadata addresses, re-check every redirect hop, verify the address each connection really reaches (DNS rebinding), and cap the download size (50 MB) and time (60 s). **Still to do:** an egress firewall rule on a shared deployment.
- **Done:** Markdown images and links are removed from the model's prose (`SynthesisAgent._strip_links_and_images`). **Still to do:** bare addresses in the prose.
- Label the evidence block as untrusted data and tell the model not to follow instructions found inside it. The current prompt says the evidence is the only allowed source but does not say that.
- Pass the planner a fixed template and the `refine_stage` enum instead of the evaluator's free text.
- Add an injection test set to the planned benchmark: evidence that contains instructions, with checks that no invalid citation survives and the answer stays on the question.
