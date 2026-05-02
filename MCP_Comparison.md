# 🔌 MCP Protocol: From Standard to Practice
## How This System Implements — and Extends — the Model Context Protocol

---

## 📖 Table of Contents
- [What is MCP?](#what-is-mcp)
- [The Problem MCP Solves: The M×N Crisis](#the-problem-mcp-solves-the-mn-crisis)
- [How Standard MCP Works](#how-standard-mcp-works)
  - [The Three Core Roles](#the-three-core-roles)
  - [The Communication Flow](#the-communication-flow)
  - [Limitations of Standard MCP](#limitations-of-standard-mcp)
- [How This System Implements MCP Principles](#how-this-system-implements-mcp-principles)
  - [Role Mapping](#role-mapping)
  - [The Dynamic Tool Selection Pipeline](#the-dynamic-tool-selection-pipeline)
  - [The Self-Correcting Feedback Loop](#the-self-correcting-feedback-loop)
- [Side-by-Side Comparison](#side-by-side-comparison)
- [What This System Adds Beyond Standard MCP](#what-this-system-adds-beyond-standard-mcp)
- [The Single Architectural Gap](#the-single-architectural-gap)
- [Conclusion](#conclusion)

---

<a id="what-is-mcp"></a>
## 🧩 What is MCP?

**Model Context Protocol (MCP)** is an open standard introduced by Anthropic that defines a universal, structured way for AI models (LLMs) to communicate with external tools, data sources, and services.

Before MCP, every team building an AI assistant had to write custom integration code for every tool they wanted to connect — ArXiv had one format, PubMed had another, a database had yet another. This was fragmented, unscalable, and created enormous engineering overhead.

MCP solves this by defining:
- **A standard wire protocol** (JSON-RPC over HTTP/SSE or stdio)
- **A universal interface contract** that every tool must implement
- **Clear roles** for who sends requests and who serves responses

> **In one sentence:** MCP is the "USB standard" for AI tools — one plug fits all.

---

<a id="the-problem-mcp-solves-the-mn-crisis"></a>
## ⚡ The Problem MCP Solves: The M×N Crisis

Imagine you have **M different AI models** (GPT-4, Claude, Gemini…) and **N different tools** (ArXiv, PubMed, OpenAlex, Materials Project…). Without a standard protocol, every model needs a custom integration for every tool.

```
WITHOUT MCP — M×N Explosion:

  GPT-4   ──── custom code ──── ArXiv
  GPT-4   ──── custom code ──── PubMed
  GPT-4   ──── custom code ──── OpenAlex
  Claude  ──── custom code ──── ArXiv
  Claude  ──── custom code ──── PubMed
  Claude  ──── custom code ──── OpenAlex
  Gemini  ──── custom code ──── ArXiv
  ...

  = M models × N tools = potentially hundreds of custom integrations 😱
```

```
WITH MCP — 1×N Simplicity:

  GPT-4  ─┐
  Claude ──┼──► MCP Client (Single Interface) ──► ArXiv MCP Server
  Gemini ─┘                                   ──► PubMed MCP Server
                                              ──► OpenAlex MCP Server
                                              ──► Materials MCP Server

  = 1 universal interface × N tools = clean, scalable ✅
```

MCP collapses the M×N matrix into a single standard. Any model that speaks MCP can use any tool that speaks MCP — with zero additional integration code.

---

<a id="how-standard-mcp-works"></a>
## ⚙️ How Standard MCP Works

<a id="the-three-core-roles"></a>
### The Three Core Roles

Standard MCP defines three clearly separated roles:

```
┌──────────────────────────────────────────────────────────┐
│                        MCP HOST                          │
│   (The application: Claude Desktop, VS Code, etc.)       │
│                                                          │
│   ┌──────────────────────────────────────────────────┐   │
│   │                   MCP CLIENT                     │   │
│   │   (Lives inside the Host; manages connections)   │   │
│   │                                                  │   │
│   │   speaks JSON-RPC ◄──────────────────────────    │   │
│   └──────────────┬───────────────────────────────┘   │   │
│                  │                                    │   │
└──────────────────┼────────────────────────────────────┘   
                   │  (over HTTP/SSE or stdio)
       ┌───────────┼───────────┐
       ▼           ▼           ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐
  │   MCP   │ │   MCP   │ │   MCP   │
  │ Server  │ │ Server  │ │ Server  │
  │ (ArXiv) │ │(PubMed) │ │(OpenAl.)│
  └─────────┘ └─────────┘ └─────────┘
```

| Role | Responsibility |
|---|---|
| **MCP Host** | The application that contains the LLM. Manages the session and lifecycle |
| **MCP Client** | Lives inside the Host. Maintains a 1:1 connection with each MCP Server and speaks the protocol |
| **MCP Server** | An external service that exposes tools, resources, or prompts via the standard protocol |

---

<a id="the-communication-flow"></a>
### The Communication Flow

A standard MCP interaction follows this sequence:

```
Step 1 — Discovery
  Client  ──► Server:  {"method": "tools/list"}
  Server  ──► Client:  {"tools": [{"name": "search_arxiv", "description": "..."}]}

Step 2 — Invocation
  Client  ──► Server:  {"method": "tools/call",
                         "params": {"name": "search_arxiv",
                                    "arguments": {"query": "perovskite stability"}}}

Step 3 — Response
  Server  ──► Client:  {"content": [{"type": "text", "text": "...results..."}]}

Step 4 — LLM Synthesis
  Client passes result back to LLM → LLM generates final response
```

The entire communication travels over a **wire protocol** — JSON-RPC formatted messages over HTTP with Server-Sent Events (SSE) for streaming, or over stdio for local processes. This means MCP Servers can live on **remote machines, in different languages, maintained by different teams.**

---

<a id="limitations-of-standard-mcp"></a>
### Limitations of Standard MCP

While powerful, standard MCP in its vanilla form has notable gaps when applied to complex research workflows:

| Limitation | Description |
|---|---|
| **No pre-tool reasoning** | The LLM decides which tool to call in a single step, without a dedicated pipeline to clean, classify, and strategically plan the query first |
| **No query specialization** | The same user query is passed to tools as-is. There is no layer that crafts a tool-specific, tiered query optimized for each database's syntax |
| **No post-retrieval intelligence** | Standard MCP returns raw tool output directly to the LLM. There is no semantic reranking, neighbor expansion, or context compression stage |
| **No output validation** | There is no built-in mechanism to verify that the LLM's final answer is actually grounded in the retrieved tool data |
| **No self-correction loop** | If the output is poor, there is no autonomous feedback mechanism. A human must re-submit the query |
| **One-shot tool selection** | Tool selection is a single LLM decision. It cannot adapt mid-cycle based on what the retrieved data reveals |

---

<a id="how-this-system-implements-mcp-principles"></a>
## 🧬 How This System Implements MCP Principles

This framework was independently designed around the same core philosophy as MCP — a single orchestration layer connecting an LLM to multiple heterogeneous tools — and implements it via **LangGraph** rather than the JSON-RPC wire protocol.

<a id="role-mapping"></a>
### Role Mapping

| MCP Standard Role | This System's Equivalent | Implementation |
|---|---|---|
| **MCP Host** | FastAPI Backend (`backend.py`) | Manages the LangGraph session, state lifecycle, and persistence |
| **MCP Client** | Supervisor Agent (`supervisor_agent.py`) | Single orchestration entry point; routes all queries dynamically |
| **MCP Server** | Each Tool Agent (ArXiv, PubMed, etc.) | Standardized via `BaseToolAgent` — universal `execute(state)` contract |
| **Wire Protocol** | `ResearchState` dictionary | Shared in-memory state passed between all agents |
| **Tool Discovery** | PlanningAgent | Dynamically selects `active_tools` at runtime per query |
| **Tool Invocation** | Supervisor executes only active tools | Never runs all tools — only the ones strategically selected |

---

<a id="the-dynamic-tool-selection-pipeline"></a>
### The Dynamic Tool Selection Pipeline

Unlike standard MCP where the LLM makes a one-shot tool selection, this system has a **dedicated multi-stage intelligence pipeline** that runs *before* any tool is ever called:

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│                   PRE-TOOL INTELLIGENCE LAYER                │
│                                                              │
│  ┌─────────────────┐                                         │
│  │ CleanQueryAgent │  Removes filler, fixes typos,           │
│  │                 │  preserves domain-specific terms        │
│  └────────┬────────┘  (e.g. "Liquidography" stays intact)   │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                         │
│  │  IntentAgent    │  Classifies query into:                 │
│  │                 │  General Research / Material Property / │
│  │                 │  Irrelevant / Casual Chat               │
│  └────────┬────────┘                                         │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                         │
│  │ PlanningAgent   │  Strategically selects active_tools,    │
│  │                 │  creates execution plan, adapts if       │
│  │                 │  this is a refinement cycle             │
│  └────────┬────────┘                                         │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                         │
│  │  QueryGenAgent  │  Crafts tool-specific tiered queries    │
│  │                 │  (strict → moderate → broad) and        │
│  │                 │  embeds optimized HTTP parameters        │
│  │                 │  into ResearchState per tool            │
│  └────────┬────────┘                                         │
└───────────┼──────────────────────────────────────────────────┘
            │
            ▼
    Supervisor routes to
    ONLY the selected tools
```

This pre-tool pipeline is the system's answer to standard MCP's one-shot tool selection. By the time a tool is invoked, the query has been cleaned, the intent understood, the strategy defined, and a custom query crafted specifically for that tool's syntax and API requirements.

---

<a id="the-self-correcting-feedback-loop"></a>
### The Self-Correcting Feedback Loop

After tool execution, this system adds a **post-retrieval intelligence layer** that standard MCP entirely lacks:

```
                    ┌─────────────────┐
                    │ Tool Agents Run │  (Only active tools)
                    │ ArXiv, PubMed,  │
                    │ OpenAlex, etc.  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ RetrievalAgent  │  Downloads full PDFs,
                    │                 │  extracts text,
                    │                 │  semantic chunking
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   RAG Agent     │  FAISS vector search
                    │                 │  + Cross-Encoder reranking
                    │                 │  + Neighbor expansion
                    │                 │  + Keyword gating
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ SynthesisAgent  │  Generates grounded report
                    │                 │  with numbered citations —
                    │                 │  strictly from retrieved context
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ EvaluationAgent │  Audits:
                    │                 │  ✓ All planned tools used?
                    │                 │  ✓ Every citation verified?
                    │                 │  ✓ Sufficient depth?
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
     ✅ COMPLETE                   ❌ NEEDS REFINEMENT
     Final report                  Supervisor re-invoked
     delivered                     with refinement reason
                                        │
                                        ▼
                                 ┌─────────────┐
                                 │  Supervisor │  Injects additional tools
                                 │             │  (SemanticScholar, ChemRxiv)
                                 │             │  adjusts plan, reruns cycle
                                 └─────────────┘
```

---

<a id="side-by-side-comparison"></a>
## 📊 Side-by-Side Comparison

| Feature | Standard MCP | This System |
|---|---|---|
| **Core philosophy** | Single interface → multiple tools | ✅ Same — Supervisor → N Tool Agents |
| **Eliminates M×N** | ✅ Yes | ✅ Yes |
| **Tool discovery** | Dynamic via `tools/list` at runtime | ✅ Dynamic via PlanningAgent per query |
| **Tool invocation** | LLM one-shot decision | ✅ Multi-stage: Intent → Plan → QueryGen → Execute |
| **Query specialization** | ❌ Raw query passed to tool | ✅ Tiered, tool-specific queries crafted per tool |
| **Communication layer** | JSON-RPC over HTTP/SSE wire protocol | In-memory `ResearchState` dictionary |
| **Tool location** | Remote servers, any language | Same Python process, LangGraph nodes |
| **Plug-and-play tools** | ✅ Drop in new MCP Server, zero code change | Requires new `BaseToolAgent` subclass |
| **Post-retrieval processing** | ❌ Raw output to LLM | ✅ FAISS search + Cross-Encoder reranking + neighbor expansion |
| **Output validation** | ❌ None built-in | ✅ EvaluationAgent audits grounding + citations |
| **Self-correction loop** | ❌ None | ✅ EvaluationAgent → Supervisor → full re-cycle |
| **Hallucination prevention** | ❌ LLM's own judgment | ✅ Zero-hallucination policy enforced by Evaluation |
| **Citation traceability** | ❌ Not enforced | ✅ Every claim mapped to a verified source |
| **RAG utilization audit** | ❌ None | ✅ CI/CD quality gate measures utilization ratio |

---

<a id="what-this-system-adds-beyond-standard-mcp"></a>
## 🚀 What This System Adds Beyond Standard MCP

This framework does not merely replicate MCP — it systematically addresses every limitation of the standard:

### 1. Pre-Tool Intelligence Pipeline
Standard MCP trusts the LLM to select the right tool with no preparation. This system runs a **3-stage reasoning pipeline** (CleanQuery → Intent → Planning → QueryGen) before a single tool is invoked, ensuring the right tool gets the right query.

### 2. Tool-Specific Query Engineering
Standard MCP passes the user's query as-is. This system's **QueryGenerationAgent** crafts tiered queries (strict → moderate → broad) customized to each tool's API syntax — PubMed gets Medical Subject Headings style queries, ArXiv gets category-aware queries, OpenAlex gets title-search optimized queries.

### 3. Two-Stage Neural Retrieval
Raw tool output in standard MCP goes directly to the LLM. This system inserts a **RAG pipeline** with FAISS bi-encoder search followed by Cross-Encoder reranking — a deep neural model that reads the query and document *together* to score true semantic relevance, catching nuances that vector similarity alone misses.

### 4. Contextual Neighbor Expansion
When a relevant text chunk is found, the RAG Agent also retrieves the chunks immediately before and after it. This ensures the LLM receives **complete surrounding context** rather than isolated fragments, preserving the document's original narrative flow.

### 5. Grounding Enforcement
The **EvaluationAgent** performs a structured audit of every generated report — verifying that all planned tools were used, all numerical citations map to a real entry in the reference list, and the depth of analysis meets the query's requirements. This is enforced autonomously, not left to the LLM's self-assessment.

### 6. Autonomous Self-Correction
If the EvaluationAgent is unsatisfied, it does not simply flag an error. It instructs the **Supervisor** with a specific refinement reason (missing data, PDF parsing failure, insufficient context, or structural weakness), which then injects additional tools or adjusts the plan and reruns the full cycle autonomously.

---

<a id="the-single-architectural-gap"></a>
## 🔧 The Single Architectural Gap

The one area where this system differs from the MCP specification is the **communication layer**:

```
Standard MCP:
  Supervisor ──► JSON-RPC message over HTTP/SSE ──► ArXiv Server (remote)
                 {"method": "tools/call",
                  "params": {"name": "search_arxiv", "arguments": {...}}}

This System:
  Supervisor ──► Python method call in memory ──► ArxivAgent.execute(state)
                 ResearchState{} passed directly
```

This architectural choice is a deliberate trade-off — **in-process communication is faster, simpler to debug, and avoids network overhead** for a single-deployment research system.

To migrate to full MCP compliance, each `BaseToolAgent` subclass would be extracted into a standalone MCP Server process, exposing a `tools/call` endpoint. The Supervisor would then discover and invoke them over the wire. The `BaseToolAgent` abstraction already provides the right interface contract to make this migration straightforward.

---

<a id="conclusion"></a>
## ✅ Conclusion

This system was designed around the same foundational insight that motivated MCP: **a single orchestration layer should connect an intelligent agent to many heterogeneous tools, eliminating redundant integrations and creating a clean, scalable architecture.**

Where standard MCP defines the **protocol contract** for this idea, this framework builds a **production research system** on top of those principles — and extends them with a pre-tool reasoning pipeline, neural post-retrieval processing, grounding enforcement, and autonomous self-correction that the standard protocol does not provide.

> **Standard MCP answers:** *"How should an AI talk to tools?"*
>
> **This system answers:** *"How should an AI think before, during, and after talking to tools — and what should it do if the answer isn't good enough?"*

The result is a system that is **MCP in philosophy, LangGraph in implementation, and research-grade in rigor.**

---

<a id="how-this-system-is-different-from-mcp"></a>
## 🆚 How This System Is Fundamentally Different From MCP

This section exists to remove any remaining ambiguity. The previous sections established where this framework and MCP share the same philosophy. This section draws the precise boundary between them.

---

### The One Fundamental Difference

> **MCP is a communication protocol. My system is an intelligent research pipeline.**

They solve different layers of the same problem.

---

### The Analogy That Makes It Crystal Clear

Think of it like a **phone call vs. a conversation:**

- **MCP** defines *how the call is made* — the network, the signal format, the handshake. It doesn't care what is said.
- **My system** defines *what happens before, during, and after the conversation* — preparation, reasoning, verification, and self-correction.

MCP is the **infrastructure layer**. My system is the **intelligence layer.**

---

### Difference 1 — 🧠 Pre-Tool Intelligence — My System Has It, MCP Doesn't

In standard MCP, the LLM picks a tool and calls it. That's it. One step.

In my system, before a single tool is ever called, a dedicated pipeline runs first:

```
MCP:       User Query ──────────────────────────────────────► Tool Call

My System: User Query → CleanQuery → Intent → Planning
                      → QueryGen ──────────────────────────► Tool Call
```

By the time my system calls a tool, it already knows:
- Exactly what the user *really* meant after cleaning and intent classification
- Which tools are *strategically* worth calling for this specific query
- What *custom, tiered query* each tool should receive, written in its own API syntax

MCP has none of this. The tool gets whatever the user originally typed.

---

### Difference 2 — 🔬 Post-Retrieval Processing — My System Has It, MCP Doesn't

Standard MCP takes the tool's raw output and hands it directly to the LLM. Done.

My system inserts an entire neural processing pipeline between the tool output and the LLM:

```
MCP:       Tool Output ──────────────────────────────────────────────► LLM

My System: Tool Output → Retrieval → FAISS Search → Cross-Encoder
                       Reranking → Neighbor Expansion → Keyword
                       Gating → Compressed High-Signal Context ──────► LLM
```

The LLM in my system never sees raw, noisy tool output. It only receives **semantically filtered, reranked, context-expanded, deduplicated evidence** — extracted from the full text of peer-reviewed papers.

---

### Difference 3 — ✅ Output Validation & Self-Correction — My System Has It, MCP Doesn't

Standard MCP has no opinion about whether the final answer is any good. That is left entirely to the LLM's own judgment.

My system has a dedicated **EvaluationAgent** that autonomously audits every output before it is delivered:

```
MCP:       LLM Answer ──────────────────────────────────────► User
                       (no verification)

My System: LLM Answer ──► EvaluationAgent
                               │
                   ┌───────────┴───────────┐
                   │                       │
            ✅ Grounded?            ❌ Not grounded?
            Citations verified?     Supervisor notified
            Tools fully used?       Full cycle reruns
                   │                       │
            Deliver to User         Try again autonomously
```

MCP cannot self-correct. My system never delivers an answer it hasn't verified.

---

### Complete Difference Summary

| Dimension | Standard MCP | My System |
|---|---|---|
| **What it is** | A wire communication protocol | An intelligent research pipeline |
| **Tool selection** | LLM one-shot decision | 3-stage reasoning: Intent → Plan → QueryGen |
| **Query per tool** | Same raw user query passed as-is | Custom tiered query per tool's API syntax |
| **After tool runs** | Raw output handed directly to LLM | Neural reranking → context compression → LLM |
| **Output quality check** | None | EvaluationAgent audits grounding + citations |
| **If output is poor** | Human must manually retry | My system autonomously reruns the full cycle |
| **Hallucination control** | LLM's own judgment | Enforced — no ungrounded claim passes |
| **Communication** | JSON-RPC over HTTP/SSE wire protocol | In-memory ResearchState via LangGraph |
| **Adding a new tool** | Drop in a new server, zero code change | New `BaseToolAgent` subclass + graph wiring |

---

### The One Thing Standard MCP Has That My System Does Not

**Network-level plug-and-play.** In true MCP, a new tool server can be dropped anywhere on the network — a different machine, a different language, a different team — and the client discovers and uses it automatically with zero code change.

In my system, adding a new tool means writing a new `BaseToolAgent` subclass and wiring it into the LangGraph graph. The abstraction is clean and well-structured, but a code change is required. This is a deliberate trade-off in favour of in-process speed, simpler debugging, and tighter state control — all appropriate for a single-deployment scientific research system.

---

### Final Clarification

These two things are not alternatives to each other. They answer different questions entirely:

> **Standard MCP asks:** *"How should an AI communicate with a tool?"*
>
> **My system asks:** *"How should an AI think before calling a tool, process what the tool returns, verify the answer it generates, and correct itself if that answer is not good enough?"*

My system could adopt MCP as its underlying communication layer — replacing in-memory `ResearchState` passing with JSON-RPC wire calls — without changing a single agent's logic. The `BaseToolAgent` abstraction is already the right interface contract for that migration. The intelligence pipeline built on top of it would remain entirely intact.

---

*For the full system architecture, agent breakdown, and LLMOps pipeline, see the sections above in this README.*
