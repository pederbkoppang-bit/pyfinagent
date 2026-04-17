# ADR 0002 -- MAS Communications: MCP for tools, A2A for agents

Status: accepted
Date: 2026-04-17
Owners: Peder (architecture) / harness (enforcement)
Phase: phase-3.7 step 3.7.0

## Context

pyfinagent's multi-agent system (MAS) needs a clear protocol boundary
between:

1. Agent-to-tool communication -- one LLM reaches out to data sources,
   brokers, or dev-tools (yfinance, BigQuery, Alpaca, GitHub).
2. Agent-to-agent communication -- one LLM-backed agent hands off a
   task to another with lifecycle semantics (retry, expiry,
   cancellation, result streaming).

Two 2024-2026 protocols compete for this surface:

- MCP (Model Context Protocol, Anthropic, spec rev 2025-11-25)
  https://modelcontextprotocol.io/specification/2025-11-25
- A2A (Agent2Agent Protocol, Google, launched April 2025)
  https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/

Phase-3.5 adopted external MCP servers (Alpaca, SEC EDGAR, FMP, FRED);
phase-4.6 hardened three in-process MCP servers (data, backtest,
signals). Open question: does the 3-node Data / Strategy / Risk MAS
(phase-3.7) use MCP for EVERYTHING, or layer A2A on top of MCP?

## Evidence surveyed

Research gate in harness_log Cycle 38 (researcher agent, 4-URL core +
9-URL comparative). Key points:

1. MCP is vertical (tool-to-agent), not horizontal. Spec 2025-11-25
   defines prompts/resources/tools only. A `Tasks` primitive (SEP-1686)
   is roadmapped for 2026 but NOT shipped.
   https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/
2. A2A is horizontal by design. April 2025 launch with 150+ orgs,
   positioned as complementary to MCP, carries native task lifecycle
   semantics (retry / expiry / cancellation).
3. Industry consensus 2026: two-protocol stack. Google Cloud's
   published financial agent uses exactly this shape -- MCP at base,
   A2A on top.
   https://medium.com/google-cloud/building-a-financial-ai-agent-with-google-adk-a2a-and-mcp-084f5937f1e8
4. MAS survey arXiv 2504.21030 recommends centralized orchestration
   for strategic decisions + decentralized MCP servers for
   data/tool execution.
   https://arxiv.org/html/2504.21030v1
5. Pure-MCP MAS failure modes (published post-mortems): tool-call
   storms, context leakage, context-window overflow.
   https://dev.to/aws/why-ai-agents-fail-3-failure-modes-that-cost-you-tokens-and-time-1flb

## Options considered

### A. Pure MCP (rejected)

Model every sub-agent as a tool the orchestrator can call.
- Pro: single protocol surface.
- Con: MCP has no task-lifecycle semantics; retries/cancellation/
  streaming become custom shims. Tool-call storms are the failure mode.
- Con: no peer-to-peer delegation; every call round-trips the
  orchestrator.

### B. Pure A2A (rejected)

Use A2A for both agent hand-offs AND tool access.
- Pro: single protocol surface with lifecycle native.
- Con: A2A does not standardize tool schemas; we would re-invent MCP
  for yfinance / BigQuery / Alpaca. Loses the phase-3.5 investment.

### C. Two-protocol stack: MCP for tools, A2A for agents (accepted)

MCP exclusively for tool access. A2A for inter-agent coordination.
One orchestrator LLM (Ford planner) is the sole consumer of both
initially; sub-agent peer-to-peer delegation happens over A2A when
phase-10.7 recursive self-modification activates.

## Decision

Decision: accept Option C -- MCP for tool-to-agent, A2A for
agent-to-agent.

The orchestrator LLM is the sole caller of MCP tool servers. When
phase-3.7 lands the 3-node Data / Strategy / Risk sub-agent layer,
those agents communicate among themselves over A2A. Paper-trading
execution via Alpaca stays MCP-first (subprocess in .mcp.json from
phase-3.5 step 3.5.3).

## Orchestrator shape chosen

```
                +-----------------------+
                |  Orchestrator LLM     |
                |  (Ford planner,       |
                |   Claude Opus)        |
                +-----------+-----------+
                            |
                     A2A over stdio
                  (task envelopes with
                   retry/expiry/cancel)
                            |
           +----------------+----------------+
           |                |                |
+----------v--+    +--------v-------+   +----v-----+
|  Data Agent |    | Strategy Agent |   | Risk Agent|
+------+------+    +--------+-------+   +----+------+
       |                    |                |
       |  MCP (tool access, stdio subprocess)
       v                    v                v
   [yfinance, FRED,    [quant_optimizer,   [kill_switch,
    SEC EDGAR,          backtest engine,    portfolio_risk,
    patents BQ,         Alpaca paper]       factor exposure]
    Vertex embeddings]
```

## Enforcement (phase-3.7 follow-on steps)

- 3.7.1-3: promote the three in-process servers (data, backtest,
  signals) to first-class MCP endpoints. Ping tools already added
  in phase-4.6.2; factory functions exist.
- 3.7.4: implement A2A task-delegation layer. If Python A2A SDK is
  not production-ready, use an AutoGen-style message bus with
  retry/expiry/cancellation until it matures. Decision deferred to
  3.7.4.
- 3.7.6: per-MCP output-size cap + debounce on duplicate tool calls
  within a 60s sliding window -- addresses tool-call-storm failure
  mode.
- 3.7.7: capability tokens scoped per session + PII filter on MCP
  input -- addresses context-leakage failure mode.

## Justification cites

- MCP spec 2025-11-25: https://modelcontextprotocol.io/specification/2025-11-25
- MCP 2026 roadmap (Tasks primitive experimental): https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/
- A2A protocol (Google, April 2025): https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/
- Google Cloud financial agent MCP+A2A+ADK: https://medium.com/google-cloud/building-a-financial-ai-agent-with-google-adk-a2a-and-mcp-084f5937f1e8
- arXiv MAS survey 2504.21030: https://arxiv.org/html/2504.21030v1
- MCP failure modes post-mortem (AWS): https://dev.to/aws/why-ai-agents-fail-3-failure-modes-that-cost-you-tokens-and-time-1flb

## Review cadence

- Revisited at phase-3.7 completion (when the MAS layer actually
  ships, we'll know whether A2A SDK ergonomics held up).
- Revisited at phase-10.7 (Meta-Evolution Engine) when recursive
  prompt optimization + sub-agent delegation become hot-path.
