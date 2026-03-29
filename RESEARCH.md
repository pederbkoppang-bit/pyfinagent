# Research Log — pyfinAgent

## Phase 3: LLM-Guided Research + MCP Integration

### 3.0: MCP Server Architecture (2026-03-29)

#### Deep Research: Model Context Protocol (MCP)

**Sources Researched:**
1. ✅ **Official Anthropic MCP Documentation** (2025-11-20 beta)
   - URL: https://platform.claude.com/docs/en/agents-and-tools/mcp-connector
   - Status: Current, production-ready
   - Key findings:
     - MCP connector supports tool calling via Messages API
     - Requires beta header: `anthropic-beta: mcp-client-2025-11-20`
     - Supports multiple servers, per-tool configuration, OAuth authentication
     - Server must be publicly exposed via HTTP (Streamable HTTP or SSE transports)
     - Only tool calls supported (not resources/prompts yet)
     - Compatible with claude-opus-4-6, claude-sonnet-4-20250514, claude-haiku-4-5

2. ✅ **MCP Specification (2025-11-25)**
   - URL: https://modelcontextprotocol.io/specification
   - Covers: tool definitions, authorization patterns, transport protocols
   - Key for pyfinAgent: SSE and Streamable HTTP are supported transports
   - Authentication: OAuth 2.0 Bearer tokens for access control

3. **MCP Python SDK** (pending fetch)
   - Will check: github.com/modelcontextprotocol/python-sdk
   - For: implementation patterns, tool definition schema

#### Key Design Decisions

**1. Server Count & Scope**
- **Decision**: Three separate FastAPI servers (pyfinagent-data, pyfinagent-backtest, pyfinagent-signals)
- **Rationale**: Separation of concerns; each serves distinct purpose
  - Data: read-only, high-volume queries → cacheable, low latency
  - Backtest: compute-intensive, experimental queries → slower, more control needed
  - Signals: real-time trading decisions → authentication, rate limiting critical
- **Implementation**: All three run on same port 8000 with /mcp/* routing

**2. Transport Protocol**
- **Decision**: Streamable HTTP (not SSE)
- **Rationale**: 
  - More compatible with existing FastAPI infrastructure
  - Better for streaming long-running operations (backtest runs)
  - Simpler to debug (curl works directly)
  - Native support in Anthropic SDK

**3. Authentication**
- **Decision**: Shared secret (API key) via `authorization_token` parameter
- **Rationale**:
  - Simpler than full OAuth (no callback URLs needed)
  - Sufficient for single-user (Peder) + Ford (trusted)
  - Easy to rotate if needed
  - Can upgrade to OAuth later if multi-user access required

**4. Tool Allowlisting**
- **Decision**: Allowlist pattern (default disabled, explicitly enable per tool)
- **Rationale**:
  - Security: Planner doesn't get access to dangerous tools (e.g., delete experiments)
  - Clarity: Explicit is better than implicit
  - Future-proof: Can easily add tools without affecting API contracts

#### Cost Analysis

**Estimated Cost per Optimization Cycle (20 cycles/month):**
- Planner LLM call: 1 call × ~500 input tokens × $3/MTok = $0.0015
  - Query experiments (1-2 tools) → ~500 output tokens max
- Evaluator LLM call: 1 call × ~1000 input tokens × $3/MTok = $0.0030
  - Run ablation tests (3-5 tool calls) → ~1000 output tokens max
- **Per-cycle cost**: ~$0.005 (well under $1/cycle budge)
- **Monthly cost**: 20 cycles × $0.005 = $0.10/month for Claude reasoning

**Why this is economical:**
- Claude reasoning is CHEAP compared to running 20x optimizer cycles (which cost $100-200/mo in compute)
- One smart suggestion worth $2-5 in compute savings pays for itself
- Previous phases got no research guidance — this adds intelligence at minimal cost

#### Implementation Roadmap

| Phase | Step | Timeline | Status |
|-------|------|----------|--------|
| 3.0 | Research gate (this doc) | Complete ✅ | Done |
| 3.0 | Build pyfinagent-data server | 2-3h | Next |
| 3.0 | Build pyfinagent-backtest server | 3-4h | Next |
| 3.1 | Planner system prompt + integration | 2-3h | Deferred |
| 3.2 | Evaluator system prompt + integration | 2-3h | Deferred |
| 3.3 | Regime detection (HMM-based) | 3-4h | Deferred |

#### Open Questions

1. **Streaming backtest results**: Should long-running backtests stream progress to Claude, or wait for completion?
   - Current plan: Wait for completion, return full result
   - Alternative: Stream progress via SSE, let Claude see realtime progress
   - Decision: Wait (simpler, less resource overhead)

2. **Tool result caching**: Should Claude cache expensive tool results (e.g., experiments list)?
   - Current plan: Let Claude handle it; Anthropic's prompt caching covers this
   - Alternative: Custom caching in MCP server
   - Decision: Let Anthropic handle it (built-in prompt caching covers us)

3. **Error handling**: What if MCP server is down when Claude tries to use it?
   - Current plan: Claude falls back to reasoning without tool; returns error
   - Alternative: Pre-warm connections before passing to Claude
   - Decision: Let Claude handle errors; it's designed for this

---

## Prior Phases

[Previous research entries from Phase 0-2 would go here; see HEARTBEAT.md for completed phases]
