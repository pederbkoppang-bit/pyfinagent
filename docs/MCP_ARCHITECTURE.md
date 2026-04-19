# MCP Server Architecture

**Phase lineage:** 3.0 (servers) -> 3.5.0-3.5.7 (external-server adoption + health cron) -> 3.7.0 (MCP-vs-A2A ADR) -> 3.7.2 (signals server promotion) -> 3.7.3 (risk server) -> 3.7.6 (supply-chain + output caps) -> 3.7.7 (capability tokens + PII scrub) -> 3.0 (this document, consolidation).

Single source of truth for how pyfinagent uses Model Context Protocol (MCP).
Companion security doc: `docs/MCP_SECURITY.md`.

## Server inventory

### Internal (stdio, trusted)

| Server | Purpose | Exposed surface | Phase |
|--------|---------|-----------------|-------|
| `backend/agents/mcp_servers/data_server.py` | Read-only market data (yfinance, FRED, BigQuery) | 7 resources + `ping` | 3.0 |
| `backend/agents/mcp_servers/backtest_server.py` | Backtest execution + run-history lookup | 4 tools + 2 resources | 3.0 |
| `backend/agents/mcp_servers/signals_server.py` | Signal generation, validation, publish | 4 tools + 3 resources | 3.0 / 3.7.2 |
| `backend/agents/mcp_servers/risk_server.py` | Risk gate: kill_switch + PBO check + 4 more | 6 tools | 3.7.3 |

All four are built on FastMCP 3.2.4 (pinned in `backend/requirements.txt` for supply-chain hardening per phase-3.7.6). Startup coordination in `backend/main.py:304-327`.

### External (registered in `.mcp.json`)

| Server | Transport | Version pin |
|--------|-----------|-------------|
| `slack` (@anthropic-ai/mcp-server-slack) | stdio (npx) | `@latest` (Anthropic-maintained) |
| `alpaca` (alpaca-mcp-server) | stdio (uvx) | **2.0.1** (pinned in phase-3.0 per phase-3.7.6 supply-chain policy) |

### Harness-injected

The BigQuery MCP server is **harness-injected**, not pinned in `.mcp.json`. See `CLAUDE.md::BigQuery Access (MCP)` for its tool surface and rules (read AND write against `sunny-might-477607-p8`). The fallback when MCP tools aren't present is the `bq` CLI or `google.cloud.bigquery` Python client.

## Transport choice

**stdio only** for all servers. Rationale:
- Local single-process deployment: no network exposure.
- MCP spec (2024-11-05 + 2025 updates) supports Streamable HTTP, but we don't need remote access in the May 2026 go-live scope.
- stdio avoids the OAuth 2.1 / SSE / Streamable-HTTP complexity budget.
- Capability negotiation at handshake still happens (FastMCP handles this).

If remote exposure is ever required, the migration path is: expose a server via Streamable HTTP behind a reverse proxy, add OAuth 2.1 auth per spec, enable rate limiting per server (see Rate-limiting gap in `MCP_SECURITY.md`).

## Agent -> server data flow

```
   Main / Slack-bot / Harness agent
         |
         |  (MCP client request with capability token)
         v
   FastMCP handshake (capabilities, tools, resources listed)
         |
         v
   mcp_capabilities.py  --- HMAC-SHA256 scope check
         |                   PII scrub on inbound args
         v
   mcp_servers/<name>_server.py  (the 4 internal servers)
         |
         v
   Tool / Resource impl (BQ, yfinance, FRED, in-proc state)
         |
         v
   Response (size-capped + debounced per phase-3.7.6)
         |
         v
   Agent receives typed response
```

Not all agent calls go through MCP. By design:
- **Internal data/risk/signals/backtest**: MCP servers above (trusted stdio).
- **External provider APIs** (Anthropic, Gemini, OpenAI): direct SDK calls via `backend/agents/llm_client.py`. These already have phase-6.7 observability (rate_limit + retry + log_llm_call).
- **OpenClaw Gateway routing**: `backend/agents/openclaw_client.py` is NOT MCP; it's a direct gateway client for agent orchestration traffic.
- **Hosted Slack MCP**: `backend/slack_bot/mcp_tools.py` integrates the Slack-hosted MCP (remote SSE, read-only channel/user/thread access).

This split is intentional: MCP for internal capability-gated infrastructure; direct SDK for external providers; Gateway for agent-to-agent traffic. Documented in phase-3.7.0 ADR.

## Tool-description principle

Anthropic's multi-agent research system reports a ~40% task-completion improvement when tool descriptions are unambiguous. The pyfinagent servers use comprehensive Python docstrings as tool descriptions (FastMCP forwards docstrings verbatim). A review checklist for any new tool or resource:

- Single-sentence purpose.
- Input parameters enumerated with types and units.
- Output schema described (not just typed).
- Clear disambiguation vs siblings (e.g., `risk_check` is NOT `validate_signal`).
- Failure modes + suggested caller responses.

This is the single most-high-impact quality lever on MCP agent behavior; new tools must have descriptions reviewed before merge.

## Health + supply-chain

`backend/services/mcp_health_cron.py` runs weekly (phase-3.5.7):
- Detects stale external-server repos (no commits in 90 days -> warning).
- License audit on registered servers (flags unlicensed or GPL).
- CVE scan via pinned version -> OSV.

Alerts route through phase-6.7 `raise_cron_alert` -> Slack. Severity P2 for stale/license, P1 for CVE matches.

## Related ADRs

- `handoff/archive/phase-3.7.0/` -- MCP (tool) vs A2A (agent) decision. TL;DR: MCP for tool surface, A2A for agent delegation.
- `handoff/archive/phase-3.7.3/` -- risk_server creation + scope.
- `handoff/archive/phase-3.7.6/` -- per-MCP output-size cap + debounce + supply-chain pin.
- `handoff/archive/phase-3.7.7/` -- capability tokens + PII filter (see `MCP_SECURITY.md`).
- `handoff/archive/phase-3.5.0/` -- MCP surface inventory (read-only).
- `handoff/archive/phase-3.5.1/` -- MCP registry crawl + adopt-now shortlist.

## Known gaps (non-blocking)

- **Rate limiting**: not implemented inside MCP servers. Acceptable for trusted-stdio deployment per research. If remote exposure is ever enabled, add aiolimiter (already in `backend/services/observability/rate_limit.py`) as a FastMCP middleware.
- **Session liveness**: `mcp_health_cron.py` checks supply-chain, not per-session liveness. Low severity for local deployment.
- **Tool-description lint**: no CI gate on docstring quality. Worth adding when tool count crosses ~20 (currently ~18 tools across 4 servers).
