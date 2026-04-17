Decision: retire

# Phase-3 Step 3.5 "Enrichment MCP Server" -- Retirement Decision

Signed: 2026-04-17 (Cycle 55 / phase-3.5 step 3.5.5)

## Context

Phase-3 (LLM-Guided Research + MCP Integration) has two MCP-related
steps scoped before the broader phase-3.5 MCP Tool Audit & Adoption
phase was added:

- phase-3 step 3.0 "MCP Server Architecture"
- phase-3 step 3.5 "Enrichment MCP Server"

When phase-3.5 was added (Cycle 38) it absorbed the substantive work
these two phase-3 steps were meant to cover.

## Decision

Retire phase-3 step 3.5 (the "Enrichment MCP Server" stub).

Superseded by:
- phase-3.5 step 3.5.3 (Alpaca MCP adoption wave 1): external broker
  execution path via MCP subprocess.
- phase-3.5 step 3.5.4 (EDGAR + FMP + FRED wave 2): three external
  data MCPs + AGPL isolation policy.
- phase-4.6 step 4.6.2 (MCP ping + list_tools): hardened the three
  in-house MCP servers (data/backtest/signals) at
  backend/agents/mcp_servers/ by adding ping tools + fixing RFC-3986
  URI schemes. Those servers now constitute the enrichment layer the
  phase-3 stub was pointing at.

## Coverage map

| Original stub scope                  | Where covered now                             |
|--------------------------------------|-----------------------------------------------|
| FastMCP enrichment server            | backend/agents/mcp_servers/ (data/backtest/signals) |
| Register in .mcp.json                | .mcp.json lists slack + alpaca externally; in-house 3 stay in-process per phase-3.7 ADR target |
| Expose tools to external LLM callers | phase-3.7 step 3.7.0 (MAS comms ADR) picks stdio vs A2A boundary |
| Parity vs existing Python client     | phase-3.5 step 3.5.3 + 3.5.4 (mcp_ab_test.py) |

## Action

1. `.claude/masterplan.json` phase-3 step 3.5 flipped to
   status=superseded with superseded_by=phase-3.5.4.
2. Phase-3 step 3.0 "MCP Server Architecture" stays pending; the
   architectural decision is the scope of phase-3.7 step 3.7.0
   (MAS comms ADR) and phase-3.7 has not yet started.
3. No code changes needed -- the superseding work already shipped.

## Residual risk

None. phase-3.5 + phase-4.6.2 already landed the code the retired
stub would have produced.
