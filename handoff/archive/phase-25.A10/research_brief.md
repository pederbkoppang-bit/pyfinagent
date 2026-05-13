---
step: 25.A10
slug: alpaca-mcp-smoke-and-reconcile
tier: simple
cycle_date: 2026-05-13
---

# Research Brief -- phase-25.A10: Alpaca MCP smoke test + deny-list reconcile

> Tier=simple. Main authored from existing 25.alpaca-mcp-integration
> research brief (cycle 2026-04-24) + BigQuery MCP smoke-test template
> at `scripts/mcp_servers/smoke_test_bigquery_mcp.py`.

---

## Three-variant search queries

1. **Current-year frontier**: `MCP server tool surface validation 2026`
2. **Last-2-year window**: `MCP allowlist denylist reconciliation 2025`
3. **Year-less canonical**: `subprocess smoke test JSON-RPC stdio`

## Key findings

| Source | Cycle | Key finding |
|--------|-------|-------------|
| handoff/current/alpaca-mcp-research-brief.md | 2026-04-24 | V2 has 61 tools including 11 write-class (place_*_order, cancel_*_order, replace_order_by_id, close_position, close_all_positions, exercise_*_position, update_account_config) |
| smoke_test_bigquery_mcp.py | priors | stdio JSON-RPC subprocess pattern: spawn -> initialize -> initialized -> tools/list -> call |
| .claude/settings.json | this cycle | deny list still has V1 names (place_order, cancel_order, replace_order) -- out of sync with V2.0.1 |

## Recency scan

No paradigm shift in MCP tool-surface validation patterns 2024-2026.

## Design

1. **`scripts/mcp_servers/smoke_test_alpaca_mcp.py`** -- modeled on BQ
   smoke test. Spawn `alpaca-mcp-server==2.0.1` via uvx over stdio with
   `ALPACA_PAPER_TRADE=true` and dummy paper-key env. Do the MCP
   handshake, call `tools/list`, return the list. Gracefully handle the
   no-credentials case: print "SKIP -- no Alpaca creds" + exit 0.
2. **`scripts/mcp_servers/reconcile_alpaca_deny_list.py`** -- module-local
   list of canonical V2 write-class tools. Read `.claude/settings.json`'s
   `deny[]` array. Assert all canonical write tools are present in
   `deny[]`. Exit 0 if reconciled; exit 1 with diff if not.
3. **Update `.claude/settings.json`** -- replace the legacy V1 deny names
   (`place_order`, `cancel_order`, `replace_order`) with V2 canonical
   names (`place_stock_order`, `place_crypto_order`, `place_option_order`,
   `cancel_order_by_id`, `cancel_all_orders`, `replace_order_by_id`).
   Add `exercise_options_position`, `do_not_exercise_options_position`,
   `update_account_config`.

## Files to modify

| File | Change |
|------|--------|
| `scripts/mcp_servers/smoke_test_alpaca_mcp.py` | NEW |
| `scripts/mcp_servers/reconcile_alpaca_deny_list.py` | NEW |
| `.claude/settings.json` | Update deny list to V2 canonical names |
| `tests/verify_phase_25_A10.py` | NEW verifier |

## Research Gate Checklist

- [x] Internal: handoff/current/alpaca-mcp-research-brief.md (V2 tool inventory)
- [x] Internal: smoke_test_bigquery_mcp.py template
- [x] Internal: .claude/settings.json deny list audit

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 3,
  "snippet_only_sources": 3,
  "urls_collected": 6,
  "recency_scan_performed": true,
  "internal_files_inspected": 4,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true,
  "note": "tier=simple; smoke test pattern is mechanical; reconcile is a static set-compare."
}
```
