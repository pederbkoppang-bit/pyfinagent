---
step: 25.A10
slug: alpaca-mcp-smoke-and-reconcile
status: in_progress
cycle_date: 2026-05-13
parent_research_brief: handoff/current/research_brief.md
---

# Contract -- phase-25.A10

## Step ID + masterplan reference

`25.A10` -- "Alpaca MCP tool-surface smoke test + deny-list reconcile"
(P2, harness_required, no dep).

## Research-gate summary

Tier=simple. Brief at `handoff/current/research_brief.md`,
`gate_passed=true`.

## Hypothesis

The Alpaca MCP deny list in `.claude/settings.json` was authored at
V1 adoption (`place_order`, `cancel_order`, `replace_order`) but the
pinned server is `alpaca-mcp-server==2.0.1` which split + renamed
those tools (`place_stock_order`, `place_crypto_order`,
`place_option_order`, `cancel_order_by_id`, `cancel_all_orders`,
`replace_order_by_id`). Today an agent could call the new tool
names and bypass the deny list. Building a smoke test + reconcile
script catches future drift.

## Success criteria (verbatim from masterplan.json)

> `scripts_mcp_servers_smoke_test_alpaca_mcp_py_exists_and_passes`
>
> `reconcile_alpaca_deny_list_py_passes_no_unauthorized_writes`

## Plan steps

1. **`scripts/mcp_servers/smoke_test_alpaca_mcp.py`** -- NEW:
   - Spawn `alpaca-mcp-server` via `uvx --from alpaca-mcp-server==2.0.1`.
   - If `ALPACA_API_KEY_ID` env is missing, print SKIP + exit 0
     (graceful degradation; the smoke test is intentionally non-blocking
     when credentials are not configured).
   - Otherwise: do MCP handshake + tools/list + assert at least 1 read
     tool (e.g., `get_account_info`) AND at least 1 write tool (e.g.,
     `place_stock_order`) is present in the live response.
2. **`scripts/mcp_servers/reconcile_alpaca_deny_list.py`** -- NEW:
   - Module-local `CANONICAL_WRITE_TOOLS = [...]` list of 11 V2 trading
     write-class tools.
   - Read `.claude/settings.json`, extract `permissions.deny[]`.
   - For each `mcp__alpaca__<tool>` in CANONICAL_WRITE_TOOLS, assert
     present in deny[]. If any missing, exit 1 + print diff.
3. **Update `.claude/settings.json`** -- replace legacy V1 names with V2
   canonical names + add `exercise_options_position`,
   `do_not_exercise_options_position`, `update_account_config`.
4. **Verifier** `tests/verify_phase_25_A10.py` with 4 claims:
   - Claim 1: smoke test file exists with the expected shape.
   - Claim 2: smoke test handles no-creds gracefully (regex match).
   - Claim 3: reconcile script exists and exits 0 on the current deny list.
   - Claim 4: deny list has all 11 canonical V2 write-tool names.

## Files

| File | Action |
|------|--------|
| `scripts/mcp_servers/smoke_test_alpaca_mcp.py` | NEW |
| `scripts/mcp_servers/reconcile_alpaca_deny_list.py` | NEW |
| `.claude/settings.json` | Update deny list |
| `tests/verify_phase_25_A10.py` | NEW |

## Verification command (immutable)

```
source .venv/bin/activate && python3 tests/verify_phase_25_A10.py
```

## Live-check

`CI smoke test enumerates Alpaca server tools and reconciles vs deny list`.
Will write `handoff/current/live_check_25.A10.md`.

## Risks + mitigations

- **Risk**: V2 tool names change in a future release (V3 etc).
  **Mitigation**: The reconcile script is the canonical drift detector;
  failing on missing names forces explicit review.
- **Risk**: Smoke test runs network/cred operations from CI.
  **Mitigation**: Smoke test gracefully skips when credentials are
  missing; only the reconcile (offline static check) runs unconditionally.

## References

- `handoff/current/research_brief.md`
- `handoff/current/alpaca-mcp-research-brief.md` (V2 tool inventory)
- `scripts/mcp_servers/smoke_test_bigquery_mcp.py` (template)
- `.claude/settings.json` (deny list)
- `.claude/masterplan.json::25.A10`
