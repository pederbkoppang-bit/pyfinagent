---
step: phase-25.A10
cycle: 100
cycle_date: 2026-05-13
result: PASS_PENDING_QA
---

# Experiment Results -- phase-25.A10

## What was built/changed

Closed audit bucket 24.10 F-2 by:

1. **`scripts/mcp_servers/smoke_test_alpaca_mcp.py`** -- NEW (~135 LOC).
   Spawns `alpaca-mcp-server==2.0.1` via uvx, performs MCP handshake
   (initialize -> initialized -> tools/list), asserts canonical V2 read
   tools (get_account_info, get_clock, get_stock_snapshot) AND write
   tools (place_stock_order, cancel_all_orders, close_position) are
   present. Gracefully accepts BOTH env var forms
   (ALPACA_API_KEY/ALPACA_SECRET_KEY OR ALPACA_API_KEY_ID/
   ALPACA_API_SECRET_KEY) and translates to the server's expected names.
   SKIPs with exit=0 when no credentials.

   **Live smoke verified locally**: server initialized, 61 tools exposed,
   all 6 sampled canonical tools present.

2. **`scripts/mcp_servers/reconcile_alpaca_deny_list.py`** -- NEW (~75 LOC).
   Static set-compare: reads `.claude/settings.json::permissions.deny[]`
   and asserts all 11 canonical V2 write-class Alpaca tools are present.
   Currently exits 0 -- deny list is reconciled.

3. **`.claude/settings.json`** -- replaced the legacy V1 deny entries
   (`mcp__alpaca__place_order`, `cancel_order`, `replace_order`) with
   V2 canonical names + added 4 more (`exercise_options_position`,
   `do_not_exercise_options_position`, `update_account_config`,
   `place_crypto_order`, `place_option_order`,
   `cancel_order_by_id`, `cancel_all_orders`, `replace_order_by_id`).
   Net: 5 -> 11 Alpaca tools denied.

## Files changed

| File | Action |
|------|--------|
| `scripts/mcp_servers/smoke_test_alpaca_mcp.py` | NEW |
| `scripts/mcp_servers/reconcile_alpaca_deny_list.py` | NEW |
| `.claude/settings.json` | Deny list updated to V2 canonical names |
| `tests/verify_phase_25_A10.py` | NEW verifier (5 claims) |

## Verification command + output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_A10.py

=== phase-25.A10 verification ===

[PASS] 1. scripts_mcp_servers_smoke_test_alpaca_mcp_py_exists_and_passes
        -> exists=True initialize=True tools_list=True uvx=True
[PASS] 2. smoke_test_skips_on_missing_creds
        -> skip_print=True env_check=True
[PASS] 3. reconcile_script_exists
        -> exists=True canonical_const=True
[PASS] 4. reconcile_alpaca_deny_list_py_passes_no_unauthorized_writes
        -> exit=0 stdout_tail=OK deny list covers all 11 canonical write tools
[PASS] 5. deny_list_has_all_11_v2_write_tools
        -> all 11 present

ALL 5 CLAIMS PASS
```

Live smoke test (with credentials):
```
$ python3 scripts/mcp_servers/smoke_test_alpaca_mcp.py
OK initialize -- protocolVersion=2024-11-05
OK tools/list -- 61 tools exposed
OK read+write tool surface confirmed (sampled 6 canonical tools present)
```

## Success criteria -> evidence

1. `scripts_mcp_servers_smoke_test_alpaca_mcp_py_exists_and_passes` -- Claims
   1 + 2 PASS: file exists with correct shape (init + tools/list + uvx
   invocation) and gracefully SKIPs on missing creds. Live invocation
   succeeds when creds present.
2. `reconcile_alpaca_deny_list_py_passes_no_unauthorized_writes` -- Claims
   3 + 4 PASS: script exists, exits 0 on current deny list, claim 5
   confirms all 11 canonical V2 write tools are denied.

## Out-of-scope / deferred

- The `.mcp.json` env-var mapping (ALPACA_API_KEY_ID -> server expects
  ALPACA_API_KEY) is a SEPARATE bug -- the smoke test works around it
  by translating in subprocess env. Fixing the .mcp.json is its own
  cycle (likely 25.A10.1).
- Watchlist mutations are NOT in the deny list -- not "trading state"
  per the audit basis.

## References

- `handoff/current/research_brief.md`
- `handoff/current/alpaca-mcp-research-brief.md` (V2 tool inventory)
- `scripts/mcp_servers/smoke_test_bigquery_mcp.py` (template)
- `.claude/masterplan.json::25.A10`
