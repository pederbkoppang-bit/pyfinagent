# Live-check placeholder -- phase-25.A10

**Step:** 25.A10 -- Alpaca MCP tool-surface smoke test + deny-list reconcile
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "CI smoke test enumerates Alpaca server tools and reconciles vs deny list"

## Pre-deployment evidence
- 5/5 verifier PASS.
- **Live smoke against alpaca-mcp-server==2.0.1** with paper credentials:
  ```
  OK initialize -- protocolVersion=2024-11-05
  OK tools/list -- 61 tools exposed
  OK read+write tool surface confirmed (sampled 6 canonical tools present)
  ```
- Reconcile script: `OK deny list covers all 11 canonical write tools`.

## Post-deployment operator workflow
1. Pull main:
   ```
   git pull origin main
   ```
2. Run the reconcile script as a static gate:
   ```
   python scripts/mcp_servers/reconcile_alpaca_deny_list.py
   ```
   Expected: `OK deny list covers all 11 canonical write tools`.
3. Run the live smoke test (requires Alpaca paper credentials):
   ```
   python scripts/mcp_servers/smoke_test_alpaca_mcp.py
   ```
   With creds: expect `61 tools exposed` + `read+write tool surface confirmed`.
   Without creds: expect `SKIP -- no Alpaca credentials ...`.

## Future-drift protection
If a future alpaca-mcp-server release adds a new write tool (e.g.,
`liquidate_position`), the reconcile script will continue to pass
(unaware of the new tool). Operators should update
`CANONICAL_WRITE_TOOLS` in the script when upgrading the package.

## Closes audit basis
bucket 24.10 F-2 RESOLVED. Deny list is now in sync with the pinned
alpaca-mcp-server==2.0.1 surface (11 write tools instead of legacy 5).

**Audit anchor for next bucket:** 25.A10.1 (.mcp.json env var name fix),
follow-ups (25.C9.1, 25.D9.1, 25.S.1, 25.B10.1).
