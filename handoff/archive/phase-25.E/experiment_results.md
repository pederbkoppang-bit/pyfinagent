---
step: phase-25.E
cycle: 95
cycle_date: 2026-05-13
result: PASS_PENDING_QA
---

# Experiment Results -- phase-25.E

## What was built/changed

Closed audit bucket 24.4 F-3 by adding a compact/full toggle to the
rationale drawer (precondition 25.C closed at cycle 94 surfaced layer-1
skills; this cycle gives operators control over view density):

1. **`backend/api/paper_trading.py::get_trade_rationale`**:
   - Added `full: bool = Query(True)` parameter (default True for
     backwards-compat with existing callers; the new frontend passes
     `full=false` by default).
   - When `full == False`, signals + tree are pruned to a compact subset:
     first Analyst + Trader + RiskJudge. Other layers reduced to [].
   - Response now includes `"full": <bool>` so the frontend can echo the
     resolved mode.
2. **`frontend/src/lib/api.ts::getPaperTradeRationale`**:
   - Extended signature: `getPaperTradeRationale(tradeId, full = true)`.
   - Appends `?full=1` (full) or `?full=0` (compact) to the URL.
3. **`frontend/src/components/AgentRationaleDrawer.tsx`**:
   - `useState<boolean>(false)` for `full` (default compact).
   - Toggle button at the top-right of the drawer body: "Show full view"
     when in compact, "Show compact view" when in full.
   - useEffect dependency includes `full`, so flipping the toggle refetches.

## Files changed

| File | Action |
|------|--------|
| `backend/api/paper_trading.py` | Add `full` param + prune branch |
| `frontend/src/lib/api.ts` | Pass `full` to URL |
| `frontend/src/components/AgentRationaleDrawer.tsx` | Toggle button + state + refetch on flip |
| `tests/verify_phase_25_E.py` | NEW (5 claims) |

## Verification command + output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_E.py

=== phase-25.E verification ===

[PASS] 1. api_paper_trading_trades_trade_id_rationale_supports_full_query_param
        -> args=['trade_id', 'full']
[PASS] 2. backend_filter_prunes_tree_when_full_false
        -> full_branch=True analyst+risk_filter=True trader_check=True
[PASS] 3. backend_returns_full_tree_when_full_true
        -> Filter only applied when full=False (else branch returns unmodified signals)
[PASS] 4. frontend_api_passes_full_query_param
        -> full_arg=True query_param=True
[PASS] 5. frontend_drawer_toggle_button_implemented
        -> useState=True toggle_button=True fetch_with_full=True

ALL 5 CLAIMS PASS
```

Backend AST: clean. Frontend tsc: clean.

## Success criteria -> evidence

1. `api_paper_trading_trades_trade_id_rationale_supports_full_query_param`
   -- Claim 1 + 2 + 3 PASS: route accepts `full: bool` param, filter logic
   present, full-mode returns unfiltered.
2. `frontend_drawer_toggle_button_implemented` -- Claim 4 + 5 PASS:
   api.ts passes `full` to URL with `?full=0/1`; drawer has `useState` +
   toggle button with the two labels + refetch on flip.

## Out-of-scope / deferred

- Persisting the toggle state across drawer opens (e.g., remember
  operator's preference): not in criterion; default-compact is a UX
  reset every time, which matches "overview first" doctrine.
- Mid-toggle skeleton state: the existing loading state handles refetch
  visually.

## References

- `handoff/current/research_brief.md`
- `backend/api/paper_trading.py:575-636`
- `frontend/src/lib/api.ts:389-392`
- `frontend/src/components/AgentRationaleDrawer.tsx:44-58, 121-133`
- `.claude/masterplan.json::25.E`
