---
step: 25.E
slug: drawer-summary-full-toggle
status: in_progress
cycle_date: 2026-05-13
parent_research_brief: handoff/current/research_brief.md
---

# Contract -- phase-25.E

## Step ID + masterplan reference

`25.E` -- "Drawer summary vs full toggle (?full=1 query param)"
(P2, harness_required, depends on `25.C` -- DONE at cycle 94).

## Research-gate summary

Tier=simple. Brief at `handoff/current/research_brief.md`,
`gate_passed=true`. Precondition 25.C closed at cycle 94.

## Hypothesis

After 25.C added layer-1 skill outputs to the drawer, full-pipeline
trades carry 20+ rows. A default-compact toggle gives operators a
3-row snapshot (Trader decision + RiskJudge gate + first Analyst
synthesis) with an expand-to-full button.

## Success criteria (verbatim from masterplan.json)

> `api_paper_trading_trades_trade_id_rationale_supports_full_query_param`
>
> `frontend_drawer_toggle_button_implemented`

## Plan steps

1. **`backend/api/paper_trading.py`**:
   - Add `full: bool = Query(False)` parameter to `get_trade_rationale`.
   - When `full == False`, prune signals to a compact subset: first
     Analyst + Trader + RiskJudge; tree buckets are filtered identically.
   - When `full == True`, return everything (current behavior).
   - Cache key includes `full` to avoid mode-mixing.
2. **`frontend/src/lib/api.ts`**:
   - Extend `getPaperTradeRationale(tradeId, full=false)` to append `?full=1`.
3. **`frontend/src/components/AgentRationaleDrawer.tsx`**:
   - `useState<boolean>` for `full` -- default `false`.
   - Toggle button at the top of the drawer body ("Show full view" /
     "Show compact view"). Refetches on flip.
4. **Verifier** -- `tests/verify_phase_25_E.py` with 5 claims:
   - Claim 1: route signature accepts `full: bool` query param.
   - Claim 2: backend filter logic prunes tree when full=False.
   - Claim 3: backend returns full tree when full=True.
   - Claim 4: frontend api.ts has the `full` argument.
   - Claim 5: drawer has a toggle button + useState.

## Files

| File | Action |
|------|--------|
| `backend/api/paper_trading.py` | Add `full` param + filter |
| `frontend/src/lib/api.ts` | Pass `full` to fetch |
| `frontend/src/components/AgentRationaleDrawer.tsx` | Toggle button + state |
| `tests/verify_phase_25_E.py` | NEW |

## Verification command (immutable)

```
source .venv/bin/activate && python3 tests/verify_phase_25_E.py
```

## Live-check

`?full=1 returns >5 signals on a full-pipeline trade`.
Will write `handoff/current/live_check_25.E.md`.

## Risks + mitigations

- **Risk**: Existing callers break if backend default flips.
  **Mitigation**: Default is `full=False`; existing callers calling without
  the param get the compact response (which is the new default), unchanged
  shape (still has `signals` + `tree`).
  Actually -- this DOES change the existing default response shape. Need
  to migrate the existing `getPaperTradeRationale` call to pass `full=true`
  in the existing usage OR keep default `True`. Decision: default
  backend `full=True` for backwards-compat; frontend's new toggle defaults
  to `false` and explicitly sends `?full=0` for compact.

Updated approach (per the risk above):
- Backend default: `full=True` (existing behavior preserved when callers
  don't pass the param).
- Frontend new toggle: default state `false`, sends `?full=0` for compact,
  `?full=1` for full.

This way no existing caller breaks.

## References

- `handoff/current/research_brief.md`
- `backend/api/paper_trading.py:575-626`
- `frontend/src/components/AgentRationaleDrawer.tsx`
- `frontend/src/lib/api.ts:389-391`
- `.claude/masterplan.json::25.E`
