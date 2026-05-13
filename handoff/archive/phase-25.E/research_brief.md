---
step: 25.E
slug: drawer-summary-full-toggle
tier: simple
cycle_date: 2026-05-13
---

# Research Brief -- phase-25.E: Drawer summary vs full toggle

> Tier=simple. Main authored from direct inspection of the rationale
> endpoint + drawer after 25.C unblocked layer-1 surface.

---

## Three-variant search queries

1. **Current-year frontier**: `progressive disclosure compact full toggle 2026 UI`
2. **Last-2-year window**: `query param compact response API 2025 mobile`
3. **Year-less canonical**: `summary vs detail drawer UI pattern`

## Key findings

| Source | Cycle | Key finding |
|--------|-------|-------------|
| Bach et al. (2022) screen-fit | priors | Default compact; expand-to-full on demand |
| Shneiderman "overview first" | priors | Show top-level (decision + gate); let operator drill |
| 25.C cycle 94 | this session | After layer-1 surfaced, drawer can have 20+ rows; need a default-compact mode |
| paper_trading.py:575 | this cycle | Rationale endpoint accepts only path param; no query knob |

## Recency scan

No paradigm shift in API-side compact/full toggle pattern 2024-2026.

## Design

1. **Backend** `GET /api/paper-trading/trades/{trade_id}/rationale?full=1`:
   - Accept `full: bool = False` query param via FastAPI Query.
   - When `full=False` (default), filter the returned `signals` + `tree` to
     a compact view: Analyst (first 1), Trader, RiskJudge. Other layers
     pruned to [].
   - When `full=True`, return everything (layer-1 + analyst + debate + quant +
     signal_stack + trader + risk).
   - Preserve the trade metadata fields regardless of mode.
2. **Frontend** `AgentRationaleDrawer.tsx`:
   - Add a `[full, setFull]` state, default `false`.
   - Add a toggle button at the top of the drawer body: "Compact view" /
     "Full view" with click handler that flips state.
   - Modify `getPaperTradeRationale(tradeId, full)` to pass the query param.
   - Reload on toggle.

## Files to modify

| File | Change |
|------|--------|
| `backend/api/paper_trading.py` | Add `full` query param + filter logic |
| `frontend/src/lib/api.ts` | `getPaperTradeRationale(tradeId, full=false)` adds `?full=1` when true |
| `frontend/src/components/AgentRationaleDrawer.tsx` | Toggle button + state + refetch on flip |
| `tests/verify_phase_25_E.py` | NEW |

## Research Gate Checklist

- [x] Internal: 25.C cycle 94 surfaced layer-1 (precondition for "full could be 20+ rows")
- [x] Endpoint inspection at `backend/api/paper_trading.py:575-626`
- [x] Drawer inspection at `frontend/src/components/AgentRationaleDrawer.tsx`

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 3,
  "snippet_only_sources": 3,
  "urls_collected": 6,
  "recency_scan_performed": true,
  "internal_files_inspected": 3,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true,
  "note": "tier=simple; toggle is mechanical; depends on 25.C which is done as of cycle 94."
}
```
