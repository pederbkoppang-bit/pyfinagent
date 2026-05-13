---
step: 25.D
slug: normalize-agent-weights-0-1
tier: simple
cycle_date: 2026-05-13
---

# Research Brief -- phase-25.D: Normalize per-agent contribution weights to 0-1 range

> Tier=simple. Main authored from direct inspection of the current
> `signal_attribution.py` weight assignments and the drawer rendering.

---

## Three-variant search queries

1. **Current-year frontier**: `unified weight scale UI dashboard 2026`
2. **Last-2-year window**: `normalize 0-1 confidence display 2025 multi-agent`
3. **Year-less canonical**: `consistent metric scale data viz`

## Key findings

| Source | Cycle | Key finding |
|--------|-------|-------------|
| Few "Information Dashboard Design" | priors | Mixed scales on one display violate at-a-glance comprehension; normalize first |
| Cleveland & McGill (1984) | priors | Position/length comparisons require shared scale |
| Current code inspection | this cycle | Trader weight = final_score (0-10), Quant weight = composite_score (0-10), SignalStack weight = conviction_score (0-10), RiskJudge weight = recommended_position_pct (0-1), Analyst = 1.0, Bull/Bear = 0.5 |

## Recency scan

No paradigm shift in normalized-weight UI display 2024-2026.

## Design

1. **Normalize in `extract_signals_from_analysis`**:
   - Trader: divide `final_score` by 10 (0-10 -> 0-1).
   - RiskJudge: already 0-1; no change.
   - Analyst: 1.0; no change (saturated; "full attention").
   - Bull/Bear: 0.5 default; allow up to 1.0; no change.
2. **Normalize in `extract_quant_signals`**:
   - Quant: divide `composite_score` by 10.
   - SignalStack: divide `conviction_score` by 10.
3. **Clamp** all final weights to [0, 1] to defend against out-of-range inputs.
4. **Update 3 unit tests** in `tests/services/test_signal_attribution.py`:
   - `test_trader_extracts_lite_full_report_reason`: `trader["weight"] == 0.7` (was 7.0).
   - `test_extract_quant_signals_full_candidate`: `quant["weight"] == 0.845` (was 8.45).
   - `test_signalstack_includes_all_overlays`: `stack["weight"] == 0.8` (was 8.0).
5. **Frontend drawer summary**: add a one-row "Total contribution weight: X.XX"
   line above the layers, computed as the sum of all signal weights.

## Files to modify

| File | Change |
|------|--------|
| `backend/services/signal_attribution.py` | Normalize 3 weight sites + clamp |
| `tests/services/test_signal_attribution.py` | Update 3 expected weights |
| `frontend/src/components/AgentRationaleDrawer.tsx` | Add total-weight summary at top |
| `tests/verify_phase_25_D.py` | NEW verifier |

## Research Gate Checklist

- [x] Internal inspection at `backend/services/signal_attribution.py:73-225`
- [x] Frontend drawer inspection at `frontend/src/components/AgentRationaleDrawer.tsx:121-178`

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
  "note": "tier=simple; normalization is mechanical division-by-10; existing test updates are aligned with the new scale."
}
```
