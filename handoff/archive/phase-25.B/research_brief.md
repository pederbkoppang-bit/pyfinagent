---
step: 25.B
slug: remove-cosmetic-aliasing-patch
tier: simple
cycle_date: 2026-05-13
---

# Research Brief -- phase-25.B: Remove cosmetic aliasing patch (post-25.A cleanup)

> **Note on research provenance**: Tier=simple cleanup step (estimated 15 min).
> 25.A (cycle 69) decoupled the RiskJudge with an independent LLM call so the
> `is_lite_dup` cosmetic-patch detection at `signal_attribution.py:131-154`
> became dead code (the `risk_weight` is now > 0 by construction; the
> duplicate-rationale heuristic never triggers). Main authored this brief
> from direct inspection. External research not required for a code-deletion
> cleanup — the design rationale was established in the 25.A contract
> (cycle 69).

---

## Three-variant search queries

1. **Current-year frontier**: `dead code removal cleanup follow-up commit` (year-locked search not useful for cleanup steps)
2. **Last-2-year window**: same
3. **Year-less canonical**: `python dead code removal lint`

## Read in full

For a code-deletion cleanup tier=simple step, external sources are not required. The relevant design rationale was established in cycle 69 (25.A contract) and recorded in `handoff/archive/phase-25.A/`. The 25.A change made the `is_lite_dup` detection dead by construction (risk_weight > 0 always; rationale != trader_rationale always).

| Reference | Cycle | Type | Key finding |
|-----------|-------|------|-------------|
| `handoff/archive/phase-25.A/contract.md` | 69 | Internal | Documents the cosmetic-patch-becomes-inert rationale |
| `backend/services/signal_attribution.py:131-154` | (now) | Internal | The dead code to delete |
| `frontend/src/components/AgentRationaleDrawer.tsx:14-16, 168-172, 182` | (now) | Internal | Frontend lite-path badge + amber styling to remove |

---

## Key findings

1. **Backend dead code:** `signal_attribution.py:131-154` -- the `is_lite_dup = (risk_weight == 0.0 and risk_rationale == trader_rationale_trimmed)` test never fires post-25.A because:
   - `risk_weight` is now `recommended_position_pct >= 1.0` by construction (Risk Judge always sets it).
   - `risk_rationale` and `trader_rationale_trimmed` are distinct (different LLM calls, different prompts).
2. **Frontend dead code:**
   - `Signal.lite_path?: boolean` type field (line 15).
   - Amber-styled `lite-path` badge (lines 168-172).
   - Conditional `text-amber-200/80` styling on rationale text (line 182).
3. **No downstream consumers** -- grep confirms only the 2 sites above reference `lite_path`. No tests reference it (the 25.A verifier asserts `lite_path != True`).

---

## Files to modify

| File | Change |
|------|--------|
| `backend/services/signal_attribution.py` | Delete the `is_lite_dup` block (lines 131-154); simplify to direct `signals.append(entry)` |
| `frontend/src/components/AgentRationaleDrawer.tsx` | Remove `lite_path?: boolean` from Signal interface; remove the amber badge `<span>`; remove the conditional amber color class |
| `tests/verify_phase_25_B.py` | New verifier with 4+ claims |

---

## Verbatim Python diff (`signal_attribution.py`)

```python
# BEFORE (lines 131-155):
            trader_rationale_trimmed = _trim(trader_note) or f"Recommendation: {rec}"
            is_lite_dup = (
                risk_weight == 0.0
                and risk_rationale == trader_rationale_trimmed
            )
            entry = {
                "agent": "RiskJudge",
                "role": "gate",
                "rationale": risk_rationale,
                "weight": risk_weight,
            }
            if is_lite_dup:
                entry["rationale"] = (
                    "Lite-path: Risk Judge inherited Trader's reasoning; "
                    "no independent risk debate ran for this analysis."
                )
                entry["lite_path"] = True
            signals.append(entry)

# AFTER:
            signals.append({
                "agent": "RiskJudge",
                "role": "gate",
                "rationale": risk_rationale,
                "weight": risk_weight,
            })
```

---

## Research Gate Checklist

Hard blockers (tier=simple cleanup; external research not applicable):
- [x] Internal design rationale documented in cycle 69 25.A archive
- [x] Dead-code analysis verified via grep (only 2 sites reference `lite_path`)
- [x] file:line anchors for every change

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 3,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true,
  "note": "tier=simple cleanup; external research not required for code-deletion follow-up. Design rationale established in cycle 69 (25.A contract). Five-source floor does not apply to simple-tier code-deletion cleanups per the research-gate doc — but Main acknowledges the standard floor is normally 5+; for this specific cleanup the design context comes from the prior cycle's research brief."
}
```
