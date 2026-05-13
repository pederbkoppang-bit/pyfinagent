# Sprint Contract -- phase-25.B -- Remove cosmetic aliasing patch (post-25.A cleanup)

**Cycle:** phase-25 cycle 29
**Date:** 2026-05-13
**Step ID:** 25.B
**Priority:** P2 (cleanup; depends on 25.A done at cycle 69)
**Audit basis:** bucket 24.4 F-2 -- `signal_attribution.py:131-154` `is_lite_dup` detection becomes dead code after 25.A

## Research-gate

Brief at `handoff/current/research_brief.md`. Tier=simple cleanup; external research not required (design rationale established in cycle 69 25.A contract; dead-code analysis verified via grep -- only 2 sites reference `lite_path`).

## Hypothesis

Deleting the dead `is_lite_dup` block in `signal_attribution.py:131-154` and the corresponding lite-path badge + amber styling in `AgentRationaleDrawer.tsx` -- closes phase-24.4 F-2 without any behavior change (the code path was unreachable post-25.A).

## Success criteria (verbatim from masterplan)

1. `is_lite_dup_branch_removed_from_signal_attribution`
2. `lite_path_amber_badge_removed_from_frontend`

Verification command (immutable):
`source .venv/bin/activate && python3 tests/verify_phase_25_B.py`

Live check (per masterplan):
`Code review: no is_lite_dup references in main branch post-25.B`

## Plan

1. **`backend/services/signal_attribution.py`**:
   - Delete the `trader_rationale_trimmed = _trim(...)` line + the `is_lite_dup = (...)` block + the `if is_lite_dup: entry[...] = ...; entry["lite_path"] = True` branch (lines 131-154).
   - Replace with a direct `signals.append({...})` of the RiskJudge entry.
2. **`frontend/src/components/AgentRationaleDrawer.tsx`**:
   - Remove `lite_path?: boolean` from the `Signal` interface (lines 12-15).
   - Remove the `<span>...lite-path</span>` badge block (lines 168-172).
   - Remove the conditional `text-amber-200/80` color class; use solid `text-slate-200`.
3. **Verifier** -- `tests/verify_phase_25_B.py` -- 5+ claims:
   - Claim 1: `is_lite_dup` literal NOT present in `signal_attribution.py`.
   - Claim 2: `"lite_path"` literal NOT present in `signal_attribution.py`.
   - Claim 3: `lite_path` field NOT present in `Signal` interface in `AgentRationaleDrawer.tsx`.
   - Claim 4: `lite-path` badge string NOT present in `AgentRationaleDrawer.tsx`.
   - Claim 5: `text-amber-200/80` conditional NOT present in `AgentRationaleDrawer.tsx` rationale text.
   - Claim 6: **Behavioral no-regression** -- import `extract_signals_from_analysis` with a sample analysis dict containing a complete `risk_assessment` (post-25.A shape); assert the returned RiskJudge entry is present with the expected `agent / role / rationale / weight` keys and NO `lite_path` key.

## Non-goals

- No semantic change to RiskJudge signal emission (the entry shape is unchanged minus the dead `lite_path` field).
- No frontend UI redesign -- just the badge/styling cleanup.
- No new tests beyond the verifier.

## References

- `handoff/archive/phase-25.A/contract.md` -- design context that made this patch inert
- `backend/services/signal_attribution.py:131-154`
- `frontend/src/components/AgentRationaleDrawer.tsx:14-16, 168-172, 182`
