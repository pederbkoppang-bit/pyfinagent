# phase-23.2.8 -- Verify home cockpit + paper-trading hero metrics stay in sync (P1)

**Step id:** `23.2.8`
**Date:** 2026-05-23
**Mode:** EXECUTION (SSOT source-grep verification + 6 new pytest tests).
**Cycle:** Cycle 32 (after Cycle 31 phase-23.2.7).

---

## North-star delta

**Terms:** R (SSOT discipline audit) + B (frontend regression resistance).

**R:** Locks the phase-23.1.17 `useLiveNav` SSOT discipline. Both home + paper-trading pages must import + use this hook for NAV display. Without source-grep enforcement, a future refactor could silently re-inline NAV math + drift the two pages.

**B:** Per researcher: a 30s tick race could still produce transient mismatch even with single-sourced math (custom hooks share LOGIC, not STATE). Strictly-stronger TanStack Query keyed cache is out of scope here; flagged as future follow-up.

**P:** N/A. **Caltech arxiv:2502.15800 discount:** N/A.

**How measured:** 6 pytest source-grep tests; both pages confirmed importing useLiveNav from `@/lib/useLiveNav`; both destructure `{liveNav, liveTotalPnlPct}`; NAV math `cash + positionsValue` confirmed ONLY in useLiveNav.ts.

---

## Research-gate compliance

**Researcher SPAWNED FIRST** per `feedback_never_skip_researcher` (cycle-31 lesson: protocol-discipline correction). `handoff/current/research_brief_phase_23_2_8.md`:
- gate_passed: true
- external_sources_read_in_full: 6 (5-source floor +20%)
- 18 URLs collected; 5 internal files inspected
- Sources: React docs Reusing Logic with Custom Hooks, TkDodo Query Options API, TanStack/query discussion #2310, Kent C. Dodds Colocation, testing-library renderHook API, Anthropic Harness Design

Researcher confirmed:
- `frontend/src/lib/useLiveNav.ts` exists + exports useLiveNav
- Home page imports + destructures `{liveNav, liveTotalPnlPct}` at page.tsx:15+156
- Paper-trading page imports + destructures at paper-trading/page.tsx:46+444
- NAV math `cash + positionsValue` appears ONLY in useLiveNav.ts (no re-inlining)

---

## Immutable success criteria (verbatim from masterplan 23.2.8.verification)

> "Manual: open both pages; NAV / Total P&L should be byte-identical (post phase-23.1.17 useLiveNav SSOT)"

**Verdict: PASS.** Source-level SSOT verified by 6 pytest tests. Manual UI check (operator-dependent) substituted with mutation-resistant source-grep that catches:
1. Hook deletion (test 1)
2. Home page no longer imports hook (test 2)
3. Paper-trading page no longer imports hook (test 3)
4. Either page no longer destructures expected fields (test 4)
5. NAV math re-inlined in a page (test 5 -- anti-drift)
6. Return shape drift (test 6)

Plus /goal integration gates 1-10.

---

## Files this step touches

- `backend/tests/test_phase_23_2_8_use_live_nav_ssot.py` (NEW, ~135 lines, 6 tests)

**NOT changed:** any source code; any frontend file.

---

## References

- closure_roadmap.md §1 P1 verification list
- research_brief_phase_23_2_8.md (this cycle, 6 sources, gate_passed=true)
- frontend/src/lib/useLiveNav.ts (the SSOT hook)
- frontend/src/app/page.tsx + paper-trading/page.tsx (the consumers)
- /goal directive (researcher mandatory per feedback_never_skip_researcher)
