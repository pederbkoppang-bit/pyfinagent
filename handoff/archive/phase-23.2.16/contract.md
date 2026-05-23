# phase-23.2.16 -- Phase 2 deferred items triage; 3-item shortlist (P2)

**Step id:** `23.2.16`
**Date:** 2026-05-23
**Mode:** EXECUTION (shortlist doc + 7 pytest tests).
**Cycle:** Cycle 40 (after Cycle 39 phase-23.2.15).

---

## North-star delta

**Terms:** R (planning-discipline audit-trail) + B (defensive next-sprint planning).

**R:** Locks the WSJF+RICE-scored 3-item shortlist (sector column / RLock audit / auto-MtM wrapper). Each item has a numeric leverage score; cross-references the 8 new tickets surfaced this session as a follow-up sprint candidate.

**B:** Next sprint's planning has measurable inputs instead of vibes. Saves ~1 day of next-cycle planning effort.

**P:** N/A. **Caltech arxiv:2502.15800 discount:** N/A (researcher applied: safety-biased ranking).

**How measured:** 7 pytest tests on the shortlist doc structure.

---

## Research-gate compliance

**Researcher SPAWNED FIRST.** `handoff/current/research_brief_phase_23_2_16.md`:
- gate_passed: true
- external_sources_read_in_full: 10 (5-floor doubled)
- 20 URLs collected; 8 internal files inspected
- Sources: Intercom RICE, SAFe WSJF, ProductPlan canonical x2, Anthropic Harness Design, Caltech arxiv:2502.15800, Centercode RICE vs WSJF, Monday.com 2026 technical debt, CTO Magazine framework, Apparity SR 11-7

---

## Immutable success criteria (verbatim from masterplan 23.2.16.verification)

> "Read Section H of phase-23.2.0 audit; rank 8 deferred items by leverage; produce a 3-item shortlist for next sprint plan"

**Verdict: PASS.**
- 8 deferred items table: PRESENT (test 2)
- 3-item shortlist: PRESENT (test 3); ranks 105.0 / 86.4 / 40.0 leverage
- Next-sprint plan: documented in `handoff/current/phase-23.2.16-shortlist.md`

Plus /goal integration gates 1-10.

---

## Files this step touches

- `handoff/current/phase-23.2.16-shortlist.md` (NEW, ~110 lines, the deliverable)
- `backend/tests/test_phase_23_2_16_shortlist_doc_presence.py` (NEW, ~95 lines, 7 tests)

---

## Shortlist (verbatim from doc)

| Rank | Item | Source cycle | Leverage | Effort |
|---|---|---|---|---|
| #1 | Add `sector` column to paper_positions | 23.1.14 | 105.0 | 2 PD |
| #2 | RLock + re-entrant lock audit follow-through | 23.1.22 | 86.4 | 4 PD |
| #3 | Auto-MtM wrapper + home Sharpe + server-side NAV | 23.1.17 | 40.0 | 3 PD |

Total sprint effort: 9 person-days.

---

## References

- closure_roadmap.md §1 P2 verification list (final 23.2.x step)
- research_brief_phase_23_2_16.md (10 sources, gate_passed=true)
- handoff/current/phase-23.2.0-internal-codebase-audit.md Section H
- /goal directive
