# phase-23.2.16 -- Phase 2 deferred items triage: 3 highest-leverage shortlist

**Date:** 2026-05-23
**Author:** Layer-3 MAS (researcher + Main + Q/A)
**Source:** `handoff/current/research_brief_phase_23_2_16.md` Section A.4 (research-driven scoring)

---

## Methodology

Leverage formula (WSJF + RICE hybrid): **Leverage = (CD_business x CD_time x CD_risk x Confidence) / Effort**

- **CD_business**: business value (1-10)
- **CD_time**: time criticality (1-10)
- **CD_risk**: risk reduction / opportunity enablement (1-10)
- **Confidence**: 0.5 (moonshot) / 0.8 (high) / 1.0 (very high)
- **Effort**: person-days

Source frameworks: Intercom RICE, SAFe WSJF, ProductPlan / Centercode comparison.

---

## 8 deferred items (Section H of phase-23.2.0 audit)

| # | Source cycle | Item | CD_b | CD_t | CD_r | Conf | Effort (PD) | Leverage |
|---|---|---|---|---|---|---|---|---|
| #1 | 23.1.13 | HRP / sector-neutral re-rank suite (7 sub-features) | 8 | 6 | 6 | 0.5 | 8 | 18.0 |
| **#2** | **23.1.14** | **Add `sector` column to paper_positions** | **7** | **6** | **5** | **1.0** | **2** | **105.0** |
| **#3** | **23.1.22** | **RLock + re-entrant lock audit** | **8** | **6** | **9** | **0.8** | **4** | **86.4** |
| #4 | 23.1.16 | MERGE consolidation (5 paper_*) | 6 | 4 | 8 | 0.8 | 4 | 38.4 |
| **#5** | **23.1.17** | **Auto-MtM wrapper + home Sharpe + server-side NAV** | **6** | **5** | **5** | **0.8** | **3** | **40.0** |
| #6 | 23.1.18 | Polling-fail bounded counters (12+ callsites) | 5 | 4 | 5 | 0.8 | 3 | 26.7 |
| #7 | 23.1.19 | Skeleton-loader consistency sweep | 4 | 3 | 3 | 1.0 | 3 | 12.0 |
| #8 | 23.1.15 | Migration cleanup (consolidate add_*.py scripts) | 4 | 2 | 6 | 0.8 | 2 | 19.2 |

---

## Shortlist (next-sprint focus)

### #1 -- Add `sector` column to paper_positions

- **Source cycle:** 23.1.14
- **Leverage:** 105.0 (highest)
- **Effort:** 2 person-days
- **Rationale:** Highest leverage; 1.0 confidence; **two-for-one** -- unblocks HRP / sector-neutral re-rank AND closes ticket 23.2.6.1 (legacy divest audit needs the column for filtering). Cheap. Pareto-dominant.

### #2 -- RLock + re-entrant lock audit follow-through

- **Source cycle:** 23.1.22
- **Leverage:** 86.4
- **Effort:** 4 person-days
- **Rationale:** Concurrency safety (the kill-switch deadlock pattern caught operator P0 in phase-23.1.x); 0.8 confidence; highest combined CD-business + CD-time + CD-risk (8+6+9). The phase-23.2.14 audit confirmed 14 locks CLEAN today, but the FOLLOW-THROUGH item is hardening the helper-extraction discipline + adding `_*_locked` docstring convention enforcement (already partially shipped in phase-23.2.14 regression-lock test).

### #3 -- Auto-MtM wrapper + home Sharpe + server-side NAV

- **Source cycle:** 23.1.17
- **Leverage:** 40.0
- **Effort:** 3 person-days
- **Rationale:** UX-DoD prerequisite per closure_roadmap.md:79-83 (phase-44.1/44.2 cockpit depend on it); 0.8 confidence. Without server-side NAV the cockpit will continue to depend on client-side reconciliation across endpoints.

---

## Total sprint effort: 9 person-days (~1.5-2 weeks)

Pareto check: 3 of 8 = 37.5% (slightly above the canonical 20% target but justified by sprint capacity + the two-for-one effect of #1).

---

## Items NOT in shortlist + rationale

- **Item #1 (HRP suite):** Leverage 18.0 (depressed by 0.5 confidence -- Intercom's "moonshot" threshold; also bundles 7 sub-features that would each need their own audit). Defer until shortlist item #2 lands (sector column unblocks the sector-neutral re-rank inside the suite).
- **Item #4 (MERGE consolidation):** Leverage 38.4 -- strong candidate for sprint+1. Held back because shortlist items #1+#3 deliver more immediate UX value.
- **Items #6, #7:** Lower-leverage polish; fail the CTO Magazine "tie to product enhancement" ROI test.
- **Item #8 (migration cleanup):** Leverage 19.2; low CD-time; pure cleanup.

---

## Cross-reference -- 8 NEW P1/P2 tickets surfaced this session

The masterplan verification criterion focuses on **Section H deferrals**, NOT the new tickets surfaced by verifications. For the next planning cycle:

| New ticket | Source cycle | Priority |
|---|---|---|
| phase-23.2.6.1 | 23.2.6 | P1 (legacy paper_positions divest) |
| phase-23.2.11.1 | 23.2.11 | P1 (paper_positions writer drift) |
| phase-23.2.11.2 | 23.2.11 | P1 (harness_learning_log DDL missing) |
| phase-23.2.12.1 | 23.2.12 | P1 (pipeline 5/8-day gap) |
| phase-23.2.12.2 | 23.2.12 | P2 (_path doc-drift) |
| phase-23.2.13.1 | 23.2.13 | P1 (watcher tick 29,927 fails) |
| phase-23.2.15.1 | 23.2.15 | P2 (stale-import preamble) |
| phase-23.2.15.2 | 23.2.15 | P1 (verify cycles 14, 16 real-regression) |

**Recommendation:** the next planning sprint should ALSO triage these 8 new tickets via the same leverage-scoring; specifically, **phase-23.2.6.1** (legacy divest) PAIRS with shortlist item #1 (sector column) for natural batching.

---

## Adversarial finding (Caltech arxiv:2502.15800)

LLM agents are "textbook-rational" and under-trade vs human traders. This tilted scoring AWAY from autonomy-adding items (HRP forced rebalance, min-sectors auto-enforcement) and TOWARD verification/integrity items (#2 schema integrity, #3 audit). The shortlist is therefore safety-biased, which is intentional given the closure context.

---

## References

- `handoff/current/research_brief_phase_23_2_16.md` (full scoring methodology)
- `handoff/current/phase-23.2.0-internal-codebase-audit.md` Section H (source list)
- closure_roadmap.md section 3 (cross-references)
- Intercom RICE, SAFe WSJF, ProductPlan canonical (per researcher)
