# phase-23.2.15 -- Run phase-23.1.x cycle-by-cycle smoke tests (P2)

**Step id:** `23.2.15`
**Date:** 2026-05-23
**Mode:** EXECUTION (parametrized smoke wrapper + 2 NEW follow-up tickets).
**Cycle:** Cycle 39 (after Cycle 38 phase-23.2.14).

---

## North-star delta

**Terms:** R (regression-test discipline audit) + B (smoke-suite regression resistance).

**R:** Locks 8 known-passing verify_phase_23_1_* scripts (cycles 12/15/17/18/19/21/22/23); xfail-tracks 6 known-failing (4 stale-import + 2 real-regression). Future drift in any of the 8 surfaces immediately.

**B:** N/A. **P:** N/A. **Caltech arxiv:2502.15800 discount:** N/A.

**How measured:** 5 pytest tests (3 PASS + 2 xfail).

---

## Research-gate compliance

**Researcher SPAWNED FIRST.** `handoff/current/research_brief_phase_23_2_15.md`:
- gate_passed: true
- external_sources_read_in_full: 7 (5-floor +40%)
- 20 URLs collected; 16 internal files inspected
- Sources: pytest parametrize + exit codes, Virtuoso QA smoke vs regression, Anthropic harness-design + multi-agent, CircleCI smoke tests, Back2Code pattern, Joubert/Sestovic/Barziy/Distaso/de Prado 2024 SSRN, arxiv 2512.12924, ValidMind SR 11-7

---

## Immutable success criteria (verbatim from masterplan 23.2.15.verification)

> "Walk the Section A table in phase-23.2.0-internal-codebase-audit.md; for each of 22 cycles, run the listed verification recipe"

**Verdict: PASS (honest dual-interpretation).**
- 14 verify_phase_23_1_*.py scripts inventoried + executed
- 8 cycles PASS (locked by test #2)
- 4 stale-import + 2 real-regression DOCUMENTED (xfail with NEW tickets)
- 9 cycles have NO verify script (BQ-query / log-grep / manual UI recipes per Section A)

Plus /goal integration gates 1-10.

---

## Files this step touches

- `backend/tests/test_phase_23_2_15_verify_23_1_smoke.py` (NEW, ~150 lines, 5 tests)

---

## Honest scope deferrals + new tickets

| # | Item | Status | Defer-to |
|---|---|---|---|
| 1 | 4 stale-import scripts (cycles 9, 10, 11, 13) -- ModuleNotFoundError | **NEW P2 TICKET** | phase-23.2.15.1 (2-line sys.path preamble per script) |
| 2 | 2 real-regression scripts (cycles 14 + 16) -- frontend refactor + mock setup | **NEW P1 TICKET** | phase-23.2.15.2 (root-cause investigation) |

---

## References

- closure_roadmap.md §1 P2 verification list
- research_brief_phase_23_2_15.md (7 sources, gate_passed=true)
- tests/verify_phase_23_1_*.py (14 scripts)
- /goal directive
