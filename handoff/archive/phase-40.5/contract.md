# phase-40.5 -- Cosmetic _LAUNCHD_JOBS description (OPEN-30) -- regression-lock

**Step id:** `phase-40.5`
**Date:** 2026-05-23
**Mode:** EXECUTION (test-only regression lock; bug pre-closed by commit 2301b977).
**Cycle:** Cycle 23 (after Cycle 22 phase-38.7).

---

## North-star delta

**Terms:** R (audit-trail integrity; documentation truthfulness).

**R:** Locks in the cleanup of a stale operator-facing string. Without the test, a future commit could re-introduce the stale text and operators would lose 1-2 hours of diagnostic time on a misleading status line (real incident in phase-23.5.19). Per Atlan 2026 "Context Drift Detection" + Forrester 2025 "agent drift": stale metadata is a documented top-3 silent killer of AI-accelerated systems. This step ships the cheapest possible regression guard.

**B:** N/A. **P:** N/A. **Caltech arxiv:2502.15800 discount:** N/A.

**How measured:** `pytest backend/tests/test_phase_40_5_launchd_descriptions.py` exit 0 (4 tests); masterplan verification command `test $(grep -rn 'FAIL'+'ING exit 127' backend/ scripts/ | wc -l) -eq 0` exits 0.

---

## Research-gate compliance

**Researcher SPAWNED** per `feedback_never_skip_researcher`. Simple-tier brief at `handoff/current/research_brief_phase_40_5.md`:
- gate_passed: true
- external_sources_read_in_full: 9 (5-source floor +80% buffer)
- 12 internal files inspected; 21 URLs collected
- 3-variant queries + recency scan performed
- Sources: launchd.plist(5), launchd.info tutorial, Lucas Pinheiro launchd Medium, Conventional Commits, TLDP bash exit codes, Snyk Python assert dangers, mobeets launchd debug, arxiv:2510.04952 (safe cross-market trade execution audit), Atlan 2026 context drift detection

Researcher delivered **verdict TRUE** on the "pre-closed" hypothesis with file:line + git SHA precision: commit `2301b977` (phase-23.6.2, 2026-05-11) updated `backend/api/cron_dashboard_api.py:120` from the stale exit-127 string to the current "exit 1 -- partial .env fix applied" + phase-23.5.19 reference. Existing verifier `tests/verify_phase_23_6_2.py:118-130` (Check 4) has been guarding the invariant since.

---

## Hypothesis

> phase-40.5 is **pre-closed at the source layer**. Required deliverable
> is a **canonical pytest-compatible regression test** so the
> "no stale exit-127 string" invariant runs in `pytest backend/`
> (instead of only the ad-hoc `tests/verify_phase_23_6_2.py` Check 4).
> No new source-code changes.

---

## Immutable success criteria (verbatim from masterplan 40.5.verification)

1. `zero_stale_FAILING_exit_127_strings_in_source` -- **PASS**

The verification command `test $(grep -rn 'FAILING exit 127' backend/ scripts/ | wc -l) -eq 0` is satisfied today (live grep returns empty stdout; exit 1 = "no matches found" per grep convention). The new pytest regression test enforces this invariant in the canonical test suite.

Plus /goal integration gates 1-10.

---

## Self-reference safety

The test file `backend/tests/test_phase_40_5_launchd_descriptions.py` deliberately AVOIDS spelling the stale string as a single literal. It uses string concatenation (`_STALE_PATTERN_WORD_1 = "FAIL" "ING"; _STALE_PATTERN = _STALE_PATTERN_WORD_1 + " exit 127"`) so the literal pattern never appears in the file's bytes -- the grep self-scan can't false-positive. This is the right defensive shape for a regression test that GUARDS a string-absence invariant.

---

## Plan steps

| # | Step | Status |
|---|---|---|
| 1 | Researcher (simple tier, 9 sources, verdict=TRUE pre-closed) | DONE |
| 2 | Verify zero matches today (grep returned empty) | DONE |
| 3 | Write contract | IN FLIGHT |
| 4 | Ship `backend/tests/test_phase_40_5_launchd_descriptions.py` (4 tests) | DONE |
| 5 | pytest verify (count >= 353; achieved 357) | DONE |
| 6 | live_check + Q/A + harness_log Cycle 23 + flip | IN FLIGHT |

---

## Files this step touches

- `backend/tests/test_phase_40_5_launchd_descriptions.py` (NEW, ~125 lines, 4 tests)

**NOT changed:** any source code (researcher confirmed bug already swept by commit 2301b977 dated 2026-05-11). ZERO frontend; ZERO scripts/; ZERO masterplan structural changes.

---

## References

- closure_roadmap.md §3 OPEN-30
- research_brief_phase_40_5.md (this cycle, 9 sources, verdict TRUE)
- git commit `2301b977` (phase-23.6.2, 2026-05-11) -- the cleanup SHA
- backend/api/cron_dashboard_api.py:108-121 -- the _LAUNCHD_JOBS dict (now clean)
- tests/verify_phase_23_6_2.py:118-130 -- the existing ad-hoc verifier
- /goal directive (researcher mandatory per feedback_never_skip_researcher)
