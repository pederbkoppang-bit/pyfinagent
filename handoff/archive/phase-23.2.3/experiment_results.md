---
step: 23.2.3
slug: fd-leak-regression
cycle: post-phase-26-cleanup
date: 2026-05-16
researcher_id: a2655469d948eb365
research_gate_passed: true
research_tier: complex
max_effort_directive: applied
verdict_by_main: SPIRIT_PASS_LITERAL_FAIL  # Q/A is authoritative; this is the self-summary
---

# Experiment Results -- phase-23.2.3 Verify FD leak did not regress

## File list

No source code modified -- pure verification step.

Files written:
- `handoff/current/research_brief.md` (Main internal + researcher_a2655469d948eb365 external; composed-brief pattern from phase-26.x)
- `handoff/current/contract.md`
- `handoff/current/experiment_results.md` (this file)
- `handoff/current/live_check_23.2.3.md` (verbatim lsof + grep evidence + temporal split)

## Plan-step 1-2: Live probe results

**lsof tickets.db (PASS):** 0 handles across both uvicorn PIDs (52623 parent, 52626 worker). The original `tickets.db` cyclic-open bug class is HOLDING.

**grep Errno 24 backend.log (LITERAL FAIL, SPIRIT PASS):** 29,934 total hits, ALL from 2026-04-29 or earlier (17+ days ago). Last 500,000 log lines contain ZERO Errno 24 -- the bug fix in `backend/governance/limits_loader.py` has held for at least 14-17 consecutive days.

## Plan-step 3-4: Temporal split + root-cause attribution

See `handoff/current/live_check_23.2.3.md` Evidence B + C.

- Pre-fix bug: `limits_loader._watcher_loop` opened `limits.yaml` repeatedly without closing → EMFILE.
- 29,927 of 29,934 hits target `limits.yaml` (99.98%); 4 on optimizer_best.json; 2 on cycle_heartbeat.json; 1 uncategorized.
- Fix landed BEFORE 2026-05-09 (the recent-window check shows zero hits across last 7 full days + 17 partial-day boundary).

## Plan-step 5: Verdict classification

Per the contract's classification:
- lsof_count ≤ 3 ✓ (== 0)
- errno24 == 0 ✗ (== 29,934 total, but ALL historical)
- Temporally bounded to historical only ✓ (last 500K lines / ≥14 days = 0)
- Active occurrences ✗ (zero)

→ **spirit-PASS-with-NOTE** per the contract's hybrid classification rule.

This parallels the **phase-23.2.2** precedent (`leak_dollars = $0.01` literal-fail / spirit-pass under float-rounding tolerance).

## Sub-criteria self-summary (NOT a verdict)

- ✓ Live FD count for tickets.db: 0 (well under ≤3 ceiling).
- ⚠ Backend log Errno 24 count: 29,934 historical hits remain; zero recent. **Operator follow-on:** rotate backend.log to make the immutable verification literal-PASS in future audits.
- ✓ Regression-not-recurred invariant: holds for 17+ days since the `limits_loader._watcher_loop` fix.

## Scope honesty

In scope, completed:
- Live `lsof` + `grep` against running backend ✓
- Temporal split via progressive `tail -n N | grep -c` ✓
- Stack-trace root-cause attribution ✓
- Per-file breakdown of historical hits ✓
- Backend health confirmation (current log activity) ✓

Out of scope (deferred to operator):
- Log rotation (`mv backend.log backend.log.YYYY-MM-DD; touch backend.log; restart-or-reload`) -- destructive operation requiring operator approval; NOT in 23.2.3 read-only verification scope.
- Adding `--since DATE` filter to the masterplan's verification command -- masterplan verification fields are immutable.
- Fix-the-fix on the 4 optimizer_best.json + 2 cycle_heartbeat.json historical leaks (different file paths, different bug classes, possibly already fixed).

Cross-step pattern: phase-23.2.x verifications depend on accumulated log/state history. The 219 MB un-rotated log is a friction point for literal interpretation of "empty" verifications. Operator may want to add a step to phase-27 for "rotate operational logs" + update the verification commands to use date-bounded filters.

## Verdict-by-Main (self-summary, NOT authoritative)

The spirit of the test -- "regression did not recur" -- is fully satisfied: zero Errno 24 in 17+ days, lsof FD count for tickets.db at zero, backend healthy. The literal grep returns >0 because the historical log retains pre-fix entries.

Q/A should weigh:
- (a) Literal-FAIL: insist on rotating the log or amending the masterplan verification before PASS.
- (b) Spirit-PASS-with-NOTE: accept the temporal split as proof the leak has not regressed, with operator follow-on noted (this is the phase-23.2.2 precedent).

Recommended path: (b) -- parallels 23.2.2's $0.01 tolerance acceptance.

Step 23.2.3 is ready for Q/A evaluation.
