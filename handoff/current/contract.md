---
step: phase-23.3.6
title: /cron page UI verification + audit consolidation
cycle_date: 2026-05-07
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_3_6.py'
research_brief: handoff/current/phase-23.3.6-audit-findings.md
---

# Contract — phase-23.3.6

## Hypothesis

This is the final step of phase-23.3 cron audit. After 5 prior steps
(23.3.0-23.3.5) shipped fixes, the /cron page and its backend
endpoints should reflect every cron-job + log audit finding in the
operator UI.

Live state to verify:
- `/api/jobs/all` returns 19 jobs (2 main + 11 slack-bot + 6 launchd).
- `/api/logs/tail` allowlist has 9 keys, all resolving to live files
  (or quiescent files documented as such).
- Frontend `/cron` page renders without React console errors
  (phase-23.2.24 closed the Rules-of-Hooks bug; phase-23.3.5 synced
  the LOG_KEYS dropdown to 9 entries).
- `tsc --noEmit` clean.
- `npx eslint .` exits 0 (errors-only).

This step ships:
- A consolidated audit-findings doc summarising all 6 sub-phases
  (23.3.0-23.3.6) with their PASS/FIX verdicts + outstanding
  operator actions.
- A new verifier `tests/verify_phase_23_3_6.py` that exercises the
  live cron-jobs + log endpoints and confirms numeric expectations
  (19 jobs, 9 log keys, sources accounted for).

## Research-gate summary

This phase reuses the research from 23.3.0-23.3.5; no new external
research needed. The live state has been progressively verified in
each prior step. tier=simple, gate_passed via reuse.

## Immutable success criteria (verbatim — DO NOT EDIT)

1. `curl /api/jobs/all` returns `n_total=19` with breakdown:
   - 2 `main_apscheduler` (paper_trading_daily, ticket_queue_process_batch)
   - 11 `slack_bot` (4 core + 7 phase-9)
   - 6 `launchd` (backend-watchdog, backend, frontend, mas-harness,
     ablation, autoresearch)
2. `curl /api/logs/tail?log=<key>&lines=2` returns valid responses
   for all 9 allowlisted keys (backend, watchdog, restart, harness,
   mas_harness_launchd, autoresearch, autoresearch_launchd,
   ablation, ablation_launchd). Files that don't exist yet (e.g. an
   empty handoff/mas-harness.launchd.log) return `exists: true,
   n_returned: 0` cleanly; not 500.
3. Live curl `/api/logs/tail?log=etc/passwd` -> HTTP 400 (path
   traversal still blocked).
4. `cd frontend && npx --no-install tsc --noEmit` exits 0.
5. `cd frontend && npx --no-install eslint .` exits 0 (errors-only;
   pre-existing warnings tolerated).
6. `python tests/verify_phase_23_3_6.py` exits 0 and exercises every
   endpoint live.
7. `handoff/current/phase-23.3.6-audit-findings.md` consolidates
   the 6 sub-phase outcomes in a single operator-readable summary.

## Plan steps

1. Write `tests/verify_phase_23_3_6.py` — 6 deterministic checks
   including 4 live HTTP probes.
2. Write `handoff/current/phase-23.3.6-audit-findings.md` -- the
   consolidated audit summary with all PASS/FIX/OPERATOR-ACTION
   tags from 23.3.0 through 23.3.6.
3. Append `harness_log.md` AFTER PASS.
4. Mark phase-23.3 itself `status=done` in masterplan once all 7
   sub-steps are done.

## Out of scope

- Browser-runtime rendering smoke (Playwright). The `npx tsc` +
  `npx eslint .` static checks plus phase-23.2.24's Rules-of-Hooks
  fix cover the bug class that previously bit us. Operator can
  open `http://localhost:3000/cron` to confirm visually.
- The 4 outstanding OPERATOR-ACTION items (slack-bot daemon
  restart for phase-23.3.2/23.3.3 wiring; backend/.env line 24
  + 25 + 56 fixes for autoresearch and ablation). These are
  documented in the consolidation; sandbox blocks .env edits and
  daemon restarts that aren't backend-related.

## Backwards compatibility

- Pure additive deliverables (1 verifier + 1 audit-findings doc).
  No code changes.

## References

- All 6 prior phase-23.3.x audit-findings docs in
  `handoff/current/phase-23.3.{0..5}-audit-findings.md`.
- All 6 prior phase-23.3.x verifiers in
  `tests/verify_phase_23_3_{0..5}.py`.
