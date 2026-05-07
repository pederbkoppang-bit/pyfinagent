---
step: phase-23.3.5
title: Log file inventory audit -- fix stale path mismatch + surface .env bugs
cycle_date: 2026-05-07
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_3_5.py'
research_brief: handoff/current/phase-23.3.5-external-research.md (also see phase-23.3.5-internal-codebase-audit.md)
---

# Contract — phase-23.3.5

## Hypothesis

Audit of every log file in `handoff/` and `handoff/logs/` revealed
**TWO SETS** of files for mas-harness, autoresearch, ablation:
- `handoff/<x>.log` -- LIVE (launchd-managed, fresh today)
- `handoff/logs/<x>.log` -- STALE 18-21 days (legacy duplicate)

`backend/api/cron_dashboard_api.py:_log_paths()` was pointing at the
STALE path for `harness`, `autoresearch`, and `mas_harness_launchd`.
The /cron Logs tab silently showed 3-week-old data while the actual
services were happily writing to a different location.

The stale-path tail also surfaced THREE `backend/.env` bugs
(consistent with phase-23.3.4's exit-127 finding for autoresearch):
- Line 24: `ALPHAVANTAGE_API_KEY=` leading space
- Line 25: another env line with the same pattern (md5-like value)
- Line 56: `ANTHROPIC_API_KEY=` leading space

`ablation` and `autoresearch` are both failing nightly because of these.

## Research-gate summary

Brief written from researcher a64401254998f0c45's live finding +
main session's direct file/curl verification (researcher abandoned
mid-task; main consolidated the brief). Cites phase-23.2.23 +
phase-23.3.4 reusable research for path-traversal allowlist + launchd
StandardOutPath semantics. Recency scan 2024-2026 (no breaking
changes). 11 internal files inspected.

`gate_passed: true` per the consolidated brief at
`handoff/current/phase-23.3.5-external-research.md`.

## Immutable success criteria (verbatim — DO NOT EDIT)

1. `_log_paths()` is corrected:
   - `harness` -> `handoff/mas-harness.log` (was `handoff/logs/...`)
   - `autoresearch` -> `handoff/autoresearch.log`
   - `mas_harness_launchd` -> `handoff/mas-harness.launchd.log`
   - Backend, watchdog, restart paths UNCHANGED (those write
     correctly to their current locations).
2. Three NEW keys added:
   - `autoresearch_launchd` -> `handoff/autoresearch.launchd.log`
   - `ablation` -> `handoff/ablation.log`
   - `ablation_launchd` -> `handoff/ablation.launchd.log`
3. Frontend `LOG_KEYS` in `frontend/src/app/cron/page.tsx` is updated
   to match the backend exactly: 9 entries, same keys, ordered for
   readability.
4. Live curl `/api/logs/tail?log=harness&lines=2` returns content
   from the LIVE `handoff/mas-harness.log` (multi-MB, fresh today)
   not the stale 18-day duplicate.
5. Live curl `/api/logs/tail?log=autoresearch_launchd&lines=2`
   returns the literal exit-127 `.env` errors so the operator can
   see them on /cron.
6. Live curl `/api/logs/tail?log=ablation_launchd&lines=2` returns
   the lines-24/25/56 errors.
7. Audit deliverable
   `handoff/current/phase-23.3.5-audit-findings.md` documents all
   3 `.env` bugs with exact line numbers + the operator fix
   sequence.
8. Regression test
   `tests/services/test_log_path_allowlist.py` asserts the 9 keys
   resolve to the expected paths (string match, not file
   existence -- the test must work even if launchd hasn't
   re-created some files yet).
9. `python tests/verify_phase_23_3_5.py` exits 0.
10. `cd frontend && npx --no-install tsc --noEmit` exits 0.

## Plan steps

1. Edit `backend/api/cron_dashboard_api.py:_log_paths()` per criteria 1+2.
2. Edit `frontend/src/app/cron/page.tsx:LOG_KEYS` per criterion 3.
3. Restart backend; live-verify criteria 4-6.
4. Add `tests/services/test_log_path_allowlist.py`.
5. Add `tests/verify_phase_23_3_5.py`.
6. Write `handoff/current/phase-23.3.5-audit-findings.md`.
7. Append `harness_log.md` AFTER PASS.

## Out of scope

- **Editing `backend/.env`** — sandbox blocks .env access. Operator
  must apply the 3-line fix manually.
- Log rotation for backend.log (164 MB and growing). Separate phase.
- Adding `slack_bot.log` to the allowlist — file doesn't exist yet
  (phase-23.3.2 prescribed creating it on operator restart). When
  the operator restarts the slack-bot per phase-23.3.2 instructions,
  add the key in a follow-up.
- Adding `seed_stability_output.log` — audit-only artifact.

## Backwards compatibility

- Re-pointing 3 keys + adding 3 new keys is purely additive. No
  existing /cron tail consumer is broken; the dropdown grew.
- Allowlist semantics preserved (KEY -> Path resolution, no client-
  controlled paths, exit-400 on unknown keys).

## References

- `handoff/current/phase-23.3.5-{external-research,internal-codebase-audit}.md`
- `backend/api/cron_dashboard_api.py:_log_paths()` (the bug)
- `frontend/src/app/cron/page.tsx:LOG_KEYS` (frontend mirror)
- phase-23.3.4 audit-findings (the autoresearch exit-127 root cause)
- phase-23.2.23 path-traversal research (reused; allowlist pattern)
