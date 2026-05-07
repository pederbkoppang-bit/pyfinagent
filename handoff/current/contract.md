---
step: phase-23.3.4
title: Launchd watchdog audit -- expand /cron manifest + flag autoresearch exit 127
cycle_date: 2026-05-07
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_3_4.py'
research_brief: handoff/current/phase-23.3.4-external-research.md (also see phase-23.3.4-internal-codebase-audit.md)
---

# Contract — phase-23.3.4

## Hypothesis

Audit of the launchd surface revealed:

1. `com.pyfinagent.backend-watchdog` IS healthy (loaded, exit 0,
   StartInterval=60s, plist in repo at
   `scripts/launchd/com.pyfinagent.backend-watchdog.plist`).
   No-recent-log entries are expected (script only logs FAIL events;
   backend's been healthy ~2 days).
2. `launchctl list` reveals 5 OTHER pyfinagent launchd services NOT
   in /cron's `_LAUNCHD_JOBS` manifest (only backend-watchdog is
   there). They're invisible to the operator on /cron.
3. `com.pyfinagent.autoresearch` shows exit 127 in `launchctl list`
   AND `handoff/logs/autoresearch.launchd.log` contains repeated
   `backend/.env: line 24: TV5O5XN8IS2NLR6X: command not found`. Root
   cause: `backend/.env` line 24 has a leading space:
   `ALPHAVANTAGE_API_KEY= TV5O5XN8IS2NLR6X`. When `set -a; .
   backend/.env; set +a` parses it, bash tokenises the trailing
   value as a command. The autoresearch nightly job has been failing
   silently every night since 2026-04-24 (13 days).

## Research-gate summary

Researcher (af3ada936ca445dd8) returned `gate_passed: true`:
- 6 sources read in full (launchd.plist man page, Jamf Nation exit-127
  thread, launchd.info tutorial, ss64 launchctl, alansiu.net 2025
  modern launchctl print, victoronsoftware launchd best practices)
- 16 URLs collected; 10 in snippet-only
- Recency scan 2024-2026 -- no semantic changes
- 11 internal files inspected -- found the .env line + traced the
  exit code chain
- Concrete recommendation: add 5 missing manifest entries, document
  + fix-instructions for the autoresearch .env bug

## Immutable success criteria (verbatim — DO NOT EDIT)

1. `backend/api/cron_dashboard_api.py::_LAUNCHD_JOBS` is extended
   from 1 entry to 6 entries, adding:
   - `com.pyfinagent.backend` (KeepAlive, RunAtLoad)
   - `com.pyfinagent.frontend` (KeepAlive, RunAtLoad)
   - `com.pyfinagent.mas-harness` (StartInterval 1800s)
   - `com.pyfinagent.ablation` (StartCalendarInterval 03:00 daily)
   - `com.pyfinagent.autoresearch` (StartCalendarInterval 02:00
     daily, **CURRENTLY FAILING — exit 127**)
   `com.pyfinagent.claude-code-proxy` is intentionally OMITTED
   (Claude Code's own service, not pyfinagent's).
2. The audit deliverable
   `handoff/current/phase-23.3.4-audit-findings.md` documents the
   autoresearch root cause (verbatim line 24 of backend/.env), the
   exact one-character fix the operator needs to apply
   (`ALPHAVANTAGE_API_KEY=TV5O5XN8IS2NLR6X` -- remove leading space),
   and a launchctl bootout/bootstrap recovery sequence.
3. Live `curl /api/jobs/all` after backend restart shows 6
   `source: "launchd"` entries (was 1).
4. Regression test `tests/api/test_launchd_manifest_count.py`
   asserts `len(_LAUNCHD_JOBS) == 6` and the 5 new ids are present.
5. `python tests/verify_phase_23_3_4.py` exits 0.
6. `python -c "import ast; ast.parse(...)"` passes for the modified
   file.

## Plan steps

1. Edit `backend/api/cron_dashboard_api.py:_LAUNCHD_JOBS` to add the
   5 entries with description text mirroring researcher's findings.
2. Restart backend to pick up the new manifest.
3. `curl /api/jobs/all | jq` to confirm 6 launchd entries.
4. New `tests/api/test_launchd_manifest_count.py` (3 tests).
5. New `tests/verify_phase_23_3_4.py` (4 deterministic checks
   including the live HTTP probe).
6. Write `handoff/current/phase-23.3.4-audit-findings.md` with the
   .env fix instructions + recovery sequence prominent.
7. Append `harness_log.md` AFTER PASS.

## Out of scope (explicitly deferred)

- **Editing `backend/.env`** -- sandbox blocks .env reads/writes from
  this session. The operator must manually delete the leading space
  on line 24. Documented prominently in audit-findings.
- Stretch goal: live `launchctl print` parsing to surface
  `last_exit_code` per service on /cron. The static manifest plus
  the autoresearch finding are enough operator value for this phase.
- Adding the user-local plists to the repo (they live in
  `~/Library/LaunchAgents/`; that's intentional per project's
  local-only deployment doctrine).
- Fixing the secondary log-path-mismatch the researcher noted -- the
  primary mas-harness/autoresearch logs are already in
  `handoff/logs/`, which `_log_paths()` already maps. Re-verify in
  phase-23.3.5 (log inventory).
- Backfilling 13 days of missed autoresearch runs (operator decision
  whether to manually re-run the nightly memo).

## Backwards compatibility

- `_LAUNCHD_JOBS` extension is additive; existing consumers indexing
  by id still work. Frontend `/cron` Jobs tab will simply show 5
  more rows under the "launchd" group.
- No code path changes; pure manifest data.

## References

- Researcher: `handoff/current/phase-23.3.4-{external-research,internal-codebase-audit}.md`
- `backend/api/cron_dashboard_api.py:84-89` (current 1-entry _LAUNCHD_JOBS)
- `handoff/logs/autoresearch.launchd.log` (exit-127 evidence)
- `backend/.env:24` (the actual bug, can't be edited from this session)
- launchd.plist(5) man page; Jamf exit-127 community thread
