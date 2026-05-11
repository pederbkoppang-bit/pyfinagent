# Phase-23.3.4 audit findings — Launchd watchdog + service inventory

**Cycle date:** 2026-05-07
**Scope:** the launchd watchdog (`com.pyfinagent.backend-watchdog`) plus
all 6 pyfinagent launchd services discovered via `launchctl list`.

## Verdict: PASS WITH FIX + 1 OPERATOR-FIX-REQUIRED

The watchdog itself is healthy. The /cron manifest was incomplete (1
entry covered, 5 services invisible). The autoresearch nightly job
has been failing silently for **13 days** with a one-character bug in
`backend/.env` line 24.

## Per-service inventory (researcher's audit)

| Label | Plist location | Schedule | Last exit | Status |
|---|---|---|---|---|
| com.pyfinagent.backend-watchdog | repo + ~/LaunchAgents | StartInterval=60s | 0 | ✓ healthy |
| com.pyfinagent.backend | ~/LaunchAgents (user-local) | KeepAlive, RunAtLoad | -15 (SIGTERM normal) | ✓ running PID 18185 |
| com.pyfinagent.frontend | ~/LaunchAgents | KeepAlive, RunAtLoad | 0 | ✓ running PID 86232 |
| com.pyfinagent.mas-harness | ~/LaunchAgents | StartInterval=1800s | 0 | ✓ idle, mas-harness.log fresh |
| com.pyfinagent.ablation | ~/LaunchAgents | StartCalendarInterval 03:00 | 0 | ✓ ran today 03:00 |
| **com.pyfinagent.autoresearch** | ~/LaunchAgents | StartCalendarInterval 02:00 | **127** | **✗ FAILING since 2026-04-24** |
| com.pyfinagent.claude-code-proxy | ~/LaunchAgents | n/a | n/a | (Claude Code's own; excluded from manifest) |

## What was changed

```diff
 # backend/api/cron_dashboard_api.py:_LAUNCHD_JOBS
-_LAUNCHD_JOBS: tuple[...] = (
-    {"id": "com.pyfinagent.backend-watchdog", ...},
-)
+_LAUNCHD_JOBS: tuple[...] = (
+    {"id": "com.pyfinagent.backend-watchdog", ...},
+    {"id": "com.pyfinagent.backend",      "schedule": "launchd KeepAlive RunAtLoad", ...},
+    {"id": "com.pyfinagent.frontend",     "schedule": "launchd KeepAlive RunAtLoad", ...},
+    {"id": "com.pyfinagent.mas-harness",  "schedule": "launchd interval 1800s", ...},
+    {"id": "com.pyfinagent.ablation",     "schedule": "launchd cron 03:00 daily", ...},
+    {"id": "com.pyfinagent.autoresearch", "schedule": "launchd cron 02:00 daily",
+     "description": "Nightly autoresearch memo (FAILING exit 127 since 2026-04-24 -- see phase-23.3.4 audit)"},
+)
```

## OPERATOR FIX REQUIRED — autoresearch exit 127

**Root cause** (researcher af3ada936ca445dd8 traced via
`handoff/logs/autoresearch.launchd.log` which contains repeated
`backend/.env: line 24: [REDACTED-phase-23.3.7]: command not found`):

`backend/.env` line 24 has a leading space:

```
ALPHAVANTAGE_API_KEY= TV5O5XN8IS2NLR6X
                    ^-- this space is the bug
```

When `scripts/autoresearch/run_nightly.sh` runs
`set -euo pipefail; set -a; . backend/.env; set +a`, bash tokenises
`TV5O5XN8IS2NLR6X` as a standalone command after the empty
assignment, can't find it, exits 127. `set -e` aborts. The job has
been failing every night since 2026-04-24 (13 days).

**Cannot be fixed from this Claude Code session** — the harness
sandbox blocks reads/writes to `backend/.env`. Operator action
required.

### One-character fix (operator runs this)

```bash
# Edit backend/.env line 24, removing the leading space:
sed -i '' 's/^ALPHAVANTAGE_API_KEY= TV5O5XN8IS2NLR6X$/ALPHAVANTAGE_API_KEY=TV5O5XN8IS2NLR6X/' backend/.env

# Verify the line is now clean:
grep '^ALPHAVANTAGE_API_KEY' backend/.env

# Recovery sequence (force the next nightly to pick up the fix):
launchctl bootout gui/501/com.pyfinagent.autoresearch 2>/dev/null
launchctl bootstrap gui/501 ~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist
launchctl kickstart gui/501/com.pyfinagent.autoresearch
sleep 5
launchctl list | grep autoresearch   # should now show exit 0 (was 127)
tail -5 handoff/logs/autoresearch.launchd.log   # should be clean now
```

If you'd rather wait for the next 02:00 UTC tick, just edit the .env
line and the fix activates automatically.

## Backwards compatibility

- `_LAUNCHD_JOBS` extension is purely additive. Existing 1-entry
  consumers still work.
- No frontend code changes needed; /cron Jobs tab will simply show
  5 more rows under the "launchd" group.

## Sibling concerns deferred

- **Live `last_exit_code` on /cron** -- researcher's stretch goal.
  Could parse `launchctl list` output via subprocess to surface the
  real exit code per service. Useful for surfacing autoresearch's
  127 visually instead of via the description string. P2 follow-up.
- **The 4 user-local plists** (backend, frontend, mas-harness,
  ablation, autoresearch) are not in the repo per local-only
  deployment doctrine. If we ever need to recreate them on a fresh
  Mac, document them in a runbook (separate phase).
- **13 days of missed autoresearch runs** -- operator decides whether
  to manually re-run any of the missed nightly memos. The script
  `scripts/autoresearch/run_nightly.sh` can be invoked manually
  after the .env fix.

## Verification

- `python tests/verify_phase_23_3_4.py` -> 4/4 OK (incl. live HTTP
  probe confirming /api/jobs/all returns 6 launchd entries).
- `pytest tests/api/test_launchd_manifest_count.py -q` -> 4 passed.
- Live `curl /api/jobs/all | jq '.jobs | map(select(.source=="launchd"))'`
  -> 6 rows.

## Q/A

Per same-session pragmatism: deterministic verifier is canonical
gate. The autoresearch fix requires operator action that cannot be
performed from this session.
