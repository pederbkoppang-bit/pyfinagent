# phase-23.3.4 -- Internal Codebase Audit: launchd Service Inventory

**Date:** 2026-05-07
**Phase:** 23.3.4 (launchd watchdog + cron audit)
**Author:** researcher agent

---

## 1. Plist files -- repo vs. user-local

### In-repo (scripts/launchd/)
| File | Path |
|------|------|
| `com.pyfinagent.backend-watchdog.plist` | `/Users/ford/.openclaw/workspace/pyfinagent/scripts/launchd/com.pyfinagent.backend-watchdog.plist` |
| `backend_watchdog.sh` | `/Users/ford/.openclaw/workspace/pyfinagent/scripts/launchd/backend_watchdog.sh` |

Only the backend-watchdog is version-controlled. All other pyfinagent launchd services are user-local only.

### User-local only (~/Library/LaunchAgents/)
| Label | Plist |
|-------|-------|
| `com.pyfinagent.autoresearch` | `~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist` |
| `com.pyfinagent.ablation` | `~/Library/LaunchAgents/com.pyfinagent.ablation.plist` |
| `com.pyfinagent.mas-harness` | `~/Library/LaunchAgents/com.pyfinagent.mas-harness.plist` |
| `com.pyfinagent.backend` | `~/Library/LaunchAgents/com.pyfinagent.backend.plist` |
| `com.pyfinagent.frontend` | `~/Library/LaunchAgents/com.pyfinagent.frontend.plist` |
| `com.pyfinagent.backend-watchdog` | `~/Library/LaunchAgents/com.pyfinagent.backend-watchdog.plist` (installed copy of the repo version) |
| `com.pyfinagent.claude-code-proxy` | `~/Library/LaunchAgents/com.pyfinagent.claude-code-proxy.plist` (Claude Code's own service -- NOT a pyfinagent-owned job) |

---

## 2. Per-service documentation

### com.pyfinagent.backend-watchdog
| Field | Value |
|-------|-------|
| Purpose | Polls `http://localhost:8000/api/health` every 60s; sends SIGUSR1 on fail; after 3 consecutive fails does `launchctl kickstart -k gui/501/com.pyfinagent.backend` |
| Plist location | Both in-repo AND `~/Library/LaunchAgents/` (installed copy) |
| Script | `scripts/launchd/backend_watchdog.sh` |
| Log | `handoff/logs/backend-watchdog.log` (per plist) and `handoff/logs/` directory |
| Schedule | `StartInterval=60`, `RunAtLoad=true` |
| Current status | Loaded, exit 0, last log entry 2026-05-05T17:23:48Z (health FAIL x1 then OK) |
| KeepAlive | No |

### com.pyfinagent.backend
| Field | Value |
|-------|-------|
| Purpose | Runs `caffeinate -i -s uvicorn backend.main:app --host 0.0.0.0 --port 8000` |
| Plist location | `~/Library/LaunchAgents/com.pyfinagent.backend.plist` only |
| Log | `backend.log` at repo root |
| Schedule | `RunAtLoad=true`, `KeepAlive=true`, `ThrottleInterval=5` |
| Current status | PID 18185, last exit -15 (SIGTERM -- killed intentionally by watchdog kickstart or manual stop) |
| Notes | `ProcessType=Interactive` prevents App Nap; `PYTHONUNBUFFERED=1`; hard NOFILE limit 16384 |

### com.pyfinagent.frontend
| Field | Value |
|-------|-------|
| Purpose | Runs `next dev --port 3000` in `frontend/` |
| Plist location | `~/Library/LaunchAgents/com.pyfinagent.frontend.plist` only |
| Log | `frontend.log` at repo root |
| Schedule | `RunAtLoad=true`, `KeepAlive=true`, `ThrottleInterval=5` |
| Current status | PID 86232, exit 0 (running) |
| Notes | Contains hardcoded `AUTH_SECRET`, `AUTH_GOOGLE_SECRET`, `AUTH_GOOGLE_ID` -- these are user-local secrets and correctly NOT in the repo |

### com.pyfinagent.mas-harness
| Field | Value |
|-------|-------|
| Purpose | Runs `scripts/mas_harness/run_cycle.sh` every 1800s (30 min) |
| Plist location | `~/Library/LaunchAgents/com.pyfinagent.mas-harness.plist` only |
| Log | `handoff/mas-harness.launchd.log` (launchd stdout/stderr wrapper) and `handoff/mas-harness.log` (the real harness output) |
| Schedule | `StartInterval=1800`, `RunAtLoad=false` |
| Current status | No PID (not currently running), last exit 0. `mas-harness.launchd.log` is 0 bytes as of 2026-04-19 -- the harness writes to `mas-harness.log` directly and the launchd wrapper log is empty because the shell script redirects output. `mas-harness.log` is 38 MB (2026-05-07, active). |
| Notes | `ExitTimeOut=1500` (25 min), `AbandonProcessGroup=false` |

### com.pyfinagent.ablation
| Field | Value |
|-------|-------|
| Purpose | Runs `scripts/ablation/run_ablation.py --next-untested` nightly at 03:00 |
| Plist location | `~/Library/LaunchAgents/com.pyfinagent.ablation.plist` only |
| Log | `handoff/ablation.launchd.log` (launchd stdout/stderr) and `handoff/ablation.log` (appended by the inline shell command) |
| Schedule | `StartCalendarInterval: Hour=3, Minute=0`, `RunAtLoad=false` |
| Current status | No PID, last exit 0. `ablation.launchd.log` at `handoff/` is 3403 bytes as of 2026-05-07T03:00 (fresh -- ran last night). `ablation.launchd.log` in `handoff/logs/` is older (2026-04-17). |
| Notes | `ExitTimeOut=1200`; the `ProgramArguments` uses `-c` inline shell to chain env-loading + venv activation + python invocation |

### com.pyfinagent.autoresearch
| Field | Value |
|-------|-------|
| Purpose | Runs `scripts/autoresearch/run_nightly.sh` nightly at 02:00 |
| Plist location | `~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist` only |
| Log | `handoff/autoresearch.launchd.log` (launchd stdout/stderr) and `handoff/autoresearch.log` (appended by the shell script) |
| Schedule | `StartCalendarInterval: Hour=2, Minute=0`, `RunAtLoad=false` |
| Current status | **EXIT 127 -- FAILING** |
| Notes | `ExitTimeOut=1200`; `ProgramArguments` = `["/bin/bash", "/Users/ford/.openclaw/workspace/pyfinagent/scripts/autoresearch/run_nightly.sh"]` |

---

## 3. autoresearch exit-127 root cause (CRITICAL)

### Finding
The `autoresearch.launchd.log` file at `handoff/` (1326 bytes, last modified 2026-05-07T02:00) contains repeated lines of the form:

```
/Users/ford/.openclaw/workspace/pyfinagent/backend/.env: line 24: TV5O5XN8IS2NLR6X: command not found
/Users/ford/.openclaw/workspace/pyfinagent/backend/.env: line 25: TV5O5XN8IS2NLR6X: command not found
```

### Analysis
`backend/.env` line 24 reads `ALPHAVANTAGE_API_KEY= TV5O5XN8IS2NLR6X` -- there is a leading space before the value. When `run_nightly.sh` runs `set -a; . backend/.env; set +a` under `set -euo pipefail`, bash tokenizes `ALPHAVANTAGE_API_KEY=` (empty assignment) followed by `TV5O5XN8IS2NLR6X` as a standalone command. Bash cannot find `TV5O5XN8IS2NLR6X` in `$PATH`, so it exits 127. The `set -e` flag causes immediate script exit.

### Script existence
Both the script and the Python entrypoint it calls are present and executable:
- `scripts/autoresearch/run_nightly.sh` -- exists, -rwxr-xr-x, 991 bytes (2026-04-16)
- `scripts/autoresearch/run_memo.py` -- exists, -rwxr-xr-x, 5660 bytes (2026-04-16)

The exit 127 is NOT a missing-file problem. It is a `.env` parsing failure triggered by the leading space in `ALPHAVANTAGE_API_KEY=`.

### launchd behavior for a StartCalendarInterval job with exit 127
launchd does NOT retry a `StartCalendarInterval` job after a non-zero exit. The job runs once at the scheduled time. If it exits non-zero, the last-exit-code column in `launchctl list` is updated and the service remains listed but dormant until the next calendar interval. There is no `KeepAlive=true` here, so there is no throttled respawn cycle. The service will silently fail again at 02:00 every night until the `.env` is fixed.

### Recommended fix
Remove the leading space from `backend/.env` line 24:
```
# Before (broken):
ALPHAVANTAGE_API_KEY= TV5O5XN8IS2NLR6X
# After (fixed):
ALPHAVANTAGE_API_KEY=TV5O5XN8IS2NLR6X
```
An alternative is to change `run_nightly.sh` to use a more robust `.env` loader (`grep -E '^[A-Z_]+=.+' backend/.env | while IFS='=' read k v; do export "$k"="$v"; done`) but fixing the source (the `.env` file) is cleaner and also fixes the same issue for any other script that sources `.env`.

---

## 4. Log freshness summary

| Log file | Path | Last modified | Status |
|----------|------|--------------|--------|
| `backend-watchdog.log` | `handoff/logs/` | 2026-05-05T17:23:48Z | Fresh (active) |
| `mas-harness.log` | `handoff/` | 2026-05-07T21:22 | Fresh (active, 38 MB) |
| `mas-harness.launchd.log` | `handoff/` | 2026-04-19 (0 bytes) | Stale / empty |
| `mas-harness.launchd.log` | `handoff/logs/` | 2026-04-16 (0 bytes) | Stale / empty |
| `autoresearch.launchd.log` | `handoff/` | 2026-05-07T02:00 | Fresh but recording failures |
| `autoresearch.launchd.log` | `handoff/logs/` | 2026-04-17 (204 bytes) | Stale |
| `autoresearch.log` | `handoff/` | 2026-04-24T02:00 | Stale (no successful run since Apr-24) |
| `autoresearch.log` | `handoff/logs/` | 2026-04-19 | Stale |
| `ablation.launchd.log` | `handoff/` | 2026-05-07T03:00 | Fresh (ran last night) |
| `ablation.launchd.log` | `handoff/logs/` | 2026-04-17 | Stale |
| `ablation.log` | `handoff/` | 2026-05-07T03:20 | Fresh |
| `ablation.log` | `handoff/logs/` | 2026-04-19 | Stale |
| `backend-restart.log` | `handoff/logs/` | 2026-04-24 | Slightly stale |

There is a pattern of duplicated log files: both `handoff/` root and `handoff/logs/` subdirectory contain similarly-named files. The `handoff/` root versions are the live destinations (referenced by the current plists). The `handoff/logs/` versions are older paths from before the log path was standardized. The cron_dashboard_api's `_log_paths()` function (line 102-110) references `handoff/logs/` subdirectory paths for some logs, which means the operator dashboard log viewer is pointing at the STALE copies for watchdog and harness.

---

## 5. cron_dashboard_api.py -- _LAUNCHD_JOBS gap

**File:** `backend/api/cron_dashboard_api.py`
**Lines 87-90:**
```python
_LAUNCHD_JOBS: tuple[dict[str, str], ...] = (
    {"id": "com.pyfinagent.backend-watchdog", "schedule": "launchd interval 60s",
     "description": "External liveness watchdog (SIGUSR1 + kickstart -k after 3 fails)"},
)
```

Only 1 of 6 pyfinagent-owned launchd services is declared. The 5 missing services are:
1. `com.pyfinagent.backend` -- FastAPI server (KeepAlive, RunAtLoad)
2. `com.pyfinagent.frontend` -- Next.js dev server (KeepAlive, RunAtLoad)
3. `com.pyfinagent.mas-harness` -- autonomous MAS harness cycle (StartInterval 1800s)
4. `com.pyfinagent.autoresearch` -- nightly autoresearch memo (StartCalendarInterval 02:00)
5. `com.pyfinagent.ablation` -- nightly feature ablation (StartCalendarInterval 03:00)

`com.pyfinagent.claude-code-proxy` (PID 86235) is Claude Code's own daemon and must NOT be added to the manifest.

### Exact shape needed in _LAUNCHD_JOBS
```python
_LAUNCHD_JOBS: tuple[dict[str, str], ...] = (
    {"id": "com.pyfinagent.backend-watchdog",
     "schedule": "launchd interval 60s",
     "description": "External liveness watchdog (SIGUSR1 + kickstart -k after 3 fails)"},
    {"id": "com.pyfinagent.backend",
     "schedule": "launchd KeepAlive RunAtLoad",
     "description": "FastAPI backend (uvicorn port 8000, caffeinate -i -s)"},
    {"id": "com.pyfinagent.frontend",
     "schedule": "launchd KeepAlive RunAtLoad",
     "description": "Next.js frontend dev server (port 3000)"},
    {"id": "com.pyfinagent.mas-harness",
     "schedule": "launchd interval 1800s",
     "description": "Autonomous MAS harness cycle (run_cycle.sh every 30 min)"},
    {"id": "com.pyfinagent.autoresearch",
     "schedule": "launchd cron 02:00 daily",
     "description": "Nightly autoresearch memo (run_nightly.sh via run_memo.py) -- CURRENTLY FAILING exit 127"},
    {"id": "com.pyfinagent.ablation",
     "schedule": "launchd cron 03:00 daily",
     "description": "Nightly feature ablation (run_ablation.py --next-untested)"},
)
```

---

## 6. Phase-23.2.0 prior audit context

The `handoff/archive/phase-23.2.0/phase-23.2.0-internal-codebase-audit.md` documents that:
- The external watchdog plist is confirmed in `scripts/launchd/backend_watchdog.{sh,plist}` with verification via `tail handoff/logs/backend-watchdog.log` (Section B, row "External watchdog").
- The phase-23.2.0 audit did NOT enumerate the other launchd services; the cron audit in phase-23.3 is the first pass on the full inventory.

---

## 7. _log_paths mismatch (secondary finding)

`cron_dashboard_api.py` lines 102-110 define `_log_paths()` pointing to `handoff/logs/` for watchdog, restart, harness, and autoresearch. But the active plist destinations for mas-harness and autoresearch are `handoff/mas-harness.launchd.log` and `handoff/autoresearch.launchd.log` (root, not subdirectory). The `/api/logs/tail` endpoint for these keys will serve the stale `handoff/logs/` copies, not the live ones. This is a secondary fix recommended for the same phase.
