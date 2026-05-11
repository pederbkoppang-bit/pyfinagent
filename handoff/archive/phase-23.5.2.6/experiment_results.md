---
step: phase-23.5.2.6
title: Investigate watchdog_health_check Slack spam (every 15 min) and fix
date: 2026-05-09
verdict_class: PASS_PENDING_QA
verification_command: 'python3 tests/verify_phase_23_5_2_6.py'
---

# Experiment Results — phase-23.5.2.6

## What was done

Two surgical changes in `backend/slack_bot/scheduler.py` plus a new
test file. No other source files touched.

### File 1 — `backend/slack_bot/scheduler.py`

**Constants added (after `_HEARTBEAT_URL` definition):**
```python
_HEALTH_PROBE_URL = "http://127.0.0.1:8000/api/health"
_watchdog_last_was_healthy: bool | None = None
```

`_BACKEND_URL = "http://backend:8000"` is **left unchanged** —
researcher noted other call sites (`_send_morning_digest`,
`_send_evening_digest`) may intentionally retain the Docker alias
for a future container deployment. Only the watchdog probe is
relocated to localhost.

**`_watchdog_health_check` rewritten** with state-transition
gating. The new state machine (per researcher's refinement on the
strict-transition pattern):

| Prior state | Current observation | Behavior |
|-------------|---------------------|----------|
| `None` | True (healthy) | log only — clean baseline |
| `None` | False (unhealthy) | **POST alert** — operator should know on first probe |
| `True` | True | log only — steady |
| `True` | False | **POST alert** — transition |
| `False` | False | log only — steady (THIS WAS THE SPAM) |
| `False` | True | **POST recovery** — closed loop |

The function now:
1. Probes `_HEALTH_PROBE_URL` (localhost).
2. Computes `is_healthy: bool`.
3. Applies the table above to decide whether to post.
4. Updates `_watchdog_last_was_healthy = is_healthy` at the end.

### File 2 — `tests/slack_bot/test_watchdog_alert_semantics.py` (new)

6 tests covering the state machine:
1. `test_steady_healthy_after_clean_start_no_post` — None→True→True→True, **0 posts**.
2. `test_first_failure_after_clean_start_posts_alert` — None→False, **1 post**.
3. `test_consecutive_failures_no_repost` — None→False→False→False, **1 post** (was 3 pre-fix — the spam).
4. `test_recovery_after_failure_posts_recovery` — None→False→True, **2 posts** (alert + recovery).
5. `test_steady_healthy_after_recovery_no_more_posts` — None→False→True→True→True, **2 posts** total.
6. `test_uses_localhost_probe_url_not_docker_alias` — regression guard: probe URL must contain `127.0.0.1` or `localhost`, MUST NOT contain `://backend:8000`.

### File 3 — `tests/verify_phase_23_5_2_6.py` (new)

4-check verifier:
1. No Docker alias hostname inside `_watchdog_health_check` body.
2. `_HEALTH_PROBE_URL` is localhost-pinned.
3. `_watchdog_last_was_healthy` symbol present.
4. The 6 unit tests pass.

### Operational step

Restarted slack-bot daemon (`pkill -f "slack_bot.app"` +
`nohup .venv/bin/python -m backend.slack_bot.app`) to deploy the
fix. New PID 49965. Startup log clean — scheduler started, all 11
jobs registered, Bolt app running.

## Verification command — verbatim from `.claude/masterplan.json::23.5.2.6`

```
python3 -c 'import sys; from pathlib import Path; src=Path("backend/slack_bot").rglob("*.py"); paths=list(src); assert any("watchdog" in p.name.lower() for p in paths) or any("watchdog" in p.read_text(encoding="utf-8") for p in paths), "no watchdog source found"; print("OK source located")' && python3 tests/verify_phase_23_5_2_6.py
```

## Verbatim result (run 2026-05-09)

```
$ <verbatim immutable command>
OK source located
=== phase-23.5.2.6 verifier ===
  [PASS] no docker alias in watchdog: watchdog body free of Docker alias
  [PASS] probe URL is localhost: _HEALTH_PROBE_URL = 'http://127.0.0.1:8000/api/health'
  [PASS] state symbol present: state-machine symbol present
  [PASS] unit tests pass: 6 passed in 0.10s

PASS (4/4)
OVERALL_EXIT=0
```

```
$ .venv/bin/python -m pytest tests/slack_bot/test_watchdog_alert_semantics.py -q
6 passed in 0.10s
```

## Sibling verifiers — no regressions

All prior verifiers re-run after the slack-bot restart:

| Verifier | Result |
|----------|--------|
| `tests/verify_phase_23_5_1.py` (paper_trading_daily) | PASS, EXIT=0 |
| `tests/verify_phase_23_5_2.py` (ticket_queue_process_batch) | PASS, EXIT=0 |
| `tests/verify_phase_23_5_2_5.py` (heartbeat bridge) | `OK 11 slack_bot; 11 non-manifest; 11 with next_run`, EXIT=0 |
| `tests/verify_phase_23_5_2_6.py` (this step) | PASS 4/4, EXIT=0 |

## Root cause documented

`backend/slack_bot/scheduler.py:24` defines `_BACKEND_URL =
"http://backend:8000"` — a Docker-compose DNS alias. Pyfinagent is
local-only on a Mac; the slack-bot runs as a host process, not in
Docker. Every 15-minute probe to `{_BACKEND_URL}/api/health`
raised a `httpx.ConnectError` (DNS failure), falling into the
`except Exception` block (lines 272-290 pre-fix) which posted
`:rotating_light: Watchdog Alert -- Backend unreachable`
unconditionally on every fire.

Confirmed by direct probe:
- `curl http://127.0.0.1:8000/api/health` → **HTTP 200**, body
  `{"status":"ok",...}`
- `curl --max-time 3 http://backend:8000/api/health` → **HTTP 000**

Backend was healthy the entire time; only the URL was wrong.

## Findings to surface to the operator

1. **The watchdog should now go quiet.** No Slack post for the
   next ~15-minute fire IF the backend stays healthy. The next
   post should only arrive on a real outage.
2. **Other Docker-alias call sites** (`_send_morning_digest`,
   `_send_evening_digest`) still use `_BACKEND_URL` — researcher
   left these unchanged on the principle that they may have been
   intentionally retained for a future Docker deployment. If
   morning/evening digests are also failing silently due to the
   same hostname bug, that's a separate substep.
3. **State resets on slack-bot restart** — by design. First post-
   restart probe behaves correctly: silent if healthy, alert if
   not (per the `None→False` row of the state machine).

## What this step does NOT do

- Touch `_send_morning_digest` / `_send_evening_digest` — they may
  be intentionally on the Docker alias.
- Add exponential backoff for repeated transitions — researcher
  ruled this out; state-transition gating is sufficient.
- Persist watchdog state across daemon restarts — out of scope;
  module-level dict resets on restart, which is the documented
  pattern.
- Tune `watchdog_interval_minutes` (stays at 15).
- Add metrics or telemetry beyond log lines.

## Artifact files

- `handoff/current/contract.md` — phase-23.5.2.6 contract.
- `handoff/current/experiment_results.md` — this file.
- `handoff/current/phase-23.5.2.6-research-brief.md` — researcher.
- `tests/verify_phase_23_5_2_6.py` — 4-check verifier (new).
- `tests/slack_bot/test_watchdog_alert_semantics.py` — 6 tests
  (new).
- `backend/slack_bot/scheduler.py` — `_HEALTH_PROBE_URL`,
  `_watchdog_last_was_healthy`, refactored
  `_watchdog_health_check`.

## How to re-run the verification

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
python tests/verify_phase_23_5_2_6.py
```

## How to monitor going forward

The watchdog will now log on every fire (~every 15 min). Tail
`handoff/logs/slack_bot.log` and look for:
- `Watchdog steady-healthy` → expected steady-state.
- `Watchdog steady-unhealthy` → backend has been down for ≥1
  interval; we suppressed the post but the log is loud.
- `Watchdog unhealthy transition` → fresh outage; one Slack post
  was sent.
- `Watchdog recovery` → backend came back; one Slack post was
  sent.
