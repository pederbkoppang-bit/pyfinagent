---
step: phase-23.5.2.6
title: Investigate watchdog_health_check Slack spam (every 15 min) and fix
cycle_date: 2026-05-09
harness_required: true
verification: "python3 -c 'import sys; from pathlib import Path; src=Path(\"backend/slack_bot\").rglob(\"*.py\"); paths=list(src); assert any(\"watchdog\" in p.name.lower() for p in paths) or any(\"watchdog\" in p.read_text(encoding=\"utf-8\") for p in paths), \"no watchdog source found\"; print(\"OK source located\")' && python3 tests/verify_phase_23_5_2_6.py"
research_brief: handoff/current/phase-23.5.2.6-research-brief.md
---

# Contract — phase-23.5.2.6

## Hypothesis

The watchdog spams Slack every 15 minutes because of a Docker-vs-Mac
hostname bug, NOT because of any "post on healthy" logic.

`backend/slack_bot/scheduler.py:24` defines:
```python
_BACKEND_URL = "http://backend:8000"
```

`backend` is a Docker-compose DNS alias. On pyfinagent's local-only
Mac deployment, the slack-bot runs as a host process (not in
Docker), so `backend` does not resolve. Every 15-minute probe at
`scheduler.py:251` (`{_BACKEND_URL}/api/health`) raises a connection
exception, falling into the `except Exception` block at lines
272-290 which **unconditionally posts** `:rotating_light: Watchdog
Alert -- Backend unreachable` to Slack.

Verified by the researcher (and reproduced by Main):
- `curl http://127.0.0.1:8000/api/health` → **HTTP 200**, body
  `{"status":"ok",...}`
- `curl --max-time 3 http://backend:8000/api/health` → **HTTP 000**
  (DNS fails; connection never opens)

The 4 core jobs' heartbeat target uses `127.0.0.1` correctly
(`scheduler.py:30` `_HEARTBEAT_URL`), so it works. Only
`_watchdog_health_check` was missed in the host-vs-Docker
migration.

**Fix has two orthogonal parts (both required per researcher):**

- **Fix A — actual bug:** change the probe URL to
  `http://127.0.0.1:8000/api/health` (do NOT change `_BACKEND_URL`
  globally, since other callers like `_send_morning_digest` may
  intentionally retain the Docker alias for a future container
  resurrection). Use a local constant `_HEALTH_PROBE_URL`.

- **Fix B — prevent spam if backend genuinely goes down:** add
  module-level `_watchdog_last_was_healthy: bool | None = None` and
  post to Slack ONLY on transitions:
  - HEALTHY → UNHEALTHY (alert)
  - UNHEALTHY → HEALTHY (recovery message)
  - Steady-state (HEALTHY→HEALTHY or UNHEALTHY→UNHEALTHY) → log
    only, no Slack post
  - First probe after daemon restart (`_watchdog_last_was_healthy
    is None`) → log only, set state silently

This combination eliminates the spam stream AND keeps the watchdog
useful for genuine outages.

## Research-gate summary

`researcher` agent `aa083d843eb04a9ea` ran tier=moderate and
returned `gate_passed: true` with:
- 6 external sources fetched in full via WebFetch (≥5 floor cleared:
  Google SRE book, OneUptime alert-fatigue Jan 2026 + dedup Jan 2026
  + AI on-call Mar 2026, Checkly alert states, Sensu alert fatigue,
  APScheduler 3.x user guide)
- 7 snippet-only + 6 read-in-full = 13 URLs (≥10 floor)
- Recency scan 2024-2026 performed (3 OneUptime 2026 articles +
  recent Checkly + APScheduler stable)
- Three-query discipline followed (current-year `2026`, last-2-year
  `2025/2024`, year-less canonical)
- 6 internal files inspected with `_watchdog_health_check` body
  quoted verbatim

Brief: `handoff/current/phase-23.5.2.6-research-brief.md`.

**Researcher's four explicit answers** (see brief Q1-Q4):
1. Root cause is the URL hostname, NOT the alert structure.
2. Fix shape: A (URL) + B (state transitions). Recovery message YES,
   backoff NO (state-transition gating alone is sufficient), persist
   across restarts NO (module-level dict; first probe is silent).
3. The `/api/health` 404 from earlier is NOT a confound — the path
   is correct (`/api/health`); only the hostname is wrong.
4. Test design: 5 new tests covering steady-healthy / first-failure /
   consecutive-failures / recovery / steady-healthy-after-recovery.

## Immutable success criteria (verbatim — DO NOT EDIT)

Copied verbatim from `.claude/masterplan.json::23.5.2.6.verification`:

```
python3 -c 'import sys; from pathlib import Path; src=Path("backend/slack_bot").rglob("*.py"); paths=list(src); assert any("watchdog" in p.name.lower() for p in paths) or any("watchdog" in p.read_text(encoding="utf-8") for p in paths), "no watchdog source found"; print("OK source located")' && python3 tests/verify_phase_23_5_2_6.py
```

The first half just confirms the watchdog source can be located
(sanity gate for any future refactor that moves the file). The
second half delegates to a project verifier, which exits 0 only
when:

1. `_watchdog_health_check` does NOT call any URL containing
   `://backend:8000` (regression guard against the Docker-hostname
   bug).
2. The probe URL contains `127.0.0.1:8000` or `localhost:8000`.
3. Module has `_watchdog_last_was_healthy` symbol (state machine
   present).
4. The 5 new pytest cases in `tests/slack_bot/test_watchdog_alert_semantics.py`
   all pass.

Decoded into deterministic checks:

1. The verification command exits 0.
2. `python3 tests/verify_phase_23_5_2_6.py` exits 0 with all 4
   sub-checks PASS.
3. `pytest tests/slack_bot/test_watchdog_alert_semantics.py -q`
   shows 5 passed.

## Plan steps

1. (DONE — RESEARCH) Researcher returned brief; gate passed.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:**
   a. Edit `backend/slack_bot/scheduler.py`:
      - Add module-level `_HEALTH_PROBE_URL = "http://127.0.0.1:8000/api/health"` constant.
      - Add module-level `_watchdog_last_was_healthy: bool | None = None`.
      - Refactor `_watchdog_health_check`:
        - Probe `_HEALTH_PROBE_URL`.
        - Compute `is_healthy: bool` from response.
        - Post to Slack ONLY on transition (None→False alert,
          False→True recovery). Log on every fire.
        - Update `_watchdog_last_was_healthy = is_healthy` at end.
   b. Add `tests/slack_bot/test_watchdog_alert_semantics.py` with
      5 tests covering the state machine.
   c. Add `tests/verify_phase_23_5_2_6.py` — replayable verifier
      that grep-checks the probe URL + state-machine symbol +
      runs the 5 pytest cases.
   d. Restart slack-bot daemon to pick up the new code.
   e. Tail `handoff/logs/slack_bot.log` and confirm `Watchdog
      health check passed` log lines start appearing without any
      Slack post.
   f. Run the verification command verbatim. Capture output.
   g. Write `experiment_results.md`.
4. **EVALUATE phase:** spawn fresh `qa` agent. 5-item harness audit
   FIRST, then deterministic re-verification, then LLM judgment.
5. **LOG phase:** append `harness_log.md` AFTER Q/A returns
   PASS/CONDITIONAL. Flip 23.5.2.6 status only after the log
   append.

## Anti-patterns guarded (≥5)

1. **Globally changing `_BACKEND_URL`** — other call sites may
   intentionally still want the Docker alias for a future
   container deployment. Only change the probe URL via a local
   constant.
2. **Silencing the watchdog entirely** — it serves a real
   purpose; the fix preserves alert-on-failure with state-
   transition gating, NOT alert suppression.
3. **Adding a heavy alerting library** (PagerDuty SDK, opsgenie,
   etc.) — pyfinagent is local-only single-Mac per
   `project_local_only_deployment.md`. Module-level state is
   correct.
4. **Persisting watchdog state across daemon restarts** —
   researcher explicitly recommends NO. First-probe-after-restart
   is silent (sets baseline without posting), per Checkly's
   canonical pattern.
5. **Adding exponential backoff** — researcher explicitly
   recommends NO. State-transition gating eliminates the stream
   without the complexity.
6. **Spawning Q/A from same prompt as bug-fix** — Q/A is a
   separate, fresh agent. Self-evaluation forbidden.

## Out of scope

- Other slack_bot jobs (morning_digest, evening_digest, etc.).
- The 17 sibling masterplan steps in phase-23.5.
- Backend `/api/health` endpoint shape (already correct; returns
  HTTP 200 with `status="ok"`).
- Refactoring `_send_*_digest` to also use `127.0.0.1` (separate
  call sites; out of scope unless the operator finds them broken).
- Watchdog interval tuning (`watchdog_interval_minutes` stays at 15).

## Backwards compatibility

- Module-level state variable is purely additive.
- New constant `_HEALTH_PROBE_URL` is additive.
- `_watchdog_health_check` signature unchanged (still
  `async def _watchdog_health_check(app: AsyncApp)`).
- `slack_channel_id` consumption unchanged.
- No `.env` changes required.
- Existing tests (if any) still pass.

## Risk

- **First-probe-after-restart is silent** — by design. If the
  backend is DOWN at slack-bot startup, the operator gets NO
  alert. Mitigation: the second probe (15 min later) WILL post,
  because state will transition from `False` to `False` (no — that
  doesn't post). **Re-checked:** the state machine is
  `None → False = silent (baseline)`, then `False → False = no
  post (steady)`. So a permanent-down at startup is silent
  forever. **DECISION:** treat first probe specially — log a
  warning even on first-fail, AND post to Slack on first-fail
  (the only "silent" case is first-pass = healthy). This is one
  difference from the strict transition pattern, justified
  because `None` baseline carries no information about prior
  state.
  - **Documented refinement:** state machine is
    - `None → True`: silent (clean startup, baseline = healthy)
    - `None → False`: POST alert (first probe failed; we don't
      know if recovery applies yet, but operator should know)
    - `True → False`: POST alert
    - `False → True`: POST recovery
    - `True → True`: silent
    - `False → False`: silent
- **Race on module-level state** — APScheduler runs
  `_watchdog_health_check` in a worker thread; concurrent fires
  are prevented by APScheduler's `max_instances=1` default for
  cron jobs (`apscheduler.userguide` cited by researcher). The
  module-level dict is single-writer per fire, so no lock needed.
- **`/api/health` shape regression** — if the response shape
  changes (e.g., key renames from `status` to `health`), our
  probe will incorrectly mark unhealthy. Mitigation: existing
  endpoint shape is well-established; a regression test on the
  health endpoint is out of scope here but should be considered.

## References

- Research brief:
  `handoff/current/phase-23.5.2.6-research-brief.md` (researcher
  `aa083d843eb04a9ea`, 6 sources read in full).
- Masterplan: `.claude/masterplan.json::23.5.2.6.verification`.
- Files to edit:
  - `backend/slack_bot/scheduler.py` (lines 24, 245-291).
- New files:
  - `tests/slack_bot/test_watchdog_alert_semantics.py` (5 tests).
  - `tests/verify_phase_23_5_2_6.py` (project verifier).
- Google SRE Practical Alerting:
  https://sre.google/sre-book/practical-alerting/
- OneUptime alert-fatigue 2026:
  https://oneuptime.com/blog/post/2026-01-24-fix-monitoring-alert-fatigue/view
- Checkly alert states (state-transition pattern):
  https://www.checklyhq.com/docs/alerting-and-retries/alert-states/
- OneUptime alert dedup 2026:
  https://oneuptime.com/blog/post/2026-01-30-alert-deduplication/view
- APScheduler User Guide (max_instances=1 default):
  https://apscheduler.readthedocs.io/en/3.x/userguide.html
