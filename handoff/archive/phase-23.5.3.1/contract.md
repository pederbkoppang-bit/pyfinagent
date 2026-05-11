---
step: phase-23.5.3.1
title: Fix Docker-alias hostname in _send_morning_digest + _send_evening_digest
cycle_date: 2026-05-09
harness_required: true
verification: 'python3 tests/verify_phase_23_5_3_1.py'
research_brief: handoff/current/phase-23.5.3.1-research-brief.md
---

# Contract — phase-23.5.3.1

## Hypothesis

`_send_morning_digest` and `_send_evening_digest` currently call
the Mac-host-process-unreachable Docker DNS alias `backend` at 4
call sites. Both fail silently every fire (fail-open `except`
hides the `httpx.ConnectError` from APScheduler so the heartbeat
listener records `status="ok"`). Operator gets ZERO digest
messages despite a green dashboard.

The fix is the smallest extension of the 23.5.2.6 watchdog
template:
- Add ONE module-level constant
  `_LOCAL_BACKEND_URL = "http://127.0.0.1:8000"` immediately after
  `_HEALTH_PROBE_URL`.
- Replace `_BACKEND_URL` with `_LOCAL_BACKEND_URL` at the 4 call
  sites (`scheduler.py:211, 214, 236, 239`).
- Leave `_BACKEND_URL = "http://backend:8000"` untouched at line
  24 with a comment that it is currently unused (kept for
  documentation / future Docker resurrection).

After the slack-bot daemon restart, the next morning_digest and
evening_digest fires will succeed: httpx hits localhost, returns
real data, formatters build a Block Kit message, Slack post lands.

## Research-gate summary

`researcher` agent `a77f33b5f4ccb9235` ran tier=simple and
returned `gate_passed: true` with:
- 6 external sources fetched in full (≥5 floor): 12factor.net
  config, Docker Compose networking docs, httpx clients,
  APScheduler events, Pydantic Settings, OneUptime httpx async
  guide Feb 2026.
- 10 snippet-only + 6 read-in-full = 16 URLs (≥10 floor).
- Recency scan 2024-2026 performed (no findings supersede
  canonical docs).
- Three-query discipline followed.
- 8 internal files inspected.

Brief: `handoff/current/phase-23.5.3.1-research-brief.md`.

**Three explicit answers from researcher:**
1. **Option B** (single `_LOCAL_BACKEND_URL`) is the recommended
   pattern — minimum blast radius, preserves `_BACKEND_URL` for
   documentation, no operator env-var setup required (which
   matters: `.env` is sandbox-blocked).
2. **No other broken call sites** outside `scheduler.py`.
   `commands.py` has its own independent `_BACKEND_URL` already at
   localhost (line 22) — not in scope.
3. **Test design** — new `tests/slack_bot/test_digest_url_semantics.py`
   with 4 tests reusing the `_FakeAsyncClient` / `_fake_response` /
   `_fake_app` / `_patch_client_with` fixtures from the watchdog
   tests.

## Immutable success criteria (verbatim — DO NOT EDIT)

Copied verbatim from `.claude/masterplan.json::23.5.3.1.verification`:

```
python3 tests/verify_phase_23_5_3_1.py
```

The verifier exits 0 only when all of the following hold:

1. `_send_morning_digest` body does NOT reference `_BACKEND_URL`
   or `://backend:8000` (regression guard).
2. `_send_evening_digest` body does NOT reference `_BACKEND_URL`
   or `://backend:8000`.
3. Both functions reference `_LOCAL_BACKEND_URL` or
   `127.0.0.1:8000`.
4. The 4 unit tests in
   `tests/slack_bot/test_digest_url_semantics.py` all pass.

## Plan steps

1. (DONE — RESEARCH) Researcher returned brief; gate passed.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:**
   a. Edit `backend/slack_bot/scheduler.py`:
      - Add `_LOCAL_BACKEND_URL = "http://127.0.0.1:8000"` right
        after `_HEALTH_PROBE_URL`.
      - Add a comment to the existing `_BACKEND_URL` line noting
        that it is unused for the watchdog and digests post
        phase-23.5.3.1.
      - Replace `_BACKEND_URL` with `_LOCAL_BACKEND_URL` at 4
        call sites: lines 211, 214, 236, 239.
   b. Add `tests/slack_bot/test_digest_url_semantics.py` with 4
      tests covering URL pinning + Slack post-on-success for both
      digest functions. Reuse fixtures from
      `test_watchdog_alert_semantics.py` (import them directly so
      they don't drift).
   c. Add `tests/verify_phase_23_5_3_1.py` — a 4-check verifier
      that grep-checks both digest function bodies + state symbol
      + runs the 4 unit tests.
   d. Restart slack-bot daemon to deploy.
   e. Verify all sibling verifiers still PASS (23.5.1, 23.5.2,
      23.5.2.5, 23.5.2.6, 23.5.3) and run the new
      `tests/verify_phase_23_5_3_1.py`.
   f. Write `experiment_results.md`.
4. **EVALUATE phase:** spawn fresh `qa` agent. 5-item harness
   audit FIRST, then deterministic re-verification, then LLM
   judgment.
5. **LOG phase:** append `harness_log.md` AFTER Q/A returns
   PASS/CONDITIONAL. Flip 23.5.3.1 status only after the log
   append.

## Anti-patterns guarded (≥3)

1. **Option C / D over Option B** — Option C (mutating
   `_BACKEND_URL`) loses the documentation value of "this is what
   it WOULD be in a Docker-compose deployment". Option D (env-
   driven) requires operator env-var setup and is overkill for a
   local-only deployment.
2. **Touching `commands.py`** — already correctly uses localhost
   in its own independent constant.
3. **Modifying the formatters** — they are tolerant of empty
   data and not implicated in the bug.
4. **Deleting `_BACKEND_URL` entirely** — preserves a useful
   doc-value constant for any future Docker resurrection. The
   researcher specifically argued for keeping it with a comment.
5. **Self-evaluation by Main** — Q/A is mandatory.

## Out of scope

- Re-fixing the watchdog (already done in 23.5.2.6).
- The 16 sibling jobs in phase-23.5 (separate substeps).
- Adding metrics/telemetry beyond log lines.
- Refactoring `_BACKEND_URL` into pydantic-settings.
- Touching the cron_dashboard_api / job_status_api bridge.

## Backwards compatibility

- `_LOCAL_BACKEND_URL` is purely additive.
- `_BACKEND_URL` is preserved (just no longer referenced from
  digest/watchdog handlers).
- Existing tests still pass: `test_watchdog_alert_semantics.py`
  uses its own URL assertion which checks `127.0.0.1` — no
  regression.
- No `.env` changes required.

## Risk

- **Backend `/api/portfolio/performance` or `/api/reports/?limit=5`
  may have schema changes** — the researcher noted formatters are
  tolerant of empty data, so even if the endpoints return `{}`
  the digest functions still complete without error. Slack post
  WILL go out, possibly with empty content, but that's not a
  regression vs the current state (which sends NOTHING).
- **Race on slack-bot startup before backend ready** — fail-open
  `except` swallows it, same as current behavior. No new risk.
- **Operator may receive a digest at the next 8 AM ET fire** that
  they weren't getting before — this is the INTENDED behavior.
  If the operator wants to mute, they can comment out the cron
  trigger separately (out of scope).

## References

- Research brief:
  `handoff/current/phase-23.5.3.1-research-brief.md` (researcher
  `a77f33b5f4ccb9235`, 6 sources read in full).
- Masterplan: `.claude/masterplan.json::23.5.3.1.verification`.
- Files to edit:
  - `backend/slack_bot/scheduler.py` (line 24 comment, new
    constant after line 36, 4 substitutions at 211/214/236/239).
- New files:
  - `tests/slack_bot/test_digest_url_semantics.py` (4 tests).
  - `tests/verify_phase_23_5_3_1.py` (4-check verifier).
- 23.5.2.6 watchdog template:
  `handoff/archive/phase-23.5.2.6/`.
- 12-factor config: https://12factor.net/config
- Docker Compose networking:
  https://docs.docker.com/compose/how-tos/networking/
- httpx clients: https://www.python-httpx.org/advanced/clients/
