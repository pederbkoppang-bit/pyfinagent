---
step: phase-23.5.3.1
title: Fix Docker-alias hostname in _send_morning_digest + _send_evening_digest
date: 2026-05-09
verdict_class: PASS_PENDING_QA
verification_command: 'python3 tests/verify_phase_23_5_3_1.py'
---

# Experiment Results — phase-23.5.3.1

## What was done

One file edited + two new test files. Per researcher's Option B
recommendation: minimum-blast-radius fix that mirrors the
phase-23.5.2.6 watchdog template.

### File 1 — `backend/slack_bot/scheduler.py`

**Constants block updated:**
- Added comment block above `_BACKEND_URL` documenting that it is
  now unused for any active handler (kept for documentation /
  future Docker resurrection).
- Added new constant `_LOCAL_BACKEND_URL = "http://127.0.0.1:8000"`
  immediately after `_HEALTH_PROBE_URL`.

**Call-site substitutions (4 lines):**
- `_send_morning_digest` line 222: `{_BACKEND_URL}` → `{_LOCAL_BACKEND_URL}`
- `_send_morning_digest` line 225: same
- `_send_evening_digest` line 247: same
- `_send_evening_digest` line 250: same

`_BACKEND_URL` itself is left in place at the constant block (with
the new comment). Verified by `grep` — only constant definition +
documentation comments reference it now; no handler references
remain.

### File 2 — `tests/slack_bot/test_digest_url_semantics.py` (new, 4 tests)
Reuses the `_FakeAsyncClient` / `_fake_response` / `_fake_app` /
`_patch_client_with` fixtures from the watchdog tests (inlined, not
imported, to avoid the `tests.slack_bot.*` package-path issue).

- `test_morning_digest_uses_localhost_not_docker_alias` — both
  httpx GETs assert `127.0.0.1:8000` AND no `://backend:8000`.
- `test_morning_digest_posts_to_slack_on_success` — confirms the
  Slack post still happens (`chat_postMessage.await_count == 1`).
- `test_evening_digest_uses_localhost_not_docker_alias` — same
  for the evening sibling.
- `test_evening_digest_posts_to_slack_on_success` — same.

### File 3 — `tests/verify_phase_23_5_3_1.py` (new, 4-check verifier)

- Body grep for both digest functions: must NOT contain
  `_BACKEND_URL` (with regex `(?<!_LOCAL)_BACKEND_URL\b` — avoids
  the substring trap with `_LOCAL_BACKEND_URL`) or
  `://backend:8000`. MUST contain `_LOCAL_BACKEND_URL` or
  `127.0.0.1`.
- Constant definition check: `_LOCAL_BACKEND_URL` resolves to a
  localhost-pinned URL.
- Unit tests pass (the 4 from File 2).

### Operational step

Restarted slack-bot daemon (`pkill -f "slack_bot.app"` +
`nohup .venv/bin/python -m backend.slack_bot.app`). New PID 63639.
Startup log clean.

## Verification command — verbatim from `.claude/masterplan.json::23.5.3.1`

```
python3 tests/verify_phase_23_5_3_1.py
```

## Verbatim result (run 2026-05-09)

```
$ python3 tests/verify_phase_23_5_3_1.py
=== phase-23.5.3.1 verifier ===
  [PASS] morning digest clean: morning digest uses _LOCAL_BACKEND_URL
  [PASS] evening digest clean: evening digest uses _LOCAL_BACKEND_URL
  [PASS] constant defined: _LOCAL_BACKEND_URL = 'http://127.0.0.1:8000'
  [PASS] unit tests pass: 4 passed in 0.10s

PASS (4/4)
EXIT=0

$ .venv/bin/python -m pytest tests/slack_bot/test_digest_url_semantics.py tests/slack_bot/test_watchdog_alert_semantics.py -q
10 passed in 0.11s
```

(4 new digest tests + 6 watchdog tests = 10 total; no regression.)

## Iteration note (verifier regex fix)

First verifier run reported FAIL on both digest checks because
the substring check `"_BACKEND_URL" in body` matched
`_LOCAL_BACKEND_URL`. Fixed by switching to a regex with a
negative-lookbehind: `(?<!_LOCAL)_BACKEND_URL\b`. Both checks
now pass cleanly. The actual code change was correct from the
first edit; the verifier was the bug. Documented here so the
fix history is traceable.

## Sibling verifiers — no regressions

| Verifier | Result |
|----------|--------|
| `tests/verify_phase_23_5_1.py` (paper_trading_daily) | PASS |
| `tests/verify_phase_23_5_2.py` (ticket_queue_process_batch) | PASS |
| `tests/verify_phase_23_5_2_5.py` (heartbeat bridge) | PASS |
| `tests/verify_phase_23_5_2_6.py` (watchdog) | PASS (4/4) |
| `tests/verify_phase_23_5_3.py` (morning_digest liveness) | PASS |
| `tests/verify_phase_23_5_3_1.py` (this step) | PASS (4/4) |

## What this step does NOT do

- Touch `commands.py` (already correct).
- Re-fix the watchdog (already done in 23.5.2.6).
- Modify formatters / settings.
- Investigate the 16 sibling jobs in phase-23.5.
- Add metrics / telemetry.
- Mutate `_BACKEND_URL` itself or remove it (Option C / D — both
  ruled out by researcher).

## Findings to surface to the operator

1. **Morning digest now hits localhost.** Tomorrow at
   `morning_digest_hour:00 ET` (default 8 AM ET = 14:00 CET),
   the operator should receive a real digest in the configured
   Slack channel — not a silent failure.
2. **Evening digest also hits localhost.** Tomorrow at
   `evening_digest_hour:00 ET` (default 17 ET = 23 CET), same.
3. **`_BACKEND_URL` is now a documentation-only constant.** No
   active handler references it. Removing it entirely is a
   future cleanup; left here per researcher's recommendation
   (kept the doc value).
4. **Phase-23.5.4 (evening_digest liveness) can now pass cleanly
   with no false-positive caveat** — the bug it would have
   masked is fixed.
5. **The slack_bot heartbeat listener will record `status="ok"`
   on the next morning fire** — and this time it will mean
   something.

## Artifact files

- `handoff/current/contract.md` — phase-23.5.3.1 contract.
- `handoff/current/experiment_results.md` — this file.
- `handoff/current/phase-23.5.3.1-research-brief.md` — researcher.
- `tests/verify_phase_23_5_3_1.py` — 4-check verifier (new).
- `tests/slack_bot/test_digest_url_semantics.py` — 4 tests (new).
- `backend/slack_bot/scheduler.py` — constants block + 4 call-site
  substitutions.

## How to re-run the verification

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
python tests/verify_phase_23_5_3_1.py
```
