---
step: phase-23.5.7.1
title: Fix format_evening_digest KeyError on dict-shaped trades_today — experiment results
date: 2026-05-09
verdict_class: PASS_PENDING_QA
verification_command: 'python3 tests/verify_phase_23_5_7_1.py'
---

# Experiment Results — phase-23.5.7.1

## What was done

One file edited (defensive boundary coerce) + 1 test file edited
(realistic envelope shape) + 2 new test files (dedicated coerce
tests + verifier).

### File 1 — `backend/slack_bot/scheduler.py:_send_evening_digest`

Replaced the bare `trades_data = trades_res.json() ...` line with
the boundary coerce:

```python
trades_res = await client.get(f"{_LOCAL_BACKEND_URL}/api/paper-trading/trades?limit=10")
# phase-23.5.7.1: /api/paper-trading/trades returns the dict envelope
# {"trades": [...], "count": N} (paper_trading.py:226). Unwrap at the
# HTTP boundary so format_evening_digest's `trades_today[:10]` slice
# gets a list, not a dict (which raises KeyError: slice(...)).
_raw = trades_res.json() if trades_res.status_code == 200 else []
trades_data = _raw.get("trades", []) if isinstance(_raw, dict) else _raw
```

`format_evening_digest` is left unchanged — formatter remains
strictly typed; fix lives upstream per Option B (researcher).

### File 2 — `tests/slack_bot/test_digest_url_semantics.py`

Updated the two `evening_digest` cases to use the realistic
dict-envelope `{"trades": [], "count": 0}` instead of a bare `[]`
— so the tests now exercise the boundary coerce on every run and
catch any future regression in the unwrap.

### File 3 (new) — `tests/slack_bot/test_evening_digest_envelope_coerce.py`

4 dedicated tests cover the boundary coerce semantics:
- `test_dict_envelope_typical_unwraps_to_inner_list`
- `test_dict_envelope_empty_unwraps_to_empty_list`
- `test_bare_list_passthrough_unchanged`
- `test_status_non_200_yields_empty_list`

Each test patches `format_evening_digest` with a recorder and
asserts the formatter received a `list`, not a dict.

### File 4 (new) — `tests/verify_phase_23_5_7_1.py`

4-check verifier:
1. `_send_evening_digest` body contains `isinstance(_raw, dict)`.
2. `format_evening_digest` body still slices `trades_today[:10]`
   (formatter strictly typed; fix did NOT couple it to envelope
   semantics).
3. The 4 coerce tests pass.
4. The 4 url-semantics tests pass (with the realistic envelope).

### Operational step

Restarted slack-bot daemon (`pkill -f "slack_bot.app"` +
`nohup .venv/bin/python -m backend.slack_bot.app`). New PID 24199.
Startup log clean: scheduler started, all 11 jobs registered, Bolt
running.

## Verification command — verbatim from `.claude/masterplan.json::23.5.7.1`

```
python3 tests/verify_phase_23_5_7_1.py
```

## Verbatim result (run 2026-05-09)

```
$ python3 tests/verify_phase_23_5_7_1.py
=== phase-23.5.7.1 verifier ===
  [PASS] evening digest has coerce: envelope coerce wired in _send_evening_digest
  [PASS] format_evening_digest unchanged: format_evening_digest still slices trades_today (formatter strictly typed; fix lives upstream)
  [PASS] coerce unit tests pass: 4 passed in 0.15s
  [PASS] url-semantics tests pass: 4 passed in 0.10s

PASS (4/4)
EXIT=0
```

## Sibling verifiers — all green (11/11 post-deploy)

Re-run after the daemon restart:

| Verifier | Result |
|----------|--------|
| 23.5.1 paper_trading_daily | PASS |
| 23.5.2 ticket_queue_process_batch | PASS |
| 23.5.2.5 heartbeat bridge | PASS (11 slack_bot, 11 non-manifest, 11 with next_run) |
| 23.5.2.6 watchdog | PASS (4/4) |
| 23.5.3 morning_digest liveness | PASS (status=ok, next_run=2026-05-10T08:00:00-04:00) |
| 23.5.3.1 digest Docker-alias fix | PASS (4/4) |
| 23.5.4 evening_digest liveness | PASS (status=ok, next_run=2026-05-10T17:00:00-04:00) |
| 23.5.5 watchdog liveness | PASS |
| 23.5.6 prompt_leak_redteam | PASS |
| 23.5.7 daily_price_refresh | PASS |
| 23.5.7.1 (this step) | PASS (4/4) |

## Why the fix is right

Per researcher's brief:
- **API shape confirmed** at `backend/api/paper_trading.py:226`:
  `result = {"trades": trades, "count": len(trades)}`. Dict
  envelope, not bare list.
- **`format_morning_digest` is NOT susceptible** — `/api/reports/`
  has `response_model=list[ReportSummary]` and returns a bare list.
  No fix needed for the morning path; this step's scope is
  evening-digest only.
- **Option B (boundary coerce, not formatter coerce)** keeps the
  formatter strictly typed (`trades_today: list`) and unwraps at
  the HTTP boundary. Cited: BetterStack pattern-matching guide,
  Real Python KeyError reference, API Response Wrapper Patterns.

## Why the false-positive in the registry persists for now

After the 23.5.7.1 deploy at 23:24 CEST, the registry still shows
`evening_digest status=ok` because the fail-open `except` in
`_send_evening_digest` previously swallowed the KeyError →
APScheduler still fired `EVENT_JOB_EXECUTED` → heartbeat listener
recorded `status=ok`. The status from THIS afternoon's crashed
fire is still in the in-memory registry (last_run=23:00 CEST).
This is expected and was documented in 23.5.4 / 23.5.7 as the
"criterion tests scheduling, not actual delivery" limitation.

The fix ensures the NEXT fire (tomorrow 23:00 CEST) won't crash;
the registry will then record an honest `status=ok`. No further
fix needed in this step.

## What this step does NOT do

- Refactor `/api/paper-trading/trades` endpoint shape (out of scope).
- Tighten the heartbeat listener to distinguish silent-failure
  from real success (separate hardening; would change EVENT_JOB_EXECUTED
  semantics). Out of scope.
- Touch `format_morning_digest` (already safe).
- Add Pydantic response_model to the trades endpoint (deferred).

## Findings to surface to the operator

1. **Tomorrow's 23:00 CEST evening_digest fire will NOT crash** —
   the boundary coerce + 4 dedicated tests prove the fix.
2. **The registry's current `status=ok` for evening_digest is a
   stale-from-this-afternoon's-crash artifact**, not a true
   in-the-wild proof. Tomorrow's clean fire will replace it.
3. **The fail-open `except` swallowing pattern is itself a
   long-term observability concern** — `EVENT_JOB_EXECUTED` should
   ideally fire only on genuine handler success. Out of scope as
   a separate enhancement.
4. **Other backend HTTP callers** of `/api/paper-trading/trades`
   may have the same dict-vs-list assumption — researcher's grep
   found only `_send_evening_digest` from the slack-bot side; out
   of scope to audit frontend / other backend callers here.

## Artifact files

- `handoff/current/contract.md`
- `handoff/current/experiment_results.md` (this file)
- `handoff/current/phase-23.5.7.1-research-brief.md`
- `tests/verify_phase_23_5_7_1.py` (new)
- `tests/slack_bot/test_evening_digest_envelope_coerce.py` (new)
- `tests/slack_bot/test_digest_url_semantics.py` (updated)
- `backend/slack_bot/scheduler.py` (1-line edit + 2 lines of
  context comment)

## How to re-run

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
python tests/verify_phase_23_5_7_1.py
```
