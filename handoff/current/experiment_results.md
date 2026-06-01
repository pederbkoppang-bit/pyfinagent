# Experiment Results — phase-54.2 (Reliable daily Slack digests for the away week)

**Date:** 2026-06-01. **Status:** complete (lifeline verified by a live delivered digest;
cron-health line shipped; bot NOT force-restarted — deliberate, documented).

## What was done

1. **Verified the supervisor** (read-only): the bot is supervised by
   `scripts/slack_bot_monitor.sh` (`*/5` user-cron, nohup-restart + iMessage); Mac
   won't sleep (caffeinate). Corrects 54.1's "unsupervised" framing.
2. **Sent a live confirmation digest** to the operator channel via the standalone
   one-shot Web-API path (no Socket Mode → cannot spawn a 2nd bot). Proof the lifeline
   delivers end-to-end.
3. **Shipped a fail-open cron-health line** into `format_morning_digest` (byte-identical
   when None) + scheduler computes it from `/api/jobs/all`.
4. **Did NOT force-restart the bot** (deliberate — avoids a false "CRASHED" iMessage to
   the remote operator + lifeline risk; line activates on next natural restart).

## Files changed

| File | Change |
|------|--------|
| `backend/slack_bot/formatters.py` | `format_morning_digest` +`cron_health: str\|None=None` kwarg → one section before the footer divider; byte-identical when None. |
| `backend/slack_bot/scheduler.py` | +`_compute_cron_health(client)` (fail-open, derives line from `/api/jobs/all`); `_send_morning_digest` passes it to the formatter. |
| `scripts/ops/send_confirmation_digest.py` | NEW — standalone one-shot `AsyncWebClient` sender (sys.path-bootstrapped); posts the digest + away-week note to the operator channel. |
| `backend/tests/test_phase_54_2_digest_cron_health.py` | NEW — 8 tests (byte-identity, single-block render, helper green/failed/fail-open/non-200/empty). |
| `handoff/current/live_check_54.2.md` | The operator-auditable lifeline verification (supervisor + delivered ts + cron-health + cadence). |

## Verification output (verbatim)

### Tests
```
python -m pytest backend/tests/test_phase_54_2_digest_cron_health.py -q
8 passed in 0.15s

# regression (existing digest tests):
python -m pytest backend/tests/test_phase_slack_digest_71.py backend/tests/test_phase_51_3_digest_guard.py -q
17 passed in 2.48s
```

### Syntax
```
python -c "import ast; ast.parse(open('backend/slack_bot/formatters.py').read()); ast.parse(open('backend/slack_bot/scheduler.py').read())"  -> both parse
python -c "import ast; ast.parse(open('scripts/ops/send_confirmation_digest.py').read())"  -> script parses
```

### Pre-flight + live send (the decisive lifeline proof)
```
channel set: True ( C...11chars ) ; bot_token set: True (xoxb-) ; slack_sdk import OK
/api/health=200 ; /api/jobs/all: 19 jobs ; statuses: failed(2), never_run(2), ok(11), running(2), scheduled(2)
  failed   : com.pyfinagent.ablation, com.pyfinagent.autoresearch   (last run; 54.1 fix clears tonight)
  never_run: evening_digest, weekly_fred_refresh                    (not yet fired this window -- normal)

# live send:
ok=True channel=C0ANTGNNK8D ts=1780324556.083759
```

## Acceptance-criteria mapping (phase-54.2)

| # | Criterion | Result |
|---|-----------|--------|
| 1 | morning+evening digests scheduled + bot running confirmed; down → fix/escalate | PASS — both scheduled; bot up + supervised (5-min cron monitor); Mac won't sleep |
| 2 | ≥1 live digest delivered + receipt (ts/channel) | PASS — `ok=True channel=C0ANTGNNK8D ts=1780324556.083759` |
| 3 | content covers NAV/P&L/positions, kill-switch/gate, 54.1 cron-health, elevation progress | PASS — delivered digest carried portfolio + cron-health line + away-week/elevation block; cron-health shipped to daily digest (byte-identical-when-None) |
| 4 | LLM-summarized body flagged op-gated (not silently spent); live_check records delivery + cadence | PASS — digest is $0/template (not op-gated); live_check_54.2.md records ts/channel/cadence |

## DO-NO-HARM / scope honesty

- `cron_health=None` default ⇒ existing digests byte-identical (proven by the
  byte-identity test). No money-path / trading-engine code touched.
- No launchd plist added (researcher: would spawn a 2nd instance → double-fired digests
  + double-fired heavy crons). Bot NOT force-restarted (false-alarm + risk avoidance).
- $0: no LLM, no pip, no BQ, no `.env`/secret edit (token via `SecretStr.get_secret_value`).
- Sending the digest is authorized: operator explicitly requested Slack updates while
  away; destination is the configured operator channel (settings, not observed content).
- The cron-health line honestly lists autoresearch/ablation as last-run-FAILED for one
  more night, with inline context that the 54.1 fix clears them tonight (no cry-wolf).

## Cycle-2 follow-up — criterion 3 fix (kill-switch + go-live-gate state)

Fresh Q/A `ad9d4c2b05d9e3a36` returned **CONDITIONAL**: criterion 3 immutably enumerates
"kill-switch + go-live-gate state" as required digest content, and that line was absent
from both the daily digest and the confirmation digest (a correct catch — for an
away-week lifeline, "is the system halted?" is the most decision-relevant signal).
Per the documented cycle-2 flow I fixed it (same fail-open precomputed-kwarg pattern):

- `format_morning_digest` gains `system_state: str | None = None` → a section after the
  portfolio block (byte-identical when None).
- `backend/slack_bot/scheduler.py`: `_compute_system_state(client)` derives the line
  from `/api/paper-trading/kill-switch` (paused / breach / active + daily-trail pcts) +
  `/api/paper-trading/gate` (ELIGIBLE/NOT + n/total), each leg fail-open;
  `_send_morning_digest` passes it in.
- `scripts/ops/send_confirmation_digest.py`: mirrors `_system_state` + passes it.
- Tests grew 8 → **14** (added: system_state byte-identity, single-block render, and
  `_compute_system_state` active+gate / paused / breach / fail-open via a URL-routing
  fake client). 17 digest regression tests still green.

**Re-delivered confirmation digest (now carrying the state line):**
```
system_state: ':large_green_circle: *Kill switch:* ACTIVE (daily -1.5%/4% | trail -0.1%/10%)
               *Go-live gate:* NOT ELIGIBLE (1/5)'
cron_health : ':warning: *Crons:* 15/19 healthy -- FAILED: com.pyfinagent.ablation, com.pyfinagent.autoresearch'
ok=True channel=C0ANTGNNK8D ts=1780325165.760459
```
Criterion 3 now PASS: the delivered digest covers NAV/P&L + kill-switch + go-live-gate
state + cron-health + the away-week/elevation block. Still $0/template; bot still not
force-restarted (the state line activates in daily digests on next natural restart;
demonstrated live via the re-delivered confirmation digest).
