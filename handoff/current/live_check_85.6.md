# live_check — phase-85.6

Required evidence shape (from `.claude/masterplan.json`, `verification.live_check`):

> live_check_85.6.md with the verbatim pre-fix 409 refusal, the verbatim post-fix
> kill-switch read showing paused=false/armed=true, the sod_snapshot audit row
> proving the anchor rolled from a real cycle (not a hand-written row), and the
> cycle_history row for the cycle that produced it

All four below, verbatim. Captured 2026-08-08 22:5x–23:0x CEST (20:5x UTC).

---

## 0. Preconditions read BEFORE acting

```
$ cat handoff/.autonomous_loop.lock          # NEVER last_result
{"pid": 11128, "cycle_id": "cycle-1786219192", "released_at": "2026-08-08T19:59:54.700415+00:00", "state": "released"}
```

`state: released`, so no cycle held the lock. pid 11128 was my own full-suite
pytest run — a test wrote to the LIVE lockfile. Pre-existing phase-36.28 class,
disclosed in `experiment_results_85.6.md` §10; it blocked nothing.

## 1. Verbatim PRE-fix 409

Backend pid **143**, started **18:54:45**, which PREDATES the fix commit
`5932ac27` (22:56:34) — so this is genuinely the old code, not an inference.

```
$ curl -s -X POST http://127.0.0.1:8000/api/paper-trading/resume \
       -H 'Content-Type: application/json' -d '{"confirmation":"RESUME"}'
HTTP 409

Cannot resume: the daily-loss anchor is STALE -- it is from '2026-08-05', not today (UTC), so the daily-loss leg cannot be verified healthy against today's open. The baselines themselves are intact (sod_nav=23830.46, peak_nav=24666.57); the trailing leg is date-independent and still armed. NO operator action is required: the daily start-of-day roll stamps today's anchor at the top of the next paper-trading cycle and this refusal clears itself. Retry the resume after that cycle.
```

Both false promises are visible in that text. State was unchanged by the refusal
(a refused resume is inert), and the audit journal did not grow:

```
{'paused': True, 'sod_date': '2026-08-05'} {'armed': False, 'daily_baseline_stale': True}
handoff/kill_switch_audit.jsonl: 52 lines before AND after
```

## 2. Backend restart — the fix put IN FORCE, not merely committed

A committed fix is not a live fix: the running process holds the pre-fix modules
in `sys.modules`.

```
$ launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend
$ launchctl list | grep 'com.pyfinagent.backend$'
23676	-15	com.pyfinagent.backend
$ ps -eo pid,lstart -p 23676
23676 lør.  8 aug. 22.57.25 2026
$ tail -200 backend.log | grep -c 'EADDRINUSE|Address already in use'
0
$ curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/api/health
200
```

pid **143 → 23676**, started **22:57:25**, after the 22:56:34 fix commit. A pid in
`launchctl list` plus no `EADDRINUSE` — not merely a 200 — is the real check.

### The corrected 409, live on the new process

```
Cannot resume: the daily-loss anchor is STALE -- it is from '2026-08-05', not today (UTC), so the daily-loss leg cannot be verified healthy against today's open. The baselines themselves are intact (sod_nav=23830.46, peak_nav=24666.57); the trailing leg is date-independent and still armed. UNBLOCK CONDITION: a paper-trading cycle must START and run its start-of-day roll (Step 0, backend/services/autonomous_loop.py, PaperTrader.roll_daily_anchor -> kill_switch.update_sod_nav). That roll now runs FIRST in the cycle, before screening and analysis, so it no longer depends on the cycle finishing. If a cycle is scheduled before you need to trade, retry the resume after it starts. If none is scheduled -- the cron is weekday-only, so this includes all weekend -- no cycle will run and this refusal will NOT clear on its own; trigger a cycle, or leave the book paused. Verify with GET /api/paper-trading/kill-switch: resume succeeds once sod_date is today's UTC date and breach.armed is true. Do NOT hand-write a sod_snapshot row into handoff/kill_switch_audit.jsonl to force this -- that forges the evidence the daily leg is measured against.
```

```
contains "NO operator action is required": False
contains "this refusal clears itself"   : False
contains "UNBLOCK CONDITION"            : True
names roll_daily_anchor                 : True
```

## 3. Verification cycle 1 of 2 — the anchor rolled at Step 0

```
$ curl -s -X POST http://127.0.0.1:8000/api/paper-trading/run-now      # 20:58:27Z
{"status":"started","started":true,"message":"Daily cycle triggered"}
HTTP 200
```

**Two seconds later**, from `backend.log`:

```
{"timestamp": "2026-08-08 22:58:29,380", "level": "INFO", "module": "paper_trader", "message": "phase-85.6: start-of-day anchor rolled '2026-08-05' -> 2026-08-08 (nav=23830.46) at cycle start, independently of the mark/trade region"}
```

Under the old code this same cycle would have had to survive ~2.4 hours of
analysis to reach the roll — and the measured 08-06/08-07 cycles did not.

## 4. The `sod_snapshot` audit row — written by the CODE PATH, not by hand

```
{"ts": "2026-08-08T19:59:35.278544+00:00", "event": "pause", "trigger": "manual", "details": {}}
{"ts": "2026-08-08T20:58:29.379594+00:00", "event": "sod_snapshot", "nav": 23830.46, "date": "2026-08-08"}
{"ts": "2026-08-08T20:58:43.463277+00:00", "event": "resume", "trigger": "manual", "details": {}}
```

The `sod_snapshot` row at `20:58:29.379594Z` lands 2 seconds after the trigger,
emitted by `PaperTrader.roll_daily_anchor -> kill_switch.update_sod_nav`, with
the log line in §3 as corroboration. It was **not** hand-written — criterion 1
forbids that and it is the ask-#21 anti-pattern.

## 5. `POST /resume` — HTTP 200. The deadlock is broken.

```
$ curl -s -X POST http://127.0.0.1:8000/api/paper-trading/resume ...   # 20:58:43Z
HTTP 200
{
 "status": "resumed",
 "state": {
  "paused": false,
  "pause_reason": null,
  "sod_nav": 23830.46,
  "sod_date": "2026-08-08",
  "peak_nav": 24666.57,
  "paused_at": null,
  "auto_resume_alerted_at": null,
  "baseline_provenance": null
 }
}
```

### The immutable verification command

```
$ bash -c 'curl -s --max-time 15 http://127.0.0.1:8000/api/paper-trading/kill-switch | python3 -c "..."'
{'paused': False, 'sod_date': '2026-08-08'} {'armed': True, 'daily_baseline_stale': False}
```

Unlike 85.4's, **this command can fail** — it reads live state over HTTP. Before
the fix it printed `{'paused': True, 'sod_date': '2026-08-05'} {'armed': False,
'daily_baseline_stale': True}`.

## 6. The `cycle_history` row for the cycle that produced it

```
$ grep c67b3b15 handoff/cycle_history.jsonl
{"cycle_id": "c67b3b15", "started_at": "2026-08-08T20:58:27.358693+00:00", "completed_at": null, "duration_ms": null, "status": "started", "n_trades": 0, "error_count": 0, "data_source_ages": {}, "bq_ingest_lag_sec": null}
```

The `started` row is stamped `20:58:27.358693Z`, one second before the
`sod_snapshot` — which is the ordering claim of this whole step: the roll happens
at cycle START, not in the mark/trade region. This cycle was still running when
the live_check was written; its terminal row is appended by the `finally` block
whatever happens to it, and that is exactly what phase-85.4 hardened.

## 7. Spend accounting

- **One** of the two authorized verification cycles used (trigger 20:58:27Z). One
  remains unspent.
- No metered LLM spend was incurred by the test-writing accident described in
  `experiment_results_85.6.md` §10: `llm_call_log` had **0 rows** in the 30
  minutes covering it.
