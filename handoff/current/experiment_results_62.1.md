# Experiment Results — masterplan step 62.1 (Slack bot under launchd + restart on current code)

**Cycle 179 | 2026-08-07 | GENERATE phase**
Contract: `handoff/current/contract_62.1.md` · Research gate:
`handoff/current/research_brief_62.1.md` (`gate_passed: true`, 9 sources)

**Headline:** this was a restart + evidence cycle, not a build. **Zero lines of
production code changed.** Criterion 1 was already satisfied before this cycle began;
criterion 2 is now satisfied by a deliberate `kickstart -k` onto current code; criterion
3 is satisfied by the naturally-scheduled 23:00 CEST evening digest (§4).

---

## 1. The step's premise was stale — measured, not assumed

The 2026-06-12 audit basis says the bot "is a manual process PID 26147 started
2026-06-05, **NOT in launchd**, no auto-restart". That has not been true for a long
time. Measured before touching anything:

```
$ launchctl print gui/$(id -u)/com.pyfinagent.slack-bot | grep -E 'state|pid|runs|program|path'
	path = /Users/ford/Library/LaunchAgents/com.pyfinagent.slack-bot.plist
	state = running
	program = /Users/ford/.openclaw/workspace/pyfinagent/.venv/bin/python
	runs = 1
	pid = 658
	last exit code = (never exited)
	properties = keepalive | runatload | legacy timer behavior | inferred program | managed LWCR | has LWCR
```

So the agent exists, `keepalive` and `runatload` are both set, and the 2026-06-05 manual
PID is long gone.

### Criterion 1's literal wording, checked literally

"KeepAlive=true, mirroring com.pyfinagent.backend.plist's environment shape":

```
$ plutil -extract KeepAlive xml1 -o - ~/Library/LaunchAgents/com.pyfinagent.slack-bot.plist
<true/>

$ # env key NAMES only -- values deliberately not printed (see §6)
$ plutil -extract EnvironmentVariables xml1 -o - ~/Library/LaunchAgents/com.pyfinagent.slack-bot.plist
slack-bot: PATH, PYTHONUNBUFFERED
$ plutil -extract EnvironmentVariables xml1 -o - ~/Library/LaunchAgents/com.pyfinagent.backend.plist
backend:   CLAUDE_CODE_OAUTH_TOKEN, DEV_LOCALHOST_BYPASS, PATH, PYTHONUNBUFFERED
```

The shared shape (`PATH`, `PYTHONUNBUFFERED`) matches; the two backend-only keys are
each correctly absent from the bot — the bot needs neither a Claude OAuth token nor the
backend's localhost auth bypass. The research gate additionally read both plists in full
and confirmed `RunAtLoad` and `WorkingDirectory` match and that `HardResourceLimits` is
backend-only. **Criterion 1 was already met on arrival.**

---

## 2. Criterion 2 — the 9-day gap, and why it was not cosmetic

Before-state, captured 2026-08-07 20:30:23 CEST:

```
OLD pid=658
OLD lstart: tir. 28 jul. 18.39.22 2026
newest backend/slack_bot/ commit: 2026-08-06 19:42:47 +0200 18659bc3
```

The process predated the newest `backend/slack_bot/` commit by ~9 days (and the 28 July
start was a reboot, not a deliberate start). Per the research gate, the running process
carried **three defects already fixed on disk**:

- **phase-82.39 + 82.48** — `nightly_outcome_rebuild` fetching non-existent BigQuery
  columns and writing a schema that never existed. That job runs **inside this process**
  at hour=3 UTC, so it had been failing silently every night.
- **phase-82.59** — two Bolt listeners raising `TypeError`.

So the restart was worth doing on its own merits, independent of the criterion.

### Pre-flight (non-skippable)

`KeepAlive` + `ThrottleInterval=5` turns a bad HEAD into a crash-loop against a
currently-working bot, so the import smoke test ran first:

```
$ .venv/bin/python -c "import backend.slack_bot.app; print('IMPORT OK')"
IMPORT OK
```

### Restart

```
$ launchctl kickstart -k gui/$(id -u)/com.pyfinagent.slack-bot
kickstart_exit=0
```

`pkill` and `bootout` were not used: both are wrong here and both are blocked by
`.claude/hooks/pre-tool-use-danger.sh`.

### After-state, captured 2026-08-07 20:30:46 CEST

```
	state = running
	runs = 2
	pid = 75468
NEW lstart: fre.  7 aug. 20.30.28 2026
old pid 658 alive? dead (kill -0 fails, as criterion 1 requires)
```

**Criterion 2 comparison, both sides verbatim:**

| | value |
|---|---|
| new process `lstart` | `fre.  7 aug. 20.30.28 2026` (2026-08-07 20:30:28 CEST) |
| newest `backend/slack_bot/` commit | `2026-08-06 19:42:47 +0200` (`18659bc3`) |
| verdict | process start is **~24h49m LATER** than the newest commit — MET |

`runs` incremented 1 → 2 and the old pid is dead, which also closes criterion 1's second
clause ("the old manual PID is dead (kill -0 fails)").

---

## 3. The new process is healthy, not merely alive

`handoff/logs/slack_bot.log`, 2026-08-07 20:30:28 onward:

```
20:30:28,612 INFO backend.slack_bot.scheduler: Scheduler started: morning digest at 8:00, evening digest at 17:00, watchdog every 15 min
20:30:28,616 INFO backend.slack_bot.scheduler: phase-9 jobs registered: ['daily_price_refresh', 'weekly_fred_refresh', 'nightly_mda_retrain', 'hourly_signal_warmup', 'nightly_outcome_rebuild', 'weekly_data_integrity', 'cost_budget_watcher']
20:30:28,692 INFO backend.slack_bot.scheduler: phase-47.1: scheduled daily_price_refresh catch-up (+20s, idempotent by day)
20:30:28,693 INFO __main__: Slack bot starting in Socket Mode...
20:30:29,193 INFO slack_bolt.AsyncApp: A new session (s_280279696) has been established
20:30:29,193 INFO slack_bolt.AsyncApp: Bolt app is running!
20:30:48,695 INFO backend.slack_bot.job_runtime: job: {'job': 'daily_price_refresh', 'status': 'started', ..., 'idempotency_key': 'daily_price_refresh:2026-08-07', 'skipped': False}
```

All 7 phase-9 jobs registered (including the previously-broken
`nightly_outcome_rebuild`), Socket Mode session established, and the price-refresh
catch-up actually executed.

---

## 4. Criterion 3 — the digest, and why it had to be the scheduled one

`digest_test.py` is **not** a dry run: it posts to the operator's channel. So a
"verification" message would have been spam manufactured to satisfy a checkbox. The only
honest evidence is a digest the system was going to send anyway.

The window is tight and was verified against the **live scheduler**, not the code:

```
$ curl -s http://127.0.0.1:8000/api/jobs/all | ... filter digest jobs
{'id': 'evening_digest', 'source': 'slack_bot', 'schedule': "cron[hour='17', minute='0']",
 'next_run': '2026-08-07T17:00:00-04:00', 'last_run': '2026-08-06T21:00:07.878920+00:00',
 'status': 'ok', 'description': 'Slack evening digest (P&L + closed trades)'}
{'id': 'morning_digest', ..., 'next_run': '2026-08-08T08:00:00-04:00',
 'last_run': '2026-08-07T12:00:17.338800+00:00', 'status': 'ok'}
```

`hour=17` on `timezone=ZoneInfo("America/New_York")` (`scheduler.py:238-248`) = **23:00
CEST tonight**. Today is Friday and a US trading day, so `_is_us_trading_day_now()`
(`scheduler.py:591`) will not skip it — Saturday and Sunday do log
`evening_digest skipped: ... is not a US trading day`, which is why missing tonight
would have cost until Monday 2026-08-10.

Note the timing detail that makes this clean evidence: today's 14:00 CEST morning digest
(`last_run: 2026-08-07T12:00:17Z`) was the **old** process's last act. Tonight's evening
digest is the **first from the new process** — precisely what the criterion asks for.

The digest's own dependencies were probed in advance rather than discovered at 23:00:

```
$ curl -s -o /dev/null -w "%{http_code} (%{time_total}s)" .../api/paper-trading/portfolio
portfolio: 200 (4.555820s)
$ curl ... "/api/paper-trading/trades?limit=10&since_today=true"
trades:    200 (1.101606s)
```

Both well inside the digest's 30s client timeout.

**RESULT: recorded in `handoff/current/live_check_62.1.md` once the 23:00 CEST run
lands.** This file is written before that moment; the live_check carries the outcome.

---

## 5. Verification — the immutable command, verbatim

```
$ launchctl print gui/$(id -u)/com.pyfinagent.slack-bot | grep -E 'state|pid' && \
  ps -o lstart= -p $(launchctl print gui/$(id -u)/com.pyfinagent.slack-bot | awk '/pid =/{print $3}') && \
  git log -1 --format=%ci -- backend/slack_bot/
	state = running
	pid = 75468
		state = active
		state = active
fre.  7 aug. 20.30.28 2026
2026-08-06 19:42:47 +0200
```

Exit 0. The three legs read: agent running, process started today, newest commit
yesterday — the ordering criterion 2 requires.

### The freeze

`backend/slack_bot/` is **frozen** between the restart and the masterplan flip: any new
commit there makes the newest-commit timestamp postdate the running process again and
re-breaks criterion 2. Checked immediately before this write:

```
$ git status --short backend/slack_bot/
(empty)
```

To be re-checked immediately before the flip.

---

## 6. Scope honesty

**Zero production code changed.** `git status --short backend/slack_bot/` is empty; the
step ships already-committed fixes by restarting, it does not re-fix them.

**Declared out of scope in the contract, before GENERATE:**

- No test message, no `digest_test.py` — both would post to the operator's Slack.
- No second instance for zero-downtime; no `pkill`; no `bootout`.
- The three defects the stale process carried are not re-fixed here.

**Discovered by the research gate, queued as their own steps rather than disclosed only
in prose** (standing rule) — and the first of these I re-verified myself, by key name
only, without printing any credential value:

- **62.1.1 (P1, SECURITY)** — plaintext `CLAUDE_CODE_OAUTH_TOKEN` in
  `com.pyfinagent.backend.plist`'s `EnvironmentVariables`. Confirmed present by name in
  §1's key comparison. The step requires resolving whether this is the *same* token as
  85.3.3 / ask #12 (the away-watchdog plist) before treating them as two findings.
- **62.1.2 (P1, SECURITY)** — `SLACK_BOT_TOKEN` exported inline by the `*/2 * * * *`
  `slack_mention_checker.sh` crontab entry, so it is visible to `crontab -l` and in the
  spawned shell's command string every two minutes.
- **62.1.3 (P2)** — operator-action step 79.15 carries a `MEASURED STATE` block claiming
  pid 83982 / 13 June / "42 days"; reality at measurement time was pid 658 / 28 July /
  9 days. Acting on it without re-measuring produces a live_check containing numbers
  that were never true — the VERIFICATION_DEFECT class the gate exists to prevent. The
  step requires deriving the whole population of such blocks, not fixing this one.
- Stale researcher memory recommending the wrong restart verb — corrected in-session by
  the researcher; the correction is committed (`1ab0b59b`).

Rotation of either credential is an **operator decision**; nothing was rotated.

---

## 7. Artifact shape

- `handoff/current/contract_62.1.md` — plan, immutable criteria verbatim, freeze declared
- `handoff/current/research_brief_62.1.md` — gate, `gate_passed: true`
- `handoff/current/experiment_results_62.1.md` — this file
- `handoff/current/evaluator_critique_62.1.md` — Q/A verdict, transcribed verbatim
- `handoff/current/live_check_62.1.md` — launchctl excerpt + lstart-vs-commit paste +
  the 23:00 CEST digest outcome
