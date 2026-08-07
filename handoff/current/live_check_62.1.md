# live_check — masterplan step 62.1 (Slack bot under launchd + restart on current code)

**Captured 2026-08-07 · Cycle 179** · `handoff/current/experiment_results_62.1.md`

All three criteria evidenced below. The step required a `launchctl print` excerpt, an
lstart-vs-commit paste, and a digest permalink.

---

## 1. Criterion 1 — launchd agent with KeepAlive, old PID dead

```
$ launchctl print gui/$(id -u)/com.pyfinagent.slack-bot
	path = /Users/ford/Library/LaunchAgents/com.pyfinagent.slack-bot.plist
	state = running
	program = /Users/ford/.openclaw/workspace/pyfinagent/.venv/bin/python
	runs = 2
	pid = 75468
	properties = keepalive | runatload | legacy timer behavior | inferred program | managed LWCR | has LWCR
```

`KeepAlive` checked literally in the plist, not inferred from the `properties` line:

```
$ plutil -extract KeepAlive xml1 -o - ~/Library/LaunchAgents/com.pyfinagent.slack-bot.plist
<true/>
```

"Mirroring `com.pyfinagent.backend.plist`'s environment shape" — env key **names** only,
values deliberately not printed (one of them is a live credential, see §5):

```
slack-bot: PATH, PYTHONUNBUFFERED
backend:   CLAUDE_CODE_OAUTH_TOKEN, DEV_LOCALHOST_BYPASS, PATH, PYTHONUNBUFFERED
```

Shared shape matches; the two backend-only keys are each correctly absent from the bot,
which needs neither a Claude OAuth token nor the backend's localhost auth bypass.

Old PID dead:

```
$ kill -0 658
old pid 658 alive? dead (kill -0 fails, as criterion 1 requires)
```

> Note: the step's 2026-06-12 audit basis ("a manual process PID 26147 started
> 2026-06-05, NOT in launchd") was **stale on arrival** — the agent already existed with
> KeepAlive. Criterion 1 was satisfied before this cycle began; what follows is what
> actually needed doing.

## 2. Criterion 2 — process start is LATER than the newest slack_bot commit

Both sides verbatim, as the criterion demands:

```
$ ps -o lstart= -p 75468
fre.  7 aug. 20.30.28 2026

$ git log -1 --format=%ci -- backend/slack_bot/
2026-08-06 19:42:47 +0200
```

Process start **2026-08-07 20:30:28** vs newest commit **2026-08-06 19:42:47** — later by
~24h49m. MET.

Before the restart it was the other way round (`pid 658`, started `tir. 28 jul. 18.39.22
2026`), i.e. the bot had been running 9-day-stale code. Not cosmetic: that process was
missing phase-82.39 + 82.48, which left `nightly_outcome_rebuild` — a job that runs
**inside this process** at 03:00 UTC — fetching non-existent BigQuery columns nightly,
and phase-82.59, which left two Bolt listeners raising `TypeError`.

Restart performed with `launchctl kickstart -k` after a mandatory import smoke test
(`KeepAlive` + `ThrottleInterval=5` would crash-loop a bad HEAD against a working bot):

```
$ .venv/bin/python -c "import backend.slack_bot.app; print('IMPORT OK')"
IMPORT OK
$ launchctl kickstart -k gui/$(id -u)/com.pyfinagent.slack-bot
kickstart_exit=0
```

`runs` incremented 1 → 2. Neither `pkill` nor `bootout` was used: both are wrong here and
both are blocked by `.claude/hooks/pre-tool-use-danger.sh`.

## 3. Criterion 3 — a digest observed in Slack from the NEW process

**The digest was not manufactured.** `scripts/.../digest_test.py` *posts* rather than
dry-running, so a "verification" message would have been spam invented to satisfy a
checkbox. The evidence is the naturally-scheduled 23:00 CEST run.

Restart banner and digest in the **same log file**, in order — the pairing is what proves
the digest came from the new process:

```
handoff/logs/slack_bot.log
:41464:  2026-08-07 20:30:28,693 INFO __main__: Slack bot starting in Socket Mode...
:41470:  2026-08-07 20:30:29,193 INFO slack_bolt.AsyncApp: Bolt app is running!
:41551:  2026-08-07 23:00:08,951 INFO backend.slack_bot.scheduler: Evening digest sent
```

(line 40820 holds the *previous* evening digest, 2026-08-06 23:00:07 — sent by the old
process, before the restart banner. The ordering is unambiguous.)

The message itself, read back from Slack read-only:

```
=== Message from PyFinAgent (U0A0CTMGF5J) at 2026-08-07 23:00:08 CEST ===
Message TS: 1786136408.876559
:city_sunset: Evening Digest — August 07, 2026
*End-of-Day Portfolio:* +$3,830.46 (+19.1%) (as of close 2026-08-05)
*Today's Trades:* No trades executed today.
PyFinAgent Evening Summary | `/portfolio` for details
```

Channel `#ford-approvals` (`C0ANTGNNK8D`, = `settings.slack_channel_id`).
**Permalink:** `https://<workspace>.slack.com/archives/C0ANTGNNK8D/p1786136408876559`
(Slack archive form: `p` + the message TS with the dot removed.)

> Method note: the digest calls `chat_postMessage` **without capturing its return value**
> (`scheduler.py:627`), so no message `ts` is logged and a permalink cannot be derived
> from the log alone. The TS above was obtained by reading the channel read-only via the
> Slack connector. No message was sent by me.

## 4. Scheduler confirmation (pre-capture, to avoid discovering a gap at 23:00)

Queried from the live scheduler rather than assumed from the code:

```
{'id': 'evening_digest', 'schedule': "cron[hour='17', minute='0']",
 'next_run': '2026-08-07T17:00:00-04:00', 'last_run': '2026-08-06T21:00:07.878920+00:00', 'status': 'ok'}
```

`hour=17` on `America/New_York` = 23:00 CEST. Today was a US trading day, so
`_is_us_trading_day_now()` did not skip it (Sat/Sun log `evening_digest skipped`, which is
why missing tonight would have cost until Monday 2026-08-10). Both endpoints the digest
depends on were probed in advance: `portfolio` 200 (4.56s), `trades` 200 (1.10s), inside
its 30s client timeout.

Note today's 14:00 CEST *morning* digest (`last_run: 2026-08-07T12:00:17Z`) was the old
process's last act — so tonight's evening digest is the **first digest from the new
process**, exactly what the criterion asks for.

## 5. Disclosures

- **A live credential sits in the backend plist.** Confirmed by name during §1's key
  comparison; the value is deliberately not reproduced anywhere. Queued as **62.1.1**
  (P1), cross-linked to **85.4** because the 68.1 gate reported it also looks malformed
  (doubled prefix, embedded newline), which would be a candidate contributor to the
  Claude-rail latency measured there. Nothing was rotated — that is an operator decision.
- **The digest content itself shows stale data**: "as of close 2026-08-05", two days old,
  reported without any error indication. That is the autonomous-cycle failure filed as
  **85.4**, not a defect in the bot or in this step.
- `backend/slack_bot/` was **frozen** between the restart and this capture — any commit
  there would have made the newest-commit timestamp postdate the running process and
  re-broken criterion 2. Verified empty (`git status --short backend/slack_bot/`) at
  capture time.
