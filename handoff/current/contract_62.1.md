# Contract — masterplan step 62.1 (Slack bot under launchd + restart on current code)

**Cycle:** 179 | **Date:** 2026-08-07 | **Priority:** P0 | **Depends on:** 62.0
**Mode:** unattended overnight drain (no AskUserQuestion; operator decisions become ask rows)

---

## 1. Research gate

`handoff/current/research_brief_62.1.md` — **gate_passed: true**, tier `moderate`,
9 external sources read in full, 24 URLs, recency scan performed, 14 internal files.

### The findings that shaped this plan

1. **The step's 2026-06-12 audit basis is STALE and its premise is already half-solved.**
   It says the bot "is a manual process PID 26147 started 2026-06-05, NOT in launchd".
   Measured today: `com.pyfinagent.slack-bot` IS a launchd agent, `state = running`,
   `pid = 658`, properties include `keepalive | runatload`. **Criterion 1 is already
   satisfied** — the researcher read both plists in full and confirmed the bot plist
   mirrors the backend plist's PATH / venv / `PYTHONUNBUFFERED` / `RunAtLoad` /
   `WorkingDirectory` shape, with the three backend-only keys
   (`CLAUDE_CODE_OAUTH_TOKEN`, `DEV_LOCALHOST_BYPASS`, `HardResourceLimits`) each
   correctly absent.
2. **Criterion 2 fails by 9 days, and it is not cosmetic.** `ps -o lstart=` on pid 658
   is `tir. 28 jul. 18.39.22 2026` (a reboot, not a deliberate start); the newest commit
   touching `backend/slack_bot/` is `2026-08-06 19:42:47 +0200`. The running process
   carries **three defects already fixed on disk**: phase-82.39 + 82.48 leave
   `nightly_outcome_rebuild` fetching non-existent BigQuery columns and writing a schema
   that never existed — and that job runs **inside this process** at hour=3 UTC — while
   phase-82.59 leaves two Bolt listeners raising `TypeError`. So the restart is worth
   doing on its merits, independent of the criterion.
3. **Criterion 3 has a zero-spam path, and it is time-boxed.** Digests fire 14:00 and
   23:00 CEST. `digest_test.py` is **not** a dry run — it posts. So the only honest,
   non-spamming evidence is the naturally-scheduled evening digest at 23:00 CEST
   tonight. Saturday and Sunday log `skipped: not a US trading day` (proven by the
   2026-08-01/02 log lines), so **if tonight's window is missed the next opportunity is
   Monday 2026-08-10 14:00 CEST**.
4. **Nothing replays on restart.** Verified against the *installed* apscheduler 3.11.2
   (`base.py:1066-1068`), not merely against the docs — so the 62.2 concern about "a
   stale RESUME re-firing" does not apply here.
5. **`launchctl kickstart -k` is the correct verb.** `pkill` and `bootout` are both
   wrong here and both blocked by `.claude/hooks/pre-tool-use-danger.sh`.

---

## 2. Hypothesis

62.1 is a **restart + evidence** cycle, not a build. Criterion 1 is already true;
criterion 2 needs a deliberate `kickstart -k` onto current code; criterion 3 needs the
naturally-scheduled 23:00 CEST digest observed from the new process. The step closes
tonight if the restart lands before 23:00 and the digest fires; otherwise it defers to
Monday with the restart already banked.

---

## 3. Immutable success criteria (copied VERBATIM from `.claude/masterplan.json`)

**verification.command:**

```
launchctl print gui/$(id -u)/com.pyfinagent.slack-bot | grep -E 'state|pid' && ps -o lstart= -p $(launchctl print gui/$(id -u)/com.pyfinagent.slack-bot | awk '/pid =/{print $3}') && git log -1 --format=%ci -- backend/slack_bot/
```

**verification.success_criteria:**

1. `com.pyfinagent.slack-bot launchd agent exists with KeepAlive=true, mirroring com.pyfinagent.backend.plist's environment shape; the old manual PID is dead (kill -0 fails)`
2. `ps lstart of the launchd-managed bot process is LATER than the newest git commit touching backend/slack_bot/ (verbatim paste of both)`
3. `a morning or evening digest observed in Slack from the NEW process (permalink or screenshot path in live_check_62.1.md)`

**verification.live_check:**

```
live_check_62.1.md with launchctl print excerpt, lstart-vs-commit paste, and the digest permalink
```

Immutable; NOT amended by this contract.

---

## 4. Plan

1. **Pre-flight import smoke test** (done before this contract was finalised —
   `IMPORT OK`). Non-skippable: `KeepAlive` + `ThrottleInterval=5` turns a bad HEAD
   into a crash-loop against a currently-working bot.
2. Record the before-state: old pid, its `lstart`, the newest `backend/slack_bot/`
   commit — criterion 2 needs both sides pasted verbatim.
3. `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.slack-bot`.
4. Verify: `runs` increments, pid changes, `state = running`, new `lstart` is later
   than the newest commit, old pid dead (`kill -0` fails — criterion 1's second clause),
   and the startup banner appears in `handoff/logs/slack_bot.log`.
5. **Freeze `backend/slack_bot/` between the restart and the masterplan flip.** Any new
   commit there re-breaks criterion 2. This is a real constraint on the rest of the
   night's drain.
6. Wait for 23:00 CEST; capture `Evening digest sent` together with the restart banner
   above it in the same log file. That pair is the criterion-3 artifact.
7. `live_check_62.1.md` → Q/A via the `qa-verdict` Workflow rail → transcribe verbatim
   → `harness_log.md` → flip to `done` **only if** all three criteria are met.

---

## 5. Scope boundaries

**In scope:** the restart, its evidence, and the live_check.

**Explicitly OUT of scope (declared now, not discovered at EVALUATE):**

- **No code change to `backend/slack_bot/`.** The three defects the stale process
  carries are already fixed on disk; this step ships them by restarting, it does not
  re-fix them.
- **No test message, no `digest_test.py`.** Both would post to the operator's Slack.
  The evidence is the naturally-scheduled digest or nothing.
- **No second instance** for zero-downtime, no `pkill`, no `bootout`.
- **The four out-of-scope defects the research gate surfaced get their own steps, not a
  prose mention** (standing rule): (i) plaintext `CLAUDE_CODE_OAUTH_TOKEN` in
  `com.pyfinagent.backend.plist`, (ii) plaintext `SLACK_BOT_TOKEN` exported inline by
  the `*/2 * * * *` crontab entry, (iii) the stale MEASURED STATE block at
  `.claude/masterplan.json:19584` (claims pid 83982 / 13 June / 42 days; reality is pid
  658 / 28 July / 9 days — acting on it without re-measuring would produce a false
  live_check), (iv) stale researcher memory recommending `pkill` (already corrected by
  the researcher in-session).
  (i) and (ii) are **credential hygiene** and are filed as P1 security steps.

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| HEAD has an import error → `KeepAlive` crash-loops against a currently-working bot | Import smoke test run FIRST; result `IMPORT OK` recorded before any restart |
| Restart drops an in-flight Socket Mode connection / unacked message | Bolt reconnects; researcher confirmed no cursor or scheduler state is lost, and apscheduler 3.11.2 does not replay missed runs |
| The 23:00 digest does not fire (bot unhealthy, or not a trading day) | Verify the startup banner and scheduler registration lines immediately after restart rather than waiting; if the digest is absent at 23:05, the step defers to Monday with the restart banked and an honest live_check |
| A later cycle tonight commits to `backend/slack_bot/` and re-breaks criterion 2 | Declared freeze in §4.5; re-check `git log -1 -- backend/slack_bot/` immediately before the flip |
| Restart re-fires a queued operator token or re-sends a digest | Researcher verified against installed apscheduler 3.11.2 `base.py:1066-1068` that nothing replays |

---

## 7. Done-definition

All three criteria met with verbatim evidence in `live_check_62.1.md`; Q/A verdict
transcribed; `harness_log.md` appended; masterplan flipped to `done`. If the 23:00
digest does not materialise, the step stays `pending` with the restart banked and the
Monday window recorded — the 61.2 deferred-close pattern.

---

## 8. References

- `handoff/current/research_brief_62.1.md` (this cycle's gate, 9 sources)
- Apple `launchd.plist(5)` / `launchctl(1)` — `KeepAlive`, `RunAtLoad`, `ThrottleInterval`, `kickstart -k`
- slack-bolt Socket Mode reconnection / graceful shutdown
- APScheduler 3.11.2 `base.py:1066-1068` (installed source, not docs) — no replay on restart
- `CLAUDE.md` harness protocol; `.claude/rules/research-gate.md`
- `handoff/current/goal_masterplan_drain_next.md` (tonight's binding rails)
