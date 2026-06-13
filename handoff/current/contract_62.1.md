# Contract -- phase-62.1: Slack bot under launchd + restart on current code (RE-EXECUTION)

Date: 2026-06-13 (AM away session, ~07:30 CEST). Goal: goal-away-ops.

## Why this is a re-execution (state delta since the 2026-06-12 contract)

The launchd CUTOVER prescribed by the Jun-12 contract ALREADY HAPPENED and is verified
live by Main before this contract:

- `~/Library/LaunchAgents/com.pyfinagent.slack-bot.plist` EXISTS (KeepAlive=true,
  RunAtLoad=true, ThrottleInterval=5, ProcessType=Interactive, LegacyTimers=true, venv
  python `-m backend.slack_bot.app`, WorkingDirectory=repo, PATH mirrors the backend
  plist, logs to handoff/logs/slack_bot.log). No edit needed.
- The agent is loaded + running (`launchctl print ... = running`, PID 38084).
- The stale manual PID 26147 is DEAD (`kill -0` -> "no such process").
- The slack_bot_monitor.sh crontab race is already removed (Jun-12 live_check_62.1.md).

**The one remaining gap = criterion 2.** The running process lstart is `Fri Jun 12
13:14:13 2026`, which is 8 seconds OLDER than the newest commit touching
backend/slack_bot/ -- `1be98e83 @ 2026-06-12 13:14:21 +0200` ("phase-62.7 prep: P1 paging
fix in alerting.py + token cursor-advance rule"). That commit updated
`backend/slack_bot/operator_tokens.py` (+6, the I-4 cursor-advance rule) and the
bot-loaded `backend/services/observability/alerting.py` (+57, the P1 paging fix). The
running bot may therefore be executing STALE token-handling + P1-paging code -- a
safety-relevant gap for a 3-week unattended window. Fix = a code-only restart in place.

## Research-gate summary

Brief: `handoff/current/research_brief_62.1.md` (REVALIDATED 2026-06-13; gate_passed:
true, 6 sources read in full, 16 URLs, three-variant queries, recency scan). Findings
that bind this GENERATE:
- `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.slack-bot` is the canonical
  "restart in place to load freshly-committed code" verb. 2026 evidence: kickstart beats
  bootout+bootstrap for in-place restart (the latter can fail silently from inside a
  managed process tree).
- **kickstart is correct ONLY because the plist is unchanged (code-only change).** This
  GENERATE does NOT edit the plist, so kickstart applies. (If the plist were edited ->
  bootout+bootstrap instead. Not the case here.)
- `-k` semantics: launchd does SIGTERM then SIGKILL after ExitTimeOut (20s). Benign here
  -- no SIGTERM handler, in-memory APScheduler jobstore (nothing to corrupt; schedule
  rebuilds at start with the documented catch-up fire).
- Crash-loop guard (research-prescribed): run a pre-restart import smoke-test
  `.venv/bin/python -c "from backend.slack_bot.app import create_app"`; if KeepAlive
  loops on a bad import, stop with `launchctl bootout`.
- Sharpest current-code check: lstart vs commit time is sound but commit-time !=
  file-edit-time; the strongest corroboration is lstart vs `stat -f %m` of the changed
  files. Both pasted in evidence.

## Immutable success criteria (verbatim from masterplan 62.1)

1. "com.pyfinagent.slack-bot launchd agent exists with KeepAlive=true, mirroring
   com.pyfinagent.backend.plist's environment shape; the old manual PID is dead (kill -0
   fails)"
2. "ps lstart of the launchd-managed bot process is LATER than the newest git commit
   touching backend/slack_bot/ (verbatim paste of both)"
3. "a morning or evening digest observed in Slack from the NEW process (permalink or
   screenshot path in live_check_62.1.md)"

verification.command (verbatim): `launchctl print gui/$(id -u)/com.pyfinagent.slack-bot |
grep -E 'state|pid' && ps -o lstart= -p $(launchctl print gui/$(id -u)/com.pyfinagent.slack-bot | awk '/pid =/{print $3}') && git log -1 --format=%ci -- backend/slack_bot/`

## Hypothesis

A single `kickstart -k` restart makes the launchd-managed bot run the current committed
slack_bot code (lstart >> 1be98e83), satisfying criteria 1 + 2 deterministically and
in-window, with a healthy Socket Mode reconnect. Criterion 3 (a digest FROM the
launchd-managed process) is satisfied by the bot's own scheduled morning digest at
08:00 ET = 14:00 CEST today -- which is OUTSIDE this AM window and is explicitly owned by
the PM session per commit 875e25d4 ("PM-session prompt also owns 62.1/62.2 evidence
closure while they remain open"). Honest expected verdict this AM: **CONDITIONAL** on
criterion 3; the step is NOT flipped to done this session.

## Plan (restart-only sequence)

1. Pre-restart import smoke-test: `.venv/bin/python -c "from backend.slack_bot.app import create_app"` -- abort if it fails (crash-loop guard).
2. Record `git log -1 --format=%ci -- backend/slack_bot/` (= 1be98e83 ts) and `stat -f %m` of the two changed files.
3. `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.slack-bot`.
4. Verify (verbatim into experiment_results_62.1.md + live_check_62.1.md):
   - state=running, single PID; `pgrep -fl backend.slack_bot.app | wc -l` == 1.
   - NEW lstart is AFTER 1be98e83 (13:14:21) AND after both files' mtime.
   - fresh handoff/logs/slack_bot.log shows "A new session ... established" + "Bolt app is running".
   - run the immutable verification.command verbatim, paste output.
5. Criterion 3: record in live_check_62.1.md that the digest evidence is the 14:00 CEST
   scheduled morning digest from the just-restarted process -- PM-owned closure. Do NOT
   trigger an off-schedule digest (a one-shot script send would NOT be "from the launchd
   bot process" -- it would prove the script, not the criterion).
6. experiment_results_62.1.md -> ONE fresh Q/A (expect CONDITIONAL on criterion 3) ->
   harness_log Cycle 64 append (result=CONDITIONAL) -> leave 62.1 status=pending.

## Handoff-file convention (this session)

Running in SUFFIXED files (`*_62.1.md`) so the rolling slots (contract.md /
research_brief.md / experiment_results.md / evaluator_critique.md) stay reserved for the
PARKED step 62.6 (CONDITIONAL, PM-owned, closes ~06-15/16 on 39.1's 3-night evidence).
No status flip this AM -> the archive hook does not fire -> no rolling/suffix conflict.

## Out of scope

Any plist edit; signal handlers in app.py; the 62.2 token handler; digest content (62.8);
phase-9 job logic; the off-schedule digest trigger (criterion 3 is PM-owned).

## References

research_brief_62.1.md (2026-06-13 revalidation); Apple launchd/launchctl docs + man
pages; openclaw kickstart-vs-bootstrap threads (#41815/#40905); docs.slack.dev Socket
Mode; APScheduler base scheduler docs. Prior: contract_62.1.md (Jun-12, superseded by
this re-execution), live_check_62.1.md (Jun-12 criteria 1-2 record).
