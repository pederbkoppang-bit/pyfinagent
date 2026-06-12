# 62.7 Dress Rehearsal -- checklist + drill script (Sunday 2026-06-14, operator present)

Built incrementally by steps 62.0-62.6 as their Q/A verdicts queued operator actions.
Every drill gets a PASS/FAIL line with a timestamp; the signed checklist closes 62.7.
Estimated operator time: ~30-40 minutes.

## A0. SATURDAY PREP (research-mandated; do NOT leave for Sunday)

- [ ] DISABLE the queued macOS 26.5.1 auto-install (R-2: AutoInstallProductKeys carries
      MSU_UPDATE_25F80_patch_26.5.1 RIGHT NOW; a clicked notification can restart the
      Mac regardless of settings): System Settings > General > Software Update > turn
      OFF automatic install; verify: defaults read /Library/Preferences/com.apple.SoftwareUpdate AutomaticallyInstallMacOSUpdates -> 0
- [ ] sudo pmset -a sleep 0  (R-3: sleep=1 today -- the Mac is held awake ONLY by the
      backend's caffeinate; if that job dies, everything sleeps in ~1 min)
- [ ] DONE BY MAIN 06-12 (verify only): alerting.py P1 fix live (single-occurrence P1
      delivers via bot-token fallback -- drill message visible in the channel); backend
      restarted with it

## A. Operator keystrokes (terminal, ~5 min)

- [ ] MAS-PLIST-ZOMBIE: mv ~/Library/LaunchAgents/com.pyfinagent.mas-harness.plist \
      ~/Library/LaunchAgents/disabled.com.pyfinagent.mas-harness.plist.bak
      (reboot would otherwise revive a 30-min loop racing the away sessions)
- [ ] AWAY_MODE_ENABLED=true appended to backend/.env (printf one-liner prepared at
      rehearsal time) + bot restart -- turns on the daily away digest sections
- [ ] pmset autorestart on (sudo pmset -a autorestart 1) -- power-loss auto-boot
- [ ] Auto-login decision: enable auto-login (System Settings > Users) OR acknowledge
      FileVault implication (a reboot without login = total outage until return)
- [ ] Defer macOS software updates 21+ days (System Settings > Software Update)
- [ ] gcloud auth application-default print-access-token (verify ADC fresh)
- [ ] gh auth status (verify GitHub auth)
- [ ] OPTIONAL (recommended SKIP): SLACK_WEBHOOK_URL for the legacy webhook paging path

## B. Slack reply tokens (phone is fine, ~3 min)

- [ ] TEST TOKEN: PING            (62.2 live round-trip -- bot must thread-ACK with a
                                   line number; if already done earlier, skip)
- [ ] AWAY DRILL: ON              (exercises the FULL token->cursor->hook-gate->.env
                                   chain once, attended; writes the no-op
                                   AWAY_DRILL_NOOP flag -- removed during the
                                   ENV-LINE-81 cleanup)
- [ ] SDK CREDIT: STOP-ON-EXHAUSTION   or   SDK CREDIT: ENABLE USAGE CREDITS <cap>
      (HARD June-15 fuse; recommendation: STOP-ON-EXHAUSTION)
- [ ] MAS PLIST: MOVED            (after the mv in section A)
- [ ] WEBHOOK: SKIP               (or CONFIGURED if section A optional item done)

## C. Drills (operator watches; Main drives, ~20 min)

- [ ] AM session real kickstart: launchctl kickstart gui/$(id -u)/com.pyfinagent.away-session-am
      -- watch it run a REAL masterplan step headlessly (or the calendar's 07:30 fire
      that morning counts); verify START/END + COST lines in session.log + the
      auto-commit/push landed
- [ ] PM session kickstart: digest arrives in Slack with away sections populated
- [ ] Token round-trip consumed: the next session reads the operator's section-B tokens,
      applies/acknowledges them, advances tokens_cursor
- [ ] Frontend bootout drill (operator keystroke -- Main is hook-blocked):
      launchctl bootout gui/$(id -u)/com.pyfinagent.frontend
      -> within 30 min the away-watchdog healthcheck detects absent, kickstart fails
      113, bootstrap fallback recovers; verify the health.jsonl line; then verify
      /login 200. (Closes the 62.5 criterion-2 residual.)
- [ ] Kill-switch drill (paper): simulated breach -> flatten+pause -> P1 alert in Slack
      -> operator replies KILL SWITCH: RESUME -> resumed; state restored to pre-drill
- [ ] healthcheck auto-restart drill already live-proven 06-12 (KeepAlive <3s); spot-check
      one watchdog line from today

## D. Sign-off

- [ ] Operator confirms: Mac stays on + plugged + lid open/clamshell-safe, network stable
- [ ] All section A-C boxes checked or explicitly waived with a note
- [ ] Main appends the signed transcript here, then closes 62.7 via its own criteria

## Reference: what runs while you are away (one-screen summary)

07:30 CEST  AM dev session (one masterplan step under the full harness loop)
14:00 CEST  morning digest (compact: asks + health)
18:00 UTC   trading cycle (mon-fri; kill-switch auto-pause 4%/10%; resume = token only)
20:00 CEST  (18:00 UTC cycle ends ~21:10 CEST)
22:00 CEST  PM evidence session (cycle evidence, healthcheck, nightly tests, notes)
23:00 CEST  evening digest (full away report: trades by market, NAV, shipped, asks,
            health, defects)
every 30min away-watchdog healthcheck (frontend-only restart authority; P1 after 2
            consecutive failed restarts via the live-proven bot-token path)
Reply HALT-DEV anytime to stop dev sessions (trading + digests continue); RESUME-DEV
to restart them. Every open decision always sits in the digest with its exact reply
string.
