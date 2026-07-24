# Contract -- phase-62.7: pre-departure dress rehearsal (prep Fri/Sat, drills Sunday)

Date: 2026-06-12. Goal: goal-away-ops. Research: research_brief_62.7.md (gate_passed,
7 in full; timed run-sheet ~85 operator-min). Per-step file (rolling slots owned by
62.6's in-flight Q/A).

## Research anchors (3 Saturday-action findings + 1 hazard)

- R-1 KILL-SWITCH P1 DEAD FOR REAL BREACHES: empty slack_webhook_url (alerting.py:
  147-156 returns False pre-send) + deduper consecutive_threshold=3 swallows one-shot
  P1s (alerting.py:111-113; P1 not threshold-exempt). A real away-window breach
  flatten+pauses (capital safe) but pages NOBODY until the next digest. FIX (prep, this
  step): P0/P1 threshold bypass + bot-token chat.postMessage fallback inside alerting
  (mirroring the live-proven healthcheck.sh:139-148 pattern), behind tests.
- R-2 macOS 26.5.1 AUTO-INSTALL QUEUED NOW (AutoInstallProductKeys carries
  MSU_UPDATE_25F80_patch_26.5.1; AutomaticallyInstallMacOSUpdates=1) -- operator
  disables Saturday; eclecticlight documents the tonight.install restart trap.
- R-3 pmset sleep=1 -- the Mac is held awake ONLY by caffeinate; if the backend job
  boots out, the machine sleeps in ~1 min and the watchdog never fires. Keystroke:
  sudo pmset -a sleep 0. (autorestart already 1; FileVault OFF -- auto-login available;
  do NOT toggle FV, Tahoe stuck-state bug.)
- I-4 TOKEN HAZARD: prompt_am.md Step-0(d) never orders cursor advance for NON-env
  tokens -- a consumed KILL SWITCH: RESUME would stay "unapplied" and could un-pause a
  FUTURE real breach. FIX (prep): Step 0 advances the cursor for EVERY processed token.
- Drill recipes: kill-switch = 4-leg composite (MCP synthetic-NAV math read-only;
  pause(trigger=drill_62_7_simulated_breach) + backend kickstart proving boot-replay
  pause survivability; P1 observation; phone RESUME -> /resume re-evaluates) -- NO
  flatten leg, NO NAV injection (update_peak is ratchet-only/unrecoverable); "restored"
  = GET /kill-switch field-identical pre/post; audit jsonl permanently gains exactly 2
  self-documenting lines (append-only, documented). POST /pause alone insufficient
  (trigger hardcoded manual = alert-silenced). AM drill: pin 63.1 via the active_goal
  calendar pointer Saturday (read-only, $0, genuinely useful); the real 07:30 Sunday
  fire counts. Token drill: KNOWN_TOKEN_ENV_MAP is EMPTY -- register a no-op AWAY DRILL
  key so the FULL semantic-cursor + hook-gate + .env chain runs once attended
  (otherwise its first live execution is 65.2's EU SCREENER: ON, unattended). Bootout
  drill: operator bootout frontend + early-fire the away-watchdog; expect
  kickstart-113-then-bootstrap in health.jsonl; 15s cold-start flake documented.

## Immutable success criteria (verbatim from masterplan 62.7)

1. "one full simulated day executed with the operator watching: AM session kickstart
   attempting a real masterplan step, PM session kickstart sending a real digest, phone
   token consumed by the next session, healthcheck auto-restart drill, kill-switch
   drill -- all PASS lines with timestamps in dress_rehearsal.md"
2. "the kill-switch drill ran in paper, evidence in kill_switch_audit.jsonl, and
   trading state was restored to pre-drill"
3. "the pre-departure checklist is signed item-by-item by the operator (auto-login or
   FileVault implications acknowledged, pmset autorestart on, updates deferred, ADC +
   gh refreshed)"

verification.command (verbatim): test -f handoff/away_ops/dress_rehearsal.md && grep -c
PASS handoff/away_ops/dress_rehearsal.md

## Plan

PREP (Fri/Sat, Main): (1) alerting.py P1 fix + tests (R-1); (2) prompt_am.md Step-0
cursor-advance-every-token fix (I-4); (3) register "AWAY DRILL" -> AWAY_DRILL_NOOP in
KNOWN_TOKEN_ENV_MAP + grandfather-exempt it; (4) dress_rehearsal.md updated with R-2/
R-3 keystrokes + the researched run-sheet + per-drill abort criteria; (5) pin 63.1 as
Sunday's AM step via active_goal.md.
SUNDAY (operator + Main): execute the run-sheet; PASS-line every drill with timestamps;
operator signs section D; close via ONE fresh Q/A -> harness_log -> flip.

## Out of scope

Editing any masterplan criteria; FileVault toggling; the macOS update itself (deferral
only); resuming autoresearch spend.
