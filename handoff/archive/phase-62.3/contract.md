# Contract -- phase-62.3: Scheduled-session plists + wrapper + kickoff prompts

Date: 2026-06-12. Goal: goal-away-ops. Research: rolling brief (tier complex,
gate_passed: true, 6 sources in full; 62.2's brief preserved separately).

## Research-gate summary (4 contract-blocking findings + anchors)

1. JUNE-15 AGENT SDK CREDIT CLIFF (plan-level discovery): from 2026-06-15, claude -p on
   subscription plans draws a separate monthly Agent SDK credit (Max 5x $100/mo, 20x
   $200/mo); on exhaustion sessions STOP unless usage credits are enabled (= metered
   spend, conflicts with the $0 decision unless operator re-decides). OPERATOR ASK
   (62.7 checklist + pending asks): confirm Max tier; choose STOP-ON-EXHAUSTION
   (recommended under $0) or enable usage credits with a cap. Wrapper logs per-session
   total_cost_usd (--output-format json) + a distinct LIMIT_HIT line so the digest
   surfaces burn rate from day 1.
2. gtimeout NOT installed -- GENERATE runs brew install coreutils (PATH already covers
   /opt/homebrew/bin). Kill semantics: TERM at cap (exit 124), -k grace KILL (137);
   never --foreground.
3. claude 2.1.173 supports -p / --dangerously-skip-permissions / --model / --max-turns
   (empirically verified; --max-turns hidden from --help but parses; negative control
   confirms). Do NOT use --bare (default -p loads hooks/CLAUDE.md/MCP -- the away design
   depends on that; 333 headless END-cycle lines + doc confirmation).
4. mas-harness ZOMBIE-REVIVAL: its plist is bootout'd but NOT disabled and still in
   ~/Library/LaunchAgents -- reboot+auto-login re-bootstraps a 30-min loop racing away
   sessions. Sessions cannot self-fix (our own 62.0 hook blocks launchctl disable on
   pyfinagent labels -- working as designed). OPERATOR pre-departure action: move/rename
   the plist (added to the 62.7 checklist).
Anchors: lockfile clone from run_cycle.sh:23-33 HARDENED (noclobber atomic create +
ps name-check; check-then-create is not atomic per Bash Hackers); invocation via stdin
prompt (run_cycle.sh:59-71); CLAUDE_BIN=/Users/ford/.local/bin/claude; EnvironmentVariables
block cloned verbatim from the mas-harness plist (:7-15). Timezone: 07:30/22:00 CEST =
05:30/20:00 UTC, no DST boundary in-window; cycle ends ~19:10 UTC = 21:10 CEST (50-min PM
margin); evening digest 23:00 CEST fires 60 min into the PM session -- acceptable (digest
reads durable state; PM front-loads pending_tokens/health writes). TOKEN-ORDER CORRECTION:
away-ops-rules.md:45-47 says "apply, then advance cursor" -- that literal order DEADLOCKS
against the 62.0 hook (cursor advance is what opens the .env gate). Operative order
(module docstring, encoded in prompts): HALT-DEV check -> validate vs KNOWN_TOKEN_ENV_MAP
-> advance_cursor -> .env write -> restart -> live_check citing the line. Rules doc gets
a wording fix with an audit note (intent unchanged; not an immutable-criteria edit).
FO-1 (binding, from 62.0): all four prompts list away-ops-rules.md FIRST and quote the
10 rails inline (cycle_prompt.md structure).

## Immutable success criteria (verbatim from masterplan 62.3)

1. "both plists lint clean, use StartCalendarInterval (AM 07:30, PM 22:00 local; tz
   cross-checked against date and the 18:00 UTC cycle in the contract), invoke the
   wrapper with am|pm, and carry the same EnvironmentVariables block as
   com.pyfinagent.mas-harness.plist"
2. "wrapper implements ALL of: shared lockfile handoff/.away-session.lock with stale-PID
   reap; sentinel pre-flight failure -> prompt_digest_only.md (never abort silently);
   dirty-tree -> prompt_recovery.md branch; git pull --rebase with offline-mode fallback;
   gtimeout 14400 (am) / 7200 (pm); claude -p --dangerously-skip-permissions --model
   claude-opus-4-8 with --max-turns 250/120; every failure path logs to
   handoff/away_ops/session.log and exits 0"
3. "a manually-kickstarted dry-run session (no-op prompt) produced START/END lines in
   session.log, and a second concurrent kickstart logged SKIP (lockfile proof)"

verification.command (verbatim): plutil -lint ~/Library/LaunchAgents/com.pyfinagent.
away-session-am.plist ~/Library/LaunchAgents/com.pyfinagent.away-session-pm.plist &&
bash -n scripts/away_ops/run_away_session.sh && grep -c 'END session'
handoff/away_ops/session.log

## Plan

1. brew install coreutils (gtimeout precondition).
2. Write both plists (clone mas-harness env block; StartCalendarInterval 07:30 / 22:00;
   StandardOut/Err -> handoff/away_ops/launchd-{am,pm}.log; RunAtLoad=false; label-only
   bootstrap AFTER the dry-run passes -- not before).
3. Write scripts/away_ops/run_away_session.sh: noclobber-atomic lockfile + stale reap
   (PID + ps name-check) + trap cleanup; HALT-DEV pre-check; sentinel pre-flight (missing
   OR failing sentinel -> digest-only prompt -- fail-closed to report-only; sentinel
   ships in 62.4, so until then the wrapper treats missing-sentinel as digest-only
   except in --dry-run mode); dirty-tree -> recovery prompt; git pull --rebase ||
   offline-mode; prompt selection am/pm/recovery/digest_only; gtimeout caps; claude -p
   invocation with --output-format json captured to per-session log (total_cost_usd +
   LIMIT_HIT detection); WIP-checkpoint instruction lives in the prompts; every failure
   logs + exits 0. --dry-run flag for the criterion-3 proof (echo-only claude substitute,
   exercising lock/log paths for real).
4. Write the four prompts under scripts/away_ops/: prompt_am.md, prompt_pm.md,
   prompt_recovery.md, prompt_digest_only.md -- rails first (FO-1), reading list, token
   procedure (corrected order), ONE-step AM scope, PM evidence list (61.1c4/65.4/35.3 +
   nightly E2E once 64.x ships), ~80% budget WIP checkpoint, exit conditions.
5. Fix away-ops-rules.md:45-47 wording (audit note in the diff).
6. Dry-run: kickstart AM manually with AWAY_SESSION_DRY_RUN=1 -> START/END lines; second
   concurrent kickstart -> SKIP line. (Real bootstrap of the calendar plists happens
   here too -- they fire next at 07:30 tomorrow, BEFORE departure, giving one live
   rehearsal day with the operator still home.)
7. ONE fresh Q/A (must carry FO-1 explicitly) -> harness_log -> flip.

## Out of scope

sentinel.sh itself (62.4); healthcheck (62.5); digest sections (62.8); enabling usage
credits (operator decision, 62.7).
