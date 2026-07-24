# Contract -- phase-62.5: healthcheck + away-watchdog (per-step file; rolling slots in use by 62.4)

Date: 2026-06-12. Goal: goal-away-ops. Research: research_brief_62.5.md (gate_passed,
7 in full, recency scan; GO).

## Research anchors (4 load-bearing)

1. P1 TRAP: raise_cron_alert_sync(severity="P1") from a one-shot process is a SILENT
   NO-OP -- P1 not in _CRITICAL_SEVERITIES (alerting.py:42) and the 3-occurrence deduper
   (:78-90) is in-memory per-process. Fix: env ALERT_CONSECUTIVE_FAILURE_THRESHOLD=1
   (settings.py:147 -> _get_default_deduper alerting.py:108-112), check the returned
   bool (:215-216), raw-webhook curl fallback (backend_watchdog.sh:61-72 pattern).
   Cross-process re-page suppression via health.jsonl p1_raised replay, NOT the deduper.
2. backend-watchdog ACTIVE (StartInterval 60) -> COEXIST with disjoint ownership: the
   away-watchdog observes everything but RESTARTS ONLY THE FRONTEND (backend recovery
   stays with the 60s agent -- two kickstart authorities on one service = double-restart
   race); slack-bot self-heals via KeepAlive.
3. Weekend false-stale: cycle is mon-fri 18:00 UTC; record cycle_age_h/cycle_fresh_26h
   HONESTLY but never page on it (criterion pages only on failed restarts). Skip
   status:"started" rows; parse completed_at (cycle_health.py:149-218 idiom).
4. kickstart on a booted-out label fails 113 -> log the kickstart attempt, then
   launchctl bootstrap fallback. launchctl print exit 0=loaded/113=not-found
   (live-verified Darwin 25.5). StartInterval fires SKIPPED during sleep (mitigated by
   backend caffeinate); RunAtLoad=true. Plist needs PATH(/opt/homebrew/bin)+HOME.
Also live-verified: /api/health 200 public; :3000/login 200 auth-free; kill-switch
endpoint 200 localhost-no-auth, backend-down fallback = replay last pause/resume in
handoff/kill_switch_audit.jsonl (no false-P1 on unknown); df -g Available (purgeable-
exclusive, conservative) for the 20GB floor; ADC print-access-token mints ~3600s tokens;
consecutive-failure state via health.jsonl replay (append-only, crash-conservative).

## Immutable success criteria (verbatim from masterplan 62.5)

1. "healthcheck verifies all listed probes and appends a structured JSON line to
   handoff/away_ops/health.jsonl"
2. "drill: a deliberately-stopped frontend (launchctl stop) is auto-recovered via
   kickstart -k with before/after states logged"
3. "com.pyfinagent.away-watchdog.plist runs healthcheck every 30 min independently of
   the two daily sessions; 2 consecutive failed restarts raise a P1 via the existing
   raise_cron_alert_sync path"

verification.command (verbatim): cd /Users/ford/.openclaw/workspace/pyfinagent && bash
scripts/away_ops/healthcheck.sh && tail -1 handoff/away_ops/health.jsonl

## Plan

1. scripts/away_ops/healthcheck.sh: probes (launchctl print x3 w/ 0/113 semantics,
   /api/health, :3000/login http code, kill-switch endpoint w/ audit-replay fallback,
   cycle freshness honest-no-page, ADC token probe, df -g >=20, gh auth status);
   frontend-only restart authority (kickstart -k, bootstrap fallback on 113, before/
   after logged); failure counting + P1 via venv python -c raise_cron_alert_sync
   (threshold env =1, bool checked, curl fallback) deduped via health.jsonl p1_raised;
   single JSON line appended per run.
2. com.pyfinagent.away-watchdog.plist: StartInterval 1800, RunAtLoad=true, env from the
   away-session plists, logs handoff/away_ops/launchd-watchdog.log; bootstrap it.
3. Drill: launchctl stop com.pyfinagent.frontend -> run healthcheck -> recovery + logs.
4. live_check_62.5.md (probe line, drill transcript, watchdog-fired line) -> ONE fresh
   Q/A -> harness_log -> flip.

## Out of scope

Backend/slack-bot restart logic (owned by backend-watchdog/KeepAlive); digest rendering
(62.8 reads health.jsonl); sentinel (62.4).
