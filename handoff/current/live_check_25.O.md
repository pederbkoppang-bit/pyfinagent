# Live-check placeholder -- phase-25.O

**Step:** 25.O -- Error escalation Slack routing (logger.exception promotion)
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "Inject scheduled-job exception; Slack alert delivered with dedup"

## Pre-deployment evidence
- 5/5 verifier PASS.
- AST clean on `backend/slack_bot/scheduler.py`.
- Behavioral round-trip patches `raise_cron_alert_sync` and confirms the
  helper invokes it with the canonical fingerprint
  (`ValueError:morning_digest`) + `severity="P1"` + `source="scheduler"`.

## Post-deployment operator workflow
1. Pull main + restart slack-bot:
   ```
   git pull origin main
   source .venv/bin/activate
   pkill -f "python -m backend.slack_bot.app" || true
   python -m backend.slack_bot.app &
   ```
2. Inject a digest failure (e.g. temporarily set `_LOCAL_BACKEND_URL`
   to an unreachable host) and trigger the morning_digest job. Expect:
   - `logger.exception("Failed to send morning digest")` stacktrace in
     `handoff/logs/slack-bot.log`.
   - P1 Slack post: `[P1] Scheduler exception in morning_digest` with
     metadata `{endpoint=morning_digest, exception_class=<...>, exception_repr=<...>}`.
3. Trigger the digest again immediately -- the second failure should be
   dedup-suppressed (same fingerprint within the AlertDeduper window).
   Wait `repeat_hours` and re-trigger -- the alert should fire again.

## Closes audit basis
bucket 24.5 F-5(f) RESOLVED. High-severity `logger.exception` sites in
`scheduler.py` now route to P1 Slack escalation with canonical
exception-class + endpoint fingerprints.

**Audit anchor for next bucket:** 25.C (Layer-1 28-skill output surfacing),
25.D (P2 backlog).
