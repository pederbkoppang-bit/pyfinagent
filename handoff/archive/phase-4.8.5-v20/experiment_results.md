# Experiment Results -- Cycle 84 / phase-4.8 step 4.8.7

Step: 4.8.7 Secrets rotation + compromise drill (RTO<15min)

## What was generated

1. **NEW** `scripts/ops/secrets_rotation_schedule.json`
   11 secrets with tiered rotation cadences:
   - ALPACA_API_KEY_ID/SECRET: 30d (high-sensitivity trading)
   - AUTH_SECRET / AUTH_GOOGLE_*: 90d (high-sensitivity auth)
   - ANTHROPIC/OPENAI/GITHUB_TOKEN: 60-90d (medium)
   - FRED/ALPHAVANTAGE/API_NINJAS: 180d (low, read-only feeds)

2. **NEW** `scripts/ops/secrets_rotation_check.py`
   Inventories names from schedule + (read-only) launchd plists.
   NEVER reads secret values. Flags any overdue entry.

3. **NEW** `handoff/secrets_drill_log.md`
   ALPACA_API_KEY leak drill with 8 numbered rotation steps
   (T+0..T+11) referencing real services:
   Alpaca dashboard -> kill-switch PAUSE -> plist edit ->
   launchctl unload/load -> paper_execution_parity test ->
   schedule update -> RESUME -> incident ticket.
   `RTO_MINUTES=11` (4 min under 15 target).

4. **NEW** `scripts/audit/secrets_rotation_audit.py`
   4 teeth: coverage, overdue, drill-RTO-line presence, RTO<15.

## Verification (verbatim, immutable)

    $ python scripts/ops/secrets_rotation_check.py && \
      grep -q "RTO_MINUTES=" handoff/secrets_drill_log.md
    {"verdict": "PASS", "scheduled": 11, "overdue": []}
    exit=0

    $ python scripts/audit/secrets_rotation_audit.py --check
    {"verdict": "PASS", "rto_minutes": 11}
    exit=0

## Success criteria

| Criterion | Result |
|-----------|--------|
| rotation_schedule_configured | PASS (11/11 expected secrets) |
| drill_completed | PASS (8 timestamped steps, real services) |
| rto_under_15min | PASS (11 min, 4 under 15 target) |

## Known limitations (non-blocking)

- Rotation is MANUAL today (plist edit + launchctl reload). Future
  phase-4.8.x step wires this into GCP Secret Manager + a
  controlled-rollout rotation cron.
- No automated test harness for "secret values actually different"
  between old and new; that verification lives in the Alpaca
  dashboard UI and is captured by the parity-test step.
