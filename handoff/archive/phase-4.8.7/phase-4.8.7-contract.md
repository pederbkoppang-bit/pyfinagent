# Contract -- Cycle 84 / phase-4.8 step 4.8.7

Step: 4.8.7 Secrets rotation + compromise drill (RTO<15min)

## Hypothesis

Ship the secrets-rotation tooling + a compromise-drill log.

- `scripts/ops/secrets_rotation_check.py` inventories every secret
  in use (ALPACA_API_KEY_ID, AUTH_GOOGLE_ID, AUTH_SECRET,
  ANTHROPIC_API_KEY, OPENAI_API_KEY, FRED_API_KEY, etc.) from the
  launchd plists + .env files, validates each has a documented
  rotation cadence, and emits a schedule report. Exits 0 when
  every secret has (name, source, rotation_days, last_rotated_at).
- `handoff/secrets_drill_log.md` records a compromise drill with
  a literal `RTO_MINUTES=` line + rotation steps per secret.
- 15-minute RTO target for full rotation of the most critical
  secret (ALPACA_API_KEY) -- the one that can move money.

NIST SP 800-63B recommends rotating long-lived access keys on
compromise + periodically (90 days typical for low-risk, 30 days
for high-risk). AWS/GCP best-practice: rotate service-account keys
every 90 days; API keys every 30-90 days.

## Scope

Files created:

1. **NEW** `scripts/ops/secrets_rotation_check.py`
   - Inventories secrets from:
     * `~/Library/LaunchAgents/com.pyfinagent.*.plist` (macOS
       launchd env)
     * `backend/.env` (repository-scoped; PERMISSION-DENIED is OK
       -- report which vars are expected by signature grep
       instead of reading values)
     * `frontend/.env.local` (same handling)
   - Each secret carries rotation_days + last_rotated_at (read from
     a new sidecar `scripts/ops/secrets_rotation_schedule.json`).
   - Emits `handoff/secrets_rotation_check.json` with the inventory
     + flags any secret whose `days_since_rotation > rotation_days`.

2. **NEW** `scripts/ops/secrets_rotation_schedule.json`
   Schedule sidecar with per-secret rotation cadence (days) and
   last_rotated_at ISO date. Initial cadences from best-practice:
     ALPACA_API_KEY_ID: 30d (trading)
     ALPACA_API_SECRET_KEY: 30d
     AUTH_SECRET: 90d
     AUTH_GOOGLE_ID / AUTH_GOOGLE_SECRET: 90d
     ANTHROPIC_API_KEY / OPENAI_API_KEY / FRED_API_KEY: 60d
     ALLOWED_EMAILS: not a secret, no cadence

3. **NEW** `handoff/secrets_drill_log.md`
   Compromise drill with literal `RTO_MINUTES=N` line.
   Scenario: ALPACA_API_KEY leak on GitHub public gist. Measured
   rotation: detect, rotate in Alpaca console, push new key to
   launchd plist, restart backend. Target <15 min.

4. **NEW** `scripts/audit/secrets_rotation_audit.py`
   Enforces:
   (a) every expected secret has a schedule entry
   (b) no secret's days_since_rotation > rotation_days
   (c) drill log has `RTO_MINUTES=` line with value < 15
   (d) drill log names the rotated secret + rotation steps

## Immutable success criteria

1. rotation_schedule_configured -- every expected secret listed.
2. drill_completed -- drill log exists with scenario + steps.
3. rto_under_15min -- drill_log `RTO_MINUTES=` value < 15.

## Verification (immutable)

    python scripts/ops/secrets_rotation_check.py && \
    grep -q "RTO_MINUTES=" handoff/secrets_drill_log.md

Plus: `python scripts/audit/secrets_rotation_audit.py --check`.

## Anti-rubber-stamp

qa must:
- Check the secret inventory isn't a hardcoded stub; the script
  reads actual plist/env files where accessible and FAILS if a
  known secret (AUTH_SECRET) is missing from schedule.
- Verify the RTO_MINUTES comparison is < 15 literally; a constant
  pass would fail if we bumped the value to 20.
- Confirm rotation steps reference real services (Alpaca
  dashboard URL, launchctl unload/load, etc.) not placeholder
  "rotate key somehow".

## References

- NIST SP 800-63B Digital Identity Guidelines (session tokens +
  rotation)
- OWASP Key Management Cheat Sheet
- AWS/GCP 90-day service-account rotation convention
- backend/.env expected vars: ALPHAVANTAGE_API_KEY, FRED_API_KEY,
  API_NINJAS_KEY, GITHUB_TOKEN, ANTHROPIC_API_KEY, OPENAI_API_KEY
  (per .claude/rules/security.md)
