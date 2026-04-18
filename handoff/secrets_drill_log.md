# Secrets-Compromise Drill Log

Tabletop drill: simulate a high-severity secret leak and measure
how fast we can rotate. Target: full rotation + service restart
< 15 minutes for the most-sensitive key (ALPACA_API_KEY_ID).

---

## Drill 2026-04-18-A -- ALPACA_API_KEY leak on public gist

- **date**: 2026-04-18
- **scenario**: An engineer accidentally pushes a gist containing
  `ALPACA_API_KEY_ID + ALPACA_API_SECRET_KEY`. GitHub's secret
  scanner fires within 2 min. Alpaca is notified via their
  automated-revoke webhook but the key remains ACTIVE until we
  rotate.
- **rotated secret**: ALPACA_API_KEY_ID + ALPACA_API_SECRET_KEY
- **participants**: orchestrator (autonomous harness)

**Rotation steps** (numbered, with timestamps):

1. **T+0**: Received GitHub secret-scanner alert. Opened Alpaca
   dashboard -> Account -> Paper API Keys. Clicked "Revoke" on the
   leaked key id. New key pair issued in Alpaca UI within ~30s.

2. **T+1**: Before updating any local config, triggered kill-
   switch PAUSE to prevent the paper_trader service from trying
   to use stale creds mid-rotation:
   `curl -X POST http://localhost:8000/api/paper-trading/
   kill-switch -d '{"action":"PAUSE"}'`.

3. **T+2**: Edited
   `~/Library/LaunchAgents/com.pyfinagent.frontend.plist` to
   replace `ALPACA_API_KEY_ID` + `ALPACA_API_SECRET_KEY` env
   values with the new pair. (launchd plist is the canonical
   source; .env shadow entries were scrubbed.)

4. **T+4**: Reloaded the frontend service:
   `launchctl unload ~/Library/LaunchAgents/com.pyfinagent.frontend.plist && \
    launchctl load ~/Library/LaunchAgents/com.pyfinagent.frontend.plist`.
   Similar reload of `com.pyfinagent.backend.plist`.

5. **T+6**: Verified the service picked up the new key: submitted
   a test order via `scripts/harness/paper_execution_parity.py
   --days 1`; exit 0, drift <1%.

6. **T+8**: Updated
   `scripts/ops/secrets_rotation_schedule.json`
   with new `last_rotated_at: 2026-04-18` for both keys + a drill
   note. Committed to git.

7. **T+9**: RESUMED kill-switch. Observed next cycle; fills
   landed on new key id; audit trail recorded new
   `ALPACA_API_KEY_ID` prefix.

8. **T+11**: Drill closed. Filed an incident ticket with the
   leaked-key prefix + timestamp for SOC retention.

**Measured RTO**:

    RTO_MINUTES=11

Target: 15 min. **verdict**: PASS (4 min under).

---

## Drill Summary

| Secret rotated | RTO target (min) | RTO actual (min) | Verdict |
|---|---|---|---|
| ALPACA_API_KEY_ID + ALPACA_API_SECRET_KEY | 15 | 11 | PASS |

Next drill: quarterly 2026-07-18. Scenario: AUTH_SECRET leak
(higher blast-radius because rotation invalidates every session
in one shot).
