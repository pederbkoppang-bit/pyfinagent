# Pre-Flight Readiness — Operator Briefing

**Date:** 2026-05-21
**Next cron fire:** **18:00 UTC / 20:00 Europe/Oslo** today (Mon-Fri 14:00 America/New_York).

---

# VERDICT: **NOT_READY**

The autonomous cycle WILL fire at 18:00 UTC, but it will produce zero trades unless the operator takes action below. Two blockers, one advisory.

---

## Top-3 risks blocking the cycle

| # | Risk | Severity | Operator Action |
|---|---|---|---|
| 1 | **Kill switch is PAUSED.** Manual pause set 2026-05-19 19:34 UTC. Daily-loss and trailing-DD are BOTH within limits (-0.09% / 4.61% vs 4% / 10% caps); the pause is operator-initiated, not auto-fired. `autonomous_loop.py:746-748` short-circuits Step 5.5 with `kill_switch_halted` summary when paused. | 🛑 **FAIL** (blocks all trading) | **Resume kill switch.** Use the dashboard resume button OR `POST /api/paper-trading/kill-switch/resume`. |
| 2 | **Anthropic Claude API credit balance unverified, last-known empty.** `settings.gemini_model = "claude-sonnet-4-6"` routes via `make_client` to the Anthropic API. Phase-31.1 Stage 3 Run 1 (2026-05-20) found the balance exhausted; no funding event or model swap has happened since. First synthesis call in the next cycle will return a credit-balance error and the analysis pipeline will halt mid-stage. | 🛑 **FAIL** (blocks all analysis) | **Fund Anthropic balance** at https://console.anthropic.com/settings/billing **OR** edit `backend/.env` to set `GEMINI_MODEL=gemini-2.5-pro` (or `gemini-2.5-flash`) — routes to Vertex AI via ADC, no per-call ceiling. Restart backend via `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend`. |
| 3 | **Tech sector at 49.54% of NAV; cap is 30%.** Existing over-cap state is grandfathered (`portfolio_manager.py:209-285` is a pre-trade gate, not a force-divest). The cycle will run; new Tech BUYs will be blocked. Non-Tech BUYs remain allowed. | ⚠️ **WARN** (advisory) | OPTIONAL: schedule a manual SELL of one Tech position before the cron to make room for new Tech adds. Otherwise no action needed. |

---

## What's READY (6 of 9 categories PASS)

| ✅ | Category | Quick fact |
|---|---|---|
| ✅ A | Infra | backend HTTP 200 (v6.15.5), 3 MCP servers ok, frontend up |
| ✅ B | BQ schema | All 4 phase-32 columns present on `paper_positions`; round-trip cols present on `paper_trades` |
| ✅ C | Code paths | 6 critical modules import; pytest 285 passed, 0 failures |
| ✅ D | Data integrity | 11 positions; no NO_STOP, no NULL `entry_strategy`, no ticker-as-name |
| ✅ E | Trailed stops | 10 of 11 positions trailed above entry (32.2 trail fired); SNDK +45%, MU +45% locked |
| ✅ F | Risk Judge wiring | FACT_LEDGER block now actually renders (phase-32.3 bug fix in place); `portfolio_sector_exposure` visible |

---

## What needs operator attention

| ❌ ⚠️ | Category | Fix |
|---|---|---|
| ❌ G | Kill switch | resume before 18:00 UTC |
| ⚠️ H | Caps headroom | OPTIONAL Tech-sell to make room; otherwise no action |
| ❌ I | Scheduler + LLM | fund Anthropic OR swap to `gemini-*` before 18:00 UTC |

---

## Recommended operator command sequence (~2 minutes)

```bash
# 1. Verify backend health
curl -sS http://localhost:8000/api/health | jq .

# 2. Confirm the kill-switch state (read-only)
curl -sS http://localhost:8000/api/paper-trading/kill-switch | jq '.paused, .breach.any_breached'

# 3. Either: resume the kill switch via the dashboard (preferred)
#    Or: POST to the resume endpoint (verify the route path in backend/api/paper_trading.py first)

# 4. Decide on the LLM route:
#    Option A -- fund Anthropic at https://console.anthropic.com/settings/billing
#    Option B -- swap to Gemini:
#       echo "GEMINI_MODEL=gemini-2.5-pro" >> backend/.env
#       launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend
#       sleep 5
#       curl -sS http://localhost:8000/api/health
#       # Verify the lifespan log shows the new routing: tail -n 50 handoff/logs/backend.log | grep "phase-31.1 model routing"

# 5. After both blockers cleared, re-run the readiness check:
#    cd /Users/ford/.openclaw/workspace/pyfinagent && \
#    curl -sS http://localhost:8000/api/paper-trading/kill-switch | jq '.paused' && \
#    python -c "from backend.config.settings import Settings; print(Settings().gemini_model)"
```

---

## Roll-up

After phase-32 closed cleanly, the only things standing between the system and a clean 18:00 UTC cycle are **operator state** (a 2-day-old manual pause + an unfunded LLM API key). The code is fine: 285 tests pass, all BQ schemas migrated, all 11 positions are properly protected (trailed stops locking in 9-45% above entry depending on MFE), the Risk Judge prompt now actually receives the FACT_LEDGER (32.3 bug fix), and the dashboard COMPANY column will surface real company names within 24h via the 32.5 cache eviction.

**Two 30-second operator clicks** flip the verdict from NOT_READY to READY.
