# live_check_58.1 — Go-live runway (window OPEN; step in progress)

**Step:** 58.1 (phase-58). **Status:** PRE-WINDOW REQUIREMENTS COMPLETE; the budgeted live window is running. This file accumulates the evidence the step's criteria require; the 58.1 harness cycle closes when the window produces its DoD re-scores.

## A. Operator's verbatim spend decision (criterion 1 — recorded BEFORE any live cycle of the window)

> `LLM SPEND: APPROVED $25`

- **Date:** 2026-06-11 (given by the operator in the local Claude Code session; mirrored verbatim to the Slack audit thread: #ford-approvals ts `1781146990.045889`, threaded under the decision block ts `1781111785.584429`).
- Companion reply (recorded in the phase-57 install commit message per the goal mechanics): `PHASE-57: FEATURE`.
- Budget: $25 for the 1-2 week window (measured burn: lite $0.05-0.17/cycle, full $1.08-4.06/cycle — even daily full-mode cycles for 2 weeks fit inside it). Daily circuit-breaker `cost_budget_daily_usd=$25` unchanged.
- Window framing (per the 55.3 chapter, accepted by the operator): sanity/stress gate, NOT a skill proof (MinTRL ≈539 trading days at backtest Sharpe 1.17).

## B. APPROVED-branch pre-window requirements (criterion 2)

**Phase-56 fixes deployed:** backend restarted via `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend` 2026-06-11; slack bot restarted (`python -m backend.slack_bot.app`, new PID 26147). Health verified:
```json
{"status":"ok","service":"pyfinagent-backend","version":"6.37.6", "mcp_servers":{"data":{"status":"ok"},"backtest":{"status":"ok"},"signals":{"status":"ok"}}}
```
(v6.37.6 = the post-56.2 build; the pre-restart process had written 2 extra corrupt KR rows on 06-10 18:39Z — the backfill migration was extended to 9 rows, commit `04d56c5d`, BEFORE the restart.)

**Kill switch verified ACTIVE before the window starts:**
```json
{"paused":false,"pause_reason":null,"sod_nav":23830.04,"sod_date":"2026-06-10","current_nav":23828.73,
 "breach":{"daily_loss_breached":false,"daily_loss_pct":0.0055,"daily_loss_limit_pct":4.0,
 "trailing_dd_breached":false,"trailing_dd_pct":0.0,"trailing_dd_limit_pct":10.0,"any_breached":false},
 "thresholds":{"daily_loss_limit_pct":4.0,"trailing_dd_limit_pct":10.0}}
```
Disclosure: `peak_nav` re-anchors after a process restart (trailing-DD ratchets from current NAV; the pre-restart peak 24,666.57 from 06-03 is in `handoff/kill_switch_audit.jsonl` — the trailing leg restarts its ratchet from ~23.8K). The F-9 SOD-anchor design limitation remains an open operator decision (live_check_56.2 §D).

**mas-harness cron re-enable:** NOT yet re-enabled — deliberately. It writes the same rolling handoff files this manual masterplan run uses (the reason it was booted). It will be re-enabled at this goal's HARD STOP; the criterion's "command echoed" placeholder:
```
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.pyfinagent.mas-harness.plist
```
(operator confirmation to be appended at HARD STOP).

## C. DoD-2/5/6/7/9 re-scores (criterion 2 — accumulate from the live window)

| DoD | Criterion | Baseline (2026-06-01) | Window evidence (append as it lands) |
|---|---|---|---|
| DoD-2 | backtest↔paper parity convergence | LIVE-BLOCKED | pending (parity tool fix B13/F-15 still open; partial evidence acceptable per 55.3) |
| DoD-5 | freshness → green | LIVE-BLOCKED | pending (needs the historical_macro refresh — operator-gated 53.3 item) |
| DoD-6 | ≥10 outcome_tracking rows from real sell-closes | LIVE-BLOCKED | pending (trade cadence ~2-4 sells/cycle → expect within the window) |
| DoD-7 | Risk-Judge production fallback rate | LIVE-BLOCKED | pending (now measurable: F-6 metering live as of this restart) |
| DoD-9 | 5 consecutive clean cron cycles + ≥1 non-HOLD | LIVE-BLOCKED | pending (count from 2026-06-11 onward; first post-fix cycle expected 18:00Z) |

Go-live gate baseline: 1/5 (06-01) → 2/5 (06-10 reading). Delta to be reported at step close.

## D. Spend ledger (window)

| Date | Cycle | Mode | Metered $ | Cumulative vs $25 |
|---|---|---|---|---|
| (append per cycle from llm_call_log — now metering the CLI rail too, F-6 fixed) | | | | |
