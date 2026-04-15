# pyfinAgent Go-Live Checklist

> Operational tracker for Phase 4.4. Every item must be verified before the May launch (Slack signals -> manual trading).
> Source of truth: `PLAN.md` section 4.4 (lines 1172-1214). This document is the executable counterpart with owner, timing, and verification recipe per item.

## How to Use This Checklist

- **WHO** -- primary owner of the item. One of `{Peder, Ford, joint}`. `Peder` is the human decision-maker and account owner. `Ford` is the autonomous remote agent / harness. `joint` means both are required.
- **WHEN** -- when the item must be checked, expressed as a timing window (`continuous` / `every harness cycle` / `launch-week` / `first-week` / `2-4 weeks pre-launch` / `one-time`).
- **HOW** -- a concrete verification recipe. Either a shell command, a file + symbol to inspect, a named artifact under `backend/backtest/experiments/results/`, or a named manual process.
- **Evidence** -- when an item is marked `[x]`, append a one-line evidence note (commit hash, artifact path, Slack message link, or date of manual verification).

## Critical Path

1. Complete 4.4.1 Statistical Validation first -- backtest evidence is the cheapest pre-gate.
2. Run 4.4.2 Paper Trading Validation for at least 2 weeks (ideally 4). This is a wall-clock gate that cannot be accelerated.
3. 4.4.3 Infrastructure Validation and 4.4.4 Risk Management Validation can run in parallel with 4.4.2.
4. 4.4.5 Human Process Validation must be complete before 4.4.6 Final Sign-Off.
5. 4.4.6 gates the actual go-live decision -- no signal flow to production until all items are `[x]`.

---

## 4.4.1 Statistical Validation

### 4.4.1.1 All evaluator criteria passing
- [ ] Evaluator scores: statistical validity >= 6, robustness >= 6, simplicity >= 6, reality gap >= 6
- **WHO**: Ford
- **WHEN**: every harness cycle (continuous)
- **HOW**: inspect the latest `handoff/current/evaluator_critique.md` and the `Plan progress` bar in the Harness tab of the backtest page. The qa-evaluator JSON verdict must show `ok: true` with all four axis scores >= 6. Cross-check the row in `backend/backtest/experiments/results/quant_results.tsv` appended by the cycle.

### 4.4.1.2 DSR >= 0.95 on out-of-sample data
- [ ] Deflated Sharpe Ratio clears the 0.95 gate on held-out data
- **WHO**: Ford
- **WHEN**: 2-4 weeks pre-launch (final validation pass)
- **HOW**: run `source .venv/bin/activate && python scripts/harness/run_validation.py` and read the `dsr` field from the resulting JSON in `backend/backtest/experiments/results/`. Confirm the DSR is computed on the OOS fold, not the full sample.

### 4.4.1.3 Sharpe stable across 5 random seeds (std < 0.1)
- [ ] Running the optimizer under 5 different seeds produces Sharpe values with std < 0.1
- **WHO**: Ford
- **WHEN**: 2-4 weeks pre-launch
- **HOW**: run `source .venv/bin/activate && python scripts/harness/run_seed_stability.py` and read the resulting TSV row. Std < 0.1 is the hard gate; std in [0.1, 0.15] is a soft flag that requires a Peder review.

### 4.4.1.4 No single walk-forward window drives > 30% of total return
- [ ] Walk-forward return distribution is not concentrated in a single window
- **WHO**: Ford
- **WHEN**: 2-4 weeks pre-launch
- **HOW**: run `source .venv/bin/activate && python scripts/harness/run_subperiod_test.py` and inspect the per-window return breakdown. Any single window contributing > 30% of total return triggers a robustness investigation before launch.

---

## 4.4.2 Paper Trading Validation

### 4.4.2.1 Paper trading ran for >= 2 weeks (ideally 4)
- [ ] Continuous paper trading run time on the latest parameter set reaches the 2-week wall-clock floor
- **WHO**: joint (Ford runs, Peder checks calendar)
- **WHEN**: 2-4 weeks pre-launch (wall-clock gate, cannot be shortened)
- **HOW**: query BigQuery `paper_snapshots` for the earliest and latest snapshot timestamps for the current parameter cohort; delta must be >= 14 days. Alternatively inspect the Paper Trading tab on the frontend for the run-start banner date.

### 4.4.2.2 Paper Sharpe >= 0.82 (70% of backtest 1.17)
- [ ] Paper trading Sharpe clears the 70% reality-gap floor
- **WHO**: Ford
- **WHEN**: continuous during the paper trading window
- **HOW**: computed by `backend/services/paper_trader.py` and surfaced on the Paper Trading tab. Also writable via `python -m backend.services.paper_trader --sharpe`. The gate is 0.82 derived from `backtest_sharpe * 0.70`; if the backtest Sharpe changes, recompute this gate.

### 4.4.2.3 Paper max drawdown < 15% (kill switch never triggered)
- [ ] The paper trading run never crossed the -15% drawdown line
- **WHO**: Ford
- **WHEN**: continuous during the paper trading window
- **HOW**: `risk_check` in `backend/agents/mcp_servers/signals_server.py` at line 723 enforces `max_drawdown_pct = -15.0` as a hard circuit breaker on BUYs. Inspect the Paper Trading tab for the max drawdown stat; any period where the kill switch fired invalidates the run and a fresh 2-week window starts.

### 4.4.2.4 No missed trading days (signal generation reliable)
- [ ] Every US market open day in the paper trading window has a signal-generation log entry
- **WHO**: Ford
- **WHEN**: continuous
- **HOW**: query BigQuery `signals_log` with `event_kind = "publish"` grouped by day; compare the count of distinct days against the NYSE trading calendar for the window. Zero gaps is the gate.

### 4.4.2.5 Paper vs backtest divergence < 20% on key metrics
- [ ] Reality-gap check: paper metrics within 20% of backtest on Sharpe / hit rate / avg return
- **WHO**: Ford
- **WHEN**: 2-4 weeks pre-launch (final comparison)
- **HOW**: compare `backend/backtest/experiments/results/quant_results.tsv` (latest row) against the paper trading aggregated stats surfaced on the Paper Trading tab. Any axis diverging >= 20% is a hard block; retune before launch.

---

## 4.4.3 Infrastructure Validation

### 4.4.3.1 MCP servers deployed and authenticated
- [ ] All three MCP servers (data / backtest / signals) are reachable and respond to health probes
- **WHO**: Ford
- **WHEN**: launch-week (plus a smoke test each morning in first-week)
- **HOW**: inspect `backend/agents/mcp_servers/` for the three server modules; run `curl -sf http://localhost:8000/api/health` and confirm each server's health subfield reports `ok`. MCP server registration is in `.mcp.json`.

### 4.4.3.2 Slack signals tested end-to-end
- [ ] Full loop: generate signal -> validate -> publish -> Slack Block Kit message received in `#pyfinagent-signals`
- **WHO**: joint
- **WHEN**: launch-week
- **HOW**: trigger a test signal via `python -m backend.slack_bot.app --dry-run-send-test` and visually confirm the Block Kit message in Slack. Verify that `backend/slack_bot/formatters.py` renders the expected blocks without Unicode rendering issues in both desktop and mobile Slack clients.

### 4.4.3.3 Gateway uptime > 99.5% over trailing 2 weeks
- [ ] Monitoring shows backend gateway uptime >= 99.5% over the preceding 14 days
- **WHO**: Ford
- **WHEN**: launch-week (plus continuous in first-week)
- **HOW**: inspect uptime counters in the SLA Monitor service; cross-check against the `/api/health` probe history. Any outage over 1 hour in the trailing window is a hard block.

### 4.4.3.4 All monitoring crons operational
- [ ] watchdog, morning, and evening crons are scheduled and have fired in the last 24 hours
- **WHO**: Ford
- **WHEN**: launch-week plus continuous
- **HOW**: inspect `backend/slack_bot/scheduler.py` for the cron registrations and confirm the latest invocation timestamps in the scheduler log. The watchdog cron should have fired within the last 15 minutes; morning and evening crons within the last 24 hours.

### 4.4.3.5 Incident log shows no unresolved P0 incidents
- [ ] Known-blockers file has zero unresolved P0 entries
- **WHO**: joint
- **WHEN**: launch-week
- **HOW**: read `.claude/context/known-blockers.md` and confirm no entry is tagged `P0` without a `resolved:` line. Any open P0 blocks launch until resolved or downgraded with Peder's explicit note.

---

## 4.4.4 Risk Management Validation

### 4.4.4.1 Kill switch tested: simulate -15% drawdown -> verify auto-liquidation
- [ ] Injecting a synthetic -15% drawdown blocks new BUYs and (per the paper trader) does not open new positions
- **WHO**: Ford
- **WHEN**: launch-week (one-time drill)
- **HOW**: write a standalone test that calls `SignalsServer.risk_check` with `portfolio={"current_drawdown_pct": -15.5, ...}` and `proposed_trade={"action": "BUY", ...}`; assert the response `allowed` is `False` and `conflicts` contains the drawdown breaker. Then flip to `-14.5` and confirm BUY is allowed. `risk_check` is at `backend/agents/mcp_servers/signals_server.py:723` and sources its threshold from `get_risk_constraints` (default `max_drawdown_pct = -15.0`).

### 4.4.4.2 Position limits tested: submit oversized position -> verify rejection
- [ ] `risk_check` rejects a BUY that would push per-ticker exposure past 10% or total past 100%
- **WHO**: Ford
- **WHEN**: launch-week (one-time drill)
- **HOW**: call `risk_check` with a proposed trade sized to breach the per-ticker 10% limit; confirm `allowed` is `False`. Repeat for the 100% total exposure limit and the max_daily_trades = 5 cap. All three thresholds live in `get_risk_constraints` in `backend/agents/mcp_servers/signals_server.py`.

### 4.4.4.3 Stop-loss tested: simulate loss > 8% -> verify auto-exit
- [ ] Paper trader or production exit logic closes a position that hits the -8% stop-loss threshold
- **WHO**: Ford
- **WHEN**: launch-week (one-time drill)
- **HOW**: inspect the exit logic in `backend/services/paper_trader.py` for the stop-loss check and write a test that marks a position at -8.5% and confirms the next tick emits a SELL. If the stop is not present, this item blocks launch until it is added as a code gate.

### 4.4.4.4 Risk limits hardcoded in `risk_check` (not configurable without code change)
- [ ] `max_exposure_per_ticker_pct`, `max_total_exposure_pct`, `max_drawdown_pct`, `max_daily_trades` are all sourced from `get_risk_constraints` and are not read from a mutable config file or env var
- **WHO**: Ford
- **WHEN**: launch-week
- **HOW**: `grep -n "max_exposure_per_ticker_pct\|max_drawdown_pct\|max_daily_trades" backend/agents/mcp_servers/signals_server.py` and confirm the values are literals in `get_risk_constraints`. Any indirection through a YAML / TOML / env loader is a hard block (it would allow accidental relaxation during an incident).

---

## 4.4.5 Human Process Validation

### 4.4.5.1 Peder's daily review process defined
- [ ] Documented daily flow: check Slack signals, approve or reject, review portfolio
- **WHO**: Peder
- **WHEN**: launch-week (one-time)
- **HOW**: this file or a sibling `docs/DAILY_REVIEW_PLAYBOOK.md` enumerates the specific Slack channels, buttons, and stats Peder reviews each morning. If the playbook is missing, this item blocks launch.

### 4.4.5.2 Escalation path defined: Ford alerts -> iMessage -> manual intervention
- [ ] Documented escalation ladder for an incident during a trading day
- **WHO**: joint
- **WHEN**: launch-week
- **HOW**: inspect `backend/slack_bot/app.py` for the escalation message helpers and confirm the iMessage bridge (or equivalent mobile push) is wired. The escalation ladder itself lives alongside this checklist or in `docs/INCIDENT_RUNBOOK.md`; Peder signs off on the ladder before launch.

### 4.4.5.3 Weekly review meeting
- [ ] Standing weekly slot on the calendar to review paper trading results, signal accuracy, and plan progress
- **WHO**: joint
- **WHEN**: continuous from launch-week onward
- **HOW**: calendar invite exists with Peder as organizer; the agenda reads from the Harness tab, the signal accuracy dashboard, and `handoff/harness_log.md`. First occurrence must be within 7 days of go-live.

### 4.4.5.4 Manual trading process
- [ ] Documented process for Peder to translate a Slack signal into a broker order
- **WHO**: Peder
- **WHEN**: launch-week
- **HOW**: `docs/MANUAL_TRADING_PLAYBOOK.md` or a named section in this file describes the exact broker, order type, and sizing rules. Peder executes a dry-run against a single signal before launch to confirm the playbook is accurate.

### 4.4.5.5 Documentation: "How to trade pyfinAgent signals" guide for Peder
- [ ] Standalone guide exists and Peder has read it end-to-end
- **WHO**: joint (Ford writes, Peder reviews)
- **WHEN**: launch-week
- **HOW**: the guide lives at `docs/TRADING_GUIDE.md` and covers: signal anatomy, confidence thresholds, sizing, stop-loss execution, and when to override Ford. Peder's sign-off is a Slack acknowledgement in `#ford-approvals`.

---

## 4.4.6 Final Sign-Off

### 4.4.6.1 Peder explicitly approves go-live
- [ ] Explicit go-live approval from Peder, not Ford's decision
- **WHO**: Peder
- **WHEN**: one-time at launch
- **HOW**: Peder posts a dated message in `#ford-approvals` saying "go-live approved for DATE". Ford records the commit hash and Slack message link as the evidence note on this item. This is the final human gate; all other items must already be `[x]` before this message is sent.

### 4.4.6.2 Budget approved for Phase 4 operational costs (~$268-348/month)
- [ ] Monthly budget envelope for Phase 4 has been approved
- **WHO**: Peder
- **WHEN**: one-time at launch
- **HOW**: budget figure in `PLAN.md` item 4.4.6 is `~$268-348/month`; Peder confirms the payment source (personal card or business account) and Ford records the confirmation in `handoff/harness_log.md`. Subsequent month-over-month drift > 15% triggers a budget review.

### 4.4.6.3 First week: extra monitoring (daily review call, immediate Slack alerts)
- [ ] First-week monitoring cadence is armed: daily review call scheduled, alert thresholds tightened
- **WHO**: joint
- **WHEN**: first-week (continuous for 7 days post-launch)
- **HOW**: daily review slot exists on the calendar for days 1-7; the SLA Monitor service has its alert thresholds tightened in `backend/services/` (e.g. drawdown alert at -5% instead of -10%, signal miss alert at 1 hour instead of 4). Revert to normal thresholds after day 7 if live Sharpe tracks the paper Sharpe.

### 4.4.6.4 Rollback plan: if live Sharpe < 0.5 in first 2 weeks -> stop signals, investigate
- [ ] Rollback criteria documented and the stop-signals command is rehearsed
- **WHO**: joint
- **WHEN**: first-week plus launch-week rehearsal
- **HOW**: rollback command is `python -m backend.slack_bot.app --pause-signals` or equivalent kill of the scheduler; the trigger is live Sharpe < 0.5 on a trailing 14-day window. Rehearse once during launch-week and record the rehearsal commit hash as evidence. If rollback fires post-launch, re-enter the harness and do not restart signals until Peder explicitly re-approves under a fresh 4.4.6.1 sign-off.

---

## Appendix: Item Count Verification

- 4.4.1 Statistical Validation: 4 items
- 4.4.2 Paper Trading Validation: 5 items
- 4.4.3 Infrastructure Validation: 5 items
- 4.4.4 Risk Management Validation: 4 items
- 4.4.5 Human Process Validation: 5 items
- 4.4.6 Final Sign-Off: 4 items
- **Total: 27 items**

This count matches `PLAN.md` section 4.4 (lines 1172-1214) verbatim. Prior session notes referenced "28 items" as a rounded figure; the authoritative count is 27.
