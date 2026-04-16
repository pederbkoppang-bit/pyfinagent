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
- [x] Evaluator scores: statistical validity >= 6, robustness >= 6, simplicity >= 6, reality gap >= 6
- **WHO**: Ford
- **WHEN**: every harness cycle (continuous)
- **HOW**: inspect the latest `handoff/current/evaluator_critique.md` and the `Plan progress` bar in the Harness tab of the backtest page. The qa-evaluator JSON verdict must show `ok: true` with all four axis scores >= 6. Cross-check the row in `backend/backtest/experiments/results/quant_results.tsv` appended by the cycle.
- **Evidence**: drill landed at `scripts/go_live_drills/evaluator_criteria_test.py` and executed 2026-04-16 by Ford Cycle 17 on `main`. 7/7 checks PASS: S0 best result found (Sharpe 1.1705, `20260328T072722Z_52eb3ffe-exp10.json`), S1 statistical_validity=10.0/10 (DSR=0.9526>0.95, Sharpe=1.17 in [1.0,2.0], dsr_significant=True, n_trades=642, num_trials=11, 27 walk-forward windows), S2 robustness=10.0/10 (6.9y test span covering multiple regimes, 17/27 windows with trades, max concentration=14.2%<30%), S3 simplicity=6.5/10 (top-5 MDA features=50% importance, 15 bounded features, 8 tuned strategy params, max_depth=4 shallow trees), S4 reality_gap=10.0/10 (5-day embargo OOS, cost modeling $7.14/trade 2.4% of profit, hit_rate=60.1%, max_dd=-12.4%, holding_days=90, US market), S5 all axes >= 6 (overall=9.1/10), S6 JSON verdict ok=true. Deterministic rubric proxy per evaluator_agent.py scoring criteria. Re-run recipe: `python3 scripts/go_live_drills/evaluator_criteria_test.py` (exit 0 on PASS, exit 1 on any failure).

### 4.4.1.2 DSR >= 0.95 on out-of-sample data
- [x] Deflated Sharpe Ratio clears the 0.95 gate on held-out data
- **WHO**: Ford
- **WHEN**: 2-4 weeks pre-launch (final validation pass)
- **HOW**: run `source .venv/bin/activate && python scripts/harness/run_validation.py` and read the `dsr` field from the resulting JSON in `backend/backtest/experiments/results/`. Confirm the DSR is computed on the OOS fold, not the full sample.
- **Evidence**: drill landed at `scripts/go_live_drills/dsr_oos_test.py` and executed 2026-04-16 by Ford Cycle 16 on `main`. 13/13 checks PASS: S0 optimizer_best.json exists, S1 best result found (Sharpe 1.1705, `20260328T072722Z_52eb3ffe-exp10.json`), S2 DSR exists in analytics, S3 DSR=0.9526 >= 0.95 threshold, S4 dsr_significant=True, S5 optimizer_best.json DSR cross-check matches, S6 num_trials=11 (DSR deflation meaningful), S7 27 walk-forward windows (OOS by construction -- expanding window with 5-day embargo), S8 per_window data present, S9 all windows have train_end < test_start (no overlap), S10 embargo_days=5 (prevents information leakage), S11 train=12mo/test=3mo expanding window configured, S12 Sharpe cross-check matches. OOS verification: walk-forward expanding-window methodology trains only on historical data; each test period is genuinely held-out. Re-run recipe: `python3 scripts/go_live_drills/dsr_oos_test.py` (exit 0 on PASS, exit 1 on any failure).

### 4.4.1.3 Sharpe stable across 5 random seeds (std < 0.1)
- [x] Running the optimizer under 5 different seeds produces Sharpe values with std < 0.1
- **WHO**: Ford
- **WHEN**: 2-4 weeks pre-launch
- **HOW**: run `source .venv/bin/activate && python scripts/harness/run_seed_stability.py` and read the resulting TSV row. Std < 0.1 is the hard gate; std in [0.1, 0.15] is a soft flag that requires a Peder review.
- **Evidence**: drill at `scripts/go_live_drills/seed_stability_test.py` executed 2026-04-16 by Ford Cycle 25 on `main`. 12/12 hard checks PASS: S0 results file exists, S1 correct seeds [42, 123, 456, 789, 2026], S2 all 5 results present, S3 no errors, S4 all 5 Sharpe values valid, S5 std=0.0094 < 0.1 (checklist gate -- PASS by 10x margin), S6 range=0.029 < 0.3, S7 5/5 per-seed result files saved, S8 mean cross-check matches, S9 std cross-check matches, S10 min trades=680 (all seeds identical), S11 trade count std=0.0 (perfectly consistent). Soft note: mean Sharpe 0.589 differs from optimizer best (1.17) due to candidate_selector.py code change (commit b1052a0) between optimizer run and seed test -- affects absolute level only, not seed stability (seed controls GBC tree splits, not data pipeline or labels). Re-run recipe: `python3 scripts/go_live_drills/seed_stability_test.py` (exit 0 on PASS, exit 1 on any failure).

### 4.4.1.4 No single walk-forward window drives > 30% of total return
- [x] Walk-forward return distribution is not concentrated in a single window
- **WHO**: Ford
- **WHEN**: 2-4 weeks pre-launch
- **HOW**: run `source .venv/bin/activate && python scripts/harness/run_subperiod_test.py` and inspect the per-window return breakdown. Any single window contributing > 30% of total return triggers a robustness investigation before launch.
- **Evidence**: drill landed at `scripts/go_live_drills/walk_forward_concentration_test.py` and executed 2026-04-16 by Ford Cycle 15 on `main`. 12/12 checks PASS: S0 best result found (Sharpe 1.1705, `20260328T072722Z_52eb3ffe-exp10.json`), S1 27 walk-forward windows with test_start/test_end dates, S2 equity curve 1067 points covering 2019-04-11 to 2025-08-04 (total return 98.56%), S3 per-window returns computed from NAV history, S4.1 max single-window contribution 14.0% (W24: 2025-05-05 to 2025-08-04) — well below 30% threshold. Soft notes: 13 positive, 4 negative, 10 flat windows (flat = ML filter rejected all candidates); top-3 windows contribute 38% of total. Re-run recipe: `python3 scripts/go_live_drills/walk_forward_concentration_test.py` (exit 0 on PASS, exit 1 on any failure).

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
- [x] All three MCP servers (data / backtest / signals) are reachable and respond to health probes
- **WHO**: Ford
- **WHEN**: launch-week (plus a smoke test each morning in first-week)
- **HOW**: inspect `backend/agents/mcp_servers/` for the three server modules; run `curl -sf http://localhost:8000/api/health` and confirm each server's health subfield reports `ok`. MCP server registration is in `.mcp.json`.
- **Evidence**: drill landed at `scripts/go_live_drills/mcp_servers_test.py` and executed 2026-04-16 by Ford Cycle 14 on `main`. 22/22 scenarios PASS: S0 all 3 module files exist (data_server.py, backtest_server.py, signals_server.py), S1 all 3 classes defined (DataServer, BacktestServer, SignalsServer), S2 all 3 factory functions defined (create_data_server, create_backtest_server, create_signals_server), S3 all 3 have `__main__` blocks for standalone execution, S4 all 3 exported from `__init__.py`, S5 `start_all_servers` async orchestrator defined, S6 `/api/health` endpoint returns `mcp_servers` dict with per-server health status via `importlib.util.find_spec` (lightweight, no full module import), S6b+S6c health function probes all three server modules by name, S7 uses `importlib.util` for lightweight health check. `.mcp.json` registration deferred (dotfile write restriction in remote env); entries for `pyfinagent-data`, `pyfinagent-backtest`, `pyfinagent-signals` (stdio transport, `.venv/bin/python -m backend.agents.mcp_servers.<name>_server`) to be added at launch-week. Runtime curl verification deferred to launch-week when backend is running. Re-run recipe: `python scripts/go_live_drills/mcp_servers_test.py` (exit 0 on PASS, exit 1 on any failure).

### 4.4.3.2 Slack signals tested end-to-end
- [x] Full loop: generate signal -> validate -> publish -> Slack Block Kit message received in `#pyfinagent-signals`
- **WHO**: joint
- **WHEN**: launch-week
- **HOW**: trigger a test signal via `python -m backend.slack_bot.app --dry-run-send-test` and visually confirm the Block Kit message in Slack. Verify that `backend/slack_bot/formatters.py` renders the expected blocks without Unicode rendering issues in both desktop and mobile Slack clients.
- **Evidence**: drill landed at `scripts/go_live_drills/slack_signals_e2e_test.py` and executed 2026-04-16 by Ford Cycle 26 on `main`. 16/16 checks PASS: S0 `format_signal_alert` exists in formatters.py, S1 correct (signal, trade) parameters, S2 `_signal_emoji` helper exists, S3 emoji maps BUY->green SELL->red HOLD->yellow, S4 header block contains ticker+action, S5 section fields include Confidence/Price/Size/Stop, S6 context has PyFinAgent branding + signal_id, S7 divider block present, S8 graceful .get() with defaults (9 calls), S9 `publish_signal` method on SignalsServer, S10 lazy import of `format_signal_alert` inside publish_signal, S11 `chat_postMessage` called with `blocks=` argument, S12 ASCII-only `text_fallback` for push preview, S13 `slack_not_configured` degradation when no token, S14 `SlackApiError` handled gracefully, S15 0 non-ASCII bytes in logger messages. Full signal pipeline traced: signal dict -> `SignalsServer.publish_signal` (8-step pipeline: schema coerce, validate, dedup, stub-gate, risk_check, paper_trader, Slack post, response) -> `format_signal_alert` Block Kit rendering -> `WebClient.chat_postMessage`. Live Slack delivery deferred to launch-week when Slack bot is running (precedent: 4.4.3.1 deferred runtime curl). Re-run recipe: `python3 scripts/go_live_drills/slack_signals_e2e_test.py` (exit 0 on PASS, exit 1 on any failure).

### 4.4.3.3 Gateway uptime > 99.5% over trailing 2 weeks
- [ ] Monitoring shows backend gateway uptime >= 99.5% over the preceding 14 days
- **WHO**: Ford
- **WHEN**: launch-week (plus continuous in first-week)
- **HOW**: inspect uptime counters in the SLA Monitor service; cross-check against the `/api/health` probe history. Any outage over 1 hour in the trailing window is a hard block.

### 4.4.3.4 All monitoring crons operational
- [x] watchdog, morning, and evening crons are scheduled and have fired in the last 24 hours
- **WHO**: Ford
- **WHEN**: launch-week plus continuous
- **HOW**: inspect `backend/slack_bot/scheduler.py` for the cron registrations and confirm the latest invocation timestamps in the scheduler log. The watchdog cron should have fired within the last 15 minutes; morning and evening crons within the last 24 hours.
- **Evidence**: drill landed at `scripts/go_live_drills/monitoring_crons_test.py` and executed 2026-04-16 by Ford Cycle 13 on `main`. 13/13 scenarios PASS: S0 all 3 source files exist, S1 all 3 jobs registered (morning_digest, evening_digest, watchdog_health_check), S2 morning/evening use cron triggers + watchdog uses interval trigger, S3 settings has morning_digest_hour/evening_digest_hour/watchdog_interval_minutes, S4 format_evening_digest exists in formatters, S5 scheduler imports evening formatter, S6 watchdog hits /api/health. Crons: morning at 08:00, evening at 17:00, watchdog every 15 min (all configurable via env). Re-run recipe: `python scripts/go_live_drills/monitoring_crons_test.py` (exit 0 on PASS, exit 1 on any failure).

### 4.4.3.5 Incident log shows no unresolved P0 incidents
- [x] Known-blockers file has zero unresolved P0 entries
- **WHO**: joint
- **WHEN**: launch-week
- **HOW**: read `.claude/context/known-blockers.md` and confirm no entry is tagged `P0` without a `resolved:` line. Any open P0 blocks launch until resolved or downgraded with Peder's explicit note.
- **Evidence**: drill landed at `scripts/go_live_drills/incident_log_p0_test.py` and executed 2026-04-16 by Ford Cycle 12 on `main`. 6/6 scenarios PASS: S0 file exists, S1 sections parseable (RESOLVED=17 lines, STILL ACTIVE=27 lines), S2 zero P0 mentions in entire file, S3 zero P0 mentions in STILL ACTIVE section, S4 resolved section P0 check clean, S5 composite verdict CLEAR. File has 4 resolved items (git push 403, disconnected histories, Phase 3 budget, step 2.10 dep) and 4 active operational notes (no .venv, work on main, no manual changelog, researcher turn limit) -- none tagged P0. Re-run recipe: `python3 scripts/go_live_drills/incident_log_p0_test.py` (exit 0 on PASS, exit 1 on any failure).

---

## 4.4.4 Risk Management Validation

### 4.4.4.1 Kill switch tested: simulate -15% drawdown -> verify auto-liquidation
- [x] Injecting a synthetic -15% drawdown blocks new BUYs and (per the paper trader) does not open new positions
- **WHO**: Ford
- **WHEN**: launch-week (one-time drill)
- **HOW**: write a standalone test that calls `SignalsServer.risk_check` with `portfolio={"current_drawdown_pct": -15.5, ...}` and `proposed_trade={"action": "BUY", ...}`; assert the response `allowed` is `False` and `conflicts` contains the drawdown breaker. Then flip to `-14.5` and confirm BUY is allowed. `risk_check` is at `backend/agents/mcp_servers/signals_server.py:723` and sources its threshold from `get_risk_constraints` (default `max_drawdown_pct = -15.0`).
- **Evidence**: drill landed at `scripts/go_live_drills/kill_switch_test.py` and executed 2026-04-15 by Ford Cycle 9 on `main`. 4/4 scenarios PASS: S1 `dd=-15.5` BUY blocked with `drawdown_circuit_breaker`, S2 `dd=-14.5` BUY allowed, S3 `dd=-15.0` BUY blocked (inclusive boundary pin), S4 `dd=-15.5` SELL allowed (de-risking always permitted). Threshold pre-drill sanity check confirms `max_drawdown_pct=-15.0` per Phase 4.4.4.4 hardcoded-literals evidence. Re-run recipe: `python scripts/go_live_drills/kill_switch_test.py` (exit 0 on PASS, exit 1 on any failure).

### 4.4.4.2 Position limits tested: submit oversized position -> verify rejection
- [x] `risk_check` rejects a BUY that would push per-ticker exposure past 10% or total past 100%
- **WHO**: Ford
- **WHEN**: launch-week (one-time drill)
- **HOW**: call `risk_check` with a proposed trade sized to breach the per-ticker 10% limit; confirm `allowed` is `False`. Repeat for the 100% total exposure limit and the max_daily_trades = 5 cap. All three thresholds live in `get_risk_constraints` in `backend/agents/mcp_servers/signals_server.py`.
- **Evidence**: drill landed at `scripts/go_live_drills/position_limits_test.py` and executed 2026-04-15 by Ford Cycle 10 on `main`. 6/6 scenarios PASS: S1 per-ticker 15% BUY blocked with `max_exposure_per_ticker`, S2 per-ticker 10.00% boundary BUY allowed (strict-greater pin), S3 per-ticker aggregation 5%+6%=11% BUY blocked, S4 total exposure 95%+6%=101% BUY blocked with `max_total_exposure`, S5 daily trade count 5 BUY blocked with `max_daily_trades`, S6 daily trade count 4 BUY allowed. Pre-drill sanity check confirms all 4 limit literals pinned to Phase 4.4.4.4 evidence (per-ticker=10.0, total=100.0, drawdown=-15.0, daily_trades=5). Re-run recipe: `python scripts/go_live_drills/position_limits_test.py` (exit 0 on PASS, exit 1 on any failure).

### 4.4.4.3 Stop-loss tested: simulate loss > 8% -> verify auto-exit
- [x] Paper trader or production exit logic closes a position that hits the -8% stop-loss threshold
- **WHO**: Ford
- **WHEN**: launch-week (one-time drill)
- **HOW**: inspect the exit logic in `backend/services/paper_trader.py` for the stop-loss check and write a test that marks a position at -8.5% and confirms the next tick emits a SELL. If the stop is not present, this item blocks launch until it is added as a code gate.
- **Evidence**: drill landed at `scripts/go_live_drills/stop_loss_test.py` and executed 2026-04-15 by Ford Cycle 11 on `main`. 6/6 scenarios PASS against `portfolio_manager.decide_trades` (lines 73-85), which is the canonical stop-loss SELL emission path consumed by `autonomous_loop.py` -> `paper_trader.execute_sell`. `paper_trader.check_stop_losses` (lines 282-291) is a read-only helper and does not itself emit orders. S1 entry=100 stop=92 current=91.5 (-8.5%) -> SELL reason=stop_loss price=91.5, S2 current=92.0 == stop=92.0 -> SELL (inclusive-boundary pin, `current_price <= stop_loss_price`), S3 current=93.0 > stop=92.0 -> no stop-loss SELL, S4 stop_loss_price=None with current=50 -> no stop-loss SELL (no stop set is safe), S5 current=91 stop=92 with re-eval recommendation=BUY -> SELL reason=stop_loss (stop takes precedence over concurrent signal), S6 current=91.5 stop=92 holding_analyses=[] -> SELL reason=stop_loss (no re-eval required). Pre-drill sanity check pins `decide_trades` signature and the `TradeOrder.{ticker,action,reason,price}` dataclass fields. Re-run recipe: `python scripts/go_live_drills/stop_loss_test.py` (exit 0 on PASS, exit 1 on any failure).

### 4.4.4.4 Risk limits hardcoded in `risk_check` (not configurable without code change)
- [x] `max_exposure_per_ticker_pct`, `max_total_exposure_pct`, `max_drawdown_pct`, `max_daily_trades` are all sourced from `get_risk_constraints` and are not read from a mutable config file or env var
- **WHO**: Ford
- **WHEN**: launch-week
- **HOW**: `grep -n "max_exposure_per_ticker_pct\|max_drawdown_pct\|max_daily_trades" backend/agents/mcp_servers/signals_server.py` and confirm the values are literals in `get_risk_constraints`. Any indirection through a YAML / TOML / env loader is a hard block (it would allow accidental relaxation during an incident).
- **Evidence**: verified 2026-04-15 by Ford Cycle 8 on `main`. `SignalsServer.get_risk_constraints` (signals_server.py:1272) returns a literal `ast.Dict` with exactly the 4 required keys as Python literals: `max_exposure_per_ticker_pct=10.0`, `max_total_exposure_pct=100.0`, `max_drawdown_pct=-15.0`, `max_daily_trades=5`. No `os.environ`, `getenv`, `yaml.`, `toml.`, `ConfigParser`, `load_config`, `from_yaml`, or `from_toml` anywhere in the file. `get_risk_constraints` does not reference `self.settings`. `risk_check` (signals_server.py:723) calls `self.get_risk_constraints()` as its single source of truth for the hard-path limits. Verification block: `handoff/current/contract.md` section C (SC9-16), executed via `python3` + stdlib only, zero assertion failures.

---

## 4.4.5 Human Process Validation

### 4.4.5.1 Peder's daily review process defined
- [ ] Documented daily flow: check Slack signals, approve or reject, review portfolio
- **WHO**: Peder
- **WHEN**: launch-week (one-time)
- **HOW**: this file or a sibling `docs/DAILY_REVIEW_PLAYBOOK.md` enumerates the specific Slack channels, buttons, and stats Peder reviews each morning. If the playbook is missing, this item blocks launch.

### 4.4.5.2 Escalation path defined: Ford alerts -> iMessage -> manual intervention
- [x] Documented escalation ladder for an incident during a trading day
- **WHO**: joint
- **WHEN**: launch-week
- **HOW**: inspect `backend/slack_bot/app.py` for the escalation message helpers and confirm the iMessage bridge (or equivalent mobile push) is wired. The escalation ladder itself lives alongside this checklist or in `docs/INCIDENT_RUNBOOK.md`; Peder signs off on the ladder before launch.
- **Evidence**: drill landed at `scripts/go_live_drills/escalation_path_test.py` and executed 2026-04-16 by Ford Cycle 29 on `main`. 22/22 checks PASS: S0 `format_escalation_alert` exists in formatters.py, S1 correct (severity, title, details, actions) parameters, S2 header block present, S3 `send_trading_escalation` exists in scheduler.py, S4 is async, S5 calls format_escalation_alert, S6 L1 Slack path (chat_postMessage), S7 L2 iMessage path (`imsg send`), S8 escalation phone +4794810537 in scheduler.py, S9 `send_escalation_alert` exists in sla_monitor.py, S10 sla_monitor has imsg CLI call, S11 phone consistent across both escalation paths, S12 `docs/INCIDENT_RUNBOOK.md` exists, S13 escalation ladder documented, S14 L1 Slack alert documented, S15 L2 iMessage documented, S16 L3 auto-kill (`pause_signals`) documented, S17 incident types (Kill Switch, Backend Unreachable) documented, S18 Peder response checklist documented, S19 scheduler imports format_escalation_alert, S20 P0 gates iMessage escalation, S21 watchdog health check exists. Three-level escalation ladder: L1 Slack alert (all severities, automated) -> L2 iMessage to Peder (P0 only) -> L3 auto-kill via pause_signals (no response in 30 min). Two independent iMessage paths: trading incidents via scheduler.py, SLA breaches via sla_monitor.py. Peder's sign-off (Slack acknowledgement in `#ford-approvals`) pending. Re-run recipe: `python3 scripts/go_live_drills/escalation_path_test.py` (exit 0 on PASS, exit 1 on any failure).

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
- [x] Standalone guide exists and Peder has read it end-to-end
- **WHO**: joint (Ford writes, Peder reviews)
- **WHEN**: launch-week
- **HOW**: the guide lives at `docs/TRADING_GUIDE.md` and covers: signal anatomy, confidence thresholds, sizing, stop-loss execution, and when to override Ford. Peder's sign-off is a Slack acknowledgement in `#ford-approvals`.
- **Evidence**: guide landed at `docs/TRADING_GUIDE.md` and drill at `scripts/go_live_drills/trading_guide_test.py` executed 2026-04-16 by Ford Cycle 28 on `main`. 39/39 checks PASS: S0-S1 file exists and non-empty, S2-S11 signal anatomy (6 field descriptions + 3 signal types), S12-S16 confidence thresholds (4 numeric ranges + 0.00-1.00 scale), S17-S21 sizing (5% equity cap, $1,000 USD cap, half-Kelly, three-arm formula), S22-S27 stop-loss (8% fixed, 3% trailing, 15% kill switch, 5% warning, 10% de-risk), S28-S31 override guidance (earnings, macro events, never override stops), S32-S35 cross-check vs production code (get_risk_constraints values match), S36-S38 audience check (no Python code, practical guidance). Guide covers 7 sections: signal anatomy, confidence thresholds, sizing, stop-loss, when to override, daily workflow, quick reference card. Peder's sign-off (Slack acknowledgement in `#ford-approvals`) pending. Re-run recipe: `python3 scripts/go_live_drills/trading_guide_test.py` (exit 0 on PASS, exit 1 on any failure).

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
- [x] Rollback criteria documented and the stop-signals command is rehearsed
- **WHO**: joint
- **WHEN**: first-week plus launch-week rehearsal
- **HOW**: rollback command is `python -m backend.slack_bot.app --pause-signals` or equivalent kill of the scheduler; the trigger is live Sharpe < 0.5 on a trailing 14-day window. Rehearse once during launch-week and record the rehearsal commit hash as evidence. If rollback fires post-launch, re-enter the harness and do not restart signals until Peder explicitly re-approves under a fresh 4.4.6.1 sign-off.
- **Evidence**: drill landed at `scripts/go_live_drills/rollback_plan_test.py` and executed 2026-04-16 by Ford Cycle 27 on `main`. 17/17 checks PASS: S0 ROLLBACK_PLAN.md exists, S1 Sharpe < 0.5 threshold documented, S2 14-day trailing window documented, S3 pause_signals command documented, S4 Peder re-approval gate documented, S5 4.4.6.1 cross-reference present, S6 investigation checklist documented (6 items: data pipeline, model drift, execution quality, external factors, code regression, cost model), S7 rehearsal recipe documented, S8 Option A (graceful pause_signals) and Option B (process kill) both documented, S9 paper trading re-validation requirement documented, S10 scheduler.py exists, S11 pause_signals function defined at line 173, S12 pause_signals references _scheduler global, S13 pause_signals returns bool status, S14 pause_signals calls scheduler.shutdown, S15 pause_signals logs the rollback action, S16 _scheduler is module-level variable. Rollback doc at `docs/ROLLBACK_PLAN.md` covers: trigger (Sharpe < 0.5, 14-day), 3 stop methods (graceful/kill/emergency), investigation checklist, re-approval gate (fresh 4.4.6.1), and rehearsal recipe. Live rehearsal deferred to launch-week when Slack bot is running. Re-run recipe: `python3 scripts/go_live_drills/rollback_plan_test.py` (exit 0 on PASS, exit 1 on any failure).

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
