# Sprint Contract -- MAS Harness Cycle 27
Generated: 2026-04-16

## Target
Phase 4.4.5.5: Documentation -- "How to trade pyfinAgent signals" guide for Peder

## Hypothesis
Peder needs a standalone operational guide at `docs/TRADING_GUIDE.md` that covers
the complete signal anatomy, confidence interpretation, position sizing, stop-loss
execution, and override scenarios. Without this guide, 4.4.5.5 blocks launch.
The guide must be accurate to the current codebase (commit `0970bd4` on main)
and written for a human trader, not a developer.

## Research Gate
WAIVED for external research. Internal codebase research completed: full read of
`signals_server.py` (publish_signal, size_position, risk_check, check_stop_loss,
track_drawdown, get_risk_constraints), `portfolio_manager.py` (decide_trades,
stop-loss path), `paper_trader.py` (execute_buy/sell), `autonomous_loop.py`
(daily cycle), `formatters.py` (format_signal_alert). All thresholds, formulas,
and flow steps documented in research notes. No academic papers needed for a
pure operational playbook.

## Success Criteria

### A. File structure (SC1-5)
- SC1: `docs/TRADING_GUIDE.md` exists and is a valid markdown file
- SC2: File has >= 6 top-level `##` sections covering all 5 required topics:
  signal anatomy, confidence thresholds, sizing, stop-loss execution, when to override
- SC3: File length >= 150 lines and <= 500 lines (concise but complete)
- SC4: File is 100% ASCII (no Unicode -- security.md rule applies to docs)
- SC5: No code blocks import non-stdlib modules (guide is human-readable, not executable)

### B. Signal anatomy section (SC6-10)
- SC6: Lists all 6 core signal fields: ticker, signal (BUY/SELL/HOLD), confidence,
  date, factors, reason
- SC7: Describes the Slack Block Kit display format: header (emoji + ticker + action),
  fields (Confidence / Price / Size / Stop), thesis section, footer (branding + signal_id)
- SC8: Explains signal_id as the dedup key (sha1[:16]) and that duplicates are suppressed
- SC9: Documents the three signal types: BUY, SELL, HOLD with their meaning
- SC10: Notes that HOLD signals post to Slack but do NOT trigger trades

### C. Confidence and thresholds section (SC11-14)
- SC11: Explains confidence is a continuous 0.0-1.0 float, not a categorical bin
- SC12: Documents the recommendation-to-action mapping: BUY/STRONG_BUY trigger buys,
  SELL/STRONG_SELL trigger sells, HOLD/downgrades trigger position reviews
- SC13: Notes that confidence feeds into the half-Kelly sizing formula
- SC14: Mentions the risk judge's `recommended_position_pct` as a secondary sizing input

### D. Position sizing section (SC15-19)
- SC15: Documents the hybrid sizing formula: min(hard_cap, kelly_arm, vol_arm)
- SC16: Lists the hard cap defaults: max_position_pct=5.0%, max_position_usd=$1,000
- SC17: Explains half-Kelly arm: 0.5 * confidence * equity
- SC18: Explains inverse-vol arm: (target_vol / annualized_vol) * equity
- SC19: Documents risk constraints: max_exposure_per_ticker=10%, max_total_exposure=100%,
  max_daily_trades=5

### E. Stop-loss section (SC20-24)
- SC20: Documents fixed stop-loss at 8% below entry price (O'Neil CAN SLIM)
- SC21: Documents trailing stop at 3% below peak price (Chandelier-lite)
- SC22: Explains the drawdown circuit breaker tiers: ok (>-5%), warning (-5%),
  derisk (-10%, halve sizes), kill (-15%, full liquidation)
- SC23: Notes that the -15% drawdown breaker blocks new BUYs but allows SELLs
- SC24: Documents that stop-loss SELL takes precedence over concurrent BUY re-evaluation signals

### F. Override and manual intervention section (SC25-28)
- SC25: Documents when Peder should NOT follow a signal (e.g., earnings day, known corporate event)
- SC26: Describes how to interpret low-confidence signals (< 0.5) vs high-confidence (> 0.8)
- SC27: Documents the kill switch: if live Sharpe < 0.5 over trailing 14 days, stop signals
- SC28: Notes that Peder is the final decision-maker -- Ford generates, Peder trades

### G. Checklist evidence (SC29-31)
- SC29: `docs/GO_LIVE_CHECKLIST.md` item 4.4.5.5 flipped from `[ ]` to `[x]`
- SC30: Evidence line appended under 4.4.5.5 citing the guide path + cycle + commit
- SC31: No other checklist items modified (4.4.5.1-4.4.5.4 and 4.4.6.* unchanged)

### H. Scope discipline (SC32-34)
- SC32: Exactly 2 files touched: `docs/TRADING_GUIDE.md` (new) and `docs/GO_LIVE_CHECKLIST.md` (edit)
- SC33: Zero `.py` files touched. Zero `backend/` files touched.
- SC34: All sibling docs (`ARCHITECTURE.md`, `UX-AGENTS.md`, `RESEARCH_3.3.1.md`) byte-identical

## Adversarial Probes (for QA evaluator)
- AP1: Guide mentions all 7 risk constraint keys from get_risk_constraints
- AP2: Stop-loss percentages match code literals (8.0% fixed, 3.0% trail, -15.0% drawdown)
- AP3: Sizing formula matches code: min(hard_cap, kelly, vol) not max or average
- AP4: Signal field list matches the Signal dataclass (6 fields, not more/fewer)
- AP5: Guide does NOT instruct Peder to modify code, config files, or env vars
- AP6: Guide does NOT contain Python import statements (human-readable, not executable)
- AP7: Daily trade limit matches code: 5 (not 10 or unlimited)
- AP8: Confidence range is [0.0, 1.0] not [0, 100] or [-1, 1]
- AP9: Kill switch threshold matches 4.4.6.4: Sharpe < 0.5 over 14 days
- AP10: Guide explicitly states Peder is the decision-maker, not an instruction-follower

## Excluded
- Peder's sign-off on the guide (that's his part of the "joint" item; Ford writes, Peder reviews)
- `docs/DAILY_REVIEW_PLAYBOOK.md` (4.4.5.1, Peder-owned)
- `docs/INCIDENT_RUNBOOK.md` (4.4.5.2, separate cycle)
- `docs/MANUAL_TRADING_PLAYBOOK.md` (4.4.5.4, Peder-owned)
- Modifications to any backend code
