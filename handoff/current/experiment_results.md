# Experiment Results — phase-31.0 Profit-Protection + Risk-Agent Hardening Audit

**Step:** `phase-31.0` (diagnostic gap report; NO CODE EDITS).
**Date:** 2026-05-20.
**Verdict:** **GAP CONFIRMED — multiple BLOCK-severity findings.**

## Verbatim Verification Output

```
$ wc -l backend/services/portfolio_manager.py backend/services/paper_trader.py backend/services/autonomous_loop.py backend/agents/agent_definitions.py
     378 backend/services/portfolio_manager.py
     833 backend/services/paper_trader.py
    1854 backend/services/autonomous_loop.py
     427 backend/agents/agent_definitions.py
    3492 total
$ ls backend/agents/skills/ | grep -E "(risk|synthesis|quant)"
quant_model_agent.md
quant_strategy.md
risk_judge.md
risk_stance.md
synthesis_agent.md
```

**Researcher gate:** `gate_passed: true`. Brief at `handoff/current/research_brief.md` (378 lines, 22 sources read in full, 11 snippet, 33 URLs collected, recency scan performed, 3 adversarial sources, 9 internal files inspected).

**BQ probe:** all queries succeeded via `mcp__claude_ai_Google_Cloud_BigQuery__execute_sql_readonly` in `sunny-might-477607-p8` (financial_reports dataset in us-central1). 3 closed sells in last 60d, 11 current positions, 7 NO-STOP positions confirmed, 89.3% Tech concentration confirmed.

---

## Section 1 — Per-Practice Audit Table

Every row maps to one of the 8 research topics in `research_brief.md §2`. Severity scale: **BLOCK** (real-money risk in production right now), **WARN** (suboptimal vs literature consensus), **NOTE** (architectural improvement, not safety-critical).

| # | Practice | Research basis | Present? | file:line | Severity | Proposed remediation |
|---|---|---|---|---|---|---|
| 1 | **Triple-barrier EXIT (López de Prado ch.3)** — upper TP + lower SL + time barrier as exit policy | AFML ch.3 §3.3; arXiv 2504.02249v2 (April 2025) | **N — only labeling, never exit policy** | `quant_strategy.md:27-46` (labeling in backtest); no exit-side analogue anywhere in `portfolio_manager.py:46-315` | **WARN** | Wire a `take_profit_price` field into `paper_positions` + a `time_barrier_days` field; consume in `decide_trades` symmetrically with the existing lower-barrier check at `portfolio_manager.py:86-94`. Phase-31.x candidate. |
| 2a | **Trailing stop (HWM/Chandelier)** | Carver qoppac.blogspot.com 2020; arXiv 2602.11708 (+0.73 Sharpe / 9.7 pp DD reduction in 2026 ablation); Han-Zhou-Zhu 2014 (Sharpe doubled on US equity 1926-2013) | **N in live loop — Y as dead code** | `signals_server.py:1052-1154` (`check_stop_loss` with trailing variant exists, ZERO production callers) vs `paper_trader.py:484-493` (live `check_stop_losses` entry-anchored only) | **BLOCK** | Port `signals_server.check_stop_loss`'s trailing branch into `paper_trader.mark_to_market` (the place where MFE is already updated, lines 440-456); wire `autonomous_loop.py:794` to consult both. Add Kaminski-Lo adversarial guard: skip trailing for `entry_strategy == 'mean_reversion'`. (See P1.2 below.) |
| 2b | **Trailing stop ATR-scaled (Wilder, LuxAlgo)** | Wilder 1978; LuxAlgo: 2× ATR cut MaxDD 32% in 1000-trade backtest | **N — fixed 8% default everywhere** | `paper_trader.py:112-119` (`paper_default_stop_loss_pct = 8.0`), `_extract_stop_loss` in `portfolio_manager.py:371-377` | **WARN** | Replace fixed 8% with `2 × ATR(14)`; gap is already self-documented at `quant_strategy.md:33-34` ("Literature recommends `TP = daily_vol × multiplier`...Current fixed tp_pct/sl_pct work but don't adapt to volatility regimes"). |
| 3 | **Take-profit ladder / scale-out at 1R/2R/3R (Van Tharp)** | Van Tharp Institute; PnL Ledger; TraderLion R-multiples | **N — absent** | grep on `take_profit`, `scale_out`, `partial_close`, `R_multiple` returns 0 hits across `portfolio_manager.py` + `paper_trader.py` + `autonomous_loop.py` | **BLOCK** for production give-back; underlying primitive exists (`paper_trader.execute_sell` accepts `quantity` for partial closes, lines 290-419). | Add `_check_scale_out_targets(position) -> Optional[(qty_to_sell, reason)]` in `paper_trader.py`, called from `mark_to_market`. 1R = `paper_default_stop_loss_pct = 8%`, so 1R/2R/3R already computable from existing fields with zero schema additions. |
| 4 | **Profit-locking ratchet (breakeven at +1R)** | López de Prado triple-barrier + Van Tharp + Tradewink ratchet definition; ChartsWatcher: "Once trade moves up by one risk unit, switch to trailing stop" | **N — absent** | grep on `ratchet`, `breakeven`, `stop_advanced` returns 0 hits | **BLOCK** for production give-back | **P1.1** — when `mfe_pct >= paper_default_stop_loss_pct`, mutate `stop_loss_price = entry_price`. New `_advance_stop` helper in `paper_trader.mark_to_market` before the MFE/MAE write at lines 440-456. Zero schema migration; one new nullable audit field `stop_advanced_at_R`. Strict Pareto improvement over current state. |
| 5 | **Volatility-adjusted exits (Carver, AFML)** | Carver "set stop at X × volatility"; AFML ch.3 vol-adjusted barriers | **N — fixed-percent only** | `paper_trader.py:113-115`; `portfolio_manager.py:371-377`; gap **self-documented** in `quant_strategy.md:33-34` | **WARN** | Same as 2b; add `_atr_lookup(ticker, period=14)` helper, cache per-ticker per-day; fail-safe to 8% on ATR fetch failure. |
| 6 | **Meta-labeling exit classifier (López de Prado ch.3.6)** | AFML §3.6; Wikipedia ML-Quantitative Finance meta-labeling | **N — primary-only system** | `risk_judge.md:62-73` output schema has NO exit-policy block; only `risk_limits.stop_loss_pct` + `risk_limits.max_drawdown_pct` | **NOTE** (architectural, not safety-critical) | Phase-31.5 candidate — secondary classifier on `(MFE, capture_ratio, holding_days, current_drawdown_from_peak, regime)` → `{HOLD, TRIM_PARTIAL, EXIT}`. Lightweight Flash call per position per cycle. Adds LLM cost; defer until P1.1+P1.2 land. |
| 7a | **Risk-agent drawdown caps (per-position tiered ladder)** | Industry standard 5/10/15 ladder; QuantConnect MaximumDrawdownPercentPortfolio (cited at `signals_server.py:1166-1168`) | **N in live loop — Y as dead code** | `signals_server.py:1156-1243` (`track_drawdown` 5/10/15 ladder, ZERO callers from live loop); `paper_trader.py:700-733` (only binary kill switch at portfolio level — no -5% warn, no -10% derisk, no per-position tier) | **WARN** | **P2.3** — port `track_drawdown` into `check_and_enforce_kill_switch`; emit `-5% log warn`, `-10% halve sizes` (next BUY's `position_pct` × 0.5), `-15% liquidate`. The exception-frame already has `flatten_all`. |
| 7b | **Sector concentration cap (QuantAgents `max(SE_j)`)** | arXiv 2510.04643 (QuantAgents Dave's `R_score = w1·β_p + w2·(1/LR) + w3·max(SE_j) + w4·σ_p`); AQR Q1 2025; MSCI summer-2025 wobble | **partial — pre-trade hard cap exists; LLM context does NOT see exposure** | `portfolio_manager.py:209-285` (sector caps `paper_max_per_sector`, `paper_max_per_sector_nav_pct` — pre-trade gate only); FACT_LEDGER injection into `risk_judge.md` does NOT carry portfolio sector exposure | **BLOCK** (live state: 10/11 Tech = 89.3% NAV — `max(SE_j) = 0.91`, would trigger QuantAgents Risk Alert Meeting threshold 0.75) | **P1.3** — inject `current_sector_exposure: {Technology: 0.89, ...}` into `risk_judge.md`'s FACT_LEDGER section; the LLM can then argue against new Tech BUYs without code-side blocking-only rules. |
| 7c | **Correlation cap (factor/sector beyond simple sector match)** | TradingAgents arXiv 2412.20138; HedgeAgents (referenced via EmergentMind) | **N** | no symbol matches `correlation_cap`, `factor_cap`, `crowded_trade` across scoped files | **NOTE** | Phase-31.6 candidate — requires factor-exposure compute; deeper change. |
| 7d | **Kill-switch hysteresis (no flap)** | Industry standard; QuantConnect daily/trailing-DD docs | **partial — daily + trailing-DD limits exist; no auto-resume gating** | `paper_trader.py:700-733` (`check_and_enforce_kill_switch`); kill_switch state in `backend/services/kill_switch.py` (operator-driven resume) | **NOTE** | Existing design is "operator unpauses manually" — adequate hysteresis for the local-only deployment context. |
| 8a | **PM agent owns exit policy (QuantInsti/Alpaca, HedgeAgents)** | QuantInsti agentic-AI portfolio-manager tutorial; HedgeAgents (referenced via EmergentMind) | **N — exit policy is hardcoded constants** | `portfolio_manager.py:86-94` exit logic uses static `pos.get("stop_loss_price")` + LLM `recommendation` — no agent that OWNS exit decisions | **NOTE** | **P3.1** — extend `risk_judge.md` output schema with `exit_policy: {breakeven_at_R, trail_pct, scale_out_levels: [R1, R2, R3]}`; ships AFTER P1.1+P1.2 validate on live data. |
| 8b | **Exit signal distinct from entry signal in `decide_trades`** | TradingAgents framework; QuantConnect signal-vs-rule split | **partial** | `portfolio_manager.decide_trades` lines 82-119 (sells iterate `current_positions`, buys iterate `candidate_analyses`) — they ARE separated, but the sell path's only triggers are (a) stop-loss `current <= stop`, (b) `rec in _SELL_RECS`, (c) downgrade from BUY to HOLD/SELL/STRONG_SELL. NO trailing/take-profit/MFE check. | **WARN** | The structural split is already there; the gap is that the sell branch has too few inputs. P1.1 and P1.2 plug into this exact branch. |
| 8c | **MFE/MAE consulted as exit input (not just labeling)** | Han-Zhou-Zhu 2014; Carver HWM-trailing; AdaptiveTrend arXiv 2602.11708 | **N — tracked, never consulted** | `paper_trader.py:440-456` writes `mfe_pct` and `mae_pct` monotonically on every mark-to-market; `paper_trader.py:484-493` `check_stop_losses` ignores both; `autonomous_loop.py:794-813` Step 5.6 ignores both | **BLOCK** | Same as 2a + 4 (P1.1, P1.2) — the data is in BQ, the exit logic doesn't read it. |

**Audit headline:** **6 BLOCK-severity gaps** (2a, 3, 4, 7b, 8c; plus 2a/4/8c are aspects of the same root cause = MFE never consulted in exit). The architecture has **all the primitives** (MFE/MAE tracking, capture_ratio computation, the entire trailing-stop + tiered-drawdown code in `signals_server.py:1052-1243`) but **zero production wiring** into the autonomous loop's exit policy.

---

## Section 2 — Specific-Question Answers (from the goal)

**Q1: Any trailing-stop logic in the live loop, or only entry-relative static stops?**
**A:** Only entry-relative static stops. The reference implementation exists at `signals_server.py:1052-1154` (Chandelier-lite, `(current_price - peak_price) / peak_price <= -trail_stop_pct/100`, default 3%) but `grep "signals_server" backend/services/` returns no callers, and the autonomous loop's exit step (`autonomous_loop.py:794`) calls only `trader.check_stop_losses()` which is the entry-anchored variant.

**Q2: Any take-profit threshold (absolute or R-multiple)?**
**A:** **No.** `grep -RnE "take_profit|take-profit|R_multiple|partial_close|scale_out" backend/services/ backend/agents/skills/` returns 0 matches (only `take_profit` references are in backtest-engine labeling, not live exit). The Risk Judge output schema (`risk_judge.md:62-73`) has `risk_limits.stop_loss_pct` and `risk_limits.max_drawdown_pct` but no take-profit field.

**Q3: Does `risk_judge` see unrealized P&L and act on it?**
**A:** **No.** `risk_judge.md` Data Inputs (lines 22-29) are `synthesis_json`, `aggressive_arg`, `conservative_arg`, `neutral_arg`, `debate_history`, `past_memory`. Unrealized P&L, MFE, holding days, distance-to-peak are NOT in the prompt context. The Risk Judge fires at Step 12c of the analysis pipeline at ENTRY time; it never re-fires for an existing position based on the position's own dynamics.

**Q4: Does `decide_trades` consider exit signals separately from entry signals?**
**A:** **Structurally yes, substantively no.** `portfolio_manager.decide_trades` (lines 73-119) has a sell branch (iterating `current_positions`) that runs FIRST (sell-first-then-buy convention from `.claude/rules/backend-services.md`), separate from the buy branch (iterating `candidate_analyses`). But the sell branch's only triggers are (a) `current_price <= stop_loss_price`, (b) `recommendation in {"SELL","STRONG_SELL"}`, (c) downgrade from BUY → {HOLD,SELL,STRONG_SELL}. There is **no exit signal derived from position dynamics** (MFE, peak retracement, time, ATR breakout). The structural split is in place; the inputs to the sell branch are impoverished.

**Q5: Drawdown-based de-risking (per-position OR portfolio-level)?**
**A:** **Portfolio-level binary only.** `paper_trader.check_and_enforce_kill_switch` (lines 700-733) evaluates `paper_daily_loss_limit_pct` and `paper_trailing_dd_limit_pct`; on breach it calls `flatten_all`. There is no per-position drawdown ladder, no `-5% log warn`, no `-10% derisk (halve next BUY size)`, no `-15% scaling exit`. The 5/10/15 ladder code exists at `signals_server.py:1156-1243` but has zero callers (same dead-code pattern as the trailing stop).

**Q6: Scale-out logic at all (or only all-or-nothing closes)?**
**A:** **Only all-or-nothing.** `portfolio_manager.decide_trades` builds `TradeOrder(action="SELL")` with `quantity=None` everywhere (lines 90, 103, 113). `paper_trader.execute_sell(quantity=None)` treats None as "full exit". The underlying primitive is partial-close-capable (the `quantity` parameter is honored, and the position-update logic at lines 290-419 handles partial reductions correctly), but no caller ever passes a fractional quantity. **The mechanism exists but is unused.**

---

## Section 3 — Live BQ Probe (verbatim)

### 3.1 Aggregate give-back ratio (`paper_trades`, last 60d, action='SELL')

| Metric | Value |
|---|---|
| sells_60d | 3 |
| avg_realized_pct | -0.418% |
| avg_mfe_pct | +7.350% |
| avg_mae_pct | -11.945% |
| avg_capture_ratio | 0.408 |
| **avg_giveback_ratio_pos_mfe** | **0.387** |
| **avg_giveback_ratio_mfe_gt5** | **0.387** |
| gave_back_50pct_count | 0 (one trade at 48.5%, just below threshold) |
| winners_with_mfe_gt5 | 2 |
| avg_holding_days | 17.33 |
| stop_loss_exits | 0 |
| stop_loss_trigger_exits | 0 |
| sell_signal_exits | **3** |
| signal_downgrade_exits | 0 |
| kill_or_flatten_exits | 0 |

**Read:** every exit in the last 60 days was a model-driven SELL signal. The entry-anchored stop did not fire once. The average winner gave back **38.7% of its best mark**. Capture ratio is **41%** — the system keeps less than half of the unrealized peak.

### 3.2 Per-trade detail (`paper_trades`, last 60d, action='SELL')

| Ticker | Reason | Held days | Realized % | MFE % | MAE % | Capture | Give-back % |
|---|---|---|---|---|---|---|---|
| CIEN | sell_signal | 20 | +6.46 | +12.56 | -9.26 | 0.515 | **48.5** |
| FIX | sell_signal | 15 | +6.75 | +9.49 | -0.07 | 0.711 | **28.9** |
| TER | sell_signal | 17 | -14.46 | 0.00 | -26.51 | 0.000 | n/a |

**Read:** CIEN ran to +12.56% then exited at +6.46% (gave back nearly half). FIX ran to +9.49% then exited at +6.75% (gave back 29%). TER never went positive (MFE=0), but its MAE was -26.51% — the position drifted 26.5% underwater before exiting at -14.46%. Even on that downside trade, the entry-anchored 8% stop should have fired at -8% and didn't (likely because the position was opened before the phase-25.2 backfill landed and had `stop_loss_price=None`, then bounced past -8% on the next mark-to-market and was held until the LLM SELL fired).

### 3.3 Current position stop coverage (`paper_positions`, all 11 rows)

| Ticker | Sector | Entry | Now | Stop | Unrealized % | MFE % | Classification | Stop vs Entry % | Stop vs Current % |
|---|---|---|---|---|---|---|---|---|---|
| SNDK | Technology | 989.90 | 1388.00 | NULL | +40.22 | +57.64 | **NO_STOP** | n/a | n/a |
| MU | Technology | 506.65 | 721.76 | 466.12 | +42.46 | +57.62 | STATIC_8PCT_ENTRY | -8.00 | **-35.42** |
| INTC | Technology | 82.57 | 112.08 | NULL | +35.74 | +53.85 | **NO_STOP** | n/a | n/a |
| COHR | Technology | 320.91 | 355.20 | 295.24 | +10.69 | +28.36 | STATIC_8PCT_ENTRY_APPROX | -8.00 | **-16.88** |
| WDC | Technology | 404.00 | 457.27 | NULL | +13.19 | +27.75 | **NO_STOP** | n/a | n/a |
| LITE | Technology | 881.64 | 894.41 | NULL | +1.45 | +19.50 | **NO_STOP** | n/a | n/a |
| ON | Technology | 98.40 | 108.12 | NULL | +9.88 | +19.49 | **NO_STOP** | n/a | n/a |
| DELL | Technology | 216.09 | 238.26 | NULL | +10.26 | +19.14 | **NO_STOP** | n/a | n/a |
| GLW | Technology | 175.89 | 177.24 | NULL | +0.77 | +19.05 | **NO_STOP** | n/a | n/a |
| KEYS | Technology | 330.19 | 338.41 | 303.78 | +2.49 | +11.47 | STATIC_8PCT_ENTRY | -8.00 | -10.23 |
| GEV | Industrials | 1078.49 | 1015.32 | 992.22 | -5.86 | +3.15 | STATIC_8PCT_ENTRY_APPROX | -8.00 | -2.28 |

**Coverage breakdown:**
- **7 of 11 positions have NO stop_loss_price** (NO_STOP): SNDK, INTC, WDC, LITE, ON, DELL, GLW. (Goal said 6; we discovered WDC also has no stop.)
- 4 of 11 have STATIC_8PCT_ENTRY (or approximately so due to rounding): MU, COHR, KEYS, GEV.
- 0 of 11 have a trailing stop (`stop_loss_price` > the static entry-anchored 8% level).
- Two positions (SNDK MFE +57.64%, INTC MFE +53.85%) have given back ~17-18 percentage points from peak with NO stop in place.
- MU has its entry-anchored stop **35.4% below current price** — the position can roundtrip to a -8% loss from entry even though it printed +57.6% MFE.

### 3.4 Sector concentration (`paper_positions`)

| Sector | Positions | Sector MV $ | % of positions value |
|---|---|---|---|
| Technology | 10 | 10,971.84 | **89.3** |
| Industrials | 1 | 1,314.81 | 10.7 |

**Read:** This is exactly the AQR Q1 2025 "new paradigm" warning and the MSCI summer-2025 quant-fund-wobble crowded-trade unwind risk. QuantAgents' `max(SE_j) = 0.89` would saturate Dave's Risk Alert Meeting trigger (`R_score > 0.75` per arXiv 2510.04643). The Layer-2 in-app MAS has pre-trade caps (`paper_max_per_sector`, `paper_max_per_sector_nav_pct`) but the Risk Judge's prompt never sees the current exposure, so it cannot reason about why an additional Tech BUY is dangerous in this regime.

### 3.5 BQ probe headline metrics — copy for the harness log

```
sells_60d=3 avg_realized=-0.42% avg_mfe=+7.35% avg_capture=0.408 avg_giveback=38.7%
positions=11 no_stop=7 static_stop=4 trailing_stop=0 max_sector=Tech_89.3%
exits_by_reason: stop_loss=0 sell_signal=3 downgrade=0 kill=0
high_mfe_positions: SNDK_+57.64%_NO_STOP, MU_+57.62%_stop_-35.4%_from_now, INTC_+53.85%_NO_STOP
```

---

## Section 4 — Ranked Remediation Proposals (P1/P2/P3)

Each proposal cites the research finding(s) that justify it, the code anchor where the gap lives, and where the Kaminski-Lo adversarial guard applies. All are CANDIDATES for future phase-31.x cycles; THIS cycle does not implement them.

### P1 — Highest impact, lowest implementation risk

#### P1.1 — Breakeven-stop ratchet at +1R (target phase-31.1)
**Hypothesis:** When `position.mfe_pct >= paper_default_stop_loss_pct` (i.e., unrealized gain ≥ 1R), advance `stop_loss_price = entry_price` (lock in zero-loss).
**Research basis:** AFML ch.3 (López de Prado) + Van Tharp R-multiples (PnL Ledger) + Tradewink ratchet definition. Adversarial caveat (Carver §4.2): "nothing special about your entry level" — Carver's HWM-trailing is theoretically superior, but breakeven is strictly better than the entry-anchored static stop currently in production. Strict Pareto improvement.
**Code anchor:** mutate `stop_loss_price` in `paper_trader.mark_to_market` BEFORE the MFE/MAE write at `paper_trader.py:440-456`. The existing `check_stop_losses` at `paper_trader.py:484-493` is unchanged — it just sees the new (higher) stop.
**Implementation site:** new `_advance_stop(pos)` helper called once per position per mark-to-market. New nullable BQ field `stop_advanced_at_R: float` (audit only). No schema migration that breaks reads.
**Adversarial guard:** unconditional — breakeven is safe even for mean-reversion entries (Kaminski-Lo Proposition 2 is about TRAILING stops degrading mean-reversion EV; a one-shot move-to-breakeven is not trailing in their sense).
**Live signal it would address:** SNDK, MU, INTC, COHR, WDC, LITE, ON, DELL each crossed +1R and would now have a breakeven floor.

#### P1.2 — HWM-trailing stop (Chandelier-lite) in the live loop (target phase-31.2)
**Hypothesis:** After P1.1 fires, replace static-stop with `stop_loss_price = max(stop_loss_price, peak_price × (1 − trail_pct/100))` where `peak_price = entry_price × (1 + mfe_pct/100)` and `trail_pct = paper_trailing_stop_pct` (new setting; start at 8% to match current stop-pct).
**Research basis:**
- Carver qoppac.blogspot.com 2020: "the stop 'trails' the price, ratcheting upwards with every new HWM."
- arXiv 2602.11708 (AdaptiveTrend, 2026) ablation: **+0.73 Sharpe**, **9.7 pp MaxDD reduction** from dynamic ATR-scaled trailing.
- Kaminski-Lo 2014 (empirical headline): "stop loss over monthly intervals in daily data can increase return by 1.5% and decrease volatility by 5% causing an increase in Sharpe by as much as 20%."
- Han-Zhou-Zhu 2014 (SSRN 2407199, 1926-2013 US equity): 10% momentum-strategy stop took MaxDD from -49.79% → -11.36% on equal-weighted, -64.97% → -23.28% on value-weighted; Sharpe more-than-doubled.
**Code anchor:** port `signals_server.py:1052-1154` (the `check_stop_loss` trailing branch) into `paper_trader.mark_to_market`. The algorithm already exists; this is wire-it-up, not derive-it.
**Implementation site:** same helper as P1.1, called AFTER MFE/MAE update so `peak_price` is fresh. Mutate `stop_loss_price` in place (monotonic max).
**Adversarial guard (Kaminski-Lo Proposition 2):** the implementation MUST check the entry strategy and SKIP the trailing logic for mean-reversion entries. `paper_positions` does not currently have an `entry_strategy` field — phase-31.2 needs to either (a) add that field (schema migration), or (b) infer it from the entry's `analysis_id` → `strategy_decisions` table. Until the adversarial guard ships, the conservative MVP is to enable trailing only on entries flagged `momentum` or `triple_barrier` via the `risk_judge` output, and default-off otherwise. This is the load-bearing safety guard.

#### P1.3 — Surface sector exposure to Risk Judge (target phase-31.3)
**Hypothesis:** Inject current portfolio sector exposure (`{Technology: 0.89, Industrials: 0.11, ...}`) into the Risk Judge's FACT_LEDGER section. Add an explicit instruction: "If sector exposure for the candidate's sector exceeds 60% of NAV, require compelling sector-specific upside or reduce position size."
**Research basis:**
- arXiv 2510.04643 (QuantAgents, Oct 2025) `R_score = w1·β_p + w2·(1/LR) + w3·max(SE_j) + w4·σ_p` — sector exposure is a first-class input.
- AQR Q1 2025 "New Paradigm in Active Equity" — Mag-7 concentration regime requires explicit guardrails.
- MSCI 2025 summer-2025 quant-wobble — crowded-trade unwind affected Two Sigma, Renaissance, Man Group.
**Code anchor:** `risk_judge.md:22-29` (Data Inputs); `portfolio_manager.py:209-285` (existing sector caps — the data is already computed at trade-decision time); `synthesis_agent.md:80-88` (add a `portfolio_concentration_warning` field to feed the Risk Judge).
**Implementation site:** orchestrator.py FACT_LEDGER builder adds `portfolio_sector_exposure` block; `risk_judge.md` prompt gains one new section ("PORTFOLIO CONTEXT — current sector weights").
**Adversarial guard:** none needed — this is a context injection, not a control change.
**Live signal it would address:** current 10/11 Tech = 89.3% NAV would surface a `max(SE_j) = 0.89` warning to the Risk Judge on every new Tech BUY proposal.

### P2 — Higher impact, medium implementation risk

#### P2.1 — Van Tharp scale-out ladder at +2R / +3R (target phase-31.4)
**Hypothesis:** When MFE crosses +2R, sell 1/3 of the position. When MFE crosses +3R, sell another 1/3. Final 1/3 rides the P1.2 trailing stop.
**Research basis:** Van Tharp Institute (Tharp Think trading concepts); PnL Ledger expectancy + R-multiples ("Partial exits reduce the effective R outcome ... scaling out improves Sharpe ratio and reduces variance even when it reduces total P&L on the biggest winners").
**Code anchor:** `paper_trader.execute_sell(quantity=…)` already supports partial closes (`paper_trader.py:290-419`). The mechanism exists; needs only a new `_check_scale_out_targets(position)` helper called from `mark_to_market` or a new Step 5.7 in `autonomous_loop.py`.
**Implementation site:** 2 new nullable BQ fields per position (`scale_out_2R_at`, `scale_out_3R_at` — timestamps, audit only). Position `quantity` already mutates correctly through partial sells.
**Adversarial guard:** Carver §2.3 rejects profit targets; we are NOT setting hard TPs, we are scaling out at MFE milestones — the final 1/3 still rides the trail (which Carver endorses). Mean-reversion entries: configure to skip scale-outs (single-shot exit at mean reversion is the canonical pattern).

#### P2.2 — ATR-scaled stops at entry (replace fixed 8% default) (target phase-31.5)
**Hypothesis:** Replace `paper_default_stop_loss_pct = 8.0` with `stop_loss_price = entry_price - 2 × ATR(14)`. Same multiplier for the trail distance in P1.2.
**Research basis:**
- LuxAlgo (snippet, multi-source): "In a study of 1,000 trades, using a 2× ATR stop-loss reduced the maximum drawdown by 32% compared to fixed stop-loss levels."
- Carver qoppac.blogspot.com 2020: "set the stop-loss at X × volatility of the market."
- AFML ch.3 (López de Prado): "the upper and lower barriers can be dynamically set based on the rolling estimate of return volatility."
- pyfinagent's own `quant_strategy.md:33-34`: "Literature recommends `TP = daily_vol × multiplier`, `SL = daily_vol × multiplier` instead of fixed percentages." Gap is already self-documented.
**Code anchor:** `paper_trader.py:112-119` (default-stop synthesis); `paper_trader.py:495-562` (`backfill_missing_stops`); `_extract_stop_loss` in `portfolio_manager.py:337-378`.
**Implementation site:** new `_atr_lookup(ticker, period=14)` helper using yfinance; cache per-ticker per-day; fail-safe to 8% if the ATR fetch errors out.

#### P2.3 — Wire the tiered drawdown ladder into the live loop (target phase-31.6)
**Hypothesis:** Port `signals_server.track_drawdown`'s 5/10/15 ladder (warning / derisk / kill) into `paper_trader.check_and_enforce_kill_switch`. Currently the helper has only the binary kill switch.
**Research basis:** `signals_server.py:1166-1168` cites QuantConnect's MaximumDrawdownPercentPortfolio model; QuantAgents (arXiv 2510.04643) ablation: "Specifically, the system showed reduced maximum drawdown improvement and volatility control" when Risk Alert Meeting was removed.
**Code anchor:** `paper_trader.py:700-733` (current binary kill switch); `signals_server.py:1156-1243` (target implementation); `autonomous_loop.py:739-760` (caller).
**Implementation site:** drop-in addition to `check_and_enforce_kill_switch`; emit `-5%` log warn, `-10%` next-cycle BUY size halver, `-15%` full liquidate. The exception-frame already calls `flatten_all`.

### P3 — Lower impact OR higher implementation risk (deferred)

#### P3.1 — Risk-Judge prompt: add exit-policy output block
Output schema extension `exit_policy: {breakeven_at_R, trail_pct, scale_out_levels: [R1, R2, R3]}`. Ships AFTER P1.1+P1.2 validate. Risk: prompt changes interact with SkillOptimizer regressions.

#### P3.2 — Meta-labeling exit classifier (per-position EXIT decision)
Lightweight Flash call per position per cycle: `(MFE, capture_ratio, holding_days, current_drawdown_from_peak, regime) → {HOLD, TRIM_PARTIAL, EXIT}`. AFML ch.3.6. Adds ~50 LLM calls/day; defer until deterministic rules prove their lift.

#### P3.3 — Strategy-conditional exit policy (full Kaminski-Lo regime guard)
Exit policy as a function of the entry strategy (momentum → trail+scale-out; mean-reversion → fixed TP at mean + time barrier, no trail; pairs → cointegration-breach exit, no trail). Requires `paper_positions.entry_strategy` field (schema migration) and per-strategy exit-policy lookup. The SAFE intermediate is P1.2's `skip-trail-on-mean-reversion` guard; full strategy-conditional logic is the long-form right answer.

---

## Section 5 — JSON-Ready phase-31 Masterplan Entries

Following the phase-23.8 schema (`id`, `name`, `status`, `description`, `acceptance_criteria` array, `verification` object with `command` and optional `live_check`). The parent entry (phase-31) defines the umbrella; phase-31.1, phase-31.2, phase-31.3 cover the P1 remediation items.

**This block is parseable as JSON** via `python -c "import json; json.loads(open('/Users/ford/.openclaw/workspace/pyfinagent/handoff/current/experiment_results.md').read().split('```json')[1].split('```')[0])"`.

```json
[
  {
    "id": "phase-31",
    "name": "Profit-Protection + Risk-Agent Hardening (umbrella)",
    "status": "pending",
    "description": "Diagnosed in phase-31.0 audit: Layer-2 autonomous loop has no profit-protection layer. MFE/MAE tracked but never consulted; entry-anchored static stop is the only price-driven exit; 7 of 11 current positions have no stop at all; sector exposure 89.3% Tech; trailing-stop + tiered-drawdown reference code at signals_server.py:1052-1243 is dead code. This umbrella covers the P1-P3 remediation cycles defined in handoff/archive/phase-31.0/experiment_results.md.",
    "acceptance_criteria": [
      "All P1 subphases (31.1, 31.2, 31.3) land status=done",
      "BQ probe re-runs show avg_giveback_ratio_pos_mfe < 0.20 (was 0.387 in phase-31.0 baseline)",
      "Current-position stop-coverage breakdown shows NO_STOP count = 0 (was 7 in phase-31.0 baseline)",
      "Sector concentration warning surfaced in Risk Judge prompt for any new BUY when max(SE_j) >= 0.60"
    ],
    "verification": {
      "command": "python -c \"import subprocess; r=subprocess.run(['grep','-RnE','trailing_stop|take_profit|_advance_stop|scale_out_2R','backend/services/'], capture_output=True, text=True); assert r.stdout, 'no profit-protection symbols found in backend/services/'; print(r.stdout[:2000])\"",
      "live_check": "BQ rows from financial_reports.paper_trades after the next 7 closed sells showing avg_giveback_ratio_pos_mfe < 0.20 OR a written explanation in handoff/current/live_check_31.md why the threshold was not met (insufficient sample size, regime change, etc.)"
    }
  },
  {
    "id": "phase-31.0",
    "name": "Audit: profit-protection + risk-agent hardening gap report",
    "status": "in_progress",
    "description": "Diagnostic-only cycle. Deep researcher brief (22 sources, 3 adversarial including Kaminski-Lo); live BQ probe (avg_giveback=38.7%, 7/11 NO_STOP, 89.3% Tech); per-practice audit table (6 BLOCK findings); P1/P2/P3 ranked remediation. NO CODE EDITS.",
    "acceptance_criteria": [
      "handoff/current/research_brief.md exists with gate_passed=true and ≥5 sources read in full",
      "handoff/current/contract.md cites the brief and lists immutable success criteria",
      "handoff/current/experiment_results.md contains all 5 required sections (audit table, specific-Q answers, BQ probe, P1/P2/P3, JSON entries)",
      "Q/A verdict: PASS (not CONDITIONAL)",
      "handoff/harness_log.md appended with the phase-31.0 cycle block",
      "phase-31, phase-31.1, phase-31.2, phase-31.3 inserted into .claude/masterplan.json"
    ],
    "verification": {
      "command": "test -f handoff/current/research_brief.md && test -f handoff/current/contract.md && test -f handoff/current/experiment_results.md && python3 -c \"import json; m=json.load(open('.claude/masterplan.json')); ids={s['id'] for s in m['steps']}; req={'phase-31','phase-31.0','phase-31.1','phase-31.2','phase-31.3'}; missing=req-ids; assert not missing, f'missing: {missing}'; print('OK', sorted(req))\""
    }
  },
  {
    "id": "phase-31.1",
    "name": "Breakeven-stop ratchet at +1R",
    "status": "pending",
    "description": "When position.mfe_pct >= paper_default_stop_loss_pct (1R), mutate stop_loss_price = entry_price. New _advance_stop helper in paper_trader.mark_to_market, called before the MFE/MAE write at paper_trader.py:440-456. New nullable BQ field stop_advanced_at_R (audit only). Justification: López de Prado AFML ch.3; Van Tharp R-multiples; Tradewink ratchet. Adversarial: safe for all strategies — breakeven is NOT trailing in Kaminski-Lo's sense.",
    "acceptance_criteria": [
      "_advance_stop helper exists in backend/services/paper_trader.py and is called from mark_to_market",
      "When position.mfe_pct >= settings.paper_default_stop_loss_pct, stop_loss_price is mutated to entry_price (and persisted via save_paper_position)",
      "New nullable column stop_advanced_at_R added to paper_positions schema (timestamp ISO string)",
      "Backfill: any current position with mfe_pct >= 8% has stop_advanced_at_R populated on the first mark-to-market after deploy",
      "Unit test backend/tests/test_phase_31_1_breakeven_ratchet.py covers: (a) mfe<1R → no advance, (b) mfe=1R → stop moves to entry, (c) mfe=2R → stop stays at entry (does NOT advance further; that's P1.2's job), (d) idempotent on repeat mark-to-market",
      "No regression in existing check_stop_losses behaviour"
    ],
    "verification": {
      "command": "python -m pytest backend/tests/test_phase_31_1_breakeven_ratchet.py -v && python -c \"import ast; ast.parse(open('backend/services/paper_trader.py').read())\"",
      "live_check": "BQ row from financial_reports.paper_positions showing stop_advanced_at_R populated for at least one current high-MFE position (SNDK, MU, INTC, COHR, WDC, LITE, ON, DELL all qualify in phase-31.0 baseline)"
    }
  },
  {
    "id": "phase-31.2",
    "name": "HWM-trailing stop (Chandelier-lite) in live loop + Kaminski-Lo adversarial guard",
    "status": "pending",
    "description": "Port signals_server.check_stop_loss trailing branch (line 1052-1154) into paper_trader. After P1.1 fires, mutate stop_loss_price = max(stop_loss_price, peak_price * (1 - trail_pct/100)) where peak_price = entry_price * (1 + mfe_pct/100). Adversarial guard: skip trailing when entry_strategy in ('mean_reversion', 'pairs'). MVP: enable trailing only on entries flagged 'momentum' or 'triple_barrier'; default-off otherwise. Justification: arXiv 2602.11708 (+0.73 Sharpe), Han-Zhou-Zhu 2014 (MaxDD -49.79% → -11.36%), Carver. Adversarial: Kaminski-Lo Proposition 2.",
    "acceptance_criteria": [
      "Trailing-stop logic ports cleanly from signals_server.py:1052-1154 into paper_trader (NOT a re-derivation — the algorithm exists)",
      "New setting paper_trailing_stop_pct (default 8.0, matches current stop_pct)",
      "stop_loss_price is monotonic max (NEVER moves down)",
      "Adversarial guard: position.entry_strategy in ('mean_reversion', 'pairs') → trailing branch is SKIPPED (still uses static + P1.1 breakeven)",
      "Unit test backend/tests/test_phase_31_2_hwm_trailing.py covers: (a) momentum entry with peak retracement → stop trails up, (b) mean-reversion entry → stop does NOT trail (Kaminski-Lo guard), (c) peak drops then recovers → stop stays at peak-anchored level (no downward move)",
      "Either: paper_positions.entry_strategy field added via migration, OR a deterministic lookup from analysis_id → strategy is implemented (with fallback default = 'momentum' to keep guard fail-CLOSED-conservative — i.e., the trail IS applied unless we KNOW it's mean-reversion)"
    ],
    "verification": {
      "command": "python -m pytest backend/tests/test_phase_31_2_hwm_trailing.py -v && python -c \"import ast; ast.parse(open('backend/services/paper_trader.py').read())\" && grep -n 'mean_reversion' backend/services/paper_trader.py",
      "live_check": "BQ row from financial_reports.paper_positions showing stop_loss_price > avg_entry_price * 0.95 for at least one position whose mfe_pct > 20% (i.e., the trail has actually moved the stop above entry-anchored)"
    }
  },
  {
    "id": "phase-31.3",
    "name": "Surface sector exposure to Risk Judge prompt",
    "status": "pending",
    "description": "Inject current portfolio sector exposure (computed from paper_positions.market_value × sector) into risk_judge.md's FACT_LEDGER section. Add instruction: 'If sector exposure for the candidate's sector exceeds 60% of NAV, require compelling sector-specific upside or reduce position size.' Justification: arXiv 2510.04643 (QuantAgents Dave's R_score with max(SE_j)); AQR Q1 2025 paradigm; MSCI summer-2025 quant-wobble.",
    "acceptance_criteria": [
      "orchestrator.py FACT_LEDGER builder includes portfolio_sector_exposure block (dict: sector → pct of NAV)",
      "risk_judge.md prompt template includes a new PORTFOLIO CONTEXT section that consumes portfolio_sector_exposure",
      "synthesis_agent.md adds portfolio_concentration_warning to its output schema (so the Risk Judge sees it pre-debate)",
      "When max(SE_j) >= 0.60, the Risk Judge prompt explicitly carries the warning; when < 0.60, no warning",
      "Unit test backend/tests/test_phase_31_3_sector_exposure.py covers: (a) 89% Tech portfolio + Tech candidate → warning fires, (b) 89% Tech + Healthcare candidate → no warning, (c) empty portfolio → no warning"
    ],
    "verification": {
      "command": "python -m pytest backend/tests/test_phase_31_3_sector_exposure.py -v && grep -n 'portfolio_sector_exposure' backend/agents/skills/risk_judge.md backend/agents/orchestrator.py",
      "live_check": "Sample Risk Judge prompt output (from a real cycle log in handoff/logs/) showing the PORTFOLIO CONTEXT section with sector exposure injected, and a Risk Judge reasoning paragraph that cites it"
    }
  }
]
```

---

## Files Touched This Cycle

**Read (verification):**
- `backend/services/portfolio_manager.py` (378 lines, full read)
- `backend/services/paper_trader.py` (lines 100-240, 420-560, 690-800)
- `backend/services/autonomous_loop.py` (lines 720-840)
- `backend/agents/skills/risk_judge.md` (140 lines, full)
- `backend/agents/skills/risk_stance.md` (69 lines, full)
- `backend/agents/skills/synthesis_agent.md` (lines 60-140)
- `backend/agents/skills/quant_strategy.md` (250 lines)
- `backend/agents/agent_definitions.py` (lines 1-120)
- `handoff/current/research_brief.md` (378 lines, full)

**Written:**
- `handoff/current/contract.md` (created this cycle)
- `handoff/current/experiment_results.md` (this file)

**Mutated (none — diagnostic only):**
- No code edits.
- No BQ writes.
- No Alpaca calls.

---

## Addendum — Masterplan ID Renumber (post-Q/A, informational only)

Discovered during masterplan insertion that **phase-31 is already taken** in `.claude/masterplan.json` (`phase-31: E2E Pipeline Smoketest (Claude Code-substituted lite path)`, status `done`, nested steps `31.0.1`-`31.0.13`). Q/A flagged this as an "unrelated naming collision" in its evaluation; the cycle still PASS'd because the JSON block in this document is a STAND-ALONE proposal that the audit identifies by the goal-given name. The cycle's HANDOFF IDENTITY remains `phase-31.0` throughout `research_brief.md`, `contract.md`, `experiment_results.md`, `evaluator_critique.md`. The MASTERPLAN INSERTION uses **phase-32** (next available umbrella ID) with the following ID mapping:

| JSON block (in this document) | Masterplan insertion ID |
|---|---|
| `phase-31` (umbrella) | `phase-32` |
| `phase-31.0` (this cycle) | nested step `32.0` |
| `phase-31.1` (Breakeven ratchet) | nested step `32.1` |
| `phase-31.2` (HWM trailing) | nested step `32.2` |
| `phase-31.3` (Sector exposure surfacing) | nested step `32.3` |

The proposal JSON block above is preserved verbatim for traceability of what Q/A PASS'd; the masterplan now carries the same content under phase-32 with the nested schema used by phase-30 / phase-31 (E2E Smoketest). Verification commands in the inserted entries reference the masterplan IDs (`phase-32`...), not the proposal IDs.

## Headline

The system tracks the data needed to take profit (`mfe_pct`, `mae_pct`, `capture_ratio` in `paper_trades` and `paper_positions`) but never consults it. The trailing-stop and tiered-drawdown reference implementations exist in the codebase as dead code (`signals_server.py:1052-1243`). The autonomous loop's exit step (`autonomous_loop.py:794`) calls only the entry-anchored `check_stop_losses`. **Confirmed:** when a position runs up and declines, the system rides it back to the original entry-anchored stop (or to the next LLM SELL signal), giving back an average **38.7% of MFE** on positive-MFE trades, capture ratio **0.408**. The P1.1 breakeven ratchet is a strict Pareto improvement, shippable in a day with zero schema break. Phase-31.1 through phase-31.3 are the recommended next implementation cycles.
