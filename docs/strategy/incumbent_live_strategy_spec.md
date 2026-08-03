# The incumbent live strategy, written down as a strategy

**Step 82.1 (phase-82). Measured 2026-08-03. Specification and diagnosis only —
no code in the live funnel was changed.**

The system has two buy lanes and they are not the same strategy. This document
specifies the one that actually spends money.

Every code claim carries a `file:line` that resolves in the current tree;
`backend/tests/test_phase_82_1_incumbent_spec.py` asserts that mechanically.

---

## 0. The lane confusion, resolved first

`backend/backtest/experiments/optimizer_best.json` declares
`strategy="triple_barrier"`, `tp_pct=10.0`, `sl_pct≈12.92`, `holding_days=90`.
Those parameters are loaded into the live cycle at
`backend/services/autonomous_loop.py:431` — and then used for **three display
purposes only**:

| what | where | use |
|---|---|---|
| `sharpe` | `backend/services/autonomous_loop.py:433` | a summary field |
| `tp_pct`, `sl_pct`, `holding_days` | `backend/services/autonomous_loop.py:434` | written into the cycle `summary` dict |
| `strategy` | `backend/services/autonomous_loop.py:1649` | a heartbeat **label** |

At `backend/services/autonomous_loop.py:1649` the label is written to
`strategy_decisions` with `decided_strategy == prior_strategy` on every cycle,
and the in-code comment at `backend/services/autonomous_loop.py:1644` says
"strategy router (deferred to phase-31)".

**No `STRATEGY_REGISTRY` label method executes in the live path.** The
registry's five strategies (`backend/backtest/backtest_engine.py:32`) score
research runs only. The registry's *exit parameters* cross over as display
fields; its *selection logic* does not.

Consequence that matters for any comparison: the backtested exit is a triple
barrier (take-profit, stop, time). The live exit is a stop only — see §6. They
are different strategies, so "the backtest says Sharpe 1.17" is not a statement
about the live book.

---

## 1. Universe

US + EU + KR equities. Recent cycles report `universe_size` between 543 and 583
names (`handoff/cycle_history.jsonl`, `funnel.universe_source = "US+EU+KR"`).

## 2. Screen — momentum, and only momentum

`backend/tools/screener.py:91` `screen_universe` computes price-momentum, RSI
and SMA-distance features. `backend/tools/screener.py:249` `rank_candidates`
takes `strategy: str = "momentum"` as its **default**
(`backend/tools/screener.py:252`), and the momentum branch at
`backend/tools/screener.py:299` builds the composite score written at
`backend/tools/screener.py:413`. A 52-week-high proximity term is available
(`backend/tools/screener.py:210`, George–Hwang 2004 anchoring).

The screen keeps `paper_screen_top_n = 10` names.

**There is no valuation, quality or leverage term in the ranking.** This is the
single most important fact about the incumbent: it is a pure cross-sectional
momentum ranker.

## 3. The throughput cap — the defining constraint

`backend/services/autonomous_loop.py:1035`:

```python
_analyze_cands = new_candidates[:settings.paper_analyze_top_n]
```

`paper_analyze_top_n = 5`. Those five tickers
(`backend/services/autonomous_loop.py:1036`) are the **only** names that
receive the 28-agent analysis in a cycle, and therefore the only names that can
possibly produce a BUY.

Three lines above, at `backend/services/autonomous_loop.py:1031`, sits the
diversification lever:

```python
_min_k = int(getattr(settings, "paper_min_k_sectors_analyzed", 0) or 0)
```

`paper_min_k_sectors_analyzed = 0`, so the `else` branch runs and the slice is a
plain top-N by momentum score.

## 4. Signal and overlays

The 28-agent pipeline emits a recommendation. Only `BUY`/`STRONG_BUY` survive:
`backend/services/portfolio_manager.py:188` drops everything else with a bare
`continue`.

Overlays — short interest, peer lead-lag, social velocity, analyst narrative,
options flow — **re-rank; they do not veto.** Several are off by default
(`analyst_narrative_enabled=False`, `options_flow_screen_enabled=False`,
`insider_signal_screen_enabled=False`).

Ranking is by `conviction_score` from the meta-scorer, invoked at
`backend/services/autonomous_loop.py:953`.

## 5. Sizing — discretionary, not risk-parity

`backend/services/portfolio_manager.py:392`:

```python
target_amount = nav * (position_pct / 100.0)
```

`position_pct` is the Risk Judge agent's `recommended_position_pct`
(`backend/agents/schemas.py:120`), resolved at
`backend/services/portfolio_manager.py:205` and **defaulting to 10% of NAV**
when the judge does not specify (`backend/services/portfolio_manager.py:391`).
The order is then capped by available cash
(`backend/services/portfolio_manager.py:393`).

Sizing is an LLM judgement, not inverse-volatility and not risk-parity. A
high-beta semiconductor and a utility receive the same 10% default.

## 6. Exit — a stop, and nothing else

- **Initial stop:** 8% below entry. `paper_default_stop_loss_pct = 8.0`. If a
  BUY arrives without a stop, one is synthesised —
  `backend/services/paper_trader.py:296`.
- **Trailing stop:** `paper_trailing_stop_pct = 8.0`, updated during
  mark-to-market at `backend/services/paper_trader.py:735`.
- **Enforcement:** `backend/services/paper_trader.py:778` `check_stop_losses`
  sells when `current <= stop`.
- **Take-profit:** the scale-out ladder exists at
  `backend/services/paper_trader.py:797` but `paper_scale_out_enabled = False`.
  **There is no take-profit.**
- **Time barrier:** none. `holding_days=90` never reaches the live exit path.
- **Re-evaluation:** held names are re-analysed every
  `paper_reeval_frequency_days = 3`.

So the live exit is: *stop out at −8%, or trail up 8% behind the high, forever.*
The optimizer's 12.92% stop and 90-day time barrier are not in force.

## 7. Capacity limits (all measured slack)

| limit | effective value | book today |
|---|---|---|
| `paper_max_positions` | 30 | 1 |
| `paper_max_per_sector` | 5 | 1 |
| `paper_swap_max_per_cycle` | 2 | — |
| `paper_swap_min_delta_pct` | 25.0 | — |

Effective values resolved through `risk_overrides.get_effective` at
`backend/services/portfolio_manager.py:345`. None is binding.

---

## 8. Measured state, reconciled to the operator screenshot

| measure | value |
|---|---|
| `paper_trades` BUY | 33 |
| `paper_trades` SELL | 32 |
| `paper_positions` open | 1 (NTAP, 5.347 @ 177.85, stop 164.62) |
| closed round-trips | 32 |

**Reconciliation.** The screenshot (2026-07-31, 17:42 local) shows
`Trades (64)` and `POSITIONS 0`. 32 BUY + 32 SELL = 64, with nothing open — the
tile was correct and the book was genuinely flat at capture. The 33rd BUY
(NTAP) executed at 2026-07-31T18:47:37Z, **after** the screenshot. Nothing is
missing and no trade was lost.

The operator's premise — "0 positions, 100% cash" — was true at that instant and
is not a stuck state.

---

## 9. Diagnosis: the ONE binding constraint on turnover

### It is `paper_analyze_top_n = 5` at `backend/services/autonomous_loop.py:1035`.

Live funnel telemetry, quoted verbatim from `handoff/cycle_history.jsonl`
(this exact row occurs 8 times; `candidates: 10 -> new_to_analyze: 5` is
invariant across 21 cycles, while `universe_size` ranged 543-583 -- the most
recent cycle, the one carrying `n_trades=1`, read 543/541):

```
{"universe_source": "US+EU+KR", "universe_size": 583, "screened": 577,
 "candidates": 10, "new_to_analyze": 5, "reeval_tickers": 0}
```

with `n_trades` of `0, 0, 0, 1` across those cycles.

583 names enter; **5 are evaluated**. At the observed BUY rate that is ~1 buy
per cycle, which reproduces 33 lifetime BUYs and the ~1–2 trades/week rate. The
book is **throttled, not blocked**, and the throttle is a cost control (28
agents per analysed name), not a risk control.

### Every competing explanation is refuted by measurement

**Kill switch — REFUTED.** Live `/api/paper-trading/kill-switch`:

```json
{"paused": false, "peak_nav": 24666.57, "current_nav": 23770.98,
 "breach": {"trailing_dd_pct": 3.6308, "trailing_dd_limit_pct": 10.0,
            "any_breached": false, "armed": false}}
```

3.63% drawdown against a 10% limit. Not paused, nothing breached. And a
*disarmed* switch does not refuse BUYs: `_kill_switch_refusal_for_buy` at
`backend/services/paper_trader.py:175` gates on `is_paused()` and
`baselines_present_in()`, and its docstring at
`backend/services/paper_trader.py:183` states it reads baselines "NEVER
`armed`" — gating on `armed` was a money-path regression caught in phase-36.9.

**"GATE NOT ELIGIBLE 2/5" — REFUTED, and it is a category error.**
`backend/services/paper_go_live_gate.py:117` `compute_gate` returns
`promote_eligible` from five booleans defined at
`backend/services/paper_go_live_gate.py:167`: `trades_ge_100`,
`psr_ge_95_sustained_30d`, `dsr_ge_95`, `sr_gap_le_30pct`,
`max_dd_within_tolerance`. This is the **paper → real-money promotion** gate. It
is not on the order path and cannot stop a paper trade. It reads 2/5 largely
because `trades_ge_100` needs 100 round-trips and the throttle has produced 32 —
the gate and the throttle are one finding seen from both ends.

**Capacity — REFUTED.** 30 positions allowed, 1 held (§7).

**"No candidate clears conviction" — NOT MEASURABLE, and that is itself a
defect.** `backend/services/portfolio_manager.py:188` drops non-BUY
recommendations with a bare `continue`, and `buy_rejections` is only appended
inside `execute_buy` (`backend/services/paper_trader.py:276`). A cycle can
analyse 5, buy 0, and report **zero rejections**. The conversion rate from
analysis to attempted order is invisible. Queued as masterplan step 82.14.

---

## 10. Why the book is concentrated in semiconductors

The same slice, ranked momentum-only, with **every** diversification lever off:

| lever | value |
|---|---|
| `sector_neutral_momentum_enabled` | False |
| `multidim_momentum_enabled` | False |
| `paper_soft_sector_diversity_enabled` | False |
| `momentum_52wh_tilt_enabled` | False |
| `paper_min_k_sectors_analyzed` | 0 |

In a semiconductor-led tape the top 10 by 1/3/6-month momentum *are* one
industry, so the five analysed names are that industry every cycle and the
funnel never sees another sector. Observed holdings: AMD, MU, SNDK, 000660.KS
(SK Hynix), NTAP, WDC, INTC, STX, 005930.KS (Samsung), PANW.

**This is not a ranker bug.** The 2025–2026 literature identifies
AI/memory/semis/Korea as the crowded trade of the period; the ranker is
faithfully expressing a crowded factor. That is a different and worse problem
than a coding error, and it is the actual "overpriced" exposure in this book —
not an expensive index, but concentration in the market's most crowded trade.

Note for anyone tempted by hard sector-neutrality: this repo already measured
**−0.166 Sharpe** for hard sector-neutralisation on a long-only book
(`backend/services/autonomous_loop.py:597`), and the external literature (MSCI,
tested 1999–2024) prescribes crowding-score caps and dynamic sizing instead.

---

## 11. What this means for the overpriced-market question

The incumbent **never forms a valuation view**. There is no valuation term in
the screen, no regime input to sizing, and no take-profit. Its cash position is
an artefact of throughput, not a defensive decision.

Any claim that the system "went defensive because the market is expensive" is
unsupported by the code. Candidate strategies (step 82.2) must be judged against
what the incumbent *is* — a five-names-a-day momentum funnel with a sub-ATR stop
— and not against a defensive posture it never had.

Two separable follow-ups, deliberately **not** bundled:

- Raising `paper_analyze_top_n` costs real LLM spend (28 agents per name),
  against the standing `$0 metered` constraint and the $25/day cap.
- Setting `paper_min_k_sectors_analyzed > 0` re-allocates the **same** five
  slots across sectors and is **free**.
