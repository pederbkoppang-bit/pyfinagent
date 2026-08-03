# experiment_results -- step 82.1

**GENERATE phase.** Contract: `handoff/current/contract.md`.
Research: `handoff/current/research_brief_82.1.md` (gate_passed=true, 6 sources
read in full, 30 URLs, 11 internal files).

## What was produced

`docs/strategy/incumbent_live_strategy_spec.md` -- the live buy funnel written
down AS a strategy (universe, screen, throughput cap, signal/overlays, ranking,
sizing, entry gates, exit), plus the turnover diagnosis. Guarded by
`backend/tests/test_phase_82_1_incumbent_spec.py`.

**No code in the live funnel was changed.** This step is specification and
measurement only.

## Verification command output (verbatim)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_1_incumbent_spec.py -q
..............                                                           [100%]
14 passed in 0.02s
```

## Headline findings

1. **The live strategy is not the backtested strategy.** `optimizer_best.json`
   (`triple_barrier`, tp 10%, sl 12.92%, 90d) reaches the live cycle at
   `autonomous_loop.py:431` and is used for THREE display purposes only --
   a summary field, a summary dict, and a heartbeat label at `:1649` where
   `decided_strategy == prior_strategy` every cycle. The live exit is a stop
   ONLY: 8% initial + 8% trailing, **no take-profit** (scale-out flag OFF) and
   **no time barrier**. The optimizer's 12.92% stop is not in force.
2. **The binding constraint on turnover is `paper_analyze_top_n = 5`**
   (`autonomous_loop.py:1035`). Of ~583 universe names, 5 per cycle receive the
   28-agent analysis, so 5 are the only names that can produce a BUY. ~1
   BUY/cycle reproduces 33 lifetime BUYs and ~1-2 trades/week. Throttled, not
   blocked -- and the throttle is a COST control, not a risk control.
3. **Concentration is generated at ranking.** `rank_candidates` defaults to
   `strategy="momentum"` (`screener.py:252`) over momentum/RSI/SMA features
   only; all five diversification levers are OFF. In a semis-led tape the top 10
   by momentum are one industry, so the 5 analysed names are that industry every
   cycle.
4. **The incumbent never forms a valuation view.** No valuation term in the
   screen, no regime input to sizing, no take-profit. Its cash position is an
   artefact of throughput, not a defensive decision. Any claim that it "went
   defensive because the market is expensive" is unsupported by the code.

## Competing explanations, each REFUTED by measurement

| explanation | verdict | evidence |
|---|---|---|
| kill switch blocking BUYs | REFUTED | live `paused:false`, `trailing_dd_pct 3.6308` vs limit 10.0, `any_breached:false`. And a DISARMED switch does not refuse: `paper_trader.py:175` gates on `is_paused()`/`baselines_present_in()`, docstring at `:182` says it reads baselines "NEVER `armed`" -- gating on `armed` was the phase-36.9 money-path regression |
| "GATE NOT ELIGIBLE 2/5" | REFUTED (category error) | `paper_go_live_gate.py:117` is the paper -> REAL-MONEY promotion gate; not on the order path. Reads 2/5 largely because `trades_ge_100` needs 100 round-trips and the throttle produced 32 |
| position / sector capacity | REFUTED | effective `paper_max_positions=30`, `paper_max_per_sector=5`; book holds 1 |
| "no candidate clears conviction" | NOT MEASURABLE -- itself a defect | `portfolio_manager.py:188` drops non-BUY with a bare `continue`; `buy_rejections` only appends inside `execute_buy`. A cycle can analyse 5, buy 0, report ZERO rejections. Queued as 82.14 |

## Research-brief corrections (Main verified rather than inherited)

The gate PASSED on process, but three of its internal code claims were WRONG and
were kept out of the spec:

| brief claim | measured |
|---|---|
| held-cap 10 blocks every buy | `paper_max_positions` = **30** via `risk_overrides.get_effective`; book holds 1 |
| `paper_max_per_sector = 2` | effective **5** |
| "a disarmed kill switch refuses every BUY" | REFUTED at `paper_trader.py:175-216`; the docstring exists specifically to warn against this misreading |

Two of its four ranked mechanisms rested on those numbers. Its M2
(momentum-only cross-section) and M4 (8% entry-relative stop is sub-1-ATR on
high-beta semis, with no TP and no time stop) verified and are carried.

Its most valuable contribution is not a code fact: the 2025-2026 literature
names AI/memory/semis/Korea as THE crowded trade of the period, and the book's
holdings are that list. So the concentration is the ranker faithfully expressing
a crowded factor -- a different and worse problem than a ranker bug. MSCI
(tested 1999-2024) prescribes crowding caps / dynamic sizing over hard
sector-neutrality, agreeing with this repo's own measured **-0.166 Sharpe** for
hard sector-neutral on a long-only book (`autonomous_loop.py:597`).

## Citation integrity

The resolver test proves a cited line EXISTS; it cannot prove the line says what
the claim says. Main therefore spot-checked the ten load-bearing citations by
printing each line. **Two were wrong and were fixed**: `autonomous_loop.py:1034`
pointed at `else:` (the slice is `:1035`), and `:588` pointed at a log statement
rather than the -0.166 sector-neutral measurement (`:597`). Both corrected and
re-verified by printing the corrected lines. A citation that resolves but
misleads is worse than none.

## Scope honesty

- No live trading behaviour changed; no gate, threshold or flag was moved.
- The diagnosis names ONE binding constraint and refutes the alternatives; it
  does NOT fix the throttle or the concentration (explicit non-scope).
- Deliberately NOT bundled: raising `paper_analyze_top_n` costs LLM spend (28
  agents/name, against the standing $0-metered constraint and the $25/day cap),
  whereas `paper_min_k_sectors_analyzed > 0` re-allocates the SAME five slots
  and is free.
- `test_spec_has_a_meaningful_number_of_citations` exists so the resolver test
  cannot pass vacuously on an empty citation set.
