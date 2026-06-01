# Experiment Results — phase-53.1 (Algorithm/quant elevation)

**Date:** 2026-06-01. **Status:** complete. Lever = no-trade rebalance band; measured
ON-vs-OFF on the $0 replay; robustness gate → **REJECT** (honest negative, valid per
criterion 3). Config-gated default-OFF (DO-NO-HARM); NO live flag flip. $0 (no LLM).

## What was done

1. Implemented the no-trade band as a standalone gated helper
   (`backend/backtest/rebalance_band.py::apply_no_trade_band` + `max_drawdown`) — the
   single source of truth used by the replay (and a future live enable).
2. Config-gated it via `rebalance_band_enabled` (default False) + `rebalance_band_pct`
   (0.2) in `settings.py` — default OFF ⇒ byte-identical full reconstitution.
3. Measured baseline vs band over 48 monthly S&P-500 rebalances (2022-2025) via the
   $0 replay, reporting Sharpe/return/turnover/maxDD GROSS and NET-of-cost.
4. Ran the 52.3 Ledoit-Wolf SR-difference gate on two pre-registered legs (gross
   do-no-harm + net promote). Honest **REJECT**.

## Files changed

| File | Change |
|------|--------|
| `backend/backtest/rebalance_band.py` | NEW — `apply_no_trade_band` (gated hysteresis; OFF=full reconstitution) + `max_drawdown`. |
| `backend/config/settings.py` | NEW gated flags `rebalance_band_enabled` (False) + `rebalance_band_pct` (0.2). Default OFF = byte-identical; not wired into live decide_trades (measure-first). |
| `scripts/ablation/no_trade_band_replay.py` | NEW $0 replay (reuses the 51.2 loaders + `sharpe_diff_test`); dual gross/net SR-diff gate. |
| `backend/tests/test_phase_53_1_rebalance_band.py` | NEW — 8 tests (OFF byte-identity, hysteresis retain/drop, ≤top_n, maxDD). |
| `handoff/current/live_check_53.1.md` | The ON-vs-OFF comparison + SR-diff stats + REJECT recommendation. |
| `handoff/current/_53_1_band_paired_returns.json` | Reproducibility dump (paired arrays + verdict). |

## Verification output (verbatim)

```
# unit tests
python -m pytest backend/tests/test_phase_53_1_rebalance_band.py -q   -> 8 passed
# settings flags default OFF
rebalance_band_enabled: False | pct: 0.2 ; sharpe_diff_test import OK
# $0 replay (48 rebalances, S&P-500, 2022-2025, top_n=10, band=0.2):
arm         grossSharpe  netSharpe   avgRet%  turnover  grossMaxDD  netMaxDD
baseline          1.388      1.351     4.054     0.555      -0.230    -0.232
band              1.399      1.366     4.085     0.489      -0.230    -0.232
GROSS (do-no-harm): dSharpe=+0.011 p=0.414 CI90=[-0.071,+0.087]  -> do-no-harm? False
NET   (promote):    dSharpe=+0.015 p=0.376 CI90=[-0.066,+0.092]  -> promote?   False
53.1 RECOMMENDATION: REJECT (not robust on the net-of-cost a-priori gate) -- honest negative
```

## Acceptance-criteria mapping (phase-53.1 — VERBATIM)

| # | Criterion | Result |
|---|-----------|--------|
| 1 | research gate passed (≥5 sources in full + recency scan) + lever justified from literature | PASS — researcher gate (7 sources, recency scan); band justified (Garleanu-Pedersen, arXiv:2412.11575, Kitces); the 4 other levers rejected from the literature |
| 2 | measured ON-vs-OFF via $0 replay on production universe (Sharpe/return/turnover/maxDD) | PASS — 48 S&P-500 rebalances; all four metrics reported gross+net |
| 3 | improvement subjected to the SAME Ledoit-Wolf SR-diff gate (a-priori rule); REJECT is valid | PASS — `analytics.sharpe_diff_test` reused verbatim, dual legs, a-priori rule; honest **REJECT** |
| 4 | config-gated, no regression (default byte-identical), NO live flip; live_check records compare+stats+rec | PASS — `rebalance_band_enabled=False` default; 8 tests pin OFF byte-identity; no live wiring; `live_check_53.1.md` written |

## DO-NO-HARM / scope honesty

- **REJECT is the honest verdict** — the band cut turnover ~12% + marginally lifted Sharpe,
  but the SR-difference is statistically within noise on this 48-month sample (p>0.37,
  delta < the 0.05 a-priori threshold). No p-hacking: the a-priori rule + dual legs were
  fixed in `contract.md` BEFORE the run.
- Default OFF ⇒ the +20% US momentum core is byte-identical; the helper is NOT wired into
  the live `decide_trades` path (a live enable would be a separate operator-gated step).
- $0: free yfinance prices, no LLM, no BQ writes, no live cycles. SR-diff gate reused
  verbatim (same a-priori rule + n_boot=5000 + seed as 52.3/52.4). No emoji; ASCII.
