# live_check 53.1 — No-trade rebalance band: $0 ON-vs-OFF + robustness gate

**Date:** 2026-06-01. **Lever:** transaction-cost-aware no-trade / rebalance buffer band
(`backend/backtest/rebalance_band.py`), band_pct=0.2. **$0** replay (free yfinance + the
51.2/52.x machinery; no LLM, no BQ, no live cycles). **NO live flag flip.**

## Measurement (48 monthly rebalances, S&P-500, 2022-2025, top_n=10)

| arm | gross Sharpe | net Sharpe | avg mo ret % | turnover | gross maxDD | net maxDD |
|-----|-------------:|-----------:|-------------:|---------:|------------:|----------:|
| baseline (full reconstitution) | 1.388 | 1.351 | 4.054 | 0.555 | -0.230 | -0.232 |
| band (b=0.2) | 1.399 | 1.366 | 4.085 | **0.489** | -0.230 | -0.232 |

The band does what the literature predicts directionally: **turnover ↓ 0.555→0.489
(~12% fewer names churned/month)**, Sharpe marginally ↑ (gross +0.011, net +0.015),
maxDD unchanged. Net > gross uplift (the cost saving), as designed.

## Robustness gate (52.3 Ledoit-Wolf SR-difference, paired stationary bootstrap, n_boot=5000, seed=42)

Two pre-registered legs (fixed BEFORE the run):

- **GROSS (do-no-harm leg):** dSharpe=**+0.011**, p=0.414, CI90=[-0.071, +0.087].
  do-no-harm (CI_low > -0.05)? **FALSE** (CI_low -0.071 — the band is statistically
  indistinguishable from baseline on gross; it neither helps nor proves-harmless within noise).
- **NET (promote leg):** dSharpe=**+0.015**, p=0.376, CI90=[-0.066, +0.092].
  a-priori rule (p<0.05 AND delta>=+0.05 AND CI_low>0)? **FALSE** (delta 0.015 < 0.05;
  p 0.376; CI_low -0.066).

## Recommendation: **REJECT** (honest negative result — valid per criterion 3)

The band's effect on THIS book/window (48 rebalances) is **directionally favorable but
statistically within noise** — it does not clear the same a-priori robustness gate that
52.3/52.4 used to reject the price-signal levers. Consistent with the research prognosis
(momentum's gross alpha is intact net of costs, so the turnover lever's edge is small;
arXiv:2412.11575's larger uplift was on a higher-turnover universe). The cited-lever
search for a robust construction edge on this book is, like the signal search (phase-52),
not yielding a gate-clearing winner on the available history.

**Disposition:** the lever ships as a **config-gated, default-OFF, tested helper**
(`rebalance_band_enabled=False` → byte-identical full reconstitution; the +20% US momentum
core is untouched). NOT promoted to live. A future revisit (longer history, or once the
live book's realized turnover/costs are higher than the replay assumes) could re-measure;
until then it stays OFF. Reproducibility dump:
`handoff/current/_53_1_band_paired_returns.json`.

## DO-NO-HARM

`rebalance_band_enabled` default False; the helper is not wired into the live
`decide_trades` path (measure-first; no live flip). 8 unit tests pin OFF byte-identity +
the hysteresis logic + maxDD. The SR-diff gate was reused verbatim (`analytics.sharpe_diff_test`,
same a-priori rule + n_boot as 52.3/52.4). $0 / no LLM.
