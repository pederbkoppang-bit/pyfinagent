# replay_60_2 results -- window 2026-05-29..2026-06-10 (generated 2026-06-11T11:38:51Z)

Recorded trades: 36; recorded swap pairs: 13; window round-trip SELLs (holding_days<=1): 6

## ARM A fidelity (flag OFF must reproduce recorded swaps): 12/13 reproduced -- FAILURES present, see table

## Per-swap replay

| day | sold | bought | OFF fired | ON verdict | basis |
|---|---|---|---|---|---|
| 2026-05-29 | KEYS | STX | False | SUPPRESSED | true scores 0.0 vs 5.0 -> delta -100% |
| 2026-06-01 | DELL | 000660.KS | True | SURVIVES | true scores 7.0 vs 4.0 -> delta 75% |
| 2026-06-02 | 000660.KS | HPE | True | SUPPRESSED | sentinel: sold ticker had NO same-day analysis (fabricated 0.0) |
| 2026-06-03 | HPE | DELL | True | SUPPRESSED | sentinel: sold ticker had NO same-day analysis (fabricated 0.0) |
| 2026-06-03 | STX | AMD | True | SURVIVES | true scores 6.0 vs 3.0 -> delta 100% |
| 2026-06-04 | AMD | 000660.KS | True | SUPPRESSED | sentinel: sold ticker had NO same-day analysis (fabricated 0.0) |
| 2026-06-04 | DELL | 005930.KS | True | SUPPRESSED | sentinel: sold ticker had NO same-day analysis (fabricated 0.0) |
| 2026-06-05 | 005930.KS | STX | True | SUPPRESSED | sentinel: sold ticker had NO same-day analysis (fabricated 0.0) |
| 2026-06-08 | STX | SNDK | True | SUPPRESSED | sentinel: sold ticker had NO same-day analysis (fabricated 0.0) |
| 2026-06-08 | DELL | MU | True | SUPPRESSED | sentinel: sold ticker had NO same-day analysis (fabricated 0.0) |
| 2026-06-09 | MU | 066570.KS | True | SUPPRESSED | sentinel: sold ticker had NO same-day analysis (fabricated 0.0) |
| 2026-06-09 | SNDK | DELL | True | SUPPRESSED | sentinel: sold ticker had NO same-day analysis (fabricated 0.0) |
| 2026-06-10 | DELL | SNDK | True | SUPPRESSED | sentinel: sold ticker had NO same-day analysis (fabricated 0.0) |

## The 3 named away-week round trips (criterion 4)

| round trip | OFF (recorded) | ON | basis |
|---|---|---|---|
| MU 06-08->06-09 | fired | SUPPRESSED | sentinel: sold ticker had NO same-day analysis (fabricated 0.0) |
| SNDK 06-08->06-09 | fired | SUPPRESSED | sentinel: sold ticker had NO same-day analysis (fabricated 0.0) |
| DELL 06-05->06-08->06-09 | fired | SURVIVES | true scores 7.0 vs 4.0 -> delta 75% |
| DELL 06-05->06-08->06-09 | fired | SUPPRESSED | sentinel: sold ticker had NO same-day analysis (fabricated 0.0) |
| DELL 06-05->06-08->06-09 | fired | SUPPRESSED | sentinel: sold ticker had NO same-day analysis (fabricated 0.0) |
| DELL 06-05->06-08->06-09 | fired | SUPPRESSED | sentinel: sold ticker had NO same-day analysis (fabricated 0.0) |

## Counterfactual ledger (one-step, suppressed pairs)

- 2026-06-02 SELL 000660.KS suppressed: hold-through -64.26 (-13.22% on 486 USD) minus bought-leg(HPE) -0.81 = -63.45 USD
- 2026-06-03 SELL HPE suppressed: hold-through -42.78 (-17.52% on 244 USD) minus bought-leg(DELL) +0.54 = -43.32 USD
- 2026-06-04 SELL AMD suppressed: hold-through -80.90 (-13.53% on 598 USD) minus bought-leg(000660.KS) -48.70 = -32.20 USD
- 2026-06-04 SELL DELL suppressed: hold-through -30.59 (-12.37% on 247 USD) minus bought-leg(005930.KS) -47.13 = +16.54 USD
- 2026-06-05 SELL 005930.KS suppressed: hold-through -54.53 (-8.05% on 677 USD) minus bought-leg(STX) +18.13 = -72.67 USD
- 2026-06-08 SELL STX suppressed: hold-through -42.61 (-6.93% on 615 USD) minus bought-leg(SNDK) -2.46 = -40.15 USD
- 2026-06-08 SELL DELL suppressed: hold-through -56.13 (-7.72% on 727 USD) minus bought-leg(MU) -44.95 = -11.18 USD
- 2026-06-09 SELL MU suppressed: hold-through -31.59 (-4.70% on 672 USD) minus bought-leg(066570.KS) -23.07 = -8.52 USD
- 2026-06-09 SELL SNDK suppressed: hold-through -1.20 (-0.20% on 595 USD) minus bought-leg(DELL) +14.73 = -15.93 USD
- 2026-06-10 SELL DELL suppressed: hold-through +0.00 (+0.00% on 730 USD) minus bought-leg(SNDK) +0.00 = +0.00 USD

Net counterfactual P&L delta (ON minus OFF): -270.86 USD
Suppressed turnover: 15,080.43 USD of 24,785.95 USD recorded (60.8%)

## Metrics (window, daily snapshots)

| arm | Sharpe(ann) | return % | maxDD % |
|---|---|---|---|
| OFF (recorded) | -1.27 | -0.81 | 3.45 |
| ON (counterfactual) | -3.16 | -1.94 | 3.49 |

sharpe_diff_test (UNDERPOWERED, T=8): {'delta': 0.0, 'p_one_sided': 1.0, 'ci_low': 0.0, 'ci_high': 0.0, 'sr_a': 0.0, 'sr_b': 0.0, 'se': 0.0, 'n': 8, 'n_boot': 2000}

## Disclosed limitations
- Decision-level replay through the production `_compute_swap_candidates`; full multi-cycle path dependence (a suppressed swap changing later candidate streams) is NOT re-simulated (Balch et al.).
- Counterfactual NAV applies the net pair delta from the first suppression day; intraday path within the window is approximate.
- T is ~9 trading days: all risk-adjusted numbers are descriptive, not inferential.
