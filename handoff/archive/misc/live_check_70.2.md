# live_check — step 70.2 (S2: soft cross-sector diversification)

Not a UI step — the live_check is the ON-vs-OFF evidence from the $0, macro-free ablation replay
(`scripts/ablation/sector_neutral_replay.py`) + deterministic tests. `historical_macro` untouched (the replay
uses only free yfinance prices + a Wikipedia sector map — no LLM, no BQ, no macro, no optimizer).

## OFF analyzed-set = monosector (reproduces today) ; ON = spans >=2 sectors

Deterministic (`backend/tests/test_phase_70_2_soft_diversity.py::test_min_k_slice_reproduces_and_diversifies`):
a monosector-heavy candidate list (5 Technology + 1 Energy + 1 Health Care) →
- **OFF** (plain top-5 analyze slice, `paper_min_k_sectors_analyzed=0`): **1 sector** — reproduces the S2 funnel.
- **ON** (`_min_k_sector_slice`, K=3): **3 distinct sectors** {Technology, Energy, Health Care}, best names first.

## ON-vs-OFF OOS basket breadth + Sharpe (47 monthly rebalances, 2022–2025, S&P 500)

| config | ann_Sharpe | Δ vs base | avg_sectors |
|---|---|---|---|
| baseline (OFF) | 1.344 | — | 4.71 |
| soft_w0.10 (ON) | 1.520 | +0.176 | 5.96 (+1.25) |
| soft_w0.20 (ON) | 1.543 | +0.200 | 6.73 (+2.02) |
| soft_w0.30 (ON) | 1.578 | +0.234 | 7.31 (+2.60) |
| sector_neutral (HARD, rejected) | 1.226 | -0.117 | 10.00 |

ON raises OOS Sharpe AND breadth at every tested w; hard neutralization lowers Sharpe (design justification).
Paired monthly returns dumped to `handoff/current/_70_2_soft_diversity_replay.json` for the activation-gate
DSR (>=0.95, de-annualized per phase-69.2) + PBO (<=0.5) computation.

## OFF byte-identical

`test_soft_off_and_w0_byte_identical`: soft flag OFF (and `w=0.0` ON) → identical ticker order + composite
scores as the no-param call, no `composite_score_raw` side-channel. Min-K K=0 → plain slice; Unknown-exempt
OFF → caps enforced as today. Every lever defaults to the identity path (DARK-until-token).

## Command evidence
```
verification command -> exit 0
pytest backend/tests/test_phase_70_2_soft_diversity.py -> 7 passed
ablation replay -> exit 0, 47 rebalances, dump written
```
