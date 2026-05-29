---
name: pbo-single-strategy-cpcv
description: phase-48.2 crux — per-strategy PBO from ONE backtest is impossible via CSCV (needs N>1 competing configs as columns); use CPCV multi-path Sharpe distribution OR small in-strategy param grid instead
metadata:
  type: project
---

# Per-strategy PBO from a single backtest (phase-48.2 BacktestEngine adapter)

The crux of phase-48.2: each rotation strategy gets ONE walk-forward backtest;
the PromotionGate (`backend/autoresearch/gate.py:21-22`) needs `dsr>=0.95 AND
pbo<=0.20`. Can a meaningful PER-STRATEGY PBO come from one backtest?

**Answer: NOT via classic CSCV.** Verbatim from Bailey/Borwein/Lopez de
Prado/Zhu "The Probability of Backtest Overfitting" (davidhbailey.com PDF,
Algorithm 2.3, read via pdfplumber 2026-05-29): matrix M is "(T x N)" where
"each column n = 1,...,N represents a vector of profits and losses ... associated
with a particular model configuration tried by the researcher." The IS/OOS split
is over ROWS (time, partitioned into S submatrices, C(S,S/2) combinations). So
**columns MUST be competing configurations, not time windows.** Feeding one
strategy's per-window OOS series as columns is a category error. Paper: "N must
be large enough ... If N is too small, omega will take only a very few values."

Internal `compute_pbo` (`backend/backtest/analytics.py:184-236`) enforces this:
`if N < 2 ... return 0.0` (line 205) -> one series => PBO=0.0 => FALSE PASS.
This is the trap to avoid.

**Recommended approach for the adapter (two valid options):**
1. SMALL IN-STRATEGY PARAM GRID (simplest, matches the producer's DEFERRED note
   at `strategy_candidate_producer.py:24-32`): for each rotation strategy run K
   (~8-16) param variants AROUND the seed (e.g. holding_days, sl_pct, z-thresh),
   stack their daily-return series as the (T x K) matrix, call `compute_pbo`.
   PBO then measures: does this strategy's IS-best variant degrade OOS? Valid &
   matches the canonical method. Cost: K backtests/strategy (mitigate via
   `skip_cache_clear=True` warm engine).
2. CPCV MULTI-PATH (de Prado AFML Ch.12; `cpcv_folds` ALREADY in gate.py:42):
   one model, many backtest PATHS. phi[N,k] = C(N,k)*k/N paths (N=6,k=2 -> 15
   splits, 5 paths). Yields a DISTRIBUTION of OOS Sharpe per strategy -> use
   10th-pctile PSR or path-Sharpe variance as robustness. CPCV is the 2024-2026
   consensus winner (Arian/Norouzi/Seco: lowest PBO, highest DSR vs KFold/WF).
   But canonical PBO formula still wants competing columns; CPCV gives a
   robustness distribution, not the rank-degradation PBO directly.

**DSR num_trials for a small strategy set:** N = number of INDEPENDENT trials,
NOT raw count. `compute_deflated_sharpe(analytics.py:239)` uses the full two-term
E[max SR] = sqrt(V)*[(1-g)*Z^-1(1-1/N) + g*Z^-1(1-1/(Ne))], g=0.5772. For ~5
correlated seeds, plain N=5 OVER-deflates DSR (the SAFE direction; producer
docstring confirms). Effective-N clustering (ONC) is the de Prado fix, deferred.
`variance_of_srs` defaults 0.5 in code but generate_report computes it from
per-window Sharpes (`analytics.py:548-549`) — needs >1 window or falls back 0.5.

See [[strategy-rotation-seed-set]] (deflate by N_eff not raw N) and
[[strategy-rotation-infra]] (selector/gate already built).
