---
name: sharpe-difference-test-methodology
description: phase-52.3 -- the RIGOROUS test for a Sharpe IMPROVEMENT (delta) is paired Ledoit-Wolf + stationary bootstrap, NOT DSR; DSR/PBO traps at small T; reusable analytics helpers
metadata:
  type: project
---

When a backtest measures one strategy's Sharpe as +X better than a baseline on the
SAME periods (e.g. the phase-52.1 52wh tilt: base ~1.39, tilt ~1.44, +0.05 over ~47
monthly rebalances), the rigorous "is the DELTA real?" test is the **paired
Ledoit-Wolf (2008) SR-difference test with a stationary-bootstrap p-value** -- NOT the
Deflated Sharpe Ratio.

**Why:** DSR (Bailey-Lopez de Prado 2014) deflates an ABSOLUTE max-of-N Sharpe for
selection bias; it answers "is 1.44 a search fluke", not "is +0.05 vs baseline real".
Smoke-proof (this repo): `compute_deflated_sharpe(1.44,N=3,T=47)=1.0` and
`(1.39,N=3,T=47)=0.9999` -- DSR literally cannot distinguish the two, because baseline
is also ~1.4. The +0.05 is a DIFFERENCE -> needs a paired test.

**The canonical difference test (Ledoit-Wolf, J.Emp.Fin 2008, econ.uzh.ch/.../iewwp320.pdf):**
delta=SR1-SR2; studentized d=|delta|/s(delta); stationary-bootstrap (Politis-Romano 1994)
p-value `(#{d~*,m>=d}+1)/(M+1)`, M>=499 (use 1000+ since $0). Resample JOINT (a_i,b_i) rows
to preserve cross-corr + autocorr. Plain paired t-test on the diff series UNDERSTATES SE
(returns autocorrelated + fat-tailed) -> only a quick companion, never the gate. LW abstract:
the older Jobson-Korkie/Memmel test is "not valid when returns have tails heavier than the
normal distribution or are of time series nature."

**N_eff for the secondary DSR check:** never use the raw count of configs tried -- they are
correlated (same composite, same universe). DSR Appendix 3 / Wikipedia: correlation matrix ->
distance -> cluster -> N_eff = #clusters. For the 5 phase-52 configs N_eff~2-3.

**Reusable repo helpers (all $0, numpy+scipy already deps):**
- `backend/backtest/analytics.py:239` compute_deflated_sharpe(observed_sr,num_trials,variance_of_srs,skewness,kurtosis,T) -- secondary absolute-Sharpe check only.
- `backend/backtest/analytics.py:184` compute_pbo(pnl_matrix,S=16) -- CSCV; at small T (e.g. 47) **use S=6 not 16** (S=16 needs T>=32 and leaves too few rows/subset). Columns must be COMPETING configs.
- `backend/backtest/analytics.py:125` compute_sharpe(returns,periods_per_year=12) for monthly.
- `backend/agents/mcp_servers/risk_server.py:133` pbo_check veto wrapper (>0.5).
- **NOT in repo:** Ledoit-Wolf / stationary bootstrap (grep=0) -- must be added (~40-60 LOC).
- Imports standalone in ~1.8s; analytics.py pulls in BacktestResult from backtest_engine.

**McLean-Pontiff (J.Finance 2016) haircut:** new in-sample edges decay ~26% OOS / ~58%
post-publication, MORE for higher in-sample edges -> a knife-edge p=0.049 should NOT enable;
require the bootstrap CI lower bound for delta > 0 with margin.

**Why:** phase-52.3 needed a defensible enable/reject gate for promoting the highest earner
(element 2 "cited research basis"); the DSR-only instinct was wrong for an improvement delta.
**How to apply:** any future "config B beats baseline by +X Sharpe, enable it?" decision --
gate on paired LW bootstrap p<0.05 + positive CI lower bound (primary), DSR>=0.95 at clustered
N_eff + PBO<=0.5 (corroborating). See [[project_pbo_single_strategy_cpcv]] (CSCV needs N>=2
competing configs) and [[project_strategy_rotation_seed_set]] (deflate by N_eff not raw N).
