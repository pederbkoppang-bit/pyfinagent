---
name: swap-churn-engine-60-2
description: phase-60.2 churn engine — sentinel/stamp/delta-scale facts, NO swap-path replay exists, fixture [0,1]-scale trap, BQ-confirmed round trips
metadata:
  type: project
---

AW-5 churn mechanism (HEAD 2026-06-11): swap path values holdings ABSENT from
same-cycle `holding_lookup` at conviction 0.0 (portfolio_manager.py:476-483);
`execute_buy` stamps last_analysis_date=now (paper_trader.py:304,:328); re-eval
gate waits `paper_reeval_frequency_days=3` (autonomous_loop.py:791-804,
settings.py:322, `.days` TRUNCATES + cycle-time drift → effectively 3-4d) → fresh
BUYs are sentinel-bait next cycle. Delta = (cand-h)/max(|h|,0.01)*100 vs 25%
(portfolio_manager.py:525-534); sentinel case = 70,000%. Code comment :526-531
believes scores are [0,1]; **lite scores are 1-10 integers**; settings.py:293
description documents a 1.0 clamp the code dropped — restoring the 1.0 clamp is
byte-identical for all real scores (>=1.0); the SENTINEL is the load-bearing bug.

**Why:** 59.3 blinded audit pinned this as the primary money bleed (81.4% weekly
turnover, MU -6.3% 1-day round trip). BQ-confirmed rows (financial_reports.
paper_trades): MU buy 06-08 18:12:05 @954.385 → sell 06-09 18:12:08 @894.53
(-$44.95); SNDK 06-08→06-09 (-$2.46, re-bought 06-10 @1656.37 ABOVE exit); DELL
06-05 19:04 → 06-08 18:11 (2d23h < 3d, +$11.17) and 06-09→06-10 (+$14.73,
PROFITABLE — report suppressed-profitable honestly). All exits literally
"swap_for_higher_conviction".

**How to apply:**
- **NO existing tool replays the swap path** — decide_trades consumers are only
  autonomous_loop.py:1155, tests, 2 go-live drill scripts;
  strategy_backtest_adapter.py:43 docstring confirms best_params NOT threaded.
  rebalance_band.py:22 apply_no_trade_band = 53.1's REJECTED monthly-universe
  band. ON-vs-OFF must be a NEW decision-replay event study from persisted BQ
  (57.1 precedent); validation arm = flag-OFF reproduces recorded orders.
- LOCF reader exists: bigquery_client.get_report(ticker) :303-358 (latest
  analysis_results row incl final_score).
- Fixture trap: test_portfolio_swap.py fixtures use [0,1] scores (0.55-0.85);
  under a 1.0-clamp the TECH1 case (0.82 vs 0.58 → 24%) stops firing — never
  apply new formula to flag-OFF path; ON tests need 1-10 integer fixtures.
- -k 'swap or sentinel or reeval' collects 9 tests incl. 4 UNRELATED name
  collisions (heartbeat "sentinel", agent-map "swappable").
- Boundary (53.1/55.3 BINDING): sentinel/stamp/scale fixes = correctness;
  forbidden = raising min_delta, absolute-point floors beyond documented intent,
  tenure shields ("no swap if held < N days" = time-domain band). LOCF equal-
  score no-swap is evidence symmetry, not tenure protection.
- Sharpe delta at T~12 cycles is UNDERPOWERED — sharpe_diff_test
  (analytics.py:239) reported descriptively; see
  [[sharpe-difference-test-methodology]].
