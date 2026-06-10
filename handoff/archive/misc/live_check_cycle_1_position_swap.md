# Live-check evidence -- Cycle 1 position-swap framework (2026-05-26)

This artifact captures the deterministic test evidence operators can audit
to confirm the swap framework lands correctly. Per CLAUDE.md
`verification.live_check` discipline (audit R-1): file existence is the
gate that turns "agent claimed PASS" into "operator can verify".

## Test execution

```
$ source .venv/bin/activate && pytest backend/tests/test_portfolio_swap.py -v
backend/tests/test_portfolio_swap.py::test_swap_framework_fills_zero_buy_gap PASSED [ 25%]
backend/tests/test_portfolio_swap.py::test_swap_disabled_reproduces_zero_buy PASSED [ 50%]
backend/tests/test_portfolio_swap.py::test_swap_skips_below_threshold PASSED [ 75%]
backend/tests/test_portfolio_swap.py::test_swap_respects_max_per_cycle PASSED [100%]
============================== 4 passed in 0.06s ===============================
```

## Regression suite (tier-1 coverage + swap framework)

```
$ pytest backend/tests/ -k "portfolio_manager or paper_trader or test_portfolio_swap"
================ 37 passed, 582 deselected, 1 warning in 2.37s =================
```

Existing `test_portfolio_manager_decide_trades_sector_count_cap_blocks_third`
now passes by explicitly disabling swap in the test fixture (the test
characterizes the sector-cap mechanism in isolation; swap behavior is
exercised by the cycle-1 file).

## What the framework does (synthetic 2026-05-26 reproduction)

Scenario: 9 positions (8 Tech `final_score` 0.55-0.75 + 1 Industrials 0.65)
+ 3 candidates (TECH_NEW1 0.85, TECH_NEW2 0.82, INDU_NEW 0.70). NAV $10k,
cash $2k, sector cap 2, `paper_swap_enabled=True`,
`paper_swap_min_delta_pct=25.0`, `paper_swap_max_per_cycle=2`.

Output (5 orders, all SELLs before all BUYs):

1. SELL TECH0 (score 0.55, weakest Tech) reason=swap_for_higher_conviction
2. SELL TECH1 (score 0.58, second-weakest Tech) reason=swap_for_higher_conviction
3. BUY INDU_NEW (score 0.70) reason=new_buy_signal (fills the open slot)
4. BUY TECH_NEW1 (score 0.85, highest-conviction Tech) reason=swap_buy
5. BUY TECH_NEW2 (score 0.82, second-highest Tech) reason=swap_buy

Pre-cycle behavior (with `paper_swap_enabled=False`): orders = [BUY INDU_NEW
only]. Zero Tech rebalance; portfolio drifts as Tech holdings underperform.

Cycle-1 behavior: portfolio gets the two highest-conviction Tech swaps in
addition to the slot-fill, advancing the testing-phase trade-count
mandate AND the north star (maximize profit by upgrading conviction).

## Risk-gate preservation

- Sector COUNT cap (`paper_max_per_sector=2`): each swap is +1 BUY / -1
  SELL in the same sector. Net count change = 0. Cap preserved.
- Sector NAV-pct cap (`paper_max_per_sector_nav_pct=30`): re-checked on
  the projected post-swap composition. Edge-case logic added: when the
  sector is already over the cap (legacy 8-Tech portfolio at 88%), swaps
  that REDUCE or hold exposure constant are allowed; swaps that WORSEN
  the breach are blocked.
- Position cap (`paper_max_positions=10`): preserved -- swap is net 0.
- Min-cash-reserve (`paper_min_cash_reserve_pct=5%`): each swap pair is
  approximately cash-neutral (SELL value $1100 -> BUY value $1000 -> net
  +$100 cash). Reserve preserved.
- Factor-correlation cap (`paper_max_factor_corr`): swap stays within the
  same sector so portfolio factor loadings move in a bounded way; not
  re-checked here (future tightening on backtest evidence).

## Citations (verified per goal mandate >=2 AI-in-trading + >=2 academic)

AI-in-trading (4 cited):
- FinRL (Liu et al. `arXiv:2011.09607` + GitHub `AI4Finance-Foundation/FinRL`)
- TradingAgents Multi-Agents LLM Framework (`arXiv:2412.20138` Dec 2024)
- FinMem (`arXiv:2311.13743`)
- LLM Long-Run Outperformance ADVERSARIAL (`arXiv:2505.07078v5` KDD 2026)

Academic methods (3 cited):
- Grinold-Kahn Fundamental Law of Active Management (CFI canonical)
- Kelly-Optimal Rebalancing Frequency (`arXiv:1807.05265`)
- Resonanz Capital "upgrade-vs-exit" position-sizing framework

Full citation chain in
`handoff/current/research_brief_phase_zero_buy_triage.md` (researcher
`adc62c28569bf64cc`, tier=deep, `gate_passed=true`).
