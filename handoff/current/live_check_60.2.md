# live_check_60.2 -- Churn-engine fix: swap sentinel + delta scale (AW-5)

**Step:** 60.2 (phase-60, P0). **Date:** 2026-06-11. **Burn:** $0 (BQ reads + yfinance closes only; no LLM calls).

## A. Regression-test output (criterion 1+2+3, verbatim)

```
$ python -m pytest backend/tests -k 'swap or sentinel or reeval or 60_2' -q
17 passed, 793 deselected, 1 warning in 2.33s     (exit 0; 9 pre-existing + 8 new)
$ python -m pytest backend/tests -q
792 passed, 12 skipped, 6 xfailed, 1 warning in 77.60s   (exit 0)
```

Key tests: `test_60_2_sentinel_regression_06_09_swap_fires_with_flag_off` (locks the defective behavior: equal-score candidate displaces a day-old unanalyzed holding at ~70,000% sentinel delta), `test_60_2_sentinel_eliminated_with_flag_on`, `test_60_2_away_week_pattern_impossible_by_construction` (NO candidate score -- 7.0/9.0/10.0 -- can displace an unanalyzed holding with the flag ON: exclusion is structural), `test_60_2_delta_boundary_on_true_1_to_10_scale` (6-vs-5 = 20% < 25% no fire; 7-vs-5 = 40% fires on true evidence; bar UNCHANGED), `test_60_2_real_score_deltas_identical_off_vs_on` (clamp inert for scores >= 1.0), `test_60_2_composes_with_57_1_binding_reject_gate`.

## B. BQ MCP rows -- the 06-08/06-09 sentinel swaps (fixture basis; job_t8OBtgMfUjeBfvZfiotfP3Q5TmXa)

| ts | ticker | action | reason | USD | holding_days | realized_pnl_pct |
|---|---|---|---|---|---|---|
| 06-08T18:11 | STX | SELL | swap_for_higher_conviction | 614.69 | 2 | +3.04 |
| 06-08T18:11 | DELL | SELL | swap_for_higher_conviction | 727.05 | 2 | +1.56 |
| 06-08T18:11 | SNDK | BUY | swap_buy | 597.24 | | |
| 06-08T18:12 | MU | BUY | swap_buy | 716.69 | | |
| 06-09T18:12 | **MU** | **SELL** | swap_for_higher_conviction | 671.74 | **1** | **-6.27** |
| 06-09T18:12 | **SNDK** | **SELL** | swap_for_higher_conviction | 594.78 | **1** | -0.41 |
| 06-09T18:12 | 066570.KS | BUY | swap_buy | 238.40 | | |
| 06-09T18:12 | DELL | BUY | swap_buy | 715.19 | | |
| 06-10T18:39 | DELL | SELL | swap_for_higher_conviction | 729.92 | 1 | +2.06 |
| 06-10T18:40 | SNDK | BUY | swap_buy | 476.60 | | |

The mechanism verbatim: MU bought 06-08 18:12 as a swap_buy, swap-SOLD 06-09 18:12 (holding_days=1, -6.27%) -- the buy-day analysis existed but the next cycle's swap path scored it sentinel 0.0.

## C. ON-vs-OFF replay (criterion 4) -- scripts/replay/replay_60_2_swap_fix.py, full report at handoff/current/replay_60_2_results.md

- **ARM A fidelity (flag OFF): 12/13 recorded swaps reproduced** through the production `_compute_swap_candidates`. The single non-reproduction (05-29 KEYS->STX) is an away-week persistence gap: the bought ticker STX has NO persisted analysis row that day (the 100%-lite-fallback period also dropped persists), so the reconstruction lacks its true score -- disclosed, not a code defect.
- **ARM B (flag ON): 11/13 swaps SUPPRESSED -- 10 of them sentinel-driven** (sold ticker had NO same-day analysis: the away-week engine), 1 score-based (-100% delta). **2 SURVIVE on true evidence** (06-01 DELL->000660.KS 75% delta; 06-03 STX->AMD 100% delta) -- the fix removes fabricated-evidence swaps only.
- **The 3 named round trips:** MU 06-08->06-09 SUPPRESSED (sentinel); SNDK 06-08->06-09 SUPPRESSED (sentinel); DELL chain: the 06-01 leg SURVIVES (true 75% delta), the 06-03/06-08->06-09/06-10 legs SUPPRESSED (sentinel).
- **Counterfactual ledger (one-step, currency-neutral pct moves x USD notionals):** net delta **-270.86 USD** (ON minus OFF) over the window; suppressed turnover 15,080 of 24,786 USD recorded (60.8%).
- **Metrics:** OFF (recorded): Sharpe -1.27, return -0.81%, maxDD 3.45% | ON (counterfactual): Sharpe -3.16, return -1.94%, maxDD 3.49%. sharpe_diff_test output degenerate at T=8 (all-zero stats) -- reported verbatim in the replay file; explicitly NOT a gate.

### Honest reading (both sides, for the operator)

1. **The mechanism is indefensible:** 10/13 recorded swaps fired against FABRICATED 0.0 conviction (BQ-verified). 55.1 measured the realized round trips at net -$132 with 81.4% weekly turnover.
2. **BUT in this specific 9-day falling window, the churn got lucky:** the swapped-OUT tickers kept falling (KEYS -13%, HPE -17%, AMD -13% hold-through), so random rotation out of losers beat holding by ~$271 one-step. The fix would have held those positions (their stop-losses still protect at -8/-10%; stops are NOT touched by this flag).
3. **The one-step ledger UNDERSTATES the fix's benefit structurally:** each bought leg became the NEXT cycle's sentinel bait (the chain DELL->000660.KS->HPE->DELL->005930.KS->STX->SNDK/MU->066570.KS/DELL->SNDK is ONE cascading chain); suppressing the first link suppresses the chain, which one-step accounting cannot capture (disclosed limitation, Balch et al.).
4. **Post-60.1 interaction:** the sentinel only binds when a holding lacks a same-cycle analysis -- the away week was 100% lite-fallback with persistence gaps. With the deep pipeline restored (60.1), holdings get scored more reliably, so exclusion binds RARELY in normal operation; the flag mainly removes the pathological failure mode.
5. n=1 regime, T~9 days: all window metrics are descriptive, not inferential.

## D. Operator promotion decision (criterion 4 -- NEVER auto-applied)

> **PENDING** -- operator: reply `60.2 FLAG: ON` or `60.2 FLAG: KEEP OFF` (or in your own words). The flag is `paper_swap_churn_fix_enabled` (default OFF; OFF path byte-identical, locked by 17 tests).

## E. Do-no-harm evidence

- Flag default OFF (`test_60_2_flag_defaults_off`); OFF path byte-identical (sentinel + 0.01 epsilon preserved verbatim; all 9 pre-existing swap/sentinel/reeval tests pass unchanged; the TECH1 [0,1]-score fixture still fires).
- Stop-losses, sell-signals, downgrades, the 57.1 binding gate: untouched (compose test passes with both flags ON).
- No live flag flips in this step; no settings changed on the running backend (code is deployed at the NEXT restart; the flag stays OFF either way).
