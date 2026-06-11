# live_check_57.1 — Binding RiskJudge gate: live evidence

**Step:** 57.1 (phase-57 FEATURE; operator reply verbatim `PHASE-57: FEATURE`, 2026-06-11; install commit `af4aa8d6`). **Required shape (masterplan):** the event-study table (BQ query + 3 rows + net P&L + selection-bias caveat) + the test outputs + confirmation the flag is default-OFF and unflipped.

## A. Event study — the 3 executed-REJECT BUYs (criterion 5; $0, stored data)

**Query 1** (run live 2026-06-11): `SELECT trade_id, created_at, ticker, action, reason, risk_judge_decision, total_value FROM financial_reports.paper_trades WHERE action='BUY' AND risk_judge_decision='REJECT' ORDER BY created_at` → exactly **3 rows, all `reason=swap_buy`** (the topology finding: a main-loop-only gate would have missed every one):

| ticker | buy (UTC) | buy value (USD, post-restatement) | reason | trade_id |
|---|---|---|---|---|
| HPE | 2026-06-02 19:18:58 | 245.04 | swap_buy | 905e08b1 |
| DELL | 2026-06-03 19:05:19 | 246.67 | swap_buy | 4f2d59e6 |
| 066570.KS (LG Electronics) | 2026-06-09 18:12:39 | 238.40 | swap_buy | a72a164e |

**Query 2** (`paper_round_trips` joined on ticker + entry-day, run live 2026-06-11):

| ticker | entry | exit | realized P&L $ | realized % | hold | exit reason |
|---|---|---|---|---|---|---|
| HPE | 06-02 | 06-03 | **−0.81** | −0.33% | 0d | swap_for_higher_conviction |
| DELL | 06-03 | 06-04 | **+0.54** | +0.22% | 0d | swap_for_higher_conviction |
| 066570.KS | 06-09 | 06-10 | **−23.18** | −9.68% | 1d | **stop_loss_trigger** |
| **NET** | | | **−$23.45** | | | |

**Selection-bias caveat (mandatory honesty):** n=3 is **descriptive of these three specific decisions, not the gate's expected value**. The sample is conditioned post-hoc on (a) REJECT verdicts that (b) executed via the swap path and (c) closed within the observed window. The gate's true EV depends on the full distribution of REJECT verdicts — including ones the judge gets wrong, where blocking would FORGO a gain. **No annualized/Sharpe extrapolation is made or implied.** (Ahern 2006 selection-bias methodology; arXiv:2511.15123 on counterfactual inconsistency.) The event study's role is regression-fixture witness + directional anecdote; EV is established only by post-flip OOS observation, which is the operator's later decision.

## B. Test outputs (criteria 1-4, verbatim)

```
$ python -m pytest backend/tests/test_phase_57_1_reject_binding.py -q
7 passed, 1 warning in 2.00s
$ python -m pytest backend/tests -k 'reject_binding or risk_judge_binding' -q     # the immutable selector
7 passed, 767 deselected, 1 warning in 2.37s
$ python -m pytest backend/tests -q                                                # full-suite regression
756 passed, 12 skipped, 6 xfailed, 1 warning in 78.22s
```

Per-criterion coverage:
- **C1 (regression fixture, both paths):** `test_reject_binding_main_path_off_emits_on_blocks` (flag OFF → advisory BUY emitted; flag ON → absent + blocked_out recorded) and `test_reject_binding_swap_path_off_emits_on_blocks` — the swap scenario reproduces the away-week vulnerability verbatim (flag OFF: the REJECT candidate swap-buys with `risk_judge_decision == "REJECT"` on the emitted order, exactly like HPE/DELL/LG; flag ON: blocked, and the next-ranked candidate takes the freed slot — budget reallocation by construction).
- **C2 (OFF byte-identity):** `test_off_identity_orders_no_reject_set` (flag ON == flag OFF on REJECT-free sets) + `test_off_identity_prompts_are_verbatim_constants` (builders return the IDENTICAL constant objects — `is` assertions — and rendered output equality). Default ships OFF: `Settings().paper_risk_judge_reject_binding is False` asserted in C1's test. **No live flag flip occurred** (settings.py Field default False; .env untouched; effective runtime value verified via the settings loader: `get_settings().paper_risk_judge_reject_binding == False`).
- **C3 (prompt-context ON):** `test_prompt_content_flag_on_real_cap_and_sector_line` — system prompt contains "exceed 30% of portfolio NAV in one sector", does NOT contain "10% of portfolio in one sector"; rendered template contains "Current portfolio context: invested-book sector weights: Technology 100.0%" from a fake-positions fixture. Plus `test_sector_context_all_cash_and_fallback_price` (all-cash + avg_entry_price fallback edges).
- **C4 (single-compute):** `test_analyzers_receive_precomputed_context_not_positions_fetch` — structural assertions: all three analyzers carry a `portfolio_context` parameter, none contains a `get_positions` call (source scan), and `run_daily_cycle`'s source contains the single `_build_portfolio_sector_context(positions)` compute site.

## C. Flag state + scope confirmation (criterion 6)

- `backend/config/settings.py::paper_risk_judge_reject_binding` — `Field(False, ...)` (default-OFF; description cites F-3/F-8 + SEC 15c3-5 rejection doctrine).
- Gate site: `portfolio_manager.py` candidate-build loop (the common ancestor of the main BUY loop and the swap path; SEC 15c3-5(d) non-bypassable placement). REJECT-only — APPROVE_REDUCED/HEDGED sizing untouched (ESMA hard-vs-soft tiering).
- Prompt builders + context threading behind the SAME flag (never bind on a blind judge — GuardAgent); flag-OFF returns verbatim constants.
- Observability: structured warning per blocked BUY + `summary["risk_judge_blocked"]` for the window's DoD-7 trail.
- NO live cycle was run by this step; NO LLM trading-cycle spend (all tests offline with fixtures).
