# Live Check — Step 69.3 (P1 signal integrity + $0 free-data lift)

Live signal-integrity fixes + a $0 free-data lift, all flag-gated default-OFF (live engine byte-identical
until the operator flips the flag). Evidence per the masterplan `live_check`: the sign-inversion test, the
100-headline parse test, the QMJ test, the regime-prompt capture proving INDPRO + net-liquidity, and the
ON-vs-OFF flag comparison.

## Test suite — `backend/tests/test_signal_integrity_69.py` (12 passed)

```
test_sign_safe_eliminates_inversion_on_negative_base   PASSED  # C1: neg-base +catalyst > neg-base -catalyst
test_sign_safe_off_byte_identical                      PASSED  # C1 do-no-harm: OFF == base*mult (grid)
test_sign_safe_positive_base_unchanged_intent          PASSED  # C1: +base reduces to base*mult
test_sign_safe_wired_into_a_real_overlay               PASSED  # C1: options_flow overlay ON/OFF flips rank
test_news_cap_no_longer_truncates_and_retries          PASSED  # C2: min(8192,...) gone; cap 48k; retry present
test_qmj_growth_assigned_before_read                   PASSED  # C3: revenue_growth_yoy assigned before read
test_indpro_now_in_fred_series                         PASSED  # C4: INDPRO in fred_data.SERIES (was dead)
test_regime_prompt_off_is_byte_identical               PASSED  # C4 do-no-harm: OFF == pre-fix prompt
test_regime_prompt_on_includes_indpro_and_netliq       PASSED  # C4: ON includes INDPRO + NET_LIQUIDITY
test_net_liquidity_unit_scaling                        PASSED  # C4: RRPONTSYD x1000 -> net = 5,900,000 M
test_flags_default_off                                 PASSED  # do-no-harm: flags exist, default OFF
test_net_liquidity_writes_no_bq                        PASSED  # do-no-harm: net-liq path writes NO BQ

12 passed in 0.04s
```

Ruff gate (qa.md §1a) on all 17 touched files + the test: **All checks passed!** (exit 0).

## ON-vs-OFF live ranking comparison (criterion 1) — $0, no LLM call

Two candidates with EQUAL negative base composite (drawdown regime); AAA gets a positive catalyst (boost),
BBB gets a negative catalyst (penalty):

```
  sign_safe_overlays=False -> AAA=-11.00  BBB=-9.00  higher-rank=BBB(-catalyst)     # INVERTED (the bug)
  sign_safe_overlays=True  -> AAA=-9.00   BBB=-11.00 higher-rank=AAA(+catalyst)     # FIXED
```
OFF ranks the NEGATIVE catalyst higher; ON ranks the POSITIVE catalyst higher — the inversion is eliminated,
live, at the real overlay call site (options_flow_screen.apply_options_surge_to_score).

## Regime-prompt capture proving INDPRO + net-liquidity (criterion 4) — $0, prompt-string render only

```
  OFF: INDPRO present=False  NET_LIQUIDITY present=False   (byte-identical to the pre-fix prompt: True)
  ON : INDPRO present=True   NET_LIQUIDITY present=True
    - INDPRO (IP): current=103.200 previous=102.9 trend=rising as_of=2026-07-01
    - NET_LIQUIDITY (Fed WALCL-TGA-RRP, $M): current=6100000 trend=rising as_of=2026-07-01 [rising -> risk_on lean]
```
(Rendered the regime prompt STRING; NO metered `ClaudeClient` regime call was made — $0.)

## Do-no-harm
- Flags `sign_safe_overlays` + `regime_net_liquidity` default-OFF → live ranking + regime prompt byte-identical.
- historical_macro FROZEN: `_fetch_net_liquidity` writes NO BQ (file cache + existing free FRED key).
- 1054 tests collect; the 6 ruff-fixed modules re-import OK. No conflict with phase-68 (overlays ≠ fills).

## Deferred / operator
- Final IC/ablation/optimizer validation of the ranking change + the net-liquidity feature waits on the
  historical_macro un-freeze token. Activation (flipping `sign_safe_overlays` / `regime_net_liquidity`) is
  the operator's call after reviewing this ON-vs-OFF evidence.

## Q/A verdict
Fresh Workflow structured-output Q/A (Opus) — see `evaluator_critique.md`.
