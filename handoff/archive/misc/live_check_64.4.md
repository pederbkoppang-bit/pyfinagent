# live_check — step 64.4 (Multi-market fixture-replay e2e)

## Green run transcript (immutable command)

```
$ source .venv/bin/activate && python -m pytest backend/tests \
    -k 'multi_market_e2e' -q -m 'not requires_live'
.....                                                                    [100%]
5 passed, 1144 deselected, 1 warning in 3.57s
# exit 0  (the 1 @pytest.mark.requires_live smoke is deselected by -m 'not requires_live')
```

## Per-market funnel counts (criterion 1: all >0 for US/KR/EU — anti-vacuous)

```
US: universe=2 screened=2 ranked=2 order_intent=2 market=['US', 'US']
KR: universe=2 screened=2 ranked=2 order_intent=2 market=['KR', 'KR']
EU: universe=2 screened=2 ranked=2 order_intent=2 market=['EU', 'EU']
```

Every funnel stage (universe → screened → ranked → order-intent) is >0 for all three markets, and the emitted
`TradeOrder.market` matches the market (derived from the ticker suffix). The seam driven is
`screen_universe` (yf.download mocked) → `rank_candidates` → `decide_trades` — the PURE path, NOT the full loop
(the loop's calendar gate would flake the funnel to 0 on weekends).

## EU "under the 65.2 thresholds via test flag" is LOAD-BEARING (criterion 1)

`test_64_4_multi_market_e2e_eu_funnel_under_lowered_thresholds` asserts EU tickers with sub-default volume (50k):
- DEFAULT thresholds (`min_avg_volume=100_000`) → screened == **0** (fails).
- Lowered test-flag kwargs (`min_avg_volume=10_000`, `min_price=1.0`) → screened > 0, ranked > 0, order-intent > 0.
So the lowered-threshold override genuinely drives the EU funnel (not a tautology). (Interpretation: "via **test**
flag" = a test-only kwarg override; the production 65.2 flag does not exist yet and 64.4's DAG dep is 66.2, not 65.2.)

## Currency invariants (criterion 2)

`test_64_4_multi_market_e2e_currency_invariants`: KR add-on avg_entry ~70000 (KRW-scale, <500 tol); EU ~150
(EUR-scale, <2 tol). Reuses the 64.3 `paper_avg_entry_fx_fix_enabled` + patched fx pattern.

## requires_live variant (criterion 3)

The file has exactly **1** `@pytest.mark.requires_live` test (`..._live_smoke`, hits real yf.download). Excluded from
the default/CI run by `-m 'not requires_live'`:
```
$ python -m pytest backend/tests -m requires_live --co -q | tail -1
12/1149 tests collected      # was 11 -> +1 (the intentional live smoke)
```

## Method / boundaries
Pure pytest, synthetic fixtures, no network in the default run. `uvx ruff check` → All checks passed. NO production
code changed (git status = only the 1 test file + handoff). $0; live book untouched; historical_macro FROZEN.
