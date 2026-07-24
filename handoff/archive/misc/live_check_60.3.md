# live_check_60.3 -- Decision-input integrity for non-USD markets (AW-9)

**Step:** 60.3 (phase-60, P1). **Date:** 2026-06-11. **Burn:** one live lite analysis for 005930.KS on the $0 flat-fee CC rail (trader call succeeded; the risk-judge CLI call flaked once -- exit 1 -> default conservative sizing, the documented fail-open; disclosed) + BQ reads. ~$0 metered.

## A. Before/after rendered prompts for the KRW fixture (criterion 1+3, verbatim from the production renderer; fixture = the persisted 06-09 066570.KS values: marketCap 44,540,606,021,632 KRW, pe 0.0)

```
=== BEFORE (flag OFF -- the away-week rendering) ===
Price: $64500.00 | Market Cap: $44540.6B | P/E: 0.0

=== AFTER (flag ON, FX available @ 0.000720) ===
Price: $46.44 (converted from KRW 64,500 @ 0.000720 KRW/USD) | Market Cap: $32.1B | P/E: 0.0
Data as-of: 2026-06-09T06:30:00+00:00 (exchange-local close; quote is 55.0h old -- NOT live)

=== AFTER (flag ON, FX unavailable -- the exact away-week state) ===
BLOCKED pre-LLM, flags: ['implausible_market_cap', 'currency_unverified']
(no prompt is built; no LLM is called; the candidate returns HOLD/REJECT in code)
```

The BEFORE line is the corruption the Risk Judge called "physically impossible... KRW/USD unit error" on 06-09 while the BUY executed anyway (stopped out -9.7%). The AFTER lines are truthful ($32.1B is LG Electronics' real converted cap) and staleness-honest (criterion 3: the as-of line replaces presented-as-live).

## B. Unit-test outputs (criteria 1-4, verbatim)

```
$ python -m pytest backend/tests -k 'prompt_fx or lite_prompt or 60_3' -q
13 passed, 810 deselected, 1 warning in 3.72s     (exit 0)
$ python -m pytest backend/tests -q
805 passed, 12 skipped, 6 xfailed, 1 warning in 77.04s   (exit 0)
```

Key tests: `test_60_3_regression_066570_away_week_state_blocks_in_code` (the 06-09 case -> blocking flags -> excluded IN CODE), `test_60_3_claude_analyzer_blocks_pre_llm` (end-to-end through the real `_run_claude_analysis` with a poisoned LLM rail that raises if touched -- the block fires BEFORE any LLM call: prose-only flagging is dead), `test_60_3_lite_prompt_krw_no_dollar_labeled_magnitude` (regex: no '$' before a KRW-scale number), `test_60_3_prompt_fx_us_byte_identity_both_flag_states` (US line byte-identical legacy == OFF == ON), `test_60_3_prompt_fx_flag_off_krw_renders_legacy` (do-no-harm locked).

FULL-suite note (honesty): the first full run had 1 failure in `test_phase_23_2_7_red_line_nav_match` (a live-BQ NAV-comparison test, unrelated surface); it passes in isolation and the full re-run is green 805/12/6 -- a live-data ordering flake, not a 60.3 regression.

## C. Post-fix BQ MCP row for a KR ticker (criterion 4; job_5z0CvcZ8VjBr0-K8sfQQ6D1vv4qG)

Live run 2026-06-11 (the REAL `_run_claude_analysis` + `_persist_analysis` against the live stack, **flag OFF = live config untouched**; provenance fields are UNGATED additive observability):

| field | value |
|---|---|
| ticker / path | 005930.KS / lite |
| recommendation / score | BUY / 7.0 |
| market_data.price (native) | 299000 (KRW) |
| market_data.currency | **KRW** |
| market_data.price_usd | **195.481** |
| market_data.market_cap_usd | **1.2836e12** |
| market_data.fx_rate | **0.00065378** |
| market_data.as_of | **2026-06-11T06:30:13+00:00** (the KRX close -- quote honestly stamped) |
| market_data.integrity_flags | [] (values internally consistent: 299,000 x 0.000654 = 195.5; $1.28T cap is under the $10T ceiling) |

Every future lite row -- US or KR, flag ON or OFF -- now carries unit-auditable provenance.

## D. In-code enforcement design (criterion 2 -- "prose-only flagging is a FAIL")

Deterministic pre-check (`backend/services/data_integrity.py::check_data_integrity`) runs in BOTH lite analyzers after the data fetch and BEFORE any LLM call. Blocking flags (implausible_market_cap > $10T post-normalization; currency_unverified -- non-US with no FX rate; currency_mismatch suffix-vs-info.currency) return a HOLD/REJECT analysis dict in code (GuardAgent chokepoint pattern, arXiv:2406.09187): HOLD is not in _BUY_RECS so decide_trades cannot buy it; recommended_position_pct 0; $0 LLM cost; the block reason lands in risk_assessment + market_data.integrity_flags (BQ-auditable). Tag-only flag: missing_pe_large_cap (P/E exactly 0 on an established large-cap = missing-data artifact -- the 066570.KS row's pe=0.0). Integrity-blocked rows count toward the 56.2 degraded-scoring guard DELIBERATELY (widespread blocks should alarm).

The unit-broken `market_cap > 5e9` BUY rule in the prompts (5e9 KRW = $3.6M -- every KR ticker passed) is cured by presentation: flag ON renders USD values, so the rule's threshold compares the correct unit. The risk-judge template receives the USD-true `market_cap_b` under the same flag (its "$2B micro-cap" heuristic becomes correct).

## E. Operator promotion decision (NEVER auto-applied)

> **PENDING** -- operator: reply `60.3 FLAG: ON` or `60.3 FLAG: KEEP OFF`. The flag is `paper_data_integrity_enabled` (default OFF; US prompts byte-identical in BOTH states -- tested; KR prompts only change when YOU promote).

## F. Do-no-harm evidence

Flag default OFF (test); US prompt line byte-identical legacy == OFF == ON (test); KR flag-OFF renders the historical line verbatim (test); provenance fields additive-only (legacy market_data keys intact -- test); stop-loss/sell paths untouched; no live flag flips (the live evidence run used the live OFF config).
