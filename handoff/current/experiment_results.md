# Experiment Results -- Step 60.3 (GENERATE)

**Step:** 60.3 -- Decision-input integrity for non-USD markets (AW-9). **Date:** 2026-06-11.

## What was built

1. **NEW `backend/services/data_integrity.py`** (pure, fail-open): `normalize_market_values` (REUSES fx_rates.get_fx_rate 6h cache + markets.market_for_symbol; as-of staleness from the quote's own regularMarketTime -- not a hardcoded close, KRX adds extended sessions 2026-06-29), `check_data_integrity` (blocking: implausible_market_cap > $10T post-normalization [NVDA $4.854T = largest real, 2x headroom], currency_unverified, currency_mismatch suffix-vs-info.currency per yfinance #2699; tag-only: missing_pe_large_cap), `render_market_lines` (US + flag-OFF byte-identical to the historical f-string; flag-ON non-US renders USD-converted + as-of line; label-native defense-in-depth branch).
2. **Both lite analyzers wired** (`_run_claude_analysis` + `_run_gemini_analysis`, shape parity): normalize+flag ALWAYS (cheap, pre-LLM); blocking flags + flag ON -> `_data_integrity_blocked_analysis` pre-LLM HOLD/REJECT (GuardAgent chokepoint; $0 LLM; counts toward the 56.2 degraded guard deliberately); prompts use the rendered market lines; risk-judge template gets USD-true `market_cap_b` under the flag (also cures the unit-broken `market_cap > 5e9` BUY rule -- 5e9 KRW = $3.6M, every KR ticker passed).
3. **UNGATED additive provenance** in lite `market_data` (currency/price_usd/market_cap_usd/fx_rate/as_of/integrity_flags) via `_integrity_market_data` -- BQ-auditable on every row regardless of flag; legacy keys intact.
4. **NEW `settings.paper_data_integrity_enabled`** (default OFF; full rationale + promotion rule in the description).
5. **Tests** `backend/tests/test_phase_60_3_data_integrity.py` (13): the 06-09 066570.KS regression (away-week state -> blocking flags -> excluded in code), end-to-end analyzer block with a poisoned LLM rail (raises if touched -- proves pre-LLM), no-'$'-KRW-magnitude regex, US byte-identity BOTH flag states, KR flag-OFF legacy rendering, currency-mismatch, ceiling boundary, FX-sane unblocked case, provenance-ungated, flag-default-OFF. One calibration fix during GENERATE: the P/E-0 floor corrected from $100B "mega-cap" to the standard $10B large-cap floor (LG converted = $32B).

## Files changed

backend/services/data_integrity.py (NEW), backend/services/autonomous_loop.py (analyzer wiring + module helpers), backend/config/settings.py (flag), backend/tests/test_phase_60_3_data_integrity.py (NEW, 13 tests).

## Verification command output (verbatim)

```
$ python -m pytest backend/tests -k 'prompt_fx or lite_prompt or 60_3' -q
13 passed, 810 deselected, 1 warning in 3.72s        (exit 0)
$ test -f handoff/current/live_check_60.3.md -> OK
```
FULL suite: first run 1 failed (`test_phase_23_2_7_red_line_nav_match` -- live-BQ NAV comparison, unrelated surface, passes in isolation), re-run **805 passed, 12 skipped, 6 xfailed exit 0** (792 post-60.2 + 13 new). Disclosed as a live-data ordering flake, not hidden.

## Live verification (live_check_60.3.md)

- Before/after prompts rendered from the production renderer for the 066570.KS fixture (the $44,540.6B corruption -> $32.1B truthful + as-of line; away-week FX-unavailable state -> BLOCKED pre-LLM).
- LIVE stack run: real `_run_claude_analysis("005930.KS")` + `_persist_analysis` with the live (flag-OFF) config -> BQ row job_5z0CvcZ8VjBr0-K8sfQQ6D1vv4qG with currency=KRW, price native 299,000 vs price_usd 195.48 @ fx 0.000654, as_of = the KRX close, flags=[] -- internally consistent, unit-auditable. One risk-judge CLI flake during the run (exit 1 -> documented fail-open default sizing) disclosed.
- Operator promotion line: PENDING in live_check §E.

## Artifact shape

- Block artifact: HOLD/REJECT analysis dict with `_data_integrity_blocked: true` + reasons in risk_assessment + `market_data.integrity_flags` (persisted).
- Provenance artifact: the six additive market_data fields on every lite row.
- Prompt artifact: USD-converted line + `Data as-of: <ISO> (... NOT live)` under the flag.
