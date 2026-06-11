# Contract -- 60.3 Decision-input integrity for non-USD markets (AW-9)

**Step:** 60.3 (phase-60, P1, harness_required, depends_on 60.2 done). **Date:** 2026-06-11.
**AW basis:** AW-9 (handoff/archive/phase-59.3/59.3-harness-free-output.md §6 spot-checks B/C/D).

## Research-gate summary (researcher abcb4dac, tier=moderate-complex, gate_passed:true)

6 sources in full (arXiv:2604.01483 deterministic guardrails, GuardAgent 2406.09187, companiesmarketcap NVDA, yfinance #2699, GIPS Q&A 5336, Korea Exchange), 40 URLs, recency scan. Brief: handoff/current/research_brief.md.

- **Corrupted sites (HEAD):** lite trader prompts `autonomous_loop.py:1898` (Claude) + `:2137` (Gemini); risk-judge template `:1627` (fed at `:1973`/`:2184`); judge SYSTEM prompt `:1619` reasons in USD terms ("market cap < $2B"). Values from `stock.info` keys (`currentPrice`/`regularMarketPrice`/`marketCap`/`trailingPE`) defaulted to 0 at `:1839-1841`/`:2104-2106`. **BONUS DEFECT in-scope for C2:** the `market_cap > 5e9` BUY rule at `:1903`/`:2142` is unit-broken for KR (5e9 KRW = $3.6M -- every KR ticker passes).
- **REUSE:** `fx_rates.get_fx_rate(ccy, "USD")` (fx_rates.py:182; 6h cache; KRW=X inversion handled; returns None on failure) + `markets.market_for_symbol`. `paper_trader._fx_local_to_usd` (:32) already converts at FILL time -- so convert the PROMPT presentation, never `price_at_analysis`/native persisted values (the tolerance gate + fills consume native).
- **Fixture (BQ-pulled live):** 066570.KS 2026-06-09T18:03:49Z row -- `market_cap=44540606021632.0`, `pe_ratio=0.0`, judge prose "physically impossible... KRW/USD unit error... reject" in `summary`, yet `recommendation=BUY` EXECUTED. Judge schema (`:1630-1637`) has NO machine-readable flag field; prose lands in reasoning/summary and NOTHING reads it.
- **External consensus:** guard verdicts must be ENFORCED by code at a chokepoint before execution (GuardAgent "intercepts... before it reaches the execution environment"; 2604.01483 "denied if O_l=1") -- prose-only flagging is the documented anti-pattern, exactly what the criterion bans.
- **Sanity bounds (cited):** largest real market cap NVDA $4.854T (2026-06-10) -> $10T post-USD-normalization ceiling (2x headroom, still catches $44.5T); P/E==0 on a mega-cap = missing-data artifact (treat as MISSING, never a real value); currency mismatch detected suffix-vs-`info.currency` (yfinance #2699: suffix is ground truth, financialCurrency unreliable).
- **Staleness:** KRX regular session 09:00-15:30 KST = 06:30 UTC close; away-week analyses ran 18:03-18:06 UTC (~11.6h stale). Label from `regularMarketTime` (as-of), NOT a hardcoded close constant (KRX adds extended sessions 2026-06-29 -- recency finding).
- **Tests:** nothing matches the -k net today; byte-identity fixture pattern reusable from test_phase_57_1_reject_binding.py:187.

## Hypothesis

Presenting non-USD inputs truthfully (USD-converted or KRW-labeled + as-of-stamped) and enforcing a deterministic integrity gate IN CODE at the candidate chokepoint removes the AW-9 corruption class (quadrillion-dollar prompts, ignored judge flags, unit-broken BUY rules) for KR tickers while leaving US prompts byte-identical; additive provenance fields make every persisted row auditable regardless of the flag.

## Immutable success criteria (verbatim from .claude/masterplan.json step 60.3)

**Command:** `cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -m pytest backend/tests -k 'prompt_fx or lite_prompt or 60_3' -q && test -f handoff/current/live_check_60.3.md`

1. "lite trader AND lite risk-judge prompts present non-USD prices/market caps either converted to USD via the existing FX helpers (_fx_local_to_usd path) or labeled with their true currency (KRW), for BOTH _run_claude_analysis and _run_gemini_analysis; a unit test with a KRW fixture asserts the rendered prompts contain no dollar-labeled KRW magnitude (no '$' immediately preceding a raw KRW-scale number)"
2. "data-integrity flags become actionable: a deterministic pre-check (or the risk-judge flag) for implausible inputs (e.g. market cap > a sanity ceiling, P/E exactly 0 on a mega-cap, price-currency mismatch) tags the analysis and excludes or floor-sizes the candidate IN CODE, covered by a unit test reproducing the 06-09 066570.KS case -- prose-only flagging is a FAIL"
3. "staleness honesty for KR quotes: prompts label the as-of time of .KS prices fetched after the 06:30 UTC KRX close instead of presenting them as current; covered by a test"
4. "post-fix evidence from the live stack: the next KR analysis row in financial_reports.analysis_results (BQ MCP) shows sane prompt-input provenance (converted/labeled values in full_report_json.market_data) and the rendered before/after prompts for one KR fixture are embedded in the live_check; no behavior change for US tickers (byte-identical prompt test with a US fixture)"

**live_check:** "REQUIRED -- before/after rendered prompts for a KRW fixture, the unit-test outputs, and a post-fix BQ MCP row for a KR ticker."

### Gating design (recorded BEFORE GENERATE; phase-60 do-no-harm header vs criterion-4 live evidence)

- **UNGATED (observability, additive only):** new `market_data` provenance fields on lite-path rows -- `currency`, `as_of` (regularMarketTime ISO), `price_usd`/`market_cap_usd` (when FX available), `integrity_flags` (tags only). These change NO decision and satisfy criterion-4's "converted/labeled values in full_report_json.market_data" on the very next live KR row with the flag OFF.
- **FLAG-GATED default OFF (`paper_data_integrity_enabled`):** the PROMPT text changes (USD-converted/KRW-labeled values + as-of staleness line), the in-code exclusion of integrity-blocked candidates (pre-LLM HOLD return + summary out-channel mirroring 57.1's blocked_out), and the USD-normalized BUY-rule comparison. Promotion = OPERATOR decision recorded in the live_check (PENDING), per the phase header "no live flag flips inside this phase".
- US tickers: byte-identical prompts with the flag OFF **AND ON** (the gate only alters non-US presentation; tested both ways).

## Plan

1. Pure helpers (new `backend/services/data_integrity.py`): `normalize_market_values(ticker, info_like) -> {currency, price_usd, market_cap_usd, as_of, fx_rate, fx_available}` (reuses fx_rates + market_for_symbol; never raises) + `check_data_integrity(ticker, normalized) -> list[flag dicts]` (ceiling $10T post-normalization; P/E==0 + mcap_usd>$100B -> missing_pe; currency mismatch suffix-vs-info.currency; non-US with FX unavailable -> currency_unverified). Unit-testable; the 066570.KS regression = the away-week state (values interpreted as USD) -> implausible_market_cap flag -> in-code exclusion (flag ON).
2. Wire into BOTH `_run_claude_analysis` + `_run_gemini_analysis`: after the info fetch -- compute normalized + flags ALWAYS (cheap); stamp into market_data (ungated, additive); flag ON -> blocking flags return a pre-LLM HOLD analysis dict (`_data_integrity_blocked: true`, reasons; HOLD is not in _BUY_RECS so decide_trades cannot buy it; no LLM spend on corrupt inputs) + `summary["data_integrity_blocked"]` out-channel; flag ON -> prompts render USD-converted (or KRW-labeled when FX unavailable) + as-of staleness line; flag ON -> the 5e9 BUY-rule compares market_cap_usd.
3. Risk-judge leg (criterion 1 "AND lite risk-judge"): the judge template's price/mcap lines go through the same normalized rendering (flag ON); judge SYSTEM prompt unchanged (its USD heuristics become correct once inputs are USD).
4. Tests `backend/tests/test_phase_60_3_data_integrity.py`: KRW-fixture rendered prompts (both analyzers) regex-assert no `\$\s?[0-9.,]{7,}` KRW-scale magnitudes (flag ON); US fixture byte-identity flag OFF AND ON (both analyzers); 066570.KS regression (away-week values -> flag -> excluded in code, flag ON; tags-only flag OFF); staleness label test (as-of line present for stale KR fixture, flag ON); P/E-0 mega-cap -> missing tag; ceiling boundary; flag-default-OFF test.
5. Live evidence: tiny script runs the REAL `_run_claude_analysis` against the live stack for 005930.KS (flag OFF -- live config untouched) + `_persist_analysis` -> BQ MCP row showing market_data provenance fields; before/after prompts from the KR fixture embedded in live_check_60.3.md; ~$0.01-0.17 burn (one lite analysis) disclosed in the 58.1 ledger.
6. live_check_60.3.md (incl. operator promotion line PENDING) -> fresh Q/A -> log -> flip.

## Do-no-harm

Decision-affecting changes flag-gated default OFF; provenance fields additive-only; US prompts byte-identical proven both flag states; no live flag flips; stop-loss/sell paths untouched; the judge schema untouched (no _LITE_RISK_DEFAULT ripple -- deterministic pre-check chosen over judge-schema surgery exactly to avoid it, per brief gotcha).

## References

handoff/current/research_brief.md (full table); GuardAgent + 2604.01483 (enforce-in-code); yfinance #2699; GIPS 5336; 57.1 blocked_out precedent; 56.2 degraded-scoring guard pattern.
