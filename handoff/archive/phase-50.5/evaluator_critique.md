# Q/A Evaluator Critique -- phase-50.5: Multi-market backtest + DATA-QUALITY gate

**Verdict: PASS** | Fresh Q/A (first for 50.5; no verdict-shopping) | 2026-05-31
**Reviewer:** Q/A subagent (merged qa-evaluator + harness-verifier), effort=max
*(This file previously held the phase-50.4 PASS critique; superseded per protocol --
the 50.4 critique is archived under `handoff/archive/phase-50.4/` on step close.)*

---

## phase-50.5 EVALUATE -- 2026-05-31

### 5-item harness-compliance audit (run FIRST)

| # | Gate | Result | Evidence |
|---|------|--------|----------|
| 1 | researcher gate BEFORE contract | PASS | `research_brief.md:252` has `## phase-50.5 RESEARCH-GATE REVALIDATION -- 2026-05-31`; envelope at :401-407 `gate_passed:true`, `external_sources_read_in_full:6`, `recency_scan_performed:true`. (An earlier 50.5 envelope at :241-246 shows 7 sources -- both >=5 floor.) Tobi Lux DAX study RE-VERIFIED live (:333: 11% deviation, 10-24 identical-OHLC days/yr, zero-volume corroborates R2); axionquant z=3 (:334) confirms `_Z_FLAG=3.0`; NEW recency hit arXiv 2603.19380 (:335) quantifies the DEFERRED PIT-membership gap. Contract cites this brief (`contract.md:8,51`). |
| 2 | contract-before-generate | PASS | `git log`: `4290c82d phase-50.5: PLAN (GENERATE pending)` precedes this GENERATE. The 4 `success_criteria` in `contract.md:23-26` are verbatim from masterplan 50.5 `verification.success_criteria` (byte-checked: all 4 PRESENT). |
| 3 | experiment_results present + complete | PASS | `experiment_results.md` present: file table (9 files), verbatim syntax+pytest (9 passed) + regression (94 passed), live_check ref, artifact shapes, disclosed deferrals. |
| 4 | log-last discipline | PASS | NO `phase=50.5` / `## Cycle.*50.5` header in `handoff/harness_log.md` (grep empty). masterplan 50.5 `status=in_progress`, `retry_count=0`. Main correctly has NOT logged/flipped before this verdict. |
| 5 | no verdict-shopping | PASS | First Q/A for 50.5 (no prior 50.5 entry in harness_log; on-disk critique before this write was the 50.4 PASS). Not a cycle-2 spawn. **Prior consecutive CONDITIONALs for 50.5 = 0** (3rd-CONDITIONAL auto-FAIL rule N/A). |

### Deterministic checks (run by Q/A, not trusting the agent)

| Check | Result |
|-------|--------|
| `ast.parse` price_quality.py | OK |
| `ast.parse` all 8 modified source files | OK (markets/backtest_engine/analytics/api.backtest/paper_trader/data_ingestion/price_quality/screener) |
| **IMMUTABLE command** (ast + pytest + test -f) | **exit=0**: 9 passed in 1.33s + `live_check_50.5.md` PRESENT |
| regression suite (50.3 + 32.1 + dod4 + 50.5) | **94 passed** in 1.63s |
| US byte-identity test uses EXACT `==` (not approx) | `:103 fx_ratio == 1.0`; `:109 spy_return_pct == raw` (exact float, not pytest.approx) |
| **INDEPENDENT live_check re-run** (Q/A ran `scripts/phase50/live_check_50_5.py` itself) | Reproduced the SHAPE: benchmark=^GDAXI; EU fx_ratio=0.99946 (!=1.0); US fx_ratio==1.0 EXACT; 15 real .DE bars dropped + 6 flagged (all R2 identical-OHLC+zero-vol); injected R1/R2/R3 all fire. Numbers match Main's (fx differs 0.99946 vs 0.99958 -- live-rate drift, expected). NOT fabricated. |
| `market_for_symbol` independent check | AAPL/MSFT/SPY->US; SAP.DE/BMW.DE->EU; 005930.KS->KR (correct) |

### LLM judgment (adversarial)

**US byte-identity (THE regression surface) -- PASS, proven 4 ways.**
The gate inserts into the live US screener + ingestion + fill paths, so US must be untouched.
1. `validate_ohlcv(df, market="US")` returns the **same object** at `price_quality.py:55` (`if df is None or market == "US": return df, report`) -- screener L1 (`screener.py`) + ingestion B (`data_ingestion.py:136-138`, `_mkt=="US"`->no-op).
2. `_get_live_price` L2 (`paper_trader.py:1208-1216`): `if market_for_symbol(ticker) != "US"` -- US tickers skip validation entirely; returns `float(hist["Close"].iloc[-1])` exactly as before. Diff touches ONLY `_get_live_price` (grep confirmed: NO kill_switch/stop_loss/execute_buy/execute_sell/backfill/paper_max_positions mutation).
3. `compute_baseline_strategies` US path: `local_currency=="USD"==base_currency` -> `fx_ratio` stays `1.0` (the `if local!=base` branch at `analytics.py:542` is not entered) -> `_to_base` short-circuits `if fx_ratio == 1.0: return r_pct` (`:559-560`) -- NO `(1+r)*1-1` float round-trip. Unit-tested with EXACT `==` (`:109`).
4. `cache.preload_prices(universe + [_benchmark])` with US benchmark `"SPY"` (`backtest_engine.py:305`) == the old hard-coded `+["SPY"]`. `get_universe_tickers(market="US")` == default (US is DEFAULT_MARKET). US benchmark stays `"SPY"` (not `^GSPC`) -> US numbers unchanged.
**No US-path behavior change.**

**Drift honesty -- PASS.** The contract (`contract.md:16`) originally fingered `backtest_engine.py:299` as the benchmark site; the researcher caught that `:299` is only a cache-preload line and the real benchmark/alpha computation lives in `analytics.compute_baseline_strategies` + `api/backtest.py`. Main patched the RIGHT sites: `analytics.py:446-573` (benchmark param + FX) and `api/backtest.py:937-947` (threads `benchmark`+`local_currency` from `get_market_config(engine.market)`). The drift is disclosed in `experiment_results.md:10-12`. Honest correction, not a hidden miss.

**Criterion #3 "EU backtest runs end-to-end" -- MET (substantiated, not hand-waved).** Main did NOT run a full `BacktestEngine(market="EU").run_backtest()` (needs BQ .DE ingestion). Instead it exercises EVERY NEW code path on REAL data: market-aware universe (`get_universe_tickers(market="EU")` -> 40 .DE/.PA tickers), per-market benchmark selection (^GDAXI), the gate on real .DE bars (15 dropped), and the FX-converted baseline (EUR->USD). I read `scripts/phase50/live_check_50_5.py` end-to-end (152 lines) -- it is genuine: live `yf.Ticker().history()` fetches, real `compute_baseline_strategies` calls, real `validate_ohlcv`, hard asserts (`:122 benchmark=="^GDAXI"`, `:146 us fx_ratio==1.0`). The ONLY un-exercised piece is the BQ-ingestion orchestration glue -- pre-existing shared machinery, NOT 50.5 code, and `live_check_50.5.md:85-91` discloses this precisely ("the full walk-forward reuses these exact wired functions; additionally requires BQ ingestion of .DE history"). Because no NEW or CHANGED logic is left unexercised, this is a **PASS, not a CONDITIONAL** -- the gap is infrastructure, not a code-coverage hole. (Same scope-honesty standard applied to 50.4.)

**Anti-rubber-stamp / no over-drop of real volatility -- PASS.** I ran an independent adversarial test: a real +8% earnings move WITH volume is PRESERVED (dropped=0, z-flagged only); a +60% glitch DROPS (R3 fires). The DROP-unambiguous / FLAG-suspicious asymmetry (`price_quality.py:78-118`) holds exactly. "No silent truncation" confirmed: counts logged (`:123-127`) AND returned in the report dict. The CFA-L2 source (`research_brief.md:336`) correctly frames this as PRE-registered data-integrity rules, not post-hoc p-hacking.

**`^GDAXI` not over-dropped -- PASS.** `market_for_symbol("^GDAXI")=="US"` but this is irrelevant: `analytics.py` NEVER imports/calls `validate_ohlcv` (grep clean) -- the benchmark series is read RAW via `prices_cache_fn`. No hidden gate interaction.

**Scope honesty (DEFERRED items) -- PASS, genuinely non-blocking.** PIT intl membership (US has the SAME gap; arXiv 2603.19380 = separate reconstruct-membership project), per-bar FX inside `_compute_nav` (even QuantConnect doesn't do cheap per-bar intl FX -- `research_brief.md:337`), simultaneous mixed-currency multi-market backtest, live per-market benchmark. None of these is required by the 4 criteria: criterion #1 says "FX-converts NAV/returns" -- the RETURNS are FX-converted (endpoint method); Sharpe-stays-local is a disclosed deferral, not a criterion miss (criterion #1 does not mandate Sharpe currency). No deferral hides a criterion gap.

### Code-review heuristics (5 dimensions evaluated -- no BLOCK, no WARN)

Diff scanned: markets.py, backtest_engine.py, analytics.py, api/backtest.py, paper_trader.py (`_get_live_price` only), data_ingestion.py, screener.py, price_quality.py (NEW), test file (NEW), live_check script (NEW).
- **financial-logic-without-behavioral-test** [BLOCK]: NOT triggered. analytics.py (FX+benchmark) gets 3 NEW behavioral tests (`test_baseline_us_byte_identical_passthrough` exact-`==`, `test_baseline_eu_fx_converted` 20%local->32%USD, `test_market_config_has_benchmarks`). backtest_engine.py change is wiring only (market param + benchmark preload; NO Sharpe/drawdown/sizing math) -- covered by the end-to-end live_check.
- **perf-metrics-bypass** [WARN]: NOT triggered. `compute_baseline_strategies` reuses the PRE-EXISTING `compute_sharpe` (analytics.py is the canonical metrics module); the FX patch wraps only `*_return_pct` via `_to_base`, never re-implements Sharpe. No inline Sharpe/drawdown/alpha.
- **kill-switch / stop-loss / max-position / backfill** [BLOCK class]: untouched (paper_trader diff is `_get_live_price`-only; grep clean for all risk-guard symbols).
- **broad-except-silences-risk-guard** [BLOCK]: NOT triggered. The `except Exception` in `validate_ohlcv:129` and `is_bad_bar:150` are deliberate fail-OPEN (return df unchanged / return False = don't drop) and LOG via `logger.warning` -- they NEVER suppress a risk guard; they prevent a validator bug from blocking the pipeline. Correct safe-by-default.
- **tautological-assertion / over-mocked-test** [BLOCK]: none. Tests assert real `validate_ohlcv` drop/flag behavior, real `compute_baseline_strategies` outputs, exact float equality. `test_baseline_eu_fx_converted` monkeypatches ONLY the FX-rate lookup (`get_fx_rate`), not the module under test -- legitimate isolation.
- **secret-in-diff / command-injection / crypto-re-enable / supply-chain-pin-removal** [BLOCK/WARN]: N/A (grep clean; no secrets, no subprocess/eval/exec, no crypto, no pin removal, no pip).
- **Frontend gate** (ESLint/tsc): N/A -- diff touches NO `frontend/**`.

### Quality criteria scoring (>=6 to pass each)
Infrastructure/data-integrity step. DSR/Sharpe-stability criteria N/A (no return-generating strategy changed; US engine byte-identical). Statistical Validity: the gate is a PRE-registered data-integrity rule (not post-hoc p-hack -- CFA-L2 compliant). Robustness: fail-open on every validator error; DROP-unambiguous/FLAG-suspicious asymmetry verified on real + adversarial data; US byte-identity proven 4 ways. Simplicity: one new pure validator + minimal wiring at 3 doors + 2 backtest sites (12-line screener, 15-line paper_trader diffs). Reality Gap: gate addresses a REAL live defect (2.5% bad .DE bars right now). No criterion below 6.

### checks_run
syntax, verification_command, pytest, regression_suite_94, independent_live_check_rerun, market_for_symbol_independent, byte_identity_trace_4way, no_over_drop_adversarial_test, gdaxi_not_overdropped, code_review_heuristics, financial_logic_behavioral_test, perf_metrics_bypass_scan, broad_except_scan, secret_command_injection_scan, drift_honesty, criterion3_end_to_end_judgment, scope_honesty_deferrals, research_gate_revalidation, contract_alignment_verbatim, harness_log_inspection, git_log_ordering, evaluator_critique, experiment_results

### Conclusion
All 4 immutable criteria GENUINELY met. (1) benchmark per-market + FX-converted returns -- wired + tested + live-verified. (2) data-quality gate DROPs unambiguous / FLAGs suspicious, counts logged+returned, no silent truncation, no over-drop of real volatility (independently adversarially tested). (3) EU path exercised end-to-end on real .DE data through every NEW code path; US byte-identical (proven 4 ways, exact-`==` tested); the un-run full-engine BQ orchestration is pre-existing shared infrastructure, not a 50.5 code-coverage gap, and is triple-disclosed. (4) live evidence reproduced by Q/A's OWN run (^GDAXI, +7.97% USD, 15 bars dropped) -- not fabricated. US byte-identity HOLDS. No BLOCK/WARN heuristic. **PASS.**

Next (Main, in order): append `handoff/harness_log.md` (LAST), then flip masterplan 50.5 -> done. Then the operator-authorized go-live flip (`settings.paper_markets -> ["US","EU","KR"]`) should be REPORTED explicitly as the final go-live action, not silently executed.
