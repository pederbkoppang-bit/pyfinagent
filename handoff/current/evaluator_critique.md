# Q/A Evaluator Critique -- phase-50.1: FX data layer

**Verdict: PASS** | Date: 2026-05-30 | Fresh Q/A (first for 50.1, no self-eval) | merged qa-evaluator + harness-verifier (deterministic-first)

## 1. Harness-compliance audit (5 items -- ALL PASS)

1. **Research gate**: `handoff/current/research_brief.md` present, a 50.1 FX-layer brief, JSON envelope `gate_passed: true` (7 external sources read in full >= 5 floor; recency scan 2024-2026 performed; 16 URLs collected; 11 internal files inspected). Cited by `contract.md` lines 6-14 + References. PASS.
2. **Contract-before-generate**: git log `phase-50.1: PLAN` (238bc024, 2026-05-30 00:08:30) precedes `phase-50.1: GENERATE` (d31069a0, 2026-05-30 00:15:05). The 4 `success_criteria` in `contract.md:20-23` are **verbatim** from masterplan step 50.1 `verification.success_criteria`. PASS.
3. **experiment_results.md**: present; lists 3 files changed (fx_rates.py NEW, migration NEW, data_ingestion.py 1-line); verbatim verification output; live evidence cross-referenced to `live_check_50.1.md`. PASS.
4. **Log-last**: NO `phase=50.1` entry in `handoff/harness_log.md` yet; masterplan 50.1 still `in_progress`. Correct ordering (log + status-flip come AFTER this PASS). PASS.
5. **No verdict-shopping**: first Q/A for 50.1; no prior CONDITIONAL/FAIL on this step-id in harness_log. PASS.

## 2. Deterministic checks (run by Q/A, cannot hallucinate)

- `ast.parse(backend/services/fx_rates.py)` -> OK
- `ast.parse(scripts/migrations/create_historical_fx_rates_table.py)` -> OK
- `ast.parse(backend/backtest/data_ingestion.py)` -> OK
- **Immutable masterplan command**: `get_fx_rate('USD','USD')==1.0` + `market_currency('EU'/'KR'/'US')==EUR/KRW/USD` -> `det OK` (exit 0)
- `test -f handoff/current/live_check_50.1.md` -> present
- `import backend.backtest.data_ingestion` -> OK (no circular dep from the new `from backend.backtest import markets` at line 17)

## 3. LIVE re-verification (independent network + BQ, run by Q/A)

- **FX direction (THE #1 risk) -- CONFIRMED CORRECT**: independent live fetch returned EUR/USD = **1.1659** (1.0 < e < 1.4 OK) and KRW/USD = **0.0006635** (0.0004 < k < 0.0015 OK -- NOT ~1300, so NOT inverted). Code uses `KRW=X` (fx_rates.py:41) and explicitly avoids `KRWUSD=X` (documented fx_rates.py:14). EUR via `EURUSD=X` (:39). Matches FRED DEXUSEU/DEXKOUS direction.
- **BQ historical_fx_rates well-formed**: EURUSD n=13 (2026-05-15..29, avg 1.1638), KRWUSD n=13 (avg 0.000665). (13 not 12 vs results -- the Q/A live re-verify write-through added today's mark; expected/correct behavior.)
- **As-of / no look-ahead**: `get_fx_rate('EUR','USD','2026-05-20')` = 1.1607, `get_fx_rate('KRW','USD','2026-05-20')` = 0.0006632. Query is parameterized `WHERE pair=@pair AND date<=@d ORDER BY date DESC LIMIT 1` (fx_rates.py:165-172). Point-in-time correct.
- **Junk-row exclusion verified**: as-of @2026-05-20 returned 1.1607 (a real backfilled row ~1.16), NOT the junk row value 1.1729 (`date='EURUSD=X'`). The `date<=` lexical comparison excludes non-ISO dates ('E' > '2'). Disclosure is accurate.
- **Append-not-upsert unaffected**: `LIMIT 1` returns a single rate despite duplicate dates from append-mode writes. Confirmed.
- **Migration**: `CREATE TABLE IF NOT EXISTS`, dry-run default + `--apply`, no `--location` pin (correctly auto-resolves financial_reports/us-central1).
- **data_ingestion stub fix**: line 147 `markets.get_market_config(market)["currency"]` (old `"USD" if market=="US" else "USD"` stub gone). market_currency EU/KR/US == EUR/KRW/USD.

## 4. Code-review heuristics (5 dimensions evaluated; no BLOCK, no degrading WARN)

- **secret-in-diff [BLOCK]**: none.
- **financial-logic-without-behavioral-test [BLOCK]**: does NOT fire -- diff touches none of perf_metrics/risk_engine/backtest_engine/backtest_trader. fx_rates.py is a NEW additive service; NAV math untouched (correctly deferred to 50.2).
- **kill-switch / stop-loss / max-position / perf-metrics-bypass [BLOCK]**: do NOT fire -- diff touches no execution/risk path.
- **SQL-injection (LLM05/insecure-output) [BLOCK]**: does NOT fire -- the f-string at fx_rates.py:165 interpolates only settings-derived constants (`proj`, `dataset`); the actual query VALUES (`pair`, `date`) ARE parameterized via `ScalarQueryParameter` (:170-171). Table names cannot be BQ-parameterized; this is the correct pattern.
- **broad-except [Dimension-3 WARN, negation-list cleared]**: 9 `except Exception` sites + one `except Exception: pass` (fx_rates.py:100-101). NONE is in a risk-guard / kill-switch / stop-loss path, so the `broad-except-silences-risk-guard [BLOCK]` heuristic does NOT fire. The pattern is the documented degradation-tolerant yfinance try/except->log->return-None idiom (mirrors data_ingestion.py:103-114); the `pass` wraps a non-critical `cache.set()` (failing to cache is harmless, the value is still returned). NOTE-level acceptable for a data fetcher; does not degrade verdict.
- **unicode-in-logger [NOTE]**: ASCII-clean.
- **print-statement [WARN]**: none in service code.

## 5. Scope-honesty assessment (3 disclosed flags -- all honest + non-blocking)

- **(a) single-ticker yf.download MultiIndex bug, caught + FIXED during verification**: the fix is present in the committed code (fx_rates.py:246-247 squeezes Close to a Series before `.items()`). This is honest disclosure of a bug that was fixed and re-verified -- NOT a live defect.
- **(b) 2 malformed streaming-buffer rows (date='EURUSD=X'/'KRW=X')**: Q/A independently confirmed exactly 2 such rows exist AND that they are excluded from every real as-of read by the `date<=` ISO comparison. DELETE blocked by BQ's ~90-min streaming-buffer rule; cleanable post-flush via `DELETE WHERE date NOT LIKE '2%'`. Harmless to reads. Honest + non-blocking.
- **(c) append-not-upsert**: confirmed the `LIMIT 1` as-of read is unaffected by duplicate dates. Hardening follow-up (load-job + MERGE) correctly deferred. Honest + non-blocking.

These are disclosed hardening items, NOT criterion violations -- the FX layer is functionally correct (direction, as-of, byte-identical USD path all verified).

Minor cosmetic NOTE (not a defect): live_check_50.1.md:17 annotation "1/~1507 KRW-per-USD" is an imprecise inverse (1/0.000665 ~= 1504); the actual stored/returned rate value 0.000665 is correct. Documentation annotation only; does not affect code or any criterion.

## 6. Success-criteria mapping (all 4 IMMUTABLE criteria MET)

1. **fx_rates.py with get_fx_rate(base,quote,date) + daily-refresh + EURUSD=X/KRW=X + cache + USD->USD=1.0** -- MET (verified live + deterministic).
2. **historical_fx_rates BQ table holds dated FX rates, backfilled EUR/USD + KRW/USD** -- MET (table created in financial_reports; EURUSD n=13 + KRWUSD n=13 over 2026-05-15..29, BQ-read confirmed).
3. **data_ingestion.py:146 currency stub fixed (per-market ISO, not 'USD' unconditionally)** -- MET (line 147 delegates to markets.get_market_config; old stub gone; import OK).
4. **live EUR/USD + KRW/USD rate fetched verbatim** -- MET (live_check_50.1.md + Q/A independent re-fetch: 1.1659 / 0.0006635).

## Verdict

**PASS.** All 4 immutable criteria met and independently re-verified (live network fetch + BQ read by Q/A, not merely trusting the generator). The #1 risk -- FX direction / KRW inversion -- is correct (`KRW=X`, not `KRWUSD=X`; KRW/USD ~0.00066, not ~1300). No look-ahead (parameterized `date<=` as-of). US-only/USD path byte-identical (`from==to->1.0`). Purely additive, no execution/risk/NAV path touched (50.2 scope correctly deferred). The 3 scope-honesty flags are honest disclosures, independently confirmed harmless, not criterion violations. No code-review BLOCK or verdict-degrading WARN. Frontend ESLint/tsc gate N/A (no frontend/** in diff).

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 4 immutable criteria met + independently re-verified (live yfinance fetch EUR/USD=1.166, KRW/USD=0.00066 NOT inverted; BQ historical_fx_rates EURUSD+KRWUSD n=13 each; parameterized date<= as-of no look-ahead; from==to->1.0 keeps US-only byte-identical). 5/5 harness-compliance pass. No code-review BLOCK/WARN. 3 disclosed hardening flags (MultiIndex bug FIXED+re-verified, 2 harmless streaming-buffer junk rows excluded by date<=, append-not-upsert unaffected by LIMIT 1) confirmed honest + non-blocking.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["syntax", "verification_command", "live_fx_direction_reverify", "bq_table_read", "asof_lookahead_check", "junk_row_exclusion", "data_ingestion_stub_fix", "code_review_heuristics", "research_brief", "contract_verbatim", "experiment_results", "harness_log_last", "scope_honesty"],
  "harness_compliance": {
    "research_gate": "PASS (gate_passed:true, 7 sources, recency scan, cited by contract)",
    "contract_before_generate": "PASS (PLAN 00:08:30 < GENERATE 00:15:05; 4 criteria verbatim)",
    "experiment_results_present": "PASS",
    "log_last": "PASS (no phase=50.1 in harness_log; masterplan 50.1 in_progress)",
    "no_verdict_shopping": "PASS (first Q/A for 50.1)"
  }
}
```
