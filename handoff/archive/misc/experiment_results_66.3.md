# Experiment results -- 66.3 Cost-truth restoration (Cycle 72, 2026-07-07)

## What was built

1. **Writer gauge-guard** (`backend/services/observability/api_call_log.py`,
   log_llm_call): failed calls with zero token movement log `session_cost_usd=0.0`
   instead of the lazy-filled cumulative cycle gauge; explicit caller values win;
   token-moving failures keep the gauge (mid-stream may bill). Docstring-level
   comment documents the gauge semantics and the 2026-06 phantom-spend incident.
2. **`scripts/away_ops/metered_spend.py`** (NEW, importable + CLI --date):
   token-derived metered dollars over metered providers with a pinned,
   source-commented price table (gemini-2.5-flash/pro, sonnet-4-6, opus-4-7/4-8,
   haiku-4-5 incl. dated-id prefix match, fable-5; Claude cache-token pricing);
   flat-fee exclusion by `agent LIKE 'cc_rail%'` + `provider='claude-code'`
   (documented billing-class choice, criterion 2); unpriced metered models count
   $0 + fail-visible warning; `rail_failures` counted from ok=false rail rows.
3. **sentinel.sh**: metered block now delegates to metered_spend.compute_for_date
   (SENTINEL_DATE env = replay hook); false-premise comment (:63-66) replaced with
   the gauge explanation; JSON gains `rail_failures_today` (init None) + a
   warnings[] entry at >=20 failures (mirrors the 66.1 breaker threshold);
   SENTINEL_TEST_METERED_USD / SENTINEL_TEST_BQ_FAIL overrides preserved.
4. NO schema migration; NO change to the gauge accumulator itself (burn-audit
   consumers keep their semantics); NO backfill (history stays as written -- the
   replay proves the new READ logic).

## Verbatim verification output (immutable command)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_66_3_cost_truth.py -q
...........                                                              [100%]
10 passed
```
Regression: 66_3 + test_phase_62_4_sentinel.py + test_observability.py together:
`32 passed, 1 warning in 18.44s`. `bash -n sentinel.sh` clean.

## Key evidence (full pastes in live_check_66.3.md)

- Replay 2026-06-18: **$0.0043** (old logic: $43.00 breach) + rail_failures=207.
  Replay 2026-06-17: **$0.0026** (old: $16.51) + rail_failures=137. Neither
  breaches the $8 baseline -- the METERED-BREACH P1 is now CLOSED with a fix, not
  just a root cause.
- Live day match: module $0.0013 == independent hand audit $0.0013 (haiku ticket
  call, 1000 in / 50 out). 06-18 hand audit likewise matches to the 4th decimal.
- Full live sentinel run: ok:true, metered 0.0013, rail_failures_today 0.

## Honest disclosures

- The metered figure is derived from tokens x a PINNED price table -- price drift
  is a maintenance liability; unknown models fail visible (warning) rather than
  silently wrong. An independent cross-check against the GCP billing export
  (all_billing_data, EU) is possible but was not run (Vertex billing lags ~1 day;
  today's figure is $0.0013 -- below billing-export granularity).
- Historical rows keep their gauge-stamped values; anyone summing
  session_cost_usd over history will still over-count. The column is now
  documented as a gauge at the writer; consumers must use per-cycle MAX or the
  new module. Register note for 63.3.
- The "$8.24 Gemini real spend" figure quoted in earlier phase-66 documents was
  itself a gauge-sum artifact; the true metered spend of the away window is
  ~$0.10-0.20 (token-derived). Corrected here; the direction of every prior
  conclusion is unchanged (real spend even lower than stated).

## File list

api_call_log.py (guard), scripts/away_ops/metered_spend.py (NEW),
scripts/away_ops/sentinel.sh (metered block + field), backend/tests/
test_phase_66_3_cost_truth.py (NEW, 10) + handoff artifacts.
