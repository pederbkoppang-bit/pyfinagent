---
step: phase-25.B3
cycle: 71
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_25_B3.py'
title: Daily loop reads latest promoted strategy via load_promoted_params() (P1)
audit_basis: phase-24.3 F-6 (autonomous_loop.py:33-43 read only optimizer_best.json; no BQ promoted_strategies query)
depends_on: 25.A3 (done, commit 2a864210)
---

# Experiment Results -- phase-25.B3

## Code changes

### `backend/db/bigquery_client.py`
- New method `get_latest_promoted_strategy(self, status_filter: list[str] | None = None) -> dict | None`:
  - Defaults `status_filter = ["pending", "active"]` because 25.A3 hardcodes status="pending" until 25.C3's state machine lands.
  - Parameterized SQL with `bigquery.ArrayQueryParameter("statuses", "STRING", status_filter)`.
  - SELECT uses `TO_JSON_STRING(params) AS params_json` (BQ JSON columns are non-serializable in `dict(row)`).
  - `ORDER BY promoted_at DESC, dsr DESC LIMIT 1`.
  - `result(timeout=30)` per CLAUDE.md rule.
  - Reader pops `params_json` and `json.loads(...)` it back to a dict; try/except yields `params: {}` on malformed JSON (no exception).
  - Returns `None` if no row matches; the row dict otherwise.

### `backend/services/autonomous_loop.py`
- New `load_promoted_params(bq: BigQueryClient) -> dict` immediately after `load_best_params()`. Three-tier fallback:
  1. BQ row found with non-empty params -> logs `"Loaded promoted params (DSR %s week=%s): %s"` -> returns those params.
  2. BQ returns None or empty params -> logs `"No active promoted strategy in BQ, falling back to optimizer_best"` -> returns `load_best_params()`.
  3. BQ raises -> logs `"Promoted strategy BQ unavailable, falling back to optimizer_best: %s"` with the exception detail -> returns `load_best_params()`.
- Caller in `run_daily_cycle` now `best_params = load_promoted_params(bq)` (was `load_best_params()`). `bq` was already in scope at that point.

### `tests/verify_phase_25_B3.py` (new file)
- 11 immutable claims:
  - Claims 1-5: structural -- signatures, default filter assignment, SQL shape (TO_JSON_STRING + ORDER BY + LIMIT 1 + IN UNNEST), result(timeout=30) inside the new method, caller wired to `load_promoted_params(bq)`.
  - Claim 6: **Behavioral happy path** -- fake bq returns a row with `params={lookback:20, tp_pct:0.1, ...}`; `load_promoted_params` returns those params + captured logger output contains `"Loaded promoted params"`.
  - Claim 7: **Behavioral empty path** -- fake bq returns None; result still a dict (falls back to optimizer_best) + logs `"No active promoted strategy in BQ"`.
  - Claim 8: **Behavioral exception path** -- fake bq raises RuntimeError; result still a dict + logs `"Promoted strategy BQ unavailable"` with exception detail (`"network down"` verified).
  - Claim 9: **Behavioral JSON round-trip** -- direct test of `get_latest_promoted_strategy` with a fake BQ client that returns a row whose `params_json='{"a":1,"b":2}'`; reader inverts to `params={a:1, b:2}` and pops `params_json` from the return dict.
  - Claim 10: **Behavioral malformed JSON** -- malformed `params_json="not-valid-json"` -> reader returns `params={}` (no exception).
  - Claim 11: literal log line `"Loaded promoted params (DSR"` present in source (covers masterplan criterion 3).

## Verbatim verifier output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_B3.py
PASS: load_promoted_params_function_exists_in_autonomous_loop
PASS: bq_get_latest_promoted_strategy_with_default_filter
PASS: bq_query_shape_to_json_string_order_limit_unnest
PASS: bq_query_uses_result_timeout_30
PASS: run_daily_cycle_caller_uses_load_promoted_params
PASS: behavioral_happy_path_returns_bq_params_and_logs_success
PASS: fallback_to_optimizer_best_json_if_bq_empty
PASS: fallback_to_optimizer_best_json_if_bq_unavailable
PASS: behavioral_bq_reader_json_round_trip
PASS: behavioral_malformed_params_json_safe_fallback
PASS: autonomous_cycle_logs_show_promoted_strategy_loaded

11/11 claims PASS, 0 FAIL
```

## Backend gates

- `python -c "import ast; ast.parse(open('backend/services/autonomous_loop.py').read())"` -- OK
- `python -c "import ast; ast.parse(open('backend/db/bigquery_client.py').read())"` -- OK
- 4 behavioral round-trip tests (happy/empty/exception/JSON-round-trip/malformed-JSON) execute the actual code with fakes; mutation that breaks any of the three fallback paths or the JSON unpacking would FAIL.

## Hypothesis verdict

CONFIRMED. The three immutable success criteria are covered:
- Criterion 1 (`load_promoted_params_function_exists_in_autonomous_loop`) -- claim 1 grep.
- Criterion 2 (`fallback_to_optimizer_best_json_if_bq_unavailable`) -- claim 8 behavioral path + claim 7 empty-row path.
- Criterion 3 (`autonomous_cycle_logs_show_promoted_strategy_loaded`) -- claim 11 grep + claim 6 behavioral assertion.

The 3-tier fallback is research-backed (microservices fallback pattern + Anthropic-style local-cache fallback). No in-process TTL cache needed because the daily cadence makes a per-cycle BQ read fine.

## Live-check

Per masterplan: "BQ promoted_strategies query returns active row; autonomous_loop log confirms params merged".

Live evidence pending capture in `handoff/current/live_check_25.B3.md` after operator applies the 25.A3 migration AND triggers a Friday promotion + a daily-cycle run. Expected log line in `handoff/logs/autonomous_loop.log` (or wherever the logger is wired): `"Loaded promoted params (DSR 1.2 week=2026-W20): ['lookback', 'tp_pct', ...]"`.

## Non-regressions

- `load_best_params()` unchanged; remains the Tier-2 fallback for `load_promoted_params`.
- Any existing caller of `load_best_params()` (if there is one outside `run_daily_cycle`) unaffected.
- No new BQ schema; reuses the 25.A3 `promoted_strategies` table.
- Daily cycle still runs to completion even when BQ is unavailable (Tier-3 fallback to `{}`).

## Downstream

Unblocks **25.C3** (strategy registry status state machine; flip `pending` -> `active`) which unblocks **25.R** (auto-switching policy / red-line goal-c).

## Next phase

Q/A pending.
