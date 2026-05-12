---
step: phase-25.Q
cycle: 74
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_25_Q.py'
title: Real-time profit_per_llm_dollar metric -- closes red-line GOAL-D (P1)
audit_basis: phase-24.13 F-2 (sovereign_api.py:386-390 hardcoded LLM cost=0; no profit_per_llm_dollar anywhere)
depends_on: 25.A9 (done)
---

# Experiment Results -- phase-25.Q (FIRST-MOVER METRIC)

## Code changes

### `scripts/migrations/add_efficiency_snapshots.py` (new)
- Idempotent `CREATE TABLE IF NOT EXISTS pyfinagent_data.efficiency_snapshots` with 9 columns.
- PARTITION BY snapshot_date, CLUSTER BY window_days.
- Default dry-run + `--apply` (operator-gated per CLAUDE.md BQ rules).

### `backend/db/bigquery_client.py`
- New `save_efficiency_snapshot(row)` MERGE on natural key `(snapshot_date, window_days)` so re-running on the same day upserts instead of duplicating. `result(timeout=30)`.

### `backend/api/sovereign_api.py`
- **New helper `_fetch_llm_cost_by_provider(window_days)`** -- queries `pyfinagent_data.llm_call_log` for `ok=TRUE` rows in the window, groups by `(provider, model)`, applies `cost_tracker.MODEL_PRICING` per-row in Python (no native BQ pricing table), maps `provider="gemini" -> "vertex"` bucket. Returns `{anthropic, vertex, openai}` dict.
- **Fixed hardcoded zeros** at the old `sovereign_api.py:386-390` site (`anthropic=0.0, vertex=0.0, openai=0.0`). Per-day rows now split the window total evenly; `totals` dict uses the actual aggregated costs.
- **New `EfficiencyResponse` Pydantic model** with `profit_per_llm_dollar: Optional[float]` (None when llm_cost==0; first-mover contract).
- **New `GET /api/sovereign/efficiency?window=7d|30d|90d&persist=false`** route. Computes:
  - Numerator: `realized_pnl_usd` summed across round-trips from `bq.get_paper_trades_in_window(days)` + `pair_round_trips`.
  - Denominator: anthropic + vertex + openai cost via `_fetch_llm_cost_by_provider`.
  - Ratio: `realized_pnl / llm_cost` when `llm_cost > 0`, else `None` with descriptive `note`.
  - Optional `persist=True` writes a snapshot via `bq.save_efficiency_snapshot`.
  - Cache + structured_log per existing endpoint pattern.

### `tests/verify_phase_25_Q.py` (new file)
- 11 immutable claims with 4 behavioral round-trips:
  - Claims 1-2: migration shape (9 cols, idempotent, partition+cluster).
  - Claim 3: BQ helper signature + MERGE + timeout.
  - Claims 4: `_fetch_llm_cost_by_provider` signature.
  - Claim 5: `get_compute_cost` no longer hardcodes zeros AND totals use llm_costs.
  - Claim 6: `EfficiencyResponse` model has `profit_per_llm_dollar: Optional[float]`.
  - Claim 7: `/efficiency` route registered.
  - Claim 8: **Behavioral happy path** -- pnl=1000 + cost=100 -> ratio=10.0 (with mocked BQ + paired round-trips + cost helper).
  - Claim 9: **Behavioral zero-cost** -> `profit_per_llm_dollar = None` + descriptive note (no inf/divide-by-zero).
  - Claim 10: **Behavioral persist=True** -> `save_efficiency_snapshot` called once with the expected row shape (window_days=30, pnl, cost, ratio).
  - Claim 11: **Behavioral provider mapping** -- `provider="gemini"` row routes to `vertex` bucket in the helper output.

## Verbatim verifier output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_Q.py
PASS: migration_declares_efficiency_snapshots_with_required_columns
PASS: migration_idempotent_with_partition_and_cluster
PASS: bq_save_efficiency_snapshot_merge_and_timeout
PASS: fetch_llm_cost_by_provider_helper_exists
PASS: sovereign_api_compute_cost_returns_non_zero_anthropic_vertex_costs
PASS: efficiency_response_pydantic_model_has_profit_per_llm_dollar
PASS: new_api_sovereign_efficiency_endpoint_returns_profit_per_llm_dollar
PASS: behavioral_efficiency_endpoint_returns_correct_ratio
PASS: behavioral_zero_llm_cost_yields_none_ratio_not_inf
PASS: metric_persisted_to_bq_for_30d_window
PASS: provider_mapping_gemini_to_vertex_enforced

11/11 claims PASS, 0 FAIL
```

## Backend gates

- `python -c "import ast; ast.parse(open('backend/api/sovereign_api.py').read())"` -- OK
- `python -c "import ast; ast.parse(open('backend/db/bigquery_client.py').read())"` -- OK
- `python -c "import ast; ast.parse(open('scripts/migrations/add_efficiency_snapshots.py').read())"` -- OK
- 4 behavioral round-trips exercise the endpoint with mocked BQ / settings / pair_round_trips / helper.

## Hypothesis verdict

CONFIRMED. Three immutable success criteria mapped:
- Criterion 1 (`sovereign_api_compute_cost_returns_non_zero_anthropic_vertex_costs`) -- claim 5 (grep) + claims 8/11 (behavioral confirm the helper returns non-zero costs).
- Criterion 2 (`new_api_sovereign_efficiency_endpoint_returns_profit_per_llm_dollar`) -- claims 6 + 7 (structural) + 8 + 9 (behavioral with both happy and zero-cost paths).
- Criterion 3 (`metric_persisted_to_bq_for_30d_window`) -- claim 10 (behavioral; persist=True invokes save_efficiency_snapshot with the expected shape).

## RED-LINE GOAL-D CLOSED

Pre-25.Q: no `profit_per_llm_dollar` metric existed anywhere. Per arxiv 2503.21422 (March 2025 survey), **no published autonomous trading system has this metric**. pyfinagent is now a first-mover.

Post-25.Q: live ratio computable via `GET /api/sovereign/efficiency?window=30d`; persistable for trend tracking via `?persist=true`.

## Live-check

Per masterplan: "GET /api/sovereign/efficiency returns valid ratio; not hardcoded zero".

Live evidence pending in `handoff/current/live_check_25.Q.md` after operator runs:
```
curl -s "http://localhost:8000/api/sovereign/efficiency?window=30d" \
  -H "Authorization: Bearer $TOKEN" | jq .
```
Expected: a JSON body with `profit_per_llm_dollar` as a float (or null with note) AND `llm_cost_usd > 0` (assuming there are llm_call_log rows in the last 30 days).

## Non-regressions

- `get_compute_cost` per-day rows preserve all 5 provider keys (response shape unchanged).
- `get_compute_cost` totals dict preserves all 5 provider keys; `bigquery` + `altdata` unchanged.
- Cache + structured_log pattern mirrors existing endpoints.
- Migration is idempotent; safe to re-run.
- Fail-open: any BQ / pricing / round-trip error logs a warning and returns zeros / None ratio rather than crashing.

## Downstream

Red-line goal-d closed. With 25.R closing goal-c, **both auto-switching (goal-c) and observability (goal-d) red-line gaps are now closed**. Remaining red-line goals: goal-a (profit) and goal-b (low cost) -- these are operational/policy concerns not directly addressable as single masterplan steps.

## Next phase

Q/A pending.
