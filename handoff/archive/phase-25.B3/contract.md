# Sprint Contract -- phase-25.B3 -- Daily loop reads latest promoted strategy via load_promoted_params()

**Cycle:** phase-25 cycle 15 (P1 sprint)
**Date:** 2026-05-12
**Step ID:** 25.B3
**Priority:** P1
**Depends on:** 25.A3 (done)
**Audit basis:** bucket 24.3 F-6 -- `autonomous_loop.py:33-43` only reads `optimizer_best.json`; no BQ promoted_strategies query

## Research-gate

Researcher spawned this cycle (agent af8733b71b4910383). Brief at
`handoff/current/research_brief.md`. Gate envelope: tier=moderate,
external_sources_read_in_full=6, urls_collected=16, recency_scan_performed=true,
internal_files_inspected=5, gate_passed=true.

Key research conclusions:
- New BQ reader `BigQueryClient.get_latest_promoted_strategy(status_filter=None)` returns the latest row by `promoted_at DESC, dsr DESC LIMIT 1` filtered to `status IN UNNEST(@statuses)`. Default filter `["pending", "active"]` -- "pending" included because 25.A3 hardcodes that status until 25.C3's state machine lands.
- BQ `JSON` column is non-serializable in `dict(row)`, so the SELECT uses `TO_JSON_STRING(params) AS params_json` and the reader inverts via `json.loads(...)`.
- 3-tier fallback: BQ row -> `optimizer_best.json` (existing `load_best_params`) -> `{}`. Microservices fallback-pattern + Anthropic-style local-cache fallback.
- No in-process TTL cache: daily cycle runs once per 24h, so fresh BQ read per cycle is correct.
- `result(timeout=30)` on the new query per CLAUDE.md rule.
- Logging shape: success path `"Loaded promoted params (DSR ...)"`; empty path `"No active promoted strategy in BQ, falling back to optimizer_best"`; exception path `"Promoted strategy BQ unavailable, falling back to optimizer_best: <exc>"`.

## Hypothesis

Adding `load_promoted_params(bq)` as a sibling to `load_best_params()`
plus a single caller-site edit at `autonomous_loop.py:100`
(`best_params = load_promoted_params(bq)`) wires the daily cycle to
the 25.A3 BQ subscriber WITHOUT touching `load_best_params`'s existing
contract or its existing tests. The 3-tier fallback ensures the cycle
keeps running even when BQ is unavailable.

## Success criteria (verbatim from masterplan)

1. `load_promoted_params_function_exists_in_autonomous_loop`
2. `fallback_to_optimizer_best_json_if_bq_unavailable`
3. `autonomous_cycle_logs_show_promoted_strategy_loaded`

Verification command (immutable):
`source .venv/bin/activate && python3 tests/verify_phase_25_B3.py`

Live check (per masterplan):
`BQ promoted_strategies query returns active row; autonomous_loop log confirms params merged`

## Plan

1. **BQ reader** -- `backend/db/bigquery_client.py`:
   - Add `get_latest_promoted_strategy(self, status_filter: list[str] | None = None) -> dict | None`.
   - Default `status_filter = ["pending", "active"]` when None.
   - Parameterized SQL with `bigquery.ArrayQueryParameter("statuses", "STRING", status_filter)`.
   - SELECT uses `TO_JSON_STRING(params) AS params_json`; reader pops the field and `json.loads(...)` it back to a dict (try/except -> `{}` on parse failure).
   - `result(timeout=30)` per CLAUDE.md rule.
   - Returns `None` if no row, the row dict otherwise.
2. **Loader** -- `backend/services/autonomous_loop.py`:
   - Add `load_promoted_params(bq: BigQueryClient) -> dict` immediately after `load_best_params()` (around line 44).
   - Calls `bq.get_latest_promoted_strategy()`; on success returns `row["params"]` and logs `"Loaded promoted params (DSR %s week=%s): %s"`.
   - Empty row -> logs `"No active promoted strategy in BQ, falling back to optimizer_best"` -> returns `load_best_params()`.
   - Exception -> logs `"Promoted strategy BQ unavailable, falling back to optimizer_best: %s"` -> returns `load_best_params()`.
3. **Wire caller** -- `backend/services/autonomous_loop.py:100`:
   - `best_params = load_best_params()` -> `best_params = load_promoted_params(bq)`.
   - `bq` is already in scope at that point (created at line 85 in `run_daily_cycle`).
4. **Verifier** -- `tests/verify_phase_25_B3.py` -- 10+ claims:
   - Claim 1: `load_promoted_params` function exists in `autonomous_loop.py` with signature `(bq: BigQueryClient) -> dict`.
   - Claim 2: `BigQueryClient.get_latest_promoted_strategy` exists with status_filter default `["pending", "active"]`.
   - Claim 3: BQ query uses `TO_JSON_STRING(params) AS params_json`, `ORDER BY promoted_at DESC, dsr DESC`, `LIMIT 1`, and `IN UNNEST(@statuses)`.
   - Claim 4: BQ query uses `result(timeout=30)`.
   - Claim 5: caller line at autonomous_loop.py uses `load_promoted_params(bq)` (NOT `load_best_params()`).
   - Claim 6: **Behavioral happy path** -- fake bq returns a row with params dict; `load_promoted_params(fake_bq)` returns those params + log captured contains "Loaded promoted params".
   - Claim 7: **Behavioral empty path** -- fake bq returns None; `load_promoted_params(fake_bq)` returns `load_best_params()` result; log captured contains "No active promoted strategy in BQ".
   - Claim 8: **Behavioral exception path** -- fake bq raises; `load_promoted_params(fake_bq)` still returns a dict (falls back to local JSON) AND logs `"Promoted strategy BQ unavailable"`.
   - Claim 9: **Behavioral JSON-round-trip** -- fake bq returns a row with `params_json='{"lookback":20}'` style raw payload (no params dict); reader inverts via json.loads.
   - Claim 10: malformed params_json -> returns dict with `params: {}` (no exception). This validates the parse-fail try/except in the reader.

## Non-goals

- No removal or behavior change of `load_best_params` -- it remains the fallback Tier-2.
- No status-state-machine logic (`pending` -> `active`); that's 25.C3.
- No live BQ execution; behavioral tests use fakes per the standard verifier pattern.
- No caching layer; daily cadence makes per-cycle read fine.

## References

- `handoff/current/research_brief.md` -- full brief this cycle
- `backend/services/autonomous_loop.py:33-43` (existing `load_best_params`)
- `backend/services/autonomous_loop.py:100` (caller wire point)
- `backend/services/autonomous_loop.py:85` (`bq` instance available at the wire point)
- `backend/db/bigquery_client.py:~660` (new reader sits right next to `save_promoted_strategy` from 25.A3)
- `backend/autoresearch/friday_promotion.py:162` (writer hardcodes `status="pending"` -> reader default includes it)
- CLAUDE.md `Critical Rules` -- 30s BQ timeout; fallback discipline
