# Sprint Contract -- phase-25.Q -- Real-time profit_per_llm_dollar (closes red-line goal-d)

**Cycle:** phase-25 cycle 18 (P1 sprint)
**Date:** 2026-05-12
**Step ID:** 25.Q
**Priority:** P1
**Depends on:** 25.A9 (done; pricing premium 2.0x verified)
**Audit basis:** bucket 24.13 F-2 -- `sovereign_api.py:386-390` hardcoded `anthropic=0.0, vertex=0.0, openai=0.0`; no `profit_per_llm_dollar` metric anywhere

## Research-gate

Researcher spawned this cycle (agent ace64aaae945ee80a). Brief at
`handoff/current/research_brief.md`. Gate envelope: 6 sources read in full,
17 URLs, recency scan performed, gate_passed=true.

Key research conclusions:
- **First-mover metric:** no published trading system has `profit_per_llm_dollar`. Closest analogue is i10x.ai's Performance-per-Dollar Index. Definition: `profit_per_llm_dollar = realized_pnl_usd / llm_cost_usd` over a configurable window (default 30 days). Zero-denominator handling: return `None` (renderer shows "n/a"), not infinity.
- **Hardcoded zeros at `sovereign_api.py:386-390`** are the immediate target for criterion 1.
- **Provider name mapping:** `llm_call_log` uses `"gemini"`; `ProviderCostPoint` uses `"vertex"`. The new helper must map.
- **P&L numerator:** reuse `bq.get_paper_trades_in_window(30)` + `pair_round_trips()` from `backend/services/paper_round_trips.py`. Don't write new SQL for P&L; the existing helper is tested.
- **Pricing join:** import `MODEL_PRICING` from `cost_tracker.py:20-76`. Apply same cache-premium math as `cost_tracker.py:147-154` (cache_read=0.1x; cache_write=2.0x). The `llm_call_log` schema does NOT have a `cost_usd` column, so the join is Python-side.
- **Endpoint:** `GET /api/sovereign/efficiency?window=7d|30d|90d&persist=false`.
- **Persistence:** new BQ table `pyfinagent_data.efficiency_snapshots` via idempotent CREATE migration; MERGE on `(snapshot_date, window_days)` for upsert semantics.

## Hypothesis

Wiring (a) a `_fetch_llm_cost_by_provider(days)` helper that queries
`llm_call_log` + joins `MODEL_PRICING` in Python; (b) using it to
replace the hardcoded zeros in `get_compute_cost`; (c) a new
`GET /api/sovereign/efficiency` endpoint that divides realized P&L by
the LLM cost; (d) optional persistence to a new `efficiency_snapshots`
table -- closes red-line goal-d. The metric is a first-mover, so the
contract definition must be unambiguous and zero-denominator-safe.

## Success criteria (verbatim from masterplan)

1. `sovereign_api_compute_cost_returns_non_zero_anthropic_vertex_costs`
2. `new_api_sovereign_efficiency_endpoint_returns_profit_per_llm_dollar`
3. `metric_persisted_to_bq_for_30d_window`

Verification command (immutable):
`source .venv/bin/activate && python3 tests/verify_phase_25_Q.py`

Live check (per masterplan):
`GET /api/sovereign/efficiency returns valid ratio; not hardcoded zero`

## Plan

1. **Migration** -- `scripts/migrations/add_efficiency_snapshots.py`:
   - Idempotent CREATE TABLE IF NOT EXISTS on `pyfinagent_data.efficiency_snapshots` with 8 columns (snapshot_date, window_days, profit_per_llm_dollar, realized_pnl_usd, llm_cost_usd, anthropic_cost_usd, vertex_cost_usd, openai_cost_usd, computed_at).
   - PARTITION BY snapshot_date, CLUSTER BY window_days.
   - Defaults to dry-run + `--apply`.
2. **BQ helper** -- `backend/db/bigquery_client.py`:
   - Add `save_efficiency_snapshot(row)` MERGE on natural key `(snapshot_date, window_days)`. Mirror `save_promoted_strategy` shape. `result(timeout=30)`.
3. **LLM-cost helper** -- `backend/api/sovereign_api.py` (new module-level function):
   - `_fetch_llm_cost_by_provider(days: int) -> dict` -- query `llm_call_log` with `WHERE ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY) AND ok = TRUE`, SELECT `provider, model, SUM(input_tok), SUM(output_tok)` GROUP BY provider, model.
   - Apply `MODEL_PRICING` lookup per (provider, model); when missing, use `_DEFAULT_PRICING`.
   - Compute cost = `(input_tok * input_price + output_tok * output_price) / 1_000_000` per model row.
   - Aggregate to per-provider sums. Map `"gemini" -> "vertex"`.
   - Return `{"anthropic": float, "vertex": float, "openai": float, "_per_model": [...]}` or similar.
   - Fail-open: any exception logs + returns zeros (preserves existing endpoint's robustness).
4. **Fix compute-cost hardcodes** -- `sovereign_api.py:386-390`:
   - Replace per-day hardcoded zeros with values from `_fetch_llm_cost_by_provider(days)`. For simplicity in this cycle: compute the WINDOW total and split evenly across days (daily-level granularity in `llm_call_log` is possible but adds query complexity; punt to a follow-up if the operator wants per-day).
   - Update `totals` dict to use the same.
5. **Efficiency endpoint** -- `sovereign_api.py`:
   - New `EfficiencyResponse` Pydantic model with fields: `window`, `profit_per_llm_dollar: Optional[float]`, `realized_pnl_usd: float`, `llm_cost_usd: float`, `anthropic_cost_usd`, `vertex_cost_usd`, `openai_cost_usd`, `computed_at: str`, `note: Optional[str]`.
   - New route `@router.get("/efficiency", response_model=EfficiencyResponse)` with `window: Literal["7d", "30d", "90d"] = Query("30d")` and `persist: bool = Query(False)`.
   - Compute realized P&L for the window via `bq.get_paper_trades_in_window(days)` + `pair_round_trips()`. Sum `realized_pnl_usd` across round-trips.
   - LLM cost via the new helper. Sum all three providers + bigquery + altdata for `llm_cost_usd` (or strictly LLM = anthropic+vertex+openai; document the choice).
   - `profit_per_llm_dollar = realized_pnl / llm_cost` when `llm_cost > 0`, else `None`.
   - If `persist=True`, call `bq.save_efficiency_snapshot(...)`.
   - Cache + structured log per existing endpoint pattern.
6. **Verifier** -- `tests/verify_phase_25_Q.py` -- 10+ claims:
   - Claim 1: migration file exists with `pyfinagent_data.efficiency_snapshots` + all 8 columns.
   - Claim 2: migration CREATE TABLE IF NOT EXISTS + PARTITION + CLUSTER.
   - Claim 3: `BigQueryClient.save_efficiency_snapshot` exists; MERGE on (snapshot_date, window_days) + `result(timeout=30)`.
   - Claim 4: `_fetch_llm_cost_by_provider(days)` exists in sovereign_api.py.
   - Claim 5: `get_compute_cost` no longer hardcodes `anthropic=0.0, vertex=0.0, openai=0.0` in the per-day rows AND `totals` dict.
   - Claim 6: `EfficiencyResponse` Pydantic model exists with profit_per_llm_dollar field.
   - Claim 7: route `@router.get("/efficiency", ...)` exists in sovereign_api.py.
   - Claim 8: **Behavioral round-trip** -- monkey-patch BQ client + pair_round_trips. Call endpoint logic. Assert: P&L computed, LLM cost > 0, ratio = P&L / cost (or None if cost==0).
   - Claim 9: **Behavioral zero-cost** -- llm_cost=0 -> profit_per_llm_dollar = None (NOT inf/0/crash). 
   - Claim 10: **Behavioral persist=True** -- call with persist=True; `save_efficiency_snapshot` mock invoked once.
   - Claim 11: provider mapping `gemini -> vertex` enforced in the helper output.

## Non-goals

- No per-day granularity for the LLM-cost split in the existing `compute-cost` endpoint (window total split evenly across days for first pass; finer granularity in a follow-up if needed).
- No frontend changes.
- No live `--apply` of the migration (operator-gated per CLAUDE.md BQ rules).
- No new pricing rows in `MODEL_PRICING` -- the existing table is the source of truth.

## References

- `handoff/current/research_brief.md` -- full brief this cycle
- `backend/api/sovereign_api.py:354-420` (`get_compute_cost`, hardcoded zeros at 386-390)
- `backend/agents/cost_tracker.py:20-76` (`MODEL_PRICING`) and 136-154 (cache-premium math)
- `backend/db/bigquery_client.py:get_paper_trades_in_window` (25.A11) for the P&L window
- `backend/services/paper_round_trips.py::pair_round_trips` for realized P&L per trade
- `scripts/migrations/add_llm_call_log.py` (existing migration; informs the new one)
- `docs/audits/phase-24-2026-05-12/24.13-redline-synthesis-findings.md` (goal-d definition)
- CLAUDE.md `Critical Rules` -- 30s BQ timeout
