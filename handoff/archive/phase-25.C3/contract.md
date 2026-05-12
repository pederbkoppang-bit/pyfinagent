# Sprint Contract -- phase-25.C3 -- Strategy registry status field; flip actual_replacement

**Cycle:** phase-25 cycle 16 (P1 sprint)
**Date:** 2026-05-12
**Step ID:** 25.C3
**Priority:** P1
**Depends on:** 25.B3 (done)
**Audit basis:** bucket 24.3 F-4 -- `monthly_champion_challenger.py:75` (and line 263) hard-codes `actual_replacement=False`; no BQ status flip on monthly HITL approval

## Research-gate

Researcher spawned this cycle (agent a2e0bccf5ac186363). Brief at
`handoff/current/research_brief.md`. Gate envelope: 7 sources read in full,
17 URLs, recency scan performed, gate_passed=true.

Key research conclusions:
- **No schema change needed.** The 25.A3 status enum (`pending|active|paused|superseded|rolled_back`) is a strict superset of the masterplan criterion's informal "active_shadow_retired". Mapping: shadow = `pending`, active = `active`, retired = `superseded`.
- **Two hardcoded `actual_replacement=False` sites:** line 75 (result dict in `run_monthly_sortino_gate`) and line 263 (notes string in `_emit_deployment_log_row`).
- **Derivation pattern:** `actual_replacement = real_capital_enabled AND approved`. The feature flag defaults to `False` at the Settings level so the paper-only invariant holds by default. Injecting via kwarg keeps the module pure-library and testable.
- **`record_approval` gets a `status_update_fn: Callable[[str, str], None] | None = None` kwarg** (same fail-open injection pattern as the existing `bq_fn`). When `status='approved'`, call `status_update_fn(challenger_id, "active")`.
- **New BQ helper `update_promoted_strategy_status(strategy_id, new_status, *, week_iso=None)`** -- parameterized UPDATE, `result(timeout=30)`.
- **Prior-active-row supersession is OUT OF SCOPE** for 25.C3 -- criterion 3 only requires the new-row flip from `pending` to `active`.

## Hypothesis

Replacing the hardcoded `actual_replacement=False` with a derived value
(driven by an injected `real_capital_enabled` flag that defaults to False),
plus wiring `record_approval` to call an injectable `status_update_fn`
that flips the BQ row's status `pending -> active` on approval, plus
adding the supporting BQ helper -- closes phase-24.3 F-4 while preserving
the SR 11-7 paper-only invariant. The Settings-level `real_capital_enabled`
flag defaults to False, so behavior is identical to today unless an operator
explicitly toggles it.

## Success criteria (verbatim from masterplan)

1. `strategy_registry_table_has_status_field_active_shadow_retired`
2. `actual_replacement_no_longer_hardcoded_false`
3. `monthly_approval_flips_status_from_shadow_to_active`

Verification command (immutable):
`source .venv/bin/activate && python3 tests/verify_phase_25_C3.py`

Live check (per masterplan):
`Monthly HITL approval test flips registry status atomically`

## Plan

1. **Settings flag** -- `backend/config/settings.py`:
   - Add `real_capital_enabled: bool = Field(False, description="SR 11-7 paper-only gate. Must remain False until compliance review wires real-capital deployment. Read by monthly_champion_challenger.run_monthly_sortino_gate.")`.
2. **BQ helper** -- `backend/db/bigquery_client.py`:
   - Add `update_promoted_strategy_status(self, strategy_id: str, new_status: str, *, week_iso: str | None = None) -> None`.
   - Parameterized UPDATE on `pyfinagent_data.promoted_strategies`.
   - When `week_iso is not None`, WHERE narrows to `strategy_id AND week_iso` (precise targeting).
   - When `week_iso is None`, updates all rows with that `strategy_id` (intended for testing / single-row scenarios; tests will pin to a specific week to verify the narrow path).
   - `result(timeout=30)` per CLAUDE.md.
3. **Module edits** -- `backend/autoresearch/monthly_champion_challenger.py`:
   - Add `real_capital_enabled: bool = False` kwarg to `run_monthly_sortino_gate` (keyword-only, defaults False -- preserves all existing callers).
   - Line 75: replace `"actual_replacement": False` with `"actual_replacement": False` only if the gate hasn't fired yet; for the post-fire path (after `result["gate_pass"] = True`), set `result["actual_replacement"] = bool(real_capital_enabled)`. (Note: the value is meaningful only when the gate fires and gets approved; the gate-fire path will set it pre-approval based on the flag.)
   - Add `status_update_fn: Callable[[str, str], None] | None = None` kwarg to `record_approval`. When `status='approved'` and the transition succeeds (not already terminal, not expired), call `status_update_fn(challenger_id, "active")` inside a try/except so the in-memory transition still completes on BQ failure.
   - Line 263: replace literal `actual_replacement=False` with derived value `actual_replacement={row.get('actual_replacement', False)}` (or thread the value through from the caller). Easiest: store the gate-fire `actual_replacement` value in the state row at line 173-182, then read it back in `_emit_deployment_log_row`.
4. **Verifier** -- `tests/verify_phase_25_C3.py` -- 10+ claims:
   - Claim 1: `Settings.real_capital_enabled` exists with default `False`.
   - Claim 2: `BigQueryClient.update_promoted_strategy_status` exists with the documented signature.
   - Claim 3: UPDATE SQL is parameterized (uses `@new_status` and `@strategy_id` named params); `result(timeout=30)` called.
   - Claim 4: `run_monthly_sortino_gate` signature accepts `real_capital_enabled: bool = False` kwarg.
   - Claim 5: line 75-area no longer has a literal `"actual_replacement": False` hardcoded (it's now derived).
   - Claim 6: line 263-area no longer hardcodes `actual_replacement=False`.
   - Claim 7: `record_approval` signature accepts `status_update_fn: Callable[[str, str], None] | None = None`.
   - Claim 8: **Behavioral approval flip** -- pass a fake `status_update_fn` MagicMock; call `record_approval(month_key, status="approved", state_path=<temp>, status_update_fn=fake_fn)`. Assert: `fake_fn.call_count == 1` AND `fake_fn.call_args == ((challenger_id, "active"),)`.
   - Claim 9: **Behavioral rejection no-flip** -- pass `status="rejected"`. Assert `fake_fn.call_count == 0`.
   - Claim 10: **Behavioral derived actual_replacement** -- call `run_monthly_sortino_gate` with `real_capital_enabled=True`; gate-fire result should have `actual_replacement == True`. With `real_capital_enabled=False` (default) the result has `actual_replacement == False`.
   - Claim 11: **Behavioral BQ UPDATE round-trip** -- direct test of `update_promoted_strategy_status` with a fake BQ client; assert `client.query` was called with the expected SQL pattern AND with parameter `new_status="active"`.
   - Claim 12: status names mentioned in 25.A3 migration include `active` and at least one of `superseded`/`retired`/`paused` (matches the criterion mapping).

## Non-goals

- No prior-active-row supersession (deferred per research finding 8).
- No real-capital deployment (flag defaults False; SR 11-7 invariant preserved).
- No multi-statement BQ transaction; single UPDATE per call is atomic enough for this scope.
- No frontend changes.

## References

- `handoff/current/research_brief.md` -- full brief this cycle
- `backend/autoresearch/monthly_champion_challenger.py:75, 173-182, 207, 263` -- edit sites
- `backend/db/bigquery_client.py:~720-760` (existing 25.A3 + 25.B3 helpers; new helper sits next to them)
- `backend/config/settings.py:141` (existing paper_trading flag; new flag mirrors shape)
- `scripts/migrations/create_promoted_strategies_table.py` -- 25.A3 schema (no change)
- CLAUDE.md `Critical Rules` -- 30s BQ timeout
