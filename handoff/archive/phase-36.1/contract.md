# phase-36.1 -- Scale-Out Take-Profit Wiring (last code BLOCK on profit-protection)

**Step id:** `phase-36.1`
**Date:** 2026-05-22
**Mode:** EXECUTION (backend code change). One harness pass per /goal directive.
**Author:** Main
**Cycle:** Cycle 14 (after Cycle 13 phase-35.1).

---

## North-star delta (mandated by /goal)

**Terms:** P (primary) + B (secondary).

**P (primary, quantified backtest estimate):** scale-out at +2R/+3R captures gains earlier vs riding trail-only-to-stop. Per closure_roadmap §5: **+0.3-0.8 Sharpe on backtest fixtures; +5-15% per-trade capture-ratio above today's 0.63 baseline** (COHR's recent close had capture_ratio=0.63 on MFE 28.36% / MAE -6.09%; a 2R/3R ladder would have locked in 50% at 2R=16% and remainder at 3R=24% rather than waiting for trail-stop at 26% which fired at 17.89% realized = capture ~0.63). Conservative discount via Caltech arxiv:2502.15800 (LLM agents systematically deviate from human traders -- but this step has NO LLM in the decision path; it's deterministic on MFE thresholds, so the Caltech discount is N/A).

**B (secondary):** ~1 extra BQ write per scale-out fire; ~0 added LLM calls. Negligible Burn impact.

**How measured:** Walk-forward backtest Sharpe pre-36.1 vs post-36.1 on the canonical 5-year fixture. Live: capture_ratio mean shift across the next 10+ closed trades. Deferred to phase-43.0 DoD aggregate measurement.

---

## Research-gate decision

**Researcher SKIPPED** (justified per /goal conditional clause). closure_roadmap §2 OPEN-2 already references 31.0-F4 with precise diagnosis ("partial-close primitive in execute_sell(quantity=...) exists; caller wiring missing"). research_brief.md cycle 12 already cited AFML triple-barrier method + AQR adaptive-regime. No new external pattern needed.

---

## Hypothesis

> If we add a `paper_scale_out_enabled` Field (default False), a
> `check_scale_out_fires()` helper in `paper_trader.py` that fires
> `execute_sell(qty=*0.5, reason="take_profit_2R")` at MFE>=2R and
> `execute_sell(qty=remainder, reason="take_profit_3R")` at MFE>=3R, an
> idempotency `scale_out_levels_hit` JSON column via an `--verify`-able
> migration, and a wiring call from autonomous_loop Step 5.4 (after
> mark_to_market, before kill-switch); THEN the next cycle that runs
> with the flag enabled will produce partial-close `paper_trades` rows
> at the right thresholds, idempotently (re-fires same cycle = no-op),
> with the existing execute_sell -> bq write path unchanged.

If true: phase-36.1 closes. OPEN-2 (last code BLOCK on profit-protection per closure_roadmap §2) cleared.

---

## Immutable success criteria (verbatim from masterplan 36.1.verification)

1. `synth_position_with_mfe_2_1R_triggers_50_percent_partial_close`
2. `synth_position_with_mfe_3_1R_triggers_remainder_close`
3. `idempotent_re_fire_in_same_cycle_is_no_op`
4. `paper_trades_emits_partial_close_row_with_reason_take_profit_2R`
5. `scale_out_levels_hit_column_added_via_idempotent_migration`

Plus /goal integration gates 1-10 (pytest>=297 baseline, flag default OFF, BQ migration idempotent, no emojis, ASCII loggers, single source of truth, log-first/flip-last, etc.).

---

## Plan steps

| # | Step | Status |
|---|---|---|
| 1 | Pre-step health + locate partial-close primitive (line 290 execute_sell exists; line 423 mark_to_market) | DONE |
| 2 | Researcher SKIP (justified) | DONE |
| 3 | Write this contract | IN FLIGHT |
| 4 | `backend/config/settings.py` -- new Field `paper_scale_out_enabled: bool = Field(False, ...)` | DONE |
| 5 | `backend/services/paper_trader.py` -- new method `check_scale_out_fires()` + `_persist_scale_out_levels()` helper | DONE |
| 6 | `backend/services/autonomous_loop.py` -- wire `check_scale_out_fires` into Step 5.4 (between mark_to_market and kill-switch eval) | DONE |
| 7 | `scripts/migrations/add_scale_out_levels_hit_column.py` -- idempotent BQ column migration with --verify mode | DONE |
| 8 | `backend/tests/test_phase_36_1_scale_out.py` -- 9 tests covering flag states + both fires + idempotency + null-column compat + migration verify | DONE (9 pass; pytest count 302->311) |
| 9 | live_check_36.1.md + Q/A + harness_log Cycle 14 + flip 36.1 status to done | IN FLIGHT |

---

## Files this step touches

- `backend/config/settings.py` (+1 Field, ~2 lines)
- `backend/services/paper_trader.py` (+110 lines: `check_scale_out_fires` + `_persist_scale_out_levels`)
- `backend/services/autonomous_loop.py` (+18 lines: Step 5.4 wiring with fail-open try/except)
- `scripts/migrations/add_scale_out_levels_hit_column.py` (NEW, 73 lines)
- `backend/tests/test_phase_36_1_scale_out.py` (NEW, 164 lines, 9 tests)
- `handoff/current/contract.md` (this)
- `handoff/current/live_check_36.1.md` (post-Q/A)
- `handoff/current/evaluator_critique.md` (Q/A overwrite)
- `handoff/harness_log.md` (Cycle 14 append)
- `.claude/masterplan.json` (status flip 36.1 at the very end)

**NOT changed:**
- `outcome_tracker.py`, `bigquery_client.py` (single source of truth preserved)
- Any frontend file
- The existing `execute_sell` partial-close primitive (line 290) is REUSED, not duplicated

---

## References

- closure_roadmap.md §2 OPEN-2 (the only OPEN code BLOCK)
- closure_roadmap.md §5 N* delta table (+0.3-0.8 Sharpe / +5-15% capture-ratio)
- research_brief.md cycle 12 (AFML triple-barrier, AQR adaptive-regime, Lopez de Prado)
- paper_trader.py:290-394 (existing execute_sell partial-close primitive)
- paper_trader.py:423-491 (mark_to_market loop with phase-32.1 + 32.2 stops)
- /goal directive (10 integration gates + N* delta mandate + circuit breakers)
