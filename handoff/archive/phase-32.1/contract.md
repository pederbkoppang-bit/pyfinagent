# Sprint Contract — phase-32.1 Breakeven-Stop Ratchet at +1R

**Step ID:** `phase-32.1`
**Date:** 2026-05-21
**Cycle type:** Implementation. Code edits + migration + tests. NOT another audit.

---

## Research-Gate Summary (per `.claude/rules/research-gate.md`)

- **Tier:** moderate.
- **Brief:** `handoff/current/research_brief.md` (researcher subagent output).
- **gate_passed:** true.
- **Headline:** Audit P1.1 stands. Direct primary-text re-read of Kaminski-Lo MIT PDF confirms Proposition 2 is about TRAILING (cumulative-loss threshold), not breakeven (one-shot move-up). No new finding 2026-05-20 → 2026-05-21 contradicts the audit's recommendation. `rg -n` confirmed NO duplicate `_advance_stop` / breakeven helper exists in `backend/` (only kill-switch peak-NAV ratchet at `paper_trader.py:710` + `kill_switch.py:185` + math-metric `break_even_win_rate` in `backtest/analytics.py:417,440` — all unrelated). Migration template = `scripts/migrations/add_external_flow_today_column.py` (phase-30.4 pattern). MFE units = percent-of-cost-basis float (e.g., 8.0 means 8%) — confirmed via direct read of `paper_trader.py:435-446`. Shares units with `settings.paper_default_stop_loss_pct`. Adversarial recheck: practitioner concern is whipsaw-at-entry; +1R trigger gives ~8% buffer (1R = default 8%) before whipsaw stop-out, mitigating the concern.

---

## Hypothesis

When `position.mfe_pct >= settings.paper_default_stop_loss_pct` (1R, default 8%), advance `stop_loss_price` from `entry × 0.92` (or whatever entry-anchored level it was) up to `entry_price` itself. Once advanced, the position can never roundtrip to a loss. Idempotent (one-shot per position via `stop_advanced_at_R` audit field). Strictly monotonic (never moves the stop down). Strict Pareto improvement over current entry-anchored static stop.

Per phase-31.0 BQ baseline, 8 of 11 current positions (SNDK +57.64% MFE, MU +57.62%, INTC +53.85%, COHR +28.36%, WDC +27.75%, LITE +19.50%, ON +19.49%, DELL +19.14% — all > 8%) crossed +1R and will gain a breakeven floor on the FIRST mark-to-market after this lands. This is the load-bearing live-check signal.

---

## Success Criteria (IMMUTABLE — copied verbatim from `.claude/masterplan.json::phase-32.1.verification.success_criteria`)

1. `_advance_stop_helper_in_paper_trader`
2. `called_from_mark_to_market_before_mfe_write`
3. `mfe_geq_1R_mutates_stop_to_entry`
4. `stop_advanced_at_R_audit_field_added_nullable`
5. `backfill_high_mfe_positions_on_first_run`
6. `unit_test_4_cases_pass` (we will ship 7 test cases — the spec said ≥4)
7. `no_regression_check_stop_losses`

Verification command (copy-paste, must pass):

```bash
python -m pytest backend/tests/test_phase_32_1_breakeven_ratchet.py -v && \
grep -n '_advance_stop' backend/services/paper_trader.py && \
python -c "import ast; ast.parse(open('backend/services/paper_trader.py').read())"
```

Live check requirement: `handoff/current/live_check_32.1.md` contains a BQ row from `financial_reports.paper_positions` showing `stop_advanced_at_R` populated AND `stop_loss_price = avg_entry_price` (NOT `entry × 0.92`) for at least one current high-MFE position. Query run AFTER migration is applied AND after at least one `mark_to_market` invocation (manual via a one-shot script if the autonomous loop is paused).

---

## Immutable Hard Guardrails (copied verbatim from `implementation_plan.hard_guardrails`)

1. NO trailing-stop logic (HWM, Chandelier, ATR-trail) — that is phase-32.2.
2. NO take-profit / scale-out logic — that is a later phase.
3. NO change to `check_stop_losses` at `paper_trader.py:484-493`.
4. NO change to `decide_trades` at `portfolio_manager.py`.
5. NO change to `risk_judge.md` or any agent skill `.md` file.
6. NO mutating Alpaca calls (read-only inspection via `mcp__alpaca__*` permitted).
7. Migration MUST be idempotent (`ADD COLUMN IF NOT EXISTS` + schema-error retry).
8. MFE/MAE write and the ratchet update MUST go through the SAME `_safe_save_position` call (no race condition).
9. Adversarial guard NOT NEEDED for breakeven specifically (Kaminski-Lo Proposition 2 is about TRAILING; breakeven is a one-shot mutation). Audit confirmed.

Additional global guardrails from the overnight goal directive:

10. NO `AskUserQuestion`.
11. NO mutating Alpaca calls (read-only `mcp__alpaca__*` OK).
12. Scope honesty: post-cycle `git diff --stat` must touch ONLY: `backend/services/paper_trader.py`, `backend/tests/test_phase_32_1_breakeven_ratchet.py` (NEW), `scripts/migrations/phase_32_1_add_stop_advanced_at_R.py` (NEW), `.claude/masterplan.json`, `handoff/current/*`, `handoff/harness_log.md`, `handoff/archive/phase-31.0/*` (the prior-cycle archive move done at pre-flight). Any out-of-scope diff → revert and re-spawn qa.

---

## Plan Steps

1. **RESEARCH** ✅ done. Brief at `handoff/current/research_brief.md`. gate_passed=true.
2. **PLAN** ✅ this file.
3. **GENERATE** — implement `_advance_stop` helper + wire into `mark_to_market` + update `_POSITION_RT_FIELDS` per `code_steps` STEP 1-5.
4. **MIGRATION** — create `scripts/migrations/phase_32_1_add_stop_advanced_at_R.py` matching the phase-30.4 template, run with `--apply`, verify via `mcp__claude_ai_Google_Cloud_BigQuery__get_table_info`, re-run for idempotency.
5. **TESTS** — write `backend/tests/test_phase_32_1_breakeven_ratchet.py` covering 7 cases per `test_specs`. Run full sweep `python -m pytest backend/tests/ -v` to confirm zero regression.
6. **VERBATIM RESULTS** — write `handoff/current/experiment_results.md` with pytest output + pre/post migration schema diff + file-touch list.
7. **EVALUATE** — spawn `qa` ONCE with 5-item harness-compliance audit first; CONDITIONAL/FAIL → fix + FRESH qa. CIRCUIT BREAKER: max 2 retries → mark `blocked` + STOP.
8. **LIVE CHECK** — invoke `PaperTrader.mark_to_market()` against the live BQ (a manual one-shot via a Python REPL one-liner, NOT via the autonomous loop which is paused), then query `paper_positions` and write `handoff/current/live_check_32.1.md` quoting verbatim rows for SNDK / MU / INTC / COHR showing `stop_advanced_at_R` populated AND `stop_loss_price = entry`.
9. **LOG** — append cycle block to `handoff/harness_log.md` BEFORE the status flip.
10. **FLIP** — `phase-32.1.status: pending → done`. Auto-commit hook; if stalled, manual commit with prefix `phase-32.1:`.

---

## Implementation crib (from researcher brief)

**Wire-in site at `paper_trader.py:435-446`** (verbatim):

```python
            market_value = pos["quantity"] * live_price
            cost_basis = pos.get("cost_basis") or (pos["quantity"] * pos["avg_entry_price"])
            pnl = market_value - cost_basis
            pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0.0

            # 4.5.2: MFE/MAE tracked monotonically across the position's holding period.
            # MFE = best unrealized_pnl_pct seen; MAE = worst (lowest). Reset only when
            # the position is fully closed (handled by execute_sell).
            prev_mfe = float(pos.get("mfe_pct") or 0.0)
            prev_mae = float(pos.get("mae_pct") or 0.0)
            new_mfe = max(prev_mfe, pnl_pct)
            new_mae = min(prev_mae, pnl_pct)
            # ── PHASE-32.1 HOOK HERE: call self._advance_stop(pos, new_mfe) ──
```

Helper goes near `_update_portfolio_cash` around line 737. `_POSITION_RT_FIELDS` at line 751 gets `"stop_advanced_at_R"` appended.

**Migration template** = `scripts/migrations/add_external_flow_today_column.py` (phase-30.4). Copy structure, change PROJECT/DATASET/TABLE/COLUMN constants and the description.

---

## References

- Researcher brief at `handoff/current/research_brief.md` (this cycle, moderate-tier, gate_passed=true).
- Phase-31.0 audit at `handoff/archive/phase-31.0/experiment_results.md` (the deep canonical reference for P1.1 with file:line anchors and BQ baseline).
- Migration template at `scripts/migrations/add_external_flow_today_column.py` (phase-30.4 canonical pattern).
