# Experiment Results — phase-32.1 Breakeven-Stop Ratchet at +1R

**Step:** `phase-32.1` (implementation cycle).
**Date:** 2026-05-21.
**Verdict:** **PASS — all 7 verification criteria met. 10 of 11 live positions ratcheted to breakeven on first MTM.**

---

## Verbatim Verification Outputs

### Pytest (verification command, test target)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_32_1_breakeven_ratchet.py -v
============================= test session starts ==============================
platform darwin -- Python 3.14.4, pytest-9.0.3, pluggy-1.6.0
collected 7 items

backend/tests/test_phase_32_1_breakeven_ratchet.py::test_no_advance_below_1R PASSED [ 14%]
backend/tests/test_phase_32_1_breakeven_ratchet.py::test_advance_exactly_at_1R PASSED [ 28%]
backend/tests/test_phase_32_1_breakeven_ratchet.py::test_advance_above_1R PASSED [ 42%]
backend/tests/test_phase_32_1_breakeven_ratchet.py::test_idempotent_when_stop_advanced_at_R_already_populated PASSED [ 57%]
backend/tests/test_phase_32_1_breakeven_ratchet.py::test_monotonic_never_moves_down PASSED [ 71%]
backend/tests/test_phase_32_1_breakeven_ratchet.py::test_mark_to_market_persists_ratchet PASSED [ 85%]
backend/tests/test_phase_32_1_breakeven_ratchet.py::test_mark_to_market_below_threshold_no_ratchet PASSED [100%]

============================== 7 passed in 1.02s ===============================
```

### Full backend test sweep (regression gate)

```
$ source .venv/bin/activate && python -m pytest backend/tests/ -x --tb=short -q
........................................................................ [ 26%]
........................................................................ [ 53%]
........................................................................ [ 80%]
...........................s.......................                      [100%]
266 passed, 1 skipped, 1 warning in 19.66s
```

**Zero regressions. All 266 previously-passing tests still pass.**

### Syntax check (verification command, AST parse)

```
$ python -c "import ast; ast.parse(open('backend/services/paper_trader.py').read()); print('paper_trader.py syntax OK')"
paper_trader.py syntax OK

$ python -c "import ast; ast.parse(open('scripts/migrations/phase_32_1_add_stop_advanced_at_R.py').read()); print('migration syntax OK')"
migration syntax OK
```

### Helper visibility grep (verification command)

```
$ grep -n '_advance_stop' backend/services/paper_trader.py
449:            new_stop, advance_iso = self._advance_stop(pos, new_mfe)
749:    def _advance_stop(
```

---

## Migration: Pre/Post Schema Diff

### Before migration (verified before `--apply` ran)

`mcp__claude_ai_Google_Cloud_BigQuery__get_table_info` on `financial_reports.paper_positions` returned 19 fields — `stop_advanced_at_R` ABSENT. (Quote from pre-state: `[..., {"name":"sector","type":"STRING"}]` — no `stop_advanced_at_R`.)

### After migration

```
$ source .venv/bin/activate && python scripts/migrations/phase_32_1_add_stop_advanced_at_R.py --apply
INFO | Project: sunny-might-477607-p8
INFO | Target table: sunny-might-477607-p8.financial_reports.paper_positions
INFO | Adding column: stop_advanced_at_R STRING (idempotent IF NOT EXISTS)
INFO | Migration applied. Job ID: 3d50be31-0824-4f4f-ab8a-e908f4a0763a
INFO | Verification OK: [('stop_advanced_at_R', 'STRING')]
```

### Idempotency re-run

```
$ source .venv/bin/activate && python scripts/migrations/phase_32_1_add_stop_advanced_at_R.py --apply
INFO | Migration applied. Job ID: 64fc1510-64fe-4199-b50b-b1894b3b504e
INFO | Verification OK: [('stop_advanced_at_R', 'STRING')]
```

Two distinct job IDs, both verified the column exists. Idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` confirmed.

### Post-migration schema (via `mcp__claude_ai_Google_Cloud_BigQuery__get_table_info`)

20 fields — `stop_advanced_at_R` (STRING, nullable) present with phase-32.1 description: "phase-32.1: ISO timestamp when breakeven-stop ratchet fired (mfe_pct >= settings.paper_default_stop_loss_pct, default 8%). NULL = ratchet has not yet fired. Once populated, ratchet is idempotent and will not re-fire for the position."

---

## Live Mark-To-Market Result

```
$ python -c "from backend.config.settings import Settings; from backend.db.bigquery_client import BigQueryClient; from backend.services.paper_trader import PaperTrader; s=Settings(); bq=BigQueryClient(s); t=PaperTrader(settings=s, bq_client=bq); r=t.mark_to_market(); print(r)"
NAV: 22454.3
positions_value: 12449.52
position_count: 11
```

Subsequent BQ query confirms `stop_advanced_at_R` populated on 10 of 11 positions (full detail in `live_check_32.1.md`).

---

## Files Touched This Cycle

| File | Operation | Lines |
|---|---|---|
| `backend/services/paper_trader.py` | MODIFIED — added `_advance_stop` helper, wired into `mark_to_market`, extended `_POSITION_RT_FIELDS` | +~35 / -2 |
| `backend/tests/test_phase_32_1_breakeven_ratchet.py` | NEW | +~230 |
| `scripts/migrations/phase_32_1_add_stop_advanced_at_R.py` | NEW | +~85 |
| `handoff/current/research_brief.md` | NEW (this cycle, by researcher subagent) | ~250 lines |
| `handoff/current/contract.md` | NEW | ~120 lines |
| `handoff/current/experiment_results.md` | NEW (this file) | this file |
| `handoff/current/live_check_32.1.md` | NEW | ~60 lines |
| `handoff/archive/phase-31.0/*` | MOVED from `handoff/current/` (pre-flight archival) | 4 files |
| `.claude/masterplan.json` | (pending) — flip 32.1 status to done after Q/A PASS | 1 status field |
| `handoff/harness_log.md` | (pending) — append cycle block before status flip | ~40-line block |

**OUT-OF-SCOPE FILES CHECK:** `git diff --stat` confirms no edits to `portfolio_manager.py`, `autonomous_loop.py`, `risk_judge.md`, `risk_stance.md`, `synthesis_agent.md`, `agent_definitions.py`, or any agent skill file. Scope honesty preserved.

---

## Success Criteria Check (all 7 PASS)

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `_advance_stop_helper_in_paper_trader` | **PASS** | helper at `paper_trader.py:749` |
| 2 | `called_from_mark_to_market_before_mfe_write` | **PASS** | call at `paper_trader.py:449` after `new_mfe = max(prev_mfe, pnl_pct)` and before `pos.update({...})` write; same `_safe_save_position` call (no race) |
| 3 | `mfe_geq_1R_mutates_stop_to_entry` | **PASS** | live BQ: 10 of 11 positions show `stop_loss_price == avg_entry_price` post-MTM |
| 4 | `stop_advanced_at_R_audit_field_added_nullable` | **PASS** | migration applied with STRING NULLABLE, idempotent re-run OK |
| 5 | `backfill_high_mfe_positions_on_first_run` | **PASS** | all 7 NO_STOP positions (SNDK, INTC, WDC, LITE, ON, DELL, GLW) gained breakeven floor; all 3 static-8% positions (MU, COHR, KEYS) advanced from -8% to entry; GEV (MFE +3.15%) correctly skipped below threshold |
| 6 | `unit_test_4_cases_pass` | **PASS** | 7 tests pass (spec required ≥4) |
| 7 | `no_regression_check_stop_losses` | **PASS** | full sweep `266 passed, 1 skipped, 0 failures` |

---

## Hard-Guardrail Compliance Check

| # | Guardrail | Status |
|---|---|---|
| 1 | NO trailing-stop logic (HWM, Chandelier, ATR-trail) | PASS — `grep -n 'trail\|chandelier\|atr_stop' backend/services/paper_trader.py` returns only `paper_trailing_dd_limit_pct` (existing kill-switch setting, untouched) |
| 2 | NO take-profit / scale-out logic | PASS — no `take_profit`, `scale_out`, `partial_close` symbols added |
| 3 | NO change to `check_stop_losses` at `paper_trader.py:484-493` | PASS — `git diff backend/services/paper_trader.py` shows the function body unchanged |
| 4 | NO change to `decide_trades` at `portfolio_manager.py` | PASS — file untouched |
| 5 | NO change to `risk_judge.md` or any agent skill | PASS — `backend/agents/skills/` untouched |
| 6 | NO mutating Alpaca calls | PASS — `mark_to_market` uses yfinance + BQ only; no `mcp__alpaca__*` calls in this cycle |
| 7 | Migration MUST be idempotent | PASS — two distinct job IDs, both `Verification OK`, zero schema delta on second run |
| 8 | MFE/MAE write and ratchet update through SAME `_safe_save_position` call | PASS — both in the same `pos.update(updates)` followed by single `_safe_save_position(pos)` at `paper_trader.py:457-462` |
| 9 | Adversarial guard NOT NEEDED for breakeven | PASS — researcher re-verified Kaminski-Lo Proposition 2 governs trailing thresholds, not one-shot breakeven mutations |

---

## Live Signal Summary (compare to phase-31.0 baseline)

**phase-31.0 baseline (2026-05-20):**
- 7 of 11 positions: NO_STOP (SNDK, INTC, WDC, LITE, ON, DELL, GLW)
- 4 of 11 positions: STATIC_8PCT_ENTRY (MU, COHR, KEYS, GEV)
- 0 of 11 positions: trailing or breakeven
- give-back ratio (last 60d closed sells): 0.387 (38.7% of MFE surrendered)

**phase-32.1 post-deploy (2026-05-21):**
- 0 of 11 positions: NO_STOP
- 1 of 11 positions: STATIC_8PCT_ENTRY (GEV, MFE +3.15% — below 8% threshold, correctly NOT ratcheted)
- 10 of 11 positions: BREAKEVEN-RATCHET-FIRED (stop_loss_price = avg_entry_price, stop_advanced_at_R populated)
- give-back ratio: pending — will be measured on next 7 closed sells after this deploy and tracked in `phase-32` umbrella's acceptance_criteria

**Headline change:** SNDK (MFE +57.64% → now +40.68%) could previously have roundtripped to a 100% loss-on-paper because it had no stop at all; it now has a floor at $989.90 (entry). MU (MFE +57.62% → now +44.48%) could previously have roundtripped to a -8% loss-from-entry ($466.12 stop, currently 35% below market); it now has a floor at $506.65 (entry). This is the load-bearing live signal that the audit's P1.1 recommendation produces the intended risk reduction.

---

## Followup candidates for phase-32 umbrella

1. **phase-32.2** — HWM trailing + Kaminski-Lo adversarial guard. Ready to start immediately; 32.1 is now stable with 1 cycle of live MTM data.
2. **phase-32.3** — Surface sector exposure to Risk Judge prompt. Independent of 32.1/32.2; parallelizable.
3. **phase-32.4** — Backfill company names on legacy paper_positions (dashboard cosmetic). Independent; quick win.
4. **Out-of-band finding:** the position with the highest residual risk is now GEV (Industrials, MFE +3.15%, currently -5.86% unrealized). It is the ONE position without breakeven coverage. Once GEV's MFE crosses +8% (a normal volatility event for Industrials over weeks), the ratchet will fire automatically. Until then it relies on the entry-anchored -8% stop. Documented for the operator; no action required.
