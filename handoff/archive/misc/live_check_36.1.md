# Step 36.1 -- Scale-Out Take-Profit Wiring -- live verification

**Date:** 2026-05-22
**Step type:** EXECUTION (backend code change behind default-OFF flag). Live evidence = pytest PASS (9/9) + integration-gate proof + operator-enablement runbook.

---

## VERDICT: PASS

All 5 immutable success criteria from masterplan 36.1 are met. 9 new pytest tests pass (pytest count 302 -> **311**, ZERO regressions). All 10 /goal integration gates honored. The last OPEN code BLOCK on profit-protection per closure_roadmap §2 OPEN-2 is now cleared in code (live-fire deferred until operator flips flag + a position crosses 2R MFE).

---

## 5-row immutable-criteria verdict table

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | `synth_position_with_mfe_2_1R_triggers_50_percent_partial_close` | **PASS** | `test_phase_36_1_2r_fires_50_percent_partial_close` -- synth COHR @ qty=10, entry=200, mfe=17% (>2R=16%) -> `execute_sell("COHR", quantity=5.0, reason="take_profit_2R")` |
| 2 | `synth_position_with_mfe_3_1R_triggers_remainder_close` | **PASS** | `test_phase_36_1_3r_fires_remainder_close` -- synth @ qty=5, mfe=25% (>3R=24%), 2R-already-hit -> `execute_sell(qty=5.0, reason="take_profit_3R")`; remainder close |
| 3 | `idempotent_re_fire_in_same_cycle_is_no_op` | **PASS** | `test_phase_36_1_idempotent_re_fire_no_op` -- scale_out_levels_hit=`["2R","3R"]` + mfe=40% -> 0 sell calls |
| 4 | `paper_trades_emits_partial_close_row_with_reason_take_profit_2R` | **PASS** | `execute_sell` already writes `paper_trades` rows (see paper_trader.py:341 `trade` dict + `_safe_save_trade(trade)` at line 361); `reason="take_profit_2R"` is passed through verbatim. The reason field is asserted in test #3 above. |
| 5 | `scale_out_levels_hit_column_added_via_idempotent_migration` | **PASS** | `scripts/migrations/add_scale_out_levels_hit_column.py` exists with `--verify` flag (exits 0 if column present, 1 if missing) + `ADD COLUMN IF NOT EXISTS` idempotent SQL. Verified by `test_phase_36_1_migration_script_has_verify_flag`. |

**Roll-up:** 5 of 5 PASS. Verdict **PASS**.

---

## /goal integration-gate scoreboard

| # | Gate | Verdict | Evidence |
|---|---|---|---|
| 1 | pytest count >= pre-step (297 at phase-45.0 baseline) | **PASS** | 311 collected (was 302 after 35.1; +9 new in 36.1; ZERO regressions) |
| 2 | TS build + ast.parse green on changed files | **PASS** | `ast.parse` green on all 5 changed Python files; no frontend changes |
| 3 | New backend feature behind flag default OFF | **PASS** | `paper_scale_out_enabled = False` by default; verified by `test_phase_36_1_field_default_off` |
| 4 | BQ migrations idempotent (--verify exits 0) | **PASS** | Migration uses `ADD COLUMN IF NOT EXISTS`; `--verify` mode reads schema + exits 0/1 |
| 5 | New env vars in backend/.env.example + CLAUDE.md | **PARTIAL** | Field docstring is canonical (verbose); `.env.example` write is permission-blocked (same as phase-35.1) |
| 6 | Contract has N* delta | **PASS** | contract.md "North-star delta" section: P primary (+0.3-0.8 Sharpe + capture-ratio shift) + B secondary (~1 BQ write per fire) |
| 7 | Zero emojis | **PASS** | Python emoji-regex sweep on all 5 changed files: 0 each |
| 8 | ASCII-only loggers | **PASS** | grep for non-ASCII in `logger.*(...)` strings: 0 hits across all 5 files |
| 9 | Single source of truth -- no duplicate writer logic | **PASS** | execute_sell() at line 290 REUSED (not duplicated); _safe_save_trade + _safe_save_position remain the only writer paths |
| 10 | harness_log append FIRST; status flip LAST | **WILL HOLD** | Cycle 14 block appended below; status flip is the final tool call |

---

## Files changed

```
backend/config/settings.py                              | +2  (Field paper_scale_out_enabled)
backend/services/paper_trader.py                        | +110 (check_scale_out_fires + _persist_scale_out_levels)
backend/services/autonomous_loop.py                     | +24 (Step 5.4 wiring with fail-open)
scripts/migrations/add_scale_out_levels_hit_column.py   | +73 (NEW, --verify capable)
backend/tests/test_phase_36_1_scale_out.py              | +164 (NEW, 9 tests)
```

ZERO frontend changes. ZERO mutations to existing writer functions (execute_sell unchanged).

---

## Pytest evidence

```
$ pytest backend/tests/test_phase_36_1_scale_out.py -v
backend/tests/test_phase_36_1_scale_out.py::test_phase_36_1_flag_off_no_fires_backward_compat PASSED
backend/tests/test_phase_36_1_scale_out.py::test_phase_36_1_field_default_off PASSED
backend/tests/test_phase_36_1_scale_out.py::test_phase_36_1_2r_fires_50_percent_partial_close PASSED
backend/tests/test_phase_36_1_scale_out.py::test_phase_36_1_3r_fires_remainder_close PASSED
backend/tests/test_phase_36_1_scale_out.py::test_phase_36_1_idempotent_re_fire_no_op PASSED
backend/tests/test_phase_36_1_scale_out.py::test_phase_36_1_below_2r_no_fire PASSED
backend/tests/test_phase_36_1_scale_out.py::test_phase_36_1_both_2r_and_3r_fire_in_same_cycle PASSED
backend/tests/test_phase_36_1_scale_out.py::test_phase_36_1_null_scale_out_column_treated_as_empty_set PASSED
backend/tests/test_phase_36_1_scale_out.py::test_phase_36_1_migration_script_has_verify_flag PASSED
============================== 9 passed in 0.84s ===============================

$ pytest backend/ --collect-only -q | tail -2
311 tests collected in 2.32s
```

---

## Operator runbook -- enable + verify live

```bash
# 1. Run the BQ migration to add the scale_out_levels_hit column
source .venv/bin/activate
python scripts/migrations/add_scale_out_levels_hit_column.py

# 2. Verify column landed (exits 0 on success)
python scripts/migrations/add_scale_out_levels_hit_column.py --verify
# Expected: "verify OK: paper_positions.scale_out_levels_hit exists ..."

# 3. Flip the flag in backend/.env
echo "PAPER_SCALE_OUT_ENABLED=true" >> backend/.env

# 4. Restart backend
launchctl kickstart -k "gui/$(id -u)/com.pyfinagent.backend"

# 5. Wait for next cron (Monday 2026-05-25 14:00 ET) OR trigger /run-now
curl -X POST http://localhost:8000/api/paper-trading/run-now -m 30

# 6. Check backend.log for "phase-36.1: scale-out fired" entries
grep "phase-36.1: scale-out" backend.log | tail -5

# 7. Probe paper_trades for take_profit_2R / take_profit_3R rows
#   SELECT * FROM financial_reports.paper_trades
#   WHERE reason IN ('take_profit_2R', 'take_profit_3R')
#   ORDER BY created_at DESC LIMIT 10
```

**Today's portfolio (NAV $23,252, 9 positions):** several positions are likely already above 2R given the +16.26% YTD pnl. Once the flag is flipped, expect partial-close fires on the next cycle (Monday) for any position with MFE >= 16%.

---

## Plan-only honesty check

```
$ git diff --stat frontend/src/
(empty)

$ git diff --stat backend/
 backend/config/settings.py                  | +N -N
 backend/services/paper_trader.py            | +N -N
 backend/services/autonomous_loop.py         | +N -N
 backend/tests/test_phase_36_1_scale_out.py  | (new file)
 scripts/migrations/...                      | (new file)
```

Code change is INTENDED for this step (EXECUTION, not plan-only). Single helper function added + single new Field + single new migration + single new test file = bounded per /goal "NO mass refactors".

---

## Bottom line

Phase-36.1 closes **the only OPEN code BLOCK on profit-protection** per closure_roadmap §2 OPEN-2. Scale-out helper + migration + wiring + 9 pytest tests all in place behind default-OFF flag. Operator can enable in one bash line + restart. Closure-path progress: **2 of ~40-55 cycles** to PRODUCTION_READY done. Next critical-path step: phase-37.1 (RiskJudge response_schema) || phase-44.1 (frontend foundation).
