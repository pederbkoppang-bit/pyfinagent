# live_check — phase-85.4

Required evidence shape (from `.claude/masterplan.json`, `verification.live_check`):

> live_check_85.4.md with the re-derived cycle_history terminal-status counts,
> the measured analysis-phase duration, the fault-injected proof that a
> mid-analysis death produces both a terminal row and an alert, and a BQ read
> showing whether paper_portfolio_snapshots resumed

All four, below, verbatim. Captured 2026-08-08 21:00–22:15 UTC+2 (cycle 183).

---

## 1. Re-derived cycle_history terminal-status counts

```
$ bash -c 'python3 -c "import json,collections;rows=[json.loads(l) for l in open(\"handoff/cycle_history.jsonl\") if l.strip()];term=[r for r in rows if r.get(\"status\") in (\"completed\",\"timeout\",\"error\")];print(collections.Counter(r[\"status\"] for r in term[-10:]))"'
Counter({'completed': 7, 'timeout': 3})
exit=0
```

The immutable verification command **cannot fail** — it prints a histogram and
exits 0 whatever the content. Recorded as a reporter, not a gate.

The counter is byte-identical to the contract's baseline because **no cycle has
run since**: the paper-trading cron is weekday-only and this work spans the
night into Sunday 2026-08-09. **No cycle was triggered by this step**, and none
of the two verification cycles authorised by the overnight goal was spent here
(85.4's criteria are provable without one; 85.6's live leg is what may need them).

---

## 2. Measured analysis-phase duration

```
$ source .venv/bin/activate && python scripts/diagnostics/measure_analysis_phase.py --log backend.log
log=backend.log  lines_parsed=162395  cycles_with_analysis_phase=3
budget_sec=7200
```

| cycle | screening | analysis phase | tickers fin/disp | mean/ticker | eff. par. | projected cycle | verdict |
|---|---|---|---|---|---|---|---|
| 2026-08-05 | 570.9s | 5098.9s (reached mark_to_market) | 6/6 | 1632.8s | 1.92 | 5670s | within budget −1530s |
| 2026-08-06 | 428.8s | 6771.3s (cycle_timeout) | 5/6 (NTAP) | 2310.6s | 1.71 | **8554s** | **OVER +1354s** |
| 2026-08-07 | 554.9s | 6645.2s (cycle_timeout) | 5/6 (NTAP) | 2319.9s | 1.75 | **8529s** | **OVER +1329s** |

Per-ticker wall-clock, 2026-08-07 (from the loop's own dispatch/persist lines):
`{'CRWD': 2176.4, 'DELL': 2468.0, 'HPE': 2313.3, 'HUM': 2048.8, 'PANW': 2593.0}`
NTAP dispatched at 21:24:33 (4517s into the phase) and never finished.

cc_rail subprocess timeout rate in-window: **14.9% / 18.1% / 23.4%** against a
150s cap (`backend/agents/claude_code_client.py:593`).

Full JSON: `handoff/current/analysis_phase_measurement_85.4.json`.

**Answer to criterion 1: 7200s is too short for the current ticker count.** The
phase does not hang. See `experiment_results_85.4.md` §2 for the 08-05
counterfactual (that cycle only fit because two analyses failed fast).

---

## 3. Fault-injected proof — terminal row AND phase-named alert

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_85_4_cycle_loudness.py -q --timeout=120
....                                                                     [100%]
4 passed, 1 warning in 13.68s
```

These drive the real `autonomous_loop.run_daily_cycle` with the fault injected
inside the analysis phase, with `cycle_history.jsonl`, the heartbeat, the cycle
lockfile and `raise_cron_alert_sync` all redirected to `tmp_path`/recorders.

| fault | terminal row written | alert |
|---|---|---|
| timeout mid-analysis (300s sleep vs 10s budget) | `status="timeout"`, `completed_at` set, exactly one row | P1, title contains `analyzing`, `details.died_in_phase == "analyzing"` |
| crash mid-analysis (`dispatch_analyses` raises) | `status="error"`, one row | P1, phase named in title and details |
| kill-switch halt (injected paused state) | `status="halted_kill_switch"`, one row — **asserted `!= "running"`** | P1, `"running"` absent from title, `died_in_phase == "kill_switch_halted"` |

The timeout test asserts its own precondition (the analysis phase was actually
entered) so it cannot pass vacuously by dying in screening instead.

### Mutation matrix — the guards can fail

```
$ source .venv/bin/activate && python scripts/qa/mutation_matrix_85_4.py
MUTATION MATRIX PASSED -- 9/9 mutations killed, tree restored byte-for-byte, suite green.
```

Two mutations were **LIVE on the first run** and exposed real defects in my own
tests (a source-scan guard, and a test coupled to the operator's live
kill-switch state). Both fixed; both written up in `experiment_results_85.4.md` §7.

---

## 4. BQ read — did `paper_portfolio_snapshots` resume?

**No.** The BigQuery MCP is not attached in this session, so this is the Python
client per CLAUDE.md BigQuery rule 6.

```
$ source .venv/bin/activate && python -c "<bigquery.Client query>"
snapshot_date  total_nav  trades_today  position_count
2026-08-05     23830.46   0             1
2026-08-03     23803.94   0             1
2026-07-31     23770.98   1             1
2026-07-30     23772.49   0             0
2026-07-29     23772.49   0             0
2026-07-28     23772.49   0             0
2026-07-27     23772.49   0             0
2026-07-24     23838.16   0             2
2026-07-22     23896.61   0             2
2026-07-21     23887.61   0             2
```

- **Latest snapshot: 2026-08-05.** Nothing for 08-06, 08-07 or 08-08 — both
  timed-out cycles died before `mark_to_market`, so no snapshot was written.
  Consistent with §2 and with the P1 `paper_portfolio_snapshots last_tick_age_sec`
  alarm in the step's audit basis.
- **The 08-05 row exists because the kill-switch halt path still snapshots**
  (`save_daily_snapshot` runs before the early return) — a halted cycle marks the
  book but trades nothing. `trades_today=0` on that row is the visible symptom.
- **Last row with `trades_today > 0` is 2026-07-31**, matching "no trade in 7 days".
- **85.4 does not and cannot change this.** Snapshots resume only when cycles
  complete; the remedies are operator asks #23/#24/#25 (config + dark flag), and
  trading additionally requires **85.6**.

---

## 5. Full-suite regression — no new failures

```
$ source .venv/bin/activate && python -m pytest backend/tests -q --timeout=120 --tb=no
26 failed, 3017 passed, 12 skipped, 5 xfailed, 1 xpassed, 48 warnings in 353.60s (0:05:53)
```

Baseline before this step (recorded in the overnight goal, proven pre-existing
by reverting the phase-85.5 production files and replaying):
**26 failed, 2985 passed**.

- **failures: 26 → 26**, count unchanged.
- **passes: 2985 → 3017 = +32**, exactly the 32 tests this step adds
  (4 loudness + 12 completed-age + 16 dispatch-barrier).

The 26 failing node ids, recorded so the next session compares a **set**, not a
count:

```
test_64_3_currency_path.py::test_64_3_currency_path_kr_avg_entry_stays_krw
test_64_3_currency_path.py::test_64_3_currency_path_eu_avg_entry_stays_eur
test_64_3_currency_path.py::test_64_3_currency_path_us_byte_identical
test_64_4_multi_market_e2e.py::test_64_4_multi_market_e2e_currency_invariants
test_book_safety_69.py::test_valid_nav_still_breaches
test_dod4_tier1_coverage_investment.py::test_paper_trader_execute_buy_average_up_recomputes_avg_entry
test_phase_23_2_15_verify_23_1_smoke.py::test_phase_23_2_15_known_pass_scripts_still_pass
test_phase_23_2_4_pause_resume_no_deadlock_live.py::test_phase_23_2_4_live_pause_resume_pause_cycle_under_5s
test_phase_23_2_6_sector_cap_emit.py::test_phase_23_2_6_backend_log_has_skipping_buy_evidence
test_phase_40_2_claude_code_v2_1_140_features.py::test_phase_40_2_settings_json_still_valid_json_after_edit
test_phase_57_1_reject_binding.py::test_reject_binding_main_path_off_emits_on_blocks
test_phase_57_1_reject_binding.py::test_reject_binding_swap_path_off_emits_on_blocks
test_phase_57_1_reject_binding.py::test_off_identity_prompts_are_verbatim_constants
test_phase_60_3_data_integrity.py::test_60_3_flag_defaults_off
test_phase_70_3_atomic_swap.py::test_avg_entry_fx_fix_local_consistent_for_kr
test_phase_70_4_gate_observability.py::test_price_tolerance_rejection_is_accumulated
test_phase_70_4_gate_observability.py::test_price_tolerance_accumulator_empty_when_within_tolerance
test_phase_75_17_verification_paths.py::test_masterplan_diff_touches_only_the_ten_sibling_insertions
test_phase_75_17_verification_paths.py::test_sweep_shape_census_matches_the_corrected_figures
test_phase_75_prompt_contracts.py::test_operator_decision_note_exists_with_token
test_phase_75_sre_ops.py::test_c1_runbook_and_operator_token_drafted
test_phase_82_39_outcome_rebuild_query.py::test_the_sweeps_recall_limit_is_recorded_not_assumed
test_portfolio_swap.py::test_swap_framework_fills_zero_buy_gap
test_price_tolerance_gate.py::test_price_tolerance_pass_1pct_deviation
test_price_tolerance_gate.py::test_price_tolerance_zero_disables_gate
test_price_tolerance_gate.py::test_price_tolerance_skipped_when_analysis_price_missing
```

**Honest limit on this claim:** I verified the failure **count** is unchanged and
that the delta in passes exactly equals my new test count. I did **not** capture
the pre-existing 26 node ids myself — the "identical set" finding is inherited
from the overnight goal's cycle-182 measurement. The list above is now recorded
so the next session can diff sets rather than counts.

---

## 6. Disclosure — this step wrote to the live kill-switch audit journal

The overnight goal predicted this and asked for disclosure if it recurred. It did.

```
BEFORE: 51 rows   handoff/kill_switch_audit.jsonl
AFTER : 52 rows
new row: {"ts": "2026-08-08T19:59:35.278544+00:00", "event": "pause", "trigger": "manual", "details": {}}
```

- **One** new row, from the full-suite run in §5. `event=pause`, `trigger=manual`.
- **Zero resumes.** The switch never left its fail-safe state; no order path opened.
- The damage is to provenance only: live `paused_at` now reads
  `2026-08-08T19:59:35Z`. The **real** operator pause remains
  **2026-08-04T11:43:31Z** per the 85.4 research gate.
- This is the defect ask #21 filed against step **36.28**; that step is still
  `pending` and this step did not widen it (it is the next item on the overnight
  goal's own list, and freezing the tree during EVALUATE means it goes in the
  next cycle, not this one).
- **My own new tests do NOT write to it** — `test_phase_85_4_cycle_loudness.py`
  injects a stubbed `kill_switch.get_state` and never touches the journal. That
  was forced by mutation M9, which proved the first draft *was* coupled to it.

### A second instance of the same class, found while committing

`handoff/.cycle_heartbeat.json` — the LIVE control-plane heartbeat — was also
overwritten by the full-suite run:

```
$ git diff handoff/.cycle_heartbeat.json
-{"cycle_id": "c2", "event": "end", "updated_at": "2026-08-08T08:36:08.427947+00:00"}
+{"cycle_id": "c2", "event": "end", "updated_at": "2026-08-08T20:00:35.252534+00:00"}
```

`cycle_id: "c2"` is a **test** cycle id, and it was already there **before** this
run (the 08:36:08Z value) — so this is pre-existing pollution of the same class,
not something 85.4 introduced. A test that isolates `_HISTORY_PATH` but not
`_HEARTBEAT_PATH` writes straight through to the operator's file.

What this affects: `compute_freshness` surfaces the heartbeat as the dead-man's
-switch signal, so a stale-but-test-refreshed heartbeat could mask a dead
emitter. Fail-safe direction is NOT guaranteed here (unlike the kill-switch
rows), because refreshing a heartbeat makes a dead system look alive.

`handoff/cycle_history.jsonl` was **NOT** modified by the run — verified with
`git status`. No fabricated cycle row entered the live ledger.

Both artifacts belong in the 36.28 widening: the fix is one autouse conftest
fixture that redirects `cycle_health._HISTORY_PATH`, `cycle_health._HEARTBEAT_PATH`
and `kill_switch._AUDIT_PATH` to tmp for the whole suite, not per-file opt-in.
