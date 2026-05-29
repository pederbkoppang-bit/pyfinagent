# Evaluator Critique — phase-48.4: Live rotation bake-off SMOKE (first REAL validation)

**Verdict: PASS** | FIRST Q/A pass on 48.4 | merged Q/A (deterministic-first + LLM judgment)
**Date:** 2026-05-29 | **LLM spend this step:** $0 (quant-only backtests; real BQ + real compute)

---

## STEP 1 — Harness-compliance audit (5 items) — ALL PASS

1. **Researcher gate — PASS.** `handoff/current/research_brief_phase_48_4_live_smoke.md` exists; JSON envelope `gate_passed: true`, `external_sources_read_in_full: 5`, `urls_collected: 13`, `recency_scan_performed: true`, `internal_files_inspected: 11`. Recency-scan section ("last 2 years, 2024-2026") present. Floor (>=5 read-in-full) met.
2. **Contract — PASS, NO clobber this cycle.** `head -1 handoff/current/contract.md` == `git show 88d770bd:handoff/current/contract.md | head -1` (both: "# Contract — phase-48.4: Live rotation bake-off SMOKE (first real validation)"). The scheduled run_harness.py did NOT re-clobber to "Sprint Contract"; the committed 48.4 PLAN at 88d770bd IS the live contract. Immutable success criteria in contract.md lines 18-22 match `.claude/masterplan.json` id="48.4" `verification.success_criteria` verbatim. (The known concurrent-writer collision did not manifest for 48.4 — verified by the git-show head match — so it is a NON-issue here, not a 48.4 protocol breach.)
3. **experiment_results.md — PASS.** Present; contains the bug-fix narrative (the live smoke CAUGHT a real bug), verbatim post-fix metrics, success-criteria mapping, and 4 FINDINGS.
4. **Log-last — PASS.** `grep -c "phase=48.4" handoff/harness_log.md` == 0. No 48.4 block yet — correct; the log append is the LAST step (after this PASS, before status flip).
5. **No verdict-shopping — PASS.** First Q/A on 48.4; no prior 48.4 verdict to overturn. The evaluator_critique.md just replaced held the 48.3 verdict (a different step; archived on step close). No sycophancy-under-rebuttal surface.

---

## STEP 2 — Deterministic checks (reproduced)

| Check | Command | Result |
|-------|---------|--------|
| Immutable verification cmd | `test -f live_check_48.4.md && tail -1 rotation_log.jsonl \| python -c "...assert allocation_pct==0.0 and status=='bakeoff_verdict'..."` | **exit 0** → `rotation_log verdict row OK: no_candidate_passed_gate triple_barrier` |
| Syntax | `ast.parse` rotation_runner.py, run_rotation_smoke.py, test_phase_48_3 | **SYNTAX OK** |
| Rotation regression | `pytest test_phase_48_3_rotation_runner test_phase_48_1_* test_phase_48_2_* test_strategy_selector -q` | **40 passed, 2 skipped** (5.03s) — matches expected |

**The target_vol FIX is real + regression-safe** (`rotation_runner.py:113-120`):
```python
def _pos(v):
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None
_tv = _pos(p.get("target_vol")) or _pos(p.get("target_annual_vol")) or 0.15
```
- 0 / missing / negative → `_pos` returns None → **0.15** (engine default, standard sizing), NOT 0. Only a POSITIVE value sets a custom target. This is exactly the documented fix: the prior `is None` mapping (visible in the removed-lines of the diff) let optimizer_best's `target_annual_vol=0` pass through to `target_vol=0` → `backtest_trader.py:89 vol_scale=min(target_vol/stock_vol,3.0)=0` → zero positions → no trades → degenerate flat NAV.
- Docstring (lines 9-19, 104-112) corrected to describe the fix + the consequence (tb_baseline + tb_risk_managed both now vol-target at 0.15, differing only by tp_pct → reseed follow-up, honestly flagged).
- **The 48.3 test now asserts the CORRECTED semantics** (`test_phase_48_3_rotation_runner.py` diff): `target_annual_vol=0 → 0.15` (was the buggy `== 0`), PLUS a NEW missing-key case (`{} → 0.15`), PLUS explicit-positive-wins (`target_vol=0.2 over target_annual_vol=0.1 → 0.2`). The corrected assertion IS the regression guard.
- The narrow `except (TypeError, ValueError)` is correct float-coercion scope — NOT a broad-except risk-guard anti-pattern (negation list explicitly allows typed narrow excepts).

**Captured live evidence — validated:**
- `rotation_smoke.log`: `[smoke] scored strategy=triple_barrier: {'dsr': 1.0, 'pbo': 0.4887334887334887, 'sharpe': 1.8232784959596307, 'n_variants': 2, 'n_windows': 6}` — REAL non-degenerate metrics.
- qm degenerate → skipped HONESTLY: `[adapter] strategy 'quality_momentum': PBO matrix undersized/degenerate (need >=2 cols and >=32 rows; got 2 usable variant(s)); emitting NO pbo so the producer SKIPS (not a false-good 0.0)` then `[candidate_producer] ... qm_trend_tilt ... skipping so the gate cannot silently drop it`. The guard fired for the degenerate seed and did NOT emit a false-good 0.0. triple_barrier was NOT flagged undersized → its PBO matrix met the T>=32 floor → pbo=0.489 is genuine.
- Selector verdict: `selected_id=triple_barrier, switched=false, reason=no_candidate_passed_gate, ranked=[], num_trials=2`. Challenger dsr=1.0 vs incumbent dsr=0.9525, but pbo 0.489 > 0.20 → vetoed.
- **Diagnostic log corroborates root cause** (`diag_backtest.log`): a direct triple_barrier backtest over the SAME window 2022-01-01..2024-06-30 was healthy — `total_trades=160 aggregate_sharpe=1.7509 n_windows=6 nav_history_len=228`, real BUY trades (VRT, KLAC...). 228 nav rows ≫ the 32 T-floor. This is the smoking gun the bug was the target_vol mapping, NOT the strategy.

---

## STEP 3 — LLM judgment (adversarial)

- **`no_candidate_passed_gate` — LEGITIMATE PASS, not masking failure.** The full chain ran end-to-end on REAL backtests (make_rotation_engine full-kwarg → 4 real walk-forward backtests → nav→returns → generate_report DSR + per-strategy (T×K) compute_pbo → producer → selector → persisted row). triple_barrier passed DSR (1.0>=0.95) but its pbo 0.489 > the strict 0.20 gate → vetoed; qm degenerate → skipped → retain incumbent. This is the gate's appropriate skepticism at K=2 (coarse PBO), NOT a hidden chain bug. For a SMOKE whose purpose is plumbing + at-least-one-seed valid metrics, both are achieved. The research brief explicitly blessed this outcome.
- **Criterion #2 (FINITE valid metrics for completing seeds) — MET.** triple_barrier: dsr=1.0 (in [0,1]), **pbo=0.489 (a REAL value from a genuine T>=32 matrix — adapter did NOT omit it, confirmed by asymmetric guard behavior vs qm + the 228-row diag nav)**, sharpe=1.82 (finite), n_windows=6. NOT a degenerate 0.0.
- **No-deploy — CONFIRMED (critical).** `git status` + `git diff --name-only`: the ONLY production change is `backend/autoresearch/rotation_runner.py` (the fix) + the test correction; new scripts `run_rotation_smoke.py` / `diag_rotation_backtest.py`; handoff/audit artifacts. ZERO edits to autonomous_loop / portfolio_manager / paper_trader / decide_trades / kill_switch / settings. Grep of the diff for `settings.paper_*` assignment / `MERGE INTO promoted_strategies` / `execute_buy|execute_sell` / `allocation_pct>0` → only documentation lines (describing audit-only nature), no actual mutation. allocation_pct=0.0 in both persisted rows. $0 LLM (quant-only).
- **Honesty — strong.** All 4 findings flagged as follow-ups in experiment_results FINDINGS (lines 28-32) + live_check: (1) target_vol bug FIXED this cycle + disclosed; (2) qm no-trades on 2022-2024 → qm-strategy investigation deferred; (3) tb_baseline/tb_risk_managed redundancy post-fix → reseed deferred; (4) K=2 PBO coarse → real bake-off needs K~8-16. Fixing a 48.3 file WITHIN 48.4 is appropriate (the live smoke is precisely the validation that surfaced the bug the $0 mock tests missed — they tested the mapping arithmetic, not the trader's target_vol=0 no-trade semantics) and is fully disclosed in the docstring + corrected test.

### Code-review heuristics (all 5 dimensions evaluated) — no BLOCK/WARN
- secret-in-diff: clean. subprocess/eval/exec: clean. print() in non-script: clean.
- broad-except-silences-risk-guard: N/A — the only added except is narrow `(TypeError, ValueError)` in `_pos()` float coercion (negation-list allowed).
- kill-switch-reachability / stop-loss / paper-trader / perf-metrics-bypass / max-position-bypass: N/A — no execution-path or perf-metrics file touched.
- financial-logic-without-behavioral-test: SATISFIED — the target_vol logic change is covered by the corrected `test_make_rotation_engine_maps_target_annual_vol_to_target_vol` (0→0.15, missing→0.15, positive-wins), which passed in the 40-test run. NOT a tautological/over-mocked test.
- sycophancy / second-opinion-shopping: N/A (first Q/A, no prior 48.4 verdict).

---

## NOTE (PASS-with-flag, does NOT degrade verdict)

- **2 rotation_log.jsonl rows, vs "exactly ONE" in criterion #3 wording.** `wc -l` = 2; both `no_candidate_passed_gate`, both allocation_pct=0.0, both window 2022-01-01..2024-06-30. These are the first (degenerate, pre-fix) run's persisted row + the post-fix run's row — both AUDIT-ONLY appends, neither a deploy side-effect. The immutable verification command checks `tail -1` (the authoritative post-fix verdict row), which is correct; experiment_results/live_check say "last row" referring to it. `_persist_verdict` appends one row per run by design, so running the bake-off twice (before + after the fix) yields two harmless rows at allocation_pct=0. Flagged for transparency; strict single-row dedup is a trivial follow-up if ever wanted. Benign — NOTE, not a blocker.

---

## Verdict rationale

**PASS.** The 48.1-48.3 rotation machinery is validated end-to-end on REAL backtests for the first time; triple_barrier produced genuine finite metrics (dsr=1.0, pbo=0.489 from a real T>=32 matrix corroborated by a 228-row healthy diagnostic backtest, sharpe=1.82, n_windows=6); the target_vol no-trade bug was a REAL bug the live smoke caught (the whole point of "verify live"), FIXED correctly (`_pos` → 0.15 floor, only positive sets a custom target), regression-guarded by the corrected 48.3 test (40 passed / 2 skipped); zero deploy side-effects ($0 LLM, allocation_pct=0, no settings/promoted_strategies/execution mutation); and all 4 findings (1 fixed, 3 deferred) are honestly flagged. The `no_candidate_passed_gate` outcome is the gate working correctly (coarse K=2 PBO appropriately vetoed triple_barrier; qm degenerate skipped) — NOT a masked chain bug. The only deviation from the literal contract ("exactly ONE row") is a benign 2nd audit-row from the before/after-fix runs, both harmless at allocation_pct=0 — a NOTE.

**violated_criteria: []** (none).

**checks_run:** harness_compliance_audit, syntax, verification_command, rotation_regression_suite (40p/2s), target_vol_fix_review, captured_live_evidence_review, diagnostic_log_corroboration, no_deploy_sideeffect_check, code_review_heuristics, evaluator_critique
