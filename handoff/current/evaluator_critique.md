# Evaluator Critique — Step 69.2 (P0 gate correctness, OFFLINE)

## Cycle 1 — Q/A verdict: FAIL (independent Q/A, workflow structured-output, 2026-07-11)

Evaluator: fresh `qa` agent run via the Workflow structured-output path (the Agent-tool subagent
path stalled 6× on the file-write this session; the workflow captured the verdict as a return
value). Read-only; Main authors this file from the agent's structured verdict.

**Verdict: FAIL** — on exactly ONE deterministic gate; *absent it, this evaluation is a PASS*.

### Harness compliance (5/5 PASS)
- **research_gate**: PASS — research_brief_69.2.md gate_passed=true, 5 external sources read in full
  (building on 8 in 69.0), recency scan, 11 internal anchors, provenance disclosed, research preceded code.
- **contract_before_generate**: DISCLOSED SLIP, judged **acceptable-not-blocking** — actual order was
  research → code → contract; the full plan pre-existed in research_brief_69.2.md §Application (written
  during the gate, before code), immutable criteria copied verbatim/unaltered, the slip is self-disclosed
  at the top of contract.md. Rule intent (no post-hoc criteria fitting) preserved. Must not recur as a pattern.
- **results_present**: PASS — complete (diff stat, 5-fix inventory, verbatim pytest, scipy DSR check,
  do-no-harm evidence, honest C4 scope note, deferral of incumbent re-validation).
- **log_last**: PASS — no 69.2 harness_log entry yet, status not flipped.
- **no_verdict_shopping**: PASS — first Q/A for 69.2.

### Immutable criteria
- **C1 DSR**: MET — both SR and V de-annualized, caller ppy=252, Bailey **0.9004880** pinned, default
  byte-identical, `dsr_52wh_verdict.py` untouched.
- **C2 Purge+embargo**: MET — purge wired into `_build_training_data` with the test window, separating the
  full 1.5·holding_days label horizon (strictly dominates the old 5-day embargo; walk_forward embargo_days
  constant itself unchanged, purge subsumes its pre-test-leakage role).
- **C3 Boundary snap**: MET — `_price_asof` at entry + liquidation; snap and no-data paths tested.
- **C4 Fracdiff-at-predict**: **ACCEPT-ON-INTENT** (non-blocking) with a mandatory tracked follow-on.
  NaN-fill half now literally identical (train medians persisted + applied at predict); windowed FFD on a
  single cross-sectional row is architecturally impossible, and the series lives in the live-adjacent
  build_feature_vector (out of a zero-live-surface step). Honest limit: predict non-stationary features are
  median-NEUTRALIZED (constant), not equivalent to train's varying fracdiff'd values — disclosed
  harm-reduction. **Condition of acceptance**: file a tracked follow-on for the true per-ticker FFD fix +
  document the neutralization. (68.5 precedent: disclosure+routing acceptable; silent reinterpretation is not.)
- **C5 Go-live booleans**: MET — true 30-day expanding-window min-PSR sustainment (fail-safe red on short
  history), DD tolerance = backtest-DD+5pp with documented 20% cap fallback, PSR via
  `services/perf_metrics.compute_psr` (no perf-metrics bypass).

### Do-no-harm: VERIFIED
DSR_THRESHOLD / PSR_THRESHOLD / MAX_DD_ABS_TOLERANCE / TRADES_THRESHOLD byte-untouched; quant_optimizer
dsr_threshold=0.95 untouched; the only external `_build_training_data` consumer (`run_ablation.py:253`)
calls positionally so new kwargs default to None = pre-fix behavior; 1028 tests still collect. Code-review
heuristics: financial-logic-with-behavioral-test satisfied (18 fixtures), no tautological assertions, no
consumer break, no security findings; `_load_backtest_max_dd` broad-except is a fail-safe config read → None
→ stricter fallback (NOTE only).

### The single blocker (violated_criteria)
`ruff_lint_gate_F401` (qa.md §1a): `uvx ruff check --select F821,F401,F811 ...` exit=1 —
verbatim: **"F401 [*] `numpy` imported but unused --> backend/tests/test_gate_correctness_69.py:13:17"**.
The three production files are lint-clean; the sole finding is the new test file's dead import. The gate text
is binding ("Non-zero exit = FAIL"); dead imports are its explicit target class → FAIL (correctly not softened).

---

## Cycle 2 — Main remediation (2026-07-11)

Per the documented cycle-2 flow (fix the blocker → update handoff files → fresh Q/A on changed evidence):

1. **F401 fixed**: removed the unused `import numpy as np` from `backend/tests/test_gate_correctness_69.py:13`
   (verified no `np.` references remain).
2. **Re-verified** (verbatim):
   - `uvx ruff check --select F821,F401,F811 <4 files>` → **"All checks passed!"** (exit 0).
   - `python -m pytest backend/tests/test_gate_correctness_69.py -q` → **18 passed in 1.48s**.
3. **C4 follow-on filed** (condition of acceptance satisfied): `handoff/current/audit_phase69/followons_69.2.md`
   (FO-69.2-A: true per-ticker time-series FFD in the feature builder, future live-adjacent step, gated on the
   historical_macro un-freeze) + the median-neutralization LIMITATION documented in the `_build_predict_features`
   docstring. Routed as a 69.4 hand-off seed.
4. **experiment_results.md updated** with the new ruff+pytest outputs and the C4 follow-on record.

Evidence changed (F401 removed; ruff now clean; follow-on filed) → a FRESH Q/A evaluates the updated state.
This is the documented file-based fresh-respawn, not verdict-shopping on unchanged evidence.

## Cycle 2 — Q/A verdict: PASS (fresh independent Q/A, workflow structured-output, 2026-07-11)

Fresh `qa` agent (Workflow structured-output). **Verdict: PASS**, `violated_criteria: []`.

Independently verified the CHANGED evidence (not sycophancy — the flip is grounded in the real code change):
- Re-ran the ruff gate BARE: `uvx ruff check --select F821,F401,F811 <4 files>` → **exit 0 / "All checks passed!"** (cycle-1 was exit 1).
- `git grep "import numpy as np"` in the test file → **no match** (import gone; no residual `np.` references).
- `pytest backend/tests/test_gate_correctness_69.py` → **18 passed**.
- C4 ACCEPT-ON-INTENT condition satisfied: `followons_69.2.md` exists (FO-69.2-A substantive, mtime post-cycle-1).
- All 5 immutable criteria re-affirmed (C1 Bailey 0.9004880 asserted behaviorally, not tautologically; C2/C3/C5 met;
  C4 accept-on-intent). Threshold constants byte-intact (`paper_go_live_gate.py:39-44`, `quant_optimizer.py:123`).
- mtime chain proves changed-evidence (followons 18:01 < critique/results 18:03, all post-cycle-1).

**Non-degrading NOTE**: the research→code→contract ordering slip was disclosed and accepted; must not recur as a pattern.
Reminder honored: harness_log append precedes the status flip.
