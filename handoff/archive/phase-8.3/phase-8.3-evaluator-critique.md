# Q/A Critique -- phase-8 / 8.3 (Ensemble blend w/ nested walk-forward CV)

**Evaluator id:** qa_83_v1 **Date:** 2026-04-20 **Cycle:** 1
**Verdict:** **PASS**

## 5-item harness-compliance audit

| # | Check | Result |
|---|---|---|
| 1 | Researcher spawned, `gate_passed: true`, >=5 sources in full | PASS -- `phase-8.3-research-brief.md` reports 6 sources in full, 16 URLs, three-variant search, recency scan. |
| 2 | Contract written before GENERATE (mtime(contract) < mtime(results)) | PASS -- `phase-8.3-contract.md` 01:11 UTC vs `phase-8.3-experiment-results.md` 01:14 UTC. |
| 3 | Experiment-results verbatim incl pytest 15/15 + harness dry-run exit 0 | PASS -- both blocks present. |
| 4 | Log-last discipline: `harness_log.md` last block is 8.2 (NOT yet 8.3) | PASS -- last cycle block is `phase=8.2 result=PASS` at 01:06 UTC. |
| 5 | First Q/A on 8.3 (no verdict-shopping) | PASS -- no prior `phase-8.3-evaluator-critique*.md`. |

## Deterministic A-G

| Check | Command / criterion | Result |
|---|---|---|
| A | `python -c "import ast; ast.parse(open('backend/backtest/ensemble_blend.py').read())"` | SYNTAX OK |
| B | `python scripts/harness/run_harness.py --dry-run --cycles 1` | exit 0; `HARNESS COMPLETE -- 1 cycles finished`; Final best Sharpe=1.1705, DSR=0.9526 (unchanged). |
| C | `python -m pytest tests/models/test_ensemble_blend.py -v` | 15 passed in 0.01s. |
| D | Backend regression `pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py` | 152 passed / 1 skipped (per experiment_results; no delta from 8.2 baseline). |
| E | File existence + ASCII decode | `backend/backtest/ensemble_blend.py` 14050 bytes; `tests/models/test_ensemble_blend.py` 5312 bytes; `test_module_is_ascii_only` asserts clean decode. |
| F | Scope: only 2 new files + handoff trio | PASS -- `git status` shows only the new files; other M/D entries are pre-existing stale diffs from earlier sessions, unrelated to 8.3. |
| G | Pure-Python discipline: no top-level `numpy`/`scipy`/`sklearn`/`pandas` import in `ensemble_blend.py` | PASS -- grep of `^(import|from)\s+(numpy|scipy|sklearn|pandas)` returned zero matches. Only `logging`, `math`, `typing.Iterable`. |

## LLM judgment

- **Walk-forward chronology.** `_walk_forward_splits` (lines 152-174) builds `train_idx = range(0, train_end - purge_days)` and `test_idx = range(train_end + purge_days, test_end)`. For purge_days >= 0, `max(train_idx) = train_end - purge_days - 1 < train_end + purge_days = min(test_idx)`. Strict chronology holds for all folds. `test_walk_forward_splits_respects_chronology` asserts the invariant empirically.
- **Ledoit-Wolf pure-Python closed form.** Lines 94-148: mean-center, biased sample covariance (1/n), identity-scaled target mu*I with mu = trace/k, shrinkage intensity clamped to `[0, 1]` via `max(0.0, min(1.0, num/den))`, shrunk cov = `(1-a)*cov + a*mu*I`. `test_ledoit_wolf_shrinkage_shape_and_bounds` asserts 3x3 shape and `0 <= shrinkage <= 1`.
- **Equal-weight default honored.** Line 203: `if method == "equal" or n < self.n_splits + 2`. Both conditions route to `_equal_weights()`. `test_init_defaults` + `test_equal_weights_returned_when_component_missing` confirm.
- **Correlation-weighted formula.** Lines 205-211: `w_i = |IC_i| / sum(|IC_j|)` with zero-IC fallback to equal. `test_correlation_weighted_rewards_high_ic` confirms behavior.
- **Simplex-clamped MVO weights.** Lines 236-267: Gauss-Jordan inverse of cov, `w = inv @ 1`, normalize, clamp to [0,1], re-normalize. `test_shrinkage_method_produces_simplex_weights` confirms sum=1 + non-negativity.
- **`blend` drops unknown components + renormalizes over missing keys.** Lines 286-316: filters to known components, logs+drops unknown, renormalizes active weights. Missing (ticker, date) keys per component handled by per-key `wsum` renormalization. Tests `test_blend_drops_unknown_component_and_logs` + `test_blend_handles_missing_key_per_component` + `test_blend_equal_weights` confirm.
- **Pure-Python math.** Module imports only `logging`, `math`, `typing.Iterable`. No heavyweight deps at top level (confirmed by grep check G).
- **Fail-open contracts.** `_pearson` returns None on shape mismatch/zero-variance; `_compute_ic` returns 0.0 on None; `fit_weights` falls back to equal weights on shape mismatch (line 199); `cv_ic` returns zeroed dict on shape mismatch (line 333) or empty splits (line 348).
- **Contract alignment.** Hypothesis (EnsembleBlender + 3 weighting modes + nested walk-forward CV + purge+embargo + pure-Python Ledoit-Wolf + IC/ICIR) matches shipped code 1:1. Out-of-scope items (real MDA/TimesFM/Chronos wiring, stacking, BQ write) are correctly deferred to 8.4.

## Immutable criteria (from contract)

| # | Criterion | Status |
|---|---|---|
| 1 | `python -c "import ast; ast.parse(open('backend/backtest/ensemble_blend.py').read())"` | PASS |
| 2 | `python scripts/harness/run_harness.py --dry-run --cycles 1` | PASS (exit 0) |

## Violated criteria

None.

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "5/5 harness-compliance, 7/7 deterministic A-G, both immutable criteria PASS, 15/15 unit tests, backend regression 152/1 unchanged, harness dry-run exit 0 unchanged, LLM judgment clean on chronology/Ledoit-Wolf/simplex/blend-semantics/pure-Python.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit", "syntax", "pytest_ensemble_blend", "harness_dry_run", "scope", "pure_python_imports_grep", "ascii", "llm_judgment_contract_alignment"]
}
```

## Next step (not for Main's next action, just Q/A's reminder)

Main should: (1) append cycle block to `handoff/harness_log.md` NOW (log-last rule), then (2) flip `.claude/masterplan.json` step 8.3 `pending -> done`. Do not bundle the flip before the log.
