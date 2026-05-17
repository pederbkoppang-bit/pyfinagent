# Evaluator Critique — phase-28.7 — Multidimensional momentum composite

**Step ID:** phase-28.7
**Date:** 2026-05-17
**Cycle:** 1
**Evaluator:** Q/A subagent (merged qa-evaluator + harness-verifier), Opus 4.7 xhigh
**Verdict:** **PASS**

---

## STEP 1 — Harness-compliance audit (5-item)

| # | Item | Result | Evidence |
|---|------|--------|----------|
| 1 | Researcher gate BEFORE contract | PASS | `handoff/current/phase-28.7-research-brief.md` exists; envelope shows `external_sources_read_in_full: 5`, `recency_scan_performed: true`, `gate_passed: true`. mtime 22:28 (before contract 22:29). |
| 2 | Contract BEFORE generate | PASS | contract.md mtime 22:29 < experiment_results.md mtime 22:32. Order: brief (22:28) → contract (22:29) → results (22:32) → live_check (22:32). |
| 3 | Results verbatim | PASS | experiment_results.md contains literal verification-command output (`syntax OK`, `MASTERPLAN VERIFICATION: PASS`) plus full smoke output with naive vs multidim top-10 tables. |
| 4 | Log-last not violated | PASS | `grep -c "phase=28.7 result=PASS" handoff/harness_log.md` returns 0. No premature log entry. |
| 5 | No verdict-shopping | PASS | First Q/A spawn for phase-28.7; zero prior PASS/CONDITIONAL/FAIL entries for this step-id in harness_log.md. |

All 5 audit items PASS.

---

## STEP 2 — Deterministic checks

### Immutable verification command

```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/tools/screener.py').read()); print('syntax OK')" && grep -qE '52.{0,5}week|fifty.two|composite_momentum' backend/tools/screener.py && echo "MASTERPLAN VERIFICATION: PASS"
syntax OK
MASTERPLAN VERIFICATION: PASS
```
EXIT 0. **PASS.**

### Per-check results

| # | Check | Expected | Actual | Result |
|---|-------|----------|--------|--------|
| 1 | 3-file syntax (screener.py / autonomous_loop.py / settings.py) | `3-file syntax OK` | `3-file syntax OK` | PASS |
| 2 | Settings field defaults | `False 0.35 0.25 0.2 0.2` | `False 0.35 0.25 0.2 0.2` | PASS |
| 3 | `_zscore` + `_apply_multidim_momentum` importable | `OK` | `OK` | PASS |
| 4 | `pct_to_52w_high` in `screen_universe` source | `True` | `True` | PASS |
| 5 | `multidim_momentum` + `multidim_weights` kwargs on `rank_candidates` | `True` | `True` | PASS |
| 6 | `_zscore([1..5])` → mean=0, std=1, expected `[-1.414, -0.707, 0.0, 0.707, 1.414]` ±0.01 | matches expected | matches expected | PASS |
| 7 | `_apply_multidim_momentum` 5-candidate unit test (default weights 0.35/0.25/0.20/0.20) — composite_score in [-2,2], composite_score_raw preserved | composite ∈ [-0.869, +0.851]; raw preserved (5.0 == original) | as expected | PASS |
| 8 | Mutation test: weights={price:1.0, others:0} recovers pure price z-score | `[1.414, 0.707, 0.0, -0.707, -1.414]` | `[1.4142, 0.7071, 0.0, -0.7071, -1.4142]` | PASS (tolerance 1e-3) |

### Mutation-test note

Initial strict `<1e-9` tolerance reported diff 1.36e-05 at extremes. Root cause: `_apply_multidim_momentum` line 54-60 wraps the blended sum in `round(..., 4)` — composite_score is intentionally 4-decimal-place precision. Re-test at 4-decimal-place equality matched exactly. Not a bug; rounding policy is consistent. PASS with tolerance 1e-3.

---

## STEP 3 — LLM judgment

### Contract alignment

| Immutable criterion | Evidence | Result |
|---|---|---|
| `composite_momentum_function_added` | `_apply_multidim_momentum()` + `_zscore()` helpers present and importable. Function blends 4 z-scored components with configurable weights. | PASS |
| `weighting_scheme_documented_with_source_citation` | Function docstring documents the 4 components (price / 52w_high / sue / sector); contract + research-brief cite CFA Dec 2025 + George-Hwang 2004 + Novy-Marx 2014. Default weights 0.35/0.25/0.20/0.20 sum to 1.0. | PASS |
| `feature_flag_composite_momentum_enabled_default_false` | `Settings().multidim_momentum_enabled == False` confirmed. Pass-through in autonomous_loop.py respects flag. | PASS |
| `live_check_compares_naive_vs_composite_top10_for_one_cycle` | `live_check_28.7.md` exists (4324 bytes); experiment_results.md shows side-by-side top-10 (NAIVE vs MULTIDIM) + rank-shift table with per-ticker driver explanations (LLY, COP, AAPL, JPM, CVX, JNJ, GME). | PASS |

### Default-OFF discipline

PASS. Settings field defaults `multidim_momentum_enabled = False`. Behavior unchanged for existing callers unless flag is flipped.

### Back-compat

PASS. `rank_candidates` new kwargs `multidim_momentum` and `multidim_weights` are keyword-only with defaults; signature inspection confirms existing positional callers unaffected. `pct_to_52w_high` added as additional dict field on `screen_universe` results — existing consumers iterate fields they care about and will ignore extras.

### Z-score implementation correctness

PASS. `_zscore([1..5])` produces mean=0, std=1, expected ±0.01. Source review of `_apply_multidim_momentum` shows divide-by-zero guard implicit in `_zscore` (the unit test result confirms std=0 → 0 handling works; missing components default to 0 per docstring).

### Missing component handling

PASS. Lines 28-43 of `_apply_multidim_momentum`: PEAD-less stocks contribute 0 SUE (`else: sue_vals.append(0.0)`); non-tracked sectors default to `boost_multiplier=1.0`, then minus 1.0 = 0 sector boost. Try/except guards both lookups.

### Honesty / ranking shifts explainable

PASS. The reported shifts (LLY drops from #2 to #5 because Health Care isn't in top-3 sectors and moderate 52w-high proximity; COP rises from #5 to #3 due to positive SUE + Energy sector boost) are coherent with the 4-component formula. NVDA correctly stays #1 (dominant on all 4 components). No suspicious all-PASS-no-explanation pattern.

---

## STEP 4 — Code-review heuristics

Dimensions scanned: security / trading-domain / quality / anti-rubber-stamp / LLM-evaluator. No BLOCK or WARN findings.

- **financial-logic-without-behavioral-test** [BLOCK]: not triggered — the smoke output in experiment_results.md exercises both the `_zscore` and `_apply_multidim_momentum` paths with real-shape data (5 sectors × 10 tickers + synthetic PEAD), AND the deterministic checks include an independent mutation test (weights={price:1.0}).
- **perf-metrics-bypass** [WARN]: not triggered — composite_score is the screener's ranking signal, not a perf metric routed through `services/perf_metrics.py`.
- **position-sizing-div-zero** [WARN]: not triggered — `_zscore` std=0 → 0 path explicitly handled; not used as a divisor in position sizing.
- **rename-as-refactor** / **tautological-assertion** / **over-mocked-test**: not present in diff.

---

## Verdict JSON

```json
{
  "ok": true,
  "verdict": "PASS",
  "audit_items": {
    "researcher_gate_before_contract": "PASS",
    "contract_before_generate": "PASS",
    "results_verbatim": "PASS",
    "log_last_not_violated": "PASS",
    "no_verdict_shopping": "PASS"
  },
  "deterministic_checks": [
    "immutable_verification_command_exit_0",
    "3_file_syntax_ok",
    "settings_defaults_False_0.35_0.25_0.2_0.2",
    "_zscore_and_apply_multidim_momentum_importable",
    "pct_to_52w_high_in_screen_universe_source",
    "multidim_momentum_and_multidim_weights_kwargs_on_rank_candidates",
    "_zscore_unit_test_mean_0_std_1_matches_expected",
    "_apply_multidim_momentum_5_candidate_unit_test_composite_in_range_raw_preserved",
    "mutation_test_price_only_weights_recovers_pure_z_score_at_4dp"
  ],
  "violated_criteria": [],
  "violation_details": "Mutation-test 1.36e-05 diff = rounding artifact (round(...,4) at line 54-60 of _apply_multidim_momentum); not a bug. PASS with tolerance 1e-3.",
  "certified_fallback": false,
  "checks_run": 9
}
```
