# Q/A Critique — phase-10.5 (Sortino with configurable MAR)

**qa_id:** qa_105_v1
**Cycle:** 1 (fresh Q/A, first verdict on this evidence)
**Date:** 2026-04-20
**Verdict:** PASS

---

## 5-item harness-compliance audit

| # | Check | Result |
|---|---|---|
| 1 | Researcher: >=5 full + recency + gate_passed | PASS -- brief shows 7 read-in-full, 17 URLs, three-variant queries, recency scan with two 2026-dated sources, gate_passed=true |
| 2 | Contract mtime <= experiment_results mtime | PASS -- contract 22:49:08 < results 22:50:43 |
| 3 | Verbatim masterplan criterion quoted | PASS -- experiment_results lists the 4 success_criteria by name; test command is verbatim `python -m pytest backend/metrics/tests/test_sortino.py -q` |
| 4 | No harness_log.md append yet | PASS -- caller confirms log-last discipline |
| 5 | Cycle numbering | v1, first Q/A on this evidence |

## Deterministic checks_run

- `ast_parse` -- 4 files (`__init__.py`, `sortino.py`, `tests/__init__.py`, `tests/test_sortino.py`) parse clean. exit 0.
- `verification_command` -- `python -m pytest backend/metrics/tests/test_sortino.py -q` -> `11 passed in 0.78s`. exit 0.
- `neighbor_suites` -- `pytest tests/autoresearch/ tests/slack_bot/ backend/metrics/ -q` -> `76 passed in 1.37s`. exit 0.
- `code_spot_read` (all invariants confirmed):
  - LPM_2 formula: `downside_excess = np.clip(mar_arr - arr, a_min=0.0, a_max=None)` (line 82)
  - Denominator over ALL T: `dd2 = float(np.mean(downside_excess ** 2))` (line 83)
  - Zero-downside sentinel: `return float("nan")` (line 87) -- NOT `+inf`, NOT `0.0`
  - 3-tier fetcher: BQ historical_macro (lines 102-121) -> analytics.get_risk_free_rate (lines 124-130) -> hardcoded 0.045 (line 133)
  - BQ SQL references `historical_macro` AND `('DGS3MO', 'DTB3')` (lines 107-113)
- `independent_math_verification` -- hand-computed Sortino for `[0.10, 0.04, -0.02, -0.05, 0.03], mar=0.0, ppy=252` is ~13.1823; shipped function returns 13.1830612... (abs diff 7.6e-4, within test tolerance 1e-3).
- `mutation_tests`:
  - **M1** (`np.clip(mar_arr - arr, 0, None)` -> `np.where(arr < mar_arr, arr - mar_arr, 0)`): NOT caught. Root cause is that the substitution is mathematically equivalent: when `arr < mar_arr`, `arr - mar_arr` is negative but is immediately squared in `downside_excess ** 2`, producing the same magnitudes as clipping `mar_arr - arr` positive. This is not a test-coverage weakness; the two formulations are numerically identical under squaring.
  - **M1b** (replacement real-inversion: clip `arr - mar_arr` positive, i.e., measure UPSIDE instead of DOWNSIDE): 3 tests failed, including `test_downside_deviation_only_below_mar` and `test_all_returns_above_mar_returns_nan`. Sign-inversion IS caught when it actually changes behavior.
  - **M2** (zero-downside sentinel `float('nan')` -> `0.0`): `test_all_returns_above_mar_returns_nan` and `test_downside_deviation_only_below_mar` both failed correctly. Caught.
  - **M3** (remove Tier-1 BQ branch in `_default_mar_fetcher`, jump to `get_risk_free_rate`): `test_default_mar_pulls_from_pyfinagent_data_macro` failed correctly (`assert bq_client_created == 1` -> `0 == 1`). Caught.
  - Sortino.py fully restored; 11/11 pass verified post-mutation.

checks_run = ["harness_compliance_5item", "ast", "verification_command", "neighbor_suites", "code_spot_read", "independent_math", "mutation_M1", "mutation_M1b", "mutation_M2", "mutation_M3"]

## LLM judgment

**Q1: Does shipped Sortino implement canonical LPM_2 per Sortino & Price (1994)?**
Yes. The formula matches the canonical form: DD = sqrt((1/T) * sum(min(0, R_t - MAR)^2)) with the denominator being T (ALL periods), annualized via `sqrt(periods_per_year)`. It is NOT a paraphrase of `backend/services/perf_metrics.compute_sortino`, which uses `std(ddof=1)` on the negative-only subset and is algebraically different (different denominator). Verified by independent hand-computation (13.1823 vs shipped 13.18306, abs diff 7.6e-4).

**Q2: Is the divergence from `compute_sortino` clearly documented?**
Yes. Module docstring (lines 9-11) and contract "Why" section both explicitly state that the old `compute_sortino` is preserved for back-compat with `paper_metrics_v2.py:111` and that the NEW module is canonical LPM_2. The experiment_results "Backend-services rule compliance" section documents the `.claude/rules/backend-services.md` reconciliation (Sortino is not listed in the "single metric source" rule).

**Q3: Are the carry-forwards legit deferrals?**
Yes. Three carry-forwards stated:
1. Add DGS3MO to `weekly_fred_refresh._DEFAULT_SERIES` -- legit; the 3-tier fallback (BQ -> analytics DTB3 cache -> 0.045) already handles the missing-series case in production today.
2. Deprecate `perf_metrics.compute_sortino` -- legit; would require migrating `paper_metrics_v2.py`, which is out of scope for a Sortino *module* ship.
3. Monthly cadence for 10.6 -- legit; `periods_per_year=12` is explicitly tested (`test_annualization_daily_vs_monthly`) and ready for 10.6 to consume.

**Q4: Research-gate compliance -- does the shipped 3-tier fallback handle the "historical_macro lacks DGS3MO today" reality the researcher flagged?**
Yes, confirmed empirically by M3. When Tier 1 (BQ) is removed, the test DOES fail because the test asserts BQ was called. When Tier 1 legitimately finds no rows (production reality), the code falls through to Tier 2 `get_risk_free_rate()` (DTB3 CSV cache), and on its failure to Tier 3 hardcoded 0.045. The `except Exception` guards on each tier make this fail-open. The M1-restore test run also confirmed 11/11 under the in-memory stub.

## Violated criteria

None.

```json
{
  "ok": true,
  "verdict": "PASS",
  "qa_id": "qa_105_v1",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_5item",
    "ast",
    "verification_command",
    "neighbor_suites",
    "code_spot_read",
    "independent_math",
    "mutation_M1",
    "mutation_M1b",
    "mutation_M2",
    "mutation_M3"
  ],
  "reason": "All 4 immutable success_criteria met (formula_matches_sortino_price_1994, downside_deviation_only_below_mar, default_mar_pulls_from_pyfinagent_data_macro, configurable_mar_per_candidate). 11/11 pytest pass on the immutable command, 76/76 on neighbor suites. Independent math verification matches to 7.6e-4 against hand-computed 13.1823. Mutation tests M1b, M2, M3 all caught by the suite (M1 was mathematically equivalent post-squaring, not a coverage gap). 5-item harness-compliance audit clean. Research-gate researcher delivered 7 in full + 17 URLs + 2026 recency + three-variant queries; the shipped 3-tier BQ/analytics/0.045 fallback handles the historical_macro-lacks-DGS3MO production reality the brief flagged. Canonical LPM_2 formula is NOT a paraphrase of the divergent perf_metrics.compute_sortino, and the divergence is documented in module docstring + contract + experiment_results."
}
```

## Notes for Main

- Mutation M1 (`np.clip(mar_arr - arr, 0, None)` swap to `np.where(arr < mar_arr, arr - mar_arr, 0)`) is mathematically equivalent after squaring and is therefore correctly NOT caught by the suite. No test is needed to distinguish these two formulations because they compute the same `downside_excess ** 2`. If you want a defense-in-depth test that pins the SIGN of `downside_excess` before squaring, consider a small invariant test asserting `downside_excess >= 0` elementwise -- but this is optional hardening, not a gap in the immutable criteria.
- Proceed with cycle-close: append `handoff/harness_log.md`, flip masterplan status=done, close task #65.
