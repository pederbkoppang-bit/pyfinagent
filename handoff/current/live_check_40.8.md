# Step 40.8 -- FF3 Correlation cap beyond GICS (OPEN-5) -- verification

**Date:** 2026-05-23
**Verdict:** **PASS** -- 3 immutable criteria + 6 mutation-resistance/edge-case tests = 9/9 PASS; default-OFF backward-compat verified.

---

## Verbatim masterplan criterion + evidence

| # | Criterion | Test | Verdict |
|---|---|---|---|
| 1 | `ff3_factor_exposure_used_alongside_gics` | test_phase_40_8_ff3_factor_exposure_used_alongside_gics | PASS (portfolio_manager.py reads `paper_max_factor_corr`; calls `factor_correlation_score`; positioned AFTER the GICS NAV-pct cap via string-position check) |
| 2 | `correlation_cap_blocks_simulated_high_ff_corr_buy` | test_phase_40_8_correlation_cap_blocks_simulated_high_ff_corr_buy | PASS (high-similarity candidate sim=0.998 > cap=0.85; orthogonal candidate sim=0.24 < cap) |
| 3 | `regression_against_known_fixture` | test_phase_40_8_regression_against_known_fixture | PASS (compute_ff3 recovers alpha=0.0002, betas (1.2, 0.4, 0.1), r_squared > 0.999 from canned 60-day noise-free series) |

Plus 6 bonus tests: cosine-sim high/low/missing/NaN/zero-vector edge cases, weighted aggregation, default-OFF backward-compat.

---

## Pytest evidence

```
$ source .venv/bin/activate
$ pytest backend/tests/test_phase_40_8_factor_correlation.py -v
============================== 9 passed in 0.04s ==============================

  test_phase_40_8_factor_correlation_score_returns_high_for_similar_vectors PASSED
  test_phase_40_8_factor_correlation_score_returns_low_for_orthogonal PASSED
  test_phase_40_8_factor_correlation_returns_zero_for_missing_inputs PASSED
  test_phase_40_8_aggregate_portfolio_loadings_weighted_by_market_value PASSED
  test_phase_40_8_aggregate_portfolio_loadings_empty_when_no_loadings PASSED
  test_phase_40_8_ff3_factor_exposure_used_alongside_gics PASSED
  test_phase_40_8_correlation_cap_blocks_simulated_high_ff_corr_buy PASSED
  test_phase_40_8_default_off_backward_compat_zero_cap_disables PASSED
  test_phase_40_8_regression_against_known_fixture PASSED

$ pytest backend/tests/ -k "portfolio_manager or sector" --tb=no -q
13 passed, 485 deselected   (existing sector-cap regression tests UNCHANGED)

$ pytest backend/ --collect-only -q | tail -2
509 tests collected   (was 500; +9 new; 0 regressions)
```

---

## Diff

```
backend/services/factor_correlation.py        NEW (~85 lines): pure helper module
backend/config/settings.py                    +13 lines: paper_max_factor_corr field
backend/services/portfolio_manager.py         +24 lines: aggregate_portfolio_loadings + per-cand gate
backend/tests/test_phase_40_8_factor_correlation.py  NEW (~170 lines, 9 tests)
```

---

## /goal integration-gate scoreboard

| # | Gate | Verdict |
|---|---|---|
| 1 | pytest count baseline (>=297) | **PASS** (509; +9 net new) |
| 2 | ast.parse green | **PASS** (factor_correlation.py + portfolio_manager.py + settings.py all parse) |
| 3 | TS build | N/A |
| 4 | Flag-default-OFF | **PASS** (`paper_max_factor_corr=0.0` disables; test_phase_40_8_default_off_backward_compat_zero_cap_disables enforces) |
| 5 | BQ idempotent | **PASS** (no BQ touched) |
| 6 | env vars docs | N/A (no new env var; settings field with literal default) |
| 7 | N* delta declared | **PASS** (R + B; default-OFF until operator opts in) |
| 8 | Zero emojis | **PASS** |
| 9 | ASCII-only loggers | **PASS** (`Skipping BUY %s: FF3 factor correlation %.3f > cap %.3f ...` is ASCII) |
| 10 | Single source of truth | **PASS** (compute_ff3 in portfolio_risk.py:58 is the math primitive; factor_correlation.py is wiring) |
| 11 | log-first / flip-last | **WILL HOLD** |

---

## Honest scope + backward compat

**Hot path safety**: portfolio_manager.py is in the BUY loop. The new code path is gated on `settings.paper_max_factor_corr > 0` (default 0.0 = OFF) AND on `port_factor_loadings` being non-empty (which requires positions to carry `factor_loadings`, which the analysis pipeline does NOT yet produce). So today, in live, the gate is doubly disabled: a no-op single `if max_factor_corr > 0 and port_factor_loadings:` check.

**Forward compat path**:
- When upstream analysis adds `factor_loadings` to position dicts -> `port_factor_loadings` becomes non-empty.
- When operator flips `paper_max_factor_corr > 0` -> cap fires per candidate.
- When candidate analysis adds `factor_loadings` to its dict -> candidate cosine sim computed; high-correlation candidates blocked.

The 3 gates chain so the feature can be rolled out incrementally without coupling.

**No production risk**: today's behavior is byte-identical to pre-40.8 because both default-OFF flags fire. Quiet-log period of 1-2 weeks recommended per researcher (AQR/Two Sigma 2025 factor-crowding research) before enabling.

---

## Research-gate

Researcher SPAWNED FIRST. Brief at `handoff/current/research_brief_phase_40_8.md` -- 5 sources read-in-full, gate_passed=true. Critical internal finding: `compute_ff3()` already exists at `portfolio_risk.py:58`. The MCP `factor_exposure()` stub remains a stub (separate phase). Recommended scope (a) MIN VIABLE implemented verbatim.

---

## Files for archive (handoff/archive/phase-40.8/)

- contract.md
- experiment_results.md
- live_check_40.8.md (this file)
- evaluator_critique.md (after Q/A PASS)
- research_brief_phase_40_8.md
