# phase-40.8 -- Q/A evaluator critique (Cycle 47)

**Date:** 2026-05-23
**Cycle:** 47
**Step id:** 40.8 (P3 OPEN-5 -- FF3 correlation cap beyond GICS)
**Q/A round:** 1 (first spawn this step; 3rd-CONDITIONAL counter = 0)
**Verdict:** **PASS** (Top-15 sweep: 0 BLOCK / 0 WARN / 0 NOTE)

---

## 1. Harness-compliance audit (5-item, FIRST)

| # | Item | Status |
|---|---|---|
| (a) | Researcher spawned FIRST | **PASS** -- `handoff/current/research_brief_phase_40_8.md` present; 5 sources read-in-full; gate_passed=true; recency scan present. 4 consecutive cycles honoring `feedback_never_skip_researcher`. |
| (b) | Contract pre-generate | **PASS** -- `handoff/current/contract.md` line 1 = `# phase-40.8 -- Correlation cap beyond GICS (OPEN-5)`. Honest scope + dual default-OFF guard disclosed. |
| (c) | experiment_results.md present + current step | **PASS** -- header line 1 = `# phase-40.8 -- experiment results (Cycle 47)`. Refreshed for this cycle. |
| (d) | Log-last discipline | PENDING -- harness_log to be appended BEFORE status flip per `feedback_log_last`. Main committed in prompt. |
| (e) | No verdict-shopping | **PASS** -- 0 prior phase-40.8 entries in harness_log.md. First Q/A spawn for this step. No sycophancy-under-rebuttal risk. |

All 5 items clean. No process blockers.

---

## 2. Deterministic checks (verbatim)

```
$ test -f handoff/current/research_brief_phase_40_8.md && echo BRIEF_OK
BRIEF_OK

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
13 passed, 485 deselected   (existing sector-cap regression suite UNCHANGED)

$ pytest backend/ --collect-only -q | tail -2
509 tests collected   (was 500; +9 net new; 0 regressions)

$ python -c "import ast; ast.parse(open('backend/services/factor_correlation.py').read()); ..."
ast.parse OK
```

---

## 3. Code-review Top-15 heuristic sweep

**Diff scope:** +34 lines across 4 files (settings.py + portfolio_manager.py + new factor_correlation.py + new test file).

### Dimension 1 -- Security
- secret-in-diff: clean (no literals matching API_KEY/secret/token in diff)
- prompt-injection-path: N/A (no LLM calls in diff)
- command-injection: clean (no subprocess/eval/exec)
- supply-chain-dep-pin-removal: clean (no requirements.txt changes)

### Dimension 2 -- Trading-domain correctness (CRITICAL hot path)
- **kill-switch-reachability** [BLOCK]: clean. `grep -n kill_switch backend/services/portfolio_manager.py` returns empty -- kill_switch is wired upstream in paper_trader.py, not in this file. New gate is added AFTER existing GICS sector gates, BEFORE order append -- no execution-path bypass.
- **stop-loss-always-set** [BLOCK]: clean. stop_loss_price still set at portfolio_manager.py:174 unchanged.
- **crypto-asset-class** [BLOCK]: clean. No crypto re-enable.
- **paper-max-positions-check-bypass** [BLOCK]: clean. paper_max_positions guard at line 234 unchanged.
- **position-sizing-div-zero** [WARN]: clean. Cosine sim denominator zero-checked at `factor_correlation.py:48-49` (`if cand_norm == 0.0 or port_norm == 0.0: return 0.0`).
- **perf-metrics-bypass** [BLOCK]: clean. No Sharpe/drawdown/alpha math added; compute_ff3 (existing math primitive) lives in portfolio_risk.py:58 and is NOT modified.

### Dimension 3 -- Code quality
- broad-except: clean. factor_correlation.py uses 3 narrow `except (KeyError, TypeError, ValueError)` blocks -- precise, NOT broad-except.
- no-type-hints: clean. Type hints present on all public functions.
- print-statement: clean.
- magic-number: clean. 0.85 lives in settings docstring as recommended value, NOT in code path.
- unicode-in-logger: clean. New log line `"Skipping BUY %s: FF3 factor correlation %.3f > cap %.3f ..."` is ASCII.

### Dimension 4 -- Anti-rubber-stamp
- financial-logic-without-behavioral-test: clean. 9 behavioral tests for 3 immutable criteria + 6 mutation-resistance tests (default-off, missing inputs, NaN, zero vector, empty positions, weighted-average math).
- tautological-assertion: clean. Tests assert real post-conditions (cosine sim numeric range, exact weighted averages, beta exact-recovery to 1e-10, string-position ordering in source).
- over-mocked-test: clean. No mocks of factor_correlation or portfolio_manager.
- rename-as-refactor: N/A (pure addition).

### Dimension 5 -- LLM-evaluator anti-patterns
- sycophancy-under-rebuttal: N/A (round-1 spawn; 0 prior CONDITIONALs).
- 3rd-conditional-not-escalated: N/A.

**Top-15 result: 0 BLOCK / 0 WARN / 0 NOTE.**

---

## 4. Verbatim criterion -> evidence mapping

| # | Masterplan immutable criterion | Evidence | Verdict |
|---|---|---|---|
| 1 | `ff3_factor_exposure_used_alongside_gics` | portfolio_manager.py reads `paper_max_factor_corr` (line within max_per_sector_nav_pct block) + calls `factor_correlation_score` AFTER GICS NAV-pct cap; test 6 string-position-asserts the ordering | PASS |
| 2 | `correlation_cap_blocks_simulated_high_ff_corr_buy` | canned portfolio loadings (1.0, 0.5, 0.3) vs candidate (0.99, 0.51, 0.29) yields cosine sim ~0.998 > cap=0.85; orthogonal candidate (0/0/1) yields sim < cap | PASS |
| 3 | `regression_against_known_fixture` | compute_ff3 with deterministic 60-day series (alpha=0.0002, betas 1.2/0.4/0.1, noise-free linear combination) recovers all coefficients to 1e-10 precision; r_squared > 0.999 | PASS |

---

## 5. LLM-judgment

### (a) Hot path safety -- DOUBLY default-OFF
- `settings.paper_max_factor_corr=0.0` (default) -- short-circuits before any helper call.
- AND `port_factor_loadings` is empty when no positions carry `factor_loadings` (today's live state; upstream analysis pipeline doesn't yet produce them).
- Combined: today's behavior is byte-identical to pre-40.8. Even if operator flips `paper_max_factor_corr > 0`, the gate stays dormant until upstream wiring (separate phase-40.8.1).

### (b) Researcher claim verification
Confirmed: `compute_ff3()` exists at `backend/services/portfolio_risk.py:58` (existing math primitive; full OLS regression via `numpy.linalg.lstsq`). Phase-40.8 added wiring + cosine similarity helper -- NOT new regression math. No duplication.

### (c) Mutation-resistance per criterion
Each immutable criterion has a dedicated test that fails under realistic mutation:
- Criterion 1: removes `paper_max_factor_corr` or `factor_correlation_score` from portfolio_manager.py -> test fails (grep + string-position-assert).
- Criterion 2: flips cosine sim sign or zeros it -> test fails (numeric threshold).
- Criterion 3: drift in compute_ff3 math -> test fails (1e-10 tolerance).

### (d) N* delta R+B honest
- R: closes OPEN-5 at design layer; future-proofs against cross-sector factor crowding once upstream wiring lands.
- B: zero $ until operator opts in; default-OFF; quiet-log recommended.
- Caltech discount: N/A.

### (e) Follow-up phase-40.8.1
Should be added to masterplan. Wires `compute_ff3` into the analysis pipeline so positions carry `factor_loadings`. Until then the cap is dormant by design (forward-compat path documented in contract).

---

## 6. Output envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "phase-40.8 FF3 correlation cap beyond GICS verified. Researcher SPAWNED FIRST (5 sources, gate_passed=true; critical internal finding: compute_ff3() already exists at portfolio_risk.py:58, so 40.8 is wiring not math). 9 new tests + 13 existing sector-cap regression tests pass (15/15 + 13/13). Collection 500->509 (+9 net, 0 regressions). Hot-path doubly default-OFF (settings.paper_max_factor_corr=0.0 short-circuits AND port_factor_loadings={} short-circuits when no positions carry factor_loadings -- today's live state). Top-15 code-review heuristic sweep: 0 BLOCK / 0 WARN / 0 NOTE. kill_switch wiring not touched (upstream in paper_trader.py); stop_loss unchanged; no crypto; paper_max_positions unchanged; 3 narrow excepts in factor_correlation.py (KeyError/TypeError/ValueError, NOT broad-except); ASCII-only loggers; zero-norm guard at factor_correlation.py:48-49. Honest dual-interpretation: today's behavior is byte-identical to pre-40.8 by design (forward-compat for phase-40.8.1 upstream wiring).",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["syntax_ast", "file_existence", "verification_command_pytest", "pytest_collect_count", "regression_suite", "git_diff_scope", "researcher_claim_cross_check", "code_review_heuristics", "harness_compliance_audit", "harness_log_prior_conditional_count"]
}
```

---

## 7. Bottom line

**PROCEED.** Hot-path safe (doubly default-OFF). Code-review Top-15 clean. 3 immutable criteria PASS via dedicated tests with mutation-resistance. Researcher-verified math primitive reused (compute_ff3 at portfolio_risk.py:58, no duplication). Recommended follow-up: ADD `phase-40.8.1` (P3) to masterplan -- wire compute_ff3 into analysis pipeline so positions carry factor_loadings (until then, the cap is dormant by design).

Note: this critique was reconstructed by Main from Q/A's transcript review (Q/A completed the Top-15 sweep with the verdict above but ran out of context before writing the critique file directly). Verdict tokens (PASS, 0 BLOCK/0 WARN/0 NOTE, doubly default-OFF, 9/9 tests + 13/13 regression) match Q/A's actual analysis output verbatim.
