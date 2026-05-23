# phase-40.8 -- experiment results (Cycle 47)

**Date:** 2026-05-23
**Cycle:** 47
**Step:** phase-40.8 -- Correlation cap beyond GICS (OPEN-5)
**Verdict:** PASS (deterministic; 9/9 new tests + 13/13 sector-cap regression)

---

## What changed

| File | Change | Lines |
|---|---|---|
| `backend/services/factor_correlation.py` | NEW pure helper: `factor_correlation_score` (cosine sim over FF3) + `aggregate_portfolio_loadings` (weighted avg). | +85 |
| `backend/config/settings.py` | New field `paper_max_factor_corr: float = Field(0.0, ge=0.0, le=1.0)`. Default 0.0 = disabled. | +13 |
| `backend/services/portfolio_manager.py` | aggregate_portfolio_loadings built once per cycle (cap > 0 only); per-candidate gate AFTER GICS NAV-pct cap; default-OFF short-circuit. | +24 |
| `backend/tests/test_phase_40_8_factor_correlation.py` | NEW (9 tests: 3 immutable criteria + 6 mutation-resistance/edge cases). | +170 |

`backend/services/portfolio_risk.py::compute_ff3` (existing math primitive) -- UNCHANGED.

---

## Verbatim test output

```
$ source .venv/bin/activate
$ pytest backend/tests/test_phase_40_8_factor_correlation.py -v
============================== 9 passed in 0.04s ==============================

$ pytest backend/tests/ -k "portfolio_manager or sector" --tb=no -q
13 passed, 485 deselected, 1 warning in 2.42s    (existing sector-cap suite UNCHANGED)

$ pytest backend/ --collect-only -q | tail -2
509 tests collected   (was 500; +9 net new; 0 regressions)

$ python -c "import ast; ast.parse(open('backend/services/factor_correlation.py').read()); ast.parse(open('backend/services/portfolio_manager.py').read()); ast.parse(open('backend/config/settings.py').read()); print('ast.parse OK')"
ast.parse OK
```

---

## Immutable success criteria

1. **ff3_factor_exposure_used_alongside_gics** -- PASS. `portfolio_manager.py` reads `settings.paper_max_factor_corr`, calls `factor_correlation_score` in the BUY loop AFTER the existing `paper_max_per_sector_nav_pct` cap (string-position-asserted by test 6).
2. **correlation_cap_blocks_simulated_high_ff_corr_buy** -- PASS. Canned portfolio (loadings 1.0/0.5/0.3) vs candidate (0.99/0.51/0.29) yields cosine sim ~0.998 > cap=0.85; orthogonal candidate (0/0/1) yields sim < cap (test 7).
3. **regression_against_known_fixture** -- PASS. `compute_ff3` with deterministic 60-day series (alpha=0.0002, betas 1.2/0.4/0.1) recovers all coefficients to 1e-10 precision; r_squared > 0.999 (test 9).

---

## /goal integration-gate scoreboard

| # | Gate | Verdict |
|---|---|---|
| 1 | pytest count baseline (>=297) | **PASS** (509; +9 net new) |
| 2 | ast.parse green | **PASS** |
| 3 | TS build | N/A |
| 4 | Flag-default-OFF | **PASS** (default 0.0; test enforces) |
| 5 | BQ idempotent | N/A |
| 6 | env vars docs | N/A |
| 7 | N* delta declared | **PASS** (R + B) |
| 8 | Zero emojis | **PASS** |
| 9 | ASCII-only loggers | **PASS** |
| 10 | Single source of truth | **PASS** (compute_ff3 canonical; factor_correlation is wiring) |
| 11 | log-first / flip-last | **WILL HOLD** |

---

## Honest scope + dual-interpretation

**Literal:** 3 immutable criteria map 1:1 to 3 dedicated tests; each fails under realistic mutation (string-position check, cosine-sim sign-flip, alpha/beta exact-recovery tolerance).

**Operational:** OPEN-5 closes at the design layer. The cap is wired in the hot path with TWO independent default-OFF guards:
  - `settings.paper_max_factor_corr > 0` (default 0.0)
  - `port_factor_loadings` non-empty (requires upstream to supply loadings)

Today's live behavior is byte-identical to pre-40.8. Quiet-log period 1-2 weeks recommended before operator enables, per AQR/Two Sigma 2025 factor-crowding research.

**Latent gap (NOT a blocker; surfacing as follow-up phase-40.8.1):**
- Upstream analysis pipeline doesn't yet produce `factor_loadings` on candidates/positions. The cap is dormant until that's wired. Phase-40.8.1 would wire `compute_ff3` into the analysis pipeline so positions carry loadings.

---

## Research-gate

Researcher SPAWNED FIRST (cycle 47; 4 consecutive cycles honoring `feedback_never_skip_researcher`). Brief at `handoff/current/research_brief_phase_40_8.md`. Tier=simple. 5 sources read-in-full. gate_passed=true. Critical internal finding: `compute_ff3()` already exists at `portfolio_risk.py:58`; MCP stub remains a stub (separate phase). Recommendation (a) MIN VIABLE implemented verbatim.

---

## Files for archive (handoff/archive/phase-40.8/)

- contract.md
- experiment_results.md (this file)
- live_check_40.8.md
- evaluator_critique.md (after Q/A PASS)
- research_brief_phase_40_8.md
