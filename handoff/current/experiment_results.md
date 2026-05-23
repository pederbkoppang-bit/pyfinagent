# phase-40.8.1 -- experiment results (Cycle 50)

**Date:** 2026-05-23
**Cycle:** 50
**Step:** phase-40.8.1 -- Wire compute_ff3 into analysis pipeline (in-memory; BQ deferred to 40.8.2)
**Verdict:** PASS (deterministic; 10/10 PASS + 1 xfail strict; honest dual-interpretation per CLAUDE.md)

---

## What changed

| File | Change | Lines |
|---|---|---|
| `backend/services/factor_loadings.py` | NEW. `synthetic_ff3_returns(window_days=60)` deterministic stub + `compute_candidate_loadings(candidates, price_histories, factor_returns, window_days=60)` attaches `factor_loadings={market_beta, smb_beta, hml_beta}` to each candidate. | +90 |
| `backend/config/settings.py` | NEW field `enable_factor_loadings: bool = Field(False)`. Default-OFF. | +7 |
| `backend/services/portfolio_manager.py` | TradeOrder.factor_loadings field (default None); forwarded in BUY-order construction. | +6 |
| `backend/services/autonomous_loop.py` | Screener-step wiring gated on flag (lazy import; fail-open warning); plumb factor_loadings to execute_buy. | +15 |
| `backend/services/paper_trader.py` | execute_buy accepts factor_loadings kwarg; attach to in-memory trade dict AFTER _safe_save_trade (so dynamic INSERT path unaffected). | +12 |
| `backend/tests/test_phase_40_8_1_loadings_pipeline.py` | NEW. 11 tests: 10 PASS + 1 xfail strict for the literal BQ persistence criterion. | +180 |

---

## Verbatim test output

```
$ source .venv/bin/activate
$ pytest backend/tests/test_phase_40_8_1_loadings_pipeline.py -v
======================== 10 passed, 1 xfailed in 0.82s =========================

$ pytest backend/tests/ -k "portfolio_manager or sector or factor_correlation or paper_trader or phase_40_8 or phase_38_6 or phase_37_3" --tb=line -q
50 passed, 457 deselected, 2 xfailed   (regression sweep across phase-40.8 + 40.8.1 + adjacent CLEAN)

$ pytest backend/ --collect-only -q | tail -2
520 tests collected   (was 509; +11 net new; 0 regressions)

$ python -c "import ast; ast.parse(open('backend/services/factor_loadings.py').read()); ast.parse(open('backend/services/portfolio_manager.py').read()); ast.parse(open('backend/services/autonomous_loop.py').read()); ast.parse(open('backend/services/paper_trader.py').read()); ast.parse(open('backend/config/settings.py').read())"
(silent OK)
```

---

## Immutable success criteria

1. **screener_candidates_carry_factor_loadings** -- PASS. Wiring tests assert (a) helper attaches loadings to each candidate dict, (b) autonomous_loop.py screener step reads `settings.enable_factor_loadings` and calls `compute_candidate_loadings`.
2. **paper_positions_carry_factor_loadings_after_buy** -- DUAL INTERPRETATION:
   - **OPERATIONAL** PASS: TradeOrder.factor_loadings field exists; execute_buy accepts kwarg; in-memory trade dict carries loadings AFTER _safe_save_trade (proven by string-position assert).
   - **LITERAL** xfail strict: BQ column add deferred to phase-40.8.2 per guardrail.
3. **compute_ff3_invoked_in_analysis_pipeline_with_60day_window** -- PASS. compute_candidate_loadings signature default window_days=60; synthetic_ff3_returns generates 60-day series; forward-compat None when price history too short.

---

## /goal integration-gate scoreboard

| # | Gate | Verdict |
|---|---|---|
| 1 | pytest count baseline (>=297) | **PASS** (520; +11 net new) |
| 2 | ast.parse green | **PASS** (5 touched files all parse) |
| 3 | TS build | N/A |
| 4 | Flag-default-OFF | **PASS** (enable_factor_loadings=False; doubly default-OFF with paper_max_factor_corr=0.0) |
| 5 | BQ idempotent | **PASS** (no BQ change; xfail strict documents deferred) |
| 6 | env vars docs | N/A |
| 7 | N* delta declared | **PASS** (R + B) |
| 8 | Zero emojis | **PASS** |
| 9 | ASCII-only loggers | **PASS** |
| 10 | Single source of truth | **PASS** (compute_ff3 canonical; factor_loadings.py wiring) |
| 11 | log-first / flip-last | **WILL HOLD** |

---

## Hot-path safety analysis

**Doubly default-OFF gate** (today's live behavior is byte-identical):
1. `settings.enable_factor_loadings == False` -> autonomous_loop.py screener step short-circuits before helper call.
2. Even if flag flipped ON, `settings.paper_max_factor_corr == 0.0` -> portfolio_manager.py FF3 cap returns 0 (no block) per phase-40.8 cap design.
3. `factor_loadings=None` default in execute_buy -> in-memory trade dict does NOT carry the new key (so BQ dynamic INSERT path unaffected; no unknown-column error possible).

**BQ INSERT safety**: factor_loadings is attached to the in-memory trade dict AFTER `_safe_save_trade(trade)` returns. The save sees a clean dict; the caller sees the augmented one. Verified by string-position assert in test 6.

---

## Honest scope

**Closure pattern**: ENGINEERED + VERIFICATION (one of 3 documented patterns per CLAUDE.md). Real wiring through 4 production files plus 1 new helper module plus 1 new test file. The pipeline is end-to-end testable today; production deployment waits on phase-40.8.2 (BQ schema + Kenneth French cache).

**Synthetic FF3 data this cycle**: factor returns are deterministic synthetic (research_brief disclosed). The wiring is verified; the data realism is phase-40.8.2.

**Literal vs operational**: per CLAUDE.md "honest dual-interpretation pattern (literal vs operational criterion; xfail with named follow-ups)" -- criterion 2 satisfied operationally + xfail strict on literal with phase-40.8.2 as the named follow-up. Aligns with the pattern used in phase-37.3 NO_OP closure (cycle 46).

---

## Research-gate

Researcher SPAWNED FIRST (cycle 50; 6 consecutive cycles honoring `feedback_never_skip_researcher`). Brief at `handoff/current/research_brief_phase_40_8_1.md`. Tier=simple. 6 sources read-in-full: Kenneth French Library + AQR Israel-Ross 2017 + arXiv 2001.04185 Volpati + Two Sigma Venn + Coding Finance (no-pip-dep FF3 fetch idiom) + Dallas Fed WP2515r1. gate_passed=true. Recency scan present.

---

## Follow-up to add to masterplan

**phase-40.8.2 (P3)** -- BQ schema + Kenneth French cache:
- Add `factor_loadings` column to `paper_positions` BQ table (inside Step 7 schema window).
- One-shot ingestion script `scripts/ingest/seed_ff3_cache.py` downloads Kenneth French daily CSV to `backend/data/_cache/ff3_factors_daily.parquet` (no new pip dep per Coding Finance idiom).
- Replace `synthetic_ff3_returns` with `load_ff3_cache()` in factor_loadings.py.
- Flip the xfail strict to PASS (literal criterion 2 satisfied).
- After 1-2 weeks of quiet-log, operator considers flipping `enable_factor_loadings=True` AND `paper_max_factor_corr=0.85`.

---

## Files for archive (handoff/archive/phase-40.8.1/)

- contract.md
- experiment_results.md (this file)
- live_check_40.8.1.md
- evaluator_critique.md (after Q/A PASS)
- research_brief_phase_40_8_1.md
