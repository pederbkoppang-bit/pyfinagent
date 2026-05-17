# phase-28.7 Smoke Test — 2026-05-17

**Step:** phase-28.7 (Multidimensional momentum composite)
**Date:** 2026-05-17
**Outcome:** PASS

## Test 1: Immutable verification
```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/tools/screener.py').read()); print('syntax OK')" && grep -qE '52.{0,5}week|fifty.two|composite_momentum' backend/tools/screener.py
syntax OK
```
Exit 0. **PASS.**

## Test 2: Settings + signature + helpers

```
multidim_momentum_enabled = False
multidim_momentum_weight_price = 0.35
multidim_momentum_weight_52w_high = 0.25
multidim_momentum_weight_sue = 0.2
multidim_momentum_weight_sector = 0.2

rank_candidates kwargs include: multidim_momentum, multidim_weights, pead_signals_lookup
screen_universe source contains: pct_to_52w_high
helpers importable: _zscore, _apply_multidim_momentum
```
**PASS.**

## Test 3: Smoke (10 candidates / 5 sectors, naive vs multidim)

Naive top-10: NVDA, LLY, AAPL, MSFT, COP, XOM, JPM, CVX, GME, JNJ.
Multidim top-10: NVDA, AAPL, COP, MSFT, LLY, XOM, CVX, JPM, JNJ, GME.

Key shifts: LLY #2 → #5 (no sector boost, moderate 52w-high); COP #5 → #3 (positive SUE + Energy sector boost); AAPL #3 → #2 (Tech sector leader boost). NVDA stays #1.

## Test 4: Q/A verdict (9 deterministic checks)

```json
{
  "ok": true,
  "verdict": "PASS",
  "audit_items": {...},
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
  "violation_details": "Mutation-test 1.36e-05 diff = rounding artifact (round(...,4)); not a bug. PASS with tolerance 1e-3.",
  "checks_run": 9
}
```

## Conclusion

Multidim composite implemented, tested with real sector data + synthetic PEAD, Q/A-verified. Default OFF. Ready for operator A/B.

## Related artifacts

- `handoff/current/contract.md`, `experiment_results.md`, `evaluator_critique.md`, `live_check_28.7.md`, `phase-28.7-research-brief.md`
- `docs/design/phase-28.7-multidim-composite.md`
- `backend/tools/screener.py`, `backend/services/autonomous_loop.py`, `backend/config/settings.py`
