# live_check 61.2 -- Decision-input integrity (DARK BUILD; live legs DEPLOY-PENDING)

Immutable shape: "BQ rows from at least one post-fix autonomous cycle:
non-null company_name on full-path rows, zero new rows with final_score=0.0
AND final_synthesis.error set, and non-constant conviction values in
paper_trades.signals."

## Status 2026-07-08 (build day)

Code committed DARK (flags OFF; no backend restart -- PID 24910 is tonight's
66.2 evidence host and must not be touched). The immutable live evidence
REQUIRES a deployed post-fix cycle, therefore this file currently records the
test-level evidence and the deploy plan; the BQ sections below are appended
after the first post-deploy cycle. Expected Q/A verdict at this stage:
CONDITIONAL (designed intermediate state, 39.1 scheduled-evidence doctrine).

## Test-level evidence (today, verbatim commands in experiment_results_61.2.md)

- Immutable pytest leg: 46 passed (33 new in
  test_phase_61_2_decision_integrity.py + 13 pre-existing matches).
- Regression: 45 passed (rail_guard/66_1/66_3/60_4/62_4). IMPORT_OK. npm
  build clean.
- Flag-OFF byte-identity asserted: legacy fabrication (HOLD/0.0), legacy
  saturated clamp (10/10/10), legacy pos_row reason-string -- all covered.

## Deploy plan (operator-visible)

1. AFTER tonight's 18:00 UTC cycle completes (~20:00), restart backend
   (bootout/bootstrap or kickstart -k) -- picks up timeout=150s + read-side
   guards + company_name fix (ungated deltas). Flags remain OFF.
2. Criterion-3 evidence (company_name) accrues from the FIRST post-restart
   scheduled cycle's full-path rows -- no flag needed.
3. Flag promotion (paper_synthesis_integrity_enabled -> ON) is an OPERATOR
   decision; on ON, the zero-fabricated-rows and non-constant-conviction legs
   become observable. paper_position_recommendation_fix_enabled staged after.

## Section A -- post-deploy cycle BQ evidence (PENDING)

(To append: company_name non-null on full-path analysis_results rows; zero
rows with final_score=0.0 AND $.final_synthesis.error; conviction spread in
paper_trades.signals; scheduled-cycle id + timestamps.)
