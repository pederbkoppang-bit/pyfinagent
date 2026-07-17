# Evaluator Critique — Step 70.3 (S3 + money-path: atomic cross-sector swap + non-US avg-entry fix)

**Evaluator:** fresh, independent Q/A via the Workflow structured-output path (Opus 4.8, `effort: max`, $0 Max
rail, stall-immune — run wf_627ff77b-3cc). Verdict transcribed VERBATIM by Main (no-self-eval guardrail).

**VERDICT: PASS** | violated_criteria: [] | off_byte_identical: true | pre_existing_fails_confirmed: true | do_no_harm_ok: true

## Checks
- verification_command_exit: 0 | pytest_70_3_passed: true (11 passed) | no_backend_only_violation: true | no_risk_threshold_moved: true
- Harness compliance 5/5: research-before-contract, contract-before-generate (mtime-proven), results present, log-last, no-verdict-shopping (first Q/A on 70.3).

## Q/A notes (verbatim)

A. HARNESS 5/5 CLEAN. Research gate: research_brief_70.3.md gate_passed=true, 8 external sources read-in-full,
recency scan §3, 3-variant queries §1. mtime order proven: research < contract < settings < paper_trader <
portfolio_manager < autonomous_loop < test < experiment_results. experiment_results lists 5 files + verbatim
verification + per-criterion evidence. No phase=70.3 in harness_log (last = Cycle 92). evaluator_critique on disk
was the 70.2 file (first Q/A on 70.3).

B. DETERMINISTIC. verification exit 0. pytest test_phase_70_3_atomic_swap.py -> 11 passed. Import-smoke of all 4
modules OK. git status: only backend/config, backend/services (x3), backend/tests (new), handoff/, audit JSONLs.

C. OFF BYTE-IDENTICAL (5 gates verified vs git diff): (a) swap emit sizing/group_id inside `if _atomic:` (OFF ->
legacy nav*pct + swap_gid=None); (b) cross-sector inside `if weakest is None and paper_cross_sector_rotation_
enabled` (OFF -> same-sector-only); (c) avg_entry inside `if paper_avg_entry_fx_fix_enabled` (OFF else-branch is
verbatim legacy new_cost/new_qty); (d) reserved_cash defaults 0.0 -> `total_cost > cash + 0.0` == `> cash`, log
fragment gated `if reserved_cash` (byte-identical); (e) autonomous_loop atomic block wholly inside
`if paper_atomic_swap_enabled` (OFF -> flat loops unchanged). settings.py purely additive (3 Field(False)).
_cross_rotation_safe reads caps read-only + only PREPENDS a fail-safe block -- no threshold value moved.
TradeOrder.swap_group_id additive (default None), keyword-only sites -> no consumer-contract-break.

D. PRE-EXISTING FAILS CONFIRMED (33 passed, 2 failed). (1) test_swap_framework_fills_zero_buy_gap: DECISIVE
causation proof -- PAPER_SWAP_CHURN_FIX_ENABLED=false makes it PASS (2 swaps); =true makes it FAIL (1 swap). Live
env has churn_fix=True (Settings default False), so the 2nd swap's 24% delta falls under the untouched 25% bar.
The SELL carries swap_group_id=None (byte-identical atomic-OFF). 70.3 does NOT touch the churn denom or exclusion
(diff-confirmed). Env-driven, not 70.3-caused. (2) test_phase_23_2_6_backend_log_has_skipping_buy_evidence: reads
backend.log/*.gz (freshly rotated, skip_count=0). Env flake, zero connection to 70.3; also flagged pre-existing
by the 70.2 Q/A.

E. LLM JUDGMENT. C1 ATOMIC: MET (BUY-first + SELL pre-check; BUY-drop -> SELL never attempted -> book unchanged;
genuine red->green; cash-bound + $50 floor drops the whole pair). C2 CROSS-SECTOR: MET (gated + OFF byte-identical;
_cross_rotation_safe blocks count-cap breach + requires HHI strictly drop; caps never moved). C3 AVG-ENTRY: MET
(KR add-on ON -> ~70000 LOCAL; OFF documents the USD-mix; US byte-identical by algebra). C4 FAIL-SAFE: MET for all
reachable paths.

NAMED WEAKNESSES (all NOTE, none block PASS):
- W1 (latent hardening): the SELL-fails-after-BUY compensation branch in _execute_swap_pair deletes the BUY via
  delete_paper_position but does NOT credit back the current_cash execute_buy debited -> IF reached, leaks cash.
  However it is provably UNREACHABLE: execute_sell has exactly two return-None guards (no position; FX None), both
  mirrored by the helper's pre-check (get_position + _fx_local_to_usd), and neither can flip between the pre-check
  and the execute_sell call inside the sync helper. No corruption possible even flag-ON. Recommendation: also
  restore cash in the compensation, OR add a test binding the pre-check to execute_sell's guard set.
- W2 (test NOTE): test_swap_atomic_cash_bounded_and_grouped's min() does not actually bind (1000 < 1200); no test
  shows the cash bound REDUCING the buy. Floor case IS covered; min() is trivially correct.
- W3 (test NOTE): cross-sector INTEGRATION (weakest-is-None end-to-end emit) untested; only _cross_rotation_safe
  unit-tested. Gate logic correct; OFF byte-identical.

DO-NO-HARM: live book byte-identical (all flags default-OFF + double-gated behind paper_swap_enabled=True,
DARK-until-token); paper-only; $0; historical_macro FROZEN; hysteresis untouched; no risk threshold moved.
VERDICT: PASS.

## Main's disposition of the non-blocking weaknesses (recorded; not a verdict edit)
- **W1 → FO-70.3-A (follow-on):** harden the (currently-unreachable) swap SELL-fail compensation to also credit
  back the debited cash + add a test binding the helper pre-check to `execute_sell`'s return-None guard set, so a
  FUTURE new `execute_sell` failure mode can't silently make the branch reachable. Deferred (dead code today;
  editing it now would require a fresh Q/A on unchanged reachable behavior).
- **W2/W3 → follow-on test coverage:** add a cash-bound-binds test (buy reduced into ($50, nav*pct)) and a
  cross-sector end-to-end emit test. Deferred to a 70.3 test-hardening follow-on or the 64.x test-matrix phase.
