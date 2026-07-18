# Evaluator Critique — Step 73.3 (D2c calibrated-sizing design)

**Evaluator:** fresh, independent Q/A via `.claude/workflows/qa-verdict.js` (Workflow structured-output, Opus 4.8, `effort:max`, `model:opus`, $0 Max rail). Verdict = captured return value; transcribed VERBATIM by Main + persisted to `handoff/current/evaluator_critique.json`. Run `wf_10bcde12-835`.

## Verdict (verbatim JSON return)

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 3 immutable criteria MET; harness compliance clean; zero production/.env change. C1: c_calibrated_sizing.md specifies the calibration method (2-bucket empirical-Bayes Beta-Binomial shrinkage toward pooled p0, size on the Wilson/posterior LOWER bound, asymmetric scalar clip(lb/p0,s_min,s_max), Jeffreys fallback, isotonic rejected), honest small-sample math against our actual count (~59 trades/30 round-trips -> 10-15/bucket; Wilson 95% lower bound 22-30pp under point estimate with arithmetic shown; calibrated-beats-uniform ~40-50/bucket ~=100-150 total coinciding w/ TRADES_THRESHOLD=100; 'many months out' -> DEFER, stated not buried), the fallback (flag OFF -> byte-identical + empirical-Bayes self-defer even flag-ON = triple safety), and the flag-gated seam. C2: A/B plan = calibrated vs uniform (s==1) on IDENTICAL signals via counterfactual sizing replay; Brier + net-P&L + maxDD; sizing is a position-size overlay that does NOT route through PromotionGate so DSR>=0.95/PBO<=0.20 stay byte-unchanged (gate.py untouched), P&L-only win rejected per charter. C3: 73.3.1/73.3.2 appended pending, executor-tagged [sonnet-4.6/high], each with an immutable live_check; no code edited (git: only handoff/**, masterplan flip+additive, audit jsonl). Verification cmd exit=0. Every load-bearing file:line anchor verified EXACT against live code (seam target_amount@pm.py:392, caps :349/:361/:393/:396/:406-422/:427-438 downstream, REJECT :246-263 upstream, meta_scorer conviction_score:int@:38 + temperature:0.0@:235, TRADES_THRESHOLD=100@paper_go_live_gate.py:39, gate.py min_dsr=0.95/max_pbo=0.20@:21-22). First 73.3 spawn, 0 prior CONDITIONALs.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5item",
    "verification_command_exit0",
    "git_scope_no_py_no_env",
    "gate_py_clean_untouched",
    "masterplan_success_criteria_byte_identical",
    "masterplan_flip_pending_to_inprogress_not_done",
    "appended_steps_executor_tagged_with_live_checks",
    "harness_log_no_cycle121_loglast",
    "harness_log_zero_prior_conditionals",
    "research_brief_gate_passed_5_sources_recency",
    "mtime_ordering_gate_before_generate",
    "seam_anchor_spotcheck_pm_388_392",
    "downstream_caps_spotcheck_349_361_406_427",
    "binding_reject_upstream_246_263",
    "meta_scorer_conviction_temp0_spotcheck",
    "go_live_trades_threshold_39",
    "save_report_confidence_columns",
    "gate_thresholds_dsr095_pbo020",
    "wilson_lower_bound_arithmetic_recompute",
    "adversarial_lenses_correctness_reproduce_scopehonesty",
    "operator_decision_smax_surfaced",
    "activation_double_gated_triple_selfdefer"
  ],
  "harness_compliance_ok": true,
  "notes": "Two non-capping NOTES surfaced for executor/transparency, neither affects the verdict: (1) SEAM TWO-BRANCH: pm.py:388-392 is now a two-way branch on paper_risk_judge_shape_fix_enabled, so the literal string `position_pct = (cand['position_pct'] or 10.0)` the design quotes exists only in the else-branch. 73.3.2's executor must insert `position_pct *= calib_scalar(cand)` AFTER the if/else (before target_amount@:392), not a naive string-replace that would miss the flag-ON branch; the design's line-range 388-392 and its non-bypassability guarantee are correct and branch-independent, and 73.3.2's live_check (scalar at design seam + all caps downstream + byte-identical-when-OFF diff/replay proof) will enforce the both-branches patch. (2) CONTRACT MTIME: contract.md final write (18:24:04) postdates the design doc (18:23:38) by 26s -- the disclosed write-first-skeleton-then-finalize pattern (flagged precedented by author + caller). The anti-retrofit guarantee holds decisively: contract immutable criteria are byte-identical to the frozen masterplan criteria (no reverse-engineering) and the research gate (18:21:45) genuinely preceded all writes. HONEST-WEIGHING items all confirmed positive: s_max is SURFACED as an operator decision (defensive 1.0 default vs 1.25-1.5) not silently decided; 'live calibrated sizing is many months out at current cadence -> DEFER' is stated in the top honest-math section AND decisions-of-record, not buried; activation is double-gated (operator token AND ~100-150-trip data bar) plus a third empirical-Bayes self-defer, consistent with dark-until-token. Design doc is 14,740 bytes (claim '14,698 chars' is chars-vs-bytes on ~42 multibyte glyphs; non-issue). This is a DESIGN step (no code shipped) so lint/tsc/runtime-smoke gates are N/A; anti-rubber-stamp was exercised by verifying every load-bearing file:line claim against ground-truth code rather than trusting the prose -- all verified exact."
}
```
