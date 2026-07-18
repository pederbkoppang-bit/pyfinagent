# Contract — phase-73.3: D2c calibrated-sizing design

**Step id:** 73.3 (phase-73, depends_on 73.2 = done/PASS @6bc67776)
**Session role:** Fable 5 + ultracode, effort MAX; RESEARCH + DESIGN ONLY. No product code, no .env, no flags, no optimizer runs, $0 metered.

## Research-gate summary (gate_passed: true)

Researcher via structured-output Workflow `wf_95688663-b4a` (opus/max, tier=moderate): 5 sources read in full (Collaborative-Calibration mechanics + App A.2.2; overconfidence metrics 2508.06225; Autorater 2510.00263; Brown-Cai-DasGupta binomial intervals; Kelly-under-estimation-error 1701.02814/MacLean-Thorp-Ziemba line), 16 URLs, recency scan (incl. the adversarial 2508.18868 shrinkage-insufficient qualifier, honestly weighed), 10 internal files. Brief: `research_brief_73.3.md`. Returned five structured `design_inputs` + `sample_size_math` — transcribed verbatim into the design doc.

Load-bearing findings:
1. **Never trust raw conviction as probability**: meta_scorer is exactly the overconfident single-call-T=0 mode; the empirical calibration map against realized win rate is mandatory.
2. **The deliberation stack is already a calibrator at zero extra agents**: every vote-share input (bull/bear/aggressive/conservative/neutral confidences, DA adjustment, moderator + risk-judge) is ALREADY persisted in `analysis_results` — computable from stored columns, fully retro-backfillable, and the pairs snapshot rides `paper_trades.signals` JSON with **zero migration**.
3. **Size on the LOWER bound** (three-source convergence): estimated probabilities cause overbetting; overbetting beats underbetting in badness; the pessimistic bound means noise can only shrink a bet; asymmetric caps (s_min≈0.5; s_max is a surfaced operator decision — 1.0 defensive vs 1.25-1.5).
4. **Sizing seam exact + non-bypassable both directions**: scalar at `portfolio_manager.py:388-392` before `target_amount`; binding REJECT strictly upstream (:246-263); $50 floor/cash/sector-NAV/FF3/count caps strictly downstream.
5. **Honest math**: at today's ~30 round trips the Wilson 95% lower bound sits 22-30pp under the point estimate — the scalar correctly stays ~1.0; calibrated beats uniform at ~40-50/bucket ≈ 100-150 total, coinciding with the go-live TRADES_THRESHOLD=100; empirical-Bayes shrinkage makes uniform the AUTOMATIC low-N behavior even flag-ON (self-deferring design). Pre-72.0.1 meta-conviction flagged degraded in backfill.

## Hypothesis

A two-bucket, shrinkage-priored, lower-bound-sized calibration layer — fed by a pairs pipeline that starts logging NOW and a scalar that self-defers to 1.0 until N clears — converts conviction into a defensible sizing edge exactly when the data supports it, with zero possibility of a small-sample overbet and zero new agents/migrations.

## Immutable success criteria (verbatim from .claude/masterplan.json step 73.3)

- "c_calibrated_sizing.md specifies the calibration method, its data requirements with honest small-sample math against our actual trade count, the fallback when insufficient data, and the flag-gated integration point at the sizing seam"
- "The A/B evaluation plan defines the promotion evidence (calibrated vs uniform sizing on identical signals) consistent with the immutable gates"
- "Executor-tagged build steps appended pending with live_checks; no code edited this session"

verification.command: `bash -c 'test -f handoff/current/design_pack_73/c_calibrated_sizing.md && grep -Eqi "isotonic|bucket|calibrat" handoff/current/design_pack_73/c_calibrated_sizing.md'`

## Plan

1. GENERATE: design doc finalized verbatim from the gate (done, 14,698 chars — sample-size math §top, five component specs, decisions of record incl. the surfaced s_max operator decision); append executor build steps 73.3.1-73.3.2 (pending, tagged, immutable live_checks).
2. `experiment_results.md` verbatim output → qa-verdict Workflow → transcribe → LOG (Cycle 121) → flip 73.3 done.

## References

- `handoff/current/research_brief_73.3.md`; `frontier_map_73.md` (#1 verdict); `design_pack_73/b_learn_loop_v2.md` (outcome substrate)
- arXiv 2404.09127, 2508.06225, 2510.00263, 1701.02814, 2508.18868; Brown-Cai-DasGupta; MacLean-Thorp-Ziemba
