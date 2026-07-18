# Contract — phase-73.0: D1 deep frontier study

**Step id:** 73.0 (phase-73 install @9489d8df; baseline @e835464b)
**Session role:** Fable 5 + ultracode, effort MAX; RESEARCH + DESIGN ONLY. No product code, no .env, no flags, no optimizer runs, $0 metered.

## Research-gate summary (gate_passed: true)

Researcher via structured-output Workflow `wf_f3f3c7ec-0fc` (opus/max, tier=complex): 5 anchors read IN FULL via the /html/+ar5iv chain (The New Quant 2510.05533; agent survey 2408.06361; Profit Mirage 2510.07920; FinMem 2311.13743; LLM-judge overconfidence 2508.06225), 40 URLs, recency scan, full syllabus validation (all 18 non-anchor refs resolve). Brief: `handoff/current/research_brief_73.0.md`.

Load-bearing findings:
1. **Profit Mirage is decisive**: frontier agents lose 50-72% of return post-knowledge-cutoff (Sharpe decay 51-62%) on near-identical tape; LLMs answer historical price trivia at 85-93% = memorization. Every syllabus return claim is pre-cutoff-inflated → mechanism evidence ONLY (hardens the #3-first chain ordering).
2. **Access + corrections handed to the fan-out**: FinMem readable only via ar5iv (broken /html/ template); Look-Ahead-Bench canonical = arXiv:2601.13770; 2605.31201 real title = "Point-in-Time Financial RAG with Frozen LLMs and Market-Feedback Adaptive Retrieval" (tighter #6 match); AlphaAgent "IR 1.5"/"KDD'25" UNVERIFIED at abstract level — must be checked in body; 2026-dated IDs are genuine preprints.
3. **Calibration scope caveat**: the calibration anchors calibrate LLM correctness, NOT trade win-rate, with explicit no-sizing-guidance — the conviction→hit-rate→size mapping is OUR build; our existing bull/bear/DA debate can double as the deliberation-calibrator at zero extra agents.
4. **Field standards**: "compute cost per bp of excess return" is now a codified minimum-reporting standard (The New Quant §7.10) — our net-of-cost north star is the field standard. Man Group's AlphaGPT runs identical gates for AI and human signals and publishes no returns — validates our human-gate posture.
5. **Reading plan returned**: 5 readers (R-A leakage, R-B memory/self-improvement, R-C calibration→sizing, R-D cost+industry, R-E pilots default-DEFER) with per-reader sources + at-our-scale adopt/reject questions honoring frozen-macro/$0/local constraints.

## Hypothesis

Reading the full syllabus through the 5-reader plan yields mechanism-level adopt/adapt/reject/defer verdicts that (a) survive the leakage-skepticism rule, (b) respect our-scale constraints, and (c) directly parameterize the D2 design steps — with the chain ordering (#3→#2→#1→#4) confirmed or honestly revised by the evidence.

## Immutable success criteria (verbatim from .claude/masterplan.json step 73.0)

- "frontier_map_73.md carries adopt/reject mechanism verdicts for the four core dimensions (leakage integrity, memory/reflection, calibrated sizing, cost-integrated promotion) plus survey/industry anchors, each verdict citing sources read IN FULL -- no verdict rests on abstract-only reads"
- "Every adopted mechanism is justified at OUR scale (2-person local paper fund) and every frontier return claim is treated as leakage-suspect mechanism evidence only"
- "Reading coverage is honest: unreachable/paywalled sources are explicitly listed with what was attempted; 2026-dated preprint IDs from the baseline are verified-or-corrected"

verification.command: `bash -c 'test -f handoff/current/frontier_map_73.md && grep -Eqi "adopt" handoff/current/frontier_map_73.md && grep -Eqi "leakage" handoff/current/frontier_map_73.md && grep -Eqi "calibrat" handoff/current/frontier_map_73.md'`

## Plan

1. GENERATE — ultracode fan-out per the reading plan: 5 parallel readers (Explore, read-only, structured verdicts; each reads its sources IN FULL via the validated access routes and answers its questions) → barrier → synthesis/consistency verifier at effort max (cross-checks verdicts against the chain ordering, the leakage rule, and our-scale constraints; flags contradictions). Main then fills `frontier_map_73.md` (write-first skeleton already on disk, all verdicts PENDING-marked).
2. `experiment_results.md` verbatim output → qa-verdict Workflow → transcribe → LOG (Cycle 118) → flip 73.0 done.

## References

- `handoff/current/research_brief_73.0.md` (anchor notes + syllabus validation + reading plan)
- `handoff/current/frontier_baseline_2026-07-18.md` (@e835464b)
- Anchors read in full: arXiv 2510.05533, 2408.06361, 2510.07920, 2311.13743 (ar5iv), 2508.06225
