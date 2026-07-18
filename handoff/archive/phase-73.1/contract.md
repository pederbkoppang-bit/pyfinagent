# Contract — phase-73.1: D2a leakage-integrity design

**Step id:** 73.1 (phase-73, depends_on 73.0 = done/PASS @83162aa0)
**Session role:** Fable 5 + ultracode, effort MAX; RESEARCH + DESIGN ONLY. No product code, no .env, no flags, no optimizer runs, $0 metered.

## Research-gate summary (gate_passed: true)

Researcher via structured-output Workflow `wf_5da65207-39a` (opus/max, tier=moderate): 5 sources read in full (Profit Mirage FactFin at implementation depth — Alg.1/Eq.4/Table 5; Detecting-Lookahead-Bias; Look-Ahead-Bench retry; AFML/CPCV canon writeup; Time-Machine-GPT/PiT line + The New Quant §7.1), 17 URLs, recency scan (skfolio now ships CombinatorialPurgedCV — C4 is wiring, not research), 8 internal files. Brief: `research_brief_73.1.md`. Returned four structured `design_inputs` (spec_points + seams + cost notes) — transcribed verbatim into the design doc.

Load-bearing findings:
1. **C1 is document+test, not build**: `_label_overlaps_test` is exactly the canonical AFML overlap predicate; no purge regression test exists anywhere in tests/ — net-new.
2. **C2 hard gap confirmed**: no knowledge-cutoff constant exists in the codebase; MODEL_CUTOFFS registry + eval-window selector are net-new; the quant-only GBM backtest is exempt — C2 governs only the live LLM signal path.
3. **LAP is BLOCKED for Claude** (requires logprobs the Anthropic API doesn't expose); FactFin PC (text-only) is the Claude-compatible substitute; IDS deferred (~10× cost).
4. **Correction to 73.0**: FactFin publishes NO formal thresholds — 'PC>0.7/IDS>0.6' was interpretive; the pilot calibrates a LOCAL cutoff (Table-5 anchors: leaky PC≈0.62 vs clean PC≈0.31).
5. **C3 cost refined + CONFIRMED METERED** (meta_scorer.py:203/:221 direct Anthropic): PC-only ≈ $0.05-0.10/candidate at M=20 hard-cap; operator cost approval required; which rail the live decision path uses is an unresolved data-gap the executor must confirm before spending.
6. **C4 is a $0 wiring job**: cpcv_folds defined-but-unused at gate.py:42; wire as an OOS-Sharpe-distribution COMPLEMENT; gate.py byte-unchanged; buildable under the macro freeze.

## Hypothesis

The four-component design (verify-the-existing-purge + post-cutoff windowing + a small Claude-compatible counterfactual pilot + CPCV complement) closes the genuinely-open leakage surface at our scale with three $0 builds and one hard-capped metered pilot — without touching gate.py's immutable thresholds or re-doing solved work.

## Immutable success criteria (verbatim from .claude/masterplan.json step 73.1)

- "a_leakage_integrity.md specifies the purge/embargo mechanics (window math for 90-135d triple-barrier horizons), the post-cutoff eval harness design, and the re-validation plan for previously-gated strategies once the leak is fixed"
- "Executor-tagged build steps appended pending with immutable live_checks (a re-run gate output on purged data; a post-cutoff eval run artifact)"
- "No product code edited by this session; historical_macro untouched"

Criterion-1 note (honest reading, criteria unamended): the purge/embargo mechanics + window math ARE specified (as verified-existing, with a regression-test spec preventing silent regression — the design's §1); the "re-validation plan once the leak is fixed" maps to §1's stale-F clearance + §2/§4's post-cutoff and CPCV validation paths, since the 73.0 study proved the leak was already fixed in phase-69.2.

verification.command: `bash -c 'test -f handoff/current/design_pack_73/a_leakage_integrity.md && grep -Eqi "purge|embargo" handoff/current/design_pack_73/a_leakage_integrity.md'`

## Plan

1. GENERATE: design doc finalized verbatim from the gate's design_inputs (done, 8,962 chars); append executor build steps 73.1.1-73.1.4 (pending, tagged, immutable live_checks).
2. `experiment_results.md` verbatim output → qa-verdict Workflow → transcribe → LOG (Cycle 119) → flip 73.1 done.

## References

- `handoff/current/research_brief_73.1.md`; `handoff/current/frontier_map_73.md` (#3 verdict + corrections)
- arXiv 2510.07920 (FactFin), 2512.23847, 2601.13770; AFML Ch.7/12 canon; The New Quant §7.1; Time-Machine-GPT principle
