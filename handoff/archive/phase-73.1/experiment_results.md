# Experiment Results — phase-73.1: D2a leakage-integrity design

Date: 2026-07-18. Session: Fable 5 + ultracode, effort MAX; RESEARCH + DESIGN ONLY ($0 metered).

## What was built

1. **Research gate** (`wf_5da65207-39a`, opus/max, tier=moderate): gate_passed=true; 5 sources read in full at implementation depth (FactFin Alg.1/Eq.4/Table 5; Detecting-Lookahead-Bias; Look-Ahead-Bench retry; AFML/CPCV canon; PiT/Time-Machine + New Quant §7.1); 8 internal files; recency scan (skfolio ships CombinatorialPurgedCV → C4 is wiring). Returned four structured `design_inputs` with spec_points/seams/cost notes. Brief: `research_brief_73.1.md`.
2. **`design_pack_73/a_leakage_integrity.md` finalized (8,962 chars)** — specs transcribed VERBATIM from the gate's design_inputs: §1 purge verification + 4-class regression-test spec (the shipped phase-69.2 purge is the canonical AFML predicate; no test exists today — net-new); §2 post-cutoff harness (MODEL_CUTOFFS registry — confirmed absent from the codebase — + labeled trusted windows; live-LLM path only); §3 FactFin PC-only counterfactual pilot (Claude-compatible; LOCAL threshold calibration — corrects 73.0's interpretive 'PC>0.7'; CONFIRMED metered at meta_scorer.py:203/:221, hard-cap M=20 ≈ $0.05-0.10/candidate, operator approval required); §4 CPCV wiring as OOS-Sharpe-distribution complement (gate.py byte-unchanged, macro-freeze-safe).
3. **Executor build steps appended pending**: 73.1.1 purge regression test [sonnet-4.6/high], 73.1.2 post-cutoff harness [sonnet-4.6/high], 73.1.3 PC pilot [opus-4.8/xhigh, metered, operator-approval-gated], 73.1.4 CPCV complement [sonnet-4.6/high] — each with an immutable live_check (pytest+mutation output; labeled eval-window artifact; PC distribution + spend ledger + approval token; CPCV run output beside the unchanged PBO scalar).
4. Criterion-1 honest mapping recorded in the contract: the "re-validation plan once the leak is fixed" maps to the stale-F clearance + §2/§4 validation paths, because 73.0 proved the leak was already fixed in phase-69.2 (criteria unamended).

## Verbatim verification output

```
$ bash -c 'test -f handoff/current/design_pack_73/a_leakage_integrity.md && grep -Eqi "purge|embargo" handoff/current/design_pack_73/a_leakage_integrity.md'
73.1 VERIFICATION COMMAND EXIT: 0 (PASS)
```

## File list

- `handoff/current/contract.md` (73.1; gate → contract → GENERATE order held — note the design-doc skeleton predated the contract as write-first discipline, disclosed here as in prior steps)
- `handoff/current/research_brief_73.1.md`
- `handoff/current/design_pack_73/a_leakage_integrity.md`
- `.claude/masterplan.json` (73.1 in-progress; 73.1.1-73.1.4 appended pending)

## Scope honesty

No product code, no .env, no flags, no optimizer runs, no metered spend (the metered PILOT is a queued executor step gated on operator approval, not executed here). The design corrects 73.0's own interpretive threshold rather than inheriting it.
