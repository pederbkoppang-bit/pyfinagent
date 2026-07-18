# Contract — phase-72.4: P4 regime deployment-policy research

**Step id:** 72.4 (phase-72, depends_on 72.3 = done/PASS @7502c664)
**Session role:** Fable 5 + ultracode, AUDIT + RESEARCH ONLY. No code, no config, no flag changes — the deliverable is a recommend-only policy in the decision sheet.

## Research-gate summary (gate_passed: true — the gate IS this step's core work)

Researcher via structured-output Workflow `wf_39390b7c-9f3` (opus/max, tier=moderate): 7 external sources read in full (Daniel-Moskowitz momentum crashes / dynamic weighting; Barroso-Santa-Clara vol-scaling; Nystrup + Shu-Mulvey regime-switching allocation; Faber trend-following; Vanguard + JPM cash-drag), 31 URLs, recency scan (DL/transformer regime models assessed and rejected as un-deployable), 6 internal files (screener composite, macro_regime multiplier chain, kill-switch, cash floor). Brief: `handoff/current/research_brief_72.4.md`. Returned a structured `recommended_policy` with statement / evidence_for / evidence_against / fit_to_pyfinagent (file:line).

Load-bearing findings:
1. **Scale, don't switch** — near-unanimous: continuous exposure scaling beats binary cash-flips (D-M dynamic weight OOS Sharpe 1.19; B-S-C 0.53→0.97; binary switching "too extreme for practical trading").
2. Binary trend/cash gating buys drawdown reduction, not return, and **whipsaws in exactly our flat regime** (Faber: identical avg returns; edge only over full cycles).
3. Cash is a funding-level decision, not a timing dial (Vanguard 2-20bps/yr drag; JPM: best/worst days cluster — a cash gate forfeits rebounds).
4. Regime detectors are least reliable in flat/choppy regimes → any regime lever must be a soft continuous down-weight on a fail-safe multiplier, never a hard switch.
5. **pyfinagent already owns the endorsed mechanism as a dark lever**: `macro_regime_filter_enabled` (settings.py:388) → continuous conviction multiplier (risk_off ×0.70, macro_regime.py:33-38) → `apply_regime_to_score` (:604-630) → momentum composite. Zero new code.
6. The recent ~100% cash was the P0 defect, not a policy — the book has no explicit deploy-vs-cash gate; it is emergent from screen + BUY seam + 5% floor.

## Hypothesis

Codifying "deploy by default; scale by regime via the existing continuous multiplier; cash only as residual; no binary gate" as a recommend-only decision-sheet item gives the operator a defensible earning definition per regime, aligned with both the literature and the mechanisms already built.

## Immutable success criteria (verbatim from .claude/masterplan.json step 72.4)

- "Researcher gate cleared per protocol (>=5 sources read in full, recency scan, envelope) on regime-conditional deployment / cash-as-a-position literature"
- "A concrete recommended policy (with the evidence for and against) lands in the decision sheet as a recommend-only item"
- "No code, config, or flag changes"

verification.command: `bash -c 'grep -Eqi "regime" handoff/current/operator_decision_sheet_72.md'`

## Plan

1. GENERATE (Main transcription/synthesis of the returned policy): write the §P4 policy section into `operator_decision_sheet_72.md` (statement, evidence for/against, fit-to-mechanisms, sequencing interplay with the P3 queue — the policy carves `macro_regime_filter_enabled` out of the P3 overlay-library HOLD with regime-specific justification and its own one-lever-measured-first discipline). Update `money_diagnosis_72.md` §P4.
2. `experiment_results.md` verbatim output → qa-verdict Workflow → transcribe → LOG (Cycle 116) → flip 72.4 done.

## References

- `handoff/current/research_brief_72.4.md` (envelope + per-source notes)
- Daniel & Moskowitz (momentum crashes); Barroso & Santa-Clara; Nystrup et al.; Shu & Mulvey; Faber GTAA; Vanguard cash-drag; JPM stay-invested (URLs in brief)
- `money_diagnosis_72.md` P0-P3 (defect-vs-policy distinction; P3 lever queue)
