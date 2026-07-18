# Contract — phase-72.3: P3 earning-capacity decision sheet (RECOMMEND-ONLY)

**Step id:** 72.3 (phase-72, depends_on 72.2 = done/PASS @665d7c0e)
**Session role:** Fable 5 + ultracode, AUDIT + RESEARCH ONLY. No product code, no .env, no flag flips, no optimizer runs, $0 metered. The operator flips levers; this step only ranks and recommends.

## Research-gate summary (gate_passed: true)

Researcher via structured-output Workflow `wf_c781c347-3ac` (opus/max, tier=moderate): 5 external sources read in full (scale-out/trend-exit evidence incl. arXiv 2604.27150 + practitioner counter-evidence; Ehsani/Harvey/Li FAJ 2023 sector-neutrality; single-variable rollout discipline incl. arXiv 2607.06117 "2 of 26 candidates survived incremental admission"), 42 URLs, recency scan, 11 internal files. Brief: `handoff/current/research_brief_72.3.md`. Returned **15 structured lever dossiers** (what-it-changes / existing evidence with file paths / expected impact / risk / rollback per lever), honest "NONE FOUND" where no internal evidence exists.

Load-bearing findings:
1. **P0 gates everything** — no lever earns until scoring is restored; all ranks are post-P0 earning-capacity.
2. **Tier-1 quantified alpha (only two)**: `paper_soft_sector_diversity` w=0.20 → **+0.20 ann Sharpe** (internal 70.2 replay `_70_2_soft_diversity_replay.json`: monotonic +0.176/+0.200/+0.234 at w=0.10/0.20/0.30, breadth +2 sectors, turnover-neutral; hard sector-neutral −0.117 WORSE; FAJ 2023 corroborates long-only sector retention); `momentum_52wh_tilt` k=0.5 → **+0.05 ann Sharpe** (`_52wh_paired_returns.json`, k=1.0 plateaus).
3. **scale_out is an honest HOLD**: zero internal backtest rows; conflicting external evidence; our own exit data (trail captured avg +17.82% on 14 trips) is trend-like → partials likely cap winners.
4. **Safety/insurance levers** (atomic_swap, avg_entry_fx_fix, sign_safe_overlays, KS-PEAK-RESET) are mechanism-proven with no $ magnitude — they protect existing profit streams, not new alpha.
5. **Dangerous-if-early**: `position_rec_fix` without synthesis-integrity ON = wrongful downgrade-SELLs on synthetic HOLDs (settings.py:203 unsafe-combination guard); `meta_scorer_enabled` is a no-op/ranking-eraser until 72.0.1 (R1) reroutes it.
6. **Rollout discipline**: ONE gated flip at a time, sequenced; never batch (single-variable rule + incremental-admission evidence).

## Hypothesis

Ranking the levers by evidence tier (quantified alpha > safety/insurance > unquantified correctness > HOLD) with an explicit one-at-a-time sequence converts the dark-flag backlog into an operator playbook that maximizes expected P&L per flip while respecting the promotion gates (DSR≥0.95 / PBO≤0.5) from the north-star charter.

## Immutable success criteria (verbatim from .claude/masterplan.json step 72.3)

- "Every dark lever in the money_recon inventory appears in operator_decision_sheet_72.md, ranked, with expected impact + risk + rollback + evidence per item"
- "All impact claims cite existing backtest results or BQ evidence -- no new optimizer runs, historical_macro untouched"
- "The sheet distinguishes recommend-ON, recommend-HOLD, and needs-more-evidence; nothing was activated by this session"

verification.command: `bash -c 'test -f handoff/current/operator_decision_sheet_72.md && grep -Eqi "rollback" handoff/current/operator_decision_sheet_72.md && grep -Eqi "scale.?out|SCALE_OUT" handoff/current/operator_decision_sheet_72.md'`

## Plan

1. GENERATE (Main synthesis from the 15 verified dossiers — no new evidence collection needed beyond one bounded BQ reconciliation query for the open $137.32 realized-P&L item from 72.2): write the P3 ranked section into `operator_decision_sheet_72.md` (tiered table + one-at-a-time sequence + HOLD list + dangerous-if-early list), covering EVERY money_recon inventory lever (incl. `paper_learn_loop_enabled` :33, which the dossiers omitted — added from the phase-69 register evidence: writer crash-dead, outcome_tracker.py:50). Update `money_diagnosis_72.md` §P3 pointer.
2. `experiment_results.md` with verbatim verification output.
3. EVALUATE via qa-verdict Workflow; transcribe verbatim.
4. LOG (Cycle 115) then flip 72.3 → done.

## References

- `handoff/current/research_brief_72.3.md` (envelope + dossiers + sources)
- `handoff/current/_70_2_soft_diversity_replay.json`, `_52wh_paired_returns.json` (the two quantified internal replays)
- `handoff/current/money_recon_2026-07-18.md` (inventory), `money_diagnosis_72.md` (P0-P2 verdicts)
- arXiv 2604.27150; arXiv 2607.06117; Ehsani/Harvey/Li FAJ 2023 (via brief)
