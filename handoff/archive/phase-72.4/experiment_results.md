# Experiment Results — phase-72.4: P4 regime deployment-policy research

Date: 2026-07-18. Session: Fable 5 + ultracode, AUDIT + RESEARCH ONLY ($0 metered; researcher via structured-output Workflow `wf_39390b7c-9f3` on the Max rail). The research gate IS this step's core work.

## What was built

1. **Research gate**: gate_passed=true, tier=moderate, **7 external sources read in full** (Daniel-Moskowitz, Barroso-Santa-Clara, Nystrup, Shu-Mulvey, Faber, Vanguard cash-drag, JPM stay-invested), 31 URLs, recency scan (DL/transformer regime models evaluated and rejected as un-deployable), 6 internal files. Brief: `handoff/current/research_brief_72.4.md` (write-first, incremental).
2. **Concrete recommend-only policy landed in `operator_decision_sheet_72.md` §P4**: deploy by default; scale down in weak regimes via the existing dark continuous `macro_regime_filter` multiplier (risk_off ×0.70 — zero new code, <$0.05/day); cash only as the residual of an empty positive-momentum screen; NO binary 200dma/trend gate (whipsaw evidence at our horizon); tail risk stays on live stops + kill-switch; one lever at a time, 3-5 measured cycles. Evidence FOR and AGAINST both stated (incl. the declined Faber drawdown edge and unvalidated-detector risk).
3. **`money_diagnosis_72.md` §P4** closed, preserving the defect-vs-policy distinction (the recent 100% cash was P0's defect, not a deployment choice).
4. Sequencing interplay documented: the policy carves `macro_regime_filter_enabled` (settings.py:388) out of the P3 overlay-library HOLD on regime-specific grounds and slots it into the one-at-a-time queue.
5. **No code, no config, no flag changes** — recommend-only throughout.

## File list

- `handoff/current/contract.md` (72.4; gate → contract → GENERATE order held)
- `handoff/current/research_brief_72.4.md`
- `handoff/current/operator_decision_sheet_72.md` (§P4 added)
- `handoff/current/money_diagnosis_72.md` (§P4 closed)
- `.claude/masterplan.json` (72.4 in-progress status flip only)

## Verbatim verification output

```
$ bash -c 'grep -Eqi "regime" handoff/current/operator_decision_sheet_72.md'
72.4 VERIFICATION COMMAND EXIT: 0 (PASS)
```

## Scope honesty

The policy's fit_to_pyfinagent chain was returned by the researcher with file:line anchors (settings.py:388 → autonomous_loop.py:422-434 → macro_regime.py:33-38,604-630 → screener.py:299-313); Main transcribed and formatted, adding only the sequencing cross-reference to P3. Counter-evidence is presented, not buried. Nothing was activated.
