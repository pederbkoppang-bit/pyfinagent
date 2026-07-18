# Experiment Results — phase-72.1: P1 approved-but-unapplied operator token audit

Date: 2026-07-18. Session: Fable 5 + ultracode, AUDIT + RESEARCH ONLY ($0 metered; researcher via structured-output Workflow `wf_ce9e1cac-e72` on the Max rail).

## What was built

1. **Research gate** (tier=simple, floor held): gate_passed=true, 6 external sources read in full (GitOps/flag-governance reconciliation literature), 19 URLs, recency scan, 14 internal files. Brief: `handoff/current/research_brief_72.1.md`. Returned a structured 15-row `token_inventory`.
2. **`handoff/current/operator_decision_sheet_72.md` created** — ACT-NOW block (credit decision / two exact `.env` lines / restart / grep) + the full 15-row P1 reconciliation table (token verbatim, date, gated flag file:line, code default, live state with *(inferred)*/UNCONFIRMED markings, applied-verdict) + recurrence-prevention section.
3. **`money_diagnosis_72.md` §P1** filled (was placeholder).
4. **Masterplan step `72.1.1` appended** (pending, [executor: sonnet-4.6/high], report-only): sentinel reverse-leg reconciliation — detect approved-but-unapplied tokens; immutable live_check = verbatim sentinel WARN output. No auto-apply; `.env` stays operator-only.
5. Headline verdicts: exactly ONE true gap (07-09 SYNTHESIS-INTEGRITY + RJ-SHAPE, double-blocked unwritten+unloaded); 06-11 batch APPLIED; phase-69 tokens owed-not-approved (correctly dark, → P3); 6 non-flag tokens classified NOT-A-FLAG with state.

## File list

- `handoff/current/contract.md` (72.1; written after gate, before GENERATE)
- `handoff/current/research_brief_72.1.md`
- `handoff/current/operator_decision_sheet_72.md` (new)
- `handoff/current/money_diagnosis_72.md` (§P1)
- `.claude/masterplan.json` (72.1 in-progress; 72.1.1 appended pending)

## Verbatim verification output

```
$ bash -c 'test -f handoff/current/operator_decision_sheet_72.md && grep -Eqi "SYNTHESIS.?INTEGRITY" handoff/current/operator_decision_sheet_72.md && grep -Eqi "token" handoff/current/operator_decision_sheet_72.md'
72.1 VERIFICATION COMMAND EXIT: 0 (PASS)
```

## Scope honesty

No product code touched, no `.env` read or written, no flags flipped. Live-state verdicts are documentary/runtime-inferred and explicitly marked; the operator grep (ACT-NOW #4) upgrades them. The 72.1.1 step is queued for a cheaper executor session — not implemented here.
