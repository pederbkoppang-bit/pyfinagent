# Experiment Results — phase-72.3: P3 earning-capacity decision sheet (RECOMMEND-ONLY)

Date: 2026-07-18. Session: Fable 5 + ultracode, AUDIT + RESEARCH ONLY ($0 metered; researcher via structured-output Workflow `wf_c781c347-3ac` on the Max rail).

## What was built

1. **Research gate** (tier=moderate): gate_passed=true, 5 external sources in full (scale-out evidence pro+con, FAJ 2023 sector-neutrality, incremental-admission/single-variable rollout), 42 URLs, recency scan, 11 internal files; returned **15 structured lever dossiers** with honest "NONE FOUND" markers where no internal evidence exists.
2. **Decision sheet §P3 written**: tiered ranking — 7-item recommend-ON sequence (one flip at a time, exact .env lines, expected impact + evidence + risk + rollback per row) and a 6-item recommend-HOLD list with evidence-based reasons; already-applied levers noted; every lever in the money_recon dark-lever inventory is covered (incl. `paper_learn_loop_enabled`, absent from the dossiers, added from the phase-69 register crash-dead-writer evidence).
3. **Impact claims cite existing evidence only**: the two quantified rows trace to `_70_2_soft_diversity_replay.json` (+0.176/+0.200/+0.234 monotonic; hard-neutral −0.117) and `_52wh_paired_returns.json` (+0.0548 at k=0.5, plateau at k=1.0); protective rows cite design-70 finding #9 and the +$1,103 swap stream; no optimizer run, historical_macro untouched.
4. **Evidence-hygiene item resolved**: one bounded BQ query reconciled the 72.2 $137.32 delta — whole-table +$3,057.36 (30 trips) vs since-05-15 +$3,194.68 (29 trips) = exactly one pre-05-15 trip at −$137.32. Verbatim CSV below.
5. **`money_diagnosis_72.md` §P3** closed.
6. Nothing was activated: no .env change, no flag flip, no product code.

## Verbatim outputs

```
$ bq query --use_legacy_sql=false --format=csv "... UNION ALL windows on financial_reports.paper_round_trips ..."
scope,n,total
all,30,3057.36
pre_0515,1,-137.32
since_0515,29,3194.68
```

```
$ bash -c 'test -f handoff/current/operator_decision_sheet_72.md && grep -Eqi "rollback" handoff/current/operator_decision_sheet_72.md && grep -Eqi "scale.?out|SCALE_OUT" handoff/current/operator_decision_sheet_72.md'
(exit 0)
```

## File list

- `handoff/current/contract.md` (72.3; gate → contract → GENERATE order held)
- `handoff/current/research_brief_72.3.md`
- `handoff/current/operator_decision_sheet_72.md` (§P3 ranking added)
- `handoff/current/money_diagnosis_72.md` (§P3 closed)
- `.claude/masterplan.json` (72.3 in-progress only — no new steps appended this step; remediation for HOLD items already exists as 72.0.1/72.2.x or is operator-gated)

## Scope honesty

Recommend-only throughout; the sequence requires the operator's own token per flip, and row #2 carries an explicit DSR/PBO-clearance precondition rather than claiming the replay alone clears the promotion gate. The one BQ query was a bounded read (3 aggregate rows).
