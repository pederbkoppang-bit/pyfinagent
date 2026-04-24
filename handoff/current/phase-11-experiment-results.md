# Experiment Results — phase-11 (Frontend coverage audit + plan)

**Step:** phase-11 (planning meta-step) **Date:** 2026-04-21

## What was done (cycle-2, post qa_11_v1 CONDITIONAL)

1. Fresh researcher (complex): 5 in full, 16 URLs, 34 internal files inspected, gate_passed=true. Brief at `handoff/current/phase-11-audit-brief.md`.
2. Q/A v1 returned CONDITIONAL with 2 real issues:
   - **Missing assumption:** no observability sub-step for the 7 new endpoints the 10 sub-steps would add
   - **Overgeneralization:** original 11.10 (`log_slot_usage` wiring into Thursday/Friday/Monthly/Rollback) is backend-only, no UI — belongs in phase-10, not phase-11
3. **Cycle-2 fixes applied:**
   - **Added new 11.10 Observability wiring** — structured logs + p50/p95/p99 latency + cost-per-call for the 7 new endpoints
   - **Renumbered old 11.10 to `phase-10.8.1`** — appended to phase-10 masterplan entry, kept as pure-backend scope
   - Strengthened verification cmds: 11.1 (UI grep for tile presence), 11.3 (POST round-trip to approval endpoint), 11.6 (`selectedWeekIso` grep, not just TSC)
   - Updated `phase-11-contract.md` + `phase-11-audit-brief.md` with cycle-2 note blocks

## Verification (no code this step — brief + contract are the deliverables)

```
$ wc -l handoff/current/phase-11-audit-brief.md
321

$ ls handoff/current/phase-11-*.md
phase-11-audit-brief.md
phase-11-contract.md
phase-11-evaluator-critique.md (qa_11_v1)
phase-11-experiment-results.md (this file)

$ git status --porcelain | grep -vE "handoff/current/" | grep -E "\.py$|\.tsx$|\.ts$"
(empty — no code/test files modified)
```

## Success criteria (self-imposed; no-code planning step)

| # | Criterion | Status |
|---|---|---|
| 1 | `brief_exists_with_coverage_matrix` | PASS — 321-line brief with full feature matrix |
| 2 | `gap_list_prioritized` | PASS — 10 sub-steps ordered operational safety → governance → transparency |
| 3 | `phase_11_block_drafts_present` | PASS — 10 sub-step JSON blocks (with cycle-2 corrections) ready to paste |
| 4 | `no_code_shipped` | PASS — only handoff/current/* touched; no .py/.tsx/.ts edits |

## Cycle-2 corrected 10 sub-steps

Each sub-step is a future harness-cycle ticket. These are NOT executed this step.

**Priority tier A — operational safety (execute first):**
- **11.1** BQ cost-budget watcher tile — daily + monthly $ spend vs caps; replaces NOK BudgetDashboard
- **11.2** Slack job heartbeat tile — 7 jobs × last-run status / last-error

**Priority tier B — governance:**
- **11.3** Monthly HITL approval UI — button to POST approve/reject; 48h countdown
- **11.4** Rollback events log viewer — `demotion_audit.jsonl` surface

**Priority tier C — transparency:**
- **11.5** Weekly ledger history viewer — past N weeks of sprint outcomes
- **11.6** Sprint tile week selector — dropdown; API already supports `?week_iso=`
- **11.7** Alt-data signal viewer — Congress/13F separate from `google_trends` field
- **11.8** Transformer signal viewer — shadow-mode; gated while phase-8.4 REJECT stands
- **11.9** Candidate-space viewer — 15,000-combo sampling DSR/PBO distribution

**Priority tier D — infra:**
- **11.10** Observability wiring — logs/metrics/latency for 11.1-11.9's 7 new endpoints

**Moved out of phase-11** (backend-only; go to phase-10 tracker):
- `phase-10.8.1` Wire `log_slot_usage` calls into 10.3/10.4/10.6/10.7 — makes 11.5/11.6 show real data

## References

- `handoff/current/phase-11-audit-brief.md` (cycle-2 revised)
- `handoff/current/phase-11-contract.md` (cycle-2 revised)
- `handoff/current/phase-11-evaluator-critique.md` (qa_11_v1 CONDITIONAL)

## Carry-forwards

- Paste the 10 sub-steps + phase-10.8.1 into `.claude/masterplan.json` (after qa_11_v2 PASS)
- Each 11.X sub-step executes as its own harness cycle later
