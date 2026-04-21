# Q/A Critique — phase-10 / 10.0 (Retire phase-8.5.7 nightly cron)

**Agent:** qa (merged qa-evaluator + harness-verifier)
**Cycle:** 1 (closure-style)
**Date:** 2026-04-20
**ID:** qa_100_v1

---

## 5-item harness-compliance audit

1. **Researcher spawn:** `handoff/current/phase-10.0-research-brief.md`
   present (mtime 2026-04-20 06:33:47), closure-style envelope
   (`external_sources_read_in_full: 0` with `"note"` field).
   Accepted on precedent `qa_78_v1`, `qa_850_v1`,
   `qa_phase5_crypto_removal_v1` (retirement/closure steps do not
   introduce new external literature). PASS.

2. **Contract PRE-commit:** `phase-10.0-contract.md` mtime
   `06:34:06` strictly earlier than
   `phase-10.0-experiment-results.md` mtime `06:34:40`
   (Δ = 34s). Contract was written before GENERATE. PASS.

3. **Experiment results:** verbatim immutable-command output +
   masterplan cross-check (`phase-8.5.7.status == "done"`) +
   pytest regression baseline (152 passed, 1 skipped) +
   caveats section explicitly flagging the non-mutation of
   8.5.7 status. PASS.

4. **Log-last:** last cycle block in `handoff/harness_log.md` is
   the phase-9.10 PASS + phase-9 closure summary. No phase-10.0
   entry yet — correct, the log is to be appended AFTER this Q/A
   returns PASS. PASS.

5. **No verdict-shopping:** first Q/A invocation on 10.0 (no prior
   critique on disk, no prior `qa_100_*` id). PASS.

All 5 harness-compliance items clean.

---

## Deterministic A–D (verbatim)

### A. Immutable verification command

```
$ test -f handoff/phase-10.0-supersede-85-7.md && grep -q 'sprint_calendar.yaml' handoff/phase-10.0-supersede-85-7.md
$ echo $?
0
```

Exit code **0** — criterion satisfied.

### B. Reference count

```
$ grep -c 'sprint_calendar.yaml' handoff/phase-10.0-supersede-85-7.md
3
```

3 references (≥1 required). PASS.

### C. Masterplan cross-reference

```
$ python3 -c "import json, ..."
8.5.7 done
phase-10 pending
10.0 pending
```

- `phase-8.5.7.status == "done"` — historical record preserved, as
  the contract stipulates.
- `phase-10.status == "pending"` — parent, will flip after all
  substeps.
- `phase-10.0.status == "pending"` — orchestrator will flip to
  `done` after appending harness_log block.

Supersede doc's rationale (nightly cron → 2 weekly slots +
monthly Sortino gate, governed by `sprint_calendar.yaml`) is
consistent with the masterplan-recorded phase-10 scope.

### D. Scope check

```
$ git status --short | grep -iE "phase-10|supersede"
?? handoff/current/phase-10.0-contract.md
?? handoff/current/phase-10.0-experiment-results.md
?? handoff/current/phase-10.0-research-brief.md
?? handoff/phase-10.0-supersede-85-7.md
```

Only handoff artifacts (plus the supersede doc it verifies). No
code changes. Scope matches contract §"Out of scope". PASS.

---

## LLM judgment

### Closure legitimacy

Retirement/supersede steps legitimately return
`external_sources_read_in_full: 0` because the research question
("should we keep running phase-8.5.7 nightly?") is already
answered by the phase-10 design doc and the
`sprint_calendar.yaml` authored in phase-10.1. Searching the
external literature for "should we keep an internal retired cron"
would produce no relevant results. Precedent: `qa_78_v1`,
`qa_850_v1`, `qa_phase5_crypto_removal_v1` all accepted closure
envelopes. Accept.

### Doc content sanity (`handoff/phase-10.0-supersede-85-7.md`)

Read in full (42 lines). Confirmed:

- (a) Names **phase-8.5.7** explicitly (title, "Superseded by",
  "What 8.5.7 was", "Masterplan reconciliation").
- (b) Names **`sprint_calendar.yaml`** 3× (header, "After" bullet,
  References section).
- (c) Explains the cadence change:
  *"Before (phase-8.5.7): unbounded nightly cron, up to 100
  experiments. After (phase-10): 2 weekly slots — Thursday batch
  trigger + Friday promotion gate — governed by
  `backend/autoresearch/sprint_calendar.yaml`. Plus monthly
  Champion/Challenger Sortino gate (HITL, phase-10.6)."*
- (d) Lists cross-references: `cron.py`, `sprint_calendar.yaml`,
  `harness_log.md`.

Content is coherent, scope is accurately bounded, and the
reconciliation section correctly notes 8.5.7 stays `done` as
historical record with the supersede doc as the cross-reference.

### No masterplan status flip on 8.5.7

Immutable criterion (§"Immutable criterion" in contract) is:

> `test -f handoff/phase-10.0-supersede-85-7.md && grep -q 'sprint_calendar.yaml' handoff/phase-10.0-supersede-85-7.md`

It does NOT require flipping `phase-8.5.7.status` to
`"superseded"`. Contract §"Plan" step 2 explicitly confirms the
expectation is `status == "done"` (confirmed above). Contract and
supersede doc agree. No violation.

### Anti-rubber-stamp

Mutation-resistance: if I remove the string `sprint_calendar.yaml`
from the supersede doc, `grep -q` returns exit 1 → criterion
FAILS. The criterion is load-bearing, not cosmetic. (Not executed
— read-only Q/A.)

---

## violated_criteria

`[]` (none)

## violation_details

`[]`

## checks_run

- `harness_compliance_audit_5`
- `immutable_verification_command`
- `reference_count_grep`
- `masterplan_status_crosscheck`
- `scope_git_status`
- `supersede_doc_content_review`
- `contract_vs_criterion_alignment`

---

## Final Decision

**PASS** — `qa_100_v1`

All 5 harness-compliance items clean. Deterministic A–D all pass
(exit 0; 3 references; 8.5.7 `done` as expected; scope
handoff-only). Supersede doc content is coherent and names all
required entities. Contract does not require (and correctly does
not attempt) a status mutation on 8.5.7.

Orchestrator next steps (order matters):
1. Append cycle block to `handoff/harness_log.md`.
2. Flip `phase-10.0.status: pending -> done` in
   `.claude/masterplan.json`.
3. `archive-handoff` hook will rotate `handoff/current/phase-10.0-*`
   into `handoff/archive/phase-10.0/` on the status flip.
