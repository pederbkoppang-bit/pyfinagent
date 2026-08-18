STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.37
WRITTEN: 2026-08-17T14:04:30Z

# Q/A write-first record -- step 86.37, cycle 4 re-evaluation

Spawned to grade cycle 4 of step 86.37 ("a dropped research gate destroys the
whole run"). Prior verdicts per prompt: cycle 1 + cycle 2 CONDITIONAL (to be
verified from the ledger, not from Main's disclosure).

## Plan
1. Harness-compliance audit (5 items)
2. Immutable command: `node --check .claude/workflows/research-gate.js && node scripts/qa/verify_research_gate_workflow.mjs`
3. Re-derive every numeric claim in live_check_86.37.md
4. Mutation / guard-vacuity work on the drop/marker/recovery guards
5. Criterion-by-criterion MET/NOT MET

## Findings (appended as established)

### F-A. Prior-attempt / sequence evidence
- `qa_wip.py 86.37 --spawned-at 2026-08-17T14:04:30Z`: `source_present=true`,
  `attempt_number=2`, `prior_attempts=1`, `attempt_number_status="ok"`,
  `attempt_number_is_lower_bound=false`, `records_retained=2` (GAUGE, not counter),
  `records_pruned_known=null`. prior_records = 1 file: the UNSTAMPED
  `verdict_wip_86.37.md` (pre-86.36 naming), so WIP coverage of this step is
  partial by construction.
- `verdict_history_86_21.py --step 86.37 --evidence-only` at 14:04:4x =
  `status: no_rows_for_step` -> ledger reported NOTHING for this step.
- MID-EVAL MUTATION: re-run at 14:06 returns `status: ok`, `verdicts: FAIL -> CONDITIONAL`.
  The two ledger rows carry `recorded_at 2026-08-17T14:04:48Z` -- i.e. Main wrote
  them ~18s AFTER my WRITTEN stamp, DURING this evaluation, via commit 13ef5bae
  "backfill 86.37/86.79 verdict-ledger rows (labelled reconstructions)". Rows are
  self-labelled `BACKFILL (reconstruction ...; run_id unrecovered)`.
- Cross-check: attempt_number(auto)=2 vs ledger count=2 -> consistent NOW, but the
  ledger rows are hand-backfilled reconstructions written mid-evaluation, and the
  authors' own artifacts describe FOUR cycles (1,2,3,4), so the ledger under-counts
  cycles 3 and 4. sequence per ledger = FAIL -> CONDITIONAL; cycles 3/4 not in it.

### F-B. Immutable command -- REPRODUCES at current HEAD
`bash -c 'node --check .claude/workflows/research-gate.js && node scripts/qa/verify_research_gate_workflow.mjs'`
EXIT=0; tail = `ALL GREEN: 124 passed, 0 failed`; derived `grep -cE '^  (ok|FAIL) ' = 124`.
Matches live_check section 6 exactly (124/124, exit 0).

### F-C. HEAD moved after the cycle-4 capture -- re-checked, capture still holds
- cycle-4 commit = `140f1ac3` (2026-08-17 12:52:45Z), 3 artifact files, +101 lines,
  NO code.
- BUT `.claude/workflows/research-gate.js` was last committed at 13:02:10Z by
  `77f15b4d phase-86.72/86.78` -- AFTER the cycle-4 capture. So the "today's tree"
  capture was taken against a research-gate.js that HEAD has since changed.
  I re-ran the command myself at 14:06Z on the post-77f15b4d tree: still 124/0,
  exit 0. Capture therefore still reproduces; noting the staleness window as a
  disclosure gap, not a numeric error.
- `git diff HEAD` on research-gate.js / verify_research_gate_workflow.mjs /
  researcher.md / rules/research-gate.md = EMPTY (no uncommitted code drift).

### F-D. FALSE ATTRIBUTION in live_check section 6 (does not reproduce)
live_check_86.37.md:105-107 claims: "The checker has since grown to 124 -- the +3
are phase-86.28's cycle-5 additions to the same file". FALSIFIED by execution:
- phase-86.28's commits to that checker are `49793961` (10:06), `a6c3c3f3`
  (10:22), `d2e987f1` (10:46) -- ALL on 2026-08-10 MORNING, i.e. BEFORE 86.37's
  own first commit `d3bb1dfb` (17:34) and before cycle 3's `23270f29` (18:03).
  They were already inside the 121 baseline and cannot be the +3.
- Symmetric difference of `check(` titles, `23270f29` vs HEAD: 104 -> 107 sites,
  ADDED = exactly 3, REMOVED = 0. The three added are:
    + "a SINGLE stochastic drop is RETRIED, not surfaced as a dropped run"
    + "...and the retried run reports NO rail_dropped (the drop was recovered)"
    + "...and the recovered run PASSES the gate, so a retry is a real recovery
       not a downgrade"
  i.e. the stage-1 RETRY assertions added 2026-08-14 by `6b4df8f9`
  (fix(harness): retry the stochastic StructuredOutput drop) and `8b520f6c`
  (phase-86.81). Attribution is wrong; the NUMBERS (121, 124, exit 0) all
  reproduce.

### F-E. STALE RESIDUAL (b) in live_check section 6
live_check §6:119-124 re-queues residual "(b) a driver-level happy-path
assertion". That item is ALREADY CLOSED at today's tree: one of the +3 above
(`verify_research_gate_workflow.mjs:534`) drives the REAL driver with
`dropsOnceThenSucceeds` and asserts `recovered.gate_passed === true`. My
M6-RETRY-REMOVED cell fails exactly that assertion, so it is load-bearing.
Cycle 3's disclosure ("no driver-level happy path has ever existed") was true
on 2026-08-10 and is no longer true; cycle 4 carried it forward unre-derived.

### F-F. MY OWN MUTATION MATRIX (hermetic mini-repo, tracked tree never written)
CONTROL: rc=0, ALL GREEN 124 passed, 0 failed (workflow md5 identical to repo).
| cell | result | killing assertions |
|---|---|---|
| M1-VALID-UNWRAP (pre-fix shape, wrapper removed, spawn literal kept) | KILLED rc=1, 108/16 | "a stage-1 DROP does not kill the workflow -- the driver RESOLVES" reports `driver threw:` + 15 more |
| M2-SELECTIVE-RESURRECT (compliant envelope injected AFTER the loop, keyed to NON-StructuredOutput spellings -- a shape absent from the author's matrix) | KILLED rc=1, 121/3 | the 3 OTHER_SHAPES cells, each `gate_passed=true` |
| M3-MARKER-FAIL-OPEN (`: 'ABSENT'` -> `: 'COMPLETE'`) | KILLED rc=1, 120/4 | "a brief with NO brief_status marker FAILS the gate" +3 |
| M4-INCOMPLETE-ADMITTED (violation -> check) | KILLED rc=1, 122/2 | "a brief declaring brief_status=INCOMPLETE FAILS the gate even when every floor is met" |
| M5-DROP-ERROR-TEXT-GONE | KILLED rc=1, 123/1 | "rail_dropped carries the ERROR TEXT" |
| M6-RETRY-REMOVED (STAGE1_MAX_ATTEMPTS 3->1) | KILLED rc=1, 121/3 | the 3 retry cells incl. the driver-level happy path |
| M8-DROP-FIELD-DELETED (`rail_dropped` removed from the return) | KILLED rc=1, 117/7 | "rail_dropped is returned as its OWN field, not folded into gate_passed" +6 |
| M7-SCHEMA-GROWS-A-REQ-FIELD | SURVIVED 124/0 in THIS checker -- but CAUGHT by the sibling `verify_workflow_args_boundary.mjs` cell "[3] fixture canary (declared)". Sibling control in the REAL repo = ALL GREEN 96/0 exit 0, so the probe discriminates. NOTE only, not a finding against 86.37. |
| M9-STAGE2-PROMPT-DROPS-MARKER (delete the `brief_status_in_brief` instruction from the stage-2 prompt) | SURVIVED 124/0 -- unguarded |
| M10-STAGE1-STEP0B-DELETED (delete the whole 9-line born-inert block from the stage-1 prompt) | SURVIVED 124/0 -- REPRODUCES the author's own disclosed survivor |
(My mini-repo run of the SIBLING checker showed a `[1]` failure that is an
ARTEFACT of the hermetic dir having no git history; in the real repo it is
ALL GREEN 96/0, exit 0. Recorded so the red is not mistaken for a defect.)

### F-G. Criterion 1 REPRODUCED by my own driver harness (not the author's checker)
Wrapped both sources in `async function __drive(args, phase, log, agent)` and
drove with a stage-1 stub that throws + a PERFECT stage-2 stub:
  PRE-FIX  (d3bb1dfb~1) -> THREW -- NO RETURN VALUE
  POST-FIX (HEAD today) -> RESOLVED {gate_passed:false,
     rail_dropped:{dropped:true,error:"agent({schema}): subagent completed
     without calling StructuredOutput"}, violations:["empty_or_errored_return"],
     brief_verification_present:true}
Matches live_check sections 1-2 exactly. Criteria 1, 2, 3 independently confirmed.

### F-H. Criterion 5 re-derived
FLOOR_SOURCES=5 / FLOOR_URLS=10 at d3bb1dfb~1, d3bb1dfb, 133060b0, 23270f29 and
HEAD -- unchanged across every commit of the step. Over-claim assertions live at
checker :304 and :387; a mutation cell 'over-claim check removed' at :702.
Control tail includes "enforceGate is pure -- no fs/process use in its body".
Immutable command re-run at final HEAD d3fa720c: EXIT=0, ALL GREEN 124/0,
derived count 124, research-gate.js md5 e26dc258bc862beead7f4a336c978480.

### F-I. HARNESS COMPLIANCE
1. RESEARCH GATE: **NOT CLEAN.** No researcher was spawned for 86.37; the
   contract reuses `research_brief_86.31.md`. I re-derived that brief's envelope
   independently: 12 sources, 64 urls_collected, recency true, gate_passed true,
   and 66 distinct URLs actually present in the file (64 <= 66 corroborates).
   The citation is accurate and the reuse is disclosed prominently. BUT the
   standing rule is ALWAYS spawn per step, both prior Q/A cycles graded it WARN
   and asked for an explicit operator ratification, the author escalated it as
   OPERATOR ASK #1 / `51-1` ("The step cannot close without a ruling"), and
   `handoff/current/operator_asks_2026-08-11.md:81` STILL reads "Carried over
   from the 2026-08-10 goal, still unanswered" -- while the same file's banner
   shows asks 06-2 / 51-4 / #20 were ANSWERED 2026-08-14. So ASK #1 is
   demonstrably still open.
   AND: the cycle-4 evidence (experiment_results tail + live_check section 6)
   does NOT mention ASK #1 at all, though cycle 3 made it the sole reason for
   PARK. That is criteria-erosion across cycles (Dim-5 WARN).
2. CONTRACT-BEFORE-GENERATE: SATISFIED. contract mtime 2026-08-10T15:25:58Z;
   first code commit d3bb1dfb 2026-08-10 17:34:06 +0200 = 15:34:06Z.
3. EXPERIMENT_RESULTS: present, 17,206 bytes, cycle-4 section at the tail.
4. LOG-LAST: masterplan `status: pending`, retry_count 0. harness_log carries
   `## Cycle 1204 -- 2026-08-10 -- phase=86.37 result=PARKED` (a disposition
   row, not a result claim). No premature flip. CLEAN.
5. NO-VERDICT-SHOPPING: evidence CHANGED since the last verdict (cycle-2
   CONDITIONAL): cycle 3 regenerated live_check + killed the 3rd evasion;
   cycle 4 commit 140f1ac3 added +14 lines to experiment_results and +30 to
   live_check. Not verdict-shopping.

### F-J. Deterministic gates
- Python lint (derived scope): `git diff --name-only HEAD -- '*.py'` = exactly
  ONE file, `backend/api/sovereign_api.py`; non-empty guard satisfied;
  `uvx ruff check --select F821,F401,F811` = "All checks passed!", exit 0.
  That file's diff adds a `1y` window to the Sovereign UI endpoint -- a
  CONCURRENT session's work, NOT attributable to 86.37.
- 1b frontend / 1c live-UI capture / 1d backend smoke: NOT APPLICABLE. The
  step's own commit 140f1ac3 touches 3 `.md` artifacts only; no frontend/**,
  no backend/**, no UI claim anywhere in the criteria.
- NO UNINTENDED PRODUCTION CHANGE attributable to this step.

### F-K. Mutation-matrix freshness (evidence-quality residual)
live_check section 5's 9-cell matrix was measured on the CYCLE-3 tree
(121 checks). The code UNDER MUTATION changed on 2026-08-14 (the stage-1 retry
loop, 6b4df8f9/8b520f6c), and cycle 4 re-captured only the CHECKER RUN
(section 6), not the matrix. Section 6 does label items 4-5 as cycle-3 history,
so this is disclosed rather than misrepresented -- and I re-ran the equivalent
cells myself at today's tree (F-F): every one still KILLS, with different
counts (e.g. the valid-unwrap cell was "2 failed" on the cycle-3 tree and is
16 failed today). Evidence-freshness gap, not a product defect.

## VERDICT REASONING
All SIX immutable criteria MET, each on my own execution rather than on the
author's report. Product code is correct and genuinely mutation-resistant:
7 of my 7 code-path mutants died, including a selective-RESURRECT shape absent
from the author's matrix. Capping issues are all evidence/compliance-side:
(1) the unratified research-gate reuse (harness compliance not clean, and the
blocker vanished from the newest evidence), (2) the false +3 attribution in
live_check section 6, (3) the stale residual (b) re-queued after it was closed.
Two prompt-side survivors (M9 stage-2, M10 stage-1) are the author's already-
disclosed residual (c), extended by me to the stage-2 half.
=> CONDITIONAL.

COMPLETED: 2026-08-17T14:16:25Z


