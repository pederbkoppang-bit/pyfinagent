STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.90
WRITTEN: 2026-08-16T08:51:22Z

# Q/A cycle-2 evaluation of 86.90 (prompt-render defect: "[object Object]")

Spawn: Workflow rail, cycle 2. Prior cycle 1 = CONDITIONAL (per Main's tasking prompt,
run wf_70a3e2c4-a6e). Evidence claimed CHANGED (code + guard + artifacts edited).

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command, git scope, node --check, re-runnable checks
C. Mutation testing of the widened walk + M5 cell + [3] CONTROL discrimination
D. Judge the 4 newly-filed steps 86.92-86.95
E. Criterion-by-criterion MET/NOT MET

## Findings log (appended as established)

### Prior-attempt evidence (gathered, NOT used as a trigger)
- `qa_wip.py 86.90 --spawned-at 2026-08-16T08:51:22Z`: source_present=true,
  attempt_number=2, attempt_number_status=ok, attempt_number_is_lower_bound=false,
  prior_attempts=1, records_retained=2 (gauge, not a counter), identity_checked=true.
- `verdict_history_86_21.py --step 86.90 --evidence-only`: status=`no_rows_for_step`,
  verdicts=(none). Ledger has no rows for this step.
- CROSS-CHECK: attempt_number (2, auto) > ledger verdict count (0) => **the ledger is
  STALE for this step**. Sequence per the authoritative source: UNKNOWN. The prior
  CONDITIONAL is known only from Main's advisory disclosure, which is not authoritative.

### A. Harness compliance (5 items)
1. research-gate-before-contract: research_brief_86.90.md exists (43,014 B, mtime
   2026-08-16T09:59:05 local) < contract_86.90.md (10:01:10) < first work commit
   a21a5889 (10:12:48). ORDER HOLDS.
2. contract-before-generate: contract 10:01:10 < qa-verdict.js edit 10:42:14 and
   verify_prompt_render_86_90.mjs 10:42:50. HOLDS.
3. experiment_results_86.90.md present (25,465 B, 10:50:26). PRESENT.
4. log-last: masterplan 86.90 status = `pending` (walked, not grepped). NOT yet flipped.
   CORRECT ORDER.
5. no-verdict-shopping: commit 98c5b6ab (10:50:45) changed qa-verdict.js (+47/-),
   research-gate.js (+47/-), verify_prompt_render_86_90.mjs (+52), masterplan.json
   (+86), and both 86.90 artifacts. EVIDENCE GENUINELY CHANGED => documented
   fresh-respawn, not a re-ask.

### B. Deterministic
- IMMUTABLE COMMAND: `bash -c 'source .venv/bin/activate && node --check
  .claude/workflows/qa-verdict.js && echo parses'` -> `parses`, **EXIT=0**.
- Re-runnable checks, all run bare (no pipe masking the exit code):
  - `node scripts/qa/verify_prompt_render_86_90.mjs` -> EXIT=0, "ALL GREEN: 78 passed,
    0 failed" (cycle 1 was 53; the +25 is real).
  - `node scripts/qa/verify_research_gate_workflow.mjs` -> EXIT=0, 124 passed.
  - `node scripts/qa/verify_escalation_86_78.mjs` -> EXIT=0, 51 checks, 0 failed.
  - `node scripts/qa/verify_rail_retry.mjs` -> EXIT=0, 38 passed.
  - `node scripts/qa/verify_workflow_args_boundary.mjs` -> **EXIT=1, "FAILED: 84
    passed, 3 failed"**. Reproduces Main's disclosed 84/3 exactly.
- UNINTENDED PRODUCTION CHANGE: `git diff --stat HEAD` shows uncommitted edits to
  backend/api/sovereign_api.py + 5 frontend files. mtimes are **2026-08-14T13:2x**,
  i.e. two days BEFORE this step's work (2026-08-16). Neither 86.90 commit contains
  them (`git show --stat` file lists checked). PRE-EXISTING peer work, NOT this step.
  No unintended production change attributable to 86.90.

### B2. Is the args-boundary RED caused by this cycle? NO -- established by execution
  + provenance, not by reading the message text:
- The 3 failures are `[3] a healthy run with a perfect envelope PASSES`, `[3] no
  regression: enforceGate without inputHealth`, `[4] drop-blind-violation: KILLED`.
  All three are enforceGate/brief-envelope assertions.
- The fixture is `handoff/current/research_brief_86.17.md` (checker line 177).
- `git log -S'carries NO brief_status marker' -- .claude/workflows/research-gate.js`
  -> **d3bb1dfb, 2026-08-10, phase-86.37**. The rule predates 86.90 by 6 days.
- 86.90's ONLY hunk near enforceGate (a21a5889, `@@ -567,6 +750,12 @@`) adds a
  `log()` WARNING about unknown arg keys -- it does not touch a gate rule.
- The cycle-2 diff to research-gate.js is confined to 3 hunks, ALL inside the render
  block (`@@ -160`, `@@ -182`, `@@ -196`), and contains **0** occurrences of
  `enforceGate`.
  => Main's "not caused by this cycle's edits" claim REPRODUCES. Note separately that
  the 86.92 audit_basis justifies it with `git worktree add --detach HEAD`, which is
  the WRONG instrument (HEAD already contains a21a5889, so that worktree cannot
  exclude 86.90); the conclusion is right, the stated proof is not. WARN-level.

### C. Sixth-hole hunt against the WIDENED walk (getOwnPropertyDescriptors)
Harness: extracted the render block from the live qa-verdict.js in-memory via
`new Function` (no file written, no tree mutation) and drove `renderArgField` +
`jsonLosslessViolation` directly.

CONTROLS FIRST (must RENDER, proving the walk is not a blanket refusal):
- C1 plain nested `{handoff:{a},list:[p,q]}` -> RENDERS, JSON faithful. GREEN.
- C2 `Object.create(null)` with a key -> RENDERS. GREEN.

The five cycle-1 findings (A1 non-enumerable data / A2 non-enumerable toJSON /
A4 getter TOCTOU / A6 nested non-enumerable / A7 array non-index own prop):
**all five now REFUSED**. Confirmed the refusal is the walk's, not an artifact.

NEW candidates probed: H1 proxy-consistent, H2 proxy non-deterministic get,
H3 revoked proxy, H4 prototype-chain toJSON, H5 sparse array, H6 non-enumerable
array index, H7 own `__proto__` key, H8 nested proxy, H9 proxy over array.
- REFUSED (correctly): H3 (loud TypeError), H4, H5, H6.
- RENDERED but NOT a loss (equivalent mutants -- walk and stringify AGREE on the
  observed [[Get]] value): H1, H7, H8, H9.
- **REAL SURVIVOR: H2.** A Proxy whose `getOwnPropertyDescriptor` trap returns a
  DATA descriptor (so `d.get || d.set` is false) while its `get` trap is
  non-deterministic. Measured: walk inspected `call1`/`call2`, the rendered JSON
  carried `call2`, an independent stringify gave `call3`. This is exactly the TOCTOU
  the accessor refusal was added to close, surviving through a shape the accessor
  check cannot see. REACHABILITY: args are JSON-derived, and a Proxy cannot arrive
  through JSON -- same reachability class as the five cycle-1 findings. Severity is
  therefore bounded, AND the cycle-2 in-code claim was narrowed to "over the value
  shapes this boundary can actually receive", so H2 does NOT falsify the stated claim.
  NOTE-level, worth queueing, not a criterion miss.

### D. The four newly filed steps -- 86.92 / 86.93 / 86.94 / 86.95
Walked `.claude/masterplan.json` (not grepped). All four EXIST, status=pending,
harness_required=true, each with success_criteria, command and live_check.
**86.94 criterion 1 is UN-MEETABLE AS WRITTEN -- measured, not argued:**
criterion text = "the 621 -> 592 -> 706 drift is REPRODUCED first, with the commands
and their verbatim output".
Run by me at 2026-08-16T08:52:52Z:
  `git log --since=2026-08-11 --format=%H | wc -l`            -> **560**  (not 621/592)
  `git log --since=2026-08-11T00:00:00 --format=%H | wc -l`   -> **712**  (not 706)
Both pinned figures fail to reproduce, and BY THE STEP'S OWN THESIS they cannot: the
bare-date count slides DOWN as the clock advances and the midnight-pinned count
climbs UP as commits land (its upper bound is still HEAD). This is the identical trap
86.91 hit -- a criterion naming a number that cannot be regenerated -- re-committed
inside the criterion written to prevent it.
86.92/86.93/86.95 criteria: substantive and meetable. NOTE on 86.93: its immutable
`verification.command` is `test -f handoff/current/experiment_results_86.90.md` -- it
depends on ANOTHER step's handoff artifact. The archive hook COPIES rather than moves
(archive-handoff.sh, "COPY (not move) so downstream verifiers keep reading the live
file"), so a flip does not break it, but the documented `handoff/current` invariant
(verify_handoff_layout.py: no files belonging to done steps) means a housekeeping pass
would turn it RED for a reason unrelated to 86.93.

### C2. MUTATION MATRIX, INSTRUMENTED PER CELL (control run first on every cell)
Method: same in-memory data:-URL import driver; for each cell I ran the cell's own
`expect()` against (a) the UNMUTATED source -- must be false -- and (b) the mutant,
recording whether `expect()` RETURNED false, RETURNED true, or THREW. The shipped
checker's `catch (_e) { survived = false }` (line 346) scores a THROW as KILLED, so a
mutant that merely fails to build is indistinguishable from a detected one.

| cell | anchor unique | control expect() | mutant expect() | verdict |
|---|---|---|---|---|
| M1 restore-concat (qa-verdict) | YES | false | true | GENUINE KILL |
| M2 restore-concat (research-gate) | YES | false | true | GENUINE KILL |
| M3 placeholder-instead-of-throw | YES | false | **THREW "Invalid or unexpected token"** | **ARTIFACT-KILL** |
| M4 identity-arg-accepts-objects | YES | false | true | GENUINE KILL |
| M5 narrow-walk-to-Object.keys | YES | false | true | GENUINE KILL |

**M5 answers Main's question directly: it is DISCRIMINATING, not an artifact.** The
mutant BUILDS and RUNS (threw=NO, spawns=1) and its prompt carries `REPLACED`, while
the unmutated control throws `args.evidence.toJSON is a NON-ENUMERABLE own property`
with spawns=0.

**M3 IS an artifact-kill.** The mutated region, verbatim:
```
    throw new Error(RENDER_SCRIPT + ': ' + where + ' cannot be rendered ... '
      + violation + '')
    return '(unrenderable)'
    void ('
      + 'phase-86.90 closed. ...')
```
`void ('` opens a single-quoted string terminated by a newline -> SyntaxError. And even
if it parsed, `return '(unrenderable)'` sits AFTER the `throw`, i.e. dead code. The cell
can never exercise a placeholder substitution.
**But the underlying guard IS load-bearing** -- I built M3-prime, a valid-syntax,
reachable substitution (`if (violation) { return '(unrenderable)'` before the throw):
threw=NO, spawns=1, prompt contains `(unrenderable)`, so section [3]'s "Map THROWS and
names the field" goes RED. Criterion 5's behavioural coverage is genuine; the defect is
confined to the mutation CELL. WARN with a named fix, not a vacuous criterion.

### C3. Do the two [3] CONTROL cases discriminate? PROVEN BY MUTATION (Main had not)
Mutant: `return path + ' BLANKET REFUSAL'` inserted at the top of the descriptor walk.
- CONTROL plain nested  -> threw=YES spawns=0 -> [3] CONTROL assert **FAILS** (RED)
- CONTROL null-proto    -> threw=YES spawns=0 -> [3] CONTROL assert **FAILS** (RED)
- UNRENDERABLE Map      -> threw=YES spawns=0 -> [3] UNRENDERABLE assert still PASSES
So the controls are the ONLY thing that catches a blanket refusal, and they do catch it.
Main's stated belief is CORRECT and is now proven. Also verified separately that the
`k === 'toJSON'` arm fires on an ENUMERABLE toJSON (message: "would REPLACE the whole
value during serialisation"), so "refused at ANY enumerability" reproduces.

### C4. Cycle-2 guard diff is purely ADDITIVE -- nothing weakened
`git diff a21a5889 98c5b6ab -- scripts/qa/verify_prompt_render_86_90.mjs`: +5
UNRENDERABLE cases, +2 CONTROLS, +1 mutation cell. Zero deletions of assertions. The
`>= 1` Main flagged is in a NEW assertion, not a loosened old one, and C3 proves it
still catches the failure it is there to catch.

### E. CRITERION 4 -- independent re-derivation of the blast radius
My operationalization differs from the author's (they parsed 583 run records and
inspected 507 prompts; I scanned the FIRST USER MESSAGE of all **1392** agent
transcripts under the project dir and grepped the received prompt for a coerced field
header).
- mine: **22** production runs (23 including the author's own pre-fix probe spawn
  `wf_4588d8a7-e70`, correctly excluded from the production table)
- theirs: **22** rows
- **SYMMETRIC DIFFERENCE: EMPTY.** Not merely equal counts -- identical members.
- 9 step ids match exactly; "6 of them also lost `extra`" reproduces (6).
- D1 rollup re-derived programmatically from the table: 22 rows = 4 PASS + 7 DROP +
  7 CONDITIONAL + 4 FAIL. The corrected sentence REPRODUCES.
- 86.86 re-grade read from the run record itself (not from Main's prose):
  `wf_a09930e2-3d7` status=completed, `verdict:"PASS"`, `ok:true`,
  `violated_criteria:[]`.
- §6 discriminating measurement REPRODUCES: naive `'[object Object]' in prompt` = True,
  but lines that ARE a coerced field = **0**; the single occurrence is inside a JSON
  string value. Headers render bare followed by a ```json block.

### F. Criterion 1 receipt, quoted from the agent's own transcript
`wf_4588d8a7-e70`, run record timestamp **2026-08-16T07:57:30Z**; fix commit a21a5889
authored 2026-08-16T10:12:48+02:00 (= 08:12:48Z). Reproduction PRECEDES the fix by
~15 min. Received prompt contains, verbatim:
    `EVIDENCE / FILES TO READ: [object Object]`

### G. Criterion 7 -- verdict semantics
Changed lines in the 86.90 diff mentioning `enforceEscalation` / `VERDICT_SCHEMA` /
`verdict_unmodified` / `consecutive_conditionals`: **0 / 0 / 0 / 0**. The only new
control-flow outcome is a throw BEFORE any spawn (verified spawns=0), which yields NO
verdict rather than a changed one.

### H. Lint + smoke
Derived scope = `git diff --name-only a21a5889^ HEAD -- '*.py'` UNION
`git diff --name-only HEAD -- '*.py'` -> 3 files (non-empty set asserted first);
`uvx ruff check --select F821,F401,F811` via xargs -0 -> "All checks passed!", exit 0.
`import backend.api.sovereign_api` OK; ast.parse OK on both scripts/qa .py files.
No frontend/** in either commit, so 1b not triggered. No UI claims, so 1c not triggered.

### I. STALE FIGURES THAT SURVIVED THE CYCLE-2 EDIT (same class as the D1 finding)
- `experiment_results_86.90.md:410`: "section [3] asserts spawns.length === 0 on all
  **14** unrenderable cases". The checker's UNRENDERABLE array has **12** entries
  (enumerated: circular, bigint, function-valued, undefined-valued, Map, NaN,
  object step_id, A1, A2, A4, A6, A7). §5 of the same document says 12.
- `experiment_results_86.90.md:423`: the "## 11. Verification commands run" block still
  records `ALL GREEN: 53 passed, 0 failed` for the guard. Actual today: **78**. §5:159
  and the cycle-2 follow-up:453 both say 78. The document states two values for one
  command inside a block presented as a command transcript.
- `live_check_86.90.md:281` carries the same stale `ALL GREEN: 53 passed, 0 failed`.

### J. WRONG INSTRUMENT for the pre-existing-RED claim (conclusion right, proof not)
`experiment_results_86.90.md` §9.1 and 86.92's audit_basis both justify "not my change"
with `git worktree add --detach <path> HEAD`. HEAD ALREADY CONTAINS a21a5889, so that
worktree cannot exclude 86.90 -- it excludes only uncommitted work.
I established the conclusion independently and it HOLDS:
- `git log -S'carries NO brief_status marker' -- research-gate.js` -> d3bb1dfb,
  2026-08-10, phase-86.37 (6 days before 86.90).
- 86.90's only hunk near enforceGate adds a `log()` warning; the cycle-2 diff to
  research-gate.js contains **0** occurrences of `enforceGate`.

## CRITERION ROLL-UP
1 REPRODUCED before change ......... MET (F)
2 LAYER localised by execution ..... MET (M1 mutant restores template concat and the
                                     literal returns with a live object in args ->
                                     marshalling innocent, template guilty)
3 research-gate checked by exec .... MET (M2 genuine kill; 0/1392 transcripts show
                                     OBJECTIVE:/INTERNAL SCOPE: coerced)
4 BLAST RADIUS enumerated .......... MET (symmetric difference EMPTY; 86.86 resolved
                                     from its own run record)
5 FAIL LOUDLY, no placeholder ...... MET (M3-prime proves the guard load-bearing)
6 regression guard + mutation ...... MET with WARN (M1/M2/M4/M5 genuine; M3 artifact)
7 verdict semantics UNCHANGED ...... MET (G)
Harness compliance: 5/5 clean. No unintended production change.

VERDICT DIRECTION: CONDITIONAL -- every criterion MET, but four WARN/NOTE findings
(M3 artifact-kill falsifying the "5 cells, all KILLED" claim; two stale figures in a
verbatim-labelled block; 86.94 criterion 1 un-meetable as written; wrong instrument
for the pre-existing-RED proof) plus the Proxy NOTE.

COMPLETED: 2026-08-16T09:04:15Z
