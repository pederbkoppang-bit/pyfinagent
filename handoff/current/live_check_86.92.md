# live_check — phase-86.92

**STATUS: IN PROGRESS.** Sections A–D are MEASURED and final (they are the
reproduction + diagnosis half, taken before any fix). Section E (post-fix exit 0
+ surviving mutation cells) is written after GENERATE.

Every block below is verbatim tool output from this session unless labelled
otherwise. Nothing here is transcribed from a previous session's notes.

---

## A. REPRODUCE the RED (criterion 1, first half)

Working tree, HEAD = `b1469a06`, 2026-08-16 20:47 CEST:

```
$ node scripts/qa/verify_workflow_args_boundary.mjs
FAILED: 84 passed, 3 failed
  - [3] a healthy run with a perfect envelope PASSES -- ["brief at handoff/current/research_brief_86.17.md carries NO brief_status marker -- it cannot be shown to be complete, so it does not pass. (Distinct from INCOMPLETE: a brief with no marker was not written by the write-first path at all.)","over-claim: recency_scan_performed=true but the brief carries NO dedicated recency-scan section (structural check -- .claude/rules/research-gate.md requires the section even when it reports no findings)","over-claim: urls_collected=40 but only -1 distinct URLs appear in the brief (the snippet-only set must be recorded there too)"]
  - [3] no regression: enforceGate without inputHealth behaves as before
  - [4] drop-blind-violation: KILLED (a blind run would pass without it)
```

Reproduced. Three failing assertions, matching the masterplan's record.

---

## B. THE CAUSE — LOCALISED BY EXECUTION, AND IT FALSIFIES THE FILED HYPOTHESIS

**The masterplan's `audit_basis` and the night-goal §3 both state the cause is
the stale on-disk brief `handoff/current/research_brief_86.17.md`.** Criterion 1
requires the cause be localised *by execution* rather than *inferred from the
message text* — and when executed, that hypothesis is **FALSE**.

`enforceGate` never opens that file. It is a pure function over
`(env, verification, opts)`; the brief path is a **string it renders into the
message**, nothing more. `verify_research_gate_workflow.mjs` already asserts
this independently (`ok enforceGate is pure -- no fs/process use in its body`),
and I re-measured it:

```
--- E: does enforceGate source contain ANY fs/require/import? ---
fs/readFile/require/open occurrences in enforceGate body: 0
```

### B1. The decisive control — point the fixture at a file that does not exist

If the on-disk brief were the cause, replacing it with a nonexistent path would
change the outcome. It does not:

```
--- A: EXACT checker fixture (stale verification stub) ---
gate_passed: false
  VIOL: brief at handoff/current/research_brief_86.17.md carries NO brief_status marker -- it cannot be shown to be co
  VIOL: over-claim: recency_scan_performed=true but the brief carries NO dedicated recency-scan section (structural ch
  VIOL: over-claim: urls_collected=40 but only -1 distinct URLs appear in the brief (the snippet-only set must be reco

--- B: SAME stale stub, but brief_path points at a FILE THAT DOES NOT EXIST ---
gate_passed: false | violation count: 3
```

Identical: 3 violations either way. **The on-disk brief is irrelevant to this
failure.** (A second, independent falsification: the `[4] drop-blind-violation`
cell — one of the three failures — uses `brief_path: 'p'`, which has never been
a file at all.)

### B2. The real stale fixture is the checker's own `verification` object literal

`enforceGate` reads seven `verification.*` fields. The checker hand-writes four:

```
verification fields READ by enforceGate: brief_exists, brief_non_empty, brief_status_in_brief, char_count, distinct_urls_in_brief, recency_section_present, urls_missing
fields the CHECKER fixture supplies      : brief_exists, brief_non_empty, char_count, urls_missing
MISSING from the stale fixture           : brief_status_in_brief, distinct_urls_in_brief, recency_section_present
```

Supplying the three missing fields — **with the same brief_path, the same
`enforceGate`, nothing in the gate weakened** — makes the healthy case pass:

```
--- C: REAL 86.17 path, verification stub COMPLETED with the 86.28/86.37 fields ---
gate_passed: true | violations: []
```

### B3. All three failures share this one cause

```
FAILURE 2 -- [3] no regression: enforceGate WITHOUT inputHealth
  stale stub    -> gate_passed: false (assertion wants true)
  complete stub -> gate_passed: true

FAILURE 3 -- [4] drop-blind-violation KILL cell (blind guard removed)
  stale stub    -> gate_passed: false (assertion wants true) residual viols: 3
  complete stub -> gate_passed: true | viols: 0
  CONTROL (unmutated, complete stub, blind) -> gate_passed: false (must be false: guard still works)
```

The last line is the one that matters for criterion 5: with the fixture
completed, the blind guard **still refuses** a blind run. Completing the fixture
restores the cell's ability to discriminate; it does not disarm the guard.

---

## C. THE `-1` (criterion 2)

**The sentinel is deliberate and documented; the MESSAGE that renders it is a
second defect, and is filed as such.**

`-1` is produced by the normaliser at `.claude/workflows/research-gate.js:632`:

```js
const n = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : -1)
```

It is the fail-closed coercion for a non-finite value: an absent
`distinct_urls_in_brief` becomes `-1`, so `urls > briefUrls` is true and the gate
**fails closed** rather than passing on an unsupplied count. That is correct
behaviour and is the same discipline as the `briefStatus` ABSENT branch
immediately above it. Measured:

```
--- D: the -1 sentinel: what does n(undefined) render as? ---
over-claim: urls_collected=40 but only -1 distinct URLs appear in the brief (the snippet-only set must be recorded there too)
```

The **defect** is that the rendered sentence states a falsehood about the
artifact. "only -1 distinct URLs appear in the brief" asserts a measurement of
the brief that was never taken; the true condition is "stage 2 supplied no count".
An operator reading that violation is told the brief is deficient when the
*verification input* was. This is the same class as 86.87 (a fallback that
fabricates its own audit trail).

**It is NOT fixed here.** Night-rail R5 puts `research-gate.js` off limits, and
the repair is a message change inside the graded gate. Filed as its own
masterplan step (see `experiment_results_86.92.md`).

---

## D. BLAST RADIUS (criterion 6) — MEASURED BY EXECUTION, NOT INFERRED

The gate did **not** break at phase-86.37 as filed. Bisected by running the
checker in a real `git worktree` at each commit (a plain file extraction is
contaminated — the checker shells out to `git show`, so section [1] fails for a
second, unrelated reason outside a repo):

```
a212dfe9   2026-08-09 22:36:45 +0200  ->  ALL GREEN: 87 passed, 0 failed
089726f9   2026-08-10 08:27:34 +0200  ->  ALL GREEN: 87 passed, 0 failed
cad38647   2026-08-10 08:51:11 +0200  ->  FAILED: 84 passed, 3 failed
d3bb1dfb   2026-08-10 17:34:06 +0200  ->  FAILED: 84 passed, 3 failed
```

- **Breaking commit: `cad38647`, phase-86.6, 2026-08-10 08:51:11 +0200** — it
  added `verification.recency_section_present` and
  `verification.distinct_urls_in_brief` to `enforceGate`.
- `d3bb1dfb` (phase-86.37) added the third field, `brief_status_in_brief`, to an
  **already-red** gate. It is not the cause; the failure count is 3 both before
  and after it, because all three missing fields land on the same three
  assertions.
- **Duration red: 2026-08-10 08:51 → 2026-08-16 20:47 ≈ 6 days 12 hours.**

### Did any step close on this gate's green signal while it was red?

**No.** The only step whose immutable command ran this checker and reached
`done` is **86.17**, which closed 2026-08-09 — *before* the break. Measured
harm is instead in two other places:

1. **86.23 is PENDING and blocked by it.** Its immutable verification command is
   `bash -c 'node scripts/qa/verify_research_gate_workflow.mjs && node scripts/qa/verify_workflow_args_boundary.mjs'`
   — it cannot go green while this checker exits 1.
2. **A CLOSED step's evidence has rotted.** 86.17 is `status: done`, and its
   immutable command shares that same shape. Run today:

   ```
   $ node scripts/qa/verify_research_gate_workflow.mjs && node scripts/qa/verify_workflow_args_boundary.mjs
   ALL GREEN: 124 passed, 0 failed
   FAILED: 84 passed, 3 failed
   $ echo $?
   1
   ```

   Its `done` is therefore no longer reproducible. Note this is **not** the same
   claim as "a step closed on a false green" — 86.17 closed 2026-08-09, a day
   *before* the break, so the gate was genuinely green when it certified. The
   damage is retrospective auditability, not a wrongly-admitted step.
3. **86.90 had to prove its own innocence manually.** With no green gate to read,
   that step established "I did not break this" with a detached worktree — and
   the first version of that proof was itself wrong (a worktree at HEAD already
   contains the commit under suspicion), which the 86.90 cycle-2 Q/A caught.

---

## E. POST-FIX EVIDENCE (criteria 3, 4, 5, 7)

### E1. The checker exits 0 (criterion 5, first half)

```
$ node scripts/qa/verify_workflow_args_boundary.mjs ; echo $?
ALL GREEN: 95 passed, 0 failed
0
```

Was `FAILED: 84 passed, 3 failed`. The immutable command, for completeness —
and with its weakness restated, since it was green all six days the gate was dead:

```
$ bash -c 'node --check scripts/qa/verify_workflow_args_boundary.mjs && echo parses'
parses
exit=0
```

### E2. Every mutation cell still KILLS (criterion 5, second half)

```
  ok   [3] fixture canary KILLED: dropping one required field breaks the healthy case
  ok   [3] fixture canary KILLED: the canary names exactly the dropped field
  ok   [4] restore-silent-catch: KILLED -- reverting it changes the outcome for malformed-json-string
  ok   [4] drop-post-parse-plain-object-check: KILLED -- reverting it changes the outcome for double-encoded-json
  ok   [4] drop-step_id-requirement: KILLED -- reverting it changes the outcome for object-without-step_id
  ok   [4] qa-restore-silent-catch: KILLED -- reverting it changes the outcome for malformed-json-string
  ok   [4] qa-drop-post-parse-plain-object-check: KILLED -- reverting it changes the outcome for double-encoded-json
  ok   [4] drop-empty-string-guard: KILLED -- reverting it changes the outcome for empty-string
  ok   [4] qa-drop-empty-string-guard: KILLED -- reverting it changes the outcome for empty-string
  ok   [4] qa-drop-step_id-requirement: KILLED -- reverting it changes the outcome for object-without-step_id
  ok   [4] drop-blind-violation: KILLED (a blind run would pass without it)
  ok   [5] qa-verdict.js: KILLED -- removing the blind early-return makes it spawn
  ok   [5] research-gate.js: KILLED -- removing the blind early-return makes it spawn
```

**`[4] drop-blind-violation` had been NON-DISCRIMINATING, not merely failing.**
That is the sharpest statement of the harm and it is measured, not asserted:

```
THE CELL ASSERTS blind.gate_passed === true ("without the guard it WOULD pass")
                          guard PRESENT   guard ABSENT   discriminates?
  STALE fixture    cell=false        cell=false     NO  <-- dead cell
  HEALTHY fixture  cell=false        cell=true      YES
```

The `guard PRESENT → false` column in the healthy row is the control: with the
fixture repaired, the blind guard **still refuses** a blind run. Completing the
fixture restored the cell's ability to discriminate; it did not disarm anything.

### E3. The canary catches the REAL historical rot (criterion 4)

Replayed in a `git worktree` (the repo file is never mutated — a disk-mutating
checker one interrupt away from `git add -A` is its own hazard) by deleting the
exact three fields `cad38647` and `d3bb1dfb` introduced:

```
=== baseline in worktree (patched checker) ===
ALL GREEN: 95 passed, 0 failed

=== REPLAY THE 2026-08-10 ROT: delete the 3 fields phase-86.6/86.37 added ===
  fixture shrunk by 155 bytes -- 3 fields removed, verified by assertion
  FAIL [3] fixture canary (declared): every BRIEF_VERIFICATION_SCHEMA.required field has a healthy value -- add a value to HEALTHY_VERIFICATION_VALUES for: brief_status_in_brief, distinct_urls_in_brief, recency_section_present
  FAIL [3] fixture canary (consumed): every verification.* field enforceGate READS is supplied -- enforceGate reads 7 field(s); missing from the fixture: brief_status_in_brief, distinct_urls_in_brief, recency_section_present
FAILED: 89 passed, 6 failed
```

The three original 2026-08-10 failures reappear alongside the two canary
failures, which is how the replay is shown to reproduce the historical state
rather than approximate it. Every anchor was `assert`ed present before
replacement and the byte delta asserted non-zero — a no-match `str.replace`
looks identical to success.

**Why this fixture cannot rot the same way.** It is no longer a transcription of
what `enforceGate` needed on the day it was written; it is derived from
`BRIEF_VERIFICATION_SCHEMA.required`, the script's own declaration of what stage
2 must return. A new required field therefore fails ONE named assertion in ONE
place with the remedy in the message, instead of three unrelated cells failing
with prose about a brief.

### E4. Nothing in the graded gate moved (criteria 3 and 7)

```
$ git status --porcelain -- scripts/qa/ .claude/workflows/ .claude/agents/
 M scripts/qa/verify_workflow_args_boundary.mjs

$ git diff --stat HEAD -- .claude/workflows/research-gate.js .claude/workflows/qa-verdict.js .claude/agents/qa.md
(empty)
```

Exactly one file changed, and it is the checker — not the thing being checked.
No `enforceGate` rule was relaxed, so no input that fails the gate today can
pass after this change. The schema is reached by appending an export to a
stripped **copy**, which is the mechanism the checker already used.

Sibling gates, re-run after the change:

```
verify_research_gate_workflow.mjs  ALL GREEN: 124 passed, 0 failed
verify_prompt_render_86_90.mjs     ALL GREEN: 95 passed, 0 failed
verify_rail_retry.mjs              ALL GREEN: 38 passed, 0 failed
verify_escalation_86_78.mjs        ALL CHECKS PASS (failed: 0)
```

### E5. The blocked step is unblocked

```
86.23 command exit code now: 0  (was 1)
```

### E6. The `-1` defect is FILED, not merely mentioned (criterion 2)

```
86.101 REPRODUCES from disk: True | status: pending | criteria: 5
$ git diff --numstat .claude/masterplan.json
20	0
```

Pure addition; no existing step mutated.
