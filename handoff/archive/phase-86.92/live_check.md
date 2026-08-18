# live_check — phase-86.92

**STATUS: COMPLETE.** All sections are written and final. (Cycle 1 left this
marker at IN PROGRESS after section E was finished — flagged by the cycle-1 Q/A
as a born-inert violation, and correct: a completed record must flip its own
marker as its final act.)

Every block below is verbatim tool output from this session, **complete and
unelided**, unless a line explicitly says otherwise.

*Cycle-2 correction:* the cycle-1 Q/A found that two blocks were declared
verbatim while silently omitting lines — 4 of 6 FAIL lines in §E3 and 5 of 11
KILLED lines in the sibling `experiment_results`. Every such block has been
**regenerated from a fresh run**, not patched up by hand. The elided lines were
all truthful, so no conclusion changed; the defect was disclosure completeness,
and it is the kind that erodes trust in every other block on the page.

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

Provenance of those three fields, re-derived per field (cycle-2 correction — an
earlier revision of this line called them "the 86.28/86.37 fields", and the
`86.28` half does not reproduce):

```
recency_section_present  cad38647 2026-08-10 phase-86.6: P1 THE CHANNELS A CONFTEST G
distinct_urls_in_brief   cad38647 2026-08-10 phase-86.6: P1 THE CHANNELS A CONFTEST G
brief_status_in_brief    d3bb1dfb 2026-08-10 phase-86.37: a dropped research gate no
```

So the correct attribution is **phase-86.6 / phase-86.37**. Worth recording
separately: `research-gate.js:715` labels that block `phase-86.28: corroborate
the two self-reports...` in its own comment, which disagrees with the subject of
the commit that introduced it. The in-code label is where my `86.28` came from.
Nothing load-bearing rests on it — the breaking-commit finding is `cad38647`
either way — but it is a live inconsistency inside the gate's own source.

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
ALL GREEN: 96 passed, 0 failed
0
```

Was `FAILED: 84 passed, 3 failed`. 96 = 87 (the pre-rot green count) + 9 new
assertions. (Cycle 1 reported 95; the cycle-2 control repair added one more —
the poison-anchor uniqueness check.) The immutable command, for completeness —
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

Complete failure list, **regenerated** from a fresh run (cycle-2: this block
previously showed 2 of the 6 lines with no ellipsis, while the page header
declared every block verbatim):

```
=== failure list ===
FAILED: 90 passed, 6 failed
  - [3] fixture canary (declared): every BRIEF_VERIFICATION_SCHEMA.required field has a healthy value -- add a value to HEALTHY_VERIFICATION_VALUES for: brief_status_in_brief, distinct_urls_in_brief, recency_section_present
  - [3] fixture canary (consumed): every verification.* field enforceGate READS is supplied -- enforceGate reads 7 field(s); missing from the fixture: brief_status_in_brief, distinct_urls_in_brief, recency_section_present
  - [3] a healthy run with a perfect envelope PASSES -- ["brief at handoff/current/research_brief_86.17.md carries NO brief_status marker -- it cannot be shown to be complete, so it does not pass. (Distinct from INCOMPLETE: a brief with no marker was not written by the write-first path at all.)","over-claim: recency_scan_performed=true but the brief carries NO dedicated recency-scan section (structural check -- .claude/rules/research-gate.md requires the section even when it reports no findings)","over-claim: urls_collected=40 but only -1 distinct URLs appear in the brief (the snippet-only set must be recorded there too)"]
  - [3] fixture canary KILLED: the canary names exactly the dropped field -- newly-missing after the mutation: (none -- the canary did not notice)
  - [3] no regression: enforceGate without inputHealth behaves as before
  - [4] drop-blind-violation: KILLED (a blind run would pass without it)
```

All six accounted for: **two** canary failures (the new guard doing its job),
**three** original 2026-08-10 failures — which is how the replay is shown to
reproduce the historical state rather than approximate it — and **one** that is
the differential canary cell honestly reporting a no-op. Deleting a field the
rotted baseline is *already* missing introduces nothing new, so the cell says
`(none -- the canary did not notice)`. That is the correct answer to the question
it asks, and it is left as-is rather than tuned to look tidier.

(The pass count is 90 here and 96 in the unmutated tree because the mutation
turns 6 assertions red; the baseline inside the same worktree run is
`ALL GREEN: 96 passed, 0 failed`.)

Every anchor was `assert`ed present before replacement and the byte delta
asserted non-zero — a no-match `str.replace` looks identical to success.

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

---

## F. CYCLE-2 REMEDIATION — the cycle-1 Q/A found a guard of mine that could not fail

Verdict `wf_1afa11f6-75a`: **CONDITIONAL**. All 7 immutable criteria met on their
letter and every headline claim independently re-derived by the evaluator, capped
by one executed finding. It was right, and the finding is the exact class this
step exists to attack — so it is recorded here rather than quietly fixed.

### F1. The vacuous positive control (WARN — `illusory-guard`)

My cycle-1 control injected `// verification.__bogusProseOnlyField__ ...`
immediately **before** `function enforceGate` — which is the slice START anchor.
The poison therefore landed *outside* the scanned region, so `stripped` was false
whether the stripper worked or not, and `naive && !stripped` was true
unconditionally. The evaluator proved it by mutation: with **both** strip
operations replaced by an inert no-op, both control assertions still printed `ok`.

The irony is the point. The in-source comment asserted *"A control that cannot
fail is not a control"* — and the control it was attached to could not fail.

**Fix, and the proof it discriminates.** The poison now goes INSIDE the region
(anchored on `const selfReported`, verified unique and at index 36913 within the
region 33457..44477), and the control is a **scan-vs-scan differential**: the same
source, sliced identically, scanned once with the stripper live and once with it
disabled, via a new `verificationFieldsReadNoStrip()`. The two must disagree.
Re-running the evaluator's own M5 mutant:

```
unmutated:
  ok   [3] fixture canary CONTROL: the poison anchor is unique and inside the region
  ok   [3] fixture canary CONTROL: the poison IS visible when stripping is disabled
  ok   [3] fixture canary CONTROL: the stripper rejects a comment-only field
ALL GREEN: 96 passed, 0 failed

M5 mutant (both strip operations neutered, -12 bytes, anchors asserted first):
  ok   [3] fixture canary CONTROL: the poison anchor is unique and inside the region
  ok   [3] fixture canary CONTROL: the poison IS visible when stripping is disabled
  FAIL [3] fixture canary CONTROL: the stripper rejects a comment-only field -- comment stripping is inert -- a field named only in prose would be demanded of the fixture
FAILED: 95 passed, 1 failed
```

M5 now **KILLS**. Note the middle assertion is what makes the first one
meaningful: if the injection ever stops landing inside the region, *that* line
goes red instead of the control silently passing.

### F2. Disclosure completeness (WARN — `scope-honesty`)

Two blocks were declared verbatim while omitting lines without an ellipsis. Both
have been **regenerated from fresh runs** rather than hand-patched (§E2, §E3, and
the sibling `experiment_results_86.92.md`). Every omitted line was truthful and no
conclusion changed — but a page that says "verbatim" and isn't undermines every
other block on it.

### F3. Provenance (NOTE) — corrected in §B2 above

`86.28/86.37` → `86.6/86.37`, re-derived per field, with the in-code `phase-86.28`
label at `research-gate.js:715` recorded as the source of the error.

### F4. Born-inert markers (NOTE)

This file's header now reads COMPLETE. Separately, the evaluator observed that the
**committed** copy of `research_brief_86.92.md` (in `687109bb`, 20:54:47) carries
`brief_status=INCOMPLETE`, because the auto-commit caught the brief 47 seconds
before its final write at 20:55:34. The gate read the finished file — run record
`wf_2ee79ffe-d4f` shows `brief_status_in_brief: COMPLETE`, `gate_passed: true` —
and the mtime ordering (research 20:55:34 < contract 20:57:37) is unaffected. The
finished brief is committed in the cycle-2 commit.

### What was NOT changed in response to the verdict

No immutable criterion was reinterpreted, and no assertion was deleted or
weakened to accommodate a finding. Assertion count went **up**:

```
b1469a06 (pre-fix):  22 check() CALL SITES (total occurrences 23 minus the 1 function definition)
HEAD     (cycle-1):  30 check() CALL SITES (total occurrences 31 minus the 1 function definition)
working tree (cyc-2): 31 check() CALL SITES (total 32 minus the definition)
```

**Counting rule stated because the numbers differ from the evaluator's.** The
cycle-1 Q/A reported `23 -> 31`; I measure `22 -> 30`. Same underlying file — the
Q/A counted every occurrence of `check(`, which includes the `function check(...)`
definition itself; I subtract it and count only CALL SITES. Neither is wrong;
the rule has to be stated with the ratio or the two look like a contradiction.

These are call sites, not executed assertions — several sites run inside loops
(one per mutant), which is why the executed totals are higher: **87 → 95 → 96**.
