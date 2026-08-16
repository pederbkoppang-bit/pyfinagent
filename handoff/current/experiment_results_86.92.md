# experiment_results — phase-86.92

## What changed

**One file.** `scripts/qa/verify_workflow_args_boundary.mjs`.

Deliberately NOT changed — verified by `git diff --stat HEAD` returning empty
for all three (rail R5 + criterion 3 + criterion 7):

```
$ git diff --stat HEAD -- .claude/workflows/research-gate.js .claude/workflows/qa-verdict.js .claude/agents/qa.md
(empty)
$ git status --porcelain -- scripts/qa/ .claude/workflows/ .claude/agents/
 M scripts/qa/verify_workflow_args_boundary.mjs
```

Plus one masterplan addition: new step **86.101** (the `-1` message defect).
`git diff --numstat .claude/masterplan.json` → `20  0` — pure addition.

---

## The finding, in one line

The gate was not red because of a stale brief on disk. It was red because **the
checker's own hand-written `verification` fixture had rotted**, and the rot was
invisible because an absent field makes `enforceGate` *fail closed* — correct
for a gate, catastrophic for a fixture.

Full measured evidence: `live_check_86.92.md` §A–D.

---

## The fix

### 1. The fixture is now synthetic and owned by the checker (criterion 4)

Both hand-written literals (`:179` and its clone at `:319`) are replaced by a
single `healthyVerification()` factory over a declared
`HEALTHY_VERIFICATION_VALUES` map.

### 2. The rot is now self-announcing, in one named place

Four new assertions plus two controls. The design point: when the next
`verification.*` field lands, **one** assertion fails **by name** and tells the
author what to do — instead of three unrelated cells failing with prose about a
brief nobody needs to touch.

- **declared** — every `BRIEF_VERIFICATION_SCHEMA.required` field has a healthy
  value. This is the assertion that would have fired on 2026-08-10.
- **consumed** — every `verification.*` field `enforceGate` actually reads is
  supplied. Not redundant with the above: it catches the opposite drift.
- **undeclared** — `enforceGate` reads nothing the schema fails to require.
- **anchored** — the scanned region really is `enforceGate`; if a boundary
  marker moves, that is *reported*, not silently scanned. (A slicing checker
  cannot cover what it slices away — the exact defect that produced 86.17's own
  cycle-2 failure.)

The schema is reached **without editing `research-gate.js`**: the checker already
appends its own `export` line to a stripped *copy* before importing, so it now
appends `export { enforceGate, BRIEF_VERIFICATION_SCHEMA }`.

### 3. Controls, because a canary that cannot fail is not a canary

The field scanner strips comments first — otherwise a field named only in prose
would be demanded of the fixture, inventing a false red. That stripping has a
**positive control**: a bogus field is injected into a comment, and the checker
asserts the raw source *does* contain it while the stripped scan *rejects* it.
If the injection ever stops landing, the control fails rather than passing
vacuously.

### 4. Mutation cells for the new guard

Dropping a required field from a copy of the fixture must (a) break the healthy
case and (b) be **named** by the canary. Assertion (b) is a **differential**
(missing-set before vs after), not an absolute count — an absolute cell is
coupled to a baseline that happens to be complete, and would report a phantom
second defect whenever the fixture is rotted for any other reason.

---

## Verification output

### The immutable command

```
$ bash -c 'node --check scripts/qa/verify_workflow_args_boundary.mjs && echo parses'
parses
```

**Disclosed:** this is a parse check. It was green throughout the six days the
gate was dead and *cannot* fail on this defect. It is reported because it is the
step's immutable command, not because it is evidence. The evidence is below.

### The checker itself (criterion 5)

```
$ node scripts/qa/verify_workflow_args_boundary.mjs ; echo $?
ALL GREEN: 95 passed, 0 failed
0
```

Was `FAILED: 84 passed, 3 failed`. 95 = 87 (the pre-rot green count) + 8 new
assertions.

### Mutation cells still KILL — including the one the rot had disabled

```
ok   [4] restore-silent-catch: KILLED -- reverting it changes the outcome for malformed-json-string
ok   [4] drop-post-parse-plain-object-check: KILLED -- reverting it changes the outcome for double-encoded-json
ok   [4] drop-step_id-requirement: KILLED -- reverting it changes the outcome for object-without-step_id
ok   [4] qa-restore-silent-catch: KILLED
ok   [4] qa-drop-post-parse-plain-object-check: KILLED
ok   [4] drop-empty-string-guard: KILLED
ok   [4] qa-drop-empty-string-guard: KILLED
ok   [4] qa-drop-step_id-requirement: KILLED
ok   [4] drop-blind-violation: KILLED (a blind run would pass without it)
ok   [5] qa-verdict.js: KILLED -- removing the blind early-return makes it spawn
ok   [5] research-gate.js: KILLED -- removing the blind early-return makes it spawn
```

**`[4] drop-blind-violation` is the important one.** The rot had not merely made
it *fail* — it had made it **non-discriminating**. Measured both ways:

```
THE CELL ASSERTS blind.gate_passed === true ("without the guard it WOULD pass")
                          guard PRESENT   guard ABSENT   discriminates?
  STALE fixture    cell=false        cell=false     NO  <-- dead cell
  HEALTHY fixture  cell=false        cell=true      YES
```

A cell whose control answer and mutant answer agree is testing nothing. This is
the concrete sense in which the rot was *worse* than a red checker.

### The canary catches the actual historical rot (criterion 4)

Replayed in a `git worktree` — the repo file is never mutated — by deleting the
exact three fields phase-86.6 and phase-86.37 added:

```
  fixture shrunk by 155 bytes -- 3 fields removed, verified by assertion
  FAIL [3] fixture canary (declared): every BRIEF_VERIFICATION_SCHEMA.required field has a healthy value
       -- add a value to HEALTHY_VERIFICATION_VALUES for: brief_status_in_brief, distinct_urls_in_brief, recency_section_present
  FAIL [3] fixture canary (consumed): every verification.* field enforceGate READS is supplied
       -- enforceGate reads 7 field(s); missing from the fixture: brief_status_in_brief, distinct_urls_in_brief, recency_section_present
```

The three original 2026-08-10 failures reappear alongside it, confirming the
replay reproduces the historical state faithfully rather than approximating it.
The mutation used an `assert` on every anchor before replacing — a no-match
`str.replace` looks identical to success — and asserts the byte count changed.

### No regression in the sibling gates

```
verify_research_gate_workflow.mjs  ALL GREEN: 124 passed, 0 failed
verify_prompt_render_86_90.mjs     ALL GREEN: 95 passed, 0 failed
verify_rail_retry.mjs              ALL GREEN: 38 passed, 0 failed
verify_escalation_86_78.mjs        ALL CHECKS PASS (failed: 0)
```

### Blast-radius payoff

86.23 is a **pending** step whose immutable command is exactly this checker. It
could not go green while the gate was dead:

```
86.23 command exit code now: 0  (was 1)
```

---

## Criterion 2 — the `-1`, and what was done about it

**Both halves of the criterion's disjunction turned out to be true**, so both are
answered rather than one being chosen:

- The **sentinel is deliberate and documented**. `n()` at
  `research-gate.js:632` coerces a non-finite value to `-1` so the gate **fails
  closed** on an unsupplied count. Same discipline as the `ABSENT` branch above it.
- The **rendering is a second defect**, and is **filed as step 86.101**. The
  message states a measurement of the brief that was never taken. It is not
  fixed here: the repair lives inside `research-gate.js`, which rail R5 puts off
  limits tonight, and it deserves its own criteria (including a sibling audit —
  `sources` and `urls` pass through the identical coercion).

Filed and confirmed to reproduce from disk:

```
86.101 REPRODUCES from disk: True | status: pending | criteria: 5
```

---

## Scope honesty — what I did NOT do

- Did not touch `research_brief_86.17.md`. It is not the cause; editing it would
  leave the checker RED (measured), and it would have been a cargo-cult fix
  following the step title.
- Did not weaken any `enforceGate` rule. Not one line of the graded gate moved.
- Did not fix the `-1` message (filed as 86.101).
- Did not close 86.23. It is unblocked by this work but is its own step.
- The `[3] fixture canary KILLED` differential cell reports
  `(none -- the canary did not notice)` when replayed against an
  already-rotted baseline. That is correct — deleting an already-absent field
  introduces nothing — and is stated rather than tuned away.

---

## Correction to the filed record

The masterplan's `audit_basis` for this step, the night goal §3 item 1, and the
step's own `name` all attribute the RED to `research_brief_86.17.md` and to
phase-86.37. **Both are false**, established by execution:

| filed claim | measured |
|---|---|
| cause is the stale on-disk brief | cause is the checker's own `verification` literal; a nonexistent path gives identical violations |
| broke at phase-86.37 (`d3bb1dfb`) | broke at phase-86.6 (`cad38647`), 2026-08-10 08:51; 86.37 joined an already-red gate |

The criteria themselves are untouched — criterion 1 is precisely what forced the
correction, by requiring localisation *by execution* rather than *from the
message text*.
