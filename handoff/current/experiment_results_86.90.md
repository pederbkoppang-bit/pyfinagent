# Experiment results -- phase-86.90

**Step:** `86.90` -- the Q/A Workflow rail stringifies nested `evidence`/`extra`
objects to the literal `[object Object]`
**Cycle:** 1 · **Written:** 2026-08-16 · **Contract:** `handoff/current/contract_86.90.md`

---

## 1. Files changed

| File | Change |
|---|---|
| `.claude/workflows/qa-verdict.js` | +160/-5 -- the phase-86.90 render boundary; `stepId`/`verification_command` via `renderIdentityArg`, `evidence`/`extra`/each criterion via `renderArgField`; unknown-arg-key warning |
| `.claude/workflows/research-gate.js` | +155/-4 -- the BYTE-IDENTICAL render block; `step_id`/`brief_path` via `renderIdentityArg`, `topic`/`internal_scope` via `renderArgField`; unknown-arg-key warning |
| `scripts/qa/verify_prompt_render_86_90.mjs` | NEW, **95** assertions -- behavioural driver + reproduce-from-git + unrenderable-throws + the criteria-CONTAINER guard + render-still-works controls + **6** mutation cells each with its own control + duplicate-integrity |
| `handoff/current/contract_86.90.md`, `research_brief_86.90.md`, `experiment_results_86.90.md`, `live_check_86.90.md`, `evaluator_critique_86.90.md` | handoff artifacts |
| `.claude/masterplan.json` | filed step `86.91` (a separate, earlier commit `c627a810`) |

No production trading code touched. No `.env`, no flag, no gate loosened.

---

## 2. Criterion 1 -- REPRODUCED before anything was changed

### 2a. Historical receipt, the real 86.86 spawn

From that agent's own transcript
(`.../subagents/workflows/wf_b1747d75-eec/agent-abeb0c1a9dca29d03.jsonl`),
prompt lines 61 and 63 verbatim:

```
  59|   9. no gate is loosened, no flag is promoted, and no .env is written in order to obtain a green result
  60|
  61| EVIDENCE / FILES TO READ: [object Object]
  62|
  63| ADDITIONAL CONTEXT: [object Object]
```

### 2b. Live minimal spawn, PRE-FIX -- Workflow run `wf_4588d8a7-e70`

One agent, `effort: low`, whose only task was to echo the line it received.
Return value verbatim:

```json
{"runtime_typeof_evidence":"object",
 "runtime_is_array":false,
 "runtime_keys":["handoff","changed_files","subject_sha256"],
 "runtime_json":"{\"handoff\":[\"handoff/current/contract_86.90.md\",\"handoff/current/experiment_results_86.90.md\"],\"changed_files\":[\".claude/workflows/qa-verdict.js\"],\"subject_sha256\":\"deadbeef\"}",
 "script_concat_result":"EVIDENCE / FILES TO READ: [object Object]",
 "agent_received":{"received_line":"[object Object]","is_literal_object_object":true}}
```

### 2c. Both shipped scripts driven pre-fix (execution, not source reading)

```
### qa-verdict.js — lines matching EVIDENCE/ADDITIONAL CONTEXT:
   "EVIDENCE / FILES TO READ: [object Object]"
   "ADDITIONAL CONTEXT: [object Object]"
   contains "[object Object]": true
### research-gate.js — lines matching OBJECTIVE/INTERNAL SCOPE:
   "OBJECTIVE: [object Object]"
   "INTERNAL SCOPE: [object Object]"
   contains "[object Object]": true
```

This reproduction is preserved re-runnably as section `[1]` of
`verify_prompt_render_86_90.mjs`, which regenerates it from the pre-fix blob at
`75831f4c` rather than transcribing it, so it cannot go stale.

---

## 3. Criterion 2 -- the LAYER, localised by execution

| Candidate layer | What was measured | Verdict |
|---|---|---|
| Workflow **args marshalling** | inside the real runtime: `typeof args.evidence === "object"`, `Array.isArray === false`, `Object.keys` intact, `JSON.stringify` round-trips the whole structure | **INNOCENT** |
| the script's **prompt template** | in that same runtime, `'EVIDENCE / FILES TO READ: ' + ev` evaluated to the literal | **GUILTY** |
| the **`agent()` call** / transport | the agent returned `received_line: "[object Object]"` -- it faithfully delivered what the template built | **INNOCENT** |

Corroborated by the official Anthropic Workflow doc, read in full by the research
gate: *"Claude passes the list as structured data, so the script can call array
and object methods on `args` directly without parsing it first."*

**The fix is applied at that layer** and nowhere else: the prompt-template field
boundary of each workflow script.

---

## 4. Criterion 5 -- the fix FAILS LOUDLY, and what it deliberately does NOT do

`renderArgField` implements **lossless-or-throw**:

| Input | Result |
|---|---|
| `undefined` / `null` / `''` | the field's documented default (unchanged behaviour) |
| `string` | itself, byte-for-byte -- **every existing caller is on this path** |
| finite number, boolean | `String(value)` |
| plain object / array, all members JSON-lossless | pretty JSON in a fenced block |
| circular, `BigInt`, function, Symbol key, `undefined` member, `Map`/`Set`/`Date`/class instance, `NaN`/`Infinity` | **THROWS**, naming `args.<field>` and the offending path |
| *(cycle 2)* any **non-enumerable** own property, any **accessor** (getter/setter), any own **`toJSON`**, a non-index own property on an array | **THROWS** -- see the correction below |

`renderIdentityArg` is stricter for `step_id`, `brief_path` and
`verification_command`: string or finite number only. An object `step_id` would
otherwise reach a **filename** (`verdict_wip_<stepId>__<stamp>.md`) and a shell
command line.

**Two research findings changed this design, and both are the difference between
a fix and a no-op:**

1. **A template literal would have been a NO-OP.** `+` coerces via ToPrimitive
   (`valueOf` first) and `` `${}` `` via ToString (`toString` first), but for a
   plain object both bottom out at `Object.prototype.toString` and produce the
   identical `[object Object]` (MDN, read in full).
2. **A bare `JSON.stringify` would have traded one silent loss for five.** It
   drops `undefined`-valued keys, function-valued keys and Symbol-keyed
   properties, collapses `Map`/`Set` to `{}`, and renders `NaN`/`Infinity` as
   `null`. Hence the explicit lossless walk before serialising, and a throw
   rather than a best-effort render.

**CYCLE-2 CORRECTION -- the walk was one seam short, and the claim was broader
than the walk.** The cycle-1 version enumerated with `Object.keys`, i.e. own
**enumerable** string keys only, and the Q/A found **five** constructions that
rendered LOSSILY WITHOUT THROWING through the gap:

| # | Construction | What was lost |
|---|---|---|
| A1 | non-enumerable own data property | silently dropped |
| A2 | **non-enumerable `toJSON`** | the WHOLE object replaced by the single string `"REPLACED"` -- a placeholder substitution reached *through* the guard written to forbid substitutions |
| A4 | non-deterministic getter | the walk reads it once, `JSON.stringify` reads it AGAIN (a TOCTOU) |
| A6 | nested non-enumerable | dropped |
| A7 | array with a non-index own property | dropped |

Controls behaved correctly throughout: an **enumerable** `toJSON` threw, and
`Object.create(null)` rendered losslessly.

**Reachability, stated so the severity is neither inflated nor deflated:** none of
the five is reachable from a real caller. `classifyArgs` either `JSON.parse`s a
string or passes the runtime object through, and a JSON-derived object has no
non-enumerables, no getters and no `toJSON`. So this was a CLAIM defect rather
than a live one -- and it is fixed anyway, because a guard whose stated rule is
broader than its measured behaviour is the failure mode this series is about.

The walk now enumerates `getOwnPropertyDescriptors` / `getOwnPropertyNames` and
**refuses accessor properties outright**, which is simultaneously the TOCTOU fix:
a pure data object cannot change between inspection and serialisation. All five
are asserted as new `[3]` cases, both controls are asserted as `[3] CONTROL`
cases so the walk cannot become a blanket refusal, and mutation cell **M5**
narrows the walk back to `Object.keys` and requires A2 to go red. The in-code
comment no longer states the rule absolutely; it states it with its bound.

**A SIXTH hole, found by the cycle-2 Q/A against the WIDENED walk, recorded
because it bounds the claim rather than refuting it.** A `Proxy` whose
`getOwnPropertyDescriptor` trap returns a DATA descriptor -- so `d.get || d.set`
is false and the accessor refusal never fires -- while its `get` trap is
non-deterministic. Measured: the walk saw call1/call2, the rendered JSON carried
call2, and an independent `JSON.stringify` gave call3. That is the exact TOCTOU
the accessor refusal was added to close, reached through a shape that check
cannot see.

It does **not** falsify the shipped claim, and the reason is that cycle 2 had
already narrowed that claim to *"over the value shapes this boundary can actually
receive"*: `args` is either `JSON.parse`d from a string or passed through as a
runtime object, and **a Proxy cannot arrive through JSON**. The Q/A also examined
and REJECTED four further candidates as **equivalent mutants** -- proxy-consistent,
own `__proto__`, nested proxy, proxy-over-array -- where the walk and
`JSON.stringify` agree so no loss occurs, and confirmed four more shapes are
correctly refused (revoked proxy with a loud `TypeError`, prototype-chain
`toJSON`, sparse array, non-enumerable array index). Left unfixed deliberately:
adding a Proxy defence would guard an unreachable path and enlarge the claim
again, which is the failure this finding is about.

Also carried: an **array** field coerces to `a,b` with **no `[object Object]`
marker at all**. Every census in this document keyed on that marker is therefore
a **floor**, never a total, and the checker asserts the array case separately.

---

## 5. Criterion 6 -- the regression guard, control GREEN first

`node scripts/qa/verify_prompt_render_86_90.mjs` -> **`ALL GREEN: 95 passed, 0 failed`**

It DRIVES the real shipped scripts with the runtime primitives stubbed and reads
the prompt actually handed to `agent()`. A source scan for `renderArgField(`
would pass on a file that never calls it on the path that matters.

| Section | What it proves |
|---|---|
| `[0]` CONTROL | a usable launch really does spawn and produce a prompt carrying the step id, and a STRING field is passed through unchanged -- without this, every "does not contain" assertion below could pass vacuously |
| `[1]` REPRODUCE | the pre-fix blob at `75831f4c` still yields `[object Object]`, on BOTH scripts |
| `[2]` FIXED | object AND array shapes render as JSON; every key and value reaches the prompt; no comma-joined collapse |
| `[3]` UNRENDERABLE | **12** cases x 2 scripts THROW naming the field AND spawn nothing, PLUS 2 controls proving the walk is not a blanket refusal |
| `[4]` research-gate | criterion 3, by execution |
| `[5]` MUTATION | **6** cells, anchor uniqueness checked first |
| `[6]` DUPLICATE INTEGRITY | the two copies of the block are byte-identical |

### Mutation matrix (6 cells, all KILLED)

| Cell | Mutation | Result |
|---|---|---|
| M1 | `qa-verdict.js`: restore `'EVIDENCE / FILES TO READ: ' + a.evidence` | **KILLED** -- `[2]` goes red |
| M2 | `research-gate.js`: restore `'OBJECTIVE: ' + a.topic` | **KILLED** -- the second copy is doing work too |
| M3 | replace the lossless-violation `throw` with `return '(unrenderable)'` | **KILLED** -- `[3]` goes red |
| M4 | `renderIdentityArg` accepts objects via `String(value)` | **KILLED** -- an object step id reaches a filename |
| M5 *(cycle 2)* | narrow the walk back to `Object.keys` | **KILLED** -- A2 renders `"REPLACED"` again without throwing |
| M6 *(cycle 4)* | revert the criteria CONTAINER guard to silent discard | **KILLED** -- `[3b]` goes red when a wrong-shaped rubric is silently discarded again |

*(cycle-5 correction, 2026-08-17: this table carried 5 rows under a "6 cells"
heading -- the escalation's "6 over 5" finding. The heading was right and the
table was short: the M6 container-guard cell, added in cycle 4 alongside section
`[3b]`, was never rowed. Verified against a live run today: section `[5]` emits
exactly 6 mutation cells, each with its control observed clean first.)*

**CYCLE-3 CORRECTION -- M3 was an ARTIFACT-KILL and this table previously
over-claimed.** The cycle-2 M3 replacement ended `void ('`, an unterminated
string, so the mutant was a **SyntaxError**; and the injected
`return '(unrenderable)'` sat AFTER the throw it was meant to replace, i.e. dead
code even had it parsed. The harness's `catch (_e) { survived = false }` scored
that crash as KILLED, so **the cell measured nothing for a whole cycle** and
"5 cells, all KILLED" did not reproduce. Found by the cycle-2 Q/A, which then
built its own valid, reachable variant and confirmed section `[3]` *does* go RED
-- so criterion 5's behavioural coverage was real; the CLAIM was not.

Two changes, because the cell and the harness were both at fault:
- M3 now injects `return '(unrenderable)'` immediately after `if (violation) {`
  -- valid syntax, and REACHABLE.
- The harness now scores **three** outcomes: `DETECTED`, `SURVIVED`, and
  `UNSCORABLE: the mutant did not build`. A mutant that never ran has been tested
  by nothing, so UNSCORABLE now **FAILS** the check instead of passing it.
- Every cell is additionally **CONTROLLED**: its own `expect()` must return false
  on the UNMUTATED source, or the cell proves nothing about the mutation.

**This matrix licenses exactly one claim: these 6 mutations were killed.** It
is not evidence that no other weakening survives.

**A checker bug caught by its own anchor-uniqueness rule, disclosed because it is
the repo's own recorded failure mode.** M1's first anchor was
`'EVIDENCE / FILES TO READ: ' + evidence` -- which also matches the phase-86.90
**comment block**, since that comment quotes the defective expression verbatim.
The checker reported `found 2 occurrences` and refused to mutate rather than
silently no-op'ing. The anchor now includes the leading indent and trailing
comma. A probe matching its own documentation is a recorded trap on this project
(`feedback_a_probe_can_match_its_own_documentation`) and it recurred here.

---

## 6. LIVE end-to-end proof on the fixed rail

Run `wf_a09930e2-3d7` (the 86.86 re-grade, see §7) is the first Q/A spawn on the
fixed rail and was given the same object-shaped `evidence`/`extra` shape that
produced `[object Object]` on 2026-08-15. From that agent's own transcript:

```
EVIDENCE / FILES TO READ:
```json
{
  "handoff": [
    "handoff/current/contract_86.86.md",
    "handoff/current/experiment_results_86.86.md",
    "handoff/current/live_check_86.86.md",
    "handoff/current/research_brief_86.86.md",
    "handoff/current/evaluator_critique_86.86.md"
  ],
  "the_commit": "e4f2e844 -- audit THE COMMIT, not a described diff: git show --stat e4f2e844",
  "changed_files": [ ... ],
  "rerunnable_checks": [ ... ],
  "subject_sha256_at_the_86_86_spawn": "5b714a9e5f43753c..."
}
```
```

Discriminating measurement, because a naive grep gives the WRONG answer here:

```
header lines rendered: ["EVIDENCE / FILES TO READ: ", "ADDITIONAL CONTEXT: "]
lines that ARE a coerced field: [] -> count 0
```

A plain `'[object Object]' in prompt` returns **True** for this prompt -- because
the `extra` object's own prose *explains* the defect and contains the phrase.
That is the self-matching-probe trap again, in the opposite direction: the string
survived **inside** the JSON block, which is itself evidence the render is
lossless. The count that discriminates is *lines that ARE a coerced field*: **0**.

---

## 7. Criterion 4 -- BLAST RADIUS, enumerated

Method: parse every Workflow run record (`~/.claude/projects/<slug>/*/workflows/wf_*.json`,
**583 records**), recover `args` (a real object on 31 records, a JSON string on
409 -- both parsed), and cross-check against what the agent ACTUALLY received by
reading the first user message of each agent transcript (**507 prompts
inspected**). The receipt, not the caller's belief, is the evidence.

**22 production spawns received a coerced field.** All are `qa-verdict`; all are
`evidence`; 6 of them also lost `extra`.

| Launched (UTC) | Run | Step | Verdict |
|---|---|---|---|
| 2026-08-08T07:34:47Z | `wf_46e96d67-b24` | 85.5 | CONDITIONAL |
| 2026-08-08T08:18:32Z | `wf_7e809394-ae8` | 85.5 | *(rail drop -- no verdict)* |
| 2026-08-08T08:43:03Z | `wf_4c70d707-88e` | 85.5 | *(rail drop)* |
| 2026-08-08T08:47:41Z | `wf_faf8bbd4-4af` | 85.5 | **PASS** |
| 2026-08-11T06:26:47Z | `wf_8a3969ee-ae0` | 86.25 | **PASS** |
| 2026-08-11T06:27:10Z | `wf_97a608dd-2a4` | 86.34 | *(rail drop)* |
| 2026-08-11T06:37:58Z | `wf_d4e2e794-567` | 86.29 | *(rail drop)* |
| 2026-08-11T06:40:16Z | `wf_9d7e0010-66f` | 86.34 | **PASS** |
| 2026-08-11T06:59:18Z | `wf_2675058b-ab3` | 86.29 | CONDITIONAL |
| 2026-08-11T07:17:17Z | `wf_2881574d-de2` | 86.38 | *(rail drop)* |
| 2026-08-11T07:17:44Z | `wf_fdc81179-861` | 86.29 | CONDITIONAL |
| 2026-08-11T07:23:27Z | `wf_982cd319-493` | 86.21 | CONDITIONAL |
| 2026-08-11T07:38:48Z | `wf_13a30a9d-33d` | 86.38 | *(rail drop)* |
| 2026-08-11T07:39:09Z | `wf_e66ad533-e61` | 86.21 | CONDITIONAL |
| 2026-08-11T07:59:36Z | `wf_468907a8-b13` | 86.38 | FAIL |
| 2026-08-11T08:15:41Z | `wf_aa7f8c4d-8bf` | 86.38 | *(rail drop)* |
| 2026-08-15T13:16:09Z | `wf_8c3730a1-32e` | 86.74 | CONDITIONAL |
| 2026-08-15T13:45:06Z | `wf_5f5ce4b6-266` | 86.85 | FAIL |
| 2026-08-15T14:01:32Z | `wf_879d28f2-9fc` | 86.85 | FAIL |
| 2026-08-15T14:19:55Z | `wf_b12cf244-d30` | 86.85 | FAIL |
| **2026-08-15T19:13:58Z** | **`wf_b1747d75-eec`** | **86.86** | **PASS** |
| 2026-08-15T19:42:44Z | `wf_769e1502-fd8` | 86.85 | CONDITIONAL |

Nine step-ids: **85.5, 86.21, 86.25, 86.29, 86.34, 86.38, 86.74, 86.85, 86.86.**
Every one of the 22 dispatched a script blob containing the defective
concatenation (checked per record from the embedded `script` field, not assumed).
Provenance: `git log -S` returns a single commit, `ccddeff4` (phase-71.1) -- the
concatenation has been there since the rail became first-class.

### Disposition, per verdict class

- **11 non-PASS verdicts (7 CONDITIONAL + 4 FAIL) and 7 rail drops.** A non-PASS
  reached under *less* evidence than intended cannot have been made *more*
  lenient by the loss. The direction of harm is one-way here, so these need no
  re-grade. Stated as a bound, not a clearance: they may have been non-PASS for
  reasons a fuller evidence set would have changed, but none of them ADMITTED
  work.

  *(CORRECTED, cycle 2. This sentence read "13 non-PASS verdicts (CONDITIONAL/FAIL)
  and 6 rail drops" -- 13+6 = 19, which EXCEEDS the 18 non-PASS-or-dropped rows in
  the table directly above it. Found by the cycle-1 Q/A and reproduced by me before
  correcting. Origin: six rows read `*(rail drop)*` and one reads
  `*(rail drop -- no verdict)*`, so a literal count misses the seventh; I then
  asserted the split instead of counting it. The table was always right. The
  figures are now COUNTED from the table itself -- 22 rows = 4 PASS + 7 drops +
  7 CONDITIONAL + 4 FAIL -- and the Q/A independently re-derived the same split
  from the run records. The disposition is unaffected: both sub-buckets are the
  "no re-grade needed" class, and the 4 PASSes were enumerated correctly.)*
- **4 PASS verdicts -- 85.5, 86.25, 86.34, 86.86 -- rested on a reconstructed
  evidence set.** This is the direction that matters, and it is stated plainly
  rather than reasoned away.

### 86.86, resolved explicitly (the named candidate)

**86.86's PASS WAS graded on a reconstructed evidence set.** Its evaluator
received `[object Object]` for both `evidence` and `extra`, and rebuilt the set
from git + `handoff/current` + `.claude/masterplan.json`.

A supporting measurement, offered as context and **not** as a substitute for the
re-grade: that evaluator's transcript independently references **all 10** items
the lost evidence object named -- every handoff file, every changed file, every
re-runnable check, and the `subject_sha256` string itself -- across 59 tool-use
blocks. Its reconstruction was, item for item, at least the intended set.

That is not enough to let the PASS stand, so **86.86 was re-graded** by a fresh
Q/A on the fixed rail, with the evidence actually delivered (§6 is the receipt).
Not verdict-shopping: the evidence DELIVERY measurably changed, which is the
documented cycle-2 condition -- and the re-grading Q/A **verified that claim
itself**, reading prompt lines 61 and 63 out of the prior run's own transcript
rather than accepting Main's word for it.

**RE-GRADE RESULT -- run `wf_a09930e2-3d7`, verdict `PASS`, `ok: true`,
`violated_criteria: []`, 27 checks run, 851 s, 237,098 tokens.** All nine
criteria MET on an independent re-derivation. The verdict is transcribed VERBATIM
into `handoff/current/evaluator_critique_86.86.md` under a `RE-GRADE` heading.
**86.86's PASS is CONFIRMED and the step stays closed.**

Three things the re-grade settled that the original evaluation had left open, all
of them tightening rather than loosening:

- It pinned the subject by sha256 (`5b714a9e...`, equal to the spawn-prompt
  value AND to the blob at `e4f2e844`) and confirmed `git diff e4f2e844..HEAD`
  over the four changed files is EMPTY -- so it graded the same artifact.
- It **ruled on the two findings Main flagged against itself**: N1 (the
  caller-side pre-mangle) is REPRODUCED but falls OUTSIDE 86.86's criteria and
  belongs to 86.88, and criterion 2's "exactly one" IS met -- verified by an AST
  walk showing no subscript write to `risk_dict` anywhere in the module.
- It ran its own novel mutation idiom the author never used (a falsy-filtering
  comprehension upstream of the resolver) -- KILLED -- plus a fixture-side
  mutation making ABSENT unexpressible, also killed.

**Harness-compliance item 4 (log-last) was disclosed, not charged:** 86.86 is
already in `harness_log` and already `done`, because this is a POST-CLOSE
re-grade rather than an in-flight EVALUATE. The original cycle respected the
order (the prior Q/A observed `pending` and no log row at its spawn time).

### The other three PASSes

85.5, 86.25 and 86.34 are **queued, not silently accepted** -- see §9. All three
are long closed and their subjects are unrelated to this rail; re-grading them
now would evaluate a tree that has moved many commits since. The honest
statement is that their PASS rests on an evidence set the evaluator rebuilt.

### Two limits of this census, stated rather than laundered

1. **It is a floor.** It counts transcripts still on disk under this project's
   session directories; pruned sessions are invisible, and the 507-prompt
   denominator is subject to the same loss.
2. **It cannot see the array variant at all.** An array-shaped field coerces to
   `a,b`, leaving no marker. The args-level scan (which keys on the VALUE's type,
   not on the marker) found no array-shaped `evidence`/`extra`, so this is
   believed to be zero -- but the two methods have different blind spots and only
   the args-level one could have detected it.

---

## 8. Criterion 3 -- research-gate.js, stated both ways

- **Vulnerable by construction, shown by execution** (not by reading the source):
  driven pre-fix with an object-shaped `topic`/`internal_scope`, it produced
  `OBJECTIVE: [object Object]` and `INTERNAL SCOPE: [object Object]`. Section
  `[1]` of the checker regenerates this from git.
- **Never triggered in practice**: 0 of 75 spawns carrying `OBJECTIVE:` and 0 of
  72 carrying `INTERNAL SCOPE:` show the marker, because every caller has passed
  strings.
- **Fixed anyway**, with the byte-identical block, because "no caller has done it
  yet" is not a guard.

---

## 9. Discovered along the way -- queued, not swept in

1. **`verify_workflow_args_boundary.mjs` has been RED since phase-86.37**, and it
   is not my change.

   *(CORRECTED, cycle 3. This paragraph originally justified the claim with
   `git worktree add --detach <path> HEAD`. That instrument does not establish
   it: **HEAD already CONTAINS `a21a5889`**, so a worktree at HEAD excludes only
   UNCOMMITTED edits and cannot exclude this step. The cycle-2 Q/A caught the
   reasoning even though the conclusion was right -- a conclusion that is correct
   for a reason that does not establish it is still a finding. Replaced with
   evidence that does establish it, re-measured by me:)*

   ```
   $ git log -S'carries NO brief_status marker' --format='%h %ad %s' --date=short \
       -- .claude/workflows/research-gate.js
   d3bb1dfb 2026-08-10 phase-86.37: a dropped research gate no longer destroys the run

   $ ls -la handoff/current/research_brief_86.17.md   # the fixture it asserts against
   ... 9 aug. 17:24 ...
   $ grep -c brief_status handoff/current/research_brief_86.17.md
   0

   $ git diff a21a5889 98c5b6ab -- .claude/workflows/research-gate.js | grep -c enforceGate
   0
   ```

   The failing rule entered on **2026-08-10** at `d3bb1dfb`; the fixture it is
   asserted against is a **2026-08-09** brief carrying **zero** `brief_status`
   markers, so it predates the requirement by a day. My cycle-2 diff touches
   `enforceGate` **zero** times, and the single occurrence in the cycle-1 diff is
   a hunk HEADER (`@@ ... function enforceGate`), i.e. context, not a change. A
   checker red for an unrelated reason is a dead gate -- filed as **86.92**.
2. **`research-gate.js` silently ignored a `questions` key on 11 runs** (phase-82
   era). Now warned via `log()`. Deliberately log-only and NOT added to the
   returned object: phase-86.78 forbids caller-authored fields appearing as
   siblings of the judge's own output, and that invariant is load-bearing.
3. **`harness-self-audit.js:68`** (`'AUDIT THIS DIMENSION: ' + d.focus`, with
   `dimensions` taken from `args`) has the same shape. Not a Layer-3 gate, no
   affected history; untouched here.
4. **`.claude/workflows/qa-verdict.js.export.mjs`** -- a gitignored generated
   artifact from phase-86.81 with no remaining referrer. Noted, untouched.
5. **The three other affected PASS verdicts** (85.5, 86.25, 86.34).

---

## 10. Criterion 7 -- verdict semantics UNCHANGED

Nothing in this change can turn a non-PASS into a PASS. The diff touches only how
caller-supplied fields are rendered INTO the prompt; it does not touch
`VERDICT_SCHEMA`, `enforceEscalation`, `enforceGate`, the no-auto-PASS clause, the
blind-run early return, or `.claude/agents/qa.md`. The only new control-flow
outcome is a **throw before any agent is spawned**, which produces no verdict at
all -- and section `[3]` asserts `spawns.length === 0` on all 12 unrenderable
cases for exactly that reason.

## 11. Verification commands run

```
$ bash -c 'source .venv/bin/activate && node --check .claude/workflows/qa-verdict.js && echo parses'
parses                                                              # exit 0

$ node --check .claude/workflows/research-gate.js && echo parses
parses                                                              # exit 0

$ node scripts/qa/verify_prompt_render_86_90.mjs
ALL GREEN: 95 passed, 0 failed                                      # exit 0   (re-run 2026-08-17, cycle 5; the count first reached 95 in cycle 4 -- the old "cycle 3" label under a cycle-4 figure was the escalation's stale-marker finding, corrected by replacement)

$ node scripts/qa/verify_research_gate_workflow.mjs
ALL GREEN: 124 passed, 0 failed                                     # exit 0

$ node scripts/qa/verify_escalation_86_78.mjs
  ALL CHECKS PASS                      # exit 0

$ node scripts/qa/verify_rail_retry.mjs
ALL GREEN: 38 passed, 0 failed                                      # exit 0

$ node scripts/qa/verify_workflow_args_boundary.mjs
  - [4] drop-blind-violation: KILLED (a blind run would pass without it)        # PRE-EXISTING -- identical at pristine HEAD, see §9.1
```


---

# Follow-up -- cycle 2 (2026-08-16)

Cycle-1 verdict was **CONDITIONAL** with three WARN findings. All three accepted
and fixed; the evidence changed, so a FRESH Q/A is spawned on the changed
evidence (the documented cycle-2 flow), not a re-ask on the same evidence.

| # | Finding | What changed |
|---|---|---|
| **D1** | `experiment_results_86.90.md:250` said "13 non-PASS verdicts and 6 rail drops"; the table gives 11 and 7 (13+6=19 > 18 rows) | The sentence is **REPLACED**, not annotated, with figures COUNTED from the table (`4 PASS + 7 drops + 7 CONDITIONAL + 4 FAIL = 22`) and the origin of the miscount recorded. I reproduced the Q/A's count myself before correcting |
| **D2** | Four follow-ups asserted "queued" with **no masterplan step in existence** | **Filed as real steps: `86.92`, `86.93`, `86.94`, `86.95`.** The masterplan's newest commit was `c627a810`, which PREDATED the work commit -- so "queued" was prose describing an intention. This is the standing project rule and I broke it while citing it |
| **E** | The in-code absolute "THE RULE IS LOSSLESS-OR-THROW" over-stated the measured guarantee; five constructions rendered lossily without throwing | The **walk was widened** (`getOwnPropertyDescriptors`; accessors refused outright, which is also the getter-TOCTOU fix; own `toJSON` refused at any enumerability; array non-index own properties refused) AND the **claim was narrowed** to state its bound. All five are now `[3]` cases; 2 controls prove the walk did not become a blanket refusal; mutation cell **M5** requires A2 to go red if the walk narrows again |

Guard after the fix: **`ALL GREEN: 95 passed, 0 failed`** (53 at cycle 1, 78 at
cycle 2, 83 at cycle 3), 6 mutation cells, each with its own control observed
GREEN first.

**A fixture bug I repeated inside the same file, disclosed rather than quietly
fixed.** The new `[3] CONTROL` cases asserted `spawns.length === 1`, which fails
on `research-gate.js` because it spawns two agents -- the *identical* mistake
section `[0]` already carries a comment about, made one section later in the same
edit. Corrected to `>= 1`, with the reason written into the file.

D2 is the one worth keeping. The other two were caught by an evaluator doing its
job; D2 was a rule I quote in my own memory (`feedback_queue_discovered_defects_in_masterplan`)
and violated in the same document that quoted it. "Queued" in prose reads as done
to the next reader, and loses the finding.


---

# Follow-up -- cycle 3 (2026-08-16)

Cycle-2 verdict was **CONDITIONAL** (run `wf_8f83d0d5-0c9`) with four WARN
findings and one NOTE. All accepted; every fix is recorded IN PLACE above rather
than appended here, because a correction must REPLACE rather than accompany.

| # | Finding | Fix, and where it lives |
|---|---|---|
| 1 | **M3 was an ARTIFACT-KILL** -- a SyntaxError mutant with the injected return placed after the throw, scored KILLED by the harness `catch`. "5 cells, all KILLED" did not reproduce | M3 now injects `return '(unrenderable)'` immediately after `if (violation) {` -- valid and REACHABLE. The harness scores **three** outcomes and `UNSCORABLE: the mutant did not build` now FAILS. **Every cell is CONTROLLED first**: its own `expect()` must be false on the unmutated source. Section 5's matrix note |
| 2 | Stale figures inside verbatim-labelled blocks (`53` vs 78; "14 unrenderable" vs 12) | **Every capture REGENERATED from a live run** by a script, not hand-edited; `14 -> 12` corrected. Guard is now **83** (53 at cycle 1, 78 at cycle 2, 83 with the per-cell controls) |
| 3 | **86.94's criterion 1 was un-meetable as filed** -- it pinned 621/592/706, which measured 560/712 the same day | Criterion 1 rewritten to demand the drift be shown **by execution with no pinned figures**. Recorded in the step's `notes` as a repair of a defective FILING (no cycle has run against the old text), not an amendment of in-flight immutable criteria |
| 4 | The pre-existing-RED claim used the **wrong instrument** -- a worktree at HEAD cannot exclude a commit HEAD contains | Replaced in section 9.1 **and** in 86.92's `audit_basis` with evidence that does establish it, re-measured by me: `git log -S` dates the rule to `d3bb1dfb` 2026-08-10; the fixture is a 2026-08-09 brief with `grep -c brief_status` = **0**; the cycle-2 diff touches `enforceGate` **0** times and the cycle-1 "occurrence" is a hunk header |
| NOTE | A **sixth hole**: a Proxy presenting a data descriptor with a non-deterministic `get` | Recorded in section 4 with its reachability bound. Left unfixed deliberately -- a Proxy cannot arrive through JSON-derived args, and adding a defence would enlarge the claim again, which is the failure this finding is about |

Finding 3 is the one worth keeping. The step I filed **to prevent** criteria that
name unreproducible numbers itself named three unreproducible numbers. Knowing a
trap and writing it down in the same hour is not the same as not falling into it.

---

# Follow-up -- cycle 4 (2026-08-16)

Cycle-3 verdict was **CONDITIONAL** (run `wf_7854f219-eaf`) with two WARN
findings. Sequence is now `[C, C, C]`, so the escalation the caller computes
carries `would_auto_fail: true` -- the next verdict is PASS or FAIL, not another
CONDITIONAL. Both findings are closed.

| # | Finding | Fix |
|---|---|---|
| **W1** | **Criterion 5 was NOT closed for the `criteria` CONTAINER.** `Array.isArray(a.criteria) ? a.criteria : []` sat UPSTREAM of the render boundary, so a present-but-wrong-shaped `criteria` was DISCARDED and the `(none passed in args)` placeholder substituted -- no throw, no log, agent spawned anyway, on **the field the evaluator grades against**. Measured by driving the real script: ARRAY -> criterion text present; STRING / OBJECT / numeric-key OBJECT -> text ABSENT, placeholder substituted | New `requireArgArray()` in the shared block: **absent stays legal** (still means "read them from the masterplan"), **present-but-wrong-shaped THROWS** naming the field. New section `[3b]` with 2 controls + 8 assertions across four wrong shapes, and mutation cell **container-guard-reverted-to-silent-discard** which KILLS |
| **W2** | `experiment_results:15` still read "NEW, **78** assertions" while four other lines said 83 -- **inside the cycle-3 row claiming every capture had been regenerated** | Every count is now **DERIVED, not typed**: assertion counts from a live run, mutation-cell counts from the run's own `: KILLED` lines. The regeneration script ends with an audit that fails if any stale count survives (`STALE COUNTS REMAINING: none`) |

**W1 is the one that matters**, and it is the sharpest finding of the four
cycles. Every other render hole found in this step (Map, `toJSON`, getter,
non-enumerable, Proxy) required an exotic construction that **cannot arrive
through JSON-derived args**. This one needs only a caller passing a string. My
own diff shows I edited that exact expression to route the ELEMENTS through
`renderArgField` and left the CONTAINER guard one line above untouched -- the
"guards stop one seam short" shape, on the seam I was building.

---

## Closure edit landed (2026-08-17): the container-bound comment, mirrored per the byte-identical design

The deferred closure edit is in: the sparse-array bound is stated AT the array
branch of the lossless walk (qa-verdict.js), and -- because the walk lives in
the delimited byte-identical phase-86.90 block that the runtime's no-import
constraint forces into BOTH workflow files -- the block was mirrored verbatim
into research-gate.js (10,601 bytes; checker section [6] re-green). The same
edit pass landed 86.96's classifyArgs comment (same block, same mirror).
Comment-only verified: `git diff` on both files shows insertions only, no
executable line changed. Checker family after the edit: prompt-render 113/113,
research-gate-workflow 124/124, args-boundary 96/96, `node --check` clean on
both scripts.
