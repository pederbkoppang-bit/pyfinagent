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
| `scripts/qa/verify_prompt_render_86_90.mjs` | NEW, 53 assertions -- behavioural driver + reproduce-from-git + unrenderable-throws + 4 mutation cells + duplicate-integrity |
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

Also carried: an **array** field coerces to `a,b` with **no `[object Object]`
marker at all**. Every census in this document keyed on that marker is therefore
a **floor**, never a total, and the checker asserts the array case separately.

---

## 5. Criterion 6 -- the regression guard, control GREEN first

`node scripts/qa/verify_prompt_render_86_90.mjs` -> **`ALL GREEN: 53 passed, 0 failed`**

It DRIVES the real shipped scripts with the runtime primitives stubbed and reads
the prompt actually handed to `agent()`. A source scan for `renderArgField(`
would pass on a file that never calls it on the path that matters.

| Section | What it proves |
|---|---|
| `[0]` CONTROL | a usable launch really does spawn and produce a prompt carrying the step id, and a STRING field is passed through unchanged -- without this, every "does not contain" assertion below could pass vacuously |
| `[1]` REPRODUCE | the pre-fix blob at `75831f4c` still yields `[object Object]`, on BOTH scripts |
| `[2]` FIXED | object AND array shapes render as JSON; every key and value reaches the prompt; no comma-joined collapse |
| `[3]` UNRENDERABLE | 7 cases x 2 scripts THROW naming the field AND spawn nothing |
| `[4]` research-gate | criterion 3, by execution |
| `[5]` MUTATION | 4 cells, anchor uniqueness checked first |
| `[6]` DUPLICATE INTEGRITY | the two copies of the block are byte-identical |

### Mutation matrix (4 cells, all KILLED)

| Cell | Mutation | Result |
|---|---|---|
| M1 | `qa-verdict.js`: restore `'EVIDENCE / FILES TO READ: ' + a.evidence` | **KILLED** -- `[2]` goes red |
| M2 | `research-gate.js`: restore `'OBJECTIVE: ' + a.topic` | **KILLED** -- the second copy is doing work too |
| M3 | replace the lossless-violation `throw` with `return '(unrenderable)'` | **KILLED** -- `[3]` goes red |
| M4 | `renderIdentityArg` accepts objects via `String(value)` | **KILLED** -- an object step id reaches a filename |

**This matrix licenses exactly one claim: these four mutations were killed.** It
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

- **13 non-PASS verdicts (CONDITIONAL/FAIL) and 6 rail drops.** A non-PASS
  reached under *less* evidence than intended cannot have been made *more*
  lenient by the loss. The direction of harm is one-way here, so these need no
  re-grade. Stated as a bound, not a clearance: they may have been non-PASS for
  reasons a fuller evidence set would have changed, but none of them ADMITTED
  work.
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

That is not enough to let the PASS stand, so **86.86 has been re-graded** by a
fresh Q/A on the fixed rail (`wf_a09930e2-3d7`), with the evidence actually
delivered (§6 shows the receipt). This is not verdict-shopping: the evidence
DELIVERY measurably changed, which is the documented cycle-2 condition. The
re-grade's verdict governs and is transcribed verbatim into
`evaluator_critique_86.90.md`; if it is not a PASS, 86.86 is reopened.

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
   is not my change: pristine `HEAD` in a scratch worktree reports the identical
   `84 passed, 3 failed`. The cause is that its section `[3]` asserts a "healthy
   run" against `handoff/current/research_brief_86.17.md`, a 2026-08-09 brief
   that predates the born-inert `brief_status` marker phase-86.37 made mandatory.
   A checker red for an unrelated reason is a dead gate.
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
all -- and section `[3]` asserts `spawns.length === 0` on all 14 unrenderable
cases for exactly that reason.

## 11. Verification commands run

```
$ bash -c 'source .venv/bin/activate && node --check .claude/workflows/qa-verdict.js && echo parses'
parses                                                              # exit 0

$ node --check .claude/workflows/research-gate.js && echo parses
parses                                                              # exit 0

$ node scripts/qa/verify_prompt_render_86_90.mjs
ALL GREEN: 53 passed, 0 failed                                      # exit 0

$ node scripts/qa/verify_research_gate_workflow.mjs
ALL GREEN: 124 passed, 0 failed                                     # exit 0

$ node scripts/qa/verify_escalation_86_78.mjs
checks run : 51   failed : 0   ALL CHECKS PASS                      # exit 0

$ node scripts/qa/verify_rail_retry.mjs
ALL GREEN: 38 passed, 0 failed                                      # exit 0

$ node scripts/qa/verify_workflow_args_boundary.mjs
FAILED: 84 passed, 3 failed        # PRE-EXISTING -- identical at pristine HEAD, see §9.1
```
