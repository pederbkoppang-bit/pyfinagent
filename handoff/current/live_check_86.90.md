# live_check -- phase-86.90

Evidence artifact for the `verification.live_check` gate. Everything below is
verbatim command output or verbatim transcript text, captured 2026-08-16.

---

## 1. The reproduced `[object Object]` prompt text

### 1a. Historical receipt -- the REAL 86.86 spawn (`wf_b1747d75-eec`)

Read from the agent's own transcript. This is what the evaluator received, not
what the caller believed it sent.

```
$ python - <<'EOF'   # extract the first user message of the agent transcript
...
--- line 61 ---
  59|   9. no gate is loosened, no flag is promoted, and no .env is written in order to obtain a green result
  60|
  61| EVIDENCE / FILES TO READ: [object Object]
  62|
  63| ADDITIONAL CONTEXT: [object Object]
--- line 63 ---
  61| EVIDENCE / FILES TO READ: [object Object]
  62|
  63| ADDITIONAL CONTEXT: [object Object]
  64|
  65| Return the verdict AS YOUR RETURN VALUE using the schema. This captured object IS the deliverable -- Main
```

### 1b. LIVE minimal spawn, PRE-FIX -- Workflow run `wf_4588d8a7-e70`

One agent (`effort: low`, 1 tool use, 2,273 ms) whose only task was to echo back
the line it received. Return value, verbatim from the task notification:

```json
{"runtime_typeof_evidence":"object",
 "runtime_is_array":false,
 "runtime_keys":["handoff","changed_files","subject_sha256"],
 "runtime_json":"{\"handoff\":[\"handoff/current/contract_86.90.md\",\"handoff/current/experiment_results_86.90.md\"],\"changed_files\":[\".claude/workflows/qa-verdict.js\"],\"subject_sha256\":\"deadbeef\"}",
 "script_concat_result":"EVIDENCE / FILES TO READ: [object Object]",
 "agent_received":{"received_line":"[object Object]","is_literal_object_object":true}}
```

### 1c. Both shipped scripts, driven pre-fix

```
$ node probe_render.mjs        # slices each script at its driver boundary, injects object-shaped args
### qa-verdict.js — lines matching EVIDENCE/ADDITIONAL CONTEXT:
   "EVIDENCE / FILES TO READ: [object Object]"
   "ADDITIONAL CONTEXT: [object Object]"
   contains "[object Object]": true
### research-gate.js — lines matching OBJECTIVE/INTERNAL SCOPE:
   "OBJECTIVE: [object Object]"
   "INTERNAL SCOPE: [object Object]"
   contains "[object Object]": true
```

---

## 2. The LOCALISED layer

`wf_4588d8a7-e70` measured all three candidate layers in one run, from inside the
real Workflow runtime:

| Layer | Measurement | Verdict |
|---|---|---|
| args marshalling | `runtime_typeof_evidence: "object"`, `runtime_is_array: false`, `runtime_keys` intact, `runtime_json` round-trips | **INNOCENT** |
| prompt template | `script_concat_result: "EVIDENCE / FILES TO READ: [object Object]"` | **GUILTY** |
| `agent()` / transport | `agent_received.received_line: "[object Object]"` -- delivered faithfully what the template built | **INNOCENT** |

Official Anthropic Workflow doc, read in full by the research gate: *"Claude
passes the list as structured data, so the script can call array and object
methods on `args` directly without parsing it first."*

---

## 3. POST-FIX, same probe

```
$ node probe_render.mjs
### qa-verdict.js — lines matching EVIDENCE/ADDITIONAL CONTEXT:
   "EVIDENCE / FILES TO READ: "
   "ADDITIONAL CONTEXT: "
   contains "[object Object]": false
### research-gate.js — lines matching OBJECTIVE/INTERNAL SCOPE:
   "OBJECTIVE: "
   "INTERNAL SCOPE: "
   contains "[object Object]": false
```

```
$ node probe2.mjs
EVIDENCE / FILES TO READ:
```json
{
  "handoff": [
    "a.md",
    "b.md"
  ],
  "changed_files": [
    "x.py"
  ],
  "subject_sha256": "5b71"
}
```

ADDITIONAL CONTEXT: plain string extra          <-- a STRING field is untouched

--- ARRAY shape (the quiet variant) ---
EVIDENCE / FILES TO READ:
```json
[
  "a.md",
  "b.md"
]
```

contains bare "a.md,b.md": false
```

---

## 4. LIVE on the fixed rail -- run `wf_a09930e2-3d7`

The 86.86 re-grade is the first Q/A spawn on the fixed rail, dispatched with the
same object-shaped `evidence`/`extra` shape that produced `[object Object]` on
2026-08-15. From that agent's own transcript:

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
  "changed_files": [
    "backend/services/autonomous_loop.py",
    "backend/tests/test_phase_66_2_risk_judge_shape.py",
    "scripts/qa/verify_lite_risk_seam_86_86.py",
    "scripts/qa/mutation_matrix_86_86.py"
  ],
  "rerunnable_checks": [
    "python scripts/qa/verify_lite_risk_seam_86_86.py",
    "python scripts/qa/mutation_matrix_86_86.py",
    "python -m pytest backend/tests/test_phase_66_2_risk_judge_shape.py -q"
  ],
  "subject_sha256_at_the_86_86_spawn": "5b714a9e5f43753c...
```

**A naive grep gives the WRONG answer on this prompt, so the discriminating
measurement is recorded instead:**

```
header lines rendered: ["EVIDENCE / FILES TO READ: ", "ADDITIONAL CONTEXT: "]
lines that ARE a coerced field: [] -> count 0
every line MENTIONING the literal (should all be my own prose):
      "why_this_is_a_RE_GRADE_and_not_verdict_shopping": "The evidence DELIVERY changed, measurably. The original 86.86 spawn (wf_b1747d75-eec, ...
```

`'[object Object]' in prompt` is **True** here -- because the `extra` object's own
prose explains the defect and contains the phrase. The string survived *inside*
the JSON block, which is itself proof the render is lossless. The count that
discriminates is **lines that ARE a coerced field: 0**.

---

## 5. research-gate.js -- the result, stated both ways

| Question | Answer | How established |
|---|---|---|
| Does it have the same defect? | **YES, by construction** at `:236` (`'OBJECTIVE: ' + topic`) and `:245` (`'INTERNAL SCOPE: ' + internalScope`) | **By EXECUTION** -- §1c above, and re-runnably as section `[1]` of the checker |
| Was it ever triggered? | **NO** -- 0 of 75 spawns carrying `OBJECTIVE: `, 0 of 72 carrying `INTERNAL SCOPE: ` | receipt scan over 507 agent prompts |
| Fixed? | **YES**, byte-identical block | section `[6]` asserts the two copies match |

---

## 6. BLAST RADIUS -- 22 production spawns, 9 step-ids

Method: 583 run records parsed (`args` recovered from 31 real objects + 409 JSON
strings); cross-checked against 507 agent prompts read from transcripts. The
receipt governs, not the caller's belief.

```
2026-08-08T07:34:47Z  wf_46e96d67-b24  85.5   CONDITIONAL
2026-08-08T08:18:32Z  wf_7e809394-ae8  85.5   (rail drop)
2026-08-08T08:43:03Z  wf_4c70d707-88e  85.5   (rail drop)
2026-08-08T08:47:41Z  wf_faf8bbd4-4af  85.5   PASS
2026-08-11T06:26:47Z  wf_8a3969ee-ae0  86.25  PASS
2026-08-11T06:27:10Z  wf_97a608dd-2a4  86.34  (rail drop)
2026-08-11T06:37:58Z  wf_d4e2e794-567  86.29  (rail drop)
2026-08-11T06:40:16Z  wf_9d7e0010-66f  86.34  PASS
2026-08-11T06:59:18Z  wf_2675058b-ab3  86.29  CONDITIONAL
2026-08-11T07:17:17Z  wf_2881574d-de2  86.38  (rail drop)
2026-08-11T07:17:44Z  wf_fdc81179-861  86.29  CONDITIONAL
2026-08-11T07:23:27Z  wf_982cd319-493  86.21  CONDITIONAL
2026-08-11T07:38:48Z  wf_13a30a9d-33d  86.38  (rail drop)
2026-08-11T07:39:09Z  wf_e66ad533-e61  86.21  CONDITIONAL
2026-08-11T07:59:36Z  wf_468907a8-b13  86.38  FAIL
2026-08-11T08:15:41Z  wf_aa7f8c4d-8bf  86.38  (rail drop)
2026-08-15T13:16:09Z  wf_8c3730a1-32e  86.74  CONDITIONAL
2026-08-15T13:45:06Z  wf_5f5ce4b6-266  86.85  FAIL
2026-08-15T14:01:32Z  wf_879d28f2-9fc  86.85  FAIL
2026-08-15T14:19:55Z  wf_b12cf244-d30  86.85  FAIL
2026-08-15T19:13:58Z  wf_b1747d75-eec  86.86  PASS      <-- the named candidate
2026-08-15T19:42:44Z  wf_769e1502-fd8  86.85  CONDITIONAL
```

Every one of the 22 dispatched a script blob containing the defective
concatenation (checked per record from the embedded `script` field). Provenance:
`git log -S` returns one commit, `ccddeff4` (phase-71.1).

**Limits, stated rather than laundered:** this is a FLOOR (pruned sessions are
invisible), and it is structurally blind to an array-shaped field, which coerces
to `a,b` with no marker at all.

### Disposition for 86.86

**86.86's PASS WAS graded on a reconstructed evidence set** -- its evaluator
received `[object Object]` for both fields and rebuilt the set from
git + `handoff/current` + `.claude/masterplan.json`.

Supporting context, NOT a substitute: that evaluator's transcript independently
references **all 10** items the lost object named -- every handoff file, every
changed file, every re-runnable check, and the `subject_sha256` string itself --
across 59 tool-use blocks.

**Re-graded** by a fresh Q/A on the fixed rail, run `wf_a09930e2-3d7`, with the
evidence actually delivered (§4 is that receipt).

```
verdict: PASS | ok: True | verdict_unmodified: True
escalation: {"sequence_supplied": ["PASS"], "sequence_status": "ok",
             "consecutive_conditionals": 0, "would_auto_fail": false,
             "attempt_number": 2, "budget_exhausted": false, "max_attempts": 5,
             "burden_on": "the party departing from the computed escalation",
             "override": null, "judge_was_told_consequence": false}
violated_criteria: []   checks_run: 27   tokens: 237,098   duration: 851 s
```

**86.86's PASS is CONFIRMED; the step stays closed.** Transcribed VERBATIM into
`handoff/current/evaluator_critique_86.86.md` under a `RE-GRADE` heading.

The re-grading Q/A independently verified the re-grade's own premise rather than
accepting it -- from its notes: *"I checked the prior run's own agent transcript
(subagents/workflows/wf_b1747d75-eec/agent-abeb0c1a9dca29d03.jsonl) and it
carries verbatim, at prompt line 61 `EVIDENCE / FILES TO READ: [object Object]`
and at line 63 `ADDITIONAL CONTEXT: [object Object]` -- exactly the lines
claimed."* That is a second, independent confirmation of this step's core finding,
produced by an agent with no stake in it.

The other three affected PASSes -- **85.5, 86.25, 86.34** -- also rest on
reconstructed evidence sets. Queued, not silently accepted.

---

## 7. Guard runs

```
$ node scripts/qa/verify_prompt_render_86_90.mjs
...
[5] MUTATION -- reverting each guard must turn the section above RED (criterion 6)

  ok   [5] restore-plus-concat: KILLED (section [2] must go RED when the raw field is concatenated again)
  ok   [5] restore-plus-concat (research-gate): KILLED (the research-gate copy must be doing the work too)
  ok   [5] placeholder-instead-of-throw: KILLED (section [3] must go RED when the throw becomes a silent placeholder)
  ok   [5] identity-arg-accepts-objects: KILLED (an object step id must not be allowed to reach a filename)

[6] DUPLICATE INTEGRITY -- the Workflow runtime forbids imports, so the block is duplicated

  ok   [6] qa-verdict.js: the block is present and delimited
  ok   [6] research-gate.js: the block is present and delimited
  ok   [6] the two copies are BYTE-IDENTICAL

ALL GREEN: 83 passed, 0 failed   (REGENERATED cycle 3)
```

Immutable command:

```
$ bash -c 'source .venv/bin/activate && node --check .claude/workflows/qa-verdict.js && echo parses'
parses
rc=0
```

---

## 8. Pre-existing red, proven NOT caused by this change

```
$ node scripts/qa/verify_workflow_args_boundary.mjs        # working tree, WITH the 86.90 change
FAILED: 84 passed, 3 failed

$ git worktree add -q --detach <scratch>/wt-head-1 HEAD    # pristine HEAD, WITHOUT it
$ (cd <scratch>/wt-head-1 && node scripts/qa/verify_workflow_args_boundary.mjs)
FAILED: 84 passed, 3 failed
```

Identical counts and identical failing case names. The cause is unrelated to this
step: section `[3]` asserts a "healthy run" against
`handoff/current/research_brief_86.17.md`, a 2026-08-09 brief that predates the
born-inert `brief_status` marker phase-86.37 later made mandatory. Queued as a
discovered defect; the worktree was removed after the measurement
(`git worktree remove --force`).
