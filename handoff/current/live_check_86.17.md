# live_check -- phase-86.17

**Required evidence (immutable, verbatim from `.claude/masterplan.json`):**
"Verbatim output of the new args-boundary checker showing, per script and per
args shape, the resolved stepId and the throw-or-run outcome; plus the verbatim
thrown message for one malformed-args case on each script; plus the verbatim
pre-existing-checker totals proving no regression against the 40/0 baseline."

---

## 1. Per-script, per-shape: resolved stepId and throw-or-run outcome

Verbatim output of `node scripts/qa/verify_workflow_args_boundary.mjs`:

```
phase-86.17 -- Workflow args-boundary verification

[1] REPRODUCE -- pre-fix blob at 178a6a59 (criterion 1)

  research-gate.js:
    plain-object             -> stepId="86.17"
    valid-json-string        -> stepId="86.17"
    malformed-json-string    -> stepId="UNSPECIFIED"
    json-string-raw-newline  -> stepId="UNSPECIFIED"
    array                    -> stepId="UNSPECIFIED"
    scalar-number            -> stepId="UNSPECIFIED"
    absent                   -> stepId="UNSPECIFIED"
    double-encoded-json      -> stepId="UNSPECIFIED"
    empty-string             -> stepId="UNSPECIFIED"
    object-without-step_id   -> stepId="UNSPECIFIED"
  qa-verdict.js:
    plain-object             -> stepId="86.17"
    valid-json-string        -> stepId="86.17"
    malformed-json-string    -> stepId="UNSPECIFIED"
    json-string-raw-newline  -> stepId="UNSPECIFIED"
    array                    -> stepId="UNSPECIFIED"
    scalar-number            -> stepId="UNSPECIFIED"
    absent                   -> stepId="UNSPECIFIED"
    double-encoded-json      -> stepId="UNSPECIFIED"
    empty-string             -> stepId="UNSPECIFIED"
    object-without-step_id   -> stepId="UNSPECIFIED"
  ok   [1] pre-fix code silently resolved stepId=UNSPECIFIED on multiple shapes

[2] FIXED -- current file (criteria 2, 3, 4)

  research-gate.js:
    plain-object             -> stepId="86.17" blind=false
  ok   [2] research-gate.js plain-object: usable args resolve the real step id
  ok   [2] research-gate.js plain-object: never resolves to UNSPECIFIED on a present-args shape
    valid-json-string        -> stepId="86.17" blind=false
  ok   [2] research-gate.js valid-json-string: usable args resolve the real step id
  ok   [2] research-gate.js valid-json-string: never resolves to UNSPECIFIED on a present-args shape
    malformed-json-string    -> THREW: research-gate: args are PRESENT but not parseable as JSON (typeof=string
  ok   [2] research-gate.js malformed-json-string: unusable/incomplete args THROW
  ok   [2] research-gate.js malformed-json-string: the throw NAMES what arrived
  ok   [2] research-gate.js malformed-json-string: never resolves to UNSPECIFIED on a present-args shape
    json-string-raw-newline  -> THREW: research-gate: args are PRESENT but not parseable as JSON (typeof=string
  ok   [2] research-gate.js json-string-raw-newline: unusable/incomplete args THROW
  ok   [2] research-gate.js json-string-raw-newline: the throw NAMES what arrived
  ok   [2] research-gate.js json-string-raw-newline: never resolves to UNSPECIFIED on a present-args shape
    array                    -> THREW: research-gate: args did not reduce to a plain object (typeof=object isAr
  ok   [2] research-gate.js array: unusable/incomplete args THROW
  ok   [2] research-gate.js array: the throw NAMES what arrived
  ok   [2] research-gate.js array: never resolves to UNSPECIFIED on a present-args shape
    scalar-number            -> THREW: research-gate: args did not reduce to a plain object (typeof=number isAr
  ok   [2] research-gate.js scalar-number: unusable/incomplete args THROW
  ok   [2] research-gate.js scalar-number: the throw NAMES what arrived
  ok   [2] research-gate.js scalar-number: never resolves to UNSPECIFIED on a present-args shape
    absent                   -> stepId="UNSPECIFIED" blind=true
  ok   [2] research-gate.js absent: absent args do NOT throw and are marked blind
  ok   [2] research-gate.js absent: never resolves to UNSPECIFIED on a present-args shape
    double-encoded-json      -> THREW: research-gate: args did not reduce to a plain object (typeof=string isAr
  ok   [2] research-gate.js double-encoded-json: unusable/incomplete args THROW
  ok   [2] research-gate.js double-encoded-json: the throw NAMES what arrived
  ok   [2] research-gate.js double-encoded-json: never resolves to UNSPECIFIED on a present-args shape
    empty-string             -> THREW: research-gate: args are PRESENT but an empty/blank string (typeof=string
  ok   [2] research-gate.js empty-string: unusable/incomplete args THROW
  ok   [2] research-gate.js empty-string: the throw NAMES what arrived
  ok   [2] research-gate.js empty-string: never resolves to UNSPECIFIED on a present-args shape
    object-without-step_id   -> THREW: research-gate: args are a plain object with NO step_id (typeof=object is
  ok   [2] research-gate.js object-without-step_id: unusable/incomplete args THROW
  ok   [2] research-gate.js object-without-step_id: the throw NAMES what arrived
  ok   [2] research-gate.js object-without-step_id: never resolves to UNSPECIFIED on a present-args shape
  qa-verdict.js:
    plain-object             -> stepId="86.17" blind=false
  ok   [2] qa-verdict.js plain-object: usable args resolve the real step id
  ok   [2] qa-verdict.js plain-object: never resolves to UNSPECIFIED on a present-args shape
    valid-json-string        -> stepId="86.17" blind=false
  ok   [2] qa-verdict.js valid-json-string: usable args resolve the real step id
  ok   [2] qa-verdict.js valid-json-string: never resolves to UNSPECIFIED on a present-args shape
    malformed-json-string    -> THREW: qa-verdict: args are PRESENT but not parseable as JSON (typeof=string is
  ok   [2] qa-verdict.js malformed-json-string: unusable/incomplete args THROW
  ok   [2] qa-verdict.js malformed-json-string: the throw NAMES what arrived
  ok   [2] qa-verdict.js malformed-json-string: never resolves to UNSPECIFIED on a present-args shape
    json-string-raw-newline  -> THREW: qa-verdict: args are PRESENT but not parseable as JSON (typeof=string is
  ok   [2] qa-verdict.js json-string-raw-newline: unusable/incomplete args THROW
  ok   [2] qa-verdict.js json-string-raw-newline: the throw NAMES what arrived
  ok   [2] qa-verdict.js json-string-raw-newline: never resolves to UNSPECIFIED on a present-args shape
    array                    -> THREW: qa-verdict: args did not reduce to a plain object (typeof=object isArray
  ok   [2] qa-verdict.js array: unusable/incomplete args THROW
  ok   [2] qa-verdict.js array: the throw NAMES what arrived
  ok   [2] qa-verdict.js array: never resolves to UNSPECIFIED on a present-args shape
    scalar-number            -> THREW: qa-verdict: args did not reduce to a plain object (typeof=number isArray
  ok   [2] qa-verdict.js scalar-number: unusable/incomplete args THROW
  ok   [2] qa-verdict.js scalar-number: the throw NAMES what arrived
  ok   [2] qa-verdict.js scalar-number: never resolves to UNSPECIFIED on a present-args shape
    absent                   -> stepId="UNSPECIFIED" blind=true
  ok   [2] qa-verdict.js absent: absent args do NOT throw and are marked blind
  ok   [2] qa-verdict.js absent: never resolves to UNSPECIFIED on a present-args shape
    double-encoded-json      -> THREW: qa-verdict: args did not reduce to a plain object (typeof=string isArray
  ok   [2] qa-verdict.js double-encoded-json: unusable/incomplete args THROW
  ok   [2] qa-verdict.js double-encoded-json: the throw NAMES what arrived
  ok   [2] qa-verdict.js double-encoded-json: never resolves to UNSPECIFIED on a present-args shape
    empty-string             -> THREW: qa-verdict: args are PRESENT but an empty/blank string (typeof=string is
  ok   [2] qa-verdict.js empty-string: unusable/incomplete args THROW
  ok   [2] qa-verdict.js empty-string: the throw NAMES what arrived
  ok   [2] qa-verdict.js empty-string: never resolves to UNSPECIFIED on a present-args shape
    object-without-step_id   -> THREW: qa-verdict: args are a plain object with NO step_id (typeof=object isArr
  ok   [2] qa-verdict.js object-without-step_id: unusable/incomplete args THROW
  ok   [2] qa-verdict.js object-without-step_id: the throw NAMES what arrived
  ok   [2] qa-verdict.js object-without-step_id: never resolves to UNSPECIFIED on a present-args shape

[3] BLIND CANNOT PASS -- enforceGate defence in depth (criterion 4)

  ok   [3] a healthy run with a perfect envelope PASSES
  ok   [3] the SAME perfect envelope CANNOT pass when the run was blind
  ok   [3] the blind refusal is NAMED in violations
  ok   [3] no regression: enforceGate without inputHealth behaves as before

[4] MUTATION -- revert each new guard (criterion 6)

  ok   [4] restore-silent-catch: CONTROL -- the guard owns its diagnosis
  ok   [4] restore-silent-catch: KILLED -- reverting it changes the outcome for malformed-json-string
  ok   [4] drop-post-parse-plain-object-check: CONTROL -- the guard owns its diagnosis
  ok   [4] drop-post-parse-plain-object-check: KILLED -- reverting it changes the outcome for double-encoded-json
  ok   [4] drop-step_id-requirement: CONTROL -- the guard owns its diagnosis
  ok   [4] drop-step_id-requirement: KILLED -- reverting it changes the outcome for object-without-step_id
  ok   [4] qa-restore-silent-catch: CONTROL -- the guard owns its diagnosis
  ok   [4] qa-restore-silent-catch: KILLED -- reverting it changes the outcome for malformed-json-string
  ok   [4] qa-drop-post-parse-plain-object-check: CONTROL -- the guard owns its diagnosis
  ok   [4] qa-drop-post-parse-plain-object-check: KILLED -- reverting it changes the outcome for double-encoded-json
  ok   [4] drop-empty-string-guard: CONTROL -- the guard owns its diagnosis
  ok   [4] drop-empty-string-guard: KILLED -- reverting it changes the outcome for empty-string
  ok   [4] qa-drop-empty-string-guard: CONTROL -- the guard owns its diagnosis
  ok   [4] qa-drop-empty-string-guard: KILLED -- reverting it changes the outcome for empty-string
  ok   [4] qa-drop-step_id-requirement: CONTROL -- the guard owns its diagnosis
  ok   [4] qa-drop-step_id-requirement: KILLED -- reverting it changes the outcome for object-without-step_id
  ok   [4] drop-blind-violation: KILLED (a blind run would pass without it)

[5] FULL DRIVER -- blind runs must spawn NOTHING (criteria 3, 4, 6)

  ok   [5] qa-verdict.js: a blind run spawns ZERO agents
  ok   [5] qa-verdict.js: a blind run cannot pass and is marked dry_run
  ok   [5] qa-verdict.js: the blind run is LOGGED
  ok   [5] qa-verdict.js: no prompt mentioning research_brief_UNSPECIFIED is ever sent
  ok   [5] research-gate.js: a blind run spawns ZERO agents
  ok   [5] research-gate.js: a blind run cannot pass and is marked dry_run
  ok   [5] research-gate.js: the blind run is LOGGED
  ok   [5] research-gate.js: no prompt mentioning research_brief_UNSPECIFIED is ever sent
  ok   [5] CONTROL: a usable launch DOES spawn
  ok   [5] qa-verdict.js: KILLED -- removing the blind early-return makes it spawn
  ok   [5] research-gate.js: KILLED -- removing the blind early-return makes it spawn

ALL GREEN: 87 passed, 0 failed
```

## 2. Verbatim thrown message, one malformed-args case per script

```
.claude/workflows/research-gate.js  [malformed-json-string] ->
  research-gate: args are PRESENT but not parseable as JSON (typeof=string isArray=false len=19 preview="{\"step_id\": \"86.17\"") -- pass a plain object (or valid JSON) carrying step_id, or omit args entirely for a dry run.

.claude/workflows/research-gate.js  [array] ->
  research-gate: args did not reduce to a plain object (typeof=object isArray=true len=9 preview="[\"86.17\"]") -- pass a plain object (or valid JSON) carrying step_id, or omit args entirely for a dry run.

.claude/workflows/qa-verdict.js  [malformed-json-string] ->
  qa-verdict: args are PRESENT but not parseable as JSON (typeof=string isArray=false len=19 preview="{\"step_id\": \"86.17\"") -- pass a plain object (or valid JSON) carrying step_id, or omit args entirely for a dry run.

.claude/workflows/qa-verdict.js  [array] ->
  qa-verdict: args did not reduce to a plain object (typeof=object isArray=true len=9 preview="[\"86.17\"]") -- pass a plain object (or valid JSON) carrying step_id, or omit args entirely for a dry run.
```

The `array` case is included alongside it to show the OTHER diagnosis: each
guard names its own failure rather than emitting one generic message, which is
what makes the mutation cells in §6 of `experiment_results_86.17.md`
distinguishable.

## 3. No regression against the 40/0 baseline

The pre-existing checker, measured BEFORE any code was written (at `178a6a59`)
and again after:

```
$ node scripts/qa/verify_research_gate_workflow.mjs
ALL GREEN: 40 passed, 0 failed
```

**Delta: 0.** The combined immutable command exits 0 with 40 + 87 = 127 passed.

This is the check that matters most for this particular fix: the checker imports
the workflow slice with `args` UNBOUND, so a bare `args === undefined` would
have raised `ReferenceError` and taken all 40 down with it.

## 4. Live-runtime proof of the dry run (criterion 4)

Not a sliced module -- a real launch, `wf_9e15e7ae-456`, **0 agents, 0 tokens**:

```
logs: ["qa-verdict: WARNING -- BLIND RUN. args were ABSENT, so there is no step,
        no criteria and no evidence to evaluate. Returning NO VERDICT (never a
        PASS) and spawning nothing."]
result: {"dry_run": true, "verdict": null, "ok": false,
          "input_health": {"status": "dry_run", "blind": true},
          "reason": "BLIND RUN: args were absent, so no step was identified.
                     This is NOT a verdict -- do not transcribe it into
                     evaluator_critique. Re-launch with args={step_id, ...}."}
```

It did NOT throw, and it CANNOT pass. Both halves of criterion 4, observed on
the real runtime at zero cost.

## 5. Live-runtime proof for the OTHER script (cycle 2)

Cycle 1 covered only `qa-verdict.js` live and justified the gap by inference
from a shared helper. The cycle-1 Q/A rejected that inference -- correctly, since
the two scripts diverge immediately after the helper -- so `research-gate.js`'s
dry run is now launched too (`wf_a5de9d05-c78`, **0 agents, 0 tokens**):

```
{"step_id": null, "gate_passed": false, "dry_run": true,
 "input_health": {"status": "dry_run", "blind": true},
 "violations": ["dry_run_no_step_id: args were ABSENT, so there is no step,
                 topic or scope to certify -- a blind run may never pass"],
 "brief_path": null, "envelope": null,
 "reason": "BLIND RUN: args were absent, so no step was identified. No
            researcher was spawned and no brief was written."}
```

No researcher spawned means no brief can be written under an UNSPECIFIED
identity -- criterion 3's second sentence, closed by construction rather than by
argument.

## 6. Scope of this evidence

In-module evidence across 10 shapes on both scripts, full-driver evidence for
both blind paths, and TWO live class-A launches (one per script). **No live
class-B or class-C launch was made** -- proving a throw that way costs a real
launch, and the in-module coverage is per-shape and exhaustive.
