# Experiment results -- phase-86.17

**Step:** 86.17 (P1) -- the Layer-3 Workflow rail silently runs a BLIND gate
when its `args` do not parse.
**Contract:** `handoff/current/contract_86.17.md` (written BEFORE any code).
**Research:** `handoff/current/research_brief_86.17.md` (gate PASSED, previous
session: 10 sources read in full, 54 URLs, recency scan performed).

---

## 1. What was built

Both `.claude/workflows/research-gate.js` and `.claude/workflows/qa-verdict.js`
carried `catch (_e) { a = {} }` followed by `|| 'UNSPECIFIED'` fallbacks, so
any unparseable input produced a gate running with no step id, no topic and no
scope -- writing `research_brief_UNSPECIFIED.md`, a name that collides across
every step that ever hits that path, and reporting nothing.

`classifyArgs` replaces that block in BOTH scripts (they cannot share a module:
the Workflow runtime forbids imports, so the duplication is deliberate and the
checker drives both copies):

| Class | Detection | Outcome |
|---|---|---|
| **A. ABSENT** | `typeof args === 'undefined'` or `null` | does NOT throw -- the dry run stays legal -- but is marked `blind` and **cannot pass** |
| **B. UNUSABLE** | present, does not reduce to a plain object (re-checked AFTER `JSON.parse`) | THROWS, naming `typeof`, `isArray`, length and a truncated preview |
| **C. INCOMPLETE** | plain object, no `step_id` | THROWS -- a present args object proves the caller meant to parameterise |

**The separation that is the whole point of this step:** not throwing and being
allowed to pass are INDEPENDENT concerns, and the old code conflated them. A dry
run has no step, no topic and no criteria, so a `gate_passed: true` would be a
certificate with no referent. `research-gate` now forces a named violation;
`qa-verdict` returns NO VERDICT AT ALL rather than a verdict-shaped object, so
Main's transcribe-VERBATIM rule has nothing it could mistake for an evaluation.

**Defence in depth, deliberately.** Classes B and C throw at the boundary and
never reach the gate. Class A does, and `enforceGate` refuses it independently
via `opts.inputHealth`. So the gate still fails closed on any future path that
bypasses the throw -- Saltzer's complete mediation applied against a regression
in this fix itself.

**Blind state is surfaced in all four places** the research prescribed: the
thrown error, a first-class `input_health` field in the return (not folded into
`gate_passed`), a WARNING log mirroring the existing `self_report_disagreed`
idiom, and -- for the dry run -- the returned `reason` string.

**Criterion 7, the stale comment.** `qa-verdict.js` carried a comment declaring
the silent fallback DELIBERATE ("fall back to {} and the prompt tells the agent
to self-recover the step context"). It is replaced, not amended: the remedy it
prescribed -- ask the agent to recover its own identity from prose -- is exactly
the prompt-level self-reflection EviBound measured at 100% false-completion
claims.

## 2. The measurement the contract required before locking a classification

The research brief could not settle `''` (empty string) from inside the script,
because it depends on how the Workflow tool represents "no args". **Measured at
$0 -- 0 agents, 4ms** (run `wf_a1b6c046-b60`):

```
phase-86.17 no-args probe -> UNBOUND (typeof args === "undefined")
{"args_is_bound": false, "is_empty_string": false,
 "conclusion": "args is UNBOUND on a no-args launch -- so the empty string is
  NOT how \"no args\" is represented, and `typeof` is mandatory"}
```

So `''` stays **class B (throw)**: it can only arrive if a caller explicitly
passed one, which is a caller bug, not a dry run. It is kept as a separately
named case in the checker so the decision stays visible.

## 3. Criterion 1 -- REPRODUCED FIRST, for BOTH scripts, across the named shapes

Regenerated from the PRE-FIX blob via `git show 178a6a59:<path>` rather than
transcribed, so this table cannot go stale:

```
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
```

**8 of 10 shapes resolved to `UNSPECIFIED` on EACH script -- 16 blind
resolutions in total**, and every one of them would have run a full
max-effort session and written an artifact under a colliding name.

Note `plain-object` and `valid-json-string` resolve correctly even pre-fix.
That is the brief's refutation, preserved: **"a string is unsupported" is
FALSE** -- a well-formed JSON string always parsed fine. The defect was the
silent catch plus the fallbacks.

## 4. Criteria 2, 3, 4 -- the fixed behaviour

Full per-script, per-shape output is in `live_check_86.17.md`. Summary: every
unusable and incomplete shape now THROWS with a message naming what arrived;
both usable shapes resolve `step_id=86.17`; and `absent` does not throw and is
marked `blind=true`.

**Criterion 4 proven on the LIVE runtime, not only in a sliced module** -- a
real no-args launch of `qa-verdict.js` (`wf_9e15e7ae-456`, **0 agents, 0
tokens**):

```
logs: ["qa-verdict: WARNING -- BLIND RUN. args were ABSENT, so there is no step,
        no criteria and no evidence to evaluate. Returning NO VERDICT (never a
        PASS) and spawning nothing."]
result: {"dry_run": true, "verdict": null, "ok": false,
          "input_health": {"status": "dry_run", "blind": true}, ...}
```

Both halves in one observation: it did **not** throw, and it **cannot** pass --
`verdict` is `null` and no agent was spawned.

## 5. Criterion 5 -- no regression, as a measured DELTA

The pre-existing checker was **40 passed / 0 failed** before this step (measured
at `178a6a59`, before any code was written) and is **40 passed / 0 failed**
after. **Delta: 0.** The new checker adds **70 passed / 0 failed**, so the
immutable command's combined total is **110 passed, 0 failed, exit 0**.

That the existing 40 survive is not incidental: the checker imports the sliced
module with `args` UNBOUND, so a bare `args === undefined` would have raised
`ReferenceError` and killed all 40. The `typeof` guard is what keeps them green.

## 6. Criterion 6 -- MUTATION, and a correction I had to make to my own method

Five boundary mutants plus one gate mutant, in both scripts.

**My first mutation assertion was WRONG, and I measured that rather than
shipping it.** I asserted "with the guard reverted, the shape no longer throws".
That failed for three of five cells -- because the guards are LAYERED: reverting
the JSON catch turns a malformed string into `{}`, which the DOWNSTREAM
`step_id` guard then rejects; reverting the post-parse plain-object check lets a
double-encoded string through to the same guard. The door never opens.

So the assertion is now the one each guard is actually responsible for: **its
own DIAGNOSIS**. Each cell runs a CONTROL first (the unmutated code must produce
the phrase that guard owns -- otherwise the cell could "pass" because the phrase
was never produced), then requires the mutant to produce a DIFFERENT outcome.
Basis: Shore's rule that an assertion message which does not put the error in
context is a worse guard even when the process still stops.

All 6 mutants killed, each with its control green.

## 7. Verification

```
$ bash -c 'node scripts/qa/verify_research_gate_workflow.mjs && node scripts/qa/verify_workflow_args_boundary.mjs'
ALL GREEN: 40 passed, 0 failed
ALL GREEN: 70 passed, 0 failed
```

exit **0**. `node --check` passes on both workflow scripts, and the pre-existing
checker independently asserts the four runtime constraints (`no fs`, `no
process`, no static import, exactly one `export`) that `node --check` cannot see.

## 8. What I could NOT verify

- **No live class-B or class-C launch was made.** Doing so means deliberately
  launching a workflow with malformed args; the throw is proven in-module across
  10 shapes on both scripts, and the class-A path IS proven live. I did not
  spend a real launch to watch a throw.
- **`research-gate.js`'s live path is proven only for class A by inference.**
  The live no-args proof was run on `qa-verdict.js`; `research-gate.js` shares
  the identical `classifyArgs` body but its own dry run was not separately
  launched.
- **The `input_health` field is not yet consumed by anything.** Main reads the
  returned object; no checker asserts a caller acts on it.
- **Empty `criteria` on a present-args `qa-verdict` launch is still accepted.**
  The research recommended treating it as an error. It is NOT in this step's
  criteria and adding a throw there could break legitimate launches, so it is
  deliberately deferred rather than smuggled in.
- **The duplication between the two scripts is unguarded.** Nothing fails if the
  two `classifyArgs` bodies drift apart; the checker drives both, so a drift
  that changes behaviour would be caught, but a cosmetic drift would not.
