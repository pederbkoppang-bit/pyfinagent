# Contract -- phase-86.90

**Step:** `86.90` (P1 by execution order in the 2026-08-16 goal; filed P2)
**Title:** the Q/A Workflow rail stringifies nested `evidence`/`extra` objects to
the literal text `[object Object]`, so structured spawn input silently never
reaches the evaluator
**Written:** 2026-08-16, AFTER the research gate returned (see below).
**Cycle:** 1

---

## 1. Research gate -- PASSED (enforced, not self-reported)

| Field | Value |
|---|---|
| Rail | `.claude/workflows/research-gate.js` launched by **scriptPath** (never by name) |
| Run | `wf_9bd7e233-f38` |
| Brief | `handoff/current/research_brief_86.90.md` (43,014 chars, `brief_status: COMPLETE`) |
| Sources read in full | **12** (floor 5) |
| URLs collected | **45** (floor 10) |
| Recency scan | performed |
| Audit-class | YES -- `coverage.dry = true` after **2 consecutive dry rounds** over 6 rounds |
| `gate_passed` | **true**, RECOMPUTED by the script; self-report agreed (`self_report_disagreed: false`) |
| Cross-check | all 12 claimed URLs found in the brief on disk; 45 <= 46 distinct URLs present |

Sources read in full: MDN Addition operator; typescript-eslint
`restrict-template-expressions`; typescript-eslint `restrict-plus-operands`;
arXiv 2508.06225v2 (judge calibration); arXiv 2502.06329 (FailSafeQA, finance,
missing context); **code.claude.com/docs/en/workflows (official Anthropic)**;
arXiv 2607.12885 (no-reference over-crediting); arXiv 2608.01000 (omission is
witness-blind); **Shore, IEEE Software 21(5):21-25 "Fail Fast"**; **RFC 9413**
(maintaining robust protocols); "Parse, don't validate" (King, year-less
canonical); arXiv 2607.19449v1 (silent tool-failure fabrication).

---

## 2. Hypothesis

`.claude/workflows/qa-verdict.js` builds its prompt by **string concatenation on
caller-supplied fields** (`'EVIDENCE / FILES TO READ: ' + evidence`). JavaScript
`+` applies ToPrimitive, so a plain object becomes the literal 15-character
string `[object Object]`. The caller believes it supplied a structured evidence
pointer; the evaluator receives nothing and **reconstructs an evidence set from
the repo**, then returns a confident verdict. Nothing throws, nothing warns, and
**the defect is invisible from the verdict** -- which is why it survived from
`ccddeff4` (phase-71.1) until an evaluator volunteered it in its own notes.

The fix is a **render boundary** for every caller-supplied field that reaches the
prompt: render losslessly where that is possible, and **throw, naming the field,
where it is not**. Never coerce, never substitute.

---

## 3. Reproduction ALREADY PERFORMED (pre-fix, before any change)

Recorded here because criterion 1 requires it before anything is changed. Full
verbatim output goes in `experiment_results_86.90.md` / `live_check_86.90.md`.

**(a) Historical receipt -- the real 86.86 spawn.** Extracted from the agent's own
transcript (`.../subagents/workflows/wf_b1747d75-eec/agent-abeb0c1a9dca29d03.jsonl`),
lines 61 and 63 of the prompt it received:

```
  61| EVIDENCE / FILES TO READ: [object Object]
  62|
  63| ADDITIONAL CONTEXT: [object Object]
```

**(b) Live minimal spawn, pre-fix** -- Workflow run `wf_4588d8a7-e70`, one agent,
`effort: low`, whose only job was to echo the line it received:

```json
{"runtime_typeof_evidence":"object","runtime_is_array":false,
 "runtime_keys":["handoff","changed_files","subject_sha256"],
 "script_concat_result":"EVIDENCE / FILES TO READ: [object Object]",
 "agent_received":{"received_line":"[object Object]","is_literal_object_object":true}}
```

**(c) Both shipped scripts driven with object-shaped fields** (module sliced at
its driver boundary, `args` injected -- execution, not source reading):

```
qa-verdict.js      "EVIDENCE / FILES TO READ: [object Object]"
                   "ADDITIONAL CONTEXT: [object Object]"
research-gate.js   "OBJECTIVE: [object Object]"
                   "INTERNAL SCOPE: [object Object]"
```

---

## 4. LAYER -- localised by execution, not guessed (criterion 2)

Three layers could have done it. The live probe settles it:

| Layer | Evidence | Verdict |
|---|---|---|
| Workflow **args marshalling** | inside the real runtime, `typeof args.evidence === "object"`, `Object.keys` intact, `JSON.stringify` round-trips | **INNOCENT** |
| The script's **prompt template** | `'EVIDENCE / FILES TO READ: ' + ev` evaluated in that same runtime produced the literal | **GUILTY** |
| The **`agent()` call** / transport | the agent returned `received_line: "[object Object]"`, i.e. it faithfully received what the template built | **INNOCENT** (faithful) |

Corroborated by the official Anthropic Workflow doc read in full: *"Claude passes
the list as structured data, so the script can call array and object methods on
`args` directly without parsing it first."*

**The fix therefore belongs in the prompt template of each workflow script**, at
`qa-verdict.js:106-108/153/161/163/164` and `research-gate.js:207/211/236/245/253`.

---

## 5. Two research findings that CHANGED the planned fix

Recorded because the obvious fix is wrong in two specific ways.

1. **Template literals are a NO-OP fix.** MDN: `+` coerces via ToPrimitive
   (`valueOf` first) while template literals coerce via ToString (`toString`
   first) -- but for a plain object **both** end at
   `Object.prototype.toString` and produce the identical `[object Object]`.
   Swapping `+` for `` `${}` `` would look like a fix and change nothing.
2. **Blind `JSON.stringify` VIOLATES criterion 5.** It silently drops
   `undefined`-valued keys, function-valued keys and Symbol keys, collapses
   `Map`/`Set` to `{}`, and renders `NaN`/`Infinity` as `null`. Rendering
   objects with a bare `JSON.stringify` would replace ONE silent loss with
   **five**. So the renderer must be **lossless-or-throw**, not
   render-best-effort.

Also carried forward: an **array**-shaped field renders as `a,b` with **no
`[object Object]` marker at all**, so any census keyed on that marker is a
FLOOR, never a total. The regression guard must cover the array shape explicitly.

---

## 6. Immutable success criteria (copied VERBATIM from `.claude/masterplan.json`)

1. the defect is REPRODUCED before anything is changed: spawn a minimal Q/A with a nested object in `evidence` and quote the prompt text the agent actually received, showing the literal "[object Object]"
2. the LAYER is localised with evidence -- prompt template, args marshalling, or the agent() call -- rather than guessed, and the fix is applied at that layer
3. research-gate.js is checked for the same defect and the result stated either way; a clean result must be shown by execution, not by reading the source
4. the BLAST RADIUS is enumerated: identify every prior spawn in this repo that passed an object-shaped evidence/extra, name them, and state for each whether its verdict rested on evidence that never arrived -- 86.86's PASS is a named candidate and must be resolved explicitly
5. the fix must FAIL LOUDLY on a value it cannot render, rather than substituting a placeholder -- a silent "[object Object]" is precisely the defect, and replacing it with a different silent fallback does not close it
6. a regression guard is added that would go RED if object-shaped input is stringified again, and it is mutation-tested with the control observed GREEN first
7. verdict semantics are UNCHANGED: nothing here may turn a non-PASS into a PASS

**Immutable verification command** (run 2026-08-16, exit 0, `parses`):

```
bash -c 'source .venv/bin/activate && node --check .claude/workflows/qa-verdict.js && echo parses'
```

Its reach is criterion 1's file-parses half and nothing more -- and a green
`node --check` is *already known on this repo* to pass on a script that cannot
run at all. The re-runnable checker below carries the rest.

---

## 7. Plan

**P1. The render boundary (both scripts).** Add `renderArgField(name, value,
fallback)` above the driver boundary of `qa-verdict.js` AND `research-gate.js`.
The two scripts **cannot share a module** -- the Workflow runtime forbids
`import` ("No module loading: a script that contains `import()` fails before the
run starts") -- so this is a deliberate duplicate, exactly as `classifyArgs`
already is, and the checker drives BOTH copies and asserts they have not drifted.

Rendering rule, stated as a closed set:

| Input | Result |
|---|---|
| `undefined` / `null` / `''` | the field's documented default (unchanged behaviour) |
| `string` | itself, unchanged -- **every current caller is on this path** |
| finite `number`, `boolean` | `String(value)` |
| plain object / array containing only JSON-lossless members | pretty JSON in a fenced block -- this is the VALUE, not a placeholder |
| anything else (`function`, `symbol`, `bigint`, `Map`, `Set`, `Date`, class instance, circular, `NaN`/`Infinity`, `undefined` *inside* a structure) | **THROW**, naming the field, the offending path, and what to pass instead |

Identity/path-shaped fields (`step_id`, `brief_path`, `verification_command`)
take a stricter renderer: string or number only, throw otherwise -- an object
step id would otherwise reach a *filename*.

`criteria` elements are rendered individually; a blank criterion throws rather
than silently numbering an empty line.

**P2. Unknown-key warning (log-only).** The blast-radius scan turned up a second
silent-loss instance: 11 `research-gate` runs passed a `questions` array the
script never reads. Unknown keys will be `log()`ged loudly. Deliberately
**log-only, not returned**: phase-86.78 forbids caller-authored fields
appearing as siblings of the judge's own output, and that invariant is
load-bearing. Stronger treatment is queued, not silently taken here.

**P3. Regression guard** -- `scripts/qa/verify_prompt_render_86_90.mjs`, driving
the REAL shipped files through the `runDriver` harness that already records the
prompt handed to `agent()` (behavioural, not a source scan):

- `[0]` CONTROL green first: a usable string-shaped launch still spawns exactly one agent with a prompt carrying the step id.
- `[1]` REPRODUCE from git: the pre-fix blob + object-shaped `evidence` -> prompt contains `[object Object]`.
- `[2]` FIXED: current files, object AND array shapes, `evidence`/`extra`/`topic`/`internal_scope` -> no `[object Object]`, no bare `a,b`, and every key of the input present in the prompt.
- `[3]` UNRENDERABLE: circular, `BigInt`, `Map`, function-valued key, `undefined`-valued key -> THROWS, message names the field.
- `[4]` research-gate.js driven identically (criterion 3, by execution).
- `[5]` MUTATION, anchor-uniqueness checked first: restore `+` in each file -> `[2]` must go RED; replace the throw with a placeholder -> `[3]` must go RED.
- `[6]` the two `renderArgField` copies are byte-identical.

**P4. Blast radius + the 86.86 disposition.** Enumerated from the agents' own
received prompts. **86.86's PASS was graded on a reconstructed evidence set** --
stated plainly, not reasoned away -- and is **re-graded** by a fresh Q/A on the
fixed rail with the evidence actually delivered. If the re-grade is not a PASS,
86.86 is reopened.

**P5. Handoff + log**, then the Q/A spawn, then the flip.

---

## 8. Out of scope (named, not silently dropped)

- Editing `.claude/agents/qa.md` to have the evaluator report unrenderable
  evidence. Rejected here: a machine-enforced throw in the script is strictly
  stronger than a prompt instruction, and CLAUDE.md's separation-of-duties rule
  discourages this session authoring an agent `.md` change that its own
  evaluator then depends on.
- `harness-self-audit.js:68` (`'AUDIT THIS DIMENSION: ' + d.focus`, where
  `dimensions` comes from `args`) has the same shape. It is **not** a Layer-3
  gate and has no affected history; queued rather than swept in.
- The stale gitignored `.claude/workflows/qa-verdict.js.export.mjs` generated
  artifact. Noted, untouched.

---

## 9. References

- `handoff/current/research_brief_86.90.md` (the gate's brief; run `wf_9bd7e233-f38`)
- Shore, *Fail Fast*, IEEE Software 21(5):21-25 -- the default-on-absent worked example
- RFC 9413 s4.2 / s5.1 -- "Tolerating unexpected input instead conceals problems"
- MDN, Addition (`+`) -- ToPrimitive vs template-literal ToString
- code.claude.com/docs/en/workflows -- args are structured data; no module loading
- arXiv 2607.19449v1 -- silent tool failure, fabrication rate 56.6%
- arXiv 2502.06329 (FailSafeQA) -- missing context, 41-70% fabrication
- CLAUDE.md "Harness Protocol"; `.claude/rules/research-gate.md`
