# Research Brief -- Step 86.17

**Topic:** Input-boundary integrity for an automated verification gate --
when a gate cannot parse its own inputs, should it fail loud, fail closed,
or silently default?

**Tier:** moderate (caller-specified)
**Audit-class:** NO (coverage reported for information only)
**Researcher:** Layer-3 Researcher via Workflow rail
**Date:** 2026-08-09
**Status:** IN PROGRESS -- write-first, appended incrementally as sources are read.

---

## Search-query variants run (three-variant discipline)

| # | Variant | Query |
|---|---------|-------|
| 1 | current-year frontier | `"The Constraint Tax" constrained decoding schema-valid wrong output 2026` |
| 2 | last-2-year window | `EviBound agent false completion claims verification gate artifact store arXiv` (2025-2026) |
| 3 | year-less canonical | (see below -- fail-fast / robustness principle / exception swallowing) |

---

## Internal code inventory (measured at source 2026-08-09, all line numbers re-derived)

| File | Lines | Role | Status |
|------|-------|------|--------|
| `.claude/workflows/research-gate.js` | 379 | Layer-3 RESEARCH gate, Workflow rail | LIVE; args block DEFECTIVE |
| `.claude/workflows/qa-verdict.js` | 129 | Layer-3 EVALUATE gate, Workflow rail | LIVE; identical args block DEFECTIVE |
| `scripts/qa/verify_research_gate_workflow.mjs` | 282 | re-runnable checker for research-gate.js | LIVE; 40 passed / 0 failed; ZERO args coverage |
| `.claude/rules/research-gate.md` | 293 | authoritative gate how-to | documents args shape, NOT mandatory-ness |
| `.claude/agents/researcher.md` | 372 | researcher role prompt | NO mention of `args` at all |
| `.claude/agents/qa.md` | (read) | Q/A role prompt | one `args=` mention at `:61` |
| `backend/tests/test_phase_75_20_qa_browser_grant.py` | -- | asserts `agentType:'qa'` in qa-verdict.js by REGEX on source | closest thing to a qa-verdict checker |
| `backend/harness_self_audit_report.py` | -- | `_EXPECTED_WORKFLOWS` presence check at `:65` | existence only |

### 1. `.claude/workflows/research-gate.js` -- the args block

Verbatim, `research-gate.js:71-88` (line numbers re-derived by direct read):

```js
71  // `args` may arrive as a parsed object OR a JSON string OR be absent (dry run).
72  let a = {}
73  try {
74    if (typeof args === 'string' && args.trim()) a = JSON.parse(args)
75    else if (args && typeof args === 'object') a = args
76  } catch (_e) { a = {} }
77
78  const stepId = a.step_id || a.stepId || 'UNSPECIFIED'
79  const topic = a.topic || '(no topic passed -- derive it from the step entry in .claude/masterplan.json)'
80  const VALID_TIERS = ['simple', 'moderate', 'complex']
81  const tierRaw = a.tier || 'moderate'
82  const tier = VALID_TIERS.includes(tierRaw) ? tierRaw : 'moderate'
83  const tierDefaulted = !a.tier || !VALID_TIERS.includes(tierRaw)
84  const internalScope = a.internal_scope || a.internalScope || '(none passed -- derive the relevant modules from the step entry)'
85  const auditClass = a.audit_class === true || a.auditClass === true
86  // The script tells the agent the EXACT path it will later verify, so write-first
87  // and the artifact cross-check cannot refer to different files.
88  const briefPath = a.brief_path || a.briefPath || `handoff/current/research_brief_${stepId}.md`
```

**Caller-supplied fields and their fallbacks (all six):**

| Field | Accepted keys | Falls back to | Disclosed to anyone? |
|-------|---------------|---------------|----------------------|
| `stepId` | `step_id`, `stepId` | `'UNSPECIFIED'` (`:78`) | **NO** |
| `topic` | `topic` | `'(no topic passed -- derive it from the step entry in .claude/masterplan.json)'` (`:79`) | self-describing in the prompt only |
| `tier` | `tier` | `'moderate'` (`:81-82`) | **YES** -- `tierDefaulted` (`:83`) is surfaced in the prompt at `:106` |
| `internalScope` | `internal_scope`, `internalScope` | `'(none passed -- derive the relevant modules from the step entry)'` (`:84`) | self-describing in the prompt only |
| `auditClass` | `audit_class`, `auditClass` | `false` (`:85`) | **NO** -- silently downgrades an audit-class step |
| `briefPath` | `brief_path`, `briefPath` | `` `handoff/current/research_brief_${stepId}.md` `` (`:88`) -> `research_brief_UNSPECIFIED.md` | **NO** |

**`tier` is existing in-file prior art for the fix.** `tierDefaulted` at `:83` is the ONE field
that already has a defaulting-disclosure mechanism, threaded into the prompt at `:106`:
`(NOT passed by the caller -- defaulted to moderate; state this assumption in the brief)`.
So the file already contains the pattern the other five fields lack. The fix generalises an
idiom that is already here; it does not invent one.

**`auditClass` silently defaulting to `false` is the most damaging single field.** Per
`researcher.md` "Adaptive coverage" and `research-gate.md:164-191`, `audit_class` is
explicitly *caller-set* and "the researcher never self-declares it to escape the loop".
A parse failure therefore silently converts an audit-class step into a non-audit step,
removing the loop-until-dry requirement from `enforceGate` at `:233-238` -- the gate becomes
strictly weaker with no diagnostic.

**Downstream reach of the blind values.** `stepId` reaches `PROMPT` (`:96`), the agent label
(`:300`), the log lines (`:356`, `:358`, `:361`), the stage-2 label (`:342`) and the RETURN
value `step_id` (`:369`). So a blind run returns `step_id:'UNSPECIFIED'` to Main and writes
`research_brief_UNSPECIFIED.md`.

**The return object (`:368-378`)** carries `step_id`, `gate_passed`, `agent_self_reported_gate_passed`,
`self_report_disagreed`, `violations`, `checks`, `brief_path`, `brief_verification`, `envelope`.
There is **no field for input health**: nothing in the return distinguishes a fully-parameterised
run from a blind one. `enforceGate` (`:201-282`) never sees `a`, `stepId` or any args-derived
value -- its signature is `(env, verification, opts)`. So a blind run can and does return
`gate_passed: true` provided the researcher self-recovers enough context to hit the floors.

**Two-stage flow.** Stage 1 `agent(PROMPT, {...})` at `:299-306`; stage 2 the read-only brief
verifier at `:321-348` wrapped in its own `try { } catch (_e) { verification = null }` at
`:349-351`. That second catch is **correctly designed**: it comments `// fail closed in enforceGate`
and `enforceGate:255-259` converts a null verification into a violation
(`'brief verification did not run ... -- failing closed rather than trusting the self-report'`).
**This is the in-file counter-example that settles the design question**: the same file already
contains a swallow-and-fail-CLOSED catch and a swallow-and-fail-OPEN catch. Only the args
catch fails open.

### 2. `.claude/workflows/qa-verdict.js` -- the identical block

The comment at `qa-verdict.js:25-29` declares the fallback DELIBERATE. **Verbatim**:

```js
25  // `args` may arrive as a parsed object OR as a JSON string (the Workflow tool
26  // stringifies scriptPath args on some paths) OR be absent (a dry-run). Handle
27  // all three so the parameterized launch actually threads its parameters; on any
28  // parse error, fall back to {} and the prompt tells the agent to self-recover
29  // the step context from .claude/masterplan.json + handoff/current/.
```

The block itself, `qa-verdict.js:30-40`:

```js
30  let a = {}
31  try {
32    if (typeof args === 'string' && args.trim()) a = JSON.parse(args)
33    else if (args && typeof args === 'object') a = args
34  } catch (_e) { a = {} }
35  const stepId = a.step_id || a.stepId || 'UNSPECIFIED'
36  const criteria = Array.isArray(a.criteria) ? a.criteria : []
37  const verificationCommand = a.verification_command || a.verificationCommand || '(none provided -- read it from .claude/masterplan.json for this step)'
38  const evidence = a.evidence || 'handoff/current/{contract.md, experiment_results.md, evaluator_critique.md} + the files changed this step (git status --short / git diff)'
39  const extra = a.extra || ''
```

The comment's claim "the prompt tells the agent to self-recover" is **true but load-bearing in a
way it does not admit**: self-recovery is a *best-effort instruction to an LLM*, not a
mechanism. Nothing verifies that recovery happened, and nothing marks the verdict as
recovered-from-blind. The fix must correct or delete this comment -- as written it
authorises the defect.

**What a verdict rendered with `criteria: []` actually asserts.** At `:68`:

```js
...(criteria.length ? criteria.map((c, i) => '  ' + (i + 1) + '. ' + c) : ['  (none passed in args -- read them from .claude/masterplan.json for this step and evaluate against them)']),
```

under the header `'IMMUTABLE SUCCESS CRITERIA for ' + stepId + ':'` (`:67`) where `stepId` is
`'UNSPECIFIED'`. So the Q/A is told to look up the immutable criteria for a step whose id it
was not given. The prompt then instructs (`:74-77`) `verdict=PASS only if EVERY immutable
criterion is met`. **A `PASS` over an empty criteria list is vacuously true** -- "every element
of the empty set is met". This is materially worse than the researcher case: the researcher's
blind run still faces `enforceGate`'s recomputation, whereas `qa-verdict.js:129` is a bare
`return verdict` with **no enforcement layer at all**. Main then transcribes that verdict
VERBATIM into `evaluator_critique.md` per CLAUDE.md, i.e. a vacuous PASS becomes the
step's official evaluation record.

### 3. `scripts/qa/verify_research_gate_workflow.mjs` -- CONFIRMED zero args coverage

Measured 2026-08-09, `grep -c -F` against the checker source:

| token | count |
|-------|-------|
| `args` | **0** |
| `JSON.parse` | **0** |
| `step_id` | **0** |
| `stepId` | **0** |
| `UNSPECIFIED` | **0** |
| `topic` | **0** |
| `dry-run` / `dry run` | **0** |

**CONFIRMED: zero args-boundary coverage.** Live run 2026-08-09: `ALL GREEN: 40 passed, 0 failed`.

**How it loads the script under test (`:47-60`) -- it IMPORTS, it does not regex or exec:**

```js
async function loadModule(sourceOverride) {
  const src = sourceOverride ?? fs.readFileSync(WORKFLOW, 'utf8')
  const idx = src.indexOf("phase('Research')")
  if (idx === -1) throw new Error("could not find the phase('Research') driver boundary")
  const tmp = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'rg-')), 'rg.mjs')
  fs.writeFileSync(tmp, src.slice(0, idx)
    + '\nexport { enforceGate, ENVELOPE_SCHEMA, BRIEF_VERIFICATION_SCHEMA, FLOOR_SOURCES, FLOOR_URLS }\n')
  const mod = await import(pathToFileURL(tmp).href)
  ...
}
```

i.e. read source -> slice everything before `phase('Research')` (`research-gate.js:298`) ->
write a temp `.mjs` -> **append an export list** -> **dynamic `import()`**. Section `[7]`
(`:215-244`) reuses the same loader with a `String.replace` mutant, so mutation testing runs
against the REAL module too. New tests MUST use this same mechanism.

**Two consequences the caller must know before specifying the fix:**

1. **The args block IS inside the imported slice.** It sits at `:71-88`, well before the
   `:298` boundary, so it executes on every `loadModule()` call. It is *reachable* by the
   checker today; it is simply never *asserted on*.
2. **The empty catch is currently LOAD-BEARING for the checker's own import.** In the temp
   module `args` is an **unbound identifier**. `typeof args === 'string'` is safe (`typeof` on
   an undeclared name yields `"undefined"` and does not throw), but the `else if (args && ...)`
   branch **dereferences it and throws `ReferenceError: args is not defined`** -- which the
   `catch (_e)` swallows. Measured directly:

   ```
   unbound `args` -> path = CAUGHT: ReferenceError: args is not defined | a = {}
   ```

   and with the try/catch naively removed, the same code **throws at import**:

   ```
   ReferenceError: args is not defined
       at .../probe_unbound2.mjs:3:6
   ```

   **A naive "just delete the catch and throw" fix therefore breaks all 40 existing checks at
   module-load time.** Whatever replaces the block must keep the unbound-identifier case
   non-throwing (that is the same case as the legitimate dry run) while making the
   present-but-unusable case loud. `typeof args === 'undefined'` is the safe discriminator;
   a bare `args === undefined` is not.
3. To assert on `stepId`/`topic`/`auditClass`/`briefPath` the checker must **add them to the
   appended export line at `:56`** -- today it exports only
   `enforceGate, ENVELOPE_SCHEMA, BRIEF_VERIFICATION_SCHEMA, FLOOR_SOURCES, FLOOR_URLS`, so
   every args-derived constant is computed but unreachable. Better still: extract the block
   into a pure named function (e.g. `parseArgs(rawArgs)`) so the checker can drive it with all
   input shapes without re-binding a module-level free variable. **The checker cannot vary a
   module-scope free variable across cases** -- ESM caches by URL, so each shape would need a
   fresh temp file. A pure exported function removes that problem entirely and matches the
   existing `enforceGate` idiom (pure, exported, mutation-testable).

**No equivalent checker exists for `qa-verdict.js`.** The whole-repo search found only:
`backend/tests/test_phase_75_20_qa_browser_grant.py:112` (a **regex over the source text**,
`assert re.search(r"agentType:\s*'qa'", src)`), `backend/harness_self_audit_report.py:65`
(`_EXPECTED_WORKFLOWS` -- file presence only) and
`backend/tests/test_phase_71_6_self_audit_cron.py:49` (writes a `// qa` stub fixture).
`scripts/` contains exactly one `.mjs` checker. So `qa-verdict.js` has **no behavioural
checker at all** -- the fix will need to create one, or generalise the existing `.mjs`.

### 4. Docs -- do they mandate args or document the dry run?

**Neither file mandates args, and NO file documents the dry-run mode.** Complete set of `args`
mentions across the three governing docs (`grep -n -i args`):

- `.claude/rules/research-gate.md:198` -- ``invoked with `args={step_id, topic, tier, internal_scope, audit_class, brief_path}`.``
- `.claude/agents/qa.md:61` -- `` `args={step_id, criteria[], verification_command, evidence, extra}`, ``
- `.claude/agents/researcher.md` -- **zero mentions of `args`**.

Both are declarative shape descriptions in prose. Neither uses MUST/REQUIRED language, neither
says what happens if a field is missing, and **the "no-args dry run" appears nowhere in any
`.md`** -- it exists only in the two source comments (`research-gate.js:71`,
`qa-verdict.js:26-27`). The dry-run mode is therefore an *undocumented* feature that the fix is
nevertheless constrained not to break. Recommend the fix also lands one line in
`research-gate.md` so the mode is specified rather than inferred.

### 5. Workflow-runtime constraint -- CONFIRMED at source

`research-gate.js:51-69` and `:284-296` state it, and the checker's header `:12-19` repeats it:

- **No filesystem / Node API access.** A static `import fs from 'node:fs'` yields
  `SyntaxError: Unexpected identifier 'fs'. import call expects one or two arguments.`
  (MEASURED 2026-08-09 per the in-file comment.)
- **Only the leading `export const meta` is accepted**; a trailing export list yields
  `SyntaxError: Unexpected keyword 'export'`.
- **`node --check` PASSES on a script that cannot run** in both cases -- so the step's
  immutable command is green on an unlaunchable file.

The checker enforces all three structurally at `:246-278`: zero static imports of any form
(`:268-270`), exactly one `export` (`:274-275`), and `enforceGate` purity (`:276-277`).

**This bounds the fix**: no `fs`, no `process`, no `import`, no new top-level `export`. The fix
must be pure JS, and any new helper must be reachable by the checker through the same
strip-and-append-export mechanism.

### 6. Reproduction -- 12 input shapes measured (caller reported 5-of-7 blind; measured 9-of-12)

Ran the block verbatim in isolation (`node`, 2026-08-09):

| # | Input shape | `stepId` | topic | tier | scope | audit | Verdict |
|---|-------------|----------|-------|------|-------|-------|---------|
| 1 | plain object | `86.17` | ok | complex | ok | true | WORKS |
| 2 | valid JSON string | `86.17` | ok | complex | ok | true | WORKS |
| 3 | malformed JSON string | `UNSPECIFIED` | lost | moderate | lost | false | **BLIND** |
| 4 | JSON string, raw newline in a value | `UNSPECIFIED` | lost | moderate | lost | false | **BLIND** |
| 5 | `undefined` (absent -- legit dry run) | `UNSPECIFIED` | lost | moderate | lost | false | **BLIND (legitimate)** |
| 6 | empty string | `UNSPECIFIED` | lost | moderate | lost | false | **BLIND** |
| 7 | double-encoded JSON string | `UNSPECIFIED` | lost | moderate | lost | false | **BLIND** |
| 8 | **array** `[{...}]` | `UNSPECIFIED` | lost | moderate | lost | false | **BLIND (new)** |
| 9 | **scalar** `42` | `UNSPECIFIED` | lost | moderate | lost | false | **BLIND (new)** |
| 10 | `null` | `UNSPECIFIED` | lost | moderate | lost | false | **BLIND (new)** |
| 11 | object missing `step_id` only | `UNSPECIFIED` | ok | complex | lost | false | **PARTIALLY BLIND (new)** |
| 12 | bogus tier `"DEEP"` | `86.17` | ok | **moderate** | ok | false | silently coerced, but DISCLOSED via `tierDefaulted` |

**Three shapes the caller's 7-shape sweep did not name, all blind:**

- **Case 8 (array).** `typeof [] === 'object'`, so `:75` *accepts* an array and assigns it to
  `a`. This is exactly the trap `enforceGate` guards against **twice** in the same file, with
  an explicit comment at `:250-254`: *"`Array.isArray` is not redundant: `typeof [] === 'object'`,
  so an array would slip this guard"*. The args block at `:75` does **not** carry that guard --
  an internal inconsistency inside one file.
- **Case 9 (scalar `42`).** `42 && typeof 42 === 'object'` is false, so it falls through
  silently; `a` stays `{}`.
- **Case 7 (double-encoded).** `JSON.parse` **succeeds** and returns a *string*. `a` is then a
  string, and `a.step_id` is `undefined` -- so this one defeats the `try/catch` entirely: there
  is no error to catch. **A fix that only hardens the catch does not cover case 7.** The
  post-parse type must be re-checked (`typeof parsed === 'object' && !Array.isArray(parsed)`).
- **Case 11** shows the failure is not all-or-nothing: a well-formed object missing only
  `step_id` writes `research_brief_UNSPECIFIED.md` while otherwise running correctly, which is
  the *hardest* variant to notice.

---

## Read in full (10; floor is 5 -- counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| 1 | https://arxiv.org/html/2511.05524 | 2026-08-09 | paper (arXiv, Nov 2025) | WebFetch (arXiv HTML) | EviBound. Prompt-level self-reflection ALONE = **100% hallucination (8/8 claimed success, 0/8 verified)**; verification-gate-only = **25%**; dual gates = **0%**. "Rather than trusting agent assertions, query external artifact stores." "No evidence, no claim." ~8.3% overhead. |
| 2 | https://arxiv.org/html/2605.26128v1 | 2026-08-09 | paper (arXiv, May 2026) | WebFetch (arXiv HTML) | The Constraint Tax. Wrong-valid-schema **49.5% -> 88.9% (+39.4pp)**, 15,000 generations. "A production system that reports only schema validity would miss the regression." "Parseability is a transport property, not a task-success metric." |
| 3 | https://www.rfc-editor.org/rfc/rfc9413.html | 2026-08-09 | standard (IETF, official) | WebFetch | Maintaining Robust Protocols -- the modern critique of Postel. "Hiding the consequences of protocol variations encourages the hiding of issues, which can conceal bugs and make them difficult to discover." Recommends *fail fast* + *virtuous intolerance*. |
| 4 | https://cwe.mitre.org/data/definitions/703.html | 2026-08-09 | official (MITRE) | WebFetch | CWE-703 pillar: "does not properly anticipate or handle exceptional conditions". Children listed: 228, 248, 391, **392**, 393, 397, 754, 755. **CWE-390 and CWE-1069 are NOT listed as children** -- correction to the spawn prompt's assumption. |
| 5 | https://cwe.mitre.org/data/definitions/392.html | 2026-08-09 | official (MITRE) | WebFetch | CWE-392 Missing Report of Error Condition -- **the sharpest match**. Demonstrative example is literally `catch (Throwable t) { logger.error(...); return; }` returning HTTP 200 OK. CVE-2004-0063: "function returns OK despite invalid PIN validation". |
| 6 | https://cwe.mitre.org/data/definitions/1188.html | 2026-08-09 | official (MITRE) | WebFetch | CWE-1188 insecure default. "Developers often choose default values that leave the product as open and easy to use as possible out-of-the-box, under the assumption that the administrator can (or should) change the default value." |
| 7 | https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html | 2026-08-09 | official (OWASP) | WebFetch | "Input validation should happen as early as possible in the data flow, preferably as soon as the data is received from the external party." Allowlist over denylist. Validation failure against a discrete option list "should be logged as a high severity event". |
| 8 | https://martinfowler.com/ieeeSoftware/failFast.pdf | 2026-08-09 | peer-reviewed (IEEE Software, Sep/Oct 2004) | WebFetch returned binary -> **pdfplumber** per research-gate.md Step 3 (13,408 chars extracted) | Shore, "Fail Fast". **The article's central worked example IS a config reader that returns a default.** "This results in the software 'failing slowly.'... fails in strange ways later on." "Search your existing code for catch-all exception handlers and either remove or refactor them." |
| 9 | https://web.mit.edu/Saltzer/www/publications/protection/Basic.html | 2026-08-09 | peer-reviewed (Saltzer & Schroeder, Proc. IEEE 1975) | WebFetch | Fail-safe defaults: "Base access decisions on permission rather than exclusion." A permission-based mistake "tends to fail by refusing permission, a safe situation, since it will be quickly detected"; exclusion-based mistakes "tend to fail by allowing access, a failure which may go unnoticed in normal use." Also *complete mediation*. |
| 10 | https://arxiv.org/html/2605.17998 | 2026-08-09 | paper (arXiv, May 2026) | WebFetch (arXiv HTML) | Verify-Gated Completion as Admission Control. "Execution may propose completion, but admission depends on explicit checks." **"Low confidence, weak evidence, missing ownership, skipped verification, unsupported states, and stale common ground all resolve fail-closed."** Blocked/failed are "explicit control outcomes, not side effects". |

**Source-quality mix:** 4 peer-reviewed/preprint (1, 2, 8, 9, 10 -- five), 5 official standards/vendor
(3, 4, 5, 6, 7). Zero community-tier in the read-in-full set.

---

## Identified but snippet-only (44; context, does NOT count toward gate)

54 unique URLs collected in total; 10 read in full, **44 snippet-only**. Notable:

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://arxiv.org/html/2607.13083 | paper (Phantom Guardrails, Jul 2026) | Strong recency hit on harness-optimizer reward hacking; time-boxed after the floor was cleared 2x |
| https://arxiv.org/html/2608.04278 | paper (EA-Graph, Aug 2026) | Artifact-anchored verification memory; adjacent, not input-boundary |
| https://arxiv.org/html/2606.25605v1 | paper (Constraint Tax in open-weight LLMs) | Replication of source 2 |
| https://arxiv.org/abs/2511.05524 , /pdf/2511.05524 | abs/pdf of source 1 | HTML read instead (research-gate.md forbids WebFetch on /pdf/) |
| https://arxiv.org/abs/2605.26128 , /pdf/2605.26128 | abs/pdf of source 2 | as above |
| https://en.wikipedia.org/wiki/Error_hiding | community | community tier -- excluded from the gate set by the hierarchy |
| https://docs.aws.amazon.com/codeguru/detector-library/jsx/swallow-exceptions | vendor detector rule | corroborates that "catch and swallow" is a first-class static-analysis detector |
| https://enterprisecraftsmanship.com/posts/fail-fast-principle/ | blog | superseded by source 8 (the primary) |
| https://www.exida.com/images/uploads/exida_Position_on_IEC_61508_2010_definitions_minimum_HFT_v4.pdf | industry (IEC 61508) | safety analogue covered by snippet; sources 9+10 carry the fail-closed argument more directly |
| https://theconfigreport.com/p/cicd-pipeline-is-a-liar | blog | CI green-build analogue; community/blog tier |
| https://gitlab.com/gitlab-org/gitlab-foss/-/issues/64461 | vendor issue tracker | "Pipelines must succeed is ignored if CI is skipped" -- the exact CI analogue |
| https://forum.gitlab.com/t/some-tests-skipped-in-reports/87453 | forum | community tier |

(Remaining 32: arXiv 2603.03305 / 2511.18335 / 2510.07248 / 2604.06066; researchgate 397479384;
lambpetros.substack.com; avench.com, risknowlogy.com, spyro-soft.com, ntnu.edu chapt10-sis.pdf,
ez.analog.com, image-ppubs.uspto.gov/10761916, alekvs.com (IEC 61508 set); medium
qbyteconsulting, houseofsoft.org, fourkitchens.com, digitaldrummerj.me, oreilly.com
testing-with-junit (fail-fast set); medium bgskinner3, devops.com, dev.to sumit_gautam_379d5,
blog.pixelfreestudio.com, squareops.com (CI set); galileo.ai, rahulkashyap.dev,
harness.io/blog/introducing-agent-trace, blog.whoisjsonapi.com, abhishek-tiwari.com
(agent-harness set); web.mit.edu/Saltzer/.../protection/ index.)

---

## Recency scan (2024-2026) -- PERFORMED

Searched the 2024-2026 window explicitly on three axes: agent self-report reliability
(`EviBound ... arXiv`), constrained-decoding validity-vs-correctness (`"The Constraint Tax" ... 2026`),
and the CI/CD silent-skip analogue (`CI pipeline silently skipped tests ... 2025 2026`), plus
`agent harness misconfigured guardrail silently disabled ... 2026`.

**Result: 5 new findings that COMPLEMENT (none supersede) the canonical prior art.**

1. **EviBound (arXiv 2511.05524, Nov 2025)** -- both citations in `research-gate.js:20-24`
   **VERIFIED AT SOURCE**, with one **material correction**: the comment says false completion
   falls "to 0% ONLY with a post-hoc gate that queries the artifact store." The paper measures
   the post-hoc **Verification Gate alone at 25%**, not 0%. 0% requires **both** gates -- and
   the second is a **pre-execution Approval Gate** that validates the acceptance contract
   *before* code runs, rejecting placeholder/underspecified inputs. **That pre-execution gate
   is the exact analogue of the args validation this step is about**, and it is the leg the
   comment omits. The paper's own framing: prompt-level techniques "can't guarantee artifacts
   actually exist."
2. **The Constraint Tax (arXiv 2605.26128, May 2026)** -- 49.5% -> 88.9% **VERIFIED** (Table 3,
   15,000 generations, CI [47.7,51.2] -> [87.8,90.0]). Adds the reporting prescription:
   "Track wrong-valid-schema rate as a first-class reliability metric."
3. **Verify-Gated Completion (arXiv 2605.17998, May 2026)** -- new. Enumerates eleven mandatory
   admission conditions and states that missing/underspecified claim packets "all resolve
   fail-closed."
4. **CI/CD analogue (2025-2026)** -- a green pipeline silently skipping 40 scenarios from a tag
   typo; "collected-but-skipped tests still count as collected, meaning minimum test count
   gates can pass even when tests are skipped"; GitLab's "Pipelines must succeed" bypassed by
   CI-skip. Directly parallel: the checker's `40 passed / 0 failed` is a minimum-count gate
   that is green while the args path is untested.
5. **Phantom Guardrails (arXiv 2607.13083, Jul 2026)** -- snippet-only. Harness optimizers
   reward "did the failure stop?" and never "was the fix warranted?" -- the failure mode a
   silently-defaulting gate manufactures.

**Nothing in the window contradicts Shore 2004, Saltzer 1975 or RFC 9413**; the 2025-2026 agent
literature independently re-derives fail-closed for LLM gates.

---

## Key findings

1. **A gate that cannot read its configuration must not certify -- this is unanimous across
   every source tier.** Saltzer & Schroeder: "Base access decisions on permission rather than
   exclusion"; a permission-based mistake "tends to fail by refusing permission, a safe
   situation, since it will be quickly detected", whereas exclusion-based mistakes "tend to
   fail by allowing access, a failure which may go unnoticed in normal use"
   (https://web.mit.edu/Saltzer/www/publications/protection/Basic.html). Verify-Gated
   Completion re-derives it for agent runtimes 51 years later: "skipped verification ...
   resolve[s] fail-closed" (https://arxiv.org/html/2605.17998). **The current block does the
   exclusion-based thing: absence of parseable input is read as permission to proceed.**

2. **The defect is CWE-392, and the canonical example is nearly identical to the code.**
   CWE-392's demonstrative example is `catch (Throwable t) { logger.error(...); return; }`
   where the servlet still returns HTTP 200 OK
   (https://cwe.mitre.org/data/definitions/392.html). `research-gate.js:76` /
   `qa-verdict.js:34` is **strictly worse**: `catch (_e) { a = {} }` does not even log. Related
   CVEs are exactly this shape -- CVE-2004-0063 "function returns OK despite invalid PIN
   validation", CVE-2002-1446 "PKCS#11 library returns OK status when detecting invalid
   signatures". A gate returning `gate_passed:true` after failing to parse its own inputs is
   the same object.

3. **Shore's canonical fail-fast article uses this exact defect as its worked example.** His
   "failing slowly" specimen is a config reader that returns a default when the property is
   absent, and his verdict on it: "For the code that returns a default value, everything will
   seem fine. But when customers start using the software, they'll encounter mysterious
   slowdowns" (https://martinfowler.com/ieeeSoftware/failFast.pdf, IEEE Software 2004). His
   closing instruction is literally the remediation: "Search your existing code for catch-all
   exception handlers and either remove or refactor them."

4. **Shore also supplies the exception that keeps the dry run alive.** "A crash is never
   appropriate. Fortunately, there's a middle ground": a *global* handler at the top, with
   catch-alls removed from everywhere else -- "If you use a global exception handler, avoid
   catch-all exception handlers in the rest of your application. They'll prevent exceptions
   from reaching your global handler." Mapped here: the throw belongs at the args boundary; the
   *runtime* is the global handler. The local `catch (_e)` is precisely the catch-all that
   prevents the error from ever reaching it.

5. **RFC 9413 rules out the "be liberal in what you accept" defence, which is the only
   principled argument for the current code.** "Hiding the consequences of protocol variations
   encourages the hiding of issues, which can conceal bugs and make them difficult to
   discover"; and *fail fast*: "Choosing to generate fatal errors for unspecified conditions
   instead of attempting error recovery can ensure that faults receive attention"
   (https://www.rfc-editor.org/rfc/rfc9413.html). Its *virtuous intolerance* argument is the
   direct answer to `qa-verdict.js:27-29`'s "the prompt tells the agent to self-recover":
   "When a new implementation encounters a peer that is intolerant of an error, it receives
   strong feedback that allows the problem to be discovered quickly." Tolerating the malformed
   caller means the caller is never fixed.

6. **The literature specifically rejects prompt-level self-recovery as a mechanism.** EviBound
   measures prompt-level self-reflection ALONE at **100% false-completion (8/8)**, and states
   "Prompt-level techniques like self-reflection and critique help with factual errors, but
   they can't guarantee artifacts actually exist" (https://arxiv.org/html/2511.05524).
   `qa-verdict.js:27-29` rests the entire correctness of a mis-parsed launch on exactly that
   mechanism. **The comment does not merely describe the defect -- it asserts a remedy the
   literature measures at 0% effective.**

7. **EviBound's Approval Gate is the missing leg, and it is a PRE-execution input check.** It
   validates the acceptance contract before code runs -- schema compliance, machine-checkability,
   and specifically "No hallucinated placeholders (rejects values like 'TBD' or placeholder
   run_ids)". `'UNSPECIFIED'` is a placeholder run_id. EviBound would reject it at the door;
   `research-gate.js` writes a brief named after it. The 25%->0% delta is attributable to
   having this gate.

8. **Reporting the degradation must be separate from the pass/fail bit.** The Constraint Tax:
   "A production system that reports only schema validity would miss the regression", and
   "needs at least four numbers" reported separately (https://arxiv.org/html/2605.26128v1).
   Applied: `gate_passed` alone cannot carry input health; a blind run and a clean run are
   currently indistinguishable in the return object (`research-gate.js:368-378`).

9. **CWE-1188 names the exact rationalisation in the code comment.** "Developers often choose
   default values that leave the product as open and easy to use as possible out-of-the-box,
   under the assumption that the administrator can (or should) change the default value.
   However, this ease-of-use comes at a cost when the default is insecure and the administrator
   does not change it" (https://cwe.mitre.org/data/definitions/1188.html). Substitute "the
   agent can self-recover" for "the administrator can change it".

10. **OWASP puts the check at the earliest point in the data flow** -- "as soon as the data is
    received from the external party" -- and treats a failure against a discrete option list as
    "a high security event [that] should be logged as a high severity event"
    (https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html). The
    args block IS that earliest point; it currently logs nothing.

---

## Consensus vs debate (external)

**Consensus (no dissent found in 10 sources across 5 decades):** an input-parse failure at a
trust boundary must be *loud*; silently substituting defaults is an anti-pattern with a CWE id;
a verification component that loses its configuration must not certify.

**The genuine debate is WHERE the loudness lands, not WHETHER.** Shore explicitly rejects
"crash the process" as the terminal answer ("a crash is never appropriate") and prescribes
throw-locally + handle-globally. RFC 9413 concedes workarounds may be *temporarily* necessary
but insists they be time-bounded and reported upstream. Verify-Gated Completion's position is
strictest: blocked/failed are "explicit control outcomes, not side effects", requiring an owner
and a recovery packet before re-entry -- i.e. a refused gate should not be silently retryable.

**No source anywhere endorses "default and continue" at a verification boundary.** The closest
thing to an adversarial voice found was the guardrail-gateway pattern in the 2026 harness
snippets ("If the Gateway is unreachable, traffic passes straight through with an annotated
span rather than blocking") -- and note that even that fail-open design **annotates the span**,
i.e. it makes the degraded mode visible. Nothing in the corpus does what the current code does:
degrade *and* stay silent.

---

## Pitfalls (from literature + measured here)

- **P1. Hardening only the `catch` does not fix case 7.** Double-encoded JSON `JSON.parse`s
  *successfully* into a string -- there is no exception to catch. The post-parse **type** must
  be re-asserted. (Measured.)
- **P2. `typeof [] === 'object'`** -- an array passes the `:75` guard. The same file guards this
  twice in `enforceGate` with an explicit comment at `:250-254`. (Measured.)
- **P3. Naively deleting the try/catch breaks the checker's own import** with
  `ReferenceError: args is not defined`, killing all 40 green checks. Use
  `typeof args === 'undefined'`, never a bare `args === undefined`. (Measured.)
- **P4. A green count is not coverage.** `40 passed / 0 failed` on a suite with **zero**
  args-boundary cases is exactly the CI failure mode in the recency scan: "collected-but-skipped
  tests still count as collected, meaning minimum test count gates can pass".
- **P5. `qa-verdict.js` has no enforcement layer at all** (`:129` is a bare `return verdict`),
  so "force the result to false" is unavailable there -- the throw is the only mechanism unless
  a wrapper is added.
- **P6. Vacuous truth.** `criteria: []` + "PASS only if EVERY immutable criterion is met"
  (`qa-verdict.js:68,74-77`) is satisfiable by the empty set. Any fix must make an empty
  criteria list an error, not merely a prompt hint.
- **P7. A warning that lives only in a log line does not survive.** Anthropic harness-design
  makes the *file* the durable state; the degraded marker must reach the artifact.

---

## Application to pyfinagent -- design recommendation (Q4)

### The three input classes, and the recommended behaviour for each

| Class | Detection (order matters) | Recommended behaviour | Basis |
|---|---|---|---|
| **A. ABSENT** -- dry run, legitimate | `typeof args === 'undefined'` (covers the unbound identifier AND explicit `undefined`) or `args === null` | **Do NOT throw.** Run in an explicit `dryRun` mode. **Force `gate_passed:false`** with violation `dry_run_no_step_id: nothing to certify`. | Shore ("a crash is never appropriate"); Saltzer fail-safe defaults; EviBound "no evidence, no claim" |
| **B. PRESENT BUT UNUSABLE** -- malformed JSON, raw-newline JSON, array, scalar, `''`, double-encoded-to-non-object | anything not class A that does not reduce to a **plain object** (`typeof x === 'object' && x !== null && !Array.isArray(x)`), re-checked **after** `JSON.parse` | **THROW**, naming the received `typeof`, `Array.isArray`, length, and a truncated preview | RFC 9413 fail fast + virtuous intolerance; CWE-392; Shore's assertion-message rule |
| **C. PRESENT BUT INCOMPLETE** -- parses to a plain object, no `step_id` | class-B check passes, `!a.step_id && !a.stepId` | **THROW.** A present args object is proof the caller *intended* to parameterise, so this is unambiguously a caller bug, not a dry run. | RFC 9413 (intolerance is what gets the caller fixed); EviBound Approval Gate rejects placeholder run_ids |

Note on `''` (empty string): classified **B (unusable)** above, but this is the ONE case whose
correct class depends on how the Workflow tool represents "no args" -- which cannot be
determined from inside the script. **Main should confirm against a live no-args launch before
locking it**; if the tool passes `''` for no-args, move it to class A. Keep it a separately
named case either way so the decision is visible.

### Explicit answer: may an absent-args (dry-run) run return `gate_passed: true`?

**No. Never.** The dry run has no step, no topic, no scope and no criteria -- there is no
subject to certify, so a `true` is a certificate with no referent. That is precisely
CVE-2004-0063's shape ("returns OK despite invalid PIN validation") under CWE-392. Saltzer
settles it: permission must be affirmatively established, not assumed from the absence of a
refusal. **The legitimacy of the dry run is about not THROWING; it is not a licence to PASS.**
The two concerns are independent and the current code conflates them.

Recommended split for `research-gate.js`: thread an `inputHealth` object into `enforceGate` (it
stays pure -- no I/O) so that `inputHealth.blind === true` appends a violation. That makes the
gate fail-closed **even on any future path that bypasses the throw** -- Saltzer's *complete
mediation*, and defence in depth against the fix itself regressing.

For `qa-verdict.js` there is no `enforceGate`, so a dry run must not return a verdict-shaped
object at all. Recommend returning `{ dry_run: true, verdict: null, ... }` so Main's
"transcribe VERBATIM" rule has nothing that could be mistaken for an evaluation.

### Tradeoffs, stated honestly

- **Throwing costs the run.** Counter: it costs it at line ~76, before a single token is spent.
  Today the same failure costs a full Opus-`max` research/Q-A session AND deposits a misfiled
  artifact (`research_brief_UNSPECIFIED.md`) in `handoff/current/`, which the archive hook
  cannot match to a phase.
- **Blast radius of the throw is bounded and measured.** The only known consumer relying on the
  swallow is the checker's own import path -- and that is **class A**, which does not throw
  under this design. No other caller was found.
- **The softer alternative** (never throw; always run; force `gate_passed:false` when blind)
  preserves the session but guarantees a wasted one. Given the shared weekly Max budget rule in
  CLAUDE.md, burning a `max`-effort session for a gate that cannot pass is pure waste.
  **Recommend both layers:** throw for B and C; force-false for anything that slips through.
- **Do not "repair" near-misses** (e.g. re-parsing a double-encoded string a second time).
  OWASP's allowlist posture and RFC 9413's anti-workaround argument both point the other way,
  and the existing stage-2 prompt already sets the house idiom: *"do not repair near-misses"*
  (`research-gate.js:336`).

### Q5 -- how to surface the blind state: ALL THREE, plus the artifact

1. **In the thrown error.** Shore: "Don't just repeat the assertion's condition... put the error
   in context." His model message names the key *and* the file. Recommended shape:
   `research-gate: args present but unusable (typeof=string, len=27, isArray=false, preview="{\"step_id\":\"86.17\", \"topic\":") -- pass a plain object or valid JSON, or omit args for a dry run.`
2. **In the return value, as a first-class field** -- not folded into `gate_passed`. Constraint
   Tax: report the degradation as its own number. Add `input_health: {status: 'ok'|'dry_run'|'defaulted', defaulted_fields: [...]}` to `research-gate.js:368-378`, which today has **no**
   input-health field at all.
3. **In the log line.** The idiom already exists: the `self_report_disagreed` WARNING at
   `:360-364`. Mirror it exactly. Note `enforceGate` must not log (it is pure); the driver
   logs.
4. **In the artifact (the addition the caller's list omits).** Per Anthropic harness-design the
   *file* is the durable handoff state; a log line is lost. `tierDefaulted` at `:83` -> `:106`
   is the **existing in-file precedent**: it tells the AGENT it is running defaulted and asks it
   to "state this assumption in the brief". Generalise that to all six fields so the brief
   itself carries the banner. Q/A then sees it during the harness-compliance audit.

### Change sites (re-derived, for the contract)

| File:line | What changes |
|---|---|
| `.claude/workflows/research-gate.js:71-88` | replace the block with a pure `parseArgs()`; classify A/B/C; keep the six fallbacks only for class A |
| `.claude/workflows/research-gate.js:106` | generalise the `tierDefaulted` disclosure idiom to every defaulted field |
| `.claude/workflows/research-gate.js:201-282` | `enforceGate(env, verification, opts)` -> accept `inputHealth`; blind => violation. Keep it PURE |
| `.claude/workflows/research-gate.js:355-364` | add the blind-run WARNING log, mirroring `self_report_disagreed` |
| `.claude/workflows/research-gate.js:368-378` | add `input_health` to the return object |
| `.claude/workflows/qa-verdict.js:25-29` | **the comment must be corrected or deleted** -- it authorises the defect and cites a remedy EviBound measures at 100% failure |
| `.claude/workflows/qa-verdict.js:30-40` | same classification; **empty `criteria` on a present-args launch is an error**, not a prompt hint |
| `scripts/qa/verify_research_gate_workflow.mjs:47-60` | extend the appended export list, or (preferred) export a pure `parseArgs` and drive all 12 shapes through it -- **ESM caches by URL, so a module-scope free variable cannot be varied across cases without a fresh temp file per shape** |
| `scripts/qa/verify_research_gate_workflow.mjs:215-244` | add args mutants to section [7] using the existing `String.replace` + `loadModule` mechanism |
| (new) a checker for `qa-verdict.js` | **none exists today** -- only a regex assertion in `backend/tests/test_phase_75_20_qa_browser_grant.py:112` |
| `.claude/rules/research-gate.md:198` | the dry-run mode is **undocumented anywhere in `.md`**; specify it and the args contract |

**Runtime constraints binding the fix (confirmed at source):** no `fs`, no `process`, no static
`import` of any form, exactly one `export` (`export const meta`). The checker already asserts
all four at `:246-278` and will catch a violation -- but `node --check` will NOT, and neither
will the step's immutable command.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **10** (5 peer-reviewed, 5 official)
- [x] 10+ unique URLs total -- **54**
- [x] Recency scan (2024-2026) performed + reported -- 5 findings, incl. a **material correction** to `research-gate.js:20-24`
- [x] Full papers / pages read, not abstracts -- arXiv HTML chain used throughout; the one PDF went through pdfplumber per `research-gate.md` Step 3
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in scope (4 named + 5 grepped)
- [x] Contradictions / consensus noted -- incl. the EviBound 25%-vs-0% correction and the CWE-703 child-list correction
- [x] All claims cited per-claim
- [x] Gap CLOSED before returning -- see "Third variant" below. `probe-qa-tool-surface.js` has
      **no** args block; `harness-self-audit.js:23` has a **third, materially different** one.

---

## LATE FINDING -- a THIRD args variant in-repo, and it already fails loud

Grepped the two remaining workflow scripts (out of the caller's stated scope, flagged as a gap,
then closed). `.claude/workflows/probe-qa-tool-surface.js` has **no args block at all**.
`.claude/workflows/harness-self-audit.js:23` has a third variant, and it is **not** a copy:

```js
23  const a = (typeof args === 'object' && args) ? args : (typeof args === 'string' && args.trim() ? JSON.parse(args) : {})
```

**There is no `try`/`catch`.** Consequences, by input class:

- **Class B (malformed JSON string): this variant THROWS** an uncaught `SyntaxError` and the run
  dies loudly -- i.e. **the behaviour this step is proposing already exists in the same
  directory**, in a script nobody flagged as broken. That is strong in-repo evidence that the
  fail-loud behaviour is operationally tolerable on this runtime, not just theoretically
  preferable.
- **Class A (unbound / absent): it does NOT throw.** Both branches are `typeof`-guarded and
  short-circuit before dereferencing `args`, so the unbound-identifier case falls through to
  `{}` safely. **This is a working, in-repo demonstration of exactly the discriminator
  recommended above** (`typeof`-first, never a bare dereference) -- and it is why this file
  would survive the checker's import mechanism where the other two only survive via the catch.
- **Class C and the array case: still silently blind.** `typeof [] === 'object'` so an array is
  accepted here too, and a missing `step_id` still defaults.

**Bearing on the recommendation.** The repo now shows three different args idioms across three
workflow scripts, of which the *only* one that fails loud on malformed input is the one written
without a catch. This reframes the fix from "introduce a new, risky throw" to **"converge three
divergent idioms on the one that already behaves correctly, and finish it"** -- adding the
class-C/array/double-encode handling that all three lack, and the disclosure that none of them
has. Main should decide whether 86.17's scope includes `harness-self-audit.js:23`; leaving it
out preserves a fourth divergent idiom, but including it widens the diff beyond the two files
the step names.

---

## Envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 10,
  "snippet_only_sources": 44,
  "urls_collected": 54,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 5,
    "dry": false
  },
  "brief_path": "handoff/current/research_brief_86.17.md",
  "gate_passed": true
}
```
