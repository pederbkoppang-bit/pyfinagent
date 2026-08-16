# Research Brief -- phase-86.90

**Topic:** Silent type-coercion of structured input in LLM prompt-template
construction -- how JS string concatenation turns a nested object into the
literal `[object Object]`, why the defect is INVISIBLE FROM THE OUTPUT (an LLM
judge reconstructs the missing context and still returns a plausible verdict),
and the engineering patterns that make it fail loudly.

**Tier:** moderate (caller-specified). **Audit-class:** YES (loop-until-dry, K=2).
**Started:** 2026-08-16.

---

## ENVELOPE (born inert -- phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 12,
  "snippet_only_sources": 33,
  "urls_collected": 45,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "coverage": {
    "audit_class": true,
    "rounds": 6,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "gate_passed": true
}
```

**Sources read in full (the 12 URLs claimed in the envelope):**

1. https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Addition
2. https://typescript-eslint.io/rules/restrict-template-expressions/
3. https://typescript-eslint.io/rules/restrict-plus-operands/
4. https://arxiv.org/html/2508.06225v2
5. https://arxiv.org/html/2502.06329
6. https://code.claude.com/docs/en/workflows
7. https://arxiv.org/html/2607.12885
8. https://arxiv.org/html/2608.01000
9. https://www.martinfowler.com/ieeeSoftware/failFast.pdf
10. https://www.rfc-editor.org/rfc/rfc9413.html
11. https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/
12. https://arxiv.org/html/2607.19449v1

---

## Status log (append-only)

- [t0] Brief created; envelope written INCOMPLETE. Read
  `.claude/agents/researcher.md` + `.claude/rules/research-gate.md` in full.
- [t1] Internal inventory + measured `node -e` reproduction of the coercion.
- Next: external round 1.

---

## Internal code inventory (Explore half)

### The defect seam -- LOCALISED TO THE PROMPT TEMPLATE, not to marshalling

| File | Lines | Role | Status |
|---|---|---|---|
| `.claude/workflows/qa-verdict.js` | 107 | `const evidence = a.evidence \|\| '<default string>'` -- accepts ANY type; `{}` is truthy so an object passes straight through | **DEFECT SOURCE (no type check)** |
| `.claude/workflows/qa-verdict.js` | 108 | `const extra = a.extra \|\| ''` -- same, ANY type | **DEFECT SOURCE** |
| `.claude/workflows/qa-verdict.js` | 163 | `'EVIDENCE / FILES TO READ: ' + evidence` -- **`+` concatenation** | **THE COERCION SITE** |
| `.claude/workflows/qa-verdict.js` | 164 | `extra ? ('\nADDITIONAL CONTEXT: ' + extra) : ''` -- **`+` concatenation**; `{}` truthy so the branch is taken | **THE COERCION SITE** |
| `.claude/workflows/qa-verdict.js` | 85-99 | `classifyArgs` -- validates the ARGS ENVELOPE (must reduce to a plain object carrying `step_id`); does **NOT** validate individual FIELD types | Guard exists, stops one seam short |
| `.claude/workflows/qa-verdict.js` | 105 | `const criteria = Array.isArray(a.criteria) ? a.criteria : []` -- the ONLY field with a type check; a non-array silently becomes `[]` (fail-safe, not fail-fast) | Partial precedent, silent |
| `.claude/workflows/qa-verdict.js` | 106 | `verification_command` -- `\|\|` default, no type check | Same class, unexercised |
| `.claude/workflows/qa-verdict.js` | 153, 161 | `'     ' + verificationCommand`, `criteria.map(c => '  ' + (i+1) + '. ' + c)` -- each element `+`-concatenated | Same class; a nested-object criterion would also stringify |
| `.claude/workflows/research-gate.js` | 146 | `const topic = a.topic \|\| '<default>'` -- no type check | **SAME DEFECT CLASS** |
| `.claude/workflows/research-gate.js` | 207 | `const internalScope = a.internal_scope \|\| a.internalScope \|\| '<default>'` -- no type check | **SAME DEFECT CLASS** |
| `.claude/workflows/research-gate.js` | 236, 245 | `'OBJECTIVE: ' + topic`, `'INTERNAL SCOPE: ' + internalScope` -- `+` concatenation | **THE COERCION SITE (mirror)** |
| `.claude/workflows/research-gate.js` | 107-139 | `classifyArgs` -- richer than the Q/A copy (Class A/B/C taxonomy, `describe()` preview, explicit "DO NOT REPAIR NEAR-MISSES") but still envelope-level only | Guard exists, stops one seam short |
| `.claude/workflows/research-gate.js` | 126-128 | `if (typeof v === 'string') { ... JSON.parse(v) }` -- the rail ALREADY anticipates that the runtime may hand `args` over as a JSON **string** | Evidence about marshalling |
| `.claude/workflows/research-gate.js` | 202-206 | `tier` -- the ONE field with a value-domain check (`VALID_TIERS`), and it **fails closed** on unsupported | **The in-repo precedent for the right fix** |

`research-gate.js` is therefore **NOT clean by construction** -- it has the same
`+`-with-unvalidated-field shape at `:236`/`:245`. It merely has not been hit,
because callers pass `topic`/`internal_scope` as strings. (The step's criterion
demands this be shown by EXECUTION, not by reading source -- that is a GENERATE
task, not a research one; this brief supplies the anchors.)

### Measured reproduction (non-mutating, `node -e`, 2026-08-16)

```
CONCAT   : EVIDENCE / FILES TO READ: [object Object]
TEMPLATE : EVIDENCE: [object Object]
JSONSTR  : EVIDENCE: {"scope":"a","handoff":["b"],"changed_files":["c"]}
ARRAY    : X: a,b
NULL     : X: null | UNDEF: X: undefined
NULLPROTO: THROWS: TypeError: Cannot convert object to primitive value
toStringOverride: X: CUSTOM
SymbolToPrimitive: X: HINT=default
truthiness of {}: true -> a.extra || "" keeps the object
```

Five load-bearing facts from that run:

1. **Template literals do NOT save you.** `` `${obj}` `` yields the identical
   `[object Object]`. Any fix phrased as "switch to template literals" is a
   no-op. (Contrast the `JSONSTR` line: `JSON.stringify` is the behaviour
   change.)
2. **Arrays coerce to a comma-join, not `[object Object]`** -- `["a","b"]`
   becomes `a,b`. So an array-shaped `evidence` would degrade *silently and
   plausibly* (looks like a real list, loses all nesting) and would NOT have
   been caught by the Q/A noticing the literal string. This widens the blast
   radius beyond object-shaped inputs.
3. `null`/`undefined` stringify to `"null"`/`"undefined"` -- but the `||`
   defaults at `:107`/`:108` intercept both first, so those are the ONE shape
   the current code handles.
4. **A null-prototype object THROWS** `TypeError: Cannot convert object to
   primitive value` -- proof that JS *can* fail loudly here; the silence is a
   property of `Object.prototype.toString`, not of the `+` operator.
5. `Boolean({}) === true`, so `a.extra || ''` at `:108` does **not** filter an
   object; the `extra ? ...` guard at `:164` is likewise satisfied by any
   object, including `{}`.

---

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| 1 | https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Addition | 2026-08-16 | Official doc (MDN) | WebFetch, full page | "Addition coerces the expression to a *primitive*, which calls `valueOf()` in priority; on the other hand, template literals and `concat()` coerce the expression to a *string*, which calls `toString()` in priority." Also: "You are advised to not use `"" + x` to perform string coercion." |
| 2 | https://typescript-eslint.io/rules/restrict-template-expressions/ | 2026-08-16 | Official doc (lint rule) | WebFetch, full page | Default `toString()` produces `"[object Object]"`, which "is often not what was intended." `allowArray` defaults to **false** and `allowAny` defaults to **true**. Requires type information. |
| 3 | https://typescript-eslint.io/rules/restrict-plus-operands/ | 2026-08-16 | Official doc (lint rule) | WebFetch, full page | "adding values that are not the same type and/or are not the same primitive type is often a sign of programmer error"; explicitly names `"[object Object]"` as the risk. `allowAny` defaults **true**. |

| 4 | https://arxiv.org/html/2508.06225v2 | 2026-08-16 | Preprint (arXiv) | WebFetch, HTML full text | "predicted confidence levels significantly overstate actual correctness". GPT-4o ECE **39.25%**, Mistral-Nemo **74.22%**; judges "cluster predictions at high confidence levels (90-100%) but achieve accuracies well below the ideal calibration line." Does NOT study evidence-sufficiency directly -- see the honest-gap note below. |
| 5 | https://arxiv.org/html/2502.06329 | 2026-08-16 | Preprint (arXiv), FINANCE domain | WebFetch, HTML full text | FailSafeQA. **"Missing context posed the biggest challenge for almost all tested models"** (range 0.21-0.68). o3-mini, the MOST robust model (robustness 0.90), still **"fabricated information in 41% of tested cases"** when context was absent or irrelevant; reasoning models fabricate in **41% to 70%** of cases. |

| 6 | https://code.claude.com/docs/en/workflows | 2026-08-16 | **Official doc (Anthropic)** | WebFetch, full page | **DECIDES OBJECTIVE (4):** "Claude passes the list as **structured data**, so the script can call **array and object methods on `args` directly without parsing it first**. If `args` is omitted, the global is `undefined` inside the script." Also: "No module loading: a script that contains `import()` fails before the run starts" and "No direct filesystem or shell access from the workflow itself". |
| 7 | https://arxiv.org/html/2607.12885 | 2026-08-16 | Preprint (arXiv) | WebFetch, HTML full text | "LLM judges tend to over-credit incorrect answers in the absence of a reference answer." Verdict flips No-Reference -> Reference-Visible range **0.09 to 0.85**; "correctness scores are highest in the NR setting and decrease when the reference is added." Reference-informed verdicts "align more closely with human judgments." |
| 8 | https://arxiv.org/html/2608.01000 | 2026-08-16 | Preprint (arXiv) | WebFetch, HTML full text | **THE EPISTEMICS OF THIS DEFECT.** "an over-inclusion is a written token a reviewer can challenge, while a missing member is an absence whose discovery is the authoring problem itself"; "Omission is witness-blind to local review"; models detect planted over-inclusions **66-7x more often** than planted omissions; production deployment of 43,227 items "fails omission-first at **10:1**". Mitigation: "discard any authored verifier that rejects a **known-correct probe**" (false rejection 58-92% -> <=5%). |
| 9 | https://www.martinfowler.com/ieeeSoftware/failFast.pdf | 2026-08-16 | **Peer-reviewed (IEEE Software 21(5):21-25, Shore 2004)** | `curl` + `pypdf` extraction (5 pages, 13,482 chars) -- NOT a WebFetch PDF summary | **THE EXACT SHAPE OF THIS DEFECT.** Shore's worked example is a config property read that returns a **default value** when absent: "For the code that returns a default value, everything will seem fine. But when customers start using the software, they'll encounter mysterious slowdowns. Figuring it out could take days of hair pulling." And: "Some people recommend making your software robust by working around problems automatically. This results in the software 'failing slowly.'" And: "Assertions shine in their ability to flush out problems in the **seams of the system**." |

| 10 | https://www.rfc-editor.org/rfc/rfc9413.html | 2026-08-16 | **Official standard (IETF)** | WebFetch, full page | s4.2: "Tolerating unexpected input instead **conceals problems, making it harder, if not impossible, to fix them later**." s6: "Hiding the consequences of protocol variations encourages the hiding of issues, which can conceal bugs and make them difficult to discover." s5.1 (virtuous intolerance): "Choosing to generate fatal errors for unspecified conditions instead of attempting error recovery can ensure that faults receive attention"; "A notification for a fatal error is best sent as **explicit error messages to the entity that made the error**." |
| 11 | https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/ | 2026-08-16 | Authoritative blog (King) -- **year-less canonical prior art** | WebFetch, full page | "the difference between validation and parsing lies almost entirely in **how information is preserved**"; "a parser is just a function that consumes less-structured input and produces more-structured output"; shotgun parsing = "parsing and input-validating code is mixed with and spread across processing code"; "**Get your data into the most precise representation you need as quickly as you can. Ideally, this should happen at the boundary of your system, before *any* of the data is acted upon.**" |

| 12 | https://arxiv.org/html/2607.19449v1 | 2026-08-16 | Preprint (arXiv) | WebFetch, HTML full text | **THE CLOSEST ANALOGUE IN THE LITERATURE.** Injects 4 silent-failure payload profiles (`empty_valid`, `malformed`, `null_field`, `truncated`) behind HTTP 200 across 12 tool stubs. Fabrication Rate (FAR) = "the agent presents data derived from or assumed from rT **without acknowledging that rT is empty or malformed**" = **56.6%** of 396 valid trajectories; Honest Surrender 43.2%. Per model: GPT-4o 55.0% FAR, Llama-3.1-8B **74.6%**. Detection: a "payload-response misalignment heuristic" that "operates at the **message boundary** and requires no access to model internals ... logs every (tool payload, final response) pair." |

*(12 sources read in full; table closed)*

---

## Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://es5.github.io/x9.html | Spec mirror | Superseded by MDN + the measured `node` run |
| https://dev.to/aman_singh/abstract-operations-the-key-to-understand-coercion-in-javascript-453i | Community | Community tier; MDN covers it authoritatively |
| https://exploringjs.com/deep-js/ch_type-coercion.html | Authoritative blog | **Attempted, HTTP 404** -- recorded as an attempt, not a read |
| https://www.jamesshore.com/v2/blog/2004/fail-fast | Blog landing page | **Attempted**; returned only a summary stub. The real article is the IEEE PDF, read in full as source #9 |
| https://dl.acm.org/doi/10.1109/MS.2004.1331296 | Peer-reviewed (paywall) | Same article as #9; the open PDF was used instead |
| https://arxiv.org/pdf/2606.07874 | Preprint | Safety-judge context rigidity; adjacent, not the mechanism |
| https://www.sciencedirect.com/science/article/pii/S2666675825004564 | Peer-reviewed survey | Survey; primary sources #7/#8/#12 read instead |
| https://arxiv.org/pdf/2412.05579 | Preprint survey | Same reason |
| https://arxiv.org/pdf/2508.03686 | Preprint (CompassVerifier) | Verifier robustness; adjacent |
| https://arxiv.org/pdf/2601.08843 | Preprint | Rubric-conditioned grading uncertainty; adjacent |
| https://arxiv.org/pdf/2602.21044 | Preprint (LogicGraph) | Source of the "filling in the gaps" framing; #8 covers the mechanism with harder numbers |
| https://arxiv.org/pdf/2602.20629 | Preprint (QEDBENCH) | Lenient-auditor finding; same class as #7 |
| https://arxiv.org/pdf/2405.04727 | Preprint | LLMs patching missing relevance judgments; adjacent |
| https://arxiv.org/pdf/2604.16383 | Preprint | Judge/clinician disagreement on completeness; adjacent |
| https://arxiv.org/html/2607.20982v1 | Preprint (GuardianAgentBench) | Guardrail benchmark; #12 is the on-point one |
| https://arxiv.org/html/2607.16215v1 | Preprint (RAIL Guard) | Remediation framework; not the detection mechanism |
| https://github.com/promptfoo/promptfoo/issues/1866 | Community (issue) | "Silent failure by promptRubric not imputing variables" -- exact analogue in another tool, but issue-tier |
| https://deepchecks.com/llm-production-challenges-prompt-update-incidents/ | Industry blog | Prompt-change incident rates; unsourced figures |
| https://medium.com/@minogin/behavioral-drift-silent-bugs-in-llm-workflows-11169ee8e66a | Community | "Behavioral drift"; anecdotal |
| https://towardsdatascience.com/prompt-engineering-fails-quietly-prompt-regression-is-why/ | Industry blog | Prompt regression framing; no measurements |
| https://futureagi.com/blog/prompt-regression-testing-2026/ | Industry blog | Same |
| https://www.promptquorum.com/prompt-engineering/prompt-audit-and-regression-risk | Industry blog | Same |
| https://www.statsig.com/perspectives/slug-prompt-regression-testing | Industry blog | Same |
| https://qaskills.sh/blog/prompt-regression-testing-guide-2026 | Industry blog | Same |
| https://testrigor.com/blog/what-is-prompt-regression-testing/ | Vendor blog | Vendor tier |
| https://en.wikipedia.org/wiki/Robustness_principle | Encyclopedia | RFC 9413 (#10) is the authoritative treatment |
| https://programmingisterrible.com/post/42215715657/postels-principle-is-a-bad-idea | Blog | **Adversarial-side prior art**; RFC 9413 supersedes it as a standard |
| https://blog.beeminder.com/postel/ | Blog | Same |
| https://www.aijsons.com/blog/zod-json-schema-validation-ai/ | Vendor blog | Zod ruled out: the Workflow runtime forbids module loading (source #6) |
| https://dev.to/pavelespitia/zod-llms-how-to-validate-ai-responses-without-losing-your-mind-4c5j | Community | Same |
| https://stacknotice.com/blog/zod-complete-guide-2026 | Blog | Same |
| https://github.com/bglow/JSTypeAsserter | Community (repo) | A dependency; forbidden by the runtime constraint |
| https://therelaymag.com/why-ui-shows-object-object-fix | Community | Restates MDN |
| https://code.claude.com/docs/llms.txt | Official index | Index only |

**URL tally: 12 read in full + 33 snippet-only = 45 unique URLs.**

---

## Search-query composition (three-variant discipline made visible)

| Variant | Queries actually run |
|---|---|
| **Current-year frontier (2026)** | `LLM-as-a-judge missing context confabulation confident verdict incomplete evidence robustness 2026`; `2026 LLM agent evaluation silent context loss prompt assembly defect detection guardrail`; `Zod JSON Schema validate at trust boundary parse don't validate runtime input validation 2026` |
| **Last-2-year window (2024-2025)** | `"[object Object]" bug postmortem production incident stringified object user facing 2024 2025`; `runtime type assertion plain JavaScript no dependencies throw on non-string template rendering 2025` |
| **Year-less canonical** | `"[object Object]" string concatenation ToPrimitive ECMAScript specification abstract operation`; `"fail fast" Jim Shore IEEE Software fail fast systems silent failure placeholder default`; `LLM judge "filling in the gaps" over-credit incomplete derivations missing premises grading`; `criticism of fail-fast input validation "be liberal in what you accept" Postel robustness principle defense counterargument`; `silent failure prompt template rendering bug LLM pipeline "fail fast" validation trust boundary`; `regression test for absent effect mutation testing prompt construction assert prompt contains rendered value`; `Claude Code Workflow tool script args agent() structured output docs` |

The year-less variant is what surfaced Shore 2004 (#9), RFC 9413 (#10) and
Parse-don't-validate (#11) -- none of which a year-locked query returned.

---

## Recency scan (last 2 years, 2024-2026)

**Performed.** Result: **four new findings in the window, all complementing
rather than superseding the canonical sources.**

1. **arXiv 2607.19449 (2026)** -- the single most on-point paper found. It is
   the only source that measures what happens when a *silently degraded payload*
   reaches an agent, rather than what happens when a judge is merely
   miscalibrated. **56.6% Fabrication Rate**; the agent "presents data derived
   from or assumed from rT without acknowledging that rT is empty or malformed."
2. **arXiv 2608.01000 (2026)** -- supplies the epistemics: omission is
   "witness-blind to local review", detected **66-7x less often** than
   over-inclusion, and a production corpus of 43,227 items "fails omission-first
   at 10:1."
3. **arXiv 2607.12885 (2026)** -- quantifies the leniency direction: verdicts
   flip up to **0.85** when the reference is restored, and scores are HIGHEST
   without it. So a judge deprived of evidence errs *generous*, not *harsh*.
4. **arXiv 2502.06329 (2025, finance)** -- missing context is the hardest
   perturbation for every model tested; even the most robust fabricates in
   **41%** of cases.

**Not superseded:** Shore 2004 (#9) and RFC 9413 (2023, #10) remain the
authoritative statements of the fail-fast argument; nothing in the 2024-2026
window contradicts them, and RFC 9413 is itself the IETF's formal
reconsideration of the "be liberal in what you accept" counter-position, so the
adversarial side is represented by a standards-track document rather than only
by blogs.

---

## Key findings

1. **`+` and template literals are equally lost for plain objects; only the
   *unrenderable* case differs.** MDN: "Addition coerces the expression to a
   *primitive*, which calls `valueOf()` in priority; on the other hand, template
   literals and `concat()` coerce the expression to a *string*, which calls
   `toString()` in priority." For a plain object both paths land on
   `Object.prototype.toString` -> `[object Object]` (measured above). They
   diverge only for objects whose `valueOf` throws (MDN's `Temporal` example:
   `"" + t` throws, `` `${t}` `` works). **A fix phrased as "use a template
   literal" is a no-op.** (MDN, accessed 2026-08-16)

2. **The silence is a property of `Object.prototype.toString`, not of `+`.** A
   null-prototype object throws `TypeError: Cannot convert object to primitive
   value` (measured). JS *can* fail loudly here; the default just doesn't.

3. **The industry-standard defences are lint rules that this codebase cannot
   currently run.** `restrict-plus-operands` exists precisely because "adding
   values that are not the same type ... is often a sign of programmer error"
   and names `"[object Object]"` as the risk; `restrict-template-expressions`
   defaults `allowArray: false`. **Both require type information**, i.e. a
   TypeScript project graph -- and both default `allowAny: true`, so even under
   TS an `any`-typed `args.evidence` would pass. (typescript-eslint docs,
   accessed 2026-08-16)

4. **The Workflow runtime does NOT stringify `args`.** Official Anthropic docs:
   "Claude passes the list as **structured data**, so the script can call array
   and object methods on `args` directly **without parsing it first**."
   Therefore the coercion **cannot** be in args marshalling -- it is in the
   prompt template, which is exactly where the source anchors put it
   (`qa-verdict.js:163`/`:164`). This is documentary, not executed, evidence;
   the step's criterion 2 still requires execution, and the existing
   `runDriver()` harness supplies it. (code.claude.com/docs/en/workflows,
   accessed 2026-08-16)

5. **The runtime forbids the obvious fix.** Same doc: "No module loading: a
   script that contains `import()` fails before the run starts." **Zod, Ajv, and
   every JSON-Schema library are unavailable inside the Workflow script.** The
   validation must be hand-rolled -- which the scripts already do, in
   `classifyArgs`.

6. **Shore's canonical fail-fast example is this exact code shape.** His
   worked example is a config read that returns a default when the property is
   absent; the fail-fast version throws naming the file path. "For the code that
   returns a default value, everything will seem fine. But when customers start
   using the software, they'll encounter mysterious slowdowns. Figuring it out
   could take days of hair pulling." `qa-verdict.js:107`'s
   `a.evidence || '<default string>'` is that method. (Shore, *IEEE Software*
   21(5), 2004)

7. **Shore also names the right place for the check.** "Assertions shine in
   their ability to flush out problems in the **seams of the system**. Use them
   to show mistakes in how the rest of the system interacts with your method."
   The args boundary IS the seam between Main (caller) and the rail.

8. **Silent tolerance is a standards-level anti-pattern, and the deciding
   variable is detectability.** RFC 9413 s4.2: "Tolerating unexpected input
   instead conceals problems, making it harder, if not impossible, to fix them
   later." s6: "Hiding the consequences of protocol variations encourages the
   hiding of issues." s5.1: "Choosing to generate fatal errors for unspecified
   conditions instead of attempting error recovery can ensure that faults
   receive attention," delivered as "explicit error messages **to the entity
   that made the error**." *(`research-gate.js:172-184` already reasons from
   exactly this RFC for the `tier` field -- the doctrine is in-house, just not
   applied to `evidence`.)*

9. **Validate at the boundary, once, and preserve the result.** "Get your data
   into the most precise representation you need as quickly as you can. Ideally,
   this should happen at the boundary of your system, before *any* of the data
   is acted upon." Shotgun parsing -- "parsing and input-validating code ...
   mixed with and spread across processing code" -- is the failure mode where a
   field is checked in some paths and not others. `classifyArgs` is the right
   boundary; it just doesn't descend into fields. (King, 2019)

10. **When degraded input reaches an agent, fabrication is the MODAL outcome,
    not the tail.** 2607.19449 injects HTTP-200 payloads that are empty, null,
    malformed or truncated and measures **FAR = 56.6%** vs Honest Surrender
    43.2% across 396 trajectories; the weakest model fabricates in **74.6%**.
    Corroborated cross-domain by FailSafeQA (finance): the *most robust* model
    still "fabricated information in 41% of tested cases", reasoning models
    41-70%. **The 86.85 Q/A that reported its own missing input is the 43%
    case, not the norm** -- and that is why the defect survived 23 spawns.

11. **The error direction is LENIENT.** Judges without a reference "over-credit
    incorrect answers"; restoring the reference flips verdicts by up to **0.85**
    and *lowers* correctness scores. (2607.12885) So a Q/A graded on a
    reconstructed evidence set is biased toward PASS -- which is the direction
    that matters for a gate.

12. **The absence leaves no artifact, which is why review cannot catch it.**
    "an over-inclusion is a written token a reviewer can challenge, while a
    missing member is an absence whose discovery is the authoring problem
    itself"; "Omission is witness-blind to local review." (2608.01000) This is
    the formal statement of why reading the verdict can never reveal this bug --
    the verdict is well-formed and often correct.

13. **`JSON.stringify` is NOT automatically a loud renderer.** Measured:
    `undefined`-, function-, and Symbol-valued keys are **silently dropped**;
    `Map`/`Set` collapse to `{}`; a null-prototype object becomes `{}`; only
    `BigInt` and circular references throw. **Swapping `+` for `JSON.stringify`
    substitutes one silent degradation for five.** The step's criterion --
    "replacing it with a different silent fallback does not close it" -- bites
    here specifically.

14. **The detection primitive the literature recommends already exists in this
    repo.** 2607.19449's heuristic "operates at the **message boundary** and
    requires no access to model internals ... logs every (tool payload, final
    response) pair." The Workflow runtime already persists every spawn's
    received prompt to `agent-*.jsonl` -- which is how the 23-spawn blast radius
    above was measured. A standing check over that corpus is a
    zero-instrumentation monitor.

15. **2608.01000's mitigation is the mutation-test discipline, stated
    independently.** "discard any authored verifier that rejects a
    **known-correct probe**" (false rejection 58-92% -> <=5%). Read against
    criterion 6 ("mutation-tested with the control observed GREEN first"): the
    control must be observed green BEFORE the mutant is judged, or the guard's
    RED is uninterpretable.

**Honest gap:** source #4 (arXiv 2508.06225) was fetched for the
overconfidence-under-missing-evidence link and **does not deliver it** -- it
studies calibration only and "does not explicitly address how judges behave when
evidence is incomplete or missing." It is retained for the calibration numbers
(ECE 39.25% / 74.22%) and explicitly NOT cited for the missing-evidence claim,
which rests on #5, #7, #8 and #12.

---

## Consensus vs debate (external)

**Consensus (strong, cross-domain, four independent literatures):**
- Silent degradation at a trust boundary is worse than a crash -- software
  engineering (#9), protocol design (#10), type-driven design (#11).
- LLM judges deprived of context fabricate rather than abstain, at rates between
  41% and 74.6% (#5, #12), and err in the LENIENT direction (#7).
- Omissions are structurally harder to detect than commissions (#8).

**Debate:**
- *Postel's robustness principle* is the standing counter-argument for lenient
  input handling. It is not fringe -- but RFC 9413 is the IETF's own
  reconsideration, and its resolution is not "be strict everywhere": it is
  **fail closed OR proceed with a machine-readable signal**, with the deciding
  variable being *whether the caller can detect the substitution from the
  response*. Both dispositions are legitimate; a *silent* third option is
  endorsed by no source found in six rounds.
- *How strict should a renderer be?* typescript-eslint ships `allowArray`,
  `allowNumber`, `allowBoolean` as knobs, conceding that not every non-string is
  a bug. There is no authority for a single right answer; the project-specific
  answer follows from what `evidence` MEANS here (a pointer set the evaluator is
  told to read), not from the literature.

---

## Pitfalls (from literature + measurement)

1. **"Use a template literal"** -- no-op (finding 1).
2. **"Just `JSON.stringify` it"** -- trades one silent loss for five (finding 13).
3. **Fixing only `evidence`** -- `extra` (`:164`), `verificationCommand`
   (`:153`), each `criteria` element (`:161`), and research-gate's `topic`
   (`:236`) / `internalScope` (`:245`) are the same shape. The repo's own
   recorded lesson is "guards stop one seam short"; a fix at one call site
   repeats it.
4. **Detecting via the `[object Object]` marker alone** -- an **array**-shaped
   field renders as `a,b` with no marker (measured). The blast-radius number
   above is a floor for that reason.
5. **A source-scan regression guard** -- `verify_research_gate_workflow.mjs:60-76`
   records that source-scan assertions were defeated **twice** by comment
   tokens; "structural is not semantic." The guard must drive the script and
   inspect the prompt handed to the `agent()` stub.
6. **A guard whose control was never observed green** -- both the repo
   (`verify_workflow_args_boundary.mjs:394-400`) and #8 insist the
   known-correct probe passes first.
7. **A no-match `String.replace` in the mutation step** -- already handled in
   `verify_workflow_args_boundary.mjs:281` ("a no-match replace looks identical
   to success"); reuse the anchor-uniqueness check.
8. **Treating the docs quote as execution evidence** -- finding 4 is
   documentary. Criterion 2 asks for the layer to be localised "with evidence";
   the `runDriver()` harness is how.

---

## Application to pyfinagent (external findings -> file:line anchors)

| Finding | Anchor | Implication for PLAN (Main owns the decision) |
|---|---|---|
| 4, 5 | `.claude/workflows/qa-verdict.js:163-164` | The layer is the **prompt template**. No library is available; the check must be hand-rolled. |
| 6, 7, 8, 9 | `qa-verdict.js:107-108`; `research-gate.js:107-139` | The house idiom for this already exists: `classifyArgs` + its `describe()` preview (`research-gate.js:108-114`) + the explicit "DO NOT REPAIR NEAR-MISSES" comment (`:102-106`). Extending it from envelope-level to field-level is a continuation of the existing design, not a new one. |
| 8 (RFC 9413 s5.1) | `research-gate.js:172-184` | The in-repo precedent -- the `tier` field **fails closed** on an unsupported value AND reports it in the response. The comment there already states the deciding rule: "The deciding variable is whether the caller can DETECT the substitution from the response." Applying the same rule to `evidence` is consistent, not novel. |
| 13 | (new code) | If any rendering of structured input is chosen, it must be paired with a loud path for what it cannot render. A bare `JSON.stringify` is a criterion-5 violation. |
| 14 | `~/.claude/projects/.../subagents/workflows/*/agent-*.jsonl` | The blast-radius probe is re-runnable and needs no new instrumentation. Candidate standing check. |
| 5 (guard) | `scripts/qa/verify_workflow_args_boundary.mjs:341-372` (`runDriver`), `:389-391` (prompt-content assertion idiom), `:394-400` (control), `:403-412` (mutation + anchor uniqueness) | The regression guard is an **addition to section [5]** of an existing checker: add a field-type shape to `SHAPES` (`:54-65`, which today varies only the envelope), drive both scripts, and assert `!/\[object Object\]/.test(spawns[0].prompt)`. Mutation: revert the new field guard; the new assertion must go RED while the control stays GREEN. |
| 11, 12 | -- | Blast-radius dispositions must be reasoned about in the LENIENT direction: a spawn that graded on a reconstructed evidence set is biased toward PASS. **86.86 is confirmed affected** (`wf_b1747d75-eec`). |
| -- | `git log -S` -> `ccddeff4` (phase-71.1) | Provenance is a single commit; the defect is coeval with the rail being made first-class. |

**Scope note for Main:** criterion 7 ("verdict semantics are UNCHANGED") is
naturally satisfied by a fix at the args boundary -- throwing before `agent()` is
called can only PREVENT a spawn, never convert a non-PASS into a PASS. That is
the same argument `verify_workflow_args_boundary.mjs` section [5] already makes
for the blind-run early return.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **12**
      (11 via WebFetch; #9 via `curl` + `pypdf` per the PDF chain in
      `.claude/rules/research-gate.md`, because it is a binary PDF)
- [x] 10+ unique URLs total -- **45** (12 read-in-full + 33 snippet-only)
- [x] Recency scan (last 2 years) performed + reported -- 4 findings, section above
- [x] Full papers / pages read, not abstracts -- every row states how
- [x] file:line anchors for every internal claim -- all inventory rows anchored

Soft checks:
- [x] Internal exploration covered every relevant module -- both Workflow
      scripts, both checkers, the masterplan entry, and the receipt corpus
- [x] Contradictions / consensus noted -- Postel vs fail-fast; the #4 honest gap
- [x] All claims cited per-claim
- [ ] **Tier length overrun, disclosed:** `moderate` nominates <=700 words. The
      prose is held near that; the tables (inventory, blast radius, source
      records) exceed it and are evidence, not prose. Not padded -- every row is
      a measurement or a quote. Flagged rather than truncated, because
      truncating the blast-radius table would delete the answer to criterion 4.

### Audit-class coverage loop (K=2)

| Round | Angle | New read-in-full findings |
|---|---|---|
| 1 | ToPrimitive / lint rules / judge calibration + finance missing-context | 5 |
| 2 | Workflow args semantics / gap-filling / silent omissions / fail-fast canon | 4 |
| 3 | Trust-boundary validation (RFC 9413, parse-don't-validate) | 2 |
| 4 | 2026 agent-eval silent-context-loss | 1 (2607.19449) |
| 5 | **Adversarial**: Postel's-law defence of lenient input handling | **0 -- DRY** (all hits community-tier; RFC 9413 is the authoritative treatment and was already read) |
| 6 | Dependency-free JS runtime assertion + `[object Object]` incident postmortems | **0 -- DRY** (all hits community/vendor tier; the runtime's no-module-loading constraint rules the libraries out anyway) |

`dry_rounds = 2 >= K_required = 2` -> `coverage.dry = true`.

---

---

## MEASURED BLAST RADIUS (internal, from the agents' own received prompts)

The Workflow runtime persists each spawned agent's transcript -- including the
verbatim prompt it received -- under
`~/.claude/projects/-Users-ford--openclaw-workspace-pyfinagent/<session>/subagents/workflows/<wf_runid>/agent-*.jsonl`.
That is the RECEIPT: it shows what the agent actually got, not what the caller
believed it sent. Measured 2026-08-16:

| Probe | Count |
|---|---|
| Agent transcripts whose prompt contains `EVIDENCE / FILES TO READ: [object Object]` | **23 files** (30 matching lines -- a prompt recurs across records, so the FILE count is the de-duplicated spawn count) |
| Agent transcripts containing the header `EVIDENCE / FILES TO READ:` at all (denominator) | **407** |
| Rate | **23 / 407 = 5.7%** of Q/A-rail spawns |
| `ADDITIONAL CONTEXT: [object Object]` | 11 occurrences |
| `OBJECTIVE: [object Object]` (research-gate.js) | **0** of **75** spawns carrying `OBJECTIVE: ` |
| `INTERNAL SCOPE: [object Object]` (research-gate.js) | **0** of **72** spawns carrying `INTERNAL SCOPE: ` |

**Affected step-ids** (extracted from `masterplan step <id>` in the same prompt):

| Step | Affected spawns |
|---|---|
| **86.86** | 1 (`wf_b1747d75-eec`) -- **the named candidate in the step's audit_basis IS AFFECTED** |
| 86.85 | 4 (`wf_b12cf244-d30`, `wf_879d28f2-9fc`, `wf_769e1502-fd8`, `wf_5f5ce4b6-266`) |
| 86.74 | 1 (`wf_8c3730a1-32e`) |
| 86.38 | 4 (`wf_aa7f8c4d-8bf`, `wf_468907a8-b13`, `wf_2881574d-de2`, `wf_13a30a9d-33d`) |
| 86.34 | 2 (`wf_9d7e0010-66f`, `wf_97a608dd-2a4`) |
| 86.29 | 3 (`wf_fdc81179-861`, `wf_d4e2e794-567`, `wf_2675058b-ab3`) |
| 86.25 | 1 (`wf_8a3969ee-ae0`) |
| 86.21 | 2 (`wf_e66ad533-e61`, `wf_982cd319-493`) |
| 85.5 | 4 (`wf_faf8bbd4-4af`, `wf_7e809394-ae8`, `wf_4c70d707-88e`, `wf_46e96d67-b24`) |
| 86.90 | 1 (`wf_9bd7e233-f38`) -- also the sole source of the third header shape `EVIDENCE: [object Object]` (2 occurrences), i.e. a probe spawn for THIS step, not a separate production call site |

**Caveats a contract must carry** (do not launder these into a clean number):
1. This is a **floor**, not a census. It counts transcripts still on disk under
   this project's session dirs; pruned/rotated sessions are invisible, and the
   407 denominator is subject to the same loss. Report it as ">=23 of 407
   surviving receipts".
2. `grep -l` (23) vs `grep -h` (30) differ because one prompt appears in
   multiple JSONL records per transcript. **23 is the spawn count**; using 30
   would double-count.
3. The measurement covers the `evidence`/`extra` headers only. Per the ToPrimitive
   facts above, an **array-shaped** field would render as `a,b` and leave NO
   `[object Object]` marker -- so this probe cannot detect that variant at all.
   Its true rate is unmeasured.

**Provenance.** `git log -S "EVIDENCE / FILES TO READ: ' + evidence"` returns a
single commit, `ccddeff4` *"phase-71.1: codify Workflow structured-output as the
FIRST-CLASS Q/A + Researcher launch (Agent-tool = fallback)"*. The concatenation
has been present since the rail became first-class; it was never introduced by a
later edit.

**research-gate.js verdict (measured, not read):** vulnerable **by construction**
(`:236` / `:245` are the same `+`-with-unvalidated-field shape) but **clean in
practice** -- 0 of 75 and 0 of 147 combined receipts carry the marker, because
every caller has passed `topic`/`internal_scope` as strings. The step's criterion
("a clean result must be shown by execution, not by reading the source") is
satisfiable by driving `research-gate.js` through the existing
`runDriver()` harness with `topic: {a:1}`; that is a GENERATE task.

---

---

## Internal code inventory, part 2 -- THE GUARD INFRASTRUCTURE ALREADY EXISTS

This is the most consequential internal finding for objective (5): the repo
already has a **behavioural** driver harness that records the prompt handed to
`agent()`. A regression guard for this defect is an ADDITION to existing
machinery, not new machinery.

| File | Lines | Role | Status |
|---|---|---|---|
| `scripts/qa/verify_workflow_args_boundary.mjs` | 1-421 (phase-86.17) | Drives **BOTH** Layer-3 scripts across arg shapes; has sections [1] REPRODUCE-from-git, [2] FIXED, [3] BLIND-CANNOT-PASS, [4] MUTATION, [5] FULL DRIVER | **BLIND TO THIS DEFECT -- see below** |
| `scripts/qa/verify_workflow_args_boundary.mjs` | 54-65 (`SHAPES`) | The 10 shapes: `plain-object`, `valid-json-string`, `malformed-json-string`, `json-string-raw-newline`, `array`, `scalar-number`, `absent`, `double-encoded-json`, `empty-string`, `object-without-step_id` | **Every shape varies the ARGS ENVELOPE. NOT ONE varies a FIELD's type.** Every "usable" shape passes `topic: "t"` -- a string. |
| `scripts/qa/verify_workflow_args_boundary.mjs` | 341-372 (`runDriver`) | Wraps the WHOLE script in `export default async function __run()`, injects stubs for `agent`/`phase`/`log`, and **records `{label, prompt}` for every spawn** | **THE HOOK FOR THE NEW GUARD** -- `r.spawns[0].prompt` is already captured |
| `scripts/qa/verify_workflow_args_boundary.mjs` | 389-391 | Existing precedent for a prompt-content assertion: `!r.spawns.some(s => /UNSPECIFIED/.test(String(s.prompt)))` | **The exact idiom to copy** for `/\[object Object\]/` |
| `scripts/qa/verify_workflow_args_boundary.mjs` | 394-400 | `CONTROL: a usable launch DOES spawn` -- guards against the checks passing vacuously | Precedent for the control-first discipline the step's criterion demands |
| `scripts/qa/verify_workflow_args_boundary.mjs` | 403-412 | MUTATION: `if (inputHealth.blind) {` -> `if (false) {`, anchor-uniqueness checked first (`n !== 1` -> explicit FAIL, "a no-match replace looks identical to success") | Precedent for mutation-testing the new guard |
| `scripts/qa/verify_research_gate_workflow.mjs` | 83-97 (`loadDriver`) + 98-104 (`driveRecording`) | Same wrap-the-whole-script trick; `const agentStub = async (prompt, opts) => { spawns.push({prompt, opts}); return null }` | Second copy of the same hook |
| `scripts/qa/verify_research_gate_workflow.mjs` | 60-76 (comment) | "Two successive Q/A passes defeated a **source-scan** assertion ... the property is BEHAVIOURAL (was `agent()` called?) ... **structural is not semantic**" | **Directly answers the step's criterion "a clean result must be shown by execution, not by reading the source"** |

**Why the existing checker missed this.** `verify_workflow_args_boundary.mjs`
exercises the boundary at the granularity of the *envelope* (`args` as a whole:
object / string / array / scalar / absent). The defect lives one level in, at
*field* granularity (`args.evidence` as object-vs-string). This is a textbook
instance of the repo's own recorded lesson "guards stop one seam short" -- and
it means the immutable verification command (`node --check`) and the two
existing checkers are ALL green today with the defect live.

---

---
