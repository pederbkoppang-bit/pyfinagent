# Research Brief -- phase-90.15

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported for information only).
**Started:** 2026-08-21. **Researcher:** Layer-3 researcher (Workflow rail).

## Envelope (born inert -- phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 28,
  "urls_collected": 37,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 3,
    "dry": false
  },
  "summary": "See the identical envelope at the tail of this brief for the full summary.",
  "brief_path": "handoff/current/research_brief_90.15.md",
  "gate_passed": true
}
```

## Objective

Four coupled questions for step 90.15:

1. **Guard/return seams and TOCTOU in object construction** -- where an invariant check
   must sit relative to the value it guards.
2. **Per-object "did MY keys leak" filters vs positive whole-key-set assertions** --
   why a negative per-source filter is structurally insufficient against a spread of an
   unknown FUTURE object, and when a positive key-set assertion is the right instrument.
3. **Ordering of specific-diagnosis vs catch-all validators** -- what published
   API/schema-validation practice says about error-message specificity.
4. **Prior art on preventing caller-authored fields from being presented as adjudicator
   output** in judge/evaluator pipelines.

## Search queries run (three-variant discipline)

(filled in below as searches execute)

Queries run (three-variant discipline visible across the tables below):
- Current-year frontier: `LLM judge evaluator output provenance separation 2026`;
  `structured output schema unknown key rejection 2026`
- Last-2-year window (2024-2026 recency scan): `TOCTOU validation object construction 2025`;
  `LLM-as-a-judge contamination caller-supplied fields 2025`
- Year-less canonical: `time-of-check-to-time-of-use TOCTOU validation object construction invariant`;
  `allowlist unknown properties validation additionalProperties JSON Schema unevaluatedProperties
  reject unknown keys`; `mass assignment over-posting`; `catch specific exception before general
  unreachable catch`; `problem details error specificity`

---

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|-----|----------|------|-------------|----------------------|
| 1 | https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html | 2026-08-21 | official/standards (OWASP) | WebFetch, full page | "Allowlist validation involves defining exactly what IS authorized, and by definition, everything else is not authorized." Denylists are "a massively flawed approach as it is trivial for an attacker to bypass such filters"; denylisting "can be useful as an additional layer of defense to catch some common malicious patterns, it should not be relied upon as the primary method." |

| 2 | https://cwe.mitre.org/data/definitions/367.html | 2026-08-21 | official/standards (MITRE CWE) | WebFetch, full entry | "The product checks the state of a resource before using that resource, but the resource's state can change between the check and the use in a way that invalidates the results of the check." Root mechanism, verbatim: "because both access() and fopen() operate on filenames rather than on file handles, there is no guarantee that the file variable still refers to the same file on disk when it is passed to fopen() that it did when it was passed to access()." Mitigations: "Ensure that locking occurs before the check, as opposed to afterwards, such that the resource, as checked, is the same as it is when in use"; "The most basic advice for TOCTOU vulnerabilities is to not perform a check before the use." |

| 3 | https://json-schema.org/understanding-json-schema/reference/object | 2026-08-21 | official docs (JSON Schema org) | WebFetch, full page | "Setting the `additionalProperties` schema to `false` means no additional properties will be allowed." And the structural limit, verbatim: "`additionalProperties` only recognizes properties declared in the same subschema as itself. So, `additionalProperties` can restrict you from 'extending' a schema using combining keywords such as allOf." The generalization: "`unevaluatedProperties` keyword is similar to `additionalProperties` except that it can recognize properties declared in subschemas" -- it collects "any properties that are successfully validated when processing the schemas and using those as the allowed list of properties." |

| 4 | https://owasp.org/API-Security/editions/2023/en/0xa3-broken-object-property-level-authorization/ | 2026-08-21 | official/standards (OWASP API Top 10 2023) | WebFetch, full entry | Consolidates 2019's "Excessive Data Exposure" (read) + "Mass Assignment" (write). Attack: attacker adds extra properties to a payload and there is "no validation...if the user should have access to the internal object property." Prevention, verbatim: "Avoid using generic methods such as `to_json()` and `to_string()`. Instead, cherry-pick specific object properties."; "If possible, avoid using functions that automatically bind a client's input into code variables, internal objects, or object properties."; "Implement a schema-based response validation mechanism...define and enforce data returned by all API methods." |

| 5 | https://pydantic.dev/docs/validation/latest/concepts/models/ | 2026-08-21 | official docs (Pydantic v2) | WebFetch, full page (via 301 from docs.pydantic.dev/latest/concepts/models/) | The permissive default is the finding: "By default, Pydantic models **won't error when you provide extra data**, and these values will simply be ignored." Three modes: `'ignore'` -- "Providing extra data is ignored (the default)"; `'forbid'` -- "Providing extra data is not permitted"; `'allow'` -- "Providing extra data is allowed and stored in the `__pydantic_extra__` dictionary attribute." Closing the set is opt-in: `model_config = ConfigDict(extra='forbid')`. NOTE (honest gap): the page does NOT argue *why* forbid is safer; it documents behaviour only. |

| 6 | https://cwe.mitre.org/data/definitions/396.html | 2026-08-21 | official/standards (MITRE CWE) | WebFetch, full entry | The ordering principle, verbatim: "catching a high-level class like Exception can obscure exceptions that deserve special treatment or that should not be caught at this point in the program"; "Catching an overly broad exception essentially defeats the purpose of a language's typed exceptions"; "Catching overly broad exceptions promotes complex error handling code that is more likely to contain security vulnerabilities." The demonstrative example contrasts one broad catch against "distinct catch blocks for each exception class". |
| 7 | https://www.rfc-editor.org/rfc/rfc9457.html | 2026-08-21 | official/standards (IETF RFC 9457) | WebFetch, full RFC | Specificity: "HTTP status codes cannot always convey enough information about errors to be helpful"; "Consumers MUST use the 'type' URI (after resolution, if necessary) as the problem type's primary identifier." **[TENSION -- reads AGAINST a naive key-set closure]** on unknown members: "Clients consuming problem details MUST ignore any such extensions that they don't recognize; this allows problem types to evolve and include additional information in the future." Security: "Risks include leaking information that can be exploited to compromise the system..." |

| 8 | https://cwe.mitre.org/data/definitions/441.html | 2026-08-21 | official/standards (MITRE CWE, "Confused Deputy") | WebFetch, full entry | The adjudicator-provenance principle, verbatim: "The product receives a request, message, or directive from an upstream component, but the product does not sufficiently preserve the original source of the request before forwarding the request to an external actor..."; "The request would appear to be coming from the product's system, not the attacker's system." Mitigation, verbatim: "The proxy core should not drop the identity of the initiator of the transaction. The immutability of the identity of the initiator must be maintained and should be forwarded all the way to the target." |
| 9 | https://arxiv.org/html/2604.05083v1 | 2026-08-21 | preprint (arXiv, 2026) | WebFetch, arXiv native HTML per the /html chain (no /pdf fetch) | Judge outputs are "highly sensitive to prompt design, language, and aggregation strategies, severely, which limits reproducibility"; "substantial variance across judges, datasets, and evaluation properties"; "Pairwise preferences can also be non-transitive, making rankings unstable"; "when a judge is not more accurate than the model being evaluated, debiasing offers only limited reductions in the need for ground-truth supervision." Architecture: "deterministic learned evaluators complement LLM-based judges with stable, efficient, and reproducible scoring." |


## Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|-----|------|--------------------------|
| https://en.wikipedia.org/wiki/Time-of-check_to_time-of-use | encyclopedia | superseded by CWE-367 (source 2), which is the normative entry |
| https://www.emergentmind.com/topics/time-of-check-to-time-of-use-toctou-races | aggregator | tertiary summary |
| https://deepstrike.io/blog/what-is-time-of-check-time-of-use-toctou | vendor blog | community tier; CWE-367 covers it |
| https://arxiv.org/pdf/2603.00476 | preprint 2026 (TOCTOU in browser-use agents) | agent-domain TOCTOU; adjacent but not object-construction; PDF URL -- would need /html chain |
| https://arxiv.org/pdf/2604.17511 | preprint 2026 (Atomic Decision Boundaries) | admissibility-at-execution-time; adjacent |
| https://dipsylala.github.io/FlawFixingGuidance/CWE-367/ | community remediation guide | derivative of CWE-367 |
| https://www.systemshardening.com/articles/cross-cutting/toctou-vulnerability-defences/ | vendor article | community tier |
| https://www.securityscientist.net/blog/12-questions-and-answers-about-toctou-time-of-check-to-time-of-use/ | blog | community tier |
| https://eastbaycyber.com/content/glossary-what-is-toctou/ | glossary | community tier |
| https://www.learnjsonschema.com/2020-12/unevaluated/unevaluatedproperties/ | reference | duplicates source 3 |
| https://opis.io/json-schema/2.x/object.html | implementation docs | duplicates source 3 |
| https://ajv.js.org/json-schema.html | implementation docs (Ajv) | `allErrors` is relevant to ordering but Ajv is not in this repo's stack |
| https://github.com/json-schema-org/json-schema-spec/issues/556 | spec issue thread | design rationale for unevaluatedProperties; source 3 carries the outcome |
| https://www.liquid-technologies.com/Reference/XmlStudio/JsonEditorNotation_UnevaluatedProperties.html | vendor reference | duplicates source 3 |
| https://medium.com/@smikulcik/bulletproof-your-input-validation-understanding-unevaluatedproperties-c6e7a0eb6ddd | blog | community tier |
| https://owasp.org/API-Security/editions/2019/en/0xa6-mass-assignment/ | official (OWASP 2019) | superseded by API3:2023 (source 4), which cites it |
| https://owasp.org/API-Security/editions/2019/en/0xa3-excessive-data-exposure/ | official (OWASP 2019) | superseded by API3:2023 (source 4) |
| https://arxiv.org/abs/2503.05965 | preprint 2025 (rating indeterminacy) | judge-validity, not output-provenance |
| https://arxiv.org/pdf/2606.19544 | preprint 2026 (Reliability without Validity) | judge agreement/bias, not the field-provenance question |
| https://arxiv.org/pdf/2603.05485 | preprint 2026 (bias-bounded judges) | bias bounding, off-question |
| https://arxiv.org/pdf/2509.19880 | preprint 2025 (Do Before You Judge) | judge self-reference, off-question |
| https://llm-as-a-judge.github.io/ | workshop hub | index page |
| https://en.wikipedia.org/wiki/LLM-as-a-Judge | encyclopedia | tertiary |
| https://www.evidentlyai.com/llm-guide/llm-as-a-judge | vendor guide | community/industry tier |
| https://deepeval.com/blog/llm-as-a-judge | vendor blog | community tier |
| https://www.braintrust.dev/articles/what-is-llm-as-a-judge | vendor blog | community tier |
| https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge | vendor docs | community tier |
| https://www.patronus.ai/llm-testing/llm-as-a-judge | vendor page | community tier |

## Recency scan (2024-2026) -- MANDATORY, PERFORMED

Searched `LLM-as-a-judge evaluator pipeline provenance separating judge output from
harness-computed fields 2026` and `"LLM-as-a-judge" separating deterministic checks from
judge scoring schema compliance 2025 survey arxiv`, plus a 2025-2026 pass on TOCTOU.

**Result: 3 new findings in the window that COMPLEMENT rather than supersede the canonical
security sources.**

1. **(2026)** arXiv:2604.05083 (read in full, source 9) establishes the *reason* the
   harness must compute fields deterministically rather than ask the judge for them: judge
   output is "highly sensitive to prompt design, language, and aggregation strategies...
   which limits reproducibility". This is the affirmative case for the caller computing
   `escalation` / `research_routing` / `severity_routing` at all -- but it says nothing
   about how to keep them *distinguishable* from judge output once computed. **That gap is
   the whole of 90.15**, and the recency scan found NO source that addresses it directly in
   an LLM-judge context. The applicable prior art is the security literature (sources 1, 2,
   4, 8), not the eval literature.
2. **(2026)** arXiv:2603.00476 "Atomicity for Agents: Exposing, Exploiting, and Mitigating
   TOCTOU Vulnerabilities in Browser-Use Agents" (snippet-only) shows TOCTOU has been
   re-derived for agent harnesses in the current window -- the class is live, not historical.
3. **(2023->present)** OWASP API3:2023 (source 4) merged the 2019 Mass Assignment and
   Excessive Data Exposure entries, i.e. the read-side and write-side of "caller properties
   bound into an object" are now treated as ONE weakness. That merge is exactly the 90.15
   shape: a caller field spread into the returned object is simultaneously an
   over-bind (write) and an over-expose (read, to Main).

**No 2024-2026 source supersedes CWE-367, CWE-396, CWE-441 or the OWASP allowlist rule.**
Those remain the governing prior art.

---

## Key findings

### F1. The 90.15 diagnosis is textbook CWE-367, and the fix matches the canonical remedy

The pre-fix shape checked `merged` and used a *different* object built one statement later.
CWE-367's own worked example is structurally identical: "because both access() and fopen()
operate on **filenames rather than on file handles**, there is no guarantee that the file
variable still refers to the same file on disk when it is passed to fopen() that it did when
it was passed to access()" (https://cwe.mitre.org/data/definitions/367.html, accessed
2026-08-21). `merged` was the filename; `{ ...merged, verdict_unmodified }` was the handle.
The remedy actually applied -- construct once, guard that object, return it unchanged -- is
CWE-367's strongest listed mitigation: "The most basic advice for TOCTOU vulnerabilities is
to not perform a check before the use", i.e. collapse the gap rather than police it. The
shipped comment at `.claude/workflows/qa-verdict.js:1060-1062` states the same reasoning
independently ("The remedy is not a fourth guard. It is to remove the seam").

### F2. Per-object leak filters are the *denylist* half; the completeness guard is the *allowlist* half. Both are prescribed, in that order.

OWASP is explicit that a negative filter cannot be primary: "Allowlist validation involves
defining exactly what IS authorized, and by definition, everything else is not authorized",
while denylisting "should not be relied upon as the primary method" though it "can be useful
as an **additional layer of defense** to catch some common malicious patterns"
(https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html, accessed
2026-08-21). That is precisely the 90.15 arrangement: three named-object denylists retained
for their *diagnostic* value, one allowlist added for *completeness*.

JSON Schema supplies the exact mechanical analogue of "a filter that only knows its own
keys": "`additionalProperties` only recognizes properties declared in the same subschema as
itself" -- so composing schemas breaks it -- whereas `unevaluatedProperties` "can recognize
properties declared in subschemas" by collecting "any properties that are successfully
validated ... and using those as the allowed list of properties"
(https://json-schema.org/understanding-json-schema/reference/object, accessed 2026-08-21).
`ALLOWED_CALLER_KEYS` + `k in verdict` at `qa-verdict.js:1107-1110` is an
`unevaluatedProperties`-shaped check: it unions the judge's *actual* key set with a fixed
caller allowlist, rather than enumerating one object's keys.

Pydantic shows the same axis with a *permissive default*, which is the real hazard: "By
default, Pydantic models **won't error when you provide extra data**, and these values will
simply be ignored"; closing the set is opt-in via `model_config = ConfigDict(extra='forbid')`
(https://pydantic.dev/docs/validation/latest/concepts/models/, accessed 2026-08-21). A plain
JS object literal is `extra='allow'` by construction -- there is no default that rejects.

### F3. Ordering (specific first, catch-all last) is the published rule, and it is normative, not stylistic.

CWE-396: "catching a high-level class like Exception **can obscure exceptions that deserve
special treatment** or that should not be caught at this point in the program"; "Catching an
overly broad exception essentially defeats the purpose of a language's typed exceptions"
(https://cwe.mitre.org/data/definitions/396.html, accessed 2026-08-21). RFC 9457 gives the
positive form for APIs: "HTTP status codes cannot always convey enough information about
errors to be helpful", and consumers "MUST use the 'type' URI ... as the problem type's
primary identifier" -- specificity is what makes an error actionable
(https://www.rfc-editor.org/rfc/rfc9457.html, accessed 2026-08-21). The
`qa-verdict.js:1100-1106` ordering comment is therefore aligned with published practice, and
the *reason* recorded there (a sibling checker asserts on the specific strings) is
corroborated in-repo: `scripts/qa/verify_prompt_render_86_90.mjs:705` asserts
`r.threw.includes('phase-86.72 invariant violated')`, and
`scripts/qa/verify_severity_routing_90_2.mjs:807` asserts `/phase-90\.2 invariant violated/`.
Reordering the catch-all first would turn both green assertions red -- a *measurable*
consequence, not a preference.

### F4. [TENSION -- the one source that reads against a key-set closure] RFC 9457 requires consumers to IGNORE unknown members.

"Clients consuming problem details MUST ignore any such extensions that they don't
recognize; this allows problem types to evolve and include additional information in the
future" (https://www.rfc-editor.org/rfc/rfc9457.html, accessed 2026-08-21). This is the
robustness/evolvability argument against strict key-set closure and it is a real
counterweight. **It does not apply here**, for a reason that should be stated in the
contract rather than assumed: RFC 9457's must-ignore binds the *consumer*; the 90.15 guard
is a *producer-side* closure at the point where an object crosses a trust boundary. The
relevant consumer here is Main, which transcribes the object VERBATIM into
`evaluator_critique.md` -- so "ignore what you don't recognize" is exactly the behaviour
that would let a caller field pass as judge output. The evolvability cost is real and is
paid by `ALLOWED_CALLER_KEYS` needing an edit when a legitimate fourth sibling is added;
that is the intended failure mode (loud, at the guard), not a defect.

### F5. The provenance question (caller field presented as adjudicator output) has direct prior art: CWE-441, confused deputy.

"The product receives a request, message, or directive from an upstream component, but the
product does not sufficiently preserve the original source of the request before forwarding
the request..." and the consequence: "The request would appear to be coming from the
product's system, not the attacker's system." The mitigation is stated as an identity
invariant: "The proxy core should not drop the identity of the initiator of the transaction.
The immutability of the identity of the initiator must be maintained and should be forwarded
all the way to the target" (https://cwe.mitre.org/data/definitions/441.html, accessed
2026-08-21). `qa-verdict.js` is exactly such an intermediary between the judge and Main.
**Nesting under a named sibling IS the "preserve the initiator's identity" mitigation** --
the key name `escalation` / `research_routing` / `severity_routing` carries the provenance,
and flattening destroys it. This is the strongest external framing for criterion 3's
judge-authored carve-out, and it also explains why the `research_routing` carve-out for
`research_needed` / `research_brief_spec` is correct rather than a weakness: those keys
genuinely originate with the judge, so preserving their identity means *allowing* them at
the top level.

OWASP API3:2023 supplies the operational rule for the same boundary: "If possible, avoid
using functions that automatically bind a client's input into code variables, internal
objects, or object properties" (the spread), and "Implement a schema-based response
validation mechanism ... define and enforce data returned by all API methods"
(https://owasp.org/API-Security/editions/2023/en/0xa3-broken-object-property-level-authorization/,
accessed 2026-08-21) -- i.e. the positive assertion belongs at the RETURN boundary, which is
where 90.15 put it.

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `.claude/workflows/qa-verdict.js` | 1120 | The rail. `merged` built at `:1041`, `untouched` at `:1045`, `returned` at `:1063`, three per-object filters at `:1068` / `:1077` / `:1095`, positive-completeness guard at `:1107-1115`, `return returned` at `:1120` | FIXED for the *reconstruction* seam; see G1 below for a seam it does NOT close |
| `scripts/qa/verify_severity_routing_90_2.mjs` | 1156 | 90.2 immutable checker. `extractLeakGuard` `:108-124`, `extractCompletenessGuard` `:131-148`, `load(sourceOverride)` `:150-180` carrying `__src` `:180`, section I source scans `:775-791`, behavioural drives of `leakGuard` `:797-815` and `completenessGuard` `:825-845`, cells `L1` `:916`, `L2` `:918`, `M18` `:922`, `M19` `:924`, matrix runner `:960-995` | GREEN at 100/100 on a clean tree (measured this session); see G1 |
| `scripts/qa/verify_escalation_86_78.mjs` | 424 | 86.78 checker | Asserts the runtime throw only as a SOURCE PATTERN: `:242-244` `SRC.includes('const leaked = Object.keys(escalation).filter') && /if \(leaked\.length > 0\) \{\s*\n\s*throw new Error/.test(SRC)`. No behavioural drive of the escalation leak guard. Its own header `:173-177` records that cell QA-F/M11 survived for exactly this reason |
| `scripts/qa/verify_prompt_render_86_90.mjs` | 730 | Whole-body driver. `runDriver` `:76-101`, `runDriverRaw` `:105-121`, LG-1 cell `:692-708` | Anchor at `:698` UPDATED by 90.15 to the 4-term merge line. **`runDriver` returns `{ threw, spawns, logs }` (`:100`) -- it DISCARDS the value `__run()` returns**, so the only whole-body driver in the repo cannot observe the returned object at all |
| `handoff/current/experiment_results_90.15.md` | 106 | GENERATE artifact | Accurate as far as it goes; states plainly "No research gate and no Q/A were spawned", step stays `pending`. Its section 5 already flags criteria 3 and 6 as uncovered |
| `.claude/agents/researcher.md` | 422 | This role's spec | read in full (STEP 0) |
| `.claude/rules/research-gate.md` | 338 | Authoritative floors | read in full (STEP 0) |

## G1 -- MEASURED THIS SESSION: the fix closes the RECONSTRUCTION seam but not the MUTATION seam

The 90.15 remedy makes `returned` the object every guard inspects *and* the object returned.
It does not make `returned` **immutable between the last guard and the `return`**. That
residual gap is still a TOCTOU window, and it is not hypothetical.

**Probe P1 (red-first, non-destructive, run 2026-08-21).** A scratch copy of the repo
(`/tmp`, tracked tree untouched) with a single mutation applied to `qa-verdict.js`, inserted
AFTER all four guards and keeping the `return returned` literal intact:

```js
returned.caller_note = 'x'
return returned
```

| Run | Result |
|-----|--------|
| CONTROL (unmutated copy) | `checks run: 100 (floor 100)  failed: 0` |
| **PROBE P1 (post-guard in-place mutation)** | **`checks run: 100 (floor 100)  failed: 0` -- SURVIVED** |

A caller-authored top-level key ships as judge output and the shipped 100-check checker is
green. Three mechanisms explain it, and each is independently worth the contract's attention:

1. **`M18` and `M19` are killed by a TEXT SCAN, not by the guard.** `verify_severity_routing_90_2.mjs:789-791`
   asserts `/\nreturn returned\n/.test(wfSrc)`. Both cells mutate the literal `return returned`,
   so both trip that scan. The checker's own comment at `:787-788` is honest about this
   ("Pinned as source too, because the behavioural drives above cannot see a THIRD
   construction step added after the guards"). But `M19`'s `why` string at `:924` claims it is
   "the only cell the positive-completeness guard alone can kill" -- **the kill is in fact
   attributable to the text scan**, and P1 is the discriminating experiment: same class of
   defect, literal preserved, guard present, checker green.
2. **The completeness-guard drive is a FIXTURE drive, not a return-path drive.**
   `:825` lifts `mod.completenessGuard` and `:837` calls `cg({ ...ok, caller_note: 'x' }, jv)` on a hand-built object. It proves
   the guard's predicate is correct; it cannot prove the guard is *positioned* to see what
   the rail actually returns. The guard extraction (`extractCompletenessGuard`, `:131-148`)
   lifts the guard out of its call site by design, which is what makes L1/L2 possible -- and
   is the same property that makes the drive blind to placement.
3. **No checker observes the returned object.** `runDriver` (`verify_prompt_render_86_90.mjs:76-101`)
   is the only end-to-end driver and it discards `__run()`'s return value (`:100` returns
   `threw`, `spawns`, `logs` only). LG-1 (`:704-707`) asserts only that a throw *occurred*.

**Instrument implied by the literature.** OWASP API3:2023's "define and enforce data returned
by all API methods" is a *return-boundary* assertion; the repo currently has none. The two
candidate remedies, in the order the sources rank them:

- **(a) Collapse the window (CWE-367's preferred form).** `Object.freeze(returned)` before
  the guards. In an ES module (strict mode) a later assignment then throws a `TypeError`
  rather than silently succeeding. **Caveat that must be measured, not assumed:**
  `qa-verdict.js` opens with `export const meta` (`:1`, ESM syntax) yet ends with a top-level
  `return` (`:1120`, illegal in an ESM body), and `node --check` accepts the file. The
  Workflow runtime's strictness is therefore NOT established by reading. What IS established:
  `runDriver` wraps the body in `export default async function __run() {` inside a `.mjs`
  (`:97`), so **under the checkers** it runs strict and a freeze violation would throw. Verify
  the production rail's mode before relying on freeze-throws there; a non-strict rail would
  make the freeze silent-but-still-effective (leak prevented, diagnosis lost).
- **(b) Assert at the boundary.** Change `runDriver` to capture `await mod.default()` and add
  a cell asserting the returned key set equals `Object.keys(judge) + ALLOWED_CALLER_KEYS`.
  This kills M18, M19 **and** P1 behaviourally, and would let the `return returned` text scan
  be retained as a cheap second layer rather than as the load-bearing detector.

Note (b) subsumes the two criteria `experiment_results_90.15.md:104-106` already lists as
uncovered, and it is the only instrument that discriminates P1.

---

## Consensus vs debate (external)

**Consensus (4 of 5 standards sources agree, from different angles):** a boundary that
receives caller-composed data must (i) close the accepted key set positively, and (ii)
preserve the originator's identity for anything it forwards. OWASP Input Validation
(allowlist primary, denylist as defence-in-depth only), OWASP API3:2023 (do not auto-bind
client input; enforce the returned schema), JSON Schema (`additionalProperties`/
`unevaluatedProperties` as the mechanical closure), and CWE-441 (do not drop the initiator's
identity) all converge.

**Debate:** RFC 9457 requires consumers to **ignore** unrecognized extension members for
evolvability -- the direct opposite of a closed key set. The resolution is the producer /
consumer split spelled out in F4; it is a genuine trade-off, not a contradiction, and the
contract should say which side of the boundary it is legislating.

**Second, quieter debate:** the eval literature (source 9) argues for *more* harness-computed
deterministic fields alongside judge output, which mechanically increases the number of
caller siblings and therefore the pressure on `ALLOWED_CALLER_KEYS`. The allowlist is the
right instrument, but it is also the thing that must be edited every time that advice is
followed. Expect churn there, and prefer a loud edit over a permissive default.

## Pitfalls (from the literature and from G1)

1. **Guarding a proxy for the value instead of the value** -- CWE-367's filename-vs-handle.
   This is the defect 90.15 fixed.
2. **Guarding the value but not the window** -- the check-then-mutate residual. Measured live
   as probe P1: 100/100 green with a caller key shipped. CWE-367's answer is to remove the
   window (freeze / construct-and-return atomically), not to add a fifth guard.
3. **Text-scan kills masquerading as behavioural kills** -- a mutation cell that changes a
   literal the checker also greps for is killed by the grep, and the cell's stated mechanism
   is then wrong. `M19`'s `why` at `verify_severity_routing_90_2.mjs:924` asserts the
   completeness guard is the killer; P1 shows it is not.
4. **Extracted-guard drives cannot test placement** -- lifting a guard out of its call site
   (necessary to defeat L1/L2's illusory-guard shapes) is exactly what makes the drive blind
   to whether the guard sits at the right seam. Both instruments are needed; neither
   substitutes.
5. **Catch-all first swallows the diagnosis** -- CWE-396. In-repo this is measurable: it
   reddens `verify_prompt_render_86_90.mjs:705` and `verify_severity_routing_90_2.mjs:807`.
6. **Permissive defaults** -- Pydantic's `extra='ignore'` default, and JS object literals,
   both accept unknown keys silently. Closure is always opt-in.

## Application to pyfinagent

| External finding | pyfinagent anchor | Implication for the 90.15 contract |
|---|---|---|
| CWE-367 "not perform a check before the use" | `qa-verdict.js:1063` -> `:1120` | The construct-once fix is correct and canonical. **Residual:** the object is mutable across `:1107-1120`. Consider `Object.freeze(returned)` at `:1063` -- but establish the rail's strict-mode first (see G1(a)); it is NOT established by reading. |
| OWASP allowlist-primary / denylist-as-extra-layer | `:1068`,`:1077`,`:1095` (denylists) + `:1107-1115` (allowlist) | The four-guard arrangement is exactly what OWASP prescribes. Keep all four; do not "simplify" by deleting the three named filters -- they carry the diagnosis. |
| CWE-396 specific-before-general | `:1100-1106` ordering comment | The ordering is normative and has a measurable in-repo consequence. Any criterion for 90.15 should assert the ORDER, not merely the presence of both. |
| CWE-441 preserve the initiator's identity | nesting under `escalation`/`research_routing`/`severity_routing` | Best external framing for criterion 3. It also justifies the `research_needed`/`research_brief_spec` carve-out at `:1078-1079` as *correct*, not as a weakness. |
| OWASP API3 "enforce data returned by all API methods" | `verify_prompt_render_86_90.mjs:76-101` | The repo has NO return-boundary assertion. Capturing `await mod.default()` in `runDriver` and asserting the key set is the instrument the literature points at, and the only one that kills probe P1. |
| Source 9: judge output is not reproducible | `qa-verdict.js:1028`,`:1038`,`:1040` | Affirms computing these fields caller-side. Says nothing about keeping them distinguishable -- 90.15 is filling a gap the eval literature does not address. |

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **9**
- [x] 10+ unique URLs total (incl. snippet-only) -- **37** (9 read-in-full + 28 snippet-only)
- [x] Recency scan (last 2 years) performed + reported -- 3 findings, none superseding
- [x] Full papers / pages read (not abstracts); arXiv fetched via `/html/`, never `/pdf/`
- [x] file:line anchors for every internal claim -- all re-derived by `grep -n` this session

Soft checks:
- [x] Internal exploration covered every module in the caller's INTERNAL SCOPE (5/5) plus the
      two instruction files
- [x] Contradictions noted -- RFC 9457's must-ignore rule is recorded as a real tension (F4),
      not omitted
- [x] Claims cited per-claim with URL + access date
- [ ] **DEVIATION, disclosed:** the brief exceeds the `moderate` tier's <=700-word guidance
      (4,732 words, measured). Cause: the internal half plus the G1 probe. Reported rather than
      trimmed, because trimming G1 would delete the session's only measured result.

## Envelope

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 28,
  "urls_collected": 37,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 3,
    "dry": false
  },
  "summary": "The 90.15 defect is textbook CWE-367 (guard evaluated on a proxy, not the used value) and the construct-once fix matches CWE-367's preferred remedy. The three per-object filters are denylists and the new key-set guard is the allowlist; OWASP prescribes exactly that pairing, and JSON Schema's additionalProperties-vs-unevaluatedProperties is the mechanical analogue of 'a filter that only knows its own keys'. Specific-before-catch-all is normative (CWE-396, RFC 9457) with a measurable in-repo consequence. CWE-441 supplies the provenance principle for judge pipelines. MEASURED THIS SESSION: the fix closes the reconstruction seam but not the mutation seam -- inserting `returned.caller_note='x'` after all four guards, keeping `return returned` intact, SURVIVES the shipped checker 100/100. M18/M19 are killed by a text scan, not by the completeness guard; no checker observes the returned object.",
  "brief_path": "handoff/current/research_brief_90.15.md",
  "gate_passed": true
}
```
