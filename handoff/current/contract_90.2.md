# Contract -- step 90.2

**Step:** 90.2 -- "route the WARN/NOTE severity the judge already emits, caller-side,
without asking it to classify anything new and without moving the verdict"
**Phase:** phase-90. **Priority:** P0. **Contract written:** 2026-08-20.
**Depends on:** 90.1 (accounting) landing first, per the phase order.

---

## 1. Research gate -- PASSED (enforced)

`Workflow({scriptPath: '.claude/workflows/research-gate.js'})`, run `wf_05a76fdf-b16`,
2 agents, 191,320 tokens, 738s. Brief: `handoff/current/research_brief_90.2.md`
(38,446 chars). Enforced return: `gate_passed: true`, `self_report_disagreed: false`,
`violations: []`; 7 sources read in full, 17 URLs, recency scan performed, all 7 claimed
URLs present in the brief.

### The finding that CHANGES this step's design

**Re-deriving severity from the judge's prose is measured at near-chance.** Replayed over
the local corpus, token-presence re-derivation agrees with the `SKILL.md` dispatch on
144/360 = **40.0%**, Cohen's kappa **0.129**, against a **56.7%** majority-class baseline
-- i.e. **16.7pp worse than a constant**. Mechanism: **130 of 183** records containing
`BLOCK` (71.0%) carry a NEGATED occurrence ("no BLOCK, no WARN fired"). The precise
literal `severity=` extractor reaches 82.5% but fires on only 15.9% of runs and captures
8 junk values in 167.

**I reproduced the mechanism independently before accepting it.** Over the run records,
8.1% of all violated_criteria entries containing a severity token carry it NEGATED, and
**7 of the 41** all-WARN/NOTE runs this step's criterion 4 names contain at least one
negated mention -- so that population is roughly **17% contaminated** by a matcher that
cannot tell `WARN: x` from `no WARN fired`.

External corroboration: AgentProp-Bench (arXiv 2604.16706) measures substring-heuristic
judging at chance (kappa 0.049) against 0.432/0.567 for real classifiers.

**This does not block the step; it determines the design.** Criterion 4 explicitly asks
for the replay proof over that measured population, and the brief's own conclusion is
that "the replay demonstrates re-derivation FAILS -- which is the evidence for the design,
not an obstacle to it."

### The tension this contract must resolve, stated rather than smoothed

The brief's recommended mechanism is to have the **judge EMIT** severity in an optional
`VERDICT_SCHEMA` field (`additionalProperties: false` means a runtime-added key is
impossible, so prose harvesting cannot be made sound without a schema edit). But **what
the judge may emit is 86.98's**, whose criterion 7 requires an **operator sign-off**, and
this step's own notes say 90.2 "must satisfy rather than pre-empt" 86.98.

**Resolution: 90.2 makes no schema edit and asks the judge for nothing new.** It ships a
caller-side reader that is *forward-compatible* with the field 86.98 may later add:

- if a judge-emitted severity is present, that governs (satisfying 86.98 criterion 5,
  "severity is taken from the judge's OWN classification rather than re-derived");
- today **0 of 969** `violation_details` rows carry one, so that branch is inert on
  arrival and changes nothing;
- otherwise the caller derives from prose **and labels the derivation as such**, carrying
  its measured reliability with it rather than presenting it as fact.

A schema edit under 90.2 would pre-empt 86.98 and bypass an operator gate. It is not made.

---

## 2. Hypothesis

The judge already distinguishes BLOCK from WARN from NOTE -- `SKILL.md:28-30` mandates the
dispatch and is preloaded into every spawn. But `VERDICT_SCHEMA.violation_details` is
`additionalProperties: false` with a closed 7-value `violation_type` enum and **no
severity key**, so that distinction survives only inside free text. Main's only defined
response to any non-PASS entry is fix-and-re-spawn, whatever its severity.

Routing that severity caller-side gives Main a second defined response -- *file it* --
without touching the verdict. The safety proof is structural: on the measured population
**all 41 candidate runs are CONDITIONAL and zero are FAIL** (I reproduced this exactly),
and a hard verdict guard makes firing on a FAIL unrepresentable rather than merely
unobserved.

---

## 3. Immutable success criteria (VERBATIM from .claude/masterplan.json)

1. a caller-side function in .claude/workflows/qa-verdict.js returns a routing object ALONGSIDE the judge's fields, never merged into them, and the existing sibling-leak invariants are extended to THROW if any of its keys appear inside the judge's own object

2. the routing can never fire on a FAIL: a test drives a FAIL whose every violated_criteria entry is WARN-prefixed and asserts the route is remediate, and a mutant removing the verdict guard is KILLED

3. the judge's `verdict` string is byte-identical before and after routing on all three values, asserted by string equality over a fixture set replaying at least 20 real returns, never by inspection

4. replay proof on the measured population: the 41 all-WARN/NOTE non-PASS runs route to queue_residual and the remaining 247 route to remediate, printed as a confusion table carrying the run ids; any run mixing a WARN entry with an untagged finding must route to remediate, and the strict-vs-permissive count difference (32 vs 41) is stated rather than silently resolved

5. Main's required response to queue_residual is to FILE a masterplan step carrying its own immutable verification command -- not an in-place fix and not a fresh spawn -- and a checker refuses the parent step's close when the filed residual is absent or does not parse

6. mutation-tested with the control observed GREEN first: a mutant letting an untagged finding route to queue_residual must be KILLED, and a mutant silently dropping any reported finding from the return must also be KILLED

**Immutable verification command** (RED at filing time by design -- it names the checker
this step must build):

```
bash -c 'node --check .claude/workflows/qa-verdict.js && node scripts/qa/verify_severity_routing_90_2.mjs --self-test'
```

**live_check:** `live_check_90.2.md`: the verbatim 41/247 replay table over real run ids,
the strict-match 32/256 table beside it, and the FAIL-immunity cell output.

---

## 4. Plan

### 4.1 `enforceSeverityRouting(verdict)` -- caller-side, pure, sibling (criterion 1)

Modelled exactly on the two functions already in the file: `enforceEscalation`
(phase-86.78) and `enforceResearchRouting` (phase-86.72). Plain function, not exported,
driven by the checker via temp re-export -- the house pattern, so the checker drives the
REAL function and never a copy.

Returned shape (names deliberately distinct from every judge field, so a leak is
detectable rather than plausible):

- `route`: `remediate` | `queue_residual`
- `severity_source`: `judge_emitted` | `derived_from_prose` | `ABSENT`
- `derived_severities`: per-entry, aligned to `violated_criteria` by index
- `disagreed`: judge-emitted vs caller-derived, when both exist
- `reliability`: the measured kappa and majority-class baseline, carried WITH the
  derivation so no consumer can read it as authoritative

Absence gets a **name** (`ABSENT`), never a zero or a silent `remediate` -- the
`sequence_status` null-never-zero idiom already in this file.

### 4.2 Sibling invariants EXTENDED, not re-invented (criterion 1)

`merged = { ...verdict, escalation, research_routing, severity_routing }`, then the
existing leak filters at the two throw-sites gain the third object. Criterion 1 says
"the existing sibling-leak invariants are extended", so a third bespoke check is the
wrong shape; the carve-out list for judge-authored fields follows the `research_routing`
precedent.

### 4.3 The FAIL guard is structural (criterion 2)

`route` is computed as `queue_residual` **only if** `verdict === 'CONDITIONAL'`. FAIL and
PASS cannot reach it. Test: a FAIL whose every entry is WARN-prefixed -> `remediate`.
Mutant removing the guard -> KILLED. This is what makes the safety property hold by
construction rather than by the observed zero-FAIL count.

### 4.4 Negation-aware derivation, with its unreliability attached (criteria 4, 6)

The derivation must not count `no WARN fired` as a WARN -- measured at 8.1% of
severity-bearing entries and 17% of the criterion-4 population. Untagged findings force
`remediate`; a mixed run (one WARN + one untagged) forces `remediate`, which criterion 4
requires explicitly and which mutation cell "untagged routes to queue_residual" kills.

### 4.5 The replay, with the count difference STATED (criterion 4)

Criterion 4 names 41 and 32. **41 reproduces exactly** on my corpus. **32 does not
reproduce under any of four plausible strict definitions** -- I measured 41 (token
anywhere), 26 (bracketed anywhere), 11 (starts-with bare), 4 (starts-with with a
separator). The criterion says the difference must be **stated rather than silently
resolved**, so the replay will print the permissive table, my strict table with its
matcher printed beside it, and an explicit note that the filing's strict definition is not
recoverable from its text. The number will not be edited to match.

Denominators will be stated: the brief measured 441 `startsWith('qa-verdict')` vs 436
exact vs 398 parseable vs 397 with a verdict. My own corpus gives 436/393. Which
denominator is in use will be printed with every ratio.

### 4.6 Verdict byte-identity over >= 20 real returns (criterion 3)

String equality across all three verdict values over a fixture set replayed from real run
records -- never inspection, and never a hardcoded `true` (the cycle-1 lesson from 86.78:
an attestation is not a check).

### 4.7 The queue_residual consumer (criterion 5)

A checker that refuses the parent step's close when the filed residual step is absent or
does not parse. This is the half that makes routing mean something: `queue_residual`
obliges Main to FILE, not to fix in place and not to re-spawn.

### 4.8 `scripts/qa/verify_severity_routing_90_2.mjs` (the immutable command)

Modelled on `verify_escalation_86_78.mjs`: real-function-via-temp-re-export, an
`EXPECTED_CHECKS` cardinality floor so a checker whose loop covers nothing cannot exit 0,
a `sourceOverride` mutation seam, and a RED cell proving the flattened-sibling mutation is
caught.

---

## 5. Explicitly NOT done here

- **No `VERDICT_SCHEMA` edit.** That is 86.98's, and its criterion 7 requires operator
  sign-off. 90.2 asks the judge for nothing new.
- **No verdict movement.** Frozen semantics; the routing is a sibling.
- 87.9 was flipped `superseded` by the operator earlier today, citing this step and 90.9.
  That flip is recorded in the masterplan, not re-litigated here.

## 6. References

- `handoff/current/research_brief_90.2.md` (gate PASSED, enforced).
- OpenTelemetry Logs Data Model -- `SeverityText` keeps the source's own string beside a
  normalized `SeverityNumber`: https://opentelemetry.io/docs/specs/otel/logs/data-model/
- OASIS SARIF v2.1.0 -- `level` / `kind` / `rank` as separate fields:
  https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/sarif-v2.1.0-errata01-os-complete.html
- RFC 9413, virtuous intolerance: https://www.rfc-editor.org/rfc/rfc9413.html
- AgentProp-Bench, substring-heuristic judging at chance: https://arxiv.org/html/2604.16706
- Judge Reliability Harness, ordinal caution: https://arxiv.org/html/2603.05399
- Anthropic, Building Effective Agents: https://www.anthropic.com/engineering/building-effective-agents

**Citation hygiene note carried forward from the brief:** the researcher recorded that a
WebFetch summariser **FABRICATED** a SARIF "SHALL NOT re-derive" clause that does not
exist in the spec. That non-existent clause is cited nowhere in this contract or in
anything this step will ship.
