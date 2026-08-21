# Experiment Results -- step 90.14

> **STATUS: BUILT AND VERIFIED, NOT EVALUATED. NOT CLOSEABLE.** No Q/A spawned.
> **Research gate: PASSED (enforced)** -- `wf_85906afa-8c5`, 10 sources read in full,
> 31 URLs, recency scan performed, `self_report_disagreed: false`, `violations: []`,
> brief `handoff/current/research_brief_90.14.md` (29,367 chars).

**Step:** 90.14 -- a completeness check bound to ONE probe input shape is not
completeness. **Date:** 2026-08-21.

---

## 1. I built this before running the gate, and the gate caught what I had missed

The operator's instruction, mid-session: *"I would like you to conduct research phase as
this is the one providing you with guidance."* I had skipped it on the grounds that the
defect was already diagnosed and reproduced. **That judgement was wrong, and the gate
proved it within one run.**

The provisional build (`d4d27b50`) varied the input across four shapes — arity and
details-shape — and **pinned `verdict: 'CONDITIONAL'` on all four.** The researcher
measured, in a shadow tree with the control observed GREEN first:

> Mutant **V1** — `derivedOnly = verdict.verdict === 'FAIL' ? derived.map(...).slice(0,-1)
> : derived.map(...)` — **SURVIVES the entire checker: 105 checks, 0 failed.** Not
> equivalent: on the checker's own 24-return fixture set it differs on **6 of 24** real
> returns (all six FAILs), route unchanged so every route assertion is structurally blind.
> The **ungated** control is KILLED, so the survival is attributable to the verdict GATE,
> not to the field being generally unguarded.

**That is criterion 6 clause 2 relocating a FIFTH time — into the one dimension my family
did not vary.** I had varied exactly the dimensions I had just been burned on and held the
rest fixed. The gate found it *before* it shipped, which is the entire argument for the
gate.

## 2. What the literature changed about the design

| Finding | Source | What it changed here |
|---|---|---|
| **Adequacy is a two-place relation over (mutants, inputs).** Fixing one and enumerating the other proves nothing about the pair. Minimality over all test sets is **undecidable**, and a richer input set **REQUIRES more mutants** | Ammann/Delamaro/Offutt 2014 (MiniMutant, ICST) | The whole framing. Four shapes + four cells was not obviously enough; cells were added with the shapes. |
| **Necessity has a published form: leave-one-out** | Gopinath 2016 | Replaced my wrapper-attribution, which gated each mutant on *its own shape's discriminant* — close to a control built from the pattern it is proving. |
| **Redundant cells INFLATE a score** — Type I error ~62%, 68% of studies vulnerable | Papadakis et al., ISSTA 2016 **[adversarial]** | Redundancy is now *declared and asserted*, not assumed absent. |
| **A survivor is a hypothesis, not a finding**; equivalence is undecidable, EMR 1–2% | Cerebro (arXiv 2112.14151), et al. | Kept the existing discipline of declaring equivalent mutants rather than padding the matrix with them. |
| **Interaction rule prices the family**: 1-way ~67%, 2-way ~93%, 3-way ~98% | NIST SP 800-142 | Chose and **stated** t = 2. |
| **"Continuous inputs must be discretised"** — the family *is* a discretisation, and adequacy is a claim about that model, not the input space | NIST SP 800-142 | **No unconditional "complete" anywhere.** |

**The retraction that follows:** my cycle-4 claim that "a fifth array field fails the
checker until it is covered" was unqualified, and **no source supports an unconditional
completeness claim.** The only completeness this step asserts is *"2-way complete over the
declared input model {arity, details, verdict}"* — and that assertion is **executable**, so
it is falsifiable rather than rhetorical.

## 3. What was built

**A DECLARED INPUT MODEL with a STATED strength.** `INPUT_MODEL = {arity: [0,2,5],
details: [none, aligned, mismatched], verdict: [CONDITIONAL, FAIL, PASS]}`,
`COVERAGE_STRENGTH = 2`. Twelve shapes, and the pairwise property is **proven by execution**
over the declared factors — an asserted covering array, never a claimed one.

**CONSERVATION LAWS instead of per-shape expected arrays.** Writing expected contents per
shape would mean re-implementing `enforceSeverityRouting` inside the checker, and *a control
built from the same walk as the code shares its bug*. Instead each returned array is
asserted against **the input array it summarises**:

- `derived_severities.length === violated_criteria.length`, index-aligned
- `emitted_severities` is `null` iff there are no details, else `=== violation_details.length`
- `governing_severities.length` is one of those two — **never a truncation of either**

Any drop, on any branch, at any arity, breaks one of these.

**LEAVE-ONE-OUT necessity, at factor-value granularity.** A pairwise covering set is *not*
a minimal basis, so asserting every shape is uniquely necessary would be false. For each
factor value: remove every shape carrying it, and show a mutant escapes.

```
  factor value           mutant that escapes without it
  arity=0               A4 empty-only fabrication
  arity=2               NOTHING
  arity=5               A1 arity-gated drop (>=4)
  details=none          NOTHING
  details=aligned       A2 aligned-only drop
  details=mismatched    A3 mismatched-only drop
  verdict=CONDITIONAL   A0c CONDITIONAL-gated drop
  verdict=FAIL          A0 FAIL-gated drop
  verdict=PASS          A0b PASS-gated drop
```

**Two values earn nothing, and the instrument found it rather than my asserting
otherwise.** `arity=2` — nothing is gated on 2. `details=none` — at arity 0 the `aligned`
shape degenerates to zero details, so `arity=0 × aligned` already *is* a no-details shape.
Both are kept because pairwise coverage needs them, both are **declared**, and the declared
list is asserted equal to the measured one, so redundancy cannot grow silently.

The **ungated** control is caught by every non-empty shape — which is what shows the gated
mutants survive because of their *gate*, not because the field is unguarded.

## 4. Cells

**27 cells.** New in this step:

```
  ok   V1   KILLED    expected KILLED   the FAIL-gated drop -- the fifth relocation, killed before it shipped
  ok   V2   KILLED    expected KILLED   the same gate on PASS
  ok   M20  KILLED    expected KILLED   the arity-gated drop that FAILED 90.2 at cycle 4
  ok   M21  KILLED    expected KILLED   branch-gated drop-last
  ok   M22  KILLED    expected KILLED   branch-gated drop-first
  ok   M23  KILLED    expected KILLED   a branch-conditional array field
  ok   N0   SURVIVED  expected SURVIVED
  ok   QX   ERROR     expected ERROR
```

## 5. Verification

```
$ bash -c 'node --check .claude/workflows/qa-verdict.js && node scripts/qa/verify_severity_routing_90_2.mjs --self-test'
  checks run: 100 (floor 100)
  failed:     0
EXIT 0
```

## 6. What is NOT done

- **No Q/A verdict.** Not closeable, not flipped.
- The brief's stronger form of necessity — **leave-one-out against the REAL cells
  M15–M23**, re-scoring the matrix with a shape removed — is **not** implemented. What ships
  is leave-one-out over *wrapper* mutants, which the brief explicitly calls a weaker
  instrument. Stated, not glossed. It would also settle whether `S3-mismatched` is subsumed
  by `S2-aligned` against the real cell set.
- `COVERAGE_STRENGTH` is 2. NIST prices 3-way at ~98% versus 2-way at ~93%; 3-way was not
  attempted and the gap is not closed by anything here.
