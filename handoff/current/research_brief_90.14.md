# Research Brief -- step 90.14

**Tier: moderate** (caller-specified). Audit-class: NO (coverage reported for information only).

**Topic:** Mutation-testing adequacy as a function of the INPUT space rather than the
output surface -- why enumerating every field of a returned object while holding one
fixed probe input leaves arity-gated and branch-gated mutants alive; principled
construction of a minimal FAMILY of test inputs (category-partition, combinatorial
interaction testing, boundary/arity classes); mutant subsumption + dominator-mutant
theory for proving each family member NECESSARY not redundant; equivalent-mutant
detection and why a surviving mutant is not automatically a finding; published guidance
on when a coverage claim may be called complete.

<!-- ENVELOPE: flipped to COMPLETE as the final act per phase-86.37. -->
```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 10,
  "snippet_only_sources": 21,
  "urls_collected": 31,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 1,
    "dry": false
  },
  "summary": "Adequacy is a two-place relation (mutants, inputs): fixing one and enumerating the other proves nothing about the pair. Ammann/Delamaro/Offutt 2014 -- minimality over ALL test sets is undecidable, and a richer input set REQUIRES more mutants. Necessity has a published form (leave-one-out against the REAL cells), which the provisional attributionMutants() wrappers approximate with wrappers gated on their own shape. A survivor is a hypothesis, not a finding (EMR 1-2% rule-based; equivalence undecidable). Redundant cells INFLATE a matrix score (Type I error ~62%, Papadakis ISSTA16 [ADVERSARIAL]). NIST interaction rule prices the family: 1-way 67%, 2-way 93%, 3-way 98%. MEASURED in a shadow tree (control green first): the shipped 4-shape family pins verdict=CONDITIONAL on all four members, so a FAIL-gated drop SURVIVES all 105 checks and is non-equivalent on 6 of 24 real fixtures -- criterion 6 clause 2 relocating a FIFTH time, now to the verdict dimension. The ungated control is KILLED. No source supports an unconditional 'complete'; the defensible claim is t-way complete over a NAMED input model.",
  "brief_path": "handoff/current/research_brief_90.14.md",
  "gate_passed": true
}
```

## Search-query composition (three-variant discipline)

_(filled in as searches run)_

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://ar5iv.labs.arxiv.org/html/1601.06466 | 2026-08-21 | paper | WebFetch (ar5iv HTML) | dynamic subsumption defined RELATIVE to a test set; |M_min| <= C(n,floor(n/2)) |
| 2 | https://rahul.gopinath.org/post/2016/10/02/minimal-mutants/ | 2026-08-21 | researcher blog | WebFetch (HTML) | 5 minimal-set definitions; a minimal suite is one where removing any test drops the score |
| 3 | https://arxiv.org/html/2406.09843 | 2026-08-21 | paper | WebFetch (arXiv HTML) | equivalent-mutant rate 1.0-10.6%, duplicate rate 0-7.5% |
| 4 | https://ar5iv.labs.arxiv.org/html/2112.14151 | 2026-08-21 | paper | WebFetch (ar5iv HTML) | only 10.2% (C) / 26.8% (Java) of mutants are subsuming; 73-90% redundant |
| 5 | https://www.albany.edu/faculty/offutt/research/papers/MiniMutant-ICST2014.pdf | 2026-08-21 | paper | pdfplumber 0.11.9 (10pp, 51,143 chars) | minimality over ALL test sets is undecidable; richer T needs MORE mutants |
| 6 | https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-142.pdf | 2026-08-21 | official doc | pdfplumber 0.11.9 (82pp) | Interaction Rule; 1-way 67% / 2-way 93% / 3-way 98%; inputs must be discretised |
| 7 | https://www.albany.edu/faculty/offutt/research/papers/calcengine-post.pdf | 2026-08-21 | paper | pdfplumber 0.11.9 (23pp, 65,263 chars) | ISP on financial calc engines: fewer tests, more defects, 5 days -> 0.5 |
| 8 | https://pmc.ncbi.nlm.nih.gov/articles/PMC5267492/ | 2026-08-21 | paper (NIST) | WebFetch (HTML) | t-way coverage is MEASURABLE on a suite not built for it; no universal sufficiency threshold |
| 9 | https://arxiv.org/html/2605.17437 | 2026-08-21 | paper (2026-07-07) | WebFetch (arXiv HTML) | adequacy is "relative to a declared stratum"; equivalents leave the denominator |
| 10 | https://discovery.ucl.ac.uk/1508136/1/ISSTA16.pdf | 2026-08-21 | paper **[ADVERSARIAL]** | pdfplumber 0.11.9 (12pp, 65,399 chars) | subsumed mutants inflate scores; Type I error ~62%; 68% of studies vulnerable |

## Progress log (write-first, appended as sources land)

### Round 1 -- searches run (three-variant discipline)
- Year-less canonical: `mutant subsumption dominator mutants minimal set of mutants`;
  `category-partition method specifying functional tests Ostrand Balcer`;
  `combinatorial interaction testing t-way pairwise fault interaction NIST`;
  `equivalent mutant problem detection undecidable trivial compiler equivalence`
- Recency (2024-2026): `mutation testing 2025 2026 test adequacy input space advances survey`
- (further variants logged below as they run)

### Source 1 -- READ IN FULL (ar5iv HTML)
`https://ar5iv.labs.arxiv.org/html/1601.06466` -- "A Theoretical Framework for
Understanding Mutation-Based Testing Methods" (Gopinath, Alipour, Ahmed, Jensen,
Groce). Accessed 2026-08-21.
- Verbatim definition: *"If a mutant m_x is killed by at least one test in a set of
  tests TS and another mutant m_y is always killed whenever m_x is killed, then m_x
  dynamically subsumes m_y."*
- *"A mutant set M_min is minimal when no distinct pair exists where one subsumes the
  other."*
- Bound: *"For an arbitrary test set TS and arbitrary mutants set M, the theoretical
  maximum size of the minimal mutant set M_min with respect to TS is C(n, floor(n/2)),
  where n is the size of a test set."* With 5 tests, at most C(5,2)=10 mutants can
  remain minimal.
- **The load-bearing point for 90.14:** subsumption is defined RELATIVE TO A TEST SET.
  Adequacy is `forall m in M, exists t in TS, d(t, p_o, m)` -- it quantifies over the
  INPUT set, and the differentiator `d` (strong vs weak mutation) is a parameter.
  A "complete" claim computed against one TS says nothing about another TS.

### Source 2 -- READ IN FULL (researcher blog, tier 3)
`https://rahul.gopinath.org/post/2016/10/02/minimal-mutants/` -- "Various Definitions
of Minimal Mutant Sets" (Rahul Gopinath). Accessed 2026-08-21.
- Five distinct notions: absolute-minimal, theoretical-minimal, disjoint, surface, and
  distinguished/unique mutants. Surface / disjoint / theoretical-minimal are shown
  functionally equivalent.
- Minimal test suite = one whose tests are *"sufficient and necessary for fulfilling
  that objective"*; removing any test drops the mutation score. **This is exactly the
  90.14 criterion-4 obligation** (each family member must kill something no other
  member kills).
- **Critical caveat, quoted:** minimality depends on the test suite used; a minimal set
  computed against one test suite may not be minimal against another, because
  subsumption relations change with the available tests.

### Source 3 -- READ IN FULL (arXiv HTML)
`https://arxiv.org/html/2406.09843` -- "A Comprehensive Study on Large Language Models
for Mutation Testing" (2024/2025). Accessed 2026-08-21.
- Equivalent-Mutant Rate (EMR) measured by manual sampling at 95% confidence / 10%
  margin: rule-based PIT 1.0%, Major 2.1%; LLM-based 3.1%-10.6%. Duplicate-Mutant Rate
  (DMR): PIT/Major 0%, LLM-based 6.9-7.5%.
- Definition used: equivalent mutants are *"those that are syntactically different but
  semantically equivalent to the original code"*, distinguished from syntactic
  duplicates.
- Relevance to 90.14: a nonzero EMR is expected even in a hand-authored matrix. **A
  surviving mutant is a hypothesis, not a finding** -- it is either a coverage gap or an
  equivalent mutant, and the two are not distinguishable without argument.

### Source 4 -- READ IN FULL (ar5iv HTML)
`https://ar5iv.labs.arxiv.org/html/2112.14151` -- "Cerebro: Static Subsuming Mutant
Selection" (Garg, Ojdanic, Degiovanni, Titcheu Chekam, Papadakis, Le Traon). Accessed
2026-08-21.
- Verbatim: *"Given two mutants M1 and M2, it is said that M1 subsumes M2 if every test
  suite T killing M1 also kills M2."* Equivalently `T1 subset-of T2`.
- Verbatim: *"subsuming mutants are the minimum subset of all mutants that when killed,
  by any possible test suite, results in killing the entire set of killable mutants."*
- Measured redundancy: **10.2% subsuming among 71,850 C mutants; 26.8% among 153,823
  Java mutants** -- i.e. 73-90% of generated mutants are redundant.
- Explicit: a mutant's classification as subsuming *depends entirely on the available
  tests*.

### Source 5 -- READ IN FULL (pdfplumber, 10pp / 51,143 chars)
`https://www.albany.edu/faculty/offutt/research/papers/MiniMutant-ICST2014.pdf` --
Ammann, Delamaro & Offutt, "Establishing Theoretical Minimal Sets of Mutants", ICST
2014. Accessed 2026-08-21. (WebFetch on a binary PDF is the documented trap; extracted
with pdfplumber 0.11.9 per `.claude/rules/research-gate.md` step 3.)
- **THE CENTRAL RESULT FOR 90.14, verbatim:** *"Computing minimal mutant sets for all
  possible test sets is clearly undecidable; it is the fact that we limit attention to a
  particular test set that makes our approach computable."*
- Verbatim: *"The richer the test set T is, the more mutants a minimal mutant set
  requires to capture the behavior exhibited by the artifact with respect to that test
  set."*
- Verbatim: *"our notion of subsumption is only assumed to hold with respect to the
  specific set of test cases under consideration, and it is possible that the subsumption
  relation would not hold for a different set of tests. Essentially, we replace the risk
  of equivalent mutants ... with the risk of incomplete test sets."*
- Verbatim: *"generating a different test set might result in a different set of minimal
  mutants."*
- Verbatim on why survivors are ambiguous: *"A live mutant m may be equivalent, or T may
  rather be missing a suitable test that kills m."*
- Verbatim on inflated scores: *"mutation scores can be inflated by redundant mutants,
  and this can make the mutation score harder to interpret."* Once redundant mutants are
  removed, *"the scores are lower, sometimes much lower."*

### Source 6 -- READ IN FULL (pdfplumber, 82pp)
`https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-142.pdf` -- Kuhn,
Kacker & Lei, NIST SP 800-142 "Practical Combinatorial Testing" (2010). Accessed
2026-08-21.
- **Interaction Rule, verbatim:** *"Most failures are induced by single factor faults or
  by the joint combinatorial effect (interaction) of two factors, with progressively
  fewer failures induced by interactions between three or more factors."*
- Measured: for the NASA application *"67% of the failures were triggered by only a
  single parameter value, 93% by 2-way combinations, and 98% by 3-way combinations"*;
  across domains all failures were triggered by at most **4-way to 6-way** interactions.
- Verbatim caution that lands directly on a 4-shape family: *"most parameters are
  continuous variables which have possible values in a very large range (+/- 2^32 or
  more). These values must be discretized to a few distinct values."* -- the family IS a
  discretisation, and its adequacy is a claim about the discretisation, not about the
  input space.
- Verbatim: the second cost is the ORACLE -- *"Generating 1,000 test data inputs is of
  little help if we cannot determine what the system under test (SUT) should produce as
  output for each of the 1,000 tests."*

### Source 7 -- READ IN FULL (pdfplumber, 23pp / 65,263 chars)
`https://www.albany.edu/faculty/offutt/research/papers/calcengine-post.pdf` -- Offutt &
Alluri, "An Industrial Study of Applying Input Space Partitioning to Test Financial
Calculation Engines". Accessed 2026-08-21.
- Directly cross-domain (finance): applying ISP to Freddie Mac calculation engines
  yielded *"fewer tests that found more defects"*, all four systems *"reported zero
  defects since release"*, test cycle 5 human-days -> 0.5.
- Establishes the practitioner claim that a SMALL, PRINCIPLED family beats a large ad-hoc
  one -- the same economics 90.14's 4-shape family is betting on.

### Source 8 -- READ IN FULL (PMC HTML)
`https://pmc.ncbi.nlm.nih.gov/articles/PMC5267492/` -- Kuhn, Dominguez Mendoza, Kacker &
Lei, "Measuring and Specifying Combinatorial Coverage of Test Input Configurations"
(NIST). Accessed 2026-08-21.
- Definitions: *simple t-way combination coverage* = *"the proportion of t-way
  combinations of n variables for which all valid variable-values configurations are
  fully covered"*; *total variable-value configuration coverage* = *"the proportion of
  all t-way variable-value configurations that are covered by at least one test case in a
  test set."*
- Key property for 90.14: these measures apply to a test suite **not built for
  combinatorial coverage** -- you can MEASURE the t-way coverage of an existing family.
- **On completeness, verbatim in effect:** no universal sufficiency threshold exists;
  the paper anchors required strength to empirical fault distribution, and states that
  coverage remains inherently incomplete -- unmeasured combinations always exist.

### Source 9 -- READ IN FULL (arXiv HTML, RECENCY WINDOW)
`https://arxiv.org/html/2605.17437` -- "A semantic mutation metric for metamorphic
relation adequacy in scientific computing programs", arXiv:2605.17437v2, **2026-07-07**.
Accessed 2026-08-21.
- *"classical Mutation Score (MS) remains syntactic. It does not say whether a
  metamorphic-relation set observes domain-semantic effects."*
- Disjoint decomposition: `mut_j(S_i) = equiv u killed u survive` -- equivalents are
  removed from the DENOMINATOR, and only after a two-layer test (`E1 ^ E2`).
- Verbatim: *"SMS therefore measures adequacy relative to a declared stratum; it is not a
  total order over all possible MR families."* -- 2026 restatement of the same
  relativity Ammann 2014 proved.

### Source 10 -- READ IN FULL (pdfplumber, 12pp / 65,399 chars) [ADVERSARIAL]
`https://discovery.ucl.ac.uk/1508136/1/ISSTA16.pdf` -- Papadakis, Henard, Harman, Jia &
Le Traon, "Threats to the Validity of Mutation-Based Test Assessment", ISSTA 2016.
Accessed 2026-08-21.
- Verbatim: *"the presence of subsumed mutants (also known as redundant mutants), can
  artificially inflate the apparent ability of a test technique to detect faults."*
- Measured: Type I errors occur *"approximately 62% of the time"* for experiments that
  take no countermeasure; **68%** of surveyed papers are vulnerable; correlations between
  MS and subsuming-MS are *"fairly weak, between 0.2 - 0.6"*.
- **Why this is the adversarial source:** it argues that a mutation matrix whose cells are
  not de-duplicated for subsumption produces a score that is *not evidence* of the thing
  it appears to measure. A checker reporting "24 of 24 cells KILLED" is exactly such a
  score unless the cells are shown to be mutually non-subsuming.

---

## Recency scan (last 2 years, 2024-2026)

Queries run: `mutation testing 2025 2026 test adequacy input space advances survey`;
`combinatorial coverage measurement 2025 2026 test suite adequacy input model
completeness`; `mutation score correlated with real fault detection criticism Papadakis
threats to validity`.

**Result: 2 new findings, both COMPLEMENTING rather than superseding the canonical
sources.** (1) arXiv:2605.17437 (2026-07-07, source 9) restates adequacy-relativity in
2026 terms -- adequacy is measured *"relative to a declared stratum"* and equivalents are
excluded from the denominator by an explicit two-layer test. (2) The 2024-2025 LLM
mutation-testing literature (source 3) supplies fresh equivalent-mutant and duplicate-
mutant base rates (EMR 1.0-10.6%, DMR 0-7.5%). **Nothing in the 2024-2026 window
supersedes** Ammann/Delamaro/Offutt 2014 on minimal mutant sets, Ostrand & Balcer 1988 on
category-partition, or the NIST interaction rule; all three remain the canonical
statements and the newer work cites them. Mutation 2026 (ICST workshop) still lists
"evaluation of mutation-based test adequacy criteria" as an OPEN topic.

## Key findings

1. **Adequacy is a two-place relation: (mutants, inputs). Fixing one and enumerating the
   other proves nothing about the pair.** *"Computing minimal mutant sets for all possible
   test sets is clearly undecidable; it is the fact that we limit attention to a
   particular test set that makes our approach computable."* (Ammann, Delamaro & Offutt
   2014, MiniMutant-ICST2014.pdf). This is the formal statement of the 90.14 filing's own
   sentence "a coverage claim is only as wide as the INPUTS it was computed over".
2. **A richer input family REQUIRES a larger mutant set, not a smaller one.** *"The richer
   the test set T is, the more mutants a minimal mutant set requires."* (ibid.) So adding
   shapes S1-S4 without adding cells leaves the family under-exercised by construction.
3. **Necessity has a published definition, and it is leave-one-out against the REAL
   mutant set** -- a minimal suite is one where *"removing any test causes mutation score
   to drop"* (Gopinath 2016). Bespoke wrappers gated on the shape they are meant to prove
   are a weaker instrument.
4. **A surviving mutant is a hypothesis, not a finding.** *"A live mutant m may be
   equivalent, or T may rather be missing a suitable test that kills m."* (Ammann 2014).
   Base rates: EMR 1.0-2.1% for rule-based generators (arXiv:2406.09843); equivalence is
   undecidable in general, and TCE recovers only ~30% of equivalents.
5. **Redundant/subsumed cells INFLATE a matrix score.** 73-90% of generated mutants are
   subsumed (Cerebro, arXiv:2112.14151); including them causes Type I errors ~62% of the
   time (Papadakis ISSTA 2016).
6. **How many shapes is an empirical question with a published answer.** NIST's Interaction
   Rule: 1-way ~67%, 2-way ~93%, 3-way ~98% cumulative fault detection; all failures
   studied were triggered by <=4-6-way interactions. A family that varies one factor at a
   time achieves only 1-way coverage of its own input model.
7. **No source in the read-in-full set offers an unconditional "complete".** NIST: coverage
   is *inherently incomplete*, strength is chosen against a fault distribution.
   arXiv:2605.17437: adequacy is *relative to a declared stratum*. The defensible claim is
   "t-way complete over a NAMED input model", never "complete".

## Internal code inventory (file:line anchors)

| File | Lines | Role | Status |
|---|---|---|---|
| `scripts/qa/verify_severity_routing_90_2.mjs` | 1016 | the checker; `--self-test` is 90.14's immutable command | **PROVISIONAL 90.14 work already committed pre-research** (`d4d27b50` "PROVISIONAL, pre-research") |
| ^ `familyShapes()` | :211-237 | S1-wide(5) / S2-comparable(2) / S3-mismatched(2) / S4-empty(0) | **all four call `V('CONDITIONAL', ...)`** -- the verdict factor is HELD FIXED |
| ^ `ARRAY_CAPABLE` | :206 | union check `derived/emitted/governing_severities` | live; catches M23 |
| ^ E1b loop | :425-452 | per-shape array-key SET + ordered content + arity-variance | live, 14 checks |
| ^ `attributionMutants()` / `shapesThatCatch()` / E1c | :242-286, :454-471 | criterion-4 necessity proof | **wrappers are authored gated on their own `onlyShape` condition** -- see finding 3 |
| ^ `MUTANTS` M15-M23 | :768-802 | the 90.14 cells | all KILLED; M20-M22 arity/branch-gated, M23 branch-conditional array key |
| ^ `UNRESOLVABLE` | :813 | ERROR discriminator (JS side) | reads the exception TYPE, incl. `is not defined` |
| ^ equivalent-mutant disclosure | :510-524 | the `entries.length > 0` clause reported as EQUIVALENT, not padded in | **literature-correct**; matches Papadakis/Ammann |
| ^ `EXPECTED_CHECKS` | :40 | cardinality floor 100 | live; run measured 105 |
| `.claude/workflows/qa-verdict.js` `enforceSeverityRouting` | :889-960+ | the subject | `derived`/`derivedOnly`/`governing`/`emitted` are 4 array channels |
| ^ single-construction seam | :1063 `const returned = {...}`, guards :1068/:1077/:1095, positive guard :1107-1118, `return returned` :1120 | 90.15 fix already landed (`d4ff4d57`) | live; M18/M19 KILLED |
| `scripts/qa/mutation_matrix_90_1.py` `_drive_unresolvable` | :353-419, `UNRESOLVABLE_ERRORS` :349 | ERROR/KILL discrimination by exception TYPE, incl. the fail-open one-liner | live (90.12 fix) |
| `scripts/qa/verify_error_discriminator_90_12.py` | 289 | red-first proof: QA1/QA1b/QA1c call-site renames -> ERROR; DOM stays KILL | live |
| `handoff/current/evaluator_critique_90.2.md` | 799 | 4 verbatim Q/A verdicts; relocation history at :545-546, :607, :788-792 | criterion 6 clause 2 relocated 4x |

## MEASURED IN A SHADOW TREE (read-only; repo untouched)

Control observed GREEN first: `/tmp/shadow90_14_a` copy of the checker + workflow +
fixtures + masterplan -> **105 checks, 0 failed**.

**A FIFTH RELOCATION IS OPEN, and it is the VERDICT dimension.** Mutant **V1** --
`const derivedOnly = verdict.verdict === 'FAIL' ? derived.map(d=>d.severity).slice(0,-1)
: derived.map(d=>d.severity)` (anchor-preserving, so every matrix cell still applies) --
**SURVIVES the entire checker: 105 checks, 0 failed.** It is NOT equivalent: driven
against the checker's own 24-return fixture set it differs on **6 of 24** real returns
(all 6 FAILs), e.g. `wf_6d4dac30-eb7` governing_severities `["UNTAGGED","UNTAGGED",
"UNTAGGED"] -> ["UNTAGGED","UNTAGGED"]`, route unchanged so every route assertion is
structurally blind -- the identical signature the cycle-4 Q/A used to FAIL 90.2.

**Discriminating control C1** -- the SAME drop UNGATED -- is **KILLED** loudly (null
mutant scored KILLED, 4 shape/bucket checks red). So V1's survival is attributable to the
verdict GATE, not to `derivedOnly` being generally unguarded.

Root cause, in the literature's terms: `familyShapes()` (:211-237) varies arity and
comparability but pins `verdict` to `CONDITIONAL` on all four members. The family is
1-way over its own input model {arity} x {details-shape} x {verdict}; pairwise coverage
of {shape} x {verdict} is NOT achieved. This is 90.14's own thesis applied one level up.

## Consensus vs debate (external)

**Consensus:** subsumption/dominator theory (Ammann 2014; Kurtz 2016; Cerebro 2021),
input-space partitioning (Ostrand & Balcer 1988; Offutt & Alluri), the NIST interaction
rule, and the 2026 SMS paper all agree that adequacy is RELATIVE to the input set and
that redundant mutants inflate scores.

**Debate:** how much a mutation score means at all. Papadakis et al. (ISSTA 2016 + ICSE
2018) argue correlations with real fault detection are weak once test-suite size is
controlled -- the adversarial position. The pyfinagent-relevant reconciliation: they do
NOT argue against mutation testing; they argue against reading an *un-deduplicated* score
as evidence. A hand-authored, individually-justified 24-cell matrix is a different
instrument from a generated 100k-mutant score -- but it inherits the obligation to show
the cells are not subsumed by one another.

## Pitfalls (from the literature, mapped to this step)

1. **Score inflation by redundant cells** -- 24/24 KILLED is not evidence unless the cells
   are mutually non-subsuming (ISSTA 2016).
2. **A necessity proof built from the pattern it is proving** -- `attributionMutants()`
   are wrappers whose gate is the shape's own discriminant. The published test is
   leave-one-out against the REAL cells (Gopinath 2016).
3. **Equivalent mutants are inevitable and must be DECLARED, not padded** -- the checker
   already does this once (:510-524); keep that discipline.
4. **Discretisation is the claim** -- NIST: continuous inputs *"must be discretized to a
   few distinct values"*; the family IS a discretisation and its adequacy is a claim about
   THAT model, not the input space.
5. **The oracle, not the input count, is usually the binding cost** (NIST SP 800-142).

## Application to pyfinagent (for Main's contract; RESEARCH ONLY -- do not treat as a plan)

- The provisional commit `d4d27b50` satisfies criteria 1-3, 5 and 6 as written, by
  execution. Criterion 4 ("the family is NECESSARY rather than decorative") is satisfied
  as written but by a weaker instrument than the literature's; a leave-one-out over the
  REAL cells M15-M23 (drop shape S_i, re-score, show >=1 cell flips KILLED->SURVIVED)
  is the published form and would also reveal whether S3 is subsumed by S2 against the
  real cell set (M15/M16 fire on any `emitted.length > 1`).
- The measured V1 survivor is a fifth relocation of criterion 6 clause 2 into the VERDICT
  dimension. The literature says the general remedy is not another shape but a NAMED INPUT
  MODEL with declared factors and a stated t (NIST); then the completeness claim becomes
  auditable and falsifiable instead of open-ended.
- Every "complete" claim in `experiment_results_90.14.md` should be written as
  "t-way complete over the declared input model {factors}", per source 8 and source 9.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL (10: 6 via WebFetch HTML, 4 via the
      sanctioned pdfplumber chain for binary PDFs)
- [x] 10+ unique URLs total (31: 10 read-in-full + 21 snippet-only)
- [x] Recency scan (2024-2026) performed + reported (2 new findings, neither supersedes)
- [x] Full papers/pages read, not abstracts (page/char counts stated per PDF)
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in the caller's INTERNAL SCOPE
- [x] Contradictions / consensus noted (Papadakis adversarial position recorded)
- [x] Claims cited per-claim with URL + access date
- [~] GAP DISCLOSED: the Ostrand & Balcer 1988 CACM primary text was NOT obtained -- the
      KAIST mirror returned a 146-byte HTML stub and ACM DL is paywalled. Its method is
      carried in this brief via secondary sources (search summaries + the Ammann/Offutt
      ISP literature, source 7 read in full), and it is listed snippet-only, not counted.

## Identified but snippet-only (does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://dl.acm.org/doi/10.1145/62959.62964 | paper (Ostrand & Balcer, CACM 1988) | ACM DL paywall |
| https://swtv.kaist.ac.kr/courses/cs453-sw-verification-tech-fall-10/category-partition.pdf | paper mirror | FETCH FAILED -- 146-byte HTML stub, not a PDF |
| https://ieeexplore.ieee.org/iel7/7517740/7528925/07528956.pdf | paper (Kurtz et al. "Are We There Yet?") | FETCH FAILED -- IEEE returned an HTML wall; content used only as a search-snippet claim (dominator mutation score > mutation score for judging completeness) |
| https://ieeexplore.ieee.org/document/7194639/ | paper (TCE, ICSE 2015) | IEEE wall |
| https://discovery.ucl.ac.uk/id/eprint/1573723/7/Jia_07882714.pdf | paper (Trivial Mutant Equivalences via Compiler Optimisations) | superseded for this brief by source 10 |
| https://dl.acm.org/doi/10.1145/3180155.3180183 | paper (Papadakis ICSE'18) | ACM paywall; position covered by source 10 |
| https://coinse.github.io/publications/pdfs/Papadakis2018hi.pdf | preprint of the above | redundant with source 10 |
| https://www.scitepress.org/PublishedPapers/2022/111667/111667.pdf | paper (Generalized Mutant Subsumption) | redundant with sources 1/4 |
| https://www.sciencedirect.com/science/article/abs/pii/S0065245818300305 | survey (Mutation Testing Advances) | Elsevier paywall |
| https://discovery.ucl.ac.uk/10056704/ | same survey, repository record | landing page only |
| https://arxiv.org/pdf/1908.02480 | survey (Constrained Combinatorial Testing) | budget; NIST sources cover the need |
| https://arxiv.org/pdf/2302.14567 | paper (Active Learning with Combinatorial Coverage) | adjacent domain, budget |
| https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=50944 | paper (IPOG t-way generation) | algorithmic, not adequacy-theoretic |
| https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=913807 | NIST note ("commonly used coverage measures do not apply well to CT") | corroborates source 8 |
| https://www.cs.cornell.edu/courses/cs5154/2021sp/resources/InputSpacePartitioning.pdf | course notes (ISP) | tier-5; textbook material |
| https://www.cambridge.org/core/books/abs/introduction-to-software-testing/input-space-partitioning/417E407C3DE99D76D5E8AE4C953ACA1D | textbook ch. (Ammann & Offutt ch.4) | paywall |
| https://dl.acm.org/doi/10.1145/3650212.3680310 | paper (Equivalent Mutants in the Wild, ISSTA 2024) | ACM paywall; recency evidence |
| https://dl.acm.org/doi/10.1145/267580.267590 | survey (Zhu/Hall/May, Software unit test coverage and adequacy, CSUR 1997) | ACM paywall; canonical year-less hit |
| https://conf.researchr.org/home/icst-2026/mutation-2026 | venue page | evidence the topic is still open in 2026 |
| https://onlinelibrary.wiley.com/doi/full/10.1002/stvr.1898 | paper (Mutation-Guided Metamorphic Testing, 2025) | Wiley; recency evidence |
| https://arxiv.org/pdf/2505.05584 | paper (PRIMG, mutant prioritization, 2025) | recency evidence |

