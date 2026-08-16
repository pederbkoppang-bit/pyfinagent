# Research Brief -- phase-86.89

**Topic:** Enumerating BEHAVIOURAL guards from source, and why a syntactic
derivation rule cannot see them.
**Tier:** moderate. **Audit-class:** YES (loop-until-dry, K_required=2).
**Started:** 2026-08-16.

## Envelope (born inert -- phase-86.37; updated in place as sources land)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 22,
  "snippet_only_sources": 23,
  "urls_collected": 45,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "coverage": {
    "audit_class": true,
    "rounds": 8,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "summary": "The 86.85 gate's blind spot is a MISSING VACUITY CHECK, not a coverage bug: both shipped artefacts mutate the SYSTEM, nothing mutates the SPECIFICATION (Kupferman CONCUR 2006). 20% of specs pass vacuously and vacuous passes always indicate a real problem. Enriching the AST rule is the wrong fix -- the misses are StaAgent Type-2 (inadequate, not over-specific), and filtering never raises inference recall (SpecFuzzer: precision 67.83->74.17%, recall 54.57% UNCHANGED). Over-crediting is detected by DISCRIMINATION (NIST paired safe/unsafe probes), not recall. '1 of 4' is Recall_SD against an author-chosen set and cannot be converted to population recall without Recall_DG (ISSTA'24 Thm 3.5); qa.md 4b already requires an author-independent known set, which the 3 FAIL ledger rows supply. ADVERSARIAL: Google raised productive mutants 15->89% with hand-declared unsound rules -- so 'never declare' is too strong; 'never let a declaration be the denominator' is defensible.",
  "brief_path": "handoff/current/research_brief_86.89.md",
  "gate_passed": true
}
```

## Status log

- [t0] Brief created, envelope born inert. Internal exploration starting.
- [t1] Internal inventory written (4 files read in full). External round 1 starting.
- [t2] Round 1: 11 sources read in full. NOT dry.
- [t3] Round 2: +4 (15). NOT dry. Round 3: +3 (18), incl. ADVERSARIAL. NOT dry.
- [t4] Round 4: +1 (19). Round 5: +2 (21), framing finding. Round 6: +1 (22). NOT dry.
- [t5] Rounds 7 and 8: ZERO new full reads each -> dry_rounds=2 = K_required. dry=true.
- [t6] 86.85 handoff artifacts + live ledger read; PART A2 CORRECTS a PART A inference.
- [t7] Envelope flipped to COMPLETE. Gate: 22 >= 5, recency done, audit-loop dry.

---

# PART A -- INTERNAL CODE INVENTORY (the Explore half)

| File | Lines | Role | Status |
|---|---|---|---|
| `scripts/qa/verify_matrix_coverage_86_85.py` | 358 | The derivation gate under audit | LIVE, called by the matrix at `mutation_matrix_86_85.py:247-254` |
| `scripts/qa/mutation_matrix_86_85.py` | 261 | 14-cell mutation matrix over the writer | LIVE |
| `scripts/qa/verdict_ledger_write.py` | 577 | The SUBJECT whose guards are enumerated | LIVE |
| `scripts/qa/verdict_history_86_21.py` | 518 | Sibling reader; carries the same behavioural-guard classes | LIVE, not currently a coverage subject |

## A1. The enumeration rule, exactly as shipped

Two clauses only (`verify_matrix_coverage_86_85.py:37-52` doc; implementation
`:117-124` and `:127-155`):

- **GUARD-RAISE** -- `_is_guard_raise` (`:117-124`): the node is an `ast.Raise`
  whose exception (unwrapping `ast.Call`) is `ast.Name` with `id == "LedgerError"`.
- **GUARD-BRANCH** -- `_direct_refusal` (`:127-155`): an `ast.If` whose own body or
  `orelse` reaches a GUARD-RAISE, **or** a `return` whose unparsed text contains
  `"EXIT_"` but not `"EXIT_OK"` (`:151`), **without descending into a nested
  `ast.If`** (`:139-140`, `continue`).

Both clauses are keyed on **refusal**: an exception type, or a non-OK exit code.
Neither clause has any way to express "this value must be ordered", "this tuple
must contain field X", "this branch must return a *distinguishing* value", or
"these two timestamps must be distinct fields".

## A2. Measured recall of the shipped rule -- 1 of 4 (25%)

Stated by the module about itself, `verify_matrix_coverage_86_85.py:4-12`:

> "Measured known-member recall against the three prior FAILs that motivated it is
> **1 of 4**: dropping the cell for the ordering guard, the step_id-in-key guard,
> or the cycle-fallback guard all leave this gate GREEN; only the fail-loud-I/O
> guard is demanded."

The four known members and why the rule sees only one:

| Known member | Cell | Target site | Node shape | Seen by the rule? |
|---|---|---|---|---|
| fail-loud I/O | M8 | `verdict_ledger_write.py:259` `raise LedgerError(f"failed to append to {path}: {exc}", EXIT_IO) from exc` | `ast.Raise`/`LedgerError` | **YES** (GUARD-RAISE) |
| ordering (oldest->newest) | M6 | `verdict_ledger_write.py:295` `return out` | bare `ast.Return` in a `FunctionDef` body, no enclosing `ast.If` | NO |
| step_id in the dedup key | M9 | `verdict_ledger_write.py:157` `return (step, f"run:{run}")` | `ast.Return` inside `if run:` -- but the returned text has no `EXIT_`, so `_direct_refusal` is False (`:151`) | NO |
| cycle fallback (M11 constant key / M12 branch deleted) | M11, M12 | `verdict_ledger_write.py:158-160` | same shape as above | NO |

A fifth behavioural guard, **event time vs write time** (M5, target
`verdict_ledger_write.py:232` `"date": event_date or stamp.date().isoformat()`), is
an `ast.Dict` value inside `build_row` -- not a raise, not an `ast.If` at all, so it
is invisible to both clauses. It is *not* in the module's own 1-of-4 denominator,
which is itself a finding: the known set was drawn from the three prior FAILs, and
M5 predates them.

**Why the misses are structural, not a tuning problem.** All four missed members are
guards over a *returned value*: its order, its composition, its distinguishing power,
its provenance. The rule's vocabulary is `raise` and non-OK `return`. There is no
predicate over "what the function returns on the success path", because on the
success path nothing refuses. A refusal is a *syntactic event*; a behaviour is a
*relation between inputs and outputs across runs*. No enrichment of the two clauses
closes that -- it is a category difference, which is the thing the external half has
to be checked against.

## A3. The `ast.Try` over-crediting, as it actually happened

`spans_with_ancestors` (`:178-205`) walks parents from the guard up to the enclosing
function, collecting spans. The measured note at `:191-201`:

> "The first version included `ast.Try`, and `main` wraps its whole body in one
> `try:`. Every anchor inside that body therefore 'overlapped' every guard in it,
> and the checker credited cell M13 with covering a raise in a completely different
> branch. The tell: removing cell M14 left the gate GREEN."

The subject site is `verdict_ledger_write.py:542-572` -- one `try:` from `:542` to the
`except LedgerError` at `:568`, enclosing the `--emit-sequence` guard (`:544-545`),
the append-arg guard (`:549-553`) and `build_row`/`append_row` dispatch. With `Try` in
the ancestor set, the containment relation "the anchor is inside something that
conditions this refusal" degenerates: a `try` is a *scope*, not a *condition*, so
containment stops discriminating and every guard in the function reads as covered by
any cell anchored anywhere in it. The current code keeps `ast.If` only (`:202`) --
`If` genuinely conditions the refusal below it.

**The detection technique that caught it is the load-bearing part**: the gate was
tested for its ability to go RED by *deleting a cell that should have been demanded*
(M14) and observing that it stayed GREEN. That is a known-member probe, run against
the checker rather than the subject.

## A4. The self-control, and its limit

`main` plants `SYNTHETIC` (`:262-269`) -- a function containing
`if value is None: raise LedgerError(...)` -- into a COPY, and requires (a) the
enumerator to produce a new fingerprint and (b) that fingerprint to be reported
UNCOVERED; otherwise exit 2 / FAILED GATE (`:290-310`).

The limit is exact and is the crux of step 86.89: **the planted guard is drawn from
the same syntactic class the rule already recognises** (a `raise LedgerError` inside
an `ast.If`). A positive control built from the pattern under test cannot measure the
pattern's recall -- it can only prove the pattern is wired up. Planting a *behavioural*
guard (e.g. a reversed return, a key with a field dropped) would currently fail the
self-control by returning "no new fingerprint" -> exit 2, which is the honest
fail-closed direction but tells you the rule's blind spot rather than fixing it.

## A5. Sibling surface not yet under any coverage gate

`verdict_history_86_21.py` carries the same behavioural classes and has NO derivation
gate over it at all:

- **ordering / reset semantics** -- `consecutive_conditionals` (`:85-106`) scans
  `reversed(self.verdicts)` and breaks on any non-CONDITIONAL. Order-dependent, and
  the module's own history records a one-sided-threshold survivor fixed at `:305-321`.
- **not-knowable vs zero** -- `:98-99` returns `None` for
  `UNPARSEABLE/LEDGER_EMPTY/LEDGER_MISSING`. This is a *value* guard (`None` not `0`),
  not a raise: invisible to a GUARD-RAISE/GUARD-BRANCH rule.
- **exact step matching** -- `:164` `sid.strip() != step_id`; a prefix mutant survived
  the cycle-1 matrix (`:357-366`).
- **printed-output guards** -- `_report` (`:201-277`): the cycle-4 Q/A built ten
  mutants against printed output and **nine survived** (`:428-437`). Every one of
  those is a behavioural guard over stdout text, with no `raise` anywhere near it.

## A6. Where the guards' pyfinagent stakes sit

M6 (ordering) is not cosmetic: `emit_sequence`'s docstring (`:349-354` of the
self-test) records that reversing `[PASS, C, C]` to `[C, C, PASS]` takes
`enforceEscalation` from `n=2 / auto_fail=true` to `n=0 / auto_fail=false` -- it
**silently disarms the 3rd-CONDITIONAL escalation**. M9/M11/M12 all fail in the
under-count / fail-OPEN direction. So the four guards the derivation rule cannot see
are precisely the ones whose failure mode is silent and permissive.

---

# PART B -- EXTERNAL RESEARCH

## B0. Search queries run (three-variant discipline, `research-gate.md`)

| Variant | Query |
|---|---|
| year-less canonical | `specification mining dynamic invariant detection Daikon recall precision evaluation` |
| year-less canonical | `mutation testing coverage over-approximation false positive "covered" guard not actually tested` |
| year-less canonical | `measuring recall static analysis rule known seeded fault benchmark chosen in advance overfitting to benchmark` |
| current-year (2026) | `AST pattern matching versus dataflow analysis recall static analysis vulnerability detection 2026` |
| current-year (2026) | `specification mining invariant inference LLM 2026 recall behavioural properties arxiv` |
| last-2-year (2025) | `requirements traceability matrix validation completeness verification technique 2025` |
| (round 2, below) | checked coverage / oracle quality; SAST recall; coverage-effectiveness correlation |

## B1. Round 1 -- read in full

### B1.1 Static analysis: what a syntactic rule provably cannot express

**CodeQL, "About data flow analysis"** (accessed 2026-08-16):
> "Nodes in the abstract syntax tree represent syntactic elements such as
> statements or expressions. Nodes in the data flow graph, on the other hand,
> represent semantic elements that carry values at runtime."

and

> "the data flow graph does not reflect the syntactic structure of the program,
> but models the way data flows through the program at runtime."

This is the formal statement of the 86.89 problem. `verify_matrix_coverage_86_85.py`
operates entirely on `ast.walk` over syntactic elements (`:208-216`). A guard over
*what a function returns* is a property of a semantic element carrying a value; there
is no AST node whose presence/absence encodes it. CodeQL also splits **local** data
flow ("within a single function") from **global** ("data flow between functions and
through object properties") -- the ordering guard M6 is even harder than local: it is
a property of a *sequence built across loop iterations*, which needs value reasoning,
not reachability.

**Semgrep, "Data-flow analysis overview"** (accessed 2026-08-16) is explicit that even
the dataflow tier is approximate:
> "All *potential* execution paths are considered, even though some may not be
> feasible."

> "Semgrep ignores the effects of `eval`-like functions on the program state. It
> doesn't make worst-case sound assumptions, but rather 'reasonable' ones."

> "Expect both false positives and false negatives."

Semgrep's engine is "an intraprocedural data-flow analysis engine", and where dataflow
support is thinner, unsupported constructs are **silently ignored**, "causing potential
false negatives". **Silent ignoring is the exact failure shape of the 86.85 gate**: an
un-enumerable guard produces no complaint, only a smaller guard set.

*Consensus note:* neither vendor doc claims a syntactic rule can be extended to cover
value properties. Both treat it as a tier change (syntax -> dataflow -> taint), each
tier bought with a different engine, and **each still unsound**.

### B1.2 Specification mining: derive properties from execution, not declaration

**Daikon manual** (accessed 2026-08-16) -- the canonical dynamic invariant detector:
> "Daikon is an implementation of dynamic detection of likely invariants; that is,
> the Daikon invariant detector reports likely program invariants."

It "runs a program, observes the values that the program computes, and then reports
properties that were true over the observed executions." Two structural limits, both
material here:

1. **The grammar is fixed in advance.** The manual's invariant list (§5.5) enumerates
   ~80 hardcoded invariant classes (`EltOneOf`, `EltRangeInt`, `CommonSequence`, ...).
   Daikon can only report a property it has a template for. *This is the same failure
   mode as the 86.85 rule, one level up*: a template list is a hand-declared register
   too -- it is just a register of PROPERTY SHAPES rather than of guard instances.
   Notably, Daikon's list DOES include sortedness/ordering and containment templates,
   which is precisely the class the pyfinagent rule cannot see. So the class is
   *known to be mineable*; it is not mineable *syntactically*.
2. **Unsound by construction.** Reported invariants "were true over the observed
   executions" -- an invariant holding across all tests may still be false in general,
   and `--conf_limit` exists to filter probable-spurious output.

**Improving Dynamic Specification Inference with LLM-Generated Counterexamples**
(arXiv:2604.10761, 2026; accessed 2026-08-16) supplies the missing NUMBERS. Over 43
methods, baseline SpecFuzzer (Daikon-lineage):

| Metric | SpecFuzzer | SpecFuzzer + LLM counterexamples |
|---|---|---|
| Precision | 67.83% | 74.17% |
| **Recall** | **54.57%** | **54.57% (unchanged)** |
| F1 | 51.39% | 53.94% |

> "Daikon is very useful... However, the technique is known to have expressiveness
> limitations, and to have its precision subject to the thoroughness of the test suite
> used for inference." (§II-A)

**The single most decision-relevant external number in this brief is that unchanged
recall.** An entire LLM counterexample loop -- 1,009 counterexamples invalidating
1,877 of 18,605 inferred assertions (~10.09%) -- moved PRECISION by ~7 points and moved
RECALL **not at all**. Counterexample filtering can only remove wrong candidates; it
cannot invent a candidate the grammar never proposed. **Recall in specification mining
is bounded by the candidate grammar, and filtering never raises it.** Any 86.89 design
that adds a *validator* on top of the existing enumeration will move precision and
leave the 1-of-4 recall exactly where it is.

**Beyond Basic Specifications?** (arXiv:2602.00715, 2026; accessed 2026-08-16) --
fetched and read; it reports aggregate metrics (NVP, NSVP, NVTC, RT) across Basic /
Verifiable / Axiom / Full configurations and does **NOT** disaggregate by logical
construct, so it does **not** answer "which construct classes are missed". Recorded as
a genuine negative result rather than mined for a quotable line. Its one transferable
finding: "CA exhibits the highest reduction rate for most models...indicating that
relying on unverifiable axiom constructs is more prone to introducing instability"
(§4.2.2) -- i.e. **unverifiable declared properties destabilise the artefact**, which
is the argument against a declared register with no verification leg.

### B1.3 Over-crediting in coverage tooling

**Stryker Mutator, "Mutant states and metrics"** (accessed 2026-08-16) gives the
canonical denominator split:

- Killed: "When at least one test failed while this mutant was active."
- Survived: "When all tests passed while this mutant was active."
- **No coverage: "The mutant isn't covered by one of your tests and survived as a result."**
- Mutation score: `detected / valid * 100`; alternative: `detected / covered * 100`.

Two separately-quotable facts: (a) "No coverage" is a **distinct state from Survived**
and both are "undetected"; (b) there exist **two scores with different denominators**,
and the `detected / covered` variant is exactly the over-crediting shape -- it scores
only over what the harness already reaches, so growing the blind spot RAISES the score.
The pyfinagent `ast.Try` bug was a `detected/covered` fallacy in a different coordinate:
by widening the *ancestor span* the checker widened what counted as "covered", so
coverage went up while real discrimination went down.

**NIST / SATE V, "Evaluating Bug Finders"** (accessed 2026-08-16) names the metric that
detects exactly this, and it is NOT precision or recall:

> "A complementary measurement is discrimination. It reflects whether a tool can detect
> a weakness when there is one, but remain silent when a similar code construct is used
> safely. For example, a tool reporting all occurrences of calls to a dangerous
> function, whether it is used correctly or not, could still achieve a decent score if
> solely based on recall and precision."

> "a tool reporting all sites would still achieve a precision of 50 %, as it would be
> right half of the time. The problem can be mitigated by introducing the
> discrimination rate, which accounts for true-positives only if the tool did not
> incorrectly report the same defect at the safe site (false-positive)."

**This is the published name for the `ast.Try` defect and for its detection.** A checker
crediting every guard inside one `try:` is a tool "reporting all sites". Recall stays
100%, precision looks fine, and only a **paired** probe -- the guard-that-should-be-
demanded next to the one-that-should-not -- separates them. The pyfinagent tell was
exactly the paired form: delete M14 and observe the gate stays GREEN.

NIST also states the operational recipe: "synthetic test cases come in pairs, one
containing a weakness and its counterpart" [safe], and discrimination counts a hit only
if the safe twin is NOT flagged.

### B1.4 Measuring recall against a known-member set

**Total Recall? How Good Are Static Call Graphs Really?** (Helm et al., **ISSTA '24**,
peer-reviewed; accessed 2026-08-16) is the methodological source for sub-question (5).

> "In order to measure these two metrics, a ground-truth CG is necessary. Constructing
> such a ground truth would entail capturing all possible program executions which is
> an undecidable problem."

Their three surveyed alternatives map one-to-one onto options available to 86.89:

1. **Micro-benchmarks** with hand-crafted expected results -- "such micro-benchmarks
   (by construction) lack insights into a CG's recall for real-world programs. A single
   commonly used, hard to analyze language feature like reflection might
   disproportionately affect the recall of static CGs of real-world programs."
   *(= the 1-of-4 known set: four hand-picked guards cannot tell you the rule's recall
   over the guard population.)*
2. **Differential size comparison** -- "A smaller CG is not necessarily more precise -- the
   smaller size could also be a result of lower recall." *(= "the gate went GREEN" is
   not evidence of coverage; it is equally evidence of a smaller enumerated set.)*
3. **Dynamic baselines** -- execute and record traces. Recommended.

And the reporting convention, stated as a theorem (Thm 3.5):

> `RecallSD * RecallDG <= RecallSG <= 1 - (1 - RecallSD) * RecallDG`

where S = the static result, D = the measurable dynamic baseline, G = the unattainable
ground truth. They first establish that **you cannot even bound recall naively**:

> "establishing a direct lower bound for recall is not possible. RecallSD could exceed
> or fall short RecallSG. These two measurements are ratios computed from different
> nominators and denominators, thus lacking an inherent relationship."

**Direct consequence for 86.89:** the "1 of 4" in
`verify_matrix_coverage_86_85.py:4-12` is `Recall_SD` -- recall against a declared known
set D of four guards. It is **NOT** an estimate of the rule's recall over the real guard
population, and by Thm 3.5 it cannot be converted into one without a second quantity
(`Recall_DG`: how completely the known set covers the population). The honest reporting
form is therefore a **two-part figure**: "1 of 4 known members demanded (25%), with the
known set drawn from N prior FAILs and of unmeasured completeness." A bare "recall =
25%" over-claims; a bare "the gate is GREEN" over-claims far worse.

**NIST/SATE V** supplies the pre-registration discipline for choosing the known set:

> "we distilled three test case characteristics required to calculate these metrics:
> statistical significance, ground truth and relevance ... the perfect test cases are a
> set of large production software, developed according to typical industry practices
> and whose defects are all identified. Unfortunately, such cases do not exist ...
> However, test cases exhibiting any two of the three characteristics are readily
> available."

So a known-member set can hold at most two of {significance, ground truth, relevance}.
NIST also measures the cost of picking synthetic: "the precision they yield from
synthetic test cases is overall higher than their precision on production" -- the
overfitting warning, from the body of the paper rather than a blog.

### B1.5 The declared-register pattern (contracts, properties, traceability)

Round 1 was **partially dry** here and the brief says so rather than padding:

- **Fowler, "Consumer-Driven Contracts"** (accessed 2026-08-16) defines the register
  ("closed and complete with respect to the entire set of functionality demanded of it
  by its existing consumers"; "derived from the union of existing consumer
  expectations") but the page **asserts** rather than verifies -- it "does not
  explicitly detail a verification process comparing declared expectations against
  actual runtime provider behavior", and does not address contract staleness. Recorded
  as read, and as a gap.
- **Pact docs** (accessed 2026-08-16) supply the verification leg Fowler's page lacks:
  contract testing checks "that all the calls to your test doubles return the same
  results as a call to the real application would" -- i.e. **the declaration is replayed
  against the real implementation**, which is the difference between a register and an
  asserted list. And the completeness caveat, verbatim: "only parts of the
  communication that are actually used by the consumer(s) get tested", so "any provider
  behaviour not used by current consumers is free to change without breaking tests."
  *A declared register is complete only over its declarers -- exactly the 86.85 hand-list
  failure, restated as a design property rather than a mistake.*
- **Hypothesis, "What is property-based testing"** (accessed 2026-08-16): "Property
  based testing is the construction of tests such that, when these tests are fuzzed,
  failures in the test reveal problems with the system under test that could not have
  been revealed by direct fuzzing of that system." The page does **not** enumerate
  property classes or limits -- recorded as read and thin. The usable point is the
  contrast it does draw: fuzzing needs "minimal understanding of ... behaviour", while
  property-based testing requires an explicitly reasoned expected behaviour. **A
  property must still be declared by a human; PBT verifies a declaration, it does not
  discover one.**

**Round 1 result: 11 sources read in full, many new findings. NOT dry.**

## B2. Round 2 -- read in full

### B2.1 The published name for "covered but not actually checked"

**Schuler & Zeller, "Checked coverage: an indicator for oracle quality"**, *Software
Testing, Verification and Reliability* 23:531-551 (2013), peer-reviewed journal
(accessed 2026-08-16; curl + pypdf, verbatim text extracted):

> "A known problem of traditional coverage metrics is that they do not assess oracle
> quality -- that is, whether the computation result is actually checked against
> expectations. In this paper, we introduce the concept of checked coverage -- the
> dynamic slice of covered statements that actually influence an oracle. Our
> experiments on seven open-source projects show that checked coverage is a sure
> indicator for oracle quality and even more sensitive than mutation testing."

> "A high coverage does not tell anything about oracle quality. It is perfectly
> possible to achieve a 100% coverage and still not have any resu[lt checked]" (§1)

The mechanism: "we compute the dynamic backward slice of test oracles -- that is, all
statements that contribute to the checked result" (§2). And the paper's own
sensitivity experiment is **oracle decay** -- "oracle quality artificially reduced by
removing checks" (§1/§3.4), i.e. **they measure the metric by deliberately deleting
checks and observing whether the metric drops.** That is precisely the probe pyfinagent
used to catch the `ast.Try` bug (delete M14, observe the gate stays GREEN) -- so the
project's ad-hoc technique has a peer-reviewed name and an established protocol.

**Oracle-based Test Adequacy Metrics: A Survey** (arXiv:2212.06118, ar5iv HTML;
accessed 2026-08-16) generalises it and supplies the discriminating definition:

> "it is entirely possible for a test suite with zero test oracles to achieve 100% code
> coverage, resulting in a poor quality test suite with low fault-detection
> effectiveness." (§1)

> "an element from the coverage domain is covered when it is executed and that code
> element affects a value checked by a test oracle via dependency chain (data and
> control dependency)" (§3)

The survey's Table 1 ranks metrics by which PIE conditions they enforce -- **regular
coverage "Ensures only execution"**, checked/state/observable coverage add propagation,
and **only mutation coverage ensures all four (execution, infection, propagation,
detection)**. Empirically (§5.2.2): "checked coverage is always lower than statement
coverage, and there is an average difference of 24%, meaning that **24% of the executed
statements do not influence any test oracle**." Also §5.5.1: "adding observability
improves fault detection by 11.94% on average, over regular coverage and 125.98% for
output-only oracles."

**Mapping to 86.85, and it is exact.** The checker's coverage rule (b) TEXTUAL
(`verify_matrix_coverage_86_85.py:253-257`) credits a guard when the cell's anchor
*overlaps a span*. Overlap is a **reachability/adjacency** relation -- the survey's "only
execution" tier. It does not establish that the mutation **propagates to** the guard's
behaviour, let alone that a check **detects** it. Rule (a) STRUCTURAL (`:246-248`) is
stronger (fingerprint disappearance ~ infection) but only fires for guards the
enumeration can name. So the gate is a *mixture* of the weakest tier and a
name-restricted stronger tier -- which is why widening the ancestor set (adding
`ast.Try`) inflated it: it added pure adjacency.

### B2.2 Coverage as a target is a weak proxy, at scale

**Inozemtseva & Holmes, "Coverage Is Not Strongly Correlated with Test Suite
Effectiveness"**, ICSE 2014 (ACM Distinguished Paper; accessed 2026-08-16; curl + pypdf):

> "we generated 31,000 test suites for five systems consisting of up to 724,000 lines of
> source code."

> "We found that there is a low to moderate correlation between coverage and
> effectiveness when the number of test cases in the suite is controlled for. In
> addition, we found that stronger forms of coverage do not provide greater insight
> into the effectiveness of the suite. Our results suggest that coverage, while useful
> for identifying under-tested parts of a program, should not be used as a quality
> target because it is not a good indicator of test suite effectiveness."

Two consequences for 86.89. (i) **"Stronger forms of coverage do not provide greater
insight"** is a direct warning against the tempting fix of enriching the AST rule
(add `ast.Return`, add `ast.Dict`, add `ast.Compare`): a richer syntactic criterion is
still a coverage criterion. (ii) The *sanctioned* use of coverage is exactly what the
86.85 gate does well -- "identifying under-tested parts of a program", i.e. finding the
guard with NO cell (it found `main`'s CLI validation). The paper's warning is against
promoting it to a quality TARGET, which is what a green gate implies if its licence
sentence is dropped. The checker's own closing lines already say this
(`verify_matrix_coverage_86_85.py:350-352`: "This licenses ONLY: 'no guard the
enumeration can see is unmutated'").

### B2.3 Published recall for pattern-based rules on real code

**A Comprehensive Study on SAST Tools for Android** (arXiv:2410.20740; accessed
2026-08-16) gives per-tool recall on four benchmarks. GHERA: AUSERA 90%, AndroBugs
73.3%, JAADAS 66.7%, APKHunt 85.7%, SUPER 33.3%, **QARK 30.8%**. CVE benchmark:
SPECK 91.8%, APKHunt 94.9%, AUSERA 89.7%, SUPER 52.3%, QARK 68.1%.

The mechanism sentence is the transferable one:

> "The tools mainly use the method as pattern-matching for vulnerability detection,
> leaving a notable gap for scenario-related logical vulnerability types."

> "79% (27/34) of unsupported types were challenging to detect without a deep
> understanding of the application's scenario logic."

**79% of the unsupported classes are unsupported because they are LOGICAL rather than
syntactic.** That is the same partition as pyfinagent's: the one demanded guard (M8) is
syntactic (`raise LedgerError`); the three-to-four missed ones are logical (ordering,
key composition, distinguishing-value, event-vs-write time). And note the spread --
30.8% to 94.9% recall across tools on the same class of task -- so "a pattern rule
gets ~25%" is not an outlier; it is the low end of a well-populated distribution, and
the high end is reached by tools that are *not* pattern-only.

Ground-truth construction, relevant to sub-question (5): they filtered 8,451 CVE
entries down to "250 Android-specific CVEs and 229 APKs, and 34 vulnerability types"
with "262 vulnerability instances", and labelled by consensus -- "three co-authors
independently labeled...In case of disagreement, the final decision was made by
majority voting." **The known set was constructed from an external corpus by
independent labellers, before and independently of the tools being scored.**

**Round 2 result: 4 more sources read in full (15 total), several new findings. NOT dry.**

## B3. Round 3 -- read in full

### B3.1 [ADVERSARIAL] Google: curated hand-written rules BEAT the derived approach

`verify_matrix_coverage_86_85.py:23-25` states the project's doctrine flatly: "**No cell
declares what it covers.** A declaration would be a hand-list wearing a different hat."
The strongest published counter-evidence is Google's own:

**Practical Mutation Testing at Scale** (Petrovic, Ivankovic, Fraser, Just;
arXiv:2102.11378, ar5iv HTML; accessed 2026-08-16):

> "developers at Google initially classified 85% of reported mutants as unproductive"
> (§I)

> "Heuristics are implemented by matching AST nodes with the full compiler information
> available...Some heuristics are unsound: they employ fuzzy name matching...but...we
> have had much more important improvements of perceived mutant usefulness from unsound
> heuristics" (§III-B5)

> "increased the ratio of productive mutants...from 15% to 89%...over six years"
> (§VII)

**This qualifies the 86.85 doctrine and must be carried into the contract.** At the
largest deployed scale in the industry, the thing that made mutation testing usable was
a **hand-curated, deliberately unsound, human-declared suppression rule set** applied on
top of AST matching -- 15% -> 89% productive. Google did not derive its way out; it
declared its way out, and accepted unsoundness for usefulness.

The reconciliation is that Google and 86.85 are curating in **opposite directions**, and
this is the key distinction for the contract: Google hand-declares what to **EXCLUDE**
(suppression, precision-raising, fail-*safe* -- a wrongly-suppressed mutant costs a
missed test); 86.85's failed hand-lists declared what to **INCLUDE** (enumeration,
recall-raising, fail-*open* -- a forgotten guard costs the whole gate). A hand-list is
sound as a *filter over a derived population* and unsound as *the population itself*.
So "never declare" is too strong; "never let a declaration BE the denominator" is the
defensible rule.

### B3.2 The 2026 framework that names this exact obligation class

**Beyond Code Reasoning: A Specification-Anchored Audit Framework** (SPECA;
arXiv:2604.26495v1, 2026; accessed 2026-08-16) is the closest published match to step
86.89 and it independently names two of pyfinagent's four missed guards:

> "When a vulnerability arises not from how code is written but from what the
> specification *requires*, these approaches lack the representational vocabulary to
> detect it" (§1)

> "implementation-pattern obligations (e.g., cache keys and deduplication keys computed
> from complete inputs)" (§3.3)

Its obligation taxonomy is: **input coverage, path coverage, concurrency safety,
temporal validity, and implementation-pattern obligations**. Map onto the four missed
pyfinagent guards:

| Missed guard | SPECA class |
|---|---|
| M9 step_id in the dedup key | implementation-pattern -- "deduplication keys computed from complete inputs" (verbatim) |
| M11/M12 cycle fallback distinguishes rows | implementation-pattern (key must remain distinguishing) |
| M5 event time vs write time | **temporal validity** |
| M6 ordering of the emitted sequence | path/temporal -- ordering of a derived sequence |

Its derivation direction is the opposite of 86.85's: SPECA is **top-down** -- "decompose
the property's assertion into verifiable sub-claims, read the enforcement code
completely" (§3.3) -- then *anchors* each sub-claim to code, and records a gap as a
finding when the proof fails. Its worked example is a dedup defect found by data-flow
tracing, not by pattern matching: "The auditor traces the data flow and discovers that
the function receives commitments_bytes (the original array with duplicates) rather
than unique_commitments (the deduplicated array)" (§3.3).

**The load-bearing consequence: the derivation SOURCE has to change, not the AST rule.**
Behavioural obligations are derivable from a SPECIFICATION and anchorable to code; they
are not derivable FROM code alone, because (SPECA §1) the code lacks the representational
vocabulary. pyfinagent has a candidate specification artefact already -- each cell's
`description` string in `mutation_matrix_86_85.py:32-153` is a prose obligation
("collapse event time into write time", "drop step_id from the dedup key") -- but it is
currently an output, not an input, and nothing anchors it.

### B3.3 Duplicated mutants inflate the denominator

**Trivial Compiler Equivalence** (Papadakis, Jia, Harman, Le Traon, **ICSE 2015**,
peer-reviewed; accessed 2026-08-16; curl + pypdf):

> "TCE is directly applicable to real-world programs and can imbue existing tools with
> the ability to detect equivalent mutants and a special form of useless mutants called
> duplicated mutants."

> "on large real-world programs, TCE can discard more than 7% and 21% of all the mutants
> as being equivalent and duplicated mutants respectively. A human-based equivalence
> verification reveals that TCE has the ability to detect approximately 30% of all the
> existing equivalent mutants."

Two transferable points. (i) **Duplicated mutants** -- distinct mutants that are mutually
equivalent -- are 21% of all mutants on real C programs. A matrix of 14 cells may
therefore hold materially fewer than 14 *distinct* obligations, which is an
over-statement of breadth independent of the coverage rule. (ii) Equivalent mutants
distort the score: "equivalent mutants should be removed from the test effectiveness
measure called mutation score, i.e., the ratio of the exposed mutants to..." -- and even
the best automatic detector gets only ~30% of them, so **the denominator of any mutation
figure is itself uncertain by a measured, non-trivial margin**. This is a second,
independent reason not to read "14/14 KILLED" as a completeness statement -- which the
matrix's own comment at `mutation_matrix_86_85.py:236-244` already says for a different
reason.

**Round 3 result: 3 more sources read in full (18 total), incl. one ADVERSARIAL. NOT dry.**

## B4. Round 4 -- read in full

**Metamorphic Testing of RESTful Web APIs** (Segura et al., IEEE TSE 2017 preprint;
accessed 2026-08-16). A metamorphic relation is "a relation among inputs and outputs
that should hold for any test case without requiring a full oracle" -- the technique for
checking a *behavioural* property when you cannot state the expected output. The paper's
MR catalogue includes ordering relations ("Relations that verify consistent behavior
under different input orderings"). The constraint that matters for 86.89:

> "metamorphic relations must be identified by humans based on problem domain knowledge"

and the paper "does not present evidence of fully automated MR derivation, positioning
MR design as a manual, expert-driven activity".

**So the literature's own answer to "can a behavioural obligation be auto-derived?" is
NO for the relation itself.** MT gives you a way to *check* an ordering property cheaply
and without a full oracle; it does not give you a way to *discover* that ordering is the
property. Combined with B1.2 (mining recall capped by the candidate grammar) and B1.5
(PBT verifies a declared property, does not discover one), three independent literatures
converge on the same split: **discovery of the obligation is human/specification work;
verification of it is machine work.**

**Round 4 result: 1 new source read in full (19 total), 1 new finding. NOT dry.**

## B5. Round 5 -- read in full. THE FRAMING FINDING

Formal verification solved this exact problem thirty years ago and gave both halves
names. The gate's blind spot is not a coverage bug; it is a **missing vacuity check**.

**Kupferman, "Sanity Checks in Formal Verification"** (CONCUR 2006; accessed 2026-08-16;
curl + pypdf):

> "Two leading sanity checks are vacuity and coverage. In vacuity, the goal is to detect
> cases where the system satisfies the specification in some unintended trivial way. In
> coverage, the goal is to increase the exhaustiveness of the specification by detecting
> components of the system that do not play a role in verification process."

> "both are based on repeating the verification process on some mutant input. **In
> vacuity, mutations are in the specifications, whereas in coverage, mutations are in
> the system.** This observation enables us to adopt work done in the context of vacuity
> to coverage, and vise versa."

And the empirical rate, attributed to Beer et al. [BBER01]:

> "vacuity is a serious problem: our experience has shown that typically **20% of
> specifications pass vacuously** during the first formal-verification runs of a new
> hardware design, and that **vacuous passes always point to a real problem** in either
> the design or its specification or environment"

The canonical instance is **antecedent failure**: verifying `AG(req -> AF grant)` on a
system "in which requests are never sent" -- the property passes because its precondition
never fires. **`AG(req -> AF grant)` passing vacuously is structurally identical to a
mutation cell being credited with covering a guard it never reaches.**

**Chockler, Kupferman & Vardi, "Coverage Metrics for Formal Verification"** (CHARME
2003; accessed 2026-08-16; curl + pypdf):

> "Even when the system is proven to be correct, there is still a question of how
> complete the specification is, and whether it really covers all the behaviors of the
> system."

> "It turns out that **no single measure can be absolute**, leading to the development of
> numerous coverage metrics whose usage is determined by industrial verification
> methodologies."

### Why this is the decision-deciding finding

Sort the two pyfinagent artefacts by which input they mutate:

| Artefact | What it mutates | FV name | What it can find |
|---|---|---|---|
| `mutation_matrix_86_85.py` (CELLS) | the SYSTEM (`verdict_ledger_write.py`) | **coverage** | a guard that cannot fail |
| `verify_matrix_coverage_86_85.py::coverage()` (`:231-257`) | the SYSTEM (in-memory mutants) | **coverage** | a guard with no cell |
| *nothing* | the SPECIFICATION (the CELLS list, the enumeration rule) | **vacuity** | *a cell/guard-pairing that passes trivially* |

**Both shipped artefacts mutate the system. Neither mutates the specification.** That is
precisely the class of defect that escaped:

- The `ast.Try` over-crediting was a **vacuous pass** -- the gate reported GREEN for a
  reason unrelated to the property. It was found by **deleting cell M14 and observing the
  gate stayed GREEN**, i.e. by a *specification-side mutation*, run once, by hand,
  ad hoc (`verify_matrix_coverage_86_85.py:198-199`, "The tell: removing cell M14 left
  the gate GREEN").
- The self-control (`:290-310`) is *also* a specification-side mutation -- it plants a
  guard -- but it plants one **from the class the rule already recognises**, so it detects
  wiring, not blindness. In FV terms it checks that the property is non-trivially
  satisfiable, not that the property set is complete.

The 20%-vacuous figure and "vacuous passes always point to a real problem" together say
the ad-hoc M14 probe should be **systematic and re-runnable**, not a one-off comment.
And "no single measure can be absolute" (CHARME 2003) is the strongest available
warning against the temptation to make one enriched AST rule the whole gate.

**Round 5 result: 2 more sources read in full (21 total), the framing finding. NOT dry.**

## B6. Round 6 -- read in full

**StaAgent: An Agentic Framework for Testing Static Analyzers** (arXiv:2507.15892;
accessed 2026-08-16) is the published method for the question "is my enumeration rule
over-specific?", and its diagnosis is a two-way classification:

> "This agent takes the seed bug and its corresponding semantically equivalent mutants
> to perform metamorphic testing on a specific bug detection rule" (§3.5)

> Type 1: "The static analyzer detects the seed bug, but not all of its semantically
> equivalent mutants" indicates "the rule is overly specific and lacks robustness."
> (§3.5)

> Type 2: "The static analyzer fails to detect both the seed bug and at least one of its
> mutants" suggests "the bug detection rule is fundamentally inadequate." (§3.5)

Results: "64 problematic rules in the latest versions of these five static analyzers
(i.e., 28 in SpotBugs, 18 in SonarQube, 6 in ErrorProne, 4 in Infer, and 8 in PMD)"
(Abstract), and §6.1 notes mutations introducing "complex branching constructs...pose
greater challenges to static analyzers and is more effective in uncovering flaws".

**Applied to `enumerate_guards`:** the pyfinagent rule is provably **Type 2** on the
behavioural class -- it misses the guard AND every semantically equivalent restatement
of it, because no restatement of "return the sequence in order" is a `raise
LedgerError` or a refusing `ast.If`. That distinction matters for the contract: Type 1
is fixed by broadening the pattern; **Type 2 is not fixable by broadening the pattern**,
because there is no pattern to broaden toward. Even seeding is instructive: the
framework's oracle is seed-bug + semantically-equivalent-mutants, which is exactly the
probe the 86.85 self-control lacks (it plants ONE guard, from the recognised class, with
no equivalent variants).

**Round 6 result: 1 more source read in full (22 total), 1 new finding. NOT dry.**

## B7-B8. Rounds 7 and 8 -- DRY (loop-until-dry satisfied, K=2)

**Round 7** probed two fresh angles -- (a) soundness/over-approximation of coverage
*measurement* tooling, (b) 2024-window specification mining for temporal/ordering
API properties. **ZERO new sources read in full.** Everything surfaced either
duplicated already-read material (the Soundiness manifesto restates the
over-approximation point already taken verbatim from Semgrep and CodeQL; the Daikon SCP
2007 paper duplicates the manual) or corroborated existing findings at snippet level
without adding a mechanism:
- API-misuse detection on MUBench (2024): **precision 72.22% / recall 43.01% / F1
  53.91%** -- corroborates SpecFuzzer's 54.57% recall band for general behavioural
  properties (B1.2).
- RRFinder mining resource-releasing specifications: **recall 94.0% / precision 86.6%**
  -- the informative contrast: recall is HIGH when the property class is narrow and
  pre-named, LOW when it is open-ended. That is the same shape as pyfinagent's own
  1-of-4 and reinforces B3.2's conclusion rather than adding to it.

**Round 8** probed (c) pre-registration / benchmark-selection bias in empirical SE and
(d) span-overlap false attribution in coverage attribution. **ZERO new sources read in
full.** (c) surfaced "Benchmarking as Empirical Standard in SE Research"
(arXiv:2105.00272), which restates NIST's trade-off without adding a mechanism; (d)
drifted off-domain entirely (source-monitoring psychology, marketing attribution). One
apt but off-domain snippet was noted and NOT promoted to a finding: an NLP redaction
paper states "Overlap-based and full-coverage scoring are identical when all matched
redactions fully cover their gold spans, meaning the reported recall does not benefit
from boundary leniency" -- an independent statement of the same over-crediting check for
span-overlap scoring, already established from NIST discrimination (B1.3).

**Rounds 7 and 8 both dry -> `dry_rounds = 2 >= K_required = 2` -> `coverage.dry = true`.**

---

# PART A2 -- INTERNAL, CORRECTED AND EXTENDED AGAINST THE 86.85 ARTIFACTS

Read after PART A; it CORRECTS one inference there and adds measured facts.

## A2.1 The three FAILs are corroborated by the ledger, not just by prose

`handoff/verdict_ledger.jsonl` (46 rows total, read 2026-08-16) carries three rows for
step 86.85, all `FAIL`, all dated 2026-08-15:

| cycle | verdict | note (truncated as stored) |
|---|---|---|
| 1 | FAIL | "C8 palindromic fixture (QA-M1 reversal SURVIVED..." |
| 2 | FAIL | "QA-M6 (fail-loud I/O guard) and QA-M4 (step_id ..." |
| 3 | FAIL | "QA-M2: the _dedup_key CYCLE-FALLBACK branch sur..." |

Cycle 4 (the C8 cycle that shipped the derivation gate) closed **CONDITIONAL with the
step ESCALATED** (commit `9a18150f`). So 86.89 inherits an escalated step, and under
CLAUDE.md F1's 3rd-CONDITIONAL rule the successor work is under pressure to be right
first time.

## A2.2 CORRECTION to PART A: the five cells cover **ZERO** guards, and no guard has more than one cell

`evaluator_critique_86.85.md:462` (cycle-4 Q/A, measured by inverting
`verify_matrix_coverage_86_85.coverage()` per cell):

> "experiment_results and live_check C8.5 both state that M5/M6/M9/M11/M12 leave the
> gate GREEN because they are 'coverage-redundant -- another cell touches the same
> guard'; **measured per-cell, those five cells cover ZERO enumerated guards and NO
> guard anywhere is covered by more than one cell.** The true reason is the second half
> of the same sentence -- their targets are invisible to the enumeration rule -- and the
> two readings differ materially: 'redundant' says the gate is still complete,
> 'invisible' says it is structurally blind."

Two facts PART A did not have, and both are design-deciding:

1. **5 of 14 cells (36%) contribute nothing to the coverage gate.** The gate reports
   "15 guards / 15 covered / 0 uncovered" while a third of the matrix is invisible to
   it. The headline is arithmetically true and rhetorically false.
2. **The guard->cell mapping is at most 1:1.** There is no redundancy anywhere, so the
   gate has **zero margin**: deleting any one of the 9 contributing cells turns exactly
   one guard RED, and deleting any of the other 5 changes nothing. A gate with no
   redundancy and a 36% inert population is brittle in both directions.

The distinction "redundant" vs "invisible" is the whole of step 86.89, stated by the
evaluator in one sentence.

## A2.3 The project already owns the pre-registration rule -- and 86.85 did not meet it

`evaluator_critique_86.85.md:476` cites the binding doctrine verbatim:

> "[WARN] qa.md 4b -- a COMPLETENESS claim must be executed against a known-member set
> **the author did not choose** and must find ALL of them. 'This cycle does not write
> another hand-list. Completeness is now DERIVED' is not supported at 1-of-4 recall."

And the recall procedure, reproducible, at `:475`:

> "drop M6 (ordering, cycle-1 QA-M1) -> rc=0 GREEN; drop M8 (fail-loud I/O, cycle-2
> QA-M6) -> rc=1 RED; drop M9 (step_id-in-key, cycle-2 QA-M4) -> rc=0 GREEN; drop
> M11+M12 (cycle-fallback, cycle-3 QA-M2) -> rc=0 GREEN. Recall = 1 of 4."

This is a **drop-one-cell sweep** -- a specification-side mutation (B5), executed by the
evaluator, against a known-member set drawn from the three prior FAILs. Note the honest
caveat the Q/A itself flagged: the set used was "the checker's OWN named member set",
i.e. author-chosen, which qa.md 4b explicitly disallows for a completeness claim. **For
86.89 the known set must come from somewhere the author does not control** -- the three
FAIL rows in `verdict_ledger.jsonl` and the QA-M* mutants in
`evaluator_critique_86.85.md` are exactly such a source, because they were written by
the evaluator before this step existed.

## A2.4 Equivalent-mutant retraction, already observed here

`evaluator_critique_86.85.md:462`: "QA-A/B/C/D KILLED; **QA-E and QA-F survived but both
fail the behavioural-differential test as defects and are retracted**." That is a
manual equivalent-mutant determination -- 2 of 6 (33%) of an independent evaluator's
cells were equivalent/unproductive. It sits inside the 21%-duplicated / 7%-equivalent
band TCE measured (B3.3) and above Google's post-curation productive rate, and it means
**any 86.89 mutation-count headline must state how equivalents were adjudicated.**

---

# PART C -- SYNTHESIS

## C1. Recency scan (last 2 years, 2024-2026) -- PERFORMED

**Result: 6 findings in the 2024-2026 window, and they COMPLEMENT rather than supersede
the canonical sources.**

| Year | Source | What it adds |
|---|---|---|
| 2026 | SPECA, arXiv:2604.26495 | Names "deduplication keys computed from complete inputs" and "temporal validity" as first-class obligation classes; states bottom-up code analysis "lack[s] the representational vocabulary" |
| 2026 | arXiv:2604.10761 | SpecFuzzer precision 67.83->74.17%, **recall 54.57% UNCHANGED** by LLM counterexample filtering |
| 2026 | arXiv:2602.00715 | Read; does NOT disaggregate by construct class (negative result, recorded) |
| 2025 | StaAgent, arXiv:2507.15892 | Type-1 (over-specific) vs Type-2 (inadequate) rule diagnosis; 64 problematic rules across 5 analyzers |
| 2024 | SAST Android, arXiv:2410.20740 | Per-tool recall 30.8%-94.9%; "79% (27/34) of unsupported types" are logical not syntactic |
| 2024 | Total Recall, ISSTA '24 | Recall bounds theorem against an unattainable ground truth; micro-benchmarks "lack insights into recall for real-world programs" |
| 2024 | MUBench API-misuse (snippet) | precision 72.22% / recall 43.01% |

**Nothing in the window supersedes the canonical sources.** Daikon (2007/manual; v5.8.23
still released 2025-06-04), Chockler-Kupferman-Vardi (2003), Kupferman (2006),
Schuler-Zeller (2013), Papadakis TCE (2015) and Inozemtseva-Holmes (2014) all remain
the primary references, and the 2024-2026 work explicitly builds on them. The one place
the new work is decisive is the **numbers**: 2026 supplies the measured fact that
filtering does not move recall, which older work only implied.

## C2. Key findings (each cited)

1. **Recall in property inference is bounded by the candidate grammar, and filtering
   never raises it.** SpecFuzzer + LLM counterexamples: precision 67.83% -> 74.17%,
   recall 54.57% -> 54.57%, 1,877 of 18,605 assertions invalidated (arXiv:2604.10761,
   2026, https://arxiv.org/html/2604.10761). *A validator bolted onto the 86.85
   enumeration will not move 1-of-4.*

2. **The pyfinagent blind spot has a formal name: this is a missing VACUITY check, not a
   coverage bug.** "In vacuity, mutations are in the specifications, whereas in
   coverage, mutations are in the system" (Kupferman, CONCUR 2006,
   https://www.cs.huji.ac.il/~ornak/publications/concur06b.pdf). Both pyfinagent
   artefacts mutate the system; nothing mutates the specification.

3. **Vacuous passes are common and always meaningful.** "typically 20% of specifications
   pass vacuously during the first formal-verification runs ... and vacuous passes
   always point to a real problem" (ibid., citing Beer et al.). *The one-off M14
   drop-probe should be a standing, re-runnable check.*

4. **No single coverage measure can be absolute.** Chockler, Kupferman & Vardi, CHARME
   2003 (https://www.cs.huji.ac.il/~ornak/publications/charme03a.pdf) -- and
   "stronger forms of coverage do not provide greater insight into the effectiveness of
   the suite" over 31,000 suites / 5 systems / 724 KLOC (Inozemtseva & Holmes, ICSE
   2014, https://www.cs.ubc.ca/~rtholmes/papers/icse_2014_inozemtseva.pdf). *Enriching
   the AST rule is the tempting fix and the literature says it under-delivers.*

5. **"Covered" that means only "adjacent" is a named, quantified failure.** Checked
   coverage is "the dynamic slice of covered statements that actually influence an
   oracle"; "24% of the executed statements do not influence any test oracle"; and "it
   is entirely possible for a test suite with zero test oracles to achieve 100% code
   coverage" (Schuler & Zeller, STVR 2013,
   https://www.st.cs.uni-saarland.de/publications/files/schuler-stvr-2013.pdf;
   survey arXiv:2212.06118, https://ar5iv.labs.arxiv.org/html/2212.06118). *Rule (b)
   TEXTUAL at `verify_matrix_coverage_86_85.py:253-257` is adjacency, the weakest tier.*

6. **The metric that detects over-crediting is DISCRIMINATION, not precision or
   recall.** "a tool reporting all sites would still achieve a precision of 50 % ... The
   problem can be mitigated by introducing the discrimination rate, which accounts for
   true-positives only if the tool did not incorrectly report the same defect at the
   safe site" (NIST/SATE V,
   https://www.nist.gov/system/files/documents/2021/03/24/Evaluating_Bug_Finders_COUFLESS_2015.pdf).
   *The `ast.Try` bug was a discrimination failure; recall stayed 100%.*

7. **Recall against a hand-picked known set is NOT recall against the population, and
   the gap is formally bounded, not hand-wavy.** `RecallSD * RecallDG <= RecallSG <=
   1 - (1 - RecallSD) * RecallDG`, with "establishing a direct lower bound for recall
   [being] not possible" without `RecallDG` (Helm et al., ISSTA '24,
   https://www.opal-project.de/articles/TotalRecall@ISSTA24.pdf).

8. **A rule that misses a guard AND every semantic restatement of it is Type-2
   INADEQUATE, not Type-1 over-specific -- and Type 2 is not fixable by broadening.**
   (StaAgent, arXiv:2507.15892, https://arxiv.org/html/2507.15892.) Corroborated by the
   SAST study: "79% (27/34) of unsupported types were challenging to detect without a
   deep understanding of the application's scenario logic"
   (https://arxiv.org/html/2410.20740).

9. **[ADVERSARIAL] Hand-declared rules are not automatically inferior -- at Google they
   were decisive.** 85% of mutants initially unproductive; hand-written, admittedly
   unsound AST-matching suppression heuristics took productive mutants from **15% to
   89%** over six years (arXiv:2102.11378,
   https://ar5iv.labs.arxiv.org/html/2102.11378). *Reconciliation: Google declares what
   to EXCLUDE (fail-safe filter over a derived population); 86.85's failed hand-lists
   declared what to INCLUDE (fail-open denominator). The defensible rule is not "never
   declare" but "never let a declaration be the denominator."*

10. **Discovery of a behavioural obligation is human/specification work; verification of
    it is machine work.** Three independent literatures agree: MRs "must be identified
    by humans based on problem domain knowledge" (TSE 2017,
    https://javiertroyauma.github.io/publications/TSE2017_REST_prePrint.pdf); PBT
    verifies a declared property (https://hypothesis.works/articles/what-is-property-based-testing/);
    mining is capped by its template grammar (Daikon manual §5.5,
    https://plse.cs.washington.edu/daikon/download/doc/daikon.html). The 2026 framework
    that closes it does so **top-down from a specification**, not bottom-up from code
    (SPECA §1/§3.3, https://arxiv.org/html/2604.26495v1).

11. **A declared register is complete only over its declarers -- by design, not by
    accident.** Pact: "only parts of the communication that are actually used by the
    consumer(s) get tested", so "any provider behaviour not used by current consumers is
    free to change without breaking tests" (https://docs.pact.io/). The verification leg
    that makes a register more than an assertion is **replay against the real
    implementation**: Pact checks "that all the calls to your test doubles return the
    same results as a call to the real application would."

12. **Mutation counts have an uncertain denominator.** TCE: >7% equivalent and 21%
    duplicated mutants on large real programs, with the best automatic detector finding
    only ~30% of equivalents (ICSE 2015,
    https://discovery.ucl.ac.uk/1499169/1/Jia_Trivial_Compiler_mutation-testing-papadakis-icse15.pdf);
    the 86.85 cycle-4 Q/A itself retracted 2 of its 6 cells as equivalent
    (`evaluator_critique_86.85.md:462`).

## C3. Consensus vs debate

**Consensus (uncontested across all 22 sources):**
- Syntactic/AST matching cannot express value-and-relation properties; a different
  engine tier is required (CodeQL, Semgrep, SAST study, SPECA).
- Every practical analysis is unsound; "expect both false positives and false negatives"
  (Semgrep) and no measure is absolute (CHARME 2003).
- Executed/adjacent != verified (Schuler & Zeller; oracle-metrics survey; Stryker's two
  denominators).
- Ground truth for real programs is unobtainable; recall must be reported against an
  explicitly-named approximation (Total Recall; NIST).

**Genuine debate, and it bears directly on the contract:**
- **Derive vs declare.** 86.85's shipped doctrine ("No cell declares what it covers")
  vs Google's measured 15%->89% from hand-curated, unsound heuristics. *Resolved above
  by direction-of-curation (exclude vs include); the contract should state which
  direction it is curating in.*
- **Bottom-up vs top-down derivation.** Daikon/SpecFuzzer mine bottom-up from executions
  and cap out near 55% recall; SPECA derives top-down from a specification and claims
  the vocabulary that bottom-up lacks. *Unresolved in the literature; pyfinagent has a
  latent spec (the cell `description` strings) that makes top-down cheap to try.*
- **Is high recall even the right target?** Inozemtseva & Holmes say coverage "should
  not be used as a quality target"; NIST says a tool may be legitimately chosen for
  recall OR precision depending on objective. *A gate demanding 100% guard coverage may
  be optimising the wrong quantity; the checker's own licence sentence
  (`:350-352`) is the right shape.*

## C4. Pitfalls (from the literature, each already half-realised in this codebase)

1. **Widening the anchor to fix a false gap re-creates the `ast.Try` bug in a new
   coordinate.** Over-crediting is the dangerous direction (checker's own note at
   `:198-200`); the literature's countermeasure is discrimination + paired probes, not
   a wider span.
2. **Adding `ast.Return`/`ast.Dict`/`ast.Compare` to the rule** buys Type-1 robustness
   and does nothing for Type-2 inadequacy (StaAgent), while "stronger forms of coverage
   do not provide greater insight" (Inozemtseva).
3. **A self-control drawn from the recognised class proves wiring, not recall.** Already
   true of `SYNTHETIC` at `:262-269`. StaAgent's oracle needs semantically-equivalent
   VARIANTS, not one instance.
4. **A known-member set the author chose cannot support a completeness claim** -- the
   project's own qa.md 4b, and the exact WARN on 86.85 (`:476`). Micro-benchmarks "lack
   insights into recall for real-world programs" (ISSTA '24).
5. **Reporting a bare "recall = 25%"** over-claims: it is `Recall_SD`, and Thm 3.5 says
   it has no direct relationship to `Recall_SG` without `Recall_DG`.
6. **A mutation headline without equivalent-mutant adjudication** is uncertain by
   7-21% (TCE) -- and 33% on the one independent sample this project has.
7. **Padding the gate with cells that cover nothing.** 5 of 14 already do
   (`evaluator_critique_86.85.md:462`); adding behavioural cells without extending the
   coverage rule would raise that fraction, not lower it.
8. **Never re-derive a claim from prose.** The "coverage-redundant" explanation in
   `experiment_results` / `live_check` was refuted by per-cell measurement -- the
   difference between "redundant" (complete) and "invisible" (blind) is the step.

## C5. Application to pyfinagent (external finding -> internal anchor)

| External finding | Internal anchor | Implication for 86.89 |
|---|---|---|
| Vacuity = mutate the SPECIFICATION (CONCUR 2006) | nothing mutates `CELLS`; the M14 drop-probe was one-off (`verify_matrix_coverage_86_85.py:198-199`) | The primary deliverable is a **standing drop-one-cell sweep**, not a richer AST rule |
| 20% pass vacuously; vacuous passes always point to a real problem | `:198-200` (the one time it was run, it found a real bug) | Make it re-runnable and part of the gate's own exit code |
| Discrimination, not recall, detects "reports all sites" (NIST) | rule (b) `spans_with_ancestors` `:178-205` | Score the gate on **paired** probes: cell-that-must-be-demanded vs cell-that-must-not |
| Recall bounds theorem (ISSTA '24 Thm 3.5) | the "1 of 4" at `:4-12` | Report as `Recall_SD = 1/4` **plus** the provenance and unmeasured completeness of D |
| Known set the author did not choose (qa.md 4b; ISSTA '24) | 3 FAIL rows in `handoff/verdict_ledger.jsonl`; QA-M* in `evaluator_critique_86.85.md` | Use the **evaluator-authored** mutants as D; they predate this step |
| Type 2 inadequate, not fixable by broadening (StaAgent) | `_is_guard_raise` `:117-124`, `_direct_refusal` `:127-155` | Do NOT try to reach ordering/key-composition by adding AST node types |
| Obligation discovery is human; verification is machine (TSE 2017, PBT, Daikon grammar) | cell `description` strings, `mutation_matrix_86_85.py:32-153` | Promote descriptions from OUTPUT to INPUT: a declared obligation register, machine-verified |
| A register is complete only over its declarers (Pact) | the same descriptions | State the register's scope limit explicitly; do not call it complete |
| Declare to EXCLUDE, derive to INCLUDE (Google) | doctrine at `:23-25` | Soften "no cell declares what it covers" to "no declaration is the denominator" |
| Checked coverage = influence, not adjacency (STVR 2013) | rule (a) `:246-248` vs rule (b) `:253-257` | Prefer a **behavioural differential** (run the mutant, compare output) over span overlap |
| 5 of 14 cells cover zero guards; no guard has >1 cell | `evaluator_critique_86.85.md:462` | Any new cell must be shown to move the coverage set, or it is inert |
| Sibling surface ungated | `verdict_history_86_21.py:85-106`, `:164`, `:201-277` (9 of 10 printed-output mutants survived) | Scope decision for Main: 86.89 may need to name whether the sibling is in scope |

**Cheapest high-value design, if it helps PLAN (Main owns the decision):** the behavioural
differential already exists in the codebase -- `mutation_matrix_86_85.py` RUNS each
mutant and observes `--self-test` go RED (`:201-211`). That is checked-coverage-grade
evidence (execution + propagation + detection). What is missing is only the
**obligation register** to score it against, and the register's members already exist as
prose in each cell's `description`. The gap is that nothing (a) requires an obligation
to have a cell, or (b) tests the register itself by removing a member. Findings 2, 3 and
10 all point at the same place.

## C6. Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL -- **22** (>=5 floor cleared 4.4x)
- [x] 10+ unique URLs total -- **45** collected (22 read in full + 23 snippet-only)
- [x] Recency scan (2024-2026) performed + reported -- §C1, 6 in-window findings
- [x] Full papers / pages read, not abstracts -- HTML/ar5iv where available; PDFs
      extracted with `pypdf` and quoted from the extracted text (NOT from a WebFetch PDF
      summary -- this project has measured fabricated quotes from that path twice)
- [x] file:line anchors for every internal claim -- PART A, A2

Soft checks:
- [x] Internal exploration covered every module in the caller's scope -- 4 scripts +
      the 86.85 critique + the live ledger
- [x] Contradictions / consensus noted -- §C3, incl. one ADVERSARIAL source and one
      correction of my own PART A inference (§A2.2)
- [x] Per-claim citation -- every quote carries URL + access date 2026-08-16
- [x] Audit-class loop-until-dry -- 8 rounds, rounds 7-8 dry, `coverage.dry = true`

## C7. Sources -- read in full (22; counts toward the gate)

| # | URL | Kind | Fetched how |
|---|---|---|---|
| 1 | https://codeql.github.com/docs/writing-codeql-queries/about-data-flow-analysis/ | official doc | WebFetch |
| 2 | https://plse.cs.washington.edu/daikon/download/doc/daikon.html | official doc | WebFetch |
| 3 | https://stryker-mutator.io/docs/mutation-testing-elements/mutant-states-and-metrics/ | official doc | WebFetch |
| 4 | https://docs.semgrep.dev/writing-rules/data-flow/data-flow-overview | official doc | WebFetch |
| 5 | https://martinfowler.com/articles/consumerDrivenContracts.html | authoritative blog | WebFetch |
| 6 | https://hypothesis.works/articles/what-is-property-based-testing/ | authoritative blog | WebFetch |
| 7 | https://arxiv.org/html/2604.10761 | preprint 2026 | WebFetch (arXiv HTML) |
| 8 | https://arxiv.org/html/2602.00715 | preprint 2026 | WebFetch (arXiv HTML) |
| 9 | https://docs.pact.io/ | official doc | WebFetch |
| 10 | https://www.opal-project.de/articles/TotalRecall@ISSTA24.pdf | **peer-reviewed ISSTA '24** | curl + pypdf |
| 11 | https://www.nist.gov/system/files/documents/2021/03/24/Evaluating_Bug_Finders_COUFLESS_2015.pdf | **NIST** | curl + pypdf |
| 12 | https://ar5iv.labs.arxiv.org/html/2212.06118 | survey preprint | WebFetch (ar5iv) |
| 13 | https://www.st.cs.uni-saarland.de/publications/files/schuler-stvr-2013.pdf | **peer-reviewed STVR 2013** | curl + pypdf |
| 14 | https://www.cs.ubc.ca/~rtholmes/papers/icse_2014_inozemtseva.pdf | **peer-reviewed ICSE 2014** | curl + pypdf |
| 15 | https://arxiv.org/html/2410.20740 | preprint 2024 | WebFetch (arXiv HTML) |
| 16 | https://ar5iv.labs.arxiv.org/html/2102.11378 | preprint (Google) **[ADVERSARIAL]** | WebFetch (ar5iv) |
| 17 | https://arxiv.org/html/2604.26495v1 | preprint 2026 | WebFetch (arXiv HTML) |
| 18 | https://discovery.ucl.ac.uk/1499169/1/Jia_Trivial_Compiler_mutation-testing-papadakis-icse15.pdf | **peer-reviewed ICSE 2015** | curl + pypdf |
| 19 | https://javiertroyauma.github.io/publications/TSE2017_REST_prePrint.pdf | **peer-reviewed IEEE TSE 2017** | WebFetch |
| 20 | https://www.cs.huji.ac.il/~ornak/publications/concur06b.pdf | **peer-reviewed CONCUR 2006** | curl + pypdf |
| 21 | https://www.cs.huji.ac.il/~ornak/publications/charme03a.pdf | **peer-reviewed CHARME 2003** | curl + pypdf |
| 22 | https://arxiv.org/html/2507.15892 | preprint 2025 | WebFetch (arXiv HTML) |

## C8. Identified but snippet-only (23; does NOT count toward the gate)

| URL | Why not fetched in full |
|---|---|
| https://homes.cs.washington.edu/~mernst/pubs/daikon-tool-scp2007.pdf | duplicates the Daikon manual (#2) |
| https://www.sciencedirect.com/science/article/pii/S016764230700161X | paywalled duplicate of the above |
| https://github.com/codespecs/daikon | tool repo; used only for the v5.8.23 / 2025-06-04 recency datum |
| https://yanniss.github.io/Soundiness-CACM.pdf | restates over-approximation already quoted from #1/#4 |
| http://soundiness.org/ | as above |
| https://arxiv.org/pdf/2401.08807 (SpecGen) | superseded for this question by #7 |
| https://arxiv.org/pdf/2606.21339 (KBSpec) | LLM spec generation, not enumeration recall |
| https://arxiv.org/abs/2606.24004 (Spec Learning) | prompt-spec alignment, off-question |
| https://arxiv.org/pdf/2509.09917 | program slicing for spec generation; adjacent |
| https://arxiv.org/pdf/2604.00280 (VeriAct) | formal spec synthesis; adjacent |
| https://dl.acm.org/doi/10.1145/3643991.3644904 (MUBench 2024) | recall 43.01% datum taken at snippet level |
| https://link.springer.com/article/10.1007/s10515-021-00294-x | API-misuse pattern mining; corroborative |
| https://arxiv.org/pdf/2402.14366 (AnnaTester) | analyzer-fault detection; #22 covers the method |
| https://arxiv.org/pdf/1812.05033 | differential testing of analyzers; adjacent |
| https://arxiv.org/pdf/2105.00272 | benchmarking standards; restates NIST trade-off |
| https://arxiv.org/pdf/1805.11683 (DeepBugs) | seeded-bug recall; older, corroborative |
| https://dl.acm.org/doi/10.1145/3324884.3416667 | adequacy vs test-set size; corroborative |
| https://stryker-mutator.io/docs/ | index page; #3 is the substantive page |
| https://about.codecov.io/blog/mutation-testing-how-to-ensure-code-coverage-isnt-a-vanity-metric/ | vendor blog, community tier |
| https://en.wikipedia.org/wiki/Metamorphic_testing | community tier; #19 is authoritative |
| https://www.faa.gov/sites/faa.gov/files/2022-02/VVSPT-E5-GDE-017_VRTM_V3.0.pdf | traceability-matrix template, process not technique |
| https://www.perforce.com/resources/alm/requirements-traceability-matrix | vendor page, community tier |
| https://link.springer.com/chapter/10.1007/978-3-540-79124-9_2 (Vacuity in Testing) | paywalled; #20 covers the duality |



