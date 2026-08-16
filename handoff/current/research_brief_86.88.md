# Research Brief -- phase-86.88

**Topic:** Guarding a seam versus guarding the ROUTES INTO it -- why a caller-side
pre-mangle defeats a value-level guard, and how to build coverage that sees it.

**Tier:** moderate (caller-specified). **Audit-class: YES** (loop-until-dry, K=2).
**Started:** 2026-08-16.

---

## ENVELOPE (born inert -- phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 15,
  "snippet_only_sources": 15,
  "urls_collected": 30,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "coverage": {
    "audit_class": true,
    "rounds": 9,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "summary": "The 86.86 fix guards the PRODUCER; four caller routes (autonomous_loop.py:3177/:3182/:3411/:3416) replace risk_dict with dict(_LITE_RISK_DEFAULT) before the producer runs, so the three-state seam resolves SIZE(3.0) instead of ABSENT -- same number, destroyed provenance. Every guard (suite, AST checker, 6-cell matrix) anchors at or below _build_lite_risk_assessment; none executes a route. CORRECTION: the checker's <whole-dict> branch is NOT dead -- measured, it fires on `x or _LITE_RISK_DEFAULT` and is blind only to the dict(...) Call shape, so the remedy is widening node shapes, not resurrection. Literature: this is Delamaro interface mutation's second operator group (mutate invocations); PIT's call operators are off-by-default, but PyTation (ICSE'26) mutates call arguments with 69% non-subsumption. Coupled-branch coverage (Cling) is the right unit; CERT OBJ06-J and parse-don't-validate give copy-then-validate; muSE makes non-detection a tool flaw; Google/Fowler say stub the LLM transport, assert state.",
  "brief_path": "handoff/current/research_brief_86.88.md",
  "gate_passed": true
}
```

*(Born INCOMPLETE at t0 and flipped to COMPLETE as the final act, per phase-86.37.)*

---

## Status log (write-first, appended as work lands)

- [t0] Brief created; envelope born INCOMPLETE. Read `.claude/agents/researcher.md`
  and `.claude/rules/research-gate.md` in full.
- [t1] Internal round 1 done: the mechanism is MEASURED, not inferred (below).
- [t2] External round 1: 1 source read in full.

---

## Internal code inventory (round 1 -- MEASURED)

Probe re-runnable; output pasted verbatim below the table.

| File | Anchor | Role | Status |
|---|---|---|---|
| `backend/services/autonomous_loop.py` | `:2304-2310` | `_LITE_RISK_DEFAULT` dict literal (pct 3.0) | live |
| same | `:2313-2362` | `_lite_position_pct` -- THE guarded seam; delegates to `_resolve_position_pct`; SIZE->judge number incl. 0.0, UNPARSEABLE->0.0 fail-closed, ABSENT->3.0 | live, correct **at the seam** |
| same | `:2365-2404` | `_build_lite_risk_assessment` -- the ONE producer; `:2401` calls the seam | live |
| same | `:3177`, `:3182` | Claude lite: `risk_dict = dict(_LITE_RISK_DEFAULT)` at no-JSON + `except` | **THE PRE-MANGLE** |
| same | `:3411`, `:3416` | Gemini lite: identical two routes | **THE PRE-MANGLE** |
| same | `:3186`, `:3422` | the only 2 calls into the producer -- both downstream of a pre-mangle route | live |
| same | `:2461` | `dict(_LITE_RISK_DEFAULT["risk_limits"])` in `_data_integrity_blocked_analysis` -- key-scoped, NOT whole-dict; already sets pct 0.0 explicitly `:2459` | live, benign |
| `scripts/qa/verify_lite_risk_seam_86_86.py` | `:53-67` | `or_default_sites` -- `ast.BoolOp`-operand walk | live |
| same | `:65-66` | the `elif isinstance(operand, ast.Name)` -> `<whole-dict>` branch | **UNEXERCISED on this tree (0 matches), but NOT structurally dead** -- see correction below |
| same | `:236-247` | prints `RESULT: OK`, 8 PASS / 0 FAIL, with the pre-mangle live | live |
| `scripts/qa/mutation_matrix_86_86.py` | (221 ll) | mutation cells; all string-substitution against the PRODUCER | pending round 2 |
| `backend/tests/test_phase_66_2_risk_judge_shape.py` | `:645`, `:658-663`, `:744-758`, `:827` | shape tests + a source-text assertion `:827` | pending round 2 |
| `backend/services/portfolio_manager.py` | `_resolve_position_pct` / `PositionVerdict` / `decide_trades` | the CONSUMER 86.74 guarded | pending round 2 |

**Measured probe output (2026-08-16):**

```
BoolOp operands that are a BARE _LITE_RISK_DEFAULT Name (the <whole-dict> branch): []
  line 3177: dict(_LITE_RISK_DEFAULT) -> node=Call, inside BoolOp? NO
  line 3182: dict(_LITE_RISK_DEFAULT) -> node=Call, inside BoolOp? NO
  line 3411: dict(_LITE_RISK_DEFAULT) -> node=Call, inside BoolOp? NO
  line 3416: dict(_LITE_RISK_DEFAULT) -> node=Call, inside BoolOp? NO
=== callers of _build_lite_risk_assessment ===
  call at line 3186
  call at line 3422
```

and `python3 scripts/qa/verify_lite_risk_seam_86_86.py` -> **`RESULT: OK`, 8 PASS / 0 FAIL**,
enumerating exactly 4 `or` sites (`reasoning` `:2392`, `decision` `:2394`, `risk_level`
`:2402`, `risk_limits` `:2403`) and **none of the four `dict(_LITE_RISK_DEFAULT)` routes.**

### The mechanism, stated precisely

`_lite_position_pct` (`:2347`) resolves a THREE-state verdict from `risk_dict`:
`SIZE` / `UNPARSEABLE` / `ABSENT`. Its correctness depends on `risk_dict` being the
**judge's own output**. At `:3177/:3182/:3411/:3416` the caller replaces `risk_dict`
wholesale with `dict(_LITE_RISK_DEFAULT)` **before** the producer is called. That dict
already contains `recommended_position_pct: 3.0`, so the seam resolves **`SIZE(3.0)`** --
a fabricated judge opinion -- instead of `ABSENT`. The numeric outcome is the same 3.0,
so no test that asserts the NUMBER can see it; what is destroyed is the **provenance**:
"the judge failed / returned no JSON" and "the judge said 3.0" become indistinguishable
downstream, exactly the collapse 86.86 removed one layer deeper. The seam is guarded;
the four routes INTO it are not. This is the `feedback_guards_stop_one_seam_short` class.

**Corollary -- an unreachable branch inside the seam.** Because every failure route now
arrives PRESENT, the `ABSENT -> 3.0` branch at `:2362` and the `UNPARSEABLE -> 0.0`
fail-closed branch at `:2352-2361` are reachable ONLY from a judge that returned parseable
JSON omitting/corrupting the key. The pre-mangle routes cannot exercise either.

### CORRECTION -- the `<whole-dict>` branch is SHAPE-SPECIFIC, not "dead"

The step's framing (and my own first measurement) called
`verify_lite_risk_seam_86_86.py:65-66` a **dead branch / zero-assertion guard**. The
86.86 Q/A critique (`handoff/current/evaluator_critique_86.86.md`, NOTE-2) says that is
FALSE. **I reproduced it rather than accepting it**, by importing the shipped
`or_default_sites` and handing it both shapes:

```
bare-Name form  `x or _LITE_RISK_DEFAULT` -> [(4, '<whole-dict>')]
Call form `dict(_LITE_RISK_DEFAULT)`      -> []
```

So the branch **fires** on `x or _LITE_RISK_DEFAULT`, and when it fires the key
`<whole-dict>` lands outside `RETAINED_KEYS` and FAILS the checker at `:194-199`. It is
blind **only** to the `dict(...)` `Call` shape. The accurate statement is therefore
*"unexercised by the current tree and blind to the shipped route shape"*, **not** *"can
never fire"*. This matters for the contract: the remedy is **widening the node shapes the
rule accepts**, not resurrecting a corpse -- and a PLAN written on the "dead branch"
premise would aim at the wrong target.

Structural census (my own AST walk, matching the Q/A's independently): **12** `ast.Name`
references to `_LITE_RISK_DEFAULT` -- 1 `Assign` (the literal `:2304`), **7** `Subscript`,
**4** `Call` (the pre-mangles). And `risk_dict` has **six rebinds** (`:3175`, `:3177`,
`:3182`, `:3409`, `:3411`, `:3416` -- all `Call` values) and **zero subscript writes**, so
today the only way the argument is corrupted is *whole-object replacement*.

### What 86.86's Q/A already established (do not re-derive in PLAN)

`evaluator_critique_86.86.md` N1: the Q/A injected a caller-side pre-mangle into
`_run_claude_analysis` and it **SURVIVED** -- *"It reintroduces the exact D6 defect (0.0 ->
3.0 -> BUY $719.93) and is caught by NEITHER the suite NOR the AST checker -- no test
drives either `_run_*_analysis` lite path end to end"*. It names the fix 86.88 should
build: *"one end-to-end test that drives the lite risk-judge block with a stubbed LLM
returning `{"recommended_position_pct": 0}`, or a checker assertion that risk_dict is not
written between parse and producer call."* Related residuals of the same class, already
enumerated there: **E2** a bare-literal pre-mangle (`or 3.0`), **E5b**
`_LITE_RISK_DEFAULT.get(...)`. N2 confirms the four routes and that they persist **3.0,
identical to the ABSENT branch** -- so this is a **provenance** defect, not a wrong number.

### Mutation matrix: every cell targets the PRODUCER, none targets a ROUTE

`scripts/qa/mutation_matrix_86_86.py:50-94` -- all five cells D6-M1..M5 replace text
INSIDE `_lite_position_pct` / `_build_lite_risk_assessment`; the extra `SEAM-M1`
(`:167-179`) mutates the producer CALL at `:3421-3422` into a parallel dict literal.
**Zero cells mutate `dict(_LITE_RISK_DEFAULT)` at `:3177/:3182/:3411/:3416`.** The file
is explicitly honest about this -- `:19-23`: *"WHAT A MATRIX LICENSES. Only 'these N
mutations were killed'. It is NOT a claim that the guards are complete"* -- so the gap is
a scope gap, not a false claim. It is nonetheless the exact gap: the defect class lives at
the call site and the operator set never reaches a call site.

### Existing in-repo precedent for a documented unreachable branch

`backend/services/portfolio_manager.py:1054-1075` keeps an unreachable `UNRECOGNISED
state` branch in PRODUCTION and documents both why it exists (fail-closed defense in
depth) and what it does **not** promise (*"The guarantee is local to this function; the
ABSENT-with-an-explicit-pct case is held safe by the CALL SITE, not by construction
here"*). That is the same caller-vs-callee split 86.88 is about, already written down by
86.74 -- and it is the argument for treating a dead branch in a **checker** differently
from one in **production**: a fail-closed production branch costs nothing if it never
fires, whereas a detection branch that never fires advertises coverage the checker does
not have.

The three-state vocabulary is shared: `portfolio_manager.py:994-996` (`SIZE`/`ABSENT`/
`UNPARSEABLE`), `PositionVerdict.blocks_buy` `:1006-1013`, `_coerce_pct` `:1078-1090`
(`raw is not None`, not `if raw:`), `_resolve_position_pct` `:1093-1108`.

---

## External sources -- READ IN FULL (counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding / quote |
|---|---|---|---|---|---|
| 1 | https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/ | 2026-08-16 | authoritative blog (Alexis King) | WebFetch, full | *"the difference between validation and parsing lies almost entirely in how information is preserved"*; **shotgun parsing**: *"Late-discovered errors in an input stream will result in some portion of invalid input having been processed"*; *"Get your data into the most precise representation you need as quickly as you can. Ideally, this should happen at the boundary of your system"*. Directly names the failure: **"Downstream re-validation or mutation before validation are antipatterns because they undo the guarantees"**. |
| 2 | https://codeql.github.com/docs/writing-codeql-queries/about-data-flow-analysis/ | 2026-08-16 | official docs (GitHub) | WebFetch, full | *"the data flow graph does not reflect the syntactic structure of the program, but models the way data flows through the program at runtime"*. Local flow *"only considers edges between data flow nodes belonging to the same function and ignores data flow between functions and through object properties"*; global flow crosses functions but is *"more time and energy intensive"*. Libraries must handle *"aliasing between variables"* -- the thing a syntactic match cannot. |
| 3 | https://docs.semgrep.dev/writing-rules/data-flow/taint-mode | 2026-08-16 | official docs (Semgrep) | WebFetch, full | *"Tainted data flows from sources to sinks through propagators, such as assignments and function calls."* Hard limit: *"Taint propagators only work intraprocedurally, that is, within a function or method. You cannot use taint propagators to propagate taint across different functions/methods."* Interprocedural needs `--pro-intrafile` / `--pro`. **Field/index sensitivity and aliasing into containers are not addressed** -- so even taint mode is not a free win for a dict passed between functions. |
| 4 | https://martinfowler.com/articles/mocksArentStubs.html | 2026-08-16 | authoritative blog (Fowler) | WebFetch, full | Double taxonomy: a **stub** provides *"canned answers to calls made during the test"*; a **fake** has *"working implementations, but usually take some shortcut"*. The coupling warning that bears on sub-question 5: *"Mockist tests are thus more coupled to the implementation of a method. Changing the nature of calls to collaborators usually cause a mockist test to break"*, and the risk that *"writing the test makes you think about the implementation of the behavior"*. Argues for **state verification** (assert the downstream effect) over behaviour verification where possible. |
| 5 | https://arxiv.org/html/2406.09843v3 | 2026-08-16 | peer-reviewed (TOSEM 2025) | WebFetch (arXiv **HTML**, per the gate's html-first chain), full | LLM-vs-classical mutation study. Scope is explicitly **function-level**: *"we generate one mutation per line"* with *"three lines around the bug location"* as context. Equivalence is *"undecidable"*; equivalent-mutant rates 1.0% (PIT) to 10.6%. "Not Detected" (surviving) mutants run **26-50%** across approaches. Confirms mainstream mutation tooling is scoped to the function under test -- a call-site fault is out of the operator set by construction. |
| 6 | https://pitest.org/quickstart/mutators/ | 2026-08-16 | official docs (PIT) | WebFetch, full | The only call-acting operators are **Void Method Call Mutator** (*"removes method calls to void methods"*, default ON), **Non Void Method Call Mutator** and **Constructor Call Mutator** (*"replaces constructor calls with `null` values"*), both **OFF by default** and flagged *"fairly unstable"*. So even the industry-standard tool's call-site coverage is opt-in, coarse (delete the call), and cannot express "the caller substituted a different argument value". |
| 7 | https://docs.python.org/3/library/copy.html | 2026-08-16 | official docs (Python) | WebFetch, full | *"A shallow copy constructs a new compound object and then (to the extent possible) inserts references into it to the objects found in the original."* Load-bearing for the internal finding: `dict(_LITE_RISK_DEFAULT)` is a **shallow** copy, so the nested `risk_limits` dict is the module-level object itself, shared by every ticker on every failure route. |
| 8 | https://arxiv.org/html/2601.19088v2 | 2026-08-16 | peer-reviewed (ICSE '26, Apr 2026) | WebFetch (arXiv HTML), full | **PyTation** -- Python-specific mutation. This is the direct counter-example to "mutation can't reach call sites": its `RemFuncArg` operator works by *"removing one optional argument at a time from function calls"*, and the tool *"logs function calls (including names and argument values)"* dynamically to find candidates. *"Python allows removing required function arguments without triggering a compile-time error ... making it easier for subtle faults to go unnoticed."* Results: **69% of PyTation mutants are not subsumed by Cosmic Ray mutants**; cross-kill 3.52%; equivalent-mutant rate 1.61% vs Cosmic Ray's 5.7%. |
| 9 | https://abseil.io/resources/swe-book/html/ch13.html | 2026-08-16 | official docs (Software Engineering at Google, ch.13) | WebFetch, full | The stubbing hazard, named: *"Stubbing leaks implementation details of your code into your test"*, and **duplicated logic** -- *"With stubbing, there is no way to ensure the function being stubbed behaves like the real implementation."* Fidelity rule: *"A fake must maintain fidelity to the API contracts of the real implementation ... but only from the perspective of the test."* Preference order **real implementation -> fake -> stub/mock**, and *"The team that owns the real implementation should write and maintain a fake."* |
| 10 | https://cmu-sei.github.io/secure-coding-standards/sei-cert-oracle-coding-standard-for-java/rules/object-orientation-obj/obj06-j | 2026-08-16 | official standard (SEI CERT, OBJ06-J) | WebFetch, full | *"a time-of-check, time-of-use (TOCTOU) vulnerability may result when a field contains a value that passes validation and security checks but changes before use."* The ordering rule: copy FIRST, then *"any input validation must be performed on the copy rather than on the original object."* Also warns that for compound data you must *"manually duplicate each element rather than relying on shallow copies."* The generalised statement of "a guard is only as good as what reaches it". |
| 11 | https://ar5iv.labs.arxiv.org/html/2102.06829 | 2026-08-16 | peer-reviewed (ACM TOPS 24(3), 2021) | WebFetch of the **ar5iv body** (the `/abs/` page was fetched first and is recorded as snippet-only, per the gate's "abstract page is not a full read" rule) | **μSE -- mutation-based SOUNDNESS evaluation of static analysers.** Exactly the method 86.88's checker needs: *"The uncaught mutants indicate flaws in the tool, and analyzing them leads to the broader discovery and awareness of the unsound assumptions of the tools."* Four mutation **schemes** decide WHERE the operator is planted (reachability-, abstraction-, taint-, scope-based) -- placement is the variable, not the fault. **13 undocumented flaws in FlowDroid**, 12 in Argus, 25 unique. The soundiness verdict: *"soundy tools certainly seem to be a practical choice: but only if the unsound choices are known, necessary, and properly documented"*, and today unsound choices *"may not be documented, and unknown to non-experts"* and *"may not even be known to tool designers (i.e. implicit assumptions)."* |
| 12 | https://ar5iv.labs.arxiv.org/html/2001.04221 | 2026-08-16 | preprint (2020), read via **ar5iv** per the gate's fallback chain | WebFetch, full | **Cling / Coupled Branches Criterion.** *"generated tests with high code coverage could be ineffective, i.e., they may not detect all faults or kill all injected mutants"*; Cling found **25 integration faults** *"(i.e., faults due to wrong assumptions about the usage of the callee class) that remain undetected when using automatically generated random and unit-level test suites"*. A "coupled branch" pair = a branch in the CALLER leading to the call site + the branch in the CALLEE it triggers -- precisely the coverage unit the 86.86 artifacts lack. |
| 13 | https://taoxie.cs.illinois.edu/publications/icsm07.pdf | 2026-08-16 | peer-reviewed (ICSM 2007, Hou/Zhang/Xie/Mei/Sun) | `curl` + **pypdf** extraction (10 pages / 50,582 chars); quotes regex-verified against the extracted text, NOT summarised by a PDF reader | The **primary-source description of Delamaro's Interface Mutation**: *"Delamaro et al. [8] developed the interface mutation (IM) approach ... designs two groups of interface mutation operators ... The first group is applied inside an interface to mutate the source code of implementation of the interface, and the second group is designed to mutate the declaration and invocations of the interfaces."* And the evaluation detail: *"To evaluate the effectiveness of interface mutation (IM), Delamaro et al. [8] seeded faults in both components and their callers."* Its opening `abs`/`log` example is the canonical statement of the blind spot: *"Statement coverage and branch coverage are not good criteria for testing f, as any input for f could achieve one hundred percent statement coverage and branch coverage."* |
| 14 | https://lincolnloop.com/blog/avoiding-mocks-testing-llm-applications-with-langchain-in-django/ | 2026-08-16 | industry blog | WebFetch, full | The concrete stub-drift instance: when LangChain replaced `__call__` with `invoke()`, *"production code would fail, yet mocked tests would still pass."* *"Mocks tend to couple tests to specific implementations, making refactoring harder, and they can reduce the surface area of code execution, hiding potential errors."* Recommends a **fake backend that invokes the real client-library API** so the production path still executes. Honest gap noted: it gives **no** guidance on preventing a double from duplicating production parsing logic. |
| 15 | https://pyre-check.org/docs/pysa-basics/ | 2026-08-16 | official docs (Meta / Pyre) | WebFetch, full | Python-specific interprocedural taint. *"Pysa works by tracking flows of data from where they originate (sources) to where they terminate in a dangerous location (sinks)."* Container handling is coarse: *"when an object is tainted, that means that all attributes of that object are also tainted"*, with `ReturnPath` to narrow it. It **is** interprocedural, but *"Pysa will only analyze the code in the repo that it runs on"*. Establishes that a Python-native tool for the "value flows through a copied dict across a function boundary" question exists and is not exotic. |

### Snippet-only / attempted but NOT read in full (does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://dl.acm.org/doi/10.1109/32.910859 | Delamaro et al., IEEE TSE 27(3) 2001 -- the IM primary | paywalled; covered via source 13's primary-source description |
| https://www.researchgate.net/publication/2594416_Integration_Testing_Using_Interface_Mutations | paper | ResearchGate wall |
| https://link.springer.com/article/10.1186/s40411-016-0031-8 | JSERD 2016, static-analysis/mutation correlation | 303 to an `idp.springer.com` auth redirect |
| https://testing.googleblog.com/2015/01/testing-on-toilet-change-detector-tests.html | Google Testing Blog | page returned nav + comments only; article body absent from the fetched HTML |
| https://web.eecs.umich.edu/~weimerw/2022-481F/readings/mutation-testing.pdf | Jia & Harman survey | superseded for this question by sources 5/8/13 |
| https://conf.researchr.org/home/icst-2026/mutation-2026 | Mutation 2026 workshop | venue page, no proceedings yet |
| https://arxiv.org/pdf/1806.09761 | Android static-analysis mutation | superseded by source 11 (same line of work, journal version) |
| https://wiki.sei.cmu.edu/confluence/display/java/SEC02-J.+Do+not+base+security+checks+on+untrusted+sources | CERT rule | same doctrine as source 10 |
| https://wiki.sei.cmu.edu/confluence/spaces/java/pages/88487571/MET52-J....clone...untrusted+method+parameters | CERT rule | ditto |
| https://learn.adacore.com/courses/SPARK_for_the_MISRA_C_Developer/chapters/08_unreachable_and_dead_code.html | vendor course | snippet already gave the operative point |
| https://www.axivion.com/en/products/code-smells-detector/detecting-unreachable-code/ | vendor | marketing-tier |
| https://arxiv.org/pdf/2506.11076 | DCE-LLM dead-code elimination | tangential (compiler DCE, not checker dead branches) |
| https://github.com/SEatSFU/pytation | tool repo for source 8 | code, not literature |
| https://dl.acm.org/doi/10.1145/125489.125473 | coupling-effect (Offutt) | paywalled; effect described via source 5 |
| https://arxiv.org/abs/2102.06829 | arXiv **abstract page** for source 11 | abstract-only fetches do not count as read-in-full per the gate; the body was then read via ar5iv and IS counted (source 11) |

---

## Internal inventory, round 2 -- the test surface

| File | Anchor | Finding |
|---|---|---|
| `backend/tests/test_phase_66_2_risk_judge_shape.py` | `:666-671` | `_judge(pct=..., decision=...)` -- builds the **raw judge JSON**, not the produced dict. Good: it does NOT duplicate the production construction. |
| same | `:776-788` | `TestLiteZeroProducesNoOrderEndToEnd._orders` calls `_build_lite_risk_assessment(_judge(pct))` and feeds the result to the **real `decide_trades`**. Genuinely end-to-end **from the producer down** -- but it *enters the system downstream of the pre-mangle*. |
| same | `:818-829` | `test_lite_position_pct_is_the_only_route_to_the_default` asserts on `inspect.getsource(al._build_lite_risk_assessment)` -- a **source-text** assertion scoped to ONE function. Structurally cannot see `:3177/:3182/:3411/:3416`. |
| same | `:770-773` | Already discloses that `paper_risk_judge_shape_fix_enabled` has **zero production readers**. |

**Net:** no test in the suite executes `_run_claude_analysis` or `_run_gemini_analysis`, so **no test executes any of the four pre-mangle routes**. Every existing guard -- suite, AST checker, mutation matrix -- is anchored at or below `_build_lite_risk_assessment`.

---

## Recency scan (last 2 years, 2024-2026) -- MANDATORY SECTION

Search-query discipline (all three variants run, per `.claude/rules/research-gate.md`):
**current-year** -- "call-site mutation testing 2026 interprocedural mutation operators caller
argument"; "unreachable guard clause ... code review 2026". **Last-2-year** -- "mutation
testing survey 2025 integration level mutants coupling effect"; "integration mutation
testing coupled branches caller callee 2024 2025". **Year-less canonical** -- "interface
mutation testing integration mutation Delamaro call site mutation operators"; "'parse
don't validate' boundary validation mutable dict defensive copy"; "detecting unreachable
code dead branch static analysis"; "validate at the producer versus validate the routes".

**Result: TWO new findings in the window that materially change the plan, plus one
non-finding.**

1. **PyTation (ICSE '26, Apr 2026 -- source 8) supersedes the assumption that mutation
   cannot reach a call site in Python.** Its `RemFuncArg` operator mutates *"function
   calls"* directly, and **69% of its mutants are not subsumed by Cosmic Ray's** -- i.e.
   the classical Python operator set demonstrably misses this fault family. This is the
   strongest single argument that a call-site mutation cell is a recognised technique and
   not an invention of this step.
2. **The LLM-mutation study (TOSEM 2025 -- source 5) confirms the *scope* limit is still
   live in 2025 tooling**: generation is *"one mutation per line"* around a target
   location, and 26-50% of mutants survive. Nothing in the 2024-2026 window relaxes the
   function-scoped default.
3. **Non-finding:** no 2024-2026 work supersedes Delamaro's Interface Mutation (2001) as
   the conceptual frame for caller-side mutation. Mutation 2026 (ICST workshop) has no
   published proceedings yet. The canonical source remains canonical.

---

## Key findings

1. **The defect class has a name and a 25-year-old literature: it is an *interface* fault,
   not a unit fault.** Delamaro's IM designs *"two groups of interface mutation operators
   ... the second group is designed to mutate the declaration and invocations of the
   interfaces"* (Hou et al. 2007, source 13). The 86.86 matrix implements only the
   first group.
2. **Callee-scoped coverage is provably blind to caller misuse.** The canonical
   illustration: *"Statement coverage and branch coverage are not good criteria for
   testing f, as any input for f could achieve one hundred percent statement coverage and
   branch coverage"* (source 13) -- 100% coverage of `f` while `f` passes 0 to a `log`
   that rejects 0. Empirically, Cling found **25 integration faults** *"that remain
   undetected when using automatically generated random and unit-level test suites"*
   (source 12).
3. **The right coverage unit is the COUPLED BRANCH -- caller branch + the callee branch it
   triggers** (source 12). `:3174`-`if risk_json_match:` / `else:` is a caller branch; the
   `SIZE`/`ABSENT` split in `_lite_position_pct` is the callee branch. Coverage that names
   only one of the pair cannot express the defect.
4. **A guard is only as strong as the value that reaches it -- and the fix ordering is
   settled doctrine.** CERT OBJ06-J: *"any input validation must be performed on the copy
   rather than on the original object"*, because *"a TOCTOU vulnerability may result when
   a field contains a value that passes validation ... but changes before use"* (source
   10). "Parse, don't validate" states the same rule constructively: *"Get your data into
   the most precise representation you need as quickly as you can. Ideally, this should
   happen at the boundary of your system"*, and names the failure mode -- **shotgun
   parsing** -- where *"some portion of invalid input"* is processed before the check
   (source 1).
5. **An `ast.BoolOp`-operand rule structurally cannot see `dict(CONST)`, and the fix is a
   different ANALYSIS, not a bigger pattern.** *"the data flow graph does not reflect the
   syntactic structure of the program"* (CodeQL, source 2); local flow *"ignores data flow
   between functions and through object properties"*, and the libraries must handle
   *"aliasing between variables"*. Semgrep's ceiling is explicit: *"Taint propagators only
   work intraprocedurally ... You cannot use taint propagators to propagate taint across
   different functions/methods"* (source 3) -- interprocedural needs Pro. Pysa (source 15)
   IS interprocedural but coarse on containers (*"when an object is tainted ... all
   attributes of that object are also tainted"*).
6. **A checker's undetected case is a TOOL FLAW, and mutation is the standard way to find
   it.** μSE: *"The uncaught mutants indicate flaws in the tool"* -- 25 unique flaws, all
   *"undocumented"* (source 11). Its four **mutation schemes** are all about WHERE the
   fault is planted; the 86.86 matrix has one placement (inside the producer). The
   soundiness rule applies directly: a soundy checker is acceptable *"only if the unsound
   choices are known, necessary, and properly documented."*
7. **A stub must not re-implement the producer.** *"With stubbing, there is no way to
   ensure the function being stubbed behaves like the real implementation"*, and
   *"Stubbing leaks implementation details of your code into your test"* (Google, source
   9); preference order **real -> fake -> stub**. Fowler (source 4) prefers **state
   verification** -- assert the downstream effect, not the calls. Source 14 supplies the
   concrete drift case: a renamed client method meant *"production code would fail, yet
   mocked tests would still pass."* The safe stub boundary here is therefore the **LLM
   transport** (`client.messages.create` / `client.generate_content`), which is the lowest
   seam that still executes every line of the pre-mangle.

---

## Consensus vs debate

**Consensus:** validate/parse at the boundary and copy before checking (sources 1, 10);
unit-scoped adequacy misses interface faults (5, 12, 13); syntactic matching cannot follow
a value across a copy or a call (2, 3, 15); non-detection by a checker is a checker defect
(11); stubs that duplicate production logic are a liability (4, 9, 14).

**Debate.** (a) *Can mutation reach call sites in practice?* PIT says barely -- its
call-acting operators are **off by default** and *"fairly unstable"* (6); PyTation says yes
and shows 69% non-subsumption (8). Resolution: it is tool-dependent, and a **hand-written
call-site cell** is the pragmatic path here. (b) *Delete an unreachable branch or make it
live?* The dead-code literature says unreachable code is a bug signal and should go, while
defensive-programming practice keeps fail-closed branches; `portfolio_manager.py:1054-1075`
already resolves this in-repo for PRODUCTION code by keeping it **with a documented
non-promise**. **This debate turns out to be largely moot here** -- the measurement above
shows the branch is not unreachable, only shape-blind, so the question is not
delete-vs-keep but *how wide the rule should be*. The transferable half of the debate does
apply to the CHECKER: an undetectable case in a detector is a soundness flaw (11), so any
shape left uncovered must be **documented**, which is the one thing the literature is
unanimous about.

---

## Pitfalls (from the literature, mapped to this step)

1. **Shotgun parsing** (1) -- adding a *second* normalisation at the caller instead of
   moving the parse to the boundary would deepen the problem.
2. **Equivalent mutants** (5, 8) -- a call-site cell that replaces `dict(_LITE_RISK_DEFAULT)`
   with something semantically identical scores a false SURVIVED. Undecidable in general
   (5); PyTation's rate is 1.61% vs Cosmic Ray's 5.7%.
3. **Soundy-but-undocumented** (11) -- the biggest risk is a *widened* checker that still
   misses a route while now looking exhaustive.
4. **Over-stubbing / change-detector tests** (9, 14) -- a stub that builds the
   `risk_assessment` dict itself would assert the test's own arithmetic.
5. **Shallow copy** (7) -- `dict(_LITE_RISK_DEFAULT)` shares the nested `risk_limits`
   object with the module-level constant. Nothing mutates it today, but any in-place edit
   would corrupt the default process-wide.
6. **Coverage-as-proof** (12, 13) -- 100% line coverage of the producer is compatible with
   zero coverage of the routes.

---

## Application to pyfinagent (external findings -> file:line)

| Finding | Anchor | Implication for the contract |
|---|---|---|
| Interface/call-site mutation is the missing operator group (13, 8) | `scripts/qa/mutation_matrix_86_86.py:50-94` | Needs a cell that mutates **`:3177/:3182/:3411/:3416`** -- e.g. `dict(_LITE_RISK_DEFAULT)` -> `{}` -- and a test that goes RED. A new `mutation_matrix_86_88.py` fits the file's own precedent (`:10-17`: separate matrices per subject are "honest"). |
| Coupled branches are the coverage unit (12) | `:3174` (`if risk_json_match` / `else`) x `:2348/:2352/:2362` | Each of the 4 routes x the resolved verdict kind is the cell grid. |
| An `ast.BoolOp`-operand rule cannot see a `Call` (2, 3, 15) | `verify_lite_risk_seam_86_86.py:53-67`, esp. the dead `:65-66` | Widen to an **assignment/argument-aware** rule: find every `Assign`/`Call` whose value is `dict(_LITE_RISK_DEFAULT)` or `_LITE_RISK_DEFAULT.copy()` or the bare Name, then check whether that binding reaches a `_build_lite_risk_assessment(...)` argument. This is reaching-definitions in miniature; keep it deliberately shallow and **document the residual unsoundness** (11). |
| Non-detection is a tool flaw; placement is a scheme (11) | `mutation_matrix_86_86.py:161-199` (`SEAM-M1`) | Add a seam-checker cell whose mutant is a **whole-dict route**, so the checker must go RED on the very shape its dead branch names. |
| Copy-then-validate (10) / parse at the boundary (1) | `:3177/:3182/:3411/:3416` | The design question for PLAN: the caller should signal **ABSENT** (e.g. `risk_dict = {}` plus an explicit failure marker) rather than fabricate a judge opinion -- preserving the evidence the seam was built to read. Note this is a **behaviour change**: it moves those routes from `SIZE(3.0)` to `ABSENT`, which resolves to the same 3.0 today, so it is value-neutral and provenance-restoring. Verify that claim by driving it. |
| Shallow copy shares nested state (7) | `:3177` etc. vs `:2309` | Worth one assertion; not the headline. |
| Real > fake > stub; state verification (4, 9, 14) | `test_phase_66_2_risk_judge_shape.py:776-788` | Extend the E2E entry point UPWARD: stub only `client.messages.create` to return an object exposing `.content[0].text` (production reads exactly that at `:3171`) and `client.generate_content` for `.text` (`:3406`), then drive the real `_run_claude_analysis` / `_run_gemini_analysis` into the real `decide_trades` and assert **no order**. The stub carries no copy of the dict construction -- it returns a string. That is a **fake transport**, not a stub of business logic. |
| Shape-blind branch: widen it, don't "resurrect" it (11 + in-repo precedent) | `verify_lite_risk_seam_86_86.py:65-66`; precedent `portfolio_manager.py:1054-1075` | The branch already fires on the bare-Name form (measured). **Widen the accepted node shapes** to cover `dict(X)` / `X.copy()` / `{**X}` and keep a positive control PER SHAPE, so "we handle whole-dict routes" is proven for each shape rather than asserted for the family. Whatever remains uncovered gets the `portfolio_manager`-style "what this does NOT promise" note (μSE's documented-unsoundness rule). |

**Scope note for PLAN.** Nothing above says 86.86 was wrong. Its checker `:19-23` and its
matrix `:19-23` both state their scope honestly; the gap is that neither scope includes a
route. 86.88 is the *next* seam outward, not a correction.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **15**
- [x] 10+ unique URLs total -- **29** (15 read in full + 14 snippet-only)
- [x] Recency scan (2024-2026) performed + reported -- 2 findings + 1 non-finding
- [x] Full papers / pages read, not abstracts -- arXiv via `/html/` and ar5iv per the
      gate's chain; the ICSM'07 PDF via `curl` + **pypdf** with regex-verified quotes; the
      `/abs/` page for 2102.06829 was demoted to snippet-only and re-read via ar5iv
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in INTERNAL SCOPE (10 files: the 5 named
      code files, the 3 handoff artifacts read by targeted grep rather than end-to-end,
      plus the 2 instructed-reading rule files). **Disclosed gap:** the 86.74 handoff set
      and `contract_86.86.md` were read by targeted grep, not in full -- adequate for the
      claims made here, but a PLAN that leans harder on 86.74's history should re-read them.
- [x] Contradictions / consensus noted (PIT vs PyTation; the delete-vs-widen question)
- [x] All claims cited per-claim
- [x] **A cross-agent claim was verified, not accepted** -- the 86.86 Q/A's NOTE-2 refuted
      this step's "dead branch" premise, and I reproduced the refutation with my own probe
      before rewriting the finding.

**Audit-class coverage accounting (stated so it cannot be over-read).** 9 external
search/fetch rounds. Rounds **8 and 9 were dry** -- zero new sources read in full beyond
de-dup -- so `dry_rounds = 2 = K_required` and `coverage.dry = true`. Honest caveat: a
**late internal** read (the 86.86 critique) DID produce a material finding, the `<whole-dict>`
correction above. That is an internal-exploration finding, not an external read-in-full
finding, so it does not reset the external dry counter under the rule as written -- but a
reader should know the internal axis went dry later than the external one.
