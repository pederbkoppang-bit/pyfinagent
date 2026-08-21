# Research Brief -- step 90.12

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported for information only).
**Date:** 2026-08-21. **Agent:** Layer-3 Researcher (Workflow rail).

**Topic.** Distinguishing "the mutant could not run" from "the mutant ran and misbehaved" in mutation
testing when a fail-open exception handler swallows the failure: published treatment of stillborn /
trivial / invalid / equivalent mutants and why they must be EXCLUDED from a mutation score rather than
counted as kills; detecting non-viability by exception TYPE vs by output SHAPE; the risk of an
over-eager exclusion silently deleting legitimate mutants; and testing guidance for code paths
deliberately designed to fail open, where the observable signal is a formatted log line rather than a
propagated exception.

---

## ENVELOPE (born inert -- phase-86.37; FLIPPED to COMPLETE as the final act)

This block was written INCOMPLETE with zeroed counts in the first tool call, before any source was
read, and flipped here at the end of the run. The authoritative copy is the identical FINAL block at
the tail of this file.

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 18,
  "urls_collected": 26,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 1,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "See the FINAL envelope at the tail of this file for the full summary string.",
  "brief_path": "handoff/current/research_brief_90.12.md",
  "gate_passed": true
}
```

---

## Status log (append-only, write-first discipline)

- 2026-08-21 -- brief created, envelope born INCOMPLETE, before any source was read.
- 2026-08-21 -- source 1 READ IN FULL (Stryker mutant states). Internal: mutation_matrix_90_1.py header + `_drive_unresolvable` + `run_cell` read.

---

## Search queries run (three-variant discipline, per research-gate.md)

| # | Variant | Query |
|---|---------|-------|
| 1 | year-less canonical | `stillborn mutants invalid mutants excluded from mutation score` |
| 2 | last-2-year window | `trivial compiler equivalent mutants mutation testing 2025 2026` |
| 3 | year-less canonical | `mutation testing mutant states timeout runtime error compile error killed survived denominator` |
| 4 | year-less canonical | `Google mutation testing arid nodes unproductive mutants suppression ICSE` |
| 5 | year-less canonical | `testing exception handlers that swallow errors assert on log output pytest caplog` |
| 6 | current-year frontier | (see Recency scan section) |

## Read in full (counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding / quote |
|---|-----|----------|------|-------------|---------------------|
| 1 | https://stryker-mutator.io/docs/mutation-testing-elements/mutant-states-and-metrics/ | 2026-08-21 | official docs (tier 2) | WebFetch (HTML) | The canonical THREE-WAY partition. Killed + Timeout = **detected**; Survived + NoCoverage = **undetected**; **Runtime error, Compile error, Ignored, Pending = EXCLUDED**. Score is `detected / valid * 100` -- errored mutants leave the DENOMINATOR entirely. Rationale given verbatim: they are excluded because "they couldn't be tested." |
| 2 | https://ar5iv.labs.arxiv.org/html/2102.11378 | 2026-08-21 | peer-reviewed (TSE 2021), Petrovic/Ivankovic/Fraser/Just, "Practical Mutation Testing at Scale: A view from Google" (tier 1) | WebFetch (ar5iv HTML) | Defines an **unproductive mutant**: "either trivially equivalent to the original program or it is detectable, but adding a test for it would not improve the test suite." Introduces **arid nodes**: "An AST node is eligible for mutation if it is covered by at least one test and if it is not arid"; "arid nodes are not considered for mutation and no mutants are produced in them ... they are never created in the first place." **Logging is the flagship arid case**: "The star example of this category is a heuristic that marks any function call arid if the function name starts with the prefix `log` or the object on which the function is invoked is called `logger`." Validated by sampling 100 nodes: "99 indeed were correctly marked." Suppression arithmetic: median 820 -> 77 -> 7 mutants per changelist; productive ratio "from 15% to 89%"; 82% of surfaced mutants labeled productive by developers. |
| 3 | https://docs.pytest.org/en/stable/how-to/logging.html | 2026-08-21 | official docs (tier 2) | WebFetch (HTML) | The canonical Python idiom for asserting on a swallowed error: `caplog.records` (LogRecord objects), `caplog.text` (rendered), `caplog.record_tuples` ((logger, level, message)), `caplog.set_level` / `at_level`, `caplog.clear()`, `caplog.get_records(when)`. Explicit hazard: "The `caplog` fixture adds a handler to the root logger ... If the root logger is modified during a test ... this handler may be removed and cause no logs to be captured." **Structured assertion (`record_tuples`/`records`) is offered as the alternative to substring-matching `text`.** |
| 4 | https://pitest.org/quickstart/basic_concepts/ | 2026-08-21 | official docs (tier 2) | WebFetch (HTML) | PIT ships a dedicated **non-viability status**, distinct from both kill and survival: **NON_VIABLE** = "one that could not be loaded by the JVM as the bytecode was in some way invalid"; **RUN_ERROR** = "something went wrong when trying to test the mutation" (docs note "Certain types of non viable mutation can currently result in a run error"); plus MEMORY_ERROR and TIMED_OUT. Notably, the quickstart page does **not** state how each status affects the score -- the exclusion arithmetic is documented by Stryker (source 1), not by PIT. |
| 5 | https://arxiv.org/html/2408.01760 | 2026-08-21 | preprint (tier 1), "Large Language Models for Equivalent Mutant Detection: How Far Are We?" (ISSTA 2024) | WebFetch (arXiv native HTML) | "A mutant is deemed equivalent if, for all possible test cases, it exhibits the same behavior as the original program under test." Best detector (fine-tuned UniXCoder) reaches **precision 94.33% / recall 81.81% / F1 86.58%** -- i.e. even the best published exclusion oracle wrongly excludes ~1 in 18. TCE baselines score far lower (TCEJavac F1 **39.31%**, TCESoot **50.80%**). The paper stresses precision because a false positive silently deletes a killable mutant; it warns equivalent mutants "introduce significant bias into mutation-based analysis". |

| 6 | https://cosmic-ray.readthedocs.io/en/latest/theory.html | 2026-08-21 | official docs (tier 2), Python mutation-testing tool | WebFetch (HTML) | The Python-ecosystem vocabulary, verbatim: "If the changes cause your code to simply crash, then we say the mutant is 'incompetent'." vs "If your test suite fails, then we say that your tests 'killed' (i.e. detected) the mutant" vs "If your test suite passes, however, we say that the mutant has 'survived'." **NEGATIVE FINDING, recorded because it is design-deciding: the page does NOT say how incompetence is detected, nor whether incompetent mutants are excluded from the score.** The Python tool ecosystem names the category and leaves the arithmetic undefined. |
| 7 | https://github.com/sixty-north/cosmic-ray/issues/310 | 2026-08-21 | community (tier 5), but the tool's own tracker | WebFetch (HTML) | **The failure mode in its OTHER direction, in the wild.** A mutant (`len(name) < 1` -> `len(name) in 1`) that raises `TypeError` at IMPORT time was reported **SURVIVED**: "There's no way this mutant should be surviving, and indeed when I manually apply the mutation, the test suite fails." Root cause: "This causes the test _collection_ phase of py.test to fail entirely, and in fact no tests are executed at all." **A non-viable mutant that crashes OUTSIDE the observation window gets whatever the runner's default is** -- SURVIVED here, KILLED in pyfinagent's matrix -- and in both cases the score is wrong for the same reason. Issue is Closed with no maintainer answer visible. |
| 8 | https://docs.python.org/3/library/exceptions.html | 2026-08-21 | official docs (tier 2) | WebFetch (HTML) | The taxonomy the TYPE-based discriminator rests on. `NameError`: "Raised when a local or global name is not found." `AttributeError`: "Raised when an attribute reference ... or assignment fails." `ImportError`: "Raised when the `import` statement has troubles trying to load a module." `ModuleNotFoundError`: "A subclass of `ImportError` which is raised by `import` when a module could not be located." Versus the domain side -- `ValueError`: "an argument that has the right type but an inappropriate value"; `AssertionError`: "Raised when an `assert` statement fails". **Hierarchy caveats that matter here: `UnboundLocalError` is a SUBCLASS of `NameError`, and `ModuleNotFoundError` is a SUBCLASS of `ImportError` -- so a `startswith`/regex match on the type NAME does not inherit; each subclass name must be listed or matched by prefix.** |

## Identified but snippet-only (context; does NOT count toward the gate)

| # | URL | Kind | Why not fetched in full |
|---|-----|------|-------------------------|
| 1 | https://stryker-mutator.io/docs/General/faq/ | official docs | Same doctrine as source 1, no additional arithmetic |
| 2 | https://ieeexplore.ieee.org/document/7194639/ | peer-reviewed (Papadakis et al., ICSE 2015, TCE) | Paywalled; TCE numbers obtained via source 5's baseline table instead |
| 3 | https://ieeexplore.ieee.org/document/7882714/ | peer-reviewed (TSE, trivial mutant equivalences via compiler optimisations) | Paywalled |
| 4 | https://conf.researchr.org/details/icst-2025/mutation-2025-papers/1/Equivalent-Mutants-Deductive-Verification-to-the-Rescue | conference page (Mutation 2025 @ ICST) | Abstract-only landing page; counting an abstract as "read in full" is forbidden |
| 5 | https://conf.researchr.org/home/icst-2026/mutation-2026 | conference CFP (Mutation 2026) | CFP, not a finding -- recorded as recency evidence that the venue is live |
| 6 | https://onlinelibrary.wiley.com/doi/abs/10.1002/stvr.1907 | peer-reviewed (MUPPAAL, STVR 2025, useless-mutant elimination) | Paywalled abstract |
| 7 | https://mutationtesting.uni.lu/survey.pdf | peer-reviewed survey (Papadakis et al., Advances in Computers 2019) | PDF; project rule + prior measurement is that WebFetch PDF summaries can fabricate quotes -- not used as a quote source |
| 8 | https://dl.acm.org/doi/10.1145/3183519.3183521 | peer-reviewed (State of Mutation Testing at Google, ICSE-SEIP 2018) | Paywalled; superseded for this topic by source 2 (TSE 2021) |
| 9 | https://arxiv.org/pdf/2010.13464 | preprint (mutation testing at Facebook) | PDF URL; adjacent, not load-bearing here |
| 10 | https://dl.acm.org/doi/10.1145/3701625.3701659 | peer-reviewed (Static and Dynamic Comparison of Mutation Testing Tools for Python, SBQS 2024) | Paywalled; relevant to the Python-tool comparison but not to the discriminator design |
| 11 | https://github.com/hcoles/pitest/issues/1352 | community (pitest tracker) | Anecdotal, superseded by source 4 |
| 12 | https://pytest-with-eric.com/fixtures/built-in/pytest-caplog/ | community tutorial | Tier 5; source 3 is the official version |
| 13 | https://qaskills.sh/blog/pytest-caplog-assert-specific-log-level | community tutorial | Tier 5 |
| 14 | https://oneuptime.com/blog/post/2026-01-30-mutation-testing-strategies/view | community blog (2026) | Tier 5; recency-scan evidence only |
| 15 | https://qaskills.sh/blog/mutation-testing-stryker-guide-2026 | community blog (2026) | Tier 5; recency-scan evidence only |
| 16 | https://pybit.es/articles/guest-mutpy-exploration/ | community blog (MutPy) | Tier 5 |
| 17 | https://www.frugaltesting.com/blog/how-to-detect-silent-failures-in-microservices-using-advanced-observability-techniques | community blog | Tier 5; silent-failure framing only |
| 18 | https://arxiv.org/pdf/2404.09952 | preprint (LLMorpheus) | PDF URL; invalid-mutant handling is incidental to its topic |

**Unique URLs collected: 26** (8 read in full + 18 snippet-only), de-duplicated by URL. Alternate
URLs for the SAME paper were dropped rather than counted twice (e.g. the UW mirror of source 2 and the
research.google landing page for #8), so this figure is the LOWER of the de-duped and naive counts.

---

## Recency scan (2024-2026) -- PERFORMED

Queries run in the window: `trivial compiler equivalent mutants mutation testing 2025 2026`;
`mutation testing 2026 mutant validity classification exception type non-viable detection`;
`testing fail-open fallback silent failure observability assert error was logged 2025`.

**Result: 3 new findings that COMPLEMENT the canonical sources; 0 that supersede them.**

1. **Exclusion doctrine is stable, not contested.** Nothing in the window revises the
   detected/undetected/**excluded** partition. The 2026-frontier searches return the Mutation 2026
   workshop CFP (https://conf.researchr.org/home/icst-2026/mutation-2026) and practitioner guides
   restating the same formula. This is a stability finding, and it is the load-bearing one for 90.12:
   **the design direction is settled prior art, not a research bet.**
2. **The exclusion ORACLE is where 2024-2026 work is happening, and it is precision-bound.**
   arXiv:2408.01760 (ISSTA 2024, read in full) reports the best fine-tuned LLM equivalent-mutant
   detector at **precision 94.33% / recall 81.81%**, against TCE baselines at F1 39-51%. Mutation 2025
   (deductive verification) and MUPPAAL (STVR 2025) both attack the same problem. Read as a design
   input: an exclusion rule with less than perfect precision **destroys mutants that were killable**,
   and the field treats that as the expensive error.
3. **Nothing in the window gives a canonical recipe for testing a deliberately fail-open handler.**
   The 2025-2026 material on "silent failures" is observability/tracing practitioner content
   (tier 5), not testing methodology. **The nearest authoritative treatment remains source 2's arid-node
   heuristic (2021) -- which says the opposite of what a naive reading suggests: Google suppresses
   mutants ON logging calls, it does not treat a log line as a non-signal.** Recorded honestly as a gap.

---

## Key findings

**F1 -- The three-way partition is the settled model, and "ERROR" is a THIRD bucket, not a soft kill.**
Stryker: Killed + Timeout = detected; Survived + NoCoverage = undetected; **Runtime error + Compile
error + Ignored + Pending = excluded**, score `detected / valid * 100`
(https://stryker-mutator.io/docs/mutation-testing-elements/mutant-states-and-metrics/, accessed
2026-08-21). PIT independently ships **NON_VIABLE** ("could not be loaded by the JVM as the bytecode
was in some way invalid") and **RUN_ERROR** ("something went wrong when trying to test the mutation")
as statuses distinct from KILLED and SURVIVED
(https://pitest.org/quickstart/basic_concepts/). Cosmic Ray names the same category **"incompetent"**:
"If the changes cause your code to simply crash, then we say the mutant is 'incompetent'"
(https://cosmic-ray.readthedocs.io/en/latest/theory.html). **Three independent tool ecosystems converge
on a third bucket.** `mutation_matrix_90_1.py`'s `ERROR` score is that bucket, and 90.1 criterion 5
clause 3 is a restatement of settled practice, not a local invention.

**F2 -- The exclusion oracle's PRECISION is the risk, and the literature says so with numbers.**
The best published equivalent-mutant oracle reaches precision 94.33% (arXiv:2408.01760,
https://arxiv.org/html/2408.01760) -- roughly 1 wrongly-excluded mutant in 18. Google's suppression is
validated the same way: 100 sampled arid nodes, "99 indeed were correctly marked"
(https://ar5iv.labs.arxiv.org/html/2102.11378). **The published discipline is: measure the false-exclusion
rate on a labelled sample and report it.** This maps exactly onto step 90.12's criteria 3 and 5.

**F3 -- The failure mode is SYMMETRIC, and it has been observed in the wild in the other direction.**
cosmic-ray issue #310 (https://github.com/sixty-north/cosmic-ray/issues/310): a mutant raising
`TypeError` during pytest COLLECTION was scored **SURVIVED** -- "There's no way this mutant should be
surviving, and indeed when I manually apply the mutation, the test suite fails ... no tests are executed
at all." pyfinagent's defect is the mirror image (non-viable scored **KILLED**). **The shared root cause
is identical: the non-viability crashes OUTSIDE the observation window, so the mutant inherits whatever
the runner's default is.** A fail-open handler is exactly such a window-collapsing device: it converts
"the code never ran" into "the code ran and returned 0".

**F4 -- Exception TYPE is a sound discriminator, but the Python hierarchy has two traps and TypeError
is a genuine hole.** Per https://docs.python.org/3/library/exceptions.html: `NameError` = "a local or
global name is not found"; `AttributeError` = "an attribute reference ... fails"; `ImportError` =
"trouble ... to load a module" -- all RESOLUTION failures. `ValueError` = "right type but an inappropriate
value"; `AssertionError` = "an `assert` statement fails" -- DOMAIN errors. **Trap 1:**
`UnboundLocalError` subclasses `NameError` and `ModuleNotFoundError` subclasses `ImportError`, and a
string match on the type NAME does not inherit -- the subclass names must be listed. **Trap 2 (from F3):**
`TypeError` is a domain error by the taxonomy, yet cosmic-ray #310's non-viable mutant died with one.
**So exception type alone is INCOMPLETE as a viability oracle** -- which is why the type test must be
the last rung of a ladder that already caught parse and import failures structurally, not a
replacement for them.

**F5 -- A log line IS a legitimate test oracle; the canonical Python idiom is structured, not
substring.** pytest offers `caplog.records` (LogRecord objects) and `caplog.record_tuples`
((logger, level, message)) alongside the rendered `caplog.text`, and warns that root-logger
reconfiguration can silently capture nothing (https://docs.pytest.org/en/stable/how-to/logging.html).
**The transferable rule: assert on the STRUCTURED field, not on prose, and prove the capture channel is
live** (the pytest doc's own warning is a "your oracle can be silently dark" warning).

**F6 -- Google suppresses mutants ON logging statements -- which is NOT the same as ignoring log
output as a signal.** "The star example ... marks any function call arid if the function name starts
with the prefix `log` or the object on which the function is invoked is called `logger`"
(https://ar5iv.labs.arxiv.org/html/2102.11378). This is about not MUTATING logging code (an
unproductive mutant), not about refusing to OBSERVE it. Naming the distinction because it is an easy
misreading and would point 90.12 in exactly the wrong direction.

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `scripts/qa/mutation_matrix_90_1.py` | 839 | The matrix. `UNRESOLVABLE_ERRORS` tuple `:349`; `_drive_unresolvable` `:353-415`; `run_cell` `:695-786`; scoring tally `:822-830`; cell `N0` `:570-572`; cell `M14` `:684`. | **ALREADY CARRIES A DRAFT FIX** (mtime 2026-08-21 10:53). `_drive_traceback` is renamed `_drive_unresolvable` and now has branch (b) at `:411-415`: `re.search(rf"\b{t}: [^\n]+", err)` over `UNRESOLVABLE_ERRORS`. Main owns whether this is the shipped design. |
| `scripts/qa/mutation_matrix_90_1.py` (ladder) | -- | Four-rung viability ladder inside `run_cell`: (1) anchor-uniqueness `:703-706`; (2) `ast.parse` -> ERROR `:712-717` (cycle 2); (3) subprocess **import probe** -> ERROR `:733-752` (cycle 3); (4) `_drive_unresolvable(obs)` -> ERROR `:769-772` (cycle 4 + this step). | Rungs 1-3 are STRUCTURAL and type-independent. **This is what makes F4's `TypeError` hole tolerable**: a non-viable mutant that dies at import is already ERROR before any type match runs. Rung 4 only needs to cover residual failures on the hook branch after a clean import. |
| `scripts/harness/attempt_gate.py` | 824 | Subject module. `handle_hook` `:382`; **the fail-open handler `:465-470`** -- `except Exception as exc:` then `print(f"[attempt-gate] INTERNAL ERROR -- {type(exc).__name__}: {exc} -- failing OPEN (the launch proceeds UNCOUNTED; ...)", file=sys.stderr)` then `return 0`. Docstring states the policy at `:36-42`. Renamed call site in the reproduction is `:393`. | **CORRECT AND MUST NOT CHANGE.** Per the hook docs only exit 2 blocks; breaking every Workflow launch on a gate bug is worse than one uncounted attempt. Note it prints via `print(..., file=sys.stderr)` -- **NOT the `logging` module**, so `caplog` (F5) is INAPPLICABLE; the observable is raw stderr captured by `subprocess.run(capture_output=True)`. `type(exc).__name__` is already interpolated, i.e. **the type is on the wire today** -- the discriminator does not need a production change to read it. |
| `scripts/qa/verify_error_discriminator_90_12.py` | 447 | The re-runnable proof. `PRE_FIX_REV = "d564ad58"` `:57`; `EXPECTED_CHECKS = 20` `:59`; `extract_pre_fix_discriminator()` `:86+` lifts the red baseline via `git show`; the domain-exception mutant helper `:129`; the six cells `:140-152` (QA1/QA1b/QA1c -> expect ERROR, QX2 control -> ERROR, DOM -> not-ERROR, N0 -> KILL_NONE); scoring loop `:195-235`; cardinality floor `:277-283`. | **ALREADY DRAFTED** (mtime 2026-08-21 10:54). Design is strong on two axes the literature endorses: the red baseline is EXTRACTED FROM GIT rather than re-typed (`:24-28`), and each mutant is observed ONCE then scored by BOTH discriminators, so the pair is a differential on identical evidence. |
| `handoff/current/evaluator_critique_90.1.md` | 447 | The cycle-5 FAIL. Verdict block `:267-330`; finding id `illusory-guard [BLOCK]` `:284`; the executed evidence `:290-291`; "WHAT WOULD CLOSE THIS" inside `notes` `:324`; Main's record `:348+`; verbatim mutant stderr `:368-371`. | The Q/A's own prescription at `:324` is: "the discriminator must read the exception TYPE out of the fail-open handler's message as well as out of a traceback ... the fix should then be proven against a call-site rename inside handle_hook, not only against a definition rename." **The drafted code follows it.** |
| `.claude/masterplan.json` | -- | Step 90.12, 6 immutable criteria + `command` + `live_check`. | `status: pending`. `audit_basis` cites `mutation_matrix_90_1.py:341` and `UNRESOLVABLE_ERRORS` at `:337` -- **both line numbers are STALE against the current file** (`:349` / `:353` today), because the draft fix moved them. Immutable criteria are unaffected; do not edit them. |

---

## Consensus vs debate (external)

**Consensus (strong, 3 independent tool ecosystems + 1 industrial paper):** a mutant that could not run
belongs in a third bucket, excluded from the mutation-score DENOMINATOR, never credited as a kill
(sources 1, 4, 6, and source 2's productive-mutant framing).

**Debate:** *how* to detect non-viability. TCE (compiler-based) reaches F1 39-51%; learned oracles reach
F1 86.58% but only 94.33% precision; deductive verification (Mutation 2025) attacks it from proof;
Google side-steps detection entirely by never GENERATING the bad mutants (arid nodes). **No published
technique claims perfect precision.** Every one of them is a heuristic with a measured false-exclusion
rate.

**Unaddressed by the literature:** none of the surveyed sources treats a *deliberately fail-open
production handler* as the mechanism that hides non-viability. The closest analogue is cosmic-ray #310's
collection-phase crash. **pyfinagent's case appears to be a genuinely under-documented variant, and that
is itself a finding** -- it means the design cannot be copied from prior art and must be argued from the
mechanism.

---

## Pitfalls (from the literature, mapped to this step)

1. **Over-eager exclusion silently deletes killable mutants** (F2). Published precision ceilings are
   ~94%. **Mitigation the literature actually uses: a labelled control sample.** Step 90.12's DOM +
   N0 + real-kill triple IS that sample -- it is small (n=3) and its smallness should be stated, not
   glossed.
2. **Type-name matching does not inherit** (F4, trap 1). `UnboundLocalError` and `ModuleNotFoundError`
   are subclasses whose NAMES do not start with their parent's name. The shipped tuple at
   `mutation_matrix_90_1.py:349` lists `ModuleNotFoundError` explicitly but **not** `UnboundLocalError`
   -- a gap worth naming even if no current cell produces one.
3. **Type is not a complete viability oracle** (F4, trap 2 / F3). `TypeError` proves it. The structural
   rungs (`ast.parse` `:712`, import probe `:733`) are what make the type test safe; **the fix must not
   be presented as if type alone decides viability.**
4. **A regex on prose is a fragile oracle** (F5). The current branch anchors on `rf"\b{t}: [^\n]+"`
   -- i.e. `<Type>: <msg>` -- which is tighter than a bare word match, but it is still a match on a
   MESSAGE FORMAT owned by production code at `attempt_gate.py:466-469`. **If that message is ever
   reworded, the discriminator goes dark silently.** The literature's answer (F5) is to assert on the
   structured field; the cheap local analogue is a check that pins the production message shape, so a
   reword turns a test red instead of turning the oracle off.
5. **A capture channel can be silently dark** (F5, pytest's own warning). The matrix's channel is
   `subprocess.run(capture_output=True).stderr`. A positive control that PROVES stderr is non-empty on
   a known-broken mutant is what keeps rung 4 from being vacuous -- QA1/QA1b/QA1c serve this role.
6. **Do not confuse "don't mutate logging" with "don't observe logging"** (F6).

---

## Application to pyfinagent (external findings -> file:line)

- **Criterion 1 (type decides, not traceback presence)** is F1 + F4. The production handler already
  interpolates `type(exc).__name__` at `attempt_gate.py:466`, so **the type is on the wire with no
  production change** -- which matters because `attempt_gate.py:36-42` documents the fail-open policy as
  deliberate and CLAUDE.md's batched-restart rule makes production edits expensive. The drafted branch
  (b) at `mutation_matrix_90_1.py:411-415` is the right shape. Pitfall 2 says consider
  `UnboundLocalError` in the `:349` tuple.
- **Criterion 2 (call-site renames, QX2 still ERROR)** is exactly F3's window argument: a definition
  rename kills the module at import (rung 3 catches it structurally), a call-site rename survives to
  runtime and is swallowed (only rung 4 catches it). The two cells prove DIFFERENT rungs, and saying so
  is stronger than treating QX2 as a regression check.
- **Criterion 3 (M14 stays KILLED, N0 SURVIVES, real-kill KILLED)** is the labelled control sample of
  F2. `AssertionError` is a DOMAIN error by https://docs.python.org/3/library/exceptions.html and is
  absent from `UNRESOLVABLE_ERRORS:349`, so M14's separation is TYPED, not incidental -- and it holds
  even when the domain exception arrives through the same fail-open handler, which is what
  `verify_error_discriminator_90_12.py:129` + the `DOM` cell `:148` drill.
- **Criterion 4 (red-first, both tables side by side)** is already structurally strong:
  `PRE_FIX_REV = "d564ad58"` `:57` extracts the real prior code from git rather than re-typing it
  (project memory: `feedback_a_control_built_from_your_own_pattern_tests_nothing`), and one observation
  scored by two discriminators is a true differential.
- **Criterion 5 (no silent cell loss)** is the direct local instance of F2, and it has ALREADY fired
  once here -- 90.1 cycle 4 flagged M14 as ERROR and deleted a legitimate cell
  (`mutation_matrix_90_1.py:363-368` records it). Printing cell count + every score before and after is
  the project's version of Google's 100-node sample.
- **Criterion 6 (verdict ledger byte-identical)** is containment, unrelated to the literature;
  `verify_error_discriminator_90_12.py:52` already pins `VERDICT_LEDGER`.
- **Stale anchors to fix in prose, not code:** step 90.12's `audit_basis` cites
  `mutation_matrix_90_1.py:341` and `:337`; the current file has `_drive_unresolvable` at `:353` and
  `UNRESOLVABLE_ERRORS` at `:349`. The criteria themselves are immutable and unaffected.
- **`caplog` is NOT the tool here** (F5). `attempt_gate.py:466-469` uses `print(..., file=sys.stderr)`,
  not `logging`. Any contract language borrowing pytest logging idiom would be wrong; the channel is raw
  stderr from `subprocess.run`.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **8** (2 peer-reviewed/preprint,
      5 official docs, 1 tool tracker)
- [x] 10+ unique URLs total -- **26**
- [x] Recency scan (2024-2026) performed + reported -- 3 queries, 3 complementary findings, 0 superseding
- [x] Full pages read (not abstracts) for the read-in-full set -- arXiv via native HTML / ar5iv per the
      PDF chain; no `arxiv.org/pdf/` fetched; no PDF used as a quote source
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in the caller's INTERNAL SCOPE (4 of 4) plus
      `.claude/masterplan.json`
- [x] Contradictions / consensus noted (F4 trap 2 vs F1; F6 vs the naive reading)
- [x] Claims cited per-claim with URL + access date
- [ ] **GAP, stated not padded:** no authoritative source was found that treats testing a deliberately
      fail-open handler as a named methodology problem. The design must be argued from mechanism
      (F3 + F4), not cited to prior art.

---

## ENVELOPE -- FINAL (supersedes the born-inert block above)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 18,
  "urls_collected": 26,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 1,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Excluding non-viable mutants from the score denominator is settled prior art (Stryker excluded-bucket, PIT NON_VIABLE, cosmic-ray incompetent, Google unproductive/arid). The published risk is the ORACLE's precision, not the doctrine: best equivalent-mutant detector 94.33% precision, Google validated arid-node suppression on 100 labelled nodes. cosmic-ray #310 shows the mirror failure (non-viable scored SURVIVED) with the same root cause -- the crash lands outside the observation window, which is precisely what a fail-open handler creates. Exception TYPE is a sound but INCOMPLETE oracle (TypeError hole; UnboundLocalError/ModuleNotFoundError subclass names do not inherit), which is why it must remain rung 4 of a parse/import/run ladder. attempt_gate.py:466 already interpolates type(exc).__name__, so no production change is needed. caplog is inapplicable -- the handler prints to raw stderr.",
  "brief_path": "handoff/current/research_brief_90.12.md",
  "gate_passed": true
}
```
