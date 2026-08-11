# Research Brief -- phase-86.5

**Topic:** Triaging a standing set of failing tests in a long-lived codebase --
classifying pre-existing failures by ROOT CAUSE rather than symptom, and deciding
fix vs delete vs quarantine.

**Tier:** moderate. **Audit-class:** YES (loop-until-dry, K=2).
**Researcher:** Layer-3 Workflow rail. **Started:** 2026-08-11.

---

## ENVELOPE (born inert -- phase-86.37; flipped to COMPLETE as the final act)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 37,
  "snippet_only_sources": 21,
  "urls_collected": 58,
  "recency_scan_performed": true,
  "internal_files_inspected": 19,
  "coverage": {
    "audit_class": true,
    "rounds": 18,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "summary": "Measured 17 failures (not 26), 100% deterministic, 0 order-dependent, 0 clock-dependent; they collapse to 8 root-cause groups. 86.3 guard held (kill-switch audit untouched).",
  "brief_path": "handoff/current/research_brief_86.5.md",
  "gate_passed": true
}
```

---

## Status log (append-only)

- 2026-08-11 -- brief created, envelope born INCOMPLETE. Beginning internal
  measurement (pytest run) + external round 1 in parallel.

---

## Constraint encountered (stated, not papered over)

`WebSearch` was **unavailable for this entire session**: the shared session budget
was already exhausted (`200 of 200 WebSearch calls`) before this agent was spawned
(matches auto-memory `reference_websearch_budget_is_session_shared`). All external
sourcing below was therefore done by **direct `WebFetch` of known-canonical URLs**
plus the **arXiv Atom API via `curl`** as the discovery substitute
(`https://export.arxiv.org/api/query?search_query=abs:"flaky test"...&sortBy=submittedDate`).
The three-variant query discipline (current-year / last-2-year / year-less canonical)
is satisfied structurally: the arXiv API sweep is the 2026 + 2025 frontier pass, and the
year-less canonical pass is the Luo-2014 / Fowler / Google-Testing-Blog set below.

## Round 1-2 -- external sources read IN FULL

### 1. Fowler, "Eradicating Non-Determinism in Tests" (2011, canonical; year-less pass)
https://martinfowler.com/articles/nonDeterminism.html -- accessed 2026-08-11, WebFetch, blog (tier 3, but canonical for the quarantine pattern)

- The erosion argument, verbatim: *"If you have a suite of 100 tests with 10
  non-deterministic tests in them, then that suite will often fail."* and
  *"Once that discipline is lost, then a failure in the healthy deterministic tests
  will get ignored too. At that point you've lost the whole game and might as well
  get rid of all the tests."* -- this is the formal statement of the broken-window
  effect on a suite: a permanently-red suite is not a degraded signal, it is a
  **destroyed** one, because triage discipline is what decays, not the tests.
- **Quarantine, with limits, and the limits are the point.** Fowler's two
  time-boxing mechanics, verbatim: *"One is a simple numeric limit: e.g. only allow
  8 tests in quarantine"* and *"Another route is to put a time limit on how long a
  test may be in quarantine, such as no longer than a week."* Quarantined tests
  **still run** (out of the deployment pipeline, not out of existence); Fowler
  **never** recommends deletion -- a quarantined test is lost regression coverage
  and is a debt with a due date.
- Taxonomy of non-determinism causes (5): **lack of isolation** (shared state),
  **asynchronous behaviour**, **remote services**, **time**, **resource leaks**.
- On defect-vs-obsolete-expectation, verbatim: *"Sometimes, of course, a test
  failure is due to a change in what the code is supposed to do, but the test hasn't
  been updated to reflect the new behavior. This is essentially a bug in the tests,
  but is equally easy to fix if it's caught right away."* -- note the qualifier
  **"if it's caught right away"**: the distinguishing evidence decays with age,
  which is exactly the position a standing set of pre-existing failures is in.

### 2. Eck, Palomba, Castelluccio, Bacchelli, "Understanding Flaky Tests: The Developer's Perspective" (ESEC/FSE 2019)
https://ar5iv.labs.arxiv.org/html/1907.01466 -- accessed 2026-08-11, WebFetch (ar5iv HTML per the arXiv chain), peer-reviewed (tier 1)

- Design: 21 professional Mozilla developers classified **200 flaky tests they had
  themselves previously fixed**; plus a survey of 121 developers (106 industrial).
- **Confirms 7 of Luo et al.'s categories** with measured counts and a fixing-effort
  score: Concurrency (61, effort 4.0), Async Wait (52, 3.0), Test Order Dependency
  (22, 2.0), Resource Leak (14, 3.0), Float Precision (16, 4.0), **Time (14, 1.0)**,
  Randomness (13, 1.0).
- **Adds 4 NEW categories**, verbatim: *"The categorization by the Mozilla developers
  uncovered four previously unreported causes of flakiness, which are also deemed as
  those requiring the most effort to fix."* -- **Too Restrictive Range** (40 cases;
  *"valid output values are outside the assertion range considered at test design
  time"*), **Test Case Timeout** (18), **Platform Dependency** (10;
  *"non-deterministic test failures occurring only on specific platforms"*),
  **Test Suite Timeout** (14).
- **Flakiness is not always a test-code problem**: in the Concurrency class, **34%
  originated in production code**, not test code. Directly relevant to the "is this
  a real defect?" question.
- **The disable-instead-of-fix pattern is measured, and is a warning:** Time category
  -- **75% were simply disabled**; Too Restrictive Range -- 16% disabled, and the
  authors comment *"in a not negligible amount of cases, developers prefer to just
  disable flaky tests rather than properly deal with them"*, attributing it to
  *"the limited time developers have"*. Test Order Dependency was fixed by
  **"Remove Dependency" 100% of the time** -- order-dependence has a known, cheap,
  universally-applied remedy.

### 3. "How Far Are We from Detecting Flaky Tests? On the Limits of Code-Based Detection" (arXiv 2607.09345v1, Jul 2026) -- [ADVERSARIAL / RECENCY]
https://arxiv.org/html/2607.09345v1 -- accessed 2026-08-11, WebFetch, preprint (tier 1)

- Thesis, verbatim: *"Flakiness is not a static property of test code, which often
  lacks the information needed to decide whether a test is flaky."*
- **This is the source that disagrees with the tempting shortcut for this step**
  (classify by reading the test file). Measured collapse once data leakage and
  fix-commit shortcuts are removed: *"flaky-class F1 scores of 0.035 and 0.070
  versus the always-flaky predictor's 0.054"*; FlakeFlagger *"collapsed from 0.79 to
  0.07"*; protocol change alone *"moved the flaky-class F1 score from 0.035 in
  (C-IDoFT, Disjoint) to 0.746 in (C-IDoFT, CV)"*. Dataset: 54,468 unit tests / 57
  projects.
- The number that governs THIS step's method: on 86 end-to-end tests, *"for 42% of
  them...we could attribute a cause from the test code and CI log. For the remaining
  58%...diagnosis required execution evidence"* -- i.e. **root cause is majority-
  unrecoverable from source reading; you must run it and capture the failure.**
- Reframing, verbatim: *"asking whether an observed failure is flaky or how likely a
  test is to fail given its execution environment"* should replace the test-level
  binary label.

### 4. "Flaky Tests in a Large Industrial Database Management System: An Empirical Study of Fixed Issue Reports" (arXiv 2602.03556v1, Feb 2026) -- [RECENCY]
(Subject system is SAP HANA. NOTE: a sibling paper, arXiv 2602.23957 "The Vocabulary of Flaky Tests in the Context of SAP HANA", is a DIFFERENT paper and is listed snippet-only below -- do not conflate them.)
https://arxiv.org/html/2602.03556v1 -- accessed 2026-08-11, WebFetch, preprint (tier 1)

- **16-category taxonomy over 559 fixed issue reports**: Concurrency, Timeout,
  Oracle-Brittleness, Configuration, Async Wait, Isolation, Platform, Application,
  Test-Framework, Error-Handling, Memory-Management, Environment, Randomization,
  Network, Numeric-Semantics, Locale.
- Distribution: **Concurrency 23% (130), Timeout 16% (87), Oracle-Brittleness 10% (57)**.
- Explicitly reconciles with the two canonical papers: Luo et al. found *"Async wait,
  Concurrency, and 'test order dependency' (Isolation in our taxonomy)"* dominant;
  Eck et al. found Async Wait + Concurrency dominant with *"'too restrictive range'
  (Oracle Brittleness) as the third common category, appearing in 40 of 234 cases (17%)"*.
- **Honest negative**: the paper does NOT cover fix-vs-quarantine-vs-delete decision
  process, fix strategies, or time-to-fix. Recorded so the brief does not over-claim it.

### 5. Fowler, "Self Testing Code" (year-less canonical pass)
https://martinfowler.com/bliki/SelfTestingCode.html -- accessed 2026-08-11, WebFetch, blog (tier 3)

- Definition, verbatim: *"You have self-testing code when you can run a series of
  automated tests against the code base and be confident that, should the tests pass,
  your code is free of any substantial defects."* -- note this defines value in terms
  of the **pass** signal, which a standing red set destroys: you can no longer run
  "the suite" and read a green.
- *"as well as building your software system, you simultaneously build a bug detector
  that's able to detect any faults inside the system"*; *"Should anyone in the team
  accidentally introduce a bug, the detector goes off."*
- Honest limit: this page does **not** directly discuss tolerating failing tests; the
  erosion argument comes from source 1, not this one.

### 6. Spotify Engineering, "Test Flakiness -- Methods for identifying and dealing with flaky tests" (2019)
https://engineering.atspotify.com/2019/11/18/test-flakiness-methods-for-identifying-and-dealing-with-flaky-tests -- accessed 2026-08-11, WebFetch, industry (tier 4)

- Operational definition, verbatim: *"a test that both passes and fails periodically
  without any code changes."* -- **this definition excludes a deterministically
  failing test**, which matters: most of pyfinagent's standing failures are
  deterministic, so the flaky-test literature's *detection* half does not apply, only
  its *taxonomy* and *disposition* halves do.
- Measured: making flakiness merely **visible** in a dashboard *"reduced test
  flakiness at Spotify from 6% to 4% in two months"* -- visibility alone is a lever.
- Honest negative: no quarantine mechanics, ownership, time limits, or deletion policy
  in this post.

### 7. Google Testing Blog, "Where do our flaky tests come from?" (Apr 2017)
https://testing.googleblog.com/2017/04/where-do-our-flaky-tests-come-from.html -- accessed 2026-08-11, WebFetch, official engineering blog (tier 2)

- Corpus: **4.2 million tests** analysed; **test size correlates with flakiness**, a
  *"linear trend"* -- larger tests are flakier -- but *"very small tests (unit tests)
  are flaky"* too, so size is a risk factor, not a cause.
- Dominant mechanism claimed: timing -- *"absolute timing (you expect something to
  complete in XX time and it doesn't) or relative timing (one thread occasionally
  executes faster than another)."*

### Partial / not counted as read-in-full
- https://testing.googleblog.com/2016/05/flaky-tests-at-google-and-how-we.html --
  **body did not render** through either WebFetch or `curl` + tag-strip (Blogger
  serves the post body dynamically); only the 36-comment thread was retrievable.
  **Not counted toward the gate.** One high-value verbatim from the Google author in
  that thread is still worth recording because it is exactly this step's thesis:
  *"From the testing system point of view a test that fails reliably is far better
  than a test that is flaky! A persistently failing test is giving a clear signal
  about what to do - even it means fixing the test."* And, on whether flaky failures
  hide real bugs: *"We do not currently keep accurate count of the number of times
  that flaky tests are really masking bugs in the code."*

## Rounds 3-8 -- further external sources read IN FULL

### 8. Meta/Facebook Engineering, "Probabilistic flakiness: How do you test your tests?" (Dec 2020)
https://engineering.fb.com/2020/12/10/developer-tools/probabilistic-flakiness/ -- accessed 2026-08-11, WebFetch, industry (tier 4)
- Reframing, verbatim: *"the right question to ask is not whether a particular test is flaky, but how flaky it is."*
- The asymmetry that governs triage, verbatim: *"A passing test indicates the absence of corresponding regression, while a failure is merely a hint to run the test again."*
- Disposition is **demotion, not deletion**, verbatim: *"Tests that deteriorate significantly and/or are not fixed on time are marked as flaky, which renders them ineligible for change-based testing."* -- i.e. a time-boxed, automatic, *visible* quarantine.
- Base rates: *"For unit tests, it is well below 1 percent, while for some end-to-end test frameworks it reaches 10 percent."*

### 9. "Cross-Project Flakiness: A Case Study of the OpenStack Ecosystem" (arXiv 2602.09311v2, Feb 2026) -- [RECENCY]
https://arxiv.org/html/2602.09311v2 -- accessed 2026-08-11, WebFetch, preprint (tier 1)
- Categories: **Event-Related 89%**, Dependency-Related 21%, Configuration-Related 21%.
- Verbatim: *"non-deterministic behavior may not be only inherited by the test logic (true behavioral inconsistency), but could also be triggered by configurations or environments."*
- Recommends shifting *"from the 'recheck and wait' paradigm to early intervention on flaky tests"*.

### 10. "A Systematic Evaluation of Environmental Flakiness in JavaScript Tests" (arXiv 2602.19098v1, Feb 2026) -- [RECENCY]
https://arxiv.org/html/2602.19098v1 -- accessed 2026-08-11, WebFetch, preprint (tier 1)
- Defines the class that dominates pyfinagent's 17: *"test failures that are caused not by issues in the test logic or application code itself, but by variations in the environment in which the tests are executed"*; operationally *"tests that fail in one environment but pass in others (with other variables in control, including code)."*
- Measured: 1,355 environment-dependent tests over 116 projects; 65 of 116 projects environment-dependent.
- Their remedy is **annotation-based, declared quarantine**: `@skipOnOs win32` / `@skipOnNodeVersion 20,22`, which *"skip and report"* rather than fail. The *report* half is the part that makes it a quarantine and not a hiding place.

### 11. Winters, Manshreck, Wright (eds.), *Software Engineering at Google*, Ch.11 "Testing Overview"
https://abseil.io/resources/swe-book/html/ch11.html -- accessed 2026-08-11, WebFetch, official/book (tier 2)
- **The threshold number this step needs**, verbatim: *"At Google, our flaky rate hovers around 0.15%, which implies thousands of flakes every day"* and *"our experience suggests that as you approach 1% flakiness, the tests begin to lose value."*
- Verbatim: *"Teams that prioritize fixing a broken test within minutes of a failure are able to keep confidence high and failure isolation fast."*
- Honest negative: this chapter has **no** explicit disable/delete policy; do not attribute one to it.

### 12. "ReproFlake: A Dataset of Reproducible Flaky-Test Failures" (arXiv 2605.21677v1, May 2026) -- [RECENCY]
https://arxiv.org/html/2605.21677v1 -- accessed 2026-08-11, WebFetch, preprint (tier 1)
- The method this step should copy: capture, per failure, **execution logs, test reports, the exception TYPE and the exception MESSAGE**, plus a passing and a failing version. Verbatim: *"the majority of exceptions in ReproFlake are assertion errors, which we further categorize based on the exception message."*
- Categories recoverable from that evidence: Timing-Dependent, Order-Dependent, Implementation-Dependent, Non-Idempotent-Outcome. Anything not resolvable is labelled *"Unreliably Reproducible (with a presumed category when available, or Unclassified otherwise)"* -- an explicit **Unclassified** bucket is legitimate practice.

### 13. pytest official docs, "How to use skip and xfail to deal with tests that cannot succeed"
https://docs.pytest.org/en/stable/how-to/skipping.html -- accessed 2026-08-11, WebFetch, official docs (tier 2)
- `skip`/`skipif` -- *"pytest should skip running the test altogether"*; the body never executes.
- `xfail` -- *"This test will run but no traceback will be reported when it fails."* The body **does** execute. This is the mechanical difference that makes `xfail` a quarantine and `skip` a hiding place.
- `xfail(strict=True)` -- *"This will make XPASS ('unexpectedly passing') results from this test to fail the test suite."* **This is the anti-silent-quarantine device**: a strict xfail turns "the bug got fixed / the expectation became valid again" into a LOUD failure instead of an ignorable `xpass`. Project-wide default via the `xfail_strict` ini option.
- `xfail(raises=...)` narrows the licence to one exception type: *"the test will be reported as a regular failure if it fails with an exception not mentioned in raises"* -- so a quarantine can be pinned to the exact measured signature and will break if the failure MODE changes.

### 14. Ahmad, Leifler, Sandahl, "Empirical Analysis of Factors and their Effect on Test Flakiness - Practitioners' Perceptions" (arXiv 1906.00673)
https://ar5iv.labs.arxiv.org/html/1906.00673 -- accessed 2026-08-11, WebFetch (ar5iv), preprint (tier 1)
- 23 perceived factors, average agreement 84%. Reports Luo et al.'s headline distribution second-hand: **async wait 45%, concurrency 20%, test order dependency 12%**.
- Verbatim: *"re-run ... is the most widely used technique to address test flakiness"*, and *"7 out of 10 participants shared that they do not use any notations to represent flaky tests"* -- i.e. **undeclared, invisible quarantine is the industry default**, which is precisely the failure mode to avoid.

### 15. "Understanding and Detecting Flaky Builds in GitHub Actions" (arXiv 2602.02307v1, Feb 2026) -- [RECENCY]
https://arxiv.org/html/2602.02307v1 -- accessed 2026-08-11, WebFetch, preprint (tier 1)
- Extends the taxonomy ABOVE the test level: Network 15.8%, Dependency Resolution 6.32%, External Environment Inconsistency 4.7%, API Service Unavailable 1.16%, ... Flaky **tests** are only *"64.99% of all flaky failures"* -- the rest are build/environment.
- Verbatim: *"Rerunning jobs is the most common mitigation pattern for flaky failures; however, reruns often serve only as a temporary workaround rather than resolving the underlying root causes."*

### 16. Fatima, Hemmati, Briand, "Systemic Flakiness: An Empirical Analysis of Co-Occurring Flaky Test Failures" (arXiv 2504.16777v1, Apr 2025) -- [RECENCY; the single most load-bearing source for this step]
https://arxiv.org/html/2504.16777v1 -- accessed 2026-08-11, WebFetch, preprint (tier 1)
- Verbatim: *"Flaky tests often exist in clusters, with co-occurring failures that share the same root causes, which we call systemic flakiness."*
- Verbatim: *"Of the 810 flaky tests, 606 (75%) belong to a cluster."* and *"The mean size over the 45 clusters is 13.5 flaky tests."*
- Verbatim: *"By recognizing systemic flakiness, developers can achieve significant cost and time savings by resolving underlying root causes that simultaneously fix multiple flaky tests, rather than inefficiently debugging and repairing them in isolation."*
- **This is the empirical warrant for the whole step**: a raw count of failing tests systematically OVERSTATES the number of distinct problems. 17 failures is not 17 problems.

### 17. Lam, Oei, Shi, Marinov, Xie, "iDFlakies: A Framework for Detecting and Partially Classifying Flaky Tests" (ICST 2019)
http://mir.cs.illinois.edu/winglam/publications/2019/LamETAL19iDFlakies.pdf -- accessed 2026-08-11, `curl` + `pypdf` full-text extraction (11 pp., 65,320 chars; sanctioned step-3 of the PDF chain), peer-reviewed (tier 1)
- The requirement this whole step defends, verbatim from the abstract: *"A desirable requirement for regression testing is that a test failure reliably indicates a problem in the code under test and not a false alarm from the test code"*.
- The canonical OD/NOD split, verbatim: *"dataset of 422 flaky tests, with 50.5% order-dependent and 49.5%"* non-order-dependent.
- Verbatim scope limit: *"our tool does not further classify the NOD tests into more precise causes of flaky tests"* -- automated classification stops at OD/NOD; finer root cause is human work.

### 18. Yaraghi, Holden, Kahani, Briand, "Automated Test Case Repair Using Language Models" (arXiv 2401.06765) -- [ADVERSARIAL on the 'just fix the test' reflex]
https://arxiv.org/pdf/2401.06765 -- accessed 2026-08-11, `curl` + `pypdf` (45 pp., 233,161 chars), preprint (tier 1)
- Prevalence of the *broken* (not flaky) class, verbatim: *"Two recent analyses of test case failures of 211 Apache software foundation projects and 61 open-source projects reported that broken test cases account for 14% and 22% of failures, respectively."*
- The definition that separates this class from flakiness, verbatim: *"false alarms resulting from broken test cases, which are failures caused by the test code rather than the SUT"*.
- **The measured risk of 'fixing' a test**, verbatim: the metric *"Plausible Repair Accuracy (PR) ... evaluates the percentage of test cases that contain at least one repair candidate that compiles successfully and passes on the updated SUT version"*, and their best model scores *"66.1% exact match accuracy and 80.0% plausible repair accuracy"*. **A 13.9-point gap between "passes" and "is right."** A test edit that turns red to green is, by construction, "plausible"; plausibility is not correctness. This is the quantified form of the hazard in this step's own scope discipline ("A green suite bought by weakening assertions is worse than 26 honest failures").

### 19. Imtiaz et al., "A Systematic Literature Review of Test Breakage Prevention and Repair Techniques" (arXiv 1909.10750)
https://arxiv.org/pdf/1909.10750 -- accessed 2026-08-11, `curl` + `pypdf` (43 pp., 107,738 chars), preprint (tier 1)
- **The three-way classification this step needs, verbatim**: *"The existing literature classifies the regression test suite as usable, unusable and obsolete test scripts. The usable test cases conform to the existing functionality because they are not affected by the changes made in the evolved (modified) version of SUT. The unusable/broken test cases contain at least one statement that cannot be executed successfully. Such un-executable statement(s) may break the whole test case but the test case can be 'fixed' by applying repairing transformations to the test case implementation. Obsolete test cases fail to execute on the updated version and are not repairable, for example, they correspond to functionality that has been removed from the SUT."*
  -> **usable / broken-but-repairable / obsolete** is exactly the fix-vs-fix-vs-delete trichotomy, and note the criterion is *whether the behaviour the test encodes still exists*, not whether the test is red.
- Against reflexive deletion, verbatim: *"Discarding broken test cases after modifications highly affect the quality of the regression test suite... Even small modifications can lead to a large number of broken test cases, in some cases up to 74% of the test suite. Discarding the broken test cases therefore leads to a significant increase in the cost of testing and may reduce the quality of the test suite."*

### 20. Hashemi, Tahir, Rasheed, "An Empirical Study of Flaky Tests in JavaScript" (arXiv 2207.01047, ICSME 2022)
https://ar5iv.labs.arxiv.org/html/2207.01047 -- accessed 2026-08-11, WebFetch (ar5iv), peer-reviewed (tier 1)
- Distribution over 358 commits: Concurrency 20.7%, Async Wait 19.6%, **OS 18.4%**, Network 12.6%, **Platform 10.3%**, UI 5.9%, Hardware 4.7%, Time 3.4%, Resource Leak 2.8%.
- **Directly contradicts the Java-centric prior**, verbatim: *"Unlike in previous studies [Luo et al.], which identify test-order dependency as one of the key causes of flakiness, we found only very few flaky test commits"* from ordering. -> the OD share is **ecosystem-specific**, not universal. (pyfinagent measures OD = 0/17; see the internal half.)
- **Measured dispositions**: Fix 295 (82%), Improve 23 (7%), **Skip/Disable 26 (7%)**, **Quarantine 8 (2%)**, **Remove 6 (2%)**. Deletion is the rarest disposition in practice.

### 21. Gruber, Fraser, "Practical Flaky Test Prediction using Common Code Evolution and Test History Data" (arXiv 2302.09330)
https://ar5iv.labs.arxiv.org/html/2302.09330 -- accessed 2026-08-11, WebFetch (ar5iv), preprint (tier 1)
- Read in full; **thin on this step's question** and recorded as such rather than inflated. Only quarantine-relevant line, verbatim: detection *"builds the necessary foundation for any form of systematic response to flakiness, such as test quarantining or automated debugging"* -- i.e. quarantine PRESUPPOSES classification. No duration/ownership/cost data.

### Not obtainable (recorded, not hidden)
- **Luo, Hariri, Eloussi, Marinov, "An Empirical Analysis of Flaky Tests" (FSE 2014)** -- the canonical reference named in this step's objective. **Could not be read in full: it has no open-access copy.** Verified via the OpenAlex API (`api.openalex.org/works?filter=title.search:An Empirical Analysis of Flaky Tests`), which returns `best_oa_location: null` and only the paywalled `doi.org/10.1145/2635868.2635920`; four mirror URLs (`mir.cs.illinois.edu`, `cs.gmu.edu/~winglam`, `cs.cornell.edu`, `people.engr.ncsu.edu`) all returned HTML 404s, not PDFs. **NOT counted toward the gate.** Its taxonomy and headline distribution are nevertheless established here by THREE independently-read-in-full sources that report it (Eck et al. #2, the SAP HANA study #4, and Ahmad et al. #14: **async wait 45%, concurrency 20%, test order dependency 12%**), so the canonical content is corroborated rather than asserted.
- **arXiv 2107.02048 -- fetched, inspected, DISCARDED.** It is a physics paper ("Classical threshold law for the formation of van der Waals molecules"), not the Parry et al. flaky-test survey; the arXiv id was my guess and it was wrong. Recorded because a discarded mis-fetch is evidence of the check, and citing it would have been fabrication.

---

# INTERNAL HALF -- MEASURED, not inferred

## Measurement provenance

| Item | Value |
|---|---|
| Command | `python -m pytest backend/tests/ -q -p no:randomly` (venv activated) |
| Tree | live working tree `/Users/ford/.openclaw/workspace/pyfinagent`, branch `main`, HEAD `5759914c` (dirty: 5 modified + 4 untracked at run start) |
| Run window | 2026-08-11 13:14 -> 13:21 local, **400.92s** |
| **Result** | **`17 failed, 3417 passed, 12 skipped, 5 xfailed, 1 xpassed, 48 warnings`** |
| Raw log | `/private/tmp/claude-501/-Users-ford--openclaw-workspace-pyfinagent/75941db6-.../scratchpad/pytest_86_5.txt` |

**The step title's "26" does not reproduce; I measure 17, independently confirming the
peer's two measurements.** Reconciling the two figures: the step's own audit_basis
records the 26 baseline as `26 failed, 3017 passed` (2026-08-08). I measure `17 failed,
**3417** passed` -- **+400 passing tests**. The populations are not comparable: 400 tests
were added between the baselines, at least one of the 26 (`test_book_safety_69.py::
test_valid_nav_still_breaches`) was fixed by step 85.5.1, and several more were addressed
by 86.3's egress guard. **The 26 is not "wrong", it is stale.** Do not treat 17 as a
correction of 26; treat it as a later measurement of a moving population, and say so.

### The measurement did NOT touch live state -- proven, not asserted

`handoff/kill_switch_audit.jsonl` after the full run: **mtime `2026-08-10 21:15`**
(i.e. ~16 hours BEFORE the run started at 13:14 on 08-11), **66 lines**,
`sha256 = ab7324ebf501e3d3886e62a5d8fd2ed4f01f675849702b6553a4df691aab455f`, newest row
`sod_snapshot 2026-08-10T19:15:08Z`. **Zero rows appended; file untouched.** The
phase-86.3 root-`conftest.py` guard held on a live tree with the backend up. (Caveat
stated honestly: this is an after-reading plus an mtime that predates the run, not a
captured before/after pair -- 86.5's criterion 5 wants the pair, so the executor must
capture the "before" hash explicitly.)

## Root-cause classification of all 17 (grouped by CAUSE, not by file)

Signature = the exception type + the assertion actually produced, transcribed from the run.

### Group A -- ENVIRONMENT-RESOLVED CONFIG vs CODE DEFAULT (4 tests). Highest-value group.
The tests assert a **code default**; `Settings()` resolves the value from `backend/.env`,
so they measure an **operator-set environment override** instead.

- `backend/config/settings.py:46` `paper_data_integrity_enabled: bool = Field(False, ...)`
- `backend/config/settings.py:342` `paper_risk_judge_reject_binding: bool = Field(False, ...)`
- **but** `backend/.env:83` `PAPER_DATA_INTEGRITY_ENABLED=true`
- **and** `backend/.env:84` `PAPER_RISK_JUDGE_REJECT_BINDING=true`

| Node id | Signature |
|---|---|
| `test_phase_60_3_data_integrity.py::test_60_3_flag_defaults_off` | `AssertionError: assert True is False` on `Settings().paper_data_integrity_enabled` |
| `test_phase_57_1_reject_binding.py::test_reject_binding_main_path_off_emits_on_blocks` | `AssertionError: assert True is False` on `Settings().paper_risk_judge_reject_binding` |
| `test_phase_57_1_reject_binding.py::test_reject_binding_swap_path_off_emits_on_blocks` | `AssertionError: flag-OFF must preserve the (vulnerable) swap BUY; swap_buys={'TECH_NEW2'}` / `assert 'TECH_NEW1' in {'TECH_NEW2'}` |
| `test_phase_57_1_reject_binding.py::test_off_identity_prompts_are_verbatim_constants` | `AssertionError` comparing `_build_risk_judge_system(Settings())` against `_LITE_RISK_JUDGE_SYSTEM` (prompt diverges because the flag is ON) |

**Disposition: this is NOT an obsolete expectation and NOT a defect -- it is a test-isolation
defect (Fowler's "lack of isolation", source #1).** The invariant the tests mean to protect
(flag-OFF is byte-identical) is still live and still valuable. The remedy is to construct
`Settings(_env_file=None)` or override the two fields explicitly, so the test asserts the
**code default** it names. **Do NOT "fix" these by flipping the defaults or the .env** --
that would silently disarm two deliberately-armed money-path features. This group is the
clearest instance in the 17 of source #18's plausible-vs-correct hazard.

### Group B -- FROZEN SNAPSHOT OF A MOVING ARTIFACT: `.claude/masterplan.json` (5 tests)
All five read the LIVE masterplan (or a `git diff` against a frozen `BASELINE_COMMIT`) and
compare it to figures frozen at authoring time. Anchors:
`test_phase_75_17_verification_paths.py:81-99` (`_masterplan_at(ref)` -> `git show
{ref}:.claude/masterplan.json`), `:186-192` (`git diff BASELINE_COMMIT -- .claude/masterplan.json`),
`:203-204`; `test_phase_82_39_outcome_rebuild_query.py:364,396` (`json.loads(REPO/".claude/masterplan.json")`).

| Node id | Signature |
|---|---|
| `test_phase_75_17...::test_sweep_over_live_masterplan_is_clean` | `unexpected genuine defects remain: {'86.31': [{'path': '.claude/hooks/lib/qa_write_guard.py', 'class': 'never-existed', ...}] }` |
| `test_phase_75_19_preflight_calibration.py::test_live_masterplan_is_currently_clean` | `unexpected genuine residue: {'86.31': [... 'ref': '.claude/hooks/lib/qa_write_guard.py', 'class': 'never-existed' ...]}` |
| `test_phase_75_17...::test_sweep_shape_census_matches_the_corrected_figures` | `assert {'dict': 1057...} == {'dict': 720...}` |
| `test_phase_75_17...::test_masterplan_diff_touches_only_the_ten_sibling_insertions` | `non-comma-artifact removal found: '  "updated_at": "2026-07-23",'` |
| `test_phase_82_39...::test_the_sweeps_recall_limit_is_recorded_not_assumed` | `no OPEN step OWNS the live phantom-column defect the sweep cannot see ... assert []` |

**Two DISTINCT sub-causes -- do not merge them:**
- **B1 (data defect, 2 tests):** step 86.31's `verification.command` references
  `.claude/hooks/lib/qa_write_guard.py`, classified `never-existed`. The test is
  **correctly failing**: this is a real masterplan-data defect and the remedy is to fix
  step 86.31's verification path, not the test. Note the step's own audit_basis flags
  these two as *"a moving target and their reproduction must name a tree"*.
- **B2 (obsolete frozen figure / brittle oracle, 3 tests):** `dict: 720 -> 1057` is the
  masterplan simply having grown; the `updated_at` diff line and the `assert []` ownership
  probe are likewise frozen assertions over a live document. This is Eck et al.'s **"Too
  Restrictive Range"** class (source #2: *"valid output values are outside the assertion
  range considered at test design time"*) / the SAP HANA study's **Oracle-Brittleness**
  (source #4, 10% of 559). Remedy: re-derive the oracle or make it a measured DELTA --
  cf. auto-memory `feedback_immutable_criteria_must_be_green_able`.

### Group C -- ARTIFACT LIFECYCLE: the file moved, the test did not (2 tests)
| Node id | Signature |
|---|---|
| `test_phase_75_prompt_contracts.py::test_operator_decision_note_exists_with_token` | `FileNotFoundError: 'handoff/current/operator_decision_75.14_schema_extension.md'` |
| `test_phase_75_sre_ops.py::test_c1_runbook_and_operator_token_drafted` | `FileNotFoundError: '.../handoff/current/ops_rotate_runbook_75.11.md'` |

**Both files EXIST** -- at `handoff/archive/misc/operator_decision_75.14_schema_extension.md`
and `handoff/archive/misc/ops_rotate_runbook_75.11.md`. The `archive-handoff` hook relocated
them on step close, exactly as `.claude/rules/research-gate.md` "Handoff folder convention"
specifies. The tests hard-code `handoff/current/`. **Correctly-intended tests with a
path assumption invalidated by a documented, deliberate lifecycle.** Remedy: search
`current/` then `archive/**`. Zero production risk.

### Group D -- THE PROBE IS WRONG, THE CODE IS RIGHT (1 test). Read this one before acting.
`test_phase_75_sre_ops.py::test_c6_no_launchctl_bootstrap_executed_in_ops_scripts`
Signature: `AssertionError: reissue_cc_oauth_token.sh: "RELOAD_HINT_2='launchctl bootstrap
gui/$(id -u) ~/Library/LaunchAgents/com.pyfinagent.backend.plist'"`.

The scanner (`test_phase_75_sre_ops.py:360-368`) walks `OPS_DIR.glob("*.sh")` line by line,
skips only lines whose `strip()` starts with `#`, and asserts `"launchctl bootstrap" not in
stripped`. **It is blind to heredocs.** In `scripts/ops/reissue_cc_oauth_token.sh` the hit at
`:17` is a *variable assignment*, and its only expansion, `:117`, sits inside a `cat <<EOF`
that OPENS at `:110` -- it is **printed to the operator, never executed**. The script says so
in prose at `:107-109`: *"bootout/bootstrap is deliberately NOT automated: away-ops rail 9
reserves it for the operator, because a bootout that succeeds followed by a bootstrap that
fails leaves paper trading UNLOADED rather than merely stopped."*
**The production code is correct and the test's INTENT is correct; the implementation of the
check is wrong.** Remedy: make the scanner heredoc-aware (and quote-aware). **Editing the
script to satisfy this test would delete the operator's restart instructions** -- the
literal "fixing a test that was correctly-shaped but wrongly-implemented" trap
(auto-memory `feedback_a_red_check_may_indict_the_probe`).

### Group E -- OBSOLETE EXPECTATION, provably deliberate (1 test)
`test_phase_40_2_claude_code_v2_1_140_features.py::test_phase_40_2_settings_json_still_valid_json_after_edit`
Signature: `AssertionError: phase-29.2 effortLevel=xhigh invariant must survive phase-40.2
edit` / `assert 'max' == 'xhigh'`.
`.claude/settings.json:2` is `"effortLevel": "max"`. CLAUDE.md's effort-policy section records
the change verbatim: *"raised xhigh -> max 2026-08-04 by direct operator instruction"*.
**The production change is intentional, documented, and attributed.** This is source #19's
*obsolete* class -- the encoded behaviour no longer exists. Remedy: re-point the assertion at
the current invariant (settings.json is valid JSON and `effortLevel` is one of the accepted
tiers), or retire the pin. This is the ONLY one of the 17 that is a textbook obsolete
expectation, and it is cheap.

### Group F -- LIVE-SYSTEM-DEPENDENT, and the declared quarantine WAS NEVER WIRED (1 test)
`test_phase_23_2_6_sector_cap_emit.py::test_phase_23_2_6_backend_log_has_skipping_buy_evidence`
Signature: `AssertionError: no 'Skipping BUY' line in backend.log OR its newest archive
(researcher counted 24 on 2026-05-23); the cap gate may be silently disabled.` / `assert 0 >= 1`.
Its own docstring (`:230-247`) already states the diagnosis and the disposition:
*"asserts LIVE backend.log evidence on THIS machine ... genuine live-system state, not a code
defect -- **quarantined per the requires_live convention (pytest.ini:9)**; set
PYFINAGENT_LIVE_TESTS=1 to run."*
**But it is not quarantined.** There is no `@pytest.mark.requires_live` on it, and its only
escape hatch is `if not backend_log.exists() or size < 100: pytest.skip(...)` -- and
`backend.log` is present at **14,071,863 bytes**, so the skip never fires and the test runs
and fails. **A quarantine that exists only in a docstring is the silent-quarantine failure
mode the literature warns about** (source #14: *"7 out of 10 participants ... do not use any
notations to represent flaky tests"*). Note also the size-only guard is a weak oracle: it
proves a file exists, not that the log covers a window in which the gate could have fired.

### Group G -- LIVE BIGQUERY-DEPENDENT BEHAVIOURAL ASSERTIONS (2 tests)
`test_phase_82_48_outcome_write_schema.py` reaches a real client at `:74`, `:228`, `:272`
(`bigquery.Client(project="sunny-might-477607-p8")`), and its `@pytest.mark.skipif` at
`:210`/`:251` is *"gated on an EXPLICIT operator opt-out, not on whether"* credentials exist
-- so **these run against live BigQuery by default**.

| Node id | Signature |
|---|---|
| `...::test_the_fetch_supplies_every_field_the_write_REQUIRES` | `with no recommendation source the outcome must be skipped` / left has one extra row `{'analysis_date': ..., 'outcome': 'win', 'pnl': 5.5, ...} == []` |
| `...::test_write_really_persists_into_bigquery` | `assert 'UNKNOWN' == 'BUY'` |

The `'UNKNOWN' == 'BUY'` signature is the **outcome-vocabulary class** already parked as step
86.25 (auto-memory `project_phase86_25_outcome_vocabulary`: *"two seams failing in OPPOSITE
directions; S2 ungated cron persisted a fabricated 'SELL'"*). **Check 86.25 before filing a
new step for it** -- this is a likely duplicate, and the 86.5 criteria explicitly require
resolving overlap rather than double-filing.

### Group H -- GENUINE BEHAVIOURAL REGRESSION (1 test)
`test_portfolio_swap.py::test_swap_framework_fills_zero_buy_gap`
Signature: `AssertionError: Expected 2 swap SELLs, got 1` / `assert 1 == 2`; the emitted orders
are `SELL TECH0 (swap_for_higher_conviction)`, `BUY INDU_NEW (new_buy_signal)`,
`BUY TECH_NEW1 (swap_buy)` -- one SELL against two BUYs, i.e. an **unbalanced swap**. This is
a real assertion about production behaviour with no environmental explanation, and it is
money-path (swap churn / atomicity -- cf. steps 60.2, 70.3). **This is the one that most
plausibly encodes a real defect** and is the strongest candidate for the "fix now, in this
step" carve-out the 86.5 scope discipline allows.

## Cross-cutting measured properties of the 17

| Property | Measured result | Method |
|---|---|---|
| **Order-dependence** | **0 of 17.** All 17 fail identically when run alone (`17 failed in 10.44s`) as in-suite (`17 failed in 400.92s`). | re-ran the exact 17 node ids in isolation |
| **Clock / timezone dependence** | **0 of 17.** Identical 17 failures under `TZ=Pacific/Kiritimati` (UTC+14) and `TZ=Etc/GMT+12` (UTC-12) -- the maximal 26-hour spread. | two shifted-zone re-runs, precondition asserted (`17` node ids) |
| **Randomness** | 0 of 17 attributable; run made with `-p no:randomly`, and the isolated re-run reproduces exactly. | -- |
| **Non-determinism (flakiness) overall** | **0 of 17.** Every failure is 100% deterministic across 4 independent runs (full suite, isolated, TZ+14, TZ-12). **NONE of the 17 is a flaky test.** | 4 runs |
| **Touches live state** | Group G (2 tests) reach live BigQuery by default; Group F reads the live 14MB `backend.log`; Groups B/B1 read the live masterplan. **None writes to the paper-trading book or the kill switch** (proven above). | grep + audit-file hash |

**A methodological warning, stated rather than glossed:** the TZ result proves these
failures are not zone-sensitive *at this wall-clock instant*. Per auto-memory
`reference_fixed_offset_tz_fixture_is_hour_dependent`, a zone shift moves the local date
for only |offset|/24 of the day, so a full proof of clock-independence needs a **shifted
clock** (e.g. `libfaketime`), not just a shifted zone. Recorded as a residual gap.

---

## Rounds 9-12 -- additional external sources read IN FULL

### 22. "Can We Classify Flaky Tests Using Only Test Code? An LLM-Based Empirical Study" (arXiv 2602.05465v1, Feb 2026) -- [RECENCY]
https://arxiv.org/html/2602.05465v1 -- accessed 2026-08-11, WebFetch, preprint (tier 1)
- Adds a **detectability spectrum keyed to the fix type**, which is finer than "code-based
  detection has limits": detectable = *"change assertion"* and *"change data structure"*
  (rating 2.2-2.7, "Likely"); undetectable = *"reset variable"* and *"reorder parameters"*
  (3.9-4.2, "Unlikely"). Verbatim: *"Whether humans consider themselves capable of
  identifying a test as flaky based on the test code depends on the type of flakiness."*
- Verbatim on the invisible class: *"flakiness may be caused by side effects of test utility
  functions whose functionality is not visible in the test code"*.
- A triage principle, verbatim: *"test-only issues can be easily identified and fixed by
  developers, more sophisticated issues can be challenging to debug and fix, as they often
  rely on more complex program states"*.

### 23. "The Vocabulary of Flaky Tests in the Context of SAP HANA" (arXiv 2602.23957v1, Feb 2026) -- [RECENCY]
https://arxiv.org/html/2602.23957v1 -- accessed 2026-08-11, WebFetch, preprint (tier 1)
- Adds a root cause absent from Luo/Eck: **Fixed Timeout**, at **17% of cases (8/47)** in
  their manual commit analysis -- ahead of concurrency at 15%.
- **The actionability finding, which is the most transferable one here**, verbatim: they
  decline to deploy a 99%-F1 model because *"the results would not be actionable for
  developers. They would just get a probability value that their change can lead to
  flakiness, but no clear instructions what to improve."* -- a classification that does not
  name a remedy is not a triage.

### 24. GitLab Handbook, "Test Quarantine Process" -- the missing operational answer to objective (c)
https://handbook.gitlab.com/handbook/engineering/testing/quarantine-process/ -- accessed 2026-08-11, `curl` + tag-strip (the page is JS-rendered; plain WebFetch returned navigation chrome only -- cf. auto-memory `feedback_gcloud_docs_fetch`), official engineering policy (tier 2)

This is the only source found that specifies a **complete, enforced, time-boxed, owned,
non-silent quarantine**. Verbatim, in order:

- Purpose + the non-negotiable temporariness: *"The quarantine process helps maintain
  pipeline stability by temporarily removing flaky or broken tests from CI execution while
  preserving them to be fixed in the future... **Quarantining a test should be temporary:
  tests must be fixed, removed, or moved to a lower test level.**"*
- **Four dispositions, decided explicitly** (not defaulted into): *"**Fix immediately**: If
  the root cause is clear and a fix is known. **Delete the test**: If it's low-value or
  redundant. **Convert to lower level**: If it can be tested more reliably at unit or
  integration level. **Quarantine**: If investigation or fix will take longer than the
  required response time."*
- **The schedule -- quarantine is a countdown, not a parking lot:**

  | Phase | Duration | Action required |
  |---|---|---|
  | Fast quarantine | **3 days maximum** | Fix, remove, or convert to long-term |
  | Long-term quarantine | **3 months maximum** | Investigation and resolution |
  | Deletion warning | **1 week** | Final opportunity to resolve |
  | Automatic deletion | **After 3 months** | Test permanently removed |

- **Ownership + escalation are mechanised**: quarantine MRs *"are created automatically for
  consistently failing flaky tests"* and assigned *"to Engineering Managers based on
  feature_category metadata"*; *"If no action is taken: The merge request is approved by the
  owning team (per feature_category) after the urgency timeline expires. The test enters
  long-term quarantine. **The 3-month deletion countdown begins.**"*
- **Owner duties, verbatim**: *"Investigate the root cause. **Update the issue with progress
  weekly.** Resolve or remove within 3 months."*
- **A quarantine requires an issue with evidence** -- prerequisites: *"The correct group::*
  label... A link to the failing pipeline or job. **The stack trace and failure pattern.**
  The ~\"failure::flaky-test\" label."*
- **Anti-silent-quarantine by construction**: *"The fast quarantine file is automatically
  cleared every Sunday at 10:00 AM UTC via scheduled pipeline"* -- a fast quarantine that
  nobody follows up on **expires and the test goes red again**. Quarantine decays to
  visible, not to permanent.
- **Dequarantine has an evidentiary bar**: *"The test passes consistently (more than 100
  local runs), and the root cause is identified and fixed."*

### Round 11 -- DRY (recorded honestly)
Two genuine gap-closing attempts, **zero** new read-in-full findings: (a)
`https://docs.gitlab.com/development/testing_guide/flaky_tests/` rendered but only
**pointed** at the handbook (no policy content of its own); (b)
`https://arxiv.org/html/2605.11482v1` (NeuroFlake) returned **HTTP 404**.
Also attempted and failed in this window: the Google 2016 post body via
`https://web.archive.org/web/2020/https://testing.googleblog.com/...` -- WebFetch is
blocked from `web.archive.org` in this environment.

### 25. "Reproducible Automated Program Repair Is Hard -- Experiences With the Defects4J Dataset" (arXiv 2604.26674v1, Apr 2026) -- [ADVERSARIAL / RECENCY]
https://arxiv.org/html/2604.26674v1 -- accessed 2026-08-11, WebFetch, preprint (tier 1)
- **The hardest number in this brief for the "just make it green" reflex:** *"9.0% of workable
  defects (59 of 655) pass their test suite when a single statement is deleted"*, despite the
  human patch being a substantive correction. Re-analysing a prior tool's evaluation, the
  *"fix-rate drops from 13.7% to 7.9% when excluding these trivial defects."*
  **A green test is compatible with the code being deleted.** Passing is a necessary, not a
  sufficient, condition for correct.
- Introduces a **"workability"** criterion stricter than reproducibility: tests must *"produce
  consistent results for the same variants"* and allow *"individual test cases to be executed
  in any order"* with identical outcomes -- and **21.6% of defects fail it**, *"primarily
  because running tests individually yields different results than executing the full suite
  (126 defects affected)."* This is the exact property I measured on pyfinagent's 17 (they
  pass it: identical in-suite and isolated).

### Round 13 -- DRY (recorded honestly)
Two full reads, **zero** new findings. `arXiv 2608.06535v1` (flaky tests in cyber-physical
systems): verbatim *"no new root-cause categories beyond existing taxonomy"* -- a
domain-specific detection method, and it *"artificially injects noise rather than analyzing
real flaky test cases."* `arXiv 2601.08998v1` (flakiness of LLM-generated DBMS tests):
*"reuses the existing Parry et al. (2021) taxonomy entirely"*; its one notable datum (63%
unordered-collection, 72/115) is a prevalence shift in a category already defined, and it
gives *"no explicit guidance"* on fix vs quarantine vs delete.

### Round 14 -- one new finding (#25 above) + one dry
`arXiv 2603.09029v1` (flaky tests in quantum software) is dry: the authors state *"We find
that we do not need to alter these classes for our extended dataset"*, and the fetch
concludes it makes *"no new contributions to flakiness taxonomy, classification criteria, or
disposition strategies"* -- only a domain prevalence shift (randomness/PRNG 19.2%).

### 26. "Detecting Flakiness in Quantum Software: A Dynamic Testing Approach" (arXiv 2512.18088v3, Dec 2025) -- [RECENCY]
https://arxiv.org/html/2512.18088v3 -- accessed 2026-08-11, WebFetch, preprint (tier 1)
- No new categories (*"use[s] the taxonomy ... proposed by Sivaloganathan et al."*), but two
  transferable **disposition** ideas: a **cross-release registry** -- *"Maintain a persistent
  registry of flaky test identifiers, together with their empirical failure rates, affected
  releases, suspected cause category, and applied fix pattern"* -- and **recurrence-aware
  tiers** (Rarely Flaky <=15% of releases / Persistently Flaky >=70% / Intermittently Flaky).
- The caution that applies to any "we fixed it" claim, verbatim: *"Zero observed failures
  should be read as 'no flakiness detected at this budget,' not necessarily as 'flakiness
  eliminated'."*

### 27. Besker, Martini et al. / "The Broken Windows Theory Applies to Technical Debt" (arXiv 2209.01549v3) -- the EMPIRICAL warrant for objective (b)
https://arxiv.org/html/2209.01549v3 -- accessed 2026-08-11, WebFetch, peer-reviewed (tier 1)
- Design: *"29 developers of varying experience levels completing system extension tasks in
  already existing systems with high or low TD density"*, repeated measures, randomised.
- Measured effects: logic reuse **beta = -1.79 (95% CI -3.12..-0.54)**; variable naming
  **beta = 2.48 (95% CI 1.41..3.64)** -- *"high TD systems strongly predicted poor variable
  naming in new code"*; SonarQube introduced issues **beta = -0.80 (95% CI -1.53..-0.08)**,
  *"high debt versions producing more defects."*
- Conclusion, verbatim: *"Three separate significant results along with a validating
  qualitative result combine to form substantial evidence of the BWT's existence in software
  engineering contexts"*; *"existing TD can have a major impact on developers propensity to
  introduce new TD of various types during development."*
- **This upgrades Fowler's broken-window claim from a well-argued assertion to a measured
  effect** -- a standing red suite is not merely untidy; pre-existing degradation
  demonstrably degrades what is built next to it.

### 28. "Broken Windows: Exploring the Applicability of a Controversial Theory on Code Quality" (arXiv 2410.13480v1, Oct 2024) -- [ADVERSARIAL]
https://arxiv.org/html/2410.13480v1 -- accessed 2026-08-11, WebFetch, preprint (tier 1)
- **The disagreeing source the deep/audit discipline requires.** Verdict: qualified support,
  not endorsement. Confirms *"The quality of an existing, initial, code body and its
  subsequent evolution are related"* and *"about 40% of the files we examined exhibit
  significant autocorrelations for lags up to ten"*.
- **But** it fails to reproduce the effect where the theory most needs it: *"We were not able
  to detect a statistical significance in the Java smells metrics of developers' commits"*,
  and *"Surprisingly ... a file's style consistency ... is associated with the developers'
  behavior at a lesser extent"* than predicted.
- Crucially, verbatim: *"we have been careful not to propose any causal relationships"* --
  only *"statistical relations"*. **So: do not argue for this step's remediation on causal
  broken-windows grounds alone.** The stronger, non-contested arguments are the direct ones
  -- Fowler's triage-discipline erosion (#1), Google's 1% value-loss threshold (#11), and the
  fact that a permanently-red suite cannot answer "did my change break anything?"

### 29. "SCOUT: A Practical Framework for Flaky Failure Triage in Distributed Database Continuous Integration" (arXiv 2603.23054v1, Mar 2026) -- [RECENCY]
https://arxiv.org/html/2603.23054v1 -- accessed 2026-08-11, WebFetch, preprint (tier 1)
- **Scope-limiting finding, recorded because the title over-promises for this step:** SCOUT is
  *"not a root-cause triage framework"* -- it is a **rerun-vs-escalate binary**
  (`p̂ >= tau* -> rerun`, else escalate), with `tau* = (c_auto + c_fp) / (c_fp + c_fn)`.
  *"SCOUT does not classify root causes or provide disposition rules beyond the
  rerun/escalate binary choice."*
- One idea does transfer: it **excludes post-failure artefacts** (*"error strings, failure
  logs, and future rerun information"*) from its main protocol to avoid leakage -- the same
  hazard 2607.09345 (#3) measured. For a human triage, by contrast, those artefacts are
  precisely the evidence you want; the lesson is that they must not be used to train or to
  justify an automated label.

### Round 15 -- one new finding (#26) + one dry
`arXiv 2511.18854v1` (LLM-assisted git bisect) is dry for this step: it localises which commit
introduced a regression but offers *"No method to determine if a failing test encodes stale
assumptions"* and *"No root-cause categorization for 'test is wrong' vs. 'code is wrong'"*.
(Noted as a possible TOOL for the executor -- bisecting a Group-B/E failure to the commit that
changed the behaviour is exactly how you separate defect from obsolete expectation -- but it
is not a finding about triage.)

### Round 17 -- DRY
`https://arxiv.org/html/2601.22264v2` (predicting intermittent job-failure categories with
few-shot LLMs) read in full: *"No new root-cause categories emerge. No new evidence
requirements for classification are identified."* It re-labels a pre-existing 46-category
TELUS catalogue. `https://arxiv.org/html/2401.15788v1` returned HTTP 404 (no HTML render).
**Zero new findings.**

### Round 18 -- DRY (probed the strongest remaining seam, not an easy one)
Both papers target objective (d) head-on and both **decline the question**:
- `https://arxiv.org/html/2411.11033v1` (REACCEPT co-evolution): its identification predicate
  is `Identify(p,p',t) = 1 if t != t' i.e., t must be updated`, which **assumes the production
  change is already correct**. Verbatim: *"There is no decision logic evaluating whether the
  production change itself might be buggy"*, and no measured data on *"instances where an
  'updated' test masks a real production defect."*
- `https://arxiv.org/html/2407.03625v2` (Synter): *"No new decision procedures are
  introduced. The authors do not establish criteria for determining whether a broken test
  indicates an actual defect versus obsolescence."* It assumes obsolescence and optimises
  repair accuracy (90.4%).

**Zero new findings. Two consecutive dry rounds -> `coverage.dry = true`.**

**This dryness is itself the headline negative result:** the automated-repair literature
systematically *presupposes* the answer to the only question that matters here -- is the test
wrong, or is the code wrong? **No source found in 18 rounds offers a mechanical decision
procedure for that.** It is human work, and the evidence it needs is named below.

---

# SYNTHESIS

## Recency scan (2024-2026) -- MANDATORY SECTION

**Performed.** Method: `WebSearch` was unavailable (session budget exhausted before spawn), so
the recency pass was run as an **arXiv Atom API sweep via curl** --
`export.arxiv.org/api/query?search_query=abs:"flaky test"&sortBy=submittedDate&sortOrder=descending`
(80 results) plus five targeted relevance queries (`flaky AND triage`, `test suite AND
technical debt`, `obsolete test`, `broken window AND software`, `test smell AND assertion`).

**Result: 16 of the 37 sources read in full are from the 2024-2026 window, and they
MATERIALLY change the guidance.** Specifically:

1. **The count-of-failures metric is superseded.** Systemic Flakiness (2025, #16) measures
   *"606 (75%) belong to a cluster"*, mean cluster **13.5**. Counting failing tests
   systematically overstates the number of distinct problems. This did not exist in the
   2014-2019 canon and it directly reshapes how 86.5 should be scoped.
2. **Code-only classification is refuted (2026, #3).** Post-leakage-correction flaky-class
   F1 collapses to **0.035-0.07**, and **58%** of failures needed execution evidence. The
   2014-era practice of reading the test to categorise it is not defensible.
3. **The taxonomy has grown well past Luo's 10.** SAP HANA (2026) runs **16 categories** and
   adds **Fixed Timeout at 17%**; JS work adds OS 18.4% / Platform 10.3%; GitHub Actions work
   shows flaky *tests* are only **64.99%** of flaky *failures*.
4. **The OD share is ecosystem-specific, not universal.** iDFlakies (2019) measured 50.5%
   order-dependent; the JS study (2022) found *"only very few"*. pyfinagent measures **0 of
   17**. Do not import a prior distribution.
5. **"Passing" was quantified as a weak oracle (2026, #25):** *"9.0% of workable defects (59
   of 655) pass their test suite when a single statement is deleted."*
6. **The broken-windows premise was tested both ways (2022 / 2024, #27 / #28)** -- confirmed
   with measured effect sizes, then partially failed to replicate, with the replication authors
   explicitly refusing a causal claim.

**No 2024-2026 source supersedes Fowler's quarantine mechanics or Google's 1% threshold**;
those remain the operative guidance and are corroborated, not contradicted, by GitLab's
current published process (#24).

## Key findings

1. **17 failures is not 17 problems.** *"Of the 810 flaky tests, 606 (75%) belong to a
   cluster"*, mean cluster size **13.5** (Fatima et al. 2025, https://arxiv.org/html/2504.16777v1).
   Measured here: the 17 collapse to **8 root-cause groups**, and 2 of those 8 (A and C)
   cover 6 tests between them.
2. **A permanently-red suite destroys triage discipline, not just tidiness.** *"Once that
   discipline is lost, then a failure in the healthy deterministic tests will get ignored
   too. At that point you've lost the whole game"* (Fowler,
   https://martinfowler.com/articles/nonDeterminism.html). Google puts a number on it:
   *"as you approach 1% flakiness, the tests begin to lose value"*
   (https://abseil.io/resources/swe-book/html/ch11.html). **pyfinagent is at 17/3434 = 0.495%
   -- below Google's 1% value-loss line but 3.3x their 0.15% operating rate.** The suite is
   not yet worthless; it is in the band where it stops being worthless only if this step happens.
3. **Root cause is majority-unrecoverable from source reading.** *"for 42% of them...we could
   attribute a cause from the test code and CI log. For the remaining 58%...diagnosis required
   execution evidence"* (https://arxiv.org/html/2607.09345v1). **Therefore the triage must key
   on the captured exception type + assertion, which is what this brief did, and 86.5's
   criterion 3 ("a MEASURED signature ... not filename similarity") is exactly right.**
4. **Quarantine is a countdown with an owner, or it is not a quarantine.** Fowler: *"only allow
   8 tests in quarantine"* / *"no longer than a week"*. GitLab operationalises it: **3 days
   fast / 3 months long-term / 1 week deletion warning / automatic deletion**, MRs
   *"created automatically ... assigned to Engineering Managers based on feature_category"*,
   owner must *"Update the issue with progress weekly"*, and the fast-quarantine file *"is
   automatically cleared every Sunday"* so neglect makes the test go **red again**
   (https://handbook.gitlab.com/handbook/engineering/testing/quarantine-process/).
5. **Deletion is legitimate but rare, and reflexive deletion is costly.** Measured
   dispositions: Fix 82%, Improve 7%, Skip/Disable 7%, Quarantine 2%, **Remove 2%**
   (https://ar5iv.labs.arxiv.org/html/2207.01047). And *"Discarding broken test cases after
   modifications highly affect the quality of the regression test suite"*
   (https://arxiv.org/pdf/1909.10750). GitLab names the one clean deletion criterion:
   *"Delete the test: If it's low-value or redundant."*
6. **"Green" is a weak oracle, and the gap is measured.** Best-model test repair scores
   *"66.1% exact match accuracy and 80.0% plausible repair accuracy"* -- a **13.9-point** gap
   where the repair passes but is not the ground truth (https://arxiv.org/pdf/2401.06765); and
   *"9.0% of workable defects (59 of 655) pass their test suite when a single statement is
   deleted"* (https://arxiv.org/html/2604.26674v1). **This is the quantitative backing for
   86.5's own ban on mass-editing tests to green.**
7. **The obsolete/broken/usable trichotomy is the disposition rule.** *"usable ... unusable/broken
   ... Obsolete test cases fail to execute on the updated version and are not repairable, for
   example, they correspond to functionality that has been removed from the SUT"*
   (https://arxiv.org/pdf/1909.10750). The discriminating question is **"does the behaviour
   this test encodes still exist, and was its removal intended?"** -- answered by evidence
   (a commit, a config change, an operator decision), never by the test's colour.
8. **Undeclared quarantine is the industry default and the thing to avoid.** *"7 out of 10
   participants shared that they do not use any notations to represent flaky tests"*
   (https://ar5iv.labs.arxiv.org/html/1906.00673). pyfinagent has a live instance: Group F's
   test documents its own quarantine in a **docstring** and is not actually marked.

## Consensus vs debate (external)

**Consensus:** (a) reruns are a workaround, not a remedy; (b) quarantine must be temporary,
owned and visible; (c) deletion is a last resort reserved for low-value/redundant tests;
(d) execution evidence, not test source, is what classifies a failure; (e) concurrency +
async-wait + timeout dominate *flakiness* (which is NOT pyfinagent's problem here).

**Genuine debate:**
- **Does prior degradation cause more degradation?** 2209.01549 says yes with effect sizes;
  2410.13480 partially fails to replicate it and *"careful not to propose any causal
  relationships"*. **Resolution for this step: do not lean on broken-windows causality.**
- **Is order-dependence a dominant class?** 50.5% (iDFlakies, Java) vs *"only very few"* (JS)
  vs **0/17** (measured here). Ecosystem-specific; must be measured, never assumed.
- **Is test-level binary "flaky/not" the right abstraction?** Meta and 2607.09345 both say no
  (*"not whether a particular test is flaky, but how flaky it is"*); the classifier literature
  keeps using it anyway.

## Pitfalls (from the literature, mapped to this step's specific hazards)

1. **Grouping by file/module.** The step already warns of this; the literature backs it --
   co-occurrence clusters cross files, and one file holds two causes (measured here:
   `test_phase_75_17_verification_paths.py` holds **both** B1 and B2).
2. **Editing the test to get green.** 13.9-pt plausible-vs-correct gap; 9.0% of suites accept
   a statement deletion. **Group A is the live trap**: "fixing" it by flipping a default or a
   `.env` line would silently disarm two armed money-path features.
3. **Fixing the subject when the probe is wrong.** Group D: acting on `test_c6` would delete
   the operator's documented restart instructions from `reissue_cc_oauth_token.sh`.
4. **Silent quarantine.** Group F is already an instance. Any quarantine filed by this step
   must be `xfail(strict=True, raises=<measured type>)` -- which still RUNS the test and turns
   an unexpected pass into a suite failure -- rather than `skip`, which does not execute at all.
5. **Freezing a criterion that is already red.** Three of the 17 (B2) are frozen figures over a
   growing document; any new step must not repeat that. Cf.
   `feedback_immutable_criteria_must_be_green_able`.
6. **Assuming zero means fixed.** *"Zero observed failures should be read as 'no flakiness
   detected at this budget'"* (https://arxiv.org/html/2512.18088v3).

## Application to pyfinagent -- mapping findings to file:line anchors

| Group | n | Root cause | Literature class | Disposition indicated | Anchors |
|---|---|---|---|---|---|
| **A** | 4 | Test reads env-resolved `Settings()`, asserts code default | Lack of isolation (Fowler); Configuration (SAP HANA) | **FIX THE TEST** (`Settings(_env_file=None)`). Never flip the flag. | `backend/config/settings.py:46,342`; `backend/.env:83,84` |
| **B1** | 2 | Real masterplan-data defect (86.31 `qa_write_guard.py`, `never-existed`) | Genuine defect -- test correctly failing | **FIX THE DATA** (step 86.31's verification path) | `test_phase_75_17_verification_paths.py:203`; `test_phase_75_19_preflight_calibration.py` |
| **B2** | 3 | Frozen census/diff oracle over a growing `masterplan.json` | Too Restrictive Range (Eck); Oracle-Brittleness (SAP HANA) | **RE-DERIVE THE ORACLE** as a delta, or retire | `test_phase_75_17_verification_paths.py:81-99,186-192`; `test_phase_82_39_outcome_rebuild_query.py:364,396` |
| **C** | 2 | Test hard-codes `handoff/current/`; `archive-handoff` hook moved the file | Environment/lifecycle | **FIX THE TEST** (search `current/` then `archive/**`) | files live at `handoff/archive/misc/{ops_rotate_runbook_75.11,operator_decision_75.14_schema_extension}.md` |
| **D** | 1 | Scanner is heredoc-blind; the code is CORRECT | Broken probe, not broken code | **FIX THE CHECK** (heredoc-aware). Do NOT edit the script. | `test_phase_75_sre_ops.py:360-368` vs `scripts/ops/reissue_cc_oauth_token.sh:17,107-110,117` |
| **E** | 1 | `effortLevel` deliberately raised `xhigh`->`max` 2026-08-04 | **Obsolete expectation** (SLR "obsolete") | **UPDATE THE EXPECTATION** | `.claude/settings.json:2`; CLAUDE.md effort-policy section |
| **F** | 1 | Live 14MB `backend.log` has no `Skipping BUY` in window; declared quarantine NEVER WIRED | Environmental flakiness + silent quarantine | **WIRE THE QUARANTINE**: real `@pytest.mark.requires_live` per `pytest.ini:9` | `test_phase_23_2_6_sector_cap_emit.py:230-247` |
| **G** | 2 | Live BigQuery reads; `'UNKNOWN' == 'BUY'` outcome vocabulary | Genuine defect (probably) | **CHECK 86.25 FIRST** -- likely duplicate | `test_phase_82_48_outcome_write_schema.py:74,210,228,251,272` |
| **H** | 1 | Unbalanced swap: 1 SELL vs 2 BUYs | Genuine behavioural regression, money-path | **FIX THE CODE** -- the one carve-out 86.5 allows | `test_portfolio_swap.py::test_swap_framework_fills_zero_buy_gap` |

**Overlap resolution required by criterion 4:** the 86.3 / 36.28 live-kill-switch-coupling
class accounts for **0 of the 17** in the current measurement -- the 86.3 guard demonstrably
held (audit file untouched) and the six files named in 86.5's audit_basis as coupled to live
pause state (`test_64_3_currency_path`, `test_64_4_multi_market_e2e`,
`test_dod4_tier1_coverage_investment`, `test_phase_70_3_atomic_swap`, `test_price_tolerance_gate`,
`test_phase_70_4_gate_observability`) **are all green now**. That is the single largest part of
the 26 -> 17 reduction and it must be recorded as *already resolved*, not re-filed.

## Recommended step shape (research input only -- Main owns PLAN)

Eight groups, but **not eight steps**: A, C, D, E, F are small, self-contained,
test-side-only, and share one property (the test is wrong, the code is right). B1, G, H are
code/data-side. A defensible filing is **4-5 steps**, not 17 and not 8 -- consistent with
finding 1. Every quarantine filed must carry: the measured signature, an owner, a review date,
and `xfail(strict=True, raises=...)` rather than `skip`.

---

## Internal code inventory

| File | Anchor | Role | Status |
|---|---|---|---|
| `conftest.py` (repo root) | whole file | phase-86.3 mutating-HTTP egress guard to the live backend origin | ACTIVE, **verified holding** (audit file untouched by a full run) |
| `backend/tests/conftest.py` | `:1-60` | `PYFINAGENT_TEST_NO_BQ=1` + phase-82.58 slack.com egress guard | ACTIVE; chains under the root guard |
| `pytest.ini` | `:1-9` | Registers `requires_live` quarantine marker; comment already states *"Classification is by ROOT CAUSE"* | **Correct prior art, under-adopted** -- Group F declares it in prose but is not marked |
| `backend/config/settings.py` | `:46`, `:342` | Code defaults `False` for the two flags | Correct; tests misread them |
| `backend/.env` | `:83`, `:84` | `=true` overrides, operator-set | Correct; is what the tests actually measure |
| `.claude/settings.json` | `:2` | `"effortLevel": "max"` | Correct; Group E asserts the superseded `xhigh` |
| `scripts/ops/reissue_cc_oauth_token.sh` | `:17`, `:107-110`, `:117` | Prints (never runs) bootout/bootstrap, per away-ops rail 9 | **Correct**; Group D's check misreads a heredoc |
| `backend/tests/test_phase_75_sre_ops.py` | `:360-368` | `.sh` scanner, skips `#` only | **Defective probe** (heredoc-blind) |
| `backend/tests/test_phase_23_2_6_sector_cap_emit.py` | `:230-247` | Live-`backend.log` evidence assertion | Quarantine declared in docstring, **never wired**; size-only skip guard is a weak oracle |
| `backend/tests/test_phase_75_17_verification_paths.py` | `:81-99`, `:186-192`, `:203` | Masterplan sweeps + git-diff shape | Holds **two** distinct causes (B1 real defect, B2 stale oracle) |
| `backend/tests/test_phase_82_48_outcome_write_schema.py` | `:74`, `:210`, `:228`, `:251`, `:272` | Real `bigquery.Client`; skipif is an explicit operator opt-out | **Reaches live BigQuery by default** |
| `backend/tests/test_phase_82_39_outcome_rebuild_query.py` | `:364`, `:396` | Reads live masterplan | B2 class |
| `handoff/kill_switch_audit.jsonl` | 66 lines, `sha256 ab7324eb...`, mtime 2026-08-10 21:15 | Live kill-switch audit trail | **Untouched by the measurement** |
| `backend.log` | 14,071,863 bytes | Live log Group F greps | Present, so Group F's skip never fires |

---

## Source tables

### Read in full (37; counts toward the gate)
| # | URL | Kind | Fetched how |
|---|---|---|---|
| 1 | https://martinfowler.com/articles/nonDeterminism.html | blog (canonical) | WebFetch |
| 2 | https://ar5iv.labs.arxiv.org/html/1907.01466 | paper | WebFetch (ar5iv) |
| 3 | https://arxiv.org/html/2607.09345v1 | paper | WebFetch |
| 4 | https://arxiv.org/html/2602.03556v1 | paper | WebFetch |
| 5 | https://martinfowler.com/bliki/SelfTestingCode.html | blog | WebFetch |
| 6 | https://engineering.atspotify.com/2019/11/18/test-flakiness-methods-for-identifying-and-dealing-with-flaky-tests | industry | WebFetch |
| 7 | https://testing.googleblog.com/2017/04/where-do-our-flaky-tests-come-from.html | official blog | WebFetch |
| 8 | https://engineering.fb.com/2020/12/10/developer-tools/probabilistic-flakiness/ | industry | WebFetch |
| 9 | https://arxiv.org/html/2602.09311v2 | paper | WebFetch |
| 10 | https://arxiv.org/html/2602.19098v1 | paper | WebFetch |
| 11 | https://abseil.io/resources/swe-book/html/ch11.html | official book | WebFetch |
| 12 | https://arxiv.org/html/2605.21677v1 | paper | WebFetch |
| 13 | https://docs.pytest.org/en/stable/how-to/skipping.html | official docs | WebFetch |
| 14 | https://ar5iv.labs.arxiv.org/html/1906.00673 | paper | WebFetch (ar5iv) |
| 15 | https://arxiv.org/html/2602.02307v1 | paper | WebFetch |
| 16 | https://arxiv.org/html/2504.16777v1 | paper | WebFetch |
| 17 | http://mir.cs.illinois.edu/winglam/publications/2019/LamETAL19iDFlakies.pdf | paper (ICST'19) | curl + pypdf (11pp) |
| 18 | https://arxiv.org/pdf/2401.06765 | paper | curl + pypdf (45pp) |
| 19 | https://arxiv.org/pdf/1909.10750 | paper (SLR) | curl + pypdf (43pp) |
| 20 | https://ar5iv.labs.arxiv.org/html/2207.01047 | paper (ICSME'22) | WebFetch (ar5iv) |
| 21 | https://ar5iv.labs.arxiv.org/html/2302.09330 | paper | WebFetch (ar5iv) |
| 22 | https://arxiv.org/html/2602.05465v1 | paper | WebFetch |
| 23 | https://arxiv.org/html/2602.23957v1 | paper | WebFetch |
| 24 | https://handbook.gitlab.com/handbook/engineering/testing/quarantine-process/ | official policy | curl + tag-strip |
| 25 | https://arxiv.org/html/2604.26674v1 | paper | WebFetch |
| 26 | https://arxiv.org/html/2512.18088v3 | paper | WebFetch |
| 27 | https://arxiv.org/html/2209.01549v3 | paper | WebFetch |
| 28 | https://arxiv.org/html/2410.13480v1 | paper [ADVERSARIAL] | WebFetch |
| 29 | https://arxiv.org/html/2603.23054v1 | paper | WebFetch |
| 30 | https://arxiv.org/html/2601.22264v2 | paper | WebFetch (dry) |
| 31 | https://arxiv.org/html/2411.11033v1 | paper | WebFetch (dry) |
| 32 | https://arxiv.org/html/2407.03625v2 | paper | WebFetch (dry) |
| 33 | https://arxiv.org/html/2608.06535v1 | paper | WebFetch (dry) |
| 34 | https://arxiv.org/html/2601.08998v1 | paper | WebFetch (dry) |
| 35 | https://arxiv.org/html/2603.09029v1 | paper | WebFetch (dry) |
| 36 | https://arxiv.org/html/2511.18854v1 | paper | WebFetch (dry) |
| 37 | https://docs.gitlab.com/development/testing_guide/flaky_tests/ | official docs | WebFetch |

### Identified but snippet-only (21; does NOT count toward the gate)
| URL | Why not read in full |
|---|---|
| https://testing.googleblog.com/2016/05/flaky-tests-at-google-and-how-we.html | body never rendered (WebFetch + curl); comments only |
| https://doi.org/10.1145/2635868.2635920 | Luo et al. FSE 2014 -- **paywalled, no OA copy** (OpenAlex `best_oa_location: null`) |
| https://arxiv.org/abs/2605.11482 | HTML 404 (NeuroFlake) |
| https://arxiv.org/abs/2401.15788 | HTML 404 |
| https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/stvr.1791 | superseded by #14 (same authors/topic) |
| https://doi.org/10.1109/icse43902.2021.00141 | UI-flaky-tests; UI is out of scope here |
| https://arxiv.org/abs/2107.02048 | **mis-fetch, discarded** (physics paper, not the Parry survey) |
| https://arxiv.org/abs/2606.20243 | Phoenix multi-agent issue resolution -- adjacent |
| https://arxiv.org/abs/2604.03035 | coding-agent evaluation -- adjacent |
| https://arxiv.org/abs/2607.12068 | agent-generated test quality -- adjacent |
| https://arxiv.org/abs/2307.14733 | StubCoder -- mock repair, out of scope |
| https://arxiv.org/abs/1811.04122 | RL test prioritisation -- out of scope |
| https://arxiv.org/abs/2401.13407 | maintainability returns -- adjacent to #27 |
| https://arxiv.org/abs/2108.04639 | PyNose test-smell detector (Python) -- tooling |
| https://arxiv.org/abs/2504.07277 | agentic test-smell hunting -- tooling |
| https://arxiv.org/abs/2108.11781 | test smells as flakiness predictors -- superseded by #3/#22 |
| https://arxiv.org/abs/2207.05539 | assertion-roulette refactoring experiment |
| https://arxiv.org/abs/2303.04234 | test smells + students |
| https://arxiv.org/abs/2410.10628 | test smells in LLM-generated tests |
| https://web.archive.org/web/2020/https://testing.googleblog.com/2016/05/flaky-tests-at-google-and-how-we.html | WebFetch blocked from web.archive.org |
| https://api.openalex.org/works?filter=title.search:An%20Empirical%20Analysis%20of%20Flaky%20Tests | discovery API, not a source |

**Search-query composition (three-variant discipline, made visible):** current-year frontier =
the arXiv `sortBy=submittedDate` sweep (2026 hits #3, #4, #9, #10, #12, #15, #22, #23, #25, #29);
last-2-year = 2024-2025 hits (#16, #26, #28); year-less canonical = the un-dated topical fetches
(#1, #5, #7, #8, #11, #13, #17, #19, #20, #24) plus the five relevance-sorted (not date-sorted)
arXiv queries in round 16.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL -- **37** (floor is 5)
- [x] 10+ unique URLs total -- **58** (37 read-in-full + 21 snippet-only)
- [x] Recency scan (last 2 years) performed + reported -- dedicated section above; 16 of 37 in window
- [x] Full papers/pages read (not abstracts) -- ar5iv/native HTML, or curl+pypdf full text; the one abstract-only candidate (Luo 2014) is explicitly EXCLUDED from the count
- [x] file:line anchors for every internal claim -- see inventory + group tables

Adaptive coverage (audit-class):
- [x] Floor met, then 18 rounds run; **rounds 17 and 18 both DRY** (zero new read-in-full findings), K=2 satisfied -> `coverage.dry = true`

Soft checks:
- [x] Internal exploration covered every failing module (11 modules, 17 node ids, 4 independent runs)
- [x] Contradictions/consensus noted (broken-windows replication; OD share; test-level binary)
- [x] Claims cited per-claim with URL
- [ ] **Residual gap, stated:** clock-independence was proven only against shifted *zones*, not a
      shifted *clock*; and the kill-switch non-touch is an after-hash + pre-run mtime, not a
      captured before/after pair (86.5 criterion 5 requires the pair).

---

## Envelope verification pass -- 2026-08-11 (append-only; brief NOT rewritten)

The original run's envelope was lost when the Workflow rail dropped before returning.
This pass re-derived every self-reported figure **from the file itself**, mechanically.
No new research was performed; nothing above this line was modified.

| Field | Claimed | Re-measured | Rule used | Verdict |
|---|---|---|---|---|
| `external_sources_read_in_full` | 37 | **37** | rows matching `^\| *[0-9]+ *\|` in the read-in-full table (L858-894); 37 unique URLs, zero dupes | **CONFIRMED** |
| `snippet_only_sources` | 21 | **21** | rows matching `^\| *https?://` in the snippet table (L899-919); 21 unique URLs | **CONFIRMED** |
| `urls_collected` | 58 | **58** | union of both tables, `sort -u`; overlap between the two tables measured = **0** rows | **CONFIRMED** |
| `recency_scan_performed` | true | **true** | dedicated section present at L681, states method + result + 6 enumerated findings | **CONFIRMED** |
| recency in-window share | "16 of 37" | **21 of 37** | arXiv ID `YYMM` prefix in {24xx,25xx,26xx} over the read-in-full URLs | **CORRECTED (undercount)** |
| `coverage.rounds` | 18 | **18** | round headings L58/L198/L482/L548/L572/L581/L643/L651/L658 + round 16 named at L925 | **CONFIRMED** |
| `coverage.dry_rounds` | 2 | **2 trailing consecutive** (4 total labelled DRY) | trailing-consecutive rule: rounds **17 and 18** are the terminal pair, both "Zero new findings" (L656, L670). Rounds 11 and 13 were also DRY but are non-terminal. | **CONFIRMED** |
| `coverage.dry` | true | **true** | `dry_rounds(2) >= K_required(2)` on the terminal run | **CONFIRMED** |
| `internal_files_inspected` | 19 | **>= 19** (24 under a generous rule) | 11 failing test modules + 8 non-test files carrying a file:line anchor or a measured property. Generous rule adds `backend.log`, `.claude/masterplan.json`, `.claude/hooks/lib/qa_write_guard.py`, and the 2 archived handoff artifacts = 24. | **CONFIRMED as a floor** |
| `brief_status` | COMPLETE | **COMPLETE** | exactly one occurrence (L16); exactly one ```json fence (L14-35); parses under `json.load` | **CONFIRMED** |

**The one correction: "16 of 37" understates the recency window.** Under the arXiv `YYMM`
rule the in-window set is 21 of 37 -- `2401.06765, 2407.03625, 2410.13480, 2411.11033,
2504.16777, 2511.18854, 2512.18088, 2601.08998, 2601.22264, 2602.02307, 2602.03556,
2602.05465, 2602.09311, 2602.19098, 2602.23957, 2603.09029, 2603.23054, 2604.26674,
2605.21677, 2607.09345, 2608.06535`. Three further sources are continuously-current docs
(`docs.pytest.org`, `handbook.gitlab.com`, `docs.gitlab.com`) and are excluded from the 21
because they carry no dated version. The error direction is **conservative** -- the brief
under-claimed its own recency coverage, so no hard blocker is affected.

**Hard blockers, re-checked against the measured figures:** 37 >= 5 read in full; 58 >= 10
URLs; recency scan present and reported; the read-in-full set is papers/full pages (ar5iv or
native arXiv HTML, or `curl` + `pypdf` full text for the 3 PDF-only items #17/#18/#19, and
`curl` + tag-strip for #24) -- no abstract-only source is counted, and the one abstract-only
candidate (Luo 2014, paywalled) is explicitly in the snippet table; internal claims carry
file:line anchors. Source-quality hierarchy holds: 27 of 37 are papers, the rest are official
docs/books (abseil, pytest, GitLab) and named-practitioner or vendor-engineering blogs
(Fowler, Google Testing, Spotify, Meta) -- zero community-tier sources in the counted set.

`gate_passed` stands at **true**, now on re-measured rather than self-reported figures.
The residual gap recorded in the soft checks (clock-independence proven against shifted
*zones* not a shifted *clock*; kill-switch non-touch is an after-hash + pre-run mtime rather
than a captured before/after pair) is a **soft** item and remains open for the executor --
86.5's criterion 5 wants the pair, so the "before" hash must be captured explicitly.
