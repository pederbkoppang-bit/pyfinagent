# Research Brief -- phase-86.94 cycle 4

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported for information only).
**Accessed:** 2026-08-17.
**Topic:** A regression guard whose green state is not a function of documentation
prose; making a structured evidence claim (a bool) falsifiable against a
measurement rather than only type-checked; a drift tripwire that re-opens on
RELEVANT corpus change (a quoted figure) and not on incidental change (a filename
in a report); mutation-testing the fail-closed branches of a static-analysis
checker; and attributing a mutation kill to the mechanism that actually fires it.

<!-- ENVELOPE:BEGIN -->
```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 16,
  "urls_collected": 23,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 3,
    "dry": false
  },
  "summary": "The guard is RED right now (42/3) and all three failures are the mentions_reviewed tripwire firing on incidental change; its corpus is 89.5% gitignored so it is unreproducible in the same class the step exists to close; and the fail-closed branch has zero mutation cells.",
  "brief_path": "handoff/current/research_brief_86.94_cycle4.md",
  "gate_passed": true
}
```
<!-- ENVELOPE:END -->

## Search queries run (three-variant discipline)

| Variant | Query |
|---|---|
| current-year frontier (2026) | `mutation testing static analysis checker fail-closed branch coverage 2026` |
| current-year frontier (2026) | `documentation drift detection executable documentation tests docs as tests 2026` |
| last-2-year window (2025) | `metamorphic testing static analyzers testing the checker soundness bugs analyzer 2025` |
| year-less canonical | `change-detector tests considered harmful Google testing blog brittle tests` |
| year-less canonical | `mutation testing trivial mutants killed for the wrong reason coincidental correctness kill attribution` |
| year-less canonical | `machine-checkable claim vs prose assertion attestation verifiable provenance predicate in-toto SLSA` |
| year-less canonical | `assertions strongly correlated test suite effectiveness Zhang Mesbah assertion strength oracle` |

## Read in full (7; >=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://abseil.io/resources/swe-book/html/ch12.html | 2026-08-17 | book chapter (official, Google/O'Reilly) | WebFetch, full | Brittle = "fails in the face of an unrelated change to production code that does not introduce any real bugs". "The ideal test is unchanging". Of four change classes (refactor / new feature / bug fix / behaviour change) only BEHAVIOUR CHANGE should force a test edit. |
| 2 | https://arxiv.org/html/2306.02319 | 2026-08-17 | preprint (arXiv) | WebFetch, arXiv HTML, full | Kill REASONS are separable: "Assertion violations would imply that the test oracles actually capture the correct program behaviour. Uncaught exceptions and timeouts, however, may only show coincidental impacts of the mutation." |
| 3 | https://arxiv.org/html/2511.11999 | 2026-08-17 | preprint (arXiv, 2025-11-15) | WebFetch, arXiv HTML, full | Same FAIL/EXC/TIME taxonomy; "despite the various reasons for mutant killing, the effectiveness of test suites is directly linked to assertion failures." |
| 4 | https://arxiv.org/html/2507.15892 | 2026-08-17 | preprint (arXiv, 2025-07-20) | WebFetch, arXiv HTML, full | Metamorphic testing OF A STATIC ANALYZER's rules: "If the analyzer detects the seed bug but fails to identify some or all of its mutants, it suggests that the rule is overly specific and lacks robustness." 64 problematic rules across SpotBugs/SonarQube/ErrorProne/Infer/PMD; 83% missed by the prior SOTA. |
| 5 | https://raw.githubusercontent.com/in-toto/attestation/main/spec/v1/statement.md | 2026-08-17 | official spec (in-toto v1) | WebFetch, full | A claim binds to a subject BY DIGEST: "Subject artifacts are matched purely by digest, regardless of content type"; "Subjects are assumed to be immutable". |
| 6 | https://slsa.dev/spec/v1.0/verifying-artifacts | 2026-08-17 | official spec (SLSA v1.0) | WebFetch, full | Verifier must independently establish "that statement's `subject` matches the digest of the artifact in question"; "Verification tools SHOULD reject unrecognized fields in `externalParameters`" -- an unexamined field is a FAILED verification, not a pass. |
| 7 | https://www.cherryleaf.com/2026/08/docs-as-tests/ | 2026-08-17 | practitioner article (2026-08-06) | WebFetch, full | Splits doc claims into testable (a checkable referent exists) and untestable (comprehension, mental models). "Passing is evidence, not absolution." |

## Identified but snippet-only (16; context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://testing.googleblog.com/2015/01/testing-on-toilet-change-detector-tests.html | official blog | WebFetch returned only page chrome + comments; article body absent from served DOM. Canonical for the term, superseded here by source #1 which restates it. |
| https://github.com/in-toto/attestation | official repo | Landing page rendered nav only; read the spec file directly instead (source #5). |
| https://link.springer.com/article/10.1007/s10009-025-00794-1 | journal survey (2025) | 303 redirect to `idp.springer.com` auth wall. |
| https://people.ece.ubc.ca/amesbah/resources/papers/fse15.pdf | peer-reviewed (FSE'15) | PDF; WebFetch PDF summaries are known in this project to fabricate quotes. Not fetched rather than risk an unverifiable quote. |
| https://www.semanticscholar.org/paper/Coverage-is-not-strongly-correlated-with-test-suite-Inozemtseva-Holmes/abd840dbcfd986e6de9102ab809c2c46e5ce47aa | peer-reviewed (ICSE'14) | Snippet sufficient; coverage-vs-effectiveness is background, not design-deciding here. |
| https://dl.acm.org/doi/fullHtml/10.1145/3425497 | peer-reviewed (TOSEM) | Stubborn mutants; adjacent, not on the attribution question. |
| https://dl.acm.org/doi/10.1145/3691620.3695034 | peer-reviewed (ASE'24) | Interrogation testing of program analyzers; paywalled. |
| https://www.shinhwei.com/statfier.pdf | peer-reviewed | PDF; superseded by source #4 which benchmarks against it. |
| https://arxiv.org/pdf/2408.13855 | preprint | PDF-only route; FN/FP census of static analyzers, background. |
| https://arxiv.org/pdf/2504.16472 | preprint (2025) | PDF-only route. |
| https://arxiv.org/pdf/2104.11767 | preprint | Mutation vs branch coverage, industrial; background. |
| https://oneuptime.com/blog/post/2026-01-24-mutation-testing/view | vendor blog (2026-01) | Community tier; no new claim beyond #2/#3. |
| https://www.docsie.io/blog/glossary/documentation-drift/ | vendor glossary (2026) | Community tier. |
| https://understandingdata.com/posts/doc-drift-detection-ci/ | practitioner blog | Community tier. |
| https://slsa.dev/spec/v0.1/provenance | official spec (superseded) | v1.0 read instead (source #6). |
| https://testing.googleblog.com/2016/05/flaky-tests-at-google-and-how-we.html | official blog | Flakiness, adjacent to brittleness but not the tripwire question. |

## Recency scan (2024-2026) -- PERFORMED

Searched the 2025 and 2026 windows explicitly (queries 1-3 above). **Four new
findings, all COMPLEMENTARY rather than superseding:**

1. **StaAgent** (arXiv:2507.15892, 2025-07-20) -- the first close external analogue
   to "mutation-test your own static-analysis checker": metamorphic testing of
   analyzer RULES, with the invariance relation stated explicitly.
2. **WITNESS** (arXiv:2511.11999, 2025-11-15) -- confirms the FAIL/EXC/TIME kill
   taxonomy is still the live framing in 2025 and that assertion kills are the
   ones tied to effectiveness.
3. **Docs-as-tests** (cherryleaf, 2026-08-06) -- current-year framing of exactly
   the "green state must not be a function of prose" question.
4. Assertions survey (Springer IJSTTT, 2025) and a 2026-01 mutation-testing
   practitioner piece -- snippet-only, no new claim.

Nothing in the window supersedes the canonical brittleness taxonomy (source #1)
or the in-toto digest-binding model (source #5). The 2025-2026 work supplies the
kill-attribution taxonomy and the checker-metamorphic pattern, both newer than
the standard mutation-testing surveys and directly on-point.

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `scripts/qa/verify_no_sliding_windows_86_94.py` | 731 | the shipped guard | **RED as of 2026-08-17: 42 passed, 3 failed** |
| `.claude/masterplan.json` (step 86.94 node) | -- | criteria + PARK note | PARKED after 3 Q/A attempts (FAIL, FAIL, CONDITIONAL) |
| `handoff/current/live_check_86.94.md` | 490 | criterion evidence | present; §A-§I |
| `handoff/current/experiment_results_86.94.md` | 223 | GENERATE record | present |
| `.gitignore:80` | 1 | `handoff/archive/_quarantine_*/` | **load-bearing, see F3** |
| `handoff/current/day_halt.md` | -- | session-ops note, ADDED after the pin commit | **the thing that turned the guard red** |
| `handoff/current/day_report_2026-08-17.md` | -- | day report, ADDED after the pin commit | second contributor |
| `.claude/agents/researcher.md`, `.claude/rules/research-gate.md` | -- | operating rules | read in full per STEP 0 |

## Key findings

**F1 -- The guard is RED right now, and all three failures are the tripwire.**
`python scripts/qa/verify_no_sliding_windows_86_94.py` -> `FAILED: 42 passed, 3
failed`. Every failure is the `mentions_reviewed` equality at
`verify_no_sliding_windows_86_94.py:544-551` disagreeing with the pins at
`:227-231`: scheduler.py 282 -> **283**, verify_decision_log_86_97.py 6 -> **9**,
frontend_route_inventory.py 49 -> **50**. (The masterplan PARK note records "ALL
GREEN 45/0"; that is no longer the state.)

**F2 -- It fired on INCIDENTAL change, exactly as the objective anticipated.**
Since the pinning commit `964b0255` (2026-08-17T00:51:13+02:00) exactly **3**
handoff files were added; two mention guarded names: `handoff/current/day_halt.md`
(all three) and `handoff/current/day_report_2026-08-17.md` (one). Both are session
narration. **Neither quotes any figure derived from any window.** Per source #1
this is textbook brittleness -- a failure "in the face of an unrelated change ...
that does not introduce any real bugs" -- and per its taxonomy no behaviour
changed, so no test should have needed to change.

**F3 -- The tripwire's own corpus is UNREPRODUCIBLE, in the same class this step
exists to close.** `MENTIONS` at `:511` walks `(REPO/'handoff').rglob('*.md')` --
the WORKING TREE, not a pinned set. Measured 2026-08-17: **49,094** `.md` files
under `handoff/`, of which **5,167** are tracked; **43,927 (89.5%) are gitignored**
via `.gitignore:80` (`handoff/archive/_quarantine_*/`). For
`frontend_route_inventory.py` only **5 of 50** mention-sites are tracked; 45 are in
the ignored quarantine. Worse: the allowlist's own smoking-gun citation at
`:203-205` --
`handoff/archive/_quarantine_2026-04-21/phase-3.7.5-v22/experiment_results.md`,
the file whose `opens_30d` / `git_activity_30d` figures justify
`quoted_as_evidence: True` -- **is itself gitignored**. On a fresh clone the count
is 5, not 50, the guard is RED for a third distinct reason, and the evidence for
the bool is absent. A count over "whatever `.md` happens to be on this disk" is a
number about a machine exactly as `--since=<bare date>` is a number about a clock.

**F4 -- The bool is the right SHAPE; the tripwire bound to it is the wrong
PREDICATE.** `quoted_as_evidence` is genuinely falsifiable in principle, and
sources #5/#6 endorse the move from prose to a structured claim. But they also
say what the claim must bind to: in-toto matches subjects **purely by digest**,
and "Subjects are assumed to be immutable"; SLSA requires the verifier to check
"that statement's `subject` matches the digest of the artifact in question".
`mentions_reviewed` binds instead to `name in text` -- a NAME over a mutable,
unbounded corpus. That is the one binding in-toto explicitly rejects.

**F5 -- The narrower predicate criterion 4 actually asks for.** Criterion 4 asks
whether a *figure* derived from a member was quoted. So the tripwire should key on
figure-bearing tokens (`opens_30d`, `usage_source: git_activity_30d`, a numeric
adjacent to the name), or on a digest over the enumerated (file, figure) pairs the
author adjudicated -- not on the bare filename. New prose that merely names the
script then becomes inert; a new quoted figure re-opens the judgement. Source #6
supplies the complementary half: a new site must not silently pass ("SHOULD reject
unrecognized fields"), so the correct shape is *narrow predicate + fail-closed on
an unrecognised match*, not *broad predicate + a number to bump*.

**F6 -- Kill attribution: my hypothesis was WRONG, and the measurement gave a
better finding.** I hypothesised that some `[4]` cells were killed by the
fail-closed `<unparsed>` branch rather than by `classify()`. MEASURED, all 11
injections differenced against the clean control: **all 11 are killed by
`classify()` on a parsed VALUE; none reaches `<unparsed>`.** The real gap is the
inverse -- **the fail-closed branch at `:374-379`, the module's central claim, has
ZERO mutation cells.** Its own motivating comment is now stale: the space form
`--since 2026-08-11` no longer reaches it, because `window_value()` returns
`('2026-08-11', True)` once `PLAUSIBLE_VALUE` matches. Four shapes were measured
to reach it (4/4, each returning `value='<unparsed>'`, verdict SLIDING):
`subprocess.run(["git","log","--since", win])`, the f-string-element form,
`--since=` with an empty value, and `--after` + variable. An argv list with a
variable window is a realistic idiom, so this is an uncovered branch, not a corner.

**F7 -- The cells cannot tell the two mechanisms apart.** Every `[4]` assertion is
`check(..., bool(hits))` over hits filtered only on `h[3] == "SLIDING"`
(`:608-610`); the value field `h[2]` is never asserted. A cell is therefore green
whether `classify()` or the fail-closed branch fired. This is precisely the
caution in sources #2/#3: non-oracle kills "may only show coincidental impacts",
and effectiveness is "directly linked to assertion failures". Cheap fix: make the
kill signal carry its mechanism -- assert `h[2] == '<unparsed>'` on fail-closed
cells and `h[2] != '<unparsed>'` on value-classification cells.

**F8 -- Prior art for the invariance the cells are approximating.** Source #4's
metamorphic relation ("detects the seed bug => detects every semantically
equivalent mutant"; a break means "the rule is overly specific and lacks
robustness") is what the argv / space / `--after` / `--before` cells are hand-
writing one at a time. Stating it as a relation over spelling variants of one
sliding value -- rather than as N hand-listed cells -- is both shorter and closes
the recall gap that cost cycles 1-2 five survivors.

## Consensus vs debate

Consensus across #1, #5, #6, #7: a check is only as good as the referent it binds
to, and a claim that cannot be contradicted by a measurement is not evidence
("Passing is evidence, not absolution"). Debate in #2 vs #3: #2 finds assertion-
only mutant filtering best for failing-test models but that unfiltered wins once
passing tests are included -- so "only assertion kills count" is a useful default,
not a law. Applied here that argues for RECORDING the mechanism rather than
discarding non-assertion kills.

## Pitfalls (from literature + measurement)

1. Replacing a prose predicate with a numeric one does not make it falsifiable if
   the number is measured over an undefined corpus (F3).
2. A tripwire tuned for sensitivity becomes a change-detector and gets switched
   off -- the same reasoning that already made this rule an allowlist not a ban.
3. A mutation cell that goes red proves *something* fired, not that the *intended*
   thing fired (F6/F7).
4. Bumping `mentions_reviewed` to 283/9/50 would be the exact anti-pattern the
   entry's own text forbids ("re-review the sites and re-state the judgement rather
   than bumping the number") -- and would leave F3 untouched.

## Application to pyfinagent (file:line)

- `verify_no_sliding_windows_86_94.py:227-231` -- keep `quoted_as_evidence` (bool);
  replace `mentions_reviewed` (count of files containing a NAME) with a claim bound
  to the adjudicated evidence set: the figure-bearing tokens, or a digest over the
  enumerated (file, figure) pairs (F4/F5, sources #5/#6).
- `:490` / `:505-512` -- pin the corpus. `QUOTE_DIRS = ["handoff"]` currently means
  the working tree (89.5% gitignored). Use `git ls-files handoff` so the denominator
  is reproducible on a fresh clone (F3). Note this CHANGES the numbers and
  invalidates the 49-mention evidence in `:192-209`, which must be re-derived from
  tracked files -- a correction that replaces, not accompanies.
- `:374-379` -- add mutation cells for the fail-closed branch using the four
  measured shapes; also fix the stale comment at `:375-376` (F6).
- `:606-610` and the `[4]` block generally -- assert `h[2]` so each cell records
  WHICH mechanism killed it (F7, sources #2/#3).
- `:571-605` -- consider replacing the hand-listed spelling cells with a stated
  invariance relation over spelling variants (F8, source #4).
- Scope note: none of the above touches verdict semantics (criterion 7).

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7)
- [x] 10+ unique URLs total (23: 7 full + 16 snippet-only)
- [x] Recency scan (2024-2026) performed + reported (4 findings)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered the caller's declared scope
- [x] Contradictions noted (#2 vs #3; and my own falsified F6 hypothesis recorded)
- [x] Claims cited per-claim
- [ ] GAP, stated: the Google change-detector post and the Springer assertions
      survey could not be retrieved (DOM / auth wall); both are covered by
      substitutes but neither is a read-in-full here.
