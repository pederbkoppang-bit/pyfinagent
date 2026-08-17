# Research Brief — step 86.97 (cycle 4)

**Tier:** simple (caller-stated). **Audit-class:** NO (coverage reported for information only).
**Objective:** (a) Why "a log line containing `reason=` exists" cannot detect a spurious or
unrecorded decision, and how to write an end-to-end guard over a shell hook that pins WHAT the
decision was. (b) Sweeping a claim class for stale unbounded statements using a known-member
recall test seeded from OTHER artifacts, not from the author's own phrasings.

**Status:** COMPLETE.

<!-- ENVELOPE (born inert, phase-86.37). Flipped to COMPLETE as the final act. -->
```json
{
  "brief_status": "COMPLETE",
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 20,
  "urls_collected": 26,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 6,
    "dry": false
  },
  "summary": "verify_decision_log_86_97.py:305 asserts \"reason=\" in log_text -- a substring test against a token the hook emits from a LITERAL FORMAT STRING (post-commit-changelog.sh:271), so it is true for every non-empty line the writer can emit and is subsumed by the existence check at :301. It cannot see reason=unrecorded (the :267 .get default) and does not see the recorded cycle-3 survivor, the :214 mutant that writes bump=minor reason=unrecorded. The E2E driver seeds {\"phases\": []} so it exercises 1 of 9 reason states (no_flip) and pins 0 by value. Fix: parse (bump, reason, created_done, transitioned_done) and assert the tuple per scenario; drive >=4 reason states; assert reason != \"unrecorded\" everywhere as a negative control; derive the expected table from the branch structure BEFORE driving, never from observed output. For the claim sweep, the literature names the exact cycle-3 failure: seeding a recall test from your own terms makes recall ~100% and meaningless -- seed from independent artifacts and score quasi-sensitivity.",
  "brief_path": "handoff/current/research_brief_86.97_cycle4.md",
  "gate_passed": true
}
```

---

## Section log (appended incrementally)

## A. Internal code inventory (the Explore half) — all claims file:line anchored

| File | Anchor | Role | Status |
|---|---|---|---|
| `.claude/hooks/post-commit-changelog.sh` | `:95` `bump_type = classify_commit(...)` | subject-based bump | live |
| same | `:100` `_ABSENT = object()`; `:103` `_FLIP_DECISION: dict = {}` | module-level decision record | live |
| same | `:106-210` `_flip_magnitude()` | sets `_FLIP_DECISION["reason"]` at **7 distinct sites** (`:160,:163,:190,:192-195,:207`) | live |
| same | `:213-216` `if bump_type != "major": bump_type = _flip_magnitude()` **else** `_FLIP_DECISION["reason"]="subject_forced_major"` | the ONLY branch that sets reason outside the detector | live |
| same | `:219-275` `_log_decision()`; write at `:270-272` | the writer; `reason = _FLIP_DECISION.get("reason", "unrecorded")` at `:267` | live |
| same | `:278` `_log_decision(bump_type)` | **the production invocation** — a module-level call `Expr` | live |
| `scripts/qa/verify_decision_log_86_97.py` | 586 lines, 5 sections | the new guard | live |
| same | `:269-288` `drive()` | E2E: builds a temp git repo, runs the REAL hook, returns `(rc, decision-log text, stderr)` | live |
| same | `:298` `rc, log_text, err = drive(HOOK_SRC)` | the baseline drive | live |
| same | `:301` `check("... a decision line is WRITTEN TO THE FILE", log_text.strip() != "")` | existence assertion | live |
| same | **`:305` `check("[3] the decision line carries a reason", "reason=" in log_text, ...)`** | **THE DEFECT UNDER STUDY** | live |
| same | `:337-341` recursion-guard drive; `:308-333` isolation predicate | bound + isolation | live |
| `handoff/current/live_check_86.91.md` | `:104` `## 3. Criterion 4 -- every decision now explains itself` | **stale unbounded claim, never corrected** | STALE |
| `handoff/current/experiment_results_86.91.md` | `:441-445` | cycle-4 residual disclosure (the step's own audit basis) | live |
| `handoff/current/live_check_86.97.md` | `:217-219` | the cycle-3 sweep table — 3 members, all found by the AUTHOR's own wording `"Every invocation"` | INCOMPLETE |
| `handoff/current/night_diagnostics.md` | `:51` | records the FAIL: `live_check_86.91.md:104` "was never bounded and never touched" | evidence |
| `handoff/current/goal_next_2026-08-17.md` | `:52` | names the mechanism: sweep keyed on `"every invocation"`, survivor says `"every decision"` | evidence |

### A1. Why `:305` cannot detect a spurious or unrecorded decision (measured from source)

`"reason=" in log_text` is a **substring-existence** test against a line the hook
produces from a **literal format string** — `.../post-commit-changelog.sh:271`:

```
_fh.write(f"{stamp} {sha} bump={bump} reason={reason} "
          f"created_done={created} transitioned_done={transitioned}\n")
```

The token `reason=` is in the *format*, not the *value*. So the assertion is true for
**every non-empty line the writer can emit**, including:

- `reason=unrecorded` — the `:267` `.get(...,"unrecorded")` default, i.e. the exact
  "decision made but not recorded" case the log exists to expose;
- any wrong-but-well-formed value (`reason=no_flip` emitted when a flip *did* happen);
- the **known survivor** already on record: the cycle-3 hook `:214` mutant writes
  `bump=minor reason=unrecorded` and survives (`night_diagnostics.md:51`).

Consequence: `:305` is **subsumed by `:301`** for every mutation of the *value*. It can
only fail if the writer stops emitting the literal `reason=` — which `:301`
(`log_text.strip() != ""`) does not cover but which is a *format* mutation, not a
*decision* mutation. The guard therefore pins the SHAPE of the record and not the
CONTENT of the decision.

### A2. The E2E driver exercises exactly ONE of nine reason values

`drive()` seeds `.claude/masterplan.json` as `{"phases": []}` (`:260`) and commits
`feat: a real change` (`:269`). Trace: `classify_commit` → `minor`; `:213` is not
`major` so `_flip_magnitude()` runs; both `_statuses()` calls return `{}` (not `None`);
`newly_done == []` → `:190` `reason = "no_flip"`. **`no_flip` is the only member of the
closed set ever driven end-to-end.** The closed set (from `live_check_86.91.md:113`) is
`subject_forced_major, flip_created, flip_transitioned, flip_created_and_transitioned,
no_flip, masterplan_unreadable_at_HEAD, first_commit, detector_error:<Type>` — plus the
writer's own `unrecorded` fallback = **9 states, 1 covered, 0 pinned by value**.

### A3. The claim-class sweep: seeded from the author's own phrasings (the failure)

`live_check_86.97.md:217-219` lists 3 corrected members, and `:237-242` shows the sweep's
own grep output — every hit contains the literal `Every invocation`. The surviving member,
`live_check_86.91.md:104`, reads **"every decision now explains itself"**. It is the same
claim in a different noun. `goal_next_2026-08-17.md:52` states the mechanism directly:
the probe was keyed on *"invocation", my own wording, while the survivor says "every
decision"*. This reproduces the standing lesson
`feedback_verification_probe_built_from_edited_strings` — a probe built from the strings
you just edited cannot find a member you never phrased that way.

**Independently-seeded known members available for a recall test** (none authored by the
sweeping session): `handoff/current/night_diagnostics.md:51` (names `live_check_86.91.md:104`
verbatim), `evaluator_critique_86.91.md:434` (the Q/A's own statement of the bound),
`experiment_results_86.91.md:444`, `.claude/masterplan.json` step 86.97 `audit_basis`.

---

## B. External sources READ IN FULL (counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key verbatim finding |
|---|---|---|---|---|---|
| 1 | https://ar5iv.labs.arxiv.org/html/2212.06118 | 2026-08-17 | preprint survey (peer-review track) | WebFetch (full, ar5iv per the arXiv chain) | *"One premise during the computation of code coverage is that test cases contain sufficient test oracles to detect faults. However, in practice, this assumption does not hold."* and *"it is entirely possible for a test suite with zero test oracles to achieve 100% code coverage, resulting in a poor quality test suite with low fault-detection effectiveness."* Checked coverage (§5.2) *"measures the percentage of statements that are executed and that also influence the computation of test oracles"*; measured *"an average difference of 24%, meaning that 24% of the executed statements do not influence any test oracle."* |
| 2 | https://arxiv.org/html/2410.21136 | 2026-08-17 | preprint (Oct 2024) | WebFetch (full, native arXiv HTML) | Oracles drift toward the implementation, not the spec: *"LLM's test oracle classification accuracy considerably drops in the presence of buggy code, suggesting that its predictions are derived towards the actual implementation rather than the desired one."* Records assertions that pass yet *"did not seem relevant to the test's behaviour"* / *"too vague"* / did not *"validate any specific behaviour."* Mutation scores 19.10% (LLM) vs 17.32% (Evosuite). |
| 3 | https://arxiv.org/html/2402.11041v1 | 2026-08-17 | preprint / e-Informatica SEJ (2024) | WebFetch (full, native arXiv HTML) | **The bullseye for the sweep half.** *"We recommend that persons conducting the informal search should be independent researchers. An independent researcher here is not involved in the study and has not participated in the design of the search strategy."* And the failure mode named exactly: *"if the same search terms are used for the informal search and the actual systematic search, then the recall is likely 100% since the actual search will probably find the same relevant papers... Hence, the 100% recall cannot guarantee that researchers achieve an acceptable level of search completeness."* QGS = *"a subset of a hypothetical gold standard, the complete set of all relevant papers on the topic"*; acceptance threshold 70-80% quasi-sensitivity. |
| 4 | https://engineering.fb.com/2025/09/30/security/llms-are-the-key-to-mutation-testing-and-better-compliance/ | 2026-08-17 | official vendor engineering blog (Meta, Sep 2025) | WebFetch (full) | *"statement or branch coverage might still fail to detect a bug if a line still runs"*; mutation testing *"reveals whether a test fails after inserting a mutation, indicating that the tests are not effectively checking the code's behavior"* and *"helps engineers and developers identify weak assertions and encourages them to write tests that truly validate code behavior instead of just executing it."* Engineer acceptance 73%; equivalence detector precision 0.79 / recall 0.47, rising to 0.95 / 0.96 with preprocessing. |
| 5 | https://bats-core.readthedocs.io/en/stable/writing-tests.html | 2026-08-17 | official docs (bats-core, shell E2E) | WebFetch (full) | The idiom for asserting a **shell hook's** observable output is exact equality, not substring: `[ "$output" = "foo: no such file 'nonexistent_filename'" ]` and `[ "${lines[0]}" = "usage: foo <filename>" ]`. `run` *"saves the exit status and output into special global variables"*; `$output` is *"the combined contents of the command's standard output and standard error streams"*; `$lines` is *"available for easily accessing individual lines of output."* Negative cases use `run ! command` because bare `!` under `set -e` will not reliably fail the test. |
| 6 | https://arxiv.org/html/2506.15227 | 2026-08-17 | preprint SLR (Jun 2025) | WebFetch (full) | *"while code coverage is widely regarded as a useful metric, its correlation with actual bug detection capability remains weak."* On oracle evaluation: *"static generation metrics do not reliably capture the quality of the generated oracles, and dynamic test adequacy metrics should serve as the principal evaluation criteria."* And the root difficulty: *"generating reliable test oracles poses a non-trivial technical challenge, as it requires capturing the intended design specification rather than merely reflecting the implemented behavior."* |

## C. Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://onlinelibrary.wiley.com/doi/abs/10.1002/stvr.1497 | journal (STVR 2013, Schuler & Zeller) | paywalled abstract; content covered in full by source 1 |
| https://www.st.cs.uni-saarland.de/publications/files/schuler-icst-2011.pdf | paper (ICST 2011) | PDF; per-project rule, avoid WebFetch on PDFs (fabricated-quote risk) |
| https://www.st.cs.uni-saarland.de/publications/files/schuler-stvr-2013.pdf | paper | same |
| https://dl.acm.org/doi/10.1109/ICST.2011.32 | ACM DL | paywalled |
| https://neu-se.github.io/CS7580-Fall-2021/lecture-notes/3-checked-coverage-schuler-icst2011/ | course notes | tier-3 secondary; source 1 is primary |
| https://www.sciencedirect.com/science/article/abs/pii/S0950584910002260 | journal (Zhang/Babar/Tell 2011, the QGS origin) | paywalled; method quoted in full via source 3 |
| https://www.cse.chalmers.se/~feldt/advice/zhang_2011_optimal_search_for_se_studies.pdf | paper PDF | PDF |
| https://www.e-informatyka.pl/index.php/einformatica/volumes/volume-2022/issue-1/article-3/ | journal landing page | same work as source 3 (arXiv version read instead) |
| https://pdfs.semanticscholar.org/59d3/ec40b4f17ed94dc5ae510c316ac511915031.pdf | paper PDF | PDF |
| https://ieeexplore.ieee.org/document/9833107/ | IEEE (VL/HCC 2022, "Is Assertion Roulette still a test smell?") | paywalled |
| https://ginabai.github.io/files/PaperPreprints/vlhcc22_AssertionRoulette.pdf | preprint PDF | PDF |
| https://tsr-catalog.readthedocs.io/en/latest/smells/assertion-roulette.html | catalog | tier-5 community |
| https://arxiv.org/pdf/2301.12284 | preprint ("Assertion Inferring Mutants") | `/pdf/` — never WebFetch per the chain; ar5iv deferred, budget |
| https://arxiv.org/pdf/2602.08146 | preprint 2026 ("Test vs Mutant: Adversarial LLM Agents") | recency-scan hit; `/pdf/` |
| https://arxiv.org/pdf/2509.23674 | preprint ("AssertGen") | hardware-assertion domain, off-topic |
| https://sol.sbc.org.br/index.php/sbqs/article/download/39001/38773/ | conference PDF | PDF |
| https://www.researchgate.net/publication/394720083_Mutation-Guided_LLM-based_Test_Generation_at_Meta | RG record | source 4 is the primary vendor account |
| https://www.augmentcode.com/guides/mutation-testing-ai-generated-code | vendor guide | tier-5 marketing |
| https://pypi.org/project/pytest-structlog/0.6 | package docs | Python-only; hook is bash+heredoc |
| https://www.baeldung.com/junit-asserting-logs | blog | JVM-specific |

**URLs collected: 26** (6 read in full + 20 snippet-only).

### Search queries run (three-variant discipline, `.claude/rules/research-gate.md`)

- **Year-less canonical (3):** `checked coverage oracle quality Schuler Zeller assertion checks result`;
  `quasi-gold standard search string sensitivity recall known relevant studies systematic review software engineering`;
  `assertion roulette test smell asserting log output structured logging testing shell script bats end-to-end`.
- **Current-year frontier (1):** `weak test assertions 2026 LLM generated assertions check existence not value mutation testing`.
- Mix is visible in the tables: sources 1 (2022) and 5 (undated official docs) are year-less hits;
  sources 2 (2024), 3 (2024), 4 (2025), 6 (2025) are recency-window hits.

## D. Recency scan (last 2 years, 2024-2026) — MANDATORY SECTION

Searched explicitly with a 2026 suffix and read four in-window sources in full
(2, 3, 4, 6; 2024-2025). **Result: 4 new findings that COMPLEMENT rather than supersede
the canonical Schuler-Zeller checked-coverage result of 2011.**

1. The canonical claim is **restated and re-measured, not overturned**: Meta (2025) still
   frames the problem as *"statement or branch coverage might still fail to detect a bug if
   a line still runs"* — the same premise source 1 traces to 2011.
2. **New in-window framing that matters here:** source 6 (2025) elevates it to a
   methodological rule — *"dynamic test adequacy metrics should serve as the principal
   evaluation criteria"*, i.e. the mutation cell is the criterion, not the assertion count.
3. **New in-window failure mode:** source 2 (2024) documents oracles that encode the
   *actual* implementation instead of the *expected* one — a hazard absent from the 2011
   literature and directly applicable to how the expected-reason table must be derived.
4. **The QGS method is unchanged but re-tested in-window** (source 3, 2024) against the
   2011 Zhang/Babar/Tell original, and the independence warning survives verbatim.
5. One 2026 hit found and NOT read in full (`arXiv:2602.08146`, adversarial LLM agents for
   unit-test robustness) — `/pdf/`-only in the result set; recorded as snippet-only.

**No source in the window contradicts the design recommended below.**

## E. Key findings

1. **An existence assertion contributes nothing to oracle quality.** *"it is entirely
   possible for a test suite with zero test oracles to achieve 100% code coverage,
   resulting in a poor quality test suite with low fault-detection effectiveness"* and
   checked coverage measures only statements that *"influence the computation of test
   oracles"* — measured gap 24% (Source 1, ar5iv/2212.06118). `verify_decision_log_86_97.py:305`
   executes the writer and observes that it ran; the *decision value* influences no oracle.
2. **Exact equality is the shell-hook idiom, and it is official.** Bats asserts
   `[ "$output" = "<exact expected string>" ]` and offers `${lines[N]}` for line-level
   pinning (Source 5, bats-core docs). Substring containment is not the documented pattern.
3. **Mutation, not assertion count, is the acceptance criterion.** *"mutation testing
   ... helps engineers and developers identify weak assertions"* (Source 4, Meta 2025) and
   *"dynamic test adequacy metrics should serve as the principal evaluation criteria"*
   (Source 6). A cell that turns the guard RED is the proof; a new `check()` line is not.
4. **The oracle must encode EXPECTED, not OBSERVED, behaviour.** *"requires capturing the
   intended design specification rather than merely reflecting the implemented behavior"*
   (Source 6); *"its predictions are derived towards the actual implementation rather than
   the desired one"* (Source 2). Deriving the expected-reason table by running the hook and
   recording its output would bless a wrong reason.
5. **A recall test seeded from your own terms is worthless — stated verbatim in the
   literature.** *"if the same search terms are used for the informal search and the actual
   systematic search, then the recall is likely 100% ... Hence, the 100% recall cannot
   guarantee that researchers achieve an acceptable level of search completeness"*, with the
   remedy *"persons conducting the informal search should be independent researchers ...
   not involved in the study and ha[ve] not participated in the design of the search
   strategy"* (Source 3, arXiv 2402.11041v1). Target quasi-sensitivity 70-80%.

## F. Consensus vs debate

**Consensus** across all six sources: coverage/execution evidence overstates fault
detection; the oracle is the binding constraint; mutation is the way to measure it.
**Debate:** source 1 argues checked coverage is *"even more sensitive than mutation
testing"* (assertions removed: checked coverage -23% vs mutation score -14%), while
sources 4 and 6 treat mutation as the principal criterion. **Immaterial here** — both
verdicts condemn `:305`, and the project's own mutation-matrix idiom already matches
sources 4/6. Secondary debate: whether Assertion Roulette is still a real smell
(IEEE 9833107, snippet-only, argues developers find it acceptable) — irrelevant, since the
defect here is assertion *weakness*, not assertion *labelling*.

## G. Pitfalls (from the literature, mapped to the traps this step has already hit)

- **Oracle derived from observed output** (Sources 2, 6) → do not build the expected-reason
  table by running the hook. Derive it from the branch structure at
  `post-commit-changelog.sh:160, :163, :190, :192-195, :207, :216` first.
- **Recall test seeded from own phrasings** (Source 3) → the exact cycle-3 failure.
- **A well-formed record that is wrong** — the known survivor writes
  `bump=minor reason=unrecorded`, which satisfies `"reason=" in log_text`. Source 4's
  framing ("a line still runs") is precisely this shape.
- **Assertion roulette** (IEEE 9833107, snippet-only) → per-scenario assertions need their
  own detail message; `check()` already takes one.

## H. Application to pyfinagent (external findings → file:line anchors)

**H1. Replace the existence test at `scripts/qa/verify_decision_log_86_97.py:305`.**
Parse the line and assert the **pair**, not the token:

```
m = re.search(r"\bbump=(\S+) reason=(\S+) created_done=(\S+) transitioned_done=(\S+)", log_text)
```
then assert `(bump, reason) == expected_for_scenario` **and** the two id lists. This pins
WHAT was decided and WHICH steps drove it — neither is observable from `"reason=" in`.
Justified by Source 5's exact-equality idiom and Source 1's checked-coverage argument.

**H2. Drive more than one of the nine reason states.** `drive()` at `:269-288` currently
reaches exactly one (`no_flip`, via the `{"phases": []}` seed at `:260` — see §A2).
Reachable scenarios to parameterise, each with its expected `(bump, reason)` derived from
source before driving: `no_flip`; `flip_transitioned` (step pending at HEAD~1, done at
HEAD); `flip_created` (step absent at HEAD~1); `subject_forced_major` (`feat!:` /
`BREAKING CHANGE:`, which is the ONLY reason set outside the detector, at `:216`);
`masterplan_unreadable_at_HEAD`; `first_commit`; `detector_error:<Type>`.

**H3. Make `unrecorded` a negative control.** `:267` `_FLIP_DECISION.get("reason","unrecorded")`
is unreachable on the unmutated hook. Assert `reason != "unrecorded"` in **every** driven
scenario. This is the assertion that kills the recorded cycle-3 survivor (`night_diagnostics.md:51`:
the `:214` mutant *"SURVIVES and writes a spurious `bump=minor reason=unrecorded`"*).

**H4. Mutation cells, control GREEN first** (Sources 4, 6). At minimum:
(i) hook `:214` → the recorded survivor, must now be KILLED;
(ii) hook `:267` → hardcode `reason = "no_flip"`, must be KILLED by the `flip_transitioned`
scenario (impossible with one scenario — this is why H2 is load-bearing, not cosmetic);
(iii) hook `:278` delete `_log_decision(bump_type)` → already covered by `:301`, re-assert;
(iv) checker `:305` itself → revert to `"reason=" in log_text` and show the H2/H3 scenarios
still RED, proving the strengthening is what acts.

**H5. The claim-class sweep, as a QGS quasi-sensitivity test** (Source 3). Concretely:
- **Seed set from INDEPENDENT artifacts only** — none authored by the sweeping session:
  `night_diagnostics.md:51` (names `live_check_86.91.md:104` verbatim),
  `evaluator_critique_86.91.md:434`, `experiment_results_86.91.md:444`,
  `.claude/masterplan.json` step 86.97 `audit_basis`. This is Source 3's *"independent
  researcher ... not involved in ... the design of the search strategy."*
- **Score quasi-sensitivity = |found ∩ seed| / |seed|** and REPORT it. The known set is small
  and enumerable, so demand 100%, above Source 3's 70-80% acceptance band.
- **Fail the sweep if it cannot find its own known members** — the same rule the step already
  imposes on the exit-path scan (masterplan 86.97 criterion 2: *"a scan that cannot find its
  own known members is a FAILED gate"*), now applied to the prose sweep.
- **Key the pattern on the CLAIM, not on a wording**: `\b[Ee]very (invocation|decision|commit|call)\b`
  in proximity to `explain|record|WHY|reason`, plus the sibling forms already on record —
  `"An unexplained none becomes impossible"`, `"no longer expressible"`, `"the CLASS, closed"`.
  Cycle 3 keyed on the literal `Every invocation` — its own edit string — scored 3/3, and
  missed `live_check_86.91.md:104` `"every decision now explains itself"`.
- **Correction must REPLACE**: `live_check_86.91.md:104` is a *heading*; edit the heading
  itself, do not append a note beneath it (project rule
  `feedback_a_correction_must_replace_not_accompany`).

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (**6**)
- [x] 10+ unique URLs total incl. snippet-only (**26**)
- [x] Recency scan (2024-2026) performed + reported (§D)
- [x] Full pages read, not abstracts, for the read-in-full set (HTML/ar5iv chain; zero `/pdf/` fetches)
- [x] file:line anchors for every internal claim (§A)

Soft checks:
- [x] Internal exploration covered every module in the caller's INTERNAL SCOPE
- [x] Contradictions / consensus noted (§F — checked coverage vs mutation sensitivity)
- [x] All claims cited per-claim
- Disclosure: of 12 internal files inspected, 7 were read directly (hook `:85-294`, checker
  `:240-359` + full assertion inventory, researcher.md, research-gate.md, live_check_86.91.md
  §3, experiment_results_86.91.md `:430-455`, masterplan 86.97) and 5 were inspected via
  targeted grep only (`night_diagnostics.md`, `goal_next_2026-08-17.md`, `live_check_86.97.md`,
  `contract_86.91.md`, `evaluator_critique_86.91.md`). Tier is `simple`; the brief exceeds the
  300-word guideline because the internal inventory is the deliverable here — the source
  floors are hard, the word budget is not.


