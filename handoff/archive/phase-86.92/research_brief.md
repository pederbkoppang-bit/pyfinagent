# Research Brief — phase-86.92

**Topic:** Durable test fixtures for evolving validators — golden-file vs
synthetic/self-owned fixtures, fixture rot, and dead-gate detection (a check
that is red for a reason nobody acts on stops signalling).

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported
for information only; `coverage.dry` is not required).
**Researcher:** Layer-3 Researcher via the Workflow rail. **Date:** 2026-08-16.

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 15,
  "urls_collected": 22,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 2,
    "dry": false
  },
  "summary": "The step's own premise is REFUTED by measurement: the RED is caused by the checker's hand-written `verification` literal (verify_workflow_args_boundary.mjs:179/:319), which omits 3 fields the schema has required since 86.28/86.37 -- not by research_brief_86.17.md. enforceGate is pure and never opens that brief; a fake path gives byte-identical violations, and adding the 3 fields turns the gate green. The `-1` is a documented sentinel (research-gate.js:632), mis-rendered as a count at :740. Fixture rot has DISABLED mutation cell [4] drop-blind-violation, which can no longer kill. 86.17 is `done` with a now-RED immutable command, and 86.92's own command is `node --check` (a parse check that cannot fail on this defect).",
  "brief_path": "handoff/current/research_brief_86.92.md",
  "gate_passed": true
}
```

---

## Search-query variants run (three-variant discipline)

| # | Variant | Query |
|---|---------|-------|
| 1 | year-less canonical | `golden file testing approval tests brittleness fixture rot` |
| 2 | year-less canonical | `test suite alerting "alert fatigue" ignored failing tests normalization of deviance` |
| 3 | current-year frontier (2026) | `snapshot testing maintenance cost empirical study obsolete test assertions 2026` |
| 4 | last-2-year window (2025) | `arXiv 2025 empirical study test rot obsolete assertions co-evolution production code test code` |

---

## Sources read in full (counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://martinfowler.com/articles/nonDeterminism.html | 2026-08-16 | authoritative blog (named researcher) | WebFetch, full | "Once that discipline is lost, then a failure in the healthy deterministic tests will get ignored too. At that point you've lost the whole game." Prescribes **quarantine with a hard bound** — a numeric cap or a time cap ("no tests remaining quarantined longer than one week"). |
| 2 | https://jestjs.io/docs/snapshot-testing | 2026-08-16 | official docs | WebFetch, full | Snapshots must be "committed and reviewed as part of your regular code review process"; the explicit goal is to "fight against the habit of regenerating snapshots when test suites fail instead of examining the root causes of their failure." Blind `-u` "lock[s] in buggy behavior". |
| 3 | https://sre.google/sre-book/monitoring-distributed-systems/ | 2026-08-16 | official docs / industry (Google SRE) | WebFetch, full | "Every page should be actionable"; "If a page merely merits a robotic response, it shouldn't be a page"; remove "signals that are collected but unused". Alerting must answer "*does this rule detect an otherwise undetected condition that is urgent, actionable…*". |
| 4 | https://arxiv.org/html/2605.06125v1 | 2026-08-16 | preprint (TEBench, arXiv HTML per the html-first chain) | WebFetch, full | Defines **Test-Stale**: "An existing test t∈T still passes after Δ is applied, but the developer nonetheless updates t… so that it better reflects the revised semantics of the code." Stale tests in **207/314 tasks (65.9%)**. Detection F1 ≈ **36%**; best structural recall **59.1%**. Root cause: "Because stale tests still pass on the updated code, **no execution signal indicates that updates are needed**." |
| 5 | https://arxiv.org/html/2511.21382v1 | 2026-08-16 | preprint (survey, LLM unit-test generation) | WebFetch, full | "Following modifications to production code, existing test cases may become obsolete or invalid." Names **mutation testing** as "a widely accepted standard for evaluating fault detection capability"; test *usability* is now 70%+ while "improving test **effectiveness** … remains a major challenge". Names the `magic number test` smell (hard-coded constants in assertions). |
| 6 | https://pmc.ncbi.nlm.nih.gov/articles/PMC2821100/ | 2026-08-16 | peer-reviewed (Banja, *normalization of deviance*) — **cross-domain** | WebFetch, full | The mechanism by which a known-bad state becomes routine: **institutionalization → socialization → rationalization**, which "dissolve anxiety … by representing deviant behaviors as thoroughly rational". Columbia: foam shedding "had occurred on every space shuttle flight" for 20 years and "its risk severity was steadily downgraded according to the illogical idea that 'if no accident has happened by now, it never will.'" Vaughan: disasters have "long incubation periods … discrepant events that accumulated unnoticed". |
| 7 | https://ieftimov.com/posts/testing-in-go-golden-files/ | 2026-08-16 | blog (named engineer, golden-file practice) | WebFetch, full | The core defect of borrowed fixtures: "the test files are not self-sufficient. This means that you will have to know the contents of another file to understand what is the expected outcome of your tests." Documents the `-update` regeneration flag and its caveat ("it must be available in all packages"). |

## Identified but snippet-only (does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://onlinelibrary.wiley.com/doi/full/10.1002/smr.70035 | peer-reviewed journal (JSEP 2025, 526 repos) | **HTTP 402 Payment Required** on WebFetch — paywalled. Snippet retained for the recency scan. |
| https://testing.googleblog.com/2015/01/testing-on-toilet-change-detector-tests.html | authoritative blog (canonical prior art) | Fetched **twice** (plain + `?m=1`); Blogger returned header/comments only, article body not in the served HTML. Not counted. |
| http://xunitpatterns.com/Fragile%20Test.html | book (Meszaros, xUnit Test Patterns — the canonical *Fragile Fixture* reference) | `connect ECONNREFUSED 52.1.13.203:443` — host down at access time. Not counted. |
| https://arxiv.org/abs/2411.11033 (REACCEPT) | preprint | Co-evolution automation; adjacent, superseded for this step by #4. |
| https://repository.ubn.ru.nl/bitstream/handle/2066/300272/300272.pdf | peer-reviewed (ICSME 2023, snapshot testing) | Binary PDF; per my own standing note PDF summarisation fabricates quotes, and re-extraction was not worth the budget at `moderate`. |
| https://ieeexplore.ieee.org/document/10336316/ | peer-reviewed (same ICSME paper) | Paywalled landing page. |
| https://dl.acm.org/doi/10.1145/3607183 | peer-reviewed (TOSEM, co-evolution identification, CHOSEN F1 0.928) | ACM paywall. |
| https://arxiv.org/pdf/2607.02469 (TestEvo-Bench) | preprint | `/pdf/` URL — forbidden as a primary fetch by `.claude/rules/research-gate.md`; no HTML render located within budget. |
| https://www.atlassian.com/incident-management/on-call/alert-fatigue | industry | Superseded by the SRE book (#3), same claim, higher tier. |
| https://www.minware.com/guide/anti-pattern/ignoring-flaky-tests | community | Lower tier; claim already covered by Fowler (#1). |
| https://jayschmidt.us/blog/deviancy-normalization/ | community | Normalization-of-deviance framing; superseded by the peer-reviewed #6. |
| https://engineering.verygood.ventures/development/testing/testing_golden_file/ | industry docs | Golden-file workflow; must be version-controlled. |
| https://github.com/oprypin/pytest-golden | tool docs | Offloads expected outputs to data files — the pattern under examination. |
| https://www.sitepoint.com/golden-master-testing-refactor-complicated-views/ | blog | Golden-master framing; Feathers' *characterization test*, Falco's Approval Tests. Lower tier than #7. |
| https://percy.io/blog/snapshot-testing | vendor blog | 2026-dated snapshot-testing framing; vendor-interested, not leaned on. |

---

## Recency scan (last 2 years, 2024-2026)

**Performed.** Two dedicated passes (queries #3 and #4 above), plus the
2026-dated frontier hits.

**Result: 3 new findings that SUPERSEDE the older canonical framing.**

1. **The 2026 literature names the exact failure class and shows it is
   majority-case, not edge-case.** TEBench (arXiv:2605.06125v1, 2026) is the
   first benchmark to separate *breaking* from **stale** tests and measures
   staleness in **65.9% of 314 real tasks** — where the older canonical
   material (Meszaros' *Fragile Fixture*, 2007; Google's change-detector post,
   2015) only ever asserted the pattern qualitatively.
2. **Detection is measured to be poor, which is new information.** Seven agent
   configurations scored **~36% F1** at identifying stale tests, with the
   paper's own diagnosis that tooling "rel[ies] almost entirely on execution
   failure signals". The corollary matters here: a stale fixture that turns a
   check **RED** is the *lucky* case — it at least produced an execution
   signal. The unlucky case is a stale fixture that keeps a check green.
3. **Test co-evolution is now studied at scale (JSEP 2025, 526 repositories)**
   and frames asynchronous test/production evolution as compromising "software
   quality and project longevity". Paywalled (HTTP 402), so held as
   snippet-only and not leaned on for any claim below.

**Not superseded:** Fowler's quarantine-with-a-bound prescription (2011),
Google SRE's "every page should be actionable", Banja's normalization-of-
deviance mechanism (2010, cross-domain) and the golden-file self-sufficiency
critique (ieftimov) remain the operative doctrine for the *dead-gate* and
*borrowed-fixture* halves. All four were reached via the **year-less canonical**
query variants precisely because a year-locked search buries them; nothing in
the 2024-2026 window replaces them, and the 2026 work (TEBench) supplies
prevalence numbers the older sources never had rather than contradicting them.

---

## Key findings

1. **A stale fixture and a real regression are indistinguishable from the exit
   code alone.** TEBench's definition is the precise one: staleness is when the
   *test's* relationship to the code has changed, independent of pass/fail —
   "no execution signal indicates that updates are needed"
   (arXiv:2605.06125v1). The diagnosis therefore has to come from *localising
   the failing assertion and driving its inputs*, never from reading the
   message. (This is exactly what 86.92's criterion 1 demands.)

2. **A red check that nobody acts on has already stopped being a check.**
   Fowler: "Once that discipline is lost, then a failure in the healthy
   deterministic tests will get ignored too. At that point you've lost the whole
   game" (martinfowler.com/articles/nonDeterminism.html). Google SRE states the
   positive form: "Every page should be actionable" and remove "signals that are
   collected but unused" (sre.google/sre-book/monitoring-distributed-systems).
   **Neither source treats "known red" as a stable state** — Fowler's remedy is
   quarantine *with a hard numeric or time bound*, precisely so that "known red"
   cannot become permanent.

3. **The tempting fix is the forbidden one, and the literature names it.** The
   Jest docs describe the failure mode verbatim: "fight against the habit of
   regenerating snapshots when test suites fail instead of examining the root
   causes of their failure"; blind regeneration "lock[s] in buggy behavior".
   Mapped to 86.92: relaxing `enforceGate` (or the fixture's expectations) so
   the old artifact passes is the `jest -u` move, and criterion 3 already
   forbids it.

4. **Self-owned synthetic fixtures beat borrowed real artifacts for
   *validator* tests.** The Jest/golden-file guidance is that a golden file is
   only sound when it is **owned and reviewed as code**; the community sources
   put the cost plainly — golden files "are not self-sufficient" (ieftimov). A
   fixture pointing at a *live, third-party-owned* artifact (here: another
   step's brief in `handoff/current/`) has no owner, no review trigger, and no
   reason to be updated when the validator's contract changes.

5. **"Known red" is not a stable classification — it is an incubation period.**
   The cross-domain source is the sharpest here: deviance normalises through
   *rationalization*, which "dissolve[s] anxiety … by representing deviant
   behaviors as thoroughly rational" (Banja, PMC2821100). Columbia's foam
   shedding "had occurred on every space shuttle flight" and "its risk severity
   was steadily downgraded according to the illogical idea that 'if no accident
   has happened by now, it never will.'" **The local instance is exact:** the
   RED was seen, correctly reproduced, and labelled "pre-existing, out of
   scope" by two independent Q/A spawns and by Main — each classification
   individually defensible, and collectively the downgrade. Vaughan's "discrepant
   events that accumulated unnoticed" is what a one-line disclosure in an
   `experiment_results` file looks like when it is never converted into a step.

6. **The specific defect is a *borrowed* fixture, and that is a named
   antipattern.** "The test files are not self-sufficient. This means that you
   will have to know the contents of another file to understand what is the
   expected outcome of your tests" (ieftimov). The checker's `verification`
   literal is worse than a golden file: it is a hand-written *stand-in* for an
   artifact produced by another agent under a schema owned by a third file, with
   no regeneration path at all (no `-update` equivalent). Every golden-file
   source assumes a regeneration mechanism exists; here none does, so the
   fixture could only ever be updated by someone remembering to.

7. **A green suite is not evidence the guards still work; mutation score is.**
   The 2026 survey names mutation testing as "a widely accepted standard for
   evaluating fault detection capability" and reports that test *usability*
   (70%+ pass) has outrun test *effectiveness* (arXiv:2511.21382v1). 86.92's
   criterion 5 ("a green checker whose mutants now survive is worse than a red
   one") is the same claim, and the measurement below shows it is not
   hypothetical here.

---

## Internal code inventory (all line numbers re-derived by direct read, 2026-08-16)

| File | Lines | Role | Status |
|------|-------|------|--------|
| `scripts/qa/verify_workflow_args_boundary.mjs` | 422 | The RED checker. Sections [1] REPRODUCE-from-git, [2] FIXED, [3] BLIND-CANNOT-PASS, [4] MUTATION, [5] FULL DRIVER | **RED: `FAILED: 84 passed, 3 failed`, exit 1.** Last touched `a212dfe9` **2026-08-09** |
| `.claude/workflows/research-gate.js` | 1101 | Layer-3 RESEARCH gate; `enforceGate` at `:567-755` | LIVE, correct. Last touched `0ecccafe` **2026-08-16** — 7 days *after* its checker |
| `.claude/workflows/qa-verdict.js` | 746 | Layer-3 EVALUATE gate; also driven by the checker | LIVE; its cells are all green |
| `handoff/current/research_brief_86.17.md` | 697 | Named by the step as "the stale fixture" | **NOT the cause — see the refutation below.** `grep -c brief_status` = **0** |
| `scripts/qa/verify_research_gate_workflow.mjs` | 1038 | The *other* checker for the same script | Green; does not cover the args boundary |
| `.claude/masterplan.json` | — | 86.17 `status: done`, 86.92 `status: pending` | 86.17's **immutable verification command is the now-RED checker** |

### The three failing assertions, localised by execution

```
FAIL [3] a healthy run with a perfect envelope PASSES
   -- ["brief at handoff/current/research_brief_86.17.md carries NO brief_status marker …",
       "over-claim: recency_scan_performed=true but the brief carries NO dedicated recency-scan section …",
       "over-claim: urls_collected=40 but only -1 distinct URLs appear in the brief …"]
FAIL [3] no regression: enforceGate without inputHealth behaves as before
FAIL [4] drop-blind-violation: KILLED (a blind run would pass without it)
```

All three are driven by **one** object literal — the `verification` fixture at
`scripts/qa/verify_workflow_args_boundary.mjs:179` (and its clone at `:319`):

```js
const verification = { brief_exists: true, brief_non_empty: true, char_count: 9000, urls_missing: [] }
```

`BRIEF_VERIFICATION_SCHEMA` (`research-gate.js:529-553`) requires **nine**
fields. Three of them — `recency_section_present`, `distinct_urls_in_brief`
(both added phase-86.28) and `brief_status_in_brief` (added phase-86.37) — are
absent from the fixture. `enforceGate` fail-closes on each, exactly as designed
(`:693-707`, `:727-732`, `:738-744`).

### REFUTATION: the step's own premise names the wrong artifact

The step title, and the phase-86.90 Q/A that filed it, attribute the RED to
`handoff/current/research_brief_86.17.md` lacking a `brief_status` marker.
**That is mechanically false and I measured it rather than reading it off the
message.** `enforceGate` is pure — no I/O (`research-gate.js:556-558`: "PURE:
no I/O, no Node APIs") — and the checker never opens that brief; its
`readFileSync` calls (`:136, :166, :278, :303, :380, :396, :404`) read only the
two workflow scripts. The path at `:177` is an inert **string**.

Driving `enforceGate` directly
(`<scratchpad>/proof.mjs`):

| Case | `brief_path` | `verification` | Result |
|---|---|---|---|
| A | `handoff/current/research_brief_86.17.md` | the checker's literal | `gate_passed=false`, 3 violations |
| B | `/nonexistent/NEVER_EXISTED.md` | the checker's literal | `gate_passed=false`, 3 violations — **violation array byte-identical to A** |
| C | `handoff/current/research_brief_86.17.md` | + the 3 missing fields | **`gate_passed=true`, 0 violations** |
| E | `'p'`, no `inputHealth` (the "legacy" cell) | the checker's literal | `gate_passed=false`, 3 violations |

So: **deleting the 86.17 brief entirely changes nothing; adding three fields to
the literal turns the checker green.** Fixing the file — the remedy the step's
name implies — would leave the checker RED. This matters for criterion 1
("localised by execution … rather than inferred from the message text") and it
inverts criterion 4: the durable fixture is not "a brief that cannot rot", it is
**a `verification` literal owned by the checker and derived from the schema.**

### The `-1` (criterion 2): a deliberate sentinel, rendered as a count

`enforceGate:632` defines `const n = (v) => (typeof v === 'number' &&
Number.isFinite(v) ? v : -1)`. It is the **same** coercion already applied to
`sources` (`:633`) and `urls` (`:634`), so `-1` is a documented "absent or
non-finite" sentinel, **not** a second arithmetic defect — measured directly:
`n(undefined) -> -1`. What *is* defective is the **message**: `:740` interpolates
the sentinel into prose that reads as a count ("only -1 distinct URLs appear in
the brief"), which is what made a fixture-shape problem look like a
brief-content problem — and is plausibly how the 86.90 Q/A reached the wrong
artifact. That is a separable one-line reporting fix (distinguish "absent" from
a count), not a loosening of the rule.

### The mutation cell that fixture rot has DISABLED (criterion 5)

`[4] drop-blind-violation` (`:302-323`) mutates `if (inputBlind) {` →
`if (false) {` and asserts the mutant then lets a blind run pass
(`blind.gate_passed === true`). It uses the **same stale literal** (`:319`), so
the mutant's `gate_passed` is held at `false` by the three unrelated violations
and the cell reports FAIL. Measured: with the fixture repaired, case D
(blind + fresh fixture) returns exactly one violation —
`dry_run_no_step_id` — so the cell becomes able to discriminate again.
**Right now that cell cannot kill anything: it would report FAIL whether the
blind guard is present or absent.** It is a false alarm masking a dead cell —
the "green checker whose mutants survive" hazard from criterion 5, in its
red-flavoured variant.

### Blast radius (criterion 6)

- The checker last changed **2026-08-09** (`a212dfe9`); `research-gate.js`
  changed as late as **2026-08-16** (`0ecccafe`). The gate has been red since
  the phase-86.37 marker landed — i.e. the *validator* moved three times
  (86.28 → 86.37 → 86.90) while the fixture stood still. Exact red-onset commit
  should be pinned by `git log -S` on the fixture literal during GENERATE;
  I did not pin it and do not assert a day count.
- **86.17 is `status: done` and its immutable verification command is
  `node scripts/qa/verify_research_gate_workflow.mjs && node
  scripts/qa/verify_workflow_args_boundary.mjs`** — which today exits 1. A
  closed step's own re-runnable evidence no longer reproduces.
- The RED was **observed and correctly disclosed** in at least three 86.90
  artifacts (`experiment_results_86.90.md:424`, `evaluator_critique_86.90.md`,
  `live_check_86.90.md:297`) and by two independent Q/A spawns — and was
  *still* carried as "pre-existing, out of scope" for a full cycle. That is
  Fowler's mechanism observed in this repo, not in the abstract: the signal was
  seen, classified as known-red, and routed around.

### A second-order dead gate, in this step's own filing

86.92's **immutable** verification command is
`bash -c 'node --check scripts/qa/verify_workflow_args_boundary.mjs && echo
parses'`. `node --check` is a **parse** check: it exits 0 on the checker as it
stands today, RED and all. The step whose entire purpose is to make a checker
green can therefore be verified green **without ever running it**. The
`live_check` field does demand the post-fix exit-0, so the evidence path exists
— but the machine-checkable half does not test the property. CLAUDE.md already
records this exact hazard for the sibling script ("the immutable `node --check`
command reaches criterion 1 only"). Criteria are immutable; the mitigation is
to make `live_check_86.92.md` carry the verbatim exit-0 run, and to say plainly
in the contract that the command does not prove the fix.

---

## Consensus vs debate (external)

**Consensus.** (a) Expected-value fixtures must be owned and reviewed as code,
not treated as data that drifts in (Jest docs; golden-file practice). (b) A
failing signal nobody acts on degrades every other signal (Fowler; Google SRE —
independently, from testing and from ops). (c) Detecting staleness from
execution alone is unreliable (TEBench, ~36% F1).

**Debate.** Whether to fix a stale fixture *in place* or regenerate it. The
snapshot camp regenerates but demands human review of the diff; the
xUnit-patterns camp argues a fixture that needs regenerating on every contract
change was **overspecified** and should assert less. For 86.92 the second view
is the stronger one: the fixture asserts a *whole* `verification` object when
what section [3] actually cares about is "a healthy run passes" — so
constructing it from a schema-derived, checker-owned helper (a minimal valid
object) is both a fix and a de-overspecification. This is the split criterion 4
is asking to be resolved explicitly.

**Non-consensus / gap.** No source I read prescribes what to do when the stale
fixture sits *inside a mutation cell*. That combination — where rot converts a
kill into a permanent false FAIL — I did not find named in the literature, and
it is the sharpest local finding.

## Pitfalls (from literature, mapped)

1. **`jest -u` reflex** — "regenerating snapshots when test suites fail instead
   of examining the root causes" (Jest docs). Local form: widening
   `enforceGate`. Forbidden by criterion 3.
2. **Quarantine without a bound** — Fowler requires a numeric or time cap;
   Banja supplies the mechanism for what happens without one (rationalization →
   "if no accident has happened by now, it never will"). Local form:
   "pre-existing, out of scope" repeated across cycles with no step filed.
   86.92 *is* the bound arriving; it should not be re-deferred.
2b. **A fixture with no regeneration path can only rot.** Every golden-file
   source assumes an `-update`-style mechanism (ieftimov). The checker has
   none — so "keep the literal, just add three fields" repairs the instance and
   leaves the class intact. Derive it from
   `BRIEF_VERIFICATION_SCHEMA.required` instead, so the *next* field addition
   fails one builder loudly rather than three assertions obscurely.
3. **Fixing the artifact the message names** — TEBench's whole point is that
   the execution signal under-determines the cause. The message named a brief;
   the cause is a literal.
4. **A green suite as evidence** — usability has outrun effectiveness
   (arXiv:2511.21382v1). After the fix, re-score the mutation cells; do not
   read exit 0 as proof.
5. **Magic-number/hard-coded fixtures** (`magic number test` smell, same
   survey). The `char_count: 9000` and `urls_collected: 40` literals are the
   same species as the three missing fields; a helper that builds a minimal
   valid `verification` from `BRIEF_VERIFICATION_SCHEMA.required` removes the
   whole class, not this instance. (Repo doctrine agrees: *a guard from the
   instance is not a guard against the class*.)

## Application to pyfinagent

| Finding | Anchor | Implication for the contract |
|---|---|---|
| Cause is the literal, not the brief | `verify_workflow_args_boundary.mjs:179`, `:319` | Criterion 1 is satisfiable only by driving `enforceGate`; the step's own title mis-names the artifact and the contract should say so |
| Schema is the source of truth | `research-gate.js:529-553` (`required` = 9 fields) | Criterion 4's "durable" answer: build the fixture **from `required`**, so the next added field breaks the *builder* once, in one place, loudly |
| Sentinel is real, message is not | `research-gate.js:632`, `:740` | Criterion 2: document the sentinel; file the message rendering as its own defect rather than folding it in |
| One mutation cell is currently dead | `verify_workflow_args_boundary.mjs:302-323` | Criterion 5: after the fix, show this cell KILLS again — it is the one the rot disabled |
| Closed step 86.17's evidence no longer reproduces | `.claude/masterplan.json` 86.17 `verification.command` | Criterion 6's blast radius has a concrete, checkable instance |
| 86.92's own command cannot fail on this defect | `.claude/masterplan.json` 86.92 `verification.command` (`node --check`) | Disclose in the contract; put the real run in `live_check_86.92.md` |

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch — **7**: Fowler, Jest docs, Google SRE book, arXiv:2605.06125v1, arXiv:2511.21382v1, PMC2821100 (peer-reviewed, cross-domain), ieftimov. Tier mix satisfies the hierarchy: 1 peer-reviewed, 2 preprint, 2 official docs, 2 authoritative blogs — **zero community-tier** in the read-in-full set.
- [x] 10+ unique URLs total (**22** distinct URLs, machine-counted in this file)
- [x] Recency scan (last 2 years) performed + reported (3 superseding findings)
- [x] Full pages read, not abstracts, for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in the caller's scope
- [x] Contradictions / consensus noted (incl. one claim REFUTED by measurement)
- [x] Claims cited per-claim
- [ ] **Gap disclosed:** three high-value sources could not be fetched
  (Wiley `402 Payment Required`, xunitpatterns `ECONNREFUSED 52.1.13.203:443`,
  Google Testing Blog body not served on two attempts). The canonical
  *Fragile Fixture* reference (Meszaros) is therefore snippet-only; **no claim
  above rests on it** — the golden-file half is carried by #7 read in full.
- [ ] **Gap disclosed:** the exact red-onset commit is not pinned (`git log -S`
  on the fixture literal, to run during GENERATE).
