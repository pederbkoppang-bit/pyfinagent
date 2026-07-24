# Research Brief — Step 75.18 (tier=complex)

**Codify the anti-vacuous-guard doctrine into the harness itself.**
Boundary: harness/docs only, no product code. Separation of duties: this
step edits `.claude/agents/qa.md`; the authoring session must NOT
self-evaluate any later step that depends on it — Main leaves a Peder
review note in `handoff/harness_log.md` (criterion 6).

Status: COMPLETE. Written 2026-07-24. gate_passed = true (6 sources read
in full, recency scan performed, taxonomy built from primary critiques).

---

## 1. Complete instance taxonomy — the core deliverable (with cycle citations)

Phase-75 produced ONE root cause — *a guard that cannot fail when its
subject is broken* — that changed shape every time. The step text
enumerates through instance (6); the session since (Cycles 135, 142, 143)
produced five more shapes. All eleven are grounded in the archived Q/A
critiques (verbatim captured return values), not inferred.

| # | Shape | Where (cycle / phase / criterion) | Concrete mechanism | Caught by |
|---|-------|-----------------------------------|--------------------|-----------|
| 1 | **Source-scan, dual-occurrence** | C129 / 75.3 / C6 | `"unwrap_secret" in SIGNALS_SRC` — the literal is on BOTH the import line (:495) and the call site (:497), so reverting ONLY :497 leaves the scan green while security-05 fully regresses | Q/A behavioral probe (fake WebClient captured `xoxb-real-token`) |
| 2 | **Source-scan, reworded-regression** | C129 / 75.3 / C3 | exact-literal scan for `resp["published"] = True`; a reworded `resp.update({"published": True})` (or a renamed var) evades the grep with the suite green | Q/A end-to-end re-fire through `publish_signal` |
| 3 | **Source-scan, literal-kept-behavior-stripped** | C129 / 75.3 / C4 | keep `"stub": True` in the dict but `pop("stub")` off every candidate before return → all 27 tests green while criterion 4 regresses | Q/A drove real tool via `create_signals_server()`+fastmcp client |
| 4 | **Tautology** | C130 / 75.2.1 (`assert wired["push"] is not None`, fixture already guarantees it) **and** C142 / 75.14:199 (`assert ... or True`, unconditionally passes) | assertion is a logical constant given the fixture / the `or True` short-circuit | Q/A read the test body; would pass with handler registered AFTER the catch-all |
| 5 | **Fixture-cannot-represent-failure** | C130 / 75.2.1 / C3 | `AsyncSay` returns `AsyncSlackResponse` (exposes `.get()`, **not** a `dict` subclass); production `isinstance(resp, dict)` registered nothing and left the push path INERT, while 22 tests passed because the `_say` stub returned a plain dict | Q/A replayed real `AsyncSlackResponse` type; **mutation of production alone MISSED it** |
| 6 | **Library-fact assertion posing as a fixture pin** | C130 / 75.2.1 / C6 | the guard added to fix #5 asserted only `not issubclass(AsyncSlackResponse, dict)` — an upstream-library fact, never references the `_FakeSlackResponse` stub it claims to pin; M9: regress stub→dict AND restore isinstance bug together → suite GREEN, production inert | Q/A M9 mutation of the **test harness itself** (never mutated by author) |
| 7 | **Re-implemented test (behavioral-looking test exercising a COPY of the logic)** | C142 / 75.14 / C6 | `test_fallback_flag_routes_verdict` re-implements the parse-fail warning inline via a monkeypatched `SimpleNamespace(get_settings)` instead of executing the real `run_risk_debate` branch → a money-path DARK-flag **if/else routing inversion** (True→REJECT vs False→APPROVE_REDUCED) passes the entire 18-test suite undetected | Q/A; fixed by EXTRACTING `_judge_parse_fail_fallback()` and executing the REAL fn both ways |
| 8 | **OR-escape-hatch / comment-token trap** | C143 / 75.15 / C5 | `('blocks the PR' not in s) or ('run_seed_stability' in s)` — the SAME diff ADDED `run_seed_stability` to the yaml COMMENT (line 20), so the 2nd clause is permanently true; re-introducing `blocks the PR` leaves the guard GREEN. (75.15 also had 2 MORE comment-token survivors M2/M6, self-caught) | Q/A mutation: overclaim re-added + token present → guard STILL passes |
| 9 | **Executor-environment / shell-and-state dependence** | C135 / 75.7 (×3), C132 / 75.5, C129 / 75.3, C130 / 75.2.1 | (a) unquoted `$(git diff --name-only …)` **word-splits in zsh** → lints only tracked files, misses the untracked new test file; (b) unquoted `$FILES` → newline-joined blob → `ruff` "No such file" → **exit 0 over ZERO files** ("All checks passed!"); (c) stale `backend` pkg in `.venv/site-packages` shadows the repo module (75.3 drill dark); (d) process-state pollution: an M10 "pass" only because a prior pytest run pre-bound the test module's `cmd` global (75.2.1); (e) `.env` flag-state flips raw pass/fail counts (75.14/75.15) | Q/A re-ran in a clean process / explicit-arg + `${=FILES}` scope; the qa.md §1a empty-set guard exists for exactly this |
| 10 | **Hand-derived-scope staleness** | C142 / 75.14 ("8 files" vs actual 9), C130 / 75.2.1 ("22 passed" vs 24; commands.py 189 vs 218 lines), C132 / 75.5 (hand-typed 10-file lint list omitting 4 of 14), C135 / 75.7 (regression 1304 vs 1305) | a list/count/scope captured BY HAND goes stale against a later edit; the "verbatim" artifact no longer reproduces | Q/A re-ran the deriving command (`git diff --stat`, `pytest -q`) |
| 11 | **Mis-attributed kill-mechanism** | C135 / 75.7 / C1 | mutation M1 IS killed — but by the completion assertion, NOT the credited `RuntimeWarning-as-error+gc.collect` leg (which is inert/swallowed as "Exception ignored while finalizing coroutine"). The guard is non-vacuous *today*, but the RECORDED reason is wrong, so a future maintainer trusting the credited leg could delete the load-bearing assertion and silently make it vacuous | Q/A re-ran M1 in-memory and observed the real failing assertion |

**Meta-pattern (the reason the rule must be about *mutation*, not a code
smell):** #5 was missed by mutation because the author mutated production
while leaving the broken stub in place; #6 was missed because the author
never mutated the **test harness itself**. Every remaining shape (#7 copy
of logic, #8 comment token, #9 environment, #10 stale scope, #11 wrong
kill reason) is a case where the SUBJECT the guard names and the THING the
guard actually observes have silently diverged.

**Companion doctrine (measure-before-assert; leg (d)) — distinct axis:**
Phase-75 shipped **three false checkable claims**, each one command from
verification: (i) DEBUG env state — "`DEBUG=true`" when `settings.debug`
was `False` (75.1); (ii) verb parity — "covers what the deleted
`handle_deploy_command` matched", missed 12 surfaces incl. bare `deploy`
(75.2); (iii) handler-registration order — written in FIVE places that the
operator-token handler registers first, when the push handler is idx 0
(75.2.1). Generalized root cause (75.5, 11 instances, `wf_b550e771-aa7`):
**a claim about a SET whose membership rule was never written down
operationally** — the code was correct every time; the harness had no
instrument on the CLAIMS. Sub-rule: a literal a criterion **source-scans
for** must not survive in explanatory COMMENTS either (3 instances: CGNAT
pattern, date cutoff, dead-branch name).

---

## 2. Per-file codification map (legs a–d → files; NO duplication, cross-link)

House convention (`research-gate.md` "Cross-references"): each rule lives
in ONE file; the others cross-link. Existing split: `qa.md` = evaluator
mechanics; `per-step-protocol.md` = orchestration sequence; the skill =
ranked code-review heuristics + severity dispatch + negation lists;
memories = operator-ratified wording. Map the four legs onto that split:

| Leg | Lands in | Concrete edit | Satisfies criterion | Cross-links (don't duplicate) |
|-----|----------|---------------|---------------------|-------------------------------|
| (a) per-criterion vacuity check | **`qa.md`** new **§4c "Guard-vacuity check"** (sits after §4b claim-auditing) | For EACH immutable criterion, NAME the concrete mutation that would make its guard fail; apply it (or show why unreachable); **"no such mutation exists" is a FINDING (`Circular_Reasoning`/`Missing_Assumption`), never a pass.** | C1 | → skill `illusory-guard` heuristic (owns the ranked shapes); → `feedback_mutation_test_guards_and_fixtures` |
| (b) mutation must cover the FIXTURE | **`qa.md` §4c** (mechanics) + **`per-step-protocol.md` §4** (one-line requirement) | qa.md: "mutation evidence MUST cover the test FIXTURE/stub as well as the code under test — assert on the object the fixture *actually returns*, not the library type it imitates. Cite 75.2.1: mutating production alone left a dict-returning stub undetected." per-step-protocol: one sentence + cross-link. | C2 | per-step-protocol → qa.md §4c (mechanics live there) |
| (c) ranked illusory-guard heuristic | **`.claude/skills/code-review-trading-domain/SKILL.md`** Dimension 4 + Top-list #17 | New ranked entry `illusory-guard` naming the **four required shapes** (source-scan-only, tautology, non-representative fixture, library-fact-assertion-posing-as-fixture-pin) with **severity dispatch** (see §2a below); RECOMMEND adding #7 re-implemented-test + #8 OR-escape-hatch as sub-shapes (criterion is a floor). | C3 | generalizes existing `tautological-assertion`/`over-mocked-test`/`rename-as-refactor` — reference, don't duplicate |
| (d) measure-before-assert + comment-literal rule | **`per-step-protocol.md`** new short subsection "Measure before asserting" | (i) a checkable factual claim must be MEASURED before it enters any artifact (contract/results/comment/commit) — 3 examples (DEBUG 75.1, verb-parity 75.2, handler-order 75.2.1); (ii) a literal a criterion source-scans for must NOT appear in explanatory comments (3 instances). | C4 | → `feedback_measure_dont_assert_claims`; qa.md §4b already enforces the reproduce-side — cross-link, don't restate |
| C5/C6 | git-diff scope + harness_log note | no product-code file touched (assert via `git diff --stat`); **Main** appends the Peder-review-of-qa.md note to `harness_log.md` | C5, C6 | CLAUDE.md "Separation of duties on agent edits" |

### 2a. Severity dispatch for the `illusory-guard` heuristic (leg c)

Align to the skill's existing BLOCK/WARN/NOTE table:
- **BLOCK** when the illusory guard is the **SOLE** coverage for a
  behavioral/money-path criterion (shapes 5, 6, 7, 8; and a source-scan
  that is the only guard for a runtime behavior).
- **WARN** when a real behavioral guard exists alongside but the scan is
  **mislabeled** as behavioral (the 75.3 rename-the-overclaiming-test fix).
- **Negation list (legitimate, do NOT flag):** a source scan that is
  **criterion-mandated verbatim** ("source no longer contains X") when
  **paired** with a behavioral guard; a scan of a **statically unreachable
  dead branch** where no runtime behavior exists to observe (75.3
  `compute_dsr_real` — a scan is the only possible guard). This negation
  list is REQUIRED so the heuristic itself is not over-broad.

---

## 3. Vacuity analysis of 75.18's OWN criteria + verification command

A doctrine step against vacuous guards must not ship vacuous guards. The
immutable `verification.command` is **itself three of the anti-patterns**:

1. `assert 'mutation' in qa.lower()` — **ALREADY TRUE on the unmodified
   qa.md** (measured: "mutation" appears on 4 lines today). This half-clause
   passes regardless of whether 75.18 does anything — the family-#3
   "literal already present" shape.
2. `'fixture' in qa.lower()` / `'fixture' in pr.lower()` — single-token
   source-scan (family #1). Measured: "fixture" is **absent** from both
   files today, so it forces *a* change — but adding the bare word
   anywhere satisfies it, WITHOUT the doctrine being present or correct.
3. `'illusory' in sk or 'vacuous' in sk` — **OR-escape-hatch** (family #8),
   satisfiable by EITHER token alone, anywhere (incl. a comment).

**Prescription:** the command is immutable and is a *necessary-not-
sufficient* smoke check — keep it, do not edit it. The REAL gate is the six
prose `success_criteria`, which are semantic ("qa.md **instructs the
evaluator** to name the concrete mutation…"). The 75.18 Q/A must satisfy
them the way `scripts/qa/verify_qa_roster_live.sh` satisfies "the section
is live": that script's `grep -qF` step is necessary, but its **step 3 is a
behavioral self-disclosure** that the section is actually operative.
The behavioral analog for a doctrine edit is a **known-member recall test
(qa.md §4b)**: take ≥1 *already-known* phase-75 vacuous guard from the
taxonomy above (e.g. instance #3 pop-key, or #8 seed OR-hatch) and confirm
the NEW qa.md §4c / skill heuristic, read as written, **would flag it**. A
doctrine that cannot catch its own corpus is `Threshold_Not_Met`. Criterion
C5 ("no backend/ or frontend/ file changed") is non-vacuously checkable by
`git diff --name-only HEAD | grep -E '^(backend|frontend)/'` returning empty.

---

## 4. Separation-of-duties mechanics for the 75.18 Q/A spawn

**The edited qa.md evaluates its own edit.** On the primary Workflow path
(`qa-verdict.js`) the Q/A "**Read[s] this file (qa.md) from disk at
runtime**" (qa.md:70) — so the moment Main writes §4c, the very next Q/A
spawn is bound by the rubric under review (unlike the Agent-tool path,
which snapshots at session start). Prescription for Main's spawn:

1. **Evaluate the DIFF against the doctrine GOALS**, not the token-scan.
   Feed the Q/A the six criteria + the two memory files' operator-ratified
   wording; instruct it to confirm the PROSE instructs each named behavior
   (read, don't grep).
2. **Flag the self-reference explicitly.** The spawn prompt must tell the
   Q/A it is grading a rubric that now includes the edit under review, and
   have it apply the NEW §4c vacuity check to the doctrine's own guards
   (the recall test in §3).
3. **Hold-flip-for-review disposition (per CLAUDE.md separation-of-duties +
   the 75.5 precedent).** Doer/judge separation holds for 75.18 itself
   (Main authors, a SEPARATE Q/A agent evaluates), so 75.18 may flip to
   `done` on a genuine PASS. The binding constraint is FORWARD: **Main
   appends a note to `harness_log.md` requesting Peder review of the qa.md
   edit, and NO later step may depend on the new rubric until that review
   lands** (criterion C6; mirrors 75.5's operator-approval gate). Do not let
   the auto-commit sweep the edit into a step that silently relies on it.

---

## 5. External research

### Read in full (6; ≥5 floor met — counts toward the gate)

| # | Source | Accessed | Kind | Fetched how | Key finding |
|---|--------|----------|------|-------------|-------------|
| 1 | Vera-Pérez, Monperrus, Baudry — *A Comprehensive Study of Pseudo-Tested Methods* (arXiv:1807.05030) | 2026-07-24 | peer-reviewed (EMSE) | ar5iv HTML | "A method is **pseudo-tested** … if the test suite **covers** the method and **does not assess any of its effects**" — covered yet no test fails when the body is stripped; present in ALL subjects even high-coverage, ratio 1%–46% |
| 2 | Betka & Wagner — *Extreme Mutation Testing in Practice* (arXiv:2103.08480) | 2026-07-24 | peer-reviewed | ar5iv HTML | 14% of methods pseudo-tested (291/2041); root causes = **weak tests with NO assertions (8)**, incomplete (3), side-effect-only (14); method-level ("extreme") mutation is the practical unit — 13 min vs 37 min |
| 3 | *Agentic Uncertainty Reveals Agentic Overconfidence* (arXiv:2602.06948) | 2026-07-24 | preprint (2026) | arxiv HTML | "some agents that succeed only **22%** of the time predict **77%** success"; "**5.5× more likely to confidently predict success on a failing task** than to doubt a successful one"; post-execution info WORSENS calibration (anchors on surface plausibility) |
| 4 | *ReVeal: Self-Evolving Code Agents via Reliable Self-Verification* (arXiv:2506.11442) | 2026-07-24 | preprint (2025) | ar5iv HTML | intrinsic self-verification → "verbose, blind self-reflection or random guessing"; reliable verification is **grounded in executable tool feedback** (Python interpreter execution), NOT model introspection |
| 5 | Shai Yallin — *Fake, Don't Mock* | 2026-07-24 | practitioner blog | direct HTML | a hand-rolled stub "might return what the test expects while the **real implementation behaves differently**"; a **fake verified by a contract test against the real type** stays faithful — the exact remedy for instances #5/#6 |
| 6 | CircleCI — *What is mutation testing?* | 2026-07-24 | vendor eng blog | direct HTML | runtime = mutants × suite-runtime (1000 mutants × 60s ≈ 16h); "**Run mutation tests on a schedule, not on every commit**"; scope to changed files; 60–80% target, not 100% |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| Descartes tool-demo arXiv:1811.03045 | paper | superseded by #1 (same authors, fuller study) |
| IEEE 10818231 / ACM 3701659 Python mutation-tool comparison | paper | tool-selection detail; snippet gave mutmut 88.5% vs cosmic-ray 82.7% |
| johal.in mutmut-2026 guide | blog | HTTP 403; CI-policy crux captured from search snippet |
| qaskills cosmic-ray / mutmut guides | blog | community tier; CI policy already covered by #6 |
| Wiley STVR 2024 competent-programmer-hypothesis | paper | coupling-effect background; covered by search synthesis |
| ceaksan LLM agentic failure modes; Zylos LLM-as-judge 2026 | blog | corroborate #3/#4 (self-eval unreliable, cross-eval needed) |
| arXiv:2601.19088 Hybrid Fault-Driven Mutation Testing for Python | paper | future-tooling; not needed for the manual-matrix verdict |

### Recency scan (2024–2026) — performed

Searched 2026-frontier + 2025 windows. **New findings that supersede/
complement the canon:** (a) *Agentic Overconfidence* (2026) quantifies the
self-eval blind spot the no-self-eval rule assumes — a fresh, citable
anchor for "agents confidently praise their own work"; (b) *ReVeal* (2025)
gives the execution-grounded-verification principle that justifies qa.md's
deterministic-FIRST order and the manual mutation matrix (execute the real
branch — instance #7's fix); (c) mutmut/cosmic-ray 2026 practice converges
on "**no new surviving mutants in protected modules**, not a 100% gate,"
and "don't assert implementation trivia to kill an equivalent mutant." The
canonical DeMillo/Lipton/Sayward coupling-effect (1978) and
Goodenough-Gerhart (1975) remain valid; the 2024 STVR reproduction confirms
the coupling effect holds while the competent-programmer hypothesis is only
partially supported — which is exactly why the doctrine targets *mutation
detection*, not a code smell.

### Key findings applied to pyfinagent

1. **"Pseudo-tested" is the academic name for the whole phase-75 root
   cause.** Definition #1 ("covers the method, does not assess its
   effects") IS instances #1–#8. The literature's detection method —
   **strip the body / mutate and require a test to fail** — is precisely
   the qa.md §4c "name the mutation that makes the guard fail" rule.
   (Source: 1807.05030 §2.1.)
2. **Mutate the harness, not just the code.** #5/#6 are the "side-effect
   methods / weak tests with no assertions" category (2103.08480) one level
   up: the *test double* had no fidelity. Remedy = contract-test the fake
   against the real type ("Fake, Don't Mock") → criterion C2's fixture-
   mutation rule.
3. **Self-verification must be execution-grounded (ReVeal) and agents are
   systematically overconfident (2602.06948, 5.5× asymmetry).** This is the
   external backbone for keeping Q/A a SEPARATE, deterministic-first agent
   and for the "no such mutation exists = FINDING" rule — the evaluator
   must EXECUTE the mutation, never reason that a guard "looks
   behavioral."
4. **Tool-adoption verdict: the manual matrix is right for this repo; do
   NOT gate CI on mutmut/cosmic-ray.** Mutation runtime = mutants × suite
   (CircleCI: ~16h for 1000 mutants); the field consensus is scheduled/
   scoped runs, not a per-PR blocking gate. pyfinagent's Q/A is a
   rare-event, per-step, read-only evaluator that already mutates the
   *specific* guards under review (scoped-to-the-diff by construction) —
   this is the "extreme/scoped" mutation the literature endorses, done by
   hand. RECOMMENDATION: keep the manual per-criterion matrix as doctrine;
   OPTIONALLY queue a *separate* future step for a nightly `mutmut` diagnostic
   (not a gate) on protected modules (`paper_trader.py`, `risk_engine.py`,
   `perf_metrics.py`) — out of scope for 75.18 (harness/docs only).

### Consensus vs debate

Consensus: pseudo-tested methods are common even at high coverage;
mutation (not coverage) measures assertion quality; self-eval is
unreliable and must be externally grounded. Debate: competent-programmer
hypothesis only partially holds (2024 STVR vs Gopinath) — doesn't affect
the doctrine, which relies on the *coupling effect* (well-supported).

### Pitfalls (from literature)

- Killing a mutant by **exception** (crash) rather than **assertion** can
  still be weak (mutmut research: "killed by exception" ≠ meaningful
  assertion). qa.md §4c should prefer assertion-kills for behavioral guards.
- **Equivalent mutants**: "no such mutation exists" can be legitimate for a
  statically-unreachable branch (75.3 dead `compute_dsr_real`) — the
  negation list (§2a) must carve this out or the rule over-fires.
- Do NOT "assert implementation trivia to kill an equivalent mutant" — that
  makes the suite worse (CircleCI/mutmut). The rule is *behavioral effect*,
  not line-coverage.

---

## 6. Internal code inventory

| File | Lines | Role | Status for 75.18 |
|------|-------|------|------------------|
| `.claude/agents/qa.md` | 447 | Q/A evaluator prompt; §4b claim-auditing already present; §4 anti-rubber-stamp mutation bullet | EDIT: add §4c (legs a,b). Contains "mutation" (4×), NO "fixture" |
| `docs/runbooks/per-step-protocol.md` | 353 | Orchestration sequence; §4 EVALUATE | EDIT: fixture line in §4 (leg b) + "Measure before asserting" subsection (leg d). NO "fixture" today |
| `.claude/skills/code-review-trading-domain/SKILL.md` | 244 | Ranked heuristics; Dim-4 has `tautological-assertion`/`over-mocked-test`/`rename-as-refactor` | EDIT: add `illusory-guard` #17 + severity + negation (leg c). NO "illusory"/"vacuous" today |
| `scripts/qa/verify_qa_roster_live.sh` | 83 | grep + git + behavioral self-disclosure | REFERENCE model for non-vacuous heading verification (§3) |
| `feedback_mutation_test_guards_and_fixtures.md` | memory | operator-ratified wording for legs a/b/c | ALIGN qa.md §4c + skill to this verbatim; it names 75.18 as the codification step |
| `feedback_measure_dont_assert_claims.md` | memory | operator-ratified wording for leg d | ALIGN per-step-protocol subsection to this |

Boundary check (C5): all four edit targets are under `.claude/` or
`docs/` — zero `backend/` or `frontend/` files. `git diff --name-only HEAD
| grep -E '^(backend|frontend)/'` must return empty.

---

## 7. Research Gate Checklist

Hard blockers — all satisfied:
- [x] ≥5 authoritative external sources READ IN FULL via WebFetch (6)
- [x] 10+ unique URLs total (≈50 across 5 searches + 6 fetches)
- [x] Recency scan (2024–2026) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (critiques cited by cycle)

Soft checks:
- [x] Internal exploration covered qa.md, per-step-protocol, skill, memories, verify script, 4 primary critiques + harness_log map
- [x] Contradictions / consensus noted (competent-programmer debate)
- [x] Claims cited per-claim

---

## 8. JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 7,
  "urls_collected": 50,
  "recency_scan_performed": true,
  "internal_files_inspected": 13,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 5,
    "dry": false
  },
  "summary": "Built the complete 11-shape guard-vacuity taxonomy (families 1-6 from 75.3/75.2.1 per the step text, plus 7 re-implemented-test/75.14, 8 OR-escape-hatch/75.15, 9 executor-environment/75.7+75.5, 10 hand-derived-scope-staleness, 11 mis-attributed-kill-mechanism/75.7) with verbatim-critique cycle citations, plus the measure-before-assert companion doctrine (3 false claims + comment-literal rule). Mapped the four codification legs onto qa.md new sec4c (per-criterion mutation naming + fixture-mutation), per-step-protocol sec4 + a Measure-before-asserting subsection, and the code-review skill illusory-guard ranked heuristic with BLOCK/WARN severity + negation list -- each in ONE file, cross-linked. External: pseudo-tested-methods (Vera-Perez), extreme mutation in practice, agentic overconfidence 2026 (22->77pct, 5.5x asymmetry), ReVeal execution-grounded verification, Fake-Dont-Mock fixture fidelity, CircleCI mutation cost. Tool-adoption verdict: KEEP the manual per-criterion matrix (scoped-by-construction), do NOT gate CI on mutmut/cosmic-ray (runtime=mutants x suite). CRITICAL: 75.18's own verification command is itself vacuous -- 'mutation' already in qa.md, and the skill clause is an OR-escape-hatch -- so the six prose criteria are the real gate and must be met via a known-member recall test against the phase-75 corpus. Separation of duties: the Workflow Q/A reads qa.md from disk at runtime so it grades its own edit; flag the self-reference, evaluate the diff vs goals, and Main holds forward-dependence for Peder review.",
  "brief_path": "handoff/current/research_brief_75.18.md",
  "gate_passed": true
}
```
