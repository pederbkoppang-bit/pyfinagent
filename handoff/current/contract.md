# Contract -- Step 75.18: codify the anti-vacuous-guard doctrine into the harness itself

- **Step id**: 75.18 (phase-75, Audit75 S18) -- **opus-tier -> Main GENERATE directly**.
- **Date**: 2026-07-24
- **BOUNDARY**: harness/docs only (qa.md, docs/runbooks/per-step-protocol.md, .claude/skills/code-review-trading-domain) -- ZERO product code (proven: `git diff --name-only HEAD | grep -E '^(backend|frontend)/'` empty). **SEPARATION OF DUTIES**: this edits .claude/agents/qa.md; per CLAUDE.md the Peder-review note lands in harness_log BEFORE any dependence; Main HOLDS the status flip (75.5 precedent) so the edits reach origin/main only after operator review; the 75.18 Q/A is spawned with an explicit self-reference disclosure (the Workflow path reads qa.md from disk live -- the edited rubric grades its own edit).

## Research-gate summary (gate PASSED)

Workflow `wf_613bed28-22d` (opus/max, complex). Envelope: `6 read-in-full (Vera-Perez pseudo-tested methods arXiv:1807.05030; Agentic Overconfidence arXiv:2602.06948; ReVeal arXiv:2506.11442; DeMillo-Lipton-Sayward; Goodenough-Gerhart revisit; mutmut/cosmic-ray CI practice), snippet=7, urls=50, recency=true, internal=13, gate_passed=true`. Brief: `research_brief_75.18.md` with the COMPLETE 11-shape taxonomy, cycle-cited.

**Corrections adopted (binding):**
1. Instance #3's mechanism is literal-kept-behavior-stripped (the `"stub": True` FIELD NAME surviving in source while pop() removed it from returns) -- distinct from #5's broken-test-double; keep them separate in the codification.
2. **The step's own verification command embodies three of the anti-patterns it codifies**: `'mutation' in qa.lower()` is ALREADY TRUE pre-edit; the skill clause is an OR-escape-hatch; all are source-scans. A green command is NOT completion evidence -- the six prose criteria are the gate, satisfied via a KNOWN-MEMBER RECALL TEST (apply the new doctrine as-written to >=1 known phase-75 vacuous guard and show it flags).
3. Catch-status nuance: the author's matrix caught 1/2/4; the INDEPENDENT Q/A caught 3/5/6 -- the codification must say the fixture/harness is mutated by the independent evaluator, not only the author.

**The 11-shape taxonomy (cycle-cited in the brief; the codification's content):** (1-3) source-scan variants; (4) tautology (incl. the 75.14 `or True`); (5) fixture-cannot-represent-failure; (6) library-fact posing as fixture pin; (7) RE-IMPLEMENTED test (75.14 routing-inversion survivor); (8) OR-escape-hatch/comment-token (75.15); (9) executor-environment non-reproducibility (zsh word-split x3, PATH, .env flag-state); (10) hand-derived-scope staleness; (11) mis-attributed kill mechanism (75.7 M1).

**Key research verdicts:** pseudo-tested methods = the academic root of #5/#6 (remedy: contract-test the fake against the real type); Agentic Overconfidence + ReVeal justify execution-grounded verification ("execute the mutation, never reason that a guard looks behavioral"); KEEP the manual scoped per-criterion matrix, NO mutmut/cosmic-ray CI gate (runtime economics; a nightly diagnostic on money modules is a possible future step, out of scope).

## Immutable criteria + command

Verbatim in the masterplan node. The command is a smoke check (partly pre-satisfied -- disclosed); the six prose criteria are satisfied non-vacuously via the recall test + Q/A prose-reads (not token-scans).

## Plan (the per-file map, NO cross-file duplication)

1. **qa.md NEW section 4c "Guard-vacuity check"** (after 4b): for EACH immutable criterion, name the concrete mutation that would make its guard fail; "no such mutation exists" is a FINDING (Circular_Reasoning/Missing_Assumption), never a pass; mutation evidence MUST cover the test FIXTURE/stub (contract-test the fake against the real type; cite the 75.2.1 dict-stub); the INDEPENDENT evaluator mutates the fixture/harness, not only the author; the full 11-shape checklist with one-line definitions + cycle citations; execution-grounded rule (run the mutation, never judge by appearance); operator-ratified memory wording VERBATIM where possible.
2. **per-step-protocol.md section 4**: one-line fixture-mutation requirement + NEW "Measure before asserting" subsection (the three phase-75 examples + the comments-carry-scanned-literals rule) + cross-links to qa.md 4c (no restating).
3. **code-review skill**: Dimension-4 + Top-list #17 "illusory guard" ranked heuristic naming the four required shapes + sub-shapes #7/#8, BLOCK/WARN dispatch (BLOCK when sole coverage for a behavioral/money-path criterion), REQUIRED negation list (criterion-mandated verbatim scans paired with behavioral guards; dead-branch scans) so it does not over-fire.
4. **Known-member recall demonstration** in experiment_results: apply 4c + the heuristic as-written to instance #3 (pop-key) AND #8 (seed OR-hatch); show each flags.
5. **Peder-review note** appended to harness_log; **status flip HELD** (the disposition note in the log + masterplan untouched at pending).
6. Q/A spawn with the self-reference disclosure + instruction to evaluate the DIFF against the six criteria + the two memories via prose-reads and to run the recall test itself.

## NOT in scope
Product code; CI mutation tooling; the preflight gate (75.19); flipping 75.18 (operator's).

## References
research_brief_75.18.md; feedback_mutation_test_guards_and_fixtures + feedback_measure_dont_assert_claims (operator-ratified wording); the 75.5 hold precedent; CLAUDE.md separation-of-duties.
