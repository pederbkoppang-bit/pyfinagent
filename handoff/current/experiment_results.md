# Experiment results -- Step 75.18 (codify the anti-vacuous-guard doctrine into the harness)

Date: 2026-07-24. **Execution model: opus-tier -> Main GENERATE directly.
SEPARATION OF DUTIES ACTIVE: this step edits `.claude/agents/qa.md`; the
Peder-review note is in harness_log; the STATUS FLIP IS HELD (75.5
precedent) -- the operator flips 75.18 to done after reviewing the qa.md
diff, which is what pushes these edits.**

## What shipped (+89 lines across EXACTLY the three doctrine files; zero product code)

1. **qa.md NEW section 4c "Guard-vacuity check"** (+64 lines, between 4b
   and 4a): the operator-ratified rule verbatim ("a guard that cannot
   fail when its subject is broken does not count"); the per-criterion
   NAME-THE-MUTATION requirement with "no such mutation exists" wired as
   a `Circular_Reasoning`/`Missing_Assumption` FINDING; the
   fixture-mutation requirement grounded in pseudo-tested-methods
   (arXiv:1807.05030) with the 75.2.1 dict-stub as the canonical
   instance; the INDEPENDENT-evaluator-mutates-the-fixture rule (history:
   author matrices caught shapes 1/2/4, the independent Q/A caught
   3/5/6); the execution-grounded rule (ReVeal arXiv:2506.11442 +
   Agentic Overconfidence arXiv:2602.06948 -- run the mutation, never
   judge by appearance); **the full 11-shape checklist with cycle
   citations** (incl. the session's new shapes: #7 re-implemented test,
   #8 OR-escape-hatch/comment-token, #9 executor-environment, #10
   hand-derived-scope staleness, #11 mis-attributed kill); verdict
   wiring (sole-coverage vacuity on behavioral/money-path = BLOCK).
2. **per-step-protocol.md** (+21): the one-line fixture-mutation
   requirement cross-linked to 4c (no duplication) + the NEW "Measure
   before asserting" subsection (three phase-75 canonical slips + the
   comments-carry-scanned-literals rule).
3. **code-review skill** (+4 net additions): ranked heuristic **#17
   illusory-guard** naming the four required shapes + sub-shapes (e)
   re-implemented test and (f) OR-escape-hatch, with the
   BLOCK-when-sole-coverage / WARN-when-paired dispatch and the
   detection question ("name the mutation that makes this fail");
   Dimension-4 table row; TWO negation entries (criterion-mandated
   verbatim scans paired with behavioral guards; dead-branch scans) so
   the heuristic does not over-fire.

## Research corrections honored

Instance #3 kept DISTINCT from #5 in the checklist (literal-kept-
behaviour-stripped vs broken-test-double -- the step text had conflated
them); the catch-status nuance encoded (author 1/2/4 vs independent
3/5/6); the tool verdict encoded implicitly (the doctrine stays a manual
scoped matrix -- no CI mutation tooling shipped, per the runtime-economics
research; a nightly diagnostic on money modules is a possible future
step, deliberately NOT queued here since it needs its own cost analysis).

## Verification -- and its honest limits (the step's own criteria audit)

- Immutable command: **exit 0** -- BUT per the research this command is
  partly PRE-SATISFIED ('mutation' already appeared in qa.md 4x before
  this step) and its skill clause is itself an OR-escape-hatch. It is a
  smoke check ONLY. Disclosed, not papered.
- **The non-vacuous evidence is the KNOWN-MEMBER RECALL TEST** (below) +
  the Q/A's prose-read of the diff against the six criteria.
- C5 boundary proven: `git diff --name-only HEAD | grep -E '^(backend|frontend)/'`
  -> EMPTY (zero product code).
- Diff stat: exactly 3 files, +89/-0 (pure additions; no existing rule
  reworded except none -- 4b/4a untouched, verified by the diff).

## KNOWN-MEMBER RECALL TEST (applying the doctrine as-written to two known members)

**Member A -- instance #3 (75.3/C129, literal-kept-behaviour-stripped):**
the guard asserted `'"stub": True' in source` while `pop("stub")`
removed the key from every returned candidate. Applying 4c: the
name-the-mutation question ("what mutation makes this guard fail?") has
the answer "only deleting the LITERAL -- stripping the BEHAVIOUR leaves
it green" -> shape #3 matches verbatim ("the scanned LITERAL survives in
source while the behaviour it names was removed") -> FINDING. The #17
heuristic detection question fails identically (no behavioral mutation
reddens it) -> illusory-guard, BLOCK if sole coverage. **FLAGGED.**
**Member B -- instance #8 (75.15/C143, OR-escape-hatch):** the guard
asserted `('blocks the PR' not in s) or ('run_seed_stability' in s)`
where the same diff added the token to a comment. Applying 4c shape #8
("a guard clause satisfiable by prose or comment tokens the same change
introduces") -> FINDING; #17 sub-shape (f) matches; the
name-the-mutation answer is "re-adding the overclaim does NOT fail it"
-> illusory-guard. **FLAGGED.**
Both known members are caught by the codification as-written. (Per
Goodenough-Gerhart, this licenses recall on these two members -- not a
completeness claim; the checklist itself says list-not-ceiling.)

## Separation-of-duties disposition

- Peder-review note: appended to harness_log (this cycle's entry).
- **STATUS FLIP HELD**: 75.18 stays `pending` after the Q/A verdict; the
  operator flips after reviewing the qa.md diff (`git diff HEAD --
  .claude/agents/qa.md` is the review surface). No later step will be
  run by this session that depends on the new 4c.
- The 75.18 Q/A is spawned WITH the self-reference disclosure (the
  Workflow path reads qa.md from disk -- the edited rubric grades its
  own edit; the Agent-tool fallback would snapshot pre-edit).

## Not verified live
Docs/harness only; nothing runtime. The next session's roster
verification (scripts/qa/verify_qa_roster_live.sh + a fresh-spawn 4c
self-disclosure probe) is the owed behavioral check AFTER the operator
approves + a restart/fresh session occurs.
