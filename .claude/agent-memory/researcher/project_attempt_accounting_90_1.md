---
name: attempt-accounting-90-1
description: Step 90.1 attempt/outcome accounting -- run records have NO outcome key (617/617), the token ceiling is structurally inert, one counter serves two budgets, and the cited tool-ralph mechanism source is uncorroborated
metadata:
  type: project
---

Measured 2026-08-20 for step 90.1 (attempt/outcome accounting on the attempt-budget rail).

**There is no `outcome` key on a Workflow run record -- 617/617.** The terminal field is
`status`, taking exactly three values: `completed` 564 / `failed` 48 / `killed` 5.
`totalTokens` IS present on 617/617 (608 non-zero). Any design that says "resolve outcome
from the run record" reads `None` on every record unless it reads `status`.

**The token half of the budget has never been able to fire.** `attempt_budget.py:64` sets
`DEFAULT_MAX_TOKENS=1_200_000` and `:128-130` ORs it into `exhausted`, but the only wired
producer -- `attempt_gate.py:190-193` -- calls `state.record(outcome)` with no `tokens=`,
and `Attempt.tokens` defaults to 0. Every escalation file on disk prints
`tokens used : 0 / 1,200,000` verbatim.

**A rail drop is mechanically separable and it is NOT cheap.** All 46
`subagent completed without calling StructuredOutput` runs carry `status:"failed"`; they
cost 8,822,653 tokens total, mean 191,796 vs 242,997 for a completed run. Two further
`failed` runs (`args are PRESENT but not parseable as JSON`) cost ZERO -- a binary
completed/not-completed vocabulary merges a free caller bug with an expensive drop.

**Why:** 90.1 is pure accounting and is the prerequisite for 90.3's rail-drop exemption,
which cannot be expressed without an outcome field.

**How to apply:** three traps to re-derive before quoting anything here.
1. **State the token population.** Restricted to the 441 `qa-verdict` runs, 13 steps
   exceed 1.2M (max 86.85 @ 2,506,619). Over all 540 step-attributed Workflow runs
   (qa-verdict 435 + research-gate 104 + 1 probe) it is 18 steps (max 86.85 @ 2,677,199).
   The masterplan `audit_basis` names the 441 population and quotes the 540 figure. Both
   numbers are right for their own population; the sentence attributes one to the other.
2. **One counter, two budgets.** `attempt_gate.build_state()` counts every `type:"attempt"`
   row ignoring `workflow`, but `research_router.py:30-35` reads the SAME stream and uses
   `workflow` to count re-research rounds. 9 of 27 step-ids have research-gate launches
   eating their five Q/A attempts (86.69 is 2 research + 1 qa).
3. **`tool-ralph` is uncorroborated.** The masterplan cites deepseek-harness `tool-ralph`
   for the closed vocabulary `complete|blocked|budget-limited`. Repo-wide grep returns ONE
   hit -- the audit_basis string itself. `docs/audits/deepseek-harness-2026-08-18.md` (450
   lines, lists every deepseek file read at commit 99f6f02) contains zero "ralph".
   deepseek.com/harness/en/ does not carry the vocabulary and the GitHub
   `docs/agent-lifecycle.md` path 404s. The triple is defensible on Temporal/k8s
   precedent alone -- do not justify it with that citation.

Verdict re-derivation matched the audit_basis exactly: 441 qa-verdict runs,
CONDITIONAL 221 / PASS 109 / FAIL 67 = 397 graded, 44 (10.0%) with no verdict.

See also [[id-collision-resolver-86-19]], [[attempt-budget-wiring-86-71]],
[[workflow-run-record-corpus-parsing]], [[url-count-must-be-re-derived]].
