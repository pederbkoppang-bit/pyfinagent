# Contract -- step 86.71

**Step:** 86.71 -- the cumulative attempt budget that CLAUDE.md says governs the
per-step loop has NO caller and NO persistence.
**Date:** 2026-08-17 · **Author:** Main (operator-attended harness-repair session)
**Research gate:** PASSED (recomputed) -- `research_brief_86.71.md`, 9 sources in
full / 30 URLs / recency scan / brief COMPLETE (33,784 chars), run `wf_77c2679f-de9`.

## Research-gate summary (what changes the plan)

- **The origin seam is real and measurable:** PreToolUse FIRES on the Workflow
  tool (655 rows in `pre_tool_use_audit.jsonl`, 2026-05-28..08-17). But the
  existing danger-hook persists only `{ts,tool,verdict,reason}` -- `tool_input`
  is received and DISCARDED (185,020/185,020 rows), so history cannot be
  backfilled and a new writer is required.
- **Hook semantics (official docs): only exit 2 blocks; timeouts, schema errors
  and missing scripts all fail OPEN** -- so the gate is exit-2-based, its
  fail-open bound is stated rather than hidden, and every internal error is
  LOUD on stderr.
- **Identifier: `run_id` exists only in the POST-launch receipt; `tool_use_id`
  is the pre-execution identifier** -- the attempt ledger keys on
  (step_id, ts, tool_use_id), never on run_id.
- **Adversarial corrections adopted:** Fowler's canonical circuit breaker DOES
  reset on success -- `attempt_budget.py:16-18` overclaims its literature and
  the docstring will be corrected; and no external source recommends the 5 /
  1.2M ceilings -- they are sourced INTERNALLY (the module's own measured
  distribution) and must never be attributed to Anthropic.
- Caller-side measurements re-derived 2026-08-17: 66.5% of step-attributable
  runs repeat an already-attempted step (320/481; qa 77.4%, researcher 20.0%);
  max 9 qa runs on one step; module confirmed caller-less by positive-controlled
  search (control: the pattern hits its own test file).

## Hypothesis

A PreToolUse hook on the Workflow tool, backed by an append-only attempt ledger
and the existing `attempt_budget.py` arithmetic, gives the loop the cumulative,
cross-session, attempt-keyed (not outcome-keyed) bound CLAUDE.md already claims
-- denying the launch at the ceiling with a written operator escalation, at zero
token cost, and never converting any verdict.

## Immutable success criteria (copied verbatim from .claude/masterplan.json)

1. the 58.4% repeat rate and the per-role split are INDEPENDENTLY re-derived before any fix is designed, with the population rule and the command stated next to each number, and any disagreement with the figures in this audit_basis reported rather than silently adopted
2. the claim that attempt_budget.py has NO runtime caller is re-verified with a positive-controlled search, and the verification names the control string used -- a bare zero from one grep is not evidence of absence
3. the budget is given a persistence mechanism that survives a session boundary, and cross-session counting is DEMONSTRATED by incrementing it in one process and reading the incremented value in a separate process invocation
4. the budget is wired into the path where runs actually originate, and the wiring is proven by driving it: show a step at the attempt ceiling produces an ESCALATION to the operator and show a step below the ceiling is unaffected
5. verdict semantics are UNCHANGED and this is demonstrated, not asserted: exhaustion must escalate and must never auto-pass, and a FAIL must remain a FAIL under every flag combination
6. the Q/A-side repeat cost is addressed, not just the researcher's -- the measured maximum is 9 Q/A runs on one step against 3 researcher runs, so a researcher-only fix does not close this step
7. NO flag is promoted and NO .env is written by this step; operator-gated changes are recorded as numbered asks
8. mutation-test every new guard: revert it and show the check goes red, with the control observed GREEN first and a byte-identical restore

## Plan

1. **Criteria 1+2 evidence** (already banked pre-design, restated with commands
   in the artifacts): 66.5% (disagreement with the filed 58.4% reported --
   corpus grew from 513 to 580+ records); positive control = `attempt_budget`
   pattern hitting `backend/tests/test_phase_86_32_attempt_budget.py`, zero
   hits on runtime surfaces.
2. **`scripts/harness/attempt_gate.py`** (new): reads the PreToolUse stdin
   JSON; extracts step_id from Workflow args (object OR string; unparseable ->
   no attribution -> allow, loudly); appends one attempt row to
   `handoff/audit/attempt_budget_audit.jsonl`; builds `BudgetState` from the
   step's rows via the EXISTING `attempt_budget.py`; on ESCALATE disposition:
   exit 2, stderr reason, and writes
   `handoff/current/escalation_attempt_budget_<sid>.md` with the operator
   instructions. Fail-open-loud on any internal error. qa-verdict AND
   research-gate launches both count (criterion 6 by construction).
3. **Wire it**: PreToolUse `Workflow` matcher in `.claude/settings.json` (via
   the update-config path; config change logged by the config-change-audit
   hook).
4. **Drive (criteria 3+4)**: cross-session counting = rows written by one
   process read by a separate invocation; ceiling drive = seed synthetic step
   999.1 to the ceiling, make a REAL Workflow launch for it, observe the DENY
   (zero tokens) + the escalation file; below-ceiling = the session's own next
   real launches proceed.
5. **Criterion 5**: the deny path produces NO verdict artifact and touches no
   verdict file; `test_exhaustion_cannot_auto_pass` + module mutation matrix
   re-run green; gate has NO disable flag (the only override is an operator
   editing the ledger, which the deny message documents).
6. **Docstring truth fix**: correct `attempt_budget.py:16-18`'s "none of them
   reset" against Fowler (gate finding), keeping the narrower true claim.
7. **Criterion 8**: mutation cells -- ceiling comparison inverted, ledger write
   dropped, step-id extraction neutered, deny turned into allow -- each with
   control green first, byte-identical restore.

## Numbered operator asks

- **ASK-1:** the ceilings stay at the module's internally-measured defaults
  (5 attempts / 1.2M tokens). Raising or lowering them is an operator decision;
  the gate reads them from `attempt_budget.py` so one edit moves both.

## References

`research_brief_86.71.md` (envelope COMPLETE; sources incl. code.claude.com
hooks doc, SRE cascading-failures, Temporal retry policies, Fowler
CircuitBreaker, resilience4j, GEP-3388 retry budgets, arXiv 2607.01641 +
2605.05846, Anthropic harness-design); `scripts/harness/attempt_budget.py` +
its tests; `scripts/qa/qa_wip.py`; `verdict_ledger_write.py` (append-only
pattern); `.claude/hooks/pre-tool-use-danger.sh` (hook I/O shape).
