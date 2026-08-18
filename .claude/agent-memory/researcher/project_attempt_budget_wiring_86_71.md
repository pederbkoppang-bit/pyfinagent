---
name: attempt-budget-wiring-86-71
description: PreToolUse DOES fire on Workflow (655 rows measured) but its audit persists no tool_input; attempt_budget.py has ZERO file I/O; and Fowler's canonical breaker REFUTES the module's "nothing resets on success" claim
metadata:
  type: project
---

Step 86.71 research gate (2026-08-17). Four findings that are not derivable from
reading the code alone.

**1. The `Workflow` seam is real and measurable.** `handoff/audit/pre_tool_use_audit.jsonl`
holds 185,014 parseable rows; **655 carry `tool == "Workflow"`**, earliest
2026-05-28T20:42:05Z, latest 2026-08-17T09:40:29Z. `Agent` = 1,226. `Task` = **0**
(99 distinct tool names). So a PreToolUse hook CAN gate a Workflow launch --
this was the open question and it is answered YES, empirically.

**2. ...but that stream can never be a retrospective attempt ledger.**
`pre-tool-use-danger.sh:53` emits exactly `{"ts","tool","verdict","reason"}` and
**185,020 of 185,020 rows carry that key-set and nothing else**. The hook *parses*
`tool_input` (`:38-44`) and then throws it away. It receives the step_id; it never
writes it. History cannot be backfilled -- a new writer is mandatory.

**Why:** two separate probes are needed to tell "the hook never sees X" from "the
hook sees X and discards it". My first probe printed `tool_input: null` and looked
like the seam was blind; it was the *audit schema* that was blind.

**How to apply:** before proposing a hook-based counter anywhere, check the hook's
`printf`/log line, not just its input parsing. And use the tool-name distribution
as the positive control that the matcher would fire at all.

**3. `attempt_budget.py` is non-persistent BY CONSTRUCTION, not by oversight.**
Grepping it for `open(` / `json.load` / `write_text` / `Path(` returns **nothing**.
`to_json()` returns a string. So "wire it up" is not the whole job -- cross-session
persistence has to be built, not connected. Zero runtime callers confirmed
positive-controlled (same grep for `qa_wip` returns `.claude/agents/qa.md`,
`.claude/workflows/qa-verdict.js` + 8 `scripts/qa/*`).

**4. The module's own docstring overstates the literature.** `attempt_budget.py:16-18`
says *"Every bound in the SRE literature is cumulative over a window, never a
consecutive streak ... None of them reset on one success."* Martin Fowler's canonical
CircuitBreaker is **exactly** a consecutive counter that resets on one success
("successful calls reset it back to zero"), and he does not discuss rolling windows
at all. resilience4j's sliding window is the rate-based alternative; GEP-3388 never
raises the question. The conclusion survives, the supporting sentence does not: the
true split is *rate-limiting bounds are cumulative* (Google SRE 60/min, GEP-3388
percentage-of-interval, resilience4j window) vs *availability breakers are streaks*,
which are health checks on a dependency rather than work accounting.

**5. Three sources the premise would lean on DECLINE to support it.** Anthropic's
harness-design blog "does not explicitly discuss retry counts, failure thresholds, or
escalation procedures for handing work back to humans". arXiv 2607.01641 ("When Agents
Do Not Stop") measures the failure class (retry-feedback-without-bound = 25.0% of 68
cases; API cost exhaustion = 95.6%) but recommends **no** cumulative budget, persistent
counter or human escalation -- only that "bounds should be enforced at the runtime scope
where feedback is created". LoopTrap (2605.05846) recommends sandboxed progress-signal
verification, not external budgets. **The 5-attempt / 1.2M-token ceiling rests on
pyfinagent's own run distribution, not on external authority -- never cite Anthropic
for it.**

**How to apply:** when a step's premise is "the literature says do X", fetch the
literature and check it says X. Here the mechanism (cumulative-over-window) is
well-supported and the *policy* (cap attempts, escalate to a human) is not supported
by anyone -- which is fine, but it must be sourced internally and said out loud.

Related: [[project_counter_correctness_86_79]] (fail-closed None-vs-zero, PERF_RECORD_LOST
loss ledger, Temporal-inclusive vs Step-Functions-exclusive), [[project_verdict_ledger_writer_86_85]]
(the append-only writer this should copy -- note its `(step_id, run_id)` key is
UNAVAILABLE at PreToolUse; `tool_use_id` is the pre-execution identifier),
[[project_retry_loop_bounding_86_32]] (the original filing).
