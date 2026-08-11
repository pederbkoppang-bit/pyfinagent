---
name: retry-loop-bounding-86-32
description: phase-86.32 -- the F1 consecutive_fails counter cannot bind (3 independent reasons); max_retries is decorative on 1078 steps; measured gate-cost distribution over 513 workflow runs
metadata:
  type: project
---

The evaluate/retry loop in this harness is **unbounded in practice**, and the
root cause has a name worth reusing.

**Root cause, one sentence: `run_harness.py` applied a HEALTH-CHECK idiom to a
WORK-ACCOUNTING problem.** Reset-on-success is correct for a circuit breaker
(question: "is the dependency healthy right now?" -- a recent success IS
evidence). It is categorically wrong for a per-work-item budget (question: "how
much have I already spent on this?" -- a success does not shrink the remaining
work). This also resolves a real source contradiction: Fowler/Nygard's breaker
resets on success; Azure's 2025-2026 revision makes the Closed-state counter
**time-based** and only lets success reset in Half-Open.

**Three independent reasons the F1 counter cannot bind** (each fatal alone):
1. `run_harness.py:1177` zeroes `consecutive_fails` on **CONDITIONAL** -- the
   dominant non-terminal verdict -- with the comment "does not count as a FAIL".
   `FAIL, CONDITIONAL, FAIL, CONDITIONAL...` tops out at 1, never 3.
2. `:1162` zeroes it on PASS (the classic reset-on-success flaw).
3. `:1109` declares it INSIDE `main()` -- **process-local**, never persisted. The
   Layer-3 manual loop spawns Q/A per step rather than `run_harness.py --cycles
   N`, so it never accumulates at all. Contrast `handoff/away_ops/
   autoresearch_fail_state.json`, the same name done durably in another subsystem.

**`max_retries` is decorative.** Present on 1078 masterplan steps, written only
by `scripts/generate_masterplan.py:202-203` + the `add_phase_27*.py` family, and
**read by nothing**. Step `75.5` sits at `retry_count == max_retries == 3` and
its status is `done`. The 3rd-CONDITIONAL rule is likewise instructions-only:
`scripts/qa/verdict_history_86_21.py` is ADVISORY by its own docstring and is
called by nothing but its own mutation harness -- and it is a DIFFERENT counter
from F1 (different source, threshold, and reset rule), with `CLAUDE.md` and
`qa.md` stating contradictory predicates.

**Measured cost (513 `wf_*.json` run records, 484 attributed to 164 steps):**
runs-per-step histogram `{1:27, 2:48, 3:38, 4:28, 5:13, 6:5, 7:2, 8:1, 9:2}`;
per-step cumulative p50 **419,739** tok, p90 **882,651**, max **1,832,223**
(step 75.5, 8 runs). **54.3% of steps take >2 runs and hold 76.0% of all gate
tokens.** 8.6% of runs (44) end non-`completed`, burning 7.65M tokens with no
verdict -- so any budget must decrement on DROPS too, or a step starves without
ever incrementing an attempt counter.

**Why:** the step text rested on two anecdotes; the numbers above are derived
from artifacts that already exist (`totalTokens` is recorded per run), so a spend
ceiling is measurable today without new instrumentation.

**How to apply:** when bounding any per-work-item loop here, make the bound
**cumulative over a window, never consecutive** (every Google SRE bound is: 3
total attempts, 10% retry ratio, 60 retries/minute -- not one is a
consecutive-failure counter). Cap rounds at ~4 (covers 93.9% of steps; the
literature plateaus after 1-2 -- Self-Refine +5.0 then +0.9 then +0.9; Huang et
al. GPT-4 GSM8K 95.5 -> 91.5 -> 89.0, and its published gains need an ORACLE stop
an LLM-judged harness does not have). On exhaustion, PARK + escalate to the
operator via the existing-but-dead `_escalate_certified_fallback` /
`## HARNESS HALT` block at `run_harness.py:1003-1035`; never fail closed silently.
Put the budget at the HARNESS layer only -- there are already three nested retry
layers (`orchestrator.py:819`, `llm_client.py:1367`, `info_gap.py:183`) and SRE
warns multi-layer retry multiplies (4^3 = 64 attempts).

**Anthropic's harness-design post prescribes NO termination rule** -- F1 is a
local invention, so there is no upstream authority to preserve when replacing it.

Brief: `handoff/current/research_brief_86.32.md`.
See [[websearch-budget-is-session-shared]], [[circuit-breaker-recovery-prior-art]].
