---
name: Anthropic agent patterns — retry loops and research-on-demand
description: Exact mechanics from Anthropic engineering posts for evaluator→generator retry loops and lead-agent spawns-researcher patterns, mapped to run_harness.py variable names
type: project
---

Key facts extracted from Anthropic's canonical engineering posts (read in full April 2026):

**Sources:** https://www.anthropic.com/engineering/multi-agent-research-system, https://www.anthropic.com/engineering/building-effective-agents (research page), https://code.claude.com/docs/en/best-practices, https://code.claude.com/docs/en/agent-teams

**Evaluator returns structured critique, not just score.** The building-effective-agents post says the evaluator "provides evaluation and feedback" — not a bare score. Our harness already does this via `_write_critique()` which writes four scored dimensions + weak_periods + verdict.

**Retry loop:** Anthropic's evaluator-optimizer pattern is "one LLM call generates a response while another provides evaluation and feedback in a loop." No explicit retry cap is stated in the canonical posts — they leave that to implementers. The evaluator feedback (structured critique including weak_periods and per-criterion scores) is passed back to the planner as `previous_critique` — this is the correct pattern.

**Research-on-demand trigger:** The multi-agent post uses scaling rules embedded in prompts: "Simple fact-finding requires just 1 agent with 3-10 tool calls, direct comparisons might need 2-4 subagents, complex research might use more than 10 subagents." There is NO explicit "SPAWN_RESEARCHER" signal — the lead decides based on complexity assessment. The agent teams doc says "Claude proposes a team" when it "determines your task would benefit from parallel work."

**Durable state between retries:** "saving its plan to Memory to persist the context" — plans written to durable storage (file or BQ) before generator runs, so a failed cycle can resume. Our `pre_cycle_best` snapshot and `write_contract()` already implement this correctly.

**Error recovery:** "we built systems that can resume from where the agent was when the errors occurred" — not full restart. This maps to our revert-to-pre_cycle_best on FAIL, not restarting from scratch.

**Why:** User wants to know whether our harness architecture matches Anthropic's canonical patterns before extending it with research-on-demand spawning.

**How to apply:** When adding a researcher agent, trigger it from the planner when hypothesis == "knowledge_gap" (not by default on every cycle). Pass a structured brief with: objective, output format, tool scope, and task boundaries. Cap at 2-3 researcher iterations before proceeding with best available info.
