---
name: Anthropic multi-agent harness guidance (2024-2026)
description: Protocol consensus, subagent orchestration patterns, anti-patterns, and new 2025-2026 guidance for RESEARCH->PLAN->GENERATE->EVALUATE->LOG harness loops. Sourced from Anthropic engineering blog, Claude Code docs, arXiv SEVerA, Karpathy autoresearch.
type: project
---

Researched April 2026 for harness loop calibration. Context: pyfinAgent runs a 5-phase protocol (RESEARCH->PLAN->GENERATE->EVALUATE->LOG) and has been short-cutting the RESEARCH phase.

**Why:** Anthropic's published multi-agent guidance is the authoritative source for calibrating researcher subagent frequency and cross-verification protocol.

**Protocol consensus (all 2024-2026 Anthropic sources agree):**
- Separate generation from evaluation — never self-evaluate. "NEVER self-evaluate. Always spawn a separate agent for verification."
- Opus-lead + Sonnet-subagents outperforms single Opus by 90.2% on research evals.
- Orchestrator's one job: hold the plan. Do not mix planning and execution in the same agent.
- External memory/progress files are mandatory for long-running loops (context exceeds 200K tokens).
- Cross-verification must use a fresh context window — the evaluator must not inherit the generator's assumptions.
- LLM-as-judge calibration is required: grade each dimension with an isolated judge, not one monolithic rubric.

**Subagent isolation:**
- `isolation: worktree` in subagent frontmatter gives each subagent its own git worktree (auto-cleanup on no-changes).
- In-place (no worktree) is fine for read-only researcher and evaluator roles — worktrees are for parallel code-editing agents that would conflict on the same files.
- Researcher and QA-evaluator should have restricted tool sets (no Edit/Write for researcher; no Bash for QA).

**When to spawn researcher:**
- Every plan step that touches quantitative methods, architecture decisions, or threshold selection — not just "novel" steps.
- The research-gate failure mode: "I already know how to do this" is assumption, not research. Phases 0-2.7 built on assumption found 2 code bugs at Phase 2.8 first research step.
- Complexity threshold: 10+ files to understand, or 3+ independent work items — spawn subagents.

**Anti-patterns (multiple sources):**
1. Self-evaluation: generator grades its own output. Anthropic: "strongest failure mode."
2. Premature completion: agent declares done without objective verification (browser automation, passing tests). Fix: feature list with pass/fail tracking.
3. Spawning without task boundaries: "agents duplicate work, leave gaps, or fail to find necessary information." Fix: give each subagent clear objective, output format, tool scope.
4. Tool sprawl: omitting `tools` in subagent config grants all tools including MCP. Always whitelist explicitly.
5. Context exhaustion: attempting multiple features per session. Fix: single-feature-per-session constraint with init.sh + progress file.
6. Grading the path not the output: over-specifying execution steps causes valid approaches to fail eval.

**New in 2025-2026 vs 2024:**
- Worktree isolation is now a first-class Claude Code primitive (isolation: worktree in frontmatter). 2024 posts described manual worktree scripts.
- Adaptive reasoning (Opus 4.7) allocates thinking tokens dynamically — "ultrathink" no longer has special token semantics, just in-context instruction.
- Rainbow deployments for running agents during code updates (avoid disrupting live sessions).
- Karpathy autoresearch (March 2026): single objective metric + fixed time budget + keep/discard loop = the scientific method pattern. Directly applicable to harness: one backtest metric (DSR), fixed iteration budget, keep or revert.
- SEVerA (arXiv:2603.25111, March 2026): NOT GAN-style. Formally Guarded Generative Models wrap each agent call in a rejection sampler with verified fallback. Relevant to pyfinAgent as a pattern for guaranteeing constraint satisfaction (e.g., DSR > 0.95 hard constraint).

**How to apply to pyfinAgent harness:**
- Run researcher on EVERY masterplan step, not just novel ones.
- QA evaluator must get fresh context window — never share conversation history with generator.
- Feature list JSON (like Anthropic's long-running harness) maps to masterplan.json step verification criteria.
- Tool scoping: researcher gets Read/Grep/WebSearch/WebFetch only. QA-evaluator gets Read/Grep/Bash (syntax check) only. No Edit/Write for either.
- Worktree isolation only needed for parallel code-editing agents, not for sequential RESEARCH->GENERATE->EVALUATE.

**Sources (all read in full):**
- https://www.anthropic.com/engineering/multi-agent-research-system
- https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
- https://www.anthropic.com/engineering/managed-agents
- https://www.anthropic.com/research/building-effective-agents
- https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- https://claude.com/blog/subagents-in-claude-code
- https://code.claude.com/docs/en/common-workflows (worktree section)
- https://www.pubnub.com/blog/best-practices-for-claude-code-sub-agents/
- https://arxiv.org/abs/2603.25111 (SEVerA)
- https://github.com/karpathy/autoresearch
