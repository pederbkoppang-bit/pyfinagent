---
name: MAS Architecture
description: Multi-Agent System design — Anthropic multi-agent research pattern with Generate->QA->Revise cycle
---

Decision made 2026-04-08: Follow Anthropic's multi-agent research pattern.

**Why:** Anthropic research shows Opus lead + Sonnet subagents outperform single Opus by 90.2%. Separation of generation from evaluation is "the strongest lever."

**Claude Code MAS (for autonomous masterplan execution) -- 3-agent topology, post-phase-4.15.0 merge:**
- Main (Lead): Opus 4.7 (orchestrator, reads masterplan, delegates, synthesizes)
- Researcher: Sonnet 4.6 (external literature + internal code exploration in one session, .claude/agents/researcher.md)
- Q/A: Opus 4.6 (merged qa-evaluator + harness-verifier; deterministic reproduction + LLM judgment, .claude/agents/qa.md)

Historical note: phase-4.15.0 merged the previous 4-agent topology (separate `Explore` + `qa-evaluator` + `harness-verifier`) into the current 3 agents. Re-splitting them is explicitly forbidden by CLAUDE.md.

**Backend MAS (for user queries via Slack/web):**
- Communication: Sonnet 4.6 (3-tier routing: trivial/simple/moderate/complex)
- Ford/Main: Opus 4.6 (orchestrator, can trigger harness)
- QA/Analyst: Opus 4.6 (quantitative analysis with anti-leniency)
- Researcher: Sonnet 4.6 (literature search, evidence-backed insights)

**Coding workflow (Generate -> QA -> Revise, max 3 iterations):**
1. Sprint contract before code work (handoff/current/contract.md)
2. Lead generates code (has Edit, Bash, Grep, git)
3. QA evaluator cross-verifies (structured JSON: {ok, reason, checks_run})
4. Lead decides: if ok:false, fix and re-evaluate (max 3 retries)
5. If 3 fails: escalate to Peder via Slack

**Critical principle:** NEVER self-evaluate. Always spawn a separate agent for verification.
