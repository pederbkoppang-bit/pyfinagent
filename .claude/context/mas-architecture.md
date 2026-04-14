---
name: MAS Architecture
description: Multi-Agent System design — Anthropic multi-agent research pattern with Generate->QA->Revise cycle
---

Decision made 2026-04-08: Follow Anthropic's multi-agent research pattern.

**Why:** Anthropic research shows Opus lead + Sonnet subagents outperform single Opus by 90.2%. Separation of generation from evaluation is "the strongest lever."

**Claude Code MAS (for autonomous masterplan execution):**
- Lead Agent: Opus 4.6 (orchestrator, reads masterplan, delegates, synthesizes)
- Researcher: Sonnet 4.6 (literature search, evidence gathering, .claude/agents/researcher.md)
- QA Evaluator: Opus 4.6 (cross-verification, .claude/agents/qa-evaluator.md)
- Harness Verifier: Sonnet 4.6 (harness results check, .claude/agents/harness-verifier.md)

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
