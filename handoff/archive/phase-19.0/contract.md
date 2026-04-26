---
step: phase-19.0
title: Feasibility study -- Claude Remote / Max programmatic handoff
cycle_date: 2026-04-26
harness_required: true
verification: "test -f docs/architecture/claude-remote-handoff-feasibility.md && grep -q 'Recommendation' docs/architecture/claude-remote-handoff-feasibility.md && grep -q 'Claude Agent SDK' docs/architecture/claude-remote-handoff-feasibility.md && grep -q 'rate limit' docs/architecture/claude-remote-handoff-feasibility.md"
research_brief: handoff/current/phase-19.0-research-brief.md
---

# Contract -- phase-19.0

## Step ID

`phase-19.0` -- "Feasibility study: Claude Remote / Max programmatic
handoff for pyfinagent heavy-lifting work". Operator-requested
2026-04-26 via /batch with explicit "do not implement" scope.

## Research-gate summary

Spawned researcher (moderate tier). Brief at
`handoff/current/phase-19.0-research-brief.md` (23KB, 283 lines).
gate: 7 external sources read in full via WebFetch (Claude Code
Authentication / Headless / Agent SDK docs, Anthropic pricing,
Portkey rate-limit blog, LiteLLM Max integration, The Register
ToS clarification), 17 URLs, recency scan present, 8 internal
files inspected. `gate_passed: true`.

Decisive findings (verified by researcher):
- **April 4, 2026: Anthropic banned third-party Max OAuth use.** The literal hypothesis is a ToS violation.
- **1M context is now standard-priced** on Sonnet 4.6 / Opus 4.7 (no `extended-context-1m-2025-08-07` beta surcharge as of 2026).
- **Sonnet 4.6: $3/$15 per Mtok input/output** -- 300K-token synthesis call ~$0.90.
- **Max plan rate limits: 200-800 prompts per 5-hour window** shared across all first-party interfaces. Irrelevant for API-key dispatch.
- **5 jobs benefit (ranked by ROI):** Layer-1 synthesis, skill optimizer global-pass, directive rewriter raw-brief mode, outcome tracker reflection, deep-dive agent.
- **Engineering cost:** 0.5 cycle spike -> 3 cycles full integration.

## Hypothesis

The operator's literal ask ("use Max OAuth for backend dispatch to
avoid additional cost") cannot be satisfied without ToS violation.
But the underlying intent (offload heavy long-context work to Claude)
is now economical via standard Anthropic API since 1M context dropped
to standard pricing in 2026. The decision document should capture
both the rejection of the literal hypothesis AND the alternative path
that achieves the goal.

## Immutable success criteria (verbatim from masterplan)

```
test -f docs/architecture/claude-remote-handoff-feasibility.md && grep -q 'Recommendation' docs/architecture/claude-remote-handoff-feasibility.md && grep -q 'Claude Agent SDK' docs/architecture/claude-remote-handoff-feasibility.md && grep -q 'rate limit' docs/architecture/claude-remote-handoff-feasibility.md
```

(File exists + contains "Recommendation" section + cites "Claude Agent SDK" + addresses "rate limit".)

## Plan steps

1. Spawn researcher (moderate tier, deep external + internal). DONE.
2. Author `docs/architecture/claude-remote-handoff-feasibility.md` covering:
   - Recommendation (REJECT literal hypothesis, ACCEPT underlying intent via API)
   - TL;DR table
   - ToS analysis with The Register + Agent SDK citations
   - What Max actually covers (table)
   - 1M context current pricing on Anthropic API (with worked examples)
   - Ranked-by-ROI table of 5 pyfinagent jobs that benefit
   - Anti-recommendations (7 jobs that do NOT benefit)
   - Recommended architecture sketch (extend `make_client()`, NOT new module)
   - Recommended budget tracker shape
   - Engineering cost estimate
   - Risk register
   - Decision: proceed with 0.5-cycle spike
3. Run immutable verification command.

NO code changes. NO masterplan changes beyond adding phase-19. NO new tests.

## References

- `handoff/current/phase-19.0-research-brief.md`
- The Register: https://www.theregister.com/2026/02/20/anthropic_clarifies_ban_third_party_claude_access/
- Anthropic pricing: https://platform.claude.com/docs/en/about-claude/pricing
- Claude Agent SDK: https://code.claude.com/docs/en/agent-sdk/overview
- Project memory: `project_local_only_deployment.md` (Max flat-fee)

## Out of scope

- The spike itself (would be phase-19.1 if approved)
- Any code changes
- Any modifications to `llm_client.py` / `orchestrator.py` / `directive_rewriter.py`
- Gemini 2.0 Pro 1M comparative evaluation (separate cycle if needed)
