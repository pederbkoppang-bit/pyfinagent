# Contract — Cycle 4.15.14 — Agent SDK + Managed Agents compliance

## Research gate
Merged researcher: Agent SDK (29 pages) + Managed Agents (9+
pages) — confirm zero SDK adoption in backend, confirm Managed
Agents pilot form not submitted.

## Hypothesis
pyfinagent uses zero `claude-agent-sdk` imports. Managed Agents
pilot not started (no access form submitted). Both intentional —
hybrid strategy: keep custom backbone, selectively adopt SDK
primitives at harness boundary + pilot Managed Agents for one
step.

## Success criteria
1. every_doc_pattern_status_evidenced
2. qa_runs_live_code_checks_not_review
3. deviations_cite_doc_page

## Scope
`docs/audits/compliance-sdk-managed-agents.md` — pattern-per-row,
feature gap list.

## References
Phase-4.11 agent_sdk.md + managed_agents.md; phase-4.12
managed_agents_gaps.md.
