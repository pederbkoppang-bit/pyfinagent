# Contract — Cycle 4.15.13 — Claude Code surfaces + CI + Slack + routines

## Research gate
Merged researcher: Claude Code CLI, headless, web, routines,
scheduled-tasks, GitHub Actions, GitLab CI, code-review, Slack,
channels, remote-control, ultraplan/ultrareview, checkpointing,
devcontainer, llm-gateway + our `.github/workflows/*.yml`,
`backend/slack_bot/*`, `.claude/cron_budget.yaml`,
`scripts/harness/run_harness.py`.

## Hypothesis
Claims to verify:
- `claude.yml` workflow missing `--model claude-opus-4-7` +
  `--allowed-tools` (MF-17)
- `cron_budget.yaml` aspirational only (MF-19)
- Custom Slack bot is 16 modules — unreplaceable
- Ultraplan/Ultrareview not in use
- No devcontainer
- LLM gateway vs our `llm_client.py` — orthogonal

## Success criteria
1. every_doc_pattern_status_evidenced
2. qa_runs_live_code_checks_not_review
3. deviations_cite_doc_page

## Scope
`docs/audits/compliance-claude-code-surfaces.md` — pattern-per-row.

## References
Phase-4.11 claude_code_surfaces.md.
