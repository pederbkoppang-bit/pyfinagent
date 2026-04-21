# Experiment Results — Cycle 4.15.13

Step: phase-4.15.13 Claude Code surfaces + CI + Slack + routines/cron

## What was built

`docs/audits/compliance-claude-code-surfaces.md` (~1480 words, 20 patterns).

## Findings

**MUST FIX (3 — reinforces phase-4.14):**
- **MF-17** `claude.yml` under-pinned — no `claude_args` at all; defaults to Sonnet; arbitrary Bash on GH runner
- **New P-03**: Permissions block in `claude.yml` is read-only (`contents: read, pull-requests: read`) — `@claude` can't commit/comment. Document as intentional or upgrade to write.
- **MF-19** `cron_budget.yaml` fully disconnected from any scheduler

## Reinforces (no action beyond phase-4.14):

- Custom Slack bot (16 modules) unreplaceable by native — different problem
- `frontend/chrome/` is Playwright binary, NOT Claude Code browser extension
- `claude-code-review.yml` uses plugin path (doc-aligned)
- No devcontainer / LLM gateway / ultraplan / ultrareview — correct absences
- Only live cron in repo: `pip-audit.yml` `schedule: cron: "0 7 * * 1"` — exactly what slots 6-15 should look like

## Routines vs cron_budget (key insight)

- **Routines** (out of preview): ideal for slots 6-15 (research, autoresearch) — 1-hour minimum, autonomous, no approval prompts
- **Slots 1-5** (trading-ops) CANNOT use Routines — they would lose human-in-loop gate. Need GH Actions `schedule:` + manual approval pattern instead
- `/loop` is session-scoped (7-day expiry, open session required) — can't satisfy any cron_budget slot

## New findings

- **New P-03** (claude.yml permissions read-only) added above
- Otherwise: no net-new MF; reinforces MF-17, MF-19

## Success criteria

1. every_doc_pattern_status_evidenced — PASS (20 patterns)
2. qa_runs_live_code_checks_not_review — PARTIAL (Q/A next)
3. deviations_cite_doc_page — PASS

## Artifact

- `docs/audits/compliance-claude-code-surfaces.md`
