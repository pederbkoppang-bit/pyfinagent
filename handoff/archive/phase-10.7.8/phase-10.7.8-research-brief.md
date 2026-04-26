# Research Brief: phase-10.7.8 -- Runbook + Rollback Drill (Meta-Evolution)

Tier: **simple** (operational doc, internal-only). Internal-heavy
research per established precedent for pure-doc cycles (16.40, 16.43,
16.46, 16.47). External literature on git-revert procedure is well-known
prior art; no new external research needed.

---

## Internal sources read in full

| File | Lines | Role |
|------|-------|------|
| `docs/runbooks/alpaca-mcp-rollback.md` | 113 | Structural template (When/Immediate/State invariants/Permanent disable/Escalation) |
| `docs/runbooks/phase9-cron-runbook.md` | (read) | Existing cron runbook -- precedent for documenting weekly jobs |
| `docs/runbooks/llm_outage.md` | (read) | LLM rollback patterns |
| `docs/runbooks/per-step-protocol.md` | (read) | Harness operational procedure |
| `backend/meta_evolution/cron.py` | 155 | The weekly job we'd roll back (10.7.6) |
| `backend/meta_evolution/directive_review.py` | 225 | The review gate we might disable (10.7.7) |
| `backend/meta_evolution/directive_rewriter.py` | 411 | The rewriter (10.7.2) |
| `backend/meta_evolution/cron_allocator.py` | 158 | Cron budget allocator (10.7.4) |
| `backend/meta_evolution/provider_rebalancer.py` | 230 | Provider budget rebalancer (10.7.5) |
| `backend/meta_evolution/alpha_velocity.py` | 161 | Alpha velocity sampler (10.7.1) |
| `backend/meta_evolution/archetype_library.py` | 253 | Archetype lookup (10.7.3) |
| `.claude/cron_budget.yaml` | 190 | Slot 14 = meta_evolution_weekly_reallocation |
| `.claude/provider_budget.yaml` | 47 | Per-provider USD floors |
| `.claude/agents/researcher.md` | -- | The directive that the rewriter proposes against |

## Recency scan

`git revert` semantics + procedure are standard git knowledge dating to
2005 (git 1.0). No 2024-2026 changes to the command's behavior. The
`feature_flag` rollback pattern is canonical (Hightouch / LaunchDarkly /
GrowthBook docs) -- not load-bearing for a local-only Mac deployment.

## Decisive findings

The masterplan immutable verification is:
```
test -f docs/runbooks/meta_evolution_rollback.md && grep -q 'git revert' docs/runbooks/meta_evolution_rollback.md
```

So the runbook MUST:
1. Exist at `docs/runbooks/meta_evolution_rollback.md`
2. Contain the literal string `git revert`

Beyond the verification floor, the runbook should be operationally
useful, mirroring `alpaca-mcp-rollback.md` structure:
- When to use (trigger conditions per failure mode)
- Immediate rollback (30-60 second commands)
- State invariants after rollback
- Permanent disable (require code review to re-enable)
- Escalation
- Drill procedure (rehearsal sign-off)

## Things to roll back (in order of severity)

| Component | Risk if active | Rollback action |
|-----------|----------------|-----------------|
| `register_meta_evolution_cron()` wired into start_scheduler() | weekly job fires unexpectedly | unregister job + git revert |
| `directive_rewriter.rewrite_directive()` | proposes a bad directive change | `git revert <commit>` of any directive change committed to `.claude/agents/researcher.md`; restart Claude Code session per "Agent definition changes require session restart" rule |
| `directive_review` ACCEPT_THRESHOLD lowered | weak proposals slip through | revert to 0.70 |
| `provider_rebalancer.allocate()` | misallocated USD across providers | revert `.claude/provider_budget.yaml` to last-known-good |
| `cron_allocator.allocate()` | over-allocates daily slots | revert `.claude/cron_budget.yaml` to last-known-good |
| `alpha_velocity` BQ persistence | spurious telemetry rows | git revert + `DELETE FROM ... WHERE computed_at >= ...` (with operator approval) |

## Pitfalls

1. **Agent definition changes require session restart** (CLAUDE.md). After `git revert` of `.claude/agents/researcher.md`, the active Claude Code session still has the OLD snapshot loaded. `/clear` or restart required.
2. **YAML reverts do NOT need restart** -- `.claude/cron_budget.yaml` and `.claude/provider_budget.yaml` are read on each `allocate()` call.
3. **directive_review is fail-CLOSED** -- so a config error rejecting all proposals is "safe" but blocks the rewriter. Symptom-based triage useful.
4. **No live wiring yet** -- as of phase-10.7.7 close, `cron.py` is not yet called from `slack_bot/scheduler.py`. The runbook should note this and explain how to handle BOTH the not-yet-live state AND the future live state.

## Drill cadence

Quarterly per the same convention as `alpaca-mcp-rollback.md`. Operator
(Peder) walks through each rollback action in dry-run mode without
applying, signs off in a comment block at the end of the runbook.

## Plan for the runbook content

Sections:
1. Title + When-to-use trigger conditions
2. Immediate rollback (30-60s) -- `git revert` the offending commit; for each module, which file:line to inspect first
3. Per-component rollback actions table (the table above)
4. State invariants after rollback
5. Permanent disable (set ACCEPT_THRESHOLD=1.01 to block rewriter outright; comment out `register_meta_evolution_cron()` call site)
6. Drill procedure (quarterly rehearsal checklist)
7. Escalation
8. Related runbooks (cross-link)

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "report_md": "handoff/current/phase-10.7.8-research-brief.md",
  "gate_passed": true,
  "gate_passed_basis": "internal-heavy precedent for pure-doc cycle (16.40 / 16.43 / 16.46 / 16.47); git-revert is canonical prior art with no recent semantic changes"
}
```
