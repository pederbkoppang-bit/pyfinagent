---
step: phase-10.7.8
title: Runbook + rollback drill for meta-evolution
cycle_date: 2026-04-26
harness_required: true
verification: "test -f docs/runbooks/meta_evolution_rollback.md && grep -q 'git revert' docs/runbooks/meta_evolution_rollback.md"
research_brief: handoff/current/phase-10.7.8-research-brief.md
---

# Contract -- phase-10.7.8

## Step ID

`phase-10.7.8` -- "Runbook and rollback drill" (`.claude/masterplan.json:3279-3287`).

## Research-gate summary

Internal-heavy brief at `handoff/current/phase-10.7.8-research-brief.md`
(tier: simple). Per established precedent for pure-doc cycles
(16.40, 16.43, 16.46, 16.47), external literature is not required: the
`git revert` rollback procedure is canonical prior art (git 1.0, 2005)
with no recent semantic changes. 14 internal files inspected
(meta_evolution package, existing runbooks, `.claude/{cron,provider}_budget.yaml`,
`.claude/agents/researcher.md`). `gate_passed: true` on internal-heavy basis.

## Hypothesis

A single operational runbook at `docs/runbooks/meta_evolution_rollback.md`
that mirrors the structure of `docs/runbooks/alpaca-mcp-rollback.md` and
documents per-component rollback actions for the 10.7.x meta-evolution
modules (cron / directive_rewriter / directive_review /
provider_rebalancer / cron_allocator / alpha_velocity / archetype_library)
gives the operator (Peder) a 60-second path back to a known-good state
when meta-evolution misbehaves.

## Immutable success criteria (verbatim from masterplan)

```
verification: test -f docs/runbooks/meta_evolution_rollback.md && grep -q 'git revert' docs/runbooks/meta_evolution_rollback.md
```

## Plan steps

1. Create `docs/runbooks/meta_evolution_rollback.md` with these sections (mirrors `alpaca-mcp-rollback.md` template):
   - Title + When-to-use (trigger conditions per failure mode)
   - Immediate rollback (30-60s) including the literal string `git revert`
   - Per-component rollback actions table (cron / directive_rewriter / directive_review / provider_rebalancer / cron_allocator / alpha_velocity)
   - State invariants after rollback
   - Permanent disable (set ACCEPT_THRESHOLD=1.01 / comment-out register call)
   - Drill procedure (quarterly rehearsal checklist with sign-off block)
   - Escalation
   - Related runbooks (cross-link)
   - **CLAUDE.md "Agent definition changes require session restart" caveat for `.claude/agents/researcher.md` reverts**

2. Run immutable verification:
   ```
   test -f docs/runbooks/meta_evolution_rollback.md && grep -q 'git revert' docs/runbooks/meta_evolution_rollback.md
   echo "exit=$?"
   ```

## References

- `.claude/masterplan.json:3279-3287` -- step entry
- `handoff/current/phase-10.7.8-research-brief.md` -- internal research gate
- `docs/runbooks/alpaca-mcp-rollback.md` -- structural template
- `backend/meta_evolution/{cron,directive_review,directive_rewriter,provider_rebalancer,cron_allocator,alpha_velocity,archetype_library}.py` -- modules covered by the runbook
- `.claude/cron_budget.yaml` (slot 14), `.claude/provider_budget.yaml`, `.claude/agents/researcher.md` -- artifacts that may need revert

## Out of scope

- Actually performing the drill (operator action; runbook documents how).
- Adding launchd plist for the meta-evolution cron (not yet wired into start_scheduler -- noted in runbook).
- BQ schema changes (no new tables this cycle).
- Code changes to any module (doc-only deliverable).
