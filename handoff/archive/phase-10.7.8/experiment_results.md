---
step: phase-10.7.8
cycle_date: 2026-04-26
forward_cycle: true
expected_verdict: PASS
deliverables:
  - docs/runbooks/meta_evolution_rollback.md (NEW, ~140 lines, 12 'git revert' occurrences)
---

# Experiment Results -- phase-10.7.8

## What was done

Wrote a single operational runbook at
`docs/runbooks/meta_evolution_rollback.md` covering rollback procedures
for all seven 10.7.x meta-evolution components: weekly cron (10.7.6),
directive review gate (10.7.7), directive rewriter (10.7.2), provider
rebalancer (10.7.5), cron budget allocator (10.7.4), alpha velocity
sampler (10.7.1), and archetype library (10.7.3).

Mirrors the structure of `docs/runbooks/alpaca-mcp-rollback.md`
(established template). Doc-only deliverable; no code changes.

## Deliverable

### `docs/runbooks/meta_evolution_rollback.md` (NEW, ~140 lines)

Sections:
1. **When to use** -- 7 trigger conditions covering each meta-evolution module + its YAML configs
2. **Immediate rollback (30-60s)** -- 4-step `git revert` procedure with explicit `git log` discovery, multi-commit handling, launchd reload, **and the CLAUDE.md "Agent definition changes require session restart" caveat for `.claude/agents/researcher.md` reverts**
3. **Per-component rollback actions** -- 7-row table (cron / rewriter / review / rebalancer / cron_allocator / alpha_velocity / archetype) with: symptom, files-to-inspect, rollback procedure
4. **State invariants after rollback** -- 5 verifiable invariants (git status clean, imports succeed, pytest sweep green, YAML re-read works, backend health endpoint responds, directive matches main)
5. **Permanent disable** -- ACCEPT_THRESHOLD=1.01 + comment-out cron registration; commit-the-disable pattern
6. **Drill procedure (quarterly)** -- 5-step dry-run rehearsal with sign-off block
7. **Escalation** -- pkill / launchctl unload + `git diff HEAD -- .claude/agents/researcher.md` safety-incident detection
8. **Related runbooks** -- 4 cross-links
9. **Notes on current wiring state** -- explicit acknowledgement that `cron.py` is not yet wired into `start_scheduler()` and `directive_review` is opt-in (not yet auto-called from rewriter), with instruction to update this runbook when the wiring lands

The literal string `git revert` appears 12 times in the runbook.

## Verification (verbatim, immutable from masterplan)

```
$ test -f docs/runbooks/meta_evolution_rollback.md && grep -q 'git revert' docs/runbooks/meta_evolution_rollback.md
$ echo "exit=$?"
exit=0
$ grep -c 'git revert' docs/runbooks/meta_evolution_rollback.md
12
```

## Files touched

| Path | Action | Note |
|------|--------|------|
| `docs/runbooks/meta_evolution_rollback.md` | CREATED | ~140 lines |
| `handoff/current/contract.md` | rewrite (rolling) | -- |
| `handoff/current/experiment_results.md` | rewrite (this) | -- |
| `handoff/current/phase-10.7.8-research-brief.md` | created (internal-heavy gate) | -- |

NO code changes. NO YAML changes. NO test additions. NO new
dependencies. Doc-only deliverable per masterplan immutable verification.

## Success criteria assessment

| # | Criterion | Result |
|---|-----------|--------|
| 1 | `docs/runbooks/meta_evolution_rollback.md` exists | PASS |
| 2 | Contains literal `git revert` | PASS (12 occurrences) |
| 3 | `test -f ... && grep -q 'git revert' ...` exit code | PASS (exit=0) |
| 4 | Mirrors structure of alpaca-mcp-rollback.md template | PASS (8 sections, same headings) |
| 5 | Covers all 7 meta-evolution modules | PASS (per-component table) |
| 6 | Includes session-restart caveat for `.claude/agents/researcher.md` reverts | PASS |
| 7 | Drill procedure with sign-off block | PASS |
| 8 | Honest about current wiring state | PASS (explicit "Notes on current wiring state" section) |

## Honest disclosures

1. **Internal-heavy research gate** (no external sources read in full) -- per established precedent for pure-doc cycles (16.40, 16.43, 16.46, 16.47). `git revert` semantics are canonical prior art (git 1.0, 2005); no recent semantic changes. JSON envelope reflects this with `gate_passed: true` on internal-only basis.

2. **No drill actually performed.** The runbook documents how to rehearse; the actual quarterly drill is a future operator action (sign-off block at the bottom of the runbook is empty by design).

3. **Wiring state called out explicitly.** As of this cycle close, neither the meta_evolution cron is wired into `start_scheduler()` nor is `directive_review` auto-called from the rewriter. The runbook says so directly and instructs updating when the wiring lands. Avoids the runbook-stale-vs-code-state drift seen in prior phase-9 cycles.

4. **No code changes.** The masterplan verification is a `test -f && grep` -- pure file existence + content check. Adding code would be scope creep.

5. **Cross-links to existing runbooks** -- alpaca-mcp-rollback.md (template), phase9-cron-runbook.md (cron procedures), llm_outage.md (LLM failure semantics), per-step-protocol.md (harness procedure). Avoids duplication.

## Closes

Task list item #77. Masterplan step phase-10.7.8.

## Next

Spawn Q/A. After PASS: log + masterplan flip + archive. End-of-batch validation.
