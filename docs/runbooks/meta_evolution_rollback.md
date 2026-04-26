# Meta-Evolution Rollback Runbook

Covers all 10.7.x meta-evolution components: weekly cron (10.7.6), directive
review gate (10.7.7), directive rewriter (10.7.2), provider rebalancer
(10.7.5), cron budget allocator (10.7.4), alpha velocity sampler (10.7.1),
archetype library (10.7.3).

## When to use

- Weekly Sunday-02:00-ET meta-evolution cron fires unexpectedly or crashes
- Rewriter proposes a directive change that the operator does not want
  applied to `.claude/agents/researcher.md`
- `directive_review` ACCEPT_THRESHOLD has been lowered and weak proposals
  are slipping through
- `provider_rebalancer.allocate()` returns USD allocations the operator
  did not approve (after editing `.claude/provider_budget.yaml`)
- `cron_allocator.allocate()` over-allocates daily slots after a
  `.claude/cron_budget.yaml` edit
- Alpha-velocity BQ rows look spurious or contaminated
- Any change committed to `backend/meta_evolution/*.py` or its YAML
  configs needs to be undone

## Immediate rollback (30-60 seconds)

The fastest path back to a known-good state is `git revert` of the
offending commit. Do NOT use `git reset --hard` against `origin/main`
without operator approval (per CLAUDE.md "Never use destructive git
commands unless explicitly requested").

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent

# 1. Identify the offending commit (recent meta_evolution changes).
git log --oneline -20 -- backend/meta_evolution/ .claude/cron_budget.yaml .claude/provider_budget.yaml .claude/agents/researcher.md

# 2. Revert it (creates a new commit that undoes the change; safe
#    on `main` because history is preserved).
git revert <commit-sha>

# 3. If multiple commits need reverting, do them in reverse order
#    (newest first). Use --no-edit to keep auto-generated messages,
#    or omit to edit each in your $EDITOR.
git revert <newest> <older> <oldest>

# 4. Backend services pick up the reverted code on next reload.
#    YAML changes are read on each allocate() call -- no restart needed.
#    Python code changes need a backend restart:
launchctl unload ~/Library/LaunchAgents/com.pyfinagent.backend.plist
launchctl load   ~/Library/LaunchAgents/com.pyfinagent.backend.plist
```

If the reverted commit touched `.claude/agents/researcher.md`, the
**active Claude Code session must be restarted** (`/clear` or relaunch)
because agent definitions are snapshotted at session start (CLAUDE.md
"Agent definition changes require session restart").

## Per-component rollback actions

| Component | Symptom | Files to inspect | Rollback |
|-----------|---------|------------------|----------|
| Weekly cron (10.7.6) | unexpected job firing or persistent crashes in scheduler logs | `backend/meta_evolution/cron.py`, `backend/slack_bot/scheduler.py` (`start_scheduler()` call site once wired) | `git revert` the wiring commit; or comment out the `register_meta_evolution_cron(scheduler)` call until investigated |
| Directive rewriter (10.7.2) | rewriter proposed a bad directive | `backend/meta_evolution/directive_rewriter.py`, `.claude/agents/researcher.md` | `git revert` the commit that applied the proposal to `researcher.md`; **restart Claude Code session** so the reverted directive becomes active |
| Directive review gate (10.7.7) | gate is accepting weak proposals | `backend/meta_evolution/directive_review.py` | confirm `ACCEPT_THRESHOLD` is `0.70`; if it has been lowered, `git revert` the change |
| Provider rebalancer (10.7.5) | misallocated USD across providers | `.claude/provider_budget.yaml`, `backend/meta_evolution/provider_rebalancer.py` | `git revert` the YAML edit; YAML is re-read on next `allocate()` call (no restart) |
| Cron budget allocator (10.7.4) | overallocated daily slots | `.claude/cron_budget.yaml`, `backend/meta_evolution/cron_allocator.py` | `git revert` the YAML edit; YAML re-read on next `allocate()` call |
| Alpha velocity (10.7.1) | spurious BQ rows in `pyfinagent_data.alpha_velocity_samples` | `backend/meta_evolution/alpha_velocity.py` | `git revert` the offending code commit; for spurious rows, run `DELETE FROM pyfinagent_data.alpha_velocity_samples WHERE computed_at >= '<ts>'` via BQ MCP **only with operator approval** (see CLAUDE.md BigQuery rule 4) |
| Archetype library (10.7.3) | bad archetype rule in active strategy lookup | `backend/meta_evolution/archetype_library.py` | `git revert` the commit |

## State invariants after rollback

- `git status` clean (revert commits are committed, not left staged)
- `python -c "from backend.meta_evolution import cron, cron_allocator, provider_rebalancer, directive_rewriter, directive_review, alpha_velocity, archetype_library"` succeeds (no ImportError on the reverted code)
- `python -m pytest tests/meta_evolution/ tests/scheduler/test_meta_cron.py tests/agents/test_evaluator_directive_review.py -q` -- all green
- For YAML reverts: `python -c "from backend.meta_evolution.cron_allocator import allocate; print(allocate('.claude/cron_budget.yaml'))"` returns sane numbers
- For directive reverts: open `.claude/agents/researcher.md` and confirm the prior text is in place
- Backend uvicorn process responding on `http://127.0.0.1:8000/api/health`

## Permanent disable (require code review to re-enable)

To pause meta-evolution entirely without removing the modules:

```bash
# 1. Block the rewriter from ever ACCEPTing -- bump threshold above 1.0.
#    Edit backend/meta_evolution/directive_review.py:
#    ACCEPT_THRESHOLD = 1.01   # was 0.70 -- temp disable
#
# 2. Comment out the cron registration call site (once wired into
#    backend/slack_bot/scheduler.py start_scheduler):
#    # register_meta_evolution_cron(scheduler)
#
# 3. Commit the disable with a clear message.
git commit -am "ops: temporarily disable meta-evolution (incident <ID>)"

# 4. To re-enable later, revert this commit (or revert the changes
#    manually) AFTER reviewing what went wrong.
```

The kill-switch / disable rule mirrors `alpaca-mcp-rollback.md` -- a
deliberate code-reviewed action both to disable and to re-enable.

## Drill procedure (quarterly rehearsal)

Mirrors the cadence of `alpaca-mcp-rollback.md`. Operator (Peder) runs
through the steps below in dry-run mode (no commits, no service restart)
and signs off in the comment block at the bottom.

1. `git log --oneline -10 -- backend/meta_evolution/` -- read recent commits
2. Pretend a recent commit is bad. Type out the `git revert <sha>` command but DO NOT execute. Confirm you understand which file(s) it would touch.
3. For each component in the table above, verify you can locate:
   - The source file
   - The relevant YAML (where applicable)
   - The test command that proves the revert is healthy
4. Confirm you can answer:
   - Which rollbacks need a backend restart? (Python code changes do; YAML re-reads do not)
   - Which rollbacks need a Claude Code session restart? (`.claude/agents/researcher.md` does; nothing else)
   - Which rollbacks need BQ DML? (only `alpha_velocity_samples` row deletion, with operator approval)
5. Sign-off block:

```
DRILL SIGN-OFF
Date: ____________
Operator: ____________
Notes: ____________
```

## Escalation

If a `git revert` does not stop the misbehavior within 5 minutes:

```bash
# 1. Hard-stop the backend (zombie-prevention: kill parent + child workers).
pkill -f "uvicorn backend.main:app"
launchctl unload ~/Library/LaunchAgents/com.pyfinagent.backend.plist

# 2. Inspect handoff/logs/ for the most recent error trace.
tail -100 handoff/logs/*.log

# 3. Inspect BQ for any unexpected rows persisted post-revert.
#    (Use the BigQuery MCP execute_sql_readonly tool; bound the query.)

# 4. If the rewriter has been applying directive changes autonomously
#    (it should NOT -- the HITL gate is enforced by NOT having auto-apply),
#    confirm the current `.claude/agents/researcher.md` matches the
#    last committed version on `main`:
git diff HEAD -- .claude/agents/researcher.md
```

If the rewriter has somehow persisted a directive change without
operator approval, this is a **safety incident** -- not just a bug.
Document it, lock the meta-evolution work down with the "Permanent
disable" steps above, and review `directive_rewriter.py` for the path
that bypassed the HITL gate.

## Related runbooks

- `docs/runbooks/alpaca-mcp-rollback.md` -- structural template for this runbook
- `docs/runbooks/phase9-cron-runbook.md` -- general APScheduler cron operational procedure
- `docs/runbooks/llm_outage.md` -- LLM provider failure (relevant when `directive_rewriter` or `directive_review` LLM calls fail; review fail-OPEN vs fail-CLOSED semantics per module)
- `docs/runbooks/per-step-protocol.md` -- harness 5-file protocol (the meta-evolution work is itself harness-driven)

## Notes on current wiring state

As of phase-10.7.7 close (2026-04-26):
- `backend/meta_evolution/cron.py` is **NOT** yet called from
  `backend/slack_bot/scheduler.py:start_scheduler()`. The weekly job
  cannot fire in production until that one-line wiring lands. If you
  receive an unexpected fire alert NOW, suspect ad-hoc invocation
  (e.g., a test or REPL call), not the scheduler.
- `directive_review.review_directive_diff()` is **NOT** yet automatically
  called from `directive_rewriter.rewrite_directive()`. The gate exists
  as a callable but is opt-in.
- All directive applications to `.claude/agents/researcher.md` remain
  HITL-gated -- the rewriter ONLY proposes; the operator + Main apply.

When this wiring lands in a future cycle, update this runbook to remove
these notes.
