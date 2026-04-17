# Per-Step Harness Protocol (operator runbook)

**Canonical references (must read before every long-running session):**
- Anthropic, "Harness Design for Long-Running Apps": https://www.anthropic.com/engineering/harness-design-long-running-apps
- Anthropic, "How We Built Our Multi-Agent Research System": https://www.anthropic.com/engineering/built-multi-agent-research-system
- Anthropic, "Building Effective Agents": https://www.anthropic.com/engineering/building-effective-agents

**This file is the executable mapping** from the Anthropic three-phase
cycle (`Plan → Generate → Evaluate`) to the concrete file-and-subagent
sequence the orchestrator must follow for every masterplan step. It
is NOT optional. Every step follows every phase. No step is exempt.

This is the checklist the *orchestrator* (main session) follows for every masterplan step. It is not an agent file -- it describes the sequence in which to spawn agents.

Source of truth: `handoff/current/phase-<id>-contract.md` + CLAUDE.md §Harness Protocol + PLAN.md §Research Gate. This file consolidates them into an executable sequence so we stop drifting.

## The five phases (in order, every step)

### 1. RESEARCH

Spawn the `researcher` subagent with an explicit effort tier:

- `simple` -- the prior substep in the same phase already cited the primary references and today's work reuses them. Example: 4.5.4 reusing the PSR/DSR formulas 4.5.1 already researched.
- `moderate` -- new subtopic with 2-3 authoritative sources to reconcile. Default.
- `complex` -- novel domain.

Pass: step id, success criteria (verbatim from `.claude/masterplan.json`), existing references from the phase-level contract.

Receive: `{report_md, urls_collected, sources_read_in_full, gate_passed}`.

If `gate_passed` is false, re-spawn with a higher tier or do the research yourself. Do not proceed to PLAN without gate_passed.

### 2. PLAN

Write `handoff/current/<step_id>-contract.md`. Required sections:

- **Hypothesis** -- falsifiable, testable by the step's verification command.
- **Success criteria** -- copied verbatim from `.claude/masterplan.json`, each annotated with the research-backed threshold (cite the source from §1).
- **Design** -- files to create/modify with absolute paths.
- **Anti-patterns guarded** -- at least 2, pulled from §1 research.
- **Out of scope** -- explicit.
- **Risk** -- what can still go wrong after this step passes.

Verification criteria are **immutable** (Anthropic: "unacceptable to remove or edit tests"). If a criterion turns out to be wrong, stop; do not silently rewrite it.

### 3. GENERATE

Do the work. Run the verification command before moving on:

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
# step's verification command from .claude/masterplan.json
```

Run inline unit/endpoint smokes too. Keep test data synthetic -- no live yfinance or BQ round-trips during GENERATE.

### 4. EVALUATE (cross-verification, both verifiers)

Never self-evaluate. Spawn both verifiers. They can run in parallel.

**Spawn order (parallel):**

1. `harness-verifier` (read-only, in-place, sonnet) -- fast deterministic checks against uncommitted files.
2. `qa-evaluator` (read-only, opus) -- deeper cross-verification. By default runs in-place; pass `isolation: "worktree"` only if the step is being evaluated post-commit.

Each returns the JSON schema documented in its agent file with `ok`, `violated_criteria`, `violation_details: [{violation_type, action, state, constraint}]`, and `certified_fallback`.

**Disagreement resolution (orchestrator's call, not the agents'):**

- Both `ok: true` -> proceed to LOG.
- Both `ok: false` -> fix or revert; increment `retry_count` in `.claude/masterplan.json`.
- Split -> treat as `ok: false` and investigate which verifier is right (usually the deterministic one). Document the split in the harness log entry.

**Certified fallback (SEVerA 2026):** when `retry_count >= max_retries` (3 in the masterplan schema), revert to the last known-good state rather than blocking -- typically `backend/backtest/experiments/optimizer_best.json` or the previous `git HEAD`.

### 5. LOG

Append to `handoff/harness_log.md` using the phase-4.5 cycle format:

```
---
## Phase 4.5 -- Step <id> <name> (YYYY-MM-DD)

**Research:** <tier>, N URLs collected, M full reads, primary refs: ...
**Plan:** contract at handoff/current/<step_id>-contract.md; key design choices.
**Generate:** files created/modified with line counts; verification command result.
**Evaluate:** harness-verifier PASS/FAIL; qa-evaluator PASS/FAIL; disagreement (if any).
**Decision:** PASS / FAIL / certified_fallback.
**Reality-gap note:** did Sharpe/DSR change? any strategy mutation?
**Next actionable step:** <id>.
```

Then (and only then) update `.claude/masterplan.json` to set the step `status: "done"`. The masterplan write triggers `archive-handoff.sh` which moves `handoff/current/<step_id>-contract.md` to `handoff/archive/phase-<id>/`.

## Anti-patterns to reject at orchestration time

1. **Skip RESEARCH** because "we've been here before" -- if the step is new, the tier can be `simple` but the phase cannot be skipped. No step is exempt.
2. **Self-evaluation** -- orchestrator directly reporting PASS without spawning the verifier pair. The TaskCompleted hook should fire; if it didn't, spawn manually.
3. **Rewrite success criteria** to fit an incomplete implementation. Criteria are immutable.
4. **Batched done-marking** -- marking multiple steps done in one masterplan write. Write once per step so the archive hook can attribute artifacts correctly.
5. **Ignore verifier disagreement** -- the orchestrator resolves splits explicitly; never pick the more convenient verdict silently.

## Hook sanity (prevents "No such file or directory" errors)

All hook commands in `.claude/settings.json` now use `"${CLAUDE_PROJECT_DIR:-$(pwd)}"/.claude/hooks/X.sh` so they resolve from any spawned subagent cwd. Hook scripts themselves already use `$CLAUDE_PROJECT_DIR` internally.

If a PostToolUse hook fires from within a subagent's cwd and `$CLAUDE_PROJECT_DIR` is unset, the fallback `$(pwd)` will fail gracefully rather than silently running against the wrong directory.

## References (read in full this cycle)

- Anthropic -- "How We Built Our Multi-Agent Research System" (2024) + "Harness Design for Long-Running Apps" + "Effective Harnesses for Long-Running Agents"
- Anthropic -- "Building Effective Agents" (seven-pattern canonical doc)
- Claude Code docs -- Agent Teams, Hooks, Memory
- VeriPlan (arXiv:2502.17898, 2025) -- violation_details output schema
- SAVeR (arXiv:2604.08401, 2026) -- 6 violation-type taxonomy
- SEVerA (arXiv:2603.25111, 2026) -- certified fallback on exhausted retries
- Google Research (2025) -- 80.9% parallel gain vs 39-70% sequential-reasoning penalty
- Kleppmann (2025) -- deterministic-first verification rationale
