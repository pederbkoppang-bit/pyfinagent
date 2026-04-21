# Per-Step Harness Protocol (operator runbook)

**Canonical references (must read before every long-running session):**
- Anthropic, "Harness Design for Long-Running Apps": https://www.anthropic.com/engineering/harness-design-long-running-apps
- Anthropic, "How We Built Our Multi-Agent Research System": https://www.anthropic.com/engineering/built-multi-agent-research-system
- Anthropic, "Building Effective Agents": https://www.anthropic.com/engineering/building-effective-agents

**This file is the executable mapping** from the Anthropic three-phase
cycle (`Plan → Generate → Evaluate`) to the concrete file-and-subagent
sequence the orchestrator must follow for every masterplan step. It
is NOT optional. Every step follows every phase. No step is exempt.

This is the checklist the *orchestrator* (Main) follows for every
masterplan step. It is not an agent file — it describes the sequence
in which to spawn agents.

Source of truth: `handoff/current/phase-<id>-contract.md` +
CLAUDE.md §Harness Protocol. This file consolidates them into an
executable sequence so we stop drifting.

## The 3-agent MAS

```
┌──────────────────────────────────────────────────────────────┐
│ Main (orchestrator — main Claude Code session)                │
│   - Plans, delegates, synthesizes, decides                    │
│   - Writes contracts, handoffs, harness log                   │
│   - Flips masterplan step status                              │
│   - NEVER self-evaluates. NEVER skips research.               │
└──────────────────┬─────────────────────┬─────────────────────┘
                   │                     │
     ┌─────────────▼──────────┐ ┌────────▼──────────┐
     │ Researcher (sonnet)    │ │ Q/A (opus)        │
     │ (merged with Explore)  │ │ (merged with      │
     │                        │ │  harness-verifier)│
     │ External docs + papers │ │ Deterministic     │
     │ + blogs + GitHub       │ │ checks + LLM      │
     │ AND                    │ │ judgment          │
     │ Internal code grep +   │ │                   │
     │ read + inventory       │ │ Reproduces the    │
     │                        │ │ immutable         │
     │ Output: report_md,     │ │ verification      │
     │ urls_collected,        │ │ command, reads    │
     │ internal_files,        │ │ evaluator_critique│
     │ gate_passed            │ │ , returns ok +    │
     │                        │ │ verdict + reason  │
     └────────────────────────┘ └───────────────────┘
```

There are exactly 3 agents. There is no separate `harness-verifier`
(merged into Q/A). There is no separate `Explore` subagent (merged
into Researcher). If anyone proposes re-adding either, reject it —
that's re-splitting for the sake of splitting.

## The five phases (in order, every step)

### 1. RESEARCH

Spawn the `researcher` subagent with an explicit effort tier:

- `simple` — the prior substep in the same phase already cited the
  primary references and today's work reuses them.
- `moderate` — new subtopic with 2–3 authoritative sources to
  reconcile. Default.
- `complex` — novel domain.

Pass: step id, success criteria (verbatim from
`.claude/masterplan.json`), existing references from the
phase-level contract, AND any internal code modules to audit.

Receive: `{report_md, urls_collected, sources_read_in_full,
internal_files_inspected, gate_passed}`.

**The Researcher covers BOTH halves in one turn** — external docs
AND internal code. Never split into researcher + Explore; that's
the old pattern.

If `gate_passed` is false, re-spawn with a higher tier or do the
research yourself. Do not proceed to PLAN without gate_passed.

### 2. PLAN

Write `handoff/current/<step_id>-contract.md`. Required sections:

- **Hypothesis** — falsifiable, testable by the step's verification
  command.
- **Success criteria** — copied verbatim from
  `.claude/masterplan.json`, each annotated with the research-backed
  threshold (cite the source from §1).
- **Design** — files to create/modify with absolute paths.
- **Anti-patterns guarded** — at least 2, pulled from §1 research.
- **Out of scope** — explicit.
- **Risk** — what can still go wrong after this step passes.

Verification criteria are **immutable** (Anthropic: "unacceptable to
remove or edit tests"). If a criterion turns out to be wrong, stop;
do not silently rewrite it.

### 3. GENERATE

Do the work. Run the verification command before moving on:

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
# step's verification command from .claude/masterplan.json
```

Run inline unit/endpoint smokes too. Keep test data synthetic — no
live yfinance or BQ round-trips during GENERATE.

### 4. EVALUATE (single Q/A agent, cross-verification)

Never self-evaluate. Spawn the Q/A agent ONCE (no more pair spawn).

Q/A runs deterministic-first:
1. Syntax / file-existence / `verification.command` exit code
2. Reads existing `handoff/current/evaluator_critique.md` +
   `experiment_results.md`
3. Optional harness dry-run (under 55s budget)
4. LLM judgment on contract alignment, scope honesty, mutation-
   resistance, and research-gate compliance

Returns the JSON schema documented in `.claude/agents/qa.md` with
`ok`, `verdict`, `violated_criteria`,
`violation_details: [{violation_type, action, state, constraint}]`,
and `certified_fallback`.

**Certified fallback (SEVerA 2026):** when `retry_count >=
max_retries` (3 in the masterplan schema), revert to the last
known-good state rather than blocking — typically
`backend/backtest/experiments/optimizer_best.json` or the previous
`git HEAD`.

**Disagreement with prior automated check:** if the step's
deterministic verification command passes but Q/A's LLM judgment
fails, the LLM judgment wins (scope honesty, anti-rubber-stamp,
mutation-resistance are load-bearing). Log the split in
`harness_log.md`.

### 5. LOG

Append to `handoff/harness_log.md` using the Cycle format:

```
## Cycle N -- YYYY-MM-DD -- phase=X.Y result=PASS/CONDITIONAL/FAIL

**Step**: <id> <name>
**Research**: tier, N URLs, M full reads, I internal files.
**Plan**: contract path; key design choices.
**Generate**: files changed, verification cmd result.
**Evaluate**: Q/A verdict + reason.
**Decision**: PASS / FAIL / certified_fallback.
**Next**: <next step id>.
```

Then (and only then) update `.claude/masterplan.json` to set the
step `status: "done"`. The masterplan write triggers
`archive-handoff.sh` which moves `handoff/current/<step_id>-
contract.md` to `handoff/archive/phase-<id>/`.

## Anti-patterns to reject at orchestration time

1. **Skip RESEARCH** because "we've been here before" — if the step
   is new, the tier can be `simple` but the phase cannot be skipped.
   No step is exempt. (The research-gate miss on 7 of 9 phase-4.8
   cycles started this way.)
2. **Self-evaluation** — Main directly reporting PASS without
   spawning Q/A. The TaskCompleted hook should fire; if it didn't,
   spawn Q/A manually.
3. **Rewrite success criteria** to fit an incomplete implementation.
   Criteria are immutable.
4. **Batched done-marking** — marking multiple steps done in one
   masterplan write. Write once per step so the archive hook can
   attribute artifacts correctly.
5. **Second-opinion shopping** — if Q/A returns CONDITIONAL, fix the
   blockers then SendMessage back to the SAME agent. Do NOT spawn a
   fresh Q/A and hope for PASS.
6. **Re-split agents** — reintroducing `Explore` as a separate
   subagent, or `harness-verifier` as a separate evaluator, after
   they've been merged. That's the old pattern. The new MAS is 3
   agents: Main + Researcher + Q/A.

## Why Main drifts: documented drift modes

From auto-memory `feedback_research_gate.md` and the phase-4.10
sub-agents audit:
- Main skips Researcher when it "feels confident" about the topic.
  Fix: `InstructionsLoaded` hook reloads this rule every session
  start; Researcher description uses "MUST BE USED" phrasing.
- Main self-evaluates under time pressure. Fix: TaskCompleted hook
  is load-bearing; never bypass.
- Main second-opinion-shops after CONDITIONAL. Fix: require
  SendMessage-to-same-agent after any fix.

## Hook sanity (prevents "No such file or directory" errors)

All hook commands in `.claude/settings.json` use
`"${CLAUDE_PROJECT_DIR:-$(pwd)}"/.claude/hooks/X.sh` so they resolve
from any spawned subagent cwd. Hook scripts themselves already use
`$CLAUDE_PROJECT_DIR` internally.

## References (read in full this cycle)

- Anthropic — "How We Built Our Multi-Agent Research System" (2024)
- Anthropic — "Harness Design for Long-Running Apps" (2025)
- Anthropic — "Building Effective Agents" (seven-pattern canonical)
- Claude Code docs — Sub-agents, Hooks, Memory
- VeriPlan (arXiv:2502.17898, 2025) — violation_details output
- SAVeR (arXiv:2604.08401, 2026) — 6 violation-type taxonomy
- SEVerA (arXiv:2603.25111, 2026) — certified fallback on exhausted
  retries
- Kleppmann (2025) — deterministic-first verification rationale
