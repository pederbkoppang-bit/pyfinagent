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
     │ Researcher             │ │ Q/A               │
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

**Launch — Workflow structured-output is FIRST-CLASS; Agent-tool is the
fallback (phase-71.1).** The primary unattended launch is the checked-in
`.claude/workflows/qa-verdict.js` script (run it via the Workflow tool
with `args={step_id, criteria[], verification_command, evidence}`, or an
equivalent inline Workflow script). It runs the Q/A role as
`agent(prompt, {schema, agentType:'general-purpose', model:'opus',
effort:'max'})`, and **the verdict is the captured return value** —
structured-outputs GA (constrained decoding) guarantees the shape, so it
does NOT depend on a subagent file-write flush. This is the
stall-immune path: the Agent-tool subagent end-flush stalled 6× on
2026-07-11 (intermittent, model-agnostic; auto-memory
`feedback_workflow_qa_when_subagents_stall`). The Agent-tool `qa`
subagent (`Agent(subagent_type:'qa')`) is the documented **fallback**
(and the worktree-isolation CI path). **Main MUST transcribe the
returned verdict VERBATIM** into `handoff/current/evaluator_critique.md`
(no editorial edits) so the no-self-eval guarantee holds — Main records
the verdict, never authors it. An errored/empty return is **NO VERDICT,
never PASS**: fall back to the Agent-tool path. The Q/A returns a verdict
and STOPS; it never loops fix→re-grade internally. Single-Q/A-per-step
and the exactly-3-agents doctrine are unchanged — the Workflow path is a
launch mechanism, not a fourth agent.

Q/A runs deterministic-first:
1. Syntax / file-existence / `verification.command` exit code
2. Reads existing `handoff/current/evaluator_critique.md` +
   `experiment_results.md`
3. Optional harness dry-run (scoped-tests tier of the verification budget)
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

#### Retry-on-FAIL loop (phase-23.2.24, formalised)

When Q/A returns `verdict: FAIL` or `verdict: CONDITIONAL`, the
canonical recovery loop is:

1. **Main reads** the verdict's `violated_criteria` +
   `violation_details` from `handoff/current/evaluator_critique.md`.
   Do NOT respawn Q/A first; you'd be second-opinion-shopping on
   unchanged evidence.
2. **Main fixes** the blockers AND updates the handoff files:
   - Code/doc changes for whatever was flagged.
   - Append a "Cycle-2 follow-up (post-Q/A-N)" section to
     `experiment_results.md` naming what changed and why. This is
     the file-based handoff Anthropic's harness-design doc
     prescribes: communication via files, fresh agent instances.
3. **Main spawns a FRESH Q/A** (new subagent instance, not
   `SendMessage` to the prior one). The fresh Q/A reads the UPDATED
   files; the new verdict reflects the FIX, not a different opinion
   on the same evidence.
4. **Loop ceiling**: max 3 retries before `certified_fallback`. The
   3rd-CONDITIONAL auto-FAIL clause below complements this — if a
   step accumulates 3 CONDITIONALs without intervening PASS/FAIL,
   the next Q/A must FAIL outright.

This is the documented pattern (Anthropic harness-design + how-we-built
multi-agent-research blog 2026 — "if more research is needed it can
create additional subagents"). It is NOT verdict-shopping. The
distinguishing test: did the FILES CHANGE between Q/A-N and Q/A-(N+1)?
If yes → legitimate retry. If no → forbidden second-opinion-shop.

GitHub Copilot Code Review (researcher's audit, phase-23.2.24) is
strictly weaker — Copilot only posts "Comment" reviews, never blocks
merges, and has no automatic re-review on push without explicit
ruleset config. The pyfinagent harness's blocking FAIL→Main→fresh-Q/A
loop is structurally stronger; do NOT downgrade to mirror Copilot.

#### CONDITIONAL escalation clause (3rd-CONDITIONAL auto-FAIL)

A CONDITIONAL verdict is appropriate when: (a) underlying functionality
is intact, (b) production code paths are unaffected, and (c) the step
was designed to discover a gap rather than deliver a fix. CONDITIONAL
is NOT an indefinite soft-pass.

If a single masterplan step-id accumulates 3 or more consecutive
CONDITIONAL verdicts without an intervening PASS or FAIL, the next
Q/A pass MUST return FAIL -- not another CONDITIONAL. This prevents
the harness from functioning as a logger rather than a corrector
(shared-evaluator-bias forcing function; see mergeshield.dev 2026
"What's Missing from Anthropic's Multi-Agent Harness").

Q/A procedure: before issuing a CONDITIONAL verdict, grep
`handoff/harness_log.md` for the current step-id and count prior
`result=CONDITIONAL` entries. If the count is already 2, the verdict
must be FAIL with `violation_type: Unjustified_Inference`.

Counter resets after: a PASS verdict, a FAIL verdict, or a new
step-id (which is a structurally distinct problem and starts fresh).

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
   spawning Q/A. Always spawn Q/A manually after every GENERATE;
   there is no hook backstop (the TaskCompleted hook was retired in
   phase-23.8.2 per audit recommendation R-2 — see
   `docs/audits/dev-mas-2026-05-11/04-remediation.md`).
3. **Rewrite success criteria** to fit an incomplete implementation.
   Criteria are immutable.
4. **Batched done-marking** — marking multiple steps done in one
   masterplan write. Write once per step so the archive hook can
   attribute artifacts correctly.
5. **Second-opinion shopping** — spawning a fresh Q/A on UNCHANGED
   evidence, hoping for a different verdict. The legitimate recovery
   (§4 Retry-on-FAIL loop, CLAUDE.md canonical cycle-2 flow) is: fix
   the blockers, update the handoff evidence files, THEN spawn a FRESH
   Q/A that reads the updated files. The distinguishing test: did the
   files change between spawns? Changed → legitimate retry; unchanged
   → forbidden verdict-shop.
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
- Main self-evaluates under time pressure. Fix: always spawn Q/A
  explicitly after every GENERATE — no automatic hook backstop.
  (The TaskCompleted hook was retired in phase-23.8.2 because the
  audit found it was a weaker, parallel evaluator that diluted
  Q/A's independence rather than reinforcing it.)
- Main second-opinion-shops after CONDITIONAL. Fix: require the
  documented retry loop — fix blockers, update the handoff evidence,
  spawn a FRESH Q/A on the changed files (respawn on unchanged
  evidence stays forbidden).

## Subagent runtime semantics (Claude Code v2.1.198+; phase-67.5)

- Subagent spawns run in the BACKGROUND by default (v2.1.198); Main is
  re-invoked on completion. Do not busy-wait or poll transcripts.
- Subagents cut off by rate limits or errors KEEP their partial work
  (v2.1.199). After a stall-class cutoff, read the partial artifacts
  (e.g. the incrementally-written research brief -- the write-first
  discipline exists exactly for this) before respawning.
- Subagents can spawn their own subagents to 5 levels (v2.1.172) when
  granted the Agent tool; the researcher deep-tier fork may be
  self-managed on a per-spawn grant (researcher.md deep-tier item 4).
  This does NOT change the 3-agent doctrine: forks are more instances
  of the SAME role, never new roles.
- settings.json `fallbackModel` (phase-67.5) covers OVERLOAD-class
  model failures only -- rate-limit/usage-limit cutoffs never trigger
  a model switch (partial-work retention covers those instead).
- **Workflow structured-output path (phase-71.1; first-class Q/A +
  Researcher launch).** A Workflow script runs each `agent()` stage in
  an isolated environment; intermediate results stay in SCRIPT
  VARIABLES, and the schema-validated return value is captured directly
  -- so the verdict/envelope does NOT depend on a subagent file-write
  flush (the fix for the model-agnostic end-flush stall). A run is
  RESUMABLE within the SAME session (`resumeFromRunId`); a new session
  restarts it. Checked-in scripts live under `.claude/workflows/` (e.g.
  `qa-verdict.js`) with a `meta{name,description}` block and become named
  commands. Workflow agents inherit Main's SESSION model unless the
  `agent()` opts route otherwise -- so the Q/A/Researcher stages set
  `model:'opus'` EXPLICITLY (never route the gate off Opus: rider-trap
  R4). Do NOT add a Monitor/transcript-mtime watchdog around it
  (rider-trap R11 -- the return-value path makes polling unnecessary and
  it contradicts the do-not-poll rule above).

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
