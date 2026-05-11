---
step: phase-23.8.2
title: Delete TaskCompleted hook (audit recommendation R-2 Option A)
cycle_date: 2026-05-11
harness_required: true
verification: 'source .venv/bin/activate && python3 tests/verify_phase_23_8_2.py'
research_brief: (researcher subagent, 2026-05-11; gate_passed=true; 6 sources read in full, 16 URLs, recency scan)
audit_basis: docs/audits/dev-mas-2026-05-11/04-remediation.md (R-2)
---

# Contract — phase-23.8.2

**Step**: phase-23.8.2 — Delete the TaskCompleted hook (R-2 Option A).

**Date**: 2026-05-11.

**Status target**: pending → done.

**Hypothesis**:
The TaskCompleted hook agent at `.claude/settings.json:101-111` is
a redundant evaluator that overlaps Q/A (per the audit's H-2
BLOCKING finding). Deleting it cleanly removes the shadow MAS
without breaking any active code path — confirmed by the
researcher: no `scripts/` or `backend/` reference, hook absence
is a graceful no-op per the official Claude Code hooks doc, and
the only "breakage" is the historical step 2.13 verification
command's assertion (which is an immutable historical artifact,
not a live workflow).

## Research-gate summary

Researcher subagent ran 2026-05-11 (tier: simple). JSON envelope:
`{"external_sources_read_in_full": 6, "snippet_only_sources": 10,
"urls_collected": 16, "recency_scan_performed": true,
"internal_files_inspected": 7, "gate_passed": true}`.

Key external citations (≥5 sources read in full via WebFetch):

1. HARNESS-DOC
   (`https://www.anthropic.com/engineering/harness-design-long-running-apps`)
   — "agents tend to respond by confidently praising the work";
   "every component in a harness encodes an assumption ... worth
   stress testing." The redundant evaluator's assumption (Q/A
   would be absent or unreliable) is obsolete now that Q/A is the
   canonical evaluator.
2. EFFECTIVE-DOC
   (`https://www.anthropic.com/research/building-effective-agents`)
   — "you should consider adding complexity *only* when it
   demonstrably improves outcomes." Two evaluators on the same
   step are added complexity with no marginal benefit.
3. Claude Code hooks doc (`https://code.claude.com/docs/en/hooks`)
   — confirmed: if a hook event key is absent from `hooks {}`, no
   error is raised; events simply fire no hooks. Graceful no-op.
4. najx.dev CI/CD anti-patterns — "A complex pipeline with too
   many stages or steps hampers understandability and
   maintainability."
5. Effective-harnesses-for-long-running-agents — Anthropic
   reaffirms minimal scaffolding.
6. Epsilla blog on Anthropic harness — GAN-style single
   Generator + single Evaluator per step.

Recency scan: no 2024-2026 source defends running multiple
redundant evaluators on the same artifact. Defense-in-depth
literature targets external adversaries, not internal harness
design.

### Research-gate red flags + action taken

The researcher found ZERO active code-path dependencies on the
TaskCompleted hook (grep across `scripts/`, `backend/`,
`.claude/agents/`, frontend). The audit's H-1 / H-2 BLOCKING
findings remain valid.

**Known controlled breakage** (documented, not blocking):

- `.claude/masterplan.json:214` — step 2.13 (`done`, historical
  Claude Code Configuration Audit) has an immutable verification
  command containing `assert 'TaskCompleted' in s['hooks']`. Per
  CLAUDE.md, verification criteria are immutable. The assertion
  PASSED at the time step 2.13 was marked done. After this cycle,
  re-running step 2.13's verification command would fail. This is
  acceptable because:
  - Step 2.13 is `done`; its verification ran once and is a
    historical record, not a recurring check.
  - The R-2 audit recommendation EXPLICITLY supersedes the
    historical assertion (the audit was written 2026-05-11, well
    after step 2.13 landed).
  - The success_criteria field for 2.13 names "TaskCompleted hook
    exists" — but the criterion was already attested at the time.
  - No automated process re-runs old verification commands.

This breakage is captured in the verifier's claim 8 (an
EXPECTED-FAIL test for step 2.13's command, with a clear comment
explaining the trade-off).

## Plan steps

### G-1 — Delete the TaskCompleted hook from `.claude/settings.json`

Remove the entire `"TaskCompleted": [...]` key + value block
(lines 100-111). The surrounding hook-event blocks remain
untouched.

### G-2 — Update `.claude/context/project.md`

Line 19 contains: "Hooks: changelog on commit, memory sync on
masterplan changes, **TaskCompleted gate**, Stop gate,
TeammateIdle". Remove "TaskCompleted gate," from this list. The
context snapshot should reflect the new hook roster.

### G-3 — Update `docs/runbooks/per-step-protocol.md`

Two specific lines reference TaskCompleted:

- **Line 225-227**: "Self-evaluation — Main directly reporting
  PASS without spawning Q/A. The TaskCompleted hook should fire;
  if it didn't, spawn Q/A manually." Reframe to: "Self-evaluation
  — Main directly reporting PASS without spawning Q/A. Always
  spawn Q/A manually after every GENERATE; there is no hook
  backstop (the TaskCompleted hook was retired in phase-23.8.2 per
  audit recommendation R-2 — see `docs/audits/dev-mas-2026-05-11/04-remediation.md`)."
- **Line 248-249**: "Main self-evaluates under time pressure.
  Fix: TaskCompleted hook is load-bearing; never bypass."
  Reframe to: "Main self-evaluates under time pressure. Fix:
  always spawn Q/A explicitly after every GENERATE — no automatic
  hook backstop. (The TaskCompleted hook was retired in
  phase-23.8.2 because the audit found it was a weaker, parallel
  evaluator that diluted Q/A's independence rather than
  reinforcing it.)"

### G-4 — Add verifier `tests/verify_phase_23_8_2.py`

10 immutable claims:

1. `.claude/settings.json` valid JSON.
2. `.claude/settings.json` has NO `TaskCompleted` key in `hooks`.
3. `.claude/settings.json` STILL has `PostToolUse`, `Stop`,
   `PreToolUse`, `InstructionsLoaded`, `ConfigChange`,
   `SubagentStop`, `TeammateIdle` (no other hooks deleted).
4. `.claude/context/project.md` line ~19 no longer contains the
   string `"TaskCompleted gate,"`.
5. `docs/runbooks/per-step-protocol.md` no longer contains
   "The TaskCompleted hook should fire" (the old line 226 prose).
6. `docs/runbooks/per-step-protocol.md` no longer contains
   "TaskCompleted hook is load-bearing" (the old line 248 prose).
7. `docs/runbooks/per-step-protocol.md` DOES contain the phrase
   "retired in phase-23.8.2" (the new replacement prose).
8. **EXPECTED-FAIL**: step 2.13's historical verification command
   would now fail if re-run. The verifier RUNS the assertion
   directly (`assert 'TaskCompleted' in s['hooks']`) and asserts
   it raises `AssertionError`. If it does NOT raise (i.e. the
   hook is still present), THIS claim fails. This is an
   expected-fail-of-old-assertion test, NOT a contradiction —
   the audit explicitly supersedes that historical check.
9. `handoff/harness_log.md` Cycle 39 contains a verbatim breakage
   disclosure for step 2.13's historical verification command,
   citing CLAUDE.md immutability + audit R-2 + audit findings
   H-1 / H-2.
10. No bash-syntax regressions on the remaining 9 hook scripts.

## Files expected to change

| File | Type | Change |
|---|---|---|
| `.claude/settings.json` | edit | remove `TaskCompleted` block (~10 lines deleted) |
| `.claude/context/project.md` | edit | remove "TaskCompleted gate," from hooks list (line 19) |
| `docs/runbooks/per-step-protocol.md` | edit | rewrite 2 prose lines (225-227, 248-249) |
| `tests/verify_phase_23_8_2.py` | NEW | 10-claim verifier |
| `handoff/current/contract.md` | NEW (this file) | contract |
| `handoff/current/experiment_results.md` | NEW (later) | by GENERATE |
| `handoff/current/evaluator_critique.md` | NEW (later) | by Q/A |
| `handoff/harness_log.md` | append | Cycle 39 with 2.13 breakage disclosure |
| `.claude/masterplan.json` | edit | add 23.8.2 pending; flip to done at end |

## Immutable success criteria

(Same as the verifier's 10 claims above.)

## Rollback note

Single-commit revert. Restoring the TaskCompleted block reverses
the change cleanly. The hook agent has no persistent state.

If a future operator needs the redundant evaluator back, they can
either restore from this commit's parent OR add a new
`TaskCompleted` block to settings.json with a tools-restricted
subagent file under `.claude/agents/` (the audit's R-2 Option B
path). Either path is reversible.

## Out of scope (explicit)

- **R-5** (qa.md fail-mode change) — needs separate session +
  Peder review.
- **R-6** (delete deprecated stubs) — needs prior
  `autonomous_loop.py` refactor.
- **qa.md update** to mention `live_check_<step_id>.md` in
  existing_results_check — deferred from cycle 38 per
  separation-of-duties.
- **Updating the Stop hook** — also an unnamed hook agent but
  fires only at session end, different role, not redundant with
  Q/A. Audit's H-3 was DEGRADES not BLOCKING. Out of scope here.
- **Updating step 2.13's verification command** — forbidden by
  CLAUDE.md immutability rule.

## References

- `docs/audits/dev-mas-2026-05-11/04-remediation.md` R-2 — the
  proposal (chose Option A: delete).
- `docs/audits/dev-mas-2026-05-11/02-per-agent.md` findings H-1
  + H-2 — BLOCKING justification.
- `.claude/settings.json:101-111` — the block being deleted.
- `.claude/masterplan.json:214` — step 2.13's historical
  verification, the controlled breakage.
- Researcher subagent JSON envelope: `gate_passed: true`.
