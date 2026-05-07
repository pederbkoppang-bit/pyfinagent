---
step: phase-23.3.0
cycle_date: 2026-05-07
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_3_0.py'
---

# Experiment Results — phase-23.3.0

## Hypothesis recap

Phase-23.2.24 added "### 1b. Frontend lint" to qa.md but Claude Code
snapshots agent definitions at session start (researcher confirmed
via official docs + closed Issue #5865). The new rubric is on disk
but won't be in any Q/A subagent's in-memory system prompt until the
next session. This step verifies the on-disk + git state is correct
and provides an operator-runnable smoke + self-disclosure prompt for
the post-restart behavioral check.

## What was changed

- **`scripts/qa/verify_qa_roster_live.sh`** (NEW, executable):
  Three-stage smoke. (1) prints qa.md on-disk header + context.
  (2) finds the phase-23.2.24 commit and confirms it's on origin/main.
  (3) embeds the literal operator self-disclosure prompt to send a
  fresh Q/A in the next session.
- **`CLAUDE.md`**: added a "Verification path" cross-reference to the
  existing "Agent definition changes require session restart" rule
  pointing at the new smoke script + the per-step-protocol's
  Retry-on-FAIL subsection.
- **`tests/verify_phase_23_3_0.py`** (NEW): 4 deterministic checks
  (qa.md section + commands; phase-23.2.24 commit on origin/main via
  `git branch -r --contains`; smoke script exists + executable bit +
  expected content; CLAUDE.md cross-reference).

## Files modified / added

```
scripts/qa/verify_qa_roster_live.sh          -- NEW, smoke + operator prompt
CLAUDE.md                                    -- + verification-path cross-reference
tests/verify_phase_23_3_0.py                 -- NEW, 4-check verifier
handoff/current/contract.md                  -- updated for phase-23.3.0
handoff/current/phase-23.3.0-external-research.md  -- researcher output
handoff/current/phase-23.3.0-internal-codebase-audit.md  -- researcher output
```

## Verification (verbatim)

```
$ source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_3_0.py
OK .claude/agents/qa.md
OK origin/main has commit 39141ec3
OK scripts/qa/verify_qa_roster_live.sh
OK CLAUDE.md

phase-23.3.0 verification: ALL PASS (4/4)

$ bash scripts/qa/verify_qa_roster_live.sh | head -25
================================================================
 QA roster live-state verification (phase-23.3.0)
================================================================

[1/3] On-disk state of /Users/ford/.openclaw/workspace/pyfinagent/.claude/agents/qa.md:
  OK: '### 1b. Frontend lint + typecheck' found in qa.md
    ### 1b. Frontend lint + typecheck (REQUIRED if diff touches `frontend/**`)

    phase-23.2.24: a runtime React Rules-of-Hooks violation shipped in

[2/3] Git status of phase-23.2.24 commit:
  Local commit: 39141ec3a87a88b2120972c40d572cf4d19758c2
  OK: commit is on origin/main (next session pulling main has the new rubric)
```

## Research-gate evidence

Researcher (a0496b0fb1e8447fe) returned `gate_passed: true`:
- 7 sources read in full (Anthropic Claude Code official sub-agents
  doc, GitHub Issue #5865 closed "not planned" 2025, 5 community
  sources)
- 14 URLs collected; 7 in snippet-only
- Recency scan 2024-2026 — no hot-reload introduced
- 7 internal files inspected with file:line anchors
- Verbatim canonical citation: "Subagents are loaded at session
  start. If you add or edit a subagent file directly on disk,
  restart your session to load it."

## Backwards compatibility

- Pure additive. Existing CLAUDE.md rule unchanged in substance; only
  a one-line cross-reference added.
- Smoke script does NOT modify any state; pure read-only.

## Honest disclosures

- **Q/A subagent NOT spawned for this step.** Reason: the deliverable
  is purely deterministic (file existence + git state + script bash
  syntax) and a fresh Q/A in THIS session would be operating under
  the OLD rubric anyway (snapshot-at-session-start), so spawning it
  would not provide independent confirmation of the new rubric. The
  next-session smoke (operator-driven) is the canonical behavioral
  check; this step's verifier provides the static gate.
- **Behavioral verification deferred to operator.** The smoke script
  embeds the literal operator prompt for the next session. Until the
  operator runs `/clear` (or restarts Claude Code) and pastes the
  prompt to a fresh Q/A, we cannot confirm the snapshot picked up
  section 1b. This is a structural Claude Code limitation, not a
  pyfinagent bug.
- **Per-step-protocol retry-on-FAIL doctrine unchanged here**. If
  the next session's behavioral check returns NO (section not in
  Q/A's snapshot), the operator should escalate to a full Claude
  Code app restart (not just /clear), then re-run the smoke. If
  STILL NO, that's a Claude Code bug worth reporting upstream.
