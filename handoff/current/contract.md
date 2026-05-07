---
step: phase-23.3.0
title: Q/A roster verification — confirm new "1b. Frontend lint" rubric will be live in next session
cycle_date: 2026-05-07
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_3_0.py'
research_brief: handoff/current/phase-23.3.0-external-research.md (also see phase-23.3.0-internal-codebase-audit.md)
---

# Contract — phase-23.3.0

## Hypothesis

Phase-23.2.24 added "### 1b. Frontend lint + typecheck" to
`.claude/agents/qa.md`. CLAUDE.md states agent definitions are
snapshotted at session start. The Q/A subagent that ran in the SAME
session as the edit could not have section 1b in its in-memory system
prompt — yet its critique reported the section present. Resolution
(per researcher): the Q/A read the FILE FROM DISK via the Read tool,
which reflects current disk state. That READ is NOT the same as its
own system prompt. The new rubric will only be reliably resident in
fresh subagent spawns starting from the NEXT session.

This step's job is verification: prove the on-disk state is correct,
prove the change is committed and pushed (so the next session picks
it up regardless of branch), and document the next-session
verification protocol the operator can run.

## Research-gate summary

Researcher (a0496b0fb1e8447fe) returned `gate_passed: true`:
- 7 sources read in full (Anthropic Claude Code official sub-agents
  doc + Issue #5865 closed "not planned" 2025 + 5 community sources)
- 14 URLs collected; 7 in snippet-only
- Recency scan 2024-2026 — no hot-reload introduced
- 7 internal files inspected
- Concrete recommendation: literal next-session verification command

Key finding (verbatim from official docs):
"Subagents are loaded at session start. If you add or edit a subagent
file directly on disk, restart your session to load it. Subagents
created through the `/agents` interface take effect immediately
without a restart." (https://code.claude.com/docs/en/sub-agents)

## Immutable success criteria (verbatim — DO NOT EDIT)

1. The on-disk state of `.claude/agents/qa.md` includes the literal
   line `### 1b. Frontend lint + typecheck` AND a literal `npx eslint .`
   AND a literal `tsc --noEmit` command in the surrounding code block.
2. `git log origin/main` includes the phase-23.2.24 commit subject
   "phase-23.2.24: fix Rules-of-Hooks bug + harden Q/A with ESLint
   coverage" so any next session pulling main will have the new rubric.
3. A new operator-runnable smoke `scripts/qa/verify_qa_roster_live.sh`
   exists and prints (a) the on-disk section header, (b) the relevant
   lines around it, and (c) a manual-procedure note instructing the
   operator to spawn Q/A in a NEW session and ask the standard
   self-disclosure question.
4. A new doc cross-reference is added to `CLAUDE.md` near the existing
   "Agent definition changes require session restart" rule pointing to
   the new smoke script and the per-step-protocol's retry-on-FAIL
   subsection, so the operator can find the verification path without
   re-deriving it.
5. `python tests/verify_phase_23_3_0.py` exits 0 and asserts:
   - Criterion 1 (on-disk section + commands)
   - Criterion 2 (commit on origin/main)
   - Criterion 3 (smoke script exists + executable bit + correct content)
   - Criterion 4 (CLAUDE.md cross-reference present)
6. `bash -n scripts/qa/verify_qa_roster_live.sh` exits 0 (script
   syntactically valid).

## Plan steps

1. **Smoke script** `scripts/qa/verify_qa_roster_live.sh`:
   - Print on-disk header + surrounding context.
   - Print git status of phase-23.2.24 commit (local vs origin/main).
   - Print the literal self-disclosure prompt the operator should
     give a fresh Q/A in their next session.
2. **CLAUDE.md addition** — one short cross-reference paragraph
   under "Agent definition changes require session restart" pointing
   to the smoke script + retry-on-FAIL subsection.
3. **Verifier** `tests/verify_phase_23_3_0.py` — 4 deterministic
   checks per criteria 1-4 + a `bash -n` syntax check on the script.
4. **Append to harness_log** AFTER Q/A PASS.

## Out of scope

- Forcing a hot-reload: Anthropic explicitly declined this in
  Issue #5865. Don't reinvent.
- Migrating qa.md to interface-managed form: separate refactor.
- Re-running phase-23.2.24's Q/A in a fresh session: that's the
  OPERATOR's job after restart, not this phase's.

## Backwards compatibility

- Pure additive: new script + CLAUDE.md paragraph + new verifier.

## References

- Researcher: `handoff/current/phase-23.3.0-external-research.md`,
  `handoff/current/phase-23.3.0-internal-codebase-audit.md`
- Anthropic Claude Code docs: https://code.claude.com/docs/en/sub-agents
- Issue #5865: https://github.com/anthropics/claude-code/issues/5865
- `.claude/agents/qa.md` (the file with section 1b)
- `CLAUDE.md` (existing snapshot rule)
- `docs/runbooks/per-step-protocol.md` (retry-on-FAIL loop, phase-23.2.24)
