---
phase: 23.3.0
type: internal-codebase-audit
date: 2026-05-05
researcher: researcher-agent
---

# Phase-23.3.0 Internal Codebase Audit

## Scope

Verify the new "1b. Frontend lint + typecheck" section in `.claude/agents/qa.md`, locate every piece of
project documentation that describes snapshot/loading semantics for agent definitions, and read the
phase-23.2.24 evaluator critique to determine whether the Q/A subagent ran ESLint because its OWN
system-prompt directed it or because Main's prompt explicitly asked.

---

## Files Inspected

| File | Lines inspected | Role | Status |
|------|----------------|------|--------|
| `.claude/agents/qa.md` | 1-201 (full) | Q/A agent definition — includes the new 1b section | Current |
| `CLAUDE.md` | 1-80 (critical rules) | Project conventions — source of "snapshotted at session start" rule | Current |
| `handoff/current/evaluator_critique.md` | 1-208 (full) | Phase-23.2.24 Q/A verdict | Current |
| `docs/runbooks/meta_evolution_rollback.md` | 1-80 | Contains the only other in-repo anchor for snapshot semantics | Current |
| `.claude/context/mas-architecture.md` | 1-31 (full) | MAS topology record | Current |
| `.claude/context/sessions/` | Referenced only | Historical session notes | Not read in full |
| `.claude/rules/research-gate.md` | Consulted via system context | Research gate protocol | Current |

---

## Finding 1: Section 1b is present and correctly located

File: `.claude/agents/qa.md`, lines 56-81

```
### 1b. Frontend lint + typecheck (REQUIRED if diff touches `frontend/**`)

phase-23.2.24: a runtime React Rules-of-Hooks violation shipped in
phase-23.2.23 (`frontend/src/app/cron/page.tsx::JobsTab` called
`useMemo` after early returns) because the prior Q/A deterministic
checks did not include ESLint. `tsc --noEmit` does NOT catch hook-order
violations -- hook-call ordering is a runtime execution-order
constraint with no model in the type system. ESLint's
`react-hooks/rules-of-hooks` rule (severity `"error"` in
`frontend/eslint.config.mjs:34`) performs AST-level control-flow
analysis and IS the canonical guard.

For ANY phase whose diff touches `frontend/**` or `.claude/agents/qa.md`,
Q/A MUST run BOTH of these and capture verbatim exit codes:

    cd /Users/ford/.openclaw/workspace/pyfinagent/frontend && npx eslint .
    cd /Users/ford/.openclaw/workspace/pyfinagent/frontend && npx tsc --noEmit

Either non-zero exit = FAIL. Note: `eslint .` defaults to errors-only
exit-1 semantics; warnings do NOT fail the gate. The hook-order rule
is set to `"error"` severity in the project config so the canonical
class of bug surfaces as an error.

Total runtime ~30-40s, well within the 55s Q/A budget.
```

Section header is at line 56. Both literal commands appear at lines 72-73. The phrase "tsc --noEmit does NOT catch hook-order violations" at lines 60-62. Gate condition "frontend/**" at lines 56 and 68. CONFIRMED PRESENT.

---

## Finding 2: Snapshot semantics documented in CLAUDE.md (line 46)

Exact quote from `CLAUDE.md` line 46:

> "Agent definition changes require session restart. `.claude/agents/*.md` files are snapshotted by the
> Agent-tool loader at session start. Adding/merging/renaming agents mid-session won't make them
> dispatchable until you `/clear` or restart Claude Code. When you edit agent files, note in the handoff
> that the next session cycle must verify the new roster is live."

And line 47:

> "Separation of duties on agent edits. The same Claude Code session should not both author an agent
> `.md` change AND self-evaluate work that depends on it. For substantive edits to `.claude/agents/`,
> leave a note in `handoff/harness_log.md` requesting Peder review before the next step depends on the
> change."

---

## Finding 3: meta_evolution_rollback.md independently confirms the same rule

File: `docs/runbooks/meta_evolution_rollback.md`, lines 52-55:

> "If the reverted commit touched `.claude/agents/researcher.md`, the
> **active Claude Code session must be restarted** (`/clear` or relaunch)
> because agent definitions are snapshotted at session start (CLAUDE.md
> "Agent definition changes require session restart")."

This confirms the snapshot rule is a well-established pattern in the project, not a one-off comment.

---

## Finding 4: How did the phase-23.2.24 Q/A subagent run ESLint?

Source: `handoff/current/evaluator_critique.md`, section "8. Self-test (meta)"

The Q/A critique at lines 161-168 states:

> "Per CLAUDE.md 'agent definition changes require session restart' the on-disk qa.md NOW has section
> '1b' (confirmed by Read), and I (this Q/A session) was briefed on the new rubric by the caller and
> ran both new checks (`npx eslint .` and `npx tsc --noEmit`) per explicit instructions. Future
> `Agent`-tool spawns of `qa` will pick up the section automatically once the next Claude Code session
> starts. This is documented behavior, not a defect."

Key phrase: "briefed on the new rubric by the caller and ran both new checks per EXPLICIT INSTRUCTIONS."

Cross-reference with the deterministic checks section (lines 29-82): `checks_run` list includes
`frontend_eslint_live` and `frontend_tsc_live`. The verifier `tests/verify_phase_23_2_24.py` also
ran ESLint via `check_eslint_exits_zero` (per critique line 96).

CONCLUSION: The Q/A subagent ran ESLint because (a) Main's explicit prompt instructed it to apply the
new rubric and (b) the verify_phase_23_2_24.py test script also ran it deterministically. The Q/A
subagent then READ qa.md from disk to confirm the section was present (file:line 95: "qa.md lines
56-81"), but that Read reflected the on-disk state, not the snapshotted system prompt at session start.

The Q/A was NOT operating under the new section as part of its own snapshotted system prompt. It was
following explicit caller instructions and running an external test script. The section in its system
prompt (as snapshotted at spawn time) had NOT been updated because the edit happened in the same session.

---

## Finding 5: The ambiguity identified by Main is confirmed

The phase-23.2.24 evaluator critique line 80 says:
```
$ grep -c "Frontend lint" .claude/agents/qa.md         => 1
```

This proves Q/A READ the file (getting a disk-time count of 1), but the Q/A's OWN PROMPT did not
include section 1b. The Q/A was correctly self-aware about this: it flagged it in section 8 and
recommended the next session verify the new roster is live.

---

## Summary of internal findings

1. `qa.md` section 1b is present at lines 56-81. (file:line `.claude/agents/qa.md:56`)
2. CLAUDE.md line 46 is the canonical, machine-verifiable statement of snapshot semantics.
3. meta_evolution_rollback.md lines 52-55 is a secondary confirmation.
4. Phase-23.2.24 Q/A ran ESLint via explicit caller instructions + external test script, NOT because
   its snapshotted system prompt included section 1b.
5. The Q/A itself disclosed this in its self-test section and recommended next-session verification.

---

## File:line anchors

| Claim | File | Lines |
|-------|------|-------|
| Section 1b present | `.claude/agents/qa.md` | 56-81 |
| Snapshot rule | `CLAUDE.md` | 46 |
| Separation of duties | `CLAUDE.md` | 47 |
| Rollback runbook snapshot confirmation | `docs/runbooks/meta_evolution_rollback.md` | 52-55 |
| Q/A self-disclosure (ran ESLint via explicit instructions) | `handoff/current/evaluator_critique.md` | 161-168 |
| Q/A grep of qa.md from disk | `handoff/current/evaluator_critique.md` | 80-82 |
| Q/A verdict PASS | `handoff/current/evaluator_critique.md` | 177-207 |
