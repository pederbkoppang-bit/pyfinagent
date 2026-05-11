---
step: phase-23.8.1
title: live_check hook gate — audit recommendation R-1 (qa.md update deferred to separate session)
cycle_date: 2026-05-11
harness_required: true
verification: 'source .venv/bin/activate && python3 tests/verify_phase_23_8_1.py'
research_brief: (researcher subagent, 2026-05-11; gate_passed=true; 7 sources read in full, 17 URLs, recency scan performed)
audit_basis: docs/audits/dev-mas-2026-05-11/04-remediation.md (R-1)
---

# Contract — phase-23.8.1

**Step**: phase-23.8.1 — live_check hook gate (R-1).

**Date**: 2026-05-11.

**Status target**: pending → done.

**Hypothesis**:
Adding an optional `verification.live_check` field to masterplan
steps + a corresponding gate inside
`.claude/hooks/auto-commit-and-push.sh` will:
1. Skip the auto-push for steps that require live-system evidence
   when `handoff/current/live_check_<step_id>.md` is absent.
2. Be **fully backward-compatible** — existing steps without
   `live_check` continue to auto-push as today.
3. Match the existing hook's failure discipline (log WARN + exit 0;
   the hook never breaks the masterplan Write itself).
4. Provide an audit-trail artifact (the live_check file) that Q/A
   can read during its existing_results_check step.

This step itself does **NOT** set `live_check` on its own
verification, so the hook must auto-push this commit normally — a
built-in regression test of backward compatibility.

## Research-gate summary

Researcher subagent ran 2026-05-11 (tier: moderate). JSON envelope:
`{"external_sources_read_in_full": 7, "snippet_only_sources": 10,
"urls_collected": 17, "recency_scan_performed": true,
"internal_files_inspected": 4, "gate_passed": true}`.

Key external citations (≥5 sources read in full via WebFetch):

1. HARNESS-DOC
   (`https://www.anthropic.com/engineering/harness-design-long-running-apps`)
   — verbatim: "**Communication was handled via files: one agent
   would write a file, another agent would read it and respond
   either within that file or with a new file that the previous
   agent would read in turn.**" Grounds the file-based artifact
   approach.
2. CLAUDE CODE HOOKS DOC
   (`https://code.claude.com/docs/en/hooks`) — clarifies hook
   exit-code semantics: exit 2 = blocking error fed as stderr;
   exit 1 = NON-blocking; exit 0 + JSON `{"decision":"block",...}`
   = graceful policy block. For the auto-commit-and-push.sh hook
   (which already exits 0 with WARN on failure), the correct
   pattern is **log WARN + exit 0 to skip push** — consistent with
   the existing git-push-failure discipline.
3. MULTI-DOC
   (`https://www.anthropic.com/engineering/multi-agent-research-system`)
   — "**The LeadResearcher synthesizes these results and decides
   whether more research is needed**" — relevant because the
   live_check file becomes an artifact Q/A reads in its
   evaluation.
4. Praetorian "Deterministic AI Orchestration" (2025) —
   "**Hard Block. The system refuses to spawn new agents until
   [precondition] runs.**" Grounds R-1's blocking-gate pattern.
5. Atlan AI Observability 2026 — "**Every agent action is logged
   with the context, policies, and data assets that governed it,
   producing audit trails for regulatory compliance.**" The
   live_check file IS the audit-trail entry for the step's
   operator verification.
6. Vadim's verification-gate post (2026) — "**An autonomous
   improvement system without verification is just autonomous
   damage.**" Grounds the gate's existence at the push layer.
7. Arize AI Observability 2026 — confirms that trace-based +
   blocking-gate are complementary, not competing.

Recency scan: no 2024-2026 source argues AGAINST file-based
pre-push gates on velocity grounds.

### Red-flag findings from research gate (action taken)

- **R-FLAG-1**: No prior live_check gating exists in
  `auto-commit-and-push.sh` (192 lines fully reviewed by
  researcher). R-1 is genuinely new — not a partial
  implementation.
- **R-FLAG-2**: masterplan.json `verification` objects today use
  ONLY `command` + `success_criteria` keys (Python grep
  confirmed). Adding `live_check` is additive-only.
- **R-FLAG-3**: No JSON-schema validator enforces the masterplan
  schema at runtime; `$schema: "masterplan-v1"` is decorative.
  Safe to add new optional field.
- **R-FLAG-4**: Editing `.claude/agents/qa.md` to mention
  `live_check_<step_id>.md` in its existing_results_check
  bullets would trigger the same separation-of-duties flag
  documented for R-5. **Action: the qa.md edit is DEFERRED to a
  separate session** (next cycle).
- **R-FLAG-5**: PostToolUse `decision: "block"` stops the NEXT
  model call, not the current tool. The hook fires AFTER the
  masterplan.json Write — correct for R-1's scope. We are not
  preventing the status flip; we are preventing the auto-push
  of that flip when live evidence is required and missing.

## Plan steps

### G-1 — Modify `.claude/hooks/auto-commit-and-push.sh` (add live_check gate)

- Insertion point: between line 116 (where `STEP_ID` is captured)
  and line 145 (where `git add -A` runs).
- Extend the existing Python heredoc (or add a second small
  Python block) to read `verification.live_check` for the
  newly-detected `$STEP_ID` from the masterplan.
- If `live_check` is non-empty AND
  `handoff/current/live_check_${STEP_ID}.md` does NOT exist:
  log `WARN: live_check field set for ${STEP_ID} but
  handoff/current/live_check_${STEP_ID}.md is missing — auto-push
  skipped. Create the file with verbatim live-system evidence
  and re-trigger by re-editing the masterplan, OR run \`git push
  origin main\` manually if appropriate.` to
  `$LOG_FILE`, then `exit 0`.
- If `live_check` is empty (or field absent): proceed as today.
- If `live_check` is set AND the file exists: log
  `INFO: live_check artifact present for ${STEP_ID} — gate
  satisfied`, then proceed as today.

### G-2 — Update `CLAUDE.md` (document the new field)

Add a short subsection under the existing
"**Per-step auto-push**" bullet that documents:
- The `verification.live_check` field's purpose, expected shape,
  and `handoff/current/live_check_<step_id>.md` artifact.
- The hook's gating behavior: WARN-and-skip, not abort.
- Reference to `docs/audits/dev-mas-2026-05-11/04-remediation.md`
  R-1 as the design source.

### G-3 — Add verifier script
`tests/verify_phase_23_8_1.py`

The verifier asserts:

1. `auto-commit-and-push.sh` contains the new gate logic (grep
   for `live_check` + the WARN message string).
2. The hook's bash syntax is valid (`bash -n
   .claude/hooks/auto-commit-and-push.sh`).
3. The hook's Python heredoc still parses (extract + `python3 -c
   "ast.parse(...)"` or equivalent).
4. Backward compatibility: a synthetic test runs the hook against
   a temp masterplan with a step that has NO `live_check` —
   confirms it does NOT trip the gate.
5. Gate fires when required: a synthetic test runs the hook
   against a temp masterplan where the newly-done step has a
   `live_check` field set AND the file is absent — confirms it
   logs the WARN line and exits 0 WITHOUT pushing.
6. Gate passes when artifact present: similar to (5) but the
   `handoff/current/live_check_<step_id>.md` file IS created in
   the temp env — confirms the gate is satisfied.
7. CLAUDE.md contains the new documentation paragraph (grep for
   key phrases: `live_check` + `handoff/current/live_check_`).
8. This step's verification has NO `live_check` field
   (regression test of backward compatibility for this very
   cycle).

### G-4 — qa.md edit DEFERRED (separation-of-duties)

Per R-FLAG-4 from the research gate: editing
`.claude/agents/qa.md` to add a `live_check_<step_id>.md` bullet
to its existing_results_check list (lines 83-93) cannot be
self-evaluated in the same session. **Action**: append a deferral
note to `handoff/harness_log.md` Cycle 38 (this cycle) stating
that the qa.md update is the natural follow-on for a future
cycle, and that Q/A's existing_results_check will not include the
new file until a session-restart-aware step adds it.

## Immutable success criteria

These are immutable once the masterplan step is written. A script
checks each one.

1. `.claude/hooks/auto-commit-and-push.sh` contains the literal
   string `live_check` AND the WARN message
   substring `auto-push skipped`.
2. `bash -n .claude/hooks/auto-commit-and-push.sh` exits 0.
3. The hook's syntax check on the embedded Python is valid (the
   verifier extracts and ast.parses the heredoc bodies).
4. **Backward compat**: synthetic-temp-masterplan test (a step
   with NO `live_check`) — the hook's gate logic returns
   "proceed".
5. **Gate fires**: synthetic-temp-masterplan test (a step WITH
   `live_check` set, file ABSENT) — the gate logic returns
   "skip-push".
6. **Gate passes**: synthetic-temp-masterplan test (a step WITH
   `live_check` set, file PRESENT) — the gate logic returns
   "proceed".
7. `CLAUDE.md` contains both `verification.live_check` and
   `handoff/current/live_check_` as substrings (the new
   documentation paragraph).
8. The verification block for THIS step (`23.8.1`) in
   `.claude/masterplan.json` does NOT contain a `live_check`
   field (otherwise the hook would block this step's own push,
   creating a chicken-and-egg).
9. `handoff/harness_log.md` Cycle 38 contains a verbatim deferral
   note for the qa.md edit referencing CLAUDE.md's "Separation of
   duties on agent edits" rule.
10. **No regressions**: `bash -n` on the other hooks in
    `.claude/hooks/` still passes
    (`.claude/hooks/post-commit-changelog.sh`,
    `.claude/hooks/archive-handoff.sh`,
    `.claude/hooks/masterplan-memory-sync.sh`,
    `.claude/hooks/commit-reminder.sh`,
    `.claude/hooks/pre-tool-use-danger.sh`,
    `.claude/hooks/config-change-audit.sh`,
    `.claude/hooks/instructions-loaded-research-gate.sh`,
    `.claude/hooks/teammate-idle-check.sh`).

## Files expected to change

| File | Type | Change |
|---|---|---|
| `.claude/hooks/auto-commit-and-push.sh` | edit | live_check gate inserted between STEP_ID detection and `git add -A` |
| `CLAUDE.md` | edit | New documentation paragraph under "Per-step auto-push" critical rule |
| `tests/verify_phase_23_8_1.py` | NEW | 10-claim verifier |
| `handoff/current/contract.md` | NEW (this file) | the contract itself |
| `handoff/current/experiment_results.md` | NEW (later) | by GENERATE |
| `handoff/current/evaluator_critique.md` | NEW (later) | by Q/A |
| `handoff/harness_log.md` | append | Cycle 38 + qa.md-deferral note (LAST before flip) |
| `.claude/masterplan.json` | edit | new step 23.8.1 pending; flip to done at end |

## Rollback note

Single-commit revert. The hook change is additive (a new gate
inserted into existing flow). If revert is needed:
`git revert <commit-sha>` returns the hook to its
pre-23.8.1 state; existing steps remain unaffected. The new
`verification.live_check` field on future steps would simply be
ignored by the older hook (silent backward compat in the other
direction).

If the live_check gate ever blocks a step that shouldn't be
blocked, the operator can:
1. Verify the issue in `handoff/logs/auto-push.log`.
2. Create the expected `handoff/current/live_check_<id>.md`
   manually with the required evidence.
3. Re-edit `.claude/masterplan.json` (no-op edit, e.g. an
   `updated_at` bump) to re-trigger the hook.
4. Or push manually: `git push origin main`.

## Out of scope (explicit)

- **R-2** (TaskCompleted hook delete/promote) — separate cycle.
- **R-5** (`.claude/agents/qa.md` fail-mode change) — separate
  session + Peder review.
- **R-6** (delete deprecated stubs) — needs prior refactor.
- **qa.md existing_results_check update** to mention
  `live_check_<step_id>.md` — DEFERRED to separate session per
  separation-of-duties (R-FLAG-4).

## References

- `docs/audits/dev-mas-2026-05-11/04-remediation.md` R-1 — the
  proposal.
- `docs/audits/dev-mas-2026-05-11/03-symptoms.md` — the
  VERIFICATION_DEFECT systemic pattern R-1 directly attacks.
- `.claude/hooks/auto-commit-and-push.sh` (192 lines) — the hook
  being modified.
- Researcher subagent JSON envelope (above): `gate_passed: true`.
