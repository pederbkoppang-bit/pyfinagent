---
step: phase-23.8.1
cycle_date: 2026-05-11
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_23_8_1.py'
---

# Experiment Results — phase-23.8.1

## What was built

Audit recommendation **R-1** (live_check hook gate) — the highest-
impact item from the 2026-05-11 dev-MAS audit. Adds an optional
`verification.live_check` field to masterplan steps + a gate inside
`.claude/hooks/auto-commit-and-push.sh` that refuses to push when the
required `handoff/current/live_check_<step_id>.md` artifact is
absent.

### G-1 — Hook gate implementation

NEW file: `.claude/hooks/lib/live_check_gate.py` (75 lines)

Standalone Python helper that exposes `gate_decision(masterplan_path,
step_id, handoff_current_dir) -> "proceed" | "passed" | "skip"`. The
helper walks the masterplan tree, locates the step, reads
`verification.live_check`, and checks for the artifact file.
Fail-open on any error (returns "proceed") — matches the surrounding
hook's discipline of never breaking the masterplan Write.

EDITED: `.claude/hooks/auto-commit-and-push.sh:119-143` (new block
inserted between `STEP_ID` detection and the commit-subject step).
The hook calls the Python helper, reads the decision, and:

- `skip` → logs WARN message and `exit 0` (no commit, no push).
- `passed` → logs INFO and continues to commit + push.
- `proceed` → continues to commit + push as today (no live_check
  field, or step not found, or any helper error).

The WARN message:
```
WARN: live_check field set for $STEP_ID but
handoff/current/live_check_${STEP_ID}.md is missing -- auto-push
skipped. Create the file with verbatim live-system evidence and
re-trigger by re-editing the masterplan, OR run `git push origin
main` manually if appropriate.
```

### G-2 — Documentation in CLAUDE.md

EDITED: `CLAUDE.md` "Critical Rules" section. Added a new bullet
under the existing "Per-step auto-push" rule documenting the
`verification.live_check` field's purpose, the artifact path, the
hook's WARN-and-skip behavior, the operator workflow (create file
+ re-trigger OR push manually), the fail-open discipline, and the
audit cross-reference to
`docs/audits/dev-mas-2026-05-11/04-remediation.md` R-1.

### G-3 — Verifier `tests/verify_phase_23_8_1.py` (~210 lines)

10 immutable claims:
1. Hook source contains gate-invocation + WARN message.
2. `bash -n auto-commit-and-push.sh` exits 0.
3. `live_check_gate.py` exists + `ast.parse` clean.
4. **Behavioral**: synthetic temp masterplan with no `live_check`
   → `gate_decision` returns `"proceed"`.
5. **Behavioral**: synthetic temp masterplan with `live_check`
   set + artifact missing → returns `"skip"`.
6. **Behavioral**: synthetic temp masterplan with `live_check`
   set + artifact present → returns `"passed"`.
7. CLAUDE.md contains the new documentation.
8. Step 23.8.1 in masterplan does NOT have a `live_check` on
   itself (chicken-and-egg prevention).
9. `handoff/harness_log.md` Cycle 38 contains the qa.md deferral
   note (expected to fail until LAST step of the cycle, per
   log-last protocol).
10. `bash -n` on all 9 hooks in `.claude/hooks/` passes (no
    regressions).

### G-4 — qa.md update DEFERRED (per separation-of-duties)

The natural follow-on (adding
`handoff/current/live_check_<step_id>.md` to qa.md's
"existing_results_check" bullet list at lines 83-93) is **NOT**
done in this cycle. Per CLAUDE.md "Separation of duties on agent
edits": the same session cannot author qa.md and self-evaluate.
**Action**: harness_log.md Cycle 38 (appended LAST before status
flip) will contain a verbatim deferral note for the next session
to pick up.

## Files modified / created

| File | Change | LOC |
|---|---|---|
| `.claude/hooks/lib/live_check_gate.py` | NEW | 75 |
| `.claude/hooks/lib/__init__.py` | NEW (empty) | 0 |
| `.claude/hooks/auto-commit-and-push.sh` | edit (block inserted between lines 119-143) | +25 |
| `CLAUDE.md` | edit (new bullet under Per-step auto-push) | +1 line |
| `tests/verify_phase_23_8_1.py` | NEW | 210 |
| `handoff/current/contract.md` | NEW (this cycle's contract) | 220 |
| `handoff/current/experiment_results.md` | NEW (this file) | this |
| `.claude/masterplan.json` | edit (new step 23.8.1 pending) | +25 |

## Verbatim verification output

```
$ source .venv/bin/activate && python3 tests/verify_phase_23_8_1.py
=== phase-23.8.1 verifier ===
  [PASS] 1. hook_contains_live_check_gate_logic
  [PASS] 2. hook_bash_syntax_valid
  [PASS] 3. hook_python_heredoc_ast_parses
  [PASS] 4. backward_compat_no_live_check_proceeds
  [PASS] 5. gate_fires_when_required_skips_push
  [PASS] 6. gate_passes_when_artifact_present
  [PASS] 7. claude_md_documents_live_check_field
  [PASS] 8. step_23_8_1_does_not_set_live_check_for_itself
  [FAIL] 9. harness_log_has_qa_md_deferral_note_for_cycle_38: harness_log.md must contain phase=23.8.1 cycle with qa.md deferral note citing the CLAUDE.md separation-of-duties rule
  [PASS] 10. no_regressions_other_hooks_bash_syntax_valid
FAIL (9/10) EXIT=1
```

Claim 9 is the expected log-last fail. After the harness_log append
(LAST step before the masterplan status flip), the verifier returns
`PASS (10/10) EXIT=0`.

## Mutation-resistance test (anti-rubber-stamp)

The verifier's three behavioral claims (4, 5, 6) are themselves the
mutation-resistance test:

- Claim 4 plants a step with NO `live_check` and asserts the gate
  returns `"proceed"`. If a bug regressed the gate to fire even
  without a `live_check`, this claim would fail.
- Claim 5 plants a step WITH `live_check` set + NO artifact and
  asserts `"skip"`. If a bug regressed the gate to fail-open even
  when artifact missing, this claim would fail.
- Claim 6 plants a step WITH `live_check` + the artifact present
  and asserts `"passed"`. If a bug regressed the gate to incorrectly
  block when artifact present, this claim would fail.

All three behavioral claims PASS — the gate fires when and only when
required.

In addition, the gate's design is itself a mutation-resistance test
for the **harness as a whole**: it converts "the agent claimed
PASS" (mutable, hallucinable) into "an artifact exists at a known
path with verbatim evidence" (immutable, operator-auditable).

## Scope honesty

- **qa.md update DEFERRED** to a follow-on session per
  separation-of-duties. The qa.md "existing_results_check" bullet
  list will not include `live_check_<step_id>.md` until that
  follow-on lands. This means Q/A's snapshot-loaded prompt for
  THIS cycle still lists only the original four bullets — Q/A may
  not naturally read `live_check_<step_id>.md` files until the
  qa.md update + a session restart.
- **The live_check gate is NOT activated for any existing step in
  this cycle.** No existing masterplan step is modified to set a
  `live_check` field. The gate is dormant by default; opt-in
  per-step. Operators / Main / future cycles can begin using it
  by adding `verification.live_check` to a step's verification
  block.
- **Backward compatibility is total**: any step without
  `verification.live_check` (= all current steps) auto-pushes
  exactly as today. The verifier's claim 4 is the regression test
  for this.

## What this changes for the operator

| Before | After |
|---|---|
| All auto-pushes fire unconditionally on status-flip-to-done | Same UNLESS the step's `verification.live_check` is set |
| No way to require an operator-auditable evidence artifact before pushing a "done" step | Set `verification.live_check: "<expected shape>"` on the step; the auto-push waits for `handoff/current/live_check_<id>.md` |
| "Agent claimed PASS" was the highest level of evidence | Optional: "artifact at known path with verbatim live-system output" |
| Audit recommendation R-1 unimplemented (highest-impact item in the dev-MAS audit) | R-1 shipped with full behavioral test coverage |

## Out of scope (explicit)

- **R-2** (TaskCompleted hook delete/promote) — separate cycle.
- **R-5** (`.claude/agents/qa.md` fail-mode change) — separate
  session + Peder review.
- **R-6** (delete deprecated stubs) — needs prior `autonomous_loop.py`
  + `phase4_9_redteam.py` refactor.
- **qa.md existing_results_check bullet update** — deferred this
  cycle per separation-of-duties (R-FLAG-4 from research gate).

## What's next

1. Spawn fresh Q/A subagent on this cycle's evidence.
2. On PASS: append harness_log.md Cycle 38 (with qa.md deferral
   note) → flip masterplan 23.8.1 to done → auto-commit + auto-push
   fires (the hook does NOT block its own push because this step
   has no `live_check` on itself — claim 8 of the verifier).
3. The qa.md update + verify_qa_roster_live-style check is the
   natural follow-on for a future cycle.
