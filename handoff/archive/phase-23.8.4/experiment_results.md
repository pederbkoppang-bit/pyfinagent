---
step: phase-23.8.4
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_23_8_4.py'
title: Auto-commit hook auto-fire diagnostic — add invocation debug log + preserve `if` predicate (observability-first)
---

# Experiment Results — phase-23.8.4

## What was built

A single observability change to the auto-commit-and-push hook + an
11-claim immutable verifier. **No wiring change**: the `if` predicates
in `.claude/settings.json` are explicitly preserved per the
research-supported observability-first plan.

### File-by-file

| File | Mode | Lines changed | Why |
|---|---|---|---|
| `.claude/hooks/auto-commit-and-push.sh` | edit | +6 / -0 | Adds one `log "INVOKED auto-commit-and-push pid=$$"` line after the `log()` helper definition (line 33-35) and BEFORE the masterplan-exists guard (line ~42), with a 5-line comment block explaining the observability rationale and pointing at the existing `newly_done` filter |
| `tests/verify_phase_23_8_4.py` | create | +280 LOC | 11-claim stdlib-only verifier (json, re, shutil, subprocess, sys, tempfile, pathlib); 3 behavioral tests (claims 7, 8, 10) including one mutation-resistance test |
| `handoff/current/research_brief.md` | overwrite | (new) | researcher subagent output, 2026-05-12, gate_passed=true |
| `handoff/current/contract.md` | overwrite | (new) | contract per protocol |
| `handoff/current/experiment_results.md` | (this file) | (new) | GENERATE artifact per protocol |
| `.claude/masterplan.json` | edit | +21 / -0 | Adds step 23.8.4 (pending state, 11 success criteria, audit_basis = cycle 40 observation) |

Files **not** modified (explicitly):

- `.claude/settings.json` — `if` predicates preserved on both Write and Edit matchers
- All sibling hooks (`masterplan-memory-sync.sh`, `archive-handoff.sh`, `commit-reminder.sh`, `post-commit-changelog.sh`)
- `.claude/hooks/lib/live_check_gate.py` (R-1 logic; orthogonal)
- `CLAUDE.md` (no operator-visible behavior change to document — observability is internal)

## Verbatim verifier output

```
=== phase-23.8.4 verifier ===
  [PASS] 1. settings_json_valid
  [PASS] 2. edit_matcher_if_predicate_preserved
  [PASS] 3. write_matcher_if_predicate_preserved
  [PASS] 4. auto_commit_hook_has_invoked_log_at_top
  [PASS] 5. auto_commit_hook_bash_syntax_valid
  [PASS] 6. auto_commit_hook_still_filters_by_newly_done
  [PASS] 7. invocation_writes_invoked_line_to_auto_push_log
  [PASS] 8. invoked_line_includes_timestamp_marker_and_hook_name
  [PASS] 9. no_regressions_other_hooks_bash_syntax_valid
  [PASS] 10. mutation_resistance_removing_invoked_line_breaks_behavioral_claim
  [FAIL] 11. harness_log_has_cycle_41_entry
         -> harness_log.md must contain `## Cycle 41 -- ... -- phase=23.8.4` entry (will FAIL at Q/A time per log-last protocol; PASSes after LOG phase)
FAIL (10/11) EXIT=1
```

**10/11 PASS** at Q/A spawn time. This is the documented log-last
intermediate state per `feedback_log_last.md` (memory) — claim 11
PASSes after the LOG phase appends the Cycle 41 entry. The verifier
must re-run AFTER the harness_log append to confirm 11/11.

## Artifact shape

### G-1: hook edit (`.claude/hooks/auto-commit-and-push.sh`)

Diff (effectively):

```diff
 mkdir -p "$LOG_DIR"
 ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
 log() { echo "[$(ts)] $*" >> "$LOG_FILE"; }
 
+# phase-23.8.4 observability: every invocation produces a log entry so the
+# next cycle that mis-fires can distinguish "hook never dispatched" (no
+# INVOKED line) from "hook dispatched but newly_done was empty" (INVOKED
+# line followed by silent exit at line ~114). Cheap; the existing
+# newly_done detection still gates the actual commit/push work.
+log "INVOKED auto-commit-and-push pid=$$"
+
 if [ ! -f "$MASTERPLAN" ]; then
     exit 0
 fi
```

Log line format produced: `[2026-05-11T22:21:25Z] INVOKED auto-commit-and-push pid=76752`

The `ts()` helper is the existing UTC ISO-8601 formatter. `$$` is the
bash PID of the hook invocation. The `auto-commit-and-push` literal
identifies this hook so a future shared log surface can distinguish
hooks.

### G-3: verifier shape (`tests/verify_phase_23_8_4.py`)

11 claims. Per-claim summary:

| # | Claim | Kind | Catches |
|---|-------|------|---------|
| 1 | `settings_json_valid` | source/JSON | settings.json structural breakage |
| 2 | `edit_matcher_if_predicate_preserved` | source/JSON | Accidental drop of `if "Edit(.claude/masterplan.json)"` |
| 3 | `write_matcher_if_predicate_preserved` | source/JSON | Accidental drop of `if "Write(.claude/masterplan.json)"` |
| 4 | `auto_commit_hook_has_invoked_log_at_top` | source/grep | INVOKED line missing, or below the masterplan-exists guard |
| 5 | `auto_commit_hook_bash_syntax_valid` | subprocess | bash -n catches syntax errors |
| 6 | `auto_commit_hook_still_filters_by_newly_done` | source/grep | Future edit accidentally removes the newly_done filter |
| 7 | `invocation_writes_invoked_line_to_auto_push_log` | **BEHAVIORAL** | Hook actually produces the log line when invoked |
| 8 | `invoked_line_includes_timestamp_marker_and_hook_name` | **BEHAVIORAL** | Tail line matches regex `[ISO-8601] INVOKED auto-commit-and-push pid=N` |
| 9 | `no_regressions_other_hooks_bash_syntax_valid` | subprocess | Sibling hook syntax breakage |
| 10 | `mutation_resistance_removing_invoked_line_breaks_behavioral_claim` | **BEHAVIORAL + MUTATION** | Proves claim 7 is anchored to the literal log line — not a no-op grep |
| 11 | `harness_log_has_cycle_41_entry` | source/grep | Log-last protocol fence |

**Mutation-resistance test detail (claim 10)**:

1. Read the hook script as `hook_text`.
2. Use `re.sub` to remove the literal `log "INVOKED auto-commit-and-push pid=$$"\n` line.
3. Write the mutated text to a tmpdir.
4. Capture `_count_invoked_lines(log_path)` before.
5. Invoke the mutated copy via `subprocess.run`.
6. Capture count after.
7. Assert delta == 0 (mutation correctly removed the line, so mutated copy did not write).

This catches a future regression where someone might write
`if "INVOKED" in hook_text` instead of the actual behavioral test —
the mutation step would still strip the literal log line, and the
behavioral test would still measure delta, so the assertion would
hold even under naive grep mutations.

**Behavioral-test safety guard** (`_newly_done_is_empty()`): claims 7,
8, 10 invoke the actual hook script as a subprocess. The hook is
fail-open + idempotent — when `newly_done` is empty, it logs INVOKED
then exits silently before any git work. The verifier ASSERTS
newly_done is empty before invoking, refusing to run the behavioral
claims otherwise. Prevents the verifier from accidentally triggering
a real auto-commit during verification.

## Surprise observation (scope-honest)

During verifier development (BEFORE the harness_log entry was written
for this cycle), the auto-push.log accumulated INVOKED entries from
**Edit** calls outside `.claude/masterplan.json` AND from **Write**
calls outside `.claude/masterplan.json`. Specifically:

- `[2026-05-11T22:17:48Z] INVOKED auto-commit-and-push pid=76339` —
  fired during this session's Edit to
  `.claude/hooks/auto-commit-and-push.sh` (i.e., when the INVOKED line
  was being added). The `if "Edit(.claude/masterplan.json)"` predicate
  should have BLOCKED this fire.
- `[2026-05-11T22:20:07Z]` — fired during this session's Write to
  `tests/verify_phase_23_8_4.py`. The `if "Write(.claude/masterplan.json)"`
  predicate should have BLOCKED this fire.

**Significance**: this is observable evidence that the `if` predicate
is **too permissive** — fires on edits/writes to paths that do NOT
match `Edit(.claude/masterplan.json)` / `Write(.claude/masterplan.json)`.
The cycle 38/39/40 reports were "hook didn't fire on Edits to
masterplan.json"; we now have a second symptom of the same
predicate's unreliability — "hook fires on Edits/Writes elsewhere."

The asymmetry is interesting:
- **Under-fires** (cycles 38/39/40): missed legitimate Edit to
  masterplan.json
- **Over-fires** (this cycle): fired on Edit/Write to hook script and
  to verifier file

The observability instrument (INVOKED line) made this asymmetry
visible — exactly the diagnostic outcome this cycle was designed to
produce. The data now exists to make the next decision (likely:
phase-23.8.5 drops the `if` predicate with empirical backing,
accepting wasteful sibling-hook churn as the cost of reliability).

This is **scope-honest disclosure**: the surprise observation is NOT
a failure of this cycle's plan. The cycle was specifically designed
to instrument first and decide later. The data already supports a
follow-up decision — but the follow-up is explicitly out of scope
per `contract.md::What this is NOT`.

## What's next

1. Spawn a fresh Q/A subagent (no second-opinion shopping per
   `feedback_harness_rigor.md`).
2. Q/A reads `contract.md`, `experiment_results.md`,
   `research_brief.md`, and runs the verifier.
3. On Q/A PASS: append `handoff/harness_log.md` Cycle 41 entry
   (LOG-LAST per `feedback_log_last.md`).
4. Re-run verifier (must return 11/11 after LOG append).
5. Flip masterplan 23.8.4 → done. This triggers the auto-commit-and-push
   hook itself — a meta-test of the observability instrument we just
   added. The auto-push.log should show an INVOKED entry followed by
   the commit / push activity.

## Audit-remediation progress after Cycle 41 (projected on PASS)

- **Shipped**: R-1 (live_check gate, cycle 38), R-2 (TaskCompleted
  delete, cycle 39), R-3 (rename Layer-2 labels, cycle 37), R-4
  (META_PLAN runtime config, cycle 37), R-6 (header correction, cycle
  40), R-7 (28→5→3 mapping paragraph, cycle 37). **Plus**: auto-commit
  hook observability (cycle 41 — out-of-original-audit, observed during
  cycles 38-40).
- **Properly deferred**: R-5 (qa.md fail-mode, needs separate session
  + Peder review per separation-of-duties); qa.md
  `existing_results_check` update for `live_check_<step_id>.md` (also
  needs separate session).
- **New observation (cycle 41)**: `if` predicate over-fires AND
  under-fires. Wiring change deferred to phase-23.8.5+ with empirical
  data from at least one more observation cycle.

## R-5 / R-6 / qa.md-follow-on still deferred (unchanged from prior cycles)

- **R-5** (qa.md fail-mode change from fail-OPEN to fail-CLOSED on
  `stop_hook_active`) — needs separate session + Peder review per
  separation-of-duties.
- **qa.md follow-on** to mention `live_check_<step_id>.md` in
  existing_results_check (lines 83-93) — deferred from cycle 38 per
  separation-of-duties.

## Honest disclosures

1. **Pre-research hypothesis was rejected.** The original plan (drop
   the `if` predicate) was overridden by the research gate. The
   masterplan criteria were updated pre-contract to match the
   research-supported observability-first plan. The criteria
   immutability rule (per CLAUDE.md "Never edit verification
   criteria in masterplan.json") applies from contract-time forward;
   the pre-contract criteria revision is documented in the contract's
   `## Hypothesis (revised after research gate)` section.
2. **No empirical confirmation of the fix yet.** This cycle adds the
   instrument. Confirmation of "hook now reliably fires on masterplan
   edits" requires observing at least one more real-world masterplan
   step flip with the INVOKED log enabled.
3. **The `if` predicate may continue to misbehave.** This cycle does
   not change wiring. If the next masterplan step still requires a
   manual trigger, the INVOKED log will provide unambiguous evidence
   (no entry = `if` blocked the dispatch; entry + silent exit =
   newly_done detection failed). Either way, the data informs the
   phase-23.8.5+ wiring decision.
4. **PIDs in INVOKED line may collide across long sessions.** Bash
   `$$` is the shell PID; if the OS recycles PIDs after process exit,
   two unrelated invocations could theoretically share a PID. In
   practice this is rare (PIDs are typically monotonic in a session)
   and the timestamp disambiguates. Not a correctness issue for the
   diagnostic.
5. **Sibling hooks run BEFORE auto-commit-and-push.sh in the chain.**
   If a sibling hook hangs or takes >timeout, auto-commit-and-push
   may not be reached. The INVOKED log catches this — if a sibling
   hook hangs, the INVOKED line for auto-commit-and-push will be
   absent even though the chain "fired." This is useful diagnostic
   information for future debugging.
