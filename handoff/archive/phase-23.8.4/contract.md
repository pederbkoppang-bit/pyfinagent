---
step: phase-23.8.4
title: Auto-commit hook auto-fire diagnostic — add invocation debug log + preserve `if` predicate (observability-first)
cycle_date: 2026-05-12
harness_required: true
verification: 'source .venv/bin/activate && python3 tests/verify_phase_23_8_4.py'
research_brief: (researcher subagent, 2026-05-12; gate_passed=true; 6 sources read in full, 15 URLs, recency scan)
audit_basis: 'observation logged in handoff/harness_log.md cycle 40 (phase-23.8.3 experiment_results.md:148 + contract.md:208) — auto-commit-and-push hook did not auto-fire on Edit calls in cycles 38/39/40; operator manually triggered.'
---

# Contract — phase-23.8.4

**Step**: phase-23.8.4 — diagnostic for the
`.claude/hooks/auto-commit-and-push.sh` hook not auto-firing on `Edit`
calls to `.claude/masterplan.json`.

**Date**: 2026-05-12.

**Status target**: pending → done.

## Hypothesis (revised after research gate)

**Initial hypothesis** (pre-research): the
`"if": "Edit(.claude/masterplan.json)"` predicate in
`.claude/settings.json` PostToolUse Edit matcher is undocumented for
file-path matching and silently fails. **Fix proposed**: drop the
`if` predicate; rely on the script's internal `newly_done` filter.

**Research-gate result (cycle 41, 2026-05-12)**: rejected the initial
plan as overly aggressive. Three findings:

1. The `if` field IS documented at `code.claude.com/docs/en/hooks`
   with permission-rule / glob syntax (`Edit(*.ts)` example), but the
   documentation does NOT show directory-prefixed exact-path patterns
   like `Edit(.claude/masterplan.json)`. The runtime semantics for
   `.claude/`-prefixed paths are under-specified.
2. Dropping the `if` predicate would make all four hooks in the
   PostToolUse Edit block (`masterplan-memory-sync`, `archive-handoff`,
   `commit-reminder`, `auto-commit-and-push`) fire on **every** Edit
   call (not just edits to masterplan.json). Sibling hooks are
   idempotent and would exit cleanly, but this is wasteful churn on
   every file edit.
3. Without an invocation log, "hook didn't fire" and "hook fired but
   silently exited at `$FLIPPED_STEP` empty" are indistinguishable in
   the existing `auto-push.log`. We cannot diagnose without
   observability.

**Revised hypothesis (observability-first)**: add a single
`log "INVOKED ..."` line near the top of `auto-commit-and-push.sh`
(after `set -euo pipefail`, before the `if [ ! -f "$MASTERPLAN" ]`
early-exit). Keep the `if` predicate as best-effort filter. **Don't
change the wiring yet** — observe next cycle (or several) to obtain
definitive evidence of which scenario actually occurs:

- **Scenario A** (no INVOKED line in auto-push.log when step flipped to
  done): `if` predicate silently blocked the hook dispatch. Future
  cycle will then aggressively drop the `if` predicate with empirical
  backing.
- **Scenario B** (INVOKED line present + silent exit at
  `$FLIPPED_STEP` empty): hook DID fire but the Python detection
  thought no step flipped. Then the bug is in the detection logic, not
  the wiring.
- **Scenario C** (INVOKED line + commit-and-push happens): hook
  fired and committed correctly. The 38/39/40 reports of "manual
  trigger needed" were false alarms or operator misattribution.

This is the disciplined engineering move: instrument first, decide
after observing real-world evidence. Per Anthropic's
building-effective-agents principle ("add complexity only when it
demonstrably improves outcomes"), removing the `if` layer requires
demonstrable evidence — which we currently lack.

## Research-gate summary

Researcher subagent ran 2026-05-12 (tier: simple). JSON envelope:
`{"external_sources_read_in_full": 6, "snippet_only_sources": 9,
"urls_collected": 15, "recency_scan_performed": true,
"internal_files_inspected": 6, "gate_passed": true}`.

Three-variant search-query discipline observed:
- Current-year frontier: `"Claude Code hooks if matcher PostToolUse Edit tool 2026"`
- Last-2-year window: `"Claude Code hooks 2025"`
- Year-less canonical: `"Claude Code hooks"`, `"event-driven hook silent failure"`

Key external citations (≥5 sources read in full via WebFetch):

1. **Claude Code hooks reference**
   (`https://code.claude.com/docs/en/hooks`, 2026-05-12) — `matcher`
   and `if` field semantics documented; `Edit(*.ts)` example only;
   explicit note that "some event types silently ignore the `matcher`
   field" establishes a documented silent-discard pattern.
2. **Anthropic harness-design**
   (`https://www.anthropic.com/engineering/harness-design-long-running-apps`)
   — file-based handoffs as durable state; defense-in-depth principle.
3. **Anthropic building-effective-agents**
   (`https://www.anthropic.com/engineering/building-effective-agents`)
   — verbatim: "you should consider adding complexity *only* when it
   demonstrably improves outcomes." Cuts BOTH ways: keeping an
   unreliable `if` predicate is complexity-without-benefit, but
   dropping it without observability is also a guess.
4. **Anthropic multi-agent research system**
   (`https://www.anthropic.com/engineering/built-multi-agent-research-system`)
   — "Adding full production tracing let us diagnose why agents
   failed and fix issues systematically." Direct support for the
   INVOKED log line.
5. **Pixelmojo practitioner guide** (2026) — ZERO examples using the
   `if` field at hook-config level; production guides route file-
   specific filtering inside command scripts.
6. **GetAIPerks guide** — confirms `CLAUDE_DEBUG=1` is the canonical
   debug tool for hook matching; documents stdin-based payload
   delivery.

Recency scan (2024-2026): no 2025-2026 source adds semantic detail on
`if` field for Edit/Write beyond official docs. Multiple practitioner
guides published in 2025-2026 omit the `if` field entirely. No source
defends preserving an unreliable filter as best practice; multiple
sources advocate moving filtering into scripts.

Hard-blocker checklist (full text in `handoff/current/research_brief.md`):
- [x] H-1: ≥5 external sources fetched in full (6)
- [x] H-2: Recency scan section present
- [x] H-3: Three-variant search-query discipline visible
- [x] H-4: Sibling-hook impact analysis (would fire on every Edit; idempotent but wasteful)
- [x] H-5: Concrete debug-log format recommendation
- [x] H-6: ≥10 verifier claim recommendation
- [x] H-7: Cross-reference cycle-38 contract (no re-research conflict)

## Immutable success criteria

Copied verbatim from `.claude/masterplan.json` phase-23.8.4
`verification.success_criteria` (these are immutable from this
contract forward — pre-contract update was permitted because the
research gate changed the plan):

1. `settings_json_valid`
2. `edit_matcher_if_predicate_preserved`
3. `write_matcher_if_predicate_preserved`
4. `auto_commit_hook_has_invoked_log_at_top`
5. `auto_commit_hook_bash_syntax_valid`
6. `auto_commit_hook_still_filters_by_newly_done`
7. `invocation_writes_invoked_line_to_auto_push_log`
8. `invoked_line_includes_timestamp_marker_and_hook_name`
9. `no_regressions_other_hooks_bash_syntax_valid`
10. `mutation_resistance_removing_invoked_line_breaks_behavioral_claim`
11. `harness_log_has_cycle_41_entry`

## Plan steps (GENERATE phase)

### G-1 — Edit `.claude/hooks/auto-commit-and-push.sh` (add INVOKED log)

Insert a single `log` call near the top of the script. Location:
after `set -euo pipefail` (line 18), after the `mkdir -p "$LOG_DIR"`
and `log()` function definition (lines 33-35), and BEFORE the
`if [ ! -f "$MASTERPLAN" ]` early-exit (line 37).

The new line:

```bash
log "INVOKED auto-commit-and-push pid=$$"
```

This produces a log entry of the form:
`[YYYY-MM-DDTHH:MM:SSZ] INVOKED auto-commit-and-push pid=12345`

The format includes:
- ISO-8601 UTC timestamp (from existing `ts()` helper)
- Literal `INVOKED` marker for grep
- Literal `auto-commit-and-push` hook identifier (so a future shared
  log surface can distinguish hooks)
- PID for distinguishing concurrent invocations (defensive)

### G-2 — DO NOT change `.claude/settings.json`

The `if` predicates on both Write and Edit matchers stay exactly as
they are. Verifier claims 2 + 3 (`edit_matcher_if_predicate_preserved`
+ `write_matcher_if_predicate_preserved`) are regression checks
asserting we did NOT accidentally drop them.

### G-3 — Create `tests/verify_phase_23_8_4.py`

Stdlib-only verifier (json, subprocess, ast, pathlib, tempfile, shutil).
11 immutable claims matching the success_criteria above. Behavioral
tests for claims 7, 8, 10:

- **Claim 7** (`invocation_writes_invoked_line_to_auto_push_log`):
  capture `wc -l auto-push.log` before, invoke
  `bash .claude/hooks/auto-commit-and-push.sh` once, capture
  `wc -l auto-push.log` after, assert delta ≥ 1 and the new tail
  contains `INVOKED auto-commit-and-push`.
- **Claim 8** (`invoked_line_includes_timestamp_marker_and_hook_name`):
  grep the most recent INVOKED line, assert it matches a regex like
  `\[20\d\d-\d\d-\d\dT\d\d:\d\d:\d\dZ\] INVOKED auto-commit-and-push pid=\d+`.
- **Claim 10** (`mutation_resistance_removing_invoked_line_breaks_behavioral_claim`):
  copy the hook script to a tmpdir, sed out the INVOKED log line,
  invoke the mutated copy, assert the mutated copy does NOT emit
  INVOKED to auto-push.log. This proves the verifier's behavioral
  claim is anchored to the literal log line — not a no-op grep.

### G-4 — Write `handoff/current/experiment_results.md`

Standard format: what was built, file list, verbatim verifier output,
artifact shape. Disclose log-last intermediate state — claim 11
(`harness_log_has_cycle_41_entry`) will fail until the harness_log
append step.

## What this is NOT

- **NOT a wiring change.** `.claude/settings.json` is not modified.
  This cycle is observability only.
- **NOT a fix for the underlying `if`-predicate-silent-failure
  hypothesis.** If the hypothesis is correct, this cycle adds the
  instrument needed to confirm it. The wiring change (drop the `if`
  predicate, or migrate to a documented filter pattern) would be a
  future cycle (phase-23.8.5+) once we have observable evidence.
- **NOT a change to the live_check gate.** The R-1 live_check logic
  at `auto-commit-and-push.sh:123-148` is untouched.
- **NOT a change to sibling hooks.** `masterplan-memory-sync.sh`,
  `archive-handoff.sh`, `commit-reminder.sh` are unmodified.

## Rollback note

Single-commit revert. The only operational risk is the INVOKED line
itself: if its addition somehow breaks the hook (e.g., bash syntax
error), `bash -n` claim 5 catches it before the masterplan flip, and
sibling-hook bash-syntax regression checks (claim 9) catch any
collateral damage. The hook's existing fail-open discipline
(exits 0 on errors, never breaks the masterplan Write) is preserved.

## Out of scope (explicit)

- Dropping the `if` predicate (deferred pending observability data).
- Adding `CLAUDE_DEBUG=1` automation (per the GetAIPerks guide, this
  is an environment-variable knob the operator can set; we don't need
  to bake it into the harness).
- In-script stdin-based file-path filtering as a belt-and-suspenders
  layer. Stdin may already be consumed by sibling hooks; safer to
  log a fixed-string INVOKED marker and rely on the existing
  `newly_done` git-diff filter.
- R-5 (qa.md fail-mode) — separate session per separation-of-duties.
- qa.md `existing_results_check` update for `live_check_<step_id>.md`
  — separate session per separation-of-duties.

## Files to be modified

| File | Mode | Why |
|---|---|---|
| `.claude/hooks/auto-commit-and-push.sh` | edit | Add one `log "INVOKED ..."` line after the `log()` helper definition |
| `tests/verify_phase_23_8_4.py` | create | 11-claim immutable verifier including 3 behavioral mutation-resistant tests |
| `handoff/current/contract.md` | (this file) | Contract per protocol |
| `handoff/current/experiment_results.md` | create | GENERATE phase artifact per protocol |
| `handoff/current/evaluator_critique.md` | create (by Q/A) | EVALUATE phase artifact per protocol |
| `handoff/harness_log.md` | append | LOG phase artifact per protocol (LAST) |
| `.claude/masterplan.json` | edit | Flip 23.8.4 status to done LAST after Q/A PASS + harness_log append |

## Files NOT modified

| File | Why |
|---|---|
| `.claude/settings.json` | Observability-first; wiring stays as today |
| `.claude/hooks/masterplan-memory-sync.sh` | Sibling hook; not in scope |
| `.claude/hooks/archive-handoff.sh` | Sibling hook; not in scope |
| `.claude/hooks/commit-reminder.sh` | Sibling hook; not in scope |
| `.claude/hooks/post-commit-changelog.sh` | Different hook chain; not in scope |
| `.claude/hooks/lib/live_check_gate.py` | R-1 logic; orthogonal |
| `CLAUDE.md` | No new operator-visible behavior to document |

## References

- `handoff/current/research_brief.md` — full research brief
  (researcher subagent output, 2026-05-12, `gate_passed: true`)
- `handoff/harness_log.md` cycle 40 (phase-23.8.3) — observation
  source for the diagnostic ("auto-commit-and-push hook not
  auto-firing on Edits in 3 consecutive cycles")
- `handoff/archive/phase-23.8.3/experiment_results.md:148` — same
  observation
- `handoff/archive/phase-23.8.3/contract.md:208` — same observation
  carried to the contract
- `handoff/archive/phase-23.8.3/evaluator_critique.md:60` — evaluator
  note: "if it doesn't auto-fire, manual trigger as in cycles 38+39"
- `https://code.claude.com/docs/en/hooks` — hooks reference (2026-05-12)
- `https://www.anthropic.com/engineering/harness-design-long-running-apps`
- `https://www.anthropic.com/engineering/building-effective-agents`
- `https://www.anthropic.com/engineering/built-multi-agent-research-system`
- `handoff/archive/phase-23.8.1/contract.md` + `experiment_results.md`
  — reference shape for hook-touching verifier with behavioral
  mutation-resistance tests (the live_check gate cycle)
- `tests/verify_phase_23_8_1.py` — reference shape for the verifier
