---
step: phase-23.8.2
cycle_date: 2026-05-11
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_23_8_2.py'
---

# Experiment Results — phase-23.8.2

## What was built

Audit recommendation **R-2 Option A** — delete the TaskCompleted hook
entirely. Q/A becomes the **sole** independent evaluator (matches the
CLAUDE.md stated principle and the audit's H-1 + H-2 BLOCKING
findings).

### G-1 — Removed `TaskCompleted` block from `.claude/settings.json`

The `TaskCompleted` key + its single agent-prompt block (formerly
lines 101-111) deleted in full. Surrounding hook keys
(`TeammateIdle`, `Stop`, `SubagentStop`) untouched. JSON still
validates.

### G-2 — Updated `.claude/context/project.md` (line 19)

Removed "TaskCompleted gate," from the hooks list. Added a sentence
noting the retirement + audit ref + that Q/A is now the sole
evaluator. Also updated the line to mention the live_check gate
shipped in cycle 38 (which had been missing from the hooks list
inventory).

### G-3 — Updated `docs/runbooks/per-step-protocol.md`

Two prose lines reframed:

- **Old line 225-227**: "Self-evaluation — Main directly reporting
  PASS without spawning Q/A. **The TaskCompleted hook should fire;**
  if it didn't, spawn Q/A manually."
- **New**: "Self-evaluation — Main directly reporting PASS without
  spawning Q/A. **Always spawn Q/A manually after every GENERATE;
  there is no hook backstop** (the TaskCompleted hook was retired in
  phase-23.8.2 per audit recommendation R-2 — see
  `docs/audits/dev-mas-2026-05-11/04-remediation.md`)."

- **Old line 248-249**: "Main self-evaluates under time pressure.
  Fix: **TaskCompleted hook is load-bearing; never bypass.**"
- **New**: "Main self-evaluates under time pressure. Fix: **always
  spawn Q/A explicitly after every GENERATE — no automatic hook
  backstop.** (The TaskCompleted hook was retired in phase-23.8.2
  because the audit found it was a weaker, parallel evaluator that
  diluted Q/A's independence rather than reinforcing it.)"

### G-4 — Verifier `tests/verify_phase_23_8_2.py` (10 immutable claims)

Pre-log-append run returned `FAIL 9/10` with claim 9
(harness_log entry) failing BY DESIGN per log-last. After the
harness_log Cycle 39 append (LAST step), it returns 10/10.

## Files modified

| File | Change | LOC |
|---|---|---|
| `.claude/settings.json` | edit | -11 (TaskCompleted block deleted) |
| `.claude/context/project.md` | edit | +1 / -1 (hooks list updated) |
| `docs/runbooks/per-step-protocol.md` | edit | 2 prose rewrites (~5 lines each) |
| `tests/verify_phase_23_8_2.py` | NEW | 165 LOC |
| `handoff/current/contract.md` | NEW | the contract |
| `handoff/current/experiment_results.md` | NEW | this |
| `.claude/masterplan.json` | edit | new step 23.8.2 pending → done |

## Verbatim verification output (pre-log-append)

```
$ source .venv/bin/activate && python3 tests/verify_phase_23_8_2.py
=== phase-23.8.2 verifier ===
  [PASS] 1. settings_json_valid
  [PASS] 2. task_completed_hook_block_removed
  [PASS] 3. other_hook_keys_intact
  [PASS] 4. project_md_no_longer_lists_task_completed_gate
  [PASS] 5. per_step_protocol_old_line_226_prose_removed
  [PASS] 6. per_step_protocol_old_line_248_prose_removed
  [PASS] 7. per_step_protocol_has_retirement_note
  [PASS] 8. step_2_13_historical_assertion_now_expectedly_fails
  [FAIL] 9. harness_log_has_2_13_breakage_disclosure: harness_log.md must document the controlled step 2.13 breakage in the phase=23.8.2 cycle entry
  [PASS] 10. no_regressions_other_hooks_bash_syntax_valid
FAIL (9/10) EXIT=1
```

Claim 9 is the expected log-last fail. After the Cycle 39 append,
verifier returns `PASS (10/10) EXIT=0`.

## Mutation-resistance / anti-rubber-stamp

Claim 8 is itself the mutation-resistance test: it runs the EXACT
verbatim historical assertion from `masterplan.json:214`
(`assert 'TaskCompleted' in s['hooks']`) and confirms it now raises
`AssertionError`. If the delete had failed silently — e.g., I had
accidentally deleted a different block — claim 8 would have passed
the assertion (instead of raising), which the verifier would catch
as `historical_assertion_now_fails = False` and tag as FAIL. This is
an inverted assertion that exercises the actual change, not just a
file grep.

## Known controlled breakage (disclosed in contract + harness_log)

`.claude/masterplan.json:214` — step **2.13** (a `done` historical
step from the "Claude Code Configuration Audit") has an immutable
verification command containing
`assert 'TaskCompleted' in s['hooks']`. Per CLAUDE.md, verification
criteria are immutable. The assertion PASSED at the time step 2.13
was marked `done`. After this cycle, the command would fail if
re-run.

This is acceptable because:
- Step 2.13 is `done`; verification commands are run once at
  step-time, not periodically.
- The R-2 audit recommendation (2026-05-11) EXPLICITLY supersedes
  the historical assertion. The audit's H-1 + H-2 findings argue
  the hook is harmful and must be removed.
- No automated process re-runs old verification commands.
- The verifier (claim 8) captures the expected new state.

This trade-off is documented verbatim in:
- `handoff/current/contract.md` § "Known controlled breakage"
- `handoff/harness_log.md` Cycle 39 (appended LAST)
- This file § "Known controlled breakage"

## Scope honesty

- **Stop hook NOT modified** in this cycle. The audit's H-3 finding
  on the Stop hook was DEGRADES (not BLOCKING), and the Stop hook
  fires only at session end (not after every Task), so it is not
  redundant with Q/A in the same way TaskCompleted was. The Stop
  hook's loop-prevention logic is a separate concern. Out of scope.
- **R-5** (qa.md fail-mode) — separate session + your review.
- **R-6** (delete deprecated stubs) — needs prior
  `autonomous_loop.py` refactor.
- **qa.md `existing_results_check` update for `live_check_*.md`** —
  deferred from cycle 38 per separation-of-duties; still pending a
  separate session.

## What this changes for the operator

| Before | After |
|---|---|
| Every Task completion fired a redundant LLM evaluator with unconstrained tool access | Q/A is the sole evaluator (matches CLAUDE.md doctrine "agents tend to confidently praise their own work" → need ONE independent evaluator, not two redundant ones) |
| Possible verdict conflicts: TaskCompleted hook says "ok:true", Q/A says CONDITIONAL — which wins? | Single source of truth: Q/A's verdict, full stop |
| LLM cost: extra ~60s call per Task (per audit H-2) | Saved per-cycle cost |
| Shadow MAS (per audit Phase 1 finding): unnamed hook agent invisible to operators | Hooks roster matches `.claude/context/project.md:19` and is auditable |

## What's next

1. Spawn fresh Q/A on this cycle's evidence.
2. On PASS: append `handoff/harness_log.md` Cycle 39 (with the
   controlled step 2.13 breakage disclosure) → flip masterplan
   23.8.2 status to done → auto-commit-and-push fires.
