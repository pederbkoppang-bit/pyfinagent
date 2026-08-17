# experiment_results -- step 86.71 (GENERATE, 2026-08-17)

Contract: `contract_86.71.md`. Research gate: PASSED (`wf_77c2679f-de9`,
recomputed; 9 sources in full, 30 URLs). Verbatim command evidence in
`live_check_86.71.md`.

## What was built

**`scripts/harness/attempt_gate.py`** -- the PreToolUse gate that finally wires
`attempt_budget.py` (which had NO runtime caller and NO persistence):

- Registered in `.claude/settings.json` as a `PreToolUse` hook with matcher
  `Workflow` -- the seam where every Layer-3 run originates (655 historical
  PreToolUse Workflow rows measured by the research gate).
- **Counts on ATTEMPT, not outcome**: one row appended to the append-only
  `handoff/audit/attempt_budget_audit.jsonl` at every allowed launch that
  carries a step_id (object OR string args; hostile ids refused, unattributable
  launches allowed-and-not-counted, loudly).
- **Denies at the ceiling** (exit 2 -- the only blocking exit code per the
  official hooks doc) at ZERO token cost, writing
  `handoff/current/escalation_attempt_budget_<sid>.md` from the module's own
  `escalation_summary()` plus the exact audited `--operator-extend` command.
- **A denial is not a verdict** -- it produces no verdict artifact and the
  module's `disposition()` has no path from exhaustion to PASS
  (`test_exhaustion_cannot_auto_pass`, still green).
- **The PASS exception**: a PASS in the verdict ledger always allows, so
  re-grades of closed steps (the 86.93 workflow) are never budget-blocked --
  it can only ever admit MORE evaluation of an already-passed step.
- **Fail-open-but-loud** on internal errors, with the bound stated in the
  docstring: hook timeouts/schema errors fail open by platform design, so the
  gate is a cost rail with an audit trail, not an unforgeable barrier.
- **Operator extension is an audited act**: `--operator-extend <sid> --by N
  --reason "..."` appends a labelled row in a git-tracked stream; the deny
  message quotes the exact command.

## The criteria, discharged

1. **Repeat rate re-derived** (population rule + command in live_check §1):
   **66.5%** (320/481 step-attributable runs; qa 302/390 = 77.4%, researcher
   18/90 = 20.0%; max 9 qa runs on one step). **Disagreement with the filed
   58.4% reported, not adopted**: the corpus grew from 513 to 580+ records
   between measurements; both figures carry their windows.
2. **No-runtime-caller re-verified with a NAMED positive control**: the pattern
   `attempt_budget` hits `backend/tests/test_phase_86_32_attempt_budget.py`
   (control, non-zero) and zero files under scripts/harness (minus the module),
   backend non-test, or .claude/hooks. *(My first control -- grepping the
   module for its own filename -- returned 0 and was itself a failed control;
   replaced, and the failure disclosed rather than deleted.)*
3. **Cross-session persistence demonstrated across separate processes**: six
   separate `python3` invocations (the pipe-test) appended rows read back by
   later invocations; the live gate then read the same file from the hook
   process. The ledger is a file, not process state.
4. **Wired at the origin and DRIVEN on REAL launches** (verbatim in live_check
   §3): step 999.2 seeded to its ceiling; an actual Workflow tool call for it
   was **DENIED by the live hook** (hot-reloaded, exit 2, zero tokens, deny
   message + escalation file); the next real launch (the 86.85 cycle-7
   respawn) was **ALLOWED and COUNTED** -- its row carries this session's id,
   `workflow: qa-verdict.js`, `attempt_number_inclusive: 1`.
5. **Exhaustion escalates and never auto-passes, under every flag combination**:
   the gate has NO disable flag; the only env overrides are testing-only ledger
   paths (documented); `test_exhaustion_cannot_auto_pass` + the 86.32 matrix
   re-run green after the docstring edit; the deny path emits no verdict key.
6. **Q/A-side repeat cost addressed by construction**: the gate counts EVERY
   Workflow launch with a step_id -- the live counted row IS a qa-verdict
   launch.
7. **No flag promoted, no .env written**: ASK-1 records the ceilings
   (5 / 1.2M) as the module's internally-measured defaults, operator-changeable
   in one place.
8. **Mutation-tested**: `scripts/qa/mutation_matrix_86_71.py` -- 6 subprocess
   cells over the gate's call sites (deny branch removed, append dropped at the
   call site, extraction neutered, corrupt-row skip, deny demoted to allow,
   disposition ignored), control observed GREEN first, md5 restore proof,
   6/6 KILLED. **The control run found a REAL bug on its first execution**:
   the deny message's unconditional `relative_to(REPO)` raised under an
   overridden escalation dir, fell into the fail-open handler, and turned a
   deny into an allow -- fixed with the story in the code comment. A matrix
   whose control catches the subject before any mutant runs is the control
   doing its job.

## Also in this step

- `attempt_budget.py`'s literature overclaim corrected at the docstring
  (research-gate adversarial finding): Fowler's canonical breaker DOES reset on
  a successful half-open probe; the true claim is narrowed to work-accounting
  budgets, and the ceilings are marked internally-sourced. Module tests (15)
  and the 86.32 matrix re-run green after the edit.

## Honest limits

- History is NOT backfilled: the pre-existing danger-hook discarded
  `tool_input` on all 185,020 historical rows, so counting starts at the
  wiring (2026-08-17). The verdict-ledger PASS exception bridges the gap for
  closed steps.
- The gate cannot bind a hook-disabled or hook-crashed session (platform
  fail-open); its guarantee is that no launch is silently uncounted WHILE the
  hook runs, and that the audit stream makes gaps visible after the fact.
- Token-ceiling enforcement uses the module's `max_tokens` only via recorded
  outcomes; per-launch token attribution at the PreToolUse seam is not possible
  (tokens are unknown at launch). The attempt ceiling is the primary bound.
