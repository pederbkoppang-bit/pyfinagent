# UNRECOGNISED STEP ID -- '86.118.1' -- OPERATOR DECISION REQUIRED

A Workflow launch claimed this step id. It is not a step in
`.claude/masterplan.json`.

## THIS IS NOT A PASS, NOT A FAIL, AND NOT AN EXHAUSTION

No budget was consumed and no verdict is implied. The launch was
stopped before any tokens were spent.

## Why an unrecognised id is refused rather than counted

Every distinct id gets its own attempt allowance. An id that names no
step therefore mints a FRESH allowance on demand -- appending `.1` to a
real step id was enough to do it, through the ordinary `args` field,
with no file edits. The live ledger already carries `999.2`, which is
absent from every masterplan step and holds 5 attempt rows.

## How to proceed (operator)

- If this IS a step attempt: correct the id to a real masterplan step.
- If it is NOT a step attempt (self-audit, ad-hoc workflow): omit
  `step_id` from `args` entirely. That path is still allowed and
  uncounted, by design.
- If the step is real but not yet filed: file it in the masterplan
  first. The plan of record is the allowance list.
The denial itself is NOT a verdict: the step remains exactly as the
last Q/A left it.

*(written 2026-08-20T19:26:29Z by attempt_gate.py at the deny, reason=unknown_step_id)*
