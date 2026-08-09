# phase-86.6 -- PARKED MID-STEP (not closed, status stays `pending`)

**Parked 2026-08-10 ~01:00 CEST** under the overnight goal's rule: do not burn
the night on one step. This file exists so the next session resumes from
measurements rather than re-deriving them. **No Q/A has graded any of this.**

## What is DONE and measured

**Research gate PASSED** (`wf_dc58bae7-aef`; 18 sources read in full, 44 URLs,
audit-class `coverage.dry` after 10 rounds). Contract written before any code.

**Criterion 1 -- the derivation method, with its recall VALIDATED.**
`scripts/qa/derive_live_state_writers_86_6.py` derives the mutating kill-switch
surface FROM `kill_switch.py` (never a hand-list: 7 methods reach
`_append_audit`) and AST-scans both test trees for invocations.

It flags the named probe, which is the whole point:

```
RECALL VALIDATION (criterion 1)
  probe: test_book_safety_69.py::test_peak_reset_dark_by_default
  FLAGGED -- calls reset_peak; redirects=True
```

**Why a static method was required:** a RUNTIME write-detector reports that probe
CLEAN. `reset_peak` returns early while `kill_switch_peak_reset_enabled` is
False, so no bytes reach the journal — the call site sits waiting for the
already-approved KS-PEAK-RESET token. The population must be derived from the
CALL, not the WRITE.

**Population: 24 tests invoke a mutating kill-switch API without a redirect my
detector can see.** PRECISION IS UNVALIDATED — the detector only recognises
`_`-prefixed module-scope fixtures, so tests isolated by a *named* fixture read
as unprotected. Measured empirically: those files run **164 passed with the live
journal byte-identical**, so none of them writes today. The risk is LATENT, which
is 86.1's landmine shape.

**Criterion 2 -- the preventer, and the trap it had to avoid.** Installed at
repo-root conftest import via `sys.addaudithook` (PEP 578).

MEASURED in the project venv, and it inverts the repo's own precedent:

```
Failed MRO: ['Failed', 'OutcomeException', 'BaseException', 'object']
kill_switch shape: logger.warning swallowed: RuntimeError
kill_switch shape: logger.warning swallowed: PermissionError
```

`kill_switch._append_audit` catches `Exception`. Both existing in-repo guards
raise `RuntimeError`, which IS an `Exception`. A refusal built to the house
pattern would be **silently absorbed — write blocked, test GREEN**. So
`LiveStateWriteRefused` derives from `BaseException`, and that property is
proven against the production shape:

```
SWALLOWED=None  ESCAPED=LiveStateWriteRefused
```

Positive and negative controls both pass: a write to the kill-switch journal is
REFUSED; a read of it succeeds (6361 bytes); a write elsewhere under `handoff/`
is not blocked, by design.

**No regression — measured as a DELTA on the full tree, not asserted.**

| run | result |
|---|---|
| baseline, guard OFF | **14 failed, 3207 passed** |
| scoped guard ACTIVE | **14 failed, 3207 passed** |
| delta | **0 added, 0 fixed** |

The 14 are pre-existing and unrelated (plus a pre-existing collection error in
`tests/services/test_persist_lite_analysis.py`, which imports
`_persist_lite_analysis` — a symbol `autonomous_loop` no longer exports).

## A number I nearly shipped wrong

Blocking the WHOLE `handoff/` tree turned the run to 21 failed. I first read that
as "the guard breaks 21 tests". **It does not.** Against the 14-failure baseline
the true cost is **+7**, and those 7 write `handoff/.autonomous_loop.lock`,
`handoff/.cycle_heartbeat.json` and one probe file under `handoff/logs/` — none
of them a kill-switch write. Blocking scope was therefore narrowed to the
kill-switch journal and its DERIVED archive dir, which is exactly what criteria
2-4 are about.

## A tier I removed rather than shipped

The first revision had a "report, don't block" tier for the rest of `handoff/`.
MEASURED: a full run emitted **zero** visible lines, because pytest captures
stderr and discards it for PASSING tests — every test that writes live state
without failing. A report tier that reports nothing is the same silent-failure
class this step exists to close, so it was removed rather than left looking
useful. The finding it was meant to carry is recorded above and needs its own
step.

## What is NOT done -- the reason this is PARKED, not closed

- **Criterion 5** — the effect of a refusal on every PRODUCTION caller of
  `_append_audit` is NOT measured. The argument that conftest is never imported
  in production is sound but **unmeasured**, and this step requires measuring it.
- **Criterion 6** — no mutation test yet (must run against a tmp COPY of the
  journal, never the live one).
- **Criteria 7 and 8 — PART B (subprocess) IS ENTIRELY UNSTARTED.** Research
  measured that child processes load NO conftest and that 72 files shell out, so
  this needs a different seam and the added-offending-call-site proof criterion 7
  demands. No caller census of `smoke_cc_rail_e2e.py` has been taken, so per
  criterion 8 the `--backend-url` default must NOT be changed.
- **Criterion 9** — the five-channel enumeration is not written up.
- **Criterion 3** — tmp-redirected tests still writing is covered incidentally by
  the delta-0 run, but has no dedicated assertion.

## Honest risk statement for the operator

The conftest guard IS committed and ACTIVE. It is measured non-breaking
(delta 0 over 3221 tests) and it closes a channel the conftest's own docstring
declared open. **But it has not been graded by a Q/A, and `sys.addaudithook`
cannot be uninstalled once installed** — PEP 578 also states plainly that it
"is not sandboxing". `PYFINAGENT_LIVE_STATE_GUARD=off` disables the refusal for
a run that genuinely must touch live state.
