# live_check_75.5.2 — the remaining gemini-2.5 behavioural pins routed to constants

All output verbatim from live runs 2026-07-24. Offline-only step (constants
refactor, zero behavioural change by construction; no UI surface).

## 1. Verification command (immutable) — exit 0

```
$ cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_75_5_2_model_pins.py -q
306 passed, 1 warning in 4.01s
$ .venv/bin/python -m pytest backend/tests/test_phase_75_5_2_model_pins.py backend/tests/test_phase_75_llm_rail.py -q
348 passed, 1 warning in 7.05s      (75.5 rail suite green -- regression proof)
```

(306 = the per-file strict scan parametrizes across the whole backend tree +
the resolution/tripwire/warning tests.)

## 2. Independent residual-literal census (Main, not the executor's claim)

```
$ grep -rn "gemini-2.5" backend/ --include="*.py" | grep -v backend/tests | grep -v model_tiers.py | grep -v cost_tracker.py | grep -v settings_api.py | wc -l
0
```

Nine behavioural pins routed (the 8 re-derived census sites + the 9th the
audit-class research gate discovered at scripts/harness/run_autonomous_loop.py:74);
the llm_client:985 family guard routed via the new GEMINI_2_5_FAMILY_PREFIX
constant; co-located prose/table literals cleaned per the strict-scan precedent.
NO tier VALUE changed anywhere — pinned by the migration-tripwire test.

## 3. Mutation matrix — 4 designed, ONE SURVIVOR FOUND AND FIXED, final 4/4 killed

First run: **M4 SURVIVED** — an aliased import (`GEMINI_DEEP_THINK as
GEMINI_WORKHORSE`) defeats a call-site name check while binding the wrong value.
Per the §4c doctrine Main STRENGTHENED the guard (import-level AST assertion:
GEMINI_WORKHORSE imported un-aliased AND GEMINI_DEEP_THINK not imported at all
— comment in the test cites this matrix finding), then re-ran the FULL matrix:

```
SUMMARY: 4 mutations, 4 killed, survivors: NONE
=== post-restore sanity: pytest exit 0 ===
```

| # | Mutation | Killed by |
|---|---|---|
| M1 | literal restored at one site (sentiment.py) | that file's parametrized C1 scan case |
| M2 | GEMINI_WORKHORSE VALUE changed | 7 failures: the tripwire value-pin + every resolution/behavioural-capture/warning test |
| M3 | **SCAN/FIXTURE**: EXCLUDE_FILES over-broadened to hide a real pin file | `test_scan_is_non_vacuous` (the superset self-test) |
| M4 | site misrouted via ALIASED import (C1 stays green — the sneaky shape) | the strengthened import-level guard (first run: SURVIVED; fixed; killed) |

Runner + both verbatim logs: scratchpad `run_mutations_75_5_2.py`,
`mutation_matrix_75_5_2.txt`.

## 4. Lint (scope = all 12 touched files, derived from git after last edit)

New test file: `All checks passed!`. All 10 edited production files:
finding-class census IDENTICAL to their `git show HEAD:` baselines (per-file
md5-compared) — zero new findings anywhere.

## 5. git diff --stat (change surface)

```
 12 files changed, 80 insertions(+), 57 deletions(-)
 (model_tiers.py: +GEMINI_2_5_FAMILY_PREFIX only; 9 pin files; the new test;
  masterplan status/queue edits)
```

## 6. Deadline context (why P1)

gemini-2.5-flash AND -pro retire 2026-10-16 (triple-confirmed official). The
migration is now a ONE-FILE change (model_tiers.py) with the value-pin tests as
the deliberate tripwire. Queued 75.5.2.1: the tripwire is never EMITTED at
runtime (zero callers), display/sample strings, and the in-file _BUILD_TIER
literal.

## 7. CYCLE 2 — the Q/A's independent probe found a SECOND surviving site pair; fixed

The cycle-1 Q/A (wf_71687e5e-c63, CONDITIONAL) re-applied the aliased-import
mutation at ALL AST-guarded sites — not just site 9 where Main fixed it — and
found it SURVIVES at sites 6,7 (`backend/services/autonomous_loop.py::
_run_gemini_analysis`): the name-reference guard is alias-defeatable and those
sites have no behavioural backstop. Named fix applied:
`test_autonomous_loop_model_tiers_import_is_alias_proof` (every model_tiers
ImportFrom in the module must bind un-aliased and never import
GEMINI_DEEP_THINK; test comment cites the Q/A finding). Matrix extended with
M5 = the exact surviving shape, full re-run verbatim:

```
SUMMARY: 5 mutations, 5 killed, survivors: NONE
=== post-restore sanity: pytest exit 0 ===
$ .venv/bin/python -m pytest backend/tests/test_phase_75_5_2_model_pins.py backend/tests/test_phase_75_llm_rail.py -q
349 passed, 1 warning
```

(cycle-2 log: scratchpad `mutation_matrix_75_5_2_cycle2.txt`)
