# Research Brief -- phase-80.46 (flaky CI gate: subprocess timeout under CPU contention)

Tier: **moderate** (caller-specified). NOT audit-class.
Started: 2026-07-26. Researcher: Layer-3 harness researcher.

## Question

A test run reported `249 passed, 1 skipped, 2060 deselected, 1 warning, 1 ERROR
in 95.89s` (normal ~16s). Five subsequent runs clean; the ERROR was never
identified. Main's UNTESTED hypothesis: `backend/tests/test_phase_75_ci_gates.py`
shells out to `pytest --collect-only` with `timeout=60`, and a
`subprocess.TimeoutExpired` surfaces in pytest as an **ERROR** (not FAILURE),
matching the observed shape.

Sub-questions:
1. Does an uncaught exception in a test body land in pytest's ERROR bucket or
   the FAILURE bucket? (load-bearing for the hypothesis)
2. Flaky-test taxonomy: where does "fixed timeout under resource contention"
   rank, and what is the recommended remedy?
3. Timeout design for subprocesses in tests: bigger constant / scaled / retry /
   eliminate?
4. In-process alternatives to shelling out to pytest (`pytest.main`, Collector
   API); is re-entrant `pytest.main()` inside a running session safe?
5. Recency scan 2025-2026.

## Status log (write-first)

- [x] skeleton written
- [ ] internal audit
- [ ] external sources
- [ ] recency scan
- [ ] recommendation
- [ ] envelope

## Queries run

(filled in below)

## HEADLINE (written early, write-first): the hypothesis is REFUTED

Two independent authoritative sources say an exception raised **inside a test
function body** is a **FAILURE**, and only setup / teardown / collection
exceptions are **ERRORS**. `subprocess.TimeoutExpired` at
`test_phase_75_ci_gates.py:120` is raised in the test BODY. Therefore a
`--collect-only` subprocess timeout would have printed `1 failed`, not
`1 error`. The observed line had **zero** `failed`. Details + the surviving
alternative hypotheses are in "Key findings" and "Recommendation".

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://github.com/pytest-dev/pytest/discussions/7950 | 2026-07-27 | official-project discussion (maintainer answer) | WebFetch (full) | "pytest considers a *failure* any assertion error or exception raised inside a test function; *errors* happen when an assertion error or exception is raised during setup/teardown/collection." junitxml uses the same mapping. |
| https://docs.pytest.org/en/stable/how-to/usage.html | 2026-07-27 | official doc | WebFetch (full) | `pytest.main()` "will not raise SystemExit but return the exit code instead"; BUT "Calling pytest.main() will result in importing your tests and any modules that they import. Due to the caching mechanism of python's import system, making subsequent calls to pytest.main() from the same process will not reflect changes to those files between the calls. For this reason, making multiple calls to pytest.main() from the same process ... is not recommended." |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|

## Recency scan (2024-2026)

(pending)

## Key findings

(pending)

## Internal code inventory

(pending)

## Recommendation

(pending)

## Research Gate Checklist

(pending)
