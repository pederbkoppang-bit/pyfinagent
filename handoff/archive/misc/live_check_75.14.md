# live_check -- Step 75.14 (gap5-07/04/05/09, gap4-11)

Date: 2026-07-24. Verbatim captures; rc=$? discipline.

## 1. Immutable verification command (exit 0)

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_prompt_contracts.py -q
18 passed in 4.02s   # cycle-2 re-run (real-branch routing tests included)
pytest_exit=0
```

## 2. Change surface + lint

```
$ git diff --stat HEAD -- backend/ | tail -1        # cycle-2 regenerated
 9 files changed, 177 insertions(+), 70 deletions(-)   (+ new 18-test file + 3 handoff docs)
$ uvx ruff check --select F821,F401,F811 <git-derived 5-file scope> backend/tests/test_phase_75_prompt_contracts.py
All checks passed!
ruff_exit=0
```

## 3. Full-suite regression (fresh run against the FINAL tree)

```
FAIL SET IDENTICAL TO BASELINE
10 failed, 1446 passed, 12 skipped, 5 xfailed, 1 xpassed, 1 warning in 93.50s (0:01:33)
```
(CYCLE-2 re-run against the final tree including the extracted
_judge_parse_fail_fallback + rewritten criterion-6 tests. Cycle-1's runs:
a mid-edit stale run was stopped and re-run -- same 10-fail baseline set
every time; measurement honesty over convenience.)

## 4. Behavioral SSTI proof (offline, the criterion-1 fixture)

test_format_skill_value_containing_placeholder_stays_inert: a kwarg VALUE
carrying '{{output_schema}}' yields exactly ONE expansion of the real
schema (the template's own) and the value renders as '{ {output_schema}}'
-- template content can no longer be pulled into external text. Mutation
M1 (un-escape) flips this red; stub-mutation M8 (fixture neutered) also
flips red, proving the fixture is load-bearing.

## 5. Flag-gated / UI / live-loop note

The ONLY flag-gated behavior (paper_risk_judge_parse_fail_reject) ships
DARK: settings-default False proven by test; OFF path byte-identical
APPROVE_REDUCED dict (field-for-field source assert). ON-vs-OFF is a $0
no-op by construction until the operator flips it (and even then REJECT
binds only with shape_fix/reject_binding ON -- documented in the flag
description). No UI surface (the frontend impact of seam alignment is
DOCUMENTED in operator_decision_75.14_schema_extension.md, not acted on).
No live LLM call was made.
