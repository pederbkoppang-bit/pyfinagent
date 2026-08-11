STATUS: INCOMPLETE -- not a verdict
STEP: 86.38
WRITTEN: 2026-08-11T07:38:52Z

# Q/A cycle 2 (first cycle to reach a verdict; cycle 1 dropped at 40 tool uses)

## Plan
- A. harness-compliance audit (5 items)
- B. deterministic: immutable parse cmd, git diff scope, ruff gate, scoped pytest, runtime smoke
- C. criteria 1-6 MET/NOT MET
- Attack list from Main: (a) extraction-vs-reimplementation of `_degradation_summary_fields`;
  (b) consistency of the "did not fire, denominator unmeasured" limit incl. docstrings;
  (c) removal of `_intended_path` on a redundancy argument; (d) paging byte-identical (M6 strict `>`).

## Findings (appended as established)

### D1. Immutable verification command -- EXIT=0
`bash -c 'source .venv/bin/activate && python -c "import ast;ast.parse(open(...autonomous_loop.py...))"'`
-> `parsed`, EXIT=0. Parse-only; proves syntax and nothing else.

### D2. Harness compliance
- research_brief_86.38.md 08:55 local < contract_86.38.md 08:58 < experiment_results 09:15
  < live_check 09:22 < evaluator_critique 09:34. Commit order c116e63a -> cef76c3b (PLAN)
  -> fd419038 (GENERATE) -> 5e97ca27 -> 07fd7c07 -> a477b74a -> da4fdb01.
- masterplan 86.38 status still `pending`; harness_log has NO `result=` line for 86.38
  (grep -nF '86.38' returns only two prose mentions in other steps' blocks).
  => log-last honoured; CONDITIONAL counter for 86.38 = 0. 3rd-CONDITIONAL rule not armed.
- cycle 1 dropped with NO verdict; evidence CHANGED (new commits a477b74a, da4fdb01) --
  not verdict-shopping.

### D3. Scope / unintended change
`git diff --name-only HEAD` = only my own WIP file + handoff/audit jsonl. The step's work
is committed. 86.38-specific files: backend/services/autonomous_loop.py,
backend/services/cycle_health.py, backend/tests/test_phase_86_38_degradation_visibility.py,
scripts/qa/{mutation_matrix,derive_lite_fallback_census}_86_38.py. The c116e63a~1..HEAD
range also carries PEER-SESSION files (86.21/86.29/86.36 scripts) -- not this step's.

### D4. Ruff F821,F401,F811 over derived scope (14 .py files, xargs -0, non-empty asserted)
`All checks passed!` exit=0.

### D5. Tests
`pytest backend/tests/test_phase_86_38_degradation_visibility.py
 backend/tests/test_phase_60_1_deep_pipeline.py -q` -> **29 passed**, exit 0. Reproduces the
handoff's claim exactly.

### D6. Consumer-contract check (I derived it myself)
grep -rn --include='*.py' --include='*.ts' --include='*.tsx' --include='*.json'
 -E 'fallback_rate|fallback_alarm_fired|fallback_reasons' backend/ frontend/src/ scripts/
=> ZERO consumers outside autonomous_loop.py itself + a settings.py docstring + the matrix.
Making those keys unconditional cannot change any downstream reader. CONFIRMED.

### D7. Paging behaviour (attack d)
`_fallback_rate_check` is NOT in the diff; `fire = n_total > 0 and (n_fallback/n_total) > threshold`
unchanged. Threshold default 0.5 (settings.py:50). I independently reproduced author cells
under an in-memory harness (see M-probe below): M6 (`>` -> `>=`) KILLED, M1 KILLED, M2 KILLED.
No risk threshold / position sizing / gate / risk-judge touched. Criterion 4 + 5 MET.

### D8. `_intended_path` removal (attack c) -- REMOVAL IS SAFE
Single write site: `_lite["_fallback_reason"] = ...` and (formerly) `_lite["_intended_path"]`
were CO-WRITTEN in the same `if isinstance(_lite, dict)` block. grep shows exactly ONE
assignment of `_fallback_reason` in production (autonomous_loop.py:2235) plus one read/copy
into full_report (:3371). So the two sets are identical BY CONSTRUCTION, not by argument --
they cannot ever differ. Main's redundancy claim holds.

### M-PROBE. Independent mutation run (in-memory; ZERO writes to the tree)
Method: read autonomous_loop.py, mutate the STRING, exec into a module injected at
sys.modules["backend.services.autonomous_loop"], patch inspect.getsource for that module so
SOURCE-SCAN guards also see the mutant, then pytest.main the 86.38 file. CONTROL (no mutation)
= 7 passed, and MC (unwire the seam) = 1 failed -> the probe DISCRIMINATES.

| cell | mutation | result |
|---|---|---|
| CONTROL | none | 7 passed (rc=0) |
| MC | `summary.update(_degradation_summary_fields(` -> `_unused_deg = (` | **KILLED** (rc=1) |
| M1 (author) | seam `if not n_total:` -> `if True:` | **KILLED** |
| M2 (author) | `"fallback_alarm_fired": bool(fire)` -> `False` | **KILLED** |
| M6 (author) | `>` -> `>=` in `_fallback_rate_check` | **KILLED** |
| **MX (mine)** | delete `degradation=_degradation,` from the `record_cycle_end(...)` call | **SURVIVED (7 passed)** |
| **MY (mine)** | drop `"fallback_rate"`,`"fallback_reasons"` from the `_degradation` key tuple | **SURVIVED (7 passed)** |

=> The summary -> `_degradation` -> `record_cycle_end` WIRING has NO guard. `grep -rn
'degradation=_degradation' backend/tests/ tests/` returns ZERO, and no test executes
`run_daily_cycle` far enough to reach that call. Under MX every future cycle persists
`degradation: {}` -- i.e. the exact defect this step exists to remove (a sub-threshold
degraded cycle leaving no durable trace where the operator looks) returns silently and the
suite stays green. This is the "guards stop one seam short" class. Prompt criterion 6
("mutation-test every new guard, including reverting the fix at the call site") is NOT MET
for this call site.

### FINDING 2 -- the boundary-causality claim is NOT consistent (attack b)
live_check §D states an HONEST LIMIT: the alarm's denominator is
`len(candidate_analyses)+len(holding_analyses)`, was NOT measured, so "missed by exactly one
ticker" is NOT claimed. But FOUR other places assert the boundary AS MEASURED CAUSE:
- autonomous_loop.py call-site comment: "MEASURED on the 2026-08-10 cycle: 3 of 6 analyses
  fell back ... `3/6 = 0.500` did not strictly exceed 0.500, the alarm correctly stayed quiet"
- cycle_health.py comment: "(measured 2026-08-10, 3 of 6 analyses on the lite fallback,
  3/6 = 0.500, no page)"
- test module docstring: "`3/6 = 0.500` does not exceed `0.500`, so the alarm correctly stayed quiet"
- test docstring:49 "3 of 6 is EXACTLY 0.5 and must not fire -- **this is the measured case**"
Six TICKERS were measured; the alarm's n_total was not. If holdings analyses are in the list
n_total > 6 and the non-firing has a different cause. The disclosure lives only in the handoff
artifact; the version that survives in production source states the unmeasured half as measured.

### Criteria (masterplan verification.success_criteria + prompt list)
1. 429 body captured verbatim -- MET (reproduced from backend.log myself, see below)
2. per-cycle full-vs-lite >=10 cycles + command -- MET (census re-run below)
3. last paper_trades date stated -- MET
4. non-scope: no risk threshold/sizing/gate/risk-judge -- MET (D7)
5. non-scope: no paid Gemini tier -- MET (no config/billing change in diff)
6. mutation-test every new guard incl. reverting the fix at the call site -- **NOT MET** (MX/MY)
