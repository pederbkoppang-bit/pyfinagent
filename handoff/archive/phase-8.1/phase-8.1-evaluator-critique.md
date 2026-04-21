# Evaluator Critique -- phase-8 / 8.1 (TimesFM shadow-logged feature pilot)

**Agent:** qa (merged qa-evaluator + harness-verifier)
**Run id:** qa_81_v1
**Cycle:** 1 (first Q/A on phase-8.1)
**Date:** 2026-04-20
**Verdict:** PASS

## 5-item harness-compliance audit

| # | Check | Result |
|---|-------|--------|
| 1 | Research gate ran, >=5 sources in full, 3-variant queries, recency scan, gate_passed:true | PASS -- `phase-8.1-research-brief.md` present (mtime 00:56) |
| 2 | Contract pre-commit (contract mtime < experiment-results mtime) | PASS -- contract 00:57 < experiment-results 00:58 |
| 3 | Experiment results verbatim, 11 tests enumerated | PASS -- pytest block lists all 11 test ids |
| 4 | Log-last: harness_log.md last entry NOT yet phase-8.1 | PASS -- latest block is phase-7 closure (00:52 UTC); log append deferred to post-Q/A |
| 5 | First Q/A on 8.1 (not verdict-shopping) | PASS -- no prior `phase-8.1-evaluator-critique.md` |

## Deterministic checks (A-F)

| Check | Command | Result |
|-------|---------|--------|
| A. Syntax | `python -c "import ast; ast.parse(open('backend/models/timesfm_client.py').read())"` | exit 0 -- SYNTAX OK |
| B. TimesFM tests | `python -m pytest tests/models/test_timesfm_client.py -v` | exit 0 -- 11 passed in 1.99s |
| C. Backend regression | `pytest backend/tests/ -q --ignore=test_paper_trading_v2.py` | exit 0 -- 152 passed, 1 skipped |
| D. File existence | `ls backend/models/{__init__,timesfm_client}.py tests/models/{__init__,test_timesfm_client}.py` | all present |
| E. ASCII decode | `open(...).read().encode('ascii')` | ASCII OK |
| F. Scope | `git status --porcelain` | `backend/models/` is new `??`; no `M ` on existing files attributable to phase-8.1 |

Per-test breakdown (B): default+custom init, empty-series fail-open,
zero-horizon fail-open, timesfm-absent fail-open, stub-model single
happy path, batch-empty, batch-per-ticker fail-open, stub-model batch
happy path, shadow_log fail-open on bad project, module-ASCII-only.

## LLM judgment

### Contract alignment

Contract immutable criteria (verbatim from masterplan.json 8.1):

- [x] `TimesFMClient` class with lazy-load, fail-open on ImportError
  or load failure -- confirmed via `_get_model` at line ~55 returning
  None on any exception; empty outputs downstream.
- [x] `forecast(ts, horizon)` + `forecast_batch(tickers, horizon)`
  matching TimesFM 2.5 API `model.forecast(horizon=N, inputs=[...])`
  returning `(point, quantile)` tuple -- grep in client confirms.
- [x] `shadow_log(ticker, as_of_date, horizon, forecast_values,
  observed_values)` writing to
  `pyfinagent_data.ts_forecast_shadow_log`, fail-open -- line 149;
  no `CREATE TABLE` DDL (table-creation deferred to 8.3 per scope).
- [x] Model name pinned to `google/timesfm-2.5-200m-pytorch` (Sept
  2025 release) -- line 25 `_MODEL_NAME` matches.
- [x] Lazy imports: `timesfm` and `numpy` BOTH inside method bodies
  -- confirmed at lines 60 (`import timesfm` inside `_get_model`),
  97 (`import numpy as np` inside `forecast`), 121 (`import numpy
  as np` inside `forecast_batch`). No top-level occurrences.
- [x] Tests never load a real TimesFM model -- stub pattern via
  monkeypatch on `_get_model`; `test_forecast_without_timesfm_installed`
  exercises the ImportError path explicitly.
- [x] `forecast` returns `list[float]` per docstring -- return type
  and stub-model tests both assert plain Python floats.
- [x] No top-level network or env-var reads -- module import has
  no side effects beyond constant binding.

### Mutation-resistance

The test suite exercises several negative paths (timesfm-absent,
empty input, zero horizon, model=None in batch, bad BQ project for
shadow_log). These are real mutation-resistant checks: flipping the
fail-open guards to raise would break `test_forecast_without_timesfm_installed_returns_empty`,
`test_forecast_batch_without_model_returns_empty_per_ticker`, and
`test_shadow_log_fail_open_no_bq`. Not rubber-stamped.

### Scope honesty

Experiment results correctly scope 8.1 as shadow-logging pilot only
(no table DDL, no feature consumption, no model download during
tests). Deferrals to 8.2/8.3 stated explicitly. No overclaim.

### Research-gate compliance

`phase-8.1-research-brief.md` exists with 7 sources read in full
(per audit item 1 expectation), recency scan performed, three-
variant queries visible. Contract's references section cites the
brief. Pass.

## Violated criteria

None.

## Follow-ups for phase-8.2

(Informational, not blockers for 8.1 PASS.)

- 8.2 should add the `ts_forecast_shadow_log` BQ table CREATE
  migration in `scripts/migrations/` so shadow_log writes land
  somewhere real.
- Consider a smoke integration test (marked `slow`, skipped by
  default) that actually loads the 200M model once per week on
  a self-hosted runner, to catch HuggingFace model-name drift.

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "run_id": "qa_81_v1",
  "reason": "All 8 contract criteria met. Deterministic A-F pass: syntax OK, 11/11 TimesFM tests, 152/1 backend regression, files present, ASCII clean, scope = 4 new files only. Lazy imports confirmed at lines 60/97/121. Model pinned to timesfm-2.5-200m-pytorch. shadow_log fail-open verified. Research gate passed with 7 in-full sources.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5",
    "syntax",
    "timesfm_tests_11",
    "backend_regression_152",
    "file_existence",
    "ascii_decode",
    "scope_git_status",
    "lazy_import_grep",
    "model_name_pin",
    "shadow_log_no_ddl",
    "contract_alignment",
    "mutation_resistance",
    "scope_honesty",
    "research_gate"
  ]
}
```
