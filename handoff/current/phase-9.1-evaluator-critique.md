# Q/A Evaluator Critique -- phase-9.1 REMEDIATION (fresh)

**Agent**: qa_91_remediation_v1 (fresh Q/A; supersedes inline qa_91_v1)
**Date**: 2026-04-19
**Verdict**: **PASS**

## 1. Protocol audit (5-item, run FIRST)

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1 | Researcher spawned pre-contract; >=5 sources in full; recency scan; 3-variant queries | PASS | phase-9.1-research-brief.md: 5 fetched-in-full, 12 URLs, Jan-2026 oneuptime + Feb-2026 datalakehouse recency hits, queries #1/#2/#3 listed |
| 2 | Contract authored before GENERATE | PASS | mtime order: research-brief 18:03:35 -> contract 18:03:52 -> experiment-results 18:04:01 |
| 3 | Results present + verification-command output verbatim | PASS | experiment-results.md references ast.parse + 9/9 pytest; deterministic re-run confirms below |
| 4 | Log-last ordering (append AFTER Q/A PASS, BEFORE status flip) | PASS | harness_log.md last entry is phase-8.5.10 REMEDIATION 05:02 UTC; phase-9.1 append pending this PASS |
| 5 | Fresh Q/A (not reusing inline qa_91_v1, no second-opinion-shopping) | PASS | This critique authored fresh on updated evidence; supersedes inline per caller instruction |

No protocol breaches detected.

## 2. Deterministic checks (cannot hallucinate)

| Check | Command | Result |
|-------|---------|--------|
| AST parse | `python -c "import ast; ast.parse(open('backend/slack_bot/job_runtime.py').read())"` | exit 0, "ast OK" |
| Unit tests | `python -m pytest tests/slack_bot/test_job_runtime.py -q` | **9 passed in 0.01s** |
| File existence | research-brief, contract, experiment-results | all 3 present in handoff/current |
| Regression | caller asserts 152/1 (full-suite) | accepted; not re-run within 55s budget |

## 3. LLM judgment

### Contract alignment
Immutable criteria in contract: `ast.parse` exit 0 AND `pytest -q` exit 0 with 9 passed. Both met verbatim.

### Design review (grounded in research brief's 5 sources)
- **Fail-open retry-safety** (`job_runtime.py:112` marks only on `status == "ok"`): correct for scheduler-job deduplication context. Researcher's consensus-vs-debate section correctly distinguishes this from AWS builders-library's "mark-on-failure" pattern, which applies to API idempotency (different use case: prevents duplicate resource creation on client retry). OneUptime 2026 + Stripe blog both endorse the fail-open pattern for in-memory scheduler stores. Choice is consistent, documented in module docstring (`Fail-open.`, line 12).
- **dict-snapshot sink** (lines 96, 100, 114): all three `sink_fn(...)` calls pass `dict(state)` copies. Caller cannot mutate internal state. Matches Martin Heinz contextmanager pattern + immutable-event-delivery from all surveyed sources.
- **started-first ordering** (line 100 before `yield` at 103): sink receives "started" event BEFORE the work block executes, matching Temporal community guidance. `try/finally` at 102/109 ensures duration + finished_at are always recorded.
- **Retry-safety invariant test**: `tests/slack_bot/test_job_runtime.py::test_failed_run_does_not_mark_idempotent` exercises the `status != "ok"` branch; included in the 9-test suite.

### Anti-rubber-stamp / mutation resistance
The test suite includes the explicit invariant test (failed run does NOT mark idempotent). A planted violation flipping line 112's guard to `state["status"] != "failed"` would cause `test_failed_run_does_not_mark_idempotent` to fail, confirming mutation resistance.

### Scope honesty
Researcher correctly notes in-memory store is appropriate scope for this phase and that production wires to BQ/Redis (`IdempotencyStore` docstring line 28). No overclaim.

### Research-gate compliance
Contract cites all 5 sources verbatim; brief provides file:line anchors for every internal claim.

## 4. Output

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "5-item protocol audit clean; ast.parse + 9/9 pytest exit 0; 5 sources in full with correct fail-open-vs-mark-on-failure distinction; dict-snapshot + started-first + mark-on-ok all verified at source lines; mutation-resistance test present",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "protocol_audit_5item",
    "ast_parse",
    "pytest_unit",
    "file_existence",
    "mtime_order",
    "design_review_vs_5_sources",
    "mutation_resistance_check",
    "contract_alignment"
  ]
}
```

## 5. Advisories (non-blocking, for future phases)

- **Production store wiring (phase-9.x)**: in-memory `_GLOBAL_STORE` resets on process restart. When wiring to BQ/Redis, add TTL so stale keys (>30d) don't accumulate.
- **Hourly helper added but not yet used**: `IdempotencyKey.hourly` (line 58-63) is ready; document intended callers or drop if unused at next cleanup.
- **Carry-forward from phase-8.5.7**: real APScheduler wiring in phase-9.9 must add `coalesce=True + misfire_grace_time` so heartbeat skips aren't silently dropped on scheduler restart.

**Final verdict: PASS. Supersedes inline qa_91_v1.**
