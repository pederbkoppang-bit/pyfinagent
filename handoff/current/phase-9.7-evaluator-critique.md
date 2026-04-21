# Q/A Critique — phase-9.7 (weekly data integrity) — REMEDIATION v1

**Verdict: PASS**  **qa_id:** qa_97_remediation_v1  **Date:** 2026-04-20

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "qa_id": "qa_97_remediation_v1",
  "violated_criteria": [],
  "checks_run": ["ast_parse", "pytest_3of3", "handoff_files_present", "spot_read_lines_49_51_52", "mutation_analysis", "contract_before_results_mtime", "verbatim_verification_quote", "log_last_rule", "researcher_gate_passed", "cross_phase_carry_disclosure", "url_plausibility"],
  "reason": "Deterministic: ast.parse OK, pytest 3/3, handoff files present (contract 18:41 < results 18:42). Lines 51-52 match brief drift-detection logic; line 49 zero-guard present. 5-item audit clean: 6 in full, 16 URLs, recency, three-variant, gate_passed=true. Cross-phase carry to phase-9.9 (scheduler.py:374 empty-dict wiring) disclosed in contract + results + brief. Other carry-forwards defensibly deferred."
}
```

## Protocol audit (5/5 PASS)

1. Researcher: 6 read-in-full, 16 URLs, three-variant, recency, gate_passed=true.
2. Contract mtime (18:41) < results mtime (18:42).
3. Verbatim verification quote matches contract criterion.
4. No 9.7 log append yet.
5. Cycle v1 qa_id.

## Deterministic reproduction

| Check | Result |
|---|---|
| ast.parse | exit 0 |
| pytest -q | 3 passed, exit 0 |
| Handoff files | all 3 present |
| Line 49 zero-guard + lines 51-52 drift logic | match brief |
| Mutation: remove `prev_n == 0` guard | `test_missing_prior_baseline_skipped` stays green (uses empty dict path); minor coverage gap, non-blocker |

## LLM judgment

Cross-phase carry to phase-9.9 (`scheduler.py:374` empty-dict wiring) disclosed in 3 places: contract line 16, results line 31, brief lines 97 + 131. Scope-honest.

Other carry-forwards defensibly deferred: direction-blind drift severity, rolling 4-week baseline (Acceldata 2025), schema/null-rate/freshness checks. All cited.

URLs real (oneuptime, montecarlodata, acceldata, synq, thedataletter).

## Q/A observations for phase-9.9 remediation

1. **Top contract item:** fix `scheduler.py:374` `add_job` to pass `current_counts`/`prior_counts` — otherwise job runs weekly with empty dicts in production, never alerts.
2. Consider adding a `test_zero_prior_baseline_skipped` fourth test to cover the `prev_n == 0` branch at line 49.

Cleared for log append + masterplan status confirmation.
