# Q/A Critique — phase-10.8 (Slot accounting to harness_learning_log)

**qa_id:** qa_108_v2
**Date:** 2026-04-20
**Cycle:** v2 (cycle-2 on updated evidence after qa_108_v1 CONDITIONAL)
**Supersedes:** qa_108_v1 (in this same file; v1 preserved in summary below)

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "qa_id": "qa_108_v2",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "syntax_ast_parse",
    "handoff_files_present",
    "source_read_line_101",
    "test_present_line_100",
    "cli_harness_4_of_4",
    "pytest_slot_accounting_10_of_10",
    "pytest_neighbors_108_of_108",
    "mutation_M3_injected",
    "mutation_M3_detected_by_new_test",
    "source_restored_pyc_cleared",
    "post_restore_pytest_10_of_10",
    "post_restore_sql_verified"
  ],
  "reason": "Cycle-2 evidence closes the v1 gap. The new test test_invariant_sql_pins_slot_id_filter captures the literal SQL string via a stub and asserts on substrings. Under M3 mutation (adding 'monthly_gate' to the IN clause in slot_accounting.py:101), the new test FAILS with a clear AssertionError on the SQL text. After restoration, 10/10 + 4/4 + 108/108 all green. Anti-rubber-stamp requirement now satisfied."
}
```

## Deterministic checks — results

| # | Check | Result |
|---|-------|--------|
| A | `ast.parse` on `slot_accounting.py` + `test_slot_accounting.py` | exit 0 |
| B | Handoff files present (contract, research-brief, experiment-results, critique) | all 4 present |
| C | Prod SQL at `slot_accounting.py:101` = `"AND slot_id IN ('thu_batch', 'fri_promotion')"` | confirmed |
| D | New test `test_invariant_sql_pins_slot_id_filter` at `test_slot_accounting.py:100` | present, 23 lines, inspects `captured["sql"]` substrings |
| E | CLI harness `scripts/harness/phase10_slot_accounting_test.py` | 4/4 PASS |
| F | `pytest tests/autoresearch/test_slot_accounting.py -q` | 10 passed |
| G | `pytest tests/autoresearch/ tests/slack_bot/ backend/metrics/ -q` | 108 passed |
| H | Mutation M3: change line 101 IN clause to include `'monthly_gate'` | test FAILED as required (assertion on SQL substring) |
| I | Restore line 101 + clear `__pycache__` | prod SQL back to canonical form |
| J | Post-restore `pytest ... -q` | 10/10 green |

Mutation M3 failure output (excerpt, from the actual run Q/A performed):

```
>       assert "IN ('thu_batch', 'fri_promotion')" in captured["sql"]
E       assert "IN ('thu_batch', 'fri_promotion')" in
E         "... AND slot_id IN ('thu_batch', 'fri_promotion', 'monthly_gate')"
tests/autoresearch/test_slot_accounting.py:116: AssertionError
```

## LLM-judgment leg

### Harness-compliance audit (5 items, per feedback_qa_harness_compliance_first.md)

1. **Researcher spawned before contract**: research-brief present at
   `handoff/current/phase-10.8-research-brief.md`. ✓
2. **Contract pre-commit (written before GENERATE)**: contract has
   immutable success criteria; experiment-results has cycle-2 patch
   section appended, not overwritten. ✓
3. **Results present and verbatim**: experiment-results includes CLI
   output and pytest output verbatim. ✓
4. **Log-last discipline**: `handoff/harness_log.md` append should
   occur AFTER this PASS and BEFORE the masterplan status flip.
   Reminder to Main: do not bundle the status flip before the log.
5. **No verdict-shopping**: this is a fresh Q/A on **updated
   evidence** (new test added, mutation independently verified,
   .pyc cleared). That is the documented canonical cycle-2 flow,
   not second-opinion-shopping. ✓

### Contract alignment

Contract's immutable criteria for phase-10.8 (slot accounting to
harness_learning_log) are met in cycle-2:
- Rows are tagged phase-10 and land in `pyfinagent_data.harness_learning_log`
- Weekly invariant `thu_batch + fri_promotion == 2` enforced via SQL
- Fail-open on BQ errors (log warning, return sum=0)
- CLI harness covers all four assertions
- **NEW**: SQL text is now pinned by a test that inspects the literal
  string, closing the v1 anti-rubber-stamp gap.

### Anti-rubber-stamp (v1's blocker) — now closed

v1 flagged that `bq_query_fn` stubs duplicated the Python-side filter
logic, so a production-SQL mutation (M3) was invisible. Q/A
re-ran M3 in cycle-2 and confirmed:
- Without the new test, M3 was silent (v1 finding reproduced).
- With the new test, M3 triggers a test failure with an assertion on
  the SQL substring itself.
- Production source was restored and 10/10 re-verified.

### Scope honesty

Experiment-results' cycle-2 patch section cleanly discloses:
- What was added (one test, 23 lines)
- The exact mutation re-run procedure
- Test count delta (9 → 10)
- No over-claiming beyond the SQL-pinning property

### Research-gate compliance

Contract references research-brief. Brief exists. Sources section
present. Not re-audited for depth in cycle-2 because v1 already
cleared the gate on research grounds; v1's blocker was code-side only.

## Verdict

**PASS**. The v1 gap is closed with a targeted test that asserts on
the SQL string itself. Mutation M3 is now caught. All deterministic
checks green post-restoration. Main may proceed to:

1. Append the cycle entry to `handoff/harness_log.md` (log-last).
2. Flip `.claude/masterplan.json` phase-10.8 → `status: done`.
3. Commit.

## Appendix — v1 summary (superseded)

qa_108_v1 verdict was CONDITIONAL with a single violation of type
`Unjustified_Inference`: `verify_weekly_invariant` SQL construction
was not exercised by any test because stubs emulated the filter in
Python. v1's required remediation: add a test that inspects the
literal SQL string. v2 confirms that remediation landed and is
mutation-resistant.
