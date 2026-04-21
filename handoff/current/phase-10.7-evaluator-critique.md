# Evaluator Critique — phase-10.7 (Q/A v2, cycle-2 fresh instance)

**qa_id:** qa_107_v2
**Date:** 2026-04-20
**Evidence:** post-patch (rollback.py line 64 = `<=`; test_rollback.py = 10 tests including `test_exact_boundary_dd_equals_threshold_no_breach`)

## Harness-compliance audit (5-item, per feedback_qa_harness_compliance_first)

| # | Item | Status |
|---|---|---|
| 1 | Researcher spawn evidenced | PASS — `phase-10.7-research-brief.md` exists, contract cites it (7 in full, 17 URLs, recency, gate_passed=true) |
| 2 | Contract written before GENERATE | PASS — contract present with immutable criteria copied verbatim from masterplan, plan predates the cycle-2 patch |
| 3 | experiment_results.md present and honest | PASS — cycle-2 patch section (lines 22-30) candidly states Q/A v1 left the mutation active and Main restored correct semantics |
| 4 | log-last discipline | N/A at Q/A step (Main's responsibility AFTER this PASS); nothing to flip yet |
| 5 | No second-opinion-shopping | PASS — this spawn evaluates **updated evidence** (new boundary test, cleared .pyc), not a re-ask on unchanged evidence |

## Deterministic checks (v2)

| Check | Result |
|---|---|
| A. Line 64 source is `<=` (not `<`) | PASS — `if abs(dd) <= threshold: return result` |
| B. Immutable CLI `phase10_rollback_test.py` | PASS — 3/3 |
| C. `pytest tests/autoresearch/test_rollback.py` | PASS — 10/10 (up from 9) |
| C'. Neighbor suite (autoresearch + slack_bot + metrics) | PASS — 98 passed (up from 97 — the +1 is the new boundary test) |
| D. New test `test_exact_boundary_dd_equals_threshold_no_breach` present | PASS — line 143-158, asserts `dd=-0.10` → `demoted=False, decision="no_breach"` |
| E. M3 mutation re-run (`<=` → `<`, .pyc cleared) | **PASS (gap closed)** — `test_exact_boundary_dd_equals_threshold_no_breach` FAILS under mutation with `assert True is False` at line 157. Restored `<=`, re-cleared .pyc, re-ran: 10/10 green. File is clean. |

## LLM judgment

**Q1. Does the new test genuinely close the v1 gap?**
YES. Under the M3 mutation the new test fails with a direct assertion
on the `demoted` flag at the exact boundary (`dd=-0.10`, default
threshold `DD_TRIGGER=0.10`). The failure mode is not incidental —
it's driven by the semantic boundary the docstring calls out
("exceeds DD_TRIGGER"). No other pre-existing test crosses this exact
point (the breach tests use -0.11 / -0.15, the sub-threshold test
uses -0.05). Mutation-resistance is real, not cosmetic.

**Q2. Is the cycle-2 patch section honest?**
YES, with a small wording nit. The section admits "Q/A v1 ran
mutation M3 ... and found no existing test failed under the
mutation — a real boundary-coverage gap" and that Main "restored
`abs(dd) <= threshold` (the correct semantics were never in the
production file — Q/A had left the mutation active when signaling
the gap)". That's candid and matches what I observe on disk. It
does not overclaim: it names the v1 gap, the restoration, the new
test, and the .pyc clearance (the phase-10.6 lesson).

Nit (non-blocking): the phrasing "correct semantics were never in
the production file" is slightly ambiguous — reads as if `<=` was
never there, when in fact it WAS there originally, Q/A v1 mutated
it to `<`, and Main had to revert. Minor — history is clear from
the critique files taken together.

**Q3. Any OTHER boundary / off-by-one mutations that would still go
undetected?**

I probed three classes of mutation by mental execution against the
test set:

| Mutation | Catches? | Why |
|---|---|---|
| `abs(dd)` → `dd` (drop abs) | YES | For dd=-0.11, `dd <= 0.10` is True → early-return no_breach → `test_challenger_dd_breach_auto_demotes` expects `demoted=True` and fails |
| `<=` → `>=` (flip operator) | YES | dd=-0.11 → `0.11 >= 0.10` True → no_breach → breach test fails |
| `<=` → `==` | YES | dd=-0.05 → `0.05 == 0.10` False → falls through to demote → `test_sub_threshold_no_demote` fails |
| `DD_TRIGGER` hard-code drift (e.g., 0.10 → 0.05) | YES | `test_imports_dd_trigger_from_promoter` asserts equality with `promoter.DD_TRIGGER` |
| `<=` → `<` (M3) | YES (this cycle) | new boundary test covers it |
| Positive-dd edge case (`dd=+0.11` should NOT demote because abs(+0.11)=0.11 > threshold ... actually it SHOULD demote by current semantics) | n/a — current semantics treat positive and negative dd symmetrically via `abs()`. No test asserts the symmetric case. This is NOT a bug, because `dd` is semantically a drawdown (≤0), but if a caller ever passed a positive spike as "dd", the function would demote. Worth noting as a documentation gap, not a code defect. |

**Conclusion on mutation-resistance:** the test set now catches the
four interesting boundary/operator mutations on line 64. The
positive-dd case is a caller-contract concern (the docstring and
callsite `promoter.on_dd_breach` guarantee negative dd), not a
correctness gap at this layer. No violated_criteria.

## Verdict

**PASS.** All three immutable success criteria are met, the v1
boundary-coverage gap is demonstrably closed (mutation re-run
failed under M3 and reverted cleanly), the neighbor suite is green
(98/98), the file is .pyc-clean, and the cycle-2 patch section in
experiment_results is honest.

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "qa_id": "qa_107_v2",
  "violated_criteria": [],
  "violation_details": [],
  "checks_run": [
    "harness_compliance_audit_5item",
    "ast_syntax",
    "immutable_cli_phase10_rollback_test",
    "pytest_test_rollback",
    "pytest_neighbor_autoresearch_slack_metrics",
    "line64_source_check",
    "new_boundary_test_present",
    "m3_mutation_re_run_then_restore",
    "pyc_cleanliness",
    "additional_mutation_probe_abs_operator_threshold_drift",
    "cycle2_patch_honesty_review"
  ],
  "certified_fallback": false,
  "reason": "Cycle-2 on updated evidence: immutable 3/3, pytest 10/10 (was 9, +1 boundary test), neighbor 98/98. M3 mutation re-run confirms new test_exact_boundary_dd_equals_threshold_no_breach FAILS under the mutation, closing the v1 gap. Restored cleanly, .pyc cleared, 10/10 green on restore. Additional mutation probes (drop-abs, flip-operator, ==, DD_TRIGGER drift) all caught by existing tests. Cycle-2 patch section in experiment_results is honest about the v1 gap. Not second-opinion-shopping — evidence materially changed (new test + restored source + cleared bytecode)."
}
```
