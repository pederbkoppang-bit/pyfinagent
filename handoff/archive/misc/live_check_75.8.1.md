# live_check_75.8.1 — shared report-integrity guard for the second gauntlet consumer

All output verbatim from live runs 2026-07-24. Offline-only step (P1 latent — no
production caller of promote_strategy; no UI surface; no flag-gated live-loop behavior).

## 1. Verification command (immutable) — exit 0

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_promotion_gate.py backend/tests/test_phase_75_8_1_harness_consumer.py -q
...............................                                          [100%]
31 passed in 0.15s
```

20 pre-existing promotion-gate tests pass UNCHANGED (the file is byte-untouched — see §3;
note the step text's "14" was the 2026-07-23 count, the suite has since grown to 20) +
11 new consumer tests.

## 2. C4 mutation matrix — 7 mutations, 7 killed, 0 survivors

Runner `run_mutations_75_8_1.py`, verbatim log `mutation_matrix_75_8_1.txt` (scratchpad);
full table in experiment_results.md. Summary verbatim:

```
SUMMARY: 7 mutations, 7 killed, survivors: NONE
=== post-restore sanity: pytest exit 0 ===
```

Consumer coverage per C4: G1 (call dropped in promotion_gate) kills through consumer 1;
G2 (call dropped in promote_strategy) kills through consumer 2; G3 (dry_run branch),
G4 (skipped-filter), G5 (empty-guard) each kill through BOTH consumers; G6 stubs the
predicate (§4c stub mutation — and incidentally demonstrated defense-in-depth: the
dry_run leg caught the real stub when the fingerprint leg was disabled, and the
reason-string pin still failed the test); G7 mutates the REALISTIC fixture itself.

## 3. C5 boundary proof — zero edits to thresholds/kill-switch/constants/limits

```
$ git diff backend/backtest/gauntlet/evaluator.py backend/governance/limits.yaml | wc -l
0
$ git diff --stat   (step files only)
 backend/autonomous_harness.py                        (integrity gate before evaluate())
 backend/backtest/gauntlet/report_integrity.py        (new shared pure-leaf module)
 scripts/risk/promotion_gate.py                       (inline 75.8 block -> shared call)
 backend/tests/test_phase_75_8_1_harness_consumer.py  (new, 11 tests)
 .claude/masterplan.json                              (75.8.1 -> in_progress)
```

`backend/tests/test_phase_75_promotion_gate.py` byte-untouched (absent from git diff).
Kill-switch code and DSR/PBO constants: no file containing them is in the diff.

## 4. Lint (scope derived from git after last edit)

New files (`report_integrity.py`, new test): `All checks passed!`. Edited files:
finding-CLASS census diffed against their `git show HEAD:` baselines — IDENTICAL for
both `scripts/risk/promotion_gate.py` and `backend/autonomous_harness.py` (zero new
findings introduced; the pre-existing classes are the big-file legacy set).

## 5. Single-implementation proof (C2) — behavioral, not source-scan

`test_monkeypatching_shared_predicate_flips_both_consumers`: one monkeypatch of
`report_integrity.check_report_integrity` makes promote_strategy REFUSE a realistic
report AND promotion_gate block the same report (rc==1, forced reason in output).
Its stub self-test (`test_shared_predicate_probe_is_not_vacuous`) proves the same
report passes BOTH consumers without the monkeypatch — the flip is caused by the
shared predicate alone.
