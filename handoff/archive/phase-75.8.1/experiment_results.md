# Experiment results — Step 75.8.1 (shared stub-fingerprint + dry-run-label guard for the SECOND gauntlet-report consumer)

Date: 2026-07-24. Execution model: opus-tagged P1 (latent) money-integrity step →
Main (Fable 5) GENERATE; Researcher gate opus/max (wf_3d29e4a9-bd7, PASSED, 7
read-in-full — reward-hacking canon: ImpossibleBench / SpecBench /
LLMs-Gaming-Verifiers + SSOT refactoring canon).

## What was built

1. **New `backend/backtest/gauntlet/report_integrity.py`** — the SINGLE shared
   implementation (pure leaf, imports only `typing`): `is_dry_run_report()`,
   `has_stub_fingerprint()` (skipped-filter + `bool(non_skipped) and` empty-guard),
   `check_report_integrity() -> (ok, reason|None)` with **fingerprint checked FIRST**
   (ordering pinned by the pre-existing test on a report that is both stub and
   dry_run:true) and the fingerprint reason string **byte-identical** to the 75.8
   inline original.
2. **`scripts/risk/promotion_gate.py`** refactored onto the shared module: the 75.8
   inline fingerprint block (:118-141) replaced by a module-attr
   `report_integrity.check_report_integrity(report)` call — and thereby GAINS the
   dry_run-label refusal it never had (the step-text correction from the research
   gate: its `dry_run` refs were the CLI's own --dry-run flag, not a label check).
3. **`backend/autonomous_harness.py::promote_strategy`** — the previously-blind
   second consumer — gains the integrity gate after report load, before
   `evaluate()`: on refusal, `_append_blocklist(strategy, reason)` +
   `raise PromotionBlocked` (parity with the two existing refusal sites; the reason
   is logged AND raised — no exception swallowed into a promote).
4. **New `backend/tests/test_phase_75_8_1_harness_consumer.py`** (11 tests, offline):
   stub-refused+blocklisted through promote; dry_run:true DIVERGENT refused through
   BOTH consumers (proves the label leg, not the fingerprint, catches it); realistic
   divergent dry_run:false PROMOTES (overall_pass True, no blocklist); empty +
   all-skipped NOT fingerprinted through BOTH consumers (the all([]) trap);
   anti-fixture-divorce (REAL `gauntlet.run(dry_run=True)` bytes refused through
   promote); the C2 single-implementation proof BY BEHAVIOR (one monkeypatch of the
   shared predicate flips BOTH consumers) plus its non-vacuity self-test.

## Files changed (exactly the contract's 4 + masterplan status)

`backend/backtest/gauntlet/report_integrity.py` (new),
`scripts/risk/promotion_gate.py`, `backend/autonomous_harness.py`,
`backend/tests/test_phase_75_8_1_harness_consumer.py` (new),
`.claude/masterplan.json` (75.8.1 → in_progress), handoff artifacts.
`test_phase_75_promotion_gate.py` BYTE-UNTOUCHED; `git diff` on evaluator.py +
limits.yaml = 0 lines (C5, verbatim in live_check).

## Verbatim verification output

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_promotion_gate.py backend/tests/test_phase_75_8_1_harness_consumer.py -q
...............................                                          [100%]
31 passed in 0.15s
```

(20 pre-existing tests unchanged-and-green — the step text's "14" was the
2026-07-23 count; measured today the suite has 20 — plus 11 new.)

Lint: new files `All checks passed!`; edited files' finding-class census diffed
against `git show HEAD:` baselines — IDENTICAL (zero new findings).

## Mutation matrix (C4 + qa.md §4c) — 7 mutations, 7 killed, 0 survivors

Runner + verbatim log in scratchpad (`run_mutations_75_8_1.py`,
`mutation_matrix_75_8_1.txt`); post-restore sanity exit 0.

| # | Mutation (executed) | Killed by (consumer coverage) |
|---|---|---|
| G1 | integrity call dropped in promotion_gate | 4 failures THROUGH CONSUMER 1 (incl. two pre-existing 75.8 tests) |
| G2 | integrity call dropped in promote_strategy | 4 failures THROUGH CONSUMER 2 |
| G3 | dry_run-label branch dropped in the shared predicate | label tests fail through BOTH consumers |
| G4 | skipped-filter dropped | all-skipped tests fail through BOTH consumers (+ the pre-existing 75.8 pin) |
| G5 | empty-list guard dropped (all([]) trap restored) | 5 failures: all-skipped AND empty params through BOTH consumers |
| G6 | **STUB**: fingerprint leg disabled in the shared predicate | 4 failures — and demonstrated defense-in-depth: the dry_run leg caught the real stub, and the reason-string pin still failed the test |
| G7 | **FIXTURE**: REALISTIC fixture made stub-shaped | the promotes-test + the probe non-vacuity self-test fail |

## Notes for Q/A

- Blocklist-parity choice (refusal also blocklists for 30 days, mirroring the two
  existing refusal sites) was flagged in the contract; raise-only would also satisfy
  C1 — the chosen shape is the consistent one.
- The label refusal makes NOTHING promotable until a live gauntlet exists (the
  writer only emits dry_run:true today; live mode raises NotImplementedError) —
  exactly the intended fail-safe posture for a latent P1; the only current caller
  is the phase4_9 redteam script, which expects refusals.
- gauntlet/__init__.py deliberately untouched (consumers import the submodule).
