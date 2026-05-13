# Live-check placeholder -- phase-25.F

**Step:** 25.F -- Byte-identical regression tests for aliasing detection
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "pytest tests/test_signal_attribution.py passes both regression tests"

## Pre-deployment evidence
- 4/4 verifier PASS.
- Direct pytest invocation also passes:
  `PYTHONPATH=. .venv/bin/python -m pytest tests/services/test_signal_attribution.py -k "test_lite_path_byte_identical_flagged or test_full_path_distinct_rationale" -v`
  -> 2 passed in 0.01s.
- Tests assert the canonical 4-key shape (`{agent, role, rationale, weight}`)
  so any future cosmetic field (lite_path or analogue) gets caught at unit
  test time.

## Post-deployment operator workflow
1. Pull main:
   ```
   git pull origin main
   ```
2. Run the full test_signal_attribution suite:
   ```
   source .venv/bin/activate
   PYTHONPATH=. python -m pytest tests/services/test_signal_attribution.py -v
   ```
   Expected: 22 passed (was 20 before this cycle).

## Closes audit basis
bucket 24.4 F-6 RESOLVED. The 25.B aliasing-detection removal is now
locked by regression tests.

**Audit anchor for next bucket:** 25.C (Layer-1 28-skill output surfacing),
25.D / 25.L (P2 backlog).
