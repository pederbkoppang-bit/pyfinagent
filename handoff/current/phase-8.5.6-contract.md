# Sprint Contract — phase-8.5 / 8.5.6 REMEDIATION v1 (full-breach)

Fresh Researcher + Fresh Q/A.

## Research findings (substantive)

5 sources in full. **Three SUBSTANTIVE design concerns surfaced:**

1. **5-day shadow window is too short** vs industry standard 30-90 days (Alpaca/3commas 2025).
2. **DSR=0.6 → 20% notional** is lenient; operator may want floor raised to 0.7.
3. **Kill-switch semantics ambiguous** — `current_dd` caller-provided; should be peak-to-trough rolling, not single-bar. Docstring at promoter.py:40 needs clarification.

All 3 immutable success_criteria still PASS literally. Advisories carry forward to a future hardening cycle.

## Immutable
`python scripts/harness/autoresearch_promotion_test.py` exit 0 + 3/3 PASS.
