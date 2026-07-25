---
name: test-that-cleans-its-own-fixture-is-inert
description: A test that performs the module's own cleaning step before calling the module asserts on its own arithmetic and cannot detect the defect it names
metadata:
  type: feedback
---

When a test pre-processes the fixture with the SAME transformation the module
under test is supposed to apply, then calls only the downstream seam, its
headline assertion is true by construction. Check this by asking: *which line
performed the cleaning — the test or the module?*

**Why:** phase-80.31 cycle 3 shipped
`test_malformed_session_is_absent_from_every_array`, whose docstring says it is
"the original defect stated positively". It did
`cleaned = frame.dropna(subset=[...])` itself, then asserted the malformed
volume was absent from `_aligned_ohlcv_arrays(cleaned)`. Measured: under
`A1_PERCOLUMN` (the original defect restored — production's volume array really
does carry the malformed bar, `len(close)=119 len(volume)=120`) the test
**passes**. Vacuity shapes #7 (re-implemented) + #4 (tautology).

**How to apply:** this is a distinct tell from the wrong-key and empty-iteration
shapes — the assertion runs, on real data, and still cannot fail. The fix is
always the same: drive the ENTRY POINT with the RAW fixture and observe what the
module built (spy on the extractor's return), so the module's own cleaning step
is the thing under test. Pairs with [[mutate-the-flag-read-not-just-the-guard]]
(test the production read, not the seam you monkeypatched).
