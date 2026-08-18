# Contract -- step 86.118

**Step:** the backend test suite has 18 PRE-EXISTING failing tests, so it cannot
detect a regression and every step's "suite is green" check is measuring a red
baseline. **P2, verification-integrity.**

## Research-gate summary (what the gate CHANGED about the plan)

Gate **PASSED** (`wf_628cc28c-e10`; **11 sources read in full**, **60 URLs**
against 60 distinct in the brief, audit-class **dry after 6 rounds with 2 dry**,
envelope `COMPLETE`, 17 internal files inspected; brief
`research_brief_86.118.md`, 42,052 chars).

**The gate refuted two of my premises and I re-verified both myself before
writing this.**

**1. My spawn prompt said "the 18 named failing FILES". It is 18 failing TESTS
across 12 files, and the full suite gives 19.** The scope statement was wrong in
a way that would have mis-sized the work.

**2. The ORDERING-ARTIFACT bucket is EMPTY inside this step's own scope.** The
gate reports zero of the 18 are order-dependent. **Measured independently**: I
ran all 19 full-suite failures one at a time --

```
FAILS_ALONE  (genuinely broken)       : 18
PASSES_ALONE (order/context-dependent):  1
  -> test_phase_86_6_subprocess_channel::test_the_optin_IS_honoured_...
```

The single order-dependent victim is the **19th**, outside the named 12 files.
So criterion 5 must be satisfied by reporting a **measured n=1 outside scope**,
not by hunting shared state among the 18 -- there is none to find.

**3. THE HEADLINE, and it is worse than "18 tests are red": log-scraping is
unsound in BOTH directions.** `23_2_6:265` asserts `count >= 1` and is RED,
while `23_2_13:136` asserts `count == 0` and **silently XPASSes**. Both strings
live only in `backend.log.20260612T104931Z.gz` (29927 / 56 occurrences), and
**29927 is the frozen number in the xfail reason**. `xfail_strict` is **not
configured anywhere** -- I confirmed that myself, and my own full-suite run
reports exactly **`1 xpassed`**. So the suite is not merely failing to detect
regressions; it is **actively hiding one test that has started passing**.

**4. Some of these are PRODUCT defects, not stale evidence.** The gate cites Luo
et al. (FSE'14): **24% of flaky-test fixes touch the code under test, and 94%
of those were real bugs**, and identifies rows 16-18 as driving product code.
The classification must therefore be able to return PRODUCT-DEFECT, and
criterion 3 already requires those be fixed or filed rather than edited away.

**5. Do not invent a fix pattern that already exists.** Four census tests pin
exact equality against a masterplan that has moved (319 commits/30d). The
**git-pin fix already exists in this repo at `75_17:81-85`** -- reuse it.

**6. A humility datapoint the gate volunteered.** TEBench (2026) measures agents
as **worst at stale-test repair, F1 35.8%**. That is this step's exact task, so
the plan leans on measurement per test rather than on pattern-matching.

**7. `23_2_6:259` has a real bug of its own**: it falls back to `archives[-1]`
(the NEWEST archive) when the evidence it wants is in the OLDEST. And both
`FileNotFoundError` tests point at files that **do exist** under
`handoff/archive/misc/`.

## Hypothesis

The 18 are dominated by **stale evidence** -- assertions that were true when
written and whose source has since rotated, been consumed, or moved -- with a
minority driving product code. None is order-dependent. Restoring green is
therefore mostly re-pointing assertions at evidence that still exists, plus a
small number of genuine fixes, and **any test made to pass by weakening it is a
worse outcome than leaving it red**.

## Immutable success criteria (copied verbatim from `.claude/masterplan.json`)

1. the failing set is RE-MEASURED by this step from a full-suite run rather than inherited from this audit_basis, with the exact pytest command and the counts stated, and run at least twice to separate deterministic failures from ordering-dependent ones
2. each failing test is classified with evidence into STALE-EVIDENCE (the assertion was true when written and its source has since rotated or been consumed), PRODUCT-DEFECT (the code is actually wrong), or ORDERING-ARTIFACT (passes in isolation), and the classification for each cites what was read or run to reach it
3. every test classified PRODUCT-DEFECT is either fixed or filed as its own numbered step -- a real defect is never closed by editing the test that found it
4. no test is made to pass by weakening it: no bulk xfail, no skip, no assertion deleted, no tolerance widened without a stated measured reason; any deletion is justified by showing the property is covered elsewhere
5. the ORDERING-ARTIFACT class is proven rather than asserted -- show the test passing in isolation AND failing in the full run, and identify the shared state responsible
6. after the work, a full-suite run is reported with its exact counts, and if the suite is still not green the remaining failures are named with their disposition rather than left as a residual total
7. mutation-test every new guard: revert it and show the check goes red, with the control observed GREEN first, the same test count collected in control and mutant, the NAMED test failing, and a byte-identical restore

**Immutable verification command:**
`bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/tests/conftest.py\").read()); print(\"parses\")"'`

**Immutable live_check:** `live_check_86.118.md` with the verbatim full-suite
output and counts from at least two runs, the per-test classification table with
its evidence, and the post-work full-suite counts.

## Plan

**P1 -- criterion 1, already begun.** Two full-suite runs recorded (**19 failed
/ 3635 passed**, twice). Because `pytest-randomly` is **absent** (filed as
**86.119**), the two runs share one fixed order, so run-to-run agreement does
NOT establish order-independence. That is why P2 uses per-test isolation
instead, and the artifact must say so rather than implying two runs proved it.

**P2 -- criterion 5, and it resolves NEGATIVE inside scope.** Per-test isolation
sweep, already run: 18 FAILS_ALONE / 1 PASSES_ALONE. Identify the shared state
behind the single victim. **Also record an anomaly I measured and have not
explained**: `test_portfolio_swap::test_swap_framework_fills_zero_buy_gap`
**passes** when the 19 are run together but **fails alone and in the full
suite** -- a masking dependency in the opposite direction. It is stated as an
open observation, not a conclusion.

**P3 -- criterion 2, one test at a time, evidence per row.** Classify each of
the 18. For log-scrapers, name the archive and the occurrence count. The
`xfail_strict` gap and the silent `1 xpassed` are reported here.

**P4 -- criterion 3.** PRODUCT-DEFECT rows fixed or filed as their own steps.
Given Luo's 24%/94%, expect a real minority; do not force everything into
STALE-EVIDENCE because that bucket is easier to close.

**P5 -- criterion 4, the discipline that decides this step's worth.** No bulk
xfail, no skip, no deleted assertion, no widened tolerance without a measured
reason. Reuse the git-pin at `75_17:81-85` for the census rows.

**P6 -- criterion 6.** Final full-suite run with exact counts; every remaining
failure named with a disposition.

**P7 -- criterion 7**, control GREEN first, same collected count, NAMED test
failing, SHA-256 restore. Per this session's R1: **every new guard is watched
going RED before it is shipped.**

## Scope honesty -- what this step does NOT do

- **It does not install `pytest-randomly`** -- that is **86.119**, and doing it
  here would make this step's before/after delta unreadable.
- **It does not fix the 19th test** (`test_phase_86_6_subprocess_channel`); it
  is outside the named 12 and is classified, measured and handed on.
- **It does not enable `xfail_strict`** as a side effect. The silent XPASS is a
  real finding, but flipping that flag suite-wide changes the outcome of every
  xfail in the repo and is its own operator-gated change.
- **It does not touch production behaviour** except where a row is classified
  PRODUCT-DEFECT and fixed under criterion 3, each named individually.
- **It does not make the suite green by weakening tests.** A smaller honest red
  count beats a green one that proves nothing.

## References

`research_brief_86.118.md` (the both-directions log-scraping finding, the
isolation census, Luo FSE'14, TEBench 2026, the `75_17:81-85` git-pin pattern);
**86.119** (`pytest-randomly` absent, so order-dependence is invisible);
**86.116** (where the 18 were discovered while attributing an unrelated
regression).
