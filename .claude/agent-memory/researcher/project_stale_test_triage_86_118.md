---
name: stale-test-triage-86-118
description: 86.118 findings -- log-scraping tests fail UNSOUND IN BOTH DIRECTIONS (same rotation reddens one, greens another); "18 failing files" was 18 tests in 12 files; ZERO order-dependent in scope while the only OD victim sits outside it
metadata:
  type: project
---

Step 86.118 (triage of long-lived failing tests, `backend/tests`). Measured
2026-08-18. Numbers drift; the MECHANISMS below do not.

**A log-scraping assertion cannot distinguish "the event stopped happening"
from "the evidence rotated away" -- and the two failure modes have opposite
SIGNS.** `test_phase_23_2_6_sector_cap_emit.py:265` asserts `count >= 1` and
goes RED. `test_phase_23_2_13_governance_watcher.py:136` asserts `count == 0`
and goes **XPASS**. Same file (`backend.log`), same cause, and only the red one
is ever triaged. Both scraped strings existed in exactly ONE archive
(`backend.log.20260612T104931Z.gz`: 29927 tick-failures, 56 "Skipping BUY");
every newer archive and the live log had **0 of both**. The `29927` quoted in
the xfail reason is a frozen quote from that dead window.

**Why:** an assertion over a rotating window is coupled to state the test does
not control (Huo & Clause's "brittle assertion"; the Mystery Guest smell).
**How to apply:** when a count-over-a-log test is red, ALWAYS grep the whole
archive set for the sibling assertion with the opposite polarity -- the green
one is the same defect and is invisible. Never "fix" a red one by widening the
threshold; that converts it into the vacuous-green twin.

**The rotation fallback reached for the WRONG archive.** `:259` used
`sorted(glob(...))[-1]` = lexicographically NEWEST; the evidence was in the
OLDEST, 7 archives back. Same newest-vs-oldest inversion as the kill-switch
archive merge (36.8). Its `else` branch `pytest.skip`s, so a rotation with no
archives makes the test vanish rather than fail -- three disease states in one
12-line block.

**Denominator + scope traps.** The prompt said "18 named failing files" and
listed 12; measured, it was **18 failing TESTS across 12 files**. The FULL
suite gave **19** -- the extra one
(`test_phase_86_6_subprocess_channel.py`) is **NOT in the named scope**, and it
is the **only order-dependent failure**: 18/18 pass in isolation, fails only in
the full suite. All 18 named failures reproduced identically alone and grouped,
so the step's "order-dependent / shared state" premise has **n=0 inside its own
scope**.

**The suite cannot detect order-dependence at all.** Only `pytest`,
`pytest-cov`, `pytest-timeout` are installed -- `pytest-randomly`,
`pytest-random-order`, `pytest-xdist`, `pytest-rerunfailures` are ABSENT, so
`-p no:randomly` is a no-op and collection order never varies. Absence of OD
findings is therefore not evidence of OD-cleanliness.

**Quarantine state:** `xfail_strict` is ABSENT from `pytest.ini`, so pytest's
`strict=False` default makes XFAIL *and* XPASS both silent -- and there is
already **1 live XPASS**. Two contradictory conventions coexist: `requires_live`
is opt-IN (fails safe, 38 refs) while
`test_phase_82_48_outcome_write_schema.py:210` uses opt-OUT
(`PYFIN_SKIP_LIVE_BQ`), hitting live BigQuery by default.

**Precedent that did not generalise:** phase-56.2 already diagnosed this exact
class in `test_phase_23_2_5_kill_switch_no_false_fires.py:271` -- reason string
says "the live log has been rotated ... so the count reflects **file state, not
code**". It fixed one file and stopped. See
[[project_kill_switch_archive_merge_36_8]] and
[[feedback_url_count_must_be_re_derived]].
