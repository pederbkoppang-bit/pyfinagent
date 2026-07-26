# live_check — phase-36.8

**Required (masterplan, verbatim):** *A test log showing: (a) the original 36.7 restore-true-peak
behavior still works, (b) the new re-anchor-respects-fresh-live-data behavior now works, both
against real archived file shapes.*

## (a) 36.7's restore-true-peak behaviour STILL works

Two assertions, one synthetic and one against the **real** corpus.

`test_phase_36_8_unmarked_rows_still_ratchet_exactly_as_phase_36_7` — archives hold `24666.57`
(2026-06-03) and `24124.77` (2026-06-22); the live file holds a later, LOWER, **unmarked**
`23838.19` (2026-07-24). Restored peak = **24666.57**. Assignment-replay would have given
`23838.19`; the ratchet gives the true mark.

`test_phase_36_8_the_real_corpus_still_restores_the_true_peak` — copies **every real audit file**
(`handoff/kill_switch_audit.jsonl` + the four under `handoff/audit/`) into a tmp tree and asserts the
restored peak equals `max(peak_update)` across them. All 20 real rows are unmarked, so this is
exactly the behaviour the live book depends on today. Originals are read-only: md5
`ce8fb93348bb9a3bbe26f2d91b1bc05e` before and after.

At suite level the immutable selector includes 36.7's whole module:

```
$ python -m pytest backend/tests/ -q -k kill_switch          # IMMUTABLE
138 passed, 1 skipped, 2126 deselected
```

## (b) The new behaviour: a fresh marked anchor now wins

**Regenerated in cycle 7, and the reason is the whole point of this section.** This block had
carried a cycle-1 row shape — `prior_peak=None` — through the cycle-5 redesign, which INVERTED
that shape's meaning: an anchor that names no superseded peak is now deliberately *non*-authoritative.
Executed at HEAD, the recorded shape returns `24666.57` — byte-identical to the outcome the block
presents as the PRE-FIX failure. It therefore demonstrated nothing: a row that fails the same way
before and after the fix cannot be a pre/post discriminator. Measured both ways at HEAD this turn:

```
prior_peak = None        -> restored peak_nav = 24666.57   (stale archive wins)
prior_peak = 24666.57    -> restored peak_nav = 18000.0    (fresh anchor wins)
```

Below is the CURRENT criterion-1 test run against the authority clause reverted to the pre-36.8
unconditional `max()` merge (in-memory; no repo write) — the real pre-fix signature:

```
>       assert ks.KillSwitchState().snapshot()["peak_nav"] == 18000.0
E       assert 24666.57 == 18000.0

backend/tests/test_phase_36_8_kill_switch_archive_merge_authority.py:103: AssertionError
FAILED backend/tests/..._archive_merge_authority.py::test_phase_36_8_a_fresh_marked_anchor_beats_a_higher_archived_peak
1 failed, 43 deselected in 0.03s
```

with the fixture rows `nav=24666.57` archived and `nav=18000.0, anchor=True, prior_peak=24666.57` live.

Whole file against that same reverted clause: **2 failed, 42 passed** (measured this turn; the
previously recorded *"10 failed, 12 passed"* was a cycle-1 figure describing a 26-test module and a
different revert). POST-FIX: **44 passed** (c1 26, c2 29, c3 32 — the module grew as each Q/A found a gap).

The boundary behaves like a boundary, not a freeze — three further assertions pin it:
later rows ratchet UP from the anchor (`18000 → 19000`, a subsequent `18500` ignored); an anchor
OLDER than a legitimate archived ratchet does **not** apply retroactively (`24666.57` still wins,
because `ts` order is the authority); and only a **literal** `True` grants authority, so a row
carrying `anchor: "yes"` / `1` / `["x"]` cannot lower the mark.

## Real archived file shapes — measured, read-only

| | |
|---|---|
| corpus | **897 rows across 5 files** |
| today's live baselines | **100% restored from the ARCHIVES** |
| true peak `24666.57` | lives in the **OLDEST** file |
| `peak_reset` rows ever | **zero** |
| boot cost | **0.95 ms total, 1.06 µs/row** |

This is why criterion 3 is answered by REFUSING a cap: an oldest-first prune would delete the row the
kill switch depends on. The archives are declared do-not-prune in both housekeeping scripts instead,
pinned by an AST test that fails if the two declarations drift.

## Criteria summary

| # | Criterion | Status |
|---|---|---|
| 1 | reproduce the failure first, verbatim | **MET** — `assert 24666.57 == 18000.0`, regenerated cycle 7 by running the CURRENT test against the reverted authority clause (see (b)) |
| 2 | 36.7's defect stays fixed | **MET** — synthetic + real-corpus assertions + 36.7's module green under the immutable selector |
| 3 | archive growth documented or capped | **MET** — cap REFUSED on measurement; do-not-prune declared in both scripts; boot cost measured |
| 4 | `reset_peak` stays DARK | **MET** — gate byte-untouched; a test asserts the call returns `None` and writes no row |
| 5 | mutation-test the fix | **MET as of cycle 5** — 13 mutations, 13 killed at baseline `44 passed`, one batch. The fifth route the cycle-4 Q/A found is closed STRUCTURALLY, not patched: authority now requires naming the superseded peak, so all five historical routes are unreachable by construction. `MSTRUCT` removes that clause and dies. Attribution, corrected: the **cycle-2** Q/A found THREE survivors (MX4, MX3, MX2); **MXP** came from the **cycle-3** Q/A. This licenses *"these 13 were killed at this baseline"* and nothing more. |

## Do-no-harm

`:8000` never restarted, never POSTed to (launchd pid `76381`, read via `launchctl print`).
`:3000` never driven. `handoff/kill_switch_audit.jsonl` md5 `ce8fb93348bb9a3bbe26f2d91b1bc05e` at
every measurement point; `git status` clean on both live audit files throughout. Kill-switch limits,
stops, sector caps, DSR and PBO byte-untouched. No peak reset performed.

**NOT LIVE:** this code is not on the operator's `:8000` — as with 36.12, the restart that would load
it is owed only after Q/A passes.


## Cycle-4 refresh, and the cycle-7 correction to it

This file carried cycle-1 numbers through two fix rounds; the cycle-2 Q/A flagged three that no longer
reproduced. Cycle-4 re-measured the counts — module **44 passed**, immutable `-k kill_switch`
**138 passed, 1 skipped**, all 44 inside that selector (cycle 1 shipped with **zero**) — and then
asserted *"all figures above are re-measured at HEAD"*. **That claim was false and is withdrawn.**
Section (b)'s capture was NOT re-measured; it stayed on a cycle-1 row shape that the cycle-5 redesign
had since inverted, and the whole-file pre-fix figure stayed on a 26-test-module number. The cycle-5
Q/A named a SET of two artifacts carrying this defect; `experiment_results_36.8.md` was regenerated
and this file was not, so a fix that was reported as complete was half-applied — and the cycle-4
blanket sentence then concealed the remaining half for two more cycles.

The lesson is the one this step keeps paying for: **a remediation list is closed member-by-member, not
by count**, and a sweeping "everything above is current" line is an unmeasurable claim that hides the
member you missed. Section (b) is regenerated above from commands run in cycle 7.
