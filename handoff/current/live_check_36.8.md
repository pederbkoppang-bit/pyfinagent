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

PRE-FIX, verbatim (recorded before any code changed):

```
        _write(archive / "kill_switch_audit-v3.jsonl",
               _row("2026-06-03T10:00:00+00:00", "peak_update", nav=24666.57))
        _write(live,
               _row("2026-07-26T10:00:00+00:00", "peak_update", nav=18000.0,
                    anchor=True, prior_peak=None))

>       assert ks.KillSwitchState().snapshot()["peak_nav"] == 18000.0
E       assert 24666.57 == 18000.0
```

Whole file pre-fix: **10 failed, 12 passed**. POST-FIX: **44 passed** (re-measured cycle 4; c1 26, c2 29, c3 32 — the module grew as each Q/A found a gap).

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
| 1 | reproduce the failure first, verbatim | **MET** — `assert 24666.57 == 18000.0`, recorded pre-fix |
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


## Cycle-4 refresh

This file carried cycle-1 numbers through two fix rounds; the cycle-2 Q/A flagged three that no longer reproduced. All figures above are re-measured at HEAD: module **44 passed**, immutable `-k kill_switch` **138 passed, 1 skipped**, and all 44 of this module's tests are inside that selector (cycle 1 shipped with **zero**).
