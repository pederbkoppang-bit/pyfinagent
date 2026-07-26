# live_check — masterplan step 36.9

Immutable requirement: *"Three test logs (one per finding) each showing the pre-fix failure
reproduced, then the post-fix behavior, against the real archived audit shapes measured 2026-07-26."*

## (0) The defect on the RUNNING system, before any code changed

GET only, no POST, no restart. `handoff/current/captures_36.9/live_kill_switch_pre_fix.json`:

```json
{"sod_nav": 23838.19, "sod_date": "2026-07-24", "peak_nav": 24666.57,
 "current_nav": 23838.16,
 "breach": {"daily_loss_pct": 0.0001, "daily_loss_limit_pct": 4.0,
            "trailing_dd_pct": 3.3584, "any_breached": false,
            "daily_baseline_missing": false, "armed": true}}
```

```
today (UTC)          = 2026-07-26
sod_date on the wire = 2026-07-24   -> STALE BY 2 DAYS
armed                = True
daily leg  : 23838.19 x 0.96 = 22884.66   <- the 4% point, against a 2-day-old anchor
trailing leg: 24666.57 x 0.90 = 22199.91
```

`22884.66` reproduces the masterplan's measured figure exactly. **F1 was live on the operator's badge
endpoint**, not hypothetical. The same capture re-verifies the standing precondition before any
kill-switch edit: the trailing leg still fires at NAV ≤ 22199.91.

## (a) F1 — stale `sod_date`

PRE-FIX (the fix reverted in memory to `daily_baseline_stale = False`, current test run):
```
>       assert r["daily_baseline_stale"] is True
E       assert False is True
FAILED ...::test_phase_36_9_the_live_2026_07_26_shape_no_longer_reports_armed
1 failed, 23 deselected in 0.03s
```
POST-FIX: `armed False`, `daily_baseline_stale True`, `daily_loss_pct 0.0` (**pre-fix this was 4.0**),
and the trailing leg still breaches on a catastrophic NAV —
`test_phase_36_9_stale_anchor_does_not_disable_the_trailing_leg`.

## (b) F2 — `nav_invalid` returned `armed:true`

PRE-FIX (`"armed": False` reverted to `bool(armed)`):
```
>       assert r["armed"] is False, (
E       AssertionError: a leg that cannot measure cannot fire -- unknown is not healthy
E       assert True is False
3 failed, 21 deselected in 0.03s
```
POST-FIX: `nav_invalid True`, `any_breached False`, `armed False`, `nav_invalid_disarmed True`, and
`daily_baseline_missing False` — the disarm is attributed to the NAV, not to healthy baselines.

## (c) F3 — `sod_nav=0.0` latched and wedged its own repair

PRE-FIX (`_coerce_nav` reverted to `float(nav)`):
```
>       assert snap["sod_nav"] is None, f"{bad!r} must not latch as a baseline"
E       AssertionError: 0.0 must not latch as a baseline
E       assert 0.0 is None
```
POST-FIX: `0.0 / -1.0 / nan / inf` are all refused — no state mutation, **no `sod_snapshot` audit
row**, and `sod_anchor_needs_reroll(snap, today)` (the REAL predicate, imported) returns `True`, so
the /resume 409's promise is kept. A book that latched `0.0` before this fix also re-anchors.

## Real archived audit shapes

The amended 36.7 test builds the genuine `-v4` row verbatim
(`{"ts": "2026-07-24T18:36:35.405342+00:00", "event": "sod_snapshot", "nav": 23838.19,
"date": "2026-07-24"}`) plus the real `-v3` `peak_update`, restores through
`_load_from_audit`, and asserts the new behaviour on that restore — the same 2-day-stale shape the
live endpoint served.

## Mutation matrix — 15 killed / 0 survived, one batch at the final baseline

`handoff/current/captures_36.9/mutation_matrix.txt`. This licenses *"these 15 were killed at baseline
`28 passed`"* and nothing more — and cycle 2 is the proof of how narrow that licence is: all 13
cycle-1 mutants passed while a live money-path regression sat in the diff, because every one of them
lived inside a module that never executed the order path. Both first-run survivors were defects in my own tests (an inline
predicate copy that could not fail, and a semantically inert mutant); both were fixed, not argued
away, and the fix for the first is why `sod_anchor_needs_reroll` exists.

## Verification

```
$ python -m pytest backend/tests/ -q -k kill_switch          # IMMUTABLE
166 passed, 1 skipped, 2126 deselected
```

## Do-no-harm

`handoff/kill_switch_audit.jsonl` md5 `ce8fb93348bb9a3bbe26f2d91b1bc05e` at every measurement point.
`:8000` GET-only, never restarted or POSTed to (launchd pid 76381). `:3000` never driven. Thresholds,
stops, sector caps, DSR, PBO byte-untouched. No peak reset.

**NOT LIVE:** not on the operator's `:8000` — the restart that would load it is owed only after Q/A
passes, and is the operator's call.


## (d) Cycle 2 -- the order-placing path, added after a Q/A caught a regression

`armed` is read by a FOURTH consumer, `paper_trader.check_and_enforce_kill_switch`, BEFORE the daily
roll. Folding staleness into it turned the ordinary first cycle of every UTC day into "lost history".

PRE-FIX (executed end-to-end, real `check_and_enforce_kill_switch`, pager captured, tmp audit path):
```
STALE anchor (yesterday)  -> blocked=True  reason=kill_switch_disarmed_lost_history  P1s=1
FRESH anchor (today)      -> blocked=False reason=None                               P1s=0
```
POST-FIX, all four states:
```
overnight stale anchor (THE REGRESSION)       blocked=False reason=None                             P1s=0
fresh anchor [CONTROL]                        blocked=False reason=None                             P1s=0
genuine LOST HISTORY (36.12 must still block) blocked=True  reason=kill_switch_disarmed_lost_history P1s=1
lost peak + stale sod (must still block)      blocked=True  reason=kill_switch_disarmed_lost_history P1s=1
```
Guarded by 4 new tests that execute the real cycle path, and by 2 new mutants
(`C2_baselines_present_refolds_staleness`, `C2_order_gate_reads_armed_again`). Matrix now
**15 killed / 0 survived** at baseline `28 passed`; immutable `166 passed, 1 skipped`.

The live audit md5 stayed `ce8fb93348bb9a3bbe26f2d91b1bc05e` across every probe and the whole matrix.


## (e) Cycle 3 -- the /resume 409 now names the actual cause

PRE-FIX, the message emitted for a merely-stale anchor (absence text, contradicted by its own
printed diagnostics):
```
Cannot resume: kill switch is DISARMED -- the loss baselines could not be restored, so neither
limit can be verified healthy (daily_baseline_missing=False, trailing_baseline_missing=False).
```
POST-FIX, driven through the real endpoint by
`test_phase_36_9_the_resume_409_names_staleness_not_absence`:
```
Cannot resume: the daily-loss anchor is STALE -- it is from '<yesterday>', not today (UTC) ...
The baselines themselves are intact (sod_nav=..., peak_nav=...); the trailing leg is
date-independent and still armed. NO operator action is required: the daily start-of-day roll
stamps today's anchor at the top of the next paper-trading cycle and this refusal clears itself.
```
Mutation: removing the branch -> `1 failed`. Immutable `167 passed, 1 skipped`.
Live audit md5 `ce8fb93348bb9a3bbe26f2d91b1bc05e` throughout.
