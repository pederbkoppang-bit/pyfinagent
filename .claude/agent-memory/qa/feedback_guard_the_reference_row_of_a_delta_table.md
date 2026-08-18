---
name: guard-the-reference-row-of-a-delta-table
description: In a table of deltas the shared baseline row is the highest-leverage target and usually the least guarded; a fix that guards one arm's input leaves every delta corruptible
metadata:
  type: feedback
---

When a deliverable reports `delta = arm - baseline`, attack the **baseline row**,
not the arms. It is subtracted from every other row, so one injection moves every
published delta at once — and it is routinely the row the guards forget.

**Why:** 86.59 cycle 3. Cycle 2's Q/A poisoned the baseline reference and the fix
added `baseline_slate_matches_an_unflagged_direct_call`, which recomputes the
baseline through a direct unflagged call and requires agreement. But that guard
compares only the `base` variable at `measure_flags():806`, which feeds **one
arm** (`min_k_sectors=3`). The row printed as `baseline` — and subtracted from all
three arms — comes from a *separate, structurally identical* `replay_session(...)`
call inside `for name, kw in FLAG_ARMS` at `:799-802`, guarded by nothing
behavioural. Adding `momentum_52wh_tilt=True, k=0.2` to that one line: **all six
guards green**, min_k's delta flipped `+2.1pp -> -2.1pp` (the figure ASK-1 rests
on) while the min_k arm itself was provably unchanged. A `w=0.05` variant left
every turnover delta reading EXACTLY as published while the baseline's top-sector
share moved `0.72 -> 0.64`. The fix had guarded the subtrahend's source and left
the minuend open.

**How to apply:**
- Find the reference the deltas are computed against and ask which call produces
  it. If two calls look identical, they are two sites — mutate each
  ([[mutate-each-duplicated-site-individually]]).
- A "recompute it independently and compare" oracle only covers the variable it
  compares. Check *which* variable, not the mechanism's name.
- The arm whose own numbers are unchanged can still have a wrong delta. Diff the
  DELTAS as well as the rows.
- Bound the oracle: 86.59's "independent" direct call re-used the same
  `build_yf_frame`, so it cannot see a corrupted frame builder.
- Related: [[a-fix-can-relocate-the-defect-one-seam-upstream]],
  [[class-guard-bound-to-the-helper-not-the-call-site]],
  [[baseline-captured-after-the-action]].
