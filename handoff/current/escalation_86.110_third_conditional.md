# ESCALATION -- step 86.110 -- PARKED on the 3rd-CONDITIONAL rule

**Status: PARKED, not done, not failed.** `.claude/masterplan.json` still
carries `"status": "pending"`. No step was flipped.

**This is the SECOND step parked on this rule tonight** (86.108 was the first),
and that pattern is itself the thing worth your attention -- see §4.

## 1. The situation

1. **All six immutable criteria are MET**, and the cycle-3 evaluator re-derived
   every one by execution rather than reading. Its words: *"No product defect
   remains: the fix, the sweep and the guard are all correct and
   mutation-proven."*
2. **The single remaining finding is closed.** It was one stale number: a
   scoped-suite block quoted `68 passed` where `71 passed` reproduces. Fixed in
   both artifacts, and the corrected figure is re-measured below.
3. **The next verdict is FAIL by rule, whatever the evidence.** The sequence is
   `[CONDITIONAL, CONDITIONAL, CONDITIONAL]`; CLAUDE.md's 3rd-CONDITIONAL rule
   requires the next pass to return FAIL. The cycle-3 return already computed
   `would_auto_fail: true`.

I did not spawn a fourth Q/A. Per the standing rule -- *all criteria MET +
starved => PARK + escalation file, never iterate* -- this file is the stop.

```
$ pytest backend/tests/ -q -k "cycle_health or heartbeat or rail_guard or 38_2 or 86_38 or 23_2_14 or 86_110"
71 passed, 3588 deselected, 1 warning in 10.99s      <- the corrected figure
```

## 2. What each cycle found

**Not one finding across three cycles was a defect in the shipped product.**
Two were real defects in guards *this step wrote*, and the rest were claims
that did not reproduce.

| Cycle | Finding | Closed by |
|---|---|---|
| 1 | **The guard could DESTROY production data.** `cycle_history.jsonl` is an append-only ledger the live loop writes from this machine; the guard restored its snapshot unconditionally, so a real append landing mid-suite would have been reverted -- a deleted cycle row, blamed on an innocent test. | Per-file rules for what a legitimate concurrent write looks like (`append_only` / `ledger_backed`), with all three arms mutated. |
| 1 | The full-suite block was measured before this step's own tests existed while labelled as the shipped tree. | Re-run, with the complete failing-ID list so drift is auditable. |
| 2 | **Only the redundant half of my own rule was tested.** Dropping `after.startswith(before)` survived 13/13, because all three fixtures were length-only discriminable. | A rewrite LONGER than the snapshot, plus cell **P11** mutating the prefix half alone. |
| 2 | My transitive-reacher census was wrong in count (6, not 3) and mechanism (two use a third idiom, stubbing `get_log`). | Census derived and pasted rather than restated. |
| 3 | The scoped-suite block still quoted the pre-cycle-2 figure -- **the same stale-capture class as cycle 1**, because I regenerated the full-suite block and left its sibling untouched in both files. | Both files corrected; figure re-measured above. |

## 3. What is shipped, and what it is worth

- The two-line isolation fix, **proven per-site**: the evaluator reverted each
  site individually and each was caught by the correct individual test.
- `scripts/qa/heartbeat_leak_sweep_86_110.py` -- a reachability-keyed
  enumeration whose result the evaluator reproduced *from the criterion's own
  literal rule* and found an exact match.
- A third pollution guard in `conftest.py`, beside the existing BQ and
  Slack-egress ones, which fires, repairs, and **cannot revert a legitimate
  concurrent write**.
- 13 tests, a 10-cell matrix (10/10), and the heartbeat regenerated
  field-for-field from the ledger -- `updated_at` equal to the last completed
  row's `completed_at` to the microsecond.
- **A real regression from 86.108 caught and repaired**: the `threading.Lock`
  roster guard had been red on main because 86.108's `-k` sweep never selected
  it. *A `-k` selection is not a regression suite.*

## 4. The decision, and a question about the rule itself

**Option A -- accept on the cycle-3 evaluator's finding.** All six criteria
MET, no product defect, and the one evidence nit is now fixed and re-measured.
Flip 86.110 to `done` on that basis. **This needs your authority: Main must
never mark a step done without a PASS**, which is why it is parked.

**Option B -- authorise attempt 4** knowing it returns FAIL by rule, resetting
the counter so attempt 5 can judge the current evidence. Costs both remaining
attempts (4 of 5 used).

**Option C -- park indefinitely.** The code is shipped and correct either way.

**And the question worth more than either step:** the 3rd-CONDITIONAL rule
exists to stop a harness *logging instead of correcting*. That is not what
happened here or on 86.108. In both, every cycle closed real findings and the
evaluator confirmed each closure by execution — the steps were terminated for
accumulating CONDITIONALs while *converging*, on evidence-class nits, with all
criteria met. Two steps in one session is a small sample, but the rule cannot
currently distinguish "stuck" from "converging", and both look identical to it.
Worth a decision independent of these two steps.

## 5. Residual, queued not fixed

`test_phase_61_2_decision_integrity.py` writes the real
`handoff/.conviction_fallback_streak.json`, which production reads
(`autonomous_loop.py:2911`, read at `:1099`/`:1113`). It is **untracked**, so it
sits outside this guard's declared git-tracked scope and no claim here is
false -- but "catches the CLASS" overstates what was measured. Not fixed
in-step: widening the guard's scope mid-step would ship an unmeasured
behaviour change.

Also open (NOTE-level, from cycle 3): deleting the `_isolates` `Assign` branch
leaves the suite green while the sweep's cross-check silently degrades. The
direction is conservative -- it can only over-report leaks, never clear one --
and the criterion's literal rule is the `setattr` shape, so the `Assign` branch
is an enhancement beyond it.
