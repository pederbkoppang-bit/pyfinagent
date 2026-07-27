# Contract — masterplan step 80.46

**[P2] A flaky CI gate: fixed subprocess timeouts on a collection that keeps growing.**

Step id: `80.46` · Phase: PLAN · Date: 2026-07-27 · HEAD: `5c489d92`

## Immutable success criteria (VERBATIM from `.claude/masterplan.json`)

1. The error is REPRODUCED under controlled contention and captured verbatim, OR the hypothesis is explicitly refuted with evidence and the real cause named -- state which
2. If confirmed: the fix does not rely on a larger fixed constant; it scales with the measured workload or removes the subprocess round-trip, and the reasoning is stated
3. The gate still fails when it should -- the phase-80.44 requires_live-deselection invariant is asserted unchanged
4. All five fixed timeouts in the file are dispositioned explicitly (hardened or justified as safe), not just the one that fired
5. MUTATION-TEST whichever guard is added

## Criterion 1: THE HYPOTHESIS IS REFUTED. Three independent ways.

I wrote this step believing a `subprocess.TimeoutExpired` at `test_phase_75_ci_gates.py:123`
caused the observed `1 error`. **That is wrong**, and the step required me to say so rather than
bend the theory to fit.

1. **Wrong bucket (my own probe).** pytest buckets by PHASE, not exception type: a test-body
   exception is a FAILURE; only setup/teardown/collection produce an ERROR. Probed directly --
   a 1s budget on a 5s sleep gives `1 failed in 1.08s`. The observed line had **zero** `failed`.
2. **A collection error is excluded** (research gate). `pytest.ini` has no
   `--continue-on-collection-errors`, so a collection error would have run **zero** tests, not 249.
3. **The arithmetic excludes a setup error** (research gate, re-verified by me):
   `250/2310 tests collected (2060 deselected)`, and the observed `249 passed + 1 skipped` = 250.
   Every selected item is accounted for -- so the error attached to a test that **PASSED**,
   i.e. a **TEARDOWN** error.

**The research gate's named culprit is ALSO refuted.** It proposed `test_phase_76_9_2_max_bridge.py`'s
`live_bridge` fixture. Measured: that file matches my `-k` expression **0 times** -- it is deselected
and its fixtures never ran. I checked rather than accepted; an agent's attribution is not evidence.

**So the cause of the observed `1 error` remains UNKNOWN.** It is a teardown error among the 250
selected `kill_switch`/`perf_metrics`/`drawdown` tests. That stays open and must NOT be claimed as
fixed by this step.

## What IS reproducible, and what this step therefore fixes

The timeout fragility is real and now reproducible on demand:

```
baseline (quiet, 10 cpus)  : 8.8s
under 30x contention       : TIMED OUT at 60s   <-- reproduced
```

`handoff/current/captures_80.46/reproduction.txt`. This is a genuine latent flake worth closing on
its own merit -- it just is not the thing I originally saw.

## Research-gate summary

`handoff/current/research_brief_80.46.md` -- `gate_passed: true`, 6 sources read in full, 18 URLs,
recency scan performed.

Its recommendation, adopted: **a LOOSE timeout derived from a measured baseline (>=20x), not a tuned
one.** SAP HANA 2026 (n=559) measured **18% timeout-flakiness** for tests calibrated close to average
execution time versus **7%** under one loose global timeout -- tight timeouts are themselves a
flakiness source. Measured headroom in the file: `:120` is 8.6x (the tightest by an order of
magnitude); the three `coverage_tier_check.py` calls sit at 30-75x.

Rejected, with reasons: **in-process `pytest.main()`** (officially discouraged for repeated calls,
and here it would be re-entrant inside a live session -- a CI gate should not be the first user of a
discouraged API); **a source scan for `@pytest.mark.requires_live`** (downgrades an end-to-end fact
to a decorator-presence pin, an antipattern this project has already flagged). Retry-on-timeout is
an acceptable addition, not a substitute.

The gate's own counter-argument, recorded: a loose timeout lowers probability without eliminating
the class, and lengthens the worst-case stall. Accepted -- the alternative eliminates the class only
by adopting a discouraged API, and a 5-minute worst-case stall in a CI gate is cheaper than a flake.

## Plan

- Raise `:123` from 60s to a value >=20x its measured 8.8s baseline, with the measurement and the
  20x rule written at the site so the next reader can re-derive it rather than guess.
- Disposition **all five** timeouts (criterion 4), not only the one that fired: state each one's
  measured cost and headroom, and harden any below the 20x rule.
- Keep `EXPECTED_REQUIRES_LIVE_DESELECTED = 16` (phase-80.44) working unchanged (criterion 3).
- Mutation-test the guard (criterion 5).
- File the unexplained teardown error as its own step -- it is a different defect and must not be
  buried inside this one.

## Do-no-harm

Test-infrastructure only; no production code, no thresholds, no kill-switch state. The live book is
untouched (`paused: false`, `armed: true`, peak `24666.57`).
