# live_check — masterplan step 80.46

Immutable: *"Verbatim capture of the reproduced error under contention, plus the post-fix run under
the SAME contention showing it gone."*

## Criterion 1 — the hypothesis is REFUTED, and the step required me to say so

I filed this step believing a `subprocess.TimeoutExpired` caused the observed `1 error`. Wrong,
three independent ways:

1. **Wrong bucket (my probe).** pytest buckets by PHASE: a test-body exception is a FAILURE; only
   setup/teardown/collection produce an ERROR. A 1s budget on a 5s sleep gives `1 failed in 1.08s`.
   The observed line had **zero** `failed`.
2. **Collection error excluded** (research gate). `pytest.ini` has no
   `--continue-on-collection-errors`, so a collection error runs **zero** tests, not 249.
3. **Arithmetic excludes a setup error.** Measured `250/2310 tests collected (2060 deselected)`;
   observed `249 passed + 1 skipped` = 250. Every selected item is accounted for, so the error
   attached to a test that PASSED -- a **TEARDOWN** error.

**The research gate's own culprit is ALSO refuted.** It named `test_phase_76_9_2_max_bridge.py`'s
`live_bridge` fixture. Measured: that file matches the `-k` expression **0 times** -- deselected, its
fixtures never ran. Checked rather than accepted.

**So the observed error's cause remains UNKNOWN** and is NOT claimed fixed here. Filed separately.

## What IS reproduced, before and after — the immutable requirement

```
BEFORE (60s budget):
  baseline (quiet, 10 cpus)  : 8.8s
  under 30x contention       : TIMED OUT at 60s      <-- reproduced on demand

AFTER (300s budget, same 30x contention):
  under 30x contention       : 284.6s -> SURVIVED
```

Honest reading of the margin: 284.6s against 300s is only **5% headroom under the synthetic 30x
oversubscription I constructed**. Under the realistic contention that triggered the original event
(~6x, measured) the job costs ~53s and the margin is 5.6x. The 20x rule is calibrated to the
realistic case, not the synthetic worst case.

## Criterion 4 — all five timeouts dispositioned, not just the one that fired

| site | was | now | measured cost | multiple |
|---|---|---|---|---|
| `:120` whole-tree collection | 60s | **300s** | 8.8s | 34x |
| `:164` coverage_tier_check | 30s | **120s** | ~1s | >=120x |
| `:187` single-file collection | 15s | **30s** | ~1s | >=30x |
| `:224` tier check | 15s | **30s** | ~1s | >=30x |
| `:255` tier check | 15s | **30s** | ~1s | >=30x |

The last four already cleared 20x at their old values; raised anyway so ONE rule covers the file
rather than four ad-hoc numbers.

## Criterion 3 — the 80.44 invariant still works

`EXPECTED_REQUIRES_LIVE_DESELECTED = 16` untouched; `17 passed` in the file.

## Criterion 5 — mutation, and the guard was BROKEN on its first two attempts

| mutant | result |
|---|---|
| `timeout` back to 60 (the reproduced flake) | **KILLED** |
| single-file check tuned back to 15 | **KILLED** |
| scope discriminator inverted | **KILLED** |

**Recorded because it is the useful part:** the first guard used a single global floor of 30s, which
**PASSED a `timeout=60`** -- it would not have caught a regression to the exact value reproduced
failing. Mutation caught that. The second attempt keyed on the `--collect-only` FLAG, which lumped
the whole-tree collection (8.8s) together with a single-file collection (~1s) and **tripped on clean
code**. Caught by running the guard against an unmutated tree before trusting it. The third keys on
collection SCOPE and is green clean, 3/3 killed.

One mutant is recorded as EQUIVALENT rather than counted: lowering `MULTIPLIER` alone cannot fail
while every timeout sits far above its floor -- a floor nothing is near is unkillable by
construction. Noted, not contrived into a kill.

## Do-no-harm

Test infrastructure only. No production code, no thresholds, no kill-switch state. Live book
untouched: `paused: false`, `armed: true`, peak `24666.57`.
