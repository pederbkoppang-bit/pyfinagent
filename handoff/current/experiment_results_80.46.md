# Experiment results — masterplan step 80.46

**Missing in cycle 1, and the Q/A was right to flag it.** The five-file protocol requires this
artifact and I shipped only contract / research_brief / live_check. Written now, cycle 2.

## What shipped

| File | Change |
|---|---|
| `backend/tests/test_phase_75_ci_gates.py` | all 5 subprocess timeouts raised to >=20x measured cost; an AST-derived policy banner; a new guard `test_phase_80_46_no_subprocess_timeout_is_tuned_tight`; calibration constants pinned to their derivation |
| `.claude/masterplan.json` | filed **80.47** for the still-unexplained teardown error |

## Criterion 1 — REFUTED, and that is an allowed outcome

The criterion says "reproduced ... OR the hypothesis is explicitly refuted". It is refuted, three
ways, all independently re-derived by the Q/A:
1. A `TimeoutExpired` in a test body reports `1 failed`, not `1 error` (probe: 1s budget on a 5s
   sleep → `1 failed in 1.08s`). The observed line had **zero** failed.
2. `pytest.ini` has no `--continue-on-collection-errors`, so a collection error runs **zero** tests.
3. `250/2310` collected; observed `249 passed + 1 skipped` = 250 → the error attached to a test that
   **passed** = a **teardown** error.

The research gate's own culprit (`max_bridge`'s `live_bridge`) is also refuted: 0 `-k` matches.

**The observed error's cause remains UNKNOWN — now FILED as 80.47**, not asserted as filed. In cycle
1 I wrote "filed separately" while no such step existed. That is the third false past-tense claim of
this session and the Q/A caught it; the step now exists and I proved it in the same turn.

## Criterion 2 + 4 — reproduced, fixed, all five dispositioned

```
BEFORE: baseline 8.8s quiet  ->  TIMED OUT at 60s under 30x oversubscription
AFTER : same 30x contention  ->  284.6s, SURVIVED at 300s
```

| site | was | now | cost | multiple |
|---|---|---|---|---|
| `:143` whole-tree collection | 60s | 300s | 8.8s | 34x |
| `:187` single-file collection | 30s | 120s | ~1s | >=120x |
| `:211` / `:248` / `:279` tier checks | 15s | 30s | ~1s | >=30x |

Call sites **keyed by function name** (see cycle 3) — cycle 1's banner carried stale numbers and quoted a 60s
budget that existed at no call site. On a step whose thesis is "a constant that assumes a static
suite", that was the worst possible defect, and the Q/A caught it.

Margin, stated honestly: 284.6s against 300s is 5% headroom under the **synthetic** 30x load I
constructed; under the realistic ~6x contention that triggered the original event the job costs ~53s
and the margin is 5.6x. The 20x rule targets the realistic case.

## Criterion 3 — the 80.44 invariant is untouched

`EXPECTED_REQUIRES_LIVE_DESELECTED = 16` unchanged; `17 passed`.

## Criterion 5 — mutation, including the one I missed

| mutant | cycle 1 | cycle 2 |
|---|---|---|
| timeout back to 60 (the reproduced flake) | KILLED | KILLED |
| single-file check back to 15 | KILLED | KILLED |
| scope discriminator inverted | KILLED | KILLED |
| `MULTIPLIER` 20→1 alone | *recorded EQUIVALENT* | **KILLED** |
| **coordinated: timeout→60 AND cost→1.0** | *not written* | **KILLED** |

**The Q/A wrote a mutant I had not, and it survived**: lowering the timeout *and* the measured cost
together restored the exact 60s budget reproduced failing while keeping the guard green. The
calibration constants lived in the file they guarded and nothing pinned them. They are now asserted
against their derivation, which also converts my "equivalent" M4 into a real kill — so the
equivalence claim was true in isolation and wrong under composition.

## The guard took four attempts, and each failure was caught by measurement

1. Global 30s floor — **passed a `timeout=60`**, i.e. blind to the exact regression it existed for.
2. Keyed on the `--collect-only` flag — lumped the 8.8s whole-tree collection with a ~1s single-file
   one and **tripped on clean code**.
3. Keyed on collection **scope** — green, 3/3 kills, but calibration unpinned.
4. Calibration pinned — 5/5 kills.

## Verification

```
$ python -m pytest backend/tests/test_phase_75_ci_gates.py -q     # IMMUTABLE
17 passed
```

**Cycle 1 shipped this command RED.** A stale gitignored `.pyc` from my own in-tree mutation runs
executed an earlier broken guard, so `17 passed` held in my shell and `1 failed, 16 passed` on a
clean checkout. The Q/A reproduced it, proved the cause by pointing `PYTHONPYCACHEPREFIX` elsewhere,
and I confirmed it before fixing. **A verification claim measured in a dirty environment is not a
verification claim** — the pyc is removed and the command now passes twice consecutively.

## Do-no-harm

Test infrastructure only. No production code, no thresholds, no kill-switch state. Live book
untouched: `paused: false`, `armed: true`, peak `24666.57`.


## Cycle 3 — two more findings, and the first one is the same defect twice

**The banner's line numbers were stale AGAIN, inside the fix for the stale line numbers.**
Cycle 2 derived them by AST and *then inserted the 7-line banner*, which shifted every site it had
just named by 7-8 (`143→150`, `211→218`, `248→255`, `279→286`). And the artifact claimed "derived by
AST, not typed" — a claim that did not reproduce.

The root cause is **derive-then-edit ordering**, so the class fix is not "derive again": it is to
key on something that does not move. The banner now lists **function names**, which are immune to
insertions above them:

```
test_backend_not_requires_live_collection_count_is_stable() -> timeout=300s
test_lock_count_guard_collected_under_not_requires_live()   -> timeout=120s
test_coverage_tier_check_*()                                -> timeout=30s  (x3)
```

**The Q/A found an escape I did not disclose, and it was the binding one.** I disclosed the
non-literal-timeout residual and argued it was acceptable. The Q/A executed a *second* escape:
deleting **one character** — `"backend/tests/"` → `"backend/tests"` — flips the discriminator,
drops the floor `176s → 20s`, and re-admits the exact 60s budget reproduced timing out. Both
spellings collect **byte-identically** (`2295/2311`), so the escape is invisible in review. My
"acceptable residual" reasoning didn't survive contact with the escape I hadn't looked for.

Both closed, both one-liners as the Q/A specified:
- The discriminator now classifies by what the argv **targets** after `rstrip("/")` — a directory
  versus a `.py` file — a semantic distinction, not a textual one.
- A non-literal timeout is now an **offender**, not a skip: an unresolvable budget is not a
  compliant one. **Fail closed.**

### Mutation, cycle 3 — run OUT OF TREE

| mutant | result |
|---|---|
| CONTROL (unmutated) | green |
| QA_B one-char path edit + `timeout=60` **(the escape I missed)** | **KILLED** |
| QA_A non-literal `timeout=_TIGHT` | **KILLED** |
| M5 coordinated (`timeout→60` AND `cost→1.0`) | **KILLED** |
| M1 plain `timeout→60` | **KILLED** |

Run on scratch copies with `PYTHONDONTWRITEBYTECODE=1` and the repo file's sha verified unchanged —
because the **in-tree** mutation habit is exactly what produced cycle 1's stale-`.pyc` defect, where
the immutable command passed in my shell and failed on a clean checkout.

### The guard's full history: six attempts

1. Global 30s floor — passed a `timeout=60`, blind to the regression it existed for.
2. Keyed on the `--collect-only` flag — tripped on clean code.
3. Keyed on collection scope — green, but calibration constants unpinned.
4. Calibration pinned — defeated by a one-character path edit.
5. Targets normalised + fail-closed on non-literals — 4/4 kills.
6. Banner re-keyed to function names so it cannot go stale on insertion.

Every one of those failures was found by **executing** something, never by reading it.
