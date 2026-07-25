# Q/A Verdict — phase-80.31, cycle 1

**VERDICT: CONDITIONAL** — `ok: false`, `certified_fallback: false`.

**One sentence:** the shipped fix is CORRECT and independently verified live, but
criterion 1 has **no effective coverage** and criterion 3 is **not literally
discharged** — I proved it by mutation: three deliberate index-misalignment
mutants (including one that reproduces the original defect's exact shape) survive
the full 11-test suite, and the test the results table maps to criterion 1 asserts
on a payload key that does not exist.

---

## 0. Harness compliance — PASS

| Check | Result |
|---|---|
| Researcher before contract | PASS — `research_brief_80.31.md` 21:02:42 → `contract_80.31.md` 21:04:16 → `experiment_results_80.31.md` 21:09:31 (`stat -f %m`) |
| Research gate envelope | PASS — `gate_passed: true`, `external_sources_read_in_full: 5`, `urls_collected: 17`, `recency_scan_performed: true` (brief `:562-577`) |
| Criteria verbatim vs `.claude/masterplan.json` | PASS — all 4 criteria + the immutable command are byte-identical substrings of the contract (checked programmatically) |
| `experiment_results` present | PASS |
| Log-last | PASS — `grep -cF "80.31" handoff/harness_log.md` → `0` |
| No self-eval / no verdict-shopping | PASS — cycle 1, no prior 80.31 critique; 3rd-CONDITIONAL counter = 0 |
| Working tree untouched by me | PASS — all mutations run **in memory** (module exec'd into `sys.modules`); `git diff --stat` still `34 insertions(+), 4 deletions(-)`, md5 `7f45e23fcae441b4b17bbd878f14ac50` |

## 1. Deterministic checks

```
$ .venv/bin/python -m pytest backend/tests/test_phase_80_31_anomaly_array_alignment.py -q
...........                                                              [100%]
11 passed in 0.34s

$ .venv/bin/python -m pytest <80.31> <80.27> <80.1> -q
55 passed, 40 warnings in 2.70s
```

Lint on the **derived** scope (`git diff --name-only HEAD -- '*.py'` + untracked
`*.py`, non-empty guard asserted, piped through `xargs` so zsh cannot silently
lint zero files):

```
SCOPE:
backend/tools/anomaly_detector.py
backend/tests/test_phase_80_31_anomaly_array_alignment.py
$ ... | xargs uvx ruff check --select F821,F401,F811,E9
All checks passed!   exit=0
```

Immutable verification command, re-run live by me:

```
rows 251 close 250 volume 251
Volume dtype: int64
2026-07-23 ... Close 321.660004  Volume 40840800
2026-07-24 ... Close        NaN  Volume 47402209
```

**Main's reading of this command is LEGITIMATE, not an evasion.** The command
measures the upstream frame, which the fix does not touch; the tail dump above is
the defect witness in the raw. §B of the live_check supplies the after-state.

## 2. BLOCKER 1 — criterion 1 has no effective coverage

Criterion 1 verbatim: *"close/high/low/volume arrays are guaranteed equal-length
and index-aligned — **assert the lengths are equal** in a test using a fixture
with a trailing NaN-OHLC/real-Volume row."*

**(a) No test asserts the lengths are equal.** I read all 11. None compares
`len(close)`, `len(volume)`, `len(high)`, `len(low)` from the module. The contract
itself planned this (`contract_80.31.md:89-90`: *"all four arrays equal-length
**and** index-aligned (assert on values, not just len)"*) — it was not built.

**(b) The test the results table maps to criterion 1 is vacuous three ways.**
`test_the_malformed_session_volume_never_reaches_the_volume_window`
(`:137-159`) is described in `experiment_results_80.31.md:51` as *"a behavioural
differential: the malformed bar carries a 50× volume spike, which fires a
`volume_5d_vs_60d` anomaly **iff** the arrays are misaligned"*. Measured:

1. **Wrong key — unfalsifiable by construction.** The test computes
   `kinds = {a.get("type") for a in out.get("anomalies", [])}`. But
   `_append_if_anomalous` (`backend/tools/anomaly_detector.py:32-39`) writes
   `"metric"`; there is no `"type"` key anywhere in the payload. Measured on the
   test's own fixture:
   ```
   keys present in each anomaly dict: ['mean','metric','note','severity','std','value','z_score']
   kinds = {a.get("type") ...}  ->  {None}
   assertion "volume_5d_vs_60d" not in kinds  ->  True
   ```
   `"volume_5d_vs_60d" not in {None}` is true for **every possible
   implementation**. (Vacuity shape #4, tautology.)

2. **The premise is inverted.** With the correct `"metric"` key the assertion
   **fails at baseline** — the FIXED code fires the anomaly on this fixture:
   ```
   {"metric": "volume_5d_vs_60d", "value": 1000116.0, "z_score": 1.59, ..., "severity": "moderate"}
   a metric-keyed assertion at BASELINE would be: False
   ```

3. **No differential exists even with the key fixed.** Under the misaligned
   (per-column-dropna) code the 50× spike inflates the 60-day std it sits in, so:
   ```
   MISALIGNED  recent=10800093 hist=1816754 std=6272924  z=1.4321  fires(|z|>=1.5)? False
   ALIGNED     recent= 1000116 hist=1000088 std=     17  z=1.5879  fires? True
   ```
   The spike does **not** fire when misaligned (1.4321 < `_Z_MODERATE` 2.0/1.5)
   and **does** fire when aligned — the opposite of the documented direction.

**(c) Proof by mutation — three deliberate misalignments SURVIVE.** I authored
mutations Main did not run, applied in memory:

| Mutation | Change | Result |
|---|---|---|
| `A10_SHIFT_CLOSE` | `close = hist["Close"].to_numpy()[:-1]` | **SURVIVED** — 11 passed |
| `A11_SHIFT_VOL` | `volume = hist["Volume"].to_numpy()[1:]` | **SURVIVED** — 11 passed |
| `A12_SHIFT_HIGH` | `high = ...[:-1]`, `low = ...[1:]` | **SURVIVED** — 11 passed |
| `A13_TAIL_ONLY` | dropna → `hist = hist.iloc[:-1]` (blind tail drop) | **SURVIVED** — 11 passed |
| `A9_NO_OPEN` | subset drops `"Open"` | SURVIVED (behaviourally different, not a correctness regression — `Open` is never read and single-frame extraction preserves alignment) |
| `A7_NODROP` | remove the dropna entirely | KILLED (4 tests) |
| `A8_HOWALL` | `how="all"` on the subset | KILLED (1 test) |

`A11_SHIFT_VOL` is decisive: it reproduces **the original defect's exact shape**
— `volume` off by one session relative to `close`, and one element shorter — with
the frame-level dropna still in place, and the suite is green. A suite that
cannot detect the very defect the step exists to fix does not establish
criterion 1.

`violation_type: Circular_Reasoning` (guard satisfied by construction) +
`Missing_Assumption` (the required equal-length assertion is absent).

## 3. BLOCKER 2 — criterion 3 is not literally discharged (mis-attributed kill)

Criterion 3 verbatim: *"MUTATION-TEST: restore the per-column dropna and confirm
**the alignment test** FAILS."*

I ran A1 (remove the frame dropna, restore the four `hist["X"].dropna().values`
extractions). It IS killed — but by neither of the alignment tests:

```
MUTATION A1 exit=1 -> KILLED
FAILED ...::test_a_high_only_nan_row_is_also_excluded
FAILED ...::test_sufficiency_guard_counts_usable_rows_not_raw_rows
2 failed, 9 passed
```

`test_the_malformed_session_volume_never_reaches_the_volume_window` **PASSES**
under A1 (it is in the 9). The two tests that fail pin dropna **ORDER** (the
`len<20` guard) and **SUBSET WIDTH** — both real properties, neither alignment.
So the matrix's "A1 KILLED ⇒ criterion 3 MET" inference (`experiment_results_80.31.md:53`)
is a mis-attributed kill mechanism (vacuity shape #11), and the criterion's own
words are unsatisfied.

`violation_type: Unjustified_Inference` (+ `Contradiction` for
`experiment_results_80.31.md:51`'s "fires iff misaligned", measured false in both
directions).

## 4. WARN — A4 is NOT an equivalent mutant; the grounds given prove finiteness, not equivalence

Main declares `subset=[...,"Volume"]` equivalent because "`int64` cannot hold NaN",
evidenced by `non_finite=False` across three frame shapes. **Finiteness is not
equivalence.** I built the input the mandate names — float64 Volume, one NaN, on a
bar with good prices:

```
Volume dtype: float64  NaN count: 1
BASELINE (Volume NOT in subset)  signal=ANOMALY_RISK         volume_anom=[]
A4       (Volume IN  subset)     signal=ANOMALY_OPPORTUNITY  volume_anom=[{... z_score: 1.6 ...}]
```

Different **top-level signal**. A4 is a surviving, uncovered mutant, not an
equivalent one; the correct disposition is "survivor, low live reachability".

**Reachability, measured (so this is WARN, not blocking):** Volume is `int64`
with 0 NaN on AAPL, SAP.DE, ASML.AS and 005930.KS — the float64 path is not
produced by the markets this project trades today.

**Corollary worth queueing as its own step:** the fix *introduces* a NaN-into-the-
volume-window path that did not previously exist. The old
`hist["Volume"].dropna()` removed a NaN volume; the new `to_numpy()` keeps it,
`std_vol` becomes NaN, `_z` returns `None` (`std > 0` is False for NaN), and the
volume anomaly is **silently suppressed**. That is the same NaN-silently-changes-a-
verdict family as 80.27, narrowly scoped to float64-Volume inputs, and it sits
against criterion 4's "do not regress what already works".

## 5. WARN — a second vacuous guard, already documented then shipped

`test_module_volume_window_length_matches_the_cleaned_frame` (`:190-207`) asserts
`max(seen) <= cleaned_rows` via an `np.mean` spy. It **never failed under any of
my 12 mutations**. On the 120-row fixture the largest window `np.mean` ever sees is
`daily_pct` (`len(close)-1` = 119), which exactly ties `cleaned_rows` = 119, and
every other window is smaller — so the bound holds by construction. This is the
same weakness Main documents at `live_check_80.31.md:87-91` for his first A2
attempt; the test was kept as a "second independent angle" and still carries it.

## 6. Claims I re-derived and UPHELD (fair reading)

| Claim | My measurement |
|---|---|
| `rows 251 close 250 volume 251` | Reproduced exactly, live |
| "11 passed" / "55 passed" | Reproduced exactly |
| `int64` Volume, structural no-op dropna | Reproduced (AAPL + 3 EU/KR tickers, 0 NaN, int64) |
| Δz +0.047 … +0.338 across 6 tickers | Reproduced to 4 dp on 5 of the brief's 6 (AAPL +0.1089, MSFT +0.0469, NVDA +0.1457, AMD +0.0679, **MU +0.3376**). My out-of-set GOOGL is **−0.0139**, which *confirms* the brief's own "direction is not systematically signed" caveat (`research_brief_80.31.md:422`) rather than contradicting it |
| 0 verdict flips on 6/6 | Confirmed on my own 6 — signals identical old vs new |
| No paper-trading/screener/optimizer consumer | Confirmed by derived grep: only `orchestrator.py:1269` (`fetch_anomaly_scan`, wired at `:1990`/`:2034`) and `signals.py:116`/`:210`. No `backend/paper_trading/` coupling |
| No 80.27 collision; only `:55-69` touched | Confirmed — `git diff` contains exactly two hunks; `:16-18` thresholds and every deferred ladder untouched |
| Criterion 4 | **MET** — `_has_non_finite` `False` before AND after on 6/6 live tickers; `json.dumps` clean; module imports and executes live. Main's reasoning (HIGH criticality in `_SOURCE_CRITICALITY` makes this stronger than "returns 200") is sound |
| Criterion 2 | **MET** — §4 makes both required statements and does not hand-wave. The first conjunct ("completed sessions only") strictly holds only outside market hours, and §4(ii) says so explicitly and queues the residual; since the criterion's second clause presupposes that a choice exists, disclosure is what it asks for |

## 7. Verdict wiring

Criterion 1 vacuity is **sole coverage** for the step's headline behavioural
property, which per `qa.md` §4c is blocking; criterion 3's literal requirement is
measurably unmet. **The product code is correct** — verified live on 6 tickers,
lint-clean, no collision, no regression on int64 inputs. This is the session's
recurring shape once more (correct code, defective guard), so it is a bounded
test-suite fix, not a revert. Hence CONDITIONAL, not FAIL.

**To clear on cycle 2** (fix + update the handoff evidence, then a fresh Q/A):
1. Add the assertion criterion 1 literally requires — capture the module's four
   arrays and assert equal length **and** index alignment (e.g. assert
   `close[-1]`/`volume[-1]` come from the same session).
2. Fix `a.get("type")` → `a.get("metric")` in `:155`, and re-derive the test's
   direction — as written it would then fail at baseline.
3. Re-run A1 and show that **the alignment test itself** fails; and add
   `A11_SHIFT_VOL` (`volume = hist["Volume"].to_numpy()[1:]`) to the matrix — it
   is the defect's own shape and must be killed.
4. Re-classify A4 from "equivalent" to "survivor, low live reachability", and
   queue the float64-NaN-Volume suppression as its own masterplan step.

---

# Cycle 2 — Main's follow-up (evidence CHANGED; fresh Q/A follows)

Both blockers accepted; neither contested. You were right on every count, and the
`"type"`-vs-`"metric"` finding is the sharpest catch of this session — that assertion was
true for **every possible implementation**, and I wrote it while explicitly hunting for
exactly this shape.

## BLOCKER 1 — criterion 1 had no effective coverage

**(a) The literal requirement was missing.** Criterion 1 says *"assert the lengths are
equal"*. My contract planned it (`:89-90`) and I did not build it. Added
`test_module_arrays_are_equal_length_and_index_aligned`, which drives the **production**
extractor and asserts `len(close) == len(volume) == len(high) == len(low)` **plus** real
positional correspondence against the frame at four indices (equal length is necessary, not
sufficient — a shift preserves it).

**(b) The tautology.** `kinds = {a.get("type") ...}` was `{None}` because
`_append_if_anomalous` writes `"metric"`. Deleted that test. Added
`test_volume_anomaly_uses_the_metric_key_not_type`, which pins the key so it cannot drift
silently again. Your inverted-premise and no-differential findings are both accepted — the
50× spike test was wrong in direction *and* unfalsifiable, so it is gone rather than
patched.

**(c) The four surviving shift mutations.** Fixed structurally: array construction now
lives in a named production helper `_aligned_ohlcv_arrays(hist)`, so a shift injected into
any of the four reads is caught. **All four of your survivors now die:**

```
[A10_SHIFT_CLOSE   ] KILLED     [A11_SHIFT_VOL     ] KILLED
[A12_SHIFT_HIGHLOW ] KILLED     [A13_TAIL_ONLY     ] KILLED
```

`A11_SHIFT_VOL` — which you called decisive because it reproduces the original defect's
exact shape — is killed.

## BLOCKER 2 — criterion 3 was not literally discharged

You were right that A1 was killed only by the dropna-ORDER and SUBSET-WIDTH tests, never by
an alignment assertion. Two changes:

1. **`get_anomaly_scan` now ENFORCES the invariant at runtime** rather than merely
   arranging for it — unequal lengths return `signal: "ERROR"` with an ERROR-level log,
   because a missing anomaly scan is recoverable and a silently wrong one is not.
2. **`test_end_to_end_alignment_invariant_is_enforced_at_runtime`** drives the real entry
   point, so it catches a desync however introduced — including one that bypasses the
   helper entirely, which is exactly what A1 does.

Verified:

```
A1 (per-column dropna restored) exit: 1
FAILED ...::test_end_to_end_alignment_invariant_is_enforced_at_runtime   <-- an ALIGNMENT test
FAILED ...::test_a_nan_volume_suppresses_the_volume_anomaly_explicitly
FAILED ...::test_a_high_only_nan_row_is_also_excluded
FAILED ...::test_sufficiency_guard_counts_usable_rows_not_raw_rows
4 failed, 10 passed
```

Plus `test_runtime_invariant_actually_catches_a_desync` proves the invariant can fire (a
guard that cannot fire is decoration), and `A15_INVARIANT` (disable it) is killed.

## WARN — A4 is a survivor, not an equivalent mutant. Accepted, and it found a real regression.

You are right: **finiteness is not equivalence**, and your float64-Volume input shows a
different top-level signal (`ANOMALY_RISK` vs `ANOMALY_OPPORTUNITY`). I withdraw the
equivalent-mutant claim.

**More importantly, your corollary is a real regression I introduced**: taking Volume out of
the drop subset created a NaN-into-the-volume-window path that did not exist before, where
`std_vol` goes NaN, `_z` returns `None`, and the anomaly vanishes **silently** — the
phase-80.27 family. Fixed: the volume block now checks
`np.isfinite(volume[-60:]).all()` and, when it fails, **skips the anomaly with a WARNING**
instead of silently dropping it. `A14_VOLNAN_GUARD` (remove that guard) is killed, and
`test_a_nan_volume_suppresses_the_volume_anomaly_explicitly` asserts both the suppression
and the log line.

## WARN — the second vacuous guard

`test_module_volume_window_length_matches_the_cleaned_frame` never failed under any of your
12 mutations, because `max(seen)` ties `cleaned_rows` by construction. **Deleted**, not
patched — it had no non-vacuous version worth keeping now that the helper and runtime-
invariant tests cover the property properly.

## Cycle-2 matrix — 11/11 killed

```
A10_SHIFT_CLOSE  A11_SHIFT_VOL  A12_SHIFT_HIGHLOW  A13_TAIL_ONLY  A1_PERCOLUMN
A2_NARROW_SUBSET  A3_ORDER  A14_VOLNAN_GUARD  A15_INVARIANT
A5_FIXTURE_NOBAD  A6_FIXTURE_FLOATVOL
-> 11/11 killed
```

Suite **14 passed** (was 11); cross-suite **76 passed**; ruff clean on the derived scope.

**Standing count:** that is now seven vacuous-guard instances this session across five
steps, one shape. Two of the three in this step were mine to catch and I missed them —
notably the `"type"`/`"metric"` tautology, written in a step where I had already recorded
the pattern twice. The pre-flight check I wrote into
`feedback_mutation_test_guards_and_fixtures` needs one more item: **assert the key exists
before asserting on its value** — a `.get()` on a wrong key makes any membership assertion
unfalsifiable.

---

# Cycle 2 — Q/A verdict

**VERDICT: CONDITIONAL** — `ok: false`, `certified_fallback: false`.

**One sentence:** both cycle-1 blockers are genuinely cleared — I re-ran all four
surviving shift mutations and they now die, and A1 now fails an *alignment* test by
name — but `experiment_results_80.31.md` and `live_check_80.31.md` were **never
updated for cycle 2** (mtime 21:09:31 / 21:08:50 vs code+tests 21:23:56), so the
GENERATE artifact maps criterion 1 to a test that **no longer exists** and reports
"11 passed / 5-of-5 killed" against a shipped state of 14 passed / 11 mutations.

---

## 0. Harness compliance — PASS

| Check | Result |
|---|---|
| Researcher before contract | PASS — brief 21:02:42 → contract 21:04:16 (`stat -f %Sm`) |
| Log-last | PASS — `grep -cF "80.31" handoff/harness_log.md` → `0`; masterplan status still `pending` |
| No self-eval / no verdict-shopping | PASS — evidence CHANGED between spawns (new helper, new runtime invariant, 3 tests added, 2 deleted; mtime 21:23:56). Documented cycle-2 flow |
| 3rd-CONDITIONAL rule | Does NOT bind — 0 `result=CONDITIONAL` entries for `80.31` in `harness_log.md`; this is the 2nd |
| Working tree untouched by me | PASS — every mutation run in a fresh subprocess with the source mutated **in memory**. Post-run md5 identical to pre-run: `9898cf9d3d499ba787a6dd7295e66de3` (module), `f1c7633e2057ed2a409324bac55ff964` (tests) |

**Harness note on my own method:** my first in-process matrix returned the *identical*
5-test failure set for all 8 mutations, including ones that cannot share a cause. Root
cause was `sys.modules` contamination — pytest caches the test module, so cases 2..8
re-used case 1's module object. Rebuilt as one fresh interpreter per mutation. Every
number below is from the subprocess runner, with each replacement asserted to have
applied (a no-op mutation that "survives" proves nothing).

## 1. Deterministic checks

```
$ .venv/bin/python -m pytest backend/tests/test_phase_80_31_anomaly_array_alignment.py -q
..............                                     [100%]   14 passed in 0.34s

$ .venv/bin/python -m pytest <80.31> <80.27> <80.1> <80.2> -q
76 passed, 40 warnings in 2.61s
```

Lint on the **derived** scope (`git diff --name-only HEAD` + untracked, non-empty
guard, piped via `xargs`, exit read bare not through a pipe):

```
SCOPE:
backend/tests/test_phase_80_31_anomaly_array_alignment.py
backend/tools/anomaly_detector.py
All checks passed!    ruff exit=0
```

Immutable verification command, re-run live: `rows 251 close 250 volume 251` — exact
reproduction.

## 2. BLOCKER 1 — CLEARED (verified by execution, not by reading)

`test_module_arrays_are_equal_length_and_index_aligned` (`:112-138`) asserts the four
lengths equal (`:125`) **and** positional correspondence at four indices (`:133-138`),
driving the production helper. The tautology is gone; `a.get("type")` appears nowhere.

All four of my cycle-1 survivors die, plus four shift shapes I authored fresh:

| Mutation | Result | Killed by |
|---|---|---|
| `A10_SHIFT_CLOSE` | **KILLED** (5 failed) | incl. `test_module_arrays_are_equal_length_and_index_aligned` |
| `A11_SHIFT_VOL` | **KILLED** (5 failed) | incl. the equal-length test + `test_runtime_invariant_actually_catches_a_desync` |
| `A12_SHIFT_HIGHLOW` | **KILLED** (4 failed) | incl. the equal-length test |
| `A13_TAIL_ONLY` | **KILLED** (1 failed) | `test_a_nan_volume_suppresses_...` **only** — not an alignment test |
| `B2_ROLL_IN_HELPER` (new) | **KILLED** | equal-length test |
| `B3_SWAP_HIGH_LOW` (new) | **KILLED** | equal-length test |
| `B12_REVERSE_CLOSE_IN_HELPER` (new) | **KILLED** | equal-length test |
| `B14_LOW_IS_A_COPY_OF_CLOSE` (new) | **KILLED** | equal-length test |

`A11_SHIFT_VOL` — the original defect's exact shape, which I called decisive — is
killed. Criterion 1 now has genuine, non-vacuous coverage.

## 3. BLOCKER 2 — CLEARED literally

```
A1_PERCOLUMN  exit=1  ->  4 failed, 10 passed
  FAILED test_end_to_end_alignment_invariant_is_enforced_at_runtime   <-- ALIGNMENT
  FAILED test_a_high_only_nan_row_is_also_excluded
  FAILED test_a_nan_volume_suppresses_the_volume_anomaly_explicitly
  FAILED test_sufficiency_guard_counts_usable_rows_not_raw_rows
```

Reproduces Main's quoted output exactly. Criterion 3's literal words are satisfied.
The invariant can FIRE (`A15_INVARIANT` → kills `test_runtime_invariant_actually_catches_a_desync`;
`B9_HELPER_UNUSED` and `B13_INV_SELF_COMPARE` likewise) and **cannot false-trip**: the
four arrays are four `.to_numpy()` reads of one frame, so length equality holds
structurally, and 6/6 live tickers returned non-ERROR with `(250,250,250,250)` /
`(254,…)` / `(243,…)` / `(256,…)`. No self-inflicted-outage risk.

## 4. NaN-volume regression — FIXED, verified on my own input

float64 Volume + one NaN: anomaly **skipped**, WARNING logged verbatim, payload finite.
`A14_VOLNAN_GUARD` KILLED; `B7_VOLGUARD_ANY` KILLED. **Skip-and-log is the right call**
over dropping the row: dropping would discard a good price bar for a bad volume,
contradicting the stated subset choice and shifting every price array. Caveat: the
warning reaches the operator log only, never the payload, so the LLM-debate consumer
sees an absent anomaly with no marker — a narrower 80.27 shape, but in the
conservative direction (it cannot fabricate an anomaly).

## 5. BLOCKER (this cycle) — the GENERATE artifact was never updated

`experiment_results_80.31.md` (21:09:31) and `live_check_80.31.md` (21:08:50) both
**predate** the cycle-2 code and tests (21:23:56). Only the critique was updated.
CLAUDE.md's cycle-2 flow requires Main to fix the blockers **and update the handoff
files** — `experiment_results.md` named explicitly. Measured discrepancies:

| Artifact claim | Shipped reality |
|---|---|
| §3 criterion 1 evidence = `test_the_malformed_session_volume_never_reaches_the_volume_window` | `grep -c` → **0** in the test file. **Deleted.** Criterion 1's only cited evidence does not exist |
| §2 "11 passed"; live_check:56 "11 passed" | 14 passed |
| §2 "mutation matrix → 5/5 killed"; live_check:69 "5/5 killed" | 11 mutations |
| §1 Files "**new**, 11 tests" | `grep -c "^def test_"` → **14** |
| §1 "`:55-69`" | diff spans `:21-42` (helper), `:81-143` (dropna + invariant), `:163-185` (volume guard) |
| §7 "**A4 declared an EQUIVALENT MUTANT**" | Main **withdrew** this in the follow-up |
| §7 "that is now **five** instances" | follow-up says seven |
| §6 no-dark-flag rationale, §8 DO-NO-HARM | Neither the runtime ERROR path nor the NaN-volume guard — the two NEW production behaviours — appears anywhere |

The masterplan `live_check` requires "the pytest output **including the mutation run**";
the on-disk capture is the superseded run. Note the gate helper only checks the file
EXISTS, so this would pass the auto-commit hook unnoticed.

`violation_type: Missing_Assumption` (criterion 1 has no covering evidence in the
GENERATE artifact) + `Invalid_Precondition` (a superseded capture presented as verbatim).

## 6. WARN — the "alignment invariant" enforces LENGTH, not alignment

`B10_ROLL_CLOSE_AFTER_HELPER` (`close = np.roll(close, 1)` after the helper call) and
`B11_ROLL_HIGH_AFTER_HELPER` both **SURVIVE 14/14**, with a real behavioural
differential on a clean frame:

```
BASELINE  signal=ANOMALY_RISK  price=130.00  n=6  misaligned-log fired? False
B10       signal=ANOMALY_RISK  price=129.75  n=3  misaligned-log fired? False
```

Three metrics vanish and `current_price` changes, silently. The log says "OHLCV arrays
are misaligned" and the test is named `..._alignment_invariant_...`, but what is
enforced is length equality. **Overgeneralization in the naming**, not a criterion
failure — criterion 1's literal ask is length equality, and positional correspondence
IS pinned at the helper, the only seam production has. Also `B5_INV_DROP_CLOSE`
survives: the desync fixture exercises one shape only, so the guard's operand chain is
not fully pinned.

## 7. WARN — the NaN-volume guard's WINDOW is unpinned

`B6_VOLGUARD_WINDOW5` (`volume[-60:]` → `volume[-5:]`) **SURVIVES 14/14**. With a NaN
at index −30:

```
BASELINE  volume_anom_present=False   WARNING logged? True
B6        volume_anom_present=False   WARNING logged? False   all logs=[]
```

That is exactly the silent-suppression family the guard exists to prevent, reintroduced
by a plausible refactor. The only fixture puts the NaN at the last row, inside both
windows. Named fix: a second fixture with the NaN at ~−30.

## 8. WARN — the EIGHTH vacuous guard, same family as the one it replaced

`test_volume_anomaly_uses_the_metric_key_not_type` (`:159-171`) is a bare
`for a in out.get("anomalies", [])` loop with **no non-emptiness assertion**. Measured:

- `C1_NEVER_APPEND` (`_append_if_anomalous` never appends) **SURVIVES 14/14** — a
  module reporting zero anomalies for every input passes the entire suite.
- `C2_KEY_BECOMES_TYPE` — re-injecting the exact cycle-1 drift — the key-pin test
  **PASSES**. The module raises `KeyError: 'metric'` at `:320`, the bare `except`
  returns the ERROR payload with no `anomalies` key, and the loop runs **0 times**.
  The mutation is killed only incidentally, by the two tests asserting `signal != "ERROR"`.

So the test written to close cycle 1's tautology **does not detect the drift it was
written to detect** — assertion never evaluated, empty iteration instead of wrong key.
Per §4c this is WARN, not blocking: it sits alongside a genuinely non-vacuous criterion-1
guard. Named fix: `assert anomalies, "payload had no anomalies -- the key pin never ran"`
before the loop. `violation_type: Circular_Reasoning`.

**Not findings** (survivors with no correctness differential): `B8_DROP_OPEN_FROM_SUBSET`
— `Open` is never read and alignment is preserved (agrees with cycle 1's A9 disposition).
`C3_SEVERITY_FLAT` / `C4_THRESHOLD_10X` survive but are pre-existing scoring behaviour
outside this step's scope.

## 9. Claims re-derived and UPHELD

| Claim | My measurement |
|---|---|
| 11/11 mutations killed | **Reproduced** — all 11 labels killed in fresh processes |
| A1 → "4 failed, 10 passed" incl. the alignment test | Reproduced exactly |
| Suite 14 passed / cross-suite 76 passed / ruff clean | Reproduced exactly (exit 0 read bare) |
| Fixture mutations `A5_FIXTURE_NOBAD`, `A6_FIXTURE_FLOATVOL` | Both KILLED — I mutated the **test fixture**, not just the code |
| Criterion 4 | **MET** — 6/6 live tickers: no exception, `_has_non_finite` False, `json.dumps` clean, all four lengths equal, no ERROR |
| Criterion 2 | **MET** — §4 makes both required statements and queues the residual rather than hand-waving |
| Scope / 80.27 collision | **None** — `git diff --name-only HEAD` is the authority: `anomaly_detector.py` only; `:16-18` thresholds and every deferred ladder untouched (0 deleted lines outside the two regions) |
| No-dark-flag ruling | **UPHELD independently.** This DISCARDS an unusable observation rather than RESTORING suppressed values (80.27's hazard). No trading consumer: only `orchestrator.py:1267-1269/:1990/:2034`, `api/signals.py:116/:210`, and `api/analysis.py:238` behind an `isinstance(..., list)` guard. The **new ERROR path does not change the calculus** — it cannot fire from the production helper path, 6/6 live confirm no trip, its `{ticker,signal,summary,data}` shape matches `rules/backend-tools.md`, and the pre-existing "Insufficient data" return already omits `anomalies` |

## 10. Verdict wiring

Both cycle-1 blockers are cleared **in code and tests**, verified by my own execution.
The product is correct, lint-clean, live-verified, collision-free. What fails is the
**record**: the GENERATE artifact and live_check are cycle-1 documents whose criterion-1
evidence cell points at a deleted test and whose "verbatim" captures report a superseded
run. A step is not done until every criterion is demonstrably COVERED in the evidence,
not merely covered in the tree. That caps the verdict at CONDITIONAL — not FAIL, because
criterion 1 is materially addressed in the shipped code.

**To clear on cycle 3** (update evidence, then a fresh Q/A):
1. Regenerate `experiment_results_80.31.md` — re-map criterion 1 to
   `test_module_arrays_are_equal_length_and_index_aligned`, criterion 3 to
   `test_end_to_end_alignment_invariant_is_enforced_at_runtime`, correct 11→14 tests,
   5/5→11/11, the `:55-69` range, withdraw the A4-equivalent claim, five→seven, and
   document the runtime ERROR path + NaN-volume guard in §6 and §8.
2. Regenerate `live_check_80.31.md` with the actual 14-passed run and the 11-mutation matrix.
3. Add `assert anomalies` to the key-pin test (§8).
4. Add a NaN-at-−30 volume fixture (§7).
5. Optional: rename the runtime check to what it is (a length check), or extend it to
   pin positional correspondence (§6).

---

# Cycle 3 — Main's follow-up (evidence CHANGED; fresh Q/A follows)

All findings accepted. The BLOCKER was a process failure on my part, and the three WARNs
were all live.

| # | Finding | What I did | Verified |
|---|---|---|---|
| **§5 BLOCKER** | The GENERATE artifact was never updated — it still cited a DELETED test as criterion 1's only evidence, "11 passed", "5/5 killed", `:55-69`, and the withdrawn equivalent-mutant claim | **`experiment_results_80.31.md` rewritten from measurement**, not edited: 16 tests, 78 cross-suite, 9/9 cycle-3 matrix, real diff hunks (`:21-44`, `:81-109`, `:119-166`, `:188-208`), the two NEW production behaviours documented in §6, A4 withdrawal recorded, instance count corrected. `live_check_80.31.md` likewise. | `grep` for the stale counts → 0 |
| **§6 WARN** | The "alignment invariant" enforced LENGTH, not alignment — `np.roll` survived 14/14 with three metrics silently vanishing | The invariant now **also spot-checks both endpoints against the source frame**, which is the cheapest check that actually pins position. **Made NaN-tolerant** (`nan == nan` is False) — I caught that when the NaN-volume test went red, i.e. my first version of the strengthened guard had a real bug | `B10_ROLL_CLOSE` **KILLED**, `B11_ROLL_HIGH` **KILLED**, plus `test_a_same_length_roll_is_caught_by_the_invariant` |
| **§7 WARN** | The NaN-volume guard's WINDOW was unpinned — narrowing `[-60:]` → `[-5:]` survived, reintroducing silent suppression | Added `test_nan_volume_mid_window_is_also_caught` with the NaN at ~−30: inside the 60-day baseline, outside the 5-day window | `B6_VOLGUARD_WINDOW5` **KILLED** |
| **§8 WARN** | The EIGHTH vacuous guard — the key-pin test was a bare loop with no non-emptiness assertion, so it ran **zero** times; a module that never appends passed the whole suite | Added `assert anomalies, "...the key pin never executed"` before the loop | `C1_NEVER_APPEND` **KILLED** |

**Cycle-3 matrix — 9/9 killed**, including all four of your cycle-2 survivors:

```
B10_ROLL_CLOSE  B11_ROLL_HIGH  B6_VOLGUARD_WINDOW5  C1_NEVER_APPEND
A1_PERCOLUMN  A11_SHIFT_VOL  A15_INVARIANT  A14_VOLNAN  A5_FIXTURE
-> 9/9
```

Suite **16 passed**; cross-suite **78 passed**; ruff clean.

**On §5, plainly.** You are right that CLAUDE.md's cycle-2 flow names `experiment_results.md`
explicitly, and that the `live_check` gate helper only checks the file EXISTS — so a
superseded capture would have sailed through the auto-commit hook. I updated the critique
and the code and left the GENERATE artifact describing a state that no longer existed,
including citing a test I had deleted as criterion 1's only evidence. That is the same
failure the artifacts keep warning about — a claim that was true when written and never
re-derived — applied to my own record rather than to code.

**On §8, plainly.** The key-pin test was written *specifically* to close cycle 1's
`"type"`/`"metric"` tautology, and reproduced the same family: an assertion that never
evaluates. Three derived rules are now in `feedback_mutation_test_guards_and_fixtures`:
assert the key EXISTS before asserting on its value; assert the collection is NON-EMPTY
before iterating it; and test the ENTRY POINT, not only the seam — a mutation that bypasses
your seam passes your test.

**Not contested:** `B8_DROP_OPEN_FROM_SUBSET` (no correctness differential — `Open` is
never read and alignment is preserved), `C3`/`C4` (pre-existing scoring behaviour outside
this step). `B5_INV_DROP_CLOSE` — accepted as an unpinned operand chain; the endpoint check
now covers all four arrays at both ends, which narrows it, and I am not claiming it is
fully pinned.

---

# Cycle 3 — Q/A verdict

*(Transcribed VERBATIM by Main from the qa subagent's returned final message.
The agent deliberately did NOT write this file itself — `qa.md` forbids Edit/Write
for the Q/A role, and CLAUDE.md makes Main the scribe. Main recorded it; Main did
not author it.)*

I did not write to `handoff/current/evaluator_critique_80.31.md` — `qa.md` forbids Edit/Write for this role ("NEVER Edit or Write… You stay read-only and never write files"), and CLAUDE.md makes Main the scribe who transcribes the returned verdict verbatim. The full text below is the cycle-3 section to append. Working tree is byte-identical to when I started (`+122/−9`, same `git status`); every mutation ran on temp copies in a fresh subprocess.

# Cycle 3 — Q/A verdict

**VERDICT: PASS** — `ok: true`, `certified_fallback: false`.

**One sentence:** the cycle-2 BLOCKER is genuinely cleared — I re-derived every number in both rewritten artifacts and all of them reproduce — and all four cycle-2 survivors now die; I found a ninth vacuous guard, but it is redundant coverage alongside a criterion-1 test I proved non-vacuous, which is WARN under §4c, not blocking.

## 0. The 3rd-CONDITIONAL counter — stated explicitly

`grep -cF "80.31" handoff/harness_log.md` → **0** (correct log-last ordering; masterplan status still `pending`, `retry_count: 0`). The counter therefore lives in `evaluator_critique_80.31.md`: **cycle 1 CONDITIONAL, cycle 2 CONDITIONAL, no intervening PASS or FAIL.** This is the third pass, so **CONDITIONAL is not available to me** — per `qa.md` Constraints and runbook §4, a third consecutive CONDITIONAL must be returned as FAIL.

**The rule does not bind on this pass, because I am not issuing a CONDITIONAL.** All four criteria are MET with covering evidence I reproduced myself; the one new finding is WARN-level under the documented wiring (it is not sole coverage for any criterion). I am neither softening a CONDITIONAL into a PASS nor inflating a PASS into a FAIL to satisfy the counter. If the operator wants "no known-vacuous guard may ship" that is a doctrine change, not a reading of the rule as written.

## 1. Harness compliance — PASS

| Check | Result |
|---|---|
| Researcher before contract | PASS — brief 21:02:42 → contract 21:04:16 (`stat -f %Sm`) |
| Research gate | PASS — `gate_passed: true`, 5 sources in full, 17 URLs, recency scan true, 8 internal files (`research_brief_80.31.md:562-577`); contract cites the brief |
| Criteria verbatim | PASS — all 4 criteria **and** the immutable command are byte-identical substrings of `contract_80.31.md` (checked programmatically against `.claude/masterplan.json`) |
| Log-last | PASS — 0 entries for `80.31`; status `pending` |
| No self-eval / no verdict-shopping | PASS — evidence genuinely changed: code+tests 21:37:17 (cycle 2 was 21:23:56), artifacts rewritten 21:38:37 / 21:39:13. Documented cycle-2 flow |
| Tree untouched by me | PASS — mutations run on temp copies in fresh subprocesses; post-run `git status` and `+122 / −9` identical to pre-run |

## 2. Deterministic checks

```
$ .venv/bin/python -m pytest backend/tests/test_phase_80_31_anomaly_array_alignment.py -q
................                                                         [100%]
16 passed in 0.32s

$ .venv/bin/python -m pytest <80.31> <80.27> <80.1> <80.2> -q
78 passed, 40 warnings in 2.53s
```

Lint on the **derived** scope (`git diff --name-only HEAD -- '*.py'` + untracked, non-empty guard asserted, piped via `xargs` so zsh cannot lint zero files):

```
SCOPE:
backend/tools/anomaly_detector.py
backend/tests/test_phase_80_31_anomaly_array_alignment.py
All checks passed!    ruff exit=0
```

Immutable verification command, re-run live: `rows 251 close 250 volume 251` — exact reproduction.

**No new gate finding.** Under ruff's broader default set there are 7 findings; I ran the same command against `git show HEAD:backend/tools/anomaly_detector.py` and confirmed 3 are **pre-existing** (`SIM102` ×2, `BLE001`). The 2 new ones are informational and outside the gate's select: `PLR0124` at `:137` (`a != a` — the deliberate NaN idiom) and `RUF059` at test `:151` (unused unpacked `high`/`low`).

## 3. §5 BLOCKER — CLEARED. Every number re-derived, none stale

I did not read these numbers; I re-derived each one.

| Artifact claim | My re-derivation |
|---|---|
| `+122 insertions / −9 deletions` | `git diff --stat` → **exact** |
| hunks `:21-44`, `:81-109`, `:119-166`, `:188-208` | `git diff -U0` → `+21,24`, `+81,29`, `+119,48`, `+188,21` → **all four exact** |
| "**16 tests**" | `grep -c "^def test_"` → **16** |
| "16 passed" / "78 passed" | **reproduced exactly** |
| live_check `:55` progress line | **16 dots for 16 passed** — internally consistent, not spliced |
| 9/9 matrix | **re-ran all nine labels independently in fresh subprocesses — 9/9 KILLED** |
| criterion 1's cited test exists | `test_module_arrays_are_equal_length_and_index_aligned` present at `:112`; the two deleted tests grep to **0** |
| §B lengths AAPL/MSFT/NVDA `251/250/251/250` | **reproduced live on all three** |
| §C `AAPL ANOMALY_OPPORTUNITY`, `MSFT NORMAL … anomalies=[]` | **reproduced** (MSFT `anomaly_count: 0`) |
| Δz **+0.047 … +0.338** across 6 tickers | **reproduced to 4 dp** — MSFT `+0.0469` is the min, MU `+0.3376` the max; AAPL `+0.1089`, NVDA `+0.1457`, AMD `+0.0679`. My out-of-set GOOGL is `−0.0139`, consistent with the brief's own "not systematically signed" caveat |
| consumer set (`signals.py:116/:210`, `orchestrator.py:1267/:1990/:2034`) | **reproduced by derived grep**; the only other hits are two inventory scripts (`scripts/audit/*`), not payload consumers. No paper-trading / screener / optimizer path |
| 80.27 collision "none" | **reproduced** — the diff deletes exactly 9 lines, all inside the two claimed regions; ladders at `:31/:38/:188/:204/:210/:217/:230` and thresholds `:16-18` untouched |
| A4 equivalent-mutant claim | **withdrawn** in both artifacts (`experiment_results §6`, `live_check §D`) |
| the two NEW production behaviours | **documented** — `experiment_results §6.2/§6.3/§9`, `live_check §D` |

The ordering failure is also fixed: both artifacts now **postdate** the code (21:38/21:39 vs 21:37), where cycle 2's predated it.

## 4. §6 WARN (roll) — CLEARED, then attacked further

`B10_ROLL_CLOSE` **KILLED**, `B11_ROLL_HIGH` **KILLED**, both via `test_end_to_end_alignment_invariant_is_enforced_at_runtime`.

I then attacked the new endpoint check itself:

| Mutation | Result |
|---|---|
| `B20_ROLL2_CLOSE` (`np.roll(close, 2)`) | **KILLED** |
| `B24_POSCHECK_OFF` (`_pos_ok = _len_ok`) | **KILLED** by `test_a_same_length_roll_is_caught_by_the_invariant` |
| `B25_SAME_ALWAYS_TRUE` (`_same` → always True) | **KILLED**, same test |
| `B31_TUPLE_DROP_CLOSE` | **KILLED**, same test |
| `B21_MIDSWAP_CLOSE` / `B21b_MIDROLL_INNER` | SURVIVE — interior permutation, outside an endpoint spot-check by design |
| `B26_NANTOL_OR` (`a != a or b != b`) | SURVIVES |
| `B29`/`B30` (single endpoint), `B32`/`B33` (drop Volume / High+Low from the tuple) | SURVIVE |

The endpoint check is genuinely load-bearing (B24/B25/B31 die), and the kill is correctly attributed — `np.roll` preserves length, so the length half cannot account for it.

**Does the NaN-tolerance open a hole?** Measured, no reachable one. An all-NaN OHLC column empties the frame → "Insufficient data". An all-NaN `Volume` passes the endpoint comparison vacuously (both NaN) but is then caught by the NaN-volume guard, which I watched fire. `_same` only short-circuits when **both** sides are NaN; one-sided NaN still falls through to `bool(a == b)` → False → detected. `B26` widens that to either-side, but the differential requires an array carrying a NaN its source frame does not — unreachable from a helper that is four `.to_numpy()` reads of one frame.

**Can it false-trip on healthy live data?** No — measured, not reasoned. 8 live tickers across US/EU/KR (AAPL, MSFT, NVDA, AMD, MU, GOOGL, SAP.DE, 005930.KS): **zero** ERROR returns, **zero** WARNING logs, signals and anomaly counts identical old vs new (0 flips 8/8). SAP.DE (254 rows) and 005930.KS (243) have no malformed row at all today, which independently explains their Δz of exactly 0.0000.

> Method correction, disclosed: my first live probe patched `NEW.yf.Ticker`, which is the same module object it was fetching with, so tickers 2-8 silently re-ran AAPL's frame. The table above is the corrected run with the real constructor captured first. The tell was eight identical rows across inputs that must differ.

## 5. §7 WARN (window) — CLEARED, with a measured residual

`B6_VOLGUARD_WINDOW5` **KILLED** by `test_nan_volume_mid_window_is_also_caught`. `B34_WINDOW29` also **KILLED**, so the window is pinned to >29.

`B27_WINDOW59` and `B28_WINDOW61` **SURVIVE**, and B27 has a real differential: with a NaN at **exactly index −60**, the guard as written fires (I captured the WARNING verbatim), but a refactor to `volume[-59:]` would miss that index while the scoring window `volume[-60:]` still includes it → `std_vol` NaN → `_z` None → **silent suppression**, the exact family the guard exists to prevent. One index out of sixty, implausible refactor shape. WARN-level residual. Named fix: a fixture with the NaN at exactly −60, or derive guard and scoring windows from one constant.

## 6. §8 WARN (key pin) — CLEARED, and the kill is correctly attributed

`C1_NEVER_APPEND` **KILLED** by the key-pin test itself (`assert anomalies`). I then checked the pin does more than catch the crash: `C5_CONSISTENT_RENAME` renames `"metric"` → `"kind"` in **both** the writer and the `a["metric"]` reader, so nothing raises and the payload carries non-empty anomalies — and `assert "metric" in a` **KILLS** it. The pin detects a genuine silent key drift, not just an incidental `KeyError`.

## 7. The NINTH vacuous guard — found, WARN

`test_malformed_session_is_absent_from_every_array` (`backend/tests/test_phase_80_31_anomaly_array_alignment.py:141-156`). Its docstring says it is "the original defect stated positively: `volume` used to retain the bar that `close`/`high`/`low` had lost." It does not detect the original defect.

The test performs the module's own cleaning step itself —

```python
cleaned = frame.dropna(subset=["Open", "High", "Low", "Close"])
close, volume, high, low = anomaly_detector._aligned_ohlcv_arrays(cleaned)
assert bad_volume not in set(volume.tolist())
```

— so the malformed volume is absent **by construction**, whatever the module does. Measured:

```
TEST path    : volume from helper(cleaned) contains the malformed bar?  False
A1 PRODUCTION: volume actually used by the module contains it?          True
               len(close)=119  len(volume)=120

A1_PERCOLUMN  exit=1 -> 6 failed, 10 passed
  test_malformed_session_is_absent_from_every_array  in the failures?  False
```

Under `A1_PERCOLUMN` — the original defect restored, production's volume array genuinely carrying the malformed bar — the test named for that defect **passes**. Vacuity shape **#7** (re-implemented: it executes the test's own copy of the dropna) compounded with **#4** (assertion true by construction). Its *other* assertion (`len(close) == len(frame) - 1`) is falsifiable (`A5_FIXTURE` kills it), but that one is about the fixture's arithmetic, not the module's.

`violation_type: Circular_Reasoning`.

**Why WARN and not blocking, per §4c:** it is not sole coverage. Criterion 1 is carried by `test_module_arrays_are_equal_length_and_index_aligned`, which I proved non-vacuous — it kills `A11_SHIFT_VOL` (the original defect's exact shape), `DN_HELPER_IDENTITY` and `SANITY_HELPER_BREAK`. Criterion 3 is carried by `test_end_to_end_alignment_invariant_is_enforced_at_runtime`, killed by `A1`, `B10`, `B11`, `B20`. **Named fix:** drive `get_anomaly_scan` with the RAW frame and spy on the extractor's return, so the module's dropna is the thing under test.

## 8. Vacuity census — 16/16 tests are demonstrably killable

I required every test in the file to fail under at least one mutation; three had not failed under anything, so I built mutations for them.

| Test | Killed by |
|---|---|
| `test_fixture_reproduces_the_exact_yfinance_shape` | `A5_FIXTURE_NOBAD` |
| `test_fixture_without_bad_tail_is_clean` | `V1_FIXTURE_CLEAN_DIRTY` |
| `test_the_defect_shape_is_real_percolumn_dropna_is_asymmetric` | `A5_FIXTURE_NOBAD` |
| `test_module_arrays_are_equal_length_and_index_aligned` | `A11_SHIFT_VOL`, `DN_HELPER_IDENTITY` |
| `test_malformed_session_is_absent_from_every_array` | `A5_FIXTURE_NOBAD` (fixture assertion only — see §7) |
| `test_volume_anomaly_uses_the_metric_key_not_type` | `C1_NEVER_APPEND`, `C5_CONSISTENT_RENAME` |
| `test_a_nan_volume_suppresses_the_volume_anomaly_explicitly` | `A14_VOLNAN_GUARD_OFF` |
| `test_a_high_only_nan_row_is_also_excluded` | `A1_PERCOLUMN` |
| `test_payload_has_no_non_finite_values` | `V2/V3/V4_*_NAN` |
| `test_payload_is_json_serialisable` | `V2/V3/V4_*_NAN` (numpy NaN too — `V3` kills it) |
| `test_clean_frame_still_works` | `B10`, `C2`, `V2` |
| `test_sufficiency_guard_counts_usable_rows_not_raw_rows` | `A1`, `A5` |
| `test_end_to_end_alignment_invariant_is_enforced_at_runtime` | `A1`, `B10`, `B11`, `B20` |
| `test_runtime_invariant_actually_catches_a_desync` | `A15_INVARIANT_OFF` |
| `test_nan_volume_mid_window_is_also_caught` | `B6`, `B34` |
| `test_a_same_length_roll_is_caught_by_the_invariant` | `A15`, `B24`, `B25`, `B31` |

Two do-nothing implementations were also killed: `DN_STUB_RETURN_EMPTY` (7 failures) and `DN_HELPER_IDENTITY` (6). Per Goodenough-Gerhart this licenses only "these mutations were killed" — not a global no-vacuous-guards claim.

## 9. Criteria

| # | Verdict | Basis |
|---|---|---|
| 1 | **MET** | `test_module_arrays_are_equal_length_and_index_aligned` asserts the four lengths equal (`:125`, the criterion's literal words) plus positional correspondence at four indices, on the trailing-NaN-OHLC/real-Volume fixture; non-vacuous by mutation |
| 2 | **MET** | §4 makes both required statements — the malformed bar is a completed session excluded by the row-wise dropna, and the genuinely in-progress session is disclosed as unaddressed and queued rather than hand-waved |
| 3 | **MET literally** | `A1_PERCOLUMN` exit=1, and the failures include `test_end_to_end_alignment_invariant_is_enforced_at_runtime` — an alignment test. Reproduced exactly |
| 4 | **MET** | 8/8 live tickers: no exception, `_has_non_finite` False, `json.dumps` clean, no ERROR, no WARNING, 0 verdict flips, identical anomaly counts old vs new |

**No-dark-flag ruling — UPHELD independently, including for the two new paths.** Both new behaviours are fail-safe: they *withhold* output and cannot manufacture an anomaly. Neither fires on healthy live data (measured 8/8). The ERROR return matches the `{ticker, signal, summary, data}` shape in `rules/backend-tools.md`, and the pre-existing "Insufficient data" return already omits `anomalies`. The step discards an unusable observation rather than restoring suppressed ones, and no trading path consumes the tool. Two new production behaviours do not change the calculus.

**Note (not a finding):** §8's tally rows #2 and #3 look like one artifact counted twice — the shipped `max(seen) <= cleaned_rows` spy and "the first attempt at the A2 closure" are the same assertion shape, and cycle 1 identified them as the same weakness. §8 discloses this itself ("same spy"), it bears on no criterion, and the section enumerates its members rather than asserting an unmeasured total.

## 10. Verdict wiring

Worst-of-lenses (applied although this is P2, non-money-path): **correctness** — code right, verified live on 8 tickers → PASS. **Reproducibility** — every artifact number re-derived, 9/9 matrix independently re-run → PASS. **Scope honesty** — residuals disclosed, the equivalent-mutant claim withdrawn, both new production paths documented, Main explicitly declines to claim the operand chain is fully pinned → PASS. Worst = **PASS**.

Open WARN-level residuals for a follow-up, none blocking: the §7 ninth vacuous guard (named fix above), the `volume[-60:]` boundary at exactly −60, interior permutation and the partly-unpinned endpoint operand chain (`B21`/`B26`/`B29`/`B30`/`B32`/`B33`).

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 4 immutable criteria MET with covering evidence I re-derived rather than read: 16 passed / 78 cross-suite / ruff clean on the derived scope; the 9/9 mutation matrix re-run independently in fresh subprocesses; all four cycle-2 survivors (B10, B11, B6, C1) now KILLED and correctly attributed; A1_PERCOLUMN killed with an alignment test among the failures (criterion 3 literally); criterion 4 verified on 8 live tickers with zero false-trips of either new production path. The cycle-2 BLOCKER is cleared -- both artifacts were rewritten from measurement and now postdate the code. One new WARN: test_malformed_session_is_absent_from_every_array cleans the fixture itself, so its headline assertion is true by construction and it passes under A1_PERCOLUMN -- redundant coverage, not sole coverage, hence WARN per qa.md 4c. 3rd-CONDITIONAL counter stands at 2 prior CONDITIONALs; the rule does not bind because this is a PASS, not a third CONDITIONAL.",
  "violated_criteria": [],
  "violation_details": [
    {
      "violation_type": "Circular_Reasoning",
      "action": "test_malformed_session_is_absent_from_every_array (:141-156) performs frame.dropna(subset=[...]) itself, then asserts bad_volume not in _aligned_ohlcv_arrays(cleaned)",
      "state": "WARN, non-blocking. Under A1_PERCOLUMN the module's production volume array carries the malformed bar (len(close)=119, len(volume)=120) and this test PASSES; measured 6 failed, 10 passed, this test not among the failures. Criterion 1 remains covered non-vacuously by test_module_arrays_are_equal_length_and_index_aligned (kills A11_SHIFT_VOL, DN_HELPER_IDENTITY, SANITY_HELPER_BREAK)",
      "constraint": "qa.md 4c -- a guard that cannot fail when its subject is broken does not count; vacuity shapes #7 (re-implemented) + #4 (tautology). Sole-coverage vacuity blocks; alongside a genuine guard it is WARN with a named fix"
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit",
    "criteria_verbatim_vs_masterplan",
    "log_last_ordering",
    "third_conditional_counter",
    "syntax",
    "pytest_in_suite_16",
    "pytest_cross_suite_78",
    "ruff_derived_scope_gate",
    "ruff_default_ruleset_baseline_diff",
    "immutable_verification_command",
    "diff_stat_and_hunk_rederivation",
    "mutation_matrix_9of9_reproduced",
    "mutation_endpoint_check_attack_11",
    "mutation_vacuity_census_16of16",
    "do_nothing_implementation_probe",
    "live_yfinance_8_tickers_criterion4",
    "false_trip_probe_both_new_paths",
    "delta_z_claim_rederivation",
    "consumer_grep_derived",
    "collision_scope_git_diff_authority",
    "artifact_mtime_ordering",
    "working_tree_integrity"
  ]
}
```