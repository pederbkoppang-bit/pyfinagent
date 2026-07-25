# Experiment Results — phase-80.31

**Step:** `80.31` (P2) — misaligned price/volume arrays in the anomaly detector.
Date 2026-07-25. Contract: `contract_80.31.md`. Gate: `research_brief_80.31.md`
(`gate_passed: true`, 5 sources in full, 17 URLs, recency scan, 8 internal files).

> **THIS ARTIFACT IS THE CYCLE-3 STATE.** Q/A cycle 2 raised a BLOCKER because the
> cycle-1 version of this file was never updated after the cycle-2 code changed — it still
> cited a test that had been deleted, "11 passed", and a 5/5 matrix. That was a real
> process failure (CLAUDE.md's cycle-2 flow names `experiment_results.md` explicitly), and
> it is why every number below is re-derived rather than carried forward.

## 1. What was built

`backend/tools/anomaly_detector.py` — `122 insertions(+), 9 deletions(-)`, in four hunks:
`:21-44` (the extraction helper), `:81-109` (the row-wise dropna), `:119-166`
(the runtime invariant), `:188-208` (the NaN-volume guard).

**1. One row-wise drop instead of four per-column drops.**
`hist = hist.dropna(subset=["Open","High","Low","Close"])`, placed **before** the
`len(hist) < 20` sufficiency guard so the check counts *usable* rows. `Volume` is
deliberately excluded from the subset (a legitimately zero/absent volume must not discard
an otherwise-good price bar); `Open` is deliberately included (it matches the measured
malformed shape and is the more conservative choice).

**Why the defect was guaranteed, not occasional:** `Volume` is `int64`, which cannot hold
NaN, so `hist["Volume"].dropna()` was a *structural no-op* while the three price columns
lost the malformed bar. Measured on AAPL/MSFT/NVDA: **251 raw rows → close/high/low/`Open`
250, volume 251.**

**2. A named extraction helper `_aligned_ohlcv_arrays(hist)`** (`:21-44`), so the
alignment invariant has a single directly-testable seam. Added in cycle 2 because four
deliberate shift mutations survived a suite that only tested the caller.

**3. A RUNTIME alignment invariant** (`:119-166`). Unequal lengths — or endpoints that do
not match the source frame — return `signal: "ERROR"` with an ERROR log rather than
scoring across mismatched sessions. A missing anomaly scan is recoverable; a silently
wrong one is not.
- Cycle 2 correctly observed the first version enforced **length only**, and `np.roll`
  (same length, every session desynced) survived. It now spot-checks **both endpoints
  against the frame**, which is the cheapest check that actually pins position.
- The endpoint comparison is **NaN-tolerant** (`nan == nan` is `False`), or a legitimately
  absent volume would be misread as a desync — a bug I introduced and caught when
  `test_a_nan_volume_suppresses_the_volume_anomaly_explicitly` went red.

**4. An explicit NaN-volume guard** (`:188-208`). Taking `Volume` out of the drop subset
created a NaN-into-the-volume-window path that did not exist before: `std_vol` goes NaN,
`_z` returns `None`, and the anomaly vanishes **silently** — the phase-80.27 family,
introduced by me while fixing its sibling. The block now checks
`np.isfinite(volume[-60:]).all()` and skips with a **WARNING** instead.

**The mechanism correction:** the malformed bar is a **COMPLETED** session (Fri
2026-07-24), not the still-forming one the step text describes. Measured by me on a
Saturday with markets closed, independently reproduced by the researcher, corroborated
upstream by **yfinance issue #2622**.

### Files

| File | Δ |
|---|---|
| `backend/tools/anomaly_detector.py` | +122 / −9 — helper, row-wise dropna, runtime invariant, NaN-volume guard |
| `backend/tests/test_phase_80_31_anomaly_array_alignment.py` | **new**, **16 tests** |

## 2. Verification — re-derived, not carried forward

```
pytest 80.31                                  ->  16 passed
pytest 80.31 + 80.27 + 80.1 + 80.2            ->  78 passed
ruff --select F401,F811,F821 (derived scope)  ->  All checks passed!
mutation matrix (cycle 3)                     ->  9/9 killed
```

## 3. Criteria → evidence

| # | Criterion | Evidence | Status |
|---|---|---|---|
| 1 | four arrays equal-length AND index-aligned, with a trailing NaN-OHLC/real-Volume fixture | `test_module_arrays_are_equal_length_and_index_aligned` — drives the **production** helper, asserts `len(close)==len(volume)==len(high)==len(low)` (the criterion's literal words) **plus** positional correspondence against the frame at four indices. Backed end-to-end by the runtime invariant and `test_a_same_length_roll_is_caught_by_the_invariant` | **MET** |
| 2 | volume z-score over completed sessions only; the in-progress choice stated explicitly | §4 — two statements | **MET** |
| 3 | MUTATION-TEST: restore the per-column dropna, confirm **the alignment test** FAILS | `A1_PERCOLUMN` **KILLED**, and the failures now include `test_end_to_end_alignment_invariant_is_enforced_at_runtime` — an alignment test. (Cycle 1 correctly flagged that A1 was previously killed only by ORDER and SUBSET-WIDTH tests.) | **MET** |
| 4 | still returns 200 / serialises cleanly | `_has_non_finite(payload) is False` + `json.dumps` clean, live and in tests. **The two NEW production paths are disclosed in §6** | **MET** |

## 4. Criterion 2 — the two statements it requires

**(i) The malformed bar is a COMPLETED session, and the row-wise dropna excludes it**, so
the volume z-score is computed over completed sessions only. **Discarding that session's
real `int64` volume is accepted deliberately** — a volume without its matching OHLC cannot
be positioned in the series, and a mis-positioned real number is worse than one fewer
observation.

**(ii) The genuinely in-progress session is a SEPARATE issue this fix does not address**,
and no special case is added. During a live session yfinance returns a partial bar with
*no* NaN, so row-wise dropna cannot see it. Every close-based metric already treats the
last bar as final, and an in-session guard needs a market-calendar dependency the module
lacks. **Queued.**

## 5. Mutation matrix — 9/9 killed (cycle 3)

```
[B10_ROLL_CLOSE      ] KILLED   np.roll(close,1) after the helper      <- Q/A c2 survivor
[B11_ROLL_HIGH       ] KILLED   np.roll(high,1) after the helper       <- Q/A c2 survivor
[B6_VOLGUARD_WINDOW5 ] KILLED   NaN-volume guard window 60 -> 5        <- Q/A c2 survivor
[C1_NEVER_APPEND     ] KILLED   _append_if_anomalous never appends     <- Q/A c2 survivor
[A1_PERCOLUMN        ] KILLED   criterion 3: restore per-column dropna
[A11_SHIFT_VOL       ] KILLED   volume shifted by one (original defect shape)
[A15_INVARIANT       ] KILLED   disable the runtime invariant
[A14_VOLNAN          ] KILLED   remove the NaN-volume guard
[A5_FIXTURE          ] KILLED   FIXTURE: remove the malformed row
```

Earlier cycles' mutations (A2 narrow-subset, A3 order, A6 fixture-float-volume,
A10/A12/A13 shifts) were killed in the cycle-2 run and are unaffected by the cycle-3
changes.

## 6. Scope honesty — three things this step changed beyond the one-line fix

1. **NOT byte-identical.** Δz **+0.047 … +0.338** across 6 tickers; 0 verdict flips on 6/6
   today, but a flip near the 1.5/2.0 boundary is possible. `volume_5d_vs_60d` is not in
   `risk_metrics`, so a newly-firing positive-z anomaly reads as *opportunity* — the
   less-conservative direction.
2. **A NEW `ERROR` return path** (the runtime invariant). It cannot fire on well-formed
   data — `test_end_to_end_alignment_invariant_is_enforced_at_runtime` asserts a clean
   frame does not trip it, and the endpoint check is NaN-tolerant so an absent volume does
   not either.
3. **A NEW suppression path** (the NaN-volume guard) — but it replaces a *silent*
   suppression with a logged one, so it is strictly more visible than what it changed.

**Why still no dark flag, unlike 80.27:** this fix **discards** a value (the malformed
bar's real volume) rather than **restoring** suppressed ones; `_has_non_finite(payload)`
is `False` before and after; and no paper-trading, screener or optimizer path consumes the
tool — it is an LLM-debate input only (`signals.py:116/:210`,
`orchestrator.py:1267/:1990/:2034`). A flag would preserve a known-wrong computation as
the default. **Both new paths above are fail-safe** (they withhold output; they never
manufacture an anomaly).

**Withdrawn:** cycle 1 declared `subset=[..., "Volume"]` an *equivalent mutant* on the
grounds that `int64` cannot hold NaN. Q/A cycle 2 refuted it with a float64-Volume input
that produces a different top-level signal. **Finiteness is not equivalence.** It is a
survivor with nil live reachability (Volume is `int64` with 0 NaN on every market this
project trades), not an equivalent mutant.

## 7. Out of scope — queued, not fixed

- **D1 — the √5 units mismatch** (`:91-94` pre-diff): the `std` of *daily* volume is used
  to z-score a *5-day mean*, so |z| is systematically ≈**2.24× too small** — an order of
  magnitude larger than the alignment offset it sits beside. Suppressive; fixing it fires
  MORE = less conservative ⇒ its own gate.
- **D2** — the self-overlapping baseline. Also suppressive; arXiv:2004.04013 finds window
  overlap can *reduce* bias, so not a reflex fix.
- **D3** — the genuinely in-progress session (§4 ii).
- **D4 — CLOSED BY THIS STEP.** "high/low align with close only coincidentally" is now
  structurally impossible; `test_a_high_only_nan_row_is_also_excluded` pins it.

## 8. The vacuous-guard tally — this step produced five of them

Recorded because the pattern, not the individual bugs, is the transferable finding.

| # | Guard | Why it could not fail |
|---|---|---|
| 1 | the 50×-spike "behavioural differential" | read `a.get("type")`; the module writes `"metric"`, so the set was `{None}` and the assertion was **true for every implementation**. Its documented premise was also inverted |
| 2 | `max(seen) <= cleaned_rows` via an `np.mean` spy | the largest window ties the bound by construction; never failed under 12 mutations |
| 3 | first attempt at the A2 closure | same spy, `60 <= 119` passed either way |
| 4 | the helper-only alignment test | A1 bypasses the helper entirely, so it passed under the very defect it names |
| 5 | the key-pin test written to close #1 | bare `for` loop with no non-emptiness assertion → ran **zero** times; a module that never appends an anomaly passed the whole suite |

**#1 and #5 are the instructive pair:** #5 was written specifically to close #1, and
reproduced the same family — an assertion that never evaluates. I caught #3 myself; Q/A
caught the rest. Derived rules now in `feedback_mutation_test_guards_and_fixtures`:
**assert the key exists before asserting on its value**, **assert the collection is
non-empty before iterating it**, and **test the entry point, not only the seam** — a
mutation that bypasses your seam passes your test.

## 9. DO-NO-HARM

| Item | Status |
|---|---|
| Live book | Cannot move — no paper-trading/screener/optimizer path reads this tool |
| 80.27 collision | **None.** `:31`, `:38`, `:188`(pre-diff), `:204`, `:210`, `:217`, `:230` ladders and the `:16-18` thresholds untouched |
| `.env` / flags / optimizer | No edit, no flag, no run; `historical_macro` FROZEN |
| Kill-switch / stops / sector caps / DSR / PBO | Not in the diff |
| 500 regression | None — payload finite, `json.dumps` clean |
| New ERROR path | Fail-safe; cannot fire on well-formed data (asserted) |
| Operator `:8000` | Not restarted (`79.55` open) — inert until then |

## 10. Tier ledger

RESEARCH **T3** (Agent-tool `researcher`, Opus 5 / max) — GENERATE **T2** (Main, Opus 5
`xhigh`) — EVALUATE **T3 ×2+** (fresh Q/A per cycle, Opus 5 / max). Fable not spent: a P2
on a non-money path. **Three Q/A cycles** rather than the usual two, because cycle 2
found a process failure (this artifact not updated) alongside three live WARNs.
