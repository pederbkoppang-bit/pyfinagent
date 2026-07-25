# live_check — phase-80.31

**Required (masterplan, verbatim):** *the measured array lengths before and after the fix,
plus the pytest output including the mutation run.*

Captured 2026-07-25. All output verbatim.

---

## §A. The immutable verification command — pre-fix baseline

```
$ cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -c "import yfinance as yf; h=yf.Ticker('AAPL').history(period='1y'); print('rows',len(h),'close',len(h['Close'].dropna()),'volume',len(h['Volume'].dropna()))"
rows 251 close 250 volume 251
```

Reproduces the step's measured figure exactly. This command reports the **upstream data
shape**, which the fix does not (and must not) change — it is the *defect witness*, not the
after-state. The after-state is §B.

## §B. Array lengths — BEFORE (per-column dropna) vs AFTER (row-wise), live

```
ticker   raw  close  vol(percol)  aligned
AAPL     251    250          251      250
MSFT     251    250          251      250
NVDA     251    250          251      250
```

- **`close` 250 vs `volume` 251** is the defect: three arrays lost the malformed bar, the
  fourth did not, so `close[i]` and `volume[i]` described different sessions.
- **`aligned` 250** is what all four arrays are now, because the row is dropped **once, on
  the frame**, before any column is read.

**Why the asymmetry was guaranteed rather than occasional:** `Volume` is `int64`, and an
`int64` column **cannot hold NaN**. So `hist["Volume"].dropna()` was a *structural no-op*
on every ticker, every time.

## §C. Live payload after the fix

```
AAPL: signal='ANOMALY_OPPORTUNITY'  non_finite=False  anomalies=[..., ...]
MSFT: signal='NORMAL'               non_finite=False  anomalies=[]
```

**Criterion 4 (sharpened):** `_has_non_finite(payload) is False`. This is stronger than
"still returns 200" and it matters now — `anomaly` is HIGH criticality in
`_SOURCE_CRITICALITY`, so under what phase-80.27 shipped an hour earlier, a single
non-finite here would make it a `critical_gap`.

## §D. Tests + mutation matrix

```
$ pytest backend/tests/test_phase_80_31_anomaly_array_alignment.py -q
................                                                         [100%]
16 passed

$ pytest 80.31 + 80.27 + 80.1 + 80.2 -q
78 passed
```

```
[B10_ROLL_CLOSE      ] KILLED   np.roll(close,1) after the helper       <- Q/A c2 survivor
[B11_ROLL_HIGH       ] KILLED   np.roll(high,1) after the helper        <- Q/A c2 survivor
[B6_VOLGUARD_WINDOW5 ] KILLED   NaN-volume guard window 60 -> 5         <- Q/A c2 survivor
[C1_NEVER_APPEND     ] KILLED   _append_if_anomalous never appends      <- Q/A c2 survivor
[A1_PERCOLUMN        ] KILLED   criterion 3: restore per-column dropna
[A11_SHIFT_VOL       ] KILLED   volume shifted by one (original defect shape)
[A15_INVARIANT       ] KILLED   disable the runtime invariant
[A14_VOLNAN          ] KILLED   remove the NaN-volume guard
[A5_FIXTURE          ] KILLED   FIXTURE: remove the malformed row

MUTATION MATRIX phase-80.31 (cycle 3): 9/9 killed
```

Earlier cycles' mutations (A2 narrow-subset, A3 order, A6 fixture-float-volume, A10/A12/A13
shifts) were killed in the cycle-2 run and are unaffected by the cycle-3 changes.

**Criterion 3 discharged literally:** under `A1_PERCOLUMN` the failures now include
`test_end_to_end_alignment_invariant_is_enforced_at_runtime` -- an ALIGNMENT test. Q/A
cycle 1 correctly flagged that A1 was previously killed only by the dropna-ORDER and
SUBSET-WIDTH tests.

### Survivors across three cycles — the recurring failure mode, recorded

**A2 initially SURVIVED.** My fixture nulls all four OHLC columns together — exactly what
Yahoo does — so narrowing the subset to `["Close"]` removed the same row and looked
correct. That is the *coincidental alignment* the research flagged: nothing enforces that
Yahoo nulls all four together, and a future NaN-High/real-Close row would leave `high`
desynced while every length still matched. Closed by
`test_a_high_only_nan_row_is_also_excluded`, which nulls **only** High and uses the
sufficiency guard as the observable (20 raw rows → 19 usable → must be rejected).

> My **first** attempt at that test also failed to kill A2: it asserted
> `max(window) <= len(frame) - 1` via an `np.mean` spy, but the spy only ever sees the
> 5- and 60-element windows, never the full array — so `60 <= 119` passed either way. The
> assertion was too weak to distinguish 119 rows from 120. Caught by re-running the
> mutation instead of trusting the green.

**A4 — `subset=[..., "Volume"]` — I called this an EQUIVALENT MUTANT. WITHDRAWN.**
Q/A cycle 2 refuted it with a float64-Volume-plus-NaN input that yields a different
top-level signal (`ANOMALY_RISK` vs `ANOMALY_OPPORTUNITY`). **Finiteness is not
equivalence** — to call a mutant equivalent you must show it cannot change observable
behaviour on ANY input, not that it did not on the three shapes you tried. Correct
disposition: a survivor with nil live reachability (Volume is `int64` with 0 NaN on every
market this project trades). Its corollary was a REAL regression I had introduced — the
NaN-volume path, now guarded and logged (`A14_VOLNAN` killed).

**Q/A cycle 2 found four more survivors**, all now killed: `B10`/`B11` (`np.roll` — same
length, every session desynced, which is why the runtime invariant now spot-checks
endpoints against the frame, NaN-tolerantly), `B6` (the guard window narrowed 60 → 5), and
`C1` (a module that never appends an anomaly passed the entire suite, because the key-pin
test iterated an empty list — the eighth vacuous guard, and the same family as the one it
was written to close).

## §E. Not byte-identical — stated, not buried

The research measured Δz **+0.047 … +0.338** across 6 tickers (MU worst — the same order as
the 1.5 → 2.0 threshold gap). **0 verdict flips on 6/6 today**, but a flip near the boundary
is possible. `volume_5d_vs_60d` is **not** in `risk_metrics`, so a newly-firing positive-z
volume anomaly reads as *opportunity* — the less-conservative direction.

**Why that is acceptable here and was not in 80.27:** this fix **discards** a value (the
malformed session's real `int64` volume); it does not **restore** suppressed ones.
`_has_non_finite(payload)` is `False` before and after, so nothing previously hidden becomes
visible. And no paper-trading, screener or optimizer path consumes this tool — it is an
LLM-debate input only (`signals.py:116/:210`, `orchestrator.py:1267/:1990/:2034`). Hence
**no dark flag**: a flag would preserve a known-wrong computation as the default.

## §F. Scope discipline

Touched: `anomaly_detector.py` only — `+122 / −9` across four hunks: `:21-44` (extraction
helper), `:81-109` (row-wise dropna), `:119-166` (runtime alignment invariant), `:188-208`
(NaN-volume guard).
**Untouched, as required** — phase-80.27's deferred ladders at `:31`, `:38`, `:188`,
`:204`, `:210`, `:217`, `:230`, and the thresholds at `:16-18`. No collision.

Operator `:8000` not restarted (`79.55` still open), so this is inert in production until
the operator acts — same standing caveat as 80.1/80.2/80.27.
