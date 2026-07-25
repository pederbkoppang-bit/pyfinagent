# Contract — phase-80.31

**Step id:** `80.31` (phase-80, **P2**, `harness_required: true`) — *[P2 — MISALIGNED
PRICE/VOLUME ARRAYS IN THE ANOMALY DETECTOR]*. Date 2026-07-25. Wave 1 tail (NaN family).
**Tier T2** — Opus 5 `high`-class work: a bounded, well-specified fix on a non-money path.

## 1. Research gate — PASSED

`handoff/current/research_brief_80.31.md`: `gate_passed: true`, 5 sources read in full,
17 URLs, `recency_scan_performed: true`, 8 internal files. Env measured: python 3.14.4,
**pandas 3.0.1**, numpy 2.4.4, yfinance 1.2.0.

**The findings that shaped the build:**

- **The misalignment is GUARANTEED, not probabilistic.** `Volume` is `int64`, which
  **cannot hold NaN**, so `hist["Volume"].dropna()` is a *structural no-op*. The other
  three columns lose the malformed row. Measured on AAPL/MSFT/NVDA: **251 raw rows →
  close/high/low/`Open` 250, volume 251.**
- **`Open` is also NaN** — the step text and my prompt both omitted it.
- **The malformed bar is a COMPLETED session** (Fri 2026-07-24), not the still-forming one.
  My own measurement, independently confirmed by the researcher and corroborated upstream
  by **yfinance issue #2622** (open since 2025-11-03: *"valid volume data… indicating it's
  a completed trading session"*).
- **Blast radius: exactly 3 offset-sensitive expressions** — `:91`, `:92`, `:93`, all in
  the volume block. The other 25 array uses are close-internal or close-derived. Membership
  rule written down in the brief, then applied.
- **high/low are aligned to close only COINCIDENTALLY** — Yahoo happens to null all four
  OHLC together, and nothing enforces it. A future NaN-Close/real-HL row would silently
  break `:170-174`. **That structural immunity is most of this fix's value**, and it closes
  the researcher's D4 outright.
- **pandas 3.x is not a hazard here.** `dropna` semantics unchanged in 3.0.0; the
  `.values`/`.to_numpy()` NaN change is opt-in behind `future.distinguish_nan_and_na` and
  nullable-dtype-only (yfinance returns NumPy-backed float64/int64). The one live pandas-3
  hazard is CoW read-only arrays from `.values` — irrelevant, this module never mutates
  them in place.

### The do-no-harm ruling that decides the shipping shape

**This fix DISCARDS a value; it does not RESTORE one** — the exact opposite of the hazard
that made 80.27 ship dark. Row-wise dropna throws away the malformed session's real
`int64` Volume (AAPL 47,402,209 on 07-24). `_has_non_finite(payload)` is `False` **before
and after** (measured live), so nothing previously NaN-suppressed becomes visible.

**Ruling: NO dark flag.** A flag would preserve a known-wrong computation as the default,
and unlike 80.27's ERROR-vs-NEUTRAL verdict this is not operator-discretionary.

**But it is NOT byte-identical, and the contract must not claim otherwise:** measured
Δz **+0.047 … +0.338** across 6 tickers (MU worst, the same order as the 1.5→2.0 threshold
gap). **0 verdict flips on 6/6 today**, but a flip is possible near the boundary. Direction
note, stated rather than buried: `volume_5d_vs_60d` is **not** in `risk_metrics`
(`:226-227`), so a newly-firing positive-z volume anomaly reads as *opportunity* — the
less-conservative direction. **Mitigant:** no paper-trading, screener or optimizer path
reads this tool; it is an LLM-debate input only (`signals.py:116/:210`,
`orchestrator.py:1267/:1990/:2034`).

## 2. Immutable success criteria — VERBATIM from `.claude/masterplan.json`

> 1. close/high/low/volume arrays are guaranteed equal-length and index-aligned -- assert the lengths are equal in a test using a fixture with a trailing NaN-OHLC/real-Volume row (the exact yfinance shape)
> 2. The volume z-score is computed over completed sessions only, and the choice about the in-progress session is stated explicitly
> 3. MUTATION-TEST: restore the per-column dropna and confirm the alignment test FAILS
> 4. The module still returns 200 / serialises cleanly (do not regress what already works)

**Immutable verification command:**
`cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -c "import yfinance as yf; h=yf.Ticker('AAPL').history(period='1y'); print('rows',len(h),'close',len(h['Close'].dropna()),'volume',len(h['Volume'].dropna()))"`
Pre-fix baseline, measured: **`rows 251 close 250 volume 251`**.

## 3. Plan

1. **`backend/tools/anomaly_detector.py`** — replace the four per-column `.dropna()` calls
   with a single row-wise `hist = hist.dropna(subset=["Open","High","Low","Close"])`,
   placed **BEFORE** the `len(hist) < 20` guard so sufficiency counts *usable* rows, then
   plain `.to_numpy()` extraction.
   - **Volume deliberately NOT in the subset:** a legitimately zero/absent volume must not
     discard an otherwise-good price bar.
   - **`Open` deliberately IS in the subset** even though the module never reads it — it
     matches the measured malformed shape and is the more conservative choice. Stated here
     rather than left silent (researcher §C5).
2. **Criterion 2 — TWO explicit statements**, per the brief:
   (i) the malformed bar is a **completed** session; row-wise dropna excludes it, and
   discarding one real volume reading is **accepted deliberately**;
   (ii) the genuinely **in-progress** session is a *separate* issue that row-wise dropna
   does **not** fix (nothing is NaN there). No special case in 80.31 — every close-based
   metric already treats the last bar as final, and an in-session guard needs a
   market-calendar dependency this module lacks. **Queued.**
3. **Tests — `backend/tests/test_phase_80_31_anomaly_array_alignment.py`** (greenfield: no
   existing test calls `get_anomaly_scan` for real; 80.1 monkeypatches it at `:186`, so
   that suite is insensitive to this change).
   - the exact yfinance shape: trailing NaN-OHLC row with a **real `int64`** Volume;
   - all four arrays equal-length **and** index-aligned (assert on *values*, not just
     `len`);
   - **criterion 4 sharpened** per the brief: assert `_has_non_finite(payload) is False`,
     not merely "no exception" — `anomaly` is HIGH criticality in `_SOURCE_CRITICALITY`, so
     a non-finite would now become a `critical_gap` under what 80.27 shipped.
   - **Fixture pins bind to the fixture builder itself** — `frame["Close"].isna().sum() == 1`,
     `frame["Volume"].notna().all()`, `frame["Volume"].dtype == "int64"` — never a library
     fact. (80.1 cycle 1 shipped that mistake; do not repeat it.)
4. **Mutation matrix** — criterion 3 explicitly requires restoring the per-column dropna
   and confirming failure. Plus a **FIXTURE mutation** (remove the bad row) and a
   **guard-order mutation** (dropna after the `len < 20` check).

## 4. OUT of scope — queued, not fixed

- **D1 — the √5 units mismatch** at `:91-94`: `std` of *daily* volume is used to z-score a
  *5-day mean*, so |z| is systematically ≈**2.24× too small**. **This is an order of
  magnitude larger than the alignment offset it sits beside.** Suppressive today; fixing it
  makes anomalies fire MORE = less conservative ⇒ its own research gate.
- **D2** — the self-overlapping baseline (`volume[-5:]` ⊂ `volume[-60:]`). Also suppressive,
  and qualified: arXiv:2004.04013 finds window overlap can *reduce* bias. Do not reflex-fix.
- **D3** — the genuinely in-progress session (needs a market calendar).
- **D4 — closed by this step** if it ships as designed (coincidental high/low alignment).
  Recorded so a later audit does not re-discover it as new.

## 5. DO-NO-HARM

Live book cannot move: no paper-trading/screener/optimizer path consumes this tool. No
`.env` edit, no flag, no optimizer run, `historical_macro` FROZEN;
kill-switch/stops/sector-caps/DSR/PBO untouched. Must not touch 80.27's deferred ladders
(`:31`, `:38`, `:188`, `:204`, `:210`, `:217`, `:230`) or the thresholds (`:16-18`).
Must not reintroduce a 500 (criterion 4). `git add -An` before the flip.
