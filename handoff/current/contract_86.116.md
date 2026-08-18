# Contract -- step 86.116

**Step:** 38% of `financial_reports.historical_prices` rows are duplicate
`(ticker,date)` keys and NOTHING under `backend/` de-duplicates them, so every
backtest lookback is positionally compressed. **P1, money path.**

## Research-gate summary (what the gate CHANGED about the plan)

Gate **PASSED** (`wf_4ec1794d-0da`; 6 sources read in full, 25 URLs against 25
distinct in the brief, envelope `COMPLETE`, 12 internal files inspected; brief
`research_brief_86.116.md`, 24,632 chars).

**The gate changed the fix itself, not just its justification.**

**1. `drop_duplicates()` is the WRONG TOOL, and this is the single most
important finding.** pandas' own documentation states that for
`DataFrame.drop_duplicates` *"Indexes, including time indexes are ignored"* --
it compares **values**, not the key. In `backend/backtest/cache.py` the date is
the **index**, so two rows for the same session that differ in any column are
both kept. **I verified this myself rather than inheriting it**: loading `AVB`
for 2026 through the real `preload_prices`, `drop_duplicates()` returns 159 rows
for **155** distinct dates -- it leaves **4 duplicate dates behind** -- while
`~df.index.duplicated(keep="first")` returns exactly 155.

*Stated precisely, because the brief's blanket phrasing did not reproduce
everywhere:* value-keyed dedup **happens to suffice** where the duplicate rows
are byte-identical (`AKAM` 2025: both methods gave 250). It is **not reliable**,
because 394,719 duplicated keys carry a differing `close`. The fix must key on
the index.

**2. The vol understatement has a closed form, so it is checkable rather than
merely measured.** Duplicating every bar makes the return series
`[r1, 0, r2, 0, ...]`, so with mean ~0 the std falls by `1/sqrt(2)` = **0.7071**
-- a **29% understatement** of annualized volatility. Measured on `AKAM`:
0.3343 as-loaded vs 0.4182 de-duplicated, ratio **0.7995**, which correctly sits
*above* the full-duplication floor because AKAM is only partially duplicated.

**3. A sharp asymmetry that changes what the test must cover.** Duplicating only
the **last 40 bars** corrupts `momentum_1m` / `momentum_3m` / `rsi_14`
identically to full duplication while volatility barely moves -- because `iloc`
lookbacks break on **recent** duplicates and `std()` responds to the **global**
rate. A fixture that only duplicates uniformly would miss half the defect.

**4. The DSR/PBO claim in my own `audit_basis` was too direct and is corrected
here.** This is **not** a Sharpe-formula bug: the engine's NAV is a per-day
dict, so the duplication does not double-count NAV points. It reaches the gates
**indirectly**, through corrupted **features** and **triple-barrier widths**.
The step must state the mechanism this way and not claim the gates are computed
on doubled NAV.

**5. It is LEGACY and therefore bounded.** 2017 90.5%, 2018-2025 ~63%, **2026
0.1%**. The write side is no longer producing duplicates, so a repair is
terminal rather than a treadmill -- and it is also why nobody noticed.

**6. A separate proven defect, recorded not absorbed:**
`backend/agents/mcp_servers/data_server.py:99` raises `KeyError('date')`.

## Hypothesis

The stored table has been ~38% duplicated since 2017 and no read path
de-duplicates, so every positional lookback in the backtest stack silently spans
roughly half the sessions it names, and realized volatility is biased downward.
De-duplicating **on read, keyed on the index** restores the intended semantics
without touching stored rows.

## Immutable success criteria (copied verbatim from `.claude/masterplan.json`)

1. the duplication is RE-MEASURED from BigQuery by the step itself rather than inherited from this audit_basis, and reported as keys/rows/tickers with the per-year breakdown and the normalisation rule stated beside every share
2. the POSITIONAL harm is demonstrated by DRIVING REAL CODE, not argued: run an actual momentum/volatility computation over a duplicated ticker's real series and over its de-duplicated series and report both numbers, so the size of the distortion is measured rather than asserted
3. it is PROVEN that no existing layer de-duplicates, with a positive control showing the probe would detect a dedupe if one were present -- a grep returning nothing is not evidence until it is shown capable of returning something
4. the fix de-duplicates on READ in backend/backtest/cache.py so that historical rows already written stay readable, and any write-side or table-level repair is proposed as a numbered operator ask rather than executed -- no DELETE or table rewrite is performed by this step
5. flag-OFF or fix-absent parity is demonstrated against an oracle: with the de-duplication disabled the returned frames are byte-identical to today's, so the change is provably inert until it is meant to fire
6. the effect on the existing gates (DSR, PBO) is reported rather than only on the price frames, because a horizon change that moves the gates is a strategy-validity question and not a data tidy-up
7. mutation-test every new guard: revert it and show the check goes red, with the control observed GREEN first, the same test count collected in control and mutant, the NAMED test failing, and a byte-identical restore

**Immutable verification command:**
`bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/backtest/cache.py\").read()); print(\"parses\")"'`

**Immutable live_check:** `live_check_86.116.md` with the re-measured
duplication census and its query, the before/after momentum and volatility
numbers from a real driven computation on a real affected ticker, the
positive-controlled proof that nothing de-duplicates today, and the parity
oracle.

## Plan

**P1 -- criterion 1, re-measured (already done, recorded here).** Census
re-run against BigQuery: 1,859,482 rows over 1,152,607 distinct keys; **706,875
duplicated keys (61.33% of keys)**; **706,875 excess rows (38.01% of rows)**;
max multiplicity 2; **336 of 513 tickers**. Per-year breakdown and the
normalisation rule (share **of keys** vs share **of rows** -- they differ and
must never be quoted interchangeably) go in the live_check.

**P2 -- criterion 2, driven against real code (already done, recorded here).**
Load a real affected ticker through the **real** `preload_prices` and run the
**real** `screener._pct_change` / `_compute_rsi` / volatility on the as-loaded
and de-duplicated series. Measured on `AKAM` 2025: `mom_1m` 0.83 -> **-0.52**
(sign flip), `mom_3m` -1.60 -> **+15.04** (sign flip), `rsi_14` 23.7 -> **54.5**,
`vol_ann` 0.3343 -> **0.4182**. A "21-period" lookback spans **12 real sessions
instead of 22**. The RSI move is money-relevant: `rank_candidates` applies
`score *= 0.8` below 20 and 23.7 sits beside that boundary.

**P3 -- criterion 3, with the positive control the criterion demands.** A grep
returning nothing proves nothing until the probe is shown capable of returning
something. Plant a `drop_duplicates` call, show the probe finds it, remove it,
show the probe returns to zero -- and assert a token that must exist as a second
control.

**P4 -- criterion 4, the fix: de-duplicate ON READ, keyed on the INDEX.**
`~df.index.duplicated(keep="first")` in `backend/backtest/cache.py`, at both
`preload_prices` and `cached_prices`. **No `DELETE`, no table rewrite**; the
table repair is ASK-1.

**P5 -- criterion 5, parity against an ORACLE.** With the de-duplication
disabled the returned frames must be **byte-identical** to today's, demonstrated
against a recorded oracle rather than two passing examples.

**P6 -- criterion 6, the gate effect, with the CORRECTED mechanism.** Report
DSR/PBO rather than only the price frames -- and describe the path honestly:
**not** a Sharpe-formula bug, but corrupted features and triple-barrier widths.
If the gates move, that is a finding to report; **no threshold is adjusted**.

**P7 -- criterion 7, mutations** with control GREEN first, the same test count
collected in control and mutant, the **named** test failing, and a
SHA-256-verified byte-identical restore. The fixture must cover the
**last-40-bars** asymmetry, not only uniform duplication.

## Scope honesty -- what this step does NOT do

- **It deletes nothing.** No `DELETE`, no table rewrite, no write-side change.
  The one-time repair is a numbered operator ask.
- **It does not claim a Sharpe-formula bug.** The gate refuted that; the path is
  features and triple-barrier widths.
- **It does not fix `data_server.py:99`'s `KeyError('date')`** -- filed
  separately rather than absorbed.
- **It does not touch the picker score** (86.59, PARKED) or the entry path
  (86.60, blocked by a peer session).
- **It does not unblock 86.117 by fiat**; that step re-measures for itself.
- **No flag is promoted and no `.env` is written.**

## References

`research_brief_86.116.md` (the `drop_duplicates`-ignores-indexes finding, the
`1/sqrt(2)` closed form, the last-40-bars asymmetry, the corrected DSR/PBO
mechanism, the legacy-window bound); `experiment_results_86.59.md` (where the
defect was discovered); `contract_86.117.md` (blocked on this step).
