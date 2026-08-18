# live_check -- step 86.59

**Required shape** (immutable): *"the per-cycle analysed-ticker list before and
after, the measured rank-stability figure, and the sector distribution."*

Everything below is produced by
`scripts/qa/rank_stability_86_59.py`, which drives the **real**
`screen_universe` + `rank_candidates` with stored BigQuery prices swapped in
for the yfinance network call. A reimplementation of the composite would
measure the script; the swap measures the picker.

Re-runnable:

```
python scripts/qa/rank_stability_86_59.py --cycles 20      # criteria 1 + 3
python scripts/qa/rank_stability_86_59.py --flags --cycles 20    # criterion 4
python scripts/qa/rank_stability_86_59.py --dispersion --cycles 20  # finding (a)
python scripts/qa/mutation_86_59.py                        # criterion 7
```

**Normalisation, stated once and applying to every figure below.** 20
**consecutive trading sessions** ending 2026-08-17, one replayed cycle per
session, each fed a trailing window of **126 sessions**. The unit is a
SESSION, never a calendar day. The universe is the **513 US tickers** BigQuery
stores; the live universe is multi-market (~583), and the gap is quantified in
the fidelity section rather than waved at. Turnover is **one-sided**: the
fraction of today's slate that was not in yesterday's.

---

## 1. Criterion 1 -- rank stability of the CURRENT score, MEASURED

```
-- criterion 1: day-over-day stability of the CURRENT score --
mean Spearman rho over the full screened cross-section : 0.9622
min  Spearman rho (the least stable adjacent pair)     : 0.9319
mean one-sided turnover, top-10 slate      : 15.8% per day
mean one-sided turnover, top-5 analysed  : 15.8% per day
sessions with ZERO top-10 turnover         : 3 of 19
```

The correlation is taken over the **full screened cross-section** (~507 names),
not over the top-10. Ranking only the top-10 would compute a correlation over a
set chosen *by the very thing being measured*.

**This partially REFUTES the step's own premise, and criterion 1 explicitly
allows that.** The step is filed as *"the picker analyses the SAME 4-6 names
every day"* because *"the ranking is effectively frozen"*. Over 20 sessions the
composite is **highly persistent but not frozen**: rho 0.9622, and only **3 of
19** adjacent sessions had zero top-10 turnover. The audit basis's own window
(08-04..08-11) does reproduce; it simply is not representative of the 20-session
span.

## 2. Criterion 1 -- the per-cycle analysed-ticker list, BEFORE and AFTER

"Before" is what the **system actually did**, read from
`financial_reports.analysis_results`. "After" is what the replayed picker
produces. There is no behaviour change in this step, so "after" is the
measurement, not a new state.

| session | LIVE analysed (ground truth) | replay top-10 ∩ live |
|---|---|---|
| 2026-08-10 | CRWD,DELL,HPE,HUM,NTAP,PANW | 5/6 |
| 2026-08-11 | 009150.KS,CRWD,DELL,HPE,NTAP,PANW | 4/6 |
| 2026-08-12 | BAX,CRWD,DELL,HPE,NTAP,PANW | 4/6 |
| 2026-08-13 | 009150.KS,DELL,HPE,HPQ,MRVL,NTAP | 4/6 |
| 2026-08-14 | HPE,MRVL,NTAP,PANW,STX,WDAY | 4/6 |
| 2026-08-17 | 009150.KS,DELL,HPE,MRVL,MU,NTAP,SNDK | 5/7 |

**18 distinct tickers were analysed LIVE over the 20 sessions**:
`009150.KS, AMD, BAX, CRWD, DDOG, DELL, DVA, FTNT, HPE, HPQ, HUM, MRVL, MU,
NTAP, PANW, SNDK, STX, WDAY`.

The step text says *"ACROSS 8 CYCLES ONLY 8 DISTINCT TICKERS WERE EVER
ANALYSED"*. Over the most recent 20 sessions it is **18**. The slate rotates
substantially from 08-12 onward -- 08-14 drops DELL entirely for STX and WDAY.

## 3. Fidelity -- what this replay is and is not

**Mean overlap between the replay's top-10 and the live analysed set: 80%.**

A replay never compared to the live system is a simulation of itself, so the
divergence is measured and its causes enumerated rather than assumed:

1. the live universe is multi-market (`009150.KS` appears three times above);
   this panel is US-only;
2. the live analysed set is candidates **∪ re-evaluations of held names**, and
   held names are *excluded* from new candidates -- NTAP recurs as a re-eval,
   by design (`autonomous_loop.py`);
3. the live cycle passes score overlays (news, PEAD, revisions, sector
   momentum) that this replay passes as `None`, so this measures the **base
   composite** -- which is exactly what criterion 1 names;
4. live yfinance uses `auto_adjust=True`; the stored close is unadjusted.

**No claim in this document is stronger than that 80%.**

## 4. Criterion 3 -- diversity as a measured distribution, N stated

**N = 20 cycles**, analysed slate = top-5, so 100 analysed slots.

```
distinct tickers ever ANALYSED over N=20 cycles: 12
  DD, DDOG, DELL, DVA, FTNT, HPE, HPQ, HUM, MU, PANW, SNDK, ZBRA
sector concentration: Information Technology holds 72.0% of analysed slots
  counts: {'Industrials': 20, 'Information Technology': 72, 'Health Care': 8}
```

12 distinct names across 100 slots = **12.0% of the ceiling**. Information
Technology at **72.0%** is the one-sector concentration the audit basis
reports, and it reproduces. Sectors come from the real `build_sector_map`
(502/513 tickers carry a GICS sector); without that lookup every slot reads
`UNKNOWN` and the figure would measure the harness rather than the book.

## 5. Criterion 4 -- what the three EXISTING dark flags already do

Measured **before** any new code, which is what the criterion requires. All
flags forced per-call in-process. **No `.env` written, no flag promoted.**

```
arm                       turnover/day  distinct  top sector
------------------------------------------------------------------------------
baseline                         15.8%        12  Information Technology 72%
sector_neutral                   28.4%        22  Industrials 20%
soft_diversity_w0.30             22.1%        17  Information Technology 40%
min_k_sectors=3                  17.9%        14  Information Technology 60%
```

**All three MOVE the slate**, so none is inert and none needs rebuilding:

- `sector_neutral`: turnover **+12.6pp**, distinct **+10**, top-sector share **-52.0pp**
- `soft_diversity_w0.30`: turnover **+6.3pp**, distinct **+5**, top-sector share **-32.0pp**
- `min_k_sectors=3`: turnover **+2.1pp**, distinct **+2**, top-sector share **-12.0pp**

`paper_min_k_sectors_analyzed` is deliberately measured through
`autonomous_loop._min_k_sector_slice` and not as a `rank_candidates` kwarg,
because it is not one -- treating it as one would measure a flag the picker
does not have.

**Window sensitivity, disclosed rather than hidden.** At 6 cycles
`soft_diversity` *reduced* turnover (-2.9pp); at 20 cycles it *raises* it
(+6.3pp). The sign flips with the window, so only the 20-cycle figures are
quoted and no 6-cycle number appears anywhere in this step's conclusions.

## 6. Finding (a) -- the declared weights are NOT the effective weights

A term's influence on a cross-sectional ranking scales with **weight x sigma**,
not with weight. Measured over the same 20 sessions:

```
mean effective share : 1m 22.6%  3m 37.0%  6m 40.4%
declared             : 1m 40.0%  3m 35.0%  6m 25.0%
gap (effective-declared): 1m -17.4pp  3m +2.0pp  6m +15.4pp
```

**The term with the smallest declared weight has the largest effective
influence.** Measured mean cross-sectional sigmas are **10.646** (1m),
**19.850** (3m), **30.441** (6m) -- the 6m term carries **2.86x** the
dispersion of the 1m term.

> *Corrected in cycle 2.* Cycle 1 quoted "~10.2 / ~19.4 / ~31.0 ... ~3.0x",
> which does **not** reproduce from the command cited for it, and was
> internally inconsistent: weight x sigma on that triple yields effective
> shares 21.9/36.5/41.6, while the artifact's own headline is 22.6/37.0/40.4 --
> which is exactly what the corrected means give. The figures above are now
> **printed by the script** (`mean cross-sectional sigma` line) rather than
> retyped into prose, so they cannot drift again. No conclusion moves; the
> load-bearing fact is that the 6m term dominates, and 2.86x carries it.

**Does the existing `multidim_momentum` flag already fix this? No -- measured,
not argued.** It *does* call `_zscore`, which makes it look like the fix. But it
z-scores the **finished composite as a single scalar**, so it cannot reweight
the horizons inside it. With `price=1.0` and every other component 0:

```
price-only multidim vs baseline: 50 of 10139 ranked positions move (0.493%),
across 20 cycles; 5 cycles are exactly identical.
```

99.507% unchanged, and **every** displacement is accounted for by the
`round(..., 4)` on the z-score creating a tie that a stable sort then reorders
-- asserted as an invariant with a paired negative fixture, not assumed.

**Recorded failure.** This began as a prediction of *exact list equality*, and
that prediction **failed** (1 of 6 cycles identical on the first run). The
structural claim survived; the test did not match it. The corrected test is
above and the failure is recorded rather than quietly replaced.

## 7. Criterion 5 -- nothing promoted, nothing written

`git show --name-only 15a817cc | grep -E '^(backend|frontend)/'` returns
**nothing** -- no production file is in this step's commit.

> *Corrected in cycle 2.* Cycle 1 cited `git status --short -- backend/`, which
> returns three modified files. All three belong to a peer session's in-flight
> work, so the substantive claim held, but the command cited did not
> demonstrate it: the working tree is shared and a tree-scoped command cannot
> attribute a change to a step. The commit-scoped command can. The only files it authors are `scripts/qa/rank_stability_86_59.py` and
`scripts/qa/mutation_86_59.py`. No `.env` write, no flag promotion, no gate
touched, no restart pending.

## 8. Criterion 7 -- mutation matrix

```
control --verify       -> rc=0 GREEN
control --dispersion   -> rc=0 GREEN
control --flags        -> rc=0 GREEN

coverage: 20 guards in target, 20 covered by a cell or an explicit transitive entry

KILLED 20 / 20   SURVIVED 0   UNSCORABLE 0
restore verified: sha256 unchanged (9282ba866f2afc87...)
```

**Cycle 1 shipped 14 cells and that was not enough.** The evaluator ran an AST
census, found 19 guards with 13 covered, and proved two of the uncovered ones
**unkillable** -- by execution, not argument. Extending coverage surfaced four
more survivors. All are now closed; the matrix additionally runs a **coverage
gate** that fails if any `_ok` guard has no cell, and it caught a real gap on
its first run. Details in `experiment_results_86.59.md` § Cycle 2.

Control observed GREEN on all three modes **first**; a non-zero exit alone is
not scored as a kill (the named guard must appear in the output); a
non-applying anchor is UNSCORABLE, never a kill.

**Two defects this matrix found in my own deliverable:**

1. **A tautological guard.** `slate_is_a_prefix_of_the_full_ranking` read
   `ranked == full[:SCREEN_TOP_N]` immediately after `ranked =
   full[:SCREEN_TOP_N]` -- a variable compared to its own definition, which
   cannot fail on any input. Replaced with an **independent**
   `rank_candidates(top_n=10)` call (cell M3).
2. **A guard that survived being weakened.** The tie-explanation rule was
   inline; mutating it to `len(moved) >= 0` left the run green, because an
   assertion cannot detect its own weakening from the inside. Fixed with a
   **paired negative fixture** of known-bad inputs the rule must reject
   (`_TIE_FIXTURE`), which turns M12 from SURVIVED into KILLED.

**A third defect, found by an accident worth recording.** A 2-minute command
timeout SIGTERMed the matrix mid-cell and **stranded a mutant on disk** --
`return moved >= 0  # MUTANT` -- which then failed the very fixture it was
added to protect. `try/finally` does not run on SIGTERM. The matrix now
installs SIGTERM/SIGINT/SIGHUP handlers that restore, and **refuses to start**
if the target already contains a `MUTANT` marker, because a matrix run from a
poisoned baseline is not a measurement.

## 9. Data-quality control -- the stored panel is NOT clean

```
duplicate (ticker,date) rows dropped before measuring: 47,880 of 200,875 (23.8%)
split-shaped bars in the panel (|1d move| > 40%): 10
```

`historical_prices` carries duplicate keys across 2017-2025 and **nothing under
`backend/` de-duplicates them**. Filed as step **86.116**; not fixed here. This
replay de-duplicates so that it measures the picker and not the table's defect.

Unadjusted splits manufacture *turnover*, so they bias this measurement
**against** the frozen-ranking premise -- a high measured stability despite them
is conservative.
