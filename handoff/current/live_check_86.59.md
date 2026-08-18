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

**This block is a single verbatim capture of one `python scripts/qa/mutation_86_59.py`
run.** Cycle 2 presented a coverage line from one run beside a sha line from
another; the evaluator caught it, and it was the second non-reproducing number
in this step. Nothing here is assembled from more than one execution.

```
sha256 : 349ea82f74680a15...
control --verify       -> rc=0 GREEN
control --dispersion   -> rc=0 GREEN
control --flags        -> rc=0 GREEN
coverage: 26 guards in target, 26 covered by a cell or an explicit transitive entry
KILLED 29 / 29   SURVIVED 0   UNSCORABLE 0
restore verified: sha256 unchanged (349ea82f74680a15...)
```

Control observed GREEN on all three modes **first**; a non-zero exit alone is
not scored as a kill (the named guard must appear in the output); a
non-applying anchor is UNSCORABLE, never a kill; restore verified by SHA-256.

**29 cells against 26 guards.** The counts differ in BOTH directions and the
reasons are different, so neither is a rounding note:

- some guards carry **more than one cell**, because one cell licenses nothing
  about a direction or a code path it never exercised.
  `min_k_arm_used_the_labelled_k` has M23 (the ARGUMENT drifts) and M24 (the
  label drifts). `sector_map_covers_the_panel_at_the_published_operating_point`
  has FOUR: M9 (total collapse), M9b (degradation to 78.2%, the level that
  actually inverts the published ordering and which the previous 50% floor did
  NOT catch), and M9c/M9d, which repeat that identical injection on the
  `--flags` and `--dispersion` paths. M9c and M9d differ from M9b **only in the
  mode**, and that is the point: cycle 5 proved the guard was absent from the
  path that publishes the criterion-4 table, so M9b's kill had been on the
  wrong path;
- some guards are covered **transitively** through the predicate they consume,
  each recorded in `COVERED_TRANSITIVELY` with a reason.

The coverage gate fails the matrix if any guard is covered by neither.

*(This paragraph was regenerated from the run captured immediately above. The
cycle-4 Q/A found the previous version asserting "the cell count (22) is lower
than the guard count (23)" nine lines below its own verbatim block reading
"24 guards in target, 24 covered" -- the block had been regenerated and the
authored prose beside it left stale, the same class as this step's own cycle-3
item 10. Both are now produced from one run, and the block is the authority.)*

**Criterion-4 cells now run at the PUBLISHED `--cycles 20`, not 4.** The cycle-2
evaluator showed kill/survive can be cycle-count dependent -- a `w=0.15`
baseline poison survived at 4 cycles and died at 20 -- so running every cell at
4 made the matrix a *weaker* oracle than the run whose numbers are published.

### What the matrix and the evaluators found in this deliverable

Recorded because most of these are the guard-vacuity class, and because two of
them were found *after* this step had already claimed credit for finding the
first two.

| # | defect | found by | fix |
|---|---|---|---|
| 1 | `slate_is_a_prefix_of_the_full_ranking` compared a variable to its own defining expression | my own matrix (cycle 1) | independent `rank_candidates(top_n=10)` call |
| 2 | the tie rule survived being weakened to always-true | my own matrix (cycle 1) | paired negative fixture `_TIE_FIXTURE` |
| 3 | a mutant stranded on disk by a SIGTERM | an accident (cycle 1) | signal handlers + refuse-to-start on a poisoned baseline |
| 4 | `panel_is_us_only` was a literal `True` | **evaluator, cycle 1** | interrogate the panel, not the query |
| 5 | `baseline_arm_is_the_unflagged_ranking` was `len(x)==len(set(x))` on a set-derived list | **evaluator, cycle 1** | assert the arm definition + distinguishability |
| 6 | four more guards died to `EXPR -> always-true` | extending coverage (cycle 2) | named predicates + known-bad fixtures |
| 7 | a cell list licenses only its own cells | **evaluator, cycle 1** | AST coverage gate; caught a real gap on its first run |
| 8 | a flag injected into the BASELINE at the replay seam left the definition intact and **flipped min_k's delta from +2.1pp to -2.1pp** | **evaluator, cycle 2** | recompute the baseline slate via a direct unflagged call (cell M20) |
| 9 | `_PREDICATE_FIXTURE` had no cell and no size assertion -- emptying it left the guard green | **evaluator, cycle 2** | `fixture_exercises_every_predicate_on_rejecting_inputs` (cell M21) |
| 10 | the §8 evidence block was spliced from two runs | **evaluator, cycle 2** | regenerated verbatim from one run, above |

**All cycle-2 AND cycle-3 attacks now KILL, control observed GREEN first at the
published `--cycles 20`** (min_k delta reproducing at +2.1pp):

```
=== CONTROL (published --cycles 20) ===
control GREEN; guards: ['fixture_exercises_every_predicate_on_rejecting_inputs',
 'predicates_reject_known_bad_inputs', 'flag_arms_all_ran',
 'baseline_arm_applies_no_flags', 'baseline_slate_matches_an_unflagged_direct_call',
 'flag_arms_are_distinguishable_from_baseline']
  min_k delta: +2.1pp

=== ATTACK 1: momentum_52wh_tilt into ONLY the baseline call, 20 cycles ===
KILLED: INVARIANT FAILED: baseline_slate_matches_an_unflagged_direct_call --
the baseline slate disagreed with a direct unflagged rank_candidates() call on
15 of 20 cycles

=== ATTACK 2: soft_sector_diversity w=0.05 into the baseline only, 20 cycles ===
KILLED: INVARIANT FAILED: baseline_slate_matches_an_unflagged_direct_call --
... on 20 of 20 cycles

=== ATTACK 3: empty the predicate fixture ===
KILLED: INVARIANT FAILED: fixture_exercises_every_predicate_on_rejecting_inputs
```

```
=== CONTROL (null mutant, published --cycles 20) ===
GREEN. baseline 15.8%/12  min_k delta +2.1pp

=== ATTACK A: momentum_52wh_tilt k=0.2 at the ARMS LOOP ===
KILLED: INVARIANT FAILED: baseline_ROW_matches_an_unflagged_direct_call

=== ATTACK B: soft_sector_diversity w=0.05 at the ARMS LOOP ===
KILLED: INVARIANT FAILED: baseline_ROW_matches_an_unflagged_direct_call

=== disk untouched ===
md5 same: True
```

The criterion-4 path now carries **seven** guards, up from two at cycle 1.

**A claim of mine that was FALSE and is now narrowed.** Cycle 3 shipped, in a
code comment *and* in both artifacts: *"An injection anywhere in the replay path
-- at the seam, in the kwargs, in a wrapper -- makes these diverge."* That was
not true. `baseline_slate_matches_an_unflagged_direct_call` compares the `base`
variable only, which feeds the **min_k arm**; the baseline **row** that every
delta is subtracted from came from a structurally identical sibling call in the
`FLAG_ARMS` loop, and injecting there survived all six guards. The comment now
states its scope narrowly, and the sibling call has its own guard
(`baseline_ROW_matches_an_unflagged_direct_call`, cell M22).

**Two structurally identical call sites, one guarded, is not guarded.** That is
the fourth appearance of one lesson in this step: a value check, then a
definition check, then a behavioural check on the wrong variable.

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
