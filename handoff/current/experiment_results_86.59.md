# experiment_results -- step 86.59

**Step:** the stock picker analyses the SAME 4-6 names every day because its
score is built only from slow trailing returns. **P1, money path.**

**Immutable verification command:**

```
$ bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/tools/screener.py\").read()); print(\"parses\")"'
parses
```

## What this step SHIPS

**A measurement, and no behaviour change.** That is the deliverable the
criteria describe: criterion 1 says the stability must be *measured*, criterion
4 says the three existing dark flags must be evaluated *before new code is
written*, and criterion 5 forbids promoting a flag. Two new files, both under
`scripts/qa/`:

| file | what it is |
|---|---|
| `scripts/qa/rank_stability_86_59.py` | the measurement -- drives the REAL `screen_universe` + `rank_candidates` with stored BigQuery prices swapped in for the yfinance network call |
| `scripts/qa/mutation_86_59.py` | criterion 7 -- **22 cells + an AST coverage gate**, control-green-first, SHA-256-verified restore |

**No production file is modified.** `git show --name-only 15a817cc | grep -E
'^(backend|frontend)/'` returns nothing; no `.env` write, no flag promotion, no
gate touched, no restart pending. (Cycle 1 cited a tree-scoped `git status`,
which cannot attribute a change to a step in a working tree shared with a peer
session.)

## Why it measures the system rather than a reimplementation

The script monkeypatches **only** `screener.yf.download`. Every factor
(`momentum_1m/3m/6m`, `rsi_14`, `volatility_ann`, `sma_50_distance_pct`,
`pct_to_52w_high`), every filter, every overlay and the sort are computed by
production code. Two details that a rewrite would have gotten wrong:

- `momentum_6m = _pct_change(close, len(close) - 1)` is anchored to the **first
  bar of the fetched window**, not to a fixed 126-day lookback, so the window
  length is a *factor definition* and is recorded beside every figure;
- `rank_candidates` only attaches a sector when the caller supplies
  `sector_lookup`, so the real `build_sector_map` is used (502/513 tickers)
  -- without it criterion 3 would report `UNKNOWN` for 100% of slots and
  measure the harness.

## Results, by criterion

Full evidence with verbatim command output: **`live_check_86.59.md`**.
Normalisation for every figure: 20 **consecutive sessions** ending 2026-08-17,
trailing window 126 sessions, 513 US tickers, one-sided turnover.

**Criterion 1 -- MEASURED, and it partially refutes the step's premise.**
Spearman rho **0.9622** mean / 0.9319 min over the full screened cross-section;
one-sided top-10 turnover **15.8%/day**; **3 of 19** adjacent sessions had zero
top-10 turnover. So the composite is *highly persistent but not frozen*. The
step is filed on *"the ranking is effectively frozen"*; over 20 sessions that
is too strong. Criterion 1 explicitly permits this outcome and requires it be
reported either way.

**Criterion 2 -- vacuously satisfied, and that is stated rather than
exploited.** No new or reweighted term is introduced, so there is nothing to
justify out-of-sample. The reason is not laziness: criterion 2 *also* demands
DSR/PBO for any such term, and those gates are computed on backtests that read
`financial_reports.historical_prices` -- which returns **duplicated rows**
(§ below). Reporting a gate number off that data would be a figure I could not
stand behind. The reweighting fix is therefore filed as **86.117**, explicitly
BLOCKED-BY **86.116**.

**Criterion 3 -- measured distribution, N = 20 stated.** 12 distinct names
across 100 analysed slots (12.0% of ceiling); **Information Technology 72.0%**
of slots. The one-sector finding reproduces. The live "before" is taken from
the system, not the replay: **18 distinct tickers analysed live** over the same
20 sessions, against the step text's *"only 8 distinct across 8 cycles"*.

**Criterion 4 -- all three existing dark flags MOVE the slate.**

| arm | turnover/day | distinct | top sector |
|---|---|---|---|
| baseline | 15.8% | 12 | Information Technology 72% |
| `sector_neutral` | 28.4% | 22 | Industrials 20% |
| `soft_diversity_w0.30` | 22.1% | 17 | Information Technology 40% |
| `min_k_sectors=3` | 17.9% | 14 | Information Technology 60% |

**This is the step's most consequential result.** The diversity mitigation is
not missing -- it is built, sitting in the tree, switched off. Writing a fourth
one is precisely the duplicate work criterion 4 exists to prevent, which is why
this step ships no new scoring code.

**Criterion 5 -- nothing promoted, nothing written.** Operator asks below.

**Criterion 6 -- parity, honestly characterised.** With no new behaviour there
is nothing to disable, so a "flag-OFF parity" claim would be vacuous. What is
actually demonstrated is stronger and checkable: the step modifies **no**
production file, so the live candidate list is unchanged by construction, and
the measurement additionally shows `rank_candidates(top_n=10)` agreeing with a
slice of the full ranking on every cycle (an independent call, cell M3).

**Criterion 7 -- 22 cells, 22 KILLED, 0 SURVIVED, 0 UNSCORABLE**, coverage
23/23, control GREEN first on all three modes, SHA-256-verified restore, plus an
**AST coverage gate** that fails the matrix if any `_ok` guard has no cell.
Criterion-4 cells now run at the published `--cycles 20`. Cycle 1 shipped 14
cells and the evaluator proved two guards unkillable; cycle 2 shipped 20 and it
proved a third attack; see the cycle-3 section.

## Finding (a): the declared weights are not the effective weights

Declared **40/35/25**; measured effective **22.6 / 37.0 / 40.4**. The term with
the *smallest* declared weight has the *largest* effective influence, because
influence scales with weight x cross-sectional sigma and the 6m horizon carries
**2.86x** the dispersion of the 1m (mean sigmas 10.646 / 19.850 / 30.441, now
PRINTED by the script rather than retyped -- cycle 1 quoted a triple that did
not reproduce; see `live_check_86.59.md` § 6).

**No existing flag fixes it, and the near-miss is worth naming**: reading the
source suggests `multidim_momentum_enabled` does, because it calls `_zscore`.
It does not -- it z-scores the *finished composite* as one scalar, so it cannot
reweight the horizons inside it. Measured: 50 of 10,139 ranked positions move
(0.493%), and every displacement is a `round(..., 4)` tie. Filed as **86.117**.

## Defects this step found in ITS OWN deliverable

Recorded because the last session's evaluations blocked on guard vacuity, and
most of these are exactly that class. **Items 1-3 were found in cycle 1; the
cycle-2 section below records the four the evaluator found that I missed.**

1. **A tautological guard.** `slate_is_a_prefix_of_the_full_ranking` compared
   `ranked` to its own defining expression on the line above. It could not fail
   on any input. Found by trying to mutate it. Replaced with an independent
   `rank_candidates(top_n=10)` call.
2. **A guard that survived being weakened.** The tie-explanation rule was
   inline; mutating it to `len(moved) >= 0` left the suite green -- an
   assertion cannot detect its own weakening from inside. Fixed with a paired
   negative fixture of known-bad inputs (`_TIE_FIXTURE`); M12 now KILLS.
3. **A stranded mutant.** A 2-minute timeout SIGTERMed the matrix mid-cell and
   left `return moved >= 0  # MUTANT` on disk, where it then failed the fixture
   it was written to protect. `try/finally` does not run on SIGTERM. The matrix
   now installs signal handlers that restore, and **refuses to start** from a
   target already containing a `MUTANT` marker.

**A prediction that failed, kept in the record.** The multidim analysis began
as a prediction of *exact list equality* and that failed (1 of 6 cycles). The
structural claim survived; the test did not match it. Both the failure and the
corrected test are in `live_check_86.59.md` § 6.

## Two defects found in the SYSTEM, filed rather than absorbed

**86.116 (P1) -- 38.0% of `financial_reports.historical_prices` rows are
duplicate `(ticker,date)` keys and nothing under `backend/` de-duplicates
them.** Found when the replay crashed on `InvalidIndexError`. Table-wide:
706,875 duplicated keys of 1,152,607; 336 of 513 tickers; **62-64% of keys in
every year 2017-2025** and 0.1% in 2026 -- which is why it was never noticed.
Not a price-accuracy defect (differing closes disagree by 0.0% at p50 *and*
p99, max 0.93%); the harm is **positional**, because `_pct_change` indexes by
row, so a "21-session" lookback spans ~12-13 real sessions, and duplicated bars
inject 0% returns that depress `volatility_ann`. `preload_prices` and
`cached_prices` both `set_index('date').sort_index()` with no
`drop_duplicates`, and `drop_duplicates` appears **nowhere** under `backend/`.
Consumers are production backtest paths. **This is why criterion 2's DSR/PBO
requirement cannot be honestly met today.**

**86.117 (P2) -- the declared/effective weight gap above**, BLOCKED-BY 86.116.

## Numbered operator asks (criterion 5)

- **ASK-1.** Promote `paper_min_k_sectors_analyzed = 3`? Measured: top-sector
  share 72% → 60%, distinct names +2, at the smallest turnover cost of the
  three arms (+2.1pp/day). The brief prefers it because it changes only which
  names reach the deep-analyse slice and does **not** mutate `composite_score`,
  so it leaves the DSR/PBO gates uncontaminated.
- **ASK-2.** Promote `paper_soft_sector_diversity_enabled` with `w = 0.30`?
  Measured: top-sector share 72% → 40%, distinct +5, turnover +6.3pp/day. A
  larger diversity gain for a larger churn cost, and it *does* overwrite
  `composite_score`.
- **ASK-3.** `sector_neutral_momentum_enabled` is measured here for
  completeness (72% → 20%, distinct +10) but is **not recommended**: this
  project's own 2026-06-01 replay measured **-0.166** long-only Sharpe for hard
  sector-neutralisation.
- **ASK-4.** Widen `paper_analyze_top_n` (currently 5)? The step does **not**
  recommend this and the research gate warns against treating slate width as a
  defect. Raised only because the criteria require operator-gated options to be
  named rather than silently declined.

## Scope honesty

This step did **not** widen the slate, promote any flag, write `.env`, add
hysteresis or an incumbent bonus, implement residual momentum, or claim the
trade drought as its consequence (86.47 owns that and is PARKED). It does not
claim 86.60's entry-path fix. Everything it discovered outside its own scope
was filed as its own step rather than absorbed.

---

## Cycle 2 -- response to the CONDITIONAL (`wf_5a3bc88c-4e1`)

**All four findings accepted; none disputed.** The evaluator did not argue that
a guard looked weak -- it **poisoned the baseline arm and showed the run stayed
green** while every criterion-4 delta collapsed to zero. That is sole-coverage
vacuity on this step's most consequential table, the one the operator asks rest
on.

The irony is recorded rather than smoothed over: cycle 1's artifact claims
credit for catching two vacuous guards, and shipped four more in the same file.
Finding a tautology does not immunise you against writing another one.

**1. `panel_is_us_only` was a literal `True`.** I wrote `_ok("panel_is_us_only",
True, ...)` -- a constant, AST-provably unkillable. The fetch SQL does carry
`WHERE market = 'US'`, but a comment about the query is not a check on the data.
Replaced with `panel_carries_no_non_us_symbols`, which interrogates the panel
itself (no ticker may carry an exchange suffix).

**2. `baseline_arm_is_the_unflagged_ranking` asserted `len(x) == len(set(x))` on
a list built as `sorted({...})`** -- true for every possible input, while its
NAME claimed criterion 4's load-bearing property. Replaced with
`baseline_arm_applies_no_flags`, which asserts against the arm **definition**
(`FLAG_ARMS[0] == ("baseline", {})`) -- the thing a poisoning actually mutates
-- plus `flag_arms_are_distinguishable_from_baseline`. The criterion-4 path went
from **2 guards to 4**.

**Both of the evaluator's exact mutations now KILL**, control observed GREEN
first:

```
=== CONTROL first ===
control GREEN, guards ran: ['predicates_reject_known_bad_inputs',
  'flag_arms_all_ran', 'baseline_arm_applies_no_flags',
  'flag_arms_are_distinguishable_from_baseline']

=== FLAG_ARMS[0] poisoned with sector_neutral=True ===
KILLED: INVARIANT FAILED: baseline_arm_applies_no_flags

=== soft-diversity arm made inert (w=0.0) ===
KILLED: INVARIANT FAILED: flag_arms_are_distinguishable_from_baseline
```

**3. The root cause, and the general fix.** Extending the matrix to cover the
six uncovered guards produced **four more survivors**, all the same shape: *any*
`_ok(name, EXPR)` can be defeated by mutating EXPR to always-true, and no
assertion detects that from the inside. So the rules are no longer written
inline. Each is now a **named predicate with a fixture of known-bad inputs it
must reject** (`_us_only`, `_enough_sessions`, `_dedup_fired`,
`_arms_distinguishable`, joined by the existing `_tie_explained`), asserted by
`predicates_reject_known_bad_inputs`. Weaken a rule and the fixture fires,
because the fixture does not depend on today's data being interesting. That
turned all four survivors into kills. One was also a plain logic bug:
`_arms_distinguishable` used `any` where it needed `all`, so emptying a single
arm's kwargs left the guard satisfied.

**4. Coverage is now itself checked.** A cell list proves things about its cells
and nothing about the guards it forgot -- which is exactly finding 2. The matrix
now runs an **AST census** of every `_ok(...)` in the target and fails if any
guard lacks a cell or an explicit `COVERED_TRANSITIVELY` entry with a reason. It
caught a real gap on its first run (`price_only_multidim_arm_ran`, now cell
M19). Adding a guard without a cell fails the matrix.

**5. The sigma triple did not reproduce, and it had reached durable storage.**
Cycle 1 quoted "~10.2 / ~19.4 / ~31.0 ... ~3.0x". Re-derived: **10.646 / 19.850
/ 30.441, ratio 2.86x**. The evaluator also caught that my own headline shares
(22.6/37.0/40.4) reproduce from the *corrected* means, not from the triple I
printed beside them -- so the artifact contradicted itself. Corrected in
`live_check`, in `experiment_results`, and in **`.claude/masterplan.json` step
86.117's `audit_basis`**, where a future step would have read it as measurement.
The script now **prints** the mean sigmas, so the number is computed rather than
retyped and cannot drift again. No conclusion moves: the load-bearing fact is
that the 6m term dominates, and 2.86x carries it.

**6. The `git status` citation was imprecise.** `git status --short -- backend/`
returns three modified files, all from a peer session's in-flight work. The
substantive claim held, but a tree-scoped command cannot attribute a change to a
step in a shared working tree. Both artifacts now cite `git show --name-only
15a817cc`, which can.

**Residual carried forward, disclosed rather than fixed mid-evaluation.** During
cycle 1's EVALUATE I positive-controlled the "no `drop_duplicates` under
`backend/`" claim (0 hits; 3 for a token that must exist; a planted
`drop_duplicates` was detected then removed). Per the freeze-the-tree rule that
evidence was held rather than edited into a file under evaluation; it is
recorded here and belongs to **86.116**, which owns that claim.

---

## Cycle 3 -- response to the second CONDITIONAL (`wf_d1d01d57-0f6`)

**All three residuals accepted.** The evaluator confirmed by execution that
every cycle-1 finding was genuinely fixed, that every published number
reproduces, and that both step commits contain zero production files -- then
found three more. Two were mine to have caught.

**1. A definition is not behaviour.** Cycle 1's guard checked a *value*
(`len(x)==len(set(x))`); I replaced it with one that checks the *definition*
(`FLAG_ARMS[0] == ("baseline", {})`). The evaluator injected `momentum_52wh_tilt`
at the `replay_session` seam, left the definition byte-identical, and the run
stayed green while **min_k's reported delta flipped from +2.1pp to -2.1pp** --
the exact figure ASK-1 rests on. A `w=0.05` variant was worse: every turnover
delta read *exactly as published* while the baseline's top-sector share silently
moved 0.72 -> 0.64.

The fix stops asserting how the baseline was configured. It **recomputes the
baseline slate through a direct, unflagged `rank_candidates` call that bypasses
`replay_session` entirely** and requires the two to agree
(`baseline_slate_matches_an_unflagged_direct_call`, cell M20). An injection
anywhere in the path -- seam, kwargs, wrapper -- makes them diverge. Both of the
evaluator's exact attacks now KILL at the published 20 cycles (15/20 and 20/20
cycles disagreeing), with the control GREEN first and min_k reproducing at
+2.1pp. The criterion-4 path now carries **six** guards, up from two at cycle 1.

I am recording the evaluator's own mitigation rather than only its finding:
`backend/tools/screener.py` contains **zero** `settings.` references, so flags
reach `rank_candidates` only as explicit caller kwargs. This was a code-edit
risk in the measurement harness, **not** something reachable by an operator
promoting ASK-1 or ASK-2.

**2. The §8 evidence block was spliced from two runs.** It carried a coverage
line reading 20/20 beside a sha line from a later run at which the command
prints 21/21. That is the **second non-reproducing number in this step, and both
were mine** -- the first was the sigma triple. §8 is now a single verbatim
capture from one execution, and it says so at the top of the block.

**3. The fixture that backs four cells had no cell of its own.** Emptying
`_PREDICATE_FIXTURE` left `predicates_reject_known_bad_inputs` green, because a
loop over an empty list finds nothing wrong -- and the AST census structurally
could not see it, since the fixture is *data*, not an `_ok(...)` call. Added
`fixture_exercises_every_predicate_on_rejecting_inputs`, which requires every
predicate to appear with a minimum number of **rejecting** cases, plus cell M21.

**4. The matrix was a weaker oracle than the published run.** Every cell ran at
`--cycles 4` while the headline numbers come from `--cycles 20`, and the
evaluator showed kill/survive is cycle-count dependent (a `w=0.15` poison
survived at 4 and died at 20). Criterion-4 cells now run at 20.

**Attempt state.** This is the third Q/A spawn on 86.59. Two CONDITIONALs stand,
so per CLAUDE.md F1 a third would force FAIL regardless of evidence. That is
stated as fact, not as an argument for leniency -- if residuals remain, FAIL is
the correct outcome and the step should park rather than iterate.
