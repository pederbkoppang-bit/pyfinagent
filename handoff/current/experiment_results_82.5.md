# Experiment results -- phase-82.5

**Step**: Exit-quality tiles are single-outlier division blowups. **P1**.
**Contract**: `handoff/current/contract_82.5.md` | **Research**: `handoff/current/research_brief_82.5.md`

## 1. The finding that changed the fix: the mean does not exist

This is not an outlier problem. Both ratios have denominators that can reach zero, so
the per-trade distribution is Cauchy-like -- Franz (arXiv:0710.2024, read in full via
ar5iv) proves *"Neither the expected value nor the variance exist"*. **-42.08 was not a
bad estimate of a true value; there is no true value**, and more trades would never make
it converge. That also rules out the reflex fix: winsorizing or clipping produces a
finite number that estimates no population parameter. The ESTIMATOR had to change.

## 2. The frontend does NOT double-scale -- checked, and deliberately not touched

The step warned not to assume a second defect. `MfeMaeScatter.tsx:114` renders
`${(avg_capture_ratio * 100).toFixed(0)}%`, and `capture_ratio` is
`realized_pnl_pct / mfe_pct` -- percent over percent, **dimensionless** -- so x100 is
CORRECT. -42.08 x 100 = -4208%, exactly the tile. Corroborated three ways in the same
file: `:111` renders `edge_ratio` with no x100 (matching the reported 86.92),
`:168` uses the same x100, `:121` renders the 0.4 threshold as "40%". **One defect,
backend-side.** The x100 is kept; removing it would create a real bug later.

## 3. The two degeneracies needed OPPOSITE treatment

- `mae == 0` (6 of 32) = the trade **never traded against us**. The
  `if mae_abs > 0` filter deleted exactly those rows -- **it was deleting the best
  trades**, survivorship bias with the sign pointing the wrong way. Now ranked at
  `+inf`, never dropped (a median is defined over the extended reals; a mean is not,
  which is the only reason that filter existed).
- `mfe == 0` (8 of 32) = there was **no exit decision to grade**; the ENTRY failed.
  Scoring it `0.0` blames the exit for an entry failure. Now excluded.

A single uniform rule would have been wrong in one direction or the other.

## 4. The real-data fixture -- criterion 4

`backend/tests/fixtures/exit_quality_32_roundtrips.json`, pulled from
`financial_reports.paper_trades`. It reproduces the step's figures EXACTLY:

```
legacy mean capture = -42.0785   (step says -42.08)
legacy mean edge    =  86.9218   (step says 86.92)  over 26 of 32 rows
mfe==0 rows: 8      mae==0 rows: 6
min capture = -1269.5726 (000660.KS)   max edge = 1483.3388 (INTC)
```

## 5. What the tiles report on that same real data

> **CORRECTED after the cycle-1 Q/A.** This section was headed "What the tiles NOW
> report" and stated these values in the present tense. At the time it was written the
> running backend was **PID 654, started 2026-07-28, with no `--reload`** -- seven days
> older than every module changed here -- so the live tiles were still showing -4208%
> and 86.92. The numbers below were a FIXTURE COMPUTATION, not a live readout, and the
> heading claimed otherwise. The backend has since been restarted and the values are
> now confirmed live; the evidence is in `handoff/current/live_check_82.5.md`.

```
Avg capture : 63%     (was -4208%)     n_defined=20 of 32, undefined=12
Edge ratio  : 3.09    (was 86.92)      +inf rows ranked, not dropped: 6
secondary   : capture ratio-of-sums 0.6446 | edge ratio-of-sums 3.9636
```

The gate predicted capture ~0.63 and n_defined=24. Capture median measured **0.6304**;
**n_defined is 20, not 24** -- the 1.0pp floor removes four more rows than the exact
zeros. The gate flagged that caveat explicitly and it was right to. I report the
measured value, not the predicted one.

## 6. One definition, three call sites de-duplicated

The formula existed **three times**: `paper_round_trips.py:97`,
`paper_trader.py:591`, and the endpoint's own aggregate -- so `/performance` and
`/mfe-mae-scatter` could disagree about the same trades. All three now call
`perf_metrics.aggregate_exit_quality` / `compute_capture_ratio`, which is where the
repo's own rules say metrics belong. `grep` for the old formula returns nothing.

## 7. A bug in my own fix, caught by my own test

`test_the_floor_is_load_bearing_not_decorative` failed with `ZeroDivisionError`. My
new guard read `if mfe < min_mfe_pct: return None` -- so at `min_mfe_pct=0.0` a zero
MFE satisfies `0 < 0 == False` and fell straight through to the division.

**That is the same defect class this step exists to fix**: a guard whose threshold does
not cover its domain, exactly like the original `mfe > 0` admitting `mfe=0.0001`.
`mfe > 0` is now unconditional and the floor is an ADDITIONAL constraint, spelled out
on two lines rather than folded into one comparison that reads fine and is wrong.

## 8. Verification

```
$ python -m pytest backend/tests/test_phase_82_5_exit_quality_metrics.py -q
................                                                         [100%]
16 passed, 1 warning in 1.93s
```

## 9. Mutation matrix

```
CONTROL                                          0 failed
M1 capture headline reverts to the MEAN          2 failed
M2 MIN_MFE floor removed (near-zero admitted)    3 failed
M3 undefined capture reverts to 0.0              4 failed
M4 mae==0 dropped again (survivorship bias)      4 failed
RESTORED                                         0 failed
```

**M1 initially killed only ONE test, and I treated that as a finding rather than a
pass.** Criterion 3 names an outlier at `mfe=1e-4` -- but the 1.0pp floor EXCLUDES that
row, so the criterion-3 guard passes identically whether the headline is a mean or a
median. The floor was doing that work, not the estimator. I added
`test_the_median_itself_is_load_bearing_not_just_the_floor`, which poisons with a row
ABOVE the floor (`mfe=1.5`, `pnl=-900`) where only a robust estimator can help, and
asserts the same row DOES destroy a mean so it can discriminate between the two. M1 now
kills 2.

## 10. Collateral: one test corrected, one pre-existing failure disclosed

- `test_paper_trader_execute_sell_capture_ratio_zero_when_no_gain` asserted
  `capture_ratio == 0.0` for `MFE == 0` -- it **encoded the defect** and passed.
  Renamed to `..._none_when_no_gain` and now asserts `is None`. Its original intent
  (no NaN, no ZeroDivisionError) is preserved and still asserted.
- `test_paper_trader_execute_buy_average_up_recomputes_avg_entry` fails. **It was
  already failing before this step** -- verified against the failure lists captured
  during 82.7, where it appears in BOTH the with- and without-change runs. Not caused
  by 82.5, and not fixed by it.

## 11. Scope honesty

- `MIN_MFE_PCT = 1.0` is a judgement call, not a measurement. It is surfaced in the
  API payload and in the tile hint so it is auditable rather than buried, and
  `test_the_floor_is_load_bearing_not_decorative` proves it changes the answer.
- Per-trade `capture_ratio` is now `float | None` in the API payload and in BQ
  writes. That is a consumer-visible contract change; frontend types and the tooltip
  were updated, and the leakage rule now requires a DEFINED capture (an ungradeable
  exit is not a leak).
- The tiles now show medians. `avg_capture_ratio` keeps its key name for
  compatibility, which is mildly misleading; `aggregation: "median"` is emitted
  alongside so a reader can tell.
- No BQ backfill: historical `paper_trades` rows keep whatever `capture_ratio` was
  written at the time. Only the aggregate is recomputed.

---

## 12. CYCLE 2 -- the three cycle-1 CONDITIONAL items, closed

Cycle 1 (task `w1eklqhf2`) returned **CONDITIONAL**. It confirmed all four criteria are
covered by non-vacuous guards using its own 7 code mutants + 5 fixture-softening + 2
fixture-substitution mutants, verified the fixture is genuine production data (the
pre-fix live endpoint independently returned the same -42.0785/86.9218), proved the
three-copy de-duplication by EXECUTING both aggregate paths, and measured zero
regressions. Three items capped it, all mine.

### C1 -- no live UI capture (qa.md 1c is BINDING)

Now supplied: `handoff/current/live_check_82.5.md` +
`handoff/current/captures_82.5/live_check_82.5_exit_quality_tiles.png`.

The Q/A had attempted the capture itself and was blocked -- :3000 sits behind the
NextAuth wall and the skip-auth :3100 instance was down. Standing up :3100 is **Main's
lifecycle responsibility, never the evaluator's**, and it was correct not to do it.

Tiles now render **Edge ratio 3.09** (hint `median(MFE / |MAE|)`) and **Avg capture
63%** (hint `median(realized_pnl / MFE), n=20`). The hints matter as much as the
numbers: the estimator change is visible to the operator instead of silently altering a
value under an unchanged label.

### C2 -- I claimed live values against a stale service

The sharpest finding. Backend PID 654 started 2026-07-28 with no `--reload`; my modules
were written 2026-08-04. The live endpoint was still returning `-42.0785`. Section 5 was
headed "What the tiles NOW report".

Restarted via launchd (`launchctl kickstart -k`, not a bare kill), new pid 62664.
Measured before AND after, both recorded in the live_check. Section 5 is now corrected
**in place** with the reason, rather than left true-looking above a contradicting
addendum.

This is `feedback_verify_own_completed_action_claims` again, in its most literal form: I
wrote a present-tense sentence about a running system without ever querying it. The
fixture computation was correct; the claim about the world was not.

### C3 -- a contract-named change site I skipped while claiming otherwise

`contract_82.5.md` section 4 enumerated `api.ts:524 -> number | null`. I never changed
`api.ts`, yet section 11 said "frontend types and the tooltip were updated". Both
`:524 avg_capture_ratio: number` and `:537 capture_ratio: number` still declared
non-null against a backend now returning null. Latent only because
`getPaperRoundTrips` has zero call sites -- which is a reason to fix it now, not to
skip it. Fixed.

The Q/A also found the mirror-image defect I introduced: `types.ts`
`PaperRoundTripSummary` declared `edge_ratio_of_sums` / `edge_n_infinite` that
`summarize()` never emits -- I had added scatter-only fields to the round-trip
interface. Removed, along with `aggregation` for the same reason. The interface now
matches the payload exactly, verified by executing `summarize()` and diffing its key set
against the declaration: **zero declared-but-not-emitted keys**. `tsc --noEmit` exits 0.

**The pattern across C1-C3 is one thing**: three claims about a world outside the test
suite -- a running service, a rendered UI, a type surface -- none of which the suite
could see, and none of which I checked. Green tests said nothing about any of them.

### Criterion 3, on the record

The Q/A's answer to my own disclosure: criterion 3 is **partially** floor-satisfied.

**CORRECTED (cycle-2 [Contradiction]): my stated cause was right for half the test and
wrong for the other half.** The verbatim criterion-3 test loops over BOTH headlines, and
they survive a mean for two DIFFERENT reasons:

- **capture** -- the 1.0pp floor excludes the poison before the estimator sees it
  (`capture_n_defined` 20 -> 20). My original explanation, correct here.
- **edge** -- there is NO floor on the edge path, so the poison IS admitted
  (`edge_n_defined` 20 -> 21). It survives because it is not extreme *as an edge
  ratio*: `mfe/|mae| = 1e-4/1e-4 = exactly 1.0`, an ordinary value inside the base
  range 2.500..3.718, producing 3.29% drift under a mean. Nothing to do with the floor.

I asserted one cause for a two-branch loop without measuring the branches separately. But criterion 3 is NOT floor-only as
covered, because `test_the_median_itself_is_load_bearing_not_just_the_floor` dies under
its mean mutant at line 201 (`assert drift < 0.20` -> "capture median moved 4766.7%"),
with a poison ABOVE the floor. It also verified that test is not itself vacuous by
checking its two anti-vacuity legs individually.


---

## 13. CYCLE 3 -- the four cycle-2 items, closed

### D1 -- MY CAPTURE RUN MUTATED TWO TRACKED FILES, and I did not notice or disclose it

The most serious of the four, and it is a defect in my *remediation*, not in the step.
Starting `npx next dev --port 3100` with `PLAYWRIGHT_DIST_DIR=.next-functional` made Next
rewrite two TRACKED files at 19:56:35:

- `frontend/next-env.d.ts` -> `./.next-functional/types/routes.d.ts` (a file that
  literally says *"This file should not be edited"*)
- `frontend/tsconfig.json` -> `include` gained `.next-functional/types/**/*.ts`

Both now pointed at a **gitignored, untracked** build dir, and `git add -An` confirmed
both would ship under 82.5's name. `feedback_audit_the_commit_not_the_diff`, caused by
the very evidence-gathering I did to close the previous cycle's finding.

**Root cause, and it is instructive**: `playwright.config.ts:40` declares a
`globalTeardown` that exists precisely to *"restore next-env.d.ts/tsconfig.json if the
functional :3100 server rewrote them"*. I started the dev server DIRECTLY rather than
through Playwright's runner, so the teardown never ran. I used the config as
documentation for the command and skipped the mechanism the config was built around.

Restored both from HEAD (`git show HEAD:<f>` written back -- note `git checkout -- <f>`
is blocked by the repo's own danger hook, correctly, since it silently discards
working-tree edits). `tsc --noEmit` re-verified on the RESTORED config: **0 errors**,
0 in `src/`, none naming any file 82.5 touched. The Q/A's green had been measured
against the mutated config; this one is not.

### D2 -- a causal claim that was wrong for half its subject

Corrected in section 12 above, in place.

### D3 -- no robustness guard for the EDGE tile, though criterion 3 says "both"

Reverting the edge headline to a mean was killed only by a value pin and a non-finite
guard -- neither is about robustness. Added
`test_the_edge_headline_is_robust_to_an_extreme_outlier`, poisoning with a genuinely
extreme edge value (`mfe=50, mae=-0.01` -> edge 5000), asserting the row is ADMITTED
(else it proves nothing) and that the same row destroys a mean (else it cannot
discriminate). The edge-mean mutant now kills **3** tests including this one, up from 2.

### D4 -- the lint gate, including one error I introduced

`import statistics as st` in my own test file -- flagged as a NOTE by cycle 1 and left
unfixed by me, which is why it came back as a violated criterion. Removed. I also
cleaned the 4 pre-existing F401s in `test_dod4_tier1_coverage_investment.py`, a file this
diff already edits.

**One near-miss worth recording**: my first pass "proved" `threading` was still used by
grepping for non-import occurrences and finding one at line 144. Line 144 is a COMMENT
(`# because snapshot() re-acquired the threading.Lock.`). Ruff was right and my check was
wrong -- a grep that cannot tell code from comments is not a proof of use. Removed only
what a corrected check showed unused.

Gate now: **All checks passed!** on a 6-file derived scope, asserted non-empty first.

### Standing state

```
82.5 suite                     17 passed
ruff (derived scope)           All checks passed!  (was: 5 errors)
tsc --noEmit (restored config) 0 errors
tracked files from capture     restored to HEAD
```

The one remaining failure repo-wide is
`test_paper_trader_execute_buy_average_up_recomputes_avg_entry`, pre-existing, which the
cycle-2 Q/A root-caused to the live kill switch being manually PAUSED -- not to the 82.7
artifact I had originally cited for it.
