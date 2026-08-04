# Contract -- phase-82.5

**Step**: Exit-quality tiles are single-outlier division blowups. **P1**.

## 1. Research gate -- PASSED

`handoff/current/research_brief_82.5.md`, Workflow rail task `ws1xvyys5`, envelope
archived at `handoff/current/qa_returns/ws1xvyys5.output.json`.

```
gate_passed=true  tier=moderate  external_sources_read_in_full=7
snippet_only=19   urls_collected=26  recency_scan_performed=true
internal_files_inspected=14
```

### The finding that changes the step: THE MEAN DOES NOT EXIST

This is not an outlier problem and it must not be fixed like one. A ratio whose
denominator can approach zero is Cauchy-like. Franz (arXiv:0710.2024, read in full via
ar5iv) proves *"Neither the expected value nor the variance exist"*, and that *"the mean
of independent, identically Cauchy-distributed variables follows the SAME Cauchy
distribution as each of the individual variables."*

So **-42.08 is not a bad estimate of a true value -- there is no true value**, and more
trades will never make it converge. That also rules out the obvious fix: winsorizing or
clipping yields a finite number that estimates no population parameter. The estimator
has to change, not the tails.

### The frontend does NOT double-scale -- and I must not "fix" it

The step told me to check this separately and not assume. Answer: **NO**.
`frontend/src/components/MfeMaeScatter.tsx:114` renders
`${(data.summary.avg_capture_ratio * 100).toFixed(0)}%`. `capture_ratio` is
`realized_pnl_pct / mfe_pct` -- percent over percent -- so it is **dimensionless**, and
x100 to render a percentage is CORRECT. -42.08 x 100 = -4208%, exactly the observed
tile. Corroborated three ways in the same file: `:111` renders `edge_ratio` with no x100
(matching the reported 86.92), `:168` uses the same x100 in the tooltip, `:121` renders
the 0.4 threshold as "40%".

**Do not touch the formatter.** If the backend were later changed to emit an
already-percent value, that multiply would become a real double-scale. One defect,
backend-side.

### The two degeneracies need OPPOSITE treatments -- this is the crux

- `mae == 0` (6 of 32) means the trade **never traded against us**. That is a genuine,
  desirable, measurable property. The `if mae_abs > 0` filter at
  `paper_trading.py:1002-1003` therefore **deletes the best trades** -- a survivorship
  bias whose sign points the wrong way. KEEP them; rank `+inf`, do not drop.
- `mfe == 0` (8 of 32) means there was **no exit decision to grade** -- the ENTRY
  failed. Including it at a fabricated `0.0` blames the exit for an entry failure.
  EXCLUDE it.

So: exclude for capture, redefine for edge. A single uniform rule would be wrong.

### `mfe == 0` is a clamp artefact, and it matches industry convention

`paper_trader.py:718-721` seeds `prev_mfe` from `or 0.0` then takes
`max(prev_mfe, pnl_pct)`, so MFE is floored at 0 by construction: a trade whose best
mark was -3% records `mfe=0` and the true MFE is unrecoverable. TradingDiaryPro (read in
full) states the same convention -- *"if the trade ... was never in the profit zone then
the MFE is zero"*. This is a **domain restriction to respect, not a bug to fix**.

### A median is defined over the extended reals; a mean is not

`+inf` rows can be RANKED rather than deleted, so a median needs no exclusion while
fewer than half the rows are degenerate (19% here). The existing filter exists only
because a mean cannot absorb `+inf`.

## 2. Immutable success criteria -- copied VERBATIM from `.claude/masterplan.json`

Command: `source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_5_exit_quality_metrics.py -q`

1. a fixture round-trip with MFE == 0 does not produce an unbounded or sign-flipped capture value in the aggregate
2. a fixture round-trip with MAE == 0 is handled explicitly rather than silently dropped from the edge-ratio aggregate
3. the reported aggregate for both tiles is robust to a single extreme outlier: adding one round-trip with mfe/mae = 1e-4 to a fixture of ordinary trades moves the reported value by less than 20 percent
4. a test pins the pre-fix behaviour by asserting the OLD mean formula would have returned a value with magnitude greater than 40 on the committed real-data fixture, so the guard cannot silently regress

## 3. The metric definition to implement

**Per-trade capture** (nullable, domain-restricted): `None` if
`mfe_pct < MIN_MFE_PCT` (= 1.0 percentage point), else `realized_pnl_pct / mfe_pct`.
The 1.0pp floor is justified because every published interpretive threshold
(0.40 / 0.60 / 0.75) is a fraction of an economically meaningful move; below ~1pp the
"available move" sits inside the round-trip cost+noise band
(`paper_transaction_cost_pct` applied twice at `paper_trader.py:575`). This ONE
threshold subsumes both pathologies -- it removes the 8 `mfe==0` rows AND the
`000660.KS` row at `mfe=0.0001`. `MIN_MFE_PCT` is the one free parameter; surface it in
the payload.

**Aggregate capture**: headline = MEDIAN over the defined subset. Secondary =
ratio-of-sums `sum(pnl_pct)/sum(mfe_pct)` over the same subset, guarded `denom > 0` else
`None`. Emit `n_defined` AND `n_undefined`; the tile shows `n_defined`, not `n_points`.

**Per-trade edge** (extended-real, NEVER dropped): `+inf` if `|mae|==0 and mfe>0`;
`None` if `|mae|==0 and mfe==0`; else `mfe_pct/|mae_pct|`.

**Aggregate edge**: headline = MEDIAN over ALL rows with `+inf` RANKED. If the median
itself lands on an `+inf` row, return `None` and disclose -- never render `Infinity`.
Secondary = `sum(mfe)/sum(|mae|)`.

Every degenerate case, enumerated: (a) `mfe>=1.0, mae<0` both numeric; (b)
`mfe>=1.0, mae==0` capture numeric, edge `+inf` ranked; (c) `0<mfe<1.0` capture `None`,
edge numeric if `mae!=0`; (d) `mfe==0, mae<0` capture `None`, edge `0.0` (a real,
meaningful zero); (e) `mfe==0 and mae==0` both `None`; (f) `n_defined==0` -> tile shows
"n/a", NOT 0%; (g) `sum(mfe)==0` -> ratio-of-sums `None`.

## 4. Change sites (re-derived by the gate; re-verify before editing)

`paper_round_trips.py:97, :141, :157` (+ payload `:159-170`); `paper_trading.py:1001`,
`:1002-1003`, `:1027`, `:1031-1032`; `paper_trader.py:591`; frontend `types.ts:766` +
`api.ts:524` -> `number | null` + a presence-discriminating tile
(`MfeMaeScatter.tsx:111,114,168`), **keeping the x100**;
`backend/tests/test_paper_trading_v2.py:235-243` (the existing test asserts key presence
only and cannot fail on any value).

**There is a SECOND, INDEPENDENT copy of the mean** at `paper_round_trips.py:157`, which
feeds `/performance -> round_trip_summary`. Patching only the endpoint would leave the
two surfaces disagreeing -- INV5 below exists to catch exactly that.

## 5. Mutation-resistant invariants (from the gate; each names what turns it red)

- **INV1** the real `000660.KS` row `{mfe_pct: 0.0001, realized_pnl_pct: -0.13}` must not
  move the aggregate outside `[-3, 3]` -- reverting `:1032` to the mean turns it red.
- **INV2** `n_defined + n_undefined == n_points`, and the edge denominator count ==
  `n_points` -- restoring `if mae_abs > 0` turns it red.
- **INV3** an all-`mfe==0` fixture returns `None` and renders "n/a" -- restoring
  `else 0.0` turns it red.
- **INV4** deleting the single most extreme row moves each headline by < 0.10 (a mean
  fails by ~40).
- **INV5** `/performance` and `/mfe-mae-scatter` agree on the same trade list --
  patching only `paper_trading.py` turns it red.

## 6. Predicted fixture values -- TO BE RE-MEASURED, NOT ASSERTED

The gate predicts capture median ~0.63 and edge median >= 0.81, and explicitly warns
these are predictions: 0.63 holds *only* if the excluded set is exactly the 8 `mfe==0`
rows, whereas the 1.0pp floor also removes `000660.KS` and can shift the median by one
order statistic. **I will measure the fixture and report what it actually returns**;
carrying a predicted number into an assertion is the `feedback_measure_dont_assert_claims`
failure.

## 7. Scope honesty

- Criterion 4 requires a **committed real-data fixture**. The 32 round-trips are live
  paper-trading data; the fixture must be derived from them and committed, and it must
  not contain anything that is not already in the repo's own data domain.
- `MIN_MFE_PCT = 1.0` is a judgement call, not a measurement. It is surfaced in the
  payload so it is auditable rather than buried.
- Changing an aggregate that other consumers read is the risk; the gate enumerated
  consumers and INV5 pins the two that exist.

## 8. References

- Franz, arXiv:0710.2024 (read in full via ar5iv) -- non-existence of the Cauchy mean
- John Sweeney, *Maximum Adverse Excursion* -- the canonical MFE/MAE text
- TradingDiaryPro MFE/MAE documentation (read in full) -- the `mfe==0` convention
- `handoff/current/research_brief_82.5.md` -- full brief, 26 URLs
