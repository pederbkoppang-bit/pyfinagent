# Contract -- phase-82.21

**Step:** 82.21 (P1) -- `historical_fundamentals` starts at 2024-06-30, so ~81%
of the 2018-2025 walk-forward window has no fundamentals at all, and nothing
records it.
**Date:** 2026-08-06. **Cycle:** 1.
**Research gate:** PASSED -- `handoff/current/research_brief_82.21.md`, envelope
`gate_passed: true`, tier `complex`, 8 external sources read in full, 41 URLs,
recency scan performed, 13 internal files inspected. Workflow rail.

---

## 1. Research-gate summary

Full brief: `handoff/current/research_brief_82.21.md`. The findings this
contract is built on, each with its anchor:

1. **The step's headline measurement reproduces exactly.** Main re-ran it live
   2026-08-06 against `financial_reports.historical_fundamentals`:
   `n_rows=4798, n_tickers=503, min(report_date)='2024-06-30',
   max='2026-02-28', COUNTIF(report_date < '2024-06-30')=0`.
2. **`report_date` is a BigQuery STRING column** (measured via
   `client.get_table(...).schema`). `MIN()` is therefore lexicographic, and it
   is only correct because the producer writes zero-padded ISO
   (`data_ingestion.py:257`, `strftime("%Y-%m-%d")`). An `isinstance(v, date)`
   guard over these rows would never fire -- the dead-guard class already
   recorded in this project's memory.
3. **THE FINDING THAT CHANGES THE DECISION: the existing table has no vintage
   at all.** `data_ingestion.py:278` writes `"filing_date": report_date` with
   the comment *"true filing date not available from yfinance"*, and
   `cache.py:612/:631` filter `report_date <= cutoff`. Measured mean
   publication lag is 66 days (median 60, p90 90). So the *covered* 2024-07+
   window also leaks look-ahead. Fundamentals there are not point-in-time
   either.
4. **~20 months cannot support inference under this project's own gates.**
   MinBTL (Bailey, Borwein, Lopez de Prado, Zhu, *Notices of the AMS* 61(5),
   2014, Thm 2) inverts at 1.67 years to roughly 2.3 independent
   configurations. The optimizer runs far more. Against DSR >= 0.95 / PBO <=
   0.5 / beat-incumbent-OOS, a 2024-07..2026-02 sample is not a shortened
   evaluation window -- it is no evaluation.
5. **EDGAR companyfacts IS point-in-time capable.** Every fact carries `filed`
   + `accn` (measured live). Restatement demonstrated on AAPL `Assets`
   @2008-09-27: `$39.572B` (10-Q filed 2009-07-22) -> `$36.171B` (10-K/A filed
   2010-01-25); 18 of 70 end-dates are multi-vintage. Cost is ~504 requests /
   a few minutes of network, but **3-5 days of tag-normalisation work** plus an
   annually recurring taxonomy obligation: `total_debt` has no single us-gaap
   tag, revenue drifts three ways across ASC 606 within one filer, and
   3/6/9/12-month duration facts share the same `end` (a ~4x magnitude trap).
6. **Exactly ONE selectable strategy is label-fundamentals-dependent: `qarp`**
   (`backtest_engine.py:1589-1592`, hard-refuses at `:1594-1595`). Derived by a
   written-down rule, not eyeballed, and recall-tested against two negative
   controls (`triple_barrier` never calls `build_feature_vector`;
   `stretch_regime` looks dependent but reads only price/vol).
   `quality_momentum` and `factor_model` are dependent but already demoted
   (`:52-67`).
7. **But ALL SIX are FEATURE-fundamentals-dependent** via `_NUMERIC_FEATURES`
   (`:124-136`), and it fails silently two different ways: on a fully-uncovered
   window `:852` drops the 15 columns and the model trains on 22 features
   instead of 37 with no record; on a straddling window `:881-882`
   `fillna(train_medians).fillna(0)` fabricates a median company for
   four-fifths of the sample -- which is strictly worse, and is the current
   behaviour on the exact window the step names.
8. **`quality_momentum:1283` is criterion 2's pathology in production code:**
   `fv.get("quality_score", 0) or 0` makes `> 0.3` unreachable and `< 0.1`
   always true when fundamentals are absent -- a structurally bearish label
   manufactured out of missing data.
9. **Criterion 3 has an existing mechanism to reuse:** phase-82.13's
   `data_availability` (default `backtest_engine.py:209`, recorded by
   `_preload_macro_and_record` `:368-400`, labelled at construction
   `:450-458`, double-surfaced at `analytics.py:843-845` because two consumers
   read only `report["analytics"]`). Do not invent a parallel one.
10. **`sec.gov` returns HTTP 403 to a default user agent**; a declared
    `User-Agent: <name> <email>` returns 200. The repo already knows this
    (`.claude/rules/security.md`).

---

## 2. The operator's decision (verbatim), and its derivation

Recorded verbatim from the operator's session directive of 2026-08-06:

> "APPROVAL. I approve every gated step. That unblocks proceeding, but two
> criteria need a recorded DECISION, not just consent -- my standing
> constraints decide them. Record my words plus the derivation; do NOT invent a
> quote from me:
> - 82.21 fundamentals source: free only -> either accept that fundamentals-
>   dependent strategies are evaluable from 2024-07 on, or build SEC EDGAR
>   XBRL. Do NOT adopt a paid source."

The source sentence again, unindented and unquoted so it is byte-exact (this
block is what the criterion-4 guard matches):

```
82.21 fundamentals source: free only -> either accept that fundamentals-
  dependent strategies are evaluable from 2024-07 on, or build SEC EDGAR XBRL.
  Do NOT adopt a paid source.
```

### The derivation, including the part the operator will not like

The instruction offers two branches and forbids a third. Measurement
**falsifies the premise of Branch A as worded**:

| Branch | Status | Evidence |
|---|---|---|
| Adopt a paid source (Sharadar / FMP / Polygon) | **FORBIDDEN** -- "Do NOT adopt a paid source" | operator, verbatim above |
| A: "accept that fundamentals-dependent strategies are **evaluable from 2024-07 on**" | **PREMISE FALSIFIED** | the covered window has no vintage (`data_ingestion.py:278`), leaking a measured 66-day mean publication lag; and MinBTL admits ~2.3 independent trials at 1.67 years against gates of DSR >= 0.95 / PBO <= 0.5. Branch A is not "a shorter evaluation window", it is **not evaluable**. |
| B: "or build SEC EDGAR XBRL" | **CHOSEN** | free; the only path that yields real history AND a true `filed` vintage; measured feasible today (`filed`+`accn` present on every fact) |

**DECISION: build SEC EDGAR XBRL.**

Derived from the operator's own standing constraints rather than invented:
`$0` spend is satisfied (EDGAR is free); the north-star charter requires
research-grade evaluation behind immutable promotion gates, and Branch A cannot
clear those gates *at any window length available from yfinance*, so choosing it
would permanently foreclose a strategy family on a premise the measurement
disproves. Branch B costs 3-5 days of free work and is strictly better on three
independent axes (history depth, true vintage, and as-filed vs standardized
signal quality -- Du/Huddart/Jiang 2021 find accruals from as-filed data predict
returns while the standardized version does not).

**What this step does NOT do.** The EDGAR ingester is not built here. None of
82.21's four criteria requires it, and its verification command is a single
pytest module. The build is **queued as its own research-gated step**; 82.21
ships the structural visibility that makes the current state honest in the
meantime. Under either branch 82.21's code would be identical -- the decision
determines which follow-on step exists, not what this step builds.

**Owed back to the operator, stated plainly:** Branch A as you worded it is not
available. If you would rather stop here than fund 3-5 days of EDGAR work, the
honest form of that choice is *"fundamentals-dependent strategies are retired,
not evaluated"* -- not *"evaluable from 2024-07"*. Say the word and the queued
EDGAR step gets dropped; this step's code is unaffected either way.

---

## 3. Hypothesis

Absent fundamentals are currently **indistinguishable from present-but-null**
fundamentals at every layer: the feature builder omits 17 keys with no signal
(`historical_data.py:140`, a bare `if fundamentals:` with no `else`), the
training matrix either drops the columns (`:852`) or imputes a fabricated median
company (`:881-882`), and the result object records nothing. Making the absence
**explicit at the feature level** and **recorded at the result level**, plus
**refusing** for the derived label-dependent set, converts a silent
data-quality hole into a visible one.

Falsifiable predictions:
- An uncovered-cutoff fixture yields `fundamentals_available is False` with the
  17 keys absent; a covered fixture yields `True` with real numeric ratios; a
  covered-but-loss-making fixture yields `True` **and** `pe_ratio is None`.
- A `qarp` backtest whose window starts before the coverage start refuses; a
  `triple_barrier` run over the same window proceeds but carries
  `data_availability["fundamentals"] is False`.

---

## 4. Immutable success criteria (verbatim from `.claude/masterplan.json`)

1. "a test asserts the measured earliest report_date in historical_fundamentals
   and fails if a later claim about coverage is made without re-measuring"
2. "the feature builder reports fundamentals-derived features as EXPLICITLY
   unavailable for a cutoff before the coverage start, rather than silently
   returning None indistinguishable from a genuine null"
3. "a backtest whose active strategy depends on fundamentals refuses to run, or
   records an explicit coverage warning in its result, when the requested
   window starts before the coverage start -- asserted on a fixture"
4. "the operator's source decision is recorded verbatim in the step artifact"

**Verification command (immutable):**
`source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_21_fundamentals_coverage.py -q`

### Seam-and-mutant map (GUARDS rule: drive the seam the criterion names)

| # | Seam | Guard drives | Production mutant that must kill it |
|---|---|---|---|
| 1 | the checked-in coverage snapshot + the constant production code believes | the snapshot loader and the drift check, plus a regex asserting the ISO format that makes a lexicographic `MIN` valid | edit the constant to a later date without re-measuring; corrupt the date format |
| 2 | `historical_data.py:140` `if fundamentals:` and the `else` that does not exist today | `build_feature_vector` itself, three-way | hardcode the flag to `False`; hardcode to `True`; flip `if fundamentals:` to `if True:`; rename `None` to a sentinel without making the two states distinguishable |
| 3 | `_preload_fundamentals_and_record()` + the refusal branch + `data_availability` | a real engine run on a fixture, per 82.13's template | delete the refusal; replace the derived dependent-set with a literal that goes stale; drop `fundamentals` from the availability dict |
| 4 | the artifact on disk | a test asserting the byte-exact operator sentence AND the recorded DECISION line | paraphrase the instruction; delete the decision line |

---

## 5. Plan

**D1 (criterion 1) -- pin the measured coverage where production can see it.**
A checked-in snapshot `backend/backtest/_fundamentals_coverage.json` holding
`{min_report_date, n_rows, n_tickers, measured_at, date_format}`, plus a
constant in production code so the test guards *the code's belief*, not just its
own literal. Modelled on `schema_oracle`'s snapshot/drift shape (`:84/:116/
:130/:136`). Default test path is offline; a live re-measure sits behind an env
flag and uses `dry_run` (`schema_oracle.py:550`) so it stays `$0`. The test
asserts the ISO format with a regex -- **not** an `isinstance(v, date)` guard,
which is dead on a STRING column.

**D2 (criterion 2) -- explicit unavailability at the feature builder.**
`historical_data.py:140`: set `features["fundamentals_available"] = bool(...)`
on BOTH branches. The discriminating predicate then exists:
`pe_ratio is None AND fundamentals_available is True` = genuine null (a
loss-making company -- `pe_ratio` is only assigned when `net_income > 0`,
`:156-158`); `pe_ratio is None AND fundamentals_available is False` =
structural. Today those two are byte-identical, which is what the criterion
forbids.

**`fundamentals_available` MUST NOT be added to `_NUMERIC_FEATURES`.** It is a
perfect proxy for `date >= 2024-07`; as a model input it is a regime dummy and
the classifier would learn the coverage boundary instead of the economics. A
guard asserts it is absent from that list.

**D3 (criterion 3) -- refuse for the dependent set, record for everyone else.**
Criterion 3 is a disjunction; the brief's evidence says do both, because the
label-level and feature-level exposures differ:
- `_preload_fundamentals_and_record()` beside `_preload_macro_and_record`,
  extending the `data_availability` default at `:209` to carry `fundamentals`
  (defaulted, so existing construction sites are untouched -- 82.13's own
  technique), labelled at construction `:450-458`, and double-surfaced at
  `analytics.py:843-845` because two consumers read only `report["analytics"]`.
- **REFUSE** when the resolved strategy is in the label-dependent set AND the
  window starts before the coverage start. The set is **derived** by the Q6 AST
  rule (keys assigned only inside the `if fundamentals:` block, intersected
  with what each label function reads), never a hardcoded `{"qarp"}` -- a literal
  goes stale the next time a strategy is added, which is the exact
  measure-don't-assert defect class.

**D4 (criterion 4)** -- the decision block in §2, with the round-trip verified
in the same turn it is written.

**Queued, not built here** (each gets its own research-gated step):
the EDGAR XBRL ingester; the 90-day publication-lag embargo (`cache.py:612/
:631`); `quality_momentum`'s `or 0` bearish-label landmine (`:1283`); the silent
feature-set shrink at `:852`; the `filing_date` projection drift
(`cache.py:283-292` vs `:626-634`).

## 6. Non-scope

No EDGAR ingester. No change to `cache.py`'s cutoff filter (the embargo is its
own step -- changing it here would alter every backtest's inputs inside a step
about *visibility*). No re-registration of the demoted strategies. No paid
source. No live positions touched; paper trading untouched.

## 7. References

- `handoff/current/research_brief_82.21.md` (the gate)
- Bailey, Borwein, Lopez de Prado, Zhu, "Pseudo-Mathematics and Financial
  Charlatanism", *Notices of the AMS* 61(5), 2014 -- MinBTL, Thm 2
- Du, Huddart, Jiang (2021) -- as-filed XBRL vs Compustat accruals
- SEC EDGAR XBRL frames/companyfacts API -- https://www.sec.gov/edgar/sec-api-documentation
- scikit-learn imputation / `MissingIndicator` docs
- Internal: `backend/backtest/historical_data.py:140-266`,
  `backend/backtest/backtest_engine.py:52-84,124-136,209,368-400,450-458,852,881-882,1273-1290,1357-1397,1572-1595`,
  `backend/backtest/data_ingestion.py:228-278`, `backend/backtest/cache.py:283-292,612,626-634`,
  `backend/db/schema_oracle.py:84,116,130,136,265,550`,
  `backend/tests/test_phase_82_13_preload_refusal_handling.py:184-218,419-425`
