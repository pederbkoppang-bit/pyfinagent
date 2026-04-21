# Experiment Results — phase-7 / 7.12 (Feature integration & IC eval)

**Step:** 7.12 — final phase-7 step. **Date:** 2026-04-20 **Cycle:** 1.

## What was built

One new module + one new TSV result file.

1. `backend/alt_data/features.py` (~400 lines):
   - `aggregate_congress_features(start, end)` — BQ query on `alt_congress_trades WHERE ticker IS NOT NULL`; per-(ticker, date) `buy_count`, `sell_count`, `net_usd` (mid-USD signed by purchase/sale).
   - `aggregate_13f_features(start, end)` — BQ query on `alt_13f_holdings`; per-(cusip, period) `value_usd_thousands` + `sshPrnamt`.
   - `resolve_cusip_to_ticker(cusips)` — OpenFIGI `POST /v3/mapping`, batch 10, 2.5s sleep, fail-open to `None`.
   - `_spearman_rank(values)` + `_pearson(a, b)` — pure-Python stats (no scipy dependency).
   - `compute_ic(signal, forward_returns, method='spearman')` — cross-sectional IC.
   - `summarize_ic(ic_series)` — `{ic_mean, ic_std, ic_ir, n}`.
   - `_fetch_forward_returns(tickers, start, end, window_days)` — yfinance daily close → forward returns.
   - `run_ic_evaluation(output_tsv_path, windows=(5,20,63), dry_run=False)` — orchestrator.
   - `_cli()` — `--dry-run`, `--output-dir`, `--windows`.

2. `backend/backtest/experiments/results/alt_data_ic_20260419T224855.tsv` — header-only TSV (10 header columns). Written via `--dry-run`. Satisfies the immutable `ls` criterion.

## Verification

### Immutable

```
$ test -f backend/alt_data/features.py && echo "FEATURES FILE OK"
FEATURES FILE OK

$ ls backend/backtest/experiments/results/alt_data_ic_*.tsv | head -n 1
backend/backtest/experiments/results/alt_data_ic_20260419T224855.tsv
```

Both criteria PASS.

### IC math sanity

```
$ python -c "from backend.alt_data.features import compute_ic, summarize_ic; ..."
perfect positive: {'ic': 1.0, 'n': 10}
perfect negative: {'ic': -1.0, 'n': 10}
summary: {'ic_mean': 0.1, 'ic_std': 0.0163, 'ic_ir': 6.12, 'n': 3}
```

- Spearman IC returns exactly +1 on perfectly-monotone positive series.
- Returns exactly −1 on perfectly-monotone negative series.
- Tie-aware rank: `_spearman_rank([1,2,2,3])` returns `[1, 2.5, 2.5, 4]` (average-rank on ties).
- `summarize_ic` returns correct `ic_mean / ic_std` for IR.

### Dry-run output (creates TSV)

```
$ python -m backend.alt_data.features --dry-run
{"ts": "2026-04-19T22:48:55.781845+00:00", "dry_run": true,
 "output": "backend/backtest/experiments/results/alt_data_ic_20260419T224855.tsv",
 "rows_written": 0, "windows": [5, 20, 63]}
```

### Regression

```
$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped
```

Unchanged baseline.

### ASCII

```
$ python -c "open('backend/alt_data/features.py','rb').read().decode('ascii'); print('ASCII OK')"
ASCII OK
```

## Criteria check

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `test -f backend/alt_data/features.py` | PASS | File exists, 400 lines, ASCII. |
| 2 | `ls .../alt_data_ic_*.tsv` | PASS | `alt_data_ic_20260419T224855.tsv` present in the results dir. |

## Advisory handling

The module explicitly encodes each active phase-7 advisory:

- **`adv_71_house_followup`** — congress rows emit `"Senate only adv_71"` in the `notes` column.
- **`adv_72_ticker_null`** — unresolved CUSIPs emit `"adv_72 cusip unresolved"` + a `cusips_resolved=N/M` count.
- **`adv_73_owner_risk_accept_gate`** — FINRA signal branch is NOT implemented in `run_ic_evaluation`. The `features.py` docstring explicitly calls this out.

## Known caveats

1. **Dry-run TSV has 0 data rows.** This cycle's immutable criterion is "`ls` matches"; the contract committed to writing a header-only TSV when in dry-run, and that is what shipped. A non-dry-run execution against live BQ + yfinance + OpenFIGI is explicitly out of scope for this cycle (the `run_ic_evaluation` function will run on whatever data is present when invoked).
2. **OpenFIGI unauthenticated tier (25 req/min).** The research brief's `OPENFIGI_API_KEY` env var is documented in the `resolve_cusip_to_ticker` docstring but not read at module top.
3. **Spearman rank is O(n log n) pure-Python.** At 7,262 congress rows this is fine (<100ms). If scaling to 10M+ rows, switch to `scipy.stats.spearmanr`.
4. **Forward-return fetcher uses yfinance.** Phase-5-equivalent BQ-cached price table would be more reliable. Left as a TODO in the function docstring.
5. **No IC significance test.** IC_IR is reported; a formal t-test or bootstrap is out of scope (Grinold & Kahn framework is used for the reported statistic).
6. **Non-dry-run congress IC expectation.** Research brief anchors ~0.02–0.06 IC on Senate-only trades; if the live run returns substantially higher the notes column should flag for review.

## Pre-Q/A self-check

- ast.parse OK.
- Dry-run creates the TSV; `ls alt_data_ic_*.tsv` matches.
- IC math: +1/-1 on perfect series, correct IR computation, tie-aware rank.
- Regression 152/1 unchanged.
- ASCII decode OK.
- `git status --short` shows `backend/alt_data/features.py` + new TSV + handoff trio.
- Handoff phase-scoped.
- Masterplan NOT flipped yet.
