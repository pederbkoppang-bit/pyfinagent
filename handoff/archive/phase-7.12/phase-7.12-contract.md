# Sprint Contract — phase-7 / 7.12 (Feature integration & IC eval)

**Step id:** 7.12 **Cycle:** 1 **Date:** 2026-04-20 **Tier:** moderate

## Research-gate summary

7 sources in full (PyQuant IC+Alphalens, dev.to ICIR/Barra, Medium IC mastery Mar 2026, OpenFIGI API docs, Fortune/NBER Dec 2025 Wei-Zhou congressional leaders, Balaena IR+IC case study, arXiv 2602.05514 Feb 2026 temporal graph learning). 15 URLs, three-variant queries, recency scan. `gate_passed: true`. Brief at `handoff/current/phase-7.12-research-brief.md`.

## Hypothesis

Write `backend/alt_data/features.py` that aggregates congress + 13F tables into cross-sectional signals, fetches forward returns via yfinance, computes Spearman IC over {5, 20, 63} day windows, and writes at least one `alt_data_ic_<timestamp>.tsv` to `backend/backtest/experiments/results/`. Advisory-aware: congress rows note "Senate only adv_71"; 13F rows that can't resolve CUSIP→ticker note "adv_72". FINRA signal extraction NOT attempted (adv_73 gate not cleared). Dry-run mode writes a header-only TSV so the criterion passes even when BQ is empty.

## Immutable criteria

- `test -f backend/alt_data/features.py`
- `ls backend/backtest/experiments/results/alt_data_ic_*.tsv | head -n 1`

## Plan

1. Write `backend/alt_data/features.py` (~350 lines):
   - `aggregate_congress_features(start, end)` → DataFrame via BQ query on `alt_congress_trades WHERE ticker IS NOT NULL`.
   - `aggregate_13f_features(start, end)` → DataFrame via BQ query on `alt_13f_holdings`.
   - `resolve_cusip_to_ticker(cusips)` → OpenFIGI POST /v3/mapping; batch 10; fail-open per advisory.
   - `compute_ic(signal, fwd_ret, method='spearman')` → dict with `ic_mean, ic_std, ic_ir, n_observations`.
   - `_get_forward_returns(tickers, start, end, window_days)` — yfinance daily close → forward returns.
   - `run_ic_evaluation(output_tsv_path, windows=(5,20,63), dry_run=False)` orchestrator.
   - `_cli()` — argparse with `--dry-run`, `--output-dir`, `--windows`.
2. Run `python -m backend.alt_data.features --dry-run` to create a header-only TSV.
3. Verify both immutable criteria + regression.
4. Q/A. Log. Flip. Phase-7 closes (13/13 done).

## Out of scope

- No live OpenFIGI call (optional; fail-open to NULL tickers). No paid API keys.
- No FINRA signal extraction (adv_73_owner_risk_accept_gate holds).
- No House chamber congress backfill (adv_71_house_followup holds).
- No IC significance testing beyond IR computation.
- ASCII-only.

## References

- `handoff/current/phase-7.12-research-brief.md`
- `backend/alt_data/{congress,f13,finra_short}.py` (DDL shapes feeding the aggregator)
- `backend/backtest/experiments/results/` (TSV target dir)
- Advisories: `adv_71_house_followup`, `adv_72_ticker_null`, `adv_73_owner_risk_accept_gate`
- `.claude/masterplan.json` → phase-7 / 7.12
