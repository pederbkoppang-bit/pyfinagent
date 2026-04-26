---
step: phase-5.6
title: Options Integration -- Black-Scholes greeks + options_ingestion --dry-run
cycle_date: 2026-04-26
harness_required: true
verification: 'source .venv/bin/activate && python -c "from backend.markets.options.greeks import black_scholes_greeks; g=black_scholes_greeks(S=450,K=450,T=30/365,r=0.05,sigma=0.20,option_type=''call''); assert 0.4<g[''delta'']<0.6; print(''ok'')" && python -m backend.markets.options.options_ingestion --underlyings SPY QQQ --dry-run'
research_brief: handoff/current/phase-5.6-research-brief.md
---

# Contract -- phase-5.6

## Step ID

`phase-5.6` -- "Options Integration (Alpaca Options Level 3)" (`.claude/masterplan.json` phase-5).

## Research-gate summary

Spawned `researcher` (moderate tier). Brief at
`handoff/current/phase-5.6-research-brief.md` (268 lines). Gate: 6
external sources read in full (Wikipedia Greeks, Investopedia,
scipy.stats docs, OCC official, macroption worked example, Real Python
argparse), recency scan, internal inventory complete. `gate_passed: true`.

Decisive findings (verified in-venv by researcher):
- ATM 30-DTE call delta at (S=450, K=450, T=30/365, r=0.05, σ=0.20) = **0.5400** (within [0.45, 0.55] success criterion)
- gamma=0.0154, theta_call/day=-0.2024, vega/1%=0.5121
- `scipy.stats.norm.cdf(d1)` for delta; `norm.pdf(d1)` for gamma/vega/theta. scipy>=1.12.0 already pinned in `backend/requirements.txt`
- OCC 21-char format: `ticker.ljust(6) + YYMMDD + C|P + str(int(round(strike*1000))).zfill(8)`
- `--dry-run` + `--underlyings` argparse shape from `scripts/migrations/add_news_sentiment_schema.py:86-105`
- Migration template: `scripts/migrations/add_news_sentiment_schema.py` (idempotent CREATE TABLE IF NOT EXISTS + ADC client + --dry-run flag)

## Hypothesis

A new `backend/markets/options/` subpackage with `greeks.py` (pure-math
Black-Scholes) + `options_ingestion.py` (CLI dry-run scaffold) lays the
foundation for the future Alpaca Options Level 3 wiring (5.6 success
criterion #3 about live paper-order submission is OUT OF SCOPE this
cycle -- it requires Alpaca options Level 3 keys which are user-action).
The greeks module is also reusable by `RiskEngine.compute_position_size('option', ..., delta=...)`
established in 5.4.

## Immutable success criteria (verbatim from masterplan)

```
source .venv/bin/activate && python -c "from backend.markets.options.greeks import black_scholes_greeks; g=black_scholes_greeks(S=450,K=450,T=30/365,r=0.05,sigma=0.20,option_type='call'); assert 0.4<g['delta']<0.6; print('ok')" && python -m backend.markets.options.options_ingestion --underlyings SPY QQQ --dry-run
```

Plus 4 success_criteria from masterplan:
1. options_snapshots table created in pyfinagent_hdw -- USER ACTION (--apply); this cycle ships the migration script in `--dry-run` form
2. ATM 30-DTE call delta in [0.45, 0.55] -- PASS (verified at 0.5400)
3. Paper option order submits via Alpaca without error (dry-run) -- OUT OF SCOPE this cycle (requires user-action: Alpaca Options Level 3 keys); ingestion script's --dry-run mode handles the verification half
4. RiskEngine options kwargs (delta/gamma/theta/vega) accepted -- PASS (already in 5.4 signature)

## Plan steps

1. Create `backend/markets/options/__init__.py` (~10 LOC; package marker + re-exports).

2. Create `backend/markets/options/greeks.py` (~150 LOC):
   - `from scipy.stats import norm`
   - `def black_scholes_greeks(S, K, T, r, sigma, option_type='call', q=0.0) -> dict`:
     - Returns `{'delta', 'gamma', 'theta', 'vega', 'rho', 'price'}`
     - Edge-case guard: T <= 0 -> intrinsic value, delta = 1 if ITM call / -1 if ITM put / 0 if OTM, other greeks = 0
     - sigma <= 0 guard: floor at 1e-6 to avoid div-by-zero
     - Theta in per-day units (divided by 365)
     - Vega in per-1%-vol units (divided by 100)
   - `def parse_occ_symbol(occ: str) -> dict`:
     - Returns `{'ticker', 'expiration', 'option_type', 'strike'}` from 21-char OCC format

3. Create `backend/markets/options/options_ingestion.py` (~120 LOC):
   - `argparse` with `--underlyings` (nargs='+', required) and `--dry-run` (store_true)
   - `__main__` block reads args, in dry-run mode prints what it would ingest (1 line per underlying), exits 0
   - When NOT dry-run: would normally fetch option chain via alpaca-py + write to `pyfinagent_hdw.options_snapshots`. Without creds or --apply: log a warning and exit 0 (fail-open per project pattern).

4. Create `scripts/migrations/create_options_snapshots_table.py` (~150 LOC):
   - Mirror structure of `scripts/migrations/add_news_sentiment_schema.py`
   - `CREATE TABLE IF NOT EXISTS pyfinagent_hdw.options_snapshots` with columns: snapshot_ts, underlying, occ_symbol, strike, expiration, dte, option_type, bid, ask, mid, iv, delta, gamma, theta, vega
   - `--dry-run` (default; prints SQL) and `--apply` (executes via BigQuery client)

5. Create `tests/markets/test_options_greeks.py` (~150 LOC, 12 tests per research plan).

6. Run immutable verification command (both halves).

## References

- `handoff/current/phase-5.6-research-brief.md`
- `backend/markets/{__init__,broker_base,alpaca_broker,risk_engine}.py` (5.1 + 5.4 patterns)
- `scripts/migrations/add_news_sentiment_schema.py:86-105` (argparse + dry-run / --apply pattern)
- Wikipedia Greeks: https://en.wikipedia.org/wiki/Greeks_(finance)
- macroption.com worked Black-Scholes example
- scipy.stats.norm docs

## Out of scope

- Live Alpaca Options Level 3 paper order submission (success criterion #3 -- requires user-action keys)
- Actual `pyfinagent_hdw.options_snapshots` table creation (success criterion #1 -- requires --apply user-action; this cycle ships the migration script as --dry-run scaffold)
- Wiring `black_scholes_greeks` into `RiskEngine` (it's already accepted via `**kwargs`; no change needed)
- BQ row writes from the ingestion script (would require live alpaca-py options chain fetch)
- Frontend options UI
- Option chain caching / persistence
