# live_check — step 63.2 (BQ cross-check of displayed numbers)

## Method (criterion 3: $0, curl + BQ only, ZERO metered LLM)

- API leg: `curl -s http://localhost:8000<ep>` (GET only, no token — `DEV_LOCALHOST_BYPASS` active; operator :3000
  NEVER touched, verified 302 after the audit).
- BQ leg: `bigquery.Client(project='sunny-might-477607-p8').query(..., location='us-central1')` (read-only, ADC).

## ≥3 side-by-side triples (verbatim API + BQ)

**Triple 1 — `/` NAV (STORED):**
```
API : curl :8000/api/paper-trading/portfolio -> portfolio.total_nav = 23874.56
BQ  : SELECT total_nav FROM financial_reports.paper_portfolio ORDER BY updated_at DESC LIMIT 1 -> 23874.56
verdict: MATCH
```

**Triple 2 — `/` cash + total P&L% + benchmark (STORED):**
```
API : portfolio.current_cash = 23214.43 | total_pnl_pct = 19.37 | benchmark_return_pct = 5.18
BQ  : paper_portfolio.current_cash = 23214.43 | total_pnl_pct = 19.37 | benchmark_return_pct = 5.18
verdict: MATCH (all three)
```

**Triple 3 — `/paper-trading/positions` AMD (STORED + identities):**
```
API : positions[0] = {ticker: AMD, quantity: 1.319955, avg_entry_price: 545.4199829,
                      cost_basis: 719.93, market_value: 660.13, unrealized_pnl: -59.80, sector: Technology}
BQ  : paper_positions -> {ticker: AMD, quantity: 1.319955, avg_entry_price: 545.4199829,
                          cost_basis: 719.93, sector: Technology}   (1 row)
verdict: MATCH (stored) + IDENTITIES HOLD (cost_basis == 1.319955*545.4199829 == 719.93;
         unrealized_pnl == market_value - cost_basis == 660.13 - 719.93 == -59.80)
```

**Triple 4 — counts:**
```
API : /status.position_count = 1 | /trades trade count = 61
BQ  : COUNT(paper_positions) = 1 | COUNT(paper_trades) = 61
verdict: MATCH
```

## The one defect (DEF-001, MEDIUM)

```
API : curl :8000/api/reports/performance -> total_recommendations=0, wins=0, losses=0, win_rate=0.0,
      benchmark_beat_rate=0.0
BQ  : SELECT COUNT(*) FROM pyfinagent_data.outcome_tracking -> 404 "Table not found in location US"
verdict: DEF-001 (MEDIUM) -- /performance + /learnings render all-0 because their source table is ABSENT.
         NOT a value mismatch (0 == no data); a data-source-availability defect. Root cause upstream
         (learn-loop writer paper_learn_loop_enabled=False, phase-35.1). Cross-ref 61.4; fix in 63.4.
```

## Immutable command output

```
$ test -f handoff/away_ops/defect_register.md && grep -c '^| DEF-' handoff/away_ops/defect_register.md
1
# exit 0  (1 DEF row = DEF-001; 24 criterion-1 triples recorded)
```

## Result
Every STORED money/position number matches API-vs-BQ EXACTLY and every computed identity holds -- the
operator-reported "dashboard numbers wrong" is NOT reproduced (the core numbers are correct as of 2026-07-17). The
sole finding is DEF-001 (a reporting-source-availability gap, cross-ref 61.4/35.1). $0; :3000 untouched; no production
code changed.
