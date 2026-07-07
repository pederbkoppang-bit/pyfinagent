# live_check 66.2 -- Redeploy capital via the NORMAL path (prep phase, 2026-07-07)

Required shape: "live_check_66.2.md with the BUY BQ row or the 5-day funnel
diagnosis, plus both integrity-check verdicts."

## 1. Criterion 1 -- OPEN (clock starts when 66.1 closes)

First post-deploy scheduled cycle: 2026-07-07 18:00 UTC (backend restarted
16:31:55 UTC holding all phase-66 fixes). Evidence lands here when it exists:
either the first ordinary-pipeline BUY row, or per-cycle funnel reports
accumulating toward the >=5-healthy-rail-day diagnosis.

Funnel tooling ACCEPTANCE (scripts/diagnostics/funnel_report.py, read-only, over
the collapse window):

```
| day        | cycle    | rail ok/fail | analyses (deg) | rec mix            | non-HOLD | trades (by reason)                              | verdict |
| 2026-06-09 | 0361d1ea | 0/0          | 5 (0)  | BUY:4 HOLD:1       | 4 | BUY:swap_buy=2 SELL:swap_for_higher_conviction=2 | GATES EVALUATED, trades executed |
| 2026-06-10 | d2a9e92b | 0/0          | 5 (0)  | BUY:4 HOLD:1       | 4 | BUY:new_buy_signal=1 BUY:swap_buy=1 SELL:stop_loss_trigger=1 SELL:swap_for_higher_conviction=1 | GATES EVALUATED, trades executed |
| 2026-06-11 | 78d253f5 | 3/2          | 8 (7)  | BUY:1 HOLD:5 N/A:2 | 3 | - | GATES EVALUATED, zero BUYs (check decide_trades per-gate logs) |
| 2026-06-12 | 5f15fdbe | 36/45        | 5 (3)  | BUY:2 HOLD:3       | 2 | - | GATES EVALUATED, zero BUYs (check decide_trades per-gate logs) |
```

The tool exposes exactly the criterion-b structure AND the known gap (universe/
screener/decide_trades counters are log-only -- reported explicitly, never
silently omitted). Refinement over the brief: on 06-11/06-12 a few BUY recs
SURVIVED the degraded scorer and were rejected inside decide_trades -- the
log-only stage; post-66.1 cycles will carry rail_skipped/breaker_tripped columns.

## 2. Criterion 3 -- CLOSED: Alpaca short exposure EXPLAINED with evidence

Read-only inspection (alpaca-py, TradingClient paper=True; NO orders placed),
account PA3VQZZLAKE2 ACTIVE, 2026-07-07 ~16:20 UTC:

```
equity=99273.30 cash=102778.62 long_mv=10909.12 short_mv=-14414.44
positions: 20 (10 long, 10 short); sum(short mv) = -14414.46 == short_market_value
shorts: ADBE AMD AVGO CSCO GOOGL IBM META MSFT PYPL UBER (qty -4..-5 each)
closed orders 2026-06-10 13:51-13:52Z: alternating 1-share buy/sell pairs with
client_order_id prefix d4-<SYM>-<n> (drill batch) + probe-alp-1
```

ROOT CAUSE: pre-departure MCP-validation drill orders (2026-06-10) -- 1-share
SELLs on symbols the account did not hold, which a margin-default Alpaca paper
account fills as SHORT OPENS. The autonomous loop CANNOT have caused it: its
launchd env has no EXECUTION_BACKEND/ALPACA keys -> ExecutionRouter defaults
bq_sim (execution_router.py:65-71; pre-prod audit 2026-05-16:161 confirms); BQ is
the authoritative ledger and the Alpaca account is a disconnected mirror.
(The -13,842.89 in the Cycle-58 finding vs -14,414.44 today = mark-to-market
drift on the same 10 shorts.) HYGIENE (operator, optional): reset the paper
account in the Alpaca dashboard; no BQ correction needed; filed for 63.3.

## 3. Criterion 4 -- CLOSED: single portfolio row is BY DESIGN; KR conversion intact

Design citation: paper_portfolio is one aggregate USD row keyed
portfolio_id='default' (bigquery_client.py:521-534, upsert :550-571); the 50.2
multi-market design puts `market`/`base_currency` on POSITIONS and converts FX
at trade/mark time (paper_trader.py:312-313/:333-334; _fx_local_to_usd :515-523;
sell credit :416). EU/KR exposure surfaces in paper_positions.market +
paper_trades, not extra portfolio rows.

USD-magnitude check on ALL 10 KR trade rows (BQ verbatim): prices are local KRW
by design (e.g. 000660.KS SELL @ 2,425,000) while total_value is correctly
USD-converted (that row: $560.80, NOT 857,843) -- the 56.1 conversion has NOT
regressed. paper_positions GROUP BY market/currency: empty (100% cash),
consistent. VERDICT: intended design; no defect. (Cosmetic: execute_sell log
line prints KRW with a '$' -- register note.)

## 4. Bonus fix shipped during prep (alert integrity)

Phantom "-61.51% drawdown" P1 (fired 2026-07-06 20:05Z on a book UP 20%):
DESC-order trap in compute_drawdown_from_snapshots (navs[-1]=OLDEST row as
"current"). Fixed (date-key ordering; refuses to guess when unknowable),
5 tests, live verification: real drawdown -2.76% (below the -3% tier,
correctly silent). Deployed in the 16:31 UTC restart.
