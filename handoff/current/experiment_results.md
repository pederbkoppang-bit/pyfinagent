# Experiment Results — Step 55.1 (GENERATE)

**Step:** 55.1 — Data-integrity + trading forensics (away week 2026-06-01 → 2026-06-10). **Date:** 2026-06-10. **Mode:** review-only, $0, NO fixes.

## What was built

Two deliverables (no production code touched):

| Artifact | Content |
|---|---|
| `handoff/current/55.1-away-week-postmortem.md` | Full primary-data post-mortem: NAV-path reconciliation (≤0.05pp), 4-point FX-conversion audit with live-row confirmation, corruption-scope classification (7/52 rows, stored vs display), penny-exact daily cash-ledger reconciliation, NAV-discrepancy root cause at `frontend/src/lib/useLiveNav.ts:34-39` (with `paper_trader.py:512-520` suspect RULED OUT), VS-KOSPI audit, turnover/round-trip/TCA, verbatim /performance + /metrics-v2, concentration (HHI) + regime-vs-skill attribution + MinTRL, kill-switch verdict CORRECTLY-DID-NOT-TRIP with arithmetic, 15-break reconciliation table (B1-B15) for 55.3 ID assignment |
| `handoff/current/live_check_55.1.md` | 6 Playwright captures (incl. all three phase-50.6 visual confirms), BQ queries + row excerpts, verbatim endpoint outputs, tool-run outputs (tca_report synthetic-labeled; paper_execution_parity honest FAIL), method + constraint-compliance disclosure |

Supporting evidence files: `handoff/current/captures_55.1/*.png` (6 screenshots), `/tmp/55_1/*.json` (raw query dumps, session-local).

## Key findings (headline)

1. **Stored NAV/cash/snapshots are CLEAN** — digest path reconciles to ≤0.05pp; daily cash-ledger identity closes to $0.01. The engine converts FX correctly internally.
2. **Trade-ledger FX defects CONFIRMED on live rows**: `paper_trades.total_value` in KRW on 7 rows (`paper_trader.py:265`), SELL `transaction_cost` in KRW on 3 rows (`:387,:413-414`). That is the complete corruption scope (13.5% of all rows, away week only).
3. **UI NAV 345,968.86 root cause = `useLiveNav.ts:34-39`** (client sums KRW live ticks as USD; display-only). Live-reproduced at 345,950.68 with exact decomposition. Same defect class in RiskMonitorCard ("Max position 1527.8%"), positions Current cell, currency-exposure %, donut center.
4. **Kill switch CORRECTLY-DID-NOT-TRIP on 06-05**: true day −2.82% < 4% daily limit; trailing 3.26% < 10%. The "−3.5%" digest figure was overstated. Structural note: the daily-loss leg is ≈0 by construction under once-daily cadence (SOD anchored at evaluation instant).
5. **Away week was net negative**: −2.26% from 06-01, 81.4% weekly turnover, 10 within-week round trips netting −$132; the +19-23% level is the April momentum book converting unrealized→realized.
6. **Concentration**: 100% Technology book all week, HHI up to 0.63; caps exist (2/sector + 30% NAV, portfolio_manager.py:223-310) but NAV%-cap is structurally diluted by 70-96% cash.
7. **MinTRL ≈ 377 daily obs vs 7 available** — all Sharpe/PSR readouts labeled insufficient-track-record; metrics-v2 DSR=0.0 is the honest summary.
8. New live-discovered breaks: profit_factor metric defect (B9), markets-toggle display/config mismatch (B14), historical_fx_rates malformed rows (B12), parity-probe non-idempotency (B13).

## Verification command output (verbatim)

```
$ cd /Users/ford/.openclaw/workspace/pyfinagent && test -f handoff/current/55.1-away-week-postmortem.md && test -f handoff/current/live_check_55.1.md && echo PASS
PASS
```

## Artifact shape

- Post-mortem: 10 sections, break list B1-B15 with taxonomy + proposed severity, every numeric claim carrying a BQ query, file:line cite, or endpoint excerpt.
- live_check: capture table + query/excerpt evidence + verbatim tool outputs + method disclosure (skip-auth :3100 instance, operator :3000 untouched).

## Honest limitations

- n=7 daily returns: attribution regression and MinTRL are reported with explicit low-power caveats; no performance claims made.
- The 2-per-sector count-cap adjudication (why 6-7 Technology positions co-existed) requires decision-time sector labels → explicitly deferred to 55.2.
- tca_report.py output is synthetic by design (script seeds deterministic fills; does not read paper_trades) — labeled as tool-smoke, real TCA computed from BQ.
- paper_execution_parity.py failed on an Alpaca client_order_id uniqueness error — reported verbatim, not retried-to-green.
