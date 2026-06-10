# Contract — Step 55.1

**Step id:** 55.1 — Data-integrity + trading forensics — PRIMARY-data post-mortem of the away week (2026-06-01 → 2026-06-10)
**Date:** 2026-06-10
**Phase:** phase-55 (away-week forensic review; $0; review-only, NO fixes, NO LLM trading-cycle spend)
**Researcher gate:** PASSED — `handoff/current/research_brief.md` (tier=complex, 6 external sources read in full, 30 URLs, recency scan done, 9 internal modules audited with file:line cites; envelope `gate_passed: true`)

## Research-gate summary

Internal audit (brief §A) confirms by code reading: BUY `total_value` stored LOCAL at `paper_trader.py:265`; SELL `total_value`+`transaction_cost` stored LOCAL at `:387,413-414`; mark-to-market converts correctly EXCEPT the FX-unavailable fallback at `:512-520` (frozen-mark / local-as-USD branch — prime NAV-inflation suspect); SELL cash credit IS converted (`:485`); kill switch consumes the same possibly-corrupted `total_nav` (`paper_trader.py:1019`, limits 4%/10% at `settings.py:453-454`); VS-KOSPI card shows holdings return not index excess (`cockpit-helpers.tsx:208-218`, tooltip already discloses); trades table renders local values with `$` (`trades-columns.tsx:106-124`). `tca_report.py` is a SYNTHETIC seeder (does not read paper_trades) — real TCA must be computed from BQ. `/api/paper-trading/reconciliation` already exists and can date corruption onset. Snapshot table resolved = `financial_reports.paper_portfolio_snapshots`. External literature (brief §B): Perold IS decomposition (fee drag + delay only in bq_sim — no market-impact claim), Bailey-LdP MinTRL/PSR/DSR (8 days << MinTRL → report N/A-by-track-record), GIPS error-correction tiers for materiality classification, reconciliation break taxonomy, alpha/beta + HHI attribution with low-power caveats.

## Hypothesis

The away-week trade ledger is currency-corrupted for KR rows (local-as-USD), the on-screen NAV=345,968.86 derives from the `:512-520` FX-fallback or a related un-converted path (NOT from `:265` alone), and the 06-05 kill-switch non-trip is explainable only after recomputing daily loss on a corrected USD NAV — all decidable from primary BQ data + live UI evidence at $0.

## Immutable success criteria (verbatim from .claude/masterplan.json, step 55.1)

1. "the post-mortem (handoff/current/55.1-away-week-postmortem.md) is built from PRIMARY data (BQ financial_reports.paper_trades/paper_positions/paper_portfolio_snapshots + /api/paper-trading/* endpoints), reconciles the Slack-digest NAV path (+21.9% 06-01, +23.4% 06-03, +19.3% 06-05, +19.2% 06-09) to within +/-0.2pp or reports the divergence as its own finding, and quantifies: weekly turnover, per-round-trip realized P&L for MU (06-08 -> 06-09), 000660.KS (06-02 -> 06-05) and DELL (4 trades), a gross-to-net TCA decomposition (implementation shortfall, fees as % of turnover, cumulative fee drag for the week; scripts/risk/tca_report.py), win_rate/profit_factor/expectancy/median_holding_days (via /api/paper-trading/performance), and the VERBATIM /api/paper-trading/metrics-v2 output -- on ~6-8 trading days its MIN_OBS_FOR_PSR=30 guard returns insufficient_data nulls, which is a VALID, honestly-reported result, not a FAIL"

2. "the code-traced FX defects are confirmed or refuted against live BQ rows (paper_trader.py:265 total_value; :386-414 SELL transaction_cost), ALL FOUR FX conversion points (trade recording, mark-to-market, cash ledger, fees) are enumerated with rate-source + rate-as-of-timestamp consistency checked, the corruption scope is classified (stored-data vs display-only; affected row count + date range), a per-snapshot-day cash-ledger reconciliation ties sum(cash movements) to NAV - sum(position market_value), and the NAV/Cash/'$10K fund' three-way discrepancy is traced to a root cause at file:line with the :512-520 FX-fallback suspect explicitly ruled in or out; the VS-KOSPI readout is audited against cockpit-helpers.tsx:197-218 (noting its existing tooltip already discloses the limitation)"

3. "concentration is measured per snapshot day (sector weights + portfolio HHI) and the report cites the config/code path of any existing concentration limit or states NONE EXISTS; the away-week return is attributed regime-vs-skill (decomposed into benchmark beta -- SPY plus a semis proxy such as SOX, and KOSPI for the KR book -- concentration tilt, and residual alpha); the kill-switch audit for 06-05 reads the configured thresholds from live config (file path cited), verifies WHICH P&L/NAV fields the switch consumes (if any are touched by the FX/NAV corruption, the non-trip may be a measurement failure rather than a threshold failure), computes the daily P&L from snapshots, and renders SHOULD-HAVE-TRIPPED (defect traced to file:line) or CORRECTLY-DID-NOT-TRIP (arithmetic shown) -- presuming either verdict in advance is a FAIL"

4. "live UI evidence is captured via Playwright MCP (the /paper-trading page behind the NextAuth wall: Value column, NAV/Cash cards, VS-KOSPI card) and embedded in live_check_55.1.md, folding in the three outstanding phase-50.6 visual confirms (/paper-trading/manage markets toggle; /paper-trading/positions currency-exposure card; /backtest US/USD/SPY strip) in the same session; the step performs NO fix work and NO LLM trading-cycle spend"

**Verification command (immutable):** `cd /Users/ford/.openclaw/workspace/pyfinagent && test -f handoff/current/55.1-away-week-postmortem.md && test -f handoff/current/live_check_55.1.md`

## Plan (ordered per brief §C — cheapest/highest-signal first)

1. Resolve the NAV contradiction: `paper_portfolio` live row (starting_capital, total_nav, current_cash) + `paper_portfolio_snapshots` 06-01..06-10 (ASC sort; DESC trap). Reconcile digest % = cumulative_pnl_pct.
2. Run existing `/api/paper-trading/reconciliation` to corroborate + date corruption onset (backend must be up; else call `compute_reconciliation` directly).
3. Pull away-week `paper_trades`; classify rows by market (column or ticker suffix); re-derive USD = qty·price·fx_asof; confirm/refute :265 + :413-414 against live rows; count affected rows + date range; classify stored-data vs display-only (GIPS materiality tier per brief F5).
4. Per-snapshot-day cash-ledger reconciliation: NAV(D) == cash(D) + Σ market_value_usd; cash(D) == starting + Σ flows + Σ SELL proceeds_usd − Σ BUY cost_usd. First broken day = onset; attribute frozen-mark vs un-converted-credit.
5. `paper_positions` mark check: market_value vs qty·current_price (local-as-USD test) per non-US position; grep logs for the `:515` FX-unavailable WARN; FX backfill coverage query on `historical_fx_rates`.
6. Turnover + round-trips (MU, 000660.KS, DELL) + gross-to-net TCA from BQ (fee drag bps; delay cost; NO market-impact claim — bq_sim fills at close). Run `tca_report.py` only as tool-smoke, LABELED synthetic. Run `paper_execution_parity.py`, report output or failure honestly.
7. `/api/paper-trading/performance` (win_rate/profit_factor/expectancy/median_holding_days) + VERBATIM `/api/paper-trading/metrics-v2` (insufficient_data nulls are VALID).
8. Kill-switch audit: read `handoff/kill_switch_audit.jsonl` 06-01..06-10; thresholds from `settings.py:453-454`; recompute 06-05 daily loss on recorded AND corrected NAV; verdict SHOULD-HAVE-TRIPPED or CORRECTLY-DID-NOT-TRIP with arithmetic — no presumption.
9. Concentration per day (position-HHI + sector-HHI); cite concentration-limit code path or state NONE EXISTS; regime-vs-skill regression (SPY + SOXX + KOSPI blend) with 8-point low-power caveat; MinTRL stated explicitly.
10. Playwright MCP live UI: /paper-trading NAV/Cash cards, KR-filtered trades Value column, VS-KOSPI card + the three phase-50.6 visual confirms (/paper-trading/manage markets toggle; /paper-trading/positions currency-exposure card; /backtest US/USD/SPY strip). Captures → live_check_55.1.md.
11. Write `55.1-away-week-postmortem.md` (formal reconciliation shape: break list w/ taxonomy, findings tagged for 55.3 ID assignment) + `live_check_55.1.md` (UI captures + BQ row evidence + queries).

## Constraints

- $0: bounded BQ reads only (date filters + LIMIT, 30s timeout); no LLM trading-cycle spend; review-only — NO fixes, no code changes outside handoff/.
- Primary data over cached endpoints (30s api_cache) — BQ direct for numbers; endpoints to confirm UI binding only.
- Sharpe/DSR/PSR labeled "insufficient track record" per MinTRL; no performance claims on 8 days.
- Verdict-neutral kill-switch audit; verdict-neutral VS-KOSPI audit (tooltip already discloses).

## References

- handoff/current/research_brief.md (researcher, 2026-06-10, gate_passed: true)
- handoff/current/goal_post_away_review.md (goal prompt; review tooling table)
- Brief sources F1-F6: Perold IS (ryanoconnellfinance.com), PSR/MinTRL (portfoliooptimizer.io), DSR (Wikipedia/Bailey-LdP SSRN 2460551), reconciliation lifecycle (solvexia.com), GIPS error correction (performancemeasurementsolutions.com), IS worked example (analystprep.com)
- Code anchors: paper_trader.py:265,387,413-414,485,512-520,556,1011-1058; kill_switch.py:230-264; settings.py:453-454; fx_rates.py:153-181; bigquery_client.py:512-513,1008,1037; cockpit-helpers.tsx:197-218; trades-columns.tsx:10-124; tca_report.py:49,110-124; reconciliation.py
