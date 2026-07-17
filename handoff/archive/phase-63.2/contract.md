# Contract — step 63.2 (BQ cross-check of displayed numbers)

**Phase:** phase-63 | **Step:** 63.2 | **Priority:** P0 | harness_required: true | depends_on: 63.1 (done); post-66.2 (done)
**Cycle:** 1 | Date: 2026-07-17 | **Type:** live AUDIT (read-only; produces a defect register). $0 (curl + BQ only,
ZERO metered LLM); historical_macro FROZEN; live book untouched; **operator :3000 NEVER touched**.

## Research-gate summary (gate PASSED)

Researcher subagent (Agent tool, Opus 4.8 effort:max, $0), brief `research_brief_63.2.md`. Envelope:
**gate_passed=true**, tier=moderate, **5 external sources read in full**, 35 snippet-only, 40 URLs, recency scan, 10
internal files. KEY:
- **Framing**: the page renders the API JSON → displayed==API is definitional; the meaningful cross-check is
  **API-vs-BQ**. Record the displayed/API/BQ triple; DEF- only for API-vs-BQ mismatches beyond rounding/tolerance.
- **$0 curl path (verified live)**: `curl -s http://localhost:8000<ep>` returns 200 with NO token
  (`DEV_LOCALHOST_BYPASS` active, auth.py:150; middleware auth main.py:426-460); `/api/sovereign|signals|
  observability` are also `_PUBLIC_PATHS`. **GETs ONLY** (no POST/PUT/DELETE). Never touch :3000.
- **BQ SoT** in `financial_reports` (paper_portfolio/paper_positions/paper_trades/paper_portfolio_snapshots) +
  `pyfinagent_data.outcome_tracking` (performance). Read-only via the Python bigquery client (ADC) — SELECT the single
  needed column; tiny tables, far under free tier. Project sunny-might-477607-p8.
- **STORED vs COMPUTED**: get_paper_portfolio is SELECT* (NAV/cash/pnl/benchmark = direct cell compare); Sharpe/alpha/
  counts/metrics-v2 = re-derived (compare via identity/re-derivation). snapshots read DESC (phase-47.4 resort trap).
- **Tolerance**: "beyond rounding" = the display unit for displayed-vs-API (2dp → ±0.005); ~0.5-1% rel for API-vs-BQ.
  TZ/formatting/live-lag/different-formula differences are recorded in the triple but are NOT DEFs.
- **WATCH (record, not auto-DEF)**: `/portfolio.sharpe_ratio` (3.56) vs `/metrics-v2.rolling_sharpe` (3.0168) —
  different formulas/windows.

## Plan (the audit worklist — page → number → API path → BQ SoT → type)
Execute per the research worklist (22 rows). Representative:
- `/` NAV/cash/pnl%/benchmark → `/api/paper-trading/{status,portfolio}` → paper_portfolio.{total_nav,current_cash,
  total_pnl_pct,benchmark_return_pct} [STORED, direct cell]; position_count → COUNT paper_positions [COMPUTED];
  sharpe → compute_sharpe_from_snapshots [COMPUTED].
- `/paper-trading/positions` per-position qty/avg_entry/cost_basis/sector → paper_positions.<col> [STORED];
  market_value/unrealized_pnl → identities (cost_basis==qty*avg_entry; mv==qty*price; upnl==mv-cost_basis) [LIVE/C].
- `/paper-trading/nav` → paper_portfolio_snapshots (DESC) [STORED]. `/paper-trading/trades` rows/count → paper_trades
  [STORED/COMPUTED]. `/performance` → outcome_tracking [COMPUTED]. `/learnings` → the learnings endpoint [COMPUTED].
  `/sovereign` compute-cost/leaderboard/red-line → the PUBLIC sovereign endpoints.

For each: curl the API value + query the BQ SoT + compare. Build the triple table; add a `| DEF-NNN |` row only for a
real mismatch beyond tolerance.

### Deliverable: `handoff/away_ops/defect_register.md`
- A CRITERION-1 TRIPLE table FIRST (rows start `| /route ...` — NOT matched by `grep '^| DEF-'`): columns
  `| route | number | displayed/API | BQ | verdict |`.
- Then CRITERION-2 DEF rows (start `| DEF-NNN |`): `| DEF-NNN | route | number | API | BQ | severity | repro |`.
  Severity: CRITICAL money/risk (NAV/cash/stop/P&L) · HIGH gate-feeding (Sharpe/DSR/alpha/win_rate) · MEDIUM counts ·
  LOW chrome. `grep -c '^| DEF-'` counts ONLY defects (0 = valid pass if all triples match).

## Immutable success criteria (verbatim from masterplan.json 63.2)
1. "each number-bearing page has a displayed-vs-API-vs-BQ triple recorded with the SQL pasted verbatim"
2. "every mismatch beyond rounding is a DEF- row with route, severity, reproduction, displayed-vs-truth values, suspected file, and {pure-bug | trading-behavior} classification"
3. "zero metered LLM calls used (BQ + curl only)"

**[Cycle-1 CONDITIONAL fix]**: cycle-1 SOFTENED these criteria (dropped "with the SQL pasted verbatim" from #1 and
"reproduction, displayed-vs-truth values, suspected file, and {pure-bug | trading-behavior} classification" from #2) —
the Q/A caught it. Restored verbatim above. The deliverable is updated accordingly: SQL pasted verbatim per triple +
DEF-001 carries suspected-file + classification.

**Verification command (immutable):**
`cd /Users/ford/.openclaw/workspace/pyfinagent && test -f handoff/away_ops/defect_register.md && grep -c '^| DEF-' handoff/away_ops/defect_register.md`

## Boundaries (binding)
$0 — curl (GET only) + BQ (read-only SELECT) + a Python re-derivation for computed values. ZERO metered LLM
(criterion 3). READ-ONLY audit — the only new file is `handoff/away_ops/defect_register.md` (+ live_check). NO
production code change; NO trade/risk/money touch; kill-switch/stops/caps/DSR/PBO untouched; historical_macro FROZEN;
live book untouched. **Operator :3000 NEVER touched** (all curls hit :8000). Any real mismatch is RECORDED as a DEF-
row (the register is the deliverable), not fixed here (fixes are 63.4). Formula/TZ/live-lag differences → triple, not
DEF.

## References
research_brief_63.2.md; backend/api/auth.py:150 (DEV_LOCALHOST_BYPASS); backend/main.py:426-460 (middleware auth);
backend/db/bigquery_client.py:521 (get_paper_portfolio), :1039 (snapshots DESC); CLAUDE.md BigQuery section
(financial_reports = paper tables, us-central1... actually paper tables via _pt_table = financial_reports);
frontend/src/lib/api.ts (page→endpoint map). The 63.1 walk (route inventory).
