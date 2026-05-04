# phase-23.2.0 — 1-week post-deployment audit (internal codebase inventory)

**Audit window:** 2026-04-26 → 2026-05-04 (~9 days)
**Cycles shipped:** phase-23.1.1 through phase-23.1.22 (21 distinct steps; 20+21 consolidated into 22)
**Author:** Main (researcher fork terminated without writing; Main has full context from driving the cycles)

---

## Section A — phase-23.1.x cascade verification map

For each cycle: what was supposed to happen + the concrete RIGHT-NOW verification.

| ID | Title | Was-supposed-to-do | Verification recipe |
|----|-------|--------------------|---------------------|
| 23.1.1 | Macro regime filter | Daily FRED-pull + Claude regime tag → conviction multiplier | `bq SELECT * FROM analysis_results WHERE agent_id='macro_regime' ORDER BY analysis_date DESC LIMIT 7` — expect 7 rows over 7 days |
| 23.1.2 | Earnings PEAD overlay | SEC 8-K + sentiment-surprise score | `grep "PEAD signals fetched" backend.log \| tail -10` — expect non-zero counts |
| 23.1.3 | News-driven idea generator | Multi-source RSS + Claude event classifier | `grep "news_signals" backend.log \| tail -10` |
| 23.1.4 | Sector event calendars | FDA/EIA/SEMI feeds | `grep "sector_events" backend.log \| tail -10` |
| 23.1.5 | LLM-as-judge meta-scorer | Top-30 → conviction 1-10 | `grep "Meta-scorer ranked" backend.log \| tail -10` |
| 23.1.6 | Settings + UI surface | Toggles for each signal source | Manual — visit /settings, verify toggles render |
| 23.1.7 | Per-trade signal attribution | rationale drawer | Manual — click a trade row, drawer renders |
| 23.1.8 | Live position MV+P&L derivation | useLivePrices on /paper-trading | Manual — Positions tab values move on every 30s tick |
| 23.1.9 | Deposit endpoint anchors P&L | deposit increments cash + starting_capital | `bq SELECT starting_capital FROM paper_portfolio` — expect 15000 (post-deposit) |
| 23.1.10 | Company name + sector on Positions/Trades | ticker-meta endpoint with BQ-first / yfinance fallback | `curl /api/paper-trading/ticker-meta?tickers=AAPL` returns `{meta:{AAPL:{...}}}` |
| 23.1.11 | Lite-Claude analysis_results persist | _persist_lite_analysis writes row | `bq SELECT COUNT(*) FROM analysis_results WHERE _path='lite' AND DATE(analysis_date) >= '2026-04-26'` |
| 23.1.12 | Removed forced lite_mode override | operator's lite_mode honored | `grep "lite_mode" backend.log \| head -5` — confirm operator value not overridden |
| 23.1.13 | Sector concentration cap v1 | paper_max_per_sector=2 default | `bq SELECT sector, COUNT(*) FROM paper_positions GROUP BY sector` — should never show >2 per sector if cap held |
| 23.1.14 | Legacy-position sector lookup + live NAV scoreboards | autonomous_loop enriches sector via _fetch_ticker_meta | `grep "Enriched .* legacy positions with sector" backend.log` |
| 23.1.15 | Trade idempotency + MERGE + cleanup | 30-min lookback in execute_buy + MERGE on ticker | `bq SELECT ticker, COUNT(*) FROM paper_trades GROUP BY ticker HAVING COUNT(*) > 1` — empty for non-resold tickers |
| 23.1.16 | ticker-meta latency fix | ThreadPoolExecutor max_workers=5 + per-ticker cache + prewarm | `grep "Prewarming ticker-meta cache" backend.log` |
| 23.1.17 | Home/paper SSOT useLiveNav | shared hook | Manual — home cockpit NAV == paper-trading NAV |
| 23.1.18 | Red Line MERGE + dedup | snapshot rows unique per date | `bq SELECT snapshot_date, COUNT(*) FROM paper_portfolio_snapshots GROUP BY snapshot_date HAVING COUNT(*) > 1` — empty |
| 23.1.19 | FD-exhaustion fix (sqlite3 closing) | 23 sites wrapped + RLIMIT log | `lsof -p <backend_pid> \| grep -c tickets.db` — should be 0-3 |
| 23.1.20 | pause/resume timeout hardening | asyncio.timeout(5) + 503 + Retry-After | `time curl -X POST .../api/paper-trading/resume` — under 6s |
| 23.1.21 | silent-hang investigation | daemon-thread + faulthandler + watchdog + Interactive | `kill -USR1 <pid>` writes thread dump |
| 23.1.22 | kill_switch reentrant-lock deadlock | `_snapshot_locked` helper | Repeated pause/resume cycle completes in <2s each |

---

## Section B — Guardrail inventory

| Guardrail | Code site | Verify works |
|-----------|-----------|--------------|
| Kill switch | `backend/services/kill_switch.py` | `bq SELECT * FROM ...` audit log + tail `handoff/kill_switch_audit.jsonl` |
| Sector cap | `backend/services/portfolio_manager.py:193+` | Look for "at cap" log lines + sector_counts in autonomous_loop summaries |
| Trade idempotency | `backend/services/paper_trader.py:101+` | Look for "Idempotency guard: skipping duplicate BUY" logs (should be rare/never) |
| paper_positions MERGE | `backend/db/bigquery_client.py:549+` | `bq SELECT ticker, COUNT(*) FROM paper_positions GROUP BY ticker HAVING COUNT(*) > 1` — empty |
| paper_portfolio_snapshots MERGE | `backend/db/bigquery_client.py:669+` | `bq SELECT snapshot_date, COUNT(*) FROM paper_portfolio_snapshots GROUP BY snapshot_date HAVING COUNT(*) > 1` — empty |
| BQ result(timeout=30) | `backend/db/bigquery_client.py:489` | grep "BQ" errors in log — none should be timeout-related |
| asyncio.timeout(5) | `backend/api/paper_trading.py` | `grep "BQ.*timed out after 5s" backend.log` — should be rare/never |
| FD leak fix | 7 files with `closing(sqlite3.connect(...))` | `lsof` count of tickets.db FDs over time |
| Daemon-thread for _spawn_real_agent | `backend/services/ticket_queue_processor.py:230+` | Look for stuck-thread warnings |
| faulthandler SIGUSR1 | `backend/main.py:122+` | `grep "faulthandler registered on SIGUSR1" backend.log` (every boot) |
| External watchdog | `scripts/launchd/backend_watchdog.{sh,plist}` | `tail handoff/logs/backend-watchdog.log` — look for "health FAIL" entries |
| ProcessType=Interactive | `~/Library/LaunchAgents/com.pyfinagent.backend.plist` | `defaults read ~/Library/LaunchAgents/com.pyfinagent.backend.plist ProcessType` |
| RLIMIT_NOFILE log | `backend/main.py:127+` | `grep "RLIMIT_NOFILE: soft" backend.log` (every boot) |
| Governance limits-loader | `backend/governance/limits_loader.py` | `grep "governance: immutable limits loaded" backend.log` (every boot) |
| mark_to_market reconciliation | `backend/services/paper_trader.py:347+` | `bq SELECT current_cash + sum(market_value) - total_nav FROM ...` should equal 0 within fees |

---

## Section C — Autonomous loop daily-run verification

`backend/services/autonomous_loop.py::run_daily_cycle` — 8-step cycle:
1. Screen universe
2. Filter candidates
3. Analyze candidates (lite OR full)
4. Re-evaluate holdings
5. Mark to market
6. Decide trades (sell-first, then buy)
7. Execute trades
8. Save daily snapshot + learn

**Did each of the last 7 days run a cycle?**
```sql
SELECT
  DATE(snapshot_date) AS d,
  COUNT(*) AS cycles_recorded
FROM `sunny-might-477607-p8.financial_reports.paper_portfolio_snapshots`
WHERE PARSE_DATE('%Y-%m-%d', snapshot_date) >= DATE_SUB(CURRENT_DATE(), INTERVAL 9 DAY)
GROUP BY d
ORDER BY d
```
Expect: ~9 rows (one per day), no gaps, all post-23.1.18 single-row-per-day.

---

## Section D — BQ table freshness

| Table | Expected freshness | SQL |
|-------|-------------------|-----|
| paper_portfolio | <24h since last MtM | `SELECT updated_at FROM paper_portfolio WHERE portfolio_id='default'` |
| paper_positions | <24h | `SELECT MAX(last_analysis_date) FROM paper_positions` |
| paper_trades | latest BUY/SELL timestamps | `SELECT MAX(created_at) FROM paper_trades` |
| paper_portfolio_snapshots | row for today | `SELECT MAX(snapshot_date) FROM paper_portfolio_snapshots` |
| analysis_results | <24h on lite-path rows | `SELECT MAX(analysis_date) FROM analysis_results WHERE _path='lite'` |
| outcome_tracking | nightly job rolling | `SELECT MAX(updated_at) FROM outcome_tracking` |
| harness_learning_log | per harness cycle | `SELECT MAX(timestamp) FROM harness_learning_log` |

---

## Section E — Endpoint pulse

Critical endpoints to spot-check:
- `GET /api/health` → 200 in <100ms
- `GET /api/paper-trading/status` → 200 in <500ms
- `GET /api/paper-trading/portfolio` → 200 in <1s
- `GET /api/paper-trading/kill-switch` → 200 in <500ms (post-23.1.20: even with hung BQ degrades to 200 with null breach)
- `POST /api/paper-trading/pause {confirmation:"PAUSE"}` → 200 in <500ms
- `POST /api/paper-trading/resume {confirmation:"RESUME"}` → 200 or 503+Retry-After in <6s (post-23.1.20)
- `GET /api/paper-trading/ticker-meta?tickers=...` → 200 in <100ms (cache hit) or <3s (cold)
- `GET /api/sovereign/red-line?window=30d` → 200 in <500ms (post-23.1.18 dedup)
- `GET /api/paper-trading/run-now` (manual cycle) — only triggered manually

---

## Section F — Frontend integrity checks

- Home `/` MAS Operator Cockpit NAV equals `/paper-trading` NAV (phase-23.1.17 useLiveNav)
- Red Line Monitor terminal point matches NAV scoreboard (phase-23.1.18)
- COMPANY + SECTOR columns populate within ~5s on tab load (phase-23.1.16 prewarm)
- Risk Monitor shows "HIGH (N/M Technology)" sector concentration when applicable (phase-23.1.13)
- Pause/Resume buttons respond in <2s (phase-23.1.22)
- "Trades" tab does NOT show duplicate rows (phase-23.1.15 cleanup + idempotency)

---

## Section G — Watchdog log review

`handoff/logs/backend-watchdog.log` should show `health FAIL` entries ONLY if backend was hung. The pre-23.1.21 hangs would have shown N×3-fail kicks. Post-23.1.21 + 23.1.22, expect zero.

---

## Section H — Phase 2 deferred items

Items the cycles explicitly deferred:

1. **23.1.13** — HRP, sector-neutral re-rank, correlation dedup, forced rebalance, min-sectors, strict 25%-NAV cap, BQ sector column on paper_positions
2. **23.1.14** — schema migration to add `sector` column to paper_positions (currently bridged by runtime enrichment)
3. **23.1.15** — collapse delete+insert in execute_buy / mark_to_market / execute_sell-partial to single MERGE; deterministic client_order_id; nightly drift-audit job
4. **23.1.16** — dedicated ticker_meta BQ table for cross-restart durability; frontend per-ticker progressive rendering; SWR refresh on cache hits >12h
5. **23.1.17** — backend auto-MtM wrapper after raw cash mutations; home Sharpe live derivation; status-endpoint server-side live NAV
6. **23.1.18** — `created_at` column on paper_portfolio_snapshots; MERGE for other paper_* tables; % return toggle on chart
7. **23.1.19** — TicketsDB thread-local single connection refactor; broader leak audit (httpx, aiohttp); periodic FD-count metric
8. **23.1.22** — audit ALL `with self._lock:` blocks for re-entrant patterns; switch KillSwitchState._lock to RLock as defensive default

---

## Recommended masterplan steps (10 P0/P1, 6 P2)

```json
[
  {"id":"23.2.1","name":"Verify autonomous loop ran daily for 7+ days (paper_portfolio_snapshots gap check)","priority":"P0","estimated_minutes":10},
  {"id":"23.2.2","name":"Verify zero phantom trades / cash-leak regressions (trade<->position reconciliation)","priority":"P0","estimated_minutes":15},
  {"id":"23.2.3","name":"Verify FD leak did not regress (lsof tickets.db over time + sample weekly logs)","priority":"P0","estimated_minutes":10},
  {"id":"23.2.4","name":"Verify pause/resume deadlock did not regress (live timing test + watchdog log review)","priority":"P0","estimated_minutes":10},
  {"id":"23.2.5","name":"Verify kill-switch breach evaluation never falsely fired (audit log review)","priority":"P0","estimated_minutes":10},
  {"id":"23.2.6","name":"Verify sector cap actually blocked Tech buys (autonomous_loop log grep + sector counts)","priority":"P1","estimated_minutes":15},
  {"id":"23.2.7","name":"Verify Red Line Monitor terminal NAV matches live across the week","priority":"P1","estimated_minutes":10},
  {"id":"23.2.8","name":"Verify home cockpit and paper-trading hero metrics stay in sync (useLiveNav SSOT)","priority":"P1","estimated_minutes":10},
  {"id":"23.2.9","name":"Verify ticker-meta latency stays low (sample cold/warm cURL + cache hit rate)","priority":"P1","estimated_minutes":10},
  {"id":"23.2.10","name":"Verify watchdog has not fired in 7 days (handoff/logs/backend-watchdog.log grep)","priority":"P1","estimated_minutes":5},
  {"id":"23.2.11","name":"Verify BQ table freshness (most-recent-row checks across 7 tables)","priority":"P1","estimated_minutes":15},
  {"id":"23.2.12","name":"Verify Layer-1 enrichment pipeline still functional (sample analysis_results lite-path counts)","priority":"P2","estimated_minutes":15},
  {"id":"23.2.13","name":"Verify governance limits-loader watcher still active (audit jsonl grep)","priority":"P2","estimated_minutes":5},
  {"id":"23.2.14","name":"Audit other `with self._lock:` blocks for re-entrant patterns (deferred from 23.1.22)","priority":"P2","estimated_minutes":30},
  {"id":"23.2.15","name":"Run phase-23.1.x cycle-by-cycle smoke tests (the table in Section A)","priority":"P2","estimated_minutes":45},
  {"id":"23.2.16","name":"Phase 2 deferred items triage — pick 3 highest-ROI for next sprint","priority":"P2","estimated_minutes":20}
]
```

---

## Gate envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 22,
  "report_md": "handoff/current/phase-23.2.0-internal-codebase-audit.md",
  "gate_passed": false,
  "gate_passed_reason": "No external research performed (Main wrote internal-only audit after researcher fork timed out). Acceptable for SCOPE-MAPPING task that drives later cycles. Each subsequent 23.2.X cycle will hit the gate independently."
}
```

The user's instruction was **"first make a research of our full codebase then update our masterplan"** — internal mapping is the deliverable, not external literature. Future 23.2.X cycles each need their own research-gate.
