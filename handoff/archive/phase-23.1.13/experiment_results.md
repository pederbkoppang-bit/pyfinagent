---
step: phase-23.1.13
cycle_date: 2026-04-28
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_13.py'
---

# Experiment Results — phase-23.1.13

## What was built

Hard sector cap of **2 positions per GICS sector** (operator-tunable via Manage tab) eliminates the 11/11-Technology bug. Risk Monitor now reflects actual sector concentration (was misleadingly green). Sector field flows end-to-end: ticker-meta lookup → screener candidate → portfolio_manager cap → trade execution.

## Files modified

| File | Change |
|---|---|
| `backend/config/settings.py` | NEW `paper_max_per_sector: int = Field(2, ge=0, le=20)` (0 disables; default 2 = ≥5 sectors for a 10-position portfolio per SEC 25% concentration threshold convention) |
| `backend/api/settings_api.py` | Expose `paper_max_per_sector` (FullSettings + SettingsUpdate + _FIELD_TO_ENV + _settings_to_full) |
| `backend/tools/screener.py` | `screen_universe` accepts optional `sector_lookup` kwarg; attaches `sector` + `company_name` to each result dict (backward compat: works without lookup) |
| `backend/services/autonomous_loop.py` | After `rank_candidates` returns top-N, calls `_fetch_ticker_meta` (BQ-first / yfinance-fallback, already 24h cached) to enrich with sector + company. Non-fatal on error. |
| `backend/services/portfolio_manager.py` | `decide_trades` resolves sector with priority: screener candidate → `analysis.full_report.market_data.sector` → `analysis.sector` → "Unknown". Builds `sector_counts` from `current_positions` (excluding sells); skips BUYs whose sector is at the cap; logs each skip with reason; increments after each accepted BUY |
| `backend/api/paper_trading.py` | `/portfolio` endpoint returns `sector_breakdown: {sector: {count, weight_pct, tickers}}` computed via `_fetch_ticker_meta` lookup |
| `frontend/src/lib/types.ts` | `FullSettings.paper_max_per_sector?` |
| `frontend/src/app/paper-trading/page.tsx` | `RiskMonitorCard` receives `tickerMeta`; computes sector counts; renders BOTH "Position size" (>20% NAV per single position, the existing check) AND "Sector concentration" (>50% of positions in one sector) as separate rows. Manage tab adds `paper_max_per_sector` `PaperSettingNum` with hint |
| `tests/services/test_sector_concentration.py` | NEW (6 tests on decide_trades cap behavior) |
| `tests/services/test_screener_sector_propagation.py` | NEW (4 tests on sector_lookup wiring) |
| `tests/verify_phase_23_1_13.py` | NEW immutable verification (7-claim source-level assertion) |

## Verbatim verification command output

```
$ source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_13.py
ok paper_max_per_sector + screen_universe.sector_lookup + decide_trades cap + autonomous_loop ticker_meta enrichment + RiskMonitor sector check + Manage tab toggle + /portfolio sector_breakdown
exit=0
```

7 claims asserted in one shot:
1. `paper_max_per_sector` field present on Settings + FullSettings + SettingsUpdate (default 2)
2. `screen_universe` accepts `sector_lookup` kwarg
3. `portfolio_manager.py` source contains `max_per_sector`, `sector_counts`, "at cap" log
4. `autonomous_loop.py` calls `_fetch_ticker_meta` to enrich candidates
5. `paper-trading/page.tsx` source contains `sectorConcentrationHigh` + uses `tickerMeta` in `RiskMonitorCard`
6. Manage tab exposes `field="paper_max_per_sector"`
7. `/portfolio` endpoint returns `sector_breakdown`

## Unit test results

```
$ source .venv/bin/activate && python -m pytest tests/api/ tests/services/ -v --no-header -q
collected 170 items
... 170 passed in 3.69s
```

10 new + 160 prior = 170/170 tests pass. Zero regression across 13 phase-23.1 cycles.

## Frontend type-check

```
$ cd frontend && npx tsc --noEmit
(silent — 0 errors)
```

## Research foundation

| Brief | Lines | Sources | Verdict |
|---|---|---|---|
| `phase-23.1.13-external-research.md` | 566 | 13 read in full + 12 snippet-only | gate_passed: true |
| `phase-23.1.13-internal-codebase-audit.md` | 351 | 13 internal files | gate_passed: true |

Both halves converged on the same v1 plan. External brief recommends "max 2 positions per GICS sector + 25% NAV cap + minimum 4 sectors + 10% per-position cap" as the canonical hard-rule set; we ship the position-count cap (2) and rely on the existing per-position 10% sizing convention to approximate the NAV cap (≤20% per sector in practice). Min-sectors enforcement deferred to Phase 2 to allow operator to ramp gradually.

## What changes for the operator at 09:30 ET tomorrow

| Before | After |
|---|---|
| Buy loop blindly fills 11/11 with momentum-clustered Tech names | Buy loop stops at 2 Tech positions; remaining slots forced into other sectors (Energy / Financials / Health / Industrials / etc.) |
| Risk Monitor "Concentration: OK" — misleading green when 11/11 Tech | Two rows: "Position size" (existing single-position >20% check) + "Sector concentration: HIGH (N/M Sector)" amber when >50% of positions in one sector |
| `/portfolio` API has no sector data | Returns `sector_breakdown` (count + weight_pct + tickers per GICS sector); used by Risk Monitor and any future dashboards/Slack |
| Operator can't tune the cap | Manage tab → "Max positions per sector" input (range 0-20; 0 disables) |
| Regime tilt + sector calendars overlays = no-ops because sector=None at ranking step | `sector` populated via `_fetch_ticker_meta` enrichment after `rank_candidates`; downstream overlays now operate on real data |

## Cost / latency

- One `_fetch_ticker_meta` call per cycle for the top-N (≤30) candidates after ranking
- 24h cache means subsequent cycles + `/portfolio` requests hit cache → ~0ms
- Cold-cache hit: ~3s for 30 yfinance lookups (already exists from phase-23.1.10; we reuse the cached result)
- Zero LLM cost added

## Out of scope (per contract; Phase-2)

- New `sector` column on `paper_positions` BQ schema (--apply migration; for v1 sector is read live via `_fetch_ticker_meta` cache lookup)
- HRP / risk-parity post-selection optimizer
- Sector-neutral re-ranking (rank within sector before merging)
- Correlation-cluster deduplication
- Forced rebalance when EXISTING positions exceed cap (cap only blocks NEW buys for v1)
- Min-sectors enforcement (≥4 distinct sectors)
- Strict 25% NAV-per-sector hard cap (position-count cap of 2 with 10% per-position cap implies ~20% max — close to canonical 25% for v1)

## Honest disclosure

- **Existing 11 Technology positions are NOT auto-liquidated.** The cap blocks NEW buys; existing concentrated positions remain until they hit a stop-loss / sell signal naturally. To force diversification immediately, the operator can manually flatten and let tomorrow's cycle build a fresh 2-per-sector portfolio under the new rules.
- **GICS classification depends on yfinance**, which uses Yahoo Finance sector labels (e.g., "Technology" not strict GICS "Information Technology"). Most US large-caps map cleanly; some edge cases (Berkshire = Financials in GICS, "Financial Services" in yfinance) work fine for cap purposes since labels are stable per ticker.
- **Default cap=2 is a practitioner consensus value**, not derived from the operator's specific risk profile. Adjustable via Manage tab. Setting to 0 disables the cap (legacy behavior).
- **The `paper_max_positions: 10` cap still applies** — combined with `paper_max_per_sector: 2`, this implies the system can only hold positions across at least 5 distinct sectors before stopping (which is the diversification goal).

## What's next

1. Spawn fresh Q/A
2. On PASS: log → flip → archive → commit → restart backend + frontend
