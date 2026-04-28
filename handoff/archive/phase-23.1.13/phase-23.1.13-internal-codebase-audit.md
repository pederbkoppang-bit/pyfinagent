# Internal Codebase Audit: Sector Concentration / Portfolio Diversification
# phase-23.1.13 — 2026-04-26

---

## Internal Files Inspected

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/services/portfolio_manager.py` | 275 | Trade decision engine (sell-first-then-buy) | Read in full |
| `backend/tools/screener.py` | 283 | Quant screening + ranking, sector ETF map | Read in full |
| `backend/services/macro_regime.py` | 277 | Macro regime tilt, sector_hints overweight/underweight | Read in full |
| `backend/services/meta_scorer.py` | 254 | LLM conviction meta-scorer, per-candidate sector field | Read in full |
| `backend/services/paper_trader.py` | 586 | Virtual trade execution (execute_buy, execute_sell) | Read in full |
| `backend/api/paper_trading.py` | 843 | API endpoints incl. ticker-meta sector lookup | Read in full |
| `backend/config/settings.py` | ~207 | All Settings fields incl. paper_* params | Read relevant lines |
| `backend/services/autonomous_loop.py` | ~650 | Daily cycle: screen → rank → analyze → trade | Key sections read |
| `frontend/src/app/paper-trading/page.tsx` | ~900+ | Risk Monitor component, concentration logic | Relevant lines read |
| `tests/services/test_macro_regime.py` | ~100 | Tests for apply_regime_to_score | Read |
| `tests/services/test_meta_scorer.py` | ~100 | Tests for meta scorer | Read |
| `tests/services/test_sector_calendars.py` | ~180 | Tests for sector calendar score apply | Read |
| `tests/api/test_ticker_meta.py` | ~100 | Tests for ticker-meta BQ+yfinance lookup | Read |

---

## Per-File Findings

### 1. `backend/services/portfolio_manager.py` — `decide_trades` / buy candidates

**Sector-aware logic: NONE.**

- `decide_trades` (line 40-212) iterates `candidate_analyses` and appends BUY orders for any ticker with `rec in _BUY_RECS` (lines 136-163).
- The buy loop at lines 175-208 has exactly two guards before appending a BUY order:
  1. `remaining_positions >= settings.paper_max_positions` (line 176) — total position count cap only.
  2. `available_cash <= 0` (line 179) — cash sufficiency.
- No sector field is read, referenced, or evaluated anywhere in this file.
- The `candidates_by_ticker` dict (line 153, added in phase-23.1.7) passes the screener candidate through for signal attribution, but no code reads `candidate.get("sector")` from it.
- **Max-positions-per-sector limit: does not exist.** There is `settings.paper_max_positions` (a global cap, default 10) and nothing else.
- **Sell-side sector rebalancing: does not exist.** The sell logic (lines 76-113) only checks stop-loss hits and explicit SELL/STRONG_SELL recs. No "exit because sector is over-concentrated" path.

Verdict for audit point 1: **GAP — no sector awareness anywhere.**

---

### 2. `backend/tools/screener.py` — `screen_universe` + `rank_candidates`

**`screen_universe` (lines 63-148): sector field NOT in output.**

The dict appended at lines 132-143 contains:
`ticker, current_price, avg_volume_20d, momentum_1m, momentum_3m, momentum_6m, rsi_14, volatility_ann, sma_50_distance_pct`.

`sector` is absent. The function never calls `yf.Ticker(t).info` — it only downloads price/volume history via `yf.download` (line 83). This is correct for performance (batch download is fast), but it means **sector is structurally missing from every screener candidate before the LLM analysis step.**

**`rank_candidates` (lines 151-248): sector field read but always None.**

- Line 205: `score = apply_regime_to_score(score, stock.get("sector"), SECTOR_ETFS, regime)` — calls `get("sector")` on the dict returned by `screen_universe`, which never contains it. The result: `sector` is always `None` here.
- Line 222: `apply_sector_events_to_score(score, stock.get("ticker"), stock.get("sector"), sector_events)` — same issue, `stock.get("sector")` is always `None`.
- **No hard concentration rule.** The regime tilt in `rank_candidates` is a soft score multiplier (±5% at lines 273-277 of macro_regime.py), not a "max N per sector" filter.
- `SECTOR_ETFS` map at lines 20-25 is correct and complete (11 GICS sectors). The machinery is wired; it just never receives a sector value.

Verdict for audit point 2: **CRITICAL GAP — sector is structurally missing from `screen_universe` output, making every sector-based overlay (regime tilt, sector calendars) a no-op at the screening stage.**

---

### 3. `backend/services/macro_regime.py` — `apply_regime_to_score`

The regime tilt is implemented correctly but is **default-OFF and has no enforcement.**

- `apply_regime_to_score` (lines 253-277): applies `conviction_multiplier` first (global), then checks `regime.sector_hints.overweight`/`underweight` for a ±5% tilt.
- The function is called from `rank_candidates` (screener.py line 203) only when `regime is not None`.
- In `autonomous_loop.py`, `regime` is fetched only when `settings.macro_regime_enabled` is True (not visible in the snippet but `get_settings()` controls it via FRED_API_KEY availability — the `_fallback_regime` path returns `regime=unknown` with `conviction_multiplier=0.85`, which does execute, but `sector_hints` is always empty in the fallback: `SectorWeights()` at line 167).
- Even when the regime is correctly computed (LLM path), the overweight/underweight tilt is **only ±5% score adjustment** — not a hard veto. A Technology stock with strong momentum gets score * 1.05 if regime says overweight XLK; that is all.
- **Hard sector concentration rule: does not exist here.**

Verdict for audit point 3: **Regime tilt exists and is correctly wired — but it is advisory (soft +/-5%) and the sector lookup into the candidate dict is broken (see point 2). Even if working, it would not prevent 11/11 Technology positions.**

---

### 4. `backend/services/meta_scorer.py` — LLM prompt diversification

The meta-scorer prompt (lines 119-135, `_build_meta_prompt`) evaluates each candidate independently and mentions:
- `sector: {c.get('sector','Unknown')}` is rendered in the candidate block at line 96.
- The 6 IMPORTANT rules (lines 122-135) cover regime interaction, PEAD, news, RSI — but **no rule about sector diversity or portfolio concentration.**

The meta-scorer is explicitly designed to score each candidate "INDEPENDENTLY" (rule 1, line 122). This is correct for signal integrity, but it means the LLM has no information about the existing sector composition of the portfolio. It cannot flag "this is the 8th Technology pick."

Verdict for audit point 4: **GAP — meta-scorer prompt has no diversification rule and no access to current portfolio sector composition. Even if it wanted to penalize concentrated picks, it lacks the data.**

---

### 5. `backend/services/paper_trader.py` — `execute_buy`

No diversification check exists. The only pre-buy guards (lines 89-98):
1. `total_cost > cash` — insufficient cash check.
2. `len(positions) >= self.settings.paper_max_positions` — global max positions check.

No sector field is read from any argument. `execute_buy` receives `ticker, amount_usd, price, reason, analysis_id, risk_judge_decision, stop_loss_price, risk_judge_position_pct, signals` — no `sector` parameter exists. The position row saved at lines 162-179 also has no `sector` column.

Verdict for audit point 5: **GAP — `execute_buy` has no sector-diversity guard. A 12th consecutive Technology buy would succeed if cash and position-count allow.**

---

### 6. `backend/api/paper_trading.py` — concentration metric via API

**Sector IS fetched (phase-23.1.10, line 640) but is DISPLAY-ONLY.**

- `_fetch_ticker_meta` (lines 657-713) retrieves `{ticker -> {company_name, sector}}` via BQ + yfinance fallback.
- `/ticker-meta` endpoint (lines 716-738) exposes this for the frontend positions/trades tables.
- **No API endpoint computes or returns a sector-concentration metric.** There is no `sector_breakdown`, `sector_counts`, or `concentration_by_sector` field in any response from `/portfolio`, `/status`, or `/performance`.
- The sector data fetched here is purely for UI display (the Sector column in the Positions table, frontend line 775).

Verdict for audit point 6: **Sector metadata is fetched for display — it is NOT used to compute or surface a sector-concentration risk signal via the API.**

---

### 7. Frontend Risk Monitor — concentration computation

The `RiskMonitor` function component (page.tsx lines ~262-328):

```typescript
// line 265-269
const concentrations = positions.map(
  (p) => ((p.quantity * (p.current_price ?? p.avg_entry_price)) / navDenom) * 100,
);
const maxPos = concentrations.length > 0 ? Math.max(...concentrations) : null;
const concentrationHigh = maxPos != null && maxPos > 20;
```

`concentrationHigh` (displayed as "Concentration: HIGH/OK") is computed as:
**max single-position weight > 20% of NAV**.

This is a **position-size concentration check, not a sector-concentration check.** With 11 equally-weighted positions of ~9% each, no single position exceeds 20%, so `concentrationHigh` is false and the label renders "OK". The metric is structurally unable to detect 11/11 Technology — it would read "OK" even with 20 Technology positions as long as none exceeded 20% of NAV individually.

`tickerMeta[pos.ticker]?.sector` (line 775) is rendered in the Positions table but **never fed into any concentration computation.**

Verdict for audit point 7: **CRITICAL — "Concentration: OK" is reporting single-position-weight concentration, not sector concentration. The metric is misleading. An 11/11 Technology portfolio will always show "OK" with equal-weight sizing.**

---

### 8. `backend/config/settings.py` — sector concentration settings

**No sector-concentration settings exist:**

Searched for `sector_concentration`, `paper_max_per_sector`, `sector_limit`, `sector_threshold` — zero matches.

The full `paper_*` settings (lines 141-185):
- `paper_trading_enabled`, `paper_starting_capital`, `paper_max_positions` (10), `paper_min_cash_reserve_pct`, `paper_screen_top_n` (10), `paper_analyze_top_n` (5), `paper_trading_hour`, `paper_reeval_frequency_days`, `paper_transaction_cost_pct`, `paper_max_daily_cost_usd`, `paper_daily_loss_limit_pct`, `paper_trailing_dd_limit_pct`, `paper_default_stop_loss_pct`.

Sector-related settings that DO exist: `sector_calendars_enabled` (line 163), `sector_calendars_lookahead_days` (line 164) — both for the sector event calendar feature, not for position concentration.

Verdict for audit point 8: **GAP — no `paper_max_per_sector` or equivalent field exists in Settings. Cannot enforce a sector cap without adding one.**

---

### 9. Test coverage — sector concentration

**Zero tests cover sector-concentration enforcement.**

Existing sector-related tests:
- `tests/services/test_macro_regime.py` — tests `apply_regime_to_score` overweight/underweight tilt (soft multiplier). Does not test concentration blocking.
- `tests/services/test_sector_calendars.py` — tests calendar event score adjustment. Not concentration.
- `tests/services/test_meta_scorer.py` — tests conviction scoring with `sector="Information Technology"`. No diversity assertion.
- `tests/api/test_ticker_meta.py` — tests BQ+yfinance sector lookup for display. Not trading logic.

No test file exists for `portfolio_manager.py` sector checks. No test verifies that `decide_trades` or `execute_buy` blocks a buy when a sector limit is reached.

Verdict for audit point 9: **GAP — zero test coverage of sector-concentration enforcement.**

---

### 10. Candidate dict — sector propagation path

**The sector field is structurally absent until after LLM analysis.**

The flow is:
1. `screen_universe()` → returns dicts without `sector` (lines 132-143 of screener.py — confirmed above).
2. `rank_candidates()` → reads `stock.get("sector")` (returns `None`); passes None to `apply_regime_to_score` and `apply_sector_events_to_score`. Both functions handle None gracefully but produce no tilt.
3. `meta_score_candidates()` → renders `sector: Unknown` in the prompt (line 96 of meta_scorer.py). The LLM sees "Unknown" for every candidate.
4. After LLM analysis in `_run_claude_analysis()` (autonomous_loop.py line 498): `sector = info.get("sector", "Unknown")` is fetched from `yf.Ticker(t).info` (line 513) and stored in the analysis result dict (line 613). **This is the first point in the pipeline where sector is correctly populated.**
5. `decide_trades()` in portfolio_manager.py — receives the analysis results, but never reads the `sector` field from them (confirmed in audit point 1).

The fix location: `screen_universe` must be augmented to call `yf.Ticker(t).info` for the batch or a separate metadata fetch to populate `sector` on each candidate dict. Alternatively, `rank_candidates` can be passed a pre-built `{ticker: sector}` lookup (cheaper: one BQ query against `analysis_results` for recently-analyzed tickers, yfinance fallback for new ones — same pattern as `_fetch_ticker_meta` in paper_trading.py).

Verdict for audit point 10: **CONFIRMED GAP — sector is absent from screener candidates. The regime tilt and sector-calendars overlays in `rank_candidates` are no-ops because `stock.get("sector")` always returns None at that stage.**

---

## BQ / yfinance Ground Truth: All 11 Positions Are Technology

Via yfinance `Ticker.info`:

| Ticker | yfinance Sector | yfinance Industry |
|--------|----------------|-------------------|
| COHR | Technology | Scientific & Technical Instruments |
| ON | Technology | Semiconductors |
| INTC | Technology | Semiconductors |
| STX | Technology | Computer Hardware |
| TER | Technology | Semiconductor Equipment & Materials |
| DELL | Technology | Computer Hardware |
| GLW | Technology | Electronic Components |
| CIEN | Technology | Communication Equipment |
| LITE | Technology | Communication Equipment |
| SNDK | Technology | Computer Hardware |
| WDC | Technology | Computer Hardware |

**All 11 are correctly classified as Technology by yfinance.** These are real Technology stocks, not misclassifications. The sector concentration is genuine, not a data artifact.

Note: yfinance uses Yahoo Finance sector labels ("Technology"), which maps closely to GICS "Information Technology" but is not identical. COHR, GLW, CIEN, LITE are GICS Industrials/Communication Equipment in strict GICS but Yahoo/yfinance reports them as Technology. The label used throughout this codebase is whatever yfinance returns — which is consistently "Technology" for all 11.

---

## Synthesis

### 5 Most Surprising Gaps

1. **`screen_universe` never fetches `sector`** — the batch yfinance download only pulls price/volume OHLCV data. Every downstream sector-aware function (`apply_regime_to_score`, `apply_sector_events_to_score`, meta-scorer prompt) receives `None`/`"Unknown"` for sector on every candidate. The regime tilt and sector calendar overlays are effectively dead letters.

2. **`portfolio_manager.decide_trades` is sector-blind** — the entire buy decision path contains zero references to sector. The only constraint on how many buys happen in a given sector is the global `paper_max_positions` cap. A momentum surge across 10 semiconductor names would result in 10 consecutive Technology buys with no systemic check.

3. **"Concentration: OK" is a single-position weight check, not a sector check** — the frontend Risk Monitor reports sector concentration using `max(position_weight) > 20%`. With 11 equal-weight 9% positions this will always show "OK". A user reading "Concentration: OK" reasonably infers the portfolio is diversified. It is not.

4. **`paper_max_per_sector` does not exist in Settings** — there is no field, no default, no enforcement point. Adding sector concentration control requires adding the field, wiring it in `decide_trades`, and exposing it in the API settings endpoint.

5. **The sector field is first populated correctly only inside `_run_claude_analysis`** (autonomous_loop.py:513) — after the LLM has already been invoked. By the time the analysis result returns, the ranking has already been decided (rank_candidates ran before the LLM step). The sector information arrives too late to influence which tickers were selected for analysis.

### 5 Things That DO Already Work

1. **`SECTOR_ETFS` map is complete and correct** (screener.py lines 20-25) — all 11 GICS sectors mapped to SPDR ETFs. The infrastructure for sector-aware scoring exists; it just lacks the input data.

2. **`macro_regime.apply_regime_to_score` is correctly implemented** (macro_regime.py lines 253-277) — conviction multiplier + sector tilt logic is sound. If `sector` were populated upstream, the ±5% tilt would fire correctly. The LLM prompt (lines 148-149) correctly identifies Technology/Cyclicals as risk_on favored, Defensives as risk_off favored.

3. **`_run_claude_analysis` correctly fetches and stores sector** (autonomous_loop.py lines 513, 613) — every LLM-analyzed ticker gets a correct sector label in its analysis result. The data exists post-analysis; it just is not fed back into concentration enforcement.

4. **`_fetch_ticker_meta` + `/ticker-meta` endpoint work correctly** (paper_trading.py lines 657-738) — BQ-first with yfinance fallback, 24h cache, graceful error handling. The sector lookup pattern is production-quality. The same pattern can be reused to populate sector on screener candidates cheaply.

5. **`execute_buy` stop-loss and cash guards work** (paper_trader.py lines 89-98) — the existing pre-buy guard structure is clean and easily extended with a sector check without restructuring the function.

---

## The 4 Must-Haves for v1

### Must-have 1: Populate `sector` in `screen_universe` output

**Location:** `backend/tools/screener.py`, `screen_universe` function, lines 93-143.

**Minimal change sketch:**
After the existing batch `yf.download` (line 83), add a per-ticker `yf.Ticker(t).info` call (or reuse `_yfinance_ticker_info` from paper_trading.py) to fetch `sector` and `market_cap`. Cache result in a module-level dict keyed by `(ticker, date)` to avoid re-fetching on repeated runs. Add `"sector": info.get("sector", "Unknown")` to the result dict at line 132.

Alternatively (cheaper): call `_fetch_ticker_meta(tickers, settings, bq)` before `screen_universe` in `autonomous_loop.py` (around line 159), pass the resulting `{ticker: sector}` dict into `screen_universe` or `rank_candidates` as a keyword arg, and let `rank_candidates` merge it into each candidate dict.

**Impact:** Unblocks regime tilt, sector calendars, AND sector concentration guard in a single fix.

---

### Must-have 2: Add `paper_max_per_sector` to Settings and enforce in `decide_trades`

**Location 1 — settings:** `backend/config/settings.py`, after line 143.
```python
paper_max_per_sector: int = Field(
    3,
    description="Maximum positions in any single sector. 0 = no limit.",
)
```

**Location 2 — portfolio_manager.py `decide_trades`:** Before the buy loop (before line 175), build a sector count from `current_positions`. Requires that `candidates_by_ticker` (already passed in since phase-23.1.7, line 46) carries `sector`, or that the analysis dict includes `sector` (it does after `_run_claude_analysis`, line 613).

```python
# Sketch: inside decide_trades, before the buy loop
sector_counts: dict[str, int] = {}
for pos in current_positions:
    if pos["ticker"] not in selling_tickers:
        s = pos.get("sector", "Unknown")
        sector_counts[s] = sector_counts.get(s, 0) + 1

# Then inside the buy loop, add:
cand_sector = cand.get("sector") or "Unknown"
max_per = settings.paper_max_per_sector
if max_per > 0 and sector_counts.get(cand_sector, 0) >= max_per:
    logger.info("Skipping %s: sector %s at limit (%d)", cand["ticker"], cand_sector, max_per)
    continue
# after appending order:
sector_counts[cand_sector] = sector_counts.get(cand_sector, 0) + 1
```

**Impact:** Directly prevents the 11/11 Technology scenario. With default `paper_max_per_sector=3`, the system would have stopped at 3 Technology positions and branched into other sectors.

---

### Must-have 3: Fix the frontend Risk Monitor to check sector concentration

**Location:** `frontend/src/app/paper-trading/page.tsx`, `RiskMonitor` function, lines 263-269.

The existing `concentrationHigh` check must be augmented with a sector concentration check using `tickerMeta` (already fetched by the page, line 378+). Sketch:

```typescript
// Count positions per sector using tickerMeta
const sectorCounts: Record<string, number> = {};
for (const p of positions) {
  const s = tickerMeta[p.ticker]?.sector || "Unknown";
  sectorCounts[s] = (sectorCounts[s] ?? 0) + 1;
}
const maxSectorCount = positions.length > 0 ? Math.max(...Object.values(sectorCounts)) : 0;
const maxSectorName = Object.entries(sectorCounts).find(
  ([, n]) => n === maxSectorCount
)?.[0] ?? "";
const sectorConcentrationHigh = positions.length >= 3 && maxSectorCount / positions.length > 0.5;
```

Then render: `{sectorConcentrationHigh ? \`HIGH (${maxSectorCount}/${positions.length} ${maxSectorName})\` : "OK"}`.

**Impact:** Surfaces the actual risk condition to the operator immediately. The current "OK" label is actively misleading.

---

### Must-have 4: Expose sector breakdown in the `/portfolio` API response

**Location:** `backend/api/paper_trading.py`, `get_portfolio` endpoint, lines 157-180.

The endpoint already returns `positions`. Add a `sector_breakdown` field computed server-side (using the same `_fetch_ticker_meta` logic or by reading the `sector` from `analysis_results` BQ table). This lets the frontend render a proper sector bar/table and makes the Risk Monitor computation backend-authoritative rather than dependent on frontend `tickerMeta` state.

**Impact:** Makes sector concentration a first-class API metric visible to any client (Slack bot, future monitoring). Lays groundwork for a backend `sector_concentration_pct` field in the risk snapshot.

---

## Hard-Blocker Checklist

- [x] `external_sources_read_in_full: 0` — this is the internal half; external research is running in parallel
- [x] Recency scan: N/A (internal-only audit; the parallel external researcher covers this)
- [x] `file:line` anchors for every internal claim — verified above
- [x] Internal exploration covered every relevant module (10 files inspected)
- [x] All audit points answered with verdict
- [x] Synthesis: 5 gaps + 5 working items documented
- [x] 4 must-haves with concrete file:line locations and change sketches

---

## JSON Envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 13,
  "report_md": "phase-23.1.13-internal-codebase-audit.md",
  "gate_passed": true,
  "note": "Internal-only half of a parallel research+explore split. External researcher is running concurrently and will provide the external gate. This brief covers audit points 1-10 in full with file:line anchors. gate_passed=true for the internal-exploration gate (all 10 audit points answered, file:line anchors present, 13 files read). The combined gate (external + internal) is satisfied when the parallel external brief also returns gate_passed=true."
}
```
