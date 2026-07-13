# Live Playwright + API evidence — trade/diversification audit (2026-07-13)

Captured by Main against a skip-auth `:3100` instance (operator `:3000` untouched,
verified 302→/login healthy after). Backend `:8000` + BQ shared with `:3000`.
`@playwright/mcp@0.0.76`. Screenshot: `handoff/current/captures_audit_2026-07-13/manage-settings-persisted-3.png`.

## S1 — "can't change MAX POSITIONS PER SECTOR": save path WORKS end-to-end

Reproduced on `/paper-trading/manage` > Trading settings:
1. Field `Max positions per sector` editable; typed 2→3; **Save button enabled** on edit (dirty detection works).
2. Clicking Save fired `PUT http://localhost:8000/api/settings/ → 200 OK` (network request #65).
3. `GET /api/settings/` immediately returned `paper_max_per_sector: 3` (persisted; cache cleared).
4. Full page reload → field still reads **3** (no UI revert).
5. `_scheduled_run()` (paper_trading.py:1327) calls `get_settings()` FRESH each cycle, and the PUT calls `get_settings.cache_clear()` → the next scheduled cycle DOES see the new value.

**Conclusion:** the setting saves + persists + propagates to the scheduled loop. The operator's
"can't change it" is therefore NOT a save bug in this running instance. It is (a) a *perception*
that nothing changes because the behavioral effect is masked by the S2/S3 root causes, and/or
(b) one of the environment-conditional S1 shadows the audit found (os.environ over .env when the
backend was started via `set -a; . backend/.env`; or a latent `risk_overrides` override with no UI
to view/clear it). Config restored to `paper_max_per_sector: 2` after the test (PUT → 200).

## S2 — "no new stock from other sectors": monosector candidate funnel (CONFIRMED live)

- Current book: **1 position — AMD (Technology)**, market_value $711. NAV ≈ $23,918, cash ≈ $23,214 (**~97% cash**).
- Lifetime trades: **61 total (31 BUY / 30 SELL) but only 20 UNIQUE tickers** → the engine churns the SAME ~20 names in and out.
- The 20 traded tickers == the live screen universe (`ticker-meta` request #54):
  `000660.KS, 005930.KS, 066570.KS, AMD, CIEN, COHR, DELL, FIX, GEV, GLW, HPE, INTC, KEYS, LITE, MU, ON, SNDK, STX, TER, WDC`
  → **~18/20 are semiconductors / tech hardware**; only GEV + FIX are industrials; the 3 `.KS` are Korean tech.
- Root cause is NOT universe size: `run_daily_cycle` builds the universe from `get_sp500_tickers()` + international (autonomous_loop.py:388). The concentration is introduced by **ranking + top-N truncation**: `paper_screen_top_n=10`, `paper_analyze_top_n=5`; only the top-5 NEW candidates are analyzed (autonomous_loop.py:838), and momentum leadership (semis) dominates the top of the ranked list, so only semis ever become BUY candidates.
- A `sector_neutral` lever EXISTS in `rank_candidates` (screener.py:258, default OFF) but a 2026-06-01 replay found HARD sector-neutral HURTS long-only returns (screener.py:71-83) — so the fix must be *soft* diversification, not a hard switch.
- The per-sector cap (default 2) then BLOCKS the 3rd+ semi and routes it to a swap path that only displaces SAME-sector holdings (portfolio_manager.py:594) → churn within tech, never rotation into a new sector. This is why 61 trades map to just 20 tickers.

## S3 — "as many trades as possible": throughput starved, not churn-starved

- ~97% cash with only 1 position: capital is barely deployed despite `min_cash_reserve=5%` (could deploy ~95%).
- Effective candidate breadth = a handful of momentum-leading semis; once the sector cap blocks them, BUYs stop → 0-trade or 1-trade cycles.
- The 30%/sector NAV cap (`paper_max_per_sector_nav_pct=30`, ON by default, settings.py:277) is a second silent BUY-blocker not surfaced in the UI.
- "More trades" should mean *more diverse capital deployment*, NOT more churn of the same 20 names (which also fights the phase-61 churn-integrity work). The lever is candidate-set diversity + cross-sector rotation, plus surfacing/relaxing the hidden gates.

## Settings surface facts
- Manage-page Trading settings → global `PUT /api/settings/` (NOT a paper-specific route).
- `risk_overrides` audit file EMPTY at audit time → no active shadow currently, but the shadow path (portfolio_manager.py:304) has NO UI to view/clear and can silently override UI-saved caps.
