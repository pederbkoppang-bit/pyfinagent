# Contract — step 64.4 (Multi-market fixture-replay e2e)

**Phase:** phase-64 | **Step:** 64.4 | **Priority:** P1 | harness_required: true | depends_on: 66.2 (done)
**Cycle:** 1 | Date: 2026-07-17 | **Type:** test-infra (1 e2e test file + synthetic fixture helper). $0; local-only;
NO production/live-loop change; historical_macro FROZEN; live book untouched. NO network (synthetic fixtures).

## Research-gate summary (gate PASSED)

Researcher subagent (Agent tool, Opus 4.8 effort:max, $0), brief `research_brief_64.4.md`. Envelope:
**gate_passed=true**, tier=complex, **7 external sources read in full**, 10 snippet-only, 17 URLs, recency scan, 12
internal files. VERDICT: **single-cycle GENERATE, NOT blocked on 65.2, NOT multi-session.** KEY:
- **Smallest offline seam** (drive once PER MARKET): `screen_universe` (`backend/tools/screener.py:91`; mock the ONLY
  net call `yf.download` at :137, `yf` module-level :11) → `rank_candidates` (:249, pure) → `decide_trades`
  (`backend/services/portfolio_manager.py:66`, pure :66-240, no bq/net/await → `list[TradeOrder]` with `.market`).
  Per-market funnel = universe→screened→ranked→order-intent counts, driven per market.
- **Fixtures: SYNTHETIC** (deterministic, no network EVER). Reuse `_clean_series`
  (`test_phase_50_5_dataquality.py:15`) which survives `validate_ohlcv` R1-R3. 2-3 tickers/market × ~60 bars (≥20
  required, screener.py:172). US bare, EU `.DE`, KR `.KS` from `INTL_UNIVERSE` (universe_lists.py:17-41). MultiIndex
  via `pd.concat({tkr:_clean_series() for tkr in tickers}, axis=1)`.
- **KEY PITFALL**: drive the PURE seam, NOT the full loop — the loop's phase-50.4 calendar gate calls
  `datetime.now()` (autonomous_loop.py:536) and drops all intl tickers on weekends (flaky funnel=0). The pure
  screen_universe seam never touches the clock.

## Criterion-1 interpretation (documented up front; flagged for Q/A)

Criterion 1: "...per-market funnel counts >0 (**EU under the 65.2 thresholds via test flag**)". Step 65.2 (per-market
threshold PRODUCTION flag) is `pending` and its code does NOT exist yet (grep = 0 hits). We read "**via test flag**"
as a TEST-ONLY override: passing lowered `min_avg_volume`/`min_price` kwargs to `screen_universe` (screener.py:93-94,
already accepted) so EU tickers pass under lowered thresholds — simulating the 65.2 concept without 65.2 production
code. Justification: (a) 64.4's DAG `depends_on` is **66.2 (done), NOT 65.2** — if 64.4 required the real 65.2 flag
the plan would gate it on 65.2; (b) the phrase "via test flag" explicitly means a test-level override, not a prod
flag; (c) 65.2 will productionize the same concept later. **Flagged for the Q/A to adjudicate** (mirrors the accepted
64.2 "(testid)" interpretation pattern).

## Plan

### `backend/tests/test_64_4_multi_market_e2e.py` (NEW)
- **Synthetic fixture helper**: `_ohlcv(tickers, bars=60)` → a `yf.download`-shaped MultiIndex DataFrame built from
  `_clean_series`-style bars (survives validate_ohlcv). Craft EU bars that would fail US-default thresholds but pass
  lowered ones.
- **Per-market cycle test** [criteria 1]: for each market in (US bare, KR `.KS`, EU `.DE`):
  patch `screener.yf.download`→the synthetic fixture; call `screen_universe(universe, market=..., [lowered kwargs for
  EU])` → assert screened count >0; `rank_candidates(screened)` → assert ranked count >0; `decide_trades(...)` →
  assert order-intent (`list[TradeOrder]`) present / funnel >0 for the market. Assert per-market funnel counts all >0
  (US/KR/EU). Drive the PURE seam only (never the loop).
- **Currency invariants** [criterion 2]: in the same file, reuse the 50.2/64.3 fx-mock pattern (`_mk_trader` +
  patch `fx_rates.get_fx_rate` + `paper_avg_entry_fx_fix_enabled`) → assert KR avg_entry stays KRW-scale, EU stays
  EUR-scale.
- **requires_live variant** [criterion 3]: one `@pytest.mark.requires_live` smoke that hits real `yf.download`
  (marker registered pytest.ini:8-9), EXCLUDED by `-m 'not requires_live'`.
- Test names contain `multi_market_e2e` for the `-k` selection. PURE (no network in the default run).

## Immutable success criteria (verbatim from masterplan.json 64.4)
1. "a fixture-replayed cycle produces screening->ranking->order-intent output for ALL THREE markets with per-market
   funnel counts >0 (EU under the 65.2 thresholds via test flag)"
2. "currency invariants asserted in the same test (KR KRW-scale, EU EUR-scale)"
3. "the requires_live variant exists and is excluded from default/CI runs"

**Verification command (immutable):**
`cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -m pytest backend/tests -k 'multi_market_e2e' -q -m 'not requires_live'`

## Boundaries (binding)
$0; local-only; test-infra ONLY (1 new test file + a synthetic-fixture helper inside it; NO production code change).
NO network in the default run (synthetic fixtures; `yf.download` mocked); the requires_live smoke is EXCLUDED from
default/CI. NO trade/risk/money touch; kill-switch/stops/caps/DSR/PBO byte-untouched; historical_macro FROZEN; live
book untouched. `requires_live` list grows by exactly 1 (the intentional live smoke) — that is the criterion-3
deliverable, not a quarantine-of-a-flaky-default. Drive the PURE seam only (calendar-gate pitfall avoided).

## References
research_brief_64.4.md; backend/tools/screener.py:11,91,93,137,172,249; backend/services/portfolio_manager.py:66,34;
backend/backtest/universe_lists.py:17-41; backend/tests/test_phase_50_5_dataquality.py:15 (_clean_series); the 50.2
_mk_trader + 64.3 fx pattern; backend/config/settings.py:455 (paper_avg_entry_fx_fix_enabled); pytest.ini:8-9
(requires_live marker); autonomous_loop.py:536,1698 (the loop path we DELIBERATELY avoid).
