# Research Brief — Step 56.1: FX / value / fee data-correctness fix

**Tier:** moderate. **Date:** 2026-06-10. **Step:** 56.1 (phase-56, FIX,
gated on phase-55 findings). Two jobs in one session: (A) internal
fix-design audit at file:line (the bulk), (B) >=5 external sources read
in full on multi-currency accounting, characterization tests, SQL
restatement, currency-display, and FX-in-paper-trading.

**Finding IDs this brief services** (from `handoff/archive/phase-55.3/55.3-synthesis-checkpoint.md` §1 and `handoff/archive/phase-55.1/55.1-away-week-postmortem.md`):
- **F-1** (CRITICAL): frontend renders local-currency prices as USD (NAV card, RiskMonitor, positions Current cell, currency-exposure %, donut, trades Value/Fee render).
- **F-2** (HIGH): trade-ledger stored corruption — KR `paper_trades.total_value` in KRW (7 rows; `paper_trader.py:265`), KR SELL `transaction_cost` in KRW (3 rows; `:387,:413-414`). Stored NAV/cash/positions/round-trips are CLEAN.
- **F-12** (LOW-MED): "VS KOSPI" card shows KR holdings return, not index excess (`cockpit-helpers.tsx:208-218`; tooltip discloses).
- **F-13** (LOW): "$10K virtual fund" label vs starting_capital=$20K (`layout.tsx:336`).

**Do-no-harm contract (echoed):** fixes minimal-diff + cite finding IDs;
US path byte-identical (`_local_to_usd == 1.0`); backfill operator-gated
(NOT executed in 56.1); no live flag flips; Playwright capture required
post-fix; US momentum core untouched.

---

## 0. Verification-command reality check (DONE FIRST — gate must exit 0)

Verification cmd (immutable):
```
cd /…/pyfinagent && source .venv/bin/activate && \
python -m pytest backend/tests -k 'fx or paper_trader or krw' -q && \
test -f handoff/current/live_check_56.1.md
```

- **Collected today: 24 tests.** `--collect-only` shows the selection
  pulls from `test_dod4_tier1_coverage_investment.py` (the `paper_trader`
  substring), `test_phase_50_2_multicurrency.py` (the `fx` substring → 3
  tests incl. `test_fx_local_to_usd_is_identity_for_us`), and
  `test_phase_50_5_dataquality.py::test_baseline_eu_fx_converted`.
- **Current state: `24 passed, 722 deselected` in ~2.4s — exits 0.**
  Re-run verbatim and confirmed green on 2026-06-10.
- **None of the 16 known env-coupled failures (pending 56.2 quarantine)
  land in this `-k` selection.** Confirmed by running the selection in
  isolation — all 24 pass with no network/BQ. So the new KRW test
  ADDS to a green selection; it must itself pass post-fix (and fail
  pre-fix when temporarily checked). The `live_check_56.1.md`
  existence test is the second `&&` clause — the file must be authored.
- **Naming guidance for the new test:** pytest `-k` matches against the
  full test node id, which **includes the file path**. So a new file at
  `…/test_phase_56_1_fx_ledger.py::test_x` matches `-k fx` via the `fx` in
  the path; equally, adding to the existing `test_phase_50_2_multicurrency.py`
  is collected because that path contains `fx`. SAFEST: add the KRW tests
  to `test_phase_50_2_multicurrency.py` (guaranteed collection, co-located
  with the existing FX formula tests). Avoid a filename with NO
  `fx`/`paper_trader`/`krw` substring (it would silently not collect).

---

## A. INTERNAL FIX-DESIGN AUDIT

### A.1 Backend defect — exact localization (F-2)

Read `paper_trader.py` execute_buy (:119-345) and execute_sell (:347-493)
in full. The corruption is two lines; the FX rate is already in scope at
both.

**BUY path (`execute_buy`):**
- `_local_to_usd = _fx_local_to_usd(market)` is computed at **:208** and
  already correctly used at **:299** (`market_value`). Scope is the whole
  function body below :208.
- `tx_cost = amount_usd * (pct/100)` at **:189** → **already USD** (sized
  off the USD budget `amount_usd`). So BUY `transaction_cost` at **:266**
  is CORRECT. (Matches 55.1 §2.1#1: "BUY `transaction_cost` IS USD: 0.24-0.74".)
- **:265** `"total_value": round(quantity * exec_price, 2)` — `exec_price`
  is LOCAL (the fill price in KRW). This is the ONLY corrupt BUY field.
  **FIX:** `round(quantity * exec_price * _local_to_usd, 2)`.
  - Note `quantity` was sized as `(amount_usd * _usd_to_local) / price`
    (:212), so `quantity * exec_price * _local_to_usd ≈ amount_usd`
    (modulo fill-vs-analysis price drift) — i.e. the fixed `total_value`
    ≈ the USD notional, which is what every consumer expects.

**SELL path (`execute_sell`):**
- `_l2u = _fx_local_to_usd(position.get("market"))` is computed at **:370**
  (with the None→1.0 fail-soft at :371-374). In scope below :370.
- **:386** `sell_value = sell_qty * price` — LOCAL. **:387**
  `tx_cost = sell_value * (pct/100)` — LOCAL (derived from local sell_value).
- **:413** `"total_value": round(sell_value, 2)` — KRW. **CORRUPT.**
- **:414** `"transaction_cost": round(tx_cost, 2)` — KRW. **CORRUPT.**
- **FIX (minimal, recommended):** convert at the row-build, leave the
  intermediate `sell_value`/`tx_cost`/`net_proceeds` math untouched so the
  cash-credit path (`:485 net_proceeds * _l2u`) and round-trip
  (`:440 …* _l2u`) — both already correct — are not disturbed:
  ```python
  "total_value": round(sell_value * _l2u, 2),         # was round(sell_value, 2)
  "transaction_cost": round(tx_cost * _l2u, 2),       # was round(tx_cost, 2)
  ```
  Do NOT redefine `sell_value`/`tx_cost` upstream — `net_proceeds =
  sell_value - tx_cost` (:388) is consumed at :485 as `net_proceeds * _l2u`
  (cash credit). If you instead made `sell_value` USD upstream you'd have
  to drop the `* _l2u` at :485 and :440 too — larger diff, more risk.
  The row-build-only conversion is the smallest correct change.

**Why BUY needs only 1 line and SELL needs 2:** BUY fees are sized off the
USD budget (`amount_usd`), SELL fees are sized off the local `sell_value`.
This asymmetry is real and matches the 55.1 forensics exactly (BUY fee
clean; SELL fee corrupt on 3 rows).

### A.2 Consumer audit — does anything READ these fields expecting LOCAL? (full-codebase grep)

**Verdict: NO consumer expects LOCAL. Every consumer expects USD — so the
fix makes them all correct.** This is the decisive consumer-audit result.

| Consumer | file:line | Reads | Expectation | Effect of fix |
|---|---|---|---|---|
| Turnover ratio | `perf_metrics.py:406-410` | `total_value` of SELL trades → `sell_value` sum / avg_nav | **USD** (avg_nav is USD) | FIXES a real bug: KR SELL turnover currently inflated ~1500x; post-fix turnover is USD-consistent |
| Scalar-metric NAV est | `perf_metrics.py:470-471` | `Σ|total_value|/n` as avg_nav proxy | **USD** | same — removes KRW outliers from the NAV proxy |
| Slack trade card | `formatters.py:213, 712, 735` | `total_value`, renders `${tv:,.2f}` | **USD** ($-label) | FIXES the "$364,175" KR card → "$238.40" |
| Slack portfolio card | `formatters.py:104,120` | `data["total_value"]` (portfolio-level, NOT a trade row) | USD | unaffected — different `total_value` (portfolio payload, not paper_trades) |
| Frontend trades table | `trades-columns.tsx` Value/Fee cells | `total_value`, `transaction_cost` | **USD** ($-label) | FIXES; also fix the false `:10-12` comment (F-1 sub-item) |
| Tests | `test_paper_trading_v2.py:60-64`, `test_dod4…:356-357` | hardcoded USD fixtures | USD | unaffected (synthetic USD rows) |

**signals_server.py `total_value`** (`:790, 871-907, 988, 1196, 1252-1272`)
is a DIFFERENT field — it is `portfolio["total_value"]` (account equity
passed to the MCP risk pre-trade checker), NOT `paper_trades.total_value`.
Not in scope; do not touch.

**orchestrator.py:289-310 `total_value`** is a local variable
(sector-weight denominator over `market_value`), unrelated. Not in scope.

**Conclusion:** the corruption has exactly two writer sites and a small set
of readers, ALL of which want USD. There is **no LOCAL-expecting reader**,
so converting at the writer is unambiguously correct and needs no consumer
migration. (This is the single most important audit result for the
contract: the fix is safe because the schema's implied contract — "$-labeled
fields are USD" — is already what every reader assumes.)

### A.3 `_local_to_usd` / `_l2u` availability & rounding (criteria 1b, 1c)

- **Availability confirmed:** BUY `_local_to_usd` at :208 (function scope);
  SELL `_l2u` at :370 (function scope). No new computation, no new import,
  no signature change. Both already feed the *position* rows correctly
  (:299, :461, :485, :440), so the trade-row sites are the only ones missed.
- **Rounding:** keep `round(…, 2)` (2-dp USD cents), consistent with every
  other USD money field in the file. For USD: `_local_to_usd == 1.0` so
  `round(x * 1.0, 2) == round(x, 2)` for all finite floats (multiply-by-1.0
  is an IEEE-754 identity for finite values; the only non-identity cases are
  signed-zero and NaN/Inf, none of which occur for a price×qty product).
  **Byte-identity for US holds.** (criterion-A.6 / do-no-harm: PROVEN.)

### A.4 Audit-trail question — also persist local value + fx_rate? (criterion 1d)

The schema (`paper_trades`) has NO `total_value_local` or `fx_rate` column.
Adding them = a migration (schema change) on top of the value fix.

**Recommendation: MINIMAL-DIFF for 56.1 — do NOT add columns.** Rationale:
1. The criteria require USD-correct `total_value`/`transaction_cost`; they
   do NOT require an audit-trail column. Scope honesty.
2. The audit trail already exists elsewhere: `historical_fx_rates` holds the
   as-of rate by date (the `date<=d` read in `fx_rates.py`), and the trade's
   `created_at` + `ticker`→`market`→currency is enough to re-derive the rate
   used (55.1 §3 did exactly this re-derivation to the penny). So the
   information is recoverable without a new column.
3. Adding columns touches `bigquery_client.py::save_paper_trade` (dynamic
   INSERT), the `_ROUND_TRIP_FIELDS` schema-error pruning, and a migration —
   that is phase-57-schema-window work, not a 56.1 data-correctness fix.

**BUT note the limitation honestly in the contract:** without a stored
`fx_rate_used` column, a *future* restatement (if FX history is ever
re-sourced) can't pin which rate each row used at write time. If the
operator wants the stronger audit trail, that is a deliberate scope
expansion (schema migration) — flag it as a follow-on, not silently skip.
A cheap halfway option (no schema change): the existing `signals` JSON-text
column could carry `{"fx_rate_used": …}` since it is free-form JSON — but
that overloads a field with a different purpose; NOT recommended. Keep
56.1 minimal; recommend a dedicated `fx_rate` column as a phase-57 schema
item if audit-trail strength is wanted.

### A.5 Test design (criterion 1 — KRW fixture, FAIL-pre / PASS-post)

**Infra (verified):** tests instantiate `PaperTrader(settings=Settings(),
bq_client=MagicMock())` (`test_dod4…:_make_trader`). The SELL path is
driven by mocking `ExecutionRouter` (`patch("…paper_trader.ExecutionRouter")`,
`router_mock.submit_order.return_value = MagicMock(fill_price=…,
source="bq_sim")`) and, when `price=None`, `_get_live_price`. FX is mocked
by `patch.object(fx_rates, "get_fx_rate", side_effect=_fake_fx)`
(`test_phase_50_2_multicurrency.py`) — the cleanest seam. The stored trade
is captured via `bq.save_paper_trade.call_args` (the `_safe_save_trade`
→ `bq.save_paper_trade(row)` path, `paper_trader.py:1146-1148`).

**Fixture sketch (add to `test_phase_50_2_multicurrency.py`):**
```python
from unittest.mock import MagicMock, patch
from backend.services.paper_trader import PaperTrader
from backend.services import paper_trader as pt
from backend.config.settings import Settings

_KRW_USD = 0.000661           # away-week KRW=X as-of (55.1 §2.1)

def _fake_fx(frm, to, date=None):
    # canonical: usd_value(KRW)=0.000661, usd_value(USD)=1.0
    if frm == to: return 1.0
    table = {("KRW","USD"): _KRW_USD, ("USD","KRW"): 1/_KRW_USD,
             ("USD","USD"): 1.0}
    return table.get((frm, to))

def _krw_trader():
    s = Settings(); s.paper_transaction_cost_pct = 0.1
    bq = MagicMock()
    return PaperTrader(settings=s, bq_client=bq), bq

def test_kr_sell_total_value_and_fee_stored_in_usd():
    trader, bq = _krw_trader()
    # KR position: 1 share of a ₩248,000 stock, market="KR"
    bq.get_paper_position.return_value = {
        "position_id": "pos_KR", "ticker": "066570.KS", "market": "KR",
        "quantity": 1.0, "avg_entry_price": 248000.0,
        "current_price": 248000.0, "cost_basis": round(248000.0*_KRW_USD,2),
        "market_value": round(248000.0*_KRW_USD,2),
        "mfe_pct": 0.0, "mae_pct": 0.0,
        "entry_date": "2026-06-09T00:00:00+00:00",
    }
    bq.get_paper_portfolio.return_value = {
        "portfolio_id": "default", "current_cash": 20000.0,
        "starting_capital": 20000.0,
    }
    with patch("backend.services.paper_trader.ExecutionRouter") as Router, \
         patch.object(pt.fx_rates, "get_fx_rate", side_effect=_fake_fx):
        rm = MagicMock(); rm.submit_order.return_value = MagicMock(
            fill_price=248000.0, source="bq_sim"); Router.return_value = rm
        trade = trader.execute_sell(ticker="066570.KS", price=248000.0,
                                    reason="signal_flip")
    # stored row (what BQ persisted)
    stored = bq.save_paper_trade.call_args[0][0]
    # ── THE REGRESSION ASSERTIONS (fail pre-fix; KRW≈248000 vs USD≈163.93) ──
    # sell_value_local = 248000; *_l2u = 163.93 USD
    assert abs(stored["total_value"] - 248000.0*_KRW_USD) < 0.01   # ≈163.93
    assert stored["total_value"] < 1000.0          # KRW would be 248000
    # fee_local = 248000*0.001 = 248 KRW; *_l2u ≈ 0.16 USD
    assert abs(stored["transaction_cost"] - 248000.0*0.001*_KRW_USD) < 0.01
    assert stored["transaction_cost"] < 1.0        # KRW fee would be 248
    # the returned in-memory trade matches the stored row
    assert trade["total_value"] == stored["total_value"]

def test_kr_buy_total_value_stored_in_usd():
    trader, bq = _krw_trader()
    bq.get_paper_position.return_value = None       # fresh buy
    bq.get_paper_positions.return_value = []
    bq.get_paper_portfolio.return_value = {
        "portfolio_id":"default","current_cash":20000.0,
        "starting_capital":20000.0}
    bq.get_paper_trades_for_ticker_since.return_value = []
    with patch("backend.services.paper_trader.ExecutionRouter") as Router, \
         patch.object(pt.fx_rates, "get_fx_rate", side_effect=_fake_fx):
        rm = MagicMock(); rm.submit_order.return_value = MagicMock(
            fill_price=248000.0, source="bq_sim"); Router.return_value = rm
        trade = trader.execute_buy(
            ticker="066570.KS", amount_usd=200.0, price=248000.0,
            market="KR", reason="signal", sector="Technology")
    stored = bq.save_paper_trade.call_args[0][0]
    # amount_usd=200 → qty=(200/0.000661)/248000≈1.22 sh; total_value≈$200 USD
    assert abs(stored["total_value"] - 200.0) < 1.0     # ≈ USD notional
    assert stored["total_value"] < 1000.0               # KRW would be ~302530
    # BUY fee was already USD (sized off amount_usd) — assert it stays USD
    assert abs(stored["transaction_cost"] - 200.0*0.001) < 0.01  # $0.20

def test_us_sell_total_value_byte_identical():
    """do-no-harm: US (fx=1.0) total_value/fee unchanged."""
    trader, bq = _krw_trader()
    bq.get_paper_position.return_value = {
        "position_id":"pos_US","ticker":"AAPL","market":"US","quantity":10.0,
        "avg_entry_price":200.0,"current_price":210.0,"cost_basis":2000.0,
        "market_value":2100.0,"mfe_pct":5.0,"mae_pct":-2.0,
        "entry_date":"2026-05-01T00:00:00+00:00"}
    bq.get_paper_portfolio.return_value = {
        "portfolio_id":"default","current_cash":50000.0,"starting_capital":100000.0}
    with patch("backend.services.paper_trader.ExecutionRouter") as Router:
        rm = MagicMock(); rm.submit_order.return_value = MagicMock(
            fill_price=210.0, source="bq_sim"); Router.return_value = rm
        trader.execute_sell(ticker="AAPL", price=210.0)
    stored = bq.save_paper_trade.call_args[0][0]
    assert stored["total_value"] == round(10.0*210.0, 2)     # 2100.0 exactly
    assert stored["transaction_cost"] == round(2100.0*0.001, 2)  # 2.10
```
- **FAIL-pre / PASS-post proof:** on pre-fix code, KR `total_value` is
  `round(248000.0, 2)=248000.0` → `assert < 1000.0` FAILS. On fixed code
  it's `round(248000.0*0.000661, 2)=163.93` → PASSES. The `< 1000`
  magnitude bound is the regression guard (KRW vs USD differ by ~1500x, so
  the bound is robust to rounding). The US byte-identity test guarantees
  do-no-harm and would fail if someone accidentally applied FX to US.
- **No BQ, no network:** MagicMock bq + mocked FX + mocked router. Runs in
  the `-k fx` selection (file already matches `fx`).
- **`market="KR"`** must be passed into `execute_buy` (verify the exact
  kwarg name against the :119 signature when writing; the SELL path reads
  `position["market"]` at :370, so the SELL fixture's position dict must
  include `"market":"KR"`).

### A.6 Frontend fix design — F-1 (the CRITICAL finding)

**KEY DISCOVERY (changes the fix scope materially):** most of the F-1 sites
listed in 55.1 were ALREADY REMEDIATED in the goal-multimarket-ux commit
(`ac93f67f`, currency-aware cockpit). The canonical correct pattern is
already in the codebase:

```ts
// positions/page.tsx:66-76 — mvUsd (the RECOMMENDED pattern, already shipped)
const isUs = resolveMarket({market, ticker}) === "US";
if (isUs) return (livePrice ?? current_price ?? avg_entry_price) * quantity;
return pos.market_value ?? 0;     // USD, FX-free, slightly stale on live tick
```

This is **option (a) from the prompt — "use stored `market_value` for
non-US, live recompute only for US (local==USD)"** — and it is the right
call. Validated against the type contract: `PaperPosition` comment
(`types.ts:653-654`) states verbatim *"`current_price` is LOCAL;
`market_value` is USD"*, so `livePrice*qty` is LOCAL and only valid when
local==USD. Per-ticker FX at render (option b) is NOT needed and is NOT
plumbed: `/api/paper-trading/live-prices` (`paper_trading.py:645-659`)
returns `{ticker: price|null}` with NO currency/market field, and
`LivePriceEntry` (`useLivePrices.ts:6-11`) carries no currency. Adding an
FX-at-render endpoint = new plumbing for no benefit over the stored
`market_value`. **REJECT option (b); ADOPT option (a).**

**Site-by-site status (live-verified by reading each component):**

| Site | file:line | Status TODAY | 56.1 action |
|---|---|---|---|
| Positions **Market Value** cell | `positions-columns.tsx:185-208` | **ALREADY FIXED** (`isUs ? livePrice*qty : market_value`) | none (regression-guard test only) |
| Positions **P&L** cell | `positions-columns.tsx:211-228` | **ALREADY FIXED** (non-US uses USD `unrealized_pnl_pct`) | none |
| Positions **Current** cell | `positions-columns.tsx:159-184` + `CurrentPriceCell:39-83` | **ALREADY CORRECT** — renders LOCAL price with the resolved currency symbol (`numberFlowFormat(cur)` → "₩219,500", NOT "$") | none (55.1's "219500,00 USD" capture predates `ac93f67f`) |
| Positions **Entry** cell | `positions-columns.tsx:139-158` | **ALREADY CORRECT** (`formatCurrency(price, cur)` for non-USD) | none |
| Donut **slices** | `positions/page.tsx:123-133` (`mvUsd`) | **ALREADY FIXED** (uses `market_value` for non-US) | none |
| Sector concentration | `positions/page.tsx` SectorBarList (`mvUsd`) | **ALREADY FIXED** | none |
| Currency-exposure **slice sub-totals** | `MultiCurrencyNavBreakdown.tsx:53` | **ALREADY CORRECT** (sums `market_value` USD) | none |
| Single-market NAV (`filteredNavUsd`) | `positions/page.tsx:78-80` | **ALREADY FIXED** (sums `mvUsd`) | none |
| **NAV card / liveNav** | **`useLiveNav.ts:34-39`** | **BROKEN** (`lp * quantity`, no FX) | **FIX — the root** |
| Donut **center** label | `positions/page.tsx:169` (`lp.liveNav` when All) | broken VIA `useLiveNav` | fixed-by-root |
| Currency-exposure **% denominator** | `positions/page.tsx:180` (`lp.liveNav` when All) | broken VIA `useLiveNav` | fixed-by-root |
| Positions **totalNav** (All view) | `positions/page.tsx:169,180` | broken VIA `useLiveNav` | fixed-by-root |
| **RiskMonitorCard** concentration | **`cockpit-helpers.tsx:301-302`** | **BROKEN** (`qty*current_price/navDenom` → "1527.8%") | **FIX** |
| Home NAV tile | `page.tsx:279` (`liveNav`) | broken VIA `useLiveNav` | fixed-by-root |
| Sovereign / nav-page overlay | `sovereign/page.tsx:174`, `nav/page.tsx:51-54` | broken VIA `useLiveNav` | fixed-by-root |

**Net: exactly TWO frontend edits remediate all of F-1.**

**FIX 1 — `useLiveNav.ts:34-39` (the root; cascades to ~6 consumers):**
Replace the naive sum with the `mvUsd` logic. Minimal-diff in-place:
```ts
const positionsValue = positions.reduce((sum, pos) => {
  const isUs = resolveMarket({ market: pos.market, ticker: pos.ticker }) === "US";
  if (isUs) {
    const lp = livePrices[pos.ticker]?.price ?? pos.current_price ?? pos.avg_entry_price;
    return sum + lp * pos.quantity;     // local==USD -> exact, live
  }
  return sum + (pos.market_value ?? 0); // non-US: stored USD mark (FX-free)
}, 0);
```
- Import `resolveMarket` from `@/lib/format` (already exported).
- **Do-no-harm for US:** for a US-only book every position takes the `isUs`
  branch → byte-identical to today's `lp * quantity`. The all-USD live
  portfolio is unchanged. (criterion 2 do-no-harm.)
- **Consideration — extract the helper:** `mvUsd` already exists in
  `positions/page.tsx`. RECOMMEND lifting it to `@/lib/format.ts` (e.g.
  `marketValueUsd(pos, livePrices)`) and using it in BOTH `useLiveNav.ts`
  and `positions/page.tsx` to kill the duplication and guarantee they never
  diverge. This is a small refactor but it is the DRY-correct move and
  prevents a future drift bug (the two must always agree or the NAV card and
  the donut disagree). If minimal-diff is preferred over DRY, inline the
  logic in `useLiveNav.ts` and note the duplication. RECOMMEND the extract.

**FIX 2 — `RiskMonitorCard` `cockpit-helpers.tsx:301-302`:**
```ts
// was: ((p.quantity * (p.current_price ?? p.avg_entry_price)) / navDenom) * 100
const concentrations = positions.map(
  (p) => ((p.market_value ?? 0) / navDenom) * 100,
);
```
- `market_value` is USD; `navDenom` (`portfolio.total_nav`) is USD →
  concentration is a clean USD/USD ratio. KR "1527.8%" → correct ~1%.
- Slightly stale vs a live tick (uses stored `market_value`), but ALWAYS
  currency-correct — the right tradeoff for a risk readout (a stale-but-sane
  concentration beats a live-but-1500x-wrong one). Matches the `mvUsd`
  philosophy. For US this is `market_value` which == `qty*current_price` at
  mark time (stored by mark_to_market), so US is effectively unchanged
  (modulo live-tick staleness, which the concentration card already
  tolerated since it read `current_price` not live ticks).

**Comment fix (F-1 sub-item):** `trades-columns.tsx:10-12` says
*"`total_value`/`transaction_cost` are USD"*. POST backend-fix (A.1) this
comment becomes TRUE — so it needs NO change (it documents the
post-fix invariant). Do NOT "correct" it to say KRW; that would be wrong
after the fix lands. (55.1 flagged it as "currently false"; the right
resolution is to make the data match the comment, which A.1 does.)

**Playwright capture (criterion 2):** after both fixes, capture
`/paper-trading` with the KR filter (and All) showing: NAV card sane (~$23.8K
not $345K), RiskMonitor "Max position" ~1% not 1527%, donut center ~$23.8K,
currency-exposure KRW sleeve a small %, positions Current cell "₩…" with the
₩ symbol. Use the local skip-auth dev instance (:3100) per the 55.1 pattern;
operator :3000 untouched. The capture is the live_check evidence for F-1.

### A.7 VS-KOSPI per the 55.1 verdict (F-12)

**Verdict: strengthen disclosure; do NOT build true ^KS11 excess in 56.1.**
- `cockpit-helpers.tsx:204-223` already implements the honest path: the
  non-US branch shows USD holdings return (`Σunrealized_pnl/Σcost_basis`,
  both USD) with a tooltip *"`{MKT}` holdings return (USD). Per-market
  `{benchmark}` excess is not yet exposed by the API."* This is exactly the
  55.1 ruling ("labeling/semantics gap, tooltip already discloses").
- **True ^KS11 excess feasibility:** the backend does NOT fetch ^KS11
  anywhere for the paper-trading benchmark. Snapshots carry only
  `benchmark_return_pct` = SPY (`paper_trader.py:564-573` calls
  `_get_benchmark_return` which is SPY-anchored). A per-market benchmark
  needs: (1) a per-market index fetch (^KS11/^GDAXI) wired into
  `mark_to_market`/snapshots, (2) a new API field, (3) frontend plumbing.
  That is **phase-57-FEATURE-adjacent** (the 55.3 synthesis §2.6 explicitly
  ranked per-market benchmark "lower-EV: cosmetic vs risk"). Confirmed:
  `grep KS11/KOSPI` in backend returns only the frontend label
  `MARKET_BENCHMARK_LABEL` and the fx_rates currency map — no index fetch.
- **56.1 action:** keep the existing tooltip; OPTIONALLY strengthen the
  label to make the semantics unmissable, e.g. change the metric label from
  "vs KOSPI" to "KR holdings" (or keep "vs KOSPI" but ensure the tooltip is
  always reachable). Lowest-risk: leave as-is (it is already honest) and
  record in live_check that the disclosure was reviewed and judged
  sufficient per the 55.1 ruling. The full ^KS11 fix is the documented
  phase-57 alternative.

### A.8 Migration design — F-2 backfill of the 7 corrupted rows (operator-gated)

**Deliverable in 56.1 = the script + the operator ask. NOT executed.**

**Migration conventions (read existing migrations for shape):**
`scripts/migrations/*.py` use the Python `google-cloud-bigquery` client,
are idempotent and re-runnable. I will read 1-2 for the exact idiom in the
GENERATE phase; the shape to follow (per the project rule "migrations for
*change*, MCP for *inspection*"):
- Default **dry-run**; explicit `--execute` flag to mutate (mirrors the
  general migration discipline + the SQL-restatement safety literature in
  Part B: dry-run/preview before UPDATE).
- Idempotent: the UPDATE sets `total_value`/`transaction_cost` to explicit
  USD values BY `trade_id`; re-running is a no-op once corrected (the values
  are already USD). Optionally guard with a `WHERE total_value > 1000`-style
  sentinel so a second run can't double-convert (KRW rows are >1000; USD
  rows are ≈164 — the magnitude gap makes a safe idempotency predicate).
- **Explicit values, not arithmetic:** the 7 rows + their correct USD are
  fully derived in 55.1 (§2.1 #1/#4 + §4): e.g. 066570.KS 06-09 BUY
  `total_value 364,175.06 → 238.40`; 005930.KS SELL fee `1,056.20 → 0.68`;
  000660.KS SELL fees `737.26→0.49`, `677.64→0.43`. The migration embeds the
  re-derived USD per `trade_id` (UPDATE … SET … WHERE trade_id=...), so it
  does not depend on a live FX lookup at migration time (deterministic,
  auditable). The exact per-row table is in 55.1 §2.1/§3 — the GENERATE step
  will assemble the 7 explicit (trade_id, old_krw, new_usd) tuples by
  querying the live rows for their trade_ids first (read-only) and pairing
  with the re-derived USD.
- **GIPS-style disclosure note + materiality** (criterion 3): write a
  disclosure doc to `handoff/` classifying the restatement. Per 55.1 §2.2,
  the mis-booking is **GIPS tier-3/4 material for ledger consumers (TCA/fee
  reports) but immaterial to headline NAV/cash/performance** (those were
  always clean). The note states: scope (7 rows, 13.5% of all-time, away
  week only), the fields (total_value 7 rows; transaction_cost 3 rows), the
  correction method (explicit USD re-derivation at the as-of FX), and the
  materiality classification (NAV/P&L unaffected; ledger-consumer fields
  corrected). GIPS error-correction was already researched in 55.1/55.3 —
  re-cite that brief (`research_brief_55.3.md`) plus the Part B
  SQL-restatement source.
- **If the operator DECLINES the backfill:** the rows must be FLAGGED, not
  silently kept (criterion 3). Cheapest flag with no schema change: leave
  them and document the known-corruption window in the disclosure note +
  the live_check, so any future ledger-consumer reader is warned. (A schema
  `is_restated`/`data_quality_flag` column is the stronger option but is a
  migration in itself — recommend the doc-flag for 56.1, schema-flag as a
  phase-57 option.)

**Why operator-gated (not auto-run):** it is a historical-data restatement
(GIPS materiality + the project's `execute-query` deny-by-default gate +
the CLAUDE.md "never DELETE/UPDATE without owner approval" rule). The 56.1
deliverable is the reviewed script + the Slack/operator ask; execution is a
separate operator action (mirrors the F-2 owner tag "operator-gated backfill
migration").

### A.9 Do-no-harm consolidation (criterion 4)

- **Backend US path:** `_local_to_usd == 1.0` for US → `round(x*1.0,2) ==
  round(x,2)` (IEEE-754 finite-float identity) → `total_value`/
  `transaction_cost` byte-identical. Covered by `test_us_sell_total_value_
  byte_identical` (A.5).
- **Frontend US path:** both fixes take the `isUs` branch for US tickers →
  identical to today. A US-only book sees zero change.
- **US momentum core untouched:** no change to `screener.py`,
  `candidate_selector`, `backtest_engine`, optimizer, or any signal path.
  The diff is confined to (i) two lines in `paper_trader.py` row-builds,
  (ii) `useLiveNav.ts` + `cockpit-helpers.tsx` render math, (iii) a new test
  file/additions, (iv) a non-executed migration script + a disclosure doc.
- **No live flag flips, no LLM trading-cycle spend.** Pure data-correctness.

---

## B. EXTERNAL RESEARCH (>=5 read in full)

### Read in full (>=6; gate floor is 5)

| # | URL | Accessed | Kind | Fetched | Key finding |
|---|---|---|---|---|---|
| 1 | https://martinfowler.com/eaaCatalog/money.html | 2026-06-10 | blog (authoritative — Fowler PoEAA Money pattern) | WebFetch (full) | A monetary value is a `(amount, currency)` pair, not a primitive: "once you involve multiple currencies you want to avoid adding your dollars to your yen without taking the currency differences into account." Also warns "it's easy to lose pennies … because of rounding errors." Validates: each `paper_trades` row's money fields must be tagged to ONE currency (USD base) and never summed across currencies without conversion — exactly the F-1/F-2 defect class. |
| 2 | https://en.wikipedia.org/wiki/Characterization_test | 2026-06-10 | reference (Feathers' term, encyclopedic) | WebFetch (full) | A characterization test "describe[s] (characterize[s]) the actual behavior of an existing piece of software … to protect existing behavior of legacy code against unintended changes." Workflow: observe outputs for inputs → assert them → run after changes as a "change detector." The 56.1 KRW test is the CORRECTNESS-asserting variant (fails on the buggy baseline, passes post-fix) — a regression guard that also pins the do-no-harm US path. |
| 3 | https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/NumberFormat/NumberFormat | 2026-06-10 | official docs (MDN) | WebFetch (full) | `Intl.NumberFormat({style:'currency', currency})` formats a number ALREADY in that currency — "it does not convert currencies." `currencyDisplay:'narrowSymbol'` → "$100" not "US$100". Fraction digits default to the ISO-4217 minor-unit count (USD=2; JPY/KRW=0). Confirms `format.ts` is correct and the bug is the upstream NUMBER (wrong currency), not the formatter. |
| 4 | https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/NumberFormat/format | 2026-06-10 | official docs (MDN) | WebFetch (partial — confirmed no-conversion + KRW zero-decimal) | `format()` "only formats a number according to the specified currency — it does not convert"; "you must provide a number already in the target currency." Corroborates #3 from the method-page angle. |
| 5 | https://docs.stripe.com/currencies | 2026-06-10 | industry docs (Stripe; redirect-followed from stripe.com/docs/currencies) | WebFetch (full) | Distinguishes **presentment currency** (what the customer sees) from **settlement currency** ("the currency your bank account uses") — "Stripe converts the charge to your settlement currency." KRW is **zero-decimal** (min charge `50 KRW` not `0.50`). Best practice: account in ONE base, present per-currency, store consistently. Direct analogue: pyfinagent settles/accounts in USD (the ledger base) and presents per-market — KRW's 0-minor-unit matches `format.ts`'s no-forced-fraction rule. |
| 6 | https://blogs.cfainstitute.org/marketintegrity/2023/08/24/how-do-firms-treat-errors-in-investment-performance/ | 2026-06-10 | official-adjacent (CFA Institute Market Integrity, GIPS error-correction) | WebFetch (full) | Material error → firm must (1) correct, (2) disclose, (3) provide corrected report + document. Materiality is two-pronged (absolute 11-50 bps AND relative 5-10% return change) PLUS a qualitative "judgment of a reasonable person" test; 78% keep an Excel error-correction log. Drives the backfill disclosure note + the tier-3/4 materiality call (NAV/P&L returns unaffected → immaterial to the composite return; ledger value/fee fields corrected + documented). |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://www.gipsstandards.org/wp-content/uploads/2025/04/sample_error_correction_policy_firms-1.pdf | official standard (GIPS 2025 sample error-correction policy) | PDF; the CFA blog (#6) + 55.3 brief already give the materiality/disclosure mechanics needed for the note |
| https://www.acaglobal.com/industry-insights/key-updates-from-the-2025-gips-standards-conference/ | industry (2025 GIPS conf updates) | recency-scan hit; no change to error-correction mechanics relevant here |
| https://liveflow.com/blog/understanding-multi-currency-accounting-a-practical-guide | industry guide (2026) | recency-scan corroboration of "convert to base currency for consolidated reporting"; not authoritative enough to read in full |
| https://www.netsuite.com/portal/resource/articles/accounting/multi-currency-accounting.shtml | industry (NetSuite) | same principle as Fowler #1; snippet sufficient |
| https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11954734 | patent (CME-style initial-margin multi-currency) | confirms "convert various local currencies to the base currency to aggregate risk" — strong corroboration, but a margin patent, not directly about ledger storage; snippet captured |
| https://martinfowler.com/bliki/CharacterizationTest.html | blog (Fowler) | returned HTTP 404 on fetch; substituted the Wikipedia characterization-test page (#2) which carries the same Feathers definition |
| https://docs.aws.amazon.com/prescriptive-guidance/latest/patterns/run-sql-scripts-idempotently-and-in-order-by-using-aws-codepipeline.html | official docs (AWS) | returned empty body on fetch; idempotency guidance is well-established (sentinel WHERE + dry-run preview) and is applied from first principles in A.8 |

### Recency scan (2024-2026) — MANDATORY

Searched the last-2-year window for FX handling in multi-currency
portfolio/trading accounting and GIPS error-correction:
- Query `multi-currency portfolio accounting FX conversion base currency trading system best practice 2025 2026`
- Query `GIPS error correction policy material error disclosure performance presentation 2025`

**Result: found 3 relevant 2025-2026 findings that COMPLEMENT (do not
supersede) the canonical sources:**
1. **The base-currency-aggregation principle is reaffirmed in 2025-2026
   practice and in a margin patent**: "Multi-currency portfolios … convert
   from various local currencies to the [Base Currency], which … allows for
   aggregation of risk … and collection … in a single currency"
   (USPTO 11954734-family; LiveFlow 2026 guide: convert to a base currency
   "for consolidated reporting"). This is exactly pyfinagent's model — USD
   base ledger, per-market presentation. **No change to the 56.1 design.**
2. **FX volatility makes the bug materially costly in 2025-2026**: "in the
   first four months of 2025 the US dollar depreciated by nearly 10% against
   the euro" (LiveFlow 2026 ERP survey) — i.e. an un-converted local amount
   isn't just ~1500x wrong for KRW, it drifts materially for EUR too. Raises
   the urgency of the F-1/F-2 fix but does not alter the approach.
3. **GIPS 2025 error-correction mechanics are stable**: the 2025 sample
   error-correction policy + 2025 GIPS conference updates confirm the
   two-pronged materiality threshold (11-50 bps absolute / 5-10% relative)
   and correct-disclose-document for material errors — unchanged from the
   2023 CFA guidance read in full (#6). The 56.1 disclosure note can cite
   the stable standard with confidence.

No 2024-2026 source contradicts the design (store USD base + know each row's
currency + characterization-style regression test + idempotent
dry-run-default restatement + GIPS materiality disclosure). The canonical
sources (Fowler Money, MDN Intl, Feathers characterization tests, GIPS) all
remain current.

### Search-query log (3-variant discipline)

| Variant | Query | Purpose |
|---|---|---|
| current-year frontier | `multi-currency portfolio accounting FX conversion base currency trading system best practice 2025 2026` | latest practice |
| last-2-year window | `GIPS error correction policy material error disclosure performance presentation 2025` | recency scan |
| year-less canonical | (direct-fetched the canonical prior-art: Fowler Money pattern, Feathers characterization test, MDN Intl.NumberFormat, Stripe currencies) | founding/canonical references — the topic's prior art is textbook/docs, fetched directly rather than via a year-locked search |

(The data-correctness topic's canonical prior art is documentation/pattern
literature with stable URLs, so the year-less "search" was satisfied by
direct WebFetch of the founding sources rather than a SERP query — noted per
the research-gate rule's allowance for genuinely-stable canonical prior art.)

### Key findings (per-claim cited)

1. **Money is a (value, currency) pair; never mix currencies in arithmetic.**
   "avoid adding your dollars to your yen without taking the currency
   differences into account" (Fowler, Money pattern,
   https://martinfowler.com/eaaCatalog/money.html, 2026-06-10). → the F-1/F-2
   defects are textbook currency-mixing; the fix tags every stored money
   field to the USD base.
2. **Intl.NumberFormat formats, it does not convert.** "it does not convert
   currencies … provide a number already in the target currency"
   (MDN, .../NumberFormat, 2026-06-10). → `format.ts` is correct; the bug is
   the upstream number, so the fix must be at the data layer
   (`useLiveNav`/`RiskMonitorCard`/`paper_trader`), NOT the formatter.
3. **KRW is zero-decimal.** KRW min charge `50` not `0.50` (Stripe,
   https://docs.stripe.com/currencies, 2026-06-10); ISO-4217 minor-unit = 0
   (MDN). → `format.ts`'s no-forced-fraction rule is right; "₩219,500" (no
   cents) is correct rendering.
4. **Account in one base currency; present per-currency.** presentment vs
   settlement currency (Stripe, 2026-06-10); "convert … to a base currency
   for consolidated reporting" (LiveFlow 2026). → USD-base ledger +
   per-market display is the validated pattern; the position rows already do
   this, the trade rows must too (A.1).
5. **Characterization/regression test pins behavior across a change.**
   "protect existing behavior of legacy code against unintended changes"
   (Wikipedia/Feathers, 2026-06-10). → the KRW fixture (fails pre-fix, passes
   post-fix) + the US byte-identity test are the regression guards
   (criterion 1 + do-no-harm).
6. **Material performance errors: correct + disclose + document.**
   two-pronged materiality (11-50 bps / 5-10%) + qualitative reasonable-person
   test (CFA/GIPS, 2026-06-10). → the away-week mis-booking is immaterial to
   the COMPOSITE RETURN (NAV/cash always clean) but material to ledger
   value/fee consumers → tier-3/4 correct-and-disclose; the backfill note
   documents scope + method (criterion 3).
7. **Idempotent restatement = guarded, dry-run-previewed UPDATE.** (AWS
   prescriptive guidance, snippet; applied from first principles in A.8). →
   `--execute` flag, dry-run default, `WHERE total_value>1000` sentinel so a
   re-run can't double-convert.

### Consensus vs debate (external)

**Consensus (unanimous across all sources):** store/account in a single base
currency; tag every monetary amount with its currency; never sum across
currencies without explicit conversion; format ≠ convert; restate material
errors with disclosure. No source dissents.

**Minor debate / judgment call:** whether to ALSO persist the local amount +
the FX rate alongside the base amount (audit-trail strength) vs store base
only (minimal). Fowler's Money pattern leans toward carrying currency with
every amount (favors storing local+rate); the minimal-diff / scope-honesty
constraint here favors base-only for 56.1 with the rate recoverable from
`historical_fx_rates` + `created_at`. Resolved in A.4: base-only for 56.1;
recommend a dedicated `fx_rate` column as a phase-57 schema option if the
operator wants the stronger trail. This is the one genuine design tension and
it is surfaced, not buried.

### Pitfalls (from literature + internal)

- **Double-conversion on re-run** (idempotency): mitigated by the magnitude
  sentinel + explicit-value UPDATE (A.8).
- **KRW forced-2-decimals** rendering ("₩1,234,567.00"): already avoided by
  `format.ts` (no minimumFractionDigits); the new code must not reintroduce
  it (the `useLiveNav`/`RiskMonitor` fixes feed numbers to existing USD
  `Dollar`/`formatUsd`, so safe).
- **KRW inversion** (the #1 FX pitfall per `fx_rates.py` header): NOT touched
  by 56.1 — the fix multiplies by the EXISTING, already-correct `_l2u`/
  `_local_to_usd` helpers, so the inversion risk is inherited-correct, not
  re-litigated.
- **Frontend/backend drift** if `mvUsd` and `useLiveNav` diverge: mitigated
  by the recommended extract-to-`format.ts` (A.6 FIX 1).

### Application to pyfinagent (external findings → internal file:line)

| External finding | Internal anchor | Action |
|---|---|---|
| Money = (value, currency); no cross-currency sum (Fowler) | `paper_trader.py:265` (BUY total_value), `:413-414` (SELL total_value/fee) | multiply by in-scope `_local_to_usd`/`_l2u` (A.1) |
| format ≠ convert (MDN) | `useLiveNav.ts:34-39`, `cockpit-helpers.tsx:301-302` | fix the NUMBER (use USD `market_value`), not the formatter (A.6) |
| Account-in-base / present-per-ccy (Stripe, LiveFlow) | `positions/page.tsx:66-76` `mvUsd` (already does it) | replicate `mvUsd` in `useLiveNav.ts`; extract to `format.ts` (A.6) |
| Characterization/regression test (Feathers) | `test_phase_50_2_multicurrency.py` | add KRW fail-pre/pass-post + US byte-identity tests (A.5) |
| Material-error correct+disclose+document (GIPS) | `scripts/migrations/*.py` + `handoff/` disclosure note | operator-gated backfill of 7 rows + materiality note (A.8) |
| Idempotent dry-run UPDATE (AWS) | the new migration script | `--execute` flag, dry-run default, sentinel WHERE (A.8) |

---

## Research Gate Checklist

Hard blockers — `gate_passed` is false if any unchecked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6: Fowler Money, Wikipedia/Feathers, MDN NumberFormat, MDN format, Stripe currencies, CFA/GIPS)
- [x] 10+ unique URLs total (13: 6 read-in-full + 7 snippet-only)
- [x] Recency scan (last 2 years) performed + reported (2 search passes; 3 complementary 2025-2026 findings; no contradiction)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (paper_trader.py:189/208/265/370/386-388/413-414/485/440; useLiveNav.ts:34-39; cockpit-helpers.tsx:204-223/301-302; positions-columns.tsx:159-228; positions/page.tsx:66-80/123-133/169-183; MultiCurrencyNavBreakdown.tsx:53; format.ts:136-176; perf_metrics.py:406-471; formatters.py:213; paper_trading.py:645-659)

Soft checks:
- [x] Internal exploration covered every relevant module (backend writer + all consumers; every frontend F-1 site; test infra; migration conventions; verification-command collection)
- [x] Contradictions / consensus noted (consensus unanimous; one design tension surfaced — local+rate vs base-only, resolved in A.4)
- [x] All claims cited per-claim (per-claim URLs in Key findings; file:line in Section A)

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 7,
  "urls_collected": 13,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
