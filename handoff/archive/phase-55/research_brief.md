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

*(Brief continues: A.6 do-no-harm, A.3-frontend F-1 fix design, F-12 VS-KOSPI,
migration sketch, then Part B external sources + recency + envelope.)*
