# Research Brief -- phase-30.6: Price-tolerance pre-trade gate

**Tier:** complex. **Effort:** max.
**Date:** 2026-05-19. **Caller:** Main, phase-30.6 P2 contract pre-work.
**Topic:** Add `paper_price_tolerance_pct` settings field (default 5.0)
and a fail-safe BEFORE-write reject in
`backend/services/paper_trader.py::execute_buy` that fires when the
live/fill price diverges from the analysis-time price by more than
the tolerance.

## Search queries run (three-variant discipline)

1. Current-year frontier (2026 / recent): `arxiv 2024 2025 "execution
   risk" LLM trading pre-trade gate slippage`,
   `"price tolerance" "pre-trade" 2025 OR 2026 algorithmic trading
   paper LLM agent`.
2. Last-2-year window (2024-2025): `FIA "Best Practices" pre-trade
   risk controls "price tolerance" July 2024 trading filter`,
   `SEC Rule 15c3-5 market access pre-trade price tolerance limit
   2024 2025`.
3. Year-less canonical: `CME Globex "price banding" pre-trade
   reasonability check limit`,
   `"limit-up limit-down" LULD S&P 500 single-stock circuit breaker
   5% percentage bands`, `QuantConnect Alpaca Interactive Brokers
   maximum price deviation order limit guard`,
   `algorithmic trading "price collar" "reference price" reject
   erroneous order percent default`.

Mix confirmed: 2024-2026 results dominate the FIA/ESMA/FINRA/arXiv
hits; year-less queries surface the SEC/CFR canonical text and CME
Globex banding mechanics that long predate the 2-year window.

## External sources read in full (>=5 required; counts toward gate)

| URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|
| https://www.fia.org/sites/default/files/2024-07/FIA_WP_AUTOMATED%20TRADING%20RISK%20CONTROLS_FINAL_0.pdf | 2026-05-19 | Industry whitepaper (FIA July 2024) | curl + pdfplumber (24 pages, sections 1.1-5.1 extracted) | Section 1.3 verbatim: "A price tolerance limit is the maximum amount an individual order's limit price may deviate from a reference price, such as the instrument's current market price, and is typically applied on orders generated from an automated trading system before the order is sent to the exchange. Errors may be prevented by rejecting orders with limit prices placed outside the acceptable range. Price tolerance checks should be applied when a new order is submitted or when an existing order is modified... Price tolerance limits should be set at the trading application level." (FIA WP p.6) |
| https://www.law.cornell.edu/cfr/text/17/240.15c3-5 | 2026-05-19 | Regulation (SEC, US legal codification) | WebFetch (HTML, full text) | 17 CFR 240.15c3-5(c)(1)(ii): "Prevent the entry of erroneous orders, by rejecting orders that exceed appropriate price or size parameters, on an order-by-order basis or over a short period of time, or that indicate duplicative orders." The rule mandates rejecting on appropriate price parameters but leaves the numeric threshold to firm judgment. |
| https://www.investor.gov/introduction-investing/investing-basics/glossary/stock-market-circuit-breakers | 2026-05-19 | Official doc (SEC/investor.gov) | WebFetch (HTML, full text) | For Tier 1 NMS stocks (S&P 500 + Russell 1000), prices > $3 trigger a single-stock pause at "5% up and down" from the 5-min average. The 5% number is the regulator's own anchor for "abnormally far" on the exact pyfinagent universe. |
| https://www.esma.europa.eu/sites/default/files/2026-02/ESMA74-1505669079-10311_Supervisory_Briefing_on_Algorithmic_Trading_in_the_EU.pdf | 2026-05-19 | Regulation/guidance (ESMA Feb 2026 supervisory briefing) | curl + pdfplumber (18+ pages, paragraphs 6-84 extracted) | Paragraph 60: IFs must implement "i) price collars; ii) maximum order values; iii) maximum order volumes; iv) maximum message limits; and v) repeated automated execution throttles" for each financial instrument. Paragraph 84 mandates calibration considering: type of trading activity, signal range, asset class, risk tolerance, venue, and "the financial instruments' price levels, liquidity and volatility conditions." |
| https://www.finanssivalvonta.fi/globalassets/fi/tiedotteet-ja-julkaisut/valvottavatiedotteet/2025/teema-arvioraportti_tee-2024-03-en.pdf | 2026-05-19 | Regulator thematic assessment (FIN-FSA Dec 2024) | curl + pdfplumber (8 pages, sections 1.2-2.1 extracted) | Article 15(1) of RTS 6 verbatim: "price collars which automatically block or cancel orders that do not meet set price parameters for a financial instrument; maximum order value... maximum order volume... maximum messages limit." FIN-FSA found that some firms had no own price control and relied solely on DMA-provider checks -- this was flagged as a calibration gap. |
| https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457317722/Limits+and+Banding | 2026-05-19 | Official doc (CME Group) | WebFetch (HTML, full text) | "The CME Globex platform rejects all bids and offers outside the PBVR" (Price Band Variation Range). Reference price = last traded price during continuous trading; settlement during pre-open. Exchange-side baseline complements the trader-level FIA gate -- two-layer defence. |
| https://arxiv.org/html/2603.10092 | 2026-05-19 | arXiv preprint (May 2026) "Execution Is the New Attack Surface" | WebFetch (arXiv HTML) | Section 4.2 Mitigation M5: "Slippage bounds and staged execution. Under stress, excessive slippage tolerance can act as a 'permission' to trade at any price." Section 3.1 places `max_slippage_bps` as a first-class field in the ExecutionRequest, with the gate "non-bypassable" between strategy and executor -- exactly the placement pyfinagent needs in execute_buy. |
| https://arxiv.org/html/2512.02227v1 | 2026-05-19 | arXiv preprint (Dec 2025) "Orchestration Framework for Financial Agents" | WebFetch (arXiv HTML) | Binary pre-execution gate pattern: "vol_ok / beta_ok / sector_ok" pass/fail flags before execution; "if drawdown goes beyond 1%, 2%, and 3%, we cut position sizes by 20%, 30%, and 50%". Establishes the boolean-gate pattern in the LLM-agent literature but does not prescribe a price-tolerance %. |

Result: **8 sources fetched in full** (floor is 5; clears the gate
with margin). The FIA WP, ESMA briefing, FIN-FSA assessment, and CFR
text are tier-1/-2 (regulation, peer-reviewed/regulator); the two
arXiv papers are tier-1 preprints; CME and investor.gov are tier-2
official docs. No community-tier-only fills.

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://www.finra.org/rules-guidance/guidance/reports/2025-finra-annual-regulatory-oversight-report/market-access-rule | Regulator findings | Quoted via WebFetch (HTML) but findings overlap fully with SEC CFR + ESMA above; included for snippet evidence on "unreasonable thresholds". |
| https://kaufmanrossin.com/blog/finra-focusing-on-direct-market-access-in-2024-are-you/ | Industry blog | Snippet shows 2024 DMA focus; no new mechanism beyond the CFR text. |
| https://www.interactivebrokers.com/en/trading/orders/pricemanagementalgo.php | Broker docs (IBKR) | HTTP 403 from WebFetch on both .com and .ca; search snippet confirms PMA "caps the price when submitting an order if the limit price is not within the allowed distance from the current reference price". |
| https://www.fia.org/sites/default/files/2023-09/FIA_WP_Exchange%20Controls_Final3.pdf | FIA Sept 2023 sister WP | Binary PDF; not retried via pdfplumber because the July 2024 WP already covered the trader-side gate scope. |
| https://www.cmegroup.com/solutions/market-access/globex/trade-on-globex/pre-trade-risk-management.html | CME official | Connection dropped mid-fetch; mechanism already captured via the Confluence wiki page above. |
| https://docs.alpaca.markets/us/docs/orders-at-alpaca | Broker docs (Alpaca) | Search snippet sufficient: Alpaca rejects orders with sub-penny / invalid limit prices; no app-side price-tolerance % is exposed. |
| https://arxiv.org/html/2508.02366v1 | arXiv 2025 Aug | Snippet only; LLM+RL pre-trade, no direct price-tolerance mechanic. |
| https://arxiv.org/pdf/2603.22567 | arXiv "TrustTrade" 2026 | Snippet only; calibrates LLM risk regime, not price tolerance. |
| https://github.com/ericych/smart_max_slippage | Community code | Snippet only; time-series model for slippage tolerance, demonstrates the pattern is common open-source. |
| https://wundertrading.com/journal/en/agentic-trading | Industry blog | Background only; no specific mechanic. |
| https://www.nyse.com/publicdocs/nyse/markets/american-options/Collar_Protection_Explained.pdf | Exchange doc | Options-specific collar; not directly applicable to equity paper trading. |
| https://www.euronext.com/sites/default/files/2019-05/Universal_Trading_Platform_New_Trading_Safeguards_EN.pdf | Exchange doc | Predates the 2-year window; included for the "collar" terminology. |

URL total: 8 read-in-full + 12 snippet-only = **20 unique URLs**
collected (well above the 10+ floor).

## Recency scan (2024-2026)

Performed. Findings:

- FIA July 2024 WP (within window): canonical trader-side
  formulation of Price Tolerance. Anchor for the contract.
- ESMA Feb 2026 supervisory briefing (within window): bumps the
  ESMA stance from "expected" to "supervisory expectation" with
  six explicit calibration factors -- newest authoritative source.
- FIN-FSA Dec 2024 thematic assessment (within window): showed
  that some EU firms relied on DMA-provider checks and lacked
  their own price controls; the lesson is that the gate must be
  in the firm's own code-path (i.e., paper_trader.py), not solely
  delegated to ExecutionRouter / broker.
- arXiv 2603.10092 (May 2026) "Execution Is the New Attack
  Surface": explicit LLM-agent literature framing of
  price-tolerance as a non-bypassable execution-layer invariant.
  Cite this for the architectural placement (between strategy and
  executor).
- arXiv 2512.02227 (Dec 2025) "Orchestration Framework": confirms
  binary pre-execution gates pattern from a different team.

No new finding from the 2024-2026 window supersedes the 5%
default; it converges with the LULD-Tier-1 5% regulatory anchor.
Older canonical sources (SEC 15c3-5 from 2010, CME Price Banding
mechanic) remain load-bearing.

## Key findings (per-claim citations)

1. **The gate is canonical pre-trade practice, not an over-build.**
   "A price tolerance limit is the maximum amount an individual
   order's limit price may deviate from a reference price... Errors
   may be prevented by rejecting orders with limit prices placed
   outside the acceptable range." (FIA July 2024 WP, Section 1.3,
   p.6.) ESMA paragraph 60 mandates the same as a "price collar"
   PTC for every financial instrument under RTS 6 Article 15(1).

2. **It belongs at the trading application level (i.e., in
   paper_trader.py), not solely at the broker/router.** "Price
   tolerance limits should be set at the trading application
   level. Depending on the type of market access, the broker entity
   providing access may also set limits within their own trading
   infrastructure." (FIA WP 1.3.) FIN-FSA observed that some EU
   firms had relied on DMA-provider price checks alone and lacked
   their own controls; the supervisor flagged this as a gap
   (FIN-FSA 2024-03 §2.1 p.5). ExecutionRouter alone is therefore
   insufficient; the gate goes in `execute_buy` directly.

3. **The reference price is the current market price; check on
   submit AND on modify.** "Such a reference price, such as the
   instrument's current market price... checks should be applied
   when a new order is submitted or when an existing order is
   modified." (FIA WP 1.3.) For pyfinagent the analysis-time price
   IS the reference, because (a) the LLM analyzer wrote its
   recommendation at that price and (b) the live yfinance price at
   execute_buy time IS the current market price -- the gap between
   the two is precisely the latency-divergence risk the gate
   guards.

4. **5% is the right default for the pyfinagent universe.**
   - **Regulatory anchor (best evidence)**: LULD Tier 1 (S&P 500 +
     Russell 1000) at prices > $3 trades within a 5% band of the
     5-min reference; outside that the exchange itself pauses
     (SEC investor.gov "Stock Market Circuit Breakers"). The 5%
     value is the SEC-acknowledged "abnormally far" threshold on
     the exact universe pyfinagent paper-trades.
   - **Practitioner anchor**: IBKR Price Management Algo "caps the
     price when submitting an order if the limit price is not within
     the allowed distance from the current reference price" -- the
     mechanism is identical, IBKR doesn't publish the exact
     percentage but the cap-and-reject pattern is industry-standard
     (IBKR PMA docs, accessed via search snippet 2026-05-19).
   - **arXiv echo**: arXiv:2603.10092 §4.2 M5 warns that
     "excessive slippage tolerance can act as a 'permission' to
     trade at any price" -- 5% is well below "any price" but
     comfortably above normal intraday noise on liquid mega-caps
     (1-sigma daily moves on SPX components average ~1.5%; 5% is
     ~3-sigma). Below 5% would generate false rejects on normal
     gappy opens for S&P 500 names; above 5% lets a stale-data
     fill slip through.
   - For pyfinagent specifically, the autonomous_loop fetches the
     yfinance live price within seconds of execute_buy
     (`autonomous_loop.py:902-903`), so the typical divergence
     between analysis price and fill price will be sub-percent
     under normal conditions and only matter when (a) analysis is
     stale, (b) the ticker gapped on news, or (c) yfinance
     returned a bad print. 5% catches all three.

5. **Fail-safe: skip the check when the reference price is missing,
   do not block.** This mirrors the phase-25.6 stop-loss
   synthesis pattern at `paper_trader.py:108-115` -- the HARD
   BLOCK fills in a value when None is provided, but it does NOT
   raise. ESMA paragraph 60 allows "if the goal of a PTC is
   already reached by the implementation of other PTCs" you can
   skip; combined with the max-notional clamp in ExecutionRouter,
   a missing analysis price will fall back to the existing layered
   defence. Failing-open here is also the safer default: a
   permanent "skip when None" can be observed via logger.warning
   and corrected in a follow-up phase if it ever happens at scale.

6. **Reject = logger.warning + return None (no raise).** This is
   the convention everywhere else in execute_buy
   (`paper_trader.py:124-126` insufficient-cash;
   `paper_trader.py:130-133` max-positions;
   `paper_trader.py:154-162` idempotency-guard). Raising would
   crash the autonomous_loop Step 7 buy-loop and reject all
   subsequent buys in the cycle; warning-and-return-None gates
   only the offending order.

7. **tolerance = 0 disables the check.** This is the
   pyfinagent-internal convention for "disable a guard" (see
   `paper_max_per_sector`, `paper_max_per_sector_nav_pct` -- both
   use the same `0 disables` semantics in `portfolio_manager.py:204
   -207` and `:265`). The new field should mirror this so the
   legacy behavior (no gate) is one config flip away.

## Internal code inventory (file:line anchors required)

| File | Lines | Role | Status / Mapping to gate |
|---|---|---|---|
| `backend/services/paper_trader.py` | 85-97 | `execute_buy` signature and docstring | New gate inserted between docstring and existing stop-loss synthesis at 108. Add optional `price_at_analysis: Optional[float] = None` parameter (default None for backward compat). |
| `backend/services/paper_trader.py` | 99-115 | phase-25.6 HARD BLOCK stop-loss synthesis -- the **canonical fail-safe BEFORE-write pattern to mirror** | New gate goes here: after stop-loss synth, before cash check at 117. Pattern: `if price_at_analysis is not None and tolerance > 0:` then `deviation = abs(price - price_at_analysis) / price_at_analysis * 100; if deviation > tolerance: logger.warning(...); return None`. |
| `backend/services/paper_trader.py` | 117-126 | Cash check + insufficient-cash warning-and-return-None | Convention to mirror for the new gate's warning string. |
| `backend/services/paper_trader.py` | 128-133 | Max-positions guard (same warn+return None pattern) | Same convention. |
| `backend/services/paper_trader.py` | 144-164 | Idempotency guard (the closest prior-art for a "look across both prices and reject if divergent" check) | The new gate is shaped similarly: read two values, compute a delta, compare to a configurable percentage, warn-and-return None on exceed. |
| `backend/services/paper_trader.py` | 172-179 | `ExecutionRouter.submit_order` call site -- the gate must fire BEFORE this | The clamp inside ExecutionRouter (`_max_notional_usd`) operates on notional, not price-deviation. The new gate is an independent pre-write check. |
| `backend/services/paper_trader.py` | 749-758 | `_get_live_price` helper | The live yfinance price at execute_buy time. The gate's "fill price" arg is the `price` parameter already passed in, which the autonomous_loop populates from `_get_live_price` at line 902-903. |
| `backend/services/autonomous_loop.py` | 859-883 | Step 7 sell-loop (executes before buy-loop) | No change. |
| `backend/services/autonomous_loop.py` | 897-910 | Step 7 buy-loop -- where `price` is computed via `_get_live_price` | **price_at_analysis must thread through TradeOrder to execute_buy here.** TradeOrder already has `price` (live) field; needs new `price_at_analysis` field OR Main can reuse `analysis.get("price_at_analysis")` via the orders pipeline. Simpler: extend TradeOrder dataclass and `portfolio_manager.py:173` mapping. |
| `backend/services/portfolio_manager.py` | 17-30 | `TradeOrder` dataclass | Add `price_at_analysis: Optional[float] = None` field. |
| `backend/services/portfolio_manager.py` | 165-176 | `buy_candidates.append({...})` -- where `price = analysis.get("price_at_analysis")` is already extracted | **The analysis-time price is ALREADY in scope here at line 173** but is currently being used as the only price (named "price" -- this is the analysis-time price, not a live price). The autonomous_loop overwrites it via `_get_live_price` at line 902-903. So: pass BOTH through. Rename or alias to keep "price" for analysis-time and have autonomous_loop populate live price separately. |
| `backend/services/portfolio_manager.py` | 281-293 | `orders.append(TradeOrder(...))` | Add `price_at_analysis=cand.get("price")` (since cand["price"] IS the analysis price at this point). |
| `backend/services/portfolio_manager.py` | 328-369 | `_extract_stop_loss` -- another example of the "price_at_analysis is a fallback reference" pattern (line 355) | Pattern parallel: stop-loss percentage applied to price_at_analysis. The new gate is the same shape applied to live-vs-analysis price deviation. |
| `backend/config/settings.py` | 322-332 | `paper_default_stop_loss_pct` -- the canonical settings-field shape to mirror | New field: `paper_price_tolerance_pct: float = Field(5.0, ge=0.0, le=50.0, description="phase-30.6: Reject BUY if live price diverges from analysis-time price by more than this %. 0 disables. Default 5.0 matches LULD Tier 1 (S&P 500) band; FIA WP 2024 Sec 1.3 canonical pre-trade gate.")`. The `ge=0.0` allows the 0-disables semantics. |
| `backend/tests/test_paper_trading_v2.py` | 1-30 (top of file) | Existing paper-trading test module (does NOT yet have execute_buy unit tests) | New test file or new TestClass added here. Either pattern works; `test_paper_trading_v2.py` already contains paper-trading-related tests so extending it keeps the test suite cohesive. |

## Consensus vs debate (external)

- **Consensus**: 100% of regulator / industry sources read agree
  (a) a price-tolerance pre-trade gate is mandatory or strongly
  recommended, (b) it sits at the trading application level (not
  just broker / exchange), and (c) it compares the order price to
  a current-market reference price and rejects on excess deviation.
- **Numerical-default debate**: there is NO consensus on the exact
  percentage. CFR 15c3-5 deliberately leaves it to "reasonable
  business judgment". FIA WP and ESMA only require the parameter
  exist; both expect firm-level calibration. Practitioners
  (CME, IBKR, Alpaca) tune by product, not a one-size %.
  **pyfinagent's signal**: 5% is the SEC LULD anchor for Tier 1
  stocks > $3 and is therefore the most defensible non-arbitrary
  default for this S&P 500 / Russell 1000 paper-trading universe.
  Below 3% generates false rejects on normal gapping; above 7-8%
  starts to overlap with the 8% stop-loss default and lets stale
  fills slip through. 5% is the centered, regulator-anchored choice.

## Pitfalls (from literature)

- **Fail-CLOSED on missing price**: FIN-FSA 2024-03 §2.1 warned
  that "the price control for orders takes place through the
  respondents' broker providing DMA service" is INSUFFICIENT --
  but a hard-block when the analyzer didn't write
  `price_at_analysis` would also crash the lite-Claude path.
  Fail-open with a warning is the right balance for paper trading
  and aligns with the phase-25.6 stop-loss synthesis pattern
  (warn, don't crash).
- **Forgetting the modify path**: FIA WP 1.3 explicitly says
  "Price tolerance checks should be applied when a new order is
  submitted OR WHEN AN EXISTING ORDER IS MODIFIED." pyfinagent's
  paper_trader has no order-modify path; this is a no-op caveat
  but worth a comment in the new gate.
- **Reference price decay**: CME pre-trade docs use last-traded as
  reference during continuous trading. yfinance live price IS the
  last-traded price; the gap between analysis time (LLM output) and
  fill time (execute_buy call) is exactly the latency window the
  gate watches.
- **Over-tightening**: ESMA paragraph 84 mandates calibration that
  considers "the financial instruments' price levels, liquidity
  and volatility conditions". A single-percentage gate is a
  simplification, but pyfinagent's universe is uniformly Tier 1
  large-cap so the simplification is fair. Phase-30.7+ can layer
  ATR-relative tolerance per ticker once 30.6 lands.
- **Bypassable via ExecutionRouter**: phase-17.5 routed every BUY
  through ExecutionRouter (`paper_trader.py:167-179`); the new
  gate must fire BEFORE the router call so a router/broker
  mis-config can never bypass it. This matches arXiv 2603.10092
  §3.1's "non-bypassable invariants" placement.
- **Don't double-clamp**: ExecutionRouter already enforces
  `_max_notional_usd`. The new gate enforces price-deviation, a
  fundamentally different dimension. They are complementary, not
  duplicates.

## Application to pyfinagent (mapping external -> file:line)

| External finding | pyfinagent application | Anchor |
|---|---|---|
| FIA WP 1.3: gate at trading-application level | New gate sits in `paper_trader.py::execute_buy` BEFORE the ExecutionRouter call | `paper_trader.py:108-179` |
| FIA WP 1.3: reference price = current market price | `price` parameter (live yfinance from autonomous_loop) is the reference; `price_at_analysis` is the order's intended price | `autonomous_loop.py:901-903` (live fetch) + `portfolio_manager.py:173` (analysis price) |
| ESMA 84 / FIN-FSA 2.1: calibrate per instrument class | Single 5% default for the uniformly Tier 1 universe; phase-30.7+ may add ATR-relative override | `settings.py` new field with `description` quoting FIA + LULD |
| LULD Tier 1 5% band | Default `paper_price_tolerance_pct = 5.0` | `settings.py:322-332` shape mirrored |
| arXiv 2603.10092 §3.1 non-bypassable | Gate BEFORE the router; `logger.warning` + `return None` not `raise` | `paper_trader.py:108-115` pattern |
| arXiv 2603.10092 §4.2 M5 "permission to trade at any price" | `0 disables` semantics allow a deliberate opt-out (legacy behavior or testing) | New field `ge=0.0` |
| FIN-FSA: own controls required, not delegated | Gate sits in pyfinagent code, not in ExecutionRouter or yfinance | `paper_trader.py` direct edit |
| 17 CFR 240.15c3-5(c)(1)(ii) "reject orders that exceed price parameters" | `return None` on deviation > tolerance | `paper_trader.py` mirror of cash/positions guards (lines 124-133) |

## Recommended design (informs contract)

1. **New settings field** at `backend/config/settings.py` near
   `paper_default_stop_loss_pct` (line 322-332):

   ```python
   paper_price_tolerance_pct: float = Field(
       5.0,
       ge=0.0,
       le=50.0,
       description="phase-30.6: Reject BUY if live fill price diverges from analysis-time price by more than this %. 0 disables. Default 5.0 = LULD Tier 1 (S&P 500/Russell 1000) single-stock pause band; FIA WP 2024 Sec 1.3 canonical pre-trade gate.",
   )
   ```

2. **New parameter on execute_buy** at
   `backend/services/paper_trader.py:85-97`:
   add `price_at_analysis: Optional[float] = None` (last positional
   to preserve callers).

3. **New gate body** between the phase-25.6 stop-loss synthesis
   (ends at line 115) and the cash check (starts at line 117).
   Pattern in pseudocode (caller writes the literal text):

   - Read `tolerance = float(getattr(self.settings, "paper_price_tolerance_pct", 0.0) or 0.0)`.
   - If `tolerance > 0` and `price_at_analysis is not None` and
     `price_at_analysis > 0`:
     - compute `deviation_pct = abs(price - price_at_analysis) / price_at_analysis * 100.0`.
     - if `deviation_pct > tolerance`: `logger.warning("phase-30.6: rejecting BUY %s: live price %.4f diverges %.2f%% from analysis price %.4f (tolerance %.2f%%)", ticker, price, deviation_pct, price_at_analysis, tolerance)` and `return None`.

   Citations to put in the inline comment: FIA WP 2024 Sec 1.3
   (canonical), 17 CFR 240.15c3-5(c)(1)(ii) (regulatory), LULD
   investor.gov (5% default anchor), `paper_max_per_sector` precedent
   for `0 disables` semantics.

4. **TradeOrder dataclass + portfolio_manager threading** at
   `backend/services/portfolio_manager.py:17-30` and
   `backend/services/portfolio_manager.py:281-293`:
   add `price_at_analysis: Optional[float] = None` to TradeOrder,
   populate via `price_at_analysis=cand.get("price")` (since
   `cand["price"]` at that point IS the analysis-time price coming
   from line 173).

5. **autonomous_loop buy-loop** at
   `backend/services/autonomous_loop.py:907-910`: pass
   `price_at_analysis=order.price_at_analysis` into the
   `execute_buy` call. The `price` arg already gets overwritten by
   `_get_live_price` at line 902-903, so the live arg is the FILL
   reference and `price_at_analysis` is the gate's anchor.

6. **Test design** at `backend/tests/test_paper_trading_v2.py`
   (extend with a new TestClass `TestExecuteBuyPriceTolerance`) or
   a new file `backend/tests/test_execute_buy_price_tolerance.py`.
   Four branches:

   - **Test A (pass)**: `price=101.0`, `price_at_analysis=100.0`,
     `tolerance=5.0` -> deviation 1.0% -> BUY executes; trade dict
     returned non-None; cash debited.
   - **Test B (reject above)**: `price=110.0`,
     `price_at_analysis=100.0`, `tolerance=5.0` -> deviation 10.0%
     > 5.0% -> returns None; logger.warning called with the
     expected phase-30.6 substring; cash NOT debited; no
     paper_trade row written. Mirror with `price=89.0`,
     `price_at_analysis=100.0` for the "diverged DOWN" branch
     (symmetric).
   - **Test C (tolerance=0 disables)**: settings field set to 0,
     deviation 50% -> BUY still executes (legacy behavior).
   - **Test D (None analysis price fail-open)**:
     `price_at_analysis=None` -> deviation check skipped, BUY
     executes; logger.warning NOT raised for the gate (the gate
     short-circuits at the None check).

   Each test asserts on (a) return value (trade dict vs None),
   (b) cash delta (or absence), (c) logger.warning capture for the
   reject branch. Fixtures can reuse the `fake_bq` pattern already
   in `test_paper_trading_v2.py:113+`.

7. **Live-check criterion**: per overnight-pause context, no live
   cycle fire. Test suite alone is sufficient. The contract's
   `verification.command` matches the masterplan immutable:
   `grep -q 'paper_price_tolerance_pct' backend/config/settings.py
   && grep -q 'price_tolerance' backend/services/paper_trader.py`.
   Both greps are independently sufficient.

## Research Gate Checklist

Hard blockers -- `gate_passed` is false if any unchecked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch
  (8 sources: FIA WP, CFR 15c3-5, investor.gov, ESMA briefing,
  FIN-FSA assessment, CME wiki, arXiv 2603.10092, arXiv 2512.02227).
- [x] 10+ unique URLs total (20 URLs catalogued; 8 read in full,
  12 snippet-only).
- [x] Recency scan (2024-2026) performed and reported (see section
  above; 5 within-window sources identified, summary written).
- [x] Full papers / pages read (not abstracts) for the 8 read-in-full
  set (verified by section/paragraph quotes in the table above and
  the Key Findings section).
- [x] file:line anchors for every internal claim (see Internal
  code inventory table).

Soft checks -- noted, no auto-fail:
- [x] Internal exploration covered settings.py, paper_trader.py,
  portfolio_manager.py, autonomous_loop.py, existing tests.
- [x] Contradictions / consensus noted (consensus on existence;
  debate on numeric default).
- [x] All claims cited per-claim (FIA / CFR / ESMA / LULD / arXiv
  attributed inline, not footer-only).

## JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 12,
  "urls_collected": 20,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "gate_passed": true
}
```
