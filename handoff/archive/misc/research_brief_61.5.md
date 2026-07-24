# Research Brief — phase-61.5 (cost-aware turnover policy, config-gated OFF)

**Status: COMPLETE — gate_passed: true (8 sources in full / ~55 URLs / recency scan / 9 internal files).**
**Tier: complex. Date: 2026-07-08. Author: researcher (Layer-3).**
**HEADLINE: FINRA TAF rate in the immutable criterion is STALE ($0.000166/$8.30 expired 2025-12-31; current = $0.000195/share cap $9.79, and it steps annually through 2029). SEC 0.00206%, KR 0.20%, EU 0.05%/EUR 1.25 all verified CURRENT. Churn-measurement window is functionally gated on 66.2 (portfolio ~100% cash; 0 post-flag sells).**

## Scope (from caller)

Pre-pay brief for the future 61.5 contract: per-market transaction-cost table
(US SEC Section 31 + FINRA TAF; KR STT; EU venue fees; optional half-spread bps;
minimum-ticket floor); >=5-day churn measurement vs the 61.1 audited baseline;
55.3-sanctioned minimum-holding-period lever ONLY if churn persists >30%;
per-market score-to-forward-return slope monitor. HYSTERESIS FAMILY BANNED
absent `HYSTERESIS: AUTHORIZE` — no score-band / no-trade-band designs in any
renamed form.

## Sections (to be filled)

1. Verified fee table (per-rate citations; drift flags vs immutable criteria)
2. Internal audit: cost-model plug-in points (paper_trader, portfolio_manager)
3. Internal audit: 61.1 churn-baseline artifacts + measurement queries
4. Internal audit: replay tooling for ON-vs-OFF restatement
5. Internal audit: forward-return data for slope monitor (join feasibility)
6. External: transaction-cost modeling literature
7. External: turnover-control levers (non-hysteresis)
8. External: IC / score-decay measurement practice
9. Recency scan (2024-2026)
10. Application to pyfinagent (design sketches)
11. Research Gate Checklist + JSON envelope

---

## 2-5. Internal code inventory (all anchors verified 2026-07-08, HEAD 07b7d9c3)

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/config/settings.py` | 337 | `paper_transaction_cost_pct: float = Field(0.1, ...)` — the flat 0.1%/side model 61.5 replaces (config-gated) | live, flat |
| `backend/services/paper_trader.py` | 189 | BUY fee: `tx_cost = amount_usd * (pct/100)` — `market` var in scope (persisted at :312/:333) | plug-in point A |
| `backend/services/paper_trader.py` | 388 | SELL fee: `tx_cost = sell_value * (pct/100)`; `sell_qty` (shares, for FINRA TAF) + `position.get("market")` (:371) + `_l2u` FX (:371) all in scope | plug-in point B |
| `backend/services/paper_trader.py` | 423, 445, 1144 | `holding_days` computed at sell (:399) and PERSISTED per SELL trade row + in BQ field list | churn query ready |
| `backend/services/paper_trader.py` | 1214 | `from backend.backtest.markets import market_for_symbol` — suffix→market mapping exists | reuse for fee dispatch |
| `backend/services/portfolio_manager.py` | ~495-575 | swap-delta: sentinel/exclusion (churn fix ON) + `denom = max(abs(holding_score), 1.0/0.01)` + `delta_pct` vs `min_delta` — **cost model does NOT plug here**; the 25.0 bar and delta math are 53.1/55.3-protected | do not touch |
| `backend/services/execution_router.py` | 85-126 | `_bq_sim_fill` — zero-slippage synthetic fill at last close; optional half-spread bps would apply to `fill_price` here OR as a fee line in paper_trader (recommend fee line: keeps fills deterministic) | plug-in point C (optional) |
| `backend/config/settings.py` | 305-311, 319 | `paper_swap_min_delta_pct=25.0` (untouchable), `paper_swap_max_per_cycle`, `paper_swap_churn_fix_enabled` doc (60.2 exclusion+clamp) | context |
| `scripts/replay/replay_60_2_swap_fix.py` | 1-50 | decision-level ON-vs-OFF replay pattern ($0: BQ reads + yfinance closes; output md in handoff/current/) | template for fee replay |
| `handoff/archive/phase-61.1/` | — | contract/results/critique/research_brief for 61.1 (Cycle 66 PASS 2026-06-15) | baseline provenance |
| `handoff/current/goal_phase61_churn_integrity.md` | 26-38, 82-86, 183-189 | fee mis-attribution ($17.14/8d vs -$139.83 churn), fee-model defect list, 61.5 token mechanics (`FEE TABLE:` / `TURNOVER LEVERS:` / optional `HYSTERESIS: AUTHORIZE`) | canonical rulings |

### (2) Where the cost model plugs in — WITHOUT touching gates

Fee charging is entirely inside `paper_trader.py` (:189 buy, :388 sell). A
`transaction_cost_model.py` (or settings-level per-market table) dispatched on
`market` replaces the two `tx_cost =` lines behind a
`paper_cost_model_enabled: bool = False` flag; OFF path byte-identical (flat
0.1%). Sell side has shares (`sell_qty`) for FINRA TAF ($0.000166/share, cap
$8.30, 1-cent round-up) and SEC Section 31 (notional bps, sells only); KR STT
0.20% sell-side; EU 0.05% min EUR 1.25 per order both sides (per-order minimum
needs LOCAL notional — available pre-`_l2u`). Minimum-ticket floor plugs into
`execute_buy` sizing (reject/skip below floor), config-gated. The swap gate
(`portfolio_manager.py:495-575`) is NOT a plug-in site — the 25% bar, denom
clamp, and exclusion logic are protected by the 53.1/55.3 rulings and the 60.2
"NOT a churn lever" doctrine (settings.py:319). The cost model changes what
trades COST, never whether they fire.

### (3) Churn baseline (61.1 / Cycle 66) + measurement queries

Baseline (immutable criteria text): 11/16 sells at <=2-day holds; 2 swap
pairs/day (06-03..06-10 fleet audit wf_7a3b2a6c-0da). 61.1 closed Cycle 66
(2026-06-15) with post-flag trades = **0** (cycle 5f15fdbe n_trades=0). The
measurement query is trivial because `holding_days` is persisted on every SELL
row (paper_trader.py:423): `SELECT COUNTIF(holding_days<=2)/COUNT(*) FROM
financial_reports.paper_trades WHERE action='SELL' AND created_at >= <flag
date>` (us-central1; ADC Python client — pinned MCP is US-locked, CLAUDE.md
BQ rule-6 fallback; created_at is STRING — use the 61.4 SAFE_CAST pattern).
**CRITICAL TIMING FINDING:** the portfolio is ~100% cash since 07-03 and had
zero BUYs 06-10..07-06 (credential outage; Cycle 67). ">=5 trading days of
post-61.1 churn measurement" is calendar-satisfied but VACUOUS (0 sells) until
66.2 redeploys capital and the loop trades again. The 61.5 contract must
define the window as >=5 trading days **containing sell activity** after
trading resumes, or explicitly invoke the Cycle-66 vacuous-pass doctrine
(disclose + interesting witness) if sells remain zero. A 0-sell window also
auto-satisfies the "<=30% => no-lever-needed" ELSE branch — Q/A will (rightly)
challenge that as vacuous; pre-register the interpretation in the contract.

### (4) Replay tooling for ON-vs-OFF restatement

`scripts/replay/replay_60_2_swap_fix.py` is the sanctioned pattern ($0,
decision-level, md output). The 61.5 fee replay is SIMPLER than 60.2's: no
decisions re-run — a pure ledger restatement. For each recorded trade in the
window: recompute `tx_cost` under the per-market table (market from the
persisted `market` column, shares from `quantity`, LOCAL notional from
`price*quantity`) vs the recorded flat 0.1%, then restate cumulative P&L.
CAVEAT: "last 30 days of recorded trades" — as of any plausible 61.5 start
date the trailing 30 calendar days contain ~0-few trades (freeze). Recommend
the contract interpret it as "the most recent 30 days that contain trades"
or the full 06-03..06-15 + post-reactivation set, disclosed to Q/A.

### (5) Forward-return data for the slope monitor

Scores: `financial_reports.analysis_results` carries per-(ticker, created_at)
`final_score` (0-10 full path / 1-10 lite; sentinel-0.0 and degraded rows must
be EXCLUDED — 61.2 adds the explicit degraded marker; until then filter
`final_score > 0`). Market: derive from ticker suffix via
`backend.backtest.markets.market_for_symbol` (paper_trader.py:1214 proves the
import path). Forward prices: no BQ price-history reader surfaced in
`bigquery_client.py` — the 60.2 replay used **yfinance closes** ($0); same
pattern works for t+5/t+10 closes. Excess return benchmark: SPY (US) is the
only wired benchmark (memory: MARKET_CONFIG has no benchmark field); use
SPY/^GDAXI/^KS11 via yfinance for per-market excess. JOIN IS FEASIBLE:
(ticker, DATE(created_at)) x yfinance daily closes. Volume caveat: analyses
exist ~since 2026-05; KR/EU rows are sparse (go-live 06-01, freeze 06-15+);
expect the KR/EU slopes to be low-power — the "statistically indistinguishable
from zero" escalation branch is LIKELY to fire on sparsity alone, so the
method doc must separate "no signal" from "insufficient n" (report n, CI, and
power alongside the slope; overlapping 5-10d windows need HAC/Newey-West SEs).

---

## 1. VERIFIED FEE TABLE (per-rate citations; accessed 2026-07-08)

| Market | Component | Current rate | Effective | Criterion text | Drift? |
|--------|-----------|--------------|-----------|----------------|--------|
| US | SEC Section 31 (sells, notional) | **$20.60/M = 0.00206%** (prior: $0.00/M through 2026-04-03) | 2026-04-04, charge date = TRADE date | 0.00206% | **CURRENT** |
| US | FINRA TAF (sells, per share) | **$0.000195/share, cap $9.79/trade** | 2026-01-01 | $0.000166/share, cap $8.30 | **STALE — criterion encodes the rate that expired 2025-12-31** |
| US | TAF forward schedule | 2027: $0.000232/$11.61; 2028: $0.000240/$12.05; 2029: $0.000249/$12.50 | Jan 1 each year | — | rates STEP ANNUALLY -> must be config, not constants |
| KR | Securities transaction tax (sells) | **0.20%** KOSPI (incl. rural-development special tax) + KOSDAQ; KONEX 0.1%; unlisted 0.35% | 2026-01-01 (was 0.15% in 2025) | +0.20% | **CURRENT** (PwC page reviewed 2026-06-04) |
| EU (Xetra) | IBKR tiered commission (both sides) | **0.05%, min EUR 1.25/order**, cap EUR 29 + exchange/clearing/regulatory pass-through; fixed plan: min EUR 3 | current | 0.05% min EUR 1.25 | **CURRENT** (min dominates below ~EUR 2,500 notional) |

Sources: FINRA fee-adjustment schedule (finra.org/rules-guidance/rule-filings/sr-finra-2024-019/fee-adjustment-schedule, read in full 2026-07-08); FINRA Information Notice 3/17/26 (finra.org/rules-guidance/notices/information-notice-20260317, in full); SEC FY2026 advisory (sec.gov/rules-regulations/fee-rate-advisories/2026-2 — 403 to WebFetch, figures corroborated by the FINRA notice + OCC memo #58530 + NYSE snippet); PwC Korea Other Taxes (taxsummaries.pwc.com/republic-of-korea/corporate/other-taxes, in full); bankeronwheels.com/ibkr-fixed-vs-tiered/ (in full).

**Drift handling (criteria are immutable):** the FINRA Section-1 rulebook page STILL codifies $0.000166/$8.30 (fetched in full — stale codified text, last amendment SR-FINRA-2023-009), which is presumably what the 2026-06-11 audit read; the SR-FINRA-2024-019 schedule supersedes it. Recommendation for the contract: implement the table as a DATED per-market rate schedule (rate rows carry effective-from dates; the replay charges the rate lawful at each trade date — all recorded June-2026 trades get $0.000195/$9.79 + $20.60/M). Pre-register in contract.md + live_check that the implementation charges the current lawful TAF rather than the criterion's expired figure, and let the operator adjudicate via the `FEE TABLE:` token. Charging a knowingly-expired rate to satisfy criterion literalism would falsify the replay. The "1-cent round-up" TAF convention comes from the 06-11 audit's FINRA-notice reading (not re-verified this pass; the rulebook page adds a de-minimis rule: no fee when execution price < the per-share rate).

## 6-8. External findings (read-in-full + key snippets)

### Read in full (8; counts toward gate)
| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| finra.org/.../sr-finra-2024-019/fee-adjustment-schedule | 2026-07-08 | official | WebFetch full | TAF 2026 $0.000195/$9.79; annual steps through 2029 |
| finra.org/rules-guidance/notices/information-notice-20260317 | 2026-07-08 | official | WebFetch full | Section 31 $20.60/M eff 2026-04-04; prior $0.00/M; charge date = trade date |
| finra.org/.../section-1-member-regulatory-fees | 2026-07-08 | official (STALE text) | WebFetch full | codified $0.000166/$8.30 + de-minimis rule — conflict resolved by the schedule above |
| finra.org/rules-guidance/guidance/trading-activity-fee | 2026-07-08 | official | WebFetch full | rate lives in Schedule A By-Laws; TAF User Guide pointer |
| taxsummaries.pwc.com/republic-of-korea/corporate/other-taxes | 2026-07-08 | industry/official-grade | WebFetch full | KR 0.20% from 2026-01-01 (2025: 0.15%; 2024: 0.18%); KONEX 0.1%, unlisted 0.35% |
| bankeronwheels.com/ibkr-fixed-vs-tiered/ | 2026-07-08 | industry | WebFetch full | Xetra 0.05% both plans; tiered min EUR 1.25 cap EUR 29; fixed min EUR 3; tiered wins small tickets |
| arxiv.org/html/2602.00196 (Rasekhschaffe) | 2026-07-08 | preprint | WebFetch full | unsmoothed daily LLM signals: 98.8% daily / 249x annual turnover, gross Sharpe 2.27 -> NET -1.24; 21d MA: turnover -82%, net Sharpe 0.78; break-even 25.4 bps; est. real cost 12.4 bps/trade (4.0 spread + 8.4 impact); "signal smoothing is essential" |
| docs.mosek.com/portfolio-cookbook/transaction.html | 2026-07-08 | official docs | WebFetch full | fixed per-order costs "discourage trading very small amounts" -> sparse portfolios; buy/sell-asymmetric linear costs; sqrt-law impact (beta=3/2) — the min-ticket floor's theoretical basis |

### Snippet-only (context; not counted)
sec.gov/rules-regulations/fee-rate-advisories/2026-2 (403); federalregister.gov 2026-04233 (bot-blocked) + 2024-27764 (TAF filing); infomemo.theocc.com #58530; NYSE regulatory-fee PDF; sec.gov 34-104909.pdf; alpaca.markets/support/regulatory-fees; investrade.com/fees; orbitax + businesskorea + biggo (KR STT news); elaw.klri.re.kr STT Act; ey.com KR tax reform; arxiv 2412.11575 (cost-aware portfolios, Dec 2024: turnover penalties raise net returns, high-dim proportional+quadratic costs); afajof Baldi-Lanfranchi "Transaction-cost-aware Factors" (2024: partial rebalancing at fixed trading intensity); Boyd et al. cvx_portfolio (multi-period convex, holding/trade costs); grahamcapital.com Transaction Costs note (2017 practitioner conventions incl. half-spread); doi.org/10.1080/1351847X.2025.2558117 (implementation-shortfall variance in construction, 2025); tandfonline 0015198X.2018.1547056 (Boudoukh et al. "Long-Horizon Predictability: A Cautionary Tale" — NW SEs underestimate at large horizon/T); nber.org w11280 (Petersen, panel SEs); stata newey manual; arxiv 2506.06356 (multi-day turnover control, A-shares); ~55 unique URLs collected across 7 searches.

### Query variants run (three-variant discipline)
Current-year: "SEC Section 31 fee rate 2026"; "FINRA TAF rate 2026"; "Korea securities transaction tax rate 2026". Year-less canonical: "transaction cost model systematic trading turnover control minimum holding period"; "information coefficient forward returns overlapping windows Newey-West standard errors". Last-2-year/frontier: "transaction cost aware portfolio construction turnover penalty 2025"; "FINRA trading activity fee increase $0.000195 effective January 1 2026".

## 9. Recency scan (2024-2026)

FOUND — multiple findings that supersede or sharpen the canonical sources: (1) FINRA SR-2024-019 (Nov 2024) replaced the decade-stable TAF with a RISING ANNUAL SCHEDULE through 2029 — fee tables must be date-parameterized; (2) KR STT restored to 0.20% effective 2026-01-01 (2025 was 0.15% — any replay of 2025 KR trades would use 0.15%); (3) SEC Section 31 had a $0.00/M holiday until 2026-04-04 — rate is appropriation-driven and resets ~annually; (4) Rasekhschaffe (arXiv 2602.00196, 2026) gives the directly-on-point LLM-signal turnover result (21d smoothing flips net sign); (5) arXiv 2412.11575 (Dec 2024) + the 2025 implementation-shortfall-variance paper support cost-aware construction with turnover penalties — noted as literature context only; a construction-stage penalty is decision-math and NOT proposed for 61.5 (adjacent to the banned band family in effect, and out of the criterion's scope).

## 10. Application to pyfinagent (design sketches)

**Cost model (criterion 1).** New `paper_cost_model_enabled: bool = False` + a dated per-market rate table (settings or `backend/services/transaction_cost_model.py`). Plug points: paper_trader.py:189 (BUY — EU commission leg + optional half-spread + min-ticket check) and :388 (SELL — SEC notional bps + TAF per-share w/ cap + KR STT + EU commission; `sell_qty`, `position.market`, LOCAL notional all in scope). Market via persisted `market` column / `market_for_symbol` (paper_trader.py:1214). Half-spread as a FEE LINE, not a fill-price mutation (keeps `_bq_sim_fill` deterministic, execution_router.py:85-126). OFF path byte-identical flat 0.1%. OPEN DESIGN QUESTION for contract: whether ON retains the flat 0.1% as commission proxy PLUS regulatory adders (conservative; matches the audit's "KR P&L overstated ~0.2%/round-trip" complaint) or substitutes realistic per-market commissions (EU 0.05% min 1.25 IS the commission; US criterion lists only regulatory fees). Recommend: retain flat proxy + adders for US/KR; EU replaces the proxy with 0.05% min EUR 1.25. Min-ticket floor: config default ~EUR/USD 2,500 equivalent (the notional where EU per-order minimum stops dominating; MOSEK fixed-cost rationale).

**Replay (criterion 1).** Pure ledger restatement per `replay_60_2_swap_fix.py` pattern ($0): recompute tx_cost per recorded trade (market, quantity, LOCAL notional, trade-date-effective rates) vs recorded flat 0.1%; restate cumulative P&L; md output. Window caveat in §4 (trade freeze — pre-register interpretation).

**Churn measurement (criterion 2).** SQL in §3; window must contain sell activity (post-66.2 reactivation) or invoke the vacuous-pass disclosure doctrine (Cycle 66 precedent).

**Min-holding lever (criterion 3, conditional).** Only if >30% of window sells close <=2-day holds: exclusion from swap displacement below N holding days + re-entry cooldown — time-axis lever, 55.3-sanctioned, flag default OFF, `TURNOVER LEVERS:` token. Literature: Rasekhschaffe smoothing result (turnover from signal autocorrelation — Qian 2007 sqrt(1-rho) in audit_basis); 21d-MA score smoothing is a listed promotable lever in the goal file (:184-185) but ONLY under the token — NOT hysteresis (no band), but disclose the family question to Q/A proactively.

**Slope monitor (criterion 4).** Per market: pooled OLS of t+5..t+10 forward excess return (bps, vs SPY/^GDAXI/^KS11 via yfinance) on `final_score` from `financial_reports.analysis_results` (exclude sentinel-0.0/degraded rows). Overlapping horizons -> Newey-West HAC with lag h-1, AND report a non-overlapping subsample check (Boudoukh et al.: NW gives "false comfort" at large h/T). Report slope, SE, CI, n per market; distinguish "slope ~ 0 with adequate n" (signal-validity alarm -> operator) from "insufficient n" (KR/EU sparse: go-live 06-01 + freeze 06-15..07-06). Standing monitor = a script + md/BQ output re-runnable per cycle, not a one-shot.

**Sequencing note for Main:** 61.5 is formally blocked by 61.4, but its two measurement criteria are ALSO functionally gated on 66.2 (capital redeployment) producing trades. Fee-table code + replay + slope estimation are buildable immediately; the churn window starts accruing only when selling resumes.

## Consensus vs debate

Consensus: costs must enter at decision/measurement level, not just accounting (Garleanu-Pedersen, Boyd, MOSEK, 2412.11575); turnover is driven by signal autocorrelation (Qian; Rasekhschaffe); fixed per-order costs make small tickets uneconomic (MOSEK; IBKR minimums). Debate/caution: NW SEs under overlapping long-horizon returns understate uncertainty (Boudoukh et al.) — hence the dual-inference design above; construction-stage turnover penalties are literature-favored but ruled out here by scope + the hysteresis-adjacency concern (operator can lift via HYSTERESIS: AUTHORIZE).

## Pitfalls (from literature + internals)

1. Hardcoding TAF/Section-31 constants — both change (TAF annually to 2029; S31 by appropriation). Dated config table mandatory.
2. Charging the criterion's expired TAF rate to satisfy literalism — falsifies the replay; pre-register the deviation.
3. Applying half-spread by mutating fill price — breaks fill determinism + notional conservation (execution_router docstring).
4. TAF/S31 are SELL-side only; EU commission is per-order BOTH sides; KR STT sell-side — misapplying sides silently doubles US costs.
5. EU minimum needs LOCAL (EUR) notional — apply before `_l2u` conversion (paper_trader.py:371).
6. NW-only inference on overlapping 5-10d returns (false comfort); report non-overlapping check + n.
7. Vacuous churn window (0 sells during freeze) — disclose, don't launder (Cycle-66 doctrine).
8. created_at is STRING in paper_trades — SAFE_CAST (61.4's fix; bigquery_client.py:955-964 precedent).
9. Sentinel-0.0/degraded analysis rows poison the slope regression — filter explicitly and report the filter.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (8)
- [x] 10+ unique URLs total (~55)
- [x] Recency scan (2024-2026) performed + reported (5 findings)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (settings, paper_trader, portfolio_manager, execution_router, replay tooling, 61.1 artifacts, harness_log)
- [x] Contradictions noted (FINRA rulebook stale text vs fee schedule — resolved; NW caution)
- [x] Per-claim citations

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 25,
  "urls_collected": 55,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "report_md": "handoff/current/research_brief_61.5.md",
  "gate_passed": true
}
```
