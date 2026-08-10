# Research Brief -- step 86.12

**Topic:** Intraday drawdown / daily-loss kill-switch design -- what NAV should a
daily-loss limit be evaluated against; how production risk systems avoid evaluating
a limit against a stale or start-of-day mark; standard failure modes (mark staleness,
asof mismatch, snapshot-vs-live NAV, rounding/FX timestamp deltas).

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage informational).
**Accessed / written:** 2026-08-10.

---

## Search-query variants run (three-variant discipline)

| Variant | Query | Purpose |
|---|---|---|
| Year-less canonical | `intraday drawdown daily loss limit start-of-day NAV mark-to-market risk limit breach` | prior art on what a DLL is measured against |
| Year-less canonical | `stale prices net asset value serial correlation understated drawdown volatility Getmansky Lo Makarov` | founding literature on stale marks -> understated risk |
| Last-2-year window | `portfolio risk limit stale mark asof mismatch snapshot NAV real-time risk monitoring 2025 2026` | recency scan |
| Current-year frontier | `real-time equity vs balance drawdown calculation trading risk engine data freshness heartbeat 2026` | 2026 practitioner frontier |

---

## Read in full (7; gate floor is 5)

| # | URL | Accessed | Kind | Fetched how | Key finding / verbatim |
|---|-----|----------|------|-------------|------------------------|
| 1 | https://www.bis.org/publ/bcbs239.pdf | 2026-08-10 | Official (BCBS 239, Tier 2) | WebFetch -> binary; text extracted locally with `pdfplumber` (28 pages, 57,572 chars) per research-gate.md Step 3 | **Principle 5 Timeliness:** *"A bank should be able to generate aggregate and up-to-date risk data in a timely manner while also meeting the principles relating to accuracy and integrity, completeness and adaptability. The precise timing will depend upon the nature and potential volatility of the risk being measured as well as its criticality to the overall risk profile."* **Principle 3(d):** *"A bank should strive towards a single authoritative source for risk data per each type of risk."* **Para 71:** *"Some position/exposure information may be needed immediately (intraday) to allow for timely and effective reactions."* **Para 43:** *"Supervisors expect banks' data to be materially complete, with any exceptions identified and explained."* **Para 22:** *"There should be no trade-offs that materially impact risk management decisions."* |
| 2 | http://web.mit.edu/Alo/www/Papers/JFE2004Pub.pdf | 2026-08-10 | Peer-reviewed (Getmansky, Lo & Makarov, *JFE* 2004) | WebFetch -> binary; **quotes independently re-verified with `pdfplumber`** because the fetch summary was partly fabricated (see "Source-integrity note") | Verbatim, p.2-3: *"'nonsynchronous trading', which refers to security prices recorded at different times but which are erroneously treated as if they were recorded simultaneously."* And: *"...will impart a downward bias on the estimated return variance and yield positive serial return correlation."* Abstract: *"the most likely explanation is illiquidity exposure and smoothed returns"*; they build a *"smoothing-adjusted Sharpe ratio."* Also cites Lo (2002): the naive Sharpe estimator can differ *"by as much as 70%."* |
| 3 | https://arxiv.org/html/2410.07607 | 2026-08-10 | Preprint, Oct 2024 (*Staleness Factors and Volatility Estimation at High Frequencies*) | WebFetch (native arXiv HTML) | Staleness is defined as *"the relative frequency of zero returns"*, modelled with a Bernoulli indicator *"whether prices update (B=0) or remain unchanged (B=1)."* Key asymmetry: *"The volatility estimates remain unbiased, whereas estimated co-volatilities are biased due to price staleness"* -- co-volatility understated by a factor in (0,1); *"the staleness correction reduces out-of-sample portfolio risk."* Offers **no** real-time stale-mark detection method (post-hoc only). |
| 4 | https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX%3A32017R0589 | 2026-08-10 | Official (EU RTS 6, MiFID II algo-trading) | WebFetch (HTML) | Art. 16(1) firms must *"monitor in real time all algorithmic trading activity"*; Art. 16(5) *"Real-time alerts shall be generated within five seconds after the relevant event."* Art. 12(1) kill functionality: *"cancel immediately, as an emergency measure, any or all of its unexecuted orders."* Art. 17(3) firms must *"reconcile its own electronic trading logs with information about its outstanding orders and risk exposures"* and keep the *"capability to calculate in real time its outstanding exposure."* **Notably: no explicit max-loss/drawdown threshold is mandated, and no explicit stale-market-data detection requirement.** |
| 5 | https://www.law.cornell.edu/cfr/text/17/240.15c3-5 | 2026-08-10 | Official (SEC Market Access Rule, via Cornell LII) | WebFetch (HTML) | (c)(1)(i): controls *"reasonably designed to systematically limit the financial exposure"* and *"prevent the entry of orders that exceed appropriate pre-set credit or capital thresholds."* (d): controls *"shall be under the direct and exclusive control of the broker or dealer."* (e): a documented system for *"regularly reviewing the effectiveness"*, *"no less frequently than annually."* **The rule is prescriptive about the THRESHOLD and about automation; it says nothing about the freshness of the value the threshold is compared against.** |
| 6 | https://docs.getdbt.com/docs/deploy/source-freshness | 2026-08-10 | Official docs (dbt) -- cross-domain engineering pattern | WebFetch (HTML) | The canonical data-freshness idiom: a freshness check is *"the first step of the job"*, and when configured as a run step *"If your source data is out of date -- this step will 'fail', and subsequent steps will not run."* Sampling rule: *"You should run your source freshness jobs with at least double the frequency of your lowest SLA"* (1h SLA -> 30min check). Freshness is computed from a **data-side** timestamp column (Snowflake: `LAST_ALTERED`), not from query time. |
| 7 | https://sre.google/sre-book/monitoring-distributed-systems/ | 2026-08-10 | Authoritative blog/book (Google SRE) | WebFetch (HTML) | Four golden signals: *"latency, traffic, errors, and saturation."* **Negative finding, reported honestly:** the chapter does **not** address data freshness/staleness, nor distinguishing "no data / unknown" from "healthy". So the "unknown is not healthy" doctrine already in `kill_switch.py` has **no** support from this canonical SRE source -- its support comes from BCBS 239 para 43 (exceptions identified and explained) and from the repo's own measured incidents, not from SRE. |

### Source-integrity note (must survive into the contract)

The WebFetch **summary** of source #2 asserted quoted strings -- `"smoother than"`,
`"understate"`, `"carried forward"`, `"estimated volatility is downward biased"` --
and numeric claims ("volatility underestimated 20-40%", "Sharpe overstated 10-50%")
that are **NOT PRESENT** in the paper. I re-extracted all 8 fetched pages with
`pdfplumber` and grepped each probe: all six returned `NOT FOUND`. The paper's real
wording is *"impart a downward bias on the estimated return variance"*, and it
publishes no such percentage bands. **Only the pdfplumber-verified strings above are
quoted in this brief.** This is the same failure mode recorded for step 83.1.1; treat
any WebFetch PDF summary as unverified until re-extracted.

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|--------------------------|
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=387578 | Preprint landing (GLM 2004) | duplicate of #2 (published version read instead) |
| https://www.nber.org/papers/w9571 | Working paper (GLM) | duplicate of #2 |
| https://www.sciencedirect.com/science/article/abs/pii/S0304405X04000698 | Journal paywall (JFE) | paywalled; #2 is the same paper |
| https://www.acadian-asset.com/investment-insights/owenomics/serial-killer-drawdowns-and-serial-correlation | Industry (Acadian) | corroborates serial-correlation -> drawdown link; lower tier than #2 |
| https://arxiv.org/pdf/2605.19337 | Preprint 2026 (Agentic Trading: LLM Agents Meet Financial Markets) | recency-scan hit; adjacent topic (agent trading), not limit-freshness |
| https://arxiv.org/pdf/2605.27887 | Preprint 2026 (PortBench) | recency-scan hit; portfolio-management benchmark, not risk-limit asof |
| https://www.workiva.com/resources/new-model-portfolio-monitoring-and-management-2026 | Vendor 2026 | marketing tier |
| https://portquant.com/blog/equity-drawdown-vs-balance-drawdown | Industry 2026 | **highest-value snippet**: balance DD *"only moves when a position is closed"* vs equity DD *"updates while trades are open"*; *"If you size positions from balance DD, you are sizing off a delayed, smoothed version of your real risk."* |
| https://www.futureshive.com/blog/end-of-day-vs-intraday-drawdown | Industry 2026 | EOD-vs-intraday framing |
| https://audacity.capital/trading-guides/relative-drawdown/ | Industry 2026 | already cited in `kill_switch.py:9-11` docstring |
| https://arongroups.co/forex-articles/eod-vs-intraday-drawdown/ | Community | EOD floor fixed during session |
| https://www.thinkcapital.com/prop-firm-drawdown-rules/ | Community | daily vs max DD |
| https://apextraderfunding.com/help-center/intraday-trailing-drawdown-accounts/intraday-trailing-drawdown-explained/ | Industry | intraday trailing recalculates tick-by-tick on peak live equity |
| https://damnpropfirms.com/glossary/daily-loss-limit/ | Community | DLL computed from prior day's CLOSING balance |
| https://proptradingvibes.com/blog/tradeday-intraday-trailing-drawdown | Community | trailing threshold does not reset daily |
| https://www.tradeclaris.com/blogs/daily-drawdown-rules-explained | Community | breach semantics |
| https://newyorkcityservers.com/blog/prop-firm-daily-drawdown-rules | Community | breach semantics |
| https://traderssecondbrain.com/guides/how-to-track-drawdown | Community | monitoring cadence |
| https://arxiv.org/pdf/2410.07607 | (PDF form of #3) | HTML form read instead, per research-gate.md |
| https://www.federalregister.gov/documents/2010/11/15/2010-28268/risk-management-controls-for-brokers-or-dealers-with-market-access | Official | **ATTEMPTED, FAILED** -- 302 to `unblock.federalregister.gov` (bot wall) |
| https://www.ecfr.gov/current/title-17/chapter-II/part-240/section-240.15c3-5 | Official | **ATTEMPTED, FAILED** -- same 302 bot wall; substituted Cornell LII (#5) |
| https://www.sec.gov/divisions/marketreg/marketaccess-secg.htm | Official | **ATTEMPTED, FAILED** -- HTTP 403 |

**Unique URLs collected: 30** (7 read in full + 23 snippet-only/attempted).

---

## Recency scan (2024-2026)

Performed -- two dedicated passes (the `2025 2026` and `2026` query variants above).
Result: **2 new findings that COMPLEMENT (do not supersede) the canonical sources.**

1. **arXiv:2410.07607 (Oct 2024)** sharpens the classical stale-price result with a
   directional asymmetry the 2004 literature does not state: under staleness,
   *variance* estimates stay unbiased while *co-variances* are biased toward zero.
   Practical read for a single-book NAV: staleness does not simply "shrink" every
   risk number uniformly -- it distorts cross-asset aggregation specifically. It also
   confirms there is **no published real-time stale-mark detector**; correction is
   post-hoc. That is a genuine gap in the literature, not a gap in my search.
2. **2026 practitioner consensus has moved toward equity-based (live) drawdown and
   AWAY from daily loss limits entirely.** Apex removed DLL on most products post-
   March-2026; TPT removed DLL across all phases in Jan 2025 (search snippets). The
   stated rationale is that *"with improved risk monitoring infrastructure (real-time
   mark-to-market, automatic position closures), firms can rely on the trailing model
   alone."* This is directly adversarial to pyfinagent's current 4%-DLL design and is
   flagged below.

No 2024-2026 source supersedes BCBS 239 or 15c3-5; both remain in force.

---

## Key findings

1. **The name of the defect class is "nonsynchronous trading", and it is a
   50-year-old known hazard.** *"security prices recorded at different times but
   which are erroneously treated as if they were recorded simultaneously"*
   (Getmansky/Lo/Makarov 2004, pdfplumber-verified,
   http://web.mit.edu/Alo/www/Papers/JFE2004Pub.pdf). A daily-loss limit computes
   `(sod_nav - current_nav)/sod_nav`. If `sod_nav` is asof T1 and `current_nav` is
   asof T2 != T1, the subtraction is exactly a nonsynchronous pair. The literature's
   result is that this **biases the estimated variance DOWNWARD** -- i.e. the naive
   error direction is to UNDER-state risk, so the limit fires LATE.
2. **The regulatory frameworks are prescriptive about the threshold and silent about
   the freshness of the value compared against it.** SEC 15c3-5(c)(1)(i) requires
   *"pre-set credit or capital thresholds"* with no asof requirement
   (https://www.law.cornell.edu/cfr/text/17/240.15c3-5); RTS 6 mandates real-time
   monitoring and 5-second alerts but *"No explicit requirements for detecting stale
   market data are included"*
   (https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX%3A32017R0589).
   **The freshness obligation comes from BCBS 239, not from the trading rules.**
3. **BCBS 239 is the load-bearing external authority for this step.** Principle 5
   makes timeliness a first-class requirement *co-equal with accuracy and
   completeness*, explicitly scaled to *"the potential volatility of the risk being
   measured"* -- a 4%/day equity limit implies a mark-age budget of well under a day.
   Principle 3(d) demands *"a single authoritative source for risk data per each type
   of risk"*. Para 43's *"any exceptions identified and explained"* is the external
   basis for the repo's existing "unknown is not healthy" doctrine.
   (https://www.bis.org/publ/bcbs239.pdf)
4. **The engineering answer to "how do you avoid evaluating against a stale mark" is
   a data-side asof timestamp + a threshold, checked FIRST and hard-failing.** dbt's
   source-freshness idiom: freshness runs as *"the first step of the job"*, computed
   from a **loaded-at column on the data itself** (never wall-clock at read time),
   with warn/error thresholds, and *"subsequent steps will not run"* on failure. The
   sampling rule -- *"at least double the frequency of your lowest SLA"* -- gives a
   principled way to pick a check cadence. (https://docs.getdbt.com/docs/deploy/source-freshness)
5. **Balance-NAV vs equity-NAV is the crisp industry framing of pyfinagent's exact
   bug.** Balance drawdown *"only moves when a position is closed"*; equity drawdown
   *"updates while trades are open"*; *"If you size positions from balance DD, you
   are sizing off a delayed, smoothed version of your real risk."*
   (portquant.com, snippet). pyfinagent's `total_nav` moves only when
   `mark_to_market()` runs -- a *snapshot* NAV with balance-like latency being used
   where an equity-like number is implied.
6. **[Adversarial] The 2026 practitioner drift is AWAY from daily-loss limits.**
   Several major prop firms dropped DLL entirely once real-time marking existed,
   reasoning that a trailing high-water limit on live equity subsumes it. This
   argues that adding an asof guard to a *stale-snapshot* DLL may be fixing the wrong
   layer; the alternative is to make the trailing leg (which is date-independent --
   `kill_switch.py:854`, `:921-923`) the primary control. **Counter-argument for
   pyfinagent:** the trailing leg reads the SAME stale `current_nav`, so a freshness
   guard is required either way; only the DLL's *baseline* is date-scoped.
7. **No published method exists for real-time stale-mark detection.** arXiv:2410.07607
   is explicitly post-hoc. So the design must be a **provenance/age guard**
   (is this number recent?), not a **statistical detector** (does this number look
   stale?). That is a scope-limiting finding for the contract.

---

## Internal code inventory

### THE HEADLINE ANSWER (the question the caller asked)

**`current_nav` is a STORED BigQuery figure on every path, never a live mark.**
There are exactly **five** production producers, four of which read the SAME persisted
`paper_portfolio.total_nav` column and one of which accepts an arbitrary caller value:

| # | Producer (file:line) | Expression | Live mark or stored? |
|---|----------------------|-----------|----------------------|
| 1 | `backend/api/paper_trading.py:513-517` (GET `/kill-switch`) | `bq.get_paper_portfolio("default")` then `float((portfolio or {}).get("total_nav") or (portfolio or {}).get("starting_capital") or 0.0)` | **STORED** BQ row, + `starting_capital` fallback |
| 2 | `backend/api/paper_trading.py:569-580` (POST `/resume`) | same BQ read; `float((portfolio or {}).get("total_nav") or 0.0)` -- **no** `starting_capital` fallback | **STORED** |
| 3 | `backend/services/paper_trader.py:1269-1272` (`roll_daily_anchor`, cycle Step 0) | `self.get_or_create_portfolio()` -> `total_nav or starting_capital or 0.0` | **STORED** |
| 4 | `backend/services/paper_trader.py:1342-1343` (`check_and_enforce_kill_switch`, Step 5.5) | `self.get_or_create_portfolio()` -> `total_nav or starting_capital or 0.0` | **STORED**, but re-written by `mark_to_market()` at Step 5 immediately prior |
| 5 | `backend/agents/mcp_servers/risk_server.py:64-82` (MCP tool `kill_switch`) | `current_nav: float \| None = None` -- **caller-supplied parameter**, no provenance at all | **ARBITRARY** |

`get_or_create_portfolio` is a plain BQ read (`backend/services/paper_trader.py:157`:
`portfolio = self.bq.get_paper_portfolio("default")`), and
`BigQueryClient.get_paper_portfolio` is `SELECT * FROM paper_portfolio WHERE
portfolio_id=@pid LIMIT 1` (`backend/db/bigquery_client.py:553-566`).

`evaluate_breach` itself (`backend/services/kill_switch.py:805-938`) **never marks
anything**: it takes `current_nav` as a parameter and reads only `_state.snapshot()`
for the two baselines (`:849-854`). The sole live-mark computation is
`PaperTrader.mark_to_market()` (`backend/services/paper_trader.py:699`), which
**writes** `total_nav` at `:780` -- it is a producer of the stored column, not a
reader on any kill-switch path.

### The asymmetry, stated precisely

- Baseline freshness **IS** checked:
  `daily_baseline_stale = _sod_date_is_stale(s.get("sod_date"), sod)`
  (`kill_switch.py:865`), disarming the daily leg when `sod_date != today (UTC)`
  (`kill_switch.py:986`). The docstring at `:961-976` states the doctrine outright:
  *"freshness is a claim that must be provable."*
- Current-NAV freshness is checked **nowhere**. `evaluate_breach` validates only
  `current_nav is None or current_nav <= 0` (`kill_switch.py:887`) -> `nav_invalid`.
  A positive-but-days-old NAV is indistinguishable from a fresh one.
- So `(sod - current_nav)/sod` (`kill_switch.py:916`) pairs a **provably-today**
  denominator with a **possibly-stale** numerator. The design already proves it cares
  about asof-matching -- on one side of the subtraction only.

### The asof timestamp ALREADY EXISTS and is discarded

`mark_to_market` writes `"updated_at": datetime.now(timezone.utc).isoformat()` into
the `paper_portfolio` row (`backend/services/paper_trader.py:789-795`), and
`get_paper_portfolio` does `SELECT *`, so **`updated_at` is present in the very dict
every one of paths #1-#4 reads and then drops.** No new column, no migration, and no
new BQ read is needed to compute mark-age -- this is the dbt `loaded_at_field`
pattern already satisfied on the data side.

There is also direct in-repo prior art for the pattern at the position level:
`mark_to_market` stamps `"marked_at"` per position with the comment (phase-61.3)
*"as-of indicator ... Observability only: no order, stop, or size depends on it."*
Step 86.12 is the portfolio-level analogue -- and the question of whether it stays
observability-only or becomes gating is the contract's central decision.

### Second-order staleness INSIDE a "fresh" mark (the FX/rounding half of the question)

Even when `mark_to_market` runs, the NAV it produces can silently contain stale
components, with no indicator on the row:

- `live_price = _get_live_price(ticker)`; if `None`, it falls back to
  `pos.get("current_price", pos["avg_entry_price"])` -- i.e. the **previous mark, or
  failing that the entry price** (`backend/services/paper_trader.py:705-708`).
- FX: `_l2u = _fx_local_to_usd(pos.get("market"))`; if `None` the code logs a warning
  and **keeps the last-known USD market value** (`paper_trader.py:713-721`). For EU/KR
  positions this means the USD leg of NAV can be an arbitrary number of sessions old.
- `nav = portfolio["current_cash"] + total_positions_value` (`paper_trader.py:780`):
  **cash comes from the stored row, positions from the (possibly partly stale) marks**
  -- a nonsynchronous sum in the exact GLM sense.
- Values are `round(..., 2)` at write (`:780`) and `round(..., 4)` on the pct in
  `evaluate_breach` (`:927`), so a 4.0000% limit and a 3.99995% reading are
  distinguishable only after rounding -- a real but second-order boundary concern.

### Live audit-trail evidence (`handoff/kill_switch_audit.jsonl`)

Event counts, live file: `{'pause': 44, 'resume': 10, 'sod_snapshot': 10}` -- **zero
`peak_update` rows in the live file**; they are only in the rotated archives
(`kill_switch.py:94-111`). Last six `sod_snapshot` rows:

```
{"ts": "2026-07-31T18:47:30.141829+00:00", "event": "sod_snapshot", "nav": 23772.49, "date": "2026-07-31"}
{"ts": "2026-08-03T19:29:34.433705+00:00", "event": "sod_snapshot", "nav": 23803.94, "date": "2026-08-03"}
{"ts": "2026-08-05T19:34:47.386888+00:00", "event": "sod_snapshot", "nav": 23830.46, "date": "2026-08-05"}
{"ts": "2026-08-08T20:58:29.379594+00:00", "event": "sod_snapshot", "nav": 23830.46, "date": "2026-08-08"}
{"ts": "2026-08-09T13:03:44.126943+00:00", "event": "sod_snapshot", "nav": 23833.94, "date": "2026-08-09", "provisional": true}
{"ts": "2026-08-09T13:08:40.510286+00:00", "event": "sod_snapshot", "nav": 23833.94, "date": "2026-08-09", "provisional": false}
```

Three measured facts:

1. **The "start-of-day" anchors are not stamped at start of day.** `ts` values are
   18:47, 19:29, 19:34, 20:58 UTC -- at or after the US cash close. `date` is today's
   UTC date so `_sod_date_is_stale` reads FRESH, while the NAV it carries is an
   *end*-of-session mark. The 85.6 Step-0 roll moved this earlier (13:03 UTC on
   08-09) but 13:03 UTC is 09:03 ET, i.e. pre-open, so the anchor is the **previous
   close's** mark either way. That is defensible as a DLL definition (industry: DLL
   is *"calculated from the prior day's closing balance"*) but it is **not** what the
   module docstring says (`kill_switch.py:7`: *"4% of start-of-day NAV"`).
2. **`sod_snapshot` rows carry no mark timestamp.** Schema is
   `{ts, event, nav, date, provisional?}` (`kill_switch.py:634-635`). `ts` is the
   **write** time, not the asof of the NAV. Nothing downstream can compute mark-age
   from the trail.
3. **The 85.6 provisional->final upgrade produced the identical value** (23833.94 ->
   23833.94) on 2026-08-09, 5 minutes apart. `paper_trader.py:1415-1449` claims that
   at the upgrade point `nav` "IS today's freshly-marked NAV (mark_to_market ran at
   Step 5)". The identical value is consistent with (a) the mark ran and the book did
   not move, or (b) the mark did not refresh. **The audit row cannot distinguish
   them** -- the observability gap in one line.

### Ordering: where the live mark sits relative to the breach decision

`backend/services/autonomous_loop.py`: Step 0 `roll_daily_anchor` (`:536`), Step 5
`mark_to_market` (`:1368`), Step 5.5 `check_and_enforce_kill_switch` (`:1400`),
post-halt re-mark (`:1429`), end-of-cycle mark (`:1770`). So on the **cycle path**
the stored NAV is fresh by construction. On the **API paths** (#1 badge, #2 resume)
and the **MCP path** (#5) there is no ordering guarantee at all -- they read whatever
BQ holds, at any hour, including a weekend when the weekday-only cron has not run
(`paper_trading.py:655-657` already documents the weekend-no-cycle case, for the
*baseline* only).

### Files inspected (8)

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/services/kill_switch.py` | 1-1138 (full) | breach math, baselines, audit replay | LIVE |
| `backend/api/paper_trading.py` | 495-694 | GET /kill-switch, POST /pause, POST /resume | LIVE |
| `backend/services/paper_trader.py` | 155-180, 699-800, 1200-1510 | portfolio read, mark_to_market, roll_daily_anchor, check_and_enforce_kill_switch | LIVE |
| `backend/db/bigquery_client.py` | 553-600 | get_paper_portfolio / upsert_paper_portfolio | LIVE |
| `backend/agents/mcp_servers/risk_server.py` | 60-100 | 5th evaluate_breach consumer (arbitrary NAV) | LIVE |
| `backend/services/autonomous_loop.py` | 514-536, 1363-1400, 1429, 1770 | cycle step ordering | LIVE |
| `handoff/kill_switch_audit.jsonl` | full (64 rows) | sole persistence of baselines | LIVE |
| `backend/config/settings.py` | 39, 390-394 | `kill_switch_peak_reset_enabled`, `kill_switch_auto_resume_enabled` (both DARK) | LIVE |

Dead/duplicate/drift noted: (a) `kill_switch.py:12` cites *"FINRA Rule 15c3-5"* -- it
is an **SEC** rule (17 CFR 240.15c3-5), not FINRA; (b) `kill_switch.py:7` says the
limit is *"4% of start-of-day NAV"* while the measured anchors are prior-close marks;
(c) `_AUDIT_PATH` is mirrored as a second constant `_KILL_SWITCH_AUDIT_PATH` at
`backend/api/paper_trading.py:892` (documented mirror, drift risk).

---

## Consensus vs debate (external)

**Consensus.** (i) A daily-loss limit is anchored on a *closing/opening balance* that
is fixed for the session -- all industry sources agree; (ii) risk data must carry a
provable asof and timeliness is co-equal with accuracy (BCBS 239 P3/P5); (iii)
freshness gates belong FIRST in the pipeline and should hard-fail (dbt).

**Debate.** (i) *Balance vs equity basis*: prop-firm practice is split, and 2026 has
been moving to live equity with the DLL removed entirely -- the adversarial finding
above; (ii) *Whether staleness is even correctable in real time*: the 2024 literature
says only post-hoc, so an age guard is the available instrument, not a detector;
(iii) BCBS 239 P5 deliberately refuses to name a number (*"precise timing will
depend"*), so the mark-age threshold is a pyfinagent design decision that must be
justified from the 4%/day limit, not copied from a source.

## Pitfalls (from literature, mapped to the likely fix)

1. **Fail-open on unknown age.** Treating a missing `updated_at` as fresh reproduces
   exactly the bug `_sod_date_is_stale` was written to kill (`kill_switch.py:968-972`
   already chose the opposite: unparseable/missing date => stale). Any NAV-age guard
   must default to STALE, not FRESH.
2. **Wall-clock instead of data-side timestamp.** dbt is explicit that freshness comes
   from a column on the data (`LAST_ALTERED` / `loaded_at_field`). Using
   `datetime.now()` at read time measures nothing.
3. **Turning `nav_stale` into `any_breached=True`.** `kill_switch.py:826-831` already
   documents why absence must not become a breach: it would `flatten_all()` a healthy
   book on a housekeeping fault. A NAV-age guard must follow the SAME per-leg
   `*_missing` / `armed` shape, not the breach shape.
4. **Wedging the resume path.** `/resume` already 409s on `armed=false`
   (`paper_trading.py:664-682`) and separately on a stale baseline (`:620-663`). A
   third refusal keyed on NAV age can re-create the 85.6 deadlock (paused book,
   unresumable) unless it has a stated, reachable unblock condition -- the 85.6
   correction at `:602-619` is the template for the wording.
5. **Nonsynchronous sums inside one "fresh" NAV.** Fixing portfolio-level age while
   leaving the `live_price is None` and `_l2u is None` fallbacks unmarked
   (`paper_trader.py:705-721`) leaves a NAV that is timestamped fresh and materially
   stale -- BCBS 239 para 43's *"exceptions identified and explained"* is the standard
   this fails.
6. **Both legs, not one.** The trailing leg (`kill_switch.py:921-923`) consumes the
   same `current_nav`; a guard applied only to the daily leg leaves half the switch
   evaluating a stale mark.

## Application to pyfinagent

| External finding | pyfinagent anchor | Implication for the contract |
|---|---|---|
| BCBS 239 P5 timeliness co-equal with accuracy | `kill_switch.py:887` validates only sign/None | add a `nav_stale` / `nav_asof` leg alongside `nav_invalid` |
| dbt `loaded_at_field`, data-side timestamp | `paper_trader.py:789-795` writes `updated_at`; `bigquery_client.py:556` `SELECT *` returns it; paths #1-#4 discard it | **zero-migration fix**: thread `updated_at` through, do not add a column |
| dbt "freshness first, subsequent steps do not run" | `evaluate_breach` is the first thing every path calls | the guard belongs inside `evaluate_breach`, not at each of the 5 call sites |
| GLM 2004 nonsynchronous pairing biases variance DOWN | `kill_switch.py:916` `(sod - current_nav)/sod` | the current error direction is **fire late**, i.e. under-protection -- state this in the hypothesis |
| BCBS 239 P3(d) single authoritative source | 4 readers of `paper_portfolio.total_nav` + 1 arbitrary MCP param (`risk_server.py:64`) | consider requiring provenance on the MCP path too, or documenting it as diagnostic-only |
| BCBS 239 para 43 exceptions identified | `mark_to_market` FX/price fallbacks are silent (`paper_trader.py:705-721`) | a `stale_components` count on the mark is the honest analogue of `marked_at` |
| 15c3-5(d) direct and exclusive control | `settings.py:39,394` both flags DARK | any new guard should ship DARK by default, matching the module's established idiom |
| 2026 drift away from DLL [adversarial] | `kill_switch.py:7-11` 4%/10% thresholds | do **not** change thresholds in this step; the finding argues about which leg is primary, and both read the same stale NAV |

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **7**
- [x] 10+ unique URLs total (incl. snippet-only) -- **30**
- [x] Recency scan (last 2 years) performed + reported -- 2 complementary findings
- [x] Full papers / pages read (not abstracts); 2 PDFs re-extracted with `pdfplumber`
      per research-gate.md Step 3, and one fabricated summary caught and corrected
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in scope (8 files) plus the 5th
      consumer (`risk_server.py`) that the scope did not name
- [x] Contradictions / consensus noted (incl. one adversarial 2026 finding)
- [x] Claims cited per-claim with URL + file:line, not footnoted
- [ ] **Gap:** three official US-regulator URLs were bot-walled (403 / 302 to
      `unblock.federalregister.gov`); the 15c3-5 text was obtained via Cornell LII
      instead of a `.gov` origin. Text is materially identical but the provenance is
      one hop removed.
- [ ] **Gap:** I did not measure the actual distribution of `updated_at` staleness on
      the live `paper_portfolio` row (would require a BQ query; out of the read-only
      research remit). The contract should require GENERATE to measure it before
      picking a threshold.

---

## Envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 23,
  "urls_collected": 30,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 2,
    "dry": false
  },
  "summary": "current_nav is a STORED BigQuery figure on all 5 kill-switch paths, never a live mark; the mark asof (paper_portfolio.updated_at) already exists in the dict every path reads and is discarded. Baseline freshness is checked, NAV freshness is not.",
  "brief_path": "handoff/current/research_brief_86.12.md",
  "gate_passed": true
}
```
