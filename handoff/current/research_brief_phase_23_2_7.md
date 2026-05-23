# Research Brief -- phase-23.2.7 (Red Line Monitor terminal NAV vs live)

**Tier:** SIMPLE
**Date:** 2026-05-23 (system date rolled mid-session)
**Spawn context:** RETROACTIVE -- Main spawned this researcher AFTER
writing test code, in real-time correction per
`feedback_never_skip_researcher` (operator override 2026-05-22).
Section H below documents the protocol-discipline breach + the
correction.

---

## Section A -- Internal code inventory (file:line)

The three NAV endpoints under test:

| Endpoint | File:line | NAV source | Cache | Notes |
|----------|-----------|------------|-------|-------|
| `GET /api/sovereign/red-line` | `backend/api/sovereign_api.py:319-357` | `financial_reports.paper_portfolio_snapshots` (BigQuery; `MAX(total_nav)` GROUP BY snapshot_date) via `_fetch_snapshots` at L140-170 | 60s in-memory (`_CACHE_TTL` L35) | Forward-fills calendar days via `_forward_fill_calendar` (L173-206); last point may be `source="filled"` (not `"actual"`) when today's snapshot hasn't been written yet. |
| `GET /api/paper-trading/portfolio` | `backend/api/paper_trading.py:160-230` | `BigQueryClient.get_paper_portfolio("default")` -> the live `paper_portfolio` row (NOT the snapshots table) | `ENDPOINT_TTLS["paper:portfolio"]` (likely 30s; service-defined) | `total_nav` is the field directly written by `paper_trader.mark_to_market` at L484-490. |
| `GET /api/paper-trading/kill-switch` | `backend/api/paper_trading.py:451-489` | Same as portfolio -- `bq.get_paper_portfolio("default")` then `nav = (portfolio).get("total_nav") or .get("starting_capital")` (L468) | None (always live) | 5s timeout on BQ; falls back to in-memory pause state on timeout (L462-467). |

NAV write path:
- `paper_trader.mark_to_market` (`backend/services/paper_trader.py:471-499`)
  computes `nav = current_cash + sum(market_value)` then writes
  `total_nav` to the `paper_portfolio` row (L484-490).
- `paper_trader.save_daily_snapshot` (`backend/services/paper_trader.py:792-840`)
  reads `total_nav` from the portfolio row (L811) and writes a row
  to `paper_portfolio_snapshots` (L839, via `bq.save_paper_snapshot`).
- The two writes are **sequential, not transactional**: the
  portfolio row is updated first, the snapshot row second. Both
  read from the same in-process `portfolio` dict so a single
  cycle's NAV is byte-identical between them -- but the snapshot
  is only as fresh as the most-recent `save_daily_snapshot` call,
  while `portfolio.total_nav` is as fresh as the most-recent
  `mark_to_market` call. These are coupled in
  `backend/autonomous_loop.py`'s daily cycle, but the red-line
  endpoint reads snapshots and the portfolio endpoint reads the
  live row -- so by construction the snapshot is at most one cycle
  behind the live portfolio NAV.

Test file:
- `backend/tests/test_phase_23_2_7_red_line_nav_match.py` -- 5
  tests, 4 PASS + 1 SKIP per the spawn brief. `NAV_MATCH_TOLERANCE_PCT
  = 1.0` (1%, L32-33).

## Section B -- External research (read in full -- 5)

| # | Source | URL | Read in full | Tier | Key finding |
|---|--------|-----|--------------|------|-------------|
| 1 | Limina -- NAV and P&L Reconciliation Guide | https://www.limina.com/blog/pnl-and-nav-reconciliation-guide | yes (WebFetch) | Industry-practitioner | No fixed numeric tolerance is industry-standard. "Set tolerances differently on different currencies, data types" -- per-feed configurable. Most-common legitimate break categories are **fees / taxes / manual adjustments / data import errors**. Recommends reconciliation at T or T+1; "Assets and Liabilities must be per trade date when they enter the NAV calculation, not per settle date". |
| 2 | Fidelity Learning Center -- Understanding an ETF's NAV | https://www.fidelity.com/learning-center/investment-products/etf/etfs-nav | yes (WebFetch) | Authoritative blog (broker-dealer) | Official NAV is struck once daily after market close (US equity: shortly after 4:00pm ET). iNAV (intraday NAV) is updated "several times a minute" with real-time data. Distinguishes the two-clock problem -- the official close-time snapshot and the real-time live valuation. Cross-asset stale-pricing risk (e.g. 4.5-hour LSE-vs-NYSE gap) is the canonical legitimate-divergence pattern. No specific tolerance %s. |
| 3 | NYIF -- Trading System Kill Switch: Panacea or Pandora's Box? | https://www.nyif.com/articles/trading-system-kill-switch-panacea-or-pandoras-box | yes (WebFetch) | Industry blog (NY Institute of Finance) | Explicitly warns: "a kill switch with one threshold based on one variable will not be viable". The piece doesn't quote NAV-source tolerance numbers but argues that kill-switch firing decisions require **outlier protection (cross-reference reported P&L against independent calculations)** before activation. Direct support for pyfinagent's 3-source cross-check pattern. |
| 4 | CrossTrade -- Trailing Drawdown Survival Guide | https://crosstrade.io/learn/risk-management/trailing-drawdown-survival-guide | yes (WebFetch) | Industry-practitioner | Differentiates **intraday** (peak NAV from live unrealized stream, "every tick raises the floor") vs **EOD** (peak NAV from a single close-of-session snapshot). pyfinagent's kill-switch uses peak from snapshots (EOD pattern) while `current_nav` is live (intraday pattern) -- the divergence is legitimate **when the snapshot is older than current_nav**, but must collapse to zero immediately after a daily snapshot. |
| 5 | Solvexia -- Mutual Fund Reconciliation glossary | https://www.solvexia.com/glossary/mutual-fund-reconciliation | yes (WebFetch) | Industry glossary | Reconciliation framework: "rule-based matching with support for tolerances and exceptions". Doesn't quote numbers; confirms the **N-way reconciliation pattern** -- internal-records vs custodian vs administrator -- which is the industry analog of pyfinagent's 3-endpoint (red-line snapshot + portfolio live + kill-switch live) cross-check. |

## Section B' -- Identified but snippet-only (8)

| # | Source | URL | Why not fetched in full |
|---|--------|-----|-------------------------|
| 1 | Nasdaq -- Difference Between NAV and Market Value Reconciliation | https://www.nasdaq.com/articles/difference-between-nav-and-market-value-reconciliation-2016-01-15 | WebFetch 60s timeout |
| 2 | GIPS Calculation Methodology PDF | https://www.gipsstandards.org/wp-content/uploads/2021/02/calculation_methodology_2.pdf | Binary PDF; no extractable text path attempted (would need pdfplumber per `.claude/rules/research-gate.md`) |
| 3 | Alpaca Paper Trading docs | https://docs.alpaca.markets/us/docs/paper-trading | Fetched; no NAV-tolerance content (simulation disclaimer only) |
| 4 | FundRecs -- NAV Reconciliation | https://www.fundrecs.com/nav-reconciliation | Fetched; promotional, no numeric tolerances |
| 5 | arXiv 1902.03457 -- "Are trading invariants really invariant?" | https://arxiv.org/abs/1902.03457 | Adjacent topic (cost-impact invariants, not NAV cross-source); snippet sufficient |
| 6 | arXiv math/0506077 -- Optimal timing of mark-to-market | https://arxiv.org/abs/math/0506077 | Adjacent topic (credit-risk MtM timing); snippet sufficient |
| 7 | NinjaTrader -- PNL Drawdown Kill Switch | https://ninjatraderecosystem.com/user-app-share-download/pnl-drawdown-kill-switch-example-for-strategy-builder/ | Vendor doc; not authoritative beyond the CrossTrade source already read |
| 8 | ISDA -- Portfolio Reconciliation in Practice | https://www.isda.org/a/crpTE/Portfolio-Reconciliation-In-Practice.pdf | Derivatives-focused (collateral PV); tangential to equity NAV |

URLs collected: 13 unique.

## Section C -- Verification protocol assessment

### Is "all 3 endpoints return the same NAV within 1% tolerance" the right invariant?

**Mostly yes, with two refinements:**

1. **1% is too loose for the same-cycle case, too tight for the
   cross-cycle case.** When all three reads happen within one
   `mark_to_market` -> `save_daily_snapshot` cycle, NAV is computed
   from the same in-memory `portfolio` dict, so the three values
   should be **byte-identical** (or differ only by Python's
   `round(nav, 2)` -- max delta is 0.5 cents on a $23K NAV =
   2.2e-5%). A 1% band hides a class of bugs where the values
   silently diverge by, say, $230 (1%); that's a real position
   miscount or stale cache, not "fee tolerance".

   But when the red-line endpoint is serving its **60s cache hit**
   while the portfolio endpoint has just been freshly recomputed
   after a `mark_to_market`, the snapshot can be up to one cycle
   stale. The autonomous loop runs daily, so the snapshot can be
   a full trading session behind -- and intraday position drift on
   a $23K paper portfolio routinely exceeds 1%.

   **Recommended tiered tolerance**:
   - Same-source (portfolio vs kill-switch): **<= 0.01% (i.e. 1
     basis point)**. Both read `paper_portfolio` row; only `round`
     errors permitted.
   - Cross-source (red-line vs portfolio): **<= 1.0% within trading
     hours, OR last-point.source must be "filled" / "actual" for
     today**. Document that the legitimate-divergence case (stale
     snapshot pre-daily-cycle) trips this, by design.

2. **The test should additionally assert `source` on the last point.**
   When last point has `source: "filled"` (forward-fill within the
   60s cache window for a day before today's snapshot has been
   written), the test should accept a larger tolerance. When
   `source: "actual"` for the current date, the tolerance should
   tighten (the snapshot was written in this cycle, so it should
   match within 1bp).

### Can the 3 endpoints legitimately diverge?

**Yes, in five named cases:**

| # | Case | Legitimate? | Detection |
|---|------|-------------|-----------|
| 1 | Red-line cache (60s TTL) serving stale data while portfolio just recomputed | Yes (cache TTL) | Wait 60s, retry; tolerance should be honored after cache expiry. |
| 2 | `save_daily_snapshot` not yet called for today; red-line last point is yesterday's snapshot forward-filled | Yes (intentional design; `_forward_fill_calendar` L173-206) | `series[-1].source == "filled"`; portfolio may have drifted intraday. |
| 3 | `mark_to_market` failed mid-cycle: portfolio.total_nav stale, snapshot newer | Bug (single-write atomicity break) | Tolerance breach **with** snapshot.date == today + source == "actual"; investigate. |
| 4 | Kill-switch 5s BQ timeout hit (L462-467); falls back to in-memory pause state with current_nav = 0 if no portfolio cache | Yes (fail-open) | `current_nav == 0` despite portfolio nav > 0; check kill-switch response shape. |
| 5 | Mid-cycle test execution: read red-line at T=0, then portfolio at T=200ms after `mark_to_market` fires at T=100ms | Yes (race) | Retry once; if persistent, treat as case 3. |

The Limina source confirms category #1 / #2 as the most common
legitimate break: "data import errors" (their phrase) is the
analog of "snapshot not yet written for today". The Fidelity
source confirms the two-clock problem (official NAV struck at
close vs live iNAV) as the analog of `red-line snapshot` vs
`paper-trading portfolio live`.

### Test coverage gaps Main may have missed

1. **No source-field assertion on last red-line point.** Today
   passed because the snapshot date matched, but the test will
   continue passing on a future day where the snapshot hasn't
   been written yet (the last point will be `source="filled"`
   carrying yesterday's NAV forward, which may still be within
   1% of today's NAV). The test should at minimum log/print the
   `source` so a human reviewer can spot the silent forward-fill.

2. **No cache-invalidation test.** The red-line endpoint caches
   60s. If the `paper_portfolio.total_nav` changes by >1% during
   that window, the test will FAIL transiently and PASS again
   60s later. Either flush the cache before the test or assert
   that POST `/api/paper-trading/portfolio` writes invalidate the
   `sovereign:red-line:*` cache keys (a brief grep of
   `sovereign_api.py` confirms it does NOT register such an
   invalidation -- this is a real cache-coherence gap, but a
   minor one given the 60s TTL).

3. **No assertion on `paper_portfolio_snapshots` row freshness.**
   The MAX(total_nav) GROUP BY snapshot_date query at L153-162
   means duplicate rows for one day get the highest NAV, not the
   most recent. If a buggy code path writes a wrong-direction NAV
   (e.g. a partial mark-to-market that under-counts a position),
   MAX may silently mask it. Test could query the snapshots table
   directly and assert at most one row per day.

4. **No assertion that kill-switch's `current_nav` is identical
   to portfolio's `total_nav` to-the-cent.** They both read from
   the same BQ row via `get_paper_portfolio("default")`. A 1%
   tolerance here is too generous; this should be `<= $0.01`
   (or `delta_pct < 0.0001`). Same-source reads MUST agree
   byte-for-byte.

5. **No regression-lock against the field-name drift.** The
   spawn brief mentions `last_nav = series[-1].get("nav")` --
   this is the field name today, but if a future refactor
   renames to `value` or `total_nav`, the test would still PASS
   when the field returns None (the assert at L65-67 checks
   None but doesn't catch a field-name typo if last_nav happens
   to default to a non-None value). Consider asserting the
   exact set of expected keys on the last point matches the
   pydantic model (`RedLinePoint`: `{"date", "nav", "source"}`).

### Verdict on Main's already-completed work

**SOUND with one minor gap.** The 5-test file correctly tests
the structural invariant (3-way NAV match), the response shape
(window + series + nav + date), the cross-endpoint check
(kill-switch vs portfolio), and the source-code wiring check
(red-line route exists). The 1% tolerance is **acceptable for
the cross-source case but too loose for the same-source case
(kill-switch vs portfolio)**. Recommended single-line tightening:
add a separate assertion in the kill-switch test with a 1bp
tolerance for the same-source comparison.

The test file is NOT broken and the spawn-brief evidence is
real (all three live endpoints returned 23184.7 today), so this
is a "harden the test for tomorrow" finding, not a "the test
PASSed for the wrong reason today" finding. Status: PASS with a
follow-up note.

## Section D -- Recency scan (last 2 years)

Searched the 2024-2026 window for new findings on:

- "test invariant NAV consistency endpoints REST API trading
  dashboard 2024 2025" -- no new findings. Eventual-consistency-
  in-REST discussions (Bogard 2013, Pillopl 2017) remain the
  canonical pattern; no NAV-specific paper or framework
  surfaced.
- "portfolio reconciliation tolerance break investment
  operations" -- Limina blog (2024) and Solvexia / FundRecs
  glossaries (undated; appear post-2023) restate the same
  industry framework as 2017-era ISDA guidance. No newer
  numeric tolerance standard has been published.

**Recency-scan result: no new findings in the 2024-2026 window
that supersede the canonical sources above.** The Fidelity and
NYIF pieces are older but remain authoritative; the Limina and
CrossTrade pieces are within the window.

## Section E -- Search queries run

1. **Current-year frontier:** "test invariant NAV consistency
   endpoints REST API trading dashboard 2024 2025" --
   year-locked, surfaced recent REST eventual-consistency
   discussions but no NAV-specific 2024-2026 finding.
2. **Last-2-year window:** "NAV reconciliation tolerance fund
   accounting cross-source invariant 2026" -- surfaced Limina
   2024, Solvexia, FundRecs.
3. **Year-less canonical:** "portfolio reconciliation tolerance
   break investment operations" -- surfaced ISDA Portfolio
   Reconciliation guidance and Limina + Solvexia as canonical
   prior-art. Properly hit older + newer mix.
4. Additional thematic: "GIPS standards NAV reconciliation
   valuation tolerance basis points" (compliance framing);
   "paper trading dashboard NAV cache staleness divergence test
   invariant" (system framing); "portfolio NAV cross-check kill
   switch threshold drawdown trading system" (kill-switch
   framing); "end-of-day mark-to-market snapshot vs live NAV
   reconciliation timing drift" (timing framing). All three
   variants of the three-query discipline covered.

## Section F -- JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 8,
  "urls_collected": 13,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "gate_passed": true
}
```

Internal files inspected: `backend/api/sovereign_api.py`,
`backend/api/paper_trading.py`,
`backend/services/paper_trader.py`,
`backend/services/kill_switch.py`,
`backend/services/paper_metrics_v2.py` (snippets),
`backend/tests/test_phase_23_2_7_red_line_nav_match.py`.

## Section G -- Application notes for pyfinagent

### Recommended test refinements (optional follow-ups, NOT blockers)

1. **Tighten the kill-switch-vs-portfolio same-source check** to
   `delta_pct < 0.0001` (1bp). They read the same BQ row; any
   divergence is a real bug.
2. **Add a `source` field assertion** to the red-line last-point
   test:
   ```python
   today = datetime.now(timezone.utc).date().isoformat()
   if series[-1]["date"] == today:
       # Today's actual snapshot must have source="actual" for
       # the strict tolerance to apply.
       if series[-1]["source"] == "actual":
           assert delta_pct <= 0.01  # 1bp -- same-cycle write
       else:
           assert delta_pct <= NAV_MATCH_TOLERANCE_PCT
   ```
3. **Add a "snapshot duplicates" check**: query
   `paper_portfolio_snapshots` directly and assert
   `COUNT(*) == COUNT(DISTINCT snapshot_date)` over the last
   30 days. Catches the `MAX(total_nav)` defense-in-depth code
   path silently masking duplicate-row corruption (L148-152
   comment acknowledges this exact failure mode is being
   defended against).
4. **Document the cache-coherence gap**: red-line cache (60s)
   does NOT get invalidated when `paper-trading/portfolio`
   writes. Decide: either add an invalidation hook (cache.set
   on portfolio writes triggers cache.invalidate
   `sovereign:red-line:*`), or document that the 60s TTL is an
   accepted bound.

### Map to verification command

The masterplan verification command is satisfied today (last
point NAV 23184.7 == portfolio.total_nav 23184.7, exact). Main's
5 tests will keep passing as long as the autonomous daily cycle
runs `mark_to_market` -> `save_daily_snapshot` in sequence
(coupled in `backend/autonomous_loop.py`). The test correctly
catches the failure mode where one of those calls breaks.

### Anti-patterns to avoid in follow-up code

- Do NOT change `MAX(total_nav)` GROUP BY to ANY_VALUE in
  `_fetch_snapshots` (sovereign_api.py L156) -- it's the existing
  defense against legacy duplicate rows per the phase-23.1.18
  comment.
- Do NOT bypass `mark_to_market` when mutating cash. The
  `adjust_cash_and_mtm` helper (paper_trader.py L872-) exists
  specifically because the phase-23.1.15 cleanup script broke
  this invariant -- "stale total_nav silently broke the
  home-cockpit Red Line".

## Section H -- Protocol-discipline correction (for harness log)

This researcher was spawned **AFTER** Main wrote
`backend/tests/test_phase_23_2_7_red_line_nav_match.py` and after
the 4-PASS-1-SKIP test run. The canonical per-step order is:
research -> contract -> generate -> qa
(`feedback_contract_before_generate`). Main wrote contract
implicitly via the masterplan spec and went straight to
GENERATE, then spawned researcher when reminded of the gate
discipline rule. This is a **protocol breach** per
`feedback_never_skip_researcher` (operator override
2026-05-22).

**What was done before the spawn:**
- 5-test file written + executed.
- Live curl probes ran; NAV matched at 23184.7 across all three
  endpoints (red-line last point, paper-trading portfolio, kill-
  switch current_nav).
- Spawn brief stated: "Researcher MUST spawn. NOTE: Main
  bypassed the gate then spawned retroactively".

**Correction in this brief:**
- Verified the test's invariant is sound (see Section C
  verdict: SOUND).
- Identified 5 coverage gaps (Section C, items 1-5), four of
  which are minor follow-ups, one of which (#4: tighten
  kill-switch-vs-portfolio to 1bp) is recommended as a
  one-line strengthening before the step is marked done.
- Identified the cache-coherence gap (Section G item 4) as a
  documentation finding.

**Recommendation to harness log:** record this as
"researcher spawned retroactively after GENERATE; verdict on
already-completed work was SOUND; one optional tightening
recommended". The protocol breach is real but the GENERATE
output happens to be defensible -- the rigor failure here is
discipline (order), not correctness (output).

**One-paragraph protocol-discipline note for the harness log:**

> phase-23.2.7 GENERATE was performed BEFORE researcher spawn,
> in violation of the research-gate discipline codified by
> `feedback_never_skip_researcher` (operator override
> 2026-05-22) and `.claude/rules/research-gate.md`. Researcher
> was spawned retroactively in real-time and verified the
> 5-test invariant is SOUND (3-way NAV cross-check with 1%
> tolerance is acceptable for the cross-source case; verdict
> details in `handoff/current/research_brief_phase_23_2_7.md`
> Section C). One optional tightening recommended (kill-switch
> vs portfolio: 1bp same-source tolerance rather than 1%
> cross-source). No GENERATE rework required; this is a process
> note for the cycle log, not a verdict-changing finding.

---

## Sources

- [NAV and P&L Reconciliation: A Comprehensive Guide -- Limina](https://www.limina.com/blog/pnl-and-nav-reconciliation-guide)
- [Understanding an ETF's NAV -- Fidelity](https://www.fidelity.com/learning-center/investment-products/etf/etfs-nav)
- [Trading System Kill Switch: Panacea or Pandora's Box? -- NYIF](https://www.nyif.com/articles/trading-system-kill-switch-panacea-or-pandoras-box)
- [NinjaTrader Trailing Drawdown Survival Guide -- CrossTrade](https://crosstrade.io/learn/risk-management/trailing-drawdown-survival-guide)
- [What is Mutual Fund Reconciliation? -- Solvexia](https://www.solvexia.com/glossary/mutual-fund-reconciliation)
- [Difference Between NAV and Market Value Reconciliation -- Nasdaq](https://www.nasdaq.com/articles/difference-between-nav-and-market-value-reconciliation-2016-01-15)
- [NAV Reconciliation -- FundRecs](https://www.fundrecs.com/nav-reconciliation)
- [Alpaca Paper Trading docs](https://docs.alpaca.markets/us/docs/paper-trading)
- [GIPS Calculation Methodology -- GIPS Standards](https://www.gipsstandards.org/wp-content/uploads/2021/02/calculation_methodology_2.pdf)
- [Are trading invariants really invariant? -- arXiv 1902.03457](https://arxiv.org/abs/1902.03457)
- [ISDA Portfolio Reconciliation In Practice](https://www.isda.org/a/crpTE/Portfolio-Reconciliation-In-Practice.pdf)
- [PNL Drawdown Kill Switch Example -- NinjaTrader](https://ninjatraderecosystem.com/user-app-share-download/pnl-drawdown-kill-switch-example-for-strategy-builder/)
- [Optimal timing of mark-to-market -- arXiv math/0506077](https://arxiv.org/abs/math/0506077)
