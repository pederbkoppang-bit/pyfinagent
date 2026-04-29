# Phase-23.1.17 External Research
# NAV Consistency in Financial Dashboards: SSOT Patterns and Snapshot vs Live NAV

Tier assumed: moderate. Accessed: 2026-04-29.

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://www.limina.com/blog/batch-processing-vs-event-driven-data-processing | 2026-04-29 | Industry/OMS-PMS doc | WebFetch full | "A position snapshot of any kind is the result of a batch process of some form; the snapshot is produced at one time, and stored away for subsequent use... such an approach is usually used by accounting systems." Cash mutations not reflected until next batch cycle. |
| https://tanstack.com/query/latest/docs/framework/react/overview | 2026-04-29 | Official docs (TanStack) | WebFetch full | Shared query key = single cache entry across all pages. "When one route updates the data, other routes automatically reflect those changes without additional requests." Prevents multi-view NAV divergence via query key identity. |
| https://swr.vercel.app/docs/mutation | 2026-04-29 | Official docs (Vercel) | WebFetch full | `mutate('/api/portfolio/nav')` triggers refetch across all mounted consumers sharing that key. "Using global mutator only with the key parameter will not update the cache or trigger revalidation unless there is a mounted SWR hook using the same key." |
| https://www.limina.com/ibor-investment-book-of-record | 2026-04-29 | Industry/IBOR doc | WebFetch full | "A live-extract IBOR doesn't store inventory at all — it doesn't keep positions or cash balances... transactions flow into the system continuously." Cash mutations that bypass position revaluation create incomplete state. Generation 3 systems reconstruct any portfolio view from the transaction ledger on demand. |
| https://bennettfinancials.com/nav-calculation-investment-funds/ | 2026-04-29 | Industry/finance blog | WebFetch full | "Integrated systems synchronize data in real time, eliminating manual errors and reducing reporting delays" — real-time computation increasingly best practice. Consistency across periods requires documented valuation methods applied uniformly. |
| https://www.limina.com/blog/portfolio-management-software-pms | 2026-04-29 | Industry/PMS doc | WebFetch full | PMS pattern: "Process for cutting a valuation and position snapshot at the NAV time (end-of-day or intraday). Storage of NAV timeseries to be used for reporting." Two-phase: compute snapshot → store timeseries. Snapshot validity ends when a cash mutation is applied out-of-band. |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://dev.to/whoffagents/react-query-vs-swr-in-2026-what-i-actually-use-and-why-3362 | Blog | Comparison article — TanStack docs are authoritative source |
| https://medium.com/@ignatovich.dm/using-swr-and-react-query-for-efficient-data-fetching-in-react-87f4256910f0 | Blog | Covered by official docs |
| https://refine.dev/blog/react-query-vs-tanstack-query-vs-swr-2025/ | Blog | Comparison; less authoritative than official docs |
| https://tanstack.com/query/v5/docs/framework/react/guides/does-this-replace-client-state | Official docs | Supplemental; main overview doc covers the key claim |
| https://navia.co.in/blog/trading-vs-live-trading/ | Blog | Paper trading intro — not authoritative on NAV architecture |
| https://phoenixstrategy.group/blog/nav-calculation-challenges-and-solutions | Blog | NAV challenges — covered by more authoritative Limina/Bennett sources |
| https://medium.com/@codeandbird/the-ultimate-guide-to-react-server-state-fetch-vs-react-query-vs-swr-b0633908194f | Blog | Covered by official docs |
| https://www.limina.com/blog/ems-vs-oms-vs-pms | Industry | System comparison — supplemental |
| https://github.com/TanStack/query/discussions/2852 | Community/GitHub | Using React Query for global state — covered by official docs |
| https://dev.to/iammuhammadarslan/mastering-react-hooks-from-basics-to-custom-hooks-2026-guide-34jc | Blog | 2026 custom hooks overview — supplemental |

---

## Recency scan (2024-2026)

Searched: "React Query TanStack Query global state shared across pages NAV portfolio 2025", "single source of truth financial dashboard NAV consistency React Query SWR 2026", "SWR React cross-page cache invalidation stale data shared query key pattern".

Result: The 2024-2026 window confirms the pattern established by TanStack Query v5 (released 2023, now canonical) and SWR. No new architecture supersedes the "shared query key = single cache entry" paradigm for cross-page consistency. The Limina IBOR content (2024-2025) reinforces that live-extract / event-driven approaches outperform batch snapshots for consistency. No findings in the window that would change the recommended approach for pyfinagent.

The recency scan also confirms: TanStack Query v5 is the current stable release (2026); SWR v2 is current. Both are in wide production use. No deprecation or breaking API change relevant to the pattern.

---

## Key Findings

### 1. Snapshot-based NAV creates consistency gaps when cash mutates out-of-band

Industry doctrine (Limina IBOR guide; Limina batch vs event-driven): "Generation 1 IBOR builds an intraday view of positions and cash from a start-of-day snapshot... The position data isn't complete, and it's often unclear for portfolio managers what is potentially missing, especially in the cash positions." Any cash mutation (deposit, cleanup refund) that does not flow through the position revaluation pipeline leaves `total_nav` stale until the next scheduled mark-to-market.

In pyfinagent, `paper_portfolio.total_nav` is authoritative only after `mark_to_market()` runs. A raw SQL UPDATE to `current_cash` produces a state where `current_cash != total_nav - positions_value`. The home page exposes this stale `total_nav` directly; the paper-trading page bypasses it by live-deriving NAV from `cash + sum(livePrice * qty)`.

Source: Limina batch vs event-driven, https://www.limina.com/blog/batch-processing-vs-event-driven-data-processing (2026-04-29)

### 2. Live-extract NAV (computed on demand) is the correct pattern for operator dashboards

The IBOR Generation 3 pattern (Limina IBOR guide): "A live-extract IBOR doesn't store inventory at all. Instead, the underlying transactions and cash movements are held, including all amendments made." Any view is reconstructed on demand. This maps directly to the paper-trading page's `useMemo` approach: recompute `liveNav = cash + sum(livePrice * qty)` on every render cycle. The stored `total_nav` is a cache that requires explicit invalidation after any mutation.

Source: Limina IBOR guide, https://www.limina.com/ibor-investment-book-of-record (2026-04-29)

### 3. React custom hooks are the canonical pattern for shared derived state across routes

The React documentation (https://react.dev/learn/reusing-logic-with-custom-hooks) and TanStack Query overview both confirm: extract shared computation into a custom hook or shared query key. A custom hook encapsulates the `useLivePrices` polling + `cash + sum(livePrice * qty)` derivation. Both pages call the hook with the same inputs, producing identical outputs by construction. This eliminates the class of divergence where page A and page B each independently derive a value from different data sources.

Source: TanStack Query overview, https://tanstack.com/query/latest/docs/framework/react/overview (2026-04-29); React docs https://react.dev/learn/reusing-logic-with-custom-hooks

### 4. For polled endpoints, backend-side live NAV has latency tradeoffs

The Bennett Financials NAV guide and Limina PMS doc acknowledge that "real-time computation eliminates errors" but require "integrated systems." For pyfinagent, the `/api/paper-trading/status` endpoint is polled by `OpsStatusBar` on every page load. Making this endpoint call `mark_to_market()` on every request would batch 5-10 yfinance HTTP calls per status poll — expensive for a potentially sub-second polling interval. The right tradeoff: keep live derivation on the frontend (already solved by `useLivePrices`) and reserve backend MtM for the scheduled autonomous cycle. The frontend hook is the right tier for intraday live NAV.

Source: Limina PMS, https://www.limina.com/blog/portfolio-management-software-pms (2026-04-29); Bennett Financials, https://bennettfinancials.com/nav-calculation-investment-funds/ (2026-04-29)

### 5. SWR/React Query shared query key guarantees SSOT — but pyfinagent uses raw fetch, not SWR

pyfinagent's `frontend/src/lib/api.ts` uses plain `apiFetch` (Bearer auth + AbortController). Neither SWR nor TanStack Query is installed. The SSOT pattern therefore cannot be achieved through shared query key cache — it must be achieved through a shared custom hook (`useLiveNav`) that both pages call with the same derivation logic. This is exactly the Fix A proposal. The SWR/TanStack Query finding confirms the principle; the implementation path in pyfinagent is a custom hook rather than a query client.

Source: SWR Mutation docs, https://swr.vercel.app/docs/mutation (2026-04-29)

### 6. Snapshot cadence: once per autonomous cycle is appropriate for historical reporting; live derivation required for operator tiles

Bennett Financials: "Most PE funds calculate NAV quarterly... Mutual funds calculate NAV daily, while hedge funds typically verify it daily or monthly." The paper_portfolio_snapshots table writing once per autonomous cycle (daily) is correct for historical charting. It is not appropriate as the data source for the operator hero tile — that tile should reflect live prices, not yesterday's snapshot. The Limina batch doc is explicit: "Batch systems suit: Final NAV calculations, period-end ledgers, regulatory reporting requiring immutable historical records. Event-driven systems suit: Front Office decision-making requiring complete, timely, accurate intraday portfolio data."

Source: Bennett Financials NAV guide, https://bennettfinancials.com/nav-calculation-investment-funds/ (2026-04-29); Limina batch vs event-driven, https://www.limina.com/blog/batch-processing-vs-event-driven-data-processing (2026-04-29)

---

## Consensus vs debate

**Consensus**: Live-derived NAV for operator dashboard tiles; stored snapshots for historical charting and period reporting. This is the dominant pattern across OMS/PMS literature and React state management literature.

**Debate**: Whether the backend `/status` endpoint should compute live NAV (Fix C) vs the frontend doing it (Fix A+E). Literature (Limina, Bennett) does not prescribe the tier. The pyfinagent-specific constraint is that `/status` is a polled endpoint — making it call yfinance on every poll is a latency/cost concern. Frontend derivation (hook) avoids this. Fix C remains architecturally desirable but operationally expensive given current polling patterns.

---

## Pitfalls from literature

1. **Stale snapshot after out-of-band mutation** (Limina IBOR guide): any direct database cash mutation that bypasses the trading system's revaluation pipeline leaves `total_nav` incorrect until the next MtM. Prophylactic: always call `mark_to_market()` after any manual cash mutation.

2. **Hybrid state: two sources for the same concept** (Limina batch): when a front-office view adds today's transactions on top of yesterday's accounting snapshot, "it's often unclear what is potentially missing." In pyfinagent, the home page reads `total_nav` (BQ snapshot-based) while the paper-trading page derives live. These are two representations of the same concept. Users lose trust when they diverge (as demonstrated by the $1,511 discrepancy).

3. **"Same query key" illusion with raw fetch** (TanStack Query / SWR docs): if different pages call the same endpoint with raw `fetch()` rather than a shared cache, they each get independent HTTP responses and can diverge if the backend state changes between calls. A shared custom hook that computes the same derivation from the same inputs is the correct mitigation when a query client is not installed.

4. **Snapshot finality** (Bennett Financials): "NAV values from prior periods should not change retroactively." Writing a new snapshot row for 2026-04-29 with the post-refund NAV is correct procedure — it creates a new authoritative point in the timeseries. It does NOT retroactively alter prior rows.

---

## Application to pyfinagent (mapping external findings to file:line anchors)

| External finding | Applies to | File:line |
|-----------------|-----------|-----------|
| Cash mutation without MtM leaves `total_nav` stale | `paper_portfolio.total_nav` stale after phase-23.1.15 refund | `backend/services/paper_trader.py:384-395` (mark_to_market recomputes nav) |
| Live-extract IBOR: recompute NAV on demand, not from stored snapshot | Home page uses stored `total_nav`; should use live derivation | `frontend/src/app/page.tsx:141-142` |
| Shared custom hook = SSOT without query client | Extract `useLiveNav` from paper-trading page to shared lib | `frontend/src/app/paper-trading/page.tsx:433-451` → `frontend/src/lib/useLiveNav.ts` |
| Snapshot cadence: daily for history, live for operator tiles | `paper_portfolio_snapshots` correct for chart; wrong for NAV hero | `backend/api/sovereign_api.py:122-147` (snapshot read) vs `backend/api/paper_trading.py:136` (stale total_nav) |
| Mutate + revalidate pattern for cache consistency | Not using SWR/React Query; shared hook is the correct substitute | `frontend/src/lib/api.ts` (raw apiFetch, no shared cache) |

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 sources)
- [x] 10+ unique URLs total (10 snippet-only + 6 read-in-full = 16 total)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (see internal audit document)

Soft checks:
- [x] Internal exploration covered every relevant module (home page, paper-trading page, sovereign_api, paper_trading API, paper_trader service, bigquery_client)
- [x] Contradictions / consensus noted (Fix C debate documented)
- [x] All claims cited per-claim (not just listed in a footer)

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "report_md": "handoff/current/phase-23.1.17-external-research.md",
  "gate_passed": true
}
```
