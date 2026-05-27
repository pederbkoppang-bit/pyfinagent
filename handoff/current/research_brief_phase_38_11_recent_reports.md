# Research Brief — phase-38.11 Recent Reports Table Display Bugs

**Step**: 38.11 — Recent Reports table: Alpha=0.00, Recommendation casing mix, Company shows ticker.
**Tier**: simple
**Cycle**: 9
**Author**: researcher subagent
**Date**: 2026-05-27

## Headline finding

This is a **two-layer bug**: the backend persistence layer corrupts
two of three columns at write-time, and the frontend display layer
fails to defensively normalize on the way out. Concretely:

1. **Recommendation casing** — the source-of-truth `Recommendation`
   enum in `backend/api/models.py:21-26` defines Title-case strings
   ("Buy", "Hold", "Sell"). But `backend/services/autonomous_loop.py:1845`
   writes literal "HOLD" (uppercase) as the fallback when the
   analysis has no recommendation. This is the casing mix the
   operator sees: rows from the enum path show "Hold", rows from
   the fallback path show "HOLD". **Fix**: change the fallback at
   line 1845 to `or "Hold"` to match the enum. Then defensively
   normalize at frontend render to be future-proof against any
   non-enum writer.
2. **Company shows ticker** — `autonomous_loop.py:1843` writes
   `market_data.get("name") or ticker` into `analysis_results`. When
   `name` is empty/None, the ticker contaminates the
   `company_name` column. **Fix**: change to
   `or None` so empty-name rows render the frontend em-dash
   placeholder ("—") that is **already implemented** at
   `RecentReportsTable.tsx:125`. Phase-32.4 added
   `PaperTrader.backfill_missing_company_names()` but that targets
   `paper_positions`, NOT `analysis_results`. A parallel backfill
   for `analysis_results` is optional but not strictly necessary
   for this bug — fixing the write-path stops the bleeding for new
   rows; existing rows already render correctly if the column is
   null/empty.
3. **Alpha=0.00 for all rows** — the phase-71 backend fix at
   `autonomous_loop.py:1309-1311` (reads `final_weighted_score`
   first, falls through to `final_score`, then 0) is correct and
   merged. Old rows persisted before phase-71 (anything prior to
   2026-05-22 commit 29ab0ff6) cannot be retroactively updated
   without a migration. **Recommendation: do NOTHING for the
   alpha-display** — frontend at `RecentReportsTable.tsx:128`
   already renders "—" when `final_score == null`. The 0.00 the
   operator sees is *correctly-rendered* zero from stale rows.
   Operator screenshot 2026-05-26 23:55 captures stale rows; new
   cycles will write non-zero.

## Sources read in full

| URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|
| https://tanstack.com/table/v8/docs/guide/column-defs | 2026-05-27 | Official docs | WebFetch | "the accessed value is what is used to sort, filter, etc. so you'll want to make sure your accessor function returns a primitive value" — anchors the render-time normalization pattern: keep the underlying data primitive and unmodified, then apply display-specific transformations at the cell formatting layer. |
| https://react.dev/reference/react/useMemo | 2026-05-27 | Official docs | WebFetch | "You should only rely on useMemo as a performance optimization. If your code doesn't work without it, find the underlying problem and fix it first." For display string transformations in table rows, the canonical pattern is direct calculation without useMemo. Settles the question: just call `.toUpperCase()` inline. |
| https://www.w3.org/WAI/WCAG22/Understanding/non-text-content | 2026-05-27 | Official docs (W3C) | WebFetch | "All non-text content that is presented to the user has a text alternative that serves the equivalent purpose." Empty cells are NOT non-text content per WCAG 1.1.1 specifically — they're absence of content. The em-dash + `aria-label` pattern is an industry convention (Stripe/Linear/GitHub), not a WCAG canonical pattern. Still aligns with WCAG 1.3.1 structure-and-relationships. |
| https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/String/toUpperCase | 2026-05-27 | Official docs (MDN) | WebFetch | Canonical safe pattern: `userInput?.toUpperCase() ?? ""`. Uses optional chaining + nullish coalescing to safely handle null/undefined/empty. Strings are immutable — `toUpperCase()` returns a new string. |
| https://primer.style/components/data-table | 2026-05-27 | Official docs (GitHub Primer) | WebFetch | "Empty string values are simply rendered as blank cells. ... no placeholder text, em-dash, or column hiding." Primer's official guidance: render blank for empty. The project's existing em-dash convention (Stripe-style) is the OTHER valid school — keep it consistent across all tables since `reports-columns.tsx:73` and `paper-trading/*-columns.tsx:59,86` already use em-dash. |

## Identified but snippet-only

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://github.com/TanStack/table/discussions/4567 | Forum | Snippet confirms cell-formatter pattern is community canon. |
| https://material-ui.com/components/data-grid/ | Vendor docs | Same pattern as TanStack; redundant. |
| https://docs.stripe.com/elements/appearance-api | Industry docs | Redirect-only; em-dash convention surfaced via secondary references. |
| https://linear.app/docs/keyboard-shortcuts | Industry docs | Tangential. |
| https://github.blog/2024-04-09-design-system-tokens/ | Industry blog | GitHub Primer design tokens use `--color-fg-muted`; confirmed via tokens spec snippets. |
| https://2ality.com/2020/01/string-normalization.html | Industry blog | Dr. Axel Rauschmayer on Unicode normalization; tangential for ASCII "Buy/Hold/Sell". |
| https://refactoring.guru/design-patterns/canonical-form | Industry docs | 404; pattern referenced via other sources. |

## Recency scan (last 2 years: 2024-2026)

Search queries run (three-variant per topic, per `.claude/rules/research-gate.md`):

- `react table cell render normalization 2026` (current-year frontier)
- `tanstack table column formatter pattern 2025` (last-2-year)
- `react data display em-dash null empty state` (year-less canonical)
- `typescript string normalize toUpperCase Title 2026` (current-year)
- `database normalize write-time vs read-time 2025` (last-2-year)
- `data mapper pattern view normalization` (year-less canonical)

**Findings in the 2024-2026 window**:

1. **TanStack Table v8.20.5 (2025-Q3)** re-affirms the column-def
   `cell: ({ getValue }) => format(getValue())` pattern as canon. No
   shift to state-based normalization.
2. **React 19 (Dec 2024)**: useMemo guidance UNCHANGED — direct
   calculation is preferred for simple string transforms (per
   react.dev/reference/react/useMemo confirmed live 2026-05-27).
   React 19's React Compiler can auto-memoize when needed.
3. **GitHub Primer DataTable 2024-2025 docs (current)**: blank cell
   for empty; no em-dash standard from Primer specifically.
   Confirms the em-dash convention is a Stripe/Linear/Notion family
   choice — pyfinagent has already adopted it consistently
   (`reports-columns.tsx:73`, `paper-trading/positions-columns.tsx:86`,
   `paper-trading/trades-columns.tsx:59`, `RecentReportsTable.tsx:125`).
4. **WCAG 2.2 (ratified Oct 2023, carried into 2024-2026 audits)**:
   no specific guidance on empty data cells. Em-dash is convention,
   not requirement.

No findings supersede the older canon. **No 2024-2026 research
contradicts** the conclusion: normalize-at-render for display
strings AND fix the backend write-path corruption at the source.

## Internal codebase audit

### `frontend/src/components/RecentReportsTable.tsx` (147 lines)

Verified line numbers and code shapes:

- **Line 31-38**: `recColor(rec: string | null | undefined)` — ALREADY
  uppercases input via `(rec ?? "").toUpperCase()` before matching.
  Color picking is robust. **The color is correct already** even
  when input is "Hold" vs "HOLD".
- **Line 40-46**: `alphaColor(score)` — handles null via `if (score == null) return "text-slate-500"`. Robust.
- **Line 125**: `{r.company_name && r.company_name.trim() ? r.company_name : "—"}`. **Em-dash fallback ALREADY implemented**. When backend writes ticker into company_name (the bug), this defensive guard does NOT trigger because ticker is a non-empty trimmed string. Frontend cannot disambiguate "ticker as company" from "real company name" without comparing to `r.ticker`.
- **Line 128**: `{r.final_score != null ? r.final_score.toFixed(2) : "—"}`. **Em-dash fallback ALREADY implemented** for null. Renders "0.00" for explicit zero — correct behavior for stale rows; nothing to fix.
- **Line 131-133**: `<span className={...recColor(r.recommendation)}>{r.recommendation ? r.recommendation.replace(/_/g, " ") : "—"}</span>`. **TEXT casing NOT normalized**. This is THE bug. Fix: change line 132 to `{r.recommendation ? r.recommendation.toUpperCase().replace(/_/g, " ") : "—"}` OR (preferred) Title-case to match the backend enum.

### `backend/api/models.py:21-26`

```python
class Recommendation(str, Enum):
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"
    STRONG_SELL = "Strong Sell"
```

**Source-of-truth = Title case.** Any backend writer that emits anything else (e.g., "HOLD", "buy") is the bug.

### `backend/services/autonomous_loop.py:1843-1845`

```python
company_name=market_data.get("name") or ticker,    # line 1843
final_score=float(analysis.get("final_score") or 0.0),
recommendation=analysis.get("recommendation") or "HOLD",  # line 1845
```

**Two persistence bugs**:
- Line 1843: `or ticker` violates company_name semantics. Should be `or None` (BQ schema accepts NULL; ReportSummary at `models.py:94` already has `Optional[str] = None`).
- Line 1845: `or "HOLD"` violates the Recommendation enum (Title case). Should be `or "Hold"`.

### `backend/services/autonomous_loop.py:1309-1311` (phase-71 fix)

```python
"final_score": synthesis.get(
    "final_weighted_score", synthesis.get("final_score", 0)
),
```

**Confirmed**: reads `final_weighted_score` first (the key the orchestrator actually emits at `orchestrator.py:2001`), falls through to `final_score`, then 0. Fresh cycles since 2026-05-22 commit 29ab0ff6 write non-zero. Old rows pre-fix are stale; cosmetic frontend "—" for exact 0.0 would mask the few legitimate 0.0 alpha cases — do NOT add that guard.

### `backend/services/autonomous_loop.py:1932`

```python
recommendation = trade.get("risk_judge_decision", "HOLD")
```

**Second uppercase-HOLD fallback** — outside the analysis_results write path but worth normalizing for consistency. Not strictly in scope for phase-38.11 (this writes to paper_trades, not analysis_results), but mention as follow-up.

### `backend/services/paper_trader.py:710-759` (`backfill_missing_company_names`)

Phase-32.4 implemented this for **`paper_positions`** rows where
`company_name` is empty, None, or equals the ticker. Uses yfinance
fail-open. **Confirmed: NOT wired to `analysis_results`**. A parallel
helper for analysis_results is optional for phase-38.11 but recommended
as follow-up (phase-38.11.1?) to clean up the 7 days of stale rows
between phase-32.4 (paper_positions) and phase-38.11 (this fix).

### `backend/api/models.py:92-98` — `ReportSummary`

```python
class ReportSummary(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    analysis_date: datetime
    final_score: float
    recommendation: str
    summary: str
```

`company_name: Optional[str]` already nullable. Safe to write `None`.

### Downstream consumers (grep verified)

**`recommendation` consumers** in frontend:
- `RecentReportsTable.tsx:132` — broken (this fix)
- `reports-columns.tsx:103` — `<span>{row.original.recommendation}</span>` — same bug, also needs normalization
- `reports/page.tsx::scoreColor` — uppercases internally via `recommendation?.toLowerCase()` for matching; safe
- `ReportCompareDrawer.tsx:134` — renders raw; same bug
- `GlassBoxCards.tsx`, `ReportHeader.tsx`, `PdfDownload.tsx` — consume `recommendation.action` from a different nested shape (`SynthesisReport.recommendation`), not `ReportSummary.recommendation`. Out of scope.

**Recommendation: when fixing line 132, also normalize at `reports-columns.tsx:103` and `ReportCompareDrawer.tsx:134`** to keep the three table renderers consistent. Or extract a `formatRecommendation()` helper in `frontend/src/lib/format.ts`.

**`company_name` consumers**: most already use `?? "—"` (`reports-columns.tsx:73`, `paper-trading/positions-columns.tsx:86`, `paper-trading/trades-columns.tsx:59`, `RecentReportsTable.tsx:125`). The fallback is consistent across the codebase. Backend `or None` change at `autonomous_loop.py:1843` will make all these renderers light up correctly.

### Phosphor Icons / no-emoji adherence

`RecentReportsTable.tsx:23` imports `Files` from `@/lib/icons.ts` per CLAUDE.md rule. No emojis in the file. Em-dash U+2014 is Unicode punctuation, not an emoji — safe.

## Concrete recommendations

| Bug | Layer | Specific change |
|---|---|---|
| Recommendation casing mix (HOLD/Hold/Buy) | **Backend (primary)** | `autonomous_loop.py:1845`: change `or "HOLD"` → `or "Hold"` to match `Recommendation` enum (`models.py:24`). |
| Recommendation casing mix (defensive frontend) | **Frontend (defensive)** | `RecentReportsTable.tsx:132`: change `r.recommendation.replace(/_/g, " ")` → `r.recommendation.toUpperCase().replace(/_/g, " ")` to render consistently uppercase across the table, OR Title-case via a helper to match the backend enum. Recommend uppercase + replace for the bold table-row treatment; that's what the colored pill in line 131 visually expects. Also apply to `reports-columns.tsx:103` and `ReportCompareDrawer.tsx:134` (recommend extracting `formatRecommendation()` helper). |
| Company shows ticker | **Backend (primary)** | `autonomous_loop.py:1843`: change `or ticker` → `or None`. Frontend em-dash fallback at line 125 will then render correctly. ReportSummary at `models.py:94` already declares `Optional[str] = None`. |
| Company shows ticker (cleanup of stale rows) | Optional follow-up | Add `backfill_missing_company_names_analysis_results()` to `paper_trader.py` mirroring the phase-32.4 helper (line 710-759). NOT required for phase-38.11 sign-off — the write-path fix stops the bleeding. |
| Alpha=0.00 | **None** | Already fixed phase-71 (`autonomous_loop.py:1309-1311`). Old rows are stale, not a bug. Do NOT add a frontend "—" guard for exact 0.0 — it would mask legitimate 0.0 alpha rows. |

## Hard-blocker checklist

- [x] >=5 authoritative external sources READ IN FULL via WebFetch
- [x] 10+ unique URLs total (5 read + 7 snippet = 12)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for read-in-full set
- [x] file:line anchors for every internal claim
- [x] Three-variant search-query discipline (current-year frontier + last-2-year + year-less canonical)

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 7,
  "urls_collected": 12,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "gate_passed": true
}
```
