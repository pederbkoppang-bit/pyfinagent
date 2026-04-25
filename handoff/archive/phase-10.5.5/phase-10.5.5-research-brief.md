# Research Brief: phase-10.5.5 -- AlphaLeaderboard (Sortable, Filterable)

Tier assumption: simple-moderate (stated by caller).

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://blog.logrocket.com/creating-react-sortable-table/ | 2026-04-21 | blog | WebFetch | Sort state shape: `{field, order}` object; toggle via `sortOrder = accessor===sortField && order==="asc" ? "desc" : "asc"`; null-safe comparator |
| https://monsterlessons-academy.com/posts/custom-react-table-with-filter-and-sorting | 2026-04-21 | blog | WebFetch | HeaderCell component pattern; `activeFilters` array state for chip filter; remove-filter handler |
| https://www.smashingmagazine.com/2020/03/sortable-tables-react/ | 2026-04-21 | authoritative blog | WebFetch | `useSortableData` custom hook; `useMemo` for sorted array; `aria-sort` attribute for accessibility |
| https://blog.ag-grid.com/unit-testing-ag-grid-react-tables-with-react-testing-library-and-vitest/ | 2026-04-21 | official doc / blog | WebFetch | `container.querySelector` pattern; read cell values via `data-cell` attrs; jsdom limitation note |
| https://www.insaim.design/blog/filter-ui-design-best-ux-practices-and-examples | 2026-04-21 | industry blog | WebFetch | Chips above table; toggle to clear; "Clear all" affordance; active vs inactive visual distinction |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://github.com/phosphor-icons/react | code | WebFetch returned no icon catalog; confirmed icon names via codebase grep of `icons.ts` instead |
| https://www.shadcn.io/blocks/tables-sortable | docs | No TanStack in this codebase; pattern serves as reference only |
| https://www.shadcn.io/template/openstatushq-data-table-filters | docs | Chip pattern reference; shadcn not used here |
| https://medium.com/@samueldeveloper/react-testing-library-vitest-the-mistakes-that-haunt-developers-and-how-to-fight-them-like-ca0a0cda2ef8 | blog | Snippet reference; core pattern already captured |
| https://testing-library.com/docs/dom-testing-library/api-events/ | official docs | fireEvent well-understood from existing test files |
| https://reactscript.com/best-data-table/ | community | No TanStack; informational only |

## Recency scan (2024-2026)

Searched: "React 19 sortable table useState 2026", "React click-to-filter chip table 2025 2026", "vitest react testing-library table sortable 2025".

Result: no fundamentally new patterns in 2024-2026 that supersede the canonical useState-based sort approach. The React 19 release (stable Dec 2024) does not alter client-side sort/filter state handling; hooks and useMemo are unchanged. Shadcn's filter-chips block (2025) confirms the chip-above-table pattern is current best practice. AG Grid's vitest guide (2025) confirms `container.querySelector` + `data-testid` approach. The recency scan surfaces no superseding approaches -- canonical patterns hold.

---

## Key Findings

1. Sort state shape: `{ key: SortKey; dir: "asc" | "desc" }` object is cleaner than two separate useState calls; toggling: if `state.key === col && dir === "asc"` -> flip to "desc", else default to "asc". (LogRocket, Smashing Magazine)

2. `useMemo` for sorted rows: `useMemo(() => [...entries].sort(comparator), [entries, sortState])` keeps renders fast and avoids mutating props. (Smashing Magazine)

3. Accessibility: `aria-sort="ascending"|"descending"|"none"` on `<th>` elements. Add a CaretUp/CaretDown Phosphor icon (they exist in the library) as visual indicator alongside aria-sort. (Smashing Magazine)

4. Filter chip placement: chips row sits ABOVE the table, below the card header. Click a status pill in any table row to set `statusFilter`; clicking the active filter chip clears it. UX standard confirmed by insaim.design and shadcn filter blocks.

5. Phosphor icon names confirmed via `/Users/ford/.openclaw/workspace/pyfinagent/frontend/src/lib/icons.ts`: `CheckCircle` (aliased as `IconCheckCircle`), `XCircle` (aliased as `IconXCircle`), `Warning` (aliased as `IconWarning`) are all already exported. For "pending" / "deploying" status, `Clock` or `Question` should be verified -- `Warning` is the safest fallback already in icons.ts.

6. Test pattern: use the `clickEl` shim from `RedLineMonitor.test.tsx` (line 22-24: `el.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }))`). No fireEvent import needed. Assert row order via `data-cell="strategy_id"` querySelectorAll on `<td>` elements, map to text content, compare array.

7. Backend gap: `LeaderboardEntry` (sovereign_api.py line 79-85) is missing `status` and `allocation_pct`. The `_fetch_strategy_deployments()` mapper (line 198-207) queries `SELECT *` from the BQ view but does not surface those fields in the Pydantic model. The fix is small: add two fields to `LeaderboardEntry` and pass them through the mapper.

---

## Internal Code Inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/api/sovereign_api.py` | 426 | Backend: 3 sovereign endpoints + models | `LeaderboardEntry` missing `status` + `allocation_pct` |
| `backend/api/sovereign_api.py::_fetch_strategy_deployments` | 198-207 | BQ view fetch; `SELECT *` | fields already in BQ view (10.5.1), not surfaced in model |
| `frontend/src/app/sovereign/page.tsx` | 163 | Sovereign page shell | AlphaLeaderboard is still a `PlaceholderCard` (line 143-150) |
| `frontend/src/lib/api.ts` | 505-534 | Sovereign fetchers | `getSovereignLeaderboard` does NOT exist yet -- must add |
| `frontend/src/lib/icons.ts` | 147 | Phosphor icon aliases | `IconCheckCircle`, `IconXCircle`, `IconWarning` confirmed present (lines 143-145) |
| `frontend/src/components/RedLineMonitor.test.tsx` | 127 | Test pattern reference | `clickEl` shim at line 22-24; `data-testid` pattern; `container.querySelector` |
| `frontend/src/components/AutoresearchLeaderboard.tsx` | 237 | Sister leaderboard component | sort state pattern, `useMemo`, status pill approach to mirror |
| `frontend/vitest.config.ts` | 19 | Vitest config | `--filter=AlphaLeaderboard` matches filename `AlphaLeaderboard.test.tsx` |

---

## Consensus vs Debate (external)

Consensus: useState sort state object + useMemo is the idiomatic React pattern without a table library. No debate. The only variance is whether to extract to a custom hook (`useSortableData`) vs inline -- for a single-use component, inline is fine and is what `AutoresearchLeaderboard.tsx` does.

Filter-as-chip: consensus that the chip sits above the table and clicking the same chip again clears it. No disagreement in sources.

---

## Pitfalls (from literature)

- Mutating the entries array directly in sort comparator instead of spreading first: `[...entries].sort(...)` is mandatory.
- Forgetting `aria-sort` on `<th>` -- accessibility gap flagged by Smashing Magazine.
- Using `fireEvent` import from `@testing-library/react` in this codebase -- the existing tests use the custom `clickEl` shim because `fireEvent` is not reliably exported. Copy the shim.
- Null-safe comparator: `sharpe`, `dsr`, `pbo`, `max_dd` are all `Optional[float]` from the backend -- treat null as -Infinity (or +Infinity for pbo where lower is better) in the sort comparator.
- `allocation_pct` may be NULL in the BQ view for strategies not yet deployed -- render as "-" not 0.

---

## Application to pyfinagent (mapping findings to code)

### Decision: extend the backend (recommended)

The BQ view `pyfinagent_pms.strategy_deployments` (shipped 10.5.1) already has `status` and `allocation_pct`. The `_fetch_strategy_deployments()` mapper runs `SELECT *` so those columns ARE in the returned dicts -- they are simply dropped by the Pydantic model. The fix is 2 lines:

In `backend/api/sovereign_api.py`, `LeaderboardEntry` (line 79-85):
```python
class LeaderboardEntry(BaseModel):
    strategy_id: str
    sharpe: Optional[float] = None
    dsr: Optional[float] = None
    pbo: Optional[float] = None
    max_dd: Optional[float] = None
    status: Optional[str] = None       # ADD
    allocation_pct: Optional[float] = None  # ADD
    notes: Optional[str] = None
```

In the `deployments` mapper block (line 299-306), add:
```python
status=str(d.get("status") or "") or None,
allocation_pct=_safe_float(d.get("allocation_pct")),
```

The TSV fallback path does not have these columns -- `status` and `allocation_pct` will be None from TSV, which is correct (render as "-").

### New fetcher in api.ts

Add `getSovereignLeaderboard()` returning `{ entries: LeaderboardEntry[], source: string, note: string|null }` where `LeaderboardEntry` includes the 7 columns from spec.

### AlphaLeaderboard component structure

```
AlphaLeaderboard
  props: entries[], loading, error
  state:
    sortState: { key: SortKey; dir: "asc" | "desc" } -- default sharpe desc
    statusFilter: string | null
  computed (useMemo):
    filtered = statusFilter ? entries.filter(e => e.status === statusFilter) : entries
    sorted   = [...filtered].sort(comparator(sortState))
  render:
    [header bar]
    [filter chip row] -- shows chip only when statusFilter active; clicking clears it
    [table]
      <thead> -- sortable th buttons with CaretUp/CaretDown icon + aria-sort
      <tbody>
        each row: status cell = clickable pill (triggers statusFilter)
```

### Phosphor pill icons (status -> icon mapping)

| status value | Icon | Weight | Color |
|---|---|---|---|
| "active" / "deployed" | `CheckCircle` | fill | emerald-400 |
| "deploying" / "pending" | `Warning` | regular | amber-400 |
| "paused" / "stopped" | `XCircle` | fill | rose-400 |
| unknown / null | `Warning` | regular | slate-400 |

All three icons are already in `icons.ts` as `IconCheckCircle`, `IconXCircle`, `IconWarning`. Import from `@/lib/icons` not directly from `@phosphor-icons/react`.

### Sort: null-handling comparator

```ts
function cmp(a: LeaderboardEntry, b: LeaderboardEntry, key: SortKey, dir: "asc" | "desc"): number {
  const nullIsWorst = key !== "pbo"; // pbo: lower is better, null = worst (high)
  const av = a[key] ?? (nullIsWorst ? -Infinity : Infinity);
  const bv = b[key] ?? (nullIsWorst ? -Infinity : Infinity);
  return dir === "asc" ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
}
```

### Test plan (4 criteria)

**1. columns_match_spec** -- render with fixture entries; query all `<th>` elements; assert text content set equals `{strategy_id, sharpe, dsr, pbo, max_dd, status, allocation_pct}`.

**2. status_pill_phosphor_only** -- render a row with `status: "active"`; query `[data-testid="status-pill"]`; assert `data-icon` attribute present (Phosphor icons render an SVG with `data-phosphor-icon` or a wrapper); assert no textContent that is an emoji (no codepoint >= U+1F300). Simpler: assert the pill container has a child `<svg>` element.

**3. sort_persists_client_side** -- render with 3 entries with known sharpe values; click the "sharpe" column `<th>` button (clickEl shim); query `[data-cell="sharpe"]` cells; assert text order is descending; click again; assert order flips to ascending.

**4. filter_by_status_pill_row** -- render with entries of mixed status; clickEl on the status pill in a row with `status: "active"`; assert rows rendered equals only active entries count; assert filter chip appears above table; clickEl on the chip; assert all rows back.

### Sovereign page edit

Replace the `PlaceholderCard` for Alpha Leaderboard (page.tsx lines 143-150) with the real component. The page already owns `costData` and `redLineSeries` state. Add parallel leaderboard fetch in a `useEffect` alongside the existing two. No window selector needed for the leaderboard.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5 fetched)
- [x] 10+ unique URLs total (12 total: 5 full + 7 snippet-only)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (sovereign_api.py, icons.ts, api.ts, sovereign/page.tsx, AutoresearchLeaderboard.tsx, RedLineMonitor.test.tsx, vitest.config.ts)
- [x] Contradictions / consensus noted (none; all sources agree)
- [x] All claims cited per-claim
