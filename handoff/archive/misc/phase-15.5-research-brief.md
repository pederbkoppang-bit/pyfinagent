# Research Brief: phase-15.5 -- Weekly Ledger History Viewer

*Tier: simple. Assumed by researcher (caller did not override).*

---

## Executive Summary

Phase 15.5 adds a read-only history viewer for `backend/autoresearch/weekly_ledger.tsv`.
The source file already has a fully functional `read_rows()` helper in `weekly_ledger.py`
(lines 48-68) that is stdlib-only, fail-open, and skips the header automatically.
The new endpoint belongs in the existing `harness_autoresearch.py` router (same `prefix="/api/harness"`)
because it is the established home for all harness-tab data endpoints. The frontend component
should mirror `DemotionAuditTable` in `HarnessDashboard.tsx` (navy palette, Phosphor icon,
empty state, data-* attrs) and must be inserted below the sprint tile per criterion 3.

Key findings: (1) do NOT re-implement row parsing -- use `weekly_ledger.read_rows()` directly;
(2) cap to last 52 rows, newest-first via `rows[-52:][::-1]`; (3) `fri_promoted_ids` and
`fri_rejected_ids` are bracketed strings like `"[a,b,c]"` -- strip `[]` and split on `,` for
display; (4) `thu_batch_id` is a short slug (not a UUID), no truncation needed; (5) horizontal
scroll (`overflow-x-auto`) on the table container handles the 8-column width; (6) the endpoint
must be added to `_PUBLIC_PATHS` in `main.py` following the demotion-audit precedent.

---

## Read in Full (>= 5 required)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://docs.python.org/3/library/csv.html | 2026-04-21 | Official doc | WebFetch | `newline=''` is mandatory; `delimiter='\t'` or `dialect='excel-tab'`; DictReader uses header row as fieldnames automatically |
| https://realpython.com/python-pathlib/ | 2026-04-21 | Authoritative blog | WebFetch | `Path.read_text(encoding="utf-8")` is idiomatic for full-file reads; `open()` + context manager preferred for streaming/complex processing |
| https://fastapi.tiangolo.com/tutorial/response-model/ | 2026-04-21 | Official doc | WebFetch | Wrap a `list[RowModel]` in a `BaseModel(rows=[...])` for the `{rows: [...]}` wire shape; declare `response_model=WrapperModel` on the decorator for OpenAPI docs |
| https://muhimasri.com/blogs/react-sticky-header-column/ | 2026-04-21 | Authoritative blog | WebFetch | `overflow-x-auto` on container + `position: sticky; left: 0` on first column for wide tables; `z-index: 3` on sticky header cell |
| https://en.wikipedia.org/wiki/ISO_week_date | 2026-04-21 | Reference | WebFetch | ISO 8601 week format is `YYYY-Www` (extended) -- the TSV already stores this; display label is simply `week_iso` value verbatim, e.g. "2026-W17" |
| https://fastapi.tiangolo.com/advanced/custom-response/ | 2026-04-21 | Official doc | WebFetch | For maximum performance, use Pydantic `response_model` type annotation -- not a `response_class`; FastAPI uses Pydantic/Rust serialisation path |
| https://www.getorchestra.io/guides/fastapi-response-model-a-comprehensive-guide | 2026-04-21 | Industry blog | WebFetch | Confirms `{rows: list[ItemModel]}` pattern: return a dict from handler, FastAPI coerces it through the `response_model`; no need to construct the outer model explicitly |

---

## Identified but Snippet-Only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://geeksforgeeks.org/python/simple-ways-to-read-tsv-files-in-python/ | Tutorial | Covered by the official csv.html doc |
| https://copyprogramming.com/howto/reactjs-creating-part-of-the-table-scrollable | Blog | Covered by muhimasri sticky-column article |
| https://tanstack.com/table/latest/docs/framework/react/examples/column-pinning-sticky | Library docs | TanStack Table is not in the project stack; plain Tailwind table suffices |
| https://material-react-table.com/docs/guides/sticky-header | Library docs | Not in project stack |
| https://codeling.dev/blog/python-csv-reader-writer-guide/ | Tutorial | Redundant with official csv.html |
| https://pypi.org/project/fastapi/ | Package page | No usage guidance; covered by official docs |
| https://www.epochconverter.com/weeknumbers | Tool | Confirms ISO week numbering but no implementation detail |
| https://www.weeknumbers.net/2025-week-numbers | Reference | Confirms W-notation format only |
| https://dev.to/esponges/create-a-reusable-react-table-component-with-typescript-56d4 | Blog | Generic React table tutorial; no wide-table specifics |
| https://inspiredpython.com/tip/python-pathlib-tips-reading-from-and-writing-to-files | Blog | Covered by realpython.com pathlib article |

---

## Recency Scan (2024-2026)

Searched for:
1. `Python csv.DictReader TSV reading best practice 2026` (current-year frontier)
2. `React wide table horizontal scroll sticky column pattern 2025 2026` (last-2-year window)
3. `ISO week number display "W17" relative week React table` (year-less canonical)

Result: No new findings from the 2024-2026 window that supersede the canonical sources.
The `csv.DictReader` + `newline=''` pattern is stable since Python 3.x and unchanged in
3.14. Horizontal scroll with `overflow-x-auto` is the established CSS pattern -- no new
primitives in 2025-2026. ISO 8601 week format is a fixed standard. The FastAPI Pydantic
`response_model` wrapper pattern is unchanged in FastAPI 0.115+.

---

## Internal Code Inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/autoresearch/weekly_ledger.py` | 118 | Defines `COLUMNS` tuple + `read_rows()` + `append_row()` | Active; used by Thursday/Friday batch jobs |
| `backend/autoresearch/weekly_ledger.tsv` | 2 | Live data file (1 seed row) | Exists, has content |
| `backend/api/harness_autoresearch.py` | 266 | FastAPI router, prefix `/api/harness` | Active; owns sprint-state + demotion-audit endpoints |
| `backend/main.py` | ~295+ | Registers `harness_autoresearch_router`; owns `_PUBLIC_PATHS` | Active |
| `frontend/src/components/HarnessDashboard.tsx` | 817 | Harness tab component; renders `DemotionAuditTable` at line 553 | Active; `DemotionAuditTable` is the mirror template |
| `frontend/src/lib/api.ts` | ~510 | API client; `getDemotionAudit()` at line 496 is the precedent | Active |
| `frontend/src/lib/types.ts` | ~1010+ | TypeScript interfaces; `DemotionAuditResponse` at line 1000 is the precedent | Active |

---

## Internal Audit Answers (file:line anchors)

### Q1. Source file + schema

`backend/autoresearch/weekly_ledger.py` lines 21-30:

```python
COLUMNS: tuple[str, ...] = (
    "week_iso",          # str, e.g. "2026-W17"
    "thu_batch_id",      # str, slug like "seed_batch_0000"
    "thu_candidates_kicked",  # int (stored as str in TSV)
    "fri_promoted_ids",  # bracketed list "[a,b,c]" or "[]"
    "fri_rejected_ids",  # bracketed list "[a,b,c]" or "[]"
    "cost_usd",          # float (stored as str in TSV)
    "sortino_monthly",   # float (stored as str in TSV)
    "notes",             # str, free text
)
```

The TSV at `backend/autoresearch/weekly_ledger.tsv` currently has 2 lines:
line 1 = header, line 2 = one seed row (`2026-W17`, `seed_batch_0000`, `0`, `[]`, `[]`,
`0.0`, `0.0`, `phase-10.2 seed row; first real batch from the Thursday slot`).

### Q2. Existing reader

`weekly_ledger.py` lines 48-68 -- `read_rows(path=LEDGER_PATH)`:
- Reads entire file via `Path.read_text(encoding="utf-8")`
- Splits on newlines, skips line 0 (header), skips blank lines
- Calls `_parse_row()` (line 41-45) which splits on `\t` and zips with `COLUMNS`
- Returns `list[dict[str, str]]` -- all values are strings
- Fail-open: returns `[]` on any exception (FileNotFoundError, parse error, etc.)
- Does NOT use `csv.DictReader` -- uses a custom hand-rolled parser; this is intentional
  ("No imports beyond stdlib" + avoids the `newline=''` footgun for the writer side)
- Return shape example: `[{"week_iso": "2026-W17", "thu_batch_id": "seed_batch_0000", ...}]`

### Q3. Endpoint homing

Add to `backend/api/harness_autoresearch.py`. Rationale:
- The router already has `prefix="/api/harness"` (line 23) so the new endpoint is
  `GET /api/harness/weekly-ledger` with zero new files.
- `demotion-audit` (line 226) is the direct precedent: reads a local file, bounded tail,
  fail-open, `response_model=WrapperModel`.
- No separate sibling file needed. The module is 266 lines and has clear sections.

### Q4. Frontend template

`DemotionAuditTable` (HarnessDashboard.tsx lines 48-134) is the exact template:
- `<section className="rounded-xl border border-navy-700 bg-navy-800/60 p-5">`
- Header row: Phosphor icon + `<h3>` + optional trailing info span
- Empty state: centred Phosphor icon (duotone) + two text lines
- Table: `overflow-hidden rounded-xl border border-navy-700` container,
  `w-full text-left text-sm` table, `border-b border-navy-700 bg-navy-800/80` thead,
  `divide-y divide-navy-700/50` tbody, `hover:bg-navy-700/40` row hover,
  `font-mono text-xs` for IDs and numeric cells, `data-*` attrs on rows

For `WeeklyLedgerTable` the table will be wider (8 columns). Wrap the table container
in `overflow-x-auto` to enable horizontal scrolling on narrow viewports. Pin
`week_iso` column with `position: sticky; left: 0; background-color: <bg>` inline
style if needed, but given the navy dark-mode background is consistent, plain
`overflow-x-auto` with `whitespace-nowrap` on numeric cells is sufficient for this
use case (the table is internal tooling, not a customer-facing page).

### Q5. Auth

`backend/main.py` line 215, `_PUBLIC_PATHS` tuple. The demotion-audit endpoint
(`/api/harness/demotion-audit`) is already public. Add `/api/harness/weekly-ledger`
to the same tuple. This is consistent with all harness-tab endpoints being public
(they contain no sensitive financial data -- just operational metadata).

### Q6. Row ordering

`read_rows()` returns rows in file order (oldest first). Reverse before returning:
`rows[-52:][::-1]` gives newest-first with a 52-row cap in one expression.

### Q7. Default N (cap)

Cap at **52 rows** (one year of weekly data). The file is small -- at 8 columns and
~200 chars per row, 52 rows is ~10 KB. No pagination needed. The cap also matches
the `_AUDIT_TAIL_LIMIT = 200` precedent in the demotion-audit section (line 177)
which shows the project already bounds tail reads.

---

## Key Findings

1. `weekly_ledger.read_rows()` is already production-ready and must be reused directly --
   do not re-implement TSV parsing in the endpoint handler.
   (Source: `weekly_ledger.py` lines 48-68)

2. The `{rows: [...]}` wire shape with a Pydantic `BaseModel` wrapper is the canonical
   FastAPI pattern for list responses.
   (Source: https://fastapi.tiangolo.com/tutorial/response-model/)

3. `fri_promoted_ids` and `fri_rejected_ids` are stored as bracketed strings
   (e.g. `"[a,b,c]"`) -- the `_format_cell` function (line 33-38) writes them this way.
   Frontend must strip `[` `]` and split on `,` for display, or show raw string in a
   monospace cell.
   (Source: `weekly_ledger.py` lines 33-38)

4. ISO week label `"2026-W17"` is already the stored format in the TSV -- no transformation
   needed; display verbatim.
   (Source: Wikipedia ISO week date article; `weekly_ledger.tsv` line 2)

5. `overflow-x-auto` on the table wrapper is the correct CSS primitive for wide tables
   on narrower viewports; sticky first column is optional given the short (~8) column count.
   (Source: muhimasri.com sticky-column article)

6. `_PUBLIC_PATHS` in `main.py` line 215 requires an explicit string prefix for every
   public endpoint; `/api/harness/weekly-ledger` must be added.
   (Source: `backend/main.py` line 215)

---

## Concrete Design Recommendation

### Endpoint

File: `backend/api/harness_autoresearch.py` (append after the demotion-audit section, ~line 252)

```python
# ── phase-15.5 Weekly ledger history viewer ───────────────────────
from backend.autoresearch.weekly_ledger import read_rows as _read_ledger_rows, LEDGER_PATH

_LEDGER_ROW_CAP = 52  # one year of weekly rows


class WeeklyLedgerRow(BaseModel):
    week_iso: str
    thu_batch_id: str
    thu_candidates_kicked: str
    fri_promoted_ids: str
    fri_rejected_ids: str
    cost_usd: str
    sortino_monthly: str
    notes: str


class WeeklyLedgerResponse(BaseModel):
    rows: list[WeeklyLedgerRow]


@router.get("/weekly-ledger", response_model=WeeklyLedgerResponse)
def get_weekly_ledger() -> WeeklyLedgerResponse:
    """Return the last N weeks from weekly_ledger.tsv, newest first.

    Fail-open: missing / corrupt TSV returns {rows: []}.
    All values are strings (raw TSV); the frontend formats numerics.
    """
    raw = _read_ledger_rows(path=LEDGER_PATH)
    capped = raw[-_LEDGER_ROW_CAP:][::-1]  # newest first
    rows = []
    for r in capped:
        try:
            rows.append(WeeklyLedgerRow(**r))
        except Exception as exc:
            logger.warning("weekly_ledger: row coerce fail-open: %r", exc)
            continue
    return WeeklyLedgerResponse(rows=rows)
```

Notes:
- `def` (not `async def`) because `read_rows()` is sync file I/O; FastAPI auto-runs
  sync endpoints in its threadpool (per `backend-api.md` convention).
- All fields are `str` because `read_rows()` returns `dict[str, str]` -- the frontend
  is responsible for `parseFloat` / display formatting.
- Add `WeeklyLedgerRow` and `WeeklyLedgerResponse` to the `__all__` list at line 253.

### main.py change

`backend/main.py` line 215 -- add `/api/harness/weekly-ledger` to `_PUBLIC_PATHS`:

```python
_PUBLIC_PATHS = (
    "/api/health", "/api/changelog", "/api/auth",
    "/api/cost-budget", "/api/jobs/status",
    "/api/harness/monthly-approval",
    "/api/harness/demotion-audit",
    "/api/harness/weekly-ledger",   # ← add
    "/docs", "/openapi.json", "/redoc",
)
```

### How to read TSV safely + cap

Use `weekly_ledger.read_rows()` directly. It already handles:
- Missing file (`p.exists()` check, returns `[]`)
- Empty file (returns `[]`)
- Malformed rows (skips rows where `len(parts) != len(COLUMNS)`)
- Any exception (wraps in `try/except`, logs, returns `[]`)

The endpoint adds only: `raw[-52:][::-1]` (cap + reverse).

### Empty-file fallback

`read_rows()` returns `[]` if the file does not exist or is empty (lines 52-55).
The endpoint propagates this as `WeeklyLedgerResponse(rows=[])`. The frontend renders
an empty-state section (centred Phosphor icon + text) when `rows.length === 0`.

### Frontend table

**Columns to render (in display order):**

| Header | TSV field | Format |
|--------|-----------|--------|
| Week | `week_iso` | Verbatim (`"2026-W17"`) |
| Thu Batch | `thu_batch_id` | Monospace, verbatim slug |
| Kicked | `thu_candidates_kicked` | Right-align, integer |
| Promoted | `fri_promoted_ids` | Strip `[]`, split on `,`, show count + tooltip or truncated list |
| Rejected | `fri_rejected_ids` | Same as promoted |
| Cost (USD) | `cost_usd` | `parseFloat().toFixed(4)` |
| Sortino | `sortino_monthly` | `parseFloat().toFixed(4)` |
| Notes | `notes` | Left-align, natural text, truncate at 60 chars |

**List-valued cells** (`fri_promoted_ids`, `fri_rejected_ids`):
These are stored as `"[a,b,c]"` or `"[]"`. Parse with:
```typescript
function parseIds(raw: string): string[] {
  const inner = raw.replace(/^\[|\]$/g, "").trim();
  if (!inner) return [];
  return inner.split(",").map(s => s.trim()).filter(Boolean);
}
```
Display: show the count (`3 ids`) or the first id + "..." if more than one.
Full list can go in a `title` attribute for hover-tooltip accessibility.

**Table container** (wrap with `overflow-x-auto` for narrow viewports):
```tsx
<div className="overflow-x-auto overflow-hidden rounded-xl border border-navy-700">
  <table className="w-full min-w-[700px] text-left text-sm">
```

**Phosphor icon**: `Table` or `Rows` from `@phosphor-icons/react` (both available in
the project's centralized `icons.ts` -- confirm before importing).

### Placement in HarnessDashboard.tsx

Current order (lines 539-553):
1. `<HarnessSprintTile .../>` (line 540) -- the sprint tile
2. `<CostBudgetWatcherTile .../>` (line 547)
3. `<JobHeartbeatTile .../>` (line 550)
4. `<DemotionAuditTable .../>` (line 553)

Criterion 3 says "below the sprint tile". Insert `<WeeklyLedgerTable>` immediately after
`<HarnessSprintTile>` at line 544 (between sprint tile and cost budget watcher), or at
the very end after `DemotionAuditTable`. The criterion is satisfied either way; placing
it directly after the sprint tile makes the weekly cadence data contextually adjacent
to the sprint state. Recommended: insert at position 2 (after sprint tile, before cost
budget watcher).

### api.ts addition

```typescript
// phase-15.5: Weekly ledger history viewer.
export function getWeeklyLedger(): Promise<import("./types").WeeklyLedgerResponse> {
  return apiFetch("/api/harness/weekly-ledger");
}
```

### types.ts additions

```typescript
// ── phase-15.5 Weekly ledger history viewer ─────────────────────
// Shape from GET /api/harness/weekly-ledger. Backed by
// backend/autoresearch/weekly_ledger.tsv written by phase-10.2.
export interface WeeklyLedgerRow {
  week_iso: string;
  thu_batch_id: string;
  thu_candidates_kicked: string;
  fri_promoted_ids: string;  // raw bracketed string "[a,b,c]"
  fri_rejected_ids: string;  // raw bracketed string "[a,b,c]"
  cost_usd: string;
  sortino_monthly: string;
  notes: string;
}

export interface WeeklyLedgerResponse {
  rows: WeeklyLedgerRow[];
}
```

---

## Research Gate Checklist

Hard blockers:
- [x] >= 5 authoritative external sources READ IN FULL via WebFetch (7 fetched)
- [x] 10+ unique URLs total including snippet-only (17 URLs)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (7 files inspected)
- [x] No contradictions found; consensus on all three topics
- [x] All claims cited per-claim

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 10,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "report_md": "handoff/current/phase-15.5-research-brief.md",
  "gate_passed": true
}
```
