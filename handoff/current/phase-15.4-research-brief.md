# Research Brief: phase-15.4 -- Rollback Events Log Viewer

Tier assumed: **simple** (stated by caller).

---

## 1. Executive Summary

Phase-15.4 adds a read-only audit-log viewer for demotion events. The data
source (`handoff/demotion_audit.jsonl`) is an append-only JSONL file written
by `backend/autoresearch/rollback.py::_append_audit` (lines 144-147). The
file does NOT currently exist (no demotion events have been triggered yet).
The implementation is small:

- **Backend**: one new `GET /api/harness/demotion-audit` route added to the
  existing `harness_autoresearch.py` router. Reads last 200 lines of the
  JSONL using `collections.deque(f, 200)`, parses each line with
  `json.loads`, returns `{events: [...]}`. Fails open to `{events: []}` when
  the file is missing or empty.
- **Frontend**: a new `DemotionAuditTable` component (self-contained, inlined
  in `HarnessDashboard.tsx` following the `JobHeartbeatTile` pattern). Four
  columns: timestamp, challenger_id, drawdown (dd), threshold. Empty-state
  message when list is empty. Fetch added to the dashboard's existing
  `Promise.all`. No pagination needed for the 200-event cap.

Auth: `/api/harness/demotion-audit` must be added to `_PUBLIC_PATHS` in
`backend/main.py` alongside the other harness-verification paths so the
`curl` verification command works without auth.

---

## 2. Read-in-Full Table (6 sources -- gate floor met)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://fastapi.tiangolo.com/tutorial/response-model/ | 2026-04-21 | Official doc | WebFetch | Return type annotation `-> list[Model]` is now preferred over `response_model=List[Model]` in FastAPI. Both generate identical OpenAPI docs. Use `response_model` only when the returned type differs from what you want to expose. |
| https://docs.python.org/3/library/collections.html | 2026-04-21 | Official doc | WebFetch | `deque(file, maxlen=N)` processes the file iterator discarding old entries automatically; O(1) per discard. `deque` provides thread-safe appends but sequential file reading is inherently single-threaded, so no locking needed. |
| https://realpython.com/python-deque/ | 2026-04-21 | Authoritative tutorial | WebFetch | Canonical tail pattern: `deque(open(f), maxlen=n)`. CPython guarantees `.append()` / `.pop()` are thread-safe (single bytecode, GIL). For multi-threaded producers, use `queue.Queue`; for read-only tail, `deque` is fine. |
| https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/DateTimeFormat/DateTimeFormat | 2026-04-21 | Official doc (MDN) | WebFetch | For compact audit-log timestamp columns: `new Intl.DateTimeFormat("en-US", { dateStyle: "short", timeStyle: "short" })` produces `"4/21/26, 2:45 PM"`. Cannot mix `dateStyle`/`timeStyle` with individual component fields. Widely available since 2017. |
| https://next-intl.dev/blog/date-formatting-nextjs | 2026-04-21 | Authoritative blog | WebFetch | Hydration mismatch from Intl in Server+Client components is a real risk. Safe mitigation: (a) format on server, embed in HTML; or (b) suppress hydration warning with `suppressHydrationWarning`; or (c) use a stable string format (`toISOString().slice(0,16).replace("T"," ")`) that is timezone-neutral. |
| https://www.scivision.dev/python-tail-end-of-file-deque/ | 2026-04-21 | Practitioner blog | WebFetch | `deque(f, N)` for tail read; raises `IndexError` on empty file -- must guard with `try/except` or test file non-empty first. No thread locking required for single-reader read path. |
| https://dev.to/awslearnerdaily/day-7-response-models-data-validation-with-pydantic-in-fastapi-39pe | 2026-04-21 | Practitioner blog | WebFetch | `response_model=List[Model]` vs return-type annotation: functionally identical for simple list endpoints. Use `response_model` explicitly when function returns raw dicts (as ours does -- we return `list[dict]` from a file read). Provides OpenAPI accuracy. |

---

## 3. Snippet-Only Table (does not count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://sqlpey.com/python/efficient-python-tail-file-methods/ | Practitioner blog | Fetched but primary new info was: deque is only suboptimal for multi-GB files; for audit logs well under 1 MB, deque is the clearest and safest option. No additional action needed. |
| https://geeksforgeeks.org/python-reading-last-n-lines-of-a-file/ | Tutorial | Search snippet: confirms deque + file iteration approach; covered by CPython docs and Real Python in more depth. |
| https://jsonl.help/use-cases/log-processing/ | Reference | Snippet: JSONL audit logs are line-delimited JSON; one `json.loads(line)` per line. No new information beyond what the project already uses in `_append_audit`. |
| https://medium.com/@sfcofc/displaying-dates-and-times-in-next-js-72889231577b | Blog | Snippet: recommends `toLocaleString()` as a simpler fallback to `Intl.DateTimeFormat`; hydration risk same. |
| https://github.com/fastapi/fastapi/issues/1608 | GitHub issue | Snippet: confirms returning `list[dict]` from FastAPI route works; FastAPI serialises dicts directly when `response_model` is set. |
| https://fastapi-events · PyPI | Package | Snippet: `fastapi-events` is for middleware-level event dispatch, not for file-backed audit log readers. Not applicable. |
| https://dev.to/amannn/reliable-date-formatting-in-nextjs-2mo7 | Blog | Snippet: same hydration mismatch advice as next-intl; recommends server-side date formatting or stable ISO strings. |

---

## 4. Recency Scan (2024-2026)

Queries run:
1. "Python tail read last N lines file safely large file 2026" (year-locked current)
2. "FastAPI Pydantic response_model list events audit log 2025" (last-2-year)
3. "Python JSONL audit log tail read deque collections bounded safe append-only" (year-less canonical)

Findings from the 2024-2026 window:

- **Python 3.14 (October 2025)**: Search snippet mentions enhanced `pathlib`
  methods in Python 3.14. This project already runs Python 3.14 (confirmed in
  CLAUDE.md stack table). No new file-tail API was introduced that supersedes
  the `deque` pattern; the enhancement is for path manipulation, not I/O.
- **FastAPI 0.115+ (2024-2025)**: No breaking changes to `APIRouter` prefix /
  `response_model` conventions observed in the 2024-2026 scan. The return-type
  annotation approach (preferred over `response_model`) has been stable since
  FastAPI 0.95 (2023) and is current best practice.
- **React 19 / Next.js 15 (December 2024)**: No changes to `Intl.DateTimeFormat`
  usage patterns. Hydration-mismatch risk from timezone differences is a known
  ongoing issue in Next.js (open GitHub issue #52698 still present). The safe
  mitigation (ISO string slice or server-side rendering) is unchanged.

Result: no findings in the 2024-2026 window that supersede the canonical
patterns recommended below.

---

## 5. Internal Audit (7 questions answered)

### Q1 -- Source of truth: `_append_audit` function

`backend/autoresearch/rollback.py`, lines 144-147:

```python
def _append_audit(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
```

Called at lines 101-111 with this record shape:

```python
{
    "ts": ts_iso,           # ISO 8601 string, UTC
    "event": "auto_demoted",
    "challenger_id": challenger_id,
    "dd": dd,               # float, realized drawdown magnitude
    "threshold": threshold, # float, DD_TRIGGER value
    "decision": "auto_demoted",
}
```

The `event` field (literal `"auto_demoted"`) is not mentioned in the success
criteria but should be surfaced anyway -- it confirms intent and future-proofs
the table for additional decision values (`"already_demoted"`, `"no_breach"`
are returned in the result dict but not written to the audit file -- only
`"auto_demoted"` rows appear in the JSONL). The endpoint can pass the field
through; the frontend tile can omit it or show it as a small badge.

### Q2 -- File path and current existence

Default path: `handoff/demotion_audit.jsonl` (relative to the process cwd,
which is the project root). Absolute: `/Users/ford/.openclaw/workspace/pyfinagent/handoff/demotion_audit.jsonl`.

**File does NOT exist** -- confirmed by `ls handoff/` output; the file is
not listed. No demotion events have been triggered. This means:
- The endpoint must handle `FileNotFoundError` gracefully and return `{events: []}`.
- The frontend empty-state branch will be the first path exercised in testing.

### Q3 -- Endpoint homing: extend existing router vs new file

Recommendation: **extend `backend/api/harness_autoresearch.py`**, not a new
file.

Rationale:
- The router already has `prefix="/api/harness"`, so the new route lands at
  `/api/harness/demotion-audit` with zero config.
- The file is small (176 lines); adding one route function and one Pydantic
  model keeps the demotion-audit code next to the sprint-state reader, which
  is its natural sibling (both are harness observability endpoints reading from
  `handoff/`).
- The phase-15.3 precedent (`monthly_approval_api.py` was a new file) was
  justified because it added two endpoints and a POST mutation. This phase adds
  one GET-only endpoint and does not warrant a new file.
- `main.py` already imports and registers `harness_autoresearch_router`
  (line 294-295). No registration change needed.

### Q4 -- Frontend template: existing tile pattern

The Harness tab renders tiles in this order (all in `HarnessDashboard.tsx`):
1. `HarnessSprintTile` (phase-10.11 / 15.3)
2. `CostBudgetWatcherTile` (phase-15.1) -- inlined in `HarnessDashboard.tsx`
3. `JobHeartbeatTile` (phase-15.2) -- inlined in `HarnessDashboard.tsx`

`DemotionAuditTable` should appear **after `JobHeartbeatTile`** and before the
`Current Contract` BentoCard section. This follows the established pattern of
operational tiles (sprint, cost, job health, demotion log) preceding the
document tiles (contract, critique, cycles).

Component model follows `JobHeartbeatTile` exactly:
- `section` wrapper with `rounded-xl border border-navy-700 bg-navy-800/60 p-5`
- Header row: Phosphor icon + title + event count badge
- `overflow-hidden rounded-xl border border-navy-700` table wrapper
- `table.w-full.text-left.text-sm` with `thead` and `tbody.divide-y`

### Q5 -- Empty state: fail-open behavior

When `handoff/demotion_audit.jsonl` does not exist OR is empty:
- Backend returns `{"events": []}` (HTTP 200, never 404 or 500).
- Frontend: when `events.length === 0`, render an empty-state block
  (Phosphor `ShieldCheck` icon + "No demotion events recorded" text + sub-line
  "Rollback kill-switch has not triggered yet"). Same empty-state structure
  used by the harness cycles section (icon + text + subtext in centered flex
  column).

### Q6 -- Bounded tail: cap at 200 lines

For an append-only audit file that could grow unboundedly:
- Use `collections.deque(f, maxlen=200)` -- reads entire file sequentially,
  retains only the last 200 lines. O(n_lines) but n_lines for audit logs is
  negligible (each demotion is rare; file will stay well under 1 MB for
  months).
- 200 events is the documented limit. Expose this as a module-level constant
  `_AUDIT_TAIL_LIMIT = 200` so tests and future callers can override.
- The `deque` approach is safe for this file size. The mmap / backward-seek
  approach (recommended for multi-GB logs) is unnecessary here.
- Guard against `IndexError` on empty file and `JSONDecodeError` on malformed
  lines with per-line `try/except` (fail-open: skip malformed lines, log a
  warning).

### Q7 -- Auth: public path addition

Add `"/api/harness/demotion-audit"` to `_PUBLIC_PATHS` in `backend/main.py`
(currently at line 215). Precedent: `"/api/harness/monthly-approval"` is
already public so the verification curl works. Same rationale here.

---

## 6. Concrete Design Recommendation

### 6.1 Backend: endpoint in `harness_autoresearch.py`

Add after the `HarnessSprintWeekState` models and before `__all__`:

```python
# ── phase-15.4: demotion audit ───────────────────────────────────
from collections import deque
from pathlib import Path

_AUDIT_TAIL_LIMIT = 200
_DEFAULT_DEMOTION_AUDIT_PATH = Path("handoff/demotion_audit.jsonl")


class DemotionAuditEvent(BaseModel):
    ts: str
    challenger_id: str
    dd: float
    threshold: float
    decision: str
    event: str = "auto_demoted"


class DemotionAuditResponse(BaseModel):
    events: list[DemotionAuditEvent]


def _read_demotion_audit(
    path: Path = _DEFAULT_DEMOTION_AUDIT_PATH,
    limit: int = _AUDIT_TAIL_LIMIT,
) -> list[dict]:
    """Tail-read the demotion audit JSONL. Fail-open: returns [] on any error."""
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            tail_lines = deque(f, maxlen=limit)
    except OSError as exc:
        logger.warning("demotion_audit: open fail-open: %r", exc)
        return []
    events = []
    for line in tail_lines:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            logger.warning("demotion_audit: bad JSON line, skipping: %r", exc)
    return events


@router.get("/demotion-audit", response_model=DemotionAuditResponse)
def get_demotion_audit():
    """Return the last 200 demotion events from handoff/demotion_audit.jsonl."""
    return DemotionAuditResponse(events=_read_demotion_audit())
```

`__all__` additions: `"DemotionAuditEvent"`, `"DemotionAuditResponse"`,
`"_read_demotion_audit"`.

### 6.2 Tail-read strategy

- Use `collections.deque(f, maxlen=200)` on an open text file handle.
- Returns the last `limit` lines without loading the full file into a Python
  list first.
- Empty file: `deque` returns an empty deque (no `IndexError` because we never
  index into it directly -- we iterate with a `for` loop).
- Missing file: `path.exists()` guard returns `[]` before attempting to open.
- Malformed JSON line: caught per-line with `json.JSONDecodeError`, logged,
  skipped. One bad line never fails the whole response.

### 6.3 Empty-file fallback path

```
demotion_audit.jsonl missing  --> path.exists() == False --> return []
demotion_audit.jsonl empty    --> deque is empty --> for loop yields nothing --> events = [] --> return []
All lines malformed           --> all caught per-line --> events = [] --> return []
OSError on open               --> caught at file-open level --> return []
```

All paths return HTTP 200 `{events: []}`.

### 6.4 Auth addition in `main.py`

```python
# Line 215 -- extend _PUBLIC_PATHS tuple:
_PUBLIC_PATHS = (
    "/api/health",
    "/api/changelog",
    "/api/auth",
    "/api/cost-budget",
    "/api/jobs/status",
    "/api/harness/monthly-approval",
    "/api/harness/demotion-audit",   # <-- add this
    "/docs",
    "/openapi.json",
    "/redoc",
)
```

### 6.5 Frontend tile: `DemotionAuditTable`

Inline in `HarnessDashboard.tsx` immediately after `JobHeartbeatTile`.

**Type** (add to `frontend/src/lib/types.ts`):

```typescript
// ── phase-15.4 Demotion Audit ──────────────────────────────────
export interface DemotionAuditEvent {
  ts: string;
  challenger_id: string;
  dd: number;
  threshold: number;
  decision: string;
  event: string;
}

export interface DemotionAuditResponse {
  events: DemotionAuditEvent[];
}
```

**Fetcher** (add to `frontend/src/lib/api.ts`):

```typescript
// phase-15.4: demotion audit log fetcher.
export function getDemotionAudit(): Promise<import("./types").DemotionAuditResponse> {
  return apiFetch("/api/harness/demotion-audit");
}
```

**Timestamp formatter** (inline helper in `HarnessDashboard.tsx`):

Use a stable ISO-slice approach to avoid Next.js hydration mismatch with
`Intl.DateTimeFormat` across server/client timezone differences:

```typescript
function fmtAuditTs(iso: string): string {
  // "2026-04-21T14:33:00.123456+00:00" -> "2026-04-21 14:33 UTC"
  return iso.slice(0, 16).replace("T", " ") + " UTC";
}
```

This is timezone-neutral (UTC, which is how the backend writes timestamps),
produces no hydration mismatch, and is compact enough for a table column.

**Component columns**:

| Column | Value | Notes |
|--------|-------|-------|
| Timestamp | `fmtAuditTs(event.ts)` | font-mono xs text-slate-400 |
| Challenger | `event.challenger_id` | font-mono xs text-slate-300 |
| DD | `event.dd.toFixed(4)` | font-mono xs text-red-400 |
| Threshold | `event.threshold.toFixed(4)` | font-mono xs text-slate-400 |

`data-*` attrs: `data-challenger={event.challenger_id}` on `<tr>` for test
selection.

**Component skeleton**:

```tsx
// ── phase-15.4 DemotionAuditTable ────────────────────────────────
import { ShieldCheck, WarningOctagon } from "@phosphor-icons/react";
// (add ShieldCheck + WarningOctagon to imports at top of file)

function fmtAuditTs(iso: string): string {
  return iso.slice(0, 16).replace("T", " ") + " UTC";
}

function DemotionAuditTable({ events }: { events: DemotionAuditEvent[] }) {
  return (
    <section className="rounded-xl border border-navy-700 bg-navy-800/60 p-5">
      <div className="mb-4 flex items-center gap-2">
        <WarningOctagon size={18} className="text-red-400" weight="fill" />
        <h3 className="text-sm font-semibold text-slate-300">
          Demotion Audit Log
        </h3>
        <span className="ml-auto text-xs text-slate-500">
          last {events.length} event{events.length !== 1 ? "s" : ""}
        </span>
      </div>
      {events.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-10 text-center">
          <ShieldCheck size={36} weight="duotone" className="text-slate-600" />
          <p className="mt-3 text-sm text-slate-400">No demotion events recorded</p>
          <p className="mt-1 text-xs text-slate-600">
            Rollback kill-switch has not triggered yet
          </p>
        </div>
      ) : (
        <div className="overflow-hidden rounded-xl border border-navy-700">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-navy-700 bg-navy-800/80">
              <tr>
                <th className="px-4 py-2.5 font-medium text-slate-400">Timestamp</th>
                <th className="px-4 py-2.5 font-medium text-slate-400">Challenger</th>
                <th className="px-4 py-2.5 text-right font-medium text-slate-400">DD</th>
                <th className="px-4 py-2.5 text-right font-medium text-slate-400">Threshold</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-navy-700/50">
              {events.map((ev, i) => (
                <tr
                  key={i}
                  data-challenger={ev.challenger_id}
                  className="transition-colors hover:bg-navy-700/40"
                >
                  <td className="px-4 py-2.5 font-mono text-xs text-slate-400">
                    {fmtAuditTs(ev.ts)}
                  </td>
                  <td className="px-4 py-2.5 font-mono text-xs text-slate-300">
                    {ev.challenger_id}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-xs text-red-400">
                    {ev.dd.toFixed(4)}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-xs text-slate-400">
                    {ev.threshold.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
```

**Dashboard integration**: add to the `Promise.all` in `HarnessDashboard`,
add `demotionAudit` state (type `DemotionAuditResponse | null`), render
`<DemotionAuditTable events={demotionAudit?.events ?? []} />` after
`<JobHeartbeatTile />`.

**Import addition**: `getDemotionAudit` added to the `@/lib/api` import.
`DemotionAuditResponse` added to the `@/lib/types` import.

**Icon additions**: `ShieldCheck` and `WarningOctagon` added to the
`@phosphor-icons/react` import block in `HarnessDashboard.tsx`. Also export
both from `frontend/src/lib/icons.ts` per the project icon convention.

---

## Research Gate Checklist

Hard blockers:

- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total (incl. snippet-only): 14 collected
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (rollback.py:144-147, main.py:215, harness_autoresearch.py:23 for router prefix)

Soft checks:

- [x] Internal exploration covered every relevant module (rollback.py, harness_autoresearch.py, main.py, HarnessDashboard.tsx, api.ts, types.ts)
- [x] Contradictions / consensus noted (deque vs seek tradeoff: deque is correct choice for files well under 1 MB)
- [x] All claims cited per-claim

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 7,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "report_md": "phase-15.4-research-brief.md",
  "gate_passed": true
}
```
