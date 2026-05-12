---
step: 25.A11
slug: wire-paper-trading-learnings-backend-endpoint
tier: moderate
cycle_date: 2026-05-12
researcher_gate: {"tier": "moderate", "external_sources_read_in_full": 6, "snippet_only_sources": 8, "urls_collected": 14, "recency_scan_performed": true, "internal_files_inspected": 10, "gate_passed": true}
---

## Research: Wire /api/paper-trading/learnings Backend Endpoint (phase-25.A11)

### Search queries run (three-variant discipline)

| Variant | Query | Purpose |
|---------|-------|---------|
| Current-year (2026) | `FastAPI analytics endpoint windowed lookback query param Pydantic validation 2026` | Latest FastAPI pattern for bounded window_days param |
| Last-2-year (2025) | `virtual fund learnings reconciliation drift MFE MAE win loss attribution paper trading analytics 2025` | Practitioner analytics shape |
| Year-less canonical | `FastAPI Query ge le bound validation window_days analytics endpoint pattern` | Prior-art for numeric bounds in FastAPI |
| Current-year (2026) | `TypeScript discriminated union empty state pattern "state" field "collected" api response frontend 2025 2026` | TS type pattern for empty-vs-populated states |
| Year-less canonical | `paper trading regime performance attribution per-regime return sharpe analytics endpoint design` | Canonical regime-bucket shape |
| Year-less canonical | `paper trading analytics dashboard "kill switch" triggers distribution reconciliation drift regime performance` | Kill-switch trigger distribution analytics |

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://fastapi.tiangolo.com/tutorial/query-param-models/ | 2026-05-12 | Official doc | WebFetch | "Since version 0.115.0, FastAPI officially supports Pydantic models to define Query parameters." Pattern: `Annotated[FilterParams, Query()]` with `Field(30, ge=1, le=365)` for window bounds. `model_config = {"extra": "forbid"}` rejects unknown params. |
| https://fastapi.tiangolo.com/async/ | 2026-05-12 | Official doc | WebFetch | "When you declare a path operation function with normal `def` it is run in an external threadpool." BigQuery Python client is blocking — use `def` or `asyncio.to_thread()`. Existing sibling endpoints in paper_trading.py use `async def` + `asyncio.to_thread(bq.method)` — this is the correct pattern for this project. |
| https://oneuptime.com/blog/post/2026-02-03-fastapi-query-parameters/view | 2026-05-12 | Authoritative blog (2026) | WebFetch | `age_min: int = Query(default=0, ge=0, le=150)` — numeric bounds pattern. Windowed analytics endpoints: group related params into Depends() class, use Query(pattern=...) for date strings. |
| https://www.developerway.com/posts/advanced-typescript-for-react-developers-discriminated-unions | 2026-05-12 | Authoritative blog | WebFetch | "When I do `setState({ status: 'loading' })`… TypeScript will not allow to set neither `data` nor `error`." Discriminated union on `status` field prevents impossible states. Directly applicable to `VirtualFundLearningsData` if a `state` field is added. |
| https://www.convex.dev/typescript/advanced/type-operators-manipulation/typescript-discriminated-union | 2026-05-12 | Authoritative doc | WebFetch | Discriminant pattern: `type ApiResponse = EmptyState \| LoadedState \| ErrorState` where each has `status: 'empty' \| 'loaded' \| 'error'`. TypeScript narrows safely in switch/if blocks. |
| https://help.tradervue.com/article/3440-mfe-and-mae-calculations | 2026-05-12 | Industry practitioner | WebFetch | MFE = "maximum interim profit during the trade (runup)"; MAE = "maximum interim loss (drawdown)". Two methods: position-based (dollar) and price-based. Used for entry/exit quality, regime attribution. MFE/MAE already computed in `backend/services/paper_round_trips.py` — no new computation needed, only aggregation. |

---

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://github.com/fastapi/fastapi/discussions/10454 | Community | Confirmed Query model validation pattern — redundant once official docs fetched |
| https://medium.com/@likhith0715/fastapi-request-body-query-parameter-validation-and-pydantic-settings-explained-e27f81ccb54c | Blog (2026) | Duplicate of bounds patterns from official docs |
| https://arxiv.org/html/2509.16707v1 | Paper (2025) | AI trading framework performance attribution — regime analysis confirmed at search-snippet level; no net-new schema findings |
| https://www.tradesviz.com/blog/advanced-stats/ | Industry | Confirmed MFE/MAE + profit factor analytics shape — fetched but content was descriptive not schema-defining |
| https://www.nyif.com/articles/trading-system-kill-switch-panacea-or-pandoras-box | Industry | Fetched — policy discussion, no trigger-distribution schema; confirmed kill switches need multi-variable triggers |
| https://www.quantifiedstrategies.com/maximum-adverse-excursion-and-maximum-favorable-excursion/ | Industry | Bot-verification wall; snippet only |
| https://rpc.cfainstitute.org/sites/default/files/-/media/documents/book/rf-lit-review/2019/rflr-performance-attribution.pdf | Peer-reviewed | Performance attribution canonical — confirmed per-regime Sharpe is standard practice; not fetched in full (PDF, large) |
| https://stevekinney.com/courses/react-typescript/typescript-discriminated-unions | Authoritative blog | Confirmed TS discriminated union pattern — redundant once convex.dev + developerway fetched |

---

### Recency scan (2024-2026)

Searched for 2024-2026 literature on: (a) FastAPI query param models, (b) TypeScript discriminated unions for API states, (c) paper trading analytics dashboard design, (d) MFE/MAE methodology updates.

**Findings:**
- FastAPI 0.115.0 (late 2024) added official `Annotated[Model, Query()]` support — **this is the current recommended pattern**. Prior to 0.115.0, query-param models required workarounds. The existing `paper_trading.py` uses the older `Query(ge=1, le=1000)` inline pattern (pre-0.115.0 style) — both are valid; the inline style is already consistent with sibling endpoints so we mirror it.
- TradesViz July 2025 updated MFE/MAE calculation to use running-PnL min/max rather than price-based — the existing `paper_round_trips.py` already uses pair-based running PnL, consistent with 2025 best practice.
- No 2024-2026 finding supersedes the core response shape: `reconciliation_divergences`, `kill_switch_triggers`, `regime_buckets` are industry-standard analytics groupings. The TypeScript discriminated-union pattern is stable since TypeScript 3.x; no paradigm shift.
- The component (`VirtualFundLearnings.tsx`) was written with the exact three-section shape — this is not a new design decision, it is a pre-committed contract from phase-4.7.7.

---

### Key findings

1. **Component contract is already locked** — `VirtualFundLearnings.tsx:26-32` exports `VirtualFundLearningsData` with exactly three arrays: `reconciliation_divergences: ReconciliationDivergence[]`, `kill_switch_triggers: KillSwitchTrigger[]`, `regime_buckets: RegimeBucket[]`, plus optional `window_days?: number` and `collected_at?: string`. The backend response shape MUST match this exactly. No schema negotiation needed.

2. **FastAPI sibling-endpoint pattern to mirror** — `backend/api/paper_trading.py:216-228` (`/trades`) uses: `Query(ge=1, le=1000)` bound, `asyncio.to_thread(bq.method)` for blocking BQ call, `cache.set(cache_key, result, TTL)` via `get_api_cache()`. The `/learnings` endpoint should mirror this pattern with `window_days: int = Query(30, ge=1, le=365)`.

3. **Kill-switch trigger data source is a local JSONL file** — `backend/services/kill_switch.py:36` writes trigger events to `handoff/kill_switch_audit.jsonl` (in-process file), not BigQuery. Each line: `{"timestamp": ..., "event": "pause"|"resume"|..., "trigger": str, "details": {...}}`. The `trigger` field on `event="pause"` rows is the reason string. Aggregation: read JSONL, filter `event == "pause"`, count by `trigger` within window, map to `KillSwitchTrigger {reason, count}`.

4. **Reconciliation divergence data source** — `backend/services/reconciliation.py:148` (`compute_reconciliation()`) already computes paper-vs-sim divergence per date. It returns `{series: [{date, paper_nav, backtest_nav, divergence_pct}], summary: {...}}`. The component wants per-trade `{symbol, side, paper_fill, sim_fill, drift_pct, ts}` — a different granularity (trade-level vs NAV-level). The BQ `paper_trades` table stores `price` (fill price). The sim fill must be computed from round-trip pairing via `backend/services/paper_round_trips.py`. Strategy: pair round-trips, for each SELL leg compare `price` (paper fill) vs the shadow-backtest yfinance close used in `reconciliation.py`, emit `drift_pct = (paper_fill - sim_fill) / sim_fill * 100`.

5. **Regime bucket data source** — `backend/services/macro_regime.py:67` defines regimes as `Literal["risk_on", "risk_off", "mixed", "unknown"]`. The `paper_trades` table rows do NOT currently store `regime_tag` per trade (confirmed by grep — the field is in `autonomous_loop.py:129` stored in `summary["macro_regime"]` in the cycle summary, not on the trade row itself). Strategy: read `paper_trades` with `created_at >= cutoff`, join to daily macro-regime snapshots by date (from the macro regime cache JSON file `backend/services/macro_regime.py:30`), or fall back to querying `paper_portfolio_snapshots` if a `macro_regime` column exists there. If no per-trade regime is stored, emit an empty `regime_buckets: []` with a note field and compute from available data.

6. **BQ tables in use** — `_pt_table()` at `backend/db/bigquery_client.py:486-487` resolves to `{gcp_project}.{bq_dataset_reports}.{table_name}`. The dataset is NOT `pyfinagent_pms` — it is the value of `settings.bq_dataset_reports` (likely `pyfinagent_data` or `pyfinagent_pms`). Tables used: `paper_trades`, `paper_portfolio_snapshots`, `paper_portfolio`, `paper_positions`.

7. **TypeScript type location** — `VirtualFundLearningsData` and its sub-interfaces (`ReconciliationDivergence`, `KillSwitchTrigger`, `RegimeBucket`) currently live in `frontend/src/components/VirtualFundLearnings.tsx:5-32`. Per success criterion 3 (`virtualfundlearningsdata_type_in_types_ts`), they must be promoted to `frontend/src/lib/types.ts` and re-exported or imported from there. The component can then import from `@/lib/types`.

8. **FastAPI `asyncio.to_thread` pattern confirmed** — All `paper_trading.py` analytics endpoints use `async def` + `await asyncio.to_thread(bq.method, args)` for the BQ call, wrapped in cache check/set. The learnings endpoint should follow the same skeleton verbatim (see response-shape section below).

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `frontend/src/components/VirtualFundLearnings.tsx` | 284 | Component with locked props contract | Active — renders empty states only |
| `frontend/src/app/paper-trading/learnings/page.tsx` | 22 | Thin page wrapper, no data fetch | Orphan — needs `useEffect`/`useState` + `getPaperLearnings()` wiring |
| `frontend/src/lib/api.ts` | ~734 | API client, 83 typed functions | Active — needs `getPaperLearnings(windowDays?)` added |
| `frontend/src/lib/types.ts` | ~1147 | Central TS type module | Active — needs `VirtualFundLearningsData` + sub-interfaces promoted here |
| `backend/api/paper_trading.py` | ~900 | FastAPI router, 23 routes | Active — needs `GET /learnings` endpoint added |
| `backend/db/bigquery_client.py` | ~750+ | BQ client, paper trading query methods | Active — needs `get_paper_learnings_data(window_days)` method |
| `backend/services/kill_switch.py` | ~200 | Kill-switch state + JSONL audit at `handoff/kill_switch_audit.jsonl:36` | Active — trigger data read from local JSONL |
| `backend/services/reconciliation.py` | ~200 | NAV-level reconciliation | Active — existing compute_reconciliation() is NAV-level; trade-level drift needs new query |
| `backend/services/paper_round_trips.py` | unknown | Round-trip pairing, MFE/MAE | Active — `pair_round_trips(trades)` used by `/mfe-mae-scatter` |
| `backend/services/macro_regime.py` | ~240 | Regime classification, cache at `_CACHE_PATH:30` | Active — per-day regime tag; no per-trade regime in paper_trades |

---

### Consensus vs debate (external)

**Consensus:**
- FastAPI `Query(default, ge, le)` inline style (mirroring existing endpoints) is correct; the newer `Annotated[Model, Query()]` style is equivalent but would be inconsistent with existing code.
- TypeScript discriminated unions on a `status` field are the 2024-2026 consensus for safe empty/loaded state handling. However, the existing component does NOT use a discriminated union — it uses optional props with array defaults. The success criteria do not require a discriminated union; this is a nice-to-have.
- `asyncio.to_thread()` for blocking BQ calls inside `async def` endpoints is the correct FastAPI pattern for this codebase.

**Debate:**
- Regime bucket strategy: if `paper_trades` has no per-trade `macro_regime` column (grep confirmed absence), regime bucketing requires either (a) joining by date to the macro_regime cache file, (b) querying `paper_portfolio_snapshots` for a `macro_regime` column, or (c) returning `regime_buckets: []` with a `note: "regime_tag not stored per trade"`. Option (c) is safest for first pass and avoids introducing a yfinance/macro dependency in the endpoint.
- Kill-switch trigger source: JSONL file vs BQ. The audit file is local and small — reading it in Python is reliable and fast. No BQ query needed for this section.

---

### Pitfalls (from literature and codebase)

1. **30s BQ timeout** — CLAUDE.md mandates `result(timeout=30)` on all fallback BQ queries. The new `get_paper_learnings_data()` BQ method must pass `timeout=30`.
2. **Empty-table graceful degradation** — if `paper_trades` has 0 rows in the window, return empty arrays not 404. The component already handles `length === 0` with friendly messaging.
3. **Kill-switch JSONL may not exist** — `kill_switch_audit.jsonl` is created on first pause event. If it does not exist, return `kill_switch_triggers: []`. Do not raise.
4. **Reconciliation drift is NAV-level in existing service** — the component wants trade-level `{symbol, side, paper_fill, sim_fill, drift_pct}`. Do not call `compute_reconciliation()` — it returns NAV series, not per-trade fills. Instead query `paper_trades` for SELL rows in the window, pair with BUY rows via `pair_round_trips`, compute `drift_pct = abs(paper_fill - avg_entry_price) / avg_entry_price * 100` as a proxy for fill drift if a true sim fill is unavailable.
5. **Cache key must include `window_days`** — sibling endpoints use `f"paper:trades:{limit}"` as cache key. The learnings key should be `f"paper:learnings:{window_days}"`.
6. **Type promotion must not break the component import** — when `VirtualFundLearningsData` moves to `types.ts`, `VirtualFundLearnings.tsx` must import it from `@/lib/types` instead of defining it locally. The component currently has a local `export interface VirtualFundLearningsData` at line 26; that definition must be removed and replaced with an import.

---

### Application to pyfinagent — response shape recommendation

#### GET /api/paper-trading/learnings?window_days=30

**Request:** `window_days: int = Query(30, ge=1, le=365)`

**Response shape** (must match `VirtualFundLearningsData` exactly — `VirtualFundLearnings.tsx:26-44`):

```python
{
    "reconciliation_divergences": [
        {
            "symbol": str,        # ticker
            "side": "buy"|"sell", # trade action lowercased
            "paper_fill": float,  # price from paper_trades row
            "sim_fill": float,    # avg_entry_price as proxy (or yfinance close if available)
            "drift_pct": float,   # (paper_fill - sim_fill) / sim_fill * 100
            "ts": str             # created_at ISO string
        },
        # ... top N by abs(drift_pct), component sorts client-side to top 10
    ],
    "kill_switch_triggers": [
        {
            "reason": str,   # trigger field from audit JSONL pause events
            "count": int     # count of pause events with this trigger in window
        }
    ],
    "regime_buckets": [
        {
            "regime": str,           # "risk_on"|"risk_off"|"mixed"|"unknown"
            "n_trades": int,
            "return_pct": float,     # avg realized_pnl_pct for round-trips in this regime
            "sharpe": float|null     # null if n_trades < 5 (insufficient for Sharpe)
        }
    ],
    "window_days": int,              # echoes the request param
    "collected_at": str              # UTC ISO timestamp of computation
}
```

**BQ query plan for reconciliation_divergences:**
```sql
SELECT ticker, action, price, created_at
FROM `{_pt_table("paper_trades")}`
WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @window_days DAY)
ORDER BY created_at DESC
LIMIT 2000
```
Then in Python: call `pair_round_trips(trades)`, for each round-trip emit one divergence row where `paper_fill = sell_price`, `sim_fill = buy_price` (approximation — best available without separate sim), `drift_pct = (paper_fill - sim_fill) / sim_fill * 100`.

**Kill-switch trigger aggregation (Python, not BQ):**
Read `handoff/kill_switch_audit.jsonl` lines, filter `event == "pause"`, filter `timestamp >= cutoff`, `Counter(row["trigger"])`, emit as `[{reason, count}]` sorted by count desc.

**Regime bucket plan:**
If no `macro_regime` column on `paper_trades`, return `regime_buckets: []`. Do not silently compute from cache — document the gap in `collected_at` note. This satisfies the component's empty state gracefully. Future step can add regime column to the trade row.

**Endpoint skeleton to mirror** (`backend/api/paper_trading.py:216-228`, `/trades` endpoint):
```python
@router.get("/learnings")
async def get_learnings(window_days: int = Query(30, ge=1, le=365)):
    cache = get_api_cache()
    cache_key = f"paper:learnings:{window_days}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    settings = get_settings()
    bq = BigQueryClient(settings)
    result = await asyncio.to_thread(_compute_learnings, bq, window_days)
    cache.set(cache_key, result, ENDPOINT_TTLS.get("paper:learnings", 300))
    return result
```

**New BQ method signature:**
```python
def get_paper_trades_in_window(self, window_days: int) -> list[dict]:
    # SELECT ticker, action, price, created_at FROM paper_trades
    # WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @window_days DAY)
    # ORDER BY created_at DESC LIMIT 2000
    # timeout=30
```

**TypeScript additions to `frontend/src/lib/types.ts`:**
```typescript
export interface ReconciliationDivergence {
  symbol: string;
  side: "buy" | "sell";
  paper_fill: number;
  sim_fill: number;
  drift_pct: number;
  ts: string;
}

export interface KillSwitchTrigger {
  reason: string;
  count: number;
}

export interface RegimeBucket {
  regime: string;
  n_trades: number;
  return_pct: number;
  sharpe: number | null;
}

export interface VirtualFundLearningsData {
  reconciliation_divergences: ReconciliationDivergence[];
  kill_switch_triggers: KillSwitchTrigger[];
  regime_buckets: RegimeBucket[];
  window_days?: number;
  collected_at?: string;
}
```
(These are currently defined at `VirtualFundLearnings.tsx:5-32` and must be removed from the component and imported from `@/lib/types`.)

**api.ts addition:**
```typescript
export function getPaperLearnings(windowDays = 30): Promise<VirtualFundLearningsData> {
  return apiFetch(`/api/paper-trading/learnings?window_days=${windowDays}`);
}
```

**page.tsx wiring:**
```tsx
"use client";
import { useState, useEffect } from "react";
import { getPaperLearnings } from "@/lib/api";
import { VirtualFundLearningsData } from "@/lib/types";
// ... useEffect fetches getPaperLearnings(30), sets state, passes to <VirtualFundLearnings data={data} loading={loading} error={error} />
```

---

### Audit cross-reference (24.11 F-1)

`docs/audits/phase-24-2026-05-12/24.11-frontend-data-wiring-findings.md:17-41` — F-1 confirms:
- `frontend/src/app/paper-trading/learnings/page.tsx:6-9`: inline comment "Live data hookup lands in a follow-up backend step"
- `frontend/src/components/VirtualFundLearnings.tsx:26-32`: component defines `VirtualFundLearningsData` locally
- `backend/api/paper_trading.py` has 23 routes — none named learnings
- `frontend/src/lib/api.ts` has no `getLearnings()` function
- Candidate 25.A11 explicitly listed at `docs/audits/phase-24-2026-05-12/24.11-frontend-data-wiring-findings.md:148-158` with exact file list matching the masterplan.

---

### Research Gate Checklist

Hard blockers — `gate_passed` is false if any unchecked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched: FastAPI query-param-models, FastAPI async, oneuptime FastAPI blog 2026, developerway discriminated-unions, convex.dev discriminated-unions, tradervue MFE/MAE)
- [x] 10+ unique URLs total incl. snippet-only (14 collected)
- [x] Recency scan (last 2 years) performed + reported (FastAPI 0.115.0 2024 pattern; TradesViz 2025 MFE/MAE update; no paradigm shifts)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks — note gaps but do not auto-fail:
- [x] Internal exploration covered every relevant module (10 files inspected)
- [x] Contradictions / consensus noted (regime bucket strategy debated; kill-switch JSONL vs BQ resolved)
- [x] All claims cited per-claim (not just listed in a footer)

---

### Files to modify (summary for contract.md author)

| File | Change |
|------|--------|
| `backend/api/paper_trading.py` | Add `GET /learnings` endpoint with `window_days: int = Query(30, ge=1, le=365)` |
| `backend/db/bigquery_client.py` | Add `get_paper_trades_in_window(window_days)` method with `timeout=30` |
| `frontend/src/lib/types.ts` | Promote `VirtualFundLearningsData`, `ReconciliationDivergence`, `KillSwitchTrigger`, `RegimeBucket` here |
| `frontend/src/lib/api.ts` | Add `getPaperLearnings(windowDays=30): Promise<VirtualFundLearningsData>` |
| `frontend/src/app/paper-trading/learnings/page.tsx` | Wire `useEffect` + `getPaperLearnings()`, pass data/loading/error to component |
| `frontend/src/components/VirtualFundLearnings.tsx` | Remove local interface definitions; import from `@/lib/types` |

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 8,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "gate_passed": true
}
```
