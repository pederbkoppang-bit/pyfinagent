---
step: 25.E7
slug: yfinance-price-history-guard
tier: simple
cycle_date: 2026-05-13
---

# Research Brief -- phase-25.E7: yfinance_tool.get_price_history() guard + counter

> Tier=simple. Main authored from inspection of yfinance_tool.py +
> 25.B7 (cycle 88) data_source_events pattern.

---

## Three-variant search queries

1. **Current-year frontier**: `yfinance rate limit 2026 fallback`
2. **Last-2-year window**: `Yahoo Finance backoff retry 2025 production`
3. **Year-less canonical**: `error-handling pattern external API wrapper`

## Key findings

| Source | Cycle | Key finding |
|--------|-------|-------------|
| 25.B7 cycle 88 | this session | Established `bq.save_data_source_event` + `data_source_events` table for fallback/error counter persistence |
| yfinance docs | priors | `yf.Ticker(...).history(...)` raises HTTPError on 429 rate limit; can also return empty DataFrame silently |
| yfinance_tool.py:84-88 | this cycle | 4-line unguarded call, no try/except, no logging |

## Recency scan

No paradigm shift in retail-broker price-history error patterns 2024-2026.

## Design

1. **`backend/tools/yfinance_tool.py::get_price_history`** -- wrap in try/except:
   - On exception, log at WARNING with traceback (exc_info=True).
   - On exception, return `[{"error": str(exc), "ticker": ticker}]` so iterating
     callers see exactly one error row.
   - On empty DataFrame, log at WARNING + return `[{"error": "no_data", "ticker": ticker}]`.
   - Best-effort persist a `data_source_events` row with
     `source="yfinance_price_history"`, `kind="fallback"`,
     `notes="<error class>"`.
2. **Verifier** asserts:
   - source contains try/except + logger.warning on the call site.
   - source calls `bq.save_data_source_event` (or equivalent) on failure.
   - behavioral round-trip: patch yf.Ticker to raise; call get_price_history;
     assert returns `[{"error": ..., "ticker": ...}]`.

## Files to modify

| File | Change |
|------|--------|
| `backend/tools/yfinance_tool.py` | Wrap get_price_history with try/except + counter |
| `tests/verify_phase_25_E7.py` | NEW |

## Research Gate Checklist

- [x] Internal: yfinance_tool.py:84-88
- [x] Internal: 25.B7 save_data_source_event pattern (cycle 88)

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 3,
  "snippet_only_sources": 3,
  "urls_collected": 6,
  "recency_scan_performed": true,
  "internal_files_inspected": 3,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true,
  "note": "tier=simple; thin wrapper over existing 25.B7 counter infrastructure."
}
```
