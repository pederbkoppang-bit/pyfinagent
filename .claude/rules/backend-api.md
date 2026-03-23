---
paths:
  - "backend/api/**"
  - "backend/config/**"
  - "backend/db/**"
  - "backend/tasks/**"
---

# Backend API & Services — FastAPI Conventions

## Stack
- FastAPI 0.115+, Python 3.14, Pydantic settings, BigQuery client
- Auth: HKDF + JWE token decrypt from NextAuth.js, email whitelist via `ALLOWED_EMAILS`
- BQ project: `sunny-might-477607-p8`, dataset: `financial_reports`

## API Structure (70+ routes across 10 routers)
- `analysis.py` — POST start + GET poll (async background tasks)
- `reports.py` — CRUD + performance stats + cost history
- `signals.py` — 12 enrichment signal endpoints + macro
- `paper_trading.py` — 8 routes for autonomous paper trading
- `backtest.py` — 15 routes for walk-forward ML backtest + quant optimizer; mutual exclusion via HTTP 409 (backtest and optimizer cannot run concurrently); `engine_source` field tracks who started the engine (`"backtest"` | `"optimizer"` | `None`); `_optimizer_state` and `_backtest_state` both carry `error` + `traceback` fields (populated via `traceback.format_exc()` on exception, returned by status endpoints); optimizer mirrors engine progress into `_backtest_state["progress"]` for unified progress panel; `engine_progress_cb` forwards engine sub-steps into both optimizer and backtest state; experiments endpoint accepts `?run_id=` filter with smart BASELINE inclusion; auto-loads latest persisted result on module init; saves result to disk after completion; `/runs` endpoints for listing/loading/deleting historical results; `/optimize/insights` endpoint returns param bounds + experiments with full params + walk-forward data scope; all `open()` calls use `encoding="utf-8"`
- `performance_api.py` — 8 routes for cache stats, latency, TTL optimizer; TSV reads use `encoding="utf-8"`
- All endpoints return JSON. Use Pydantic models for request/response validation.

## BigQuery Schema
- `analysis_results` — 88-column ML training schema (managed by `migrate_bq_schema.py`)
- `outcome_tracking` — Actual price performance vs recommendations
- `agent_memories` — BM25-retrievable learned lessons
- 4 paper trading tables, 3 historical backtest data tables
- All migrations idempotent (safe to re-run)

## Services
- `api_cache.py` — Thread-safe TTL response cache, endpoint-specific TTLs
- `perf_tracker.py` — Per-endpoint latency recording (p50/p95/p99)
- `perf_optimizer.py` — Autoresearch-style TTL optimizer
- `paper_trader.py` — Virtual trade execution, `portfolio_manager.py` — sell-first-then-buy logic
- `perf_metrics.py` — Single source of truth for P&L, Sharpe, alpha, scalar metric

## Conventions
- Cache invalidation on write endpoints (POST/PUT/DELETE)
- Async background tasks for long operations (analysis, backtest, optimization)
- Input validation on all endpoints. Ticker symbols sanitized.
- OWASP security headers on all responses
- **Encoding**: Always use `encoding="utf-8"` on all `open()` and `Path.write_text()` calls (Windows default is CP1252 — charmap errors on Unicode characters)
- **Logging**: `setup_logging()` in `main.py` wraps stderr with `io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")` — do not replace this with a plain `StreamHandler()`
- **Error state dicts**: Background tasks store `traceback.format_exc()` on exception so status endpoints can return full tracebacks to the UI
- **Mutual exclusion**: Backtest and optimizer share an implicit engine lock — `run_backtest()` rejects with 409 if optimizer running and vice versa; `_is_engine_busy()` checks both states; `engine_source` field in `_backtest_state` tracks the active caller
