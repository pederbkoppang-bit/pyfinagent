---
paths:
  - "backend/api/**"
  - "backend/config/**"
  - "backend/services/**"
  - "backend/db/**"
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
- `backtest.py` — 11 routes for walk-forward ML backtest + quant optimizer
- `performance_api.py` — 8 routes for cache stats, latency, TTL optimizer
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
