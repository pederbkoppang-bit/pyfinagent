---
applyTo: "backend/services/**"
---

# Backend Services — Conventions

## Modules

- `paper_trader.py` — Virtual trade execution backed by BigQuery. No real money.
- `portfolio_manager.py` — Sell-first-then-buy logic with Risk Judge position sizing.
- `perf_metrics.py` — Single source of truth for P&L, Sharpe, max drawdown, alpha, scalar metric (risk-adjusted return + turnover penalty). All optimizers, API endpoints, and paper-trading use these canonical formulas.
- `perf_tracker.py` — Thread-safe per-endpoint latency recorder. Middleware collects timing, exports p50/p95/p99 + TSV for optimizer.
- `perf_optimizer.py` — Autoresearch-style TTL tuner. Propose → apply → measure → keep/discard. Logs to `perf_results.tsv`.
- `api_cache.py` — Thread-safe in-memory TTL cache. Per-endpoint configurable TTLs. Write endpoints invalidate relevant keys.
- `autonomous_loop.py` — Daily cycle: Screen → Analyze → Decide → Trade → Snapshot → Learn. APScheduler cron via MetaCoordinator + PaperTrader.
- `outcome_tracker.py` — Evaluates past recommendations vs actual prices (7/30/90/180/365-day windows). Generates LLM reflections, persists to BigQuery `agent_memories` for BM25 retrieval.

## Conventions

- **Sell-first-then-buy**: `portfolio_manager.py` always executes sells before buys to free capital.
- **Single metric source**: Never compute Sharpe, drawdown, or alpha outside `perf_metrics.py`. Import from there.
- **Cache invalidation**: Write endpoints (POST/PUT/DELETE) must invalidate relevant `api_cache` keys.
- **Thread safety**: `api_cache`, `perf_tracker`, and `perf_optimizer` are module-level singletons accessed from async handlers — all use `threading.Lock`.
- **TSV logging**: Both `perf_optimizer` and `outcome_tracker` append to TSV files for auditability. Always use `encoding="utf-8"` on all `open()` and `write_text()` calls (Windows CP1252 default causes charmap errors).
- **No real money**: Paper trading is virtual only. All positions tracked in BigQuery `paper_*` tables.
