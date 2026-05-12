# Live-check evidence — phase-24.7 — Data Quality

**Step:** 24.7 — Data quality + BQ freshness + yfinance fallback audit (P1)
**Date:** 2026-05-12

```
$ ls docs/audits/phase-24-2026-05-12/24.7-data-quality-findings.md
docs/audits/phase-24-2026-05-12/24.7-data-quality-findings.md

$ head -30 docs/audits/phase-24-2026-05-12/24.7-data-quality-findings.md
---
bucket: 24.7
slug: data-quality
cycle: 7
cycle_date: 2026-05-12
researcher_gate: {"tier": "moderate", "external_sources_read_in_full": 6, "snippet_only_sources": 10, "urls_collected": 16, "recency_scan_performed": true, "internal_files_inspected": 9, "gate_passed": true}
---

# Findings — phase-24.7 — Data Quality + BQ Freshness + yfinance Fallback

## Executive summary

The `/freshness` endpoint at `backend/services/cycle_health.py:214-228` covers only `paper_trades` + `paper_portfolio_snapshots`, leaving five critical tables (`historical_prices`, `historical_fundamentals`, `historical_macro`, `signals_log`, plus paper-trading positions) unmonitored.
```

**Audit anchor for next bucket:** 24.8 (P1 — observability + safety rails — watchdog, kill-switch, SLA, cost-budget).
