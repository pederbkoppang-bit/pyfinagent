# Live-check evidence — phase-24.2 — Pipeline Routing

**Step:** 24.2 — Pipeline routing + report persistence audit (P1)
**Date:** 2026-05-12
**Live-check field:** `ls docs/audits/phase-24-2026-05-12/24.2-pipeline-routing-findings.md && head -30 docs/audits/phase-24-2026-05-12/24.2-pipeline-routing-findings.md`

---

## Verbatim command output

```
$ ls docs/audits/phase-24-2026-05-12/24.2-pipeline-routing-findings.md
docs/audits/phase-24-2026-05-12/24.2-pipeline-routing-findings.md

$ head -30 docs/audits/phase-24-2026-05-12/24.2-pipeline-routing-findings.md
---
bucket: 24.2
slug: pipeline-routing
cycle: 5
cycle_date: 2026-05-12
researcher_gate: {"tier": "complex", "external_sources_read_in_full": 5, "snippet_only_sources": 10, "urls_collected": 16, "recency_scan_performed": true, "internal_files_inspected": 7, "gate_passed": true}
---

# Findings — phase-24.2 — Pipeline Routing + Report Persistence

## Executive summary

The hypothesis is confirmed with one important correction: the `lite_mode` branch at `backend/services/autonomous_loop.py:575` correctly routes between the lite path (`_run_claude_analysis`) and the full path (`AnalysisOrchestrator.run_full_analysis`), BUT the empty `/reports` page is NOT caused by lite mode skipping persistence — it is caused by the FULL path also never persisting from paper trading.
```

**Audit anchor for next bucket:** 24.3 (P1 — autoresearch daily-loop wiring — Sunday-only cron isolation).
