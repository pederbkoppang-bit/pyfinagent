# Live-check evidence — phase-24.5 — Slack Notifications

**Step:** 24.5 — Slack notifications + operator alerting audit (P0)
**Date:** 2026-05-12
**Live-check field:** `ls docs/audits/phase-24-2026-05-12/24.5-slack-notifications-findings.md && head -30 docs/audits/phase-24-2026-05-12/24.5-slack-notifications-findings.md`

---

## Verbatim command output

```
$ ls docs/audits/phase-24-2026-05-12/24.5-slack-notifications-findings.md
docs/audits/phase-24-2026-05-12/24.5-slack-notifications-findings.md

$ head -30 docs/audits/phase-24-2026-05-12/24.5-slack-notifications-findings.md
---
bucket: 24.5
slug: slack-notifications
cycle: 4
cycle_date: 2026-05-12
researcher_gate: {"tier": "complex", "external_sources_read_in_full": 5, "snippet_only_sources": 10, "urls_collected": 15, "recency_scan_performed": true, "internal_files_inspected": 10, "gate_passed": true}
---

# Findings — phase-24.5 — Slack Notifications + Operator Alerting

## Executive summary

All four operator-reported Slack bugs are confirmed and rooted to specific file:line anchors in `backend/slack_bot/`. The `Portfolio: +$0.00 (+0.0%)` digest output is a TWO-LEVEL bug: `scheduler.py:235` calls `/api/portfolio/performance` (the legacy in-memory portfolio at `backend/api/portfolio.py` whose `_positions: dict` is always empty after restart) AND `formatters.py:322` reads `total_return_pct` while the endpoint returns `total_pnl_pct`.
```

**Audit anchor for next bucket:** 24.2 (P1 — pipeline routing + report persistence — lite-vs-full branching, /reports empty).
