# Live-check evidence — phase-24.8 — Observability + Safety Rails

**Step:** 24.8 (P1)
**Date:** 2026-05-12

```
$ ls docs/audits/phase-24-2026-05-12/24.8-observability-findings.md
docs/audits/phase-24-2026-05-12/24.8-observability-findings.md

$ head -30 docs/audits/phase-24-2026-05-12/24.8-observability-findings.md
---
bucket: 24.8
slug: observability
cycle: 8
cycle_date: 2026-05-12
researcher_gate: {"tier": "complex", "external_sources_read_in_full": 6, ..., "gate_passed": true}
---

# Findings — phase-24.8 — Observability + Safety Rails

## Executive summary

Observability + safety rails in pyfinagent are in better shape than buckets 24.1/24.4/24.5 might suggest, but three critical gaps remain. The kill-switch is operator-reachable: `frontend/src/components/OpsStatusBar.tsx:96-130` exposes PAUSE / RESUME / FLATTEN_ALL buttons.
```

**Audit anchor for next bucket:** 24.10 (P1 — MCP infrastructure + security).
