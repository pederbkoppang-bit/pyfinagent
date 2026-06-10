# Live-check evidence — phase-24.11 — Frontend↔Backend Wiring

**Step:** 24.11 (P2)
**Date:** 2026-05-12

```
$ ls docs/audits/phase-24-2026-05-12/24.11-frontend-data-wiring-findings.md
docs/audits/phase-24-2026-05-12/24.11-frontend-data-wiring-findings.md

$ head -30 docs/audits/phase-24-2026-05-12/24.11-frontend-data-wiring-findings.md
---
bucket: 24.11
slug: frontend-data-wiring
cycle: 10
cycle_date: 2026-05-12
researcher_gate: {"tier": "moderate", "external_sources_read_in_full": 5, ..., "gate_passed": true}
---

# Findings — phase-24.11 — Frontend↔Backend Wiring Data Layer

## Executive summary

Frontend↔backend wiring is mostly clean. 119 backend routes are registered in `backend/main.py:379-413`, and `frontend/src/lib/api.ts` exposes 83 typed API functions covering the operator-visible surface.
```

**Audit anchor for next bucket:** 24.12 (P2 — frontend UI/UX presentation layer).
