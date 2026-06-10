# Live-check evidence — phase-24.10 — MCP Security

**Step:** 24.10 (P1)
**Date:** 2026-05-12

```
$ ls docs/audits/phase-24-2026-05-12/24.10-mcp-security-findings.md
docs/audits/phase-24-2026-05-12/24.10-mcp-security-findings.md

$ head -30 docs/audits/phase-24-2026-05-12/24.10-mcp-security-findings.md
---
bucket: 24.10
slug: mcp-security
cycle: 9
cycle_date: 2026-05-12
researcher_gate: {"tier": "moderate", "external_sources_read_in_full": 6, ..., "gate_passed": true}
---

# Findings — phase-24.10 — MCP Infrastructure + Security

## Executive summary

pyfinagent's MCP setup is principle-aligned with 2026 best practices: two pinned servers (`alpaca-mcp-server==2.0.1`, `mcp-server-bigquery==0.3.2`) at `.mcp.json`, deny-listed write paths at `.claude/settings.json:153-158`, stdio transport only.
```

**Audit anchor for next bucket:** 24.11 (P2 — frontend↔backend data wiring).
