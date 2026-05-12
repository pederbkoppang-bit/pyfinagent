# Live-check evidence — phase-24.0 — Audit Charter

**Step:** 24.0 — Phase-24 audit charter + red-line invariants
**Date:** 2026-05-12
**Live-check field (from masterplan.json):** `ls docs/audits/phase-24-2026-05-12/24.0-charter-findings.md && head -30 docs/audits/phase-24-2026-05-12/24.0-charter-findings.md`

---

## Verbatim command output

```
$ ls docs/audits/phase-24-2026-05-12/24.0-charter-findings.md
docs/audits/phase-24-2026-05-12/24.0-charter-findings.md

$ head -30 docs/audits/phase-24-2026-05-12/24.0-charter-findings.md
---
bucket: 24.0
slug: charter
cycle: 1
cycle_date: 2026-05-12
researcher_gate: {"tier": "moderate", "external_sources_read_in_full": 6, "snippet_only_sources": 10, "urls_collected": 16, "recency_scan_performed": true, "internal_files_inspected": 10, "gate_passed": true}
---

# Findings — phase-24.0 — Audit Charter + Red-Line Invariants

## Executive summary

Phase-24 is a READ-ONLY application-wide audit of the pyfinagent codebase split into 15 buckets (24.0 charter + 24.1-24.14 = 14 audit buckets). This charter doc establishes the red-line goal verbatim from `project_system_goal.md` ("maximize profit at lowest cost live; dynamically shift strategy to whichever is making the most money"), the canonical-URL whitelist that every subsequent bucket's researcher will read in full, the findings-doc path convention (`docs/audits/phase-24-2026-05-12/24.<N>-<slug>-findings.md`), and a coverage matrix mapping every backend/frontend/infrastructure subdir to a bucket so no codebase region is unaudited. The 15-bucket structure is **confirmed sufficient and non-overlapping** by the researcher's gate (`gate_passed: true`, 6 sources read in full); the only documentation gap is a `depends_on_step: "24.9"` field on bucket 24.13 in `.claude/masterplan.json` while the master prompt describes 24.13 depending on the union of 24.1-24.9 — a non-structural note. Three phase-25 candidate charter-level improvements are proposed below.

## Code-grounded findings

### F-1: Phase-24 has exactly 15 children, 24.0 through 24.14

Confirmed via direct read of `.claude/masterplan.json` lines 7895-8399. Step IDs and priorities:

| Step ID | Name | Priority | depends_on_step |
|---|---|---|---|
| 24.0 | Charter + red-line invariants | P2 | `null` |
| 24.1 | Trading-execution + governance | **P0** | 24.0 |
| 24.2 | Pipeline routing + report persistence | P1 | 24.0 |
| 24.3 | Autoresearch ↔ daily-loop wiring | P1 | 24.0 |
| 24.4 | Agent topology + per-agent rationale flow | **P0** | 24.0 |
| 24.5 | Slack notifications + operator alerting | **P0** | 24.0 |
| 24.6 | Backtest engine + walk-forward | P2 | 24.0 |
| 24.7 | Data quality + BQ freshness | P1 | 24.0 |
```

The findings doc exists at the expected path; the head-30 output shows the YAML frontmatter with `gate_passed: true`, the executive summary documenting the 15-bucket structure, and the code-grounded findings table listing the first 8 steps. This satisfies the live_check requirement that an operator can independently audit the artifact on disk.

**Audit anchor for next bucket:** 24.1 (P0 — trading-execution + governance — stop-loss orphan, missing-stops-on-entry).
