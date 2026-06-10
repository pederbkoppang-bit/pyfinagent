# Live-check evidence — phase-24.4 — Agent Rationale Flow

**Step:** 24.4 — Agent topology + per-agent rationale flow audit (P0)
**Date:** 2026-05-12
**Live-check field:** `ls docs/audits/phase-24-2026-05-12/24.4-agent-rationale-findings.md && head -30 docs/audits/phase-24-2026-05-12/24.4-agent-rationale-findings.md`

---

## Verbatim command output

```
$ ls docs/audits/phase-24-2026-05-12/24.4-agent-rationale-findings.md
docs/audits/phase-24-2026-05-12/24.4-agent-rationale-findings.md

$ head -30 docs/audits/phase-24-2026-05-12/24.4-agent-rationale-findings.md
---
bucket: 24.4
slug: agent-rationale
cycle: 3
cycle_date: 2026-05-12
researcher_gate: {"tier": "complex", "external_sources_read_in_full": 6, "snippet_only_sources": 10, "urls_collected": 16, "recency_scan_performed": true, "internal_files_inspected": 8, "gate_passed": true}
---

# Findings — phase-24.4 — Agent Topology + Per-Agent Rationale Flow

## Executive summary

The byte-identical Trader/RiskJudge rationale bug is structural, not cosmetic. The aliasing occurs in `backend/services/autonomous_loop.py:719` — the lite-path `_run_claude_analysis` makes ONE LLM call producing ONE `reason` string, then writes it to BOTH `analysis.reason` (consumed by Trader) and `risk_assessment.reason` (consumed by RiskJudge). No independent risk-specific LLM call exists. A patch at `backend/services/signal_attribution.py:131-154` detects the byte-identical pair (weight=0.0 + rationale match) and displays an amber "lite-path" badge — this is cosmetic, not a fix. The sparse-drawer symptom (3 of ~20 agents) is downstream of bucket 24.2 (pipeline routing): the lite path is the default, and lite mode produces only 2-3 signal rows. The full 28-skill Layer-1 pipeline exists in code but its individual skill rationales (NLP sentiment, scenario weights, bias detector outputs) are never persisted into the `signals` JSON column even when the full pipeline runs.
```

The findings doc exists at the expected path; head-30 shows the frontmatter with `gate_passed: true` and the full executive summary documenting the smoking gun `autonomous_loop.py:719` aliasing.

**Audit anchor for next bucket:** 24.5 (P0 — Slack notifications + operator alerting — wrong P&L digests + 5×SNDK recent-analyses + mis-scheduled morning digest).
