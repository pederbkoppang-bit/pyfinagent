---
name: gemini-lifecycle-pipeline-restoration
description: phase-60.1 research — gemini-2.0-flash discontinued 2026-06-01, migration candidates + dates, KR CIK abort in quant CF, fallback-alarm wiring points
metadata:
  type: project
---

phase-60.1 (2026-06-11) research facts with reuse value:

- **gemini-2.0-flash-001/-lite-001 DISCONTINUED 2026-06-01** on Vertex (404 from 06-02 in backend.log). Bare aliases track latest stable and die at family retirement — always record the retirement date next to any Gemini pin. Google's named replacement: gemini-3.1-flash-lite. **2.5 family (pro/flash/flash-lite) retires 2026-10-16** (one capture says flash-lite 2026-07-22 — conflicting); settings.py deep_think default gemini-2.5-pro is therefore ALSO on the clock. Pricing/Mtok: 2.5-flash-lite $0.10/$0.40 (parity w/ 2.0-flash), 2.5-flash $0.30/$2.50, 3.1-flash-lite $0.25/$1.50, 3.5-flash $1.50/$9. Gemini 3.x via the legacy `vertexai` SDK is unproven in this repo (SDK modules removed from releases after 2026-06-24; google-genai is the successor).
- **KR (.KS) full-pipeline abort** = `functions/quant/main.py:88` "not found in SEC CIK mapping" -> quant CF streams ERROR -> orchestrator.run_quant_agent raises (hard step-2 dep at orchestrator.py:~1536). Ingestion CF is best-effort since 27.6.6; quant is NOT. yfinance leg merges at orchestrator.py:951-953 (market-agnostic). OpenDART: free key, ~10k req/day, corpCode.xml maps 6-digit ticker -> 8-digit corp_code; English platform live since 2025-02-10 but filing bodies are Korean.
- **Silent full->lite fallback** = autonomous_loop.py:1529-1541 bare `except Exception` (anchors drift — step-context said 1411-1419). Failure reasons are logger-only, never persisted. Provenance IS persisted (`_path` via _persist_analysis :2180+, gate :875-877) but NO surface shows it (formatters.py:384-397 digest, backend/api/reports.py both blind).
- **Alert plumbing** = backend/services/observability/alerting.py:119 raise_cron_alert / :185 sync (import path `backend.services.alerting` resolves there). 56.2 degraded guard block at autonomous_loop.py:898-925 + predicate :1669-1696 is the wiring site for any new cycle-level alarm. SRE workbook low-traffic rule: at 1 cycle/day the cycle IS the alert window; multiwindow burn rates degenerate.
- **No one-shot Vertex smoke script exists** under scripts/ (phase-26 smoke was ad-hoc, harness_log.md:18686). 6 test files assert the literal "gemini-2.0-flash" (test_agent_map_live_model, test_apply_model_to_all_agents, test_evaluator_agent, verify_phase_25_Q, test_paper_trading_deposit, test_settings_api_signal_stack) — repins false-red the suite unless updated.

**Why:** AW-4 P0 — away week 05-29..06-10 ran 100% lite while operator believed the 28-agent pipeline was live.
**How to apply:** any phase-60.x step touching models, KR, or alarms starts from these anchors; re-verify line numbers (they drift). Related: [[cost-pricing-tables-inventory]], [[multimarket-universe-wiring]].
