---
name: dod-production-ready-gate
description: phase-43.0 DoD gate structure — 26 criteria (14 backend + 12 UX), source-of-truth file locations, which are deterministic vs live-blocked, and the 2026-06-01 PASS tally
metadata:
  type: project
---

phase-43.0 is the PRODUCTION_READY closure gate = a **26-criterion DoD audit
(14 backend + 12 UX)**. It CANNOT fully close autonomously: immutable
success_criteria are `all_14_DoD_criteria_PASS` + `audit_file_carries_verbatim_evidence_per_criterion`
+ `qa_confirms_no_silent_drops` + `operator_approval_recorded_for_PRODUCTION_READY_declaration`.
Deliverable each cycle = `handoff/current/production_ready_audit_<date>.md` with
PER-CRITERION verbatim evidence + class PASS / LIVE-BLOCKED / OPERATOR-GATED.

**Why:** the operator is remote (can't type PRODUCTION_READY) and 5 backend
criteria need 1-2 weeks of live trading cycles (LLM spend = operator-gated).
The honest deliverable is the refreshed audit, NOT a forced close.

**How to apply (canonical locations — verify before citing, may drift):**
- Backend 14 verbatim defs: `handoff/current/master_roadmap_to_production.md §6` (lines ~320-333, `| **DoD-N** |` table).
- UX 12 verbatim defs: `handoff/current/frontend_ux_master_design.md §6` (lines ~507-520, `| **UX-N** |` table).
- Masterplan: `.claude/masterplan.json` phase-43 → 43.0 (pending) + 43.0.1/43.0.2 (done, lifted DoD-4 coverage to STRICT ≥75%).
- Prior audits: `production_ready_audit_2026-05-28.md` (cycle 12), `_2026-05-23.md`.

**Deterministic ($0, RUN THESE) vs LIVE-BLOCKED split (as of 2026-06-01):**
- PASS-now / deterministic: DoD-3 (kill_switch.py check_auto_resume), DoD-4
  (cov on Tier-1 STRICT modules, tiered policy `docs/coverage_tier_overrides.md`),
  DoD-8 (paper_trader.py check_scale_out_fires), DoD-10 (model_tiers.py:66 +
  settings.py:30 gemini-2.5-pro), DoD-11 (OPEN-id silent-drop grep+comm),
  DoD-12 (`scripts/qa/ascii_logger_check.py` exit 0), DoD-13 (cycle_lock.py),
  DoD-14 (OWASP LLM01-10 grep in `.claude/skills/code-review-trading-domain/SKILL.md`).
- LIVE-BLOCKED (need live cycles/BQ = operator LLM spend): DoD-2 (bt↔paper
  Sharpe parity, `/api/paper-trading/reconciliation`, threshold gap_rel≤0.30
  at perf_metrics.py:128 — NOT the deprecated <0.01), DoD-5 (`/freshness`
  Unknown bands), DoD-6 (outcome_tracking+agent_memories ≥10/≥5 rows), DoD-7
  (Risk Judge JSON-fallback rate from live backend.log), DoD-9 (5 consecutive
  `completed` in cycle_history.jsonl + ≥1 non-HOLD per arXiv:2502.15800).
- OPERATOR-GATED: DoD-1 (launchctl cron SLA — autoresearch fix is phase-39.1).
- UX 0/12: all need phase-44.x build + Playwright/Lighthouse behind NextAuth
  (OPERATOR-GATED); UX-4/6/8 become grep-verifiable once built.

**2026-06-01 tally: 8/14 backend PASS** (3,4,8,10,11,12,13,**14**). vs
2026-05-28: **DoD-14 newly closed** (LLM04/05/09 tags + "v2.0"→"2025" label
added; was 7/10), DoD-4 firmed CONDITIONAL→PASS. Tests grew 614→738 backend,
62→178 frontend. **WATERMELON RISK:** full backend run = 16 failed / 711
passed — all env-coupled (live-BQ freshness probes test_phase_23_2_11/12/10,
a moved fixture-doc test_phase_23_2_16_shortlist_doc_presence ×7,
test_rainbow_canary, test_agent_map_live_model), NOT logic regressions.
Don't claim a fully-green suite; quarantine/document these 16.

External grounding (read-in-full): production-readiness is a SCORED/tri-state
gate, "done"=executed-and-evidenced not claimed (Google ML Test Score, Breck
et al.); open criteria are legit IF they carry an agreed remediation plan
(Google SRE PRR — negotiate deficits); honest red beats padded green
("problems at week 3 cost a fraction of week 12", Cultivated watermelon);
DoD-2 relative-30% is statistically sound, <0.01 infeasible on 30d
([[psr-dsr-formulas]] Bailey-LdP MinTRL). Backtest-live equity-curve
reconciliation is the canonical trading go-live gate (NautilusTrader,
QuantConnect 2025-26).
