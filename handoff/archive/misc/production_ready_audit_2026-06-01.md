# Production-Ready DoD Audit — 2026-06-01 (phase-43.0)

**Verdict: NOT_PRODUCTION_READY.** Backend **8/14 PASS**, UX **0/12**. Immutable
criterion #1 (`all_14_DoD_criteria_PASS`) is NOT met; criterion #4
(`operator_approval_recorded`) requires the operator (REMOTE this week) to type
"PRODUCTION_READY: APPROVED". This is the honest refreshed audit (Message-B step 2:
"run the audit; close what's autonomously closable; honestly mark the live-blocked
criteria"). Source of truth: `master_roadmap_to_production.md §6` (backend),
`frontend_ux_master_design.md §6` (UX). Full per-criterion evidence + commands:
`handoff/current/research_brief.md`.

## Delta since 2026-05-28

| Signal | 05-28 | 06-01 | Δ |
|---|---|---|---|
| Backend tests collected | 614 | 738 | +124 |
| Frontend vitest | 62 | 178 (23 files) | +116 |
| DoD-14 OWASP | 7/10 + "v2.0" | **10/10 + "2025"** | **CLOSED** |
| DoD-4 Tier-1 coverage | ~79.1% | all STRICT ≥75% (total 79.8%) | firmed PASS |
| Backend full-run | "clean" | **16 failed / 711 passed** (env-coupled) | honesty finding |

## Backend DoD — 8/14 PASS

| # | Criterion | Verdict | Evidence (verbatim cmd in research_brief.md) |
|---|-----------|---------|-----|
| DoD-1 | All cron jobs last-run within SLA (0 exit≠0) | **OPERATOR-GATED** | `autoresearch` + `ablation` last-exit=1 (`launchctl list`). autoresearch = owner-gated huggingface gap (phase-39.1); ablation = NEW 2nd failure. NOTE: the 54.1 `paper_markets` fix re-verifies on tonight's fire. |
| DoD-2 | Backtest↔paper Sharpe/P&L parity (≤30% IS-OOS decay) | **LIVE-BLOCKED** | `/reconciliation`: early NAV divergence 52.5% > 30% (seed divergence $9,499 vs $20,000); needs live convergence cycles. |
| DoD-3 | Kill-switch hysteresis tested | **PASS** | `kill_switch.py check_auto_resume` (2h/1h thresholds, idempotent) + unit test. |
| DoD-4 | Coverage ≥70%/layer (tiered policy) | **PASS** | Tier-1 STRICT: paper_trader 78.2 / portfolio_mgr 82.0 / perf_metrics 79.8 / cycle_health 72.8 / kill_switch 90.7. |
| DoD-5 | 0 Unknown freshness bands | **LIVE-BLOCKED** | No literal `Unknown` today (red/amber/green now), but the read was SIGTERM-tainted; re-probe on a stable backend + a fresh cycle. |
| DoD-6 | Learn-loop alive (≥10 outcome rows, ≥5 memories) | **LIVE-BLOCKED** | Writer wired (`autonomous_loop.py:1961-2042`); needs ≥10 real sell-closes (recent cycles n_trades=0). |
| DoD-7 | Risk Judge structured-output >95% | **LIVE-BLOCKED** | Schema enforced (`orchestrator.py`+`risk_debate.py`, phase-37.1); production fallback-rate needs live backend.log. |
| DoD-8 | Profit-protection scale-out closed | **PASS** | `paper_trader.check_scale_out_fires` (idempotent, gated, unit-tested). |
| DoD-9 | 5 consecutive clean cron cycles | **LIVE-BLOCKED** | Streak=4 (a `timeout` breaks the run); needs 5 + ≥1 non-HOLD decide_trades proposal. |
| DoD-10 | Source defaults match prod env | **PASS** | `model_tiers.py:66` + `settings.py:30` = `gemini-2.5-pro`. |
| DoD-11 | 0 silent audit-finding drops (OPEN-1..33) | **PASS** | All 33 mapped (OPEN-19/21/27 → roadmap rows + named auto-memories that EXIST). |
| DoD-12 | ASCII-only loggers | **PASS** | `ascii_logger_check.py` → 576 files / 1830 calls / 0 violations, EXIT 0. |
| DoD-13 | Restart-survivable cycle state | **PASS** | `cycle_lock.py` + `clean_stale_lock` in `main.py` lifespan (today's SIGTERM is a live demo of the recovery path). |
| DoD-14 | OWASP LLM Top-10 compliance | **PASS (newly closed)** | LLM01-LLM10 all 10/10 tagged in the trading-domain skill; "2025" label. |

**5 LIVE-BLOCKED** (DoD-2/5/6/7/9) need 1-2 weeks of live trading cycles = **operator
LLM spend**. **1 OPERATOR-GATED** (DoD-1) needs the owner-gated autoresearch fix + ablation
triage.

## UX DoD — 0/12 (all OPERATOR-GATED or FAIL-fixable + operator-verify)

Source: `frontend_ux_master_design.md §6`. These close under phase-44.x (cockpit
foundation, largely unbuilt) and ALL require real-browser verification (Playwright +
Lighthouse + axe) behind the NextAuth wall. Grep-verifiable FAIL-fixable subset:
UX-4 (settings DRY dup), UX-6 (DataTable adoption — foundation `@tanstack/react-table`
IS installed), UX-8 (inline states → States library). The rest (UX-1/2/5/7/9/11/12) need
operator-run Lighthouse/Playwright; UX-10 needs an SSE backend (LIVE-BLOCKED). Full table
in `research_brief.md`.

## Honesty findings (anti-watermelon)

1. **Full backend run = 16 failed / 711 passed.** All 16 are ENVIRONMENT-COUPLED (live-BQ
   freshness probes, a moved fixture-doc `test_phase_23_2_16_shortlist_doc_presence` ×7,
   canary/wiring) — **NOT logic regressions**. Do NOT claim a fully-green suite from the
   738 collect-only count. Recommend: quarantine/mark these (a follow-up hygiene task).
2. **Restart-tainted probe.** DoD-5 freshness + DoD-2 reconciliation were read while the
   backend was under SIGTERM (`-15`). Re-probe on a stable backend before any PASS claim.
3. **DoD-4 wording.** Master-roadmap §6 DoD-4 should cite the tiered-coverage policy
   explicitly (1-line doc edit) to remove the "every broad layer ≥70%" ambiguity.

## Operator asks (carried to cycle_block_summary.md)

To reach PRODUCTION_READY, the operator must (after returning):
1. **Approve LLM spend for live cycles** → closes DoD-2/5/6/7/9 over 1-2 weeks
   (convergence + ≥10 sell-closes + 5 clean consecutive cycles + Risk-Judge fallback rate).
2. **Owner-gate fixes for DoD-1**: the autoresearch `langchain_huggingface` install
   (phase-39.1, pip — operator-gated) + ablation exit-1 triage.
3. **Build + visually verify the UX DoD** (phase-44.x) behind the NextAuth wall
   (Playwright + Lighthouse ≥95 a11y / ≥90 perf).
4. **Type "PRODUCTION_READY: APPROVED"** once 1-4 are green (criterion #4).

phase-43.0 stays **pending** (verdict NOT_PRODUCTION_READY). The audit (criteria #2
verbatim-evidence + #3 no-silent-drops) is the delivered artifact.
