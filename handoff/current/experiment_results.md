# Experiment Results — phase-43.0 (Production-Ready DoD audit)

**Date:** 2026-06-01. **Status:** audit COMPLETE + honest. Verdict =
**NOT_PRODUCTION_READY** (backend 8/14, UX 0/12). The step stays **pending** (criteria #1
all-14-PASS + #4 operator-approval cannot be met autonomously). $0 / read-only.

## What was done

Refreshed the 26-criterion DoD audit (14 backend + 12 UX) with verbatim per-criterion
definitions + CURRENT-state classification (PASS / LIVE-BLOCKED / OPERATOR-GATED) + the
exact evidence command per criterion. Ran the cheap deterministic checks ($0, read-only).
Deliverable: `handoff/current/production_ready_audit_2026-06-01.md`. Operator asks seeded
in `handoff/current/cycle_block_summary.md`.

## Files changed

| File | Change |
|------|--------|
| `handoff/current/production_ready_audit_2026-06-01.md` | NEW — the audit deliverable (the masterplan `live_check`). |
| `handoff/current/cycle_block_summary.md` | NEW — operator asks (43.0 + run-wide). |
| `handoff/current/{research_brief,contract,experiment_results}.md` | Cycle artifacts (brief restored after the optimizer-cron clobber; cron booted out). |

## Verification output (verbatim, ran this cycle)

```
pytest backend/tests/ (collect)                 -> 738 collected ; FULL RUN: 16 failed / 711 passed
   (the 16 are ENVIRONMENT-COUPLED: live-BQ freshness probes, a moved fixture-doc x7,
    canary/wiring -- NOT logic regressions; surfaced honestly, not claimed-green)
frontend vitest                                  -> 23 files / 178 tests pass
scripts/qa/ascii_logger_check.py                 -> OK 576 files / 1830 calls / 0 violations, EXIT 0
launchctl list | grep pyfinagent                 -> autoresearch + ablation last-exit=1 (DoD-1)
/api/paper-trading/reconciliation                -> early NAV divergence 52.5% > 30% (DoD-2)
OWASP grep (trading-domain skill)                -> LLM01-LLM10 all 10/10 (DoD-14 CLOSED)
DoD-4 coverage (Tier-1 STRICT)                   -> 78.2/82.0/79.8/72.8/90.7 all >=75% (PASS)
```

## Acceptance-criteria mapping (phase-43.0 — VERBATIM)

| # | Criterion | Result |
|---|-----------|--------|
| 1 | all_14_DoD_criteria_PASS | **NOT MET** (8/14 backend; 5 live-blocked + 1 operator-gated; UX 0/12) — honestly recorded |
| 2 | audit_file_carries_verbatim_evidence_per_criterion | **PASS** — `production_ready_audit_2026-06-01.md` (verbatim def + state + cmd per criterion) |
| 3 | qa_confirms_no_silent_drops | pending fresh Q/A (DoD-11 itself shows 0 silent drops; Q/A confirms the audit hid nothing) |
| 4 | operator_approval_recorded_for_PRODUCTION_READY_declaration | **NOT MET** — operator REMOTE; ask seeded in cycle_block_summary.md |

## Honesty / scope

- NOT_PRODUCTION_READY is the honest verdict (anti-watermelon: the 16 env-coupled test
  failures + the SIGTERM-tainted freshness probe are surfaced, not hidden).
- $0: no live cycles, no LLM spend, no BQ writes, no `.env`/secret edit. Did NOT seek or
  forge the operator approval.
- The step is GATED on the operator (approval + LLM-spend for live cycles + UX build/verify)
  — it stays `pending`. The run continues to the autonomously-closable phase-53.x; the
  consolidated operator asks live in `cycle_block_summary.md`.
