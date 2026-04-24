# Full-Application UAT Runbook — phase-16

One-page operator summary for the end-to-end UAT that exercises every
shipped subsystem in pyfinagent before go-live.

## Purpose

Prove the whole application works together. Hermetic drills already
prove each subsystem in isolation; phase-16 proves they work plugged
into the rest of the system RIGHT NOW, under live configuration.

## Prerequisites (must be true before starting)

- Backend + frontend + mas-harness launchd agents all active (phase-16.1).
- Backend process started AFTER the latest pre-prod commit on main.
- `ALPACA_PAPER_TRADE` is NOT set to `false`.
- `BackendMode == bq_sim` (i.e., no real Alpaca keys in env).
- Working tree clean (the harness-revert-hygiene gate in `run_cycle.sh`
  refuses to run over dirty edits, so keep it clean).

## How to run

The 15 sub-steps are designed to run sequentially. Each has a
copy-pasteable `verification.command` in `.claude/masterplan.json`.

Recommended order:

1. **16.1 Infra** → fast sanity checks (launchctl, /api/health, BQ round-trip, disk).
2. **16.2 Analysis pipeline** → one full AAPL ticker run through the 15-step pipeline.
3. **16.3 MAS orchestrator** → planner↔evaluator round-trip.
4. **16.4 Autonomous paper cycle** → CRITICAL: assert `ALPACA_PAPER_TRADE` lockout first, then one `run_autonomous_cycle()`.
5. **16.5 Self-improving loops** → MetaCoordinator.gather_health + 1-iter skill_optimizer + 1-iter perf_optimizer.
6. **16.6 Kill switch** → pause → flatten → resume → zero-orders drill re-run.
7. **16.7 HITL C/C gate** → hitl_gate_drill.py + BQ audit row assert.
8. **16.8 Slack + crons** → [UAT-16.8] Slack ping + APScheduler next-fire assert.
9. **16.9 Backtest + quant opt** → **MUST call `cache.preload_macro()` first** (silent-hang gotcha); 2-iter walkforward.
10. **16.10 Frontend sweep** → curl all 10 pages for 200 + non-blank body.
11. **16.11 Auth + OWASP** → session roundtrip + 401 on protected route + OWASP headers.
12. **16.12 Observability** → cycle_health freshness + perf_tracker + harness_log non-empty.
13. **16.13 Drills aggregate** → aggregate_gate_check + 3 individual drills.
14. **16.14 Harness MAS dry-run** → `run_harness.py --cycles 1 --dry-run` + all 5 handoff files produced.
15. **16.15 Go/No-Go verdict** → **SPAWN Q/A; self-evaluation is forbidden**.

## Critical gotchas

1. **`cache.preload_macro()` before any backtest.** Missing this causes a silent ~40 min hang. Baked into 16.9's criterion #1.
2. **Assert paper-only BEFORE triggering a paper cycle.** `execution_router._refuse_live_keys()` is a safety layer, but the UAT should verify `ALPACA_PAPER_TRADE` is not `"false"` at the env level too. Baked into 16.4's criterion #1 and #2.
3. **Q/A spawn is mandatory on 16.15.** Main cannot self-declare PASS. This is the immutable gate that prevents a corrupted go-live where the system declares itself ready. Baked into 16.15's criterion #5 (immutable).

## Expected duration

- 16.1–16.3, 16.10–16.12: few minutes each.
- 16.4 (live cycle): 3–10 min depending on tickers.
- 16.9 (backtest): 10–20 min on 2-iter walkforward.
- 16.13–16.14: 5–15 min total.
- 16.15 (Q/A): 2–5 min.
- **Total:** ~1–2 hours if each step passes on first try.

## Failure discipline

- If a sub-step CONDITIONAL or FAIL: stop, fix the root cause, re-run
  that sub-step (not the whole phase).
- Three consecutive sub-step failures on the same ID → escalate to a
  certified_fallback follow-up cycle, don't thrash.
- Never flip a sub-step `done` by hand — always via Q/A reproduction
  of its `verification.command`.

## Post-UAT

On phase-16 PASS → update `handoff/harness_log.md` with the cycle
entry, then move to BLOCKER-4 (paper→live transition) with Peder
typed approval per the pre-production audit brief.
