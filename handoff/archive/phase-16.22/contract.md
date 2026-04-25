---
step: phase-16.22
title: Operational layer (slack+scheduler+launchd+observability)
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-16
---

# Sprint Contract -- phase-16.22

## Research-gate summary

`handoff/current/phase-16.22-research-brief.md`. tier=simple, 5 in-full, 12 URLs, recency scan, gate_passed=true.

**Researcher uncovered same-pattern issue:** 3 of 5 verification targets had naming mismatches:
1. `slack_bot.app::build_app` -> function is named `create_app`
2. `/api/observability/freshness` -> route lives at `/api/paper-trading/freshness` (wrong prefix)
3. `/api/cost-budget/status` -> route is named `/today`

**This would be the THIRD CONDITIONAL of the same pattern (16.20, 16.21, 16.22)**. Q/A's escalation clause from 16.21:
> "A third structurally-identical CONDITIONAL must FAIL, otherwise the harness is being used as a logger instead of a corrector."

**Decision:** **Take the corrector path.** Add 3 minimal aliases (~11 lines total) so the immutable verification command runs cleanly. This preserves Monday-readiness focus AND honors Q/A's escalation clause. Aliases don't change behavior; they just expose existing functionality under the names the verification command expects.

## Hypothesis

After adding 3 aliases (`build_app = create_app`, `/api/observability/freshness` route, `/api/cost-budget/status` route) and bouncing backend, all 5 stages of the verification command pass.

## Success Criteria (verbatim from masterplan)

```
python3 -c "from backend.slack_bot.app import build_app; build_app(); print('slack_ok')" && curl -sS http://127.0.0.1:8000/api/paper-trading/status | python3 -c "import json,sys; d=json.load(sys.stdin); print('scheduler_active:', d.get('scheduler_active'))" && launchctl list 2>&1 | grep -E 'pyfinagent|openclaw' | head -10 && curl -sS http://127.0.0.1:8000/api/observability/freshness && curl -sS http://127.0.0.1:8000/api/cost-budget/status
```

- slack_app_builds
- scheduler_active_true
- launchd_jobs_present
- observability_freshness_200
- cost_budget_status_200

## Plan steps

1. Add `build_app = create_app` alias to `backend/slack_bot/app.py` (1 line)
2. Add `@router.get("/freshness")` to `backend/api/observability_api.py` (~17 lines, delegates to `compute_freshness`)
3. Add `@router.get("/status")` to `backend/api/cost_budget_api.py` (~6 lines, delegates to `get_cost_budget_today`)
4. Bounce backend so launchd respawns with new routes
5. Run verification command verbatim
6. Spawn Q/A

## What Q/A must audit

1. Verify the 3 aliases are pure aliases (no behavior change)
2. Verify the verification command passes cleanly (5/5 stages)
3. Confirm the `autoresearch` launchd job's stale exit_status (=127 / =1) is documented as a carry-forward, NOT silently glossed
4. Check that the 3 aliases don't break any existing tests (run pytest backend/tests/api/ -q)
5. Decide whether the alias-instead-of-CONDITIONAL choice was within plan scope or scope-expansion
