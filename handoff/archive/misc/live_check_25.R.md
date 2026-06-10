# Live-check placeholder -- phase-25.R

**Step:** 25.R -- Strategy auto-switching policy (closes red-line goal-c)
**Date:** 2026-05-12

## Live-check field (per masterplan)
> "Live: a strategy switch event posts P0 Slack alert and is reflected in next-cycle decisions"

## Pre-deployment evidence
- 11/11 verifier PASS (`source .venv/bin/activate && python3 tests/verify_phase_25_R.py`)
- 6 behavioral round-trips: happy / gate-fail / first-promotion / BQ-fail-open / formatter-shape / None-prior.
- Backend AST clean for both touched files (`promoter.py`, `formatters.py`).
- `@dataclass(frozen=True)` invariant preserved (claim 10).

## Post-deployment operator workflow
1. (Prereq) 25.A3 migration applied; 25.B3 + 25.C3 + 25.R commits all on main.
2. From the scheduler or an ad-hoc script, when a shadow trial clears the gate (`shadow_trading_days >= 5` AND `dsr >= 0.95`):
   ```python
   from backend.autoresearch.promoter import Promoter
   from backend.db.bigquery_client import BigQueryClient
   from backend.config.settings import get_settings
   from backend.slack_bot.app import post_to_channel  # or appropriate Slack sender

   settings = get_settings()
   bq = BigQueryClient(settings)
   promoter = Promoter()
   trial = {
       "trial_id": "trial_42",
       "shadow_trading_days": 7,
       "dsr": 1.10,
       "pbo": 0.10,
       "params": {"lookback": 20, "tp_pct": 0.10},
       "sortino_monthly": 0.42,
   }
   result = promoter.write_to_registry(
       bq, trial, week_iso="2026-W20",
       slack_fn=lambda blocks: post_to_channel(settings.slack_channel_id, blocks=blocks),
   )
   ```
3. Expected:
   - `result["promoted"] == True`
   - `result["alert_sent"] == True`
   - Slack channel receives a P0 message with header ":rotating_light: Strategy Auto-Switch (P0)" and structured fields.
4. Verify the BQ state machine flipped atomically:
   ```sql
   SELECT strategy_id, status, allocation_pct, promoted_at
   FROM `sunny-might-477607-p8.pyfinagent_data.promoted_strategies`
   WHERE status IN ('active', 'superseded')
   ORDER BY promoted_at DESC
   LIMIT 5;
   ```
   Expected: the new strategy row has `status='active'` and the prior row has `status='superseded'`.
5. Verify next daily cycle picks up the new strategy:
   ```
   curl -s -X POST http://localhost:8000/api/paper-trading/run-now \
     -H "Authorization: Bearer $TOKEN"
   ```
   The autonomous_loop log should emit (25.B3 wiring):
   ```
   Loaded promoted params (DSR 1.10 week=2026-W20): ['lookback', 'tp_pct']
   ```

## Closes red-line goal-c
Goal-c: "dynamically shift strategy to whichever is making the most money".
Complete pipeline:
- friday_promotion writes `pending` (25.A3)
- monthly HITL approval flips to `active` (25.C3) -- OR --
- **Promoter.write_to_registry auto-flips to `active` + supersedes prior + fires P0 Slack (25.R)**
- daily loop reads via load_promoted_params (25.B3) -> trades on the new params

Goal-d (real-time `profit_per_llm_dollar` metric) is the remaining red-line gap;
covered by step 25.Q.

**Audit anchor for next bucket:** 25.Q (closes red-line goal-d) or 25.A6 (live-vs-backtest Sharpe reconciliation).
