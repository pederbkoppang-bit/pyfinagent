---
name: project-bq-dataset-locations
description: Critical BigQuery dataset locations + which tables live where (resolved 2026-05-22 during phase-45.0 BQ probes)
metadata:
  type: project
---

When probing BigQuery for autonomous-loop / learn-loop / paper-trading evidence,
use the correct dataset location -- they differ across regions:

## Dataset -> location map (verified 2026-05-22)

| Dataset | Location | Tables of interest |
|---|---|---|
| `pyfinagent_data` | US | `llm_call_log`, `strategy_decisions`, `risk_intervention_log`, `scraper_audit_log`, `sla_alerts`, `unified_sar_log`, `alt_13f_holdings`, `alt_congress_trades`, `alt_finra_short_volume` |
| `financial_reports` | **us-central1** | `agent_memories`, `outcome_tracking`, `paper_trades`, `paper_positions`, `paper_portfolio`, `paper_portfolio_snapshots`, `paper_round_trips`, `signals_log`, `analysis_results`, `historical_prices`, `historical_fundamentals`, `historical_macro` |
| `pyfinagent_pms` | US | `active_holdings_view`, `alpha_velocity_samples`, `directive_versions`, `portfolio_status_snapshot`, `portfolio_transactions`, `strategy_deployments`, `strategy_deployments_log` |
| `pyfinagent_hdw` | US | historical data warehouse |
| `pyfinagent_staging` | US | staging / pre-prod |

**Why:** `Backend: financial_reports = "financial_reports"` dataset in `backend/db/bigquery_client.py:486` lives in `us-central1` per CLAUDE.md BigQuery Access section.

**How to apply:**
1. When opening a `bigquery.Client`, set `location="us-central1"` if querying any `financial_reports.*` table.
2. For `pyfinagent_data.*` and `pyfinagent_pms.*` use `location="US"`.
3. If you get `404 Not found: Table sunny-might-477607-p8:financial_reports.X was not found in location US` -- you used the wrong location.

## Timestamp column conventions

- `pyfinagent_data.llm_call_log`: `ts` (TIMESTAMP)
- `pyfinagent_data.strategy_decisions`: `ts`
- `financial_reports.paper_trades`: `created_at`
- `financial_reports.outcome_tracking`: `evaluated_at` (+ `analysis_date`)
- `financial_reports.agent_memories`: `created_at`
- `financial_reports.paper_portfolio_snapshots`: `snapshot_date` (DATE, not TIMESTAMP)

## Critical state observed 2026-05-22 (phase-45.0)

- `financial_reports.outcome_tracking` has SCHEMA but **0 rows** (never written to)
- `financial_reports.agent_memories` has SCHEMA but **0 rows** (BM25 over an empty index = no-op)
- `pyfinagent_data.llm_call_log` has 138 rows but only 5 unique cycle_ids, latest 2026-05-21 05:15 -- the autonomous_loop's Risk-Judge invocations are NOT being logged
- `pyfinagent_data.strategy_decisions` has heartbeat-only rows per cycle (`trigger=cycle_heartbeat`, `rationale="per-cycle heartbeat..."`)
- `financial_reports.paper_trades` correctly captures stop_loss_trigger SELLs with mfe/mae/capture_ratio populated, BUT `risk_judge_decision` and `signals` fields are EMPTY on stop-loss-triggered SELLs (gap to investigate when phase-35.2 lands)

**Implication for phase-35.1 (learn-loop alive):** the missing piece is the
WRITER that converts a closed stop_loss_trigger SELL into an
outcome_tracking row + an agent_memories.lesson row. Stop-loss fires
are happening (LITE + COHR today, both 25-day holds at +9.5% / +17.9% pnl)
but no learn-loop write is occurring -- this is a code gap, not an
unfired-trigger gap.
