---
step: 24.14
slug: final-synthesis-phase-25-candidates
tier: moderate
cycle_date: 2026-05-12
researcher_gate: {"tier": "moderate", "external_sources_read_in_full": 5, "snippet_only_sources": 10, "urls_collected": 20, "recency_scan_performed": true, "internal_files_inspected": 14, "gate_passed": true}
---

# Research Brief — phase-24.14 — Final Synthesis + Ranked phase-25.x Candidate List

## Read in full (>=5 required; counts toward gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-05-12 | Official doc | WebFetch | "Every component encodes an assumption... worth stress testing"; sprint contracts; Plan→Generate→Evaluate |
| https://www.anthropic.com/engineering/built-multi-agent-research-system | 2026-05-12 | Official doc | WebFetch | "Scale effort to query complexity"; parallel tool calling cuts time 90%; synchronous execution simplifies dependency ordering |
| https://www.fygurs.com/blog/product-prioritization-frameworks-compared | 2026-05-12 | Blog/practitioner | WebFetch | WSJF = Cost of Delay / Job Duration; RICE = (Reach × Impact × Confidence) / Effort; "most dangerous: adopt single framework as gospel" |
| https://agileseekers.com/blog/structuring-a-scalable-product-backlog-with-dependency-mapping | 2026-05-12 | Blog/practitioner | WebFetch | "If item A must be done before B, reorder even if scores differ"; dependency matrix; pitfall = hard-coding without alternatives |
| https://ctomagazine.com/prioritize-technical-debt-ctos/ | 2026-05-12 | Industry blog | WebFetch | Critical→High→Medium→Low tier model; 80/20 rule; 20% sprint allocation to debt; security/compliance > velocity > legacy-stable |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://www.sciencedirect.com/science/article/pii/S016412122030220X | Peer-reviewed | Paywall; covered by CTO Magazine synthesis |
| https://leadership.garden/tips-on-prioritizing-tech-debt/ | Blog | HTTP 403 |
| https://www.em-tools.io/frameworks/moscow-prioritization | Blog | Covered by Fygurs comparison |
| https://ieeexplore.ieee.org/document/7070428/ | Peer-reviewed | Paywall; covered by canonical backlog refs |
| https://monday.com/blog/project-management/dependencies-diagram/ | Blog | Covered by AgileSeekers |
| https://asana.com/resources/critical-path-method | Blog | Covered by AgileSeekers |
| https://newsroom.planview.com/planview-reinvents-how-enterprises-make-critical-decisions-with-connected-work-graph-ai-powered-dependency-intelligence/ | Press release | Informational only |
| https://help.ducalis.io/frameworks/technical-debt-prioritization/ | Tool docs | Covered by CTO Magazine |
| https://www.tiny.cloud/blog/technical-debt-tracking/ | Blog | Covered by CTO Magazine |
| https://link.springer.com/chapter/10.1007/978-3-032-10721-3_49 | Peer-reviewed | Paywall |

## Recency scan (2024-2026)

Searched with three-variant discipline:
1. Current-year frontier: `"backlog prioritization 2026"` and `"dependency graph backlog prioritization critical path 2026"` — found Planview Connected Work Graph (Feb 2026) using AI dependency intelligence; Monday.com 2026 backlog guide.
2. Last-2-year window: `"technical debt ranking prioritization WSJF RICE MoSCoW backlog 2026"` and `"dependency graph prioritization software backlog 2025"` — found WSJF/RICE practitioner comparison consolidating 2025 consensus.
3. Year-less canonical: `"technical debt prioritization framework software engineering"` — found ScienceDirect systematic review (2020) and IEEE predictive analytics for tech debt paper.

**Result:** No new 2024-2026 finding supersedes the WSJF / dependency-ordering canonical frameworks. The 2026 additions are tooling improvements (AI-powered dependency graphs, Planview Connected Work Graph) but the underlying decision logic — WSJF Cost-of-Delay / Job-Duration plus topological ordering by dependency — remains unchanged. The practitioner consensus (20% sprint capacity for debt; critical-path dependencies override score-only sequencing) is stable.

---

## Search queries run

1. `"technical debt ranking prioritization WSJF RICE MoSCoW backlog 2026"` (current-year + last-2-year)
2. `"dependency graph prioritization software backlog 2025"` (last-2-year)
3. `"technical debt prioritization framework software engineering"` (year-less canonical)
4. `"dependency graph backlog prioritization critical path 2026"` (year-less + current-year)
5. `"backlog prioritization 2026"` (current-year frontier)

---

## Internal code inventory

| File | Role | Bucket(s) | Status |
|---|---|---|---|
| `docs/audits/phase-24-2026-05-12/24.0-charter-findings.md` | Charter + coverage matrix | 24.0 | READ |
| `docs/audits/phase-24-2026-05-12/24.1-execution-trading-findings.md` | Stop-loss orphan + governance gaps | 24.1 | READ |
| `docs/audits/phase-24-2026-05-12/24.2-pipeline-routing-findings.md` | Full pipeline persistence missing | 24.2 | READ |
| `docs/audits/phase-24-2026-05-12/24.3-autoresearch-wiring-findings.md` | Strategy-switch mechanism absent | 24.3 | READ |
| `docs/audits/phase-24-2026-05-12/24.4-agent-rationale-findings.md` | RiskJudge byte-identical to Trader | 24.4 | READ |
| `docs/audits/phase-24-2026-05-12/24.5-slack-notifications-findings.md` | Wrong P&L endpoint + 5 missing alert types | 24.5 | READ |
| `docs/audits/phase-24-2026-05-12/24.6-backtest-engine-findings.md` | 62-experiment plateau; no live→backtest feedback | 24.6 | READ |
| `docs/audits/phase-24-2026-05-12/24.7-data-quality-findings.md` | Freshness blind spots; yfinance unguarded | 24.7 | READ |
| `docs/audits/phase-24-2026-05-12/24.8-observability-findings.md` | Cost budget honor-system; imsg-only SLA | 24.8 | READ |
| `docs/audits/phase-24-2026-05-12/24.9-llm-conformance-findings.md` | Cache-write premium 1.25x bug; Batch/Files unused | 24.9 | READ |
| `docs/audits/phase-24-2026-05-12/24.10-mcp-security-findings.md` | Alpaca smoke test missing; SecretStr gap | 24.10 | READ |
| `docs/audits/phase-24-2026-05-12/24.11-frontend-data-wiring-findings.md` | Learnings orphan; 7 `unknown` return types | 24.11 | READ |
| `docs/audits/phase-24-2026-05-12/24.12-ui-ux-presentation-findings.md` | Degraded states; tab icons; Sharpe mismatch | 24.12 | READ |
| `docs/audits/phase-24-2026-05-12/24.13-redline-synthesis-findings.md` | Red-line synthesis; all 4 sub-goals misaligned | 24.13 | READ |

---

## Key findings

1. **P0 cluster — live money at risk.** Stop-loss orphan (`paper_trader.py:414`, zero callers); `portfolio_manager.py:82-88` silently bypasses None stops via Python truthiness; TER at -12.30% unrealized (~-$1,107 accruing) with 6 of 11 positions stop-less. Per CTO Magazine Critical tier: "Address immediately." (24.1 F-1, F-3)

2. **P0 cluster — operator blindness.** Digest P&L reads wrong endpoint (`scheduler.py:235` → `/api/portfolio/performance` in-memory dict, always $0.00) AND wrong field key (`formatters.py:322` reads `total_return_pct` vs actual `total_pnl_pct`). Kill-switch fires with only `logger.info`. (24.5 F-1, F-2, F-5b)

3. **P0 cluster — cost enforcement absent.** Cost budget tracked in `cost_budget_api.py` but `llm_client.py` never checks `tripped` flag before API calls. `cost_tracker.py:147` under-reports cache-write cost 60% (1.25x vs actual 2.0x for 1h-TTL). (24.8 F-4, 24.9 F-1)

4. **P1 cluster — strategy-switch mechanism does not exist.** Zero `autoresearch|meta_evolution` imports in `autonomous_loop.py`; `monthly_champion_challenger.py:76` hard-codes `actual_replacement: False`; `autoresearch/cron.py:29-38` is `lambda: None` stub; promoted strategies go to flat TSV with no listener. Full pipeline outputs (~$1.50/day) never persist (`orchestrator.py` has zero `save_report` calls). Red-line goal-c unimplemented. (24.3 F-1 through F-6, 24.2 F-2)

5. **P1 cluster — observability gaps.** `/freshness` covers only 2 of 7 critical tables; `profit_per_llm_dollar` absent codebase-wide; `sovereign_api.py:394-395` hardcodes zero LLM costs; `pyfinagent_pms.strategy_deployments` BQ view does not exist (self-disclosed at `sovereign_api.py:336`). (24.7, 24.13 F-2)

6. **P1 cluster — LLM cost optimizations.** Batch API (50% discount) unused on 28-skill pipeline; Files API (~97% skill-token reduction) unused; native Citations (eliminates one LLM call per Q&A) unused; system prompt below 4096-token cache threshold so caching silently no-ops. (24.9 F-4, F-5, F-6, F-2)

7. **Deduplication.** 24.13 candidates 25.T (cost-budget hard-block) = 24.8 candidate 25.A8; 25.U (plateau-detection enforcement) = 24.6 candidate 25.D6. Merged — no duplicate entries in final list.

8. **Dependency ordering overrides WSJF score.** Per AgileSeekers: "if A must be done before B, reorder even if scores differ." Critical chains: (a) 25.A9 → 25.A8 (accurate cost tracking before hard-block); (b) 25.A3 → 25.B3/25.C3/25.F3 → 25.R (registry before daily loop reads before strategy flip); (c) 25.A → 25.B (decouple before removing cosmetic patch).

---

## Consensus vs debate

**Consensus**: WSJF analysis confirms 25.G, 25.H, 25.1, 25.K as highest Cost-of-Delay / lowest effort — all P0. CTO Magazine Critical tier = immediate for stop-loss orphan, P&L blindness, cost budget. Anthropic harness-design doc confirms stale-scaffolding diagnosis for the autoresearch stub.

**Debate**: 25.B9 (bump system prompt to 4096 tokens) and 25.C9 (Batch API) could be P0 if cycle volume is high enough; left at P1/P2 because the volume data was not conclusive in bucket 24.9.

## Pitfalls (from literature)

1. WSJF without dependency adjustment leads to suboptimal ordering (AgileSeekers). Applied above.
2. MoSCoW alone is insufficient for ranking within tiers; WSJF gives continuous ordering within P0/P1/P2.
3. 20% sprint capacity rule (CTO Magazine): P0 blockers should not crowd out all capacity; P1/P2 need ongoing slots.

## Application to pyfinagent (mapping to file:line anchors)

| Candidate | Primary file anchor | Bucket source |
|---|---|---|
| 25.1 | `backend/services/autonomous_loop.py` (new Step 5.6) + `paper_trader.py:414` | 24.1 |
| 25.2 | `backend/services/paper_trader.py` (new `backfill_missing_stops()`) | 24.1 |
| 25.6 | `backend/services/paper_trader.py` (`execute_buy()`) | 24.1 |
| 25.G | `backend/slack_bot/scheduler.py:235,260` + `formatters.py:322` + `commands.py:138` | 24.5 |
| 25.H | `backend/db/bigquery_client.py:258-268` | 24.5 |
| 25.J | `backend/services/paper_trader.py` (hook after execute_buy/sell) | 24.5 |
| 25.K | `backend/slack_bot/scheduler.py:353-366` + `kill_switch.py` | 24.5 |
| 25.A9 | `backend/services/cost_tracker.py:147` | 24.9 |
| 25.A8 | `backend/services/llm_client.py` (budget check before each call) | 24.8 |
| 25.A | `backend/services/autonomous_loop.py:619-740` (second LLM call) | 24.4 |
| 25.B | `backend/services/signal_attribution.py:117-157` (delete `is_lite_dup`) | 24.4 |
| 25.A2 | `backend/services/autonomous_loop.py:575-615` + fix comment L273 | 24.2 |
| 25.A3 | `backend/autoresearch/friday_promotion.py:121-131` + BQ DDL | 24.3 |
| 25.B3 | `backend/services/autonomous_loop.py:33-43` | 24.3 |
| 25.C3 | `backend/autoresearch/monthly_champion_challenger.py:76` + `promoter.py` | 24.3 |
| 25.F3 | `backend/autoresearch/cron.py:29-38` | 24.3 |
| 25.Q | `backend/api/sovereign_api.py:394-395` | 24.13 |
| 25.R | `backend/services/autonomous_loop.py:33-43` + `promoter.py` | 24.13 |

---

## Ranked phase-25.x candidate list

### Prioritization method

Hybrid: P0/P1/P2 tiers from CTO Magazine Critical/High/Medium/Low severity mapping; WSJF ordering within tiers (Cost of Delay / Job Duration); topological dependency override where applicable.

---

### P0 candidates (8 steps — live financial / operational impact)

| Step | Name | Files | Effort | Depends on |
|---|---|---|---|---|
| 25.1 | Wire `check_stop_losses()` into daily loop with auto-sell | `autonomous_loop.py` (Step 5.6), `paper_trader.py` | S | none |
| 25.2 | Backfill missing stops with same-cycle re-check | `paper_trader.py` (new method), `scripts/maintenance/backfill_stops.py` | S | 25.1 |
| 25.6 | "No-stop-on-entry" hard block in `execute_buy()` | `paper_trader.py` | S | 25.1 |
| 25.G | Fix digest P&L data source (endpoint + field key) | `scheduler.py:235,260`, `formatters.py:322`, `commands.py:138` | S | none |
| 25.H | Fix "Recent Analyses" 5x SNDK with ticker dedup | `bigquery_client.py:258-268` | S | none |
| 25.J | Add trade confirmation notifications | `paper_trader.py` hook + `formatters.py` + `scheduler.py` | S | 25.1 |
| 25.K | Wire kill-switch state changes to Slack | `scheduler.py:353-366`, `kill_switch.py` | S | none |
| 25.A8 | Cost-budget HARD-BLOCK in `llm_client.py` | `llm_client.py`, `cost_budget_api.py`, `autonomous_loop.py` | M | 25.A9 |

### P1 candidates (19 steps — critical mechanism or significant quality gap)

| Step | Name | Files | Effort | Depends on |
|---|---|---|---|---|
| 25.A9 | Fix cache-write cost premium (1.25x → 2.0x) | `cost_tracker.py:147` | S | none |
| 25.A | Decouple RiskJudge in lite path (independent LLM call) | `autonomous_loop.py:619-740`, `llm_client.py` | M | none |
| 25.B | Remove cosmetic aliasing patch after 25.A | `signal_attribution.py:117-157` | S | 25.A |
| 25.A2 | Wire `bq.save_report()` into full pipeline | `autonomous_loop.py:575-615`, fix comment L273 | S | none |
| 25.A3 | Write promoted strategies to BQ `promoted_strategies` table | `friday_promotion.py:121-131`, `bigquery_client.py`, DDL migration | M | none |
| 25.B3 | Daily loop reads latest promoted strategy | `autonomous_loop.py:33-43` | M | 25.A3 |
| 25.C3 | Strategy registry with `status` field + flip `actual_replacement` | `monthly_champion_challenger.py:76`, `promoter.py` | M | 25.A3 |
| 25.F3 | Replace `autoresearch/cron.py` stub with real APScheduler wiring | `autoresearch/cron.py:29-38`, `slack_bot/scheduler.py` | M | 25.A3 |
| 25.Q | Real-time `profit_per_llm_dollar` metric | `sovereign_api.py:394-395`, new `/api/sovereign/efficiency`, `cost_tracker.py` | M | 25.A9 |
| 25.R | Strategy auto-switching policy | `autonomous_loop.py:33-43`, `promoter.py`, Slack formatters | L | 25.C3 + 25.F3 |
| 25.A7 | Per-table freshness endpoint covering all 5 data tables | `cycle_health.py:214-228`, `observability_api.py` | S | none |
| 25.B7 | yfinance-fallback counter + WARNING log promotion | `orchestrator.py:1140-1141`, `bigquery_client.py`, DDL migration | S | none |
| 25.D7 | `preload_macro()` max-age guard | `backtest/cache.py:184-228` | S | none |
| 25.E7 | `yfinance_tool.get_price_history()` try/except + counter | `tools/yfinance_tool.py:84-88` | S | none |
| 25.A11 | Wire the learnings backend | `paper_trading.py` (new route), `bigquery_client.py`, `api.ts`, `types.ts`, `learnings/page.tsx` | M | none |
| 25.A12 | Playwright visual regression CI baseline | `playwright.config.ts`, `tests/visual/*.spec.ts`, `.github/workflows/visual-regression.yml` | M | none |
| 25.B12 | Missing-states + tab-icon sweep | `performance/page.tsx:65-66`, `sovereign/page.tsx:63-68`, `paper-trading/page.tsx:383-390` | S | none |
| 25.C12 | Cross-tab Sharpe KPI reconciliation | `paper_trading.py` (add sharpe to /portfolio), `frontend/app/page.tsx` | S | none |
| 25.A10 | Alpaca MCP tool-surface smoke test | `scripts/mcp_servers/smoke_test_alpaca_mcp.py` (new), `reconcile_alpaca_deny_list.py` (new) | S | none |

### P2 candidates (18 steps — hygiene, polish, scaffolding)

| Step | Name | Effort | Depends on |
|---|---|---|---|
| 25.3 | "No-sells-in-N-days" anomaly watchdog | S | none |
| 25.4 | Connect `limits.yaml:max_sector_weight_pct` to `decide_trades()` | M | none |
| 25.5 | Enforce `max_position_notional_pct` from `limits.yaml` in `execute_buy()` | S | none |
| 25.I | Fix morning digest schedule + startup-log echo | S | none |
| 25.L | Add drawdown alarm | S | none |
| 25.M | Cost-budget breach alert wire repair | S | 25.A8 |
| 25.N | Add cycle-completion summary notification | S | 25.A2 |
| 25.O | Error-escalation routing | S | none |
| 25.A6 | Explicit live-vs-backtest Sharpe reconciliation | M | none |
| 25.D6 | Planner plateau-detection enforcement (lock-file gate) | S | none |
| 25.B8 | SLA Slack fallback (replace imsg-only) | S | none |
| 25.C8 | Governance watcher pre-exit Slack alert | S | none |
| 25.D8 | Kill-switch hot-key in Slack | S | none |
| 25.B9 | Bump system prompt above 4096-token cache threshold | M | none |
| 25.B10 | SecretStr migration for sensitive fields | M | none |
| 25.B11 | OpenAPI-based TS codegen for drift prevention | L | none |
| 25.S | Daily P&L attribution report | M | 25.Q |
| 25.0.C | Wire red-line goal into Q/A's LLM-judgment leg | M | none |

---

## Proposed masterplan.json step entries (JSON literals — P0 + high-P1 only)

```json
[
  {
    "id": "25.1",
    "name": "Wire check_stop_losses() into daily loop with auto-sell",
    "status": "pending",
    "priority": "P0",
    "depends_on_step": null,
    "harness_required": true,
    "effort": "S",
    "verification": {
      "command": "python3 tests/verify_phase_25_1.py",
      "live_check": "BQ query on paper_trades showing stop_loss_triggered sells after next autonomous cycle"
    }
  },
  {
    "id": "25.2",
    "name": "Backfill missing stops with same-cycle re-check",
    "status": "pending",
    "priority": "P0",
    "depends_on_step": "25.1",
    "harness_required": true,
    "effort": "S",
    "verification": {
      "command": "python3 tests/verify_phase_25_2.py",
      "live_check": "BQ query showing all paper_positions.stop_loss_price non-null after backfill"
    }
  },
  {
    "id": "25.6",
    "name": "No-stop-on-entry hard block in execute_buy()",
    "status": "pending",
    "priority": "P0",
    "depends_on_step": "25.1",
    "harness_required": true,
    "effort": "S",
    "verification": {
      "command": "python3 tests/verify_phase_25_6.py"
    }
  },
  {
    "id": "25.G",
    "name": "Fix digest P&L data source (endpoint + field key)",
    "status": "pending",
    "priority": "P0",
    "depends_on_step": null,
    "harness_required": true,
    "effort": "S",
    "verification": {
      "command": "python3 tests/verify_phase_25_G.py",
      "live_check": "Slack morning digest showing non-zero portfolio P&L matching /api/paper-trading/portfolio"
    }
  },
  {
    "id": "25.H",
    "name": "Fix Recent Analyses 5x SNDK with ticker dedup",
    "status": "pending",
    "priority": "P0",
    "depends_on_step": null,
    "harness_required": true,
    "effort": "S",
    "verification": {
      "command": "python3 tests/verify_phase_25_H.py"
    }
  },
  {
    "id": "25.J",
    "name": "Add trade confirmation notifications",
    "status": "pending",
    "priority": "P0",
    "depends_on_step": "25.1",
    "harness_required": true,
    "effort": "S",
    "verification": {
      "command": "python3 tests/verify_phase_25_J.py"
    }
  },
  {
    "id": "25.K",
    "name": "Wire kill-switch state changes to Slack",
    "status": "pending",
    "priority": "P0",
    "depends_on_step": null,
    "harness_required": true,
    "effort": "S",
    "verification": {
      "command": "python3 tests/verify_phase_25_K.py"
    }
  },
  {
    "id": "25.A9",
    "name": "Fix cache-write cost premium (1.25x to 2.0x)",
    "status": "pending",
    "priority": "P1",
    "depends_on_step": null,
    "harness_required": true,
    "effort": "S",
    "verification": {
      "command": "python3 tests/verify_phase_25_A9.py"
    }
  },
  {
    "id": "25.A8",
    "name": "Cost-budget HARD-BLOCK in llm_client",
    "status": "pending",
    "priority": "P0",
    "depends_on_step": "25.A9",
    "harness_required": true,
    "effort": "M",
    "verification": {
      "command": "python3 tests/verify_phase_25_A8.py",
      "live_check": "POST /cost-budget/set-tripped then confirm next cycle aborts with BudgetBreachError in logs"
    }
  },
  {
    "id": "25.A",
    "name": "Decouple RiskJudge in lite path (independent LLM call)",
    "status": "pending",
    "priority": "P1",
    "depends_on_step": null,
    "harness_required": true,
    "effort": "M",
    "verification": {
      "command": "python3 tests/verify_phase_25_A.py"
    }
  },
  {
    "id": "25.B",
    "name": "Remove cosmetic aliasing patch after 25.A",
    "status": "pending",
    "priority": "P1",
    "depends_on_step": "25.A",
    "harness_required": true,
    "effort": "S",
    "verification": {
      "command": "python3 tests/verify_phase_25_B.py"
    }
  },
  {
    "id": "25.A2",
    "name": "Wire bq.save_report() into full pipeline",
    "status": "pending",
    "priority": "P1",
    "depends_on_step": null,
    "harness_required": true,
    "effort": "S",
    "verification": {
      "command": "python3 tests/verify_phase_25_A2.py",
      "live_check": "BQ query on reports table showing new rows after next autonomous cycle"
    }
  },
  {
    "id": "25.A3",
    "name": "Write promoted strategies to BQ promoted_strategies table",
    "status": "pending",
    "priority": "P1",
    "depends_on_step": null,
    "harness_required": true,
    "effort": "M",
    "verification": {
      "command": "python3 tests/verify_phase_25_A3.py"
    }
  },
  {
    "id": "25.B3",
    "name": "Daily loop reads latest promoted strategy",
    "status": "pending",
    "priority": "P1",
    "depends_on_step": "25.A3",
    "harness_required": true,
    "effort": "M",
    "verification": {
      "command": "python3 tests/verify_phase_25_B3.py"
    }
  },
  {
    "id": "25.C3",
    "name": "Strategy registry with status field + flip actual_replacement",
    "status": "pending",
    "priority": "P1",
    "depends_on_step": "25.A3",
    "harness_required": true,
    "effort": "M",
    "verification": {
      "command": "python3 tests/verify_phase_25_C3.py"
    }
  },
  {
    "id": "25.F3",
    "name": "Replace autoresearch/cron.py stub with real APScheduler wiring",
    "status": "pending",
    "priority": "P1",
    "depends_on_step": "25.A3",
    "harness_required": true,
    "effort": "M",
    "verification": {
      "command": "python3 tests/verify_phase_25_F3.py"
    }
  },
  {
    "id": "25.Q",
    "name": "Real-time profit_per_llm_dollar metric",
    "status": "pending",
    "priority": "P1",
    "depends_on_step": "25.A9",
    "harness_required": true,
    "effort": "M",
    "verification": {
      "command": "python3 tests/verify_phase_25_Q.py",
      "live_check": "GET /api/sovereign/efficiency returning non-zero pnl_per_llm_dollar_30d"
    }
  },
  {
    "id": "25.R",
    "name": "Strategy auto-switching policy",
    "status": "pending",
    "priority": "P1",
    "depends_on_step": "25.C3",
    "harness_required": true,
    "effort": "L",
    "verification": {
      "command": "python3 tests/verify_phase_25_R.py",
      "live_check": "Slack notification showing strategy flip from shadow to active after DSR gate passes"
    }
  }
]
```

---

## Research-gate checklist

### Hard blockers
- [x] >=5 authoritative external sources READ IN FULL via WebFetch
- [x] 10+ unique URLs total (incl. snippet-only) — 20 collected
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (all sourced to prior findings docs with original anchors preserved)

### Soft checks
- [x] Internal exploration covered every relevant module (all 14 findings docs read in full)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim

---

## Summary (<=200 words)

Phase-24's 14-bucket audit surfaced 45 distinct phase-25 candidates (after deduplicating 2 overlapping entries from bucket 24.13). P0 (8 steps): live financial losses from the stop-loss orphan (`paper_trader.py:414`); operator blindness from the P&L-wrong-endpoint double bug (`scheduler.py:235` + `formatters.py:322`); silent kill-switch; cost budget tracked but not enforced in `llm_client.py`. P1 (19 steps): the red-line strategy-switching mechanism is entirely absent (autoresearch chain is a `lambda: None` stub; `actual_replacement` hard-coded False; zero autoresearch imports in `autonomous_loop.py`); full pipeline never persists output; RiskJudge byte-identical to Trader; LLM cost under-counted 60%; `profit_per_llm_dollar` absent codebase-wide. P2 (18 steps): harness scaffolding, frontend hygiene, security hardening, advanced LLM cost optimizations. Dependency chains require topological ordering: 25.A9 before 25.A8; 25.A3 → 25.B3/C3/F3 → 25.R for the full autoresearch/strategy-switching chain; 25.A before 25.B for RiskJudge decoupling. The most leverage-per-effort candidates are the 7 small-effort P0 items (each a single-file fix) and 25.A9 (one-line cost-tracking fix that unblocks two downstream steps).

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 10,
  "urls_collected": 20,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
