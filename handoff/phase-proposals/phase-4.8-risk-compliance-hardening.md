# Phase 4.8 - Pre-Go-Live Risk & Compliance Hardening

Status: proposal (pending)
Owner: Peder / harness
Depends on: phase-4.7 (UI audit must land so the risk dashboard has a home)
Gate: none (this IS the gate for phase-4 go-live)

## Goal

Close the top 10 profit-destroying gaps that small quant teams systematically
under-build before going live with real capital. Each gap has a published
post-mortem or academic paper quantifying its P&L impact, and each fix is a
small-to-medium build relative to the alpha it defends. This phase is the
last substantive hardening before the phase-4 aggregate smoketest flips the
go-live flag.

## Red Line contract

- **Profit**: every item below defends at least 20-80% of theoretical alpha
  that would otherwise be destroyed in production (documented in references).
- **Cost**: total incremental compute is negligible (all fixes are either
  additional BQ queries at existing cost or small Python modules). No new
  paid data vendors.
- **Risk**: this phase exists specifically to cap tail risk (CVaR + factor
  exposure), operational risk (DR runbooks + key rotation), regulatory risk
  (FINRA GenAI + PDT/T+1 + wash-sale + EU AI Act), and selection-bias risk
  (survivorship audit + champion-challenger rollout).

## Success criteria

1. Implementation shortfall logged per fill; weekly median IS on liquid
   symbols < 10 bps; alert fires above 15 bps.
2. Backtest universe passes survivorship-bias audit (delisted symbols
   present, `as_of_date` enforced on every fundamentals query).
3. Daily portfolio CVaR at 97.5% computed and gated (no new positions when
   CVaR > 2% of NAV); weekly Fama-French 3-factor regression runs with
   hard caps beta <= 1.2 and |momentum loading| <= 0.8.
4. Fractional-Kelly multi-strategy allocator produces per-strategy
   allocation_pct; single-strategy cap at 30% of NAV enforced.
5. PSI + 20-day rolling IC drift monitor running; auto-freeze fires on
   PSI > 0.25 or 5 consecutive days of negative IC.
6. Champion-challenger rollout enforced via `allocation_pct` in
   `optimizer_best.json`; promotion gates (5% -> 25% -> 100%) with named
   live-Sharpe / live-IS / live-CVaR thresholds.
7. Three disaster-recovery runbooks landed (broker outage / data-feed
   outage / LLM-provider outage) + one tabletop drill each before go-live.
8. Secrets rotation schedule every 90 days; one full key-compromise drill
   completed with restored-function-RTO < 15 min.
9. FINRA GenAI compliance: every LLM-generated trade rationale is
   queryable by `trade_id` for >= 3 years; WORM-grade storage.
10. 2026 regulatory memo (one page) covering PDT elimination + FINRA
    4210 real-time margin, T+1 settlement funding, wash-sale filter on
    any tax-loss harvesting candidate, and EU AI Act Article 10 data
    provenance if any EU-adjacent capital.

## Step-by-step plan

### 4.8.0 - Transaction Cost Analysis

Log implementation shortfall per fill:
`IS = (fill_price - arrival_price) / arrival_price * side_sign`.
Emit weekly report with median/p95 IS per symbol-liquidity bucket. Alert
when weekly median IS > 10 bps on names tagged liquid.

Verification: `python scripts/risk/tca_report.py --week last && python -c "import json; r=json.load(open('handoff/tca_last_week.json')); assert r['median_bps_liquid'] < 15"`

Criteria: `tca_logged_per_fill`, `weekly_report_generated`, `alert_fires_above_15bps_liquid`.

### 4.8.1 - Survivorship-bias + point-in-time audit

Add `delisted_at` flag to universe tables. Enforce `as_of_date` kwarg
across every fundamentals and sentiment signal reader. Re-run one
representative backtest pre-fix and post-fix; report Sharpe delta.

Verification: `python scripts/audit/survivorship_audit.py && python -c "import json; r=json.load(open('handoff/survivorship_audit.json')); assert r['pit_enforced_pct'] == 1.0"`

Criteria: `delisted_at_populated`, `pit_kwarg_enforced_100pct`, `sharpe_delta_documented`.

### 4.8.2 - Portfolio CVaR + factor-exposure gate

Daily 97.5% parametric CVaR over 60-day covariance. Weekly Fama-French
3-factor regression; hard caps beta <= 1.2, |momentum| <= 0.8. Block new
positions when CVaR > 2% of NAV.

Verification: `python -c "from backend.services.portfolio_risk import daily_check; r=daily_check(); assert 'cvar_97_5' in r and 'ff3' in r"`

Criteria: `cvar_daily_computed`, `ff3_weekly_computed`, `new_positions_blocked_when_cvar_over_2pct`, `beta_cap_enforced`.

### 4.8.3 - Fractional-Kelly multi-strategy allocator

Per-strategy `allocation_pct` from OOS Sharpe via fractional Kelly
(default f=0.5). Multi-strategy mixing via covariance matrix. Hard single-
strategy cap at 30% of NAV.

Verification: `python scripts/risk/kelly_allocator.py --dry-run && python -c "import json; r=json.load(open('handoff/allocator_output.json')); assert max(s['alloc_pct'] for s in r['strategies']) <= 0.30"`

Criteria: `per_strategy_alloc_computed`, `single_strategy_cap_30pct`, `covariance_based_mixing`.

### 4.8.4 - Drift monitor (PSI + rolling IC)

Weekly PSI of each model's output distribution vs its training
distribution; daily rolling 20-day IC vs 1-day forward return. Auto-freeze
signal on PSI > 0.25 or 5 consecutive negative IC days.

Verification: `python -c "from backend.services.drift_monitor import run; r=run(); assert 'models' in r and all('psi' in m and 'ic_20d' in m for m in r['models'])"`

Criteria: `psi_weekly_logged`, `ic_20d_rolling_logged`, `auto_freeze_fires_at_thresholds`.

### 4.8.5 - Champion-challenger gradual rollout

Extend `optimizer_best.json` with `allocation_pct` per strategy. Promotion
gate: 5% -> 25% -> 100% based on 30-day live Sharpe > 0.5 AND live IS <
10 bps AND no CVaR breach.

Verification: `python scripts/risk/promotion_gate.py --dry-run && grep -q '"allocation_pct"' backend/backtest/experiments/optimizer_best.json`

Criteria: `allocation_pct_field_present`, `promotion_gate_enforced`, `initial_live_allocation_5pct_default`.

### 4.8.6 - Disaster-recovery runbooks + tabletop

Three one-page runbooks in `docs/runbooks/`:
- `broker_outage.md` (flatten via secondary endpoint, then hold)
- `data_feed_outage.md` (freeze signal generation, use cached snapshot)
- `llm_outage.md` (fail harness cycle cleanly, resume next cycle)

One 30-minute tabletop drill per runbook logged in
`handoff/dr_drill_log.md` before go-live.

Verification: `for f in broker_outage data_feed_outage llm_outage; do test -f docs/runbooks/$f.md || exit 1; done && test -f handoff/dr_drill_log.md`

Criteria: `three_runbooks_landed`, `three_tabletop_drills_logged`, `rto_per_scenario_measured`.

### 4.8.7 - Secrets rotation + compromise drill

90-day rotation schedule in GCP Secret Manager. One full compromise drill:
revoke all vendor keys simultaneously in staging, time full restore, target
RTO < 15 min. Log to `handoff/secrets_drill_log.md`.

Verification: `python scripts/ops/secrets_rotation_check.py && grep -q 'RTO_MINUTES=' handoff/secrets_drill_log.md`

Criteria: `rotation_schedule_configured`, `drill_completed`, `rto_under_15min`.

### 4.8.8 - Supply-chain hardening

Pin LLM client library versions in `requirements.txt`. Add `pip-audit` to
CI. Weekly cron re-runs `pip-audit` and posts critical advisories to
Slack. (Triggered by LiteLLM Feb-2026 compromise post-mortem.)

Verification: `pip-audit --requirement requirements.txt --strict`

Criteria: `llm_clients_pinned`, `pip_audit_in_ci`, `weekly_pip_audit_cron`.

### 4.8.9 - FINRA GenAI compliance

Every LLM-generated trade rationale queryable by `trade_id` for 3+ years;
WORM-grade retention (immutable object lock in GCS). HITL approval logged
for every capital-promotion decision. Extends phase-4.5.5 signal_attribution.

Verification: `python scripts/compliance/finra_rationale_audit.py --sample 10 && python -c "import json; r=json.load(open('handoff/finra_audit.json')); assert r['sample_retrieval_success_rate'] == 1.0"`

Criteria: `rationale_queryable_by_trade_id`, `worm_retention_3y`, `hitl_approvals_logged`.

### 4.8.10 - 2026 regulatory memo + tax-lot / wash-sale filter

One-page memo in `docs/compliance/2026-regulatory-memo.md` covering:
- PDT elimination (Apr 2026) + real-time FINRA 4210 margin handling.
- T+1 settlement funding windows for Alpaca ACH-funded accounts.
- Wash-sale filter: reject any candidate proposing buy-back within 30
  days of a realized loss at the same symbol.
- EU AI Act Article 10 data-provenance: applicability memo (one page).

Verification: `test -f docs/compliance/2026-regulatory-memo.md && python scripts/compliance/wash_sale_filter.py --test`

Criteria: `memo_landed`, `wash_sale_filter_active`, `t1_funding_guard_active`, `realtime_margin_handler_active`.

## Proposed masterplan.json snippet

```json
{
  "id": "phase-4.8",
  "name": "Pre-Go-Live Risk & Compliance Hardening",
  "status": "pending",
  "depends_on": ["phase-4.7"],
  "gate": null,
  "steps": [
    {"id": "4.8.0", "name": "Transaction Cost Analysis (implementation shortfall)", "status": "pending", "harness_required": true, "verification": {"command": "python scripts/risk/tca_report.py --week last && python -c \"import json; r=json.load(open('handoff/tca_last_week.json')); assert r['median_bps_liquid'] < 15\"", "success_criteria": ["tca_logged_per_fill", "weekly_report_generated", "alert_fires_above_15bps_liquid"]}, "contract": null, "retry_count": 0, "max_retries": 3},
    {"id": "4.8.1", "name": "Survivorship-bias + point-in-time audit", "status": "pending", "harness_required": true, "verification": {"command": "python scripts/audit/survivorship_audit.py && python -c \"import json; r=json.load(open('handoff/survivorship_audit.json')); assert r['pit_enforced_pct'] == 1.0\"", "success_criteria": ["delisted_at_populated", "pit_kwarg_enforced_100pct", "sharpe_delta_documented"]}, "contract": null, "retry_count": 0, "max_retries": 3},
    {"id": "4.8.2", "name": "Portfolio CVaR + factor-exposure gate", "status": "pending", "harness_required": true, "verification": {"command": "python -c \"from backend.services.portfolio_risk import daily_check; r=daily_check(); assert 'cvar_97_5' in r and 'ff3' in r\"", "success_criteria": ["cvar_daily_computed", "ff3_weekly_computed", "new_positions_blocked_when_cvar_over_2pct", "beta_cap_enforced"]}, "contract": null, "retry_count": 0, "max_retries": 3},
    {"id": "4.8.3", "name": "Fractional-Kelly multi-strategy allocator", "status": "pending", "harness_required": true, "verification": {"command": "python scripts/risk/kelly_allocator.py --dry-run && python -c \"import json; r=json.load(open('handoff/allocator_output.json')); assert max(s['alloc_pct'] for s in r['strategies']) <= 0.30\"", "success_criteria": ["per_strategy_alloc_computed", "single_strategy_cap_30pct", "covariance_based_mixing"]}, "contract": null, "retry_count": 0, "max_retries": 3},
    {"id": "4.8.4", "name": "Drift monitor (PSI + rolling IC)", "status": "pending", "harness_required": true, "verification": {"command": "python -c \"from backend.services.drift_monitor import run; r=run(); assert 'models' in r and all('psi' in m and 'ic_20d' in m for m in r['models'])\"", "success_criteria": ["psi_weekly_logged", "ic_20d_rolling_logged", "auto_freeze_fires_at_thresholds"]}, "contract": null, "retry_count": 0, "max_retries": 3},
    {"id": "4.8.5", "name": "Champion-challenger gradual rollout", "status": "pending", "harness_required": true, "verification": {"command": "python scripts/risk/promotion_gate.py --dry-run && grep -q '\"allocation_pct\"' backend/backtest/experiments/optimizer_best.json", "success_criteria": ["allocation_pct_field_present", "promotion_gate_enforced", "initial_live_allocation_5pct_default"]}, "contract": null, "retry_count": 0, "max_retries": 3},
    {"id": "4.8.6", "name": "Disaster-recovery runbooks + tabletop drills", "status": "pending", "harness_required": true, "verification": {"command": "for f in broker_outage data_feed_outage llm_outage; do test -f docs/runbooks/$f.md || exit 1; done && test -f handoff/dr_drill_log.md", "success_criteria": ["three_runbooks_landed", "three_tabletop_drills_logged", "rto_per_scenario_measured"]}, "contract": null, "retry_count": 0, "max_retries": 3},
    {"id": "4.8.7", "name": "Secrets rotation + compromise drill", "status": "pending", "harness_required": true, "verification": {"command": "python scripts/ops/secrets_rotation_check.py && grep -q 'RTO_MINUTES=' handoff/secrets_drill_log.md", "success_criteria": ["rotation_schedule_configured", "drill_completed", "rto_under_15min"]}, "contract": null, "retry_count": 0, "max_retries": 3},
    {"id": "4.8.8", "name": "Supply-chain hardening (pin + pip-audit)", "status": "pending", "harness_required": true, "verification": {"command": "pip-audit --requirement requirements.txt --strict", "success_criteria": ["llm_clients_pinned", "pip_audit_in_ci", "weekly_pip_audit_cron"]}, "contract": null, "retry_count": 0, "max_retries": 3},
    {"id": "4.8.9", "name": "FINRA GenAI compliance (3-yr WORM rationale)", "status": "pending", "harness_required": true, "verification": {"command": "python scripts/compliance/finra_rationale_audit.py --sample 10 && python -c \"import json; r=json.load(open('handoff/finra_audit.json')); assert r['sample_retrieval_success_rate'] == 1.0\"", "success_criteria": ["rationale_queryable_by_trade_id", "worm_retention_3y", "hitl_approvals_logged"]}, "contract": null, "retry_count": 0, "max_retries": 3},
    {"id": "4.8.10", "name": "2026 regulatory memo + wash-sale filter", "status": "pending", "harness_required": true, "verification": {"command": "test -f docs/compliance/2026-regulatory-memo.md && python scripts/compliance/wash_sale_filter.py --test", "success_criteria": ["memo_landed", "wash_sale_filter_active", "t1_funding_guard_active", "realtime_margin_handler_active"]}, "contract": null, "retry_count": 0, "max_retries": 3}
  ]
}
```

## References

Key citations (full list in the gap-analysis chat log, 2026-04-17):

1. https://www.cis.upenn.edu/~mkearns/finread/impshort.pdf - Implementation Shortfall (Kearns/UPenn)
2. https://www.hbs.edu/faculty/Shared%20Documents/events/328/TradingCostEfficiency_FULL_112912.pdf - Trading Costs of Asset Pricing Anomalies
3. https://www.luxalgo.com/blog/survivorship-bias-in-backtesting-explained/ - Survivorship Bias
4. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 - DSR (Bailey & Lopez de Prado 2014)
5. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253 - PBO (Bailey et al.)
6. https://www.interactivebrokers.com/campus/ibkr-quant-news/expected-shortfall-es/ - Expected Shortfall
7. https://blog.quantinsti.com/risk-constrained-kelly-criterion/ - Risk-Constrained Kelly
8. https://arxiv.org/pdf/1710.00431 - Kelly Portfolio
9. https://blog.quantinsti.com/autoregressive-drift-detection-method/ - Drift detection
10. https://www.evidentlyai.com/ml-in-production/data-drift - PSI in production
11. https://alpaca.markets/learn/paper-trading-vs-live-trading-a-data-backed-guide-on-when-to-start-trading-real-money - Paper-to-live champion-challenger
12. https://community.portfolio123.com/uploads/short-url/3WHpAUOzhCG8QAUez71HpoWnA62.pdf - Backtest vs live R^2 < 0.025
13. https://blog.purestorage.com/perspectives/operational-resilience-learning-from-cme-outage/ - CME outage post-mortem
14. https://www.trendmicro.com/en_us/research/26/c/inside-litellm-supply-chain-compromise.html - LiteLLM compromise
15. https://blog.gitguardian.com/secrets-api-management/ - Secrets rotation
16. https://www.finra.org/sites/default/files/2025-12/2026-annual-regulatory-oversight-report.pdf - FINRA 2026 GenAI
17. https://www.britannica.com/money/pattern-day-trader-rule - PDT elimination (April 2026)
18. https://www.sec.gov/newsroom/press-releases/2024-62 - T+1 settlement
19. https://www.eba.europa.eu/sites/default/files/2025-11/d8b999ce-a1d9-4964-9606-971bbc2aaf89/AI%20Act%20implications%20for%20the%20EU%20banking%20sector.pdf - EU AI Act
20. https://portkey.ai/blog/ai-cost-observability-a-practical-guide-to-understanding-and-managing-llm-spend/ - AI cost observability
