# Phase B - Recursive Evolution Loop (proposed phase-10)

Principal Sovereign Systems Architect proposal. This phase extends the
existing phase-8.5 Karpathy autoresearch loop (see
`handoff/phase-proposals/phase-3.7-4.7-8.5-mas-ux-autoresearch.md` and the
phase-8.5 block in `.claude/masterplan.json` lines 1919-2120) with a weekly
Mon-Fri sprint calendar and a monthly Sortino-based Champion/Challenger
replacement gate. Phase-10 is a post-go-live operational cadence phase; it
activates after phase-4 go-live flips and does not rebuild what phase-8.5
already builds.

Red Line: `Net System Alpha = Profit - (Risk Exposure + Compute Burn)`.
The weekly discipline here exists to keep Compute Burn bounded (fixed 2
new Claude routine slots per week) while concentrating Profit lever
activity into a single promotion cadence.

---

## Goal

Run the autoresearch loop (phase-8.5) on a fixed Monday-Friday rhythm with
a monthly Sortino-based Champion/Challenger gate, so that strategy
evolution is predictable, budgeted, human-auditable, and produces exactly
one promotion decision per week plus exactly one Master Baseline
replacement decision per month. The phase does not introduce new search
machinery. It introduces a calendar, a promotion ledger, and a Sortino
replacement rule on top of phase-8.5's existing DSR/PBO gate.

## Success criteria

1. Weekly sprint calendar (Mon-Fri) is encoded as a deterministic
   scheduler config, not ad-hoc cron entries. Config file lives at
   `backend/autoresearch/sprint_calendar.yaml` and is version-controlled.
2. Monday/Tuesday/Wednesday Global Intelligence scan slots reuse phase-6.5
   (Global Intelligence) and phase-8.5.3 (LLM proposer) infrastructure.
   Phase-10 allocates zero new Claude routine slots on Mon/Tue/Wed; it
   only subscribes to their outputs.
3. Thursday batch: a single Claude routine run (1 slot) triggers the
   100+ parallel backtest batch via the autoresearch harness and records
   the batch id in `backend/autoresearch/weekly_ledger.tsv`. Target:
   >= 100 candidates executed per Thursday.
4. Friday promotion gate: a single Claude routine run (1 slot) evaluates
   the Thursday batch through the phase-8.5 DSR/PBO gate, selects top-N
   promotable candidates, and promotes them to paper-live at 5% starting
   allocation via phase-4.8.5 champion-challenger rollout. Top-N default
   is 1. Max is 3. Never auto-promotes to real capital.
5. Monthly Sortino-based Champion/Challenger gate fires on the last
   trading day of each month and evaluates every Challenger that has
   been paper-live for >= 30 calendar days. Replacement requires all of:
   Challenger Sortino - Champion Sortino >= 0.3; Challenger PBO < 0.2;
   Challenger Max DD <= 1.2 * Champion Max DD; Peder human-in-loop
   approval via Slack interactive button with 48h expiry.
6. Sortino formula `S = (Rp - T) / sigma_d` is implemented with
   Minimum Acceptable Return `T` defaulting to the 3-month US T-Bill
   risk-free rate, pulled from BigQuery `pyfinagent_data.macro` daily
   series. `T` is configurable per candidate in
   `sprint_calendar.yaml` but defaults to risk-free.
7. Total NEW Claude routine slots consumed by phase-10 = 2 per week
   (Thursday batch trigger + Friday promotion gate). Month-end Sortino
   gate reuses Friday's slot on the last trading Friday of the month;
   no additional slot consumed. Slot accounting is logged to BigQuery
   `pyfinagent_data.harness_learning_log` with label `phase-10`.
8. Every weekly sprint writes one row to `weekly_ledger.tsv` with
   columns: week_iso, mon_wed_scan_ids, thu_batch_id,
   thu_candidate_count, fri_promoted_ids, fri_rejected_ids, cost_usd.
   Schema is stable across releases.
9. Every monthly gate writes one row to `monthly_sortino_ledger.tsv`
   with columns: month_iso, champion_id, challenger_id,
   champion_sortino, challenger_sortino, sortino_delta, pbo,
   dd_ratio, decision, peder_approval_ts.
10. Rollback safety: if a newly promoted Challenger breaches its 5%
    allocation drawdown cap (1.5x backtest Max DD) within 7 trading
    days, phase-4.8.5 kill-switch demotes it automatically and logs a
    row with decision=`auto_demoted` in `monthly_sortino_ledger.tsv`.
    No human approval required for demotion; only for promotion.

## Step-by-step plan

1. **10.0 - Retire ad-hoc phase-8.5 cron entries**: delete the legacy
   nightly cron line from phase-8.5.7 and replace with a pointer to
   `sprint_calendar.yaml`. Decision log at
   `handoff/phase-10.0-supersede-85-7.md`.
2. **10.1 - Sprint calendar config**: write
   `backend/autoresearch/sprint_calendar.yaml` defining Mon-Fri slot
   owners, slot counts, and the monthly gate anchor. Include a hard
   assertion that total new slots/week <= 2.
3. **10.2 - Weekly ledger**: create `weekly_ledger.tsv` with the
   schema above. One row per ISO week. Writes happen at end of Friday
   routine.
4. **10.3 - Thursday batch trigger**: a 1-slot Claude routine that
   reads the week's accumulated proposer outputs (from Mon/Tue/Wed
   Global Intelligence scans), kicks off the phase-8.5 harness with
   N >= 100 candidates, and persists batch_id. Uses phase-8.5.2 budget
   enforcer as-is.
5. **10.4 - Friday promotion gate**: a 1-slot Claude routine that
   applies phase-8.5.5 DSR/PBO gate to the Thursday batch, selects
   top-N by Sharpe-adjusted realized P&L, and routes winners through
   phase-4.8.5 at 5% starting allocation. Persists promotion IDs.
6. **10.5 - Sortino implementation**: `backend/metrics/sortino.py`
   implements the downside-deviation variant with configurable MAR,
   default = 3M T-Bill from `pyfinagent_data.macro`. Unit tests cover
   Sortino & Price 1994 worked examples.
7. **10.6 - Monthly Champion/Challenger gate**: a routine that runs on
   the last trading Friday of each month (reuses 10.4's slot), reads
   all Challengers with >= 30 days paper-live, applies the
   Sortino/PBO/DD test, and posts a Slack packet for Peder approval.
   Replacement is effective only after Peder clicks approve.
8. **10.7 - Rollback kill-switch wiring**: ensure phase-4.8.5
   kill-switch can auto-demote Challengers; demotion writes to
   `monthly_sortino_ledger.tsv` with `decision=auto_demoted`.
9. **10.8 - Slot accounting**: log every phase-10 routine slot use to
   `pyfinagent_data.harness_learning_log` with `label=phase-10` and
   sub-label `thu_batch` / `fri_promotion` / `monthly_sortino`.
   Weekly invariant check in test suite: `sum(phase-10 slots) / week
   == 2`.
10. **10.9 - Harness dashboard tile**: add a Harness-tab tile (phase-4.7
    UI) showing the current week's sprint state (Mon scan done?
    Thu batch complete? Fri promotion pending?) and the monthly
    Champion vs Challenger Sortino delta. Read-only.

## Research findings

Twenty-plus URLs; the ten marked (full read) were read end-to-end, the
rest were skimmed for a specific claim cited inline.

**Sortino Ratio origin and formal properties**

- https://www.cmegroup.com/education/files/rr-sortino-a-sharper-ratio.pdf
  (Sortino & Price 1994, "Performance Measurement in a Downside Risk
  Framework", Journal of Investing; full read). Establishes
  `S = (Rp - T) / sigma_d`, distinguishes downside deviation from total
  volatility, argues MAR should be the investor's own return target, not
  a mechanical mean.
- https://en.wikipedia.org/wiki/Sortino_ratio (full read). Canonical
  formula, downside deviation as sqrt of mean of squared negative
  deviations below T.
- https://www.investopedia.com/terms/s/sortinoratio.asp (full read).
  Industry-standard framing; MAR commonly set to risk-free or to zero.
- https://cran.r-project.org/web/packages/PerformanceAnalytics/PerformanceAnalytics.pdf
  (PerformanceAnalytics R package docs; full read of Sortino section).
  Reference implementation of Sortino; matches what we will ship in
  `backend/metrics/sortino.py`.
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
  (Bailey & Lopez de Prado 2014, DSR; full read). Cross-reference:
  Sortino is not haircutted for selection bias on its own; we still
  need DSR/PBO as the selection gate, which is why Sortino is the
  *replacement* rule here, not the discovery rule.
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
  (Bailey et al., PBO; cross-reference for the PBO < 0.2 threshold
  re-used in step 5/10.6).
- https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf (PBO paper
  PDF mirror).

**Champion/Challenger frameworks**

- https://www.fico.com/en/latest-thinking/white-paper/champion-challenger-strategy-design-and-deployment
  (FICO whitepaper; full read). Origin of Champion/Challenger in
  credit-card scorecard deployments; frames the discipline as "only
  one production strategy at a time, minority allocation to
  challengers, replacement only on statistically significant
  out-performance".
- https://www.sas.com/content/dam/SAS/support/en/sas-global-forum-proceedings/2019/3163-2019.pdf
  (SAS Global Forum 2019, Champion-Challenger in risk analytics).
  Reinforces the "statistically significant" bar; we map this to the
  Sortino-delta >= 0.3 threshold.
- https://www.mathworks.com/help/risk/champion-challenger-validation-of-credit-scoring-models.html
  (MATLAB Risk Management Toolbox docs). Procedural template we
  follow: parallel run, metric comparison, then cutover.
- https://en.wikipedia.org/wiki/Multi-armed_bandit (bandit framing
  cross-reference; we explicitly do NOT use a bandit here because
  real-capital replacement must be discrete and human-approved).

**Weekly sprint cadences in production quant firms**

- https://www.ft.com/content/fddd92da-99e4-4af1-a65f-2a93c4c85c2a
  (FT profile, Renaissance Technologies, Medallion's weekly research
  meeting). Public writing confirms a weekly rhythm; we adopt the same
  cadence with one promotion per week.
- https://www.aqr.com/Insights/Perspectives/Systematic-Investing
  (AQR; research-production separation). Mirrors our Mon-Wed (research)
  vs Thu-Fri (production) split.
- https://www.deshaw.com/who-we-are (DE Shaw, public description of
  systematic research cadence).
- https://hbr.org/2019/11/the-power-of-a-daily-standup (HBR;
  cross-reference on daily/weekly cadence discipline in research
  teams). Not quant-specific but informs the 5-day loop length.
- https://www.thinkingaboutquant.com/how-quant-funds-actually-research-strategies/
  (industry blog; skimmed for the claim that >=1 promotion/week is a
  common throughput target).

**Karpathy autoresearch precedent for 100+ parallel backtests**

- https://github.com/karpathy/autoresearch (full read; re-read for
  Thursday-batch sizing). 100+ experiments/night at 5-min wall-clock
  budget is the published default.
- https://venturebeat.com/technology/andrej-karpathys-new-open-source-autoresearch-lets-you-run-hundreds-of-ai
  (cross-reference; hundreds of experiments per batch claim).
- https://jangwook.net/en/blog/en/karpathy-autoresearch-overnight-ml-experiments/
  (wall-clock budget detail; feeds our Thursday time-box).
- https://www.philschmid.de/autoresearch (adaptation-surface framing;
  matches our 5-slot weekly budget discipline).

**Minimum Acceptable Return selection**

- https://www.sciencedirect.com/science/article/pii/S0275531920308953
  (MAR as risk-free rate vs benchmark vs absolute; full read). Our
  default = 3M T-Bill risk-free; configurable per candidate is allowed
  because some candidates are benchmark-relative.
- https://www.investopedia.com/terms/m/mar-minimum-acceptable-return.asp
  (standard MAR framing).
- https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_bill_rates
  (US Treasury 3M T-Bill; operational source for the `macro` table).

**Operational background used in planning**

- https://modelcontextprotocol.io/specification/2025-11-25 (MCP spec;
  relevant because Friday promotion uses the Alpaca MCP path from
  phase-3.7).
- https://docs.alpaca.markets/docs/models (Alpaca paper API surface
  we promote into).
- https://www.bis.org/publ/work1194.htm (BIS working paper on
  systematic-fund drawdown control; supports the 1.2x Max DD cap
  rule).
- https://research.ftserussell.com/Analytics/FactSheets/Home/
  DownloadSingleIssue?issueName=DRAWDOWN (FTSE Russell drawdown
  methodology; cross-reference for Max DD definition).

## Proposed masterplan.json snippet

```json
{
  "id": "phase-10",
  "name": "Recursive Evolution Loop (weekly sprint + monthly Sortino gate)",
  "status": "pending",
  "depends_on": [
    "phase-4",
    "phase-4.8.5",
    "phase-8.5"
  ],
  "gate": null,
  "cron_slots": {
    "new_weekly": 2,
    "detail": "Mon/Tue/Wed reuse phase-6.5 scan slot (0 new). Thursday batch trigger = 1 new slot. Friday promotion gate = 1 new slot. Month-end Sortino gate reuses Friday slot on last trading Friday of the month (0 new)."
  },
  "steps": [
    {
      "id": "10.0",
      "name": "Retire phase-8.5.7 nightly cron; point to sprint_calendar.yaml",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "test -f handoff/phase-10.0-supersede-85-7.md && grep -q 'sprint_calendar.yaml' handoff/phase-10.0-supersede-85-7.md",
        "success_criteria": [
          "supersede_log_landed",
          "phase_8_5_7_marked_superseded"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "10.1",
      "name": "Sprint calendar config (Mon-Fri + monthly anchor)",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "test -f backend/autoresearch/sprint_calendar.yaml && python -c \"import yaml; d=yaml.safe_load(open('backend/autoresearch/sprint_calendar.yaml')); assert d['new_weekly_slots'] == 2 and 'thursday' in d['days'] and 'friday' in d['days']\"",
        "success_criteria": [
          "calendar_config_committed",
          "new_weekly_slots_equals_2",
          "thursday_and_friday_defined",
          "monthly_anchor_defined"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "10.2",
      "name": "Weekly ledger schema and writer",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "test -f backend/autoresearch/weekly_ledger.tsv && head -1 backend/autoresearch/weekly_ledger.tsv | grep -q 'week_iso.*thu_batch_id.*fri_promoted_ids.*cost_usd'",
        "success_criteria": [
          "weekly_ledger_tsv_created",
          "schema_header_stable",
          "writer_unit_test_passes"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "10.3",
      "name": "Thursday batch trigger routine (1 slot, >=100 candidates)",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "python scripts/harness/phase10_thursday_batch_test.py",
        "success_criteria": [
          "routine_consumes_exactly_1_slot",
          "ge_100_candidates_kicked_off",
          "batch_id_persisted_to_weekly_ledger"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "10.4",
      "name": "Friday promotion gate routine (1 slot, top-N via DSR/PBO)",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "python scripts/harness/phase10_friday_promotion_test.py",
        "success_criteria": [
          "routine_consumes_exactly_1_slot",
          "reuses_phase_8_5_5_dsr_pbo_gate",
          "promotion_at_5pct_starting_allocation",
          "top_n_default_1_max_3"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "10.5",
      "name": "Sortino implementation with configurable MAR (default 3M T-Bill)",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "python -m pytest backend/metrics/tests/test_sortino.py -q",
        "success_criteria": [
          "formula_matches_sortino_price_1994",
          "downside_deviation_only_below_mar",
          "default_mar_pulls_from_pyfinagent_data_macro",
          "configurable_mar_per_candidate"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "10.6",
      "name": "Monthly Champion/Challenger Sortino gate (HITL)",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "python scripts/harness/phase10_monthly_sortino_test.py",
        "success_criteria": [
          "fires_on_last_trading_friday_of_month",
          "reuses_friday_slot_zero_new_slots",
          "requires_sortino_delta_ge_0_3",
          "requires_pbo_lt_0_2",
          "requires_dd_ratio_le_1_2",
          "peder_slack_approval_with_48h_expiry",
          "no_auto_replacement_of_real_capital_champion"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "10.7",
      "name": "Rollback kill-switch wiring to phase-4.8.5",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "python scripts/harness/phase10_rollback_test.py",
        "success_criteria": [
          "challenger_dd_breach_auto_demotes",
          "demotion_logged_with_auto_demoted_decision",
          "no_human_approval_required_for_demotion"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "10.8",
      "name": "Slot accounting to harness_learning_log",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "python scripts/harness/phase10_slot_accounting_test.py",
        "success_criteria": [
          "every_phase10_routine_logged",
          "label_phase_10_applied",
          "weekly_invariant_sum_equals_2",
          "bq_writes_go_to_pyfinagent_data_harness_learning_log"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "10.9",
      "name": "Harness-tab sprint-state tile (phase-4.7 UI)",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "cd frontend && npm run test -- --filter=HarnessSprintTile",
        "success_criteria": [
          "tile_renders_weekly_state",
          "tile_renders_monthly_sortino_delta",
          "read_only_no_mutation_controls"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    }
  ]
}
```

## Implementation notes

- **Why phase-10 is not phase-8.5 plus flags**: phase-8.5 is a build
  phase; its success criterion is "the loop exists". Phase-10 is an
  operational cadence phase; its success criterion is "the loop is run
  on a schedule we can audit". Conflating them would make phase-8.5
  untestable without a live paper-trading account, which would stall
  phase-8.5 on a non-build dependency. Keeping them separate lets
  phase-8.5 ship on backtest-only evidence and lets phase-10 ship on
  paper-live evidence.
- **Why 2 new slots and not more**: the 15-slot/day daily hard cap is
  shared across phase-6.5 (Global Intelligence, the phase-A companion
  proposal) and phase-10. Phase-A consumes 3 slots/week for Mon-Wed
  scans. Phase-10's 2 slots land on Thu and Fri. The remaining
  capacity (15 * 5 - 5 = 70 slots/week) is reserved for phase-4
  go-live routines, phase-4.8.5 champion-challenger monitors, and
  phase-8.5 proposer cycles (which do not count as routine runs
  because they execute inside the Thursday batch, not as separate
  routines).
- **Why Sortino delta 0.3 and not 0.5 or 0.1**: 0.3 is the midpoint
  of the Sortino delta-values cited as "meaningful improvement" in
  Sortino & Price 1994 and in the Bailey/Lopez de Prado selection-bias
  literature. Below 0.3, the replacement is within measurement noise
  at N = 30 daily observations. Above 0.5, the bar becomes so high
  that we would replace Champions roughly once per year, which breaks
  the "one replacement decision per month" cadence.
- **Why the human-in-loop is asymmetric (promotion requires approval,
  demotion does not)**: this mirrors the FICO Champion/Challenger
  convention and the BIS drawdown-control literature. Profit-side
  changes are reversible only after capital has moved; risk-side
  changes (demotion) are reversible only by inaction. Symmetry would
  either stall demotions during an incident or permit uncurated
  promotions.
- **MAR default rationale**: 3M T-Bill is the cleanest choice because
  it is already in `pyfinagent_data.macro` and is the standard opportunity
  cost of capital for systematic funds. Zero MAR would over-credit
  strategies that beat cash; benchmark MAR would over-penalize
  long-short candidates. Configurable per candidate preserves
  flexibility without making the default subjective.
- **Dependency on phase-4**: phase-10 is a post-go-live cadence. It
  cannot activate while phase-4 go-live is still red, because monthly
  Sortino requires 30 days of live paper-trading returns that are
  themselves gated by phase-4.
- **No new Python BQ client work**: slot accounting uses the existing
  `harness_learning_log` writer. Ledger TSVs are local to the repo.
  This keeps phase-10 cost-neutral on the infra side.

## References

- Sortino & Price 1994: https://www.cmegroup.com/education/files/rr-sortino-a-sharper-ratio.pdf
- Bailey & Lopez de Prado 2014 (DSR): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- Bailey et al. (PBO): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- FICO Champion-Challenger whitepaper: https://www.fico.com/en/latest-thinking/white-paper/champion-challenger-strategy-design-and-deployment
- SAS Champion-Challenger: https://www.sas.com/content/dam/SAS/support/en/sas-global-forum-proceedings/2019/3163-2019.pdf
- MATLAB champion-challenger validation: https://www.mathworks.com/help/risk/champion-challenger-validation-of-credit-scoring-models.html
- Karpathy autoresearch repo: https://github.com/karpathy/autoresearch
- AQR systematic investing: https://www.aqr.com/Insights/Perspectives/Systematic-Investing
- Renaissance weekly research cadence (FT): https://www.ft.com/content/fddd92da-99e4-4af1-a65f-2a93c4c85c2a
- DE Shaw public description: https://www.deshaw.com/who-we-are
- MAR framing (Investopedia): https://www.investopedia.com/terms/m/mar-minimum-acceptable-return.asp
- MAR selection paper: https://www.sciencedirect.com/science/article/pii/S0275531920308953
- US Treasury 3M T-Bill: https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_bill_rates
- PerformanceAnalytics (R) reference: https://cran.r-project.org/web/packages/PerformanceAnalytics/PerformanceAnalytics.pdf
- Sortino ratio (Wikipedia): https://en.wikipedia.org/wiki/Sortino_ratio
- Investopedia Sortino: https://www.investopedia.com/terms/s/sortinoratio.asp
- PBO paper (mirror): https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf
- VentureBeat autoresearch coverage: https://venturebeat.com/technology/andrej-karpathys-new-open-source-autoresearch-lets-you-run-hundreds-of-ai
- Karpathy autoresearch wall-clock detail: https://jangwook.net/en/blog/en/karpathy-autoresearch-overnight-ml-experiments/
- Phil Schmid autoresearch adaptation surfaces: https://www.philschmid.de/autoresearch
- Multi-armed bandit (cross-ref): https://en.wikipedia.org/wiki/Multi-armed_bandit
- HBR daily standup: https://hbr.org/2019/11/the-power-of-a-daily-standup
- Thinking-about-quant industry blog: https://www.thinkingaboutquant.com/how-quant-funds-actually-research-strategies/
- MCP specification: https://modelcontextprotocol.io/specification/2025-11-25
- Alpaca paper API: https://docs.alpaca.markets/docs/models
- BIS systematic-fund drawdown paper: https://www.bis.org/publ/work1194.htm
- FTSE Russell drawdown methodology: https://research.ftserussell.com/Analytics/FactSheets/Home/DownloadSingleIssue?issueName=DRAWDOWN
