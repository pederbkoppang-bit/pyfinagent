# Phase A (phase-4.9) - Immutable Core and Risk Guard

Status: proposal (pending)
Owner: Peder / harness
Depends on: phase-4.8 (risk and compliance hardening lands the primitives this phase makes immutable)
Gate: HARD GATE for phase-4 go-live. The Gauntlet gates every autoresearch-promoted strategy (phase-8.5) entering paper-live.

## Goal

Phase 4.8 (see `handoff/phase-proposals/phase-4.8-risk-compliance-hardening.md`)
delivers the runtime risk primitives - CVaR gate, factor-exposure cap,
Kelly allocator, drift monitor, champion-challenger promotion. Phase A
EXTENDS that baseline by making the specific numerical thresholds
NON-HOT-RELOADABLE and by forcing every self-evolved strategy through a
7-regime black-swan Gauntlet before any virtual-money allocation.

The Red Line for this phase: `Net System Alpha = Profit - (Risk Exposure +
Compute Burn)`. An immutable limits file bounds Risk Exposure with a
tamper-evident audit trail; the Gauntlet bounds tail risk by rejecting
strategies whose drawdown in a stress window exceeds 1.5x their backtest
maximum; neither subsystem consumes any of the 15 daily Claude routine
slots (config is read at process start, Gauntlet runs on-demand only).

This is the "constitutional" layer of the Intelligence Engine. Phase 4.8
says "we compute CVaR and block trades above 2% of NAV". Phase A says
"the 2% is frozen in a signed file and any change requires a tagged
governance commit by Peder, independently verified at process start".

## Success criteria

1. `docs/governance/immutable_limits.yaml` exists, is schema-validated on
   every process start, and contains the six required limits: max per-
   position notional, max portfolio leverage, max daily drawdown, max
   single-name concentration, max sector concentration, min liquidity
   filter.
2. Any commit that modifies `immutable_limits.yaml` is a signed, annotated
   git tag (`git tag -s limits-YYYYMMDD-NN`) authored by the governance
   owner (`peder.bkoppang@hotmail.no`); CI rejects unsigned or unowned
   mutations.
3. Backend refuses to start if (a) the file is missing, (b) the file is
   not covered by a signed tag, (c) the schema does not validate, or (d)
   the in-process loader detects a mid-run mutation (hot-reload is
   explicitly disabled - SIGHUP does NOT re-read the file).
4. The Gauntlet runner (`scripts/risk/gauntlet.py`) executes all 7
   historical regimes plus a 1000-path Monte Carlo tail simulation at
   the 99th percentile against any candidate strategy and produces a
   deterministic pass/fail report.
5. Pass criteria per regime: strategy max drawdown <= 1.5x backtest-
   window max drawdown AND zero liquidity-driven forced exits (defined
   as any fill at price outside +/-5 sigma of prior-minute mid on a
   sized-below-ADV20 name).
6. Monte Carlo pass criterion: 99th-percentile simulated drawdown <=
   2.0x observed backtest drawdown AND no simulated path breaches the
   max-daily-drawdown immutable limit.
7. Autoresearch promotion pipeline (phase-8.5) refuses to advance any
   strategy to `allocation_pct > 0` until `handoff/gauntlet/<strategy_id>
   /report.json` shows `overall_pass == true`; promotion writes the
   gauntlet report hash into `optimizer_best.json` as provenance.
8. Gauntlet results are logged to BigQuery (`pyfinagent_data
   .gauntlet_results`) with the strategy id, regime id, drawdown, forced-
   exit count, and Monte Carlo percentile for audit.
9. `cron_slots: 0` - the phase adds zero scheduled Claude routine runs;
   the Gauntlet is on-demand (triggered by the phase-8.5 promotion
   hook), and the immutable-limits check runs in-process on startup.
10. Phase completion gate: a red-team attempt to mutate
    `immutable_limits.yaml` without a signed tag must fail closed in CI,
    and an intentional Gauntlet-failing strategy must be rejected by the
    promotion pipeline. Both negative tests logged in
    `handoff/phase-4.9-redteam.md`.

## Step-by-step plan

### 4.9.0 - Schema and file for immutable limits

Create `docs/governance/immutable_limits.yaml` with the six bounded
numeric limits plus metadata (owner, approved_at, tag_ref, hash). Create
`backend/governance/limits_schema.py` with a pydantic v2 model. Values
are the ones already decided in phase 4.8 (NAV <= 2% CVaR, beta <= 1.2,
single-strategy <= 30%, momentum <= 0.8) plus new ones: per-position
notional <= 5% of NAV, portfolio leverage <= 1.5x, daily drawdown <=
3% of NAV, single-name concentration <= 8% of NAV, sector concentration
<= 35% of NAV, min liquidity = ADV20 >= 2,000,000 USD.

Verification: `python -c "from backend.governance.limits_schema import load; l=load(); assert l.max_position_notional_pct == 0.05 and l.max_portfolio_leverage == 1.5"`

Criteria: `limits_file_exists`, `schema_validates`, `six_limits_present`.

### 4.9.1 - Tag-signed-commit enforcement in CI

Add `.github/workflows/immutable_limits_guard.yml` which on any PR
touching `docs/governance/immutable_limits.yaml` requires: (a) the head
commit be pointed at by a signed annotated git tag
(`git verify-tag $(git describe --exact-match)`), (b) the tagger email
match the single allowed governance owner list in
`docs/governance/OWNERS`, (c) the tag message contain an `approval:`
line with rationale. Fails the PR otherwise.

Verification: `bash scripts/governance/verify_limits_tag.sh --dry-run`

Criteria: `ci_workflow_landed`, `unsigned_push_rejected`, `wrong_owner_rejected`, `approval_message_required`.

### 4.9.2 - Startup loader with no hot-reload

`backend/governance/limits_loader.py` reads the YAML once at process
start, computes a SHA-256 of file contents, stores the digest in
`backend.state.LIMITS_DIGEST`, and registers a filesystem watcher that
raises `ImmutableLimitsMutated` and force-exits the process on any
post-start mutation. SIGHUP is explicitly ignored (contrasted with
tunable configs which DO hot-reload). A unit test confirms SIGHUP does
not re-read the file.

Verification: `python -c "from backend.governance.limits_loader import load_once, get_digest; load_once(); d=get_digest(); assert len(d) == 64"`

Criteria: `load_once_pattern`, `sighup_ignored`, `mutation_kills_process`, `digest_exposed_to_healthcheck`.

### 4.9.3 - Runtime enforcement hooks

Wire the six limits into existing gates: CVaR/FF3 gate
(`backend/services/portfolio_risk.py`), Kelly allocator
(`scripts/risk/kelly_allocator.py`), order router
(`backend/services/order_router.py`). Each call-site reads from the
frozen-at-startup `LimitsSnapshot` rather than from env vars or live
YAML. Any path that takes a limit value from anywhere other than the
snapshot is a lint violation.

Verification: `python scripts/governance/lint_limits_usage.py --strict`

Criteria: `all_callsites_use_snapshot`, `no_env_var_fallback`, `lint_in_ci`.

### 4.9.4 - Gauntlet regime catalog

Create `backend/backtest/gauntlet/regimes.py` with the seven historical
windows hard-coded as date ranges and per-regime asset-universe hints:
2008 GFC (2008-09-01..2008-12-31), 2010 Flash Crash (2010-05-06
intraday + T-5 to T+5 daily), 2015 SNB un-peg (2015-01-14..2015-01-22
with CHF pairs flagged), 2020 COVID crash (2020-02-15..2020-03-31),
2022 rate-shock (2022-01-03..2022-06-30), 2024 yen carry unwind
(2024-07-29..2024-08-12), 2025 tariff spike (2025-04-02..2025-04-18).

Verification: `python -c "from backend.backtest.gauntlet.regimes import REGIMES; assert len(REGIMES) == 7 and all('start' in r and 'end' in r for r in REGIMES)"`

Criteria: `seven_regimes_defined`, `date_ranges_immutable`, `universe_hints_present`.

### 4.9.5 - Gauntlet runner

`scripts/risk/gauntlet.py <strategy_id>` loads the strategy spec,
replays it through each regime via the existing `backtest_engine.py`,
computes per-regime max drawdown and forced-exit count, runs a 1000-
path Monte Carlo tail simulation (block bootstrap on the full history
with tail-amplification scalar = 1.5 at the 99th percentile), emits
`handoff/gauntlet/<strategy_id>/report.json`, and appends a row to
`pyfinagent_data.gauntlet_results` in BigQuery. Deterministic seed
derived from `strategy_id` hash for reproducibility.

Verification: `python scripts/risk/gauntlet.py --strategy baseline --dry-run && python -c "import json; r=json.load(open('handoff/gauntlet/baseline/report.json')); assert 'per_regime' in r and len(r['per_regime']) == 7 and 'monte_carlo' in r"`

Criteria: `seven_regimes_executed`, `monte_carlo_1000_paths`, `bq_row_appended`, `deterministic_seed`, `report_schema_valid`.

### 4.9.6 - Pass-criteria enforcement

`backend/backtest/gauntlet/evaluator.py` compares per-regime drawdown
against the 1.5x backtest-window threshold, counts forced exits, checks
the 99th-percentile simulated drawdown against 2.0x observed, and
verifies no simulated path breaches the max-daily-drawdown immutable
limit. Emits `overall_pass: bool` and `reason: str` in the report.

Verification: `python -c "from backend.backtest.gauntlet.evaluator import evaluate; r=evaluate({'per_regime':[{'id':'gfc_2008','drawdown':0.12,'bt_drawdown':0.10,'forced_exits':0}]*7,'monte_carlo':{'p99_drawdown':0.18,'bt_drawdown':0.10,'breaches':0}}); assert r['overall_pass'] == True"`

Criteria: `drawdown_ratio_enforced`, `forced_exits_zero_required`, `mc_p99_checked`, `immutable_limit_breach_blocks`.

### 4.9.7 - Promotion-pipeline integration

Modify `scripts/risk/promotion_gate.py` (from phase 4.8.5) to refuse
any `allocation_pct > 0` without a passing Gauntlet report. Write the
report hash into `optimizer_best.json` alongside `allocation_pct`.
The first promotion rung (5%) now requires: live-Sharpe > 0.5 AND
live-IS < 10 bps AND no CVaR breach AND `gauntlet.overall_pass ==
true AND gauntlet.report_hash == file_hash`.

Verification: `python scripts/risk/promotion_gate.py --dry-run --require-gauntlet && grep -q '"gauntlet_report_hash"' backend/backtest/experiments/optimizer_best.json`

Criteria: `gauntlet_required_for_promotion`, `hash_recorded_in_best_json`, `promotion_fails_without_gauntlet`.

### 4.9.8 - Autoresearch wiring (phase-8.5 hook)

`backend/autonomous_harness.py` and the phase-8.5 autoresearch flow
trigger `gauntlet.py` as the final synchronous step before any
virtual-money allocation. Failure writes to
`handoff/harness_log.md` with `gauntlet_result: FAIL <reason>` and
blocks further promotion of that strategy for 30 days (stored in
BQ `pyfinagent_data.gauntlet_blocklist`).

Verification: `python -c "from backend.autonomous_harness import promote_strategy; import pytest; pytest.raises(Exception, promote_strategy, 'intentionally_bad_strategy')"`

Criteria: `autoresearch_calls_gauntlet`, `30day_blocklist`, `harness_log_annotated`.

### 4.9.9 - Red-team negative tests

Two scripted adversarial tests in `handoff/phase-4.9-redteam.md`:
(1) submit a PR that edits `immutable_limits.yaml` without a signed
tag and confirm CI rejects; (2) construct an intentionally Gauntlet-
failing strategy (5x leverage, no stop loss) and confirm
`promote_strategy` raises. Both results logged with timestamps.

Verification: `test -f handoff/phase-4.9-redteam.md && grep -q 'REDTEAM_PASS' handoff/phase-4.9-redteam.md`

Criteria: `unsigned_mutation_blocked`, `bad_strategy_blocked`, `evidence_logged`.

## Research findings

Every numeric threshold in this phase is grounded in a specific regime
or academic finding. Citations are indexed against the `References`
section below.

### Adversarial backtesting and why 7 regimes is the minimum (not 1)

Bailey, Borwein, Lopez de Prado, and Zhu [1, 2, 3] formalize the
probability of backtest overfitting (PBO) as the fraction of in-sample
"best" strategies that underperform out-of-sample. Their Combinatorially
Symmetric Cross-Validation framework demonstrates that a single
historical path is pathologically insufficient: the more configurations
tried, the more certain the backtest is overfit. Lopez de Prado [4, 5]
extends this to the Deflated Sharpe Ratio and explicitly argues that
self-evolving quant systems MUST test against regimes the optimizer
never saw, not just resampled draws of the training regime.

The implication for pyfinagent: our autoresearch loop (phase-8.5) is a
PBO factory unless every promotion is gated on out-of-regime evidence.
Seven disjoint stress regimes plus 1000 Monte Carlo tail paths is the
minimum to push estimated PBO below 0.15 at a 5% haircut on Sharpe,
per the reference R and Python implementations [5, 7].

### SEC 15c3-5 and FINRA 3110: hard pre-trade limits must be immutable in spirit

SEC Rule 15c3-5 (Market Access Rule) [8, 9, 10] requires every broker-
dealer to maintain pre-trade financial and regulatory controls under
"direct and exclusive control" and reviewed on a regular schedule. The
Wilmer Hale FAQ analysis [11] and NASDAQ trading note [12] both
emphasize that a limit bypass - even intraday, even by an algorithm -
is a rule violation, not a technicality. Unfiltered or "naked" access
is effectively prohibited: every order must pass through the firm's
own limit layer on a pre-trade basis.

FINRA Rule 3110 [13, 14] requires a documented supervisory system with
change-control evidence (review logs, escalations, corrective actions).
Our `immutable_limits.yaml` + signed-tag + CI verify mapping satisfies
both the "pre-set credit or capital thresholds" clause of 15c3-5(c)(1)
and the "evidence that supervision has occurred" clause of 3110.

The non-hot-reload choice is informed by Flash Crash post-mortems: a
limit that can be changed mid-session by a production process is, in
practice, a limit that WILL be changed when someone panics.

### 2010 Flash Crash - why "min liquidity" is a hard limit

The SEC/CFTC joint report [15, 16, 17] documents that the 2010 Flash
Crash was triggered by a single large sell program (Waddell & Reed's
75,000 E-Mini S&P contracts) executed without a price or liquidity
constraint - "9% of trailing minute volume, no matter what". At
2:45:28 pm EDT a 5-second CME Stop Logic pause arrested the cascade.
The Berman SEC speech [18] and subsequent academic review [19] both
identify the missing control as a minimum-liquidity floor at the
executing-venue level.

This is why `min_liquidity_filter` (ADV20 >= $2M) is in the immutable
set. The Gauntlet's 2010 regime specifically stress-tests any strategy
for behavior on thinly traded names during the 2:32-2:48 pm window;
candidates that route size into a vanishing book get flagged as
liquidity-driven forced exits.

### 2015 SNB un-peg - why leverage must be capped ABOVE the single-position view

FXCM's second-by-second data [20, 21, 22] shows EUR/CHF dropping 40%
in seconds at the 04:30 EST SNB announcement, with liquidity gone
from EUR/CHF quotes for 45 minutes. Spreads widened to 2000-3000 pips
on an average 6000 pip range. The Hedgeweek post-mortem [23]
documents that FX brokers with apparently-safe strategy-level limits
but NO portfolio-leverage cap were insolvent before the first hour
was out.

This drives two immutable limits: `max_portfolio_leverage <= 1.5x`
(an absolute cap that ignores Kelly's suggestion, which would allow
higher on high-Sharpe regimes) and `single_name_concentration <= 8%`.
The 2015 regime in the Gauntlet specifically checks that strategies
holding implicit FX exposure via ADR or cross-listed names do not
silently breach `portfolio_leverage` when a USD-denominated carry
instantly becomes a 1.3x-leveraged short.

### 2020 COVID dash-for-cash - Treasury-base liquidity IS an equity risk

The BIS working paper [24], Liberty Street post [25], OFR analysis
[26], and Fed staff note [27] document the March 2020 Treasury
dysfunction: 10-year yields rose 64 bps March 9-18 while equities
fell, breaking the standard risk-off correlation. Mutual-fund
outflows, foreign official selling, and hedge-fund deleveraging
simultaneously hit dealer balance sheets. The Fed's unprecedented
$125B/day purchases restored function; without them the cascade
would have been orders worse.

This is why `max_daily_drawdown <= 3% of NAV` is non-negotiable:
a strategy that "should" only be down 1% can absolutely be down 5%
on a correlation-regime-break day, and the immutable limit is what
prevents the algorithm from sizing back in at the false bottom. The
2020 regime in the Gauntlet explicitly checks behavior when historical
correlation assumptions invert.

### 2024 yen carry unwind - Black Monday 1987 intensity in 2 days

BIS Bulletin 90 [28], Wellington post-mortem [29], and ION Group
analysis [30] document the August 5 2024 Nikkei single-day decline
(biggest since 1987 Black Monday), driven by a 15 bp BOJ hike
compounding expected Fed cuts. Roughly $250B in carry positions
unwound. The critical finding [29, 30]: strategies predicated on
contained volatility were FORCED to unwind by deleveraging pressure
and margin increases, creating the textbook vol-positive-feedback
loop.

This drives the Gauntlet's forced-exit criterion. A strategy that
"passes" historical max-drawdown but exited at -8 sigma during the
2024 regime is NOT promotable - the exit was margin-driven, not
signal-driven. We count any fill > 5 sigma from prior-minute mid on
a sub-ADV20 size as liquidity-driven.

### April 2025 tariff spike - the most recent regime IS the most valuable test

The Liberation Day tariff announcement on April 2 2025 [31, 32, 33]
triggered the largest 2-day US market loss in history (Dow -4000+
points, S&P 500 -10%, VIX peaked at 45.31). Because this regime is
(a) very recent and (b) almost certainly NOT in the training data
for any strategy our autoresearch loop has proposed to date (all
optimizer runs to date terminated before April 2025 data was
ingested per the data audit in phase 5.5), it is the highest-
signal out-of-sample test available.

### Monte Carlo choice: 1000 paths, 99th percentile, block bootstrap

The MDPI Basel III paper [34] and the GARP survey [35] both argue
that for regulatory-grade backtesting 1,000 paths is the floor for
a stable 99th percentile estimate (they recommend 5k-25k for capital
models but 1k for strategy-gate use). The QuantVPS [36] and Faccia
book [37] confirm that a block-bootstrap with tail amplification
at 1.5x is the simplest defensible estimator that doesn't assume
normality - critical because every citation above [1, 15, 20, 24,
28, 31] documents fat tails that Gaussian VaR would miss entirely.

The choice is: 1000 paths (defensible floor per [34, 35]), 99th
percentile (fat-tail aware, avoiding the statistically shaky 99.99th),
and 2.0x vs observed drawdown as the Monte Carlo pass threshold -
sitting between the 1.5x historical-regime ratio and the Basel III
stressed-VaR multiplier of ~3.0x. Calibration is re-validated in step
4.9.9 against the known-good baseline strategy before the gate is
considered binding.

### Immutable-infrastructure pattern for limits

The IBM and Spacelift immutable-infrastructure write-ups [38, 39]
and the QodeQuay risk framing [40] establish the pattern: config
that can be changed only by replacing the artifact, with the artifact
identified by a content-addressable tag (Git SHA for us). The Legit
Security analysis [41] explicitly calls out signed commits as the
cryptographic root of an audit trail that stands up in regulatory
examination. Basel III's internal-ratings-based floors [42, 43]
provide the regulatory precedent: parameters used in capital
calculation cannot be hot-reloaded; they require governance
approval and a version trail.

The key design choice: SIGHUP does NOT re-read. Every limits
consumer reads from a frozen snapshot. The filesystem watcher
is there only to force-EXIT on mutation, not to re-load. This
is the exact opposite of the pattern for tunable parameters
(strategy weights, regime-switch thresholds) which DO hot-reload.

## Proposed masterplan.json snippet

```json
{
  "id": "phase-4.9",
  "name": "Immutable Core and Risk Guard",
  "status": "pending",
  "depends_on": ["phase-4.8"],
  "gate": "hard_gate_for_phase_4_go_live_and_phase_8_5_promotion",
  "steps": [
    {"id": "4.9.0", "name": "Schema and file for immutable limits", "status": "pending", "harness_required": false, "verification": {"command": "python -c \"from backend.governance.limits_schema import load; l=load(); assert l.max_position_notional_pct == 0.05 and l.max_portfolio_leverage == 1.5\"", "success_criteria": ["limits_file_exists", "schema_validates", "six_limits_present"]}, "contract": null, "retry_count": 0, "max_retries": 3},
    {"id": "4.9.1", "name": "Tag-signed-commit enforcement in CI", "status": "pending", "harness_required": false, "verification": {"command": "bash scripts/governance/verify_limits_tag.sh --dry-run", "success_criteria": ["ci_workflow_landed", "unsigned_push_rejected", "wrong_owner_rejected", "approval_message_required"]}, "contract": null, "retry_count": 0, "max_retries": 3},
    {"id": "4.9.2", "name": "Startup loader with no hot-reload", "status": "pending", "harness_required": false, "verification": {"command": "python -c \"from backend.governance.limits_loader import load_once, get_digest; load_once(); d=get_digest(); assert len(d) == 64\"", "success_criteria": ["load_once_pattern", "sighup_ignored", "mutation_kills_process", "digest_exposed_to_healthcheck"]}, "contract": null, "retry_count": 0, "max_retries": 3},
    {"id": "4.9.3", "name": "Runtime enforcement hooks wired to snapshot", "status": "pending", "harness_required": false, "verification": {"command": "python scripts/governance/lint_limits_usage.py --strict", "success_criteria": ["all_callsites_use_snapshot", "no_env_var_fallback", "lint_in_ci"]}, "contract": null, "retry_count": 0, "max_retries": 3},
    {"id": "4.9.4", "name": "Gauntlet regime catalog (7 historical windows)", "status": "pending", "harness_required": false, "verification": {"command": "python -c \"from backend.backtest.gauntlet.regimes import REGIMES; assert len(REGIMES) == 7 and all('start' in r and 'end' in r for r in REGIMES)\"", "success_criteria": ["seven_regimes_defined", "date_ranges_immutable", "universe_hints_present"]}, "contract": null, "retry_count": 0, "max_retries": 3},
    {"id": "4.9.5", "name": "Gauntlet runner (7 regimes + 1000 MC paths)", "status": "pending", "harness_required": true, "verification": {"command": "python scripts/risk/gauntlet.py --strategy baseline --dry-run && python -c \"import json; r=json.load(open('handoff/gauntlet/baseline/report.json')); assert 'per_regime' in r and len(r['per_regime']) == 7 and 'monte_carlo' in r\"", "success_criteria": ["seven_regimes_executed", "monte_carlo_1000_paths", "bq_row_appended", "deterministic_seed", "report_schema_valid"]}, "contract": null, "retry_count": 0, "max_retries": 3},
    {"id": "4.9.6", "name": "Pass-criteria evaluator", "status": "pending", "harness_required": false, "verification": {"command": "python -c \"from backend.backtest.gauntlet.evaluator import evaluate; r=evaluate({'per_regime':[{'id':'gfc_2008','drawdown':0.12,'bt_drawdown':0.10,'forced_exits':0}]*7,'monte_carlo':{'p99_drawdown':0.18,'bt_drawdown':0.10,'breaches':0}}); assert r['overall_pass'] == True\"", "success_criteria": ["drawdown_ratio_enforced", "forced_exits_zero_required", "mc_p99_checked", "immutable_limit_breach_blocks"]}, "contract": null, "retry_count": 0, "max_retries": 3},
    {"id": "4.9.7", "name": "Promotion-pipeline integration", "status": "pending", "harness_required": false, "verification": {"command": "python scripts/risk/promotion_gate.py --dry-run --require-gauntlet && grep -q '\"gauntlet_report_hash\"' backend/backtest/experiments/optimizer_best.json", "success_criteria": ["gauntlet_required_for_promotion", "hash_recorded_in_best_json", "promotion_fails_without_gauntlet"]}, "contract": null, "retry_count": 0, "max_retries": 3},
    {"id": "4.9.8", "name": "Autoresearch wiring (phase-8.5 hook)", "status": "pending", "harness_required": true, "verification": {"command": "python -c \"from backend.autonomous_harness import promote_strategy; import pytest; pytest.raises(Exception, promote_strategy, 'intentionally_bad_strategy')\"", "success_criteria": ["autoresearch_calls_gauntlet", "30day_blocklist", "harness_log_annotated"]}, "contract": null, "retry_count": 0, "max_retries": 3},
    {"id": "4.9.9", "name": "Red-team negative tests", "status": "pending", "harness_required": true, "verification": {"command": "test -f handoff/phase-4.9-redteam.md && grep -q 'REDTEAM_PASS' handoff/phase-4.9-redteam.md", "success_criteria": ["unsigned_mutation_blocked", "bad_strategy_blocked", "evidence_logged"]}, "contract": null, "retry_count": 0, "max_retries": 3}
  ]
}
```

## Implementation notes

### Files to create

- `docs/governance/immutable_limits.yaml` - the frozen limits
- `docs/governance/OWNERS` - one-line governance owner list
- `backend/governance/__init__.py`
- `backend/governance/limits_schema.py` - pydantic model
- `backend/governance/limits_loader.py` - load_once + digest + watcher
- `.github/workflows/immutable_limits_guard.yml`
- `scripts/governance/verify_limits_tag.sh`
- `scripts/governance/lint_limits_usage.py`
- `backend/backtest/gauntlet/__init__.py`
- `backend/backtest/gauntlet/regimes.py`
- `backend/backtest/gauntlet/evaluator.py`
- `scripts/risk/gauntlet.py`
- `handoff/phase-4.9-redteam.md` (populated in 4.9.9)

### Files to modify

- `scripts/risk/promotion_gate.py` (from phase 4.8.5) - add
  `--require-gauntlet` flag
- `backend/autonomous_harness.py` - call Gauntlet synchronously
  before any `allocation_pct > 0` promotion
- `backend/backtest/experiments/optimizer_best.json` - add
  `gauntlet_report_hash` field schema

### BigQuery tables to create

- `pyfinagent_data.gauntlet_results` - one row per (strategy_id,
  regime_id) with drawdown, forced_exits, pass/fail
- `pyfinagent_data.gauntlet_blocklist` - strategy_id with 30-day
  TTL after a Gauntlet failure

### Rollout risk

- **Low risk**: steps 4.9.0-4.9.3 are additive. Existing gates keep
  working; the snapshot is an extra constraint layer.
- **Medium risk**: step 4.9.5 (Gauntlet runner) could reveal that
  the current baseline strategy itself fails one or more regimes.
  That is EXPECTED and is the point - remediate before go-live.
  Budget: if baseline fails >= 2 regimes, treat as phase-4.8.2 CVaR
  regression and fix before closing phase-4.9.
- **Low risk**: step 4.9.7 promotion integration is guarded behind
  `--require-gauntlet` flag; can be feature-flipped.

### Cost estimate

- **Compute**: one Gauntlet run ~ 7 * full-history backtest + 1000
  MC paths. On current infra ~ 45-60 min per candidate. Runs only
  on-demand (strategy promotion), not on cron. Typical cadence:
  1-3 runs per week. BQ query cost ~$0.05 per run (historical
  data is cached).
- **LLM cost**: zero. Gauntlet is numeric only, no LLM calls.
- **Cron slots consumed: 0 (on-demand only)**. No impact on the 15
  daily Claude scheduled routine budget - the Gauntlet is triggered
  by the phase-8.5 promotion hook and the immutable-limits check
  runs in-process on backend startup.
- **One-time engineering**: ~3-5 days for the governance loader +
  signed-tag CI, ~5-8 days for the Gauntlet runner including regime
  catalog and evaluator.

### Interaction with phase 4.8

Phase 4.8 delivers the measurements (CVaR, factor exposure, Kelly,
PSI, drift, runbooks). Phase 4.9 immobilizes the thresholds on those
measurements and adds the out-of-regime stress test. Phase 4.9 does
not re-implement any 4.8 primitive - it wraps them.

| Concept | 4.8 | 4.9 |
|---|---|---|
| CVaR 97.5% gate at 2% NAV | computed and enforced | threshold frozen in `immutable_limits.yaml` |
| Single-strategy 30% cap | enforced in allocator | threshold frozen in `immutable_limits.yaml` |
| Beta <= 1.2, momentum <= 0.8 | enforced in FF3 gate | thresholds frozen in `immutable_limits.yaml` |
| Champion-challenger promotion | 5%->25%->100% with live metrics | adds Gauntlet pass as mandatory precondition |
| Stress testing | single walk-forward | 7 regimes + 1000-path MC |

## References

1. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253 - Bailey, Borwein, Lopez de Prado, Zhu, "The Probability of Backtest Overfitting" (SSRN, 2015). Accessed 2026-04-17.
2. https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf - Full preprint of Bailey et al., PBO paper. Accessed 2026-04-17.
3. https://sdm.lbl.gov/oapapers/ssrn-id2507040-bailey.pdf - Bailey et al., "Statistical Overfitting and Backtest Performance" (LBL). Accessed 2026-04-17.
4. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 - Bailey and Lopez de Prado, "The Deflated Sharpe Ratio". Accessed 2026-04-17.
5. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678 - Lopez de Prado, "Building Diversified Portfolios that Outperform Out-of-Sample". Accessed 2026-04-17.
6. https://www.risk.net/investing/quant-investing/6993166/some-quant-shops-doomed-to-struggle-lopez-de-prado - Risk.net interview on self-evolving quant systems and PBO. Accessed 2026-04-17.
7. https://cran.r-project.org/web/packages/pbo/vignettes/pbo.html - CRAN `pbo` package reference implementation. Accessed 2026-04-17.
8. https://www.law.cornell.edu/cfr/text/17/240.15c3-5 - 17 CFR 240.15c3-5, Market Access Rule full text. Accessed 2026-04-17.
9. https://www.sec.gov/files/rules/final/2010/34-63241-secg.htm - SEC Small-Entity Compliance Guide for 15c3-5. Accessed 2026-04-17.
10. https://www.sec.gov/rules-regulations/2011/06/risk-management-controls-brokers-or-dealers-market-access - SEC press release on 15c3-5 adoption. Accessed 2026-04-17.
11. https://www.wilmerhale.com/en/insights/client-alerts/sec-staff-issues-first-set-of-faqs-on-rule-15c3-5-risk-management-controls-for-brokers-or-dealers-with-market-access - WilmerHale client alert on 15c3-5 FAQ. Accessed 2026-04-17.
12. https://www.nasdaqtrader.com/content/productsservices/trading/ften/sec_mar.pdf - NASDAQ note on 15c3-5 implementation. Accessed 2026-04-17.
13. https://www.finra.org/rules-guidance/rulebooks/finra-rules/3110 - FINRA Rule 3110 Supervision. Accessed 2026-04-17.
14. https://www.innreg.com/resources/finra-rules/3110-supervision - InnReg explanation of Rule 3110. Accessed 2026-04-17.
15. https://www.sec.gov/news/studies/2010/marketevents-report.pdf - SEC/CFTC joint report, "Findings Regarding the Market Events of May 6, 2010" (Flash Crash). Accessed 2026-04-17.
16. https://www.sec.gov/sec-cftc-prelimreport.pdf - SEC/CFTC preliminary Flash Crash report. Accessed 2026-04-17.
17. https://www.cftc.gov/sites/default/files/idc/groups/public/@economicanalysis/documents/file/oce_flashcrash0314.pdf - CFTC OCE analysis of HFT and the Flash Crash. Accessed 2026-04-17.
18. https://www.sec.gov/news/speech/2010/spch101310geb.htm - SEC Berman speech on Flash Crash participants. Accessed 2026-04-17.
19. https://www.econstor.eu/bitstream/10419/313247/1/1676811400.pdf - Akansu, "The Flash Crash: A Review". Accessed 2026-04-17.
20. https://www.globenewswire.com/news-release/2015/03/12/714464/33730/en/FXCM-Releases-Detailed-Data-on-the-SNB-Flash-Crash.html - FXCM data release on SNB un-peg. Accessed 2026-04-17.
21. https://www.financemagnates.com/forex/brokers/fxcm-publishes-data-of-snb-mishandling-of-the-swiss-franc/ - FinanceMagnates second-by-second FXCM data. Accessed 2026-04-17.
22. https://link.springer.com/chapter/10.1007/978-3-031-23194-0_7 - Springer chapter on SNB and CHF role in global markets. Accessed 2026-04-17.
23. https://www.hedgeweek.com/absorbing-currency-market-liquidity-shocks/ - Hedgeweek post-mortem on SNB liquidity gap. Accessed 2026-04-17.
24. https://www.bis.org/publ/work966.pdf - BIS Working Paper 966, "The Treasury market in spring 2020 and the response of the Federal Reserve". Accessed 2026-04-17.
25. https://libertystreeteconomics.newyorkfed.org/2020/04/treasury-market-liquidity-during-the-covid-19-crisis/ - NY Fed Liberty Street post on March 2020 Treasury liquidity. Accessed 2026-04-17.
26. https://www.financialresearch.gov/the-ofr-blog/2023/04/03/ofr-models-one-theory-on-the-cause-of-march-2020s-treasury-market-fragility/ - OFR analysis of March 2020 Treasury fragility. Accessed 2026-04-17.
27. https://www.federalreserve.gov/econres/notes/feds-notes/treasury-market-functioning-during-the-covid-19-outbreak-evidence-from-collateral-re-use-20201204.html - Fed FEDS Note on COVID Treasury functioning. Accessed 2026-04-17.
28. https://www.bis.org/publ/bisbull90.pdf - BIS Bulletin 90, "The market turbulence and carry trade unwind of August 2024". Accessed 2026-04-17.
29. https://www.wellington.com/en/insights/the-yen-carry-trade-unwind - Wellington Management analysis of yen carry unwind. Accessed 2026-04-17.
30. https://iongroup.com/blog/markets/yen-carry-trade-volatility-a-reminder-that-trusted-technology-and-partnerships-matter/ - ION Group post on Aug 2024 vol mechanics. Accessed 2026-04-17.
31. https://en.wikipedia.org/wiki/2025_stock_market_crash - Wikipedia summary of April 2025 Liberation Day crash. Accessed 2026-04-17.
32. https://en.wikipedia.org/wiki/Liberation_Day_tariffs - Wikipedia Liberation Day tariffs entry. Accessed 2026-04-17.
33. https://internationalbanker.com/banking/how-trumps-liberation-day-tariffs-sparked-a-global-credit-risk-shock/ - International Banker, Liberation Day credit-risk shock. Accessed 2026-04-17.
34. https://www.mdpi.com/2227-9091/13/8/146 - MDPI, "Monte Carlo-Based VaR Estimation and Backtesting Under Basel III". Accessed 2026-04-17.
35. https://www.garp.org/risk-intelligence/credit/the-case-for-monte-carlo-simulations - GARP, "The Case for Monte Carlo Simulations". Accessed 2026-04-17.
36. https://www.quantvps.com/blog/yen-carry-trade-unwind-explained - QuantVPS, yen carry unwind mechanics (also used for MC discussion). Accessed 2026-04-17.
37. https://www.amazon.com/Black-Swan-Monte-Carlo-Simulations-ebook/dp/B0GTMZR1RC - Faccia, "Black Swan Monte Carlo Simulations" (reference text). Accessed 2026-04-17.
38. https://www.ibm.com/think/topics/immutable-infrastructure - IBM, "What is Immutable Infrastructure?" Accessed 2026-04-17.
39. https://spacelift.io/blog/what-is-immutable-infrastructure - Spacelift, immutable-infra best practices. Accessed 2026-04-17.
40. https://www.qodequay.com/immutable-infrastructure-risk-reduction - QodeQuay, risk reduction via immutable infra. Accessed 2026-04-17.
41. https://www.legitsecurity.com/aspm-knowledge-base/what-is-immutable-infrastructure - Legit Security, immutable-infra audit trail implications. Accessed 2026-04-17.
42. https://www.bis.org/bcbs/publ/d424.pdf - Basel Committee, "Basel III: Finalising post-crisis reforms" (IRB parameter floors). Accessed 2026-04-17.
43. https://www.bis.org/bcbs/publ/d424_hlsummary.pdf - BIS Basel III reforms high-level summary. Accessed 2026-04-17.
44. https://www.finra.org/rules-guidance/rulebooks/finra-rules/4210 - FINRA Rule 4210 margin requirements. Accessed 2026-04-17.
45. https://www.finra.org/rules-guidance/guidance/reports/2022-finras-examination-and-risk-monitoring-program/portfolio-margin-intraday-trading - FINRA guidance on portfolio margin concentration. Accessed 2026-04-17.
