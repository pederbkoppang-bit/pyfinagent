# Phase 10.7 Proposal - Meta-Evolution Engine

**Drafted:** 2026-04-17
**Author:** Principal Sovereign Systems Architect (harness)
**Status:** proposed (not yet written to `.claude/masterplan.json`)
**Proposed id:** phase-10.7
**Depends on:** phase-8.5 (Karpathy autoresearch baseline), phase-4.6 (smoketest),
phase-4.8 (risk/compliance hardening)

## The Red Line

> `Net System Alpha = Profit - (Risk Exposure + Compute Burn)`

Phase-10.7 does NOT search for new alpha directly. It owns the meta-loop
that decides **HOW** the system searches (prompts, archetypes, slot
allocation, provider mix), so every downstream search is more
alpha-per-dollar efficient. The 15 Claude scheduled routine runs per day
is a hard environmental cap; this phase is the governance authority for
the 10 research slots (the 5 trading-ops slots are reserved and
untouchable by this phase).

Baseline: phase-8.5 already provides the Karpathy-style autoresearch
inner loop (propose diff -> backtest -> scalar gate -> commit/revert).
Phase-10.7 is the **outer** loop that rewrites the Research Directive
itself, seeds the archetype space, reallocates slots, and rebalances
providers based on realized Alpha Velocity.

---

## Goal

Build a recursive, self-improving meta-layer that (a) continuously
rewrites the Research Directive prompt when alpha-per-compute falls
below threshold, (b) seeds the phase-8.5 proposer with a diverse set of
non-obvious strategy archetypes, and (c) governs the 15-slot/day cron
budget and per-provider API-credit mix so Net System Alpha is maximized
without single-provider lockout or overrun of the slot cap.

Specifically:
1. A weekly job that scores research sessions by
   `alpha_found / compute_cost`, and when a 4-week moving average falls
   below threshold, emits a proposed git diff to
   `backend/agents/skills/researcher.md`.
2. An archetype seed library covering six non-obvious strategy families
   beyond momentum / mean-reversion.
3. An Alpha Velocity metric, computed per research branch weekly, that
   drives two reallocations: cron slots and API credits.
4. A Cron Budget Allocator that reads and proposes diffs to
   `.claude/cron_budget.yaml` with human-in-loop approval for any diff
   touching trading-ops slots and auto-apply for research-slot-only
   diffs, enforcing total slots <= 15.

## Success criteria

1. `backend/meta_evolution/alpha_velocity.py` (NEW) computes Alpha
   Velocity = realized USD alpha / USD compute cost per branch, writes
   weekly rows to `pyfinagent_data.alpha_velocity_weekly` partitioned on
   `week_ending_date`.
2. `backend/meta_evolution/directive_rewriter.py` (NEW) scans the last
   28 days of planner/researcher transcripts in
   `pyfinagent_data.harness_learning_log`, computes a 4-week MA of
   `alpha_found_usd / compute_cost_usd`, and when below configurable
   threshold (default: 1.5x) emits a git diff proposal file at
   `handoff/meta/proposed_directive_diff.patch`.
3. `backend/meta_evolution/archetype_library.py` (NEW) exposes exactly
   six archetype seeds (names listed in plan below). `len(ARCHETYPES)
   == 6` asserted in a unit test.
4. `.claude/cron_budget.yaml` (NEW) is the single source of truth for
   the 15-slot/day budget, with `reserved_trading_ops` (5 slots) and
   `research` (10 slots) sections. Total slots <= 15 enforced by
   `scripts/meta/validate_cron_budget.py`.
5. `backend/meta_evolution/cron_allocator.py` (NEW) reads current YAML,
   computes Alpha Velocity per branch, emits proposed YAML diff.
   Research-only diffs auto-apply; trading-ops diffs block on
   `handoff/meta/cron_diff_pending.yaml` until Peder signs off.
6. `backend/meta_evolution/provider_rebalancer.py` (NEW) reallocates
   Anthropic / Vertex / OpenAI credit split with hard floors:
   Anthropic >= 20%, Vertex >= 20%, OpenAI >= 10% (sum = 100%), so no
   single outage can zero out the system.
7. Weekly APScheduler job registered in
   `backend/scheduler/meta_cron.py` (NEW) that runs Sundays 06:00 UTC,
   consuming ~1/7 of a daily research slot (co-scheduled with an
   existing idle slot; never an incremental slot).
8. Evaluator agent (`backend/agents/evaluator_agent.py`) gets a new
   entrypoint `review_directive_diff(diff_path)` that returns
   `{approve: bool, rationale: str}`; any diff touching the Research
   Directive must pass this gate before human review.
9. Rollback procedure documented and tested:
   `git revert <sha>` on `backend/agents/skills/researcher.md` restores
   prior directive and the next weekly Alpha Velocity snapshot confirms
   recovery. Runbook file `docs/runbooks/meta_evolution_rollback.md`.
10. Verification commands below are real, executable, and target the
    new files.

## Step-by-step plan

### 10.7.1 Alpha Velocity metric

- **Research gate:** Bailey and Lopez de Prado on trial deflation (for
  denominator choice), Two Sigma and Citadel 2024-2026 public writing on
  cost-adjusted alpha, Weights and Biases compute-ROI framework, Bostrom
  chapter on resource-bounded recursive self-improvement. >=3 sources,
  >=10 URLs, 2 full-paper reads.
- **Plan:** Define
  `alpha_velocity = (USD realized alpha over 4 weeks) / (USD compute
  cost over same window)`. Compute cost = LLM spend + BQ scan bytes *
  $5/TiB + cron-slot opportunity cost ($X/slot/day). Store weekly in
  `pyfinagent_data.alpha_velocity_weekly` (schema: week_ending_date
  DATE, branch_id STRING, alpha_usd FLOAT64, compute_usd FLOAT64,
  velocity FLOAT64, trials INT64).
- **Generate:** `backend/meta_evolution/alpha_velocity.py` with a
  `compute_weekly()` entrypoint.
- **Evaluate:** Backfill 8 weeks from existing
  `harness_learning_log`; spot-check against a hand-computed week.
- **Verification command:**
  ```bash
  python -c "import ast; ast.parse(open('backend/meta_evolution/alpha_velocity.py').read())"
  python -m pytest tests/meta_evolution/test_alpha_velocity.py -v
  ```

### 10.7.2 Recursive Prompt Optimization (Research Directive rewriter)

- **Research gate:** DSPy (Khattab et al., Stanford), TextGrad (Yuksekgonul
  et al., Stanford 2024), Anthropic "Building Effective Agents" (2024) on
  tool-call economy, OpenAI 2025 self-refining-prompts work, Yudkowsky
  on safe bounded recursion.
- **Plan:** New
  `backend/meta_evolution/directive_rewriter.py`. Pull last 28 days of
  planner/researcher transcripts; compute rolling 4-week MA of
  `alpha_found / compute_cost`; if below configurable threshold
  (default 1.5x historical median), call an LLM with the transcripts +
  current directive + alpha-velocity report and request a unified diff
  against `backend/agents/skills/researcher.md`. Write diff to
  `handoff/meta/proposed_directive_diff.patch`. Never mutate the prompt
  file in-place.
- **Generate:** Diff goes to the Evaluator agent
  (`review_directive_diff`) which must return `approve: true` with a
  one-paragraph rationale referencing measurable expected change.
  Approved diffs are posted to Slack for Peder.
- **Evaluate:** A/B: if merged, next 4-week MA must exceed prior 4-week
  MA by >=10% or auto-revert is queued.
- **Rollback:** `git revert <sha>` on the single directive file;
  documented in `docs/runbooks/meta_evolution_rollback.md`.
- **Verification command:**
  ```bash
  python -c "import ast; ast.parse(open('backend/meta_evolution/directive_rewriter.py').read())"
  python -m pytest tests/meta_evolution/test_directive_rewriter.py -v
  test -f docs/runbooks/meta_evolution_rollback.md
  ```

### 10.7.3 Algorithm Discovery - archetype seed library

- **Research gate:** Kelly stochastic programming, DeepMind MuZero,
  FinRL (arXiv 2011.09607), TradeMaster (NeurIPS 2023), Euan Sinclair
  volatility arbitrage writings, 13D Monitor factor-rotation reports,
  Nasdaq TotalView microstructure docs, cross-chain arb papers
  (Chainalysis 2025).
- **Plan:** Six archetype seeds in
  `backend/meta_evolution/archetype_library.py`:
  1. Cross-chain liquidity arbitrage (perp-spot basis, BTC/ETH
     perpetual futures vs spot across CEX/DEX).
  2. LLM-sentiment volatility (IV skew vs news-sentiment imbalance on
     single-name equities).
  3. Event-driven microstructure (13D/13F filing deltas as pre-move
     signal on 10-day horizon).
  4. Lead-lag cross-asset (FX vol -> equity sector rotation, DXY ->
     EM/DM).
  5. Factor rotation (momentum vs quality regime shifts tied to yield
     curve slope).
  6. Volatility surface arbitrage (realized vs implied vol spread
     across tenors / skew).
- **Generate:** Each archetype exposes
  `{name, hypothesis, feature_inputs, data_deps, expected_sharpe_range,
  disqualifying_condition}`. Phase-8.5 proposer iterates this library,
  not a free-text universe.
- **Evaluate:** Archetype is "live" only if it produces at least one
  candidate that clears phase-8.5 DSR and CPCV gates within 4 weeks.
  Dead archetypes are flagged, not removed (reason: regimes return).
- **Verification command:**
  ```bash
  python -c "import ast; ast.parse(open('backend/meta_evolution/archetype_library.py').read())"
  python -c "from backend.meta_evolution.archetype_library import ARCHETYPES; assert len(ARCHETYPES) == 6"
  ```

### 10.7.4 Cron Budget Allocator (MANDATORY - slot governance authority)

- **Research gate:** 15-slot Claude schedule doc, MLOps compute-ROI
  literature, APScheduler max_instances / coalesce docs, GCP outage
  post-mortems (us-east1 Oct 2025), Anthropic status page history, Nov
  2023 OpenAI outage post-mortem.
- **Plan:** Create `.claude/cron_budget.yaml` with exactly two top-level
  sections:
  ```yaml
  # .claude/cron_budget.yaml - authoritative slot budget
  total_daily_slots: 15
  reserved_trading_ops:
    # 5 reserved slots (ids and names; phase-10.7 MAY NOT auto-modify)
    - {slot: T1, cadence: pre_market,   task: market_open_checks}
    - {slot: T2, cadence: mid_day,      task: midday_risk_gate}
    - {slot: T3, cadence: close,        task: eod_reconciliation}
    - {slot: T4, cadence: nightly,      task: paper_trading_cycle}
    - {slot: T5, cadence: nightly,      task: sla_monitor_sweep}
  research:
    # 10 slots; phase-10.7 auto-reallocates these based on Alpha Velocity
    - {slot: R1,  branch: macro_regime}
    - {slot: R2,  branch: news_sentiment}
    - {slot: R3,  branch: cross_asset_lead_lag}
    - {slot: R4,  branch: transformer_signals}
    - {slot: R5,  branch: vol_surface_arb}
    - {slot: R6,  branch: factor_rotation}
    - {slot: R7,  branch: event_microstructure}
    - {slot: R8,  branch: llm_iv_sentiment}
    - {slot: R9,  branch: cross_chain_basis}
    - {slot: R10, branch: meta_evolution_self}   # hosts 10.7's own weekly run
  ```
- **Generate:** `backend/meta_evolution/cron_allocator.py` reads YAML,
  reads latest Alpha Velocity per branch from BQ, computes proposed
  reallocation (e.g. reassign lowest-velocity research slot to highest
  backlog), writes `handoff/meta/cron_diff_pending.yaml` plus a human
  summary. If the diff touches ONLY research slots, auto-applies via
  `scripts/meta/apply_cron_diff.py`. If it touches any
  `reserved_trading_ops` slot, it halts and posts to Slack requiring
  Peder approval. `scripts/meta/validate_cron_budget.py` asserts
  `total_daily_slots <= 15` on every write.
- **Evaluate:** Weekly dry-run for 4 weeks before first live
  auto-application; simulator emits what would have been applied and
  compares to manual baseline.
- **Verification command:**
  ```bash
  test -f .claude/cron_budget.yaml
  python scripts/meta/validate_cron_budget.py .claude/cron_budget.yaml
  python -c "import yaml; d=yaml.safe_load(open('.claude/cron_budget.yaml')); assert d['total_daily_slots'] <= 15; assert len(d['reserved_trading_ops']) == 5; assert len(d['research']) == 10"
  python -m pytest tests/meta_evolution/test_cron_allocator.py -v
  ```

### 10.7.5 API-credit reallocator (multi-provider, with floors)

- **Research gate:** OpenAI Nov 2023 outage post-mortem, Anthropic
  status page incidents 2024-2026, GCP us-east1 Oct 2025 incident,
  multi-cloud LLM routing papers (Martian, Not Diamond), Vertex AI
  quota docs.
- **Plan:** `backend/meta_evolution/provider_rebalancer.py` reads
  7-day provider-level alpha velocity from
  `pyfinagent_data.harness_learning_log` joined with cost ledger,
  proposes new weights in `backend/config/provider_weights.yaml` (NEW).
  Hard floors: Anthropic >= 20%, Vertex >= 20%, OpenAI >= 10%, sum =
  100%. Any proposed weight below floor is clipped and the residual is
  redistributed pro-rata.
- **Generate:** Rebalancer writes
  `handoff/meta/provider_rebalance_proposal.md` with before/after
  weights + expected $ savings. Auto-applies if change per provider is
  <= 10% absolute; otherwise requires human review.
- **Evaluate:** Weekly, and any emergency rebalance triggered if a
  provider 5xx rate exceeds 10% over 1 hour.
- **Verification command:**
  ```bash
  python -c "import ast; ast.parse(open('backend/meta_evolution/provider_rebalancer.py').read())"
  python -m pytest tests/meta_evolution/test_provider_rebalancer.py -v
  ```

### 10.7.6 Weekly scheduler wiring

- **Research gate:** APScheduler docs on misfire_grace_time and
  coalesce, pyfinagent existing `backend/scheduler/`.
- **Plan:** `backend/scheduler/meta_cron.py` registers one weekly job
  `meta_evolution_weekly` at Sunday 06:00 UTC. It runs inside an
  existing idle research slot (R10 per the YAML above) and does NOT
  consume an incremental slot. Job orchestration:
  alpha_velocity.compute_weekly -> directive_rewriter.maybe_propose ->
  cron_allocator.propose -> provider_rebalancer.propose -> Slack
  summary.
- **Generate:** APScheduler `CronTrigger(day_of_week='sun', hour=6,
  minute=0)`, `max_instances=1`, `coalesce=True`.
- **Evaluate:** Dry-run with `--simulate` flag for 2 cycles before
  live.
- **Verification command:**
  ```bash
  python -c "import ast; ast.parse(open('backend/scheduler/meta_cron.py').read())"
  python -m pytest tests/scheduler/test_meta_cron.py -v
  ```

### 10.7.7 Evaluator review gate for directive diffs

- **Research gate:** Anthropic 2024 "Building Effective Agents"
  (evaluator-optimizer pattern), constitutional-AI critique-and-revise
  literature.
- **Plan:** Extend `backend/agents/evaluator_agent.py` with
  `review_directive_diff(diff_path: str) -> {approve: bool, rationale:
  str}`. The evaluator must reject any diff that (a) removes
  falsifiability language, (b) increases prompt token count > 20%, or
  (c) eliminates any of the 6 archetypes.
- **Generate:** Rejection writes to
  `handoff/meta/directive_diff_rejected.md`; approval writes to
  `handoff/meta/directive_diff_approved.md` and pings Peder via Slack.
- **Evaluate:** Unit test feeds three synthetic diffs (one good, two
  that violate each guardrail) and asserts the expected outcome.
- **Verification command:**
  ```bash
  python -m pytest tests/agents/test_evaluator_directive_review.py -v
  ```

### 10.7.8 Runbook and rollback drill

- **Plan:** `docs/runbooks/meta_evolution_rollback.md` documents:
  identify bad directive commit -> `git revert <sha>` ->
  `python scripts/meta/verify_directive_restore.py` -> watch
  `alpha_velocity_weekly` for one week.
- **Generate:** Schedule a quarterly rollback drill in the runbook.
- **Verification command:**
  ```bash
  test -f docs/runbooks/meta_evolution_rollback.md
  grep -q "git revert" docs/runbooks/meta_evolution_rollback.md
  ```

---

## Research findings

### Recursive self-improvement and bounded recursion

- Bostrom N., "Superintelligence", chapter on recursive self-improvement
  dynamics. https://global.oup.com/academic/product/superintelligence-9780199678112
- Yudkowsky E., "Levels of Organization in General Intelligence" (MIRI,
  foundational on bounded recursion).
  https://intelligence.org/files/LOGI.pdf
- Anthropic, "Alignment faking in LLMs" (2024) - why fully-autonomous
  self-rewriting is dangerous without human review.
  https://www.anthropic.com/research/alignment-faking
- DeepMind, "Sparrow" and "agent-from-scratch" (2022-2024) on
  iterated-distillation-and-amplification.
  https://www.deepmind.com/blog/building-safer-dialogue-agents
- METR, "Evaluations of advanced AI agents" (2024) - 4-hour task
  horizons growing at 7 months doubling.
  https://metr.org/blog/2024-11-22-evaluating-r-d-capabilities-of-llms/

### Prompt optimization frameworks

- Khattab O. et al., "DSPy: Compiling Declarative Language Model Calls"
  (Stanford, 2023). https://arxiv.org/abs/2310.03714
- Yuksekgonul M. et al., "TextGrad: Automatic Differentiation via Text"
  (Stanford, 2024). https://arxiv.org/abs/2406.07496
- Anthropic, "Building Effective Agents" (Dec 2024) - tool-call economy
  and evaluator-optimizer pattern.
  https://www.anthropic.com/research/building-effective-agents
- OpenAI research, "Self-refining prompts" (2025).
  https://openai.com/research/
- Fernando C. et al., "Promptbreeder" (DeepMind 2023).
  https://arxiv.org/abs/2309.16797
- Zhou Y. et al., "Large Language Models Are Human-Level Prompt
  Engineers" (APE, ICLR 2023). https://arxiv.org/abs/2211.01910

### Algorithm discovery in quant

- Kelly J. L., "A New Interpretation of Information Rate" (1956), the
  original stochastic-programming allocation.
  https://www.princeton.edu/~wbialek/rome/refs/kelly_56.pdf
- Schrittwieser J. et al., "MuZero" (DeepMind, Nature 2020).
  https://www.nature.com/articles/s41586-020-03051-4
- Liu X.-Y. et al., "FinRL: Deep RL for Finance" (NeurIPS 2020
  workshop). https://arxiv.org/abs/2011.09607
- Sun S. et al., "TradeMaster" (NeurIPS 2023). https://arxiv.org/abs/2306.17120
- Sinclair E., "Volatility Trading" (Wiley, 2nd ed.) - IV/RV spread as
  structural edge.
- Bailey D., Lopez de Prado M., "The Deflated Sharpe Ratio" (2014).
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- Lopez de Prado M., "Advances in Financial Machine Learning" (Wiley
  2018) - PBO, CPCV.

### Alpha Velocity / profit-per-compute

- Two Sigma Engineering, "Efficient trading research" (2024-2025
  public writing). https://www.twosigma.com/insights/
- Citadel GQS, 2024-2026 public talks on compute-adjusted Sharpe.
  https://www.citadelsecurities.com/about-us/insights/
- Weights and Biases, "Compute ROI in ML" framework.
  https://wandb.ai/site/articles/compute-roi
- AWS ML blog, "Cost-aware hyperparameter optimization" (2024).
  https://aws.amazon.com/blogs/machine-learning/

### Multi-provider LLM routing and single-provider risk

- OpenAI, "Nov 8 2023 incident post-mortem".
  https://status.openai.com/incidents/
- Anthropic status page, incident archive 2024-2026.
  https://status.anthropic.com/history
- Google Cloud, "us-east1 incident Oct 2025" post-mortem.
  https://status.cloud.google.com/summary
- Martian, "Model Router" docs. https://withmartian.com/docs
- Not Diamond, router benchmarks (2024-2025).
  https://www.notdiamond.ai/
- Vertex AI quota and regional failover.
  https://cloud.google.com/vertex-ai/docs/quotas

### Compute-budget governance / MLOps

- Sculley D. et al., "Hidden Technical Debt in ML Systems" (NeurIPS
  2015). https://papers.nips.cc/paper/5656
- APScheduler docs on `max_instances`, `coalesce`,
  `misfire_grace_time`.
  https://apscheduler.readthedocs.io/en/3.x/userguide.html
- Karpathy A., "autoresearch" (GitHub, 2025) - the baseline inner loop
  that phase-8.5 adopts. https://github.com/karpathy/autoresearch

---

## Proposed masterplan.json snippet

```json
{
  "id": "phase-10.7",
  "name": "Meta-Evolution Engine",
  "status": "proposed",
  "depends_on": ["phase-8.5", "phase-4.6", "phase-4.8"],
  "cron_slots": "1/7 weekly (shares research slot R10, zero incremental consumption)",
  "red_line_contract": {
    "objective": "Net System Alpha = Profit - (Risk Exposure + Compute Burn)",
    "cost_introduced": "One weekly LLM rewrite call (~$1) + one weekly BQ scan (<1 GiB)",
    "profit_edge_defended": "Continuous improvement of research directive + archetype seeding -> higher alpha per trial",
    "risk_cap": "Total daily slots <= 15; trading-ops slots (5) immutable by this phase; provider floors 20/20/10"
  },
  "steps": [
    {
      "id": "10.7.1",
      "name": "Alpha Velocity metric + BQ table",
      "files_new": [
        "backend/meta_evolution/alpha_velocity.py",
        "tests/meta_evolution/test_alpha_velocity.py"
      ],
      "bq_new": ["pyfinagent_data.alpha_velocity_weekly"],
      "verification": "python -m pytest tests/meta_evolution/test_alpha_velocity.py -v"
    },
    {
      "id": "10.7.2",
      "name": "Recursive Prompt Optimization (Research Directive rewriter)",
      "files_new": [
        "backend/meta_evolution/directive_rewriter.py",
        "tests/meta_evolution/test_directive_rewriter.py",
        "docs/runbooks/meta_evolution_rollback.md"
      ],
      "files_modified": ["backend/agents/skills/researcher.md (via git diff only, never in-place)"],
      "verification": "python -m pytest tests/meta_evolution/test_directive_rewriter.py -v"
    },
    {
      "id": "10.7.3",
      "name": "Algorithm Discovery archetype seed library",
      "files_new": [
        "backend/meta_evolution/archetype_library.py",
        "tests/meta_evolution/test_archetype_library.py"
      ],
      "archetypes": [
        "cross_chain_basis",
        "llm_iv_sentiment",
        "event_microstructure",
        "cross_asset_lead_lag",
        "factor_rotation",
        "vol_surface_arb"
      ],
      "verification": "python -c \"from backend.meta_evolution.archetype_library import ARCHETYPES; assert len(ARCHETYPES) == 6\""
    },
    {
      "id": "10.7.4",
      "name": "Cron Budget Allocator (slot governance authority)",
      "files_new": [
        ".claude/cron_budget.yaml",
        "backend/meta_evolution/cron_allocator.py",
        "scripts/meta/validate_cron_budget.py",
        "scripts/meta/apply_cron_diff.py",
        "tests/meta_evolution/test_cron_allocator.py"
      ],
      "invariants": [
        "total_daily_slots <= 15",
        "len(reserved_trading_ops) == 5 and is immutable by auto-apply",
        "len(research) == 10",
        "Any diff touching reserved_trading_ops blocks on human approval"
      ],
      "verification": "python scripts/meta/validate_cron_budget.py .claude/cron_budget.yaml && python -m pytest tests/meta_evolution/test_cron_allocator.py -v"
    },
    {
      "id": "10.7.5",
      "name": "API-credit reallocator with per-provider floors",
      "files_new": [
        "backend/meta_evolution/provider_rebalancer.py",
        "backend/config/provider_weights.yaml",
        "tests/meta_evolution/test_provider_rebalancer.py"
      ],
      "floors": {"anthropic": 0.20, "vertex": 0.20, "openai": 0.10},
      "verification": "python -m pytest tests/meta_evolution/test_provider_rebalancer.py -v"
    },
    {
      "id": "10.7.6",
      "name": "Weekly APScheduler wiring",
      "files_new": [
        "backend/scheduler/meta_cron.py",
        "tests/scheduler/test_meta_cron.py"
      ],
      "schedule": "Sunday 06:00 UTC, runs inside research slot R10, zero incremental slot cost",
      "verification": "python -m pytest tests/scheduler/test_meta_cron.py -v"
    },
    {
      "id": "10.7.7",
      "name": "Evaluator review gate for directive diffs",
      "files_modified": ["backend/agents/evaluator_agent.py"],
      "files_new": ["tests/agents/test_evaluator_directive_review.py"],
      "guardrails": [
        "reject diff that removes falsifiability language",
        "reject diff that grows prompt token count > 20%",
        "reject diff that eliminates any of the 6 archetypes"
      ],
      "verification": "python -m pytest tests/agents/test_evaluator_directive_review.py -v"
    },
    {
      "id": "10.7.8",
      "name": "Runbook and rollback drill",
      "files_new": ["docs/runbooks/meta_evolution_rollback.md"],
      "drill_cadence": "quarterly",
      "verification": "test -f docs/runbooks/meta_evolution_rollback.md && grep -q 'git revert' docs/runbooks/meta_evolution_rollback.md"
    }
  ]
}
```

---

## Implementation notes

- ASCII-only throughout. No emojis in any file created by this phase.
- Python 3.14 target. Dependencies already present in `requirements.txt`:
  `apscheduler`, `pyyaml`, `google-cloud-bigquery`, `anthropic`.
- The Research Directive file (`backend/agents/skills/researcher.md`)
  is modified ONLY by producing a unified diff; never `open(..., 'w')`.
  The apply step is an explicit `git apply` invoked by
  `scripts/meta/apply_directive_diff.py` after evaluator + human
  approval.
- `.claude/cron_budget.yaml` is the SINGLE source of truth for slot
  count. Any other cron-defining file (e.g. `cloud-scheduler-*.yaml`)
  must `ref:` a slot id from this file. Validation script enforces
  referential integrity.
- Alpha Velocity uses gross realized alpha (pre-tax, post-slippage) to
  avoid double-counting risk penalties that the Evaluator already
  applies.
- Provider floors (20/20/10) are chosen so that any single-provider
  full outage loses <= 70% capacity and the orchestrator can continue
  with degraded throughput. Floors are config, not code constants.
- The 15-slot cap is enforced in THREE places: YAML schema validator,
  `cron_allocator.propose()` hard check, and APScheduler job count on
  startup.
- Weekly job is intentionally low-frequency: meta-evolution must be
  slower than the phase-8.5 inner loop to avoid the meta-overfitting
  analog of Bailey / Lopez de Prado's Probability of Backtest
  Overfitting at the directive layer.
- All rewrites produce a SHA in the commit message referencing the
  Alpha Velocity snapshot that motivated them, enabling forensic
  reconstruction.

---

## References

See Research findings section above. URL count target: >= 20 distinct
sources across recursive self-improvement, prompt optimization,
algorithm discovery, alpha-velocity, multi-provider risk, and
compute-budget governance literature.

Key citations for quick review:
- https://arxiv.org/abs/2310.03714 (DSPy)
- https://arxiv.org/abs/2406.07496 (TextGrad)
- https://arxiv.org/abs/2309.16797 (Promptbreeder)
- https://arxiv.org/abs/2211.01910 (APE)
- https://arxiv.org/abs/2011.09607 (FinRL)
- https://arxiv.org/abs/2306.17120 (TradeMaster)
- https://www.nature.com/articles/s41586-020-03051-4 (MuZero)
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 (DSR)
- https://www.anthropic.com/research/building-effective-agents
- https://www.anthropic.com/research/alignment-faking
- https://github.com/karpathy/autoresearch
- https://status.openai.com/incidents/
- https://status.anthropic.com/history
- https://status.cloud.google.com/summary
- https://apscheduler.readthedocs.io/en/3.x/userguide.html
- https://intelligence.org/files/LOGI.pdf
- https://papers.nips.cc/paper/5656 (Hidden Technical Debt)
- https://metr.org/blog/2024-11-22-evaluating-r-d-capabilities-of-llms/
- https://withmartian.com/docs
- https://www.notdiamond.ai/
- https://cloud.google.com/vertex-ai/docs/quotas
- https://www.twosigma.com/insights/
- https://www.citadelsecurities.com/about-us/insights/
- https://wandb.ai/site/articles/compute-roi
- https://www.princeton.edu/~wbialek/rome/refs/kelly_56.pdf
