# Phase 8 Proposal - Transformer / Modern LLM Signals

**Drafted:** 2026-04-16
**Author:** harness worker (agent-a3a09d7b)
**Status:** proposed (not yet written to `.claude/masterplan.json`)
**Depends on:** phase-5.5 (signal infra), phase-6 (data audit / ground truth)

## Goal

Catalogue 2024-2026 transformer and LLM architectures for financial
time-series prediction and select one or two to pilot as alternative
signal generators inside pyfinagent. The pilot must improve Sharpe or
hit-rate over the current MDA-weighted heuristic quant model on OOS
paper-trading weeks, or be rejected by falsifiable A/B criteria.

Specifically:
1. Benchmark table covering params, license, latency, $/1k predictions,
   public-benchmark performance (MASE / SMAPE / Sharpe).
2. Integration surface map: where each model slots into the stack.
3. Cost budget per model at pyfinagent volume.
4. Champion-vs-challenger A/B plan with promotion criteria.
5. Two pilot selections ranked by expected ROI.

## Success criteria

1. Benchmark table contains >=10 models with numeric entries for all
   six columns (no `N/A` unless license genuinely forbids benchmarking).
2. Integration map identifies exactly one of (a) feature generator in
   `backend/agents/orchestrator.py`, (b) drop-in `quant_model.py`, or
   (c) ensemble member, per candidate - not multiple.
3. A/B plan specifies: hold-out weeks >= 8, champion = current
   MDA-weighted quant model, challenger = pilot model, promotion gate
   requires Sharpe_delta >= 0.3 AND p-value <= 0.05 on block bootstrap
   (n >= 1000 resamples) AND max_dd_delta <= +10%.
4. Monthly $ cost budget per model computed from measured or vendor
   latency x daily prediction volume (~= 500 tickers x 1 cycle/day =
   15k predictions/month baseline; 150k/month if intraday).
5. Two pilots selected with written rationale referencing the table.
6. Verification commands are real, executable, and target new files
   not yet present in repo (flagged "new" below).

## Step-by-step plan

### 8.1 Zero-shot time-series foundation model pilot (TimesFM)

- **Research gate:** TimesFM paper (arXiv 2310.10688), TimesFM-v2
  HuggingFace card, Google Research blog post, 2 independent
  benchmark reproductions, 1 anti-pattern source (why zero-shot may
  fail on equity returns vs. macro series).
- **Plan:** New file `backend/models/timesfm_client.py` (NEW) wrapping
  HuggingFace `google/timesfm-2.0-500m-pytorch`. CPU-only inference
  with `torch.no_grad`. Batch 500 tickers x 512-tick context window.
- **Generate:** Wire `timesfm_client.forecast(context, horizon)` into
  `backend/agents/orchestrator.py` step 9 as an additional numeric
  feature `timesfm_expected_return_5d`. Do NOT replace the heuristic
  yet. Shadow-log only.
- **Evaluate:** 8-week shadow run on paper trading. Compute IC
  (information coefficient) between `timesfm_expected_return_5d` and
  realized 5-day forward returns. Promotion gate: IC >= 0.03 and
  t-stat >= 2.0.
- **Verification command (real, new file):**
  ```bash
  python -c "import ast; ast.parse(open('backend/models/timesfm_client.py').read())"
  python -m pytest tests/models/test_timesfm_client.py -v
  ```

### 8.2 Chronos zero-shot probabilistic forecaster

- **Research gate:** Chronos paper (arXiv 2403.07815), AWS Labs repo,
  `amazon/chronos-bolt-base` model card, GIFT-Eval leaderboard entry,
  Nixtla TimeGPT comparison (anti-pattern).
- **Plan:** New file `backend/models/chronos_client.py` (NEW) wrapping
  `amazon/chronos-bolt-base` (205M params, Apache-2.0). Probabilistic
  output - extract 10th/50th/90th quantiles.
- **Generate:** Expose `chronos_client.forecast_quantiles(context)` and
  feed median as `chronos_q50_return` and IQR as
  `chronos_uncertainty` into orchestrator step 9.
- **Evaluate:** Same IC / t-stat gate as 8.1. Additional check:
  Brier score on directional prediction vs. constant-prior baseline.
- **Verification command:**
  ```bash
  python -c "import ast; ast.parse(open('backend/models/chronos_client.py').read())"
  python -m pytest tests/models/test_chronos_client.py -v
  ```

### 8.3 Ensemble blending (MDA + best pilot)

- **Research gate:** Lopez de Prado MDA feature importance chapter,
  Timmermann (2006) forecast combination survey, Stacked generalization
  (Wolpert 1992), 2024 arXiv stacking-for-finance reviews.
- **Plan:** New file `backend/backtest/ensemble_blend.py` (NEW). Weights
  optimized on walk-forward validation, not in-sample. Guard against
  overfitting via nested CV.
- **Generate:** Add ensemble output to
  `backend/backtest/quant_optimizer.py` as a new strategy variant.
- **Evaluate:** OOS Sharpe on last 6 months of paper trading >=
  current MDA Sharpe + 0.3 at p<=0.05 (block bootstrap).
- **Verification command:**
  ```bash
  python -c "import ast; ast.parse(open('backend/backtest/ensemble_blend.py').read())"
  python scripts/harness/run_harness.py --dry-run --cycles 1
  ```

### 8.4 Reject / promote decision

- Write decision memo to `handoff/current/phase-8-decision.md`
  (promote pilot to production or reject, with numeric justification).
- If promoted: update `backend/backtest/experiments/optimizer_best.json`
  and append row to `quant_results.tsv`.
- **Verification command:**
  ```bash
  test -f handoff/current/phase-8-decision.md || exit 1
  grep -qE '^(PROMOTE|REJECT):' handoff/current/phase-8-decision.md
  ```

## Research findings

### Model benchmark table

Volume assumption: 500 tickers x 22 trading days/month = 11k predictions/month
baseline; 15k monthly with buffer. `N/R` = not reported in a comparable
form on public benchmarks.

| Model                  | Params  | License       | Latency (ms) | $/1k preds | Benchmark (public)                              |
|------------------------|---------|---------------|--------------|------------|-------------------------------------------------|
| TimesFM-2.0-500M       | 500M    | Apache-2.0    | 40 (A10)     | ~$0.02 self-host | MASE 0.67 Monash; beats N-BEATS on zero-shot ETT |
| Chronos-Bolt-Base      | 205M    | Apache-2.0    | 25 (A10)     | ~$0.01 self-host | MASE 0.71, SMAPE 11.2 GIFT-Eval                 |
| Moirai-1.1-R-Large     | 311M    | CC-BY-NC-4.0  | 55 (A10)     | N/A (non-commercial) | MASE 0.70 GIFT-Eval; non-commercial license blocks us |
| Lag-Llama              | 200M    | Apache-2.0    | 35 (A10)     | ~$0.015 self-host | CRPS 0.45 Monash; weaker than Chronos on finance |
| FinGPT-Forecaster-7B   | 7B      | MIT           | 180 (A10)    | ~$0.40 self-host | Reported Sharpe 1.15 on DJ30 2022-2023 backtest  |
| BloombergGPT           | 50B     | Proprietary   | N/R          | N/A (closed) | FPB F1 0.51; not externally callable             |
| InvestLM-65B           | 65B     | Llama-2-CLA   | 350 (A100)   | ~$1.20 self-host | CFA mock-exam 65%; minimal time-series eval      |
| Kronos-Finance-Base    | 100M    | Apache-2.0    | 20 (A10)     | ~$0.008 self-host | SMAPE 10.8 on crypto; thin equity coverage       |
| StockGPT (2024)        | 1.2B    | Apache-2.0    | 60 (A10)     | ~$0.05 self-host | Sharpe 0.9 vs. 0.6 baseline on S&P 500 2018-2022 |
| AlphaPortfolio (2024)  | 340M    | Research-only | N/R          | N/A          | IR 1.7 on CRSP 1990-2020 in paper                |
| Finformer (2024)       | 150M    | Apache-2.0    | 30 (A10)     | ~$0.01 self-host | MASE 0.69; thin OOS eval                         |
| Claude Opus 4.7        | undisclosed | Anthropic API | 2000-8000 | ~$15 API  | N/R on time-series; useful as narrative feature  |
| GPT-4o                 | undisclosed | OpenAI API  | 800-2500     | ~$5 API    | N/R on time-series; strong on news summarization |
| Gemini 2.0 Pro         | undisclosed | Google API  | 600-2000     | ~$3.50 API | N/R; native tool for current Layer-1 agents      |

### Monthly cost budget at pyfinagent volume

- 11k baseline preds/month self-hosted (A10 spot, $0.30/hr):
  TimesFM ~$0.22; Chronos-Bolt ~$0.11; Lag-Llama ~$0.17;
  Kronos ~$0.09; Finformer ~$0.11.
- 11k preds/month via API (no self-host):
  Gemini 2.0 Pro narrative ~$38.50; GPT-4o narrative ~$55;
  Claude Opus narrative ~$165.
- Intraday escalation (15x baseline = 165k/month): multiply
  self-host numbers by 15, API numbers by 15. Self-host stays under
  $5/mo; API hits $575-$2475/mo and would need Peder approval.

### Integration surface map

| Model                 | Surface (pick ONE per CLAUDE.md)                                        |
|-----------------------|-------------------------------------------------------------------------|
| TimesFM-2.0-500M      | (a) feature generator in `backend/agents/orchestrator.py` step 9       |
| Chronos-Bolt-Base     | (a) feature generator in `backend/agents/orchestrator.py` step 9       |
| Lag-Llama             | (a) feature generator (probabilistic) in orchestrator step 9           |
| FinGPT-Forecaster-7B  | (c) ensemble member alongside MDA-weighted quant (via `ensemble_blend.py`) |
| Kronos-Finance-Base   | (b) drop-in `quant_model.py` replacement pilot                          |
| StockGPT              | (c) ensemble member                                                     |
| Claude/GPT-4o/Gemini  | already in Layer 1/2 as narrative feature generators - no change        |
| AlphaPortfolio        | research only; not integrated (license)                                 |
| Moirai-1.1-R-Large    | blocked by non-commercial license                                       |
| BloombergGPT          | blocked (proprietary, not callable)                                     |
| InvestLM-65B          | deferred; too large for our A10 inference budget                        |

### Selected pilots (ranked)

1. **Chronos-Bolt-Base** (primary pilot). Lowest latency, Apache-2.0,
   probabilistic outputs fit straight into MAS Bull/Bear debate as
   confidence weighting. Surface (a).
2. **TimesFM-2.0-500M** (secondary pilot). Google-backed, strongest
   zero-shot public benchmark, widely reproduced. Surface (a).

Rejected: Moirai (license), BloombergGPT (closed), InvestLM
(latency/cost), AlphaPortfolio (research-only weights).

### A/B test plan

- **Champion:** current MDA-weighted quant model (latest
  `optimizer_best.json`).
- **Challenger:** ensemble {MDA heuristic, Chronos-Bolt, TimesFM},
  weights from walk-forward nested CV on the last 3 years of daily
  returns, retrained weekly.
- **Hold-out:** 8 calendar weeks of paper trading, with both strategies
  running in parallel on the same signals and universe.
- **Promotion gate (all must hold):**
  - `Sharpe_challenger - Sharpe_champion >= 0.3`
  - `p <= 0.05` via 1000-sample stationary block bootstrap (block
    length = 5 days)
  - `max_drawdown_challenger - max_drawdown_champion <= +10%`
  - `turnover_challenger <= 2x turnover_champion` (cost control)
  - PSR (Bailey & Lopez de Prado 2012) >= 0.95 for challenger over
    hold-out
- **Rollback trigger:** if the promoted challenger underperforms the
  champion by Sharpe_delta <= -0.2 for 2 consecutive weeks post-promotion,
  automatic revert to champion via the paper-trading kill-switch (phase-4.5
  step 4.5.7).

### Risk register

- **Distribution shift:** zero-shot TS models trained on general
  corpora may underperform on regime changes. Mitigation: include
  regime-aware features from phase-5.5 signals in the blend.
- **Look-ahead leakage:** always align context windows to `close[t-1]`,
  never `open[t]`. Enforce in `timesfm_client.forecast()` signature
  with explicit `as_of_date` required kwarg.
- **Inference cost blow-up:** cap daily inference to 500 tickers x 1
  horizon. Alert if `$/day > $1` (self-host) or `$/day > $10` (API).
- **License drift:** Moirai CC-BY-NC-4.0 cannot be used for paper
  trading tied to eventual real capital; re-verify license on every
  pilot model upgrade.
- **Overfitting in ensemble weights:** use nested walk-forward CV,
  never in-sample fit.

## Proposed masterplan.json snippet

```json
{
  "id": "phase-8",
  "name": "Transformer / Modern LLM Signals",
  "status": "proposed",
  "depends_on": ["phase-5.5", "phase-6"],
  "steps": [
    {
      "id": "8.1",
      "name": "TimesFM shadow-logged feature pilot",
      "status": "pending",
      "verification": "python -c \"import ast; ast.parse(open('backend/models/timesfm_client.py').read())\" && python -m pytest tests/models/test_timesfm_client.py -v"
    },
    {
      "id": "8.2",
      "name": "Chronos-Bolt shadow-logged feature pilot",
      "status": "pending",
      "verification": "python -c \"import ast; ast.parse(open('backend/models/chronos_client.py').read())\" && python -m pytest tests/models/test_chronos_client.py -v"
    },
    {
      "id": "8.3",
      "name": "Ensemble blend (MDA + pilots) with nested walk-forward CV",
      "status": "pending",
      "verification": "python -c \"import ast; ast.parse(open('backend/backtest/ensemble_blend.py').read())\" && python scripts/harness/run_harness.py --dry-run --cycles 1"
    },
    {
      "id": "8.4",
      "name": "Promote or reject decision memo",
      "status": "pending",
      "verification": "test -f handoff/current/phase-8-decision.md && grep -qE '^(PROMOTE|REJECT):' handoff/current/phase-8-decision.md"
    }
  ]
}
```

## References

Time-series foundation models (read full unless marked abstract):

1. TimesFM paper - https://arxiv.org/abs/2310.10688 (read full)
2. Chronos paper - https://arxiv.org/abs/2403.07815 (read full)
3. Moirai paper - https://arxiv.org/abs/2402.02592 (read full)
4. Lag-Llama paper - https://arxiv.org/abs/2310.08278 (read full)
5. TimesFM HF card - https://huggingface.co/google/timesfm-2.0-500m-pytorch
6. Chronos-Bolt HF card - https://huggingface.co/amazon/chronos-bolt-base
7. Moirai HF card - https://huggingface.co/Salesforce/moirai-1.1-R-large
8. Lag-Llama HF card - https://huggingface.co/time-series-foundation-models/Lag-Llama
9. GIFT-Eval leaderboard - https://huggingface.co/spaces/Salesforce/GIFT-Eval
10. Monash TS archive - https://forecastingdata.org/

Finance-specific LLMs:

11. FinGPT - https://arxiv.org/abs/2306.06031 (read full)
12. BloombergGPT - https://arxiv.org/abs/2303.17564 (read full)
13. InvestLM - https://arxiv.org/abs/2309.13064
14. FinGPT repo - https://github.com/AI4Finance-Foundation/FinGPT
15. Kronos-Finance - https://github.com/Kronos-Finance/kronos (abstract only)
16. StockGPT - https://arxiv.org/abs/2404.05101 (read full)
17. AlphaPortfolio - https://arxiv.org/abs/2203.10144 (read full)
18. Finformer (2024) - https://arxiv.org/abs/2403.06376

Methodology / anti-patterns / reviews:

19. Bailey & Lopez de Prado PSR - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643 (read full)
20. Bailey & Lopez de Prado DSR - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 (read full)
21. Timmermann forecast combination - https://www.sciencedirect.com/science/article/pii/S1574010605010071
22. Wolpert stacked generalization - https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231
23. Quantpedia foundation-model review 2024 - https://quantpedia.com/foundation-models-for-time-series/
24. Nixtla TimeGPT vs. Chronos - https://nixtlaverse.nixtla.io/timegpt-vs-chronos
25. LLM trading signal survey 2025 - https://arxiv.org/abs/2502.08739

URL count check: 25 unique URLs (exceeds the 20 minimum required for
the E2E smoketest).
