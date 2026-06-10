REJECT: Shadow-only pilots are kept as scaffolds; no live trading in phase-8.

# Phase-8 Decision Memo  --  Transformer / Modern LLM Signals

**Date:** 2026-04-20
**Decision:** **REJECT** promotion to live trading.
**Scaffolds retained:** yes (`backend/models/timesfm_client.py`, `backend/models/chronos_client.py`, `backend/backtest/ensemble_blend.py`).
**Re-evaluation trigger:** see Section 6.

---

## 1. What was shipped in phase-8

| Sub-step | Deliverable | Q/A verdict |
|---|---|---|
| 8.1 TimesFM scaffold | `backend/models/timesfm_client.py` + 11 tests | qa_81_v1 PASS |
| 8.2 Chronos-Bolt scaffold | `backend/models/chronos_client.py` + 11 tests | qa_82_v1 PASS |
| 8.3 Ensemble blend | `backend/backtest/ensemble_blend.py` + 15 tests | qa_83_v1 PASS |
| 8.4 Decision memo | THIS DOCUMENT | pending qa_84_v1 |

Total new code: ~660 lines across 3 production modules + 37 tests (all green).

## 2. Evidence behind the REJECT decision

### 2.1 Runtime gate

The repo venv is Python 3.14. Both `timesfm` (>=3.10,<3.12) and `chronos-forecasting` + `torch` are not installed in this interpreter. **Zero live forecasts have been produced** by either client during phase-8. Tests validate the API surface only, via monkeypatched stub models.

### 2.2 Published evidence  --  zero-shot TSFMs underperform on equities

| Source | Finding |
|---|---|
| arXiv 2511.18578 (Nov 2025) | Zero-shot TimesFM(500M): R-squared = -2.80%, directional accuracy <50%, annualised return -1.47% vs CatBoost 46.50% on daily excess returns |
| arXiv 2511.18578 (Nov 2025) | Zero-shot Chronos(large): ~51% directional accuracy; essentially coin-flip |
| tech.preferred.jp blog (2024-2025) | Zero-shot TimesFM Sharpe 0.42 vs AR(1) Sharpe 1.58 on S&P 500 |
| arXiv 2412.09880 (Dec 2024) | Fine-tuned TimesFM on 100k+ financial series reaches Sharpe 1.68; zero-shot does NOT beat AR(1) |

Conclusion: the models ship as generic time-series foundations; domain-specific fine-tuning is required to beat the project's existing MDA + AR(1) baseline.

### 2.3 Ensemble blender has no data to blend

`backend/backtest/ensemble_blend.py` supports equal / correlation / Ledoit-Wolf-shrinkage weighting with nested walk-forward CV. Without live TimesFM/Chronos forecasts (runtime gate, Sec. 2.1), the blender's only non-trivial component is the existing MDA baseline  --  which is already integrated end-to-end by phase-1. Promoting the blender now would be a no-op over status quo.

## 3. What promotion would have required

A PROMOTE decision would have needed ALL of:

1. Runtime gate cleared (Python 3.11 sub-env OR a docker-ized inference service).
2. At least one of TimesFM / Chronos producing measurable IC > 0 on a shadow-log window of >= 60 trading days.
3. Ensemble IR > MDA-baseline IR on out-of-sample nested walk-forward CV.
4. Cost-efficiency: inference latency + compute cost that fits within the 15-slot daily Claude-routine budget.

None of the four is currently met.

## 4. What stays on disk

- `backend/models/timesfm_client.py`  --  lazy-imports `timesfm`; fail-open when the package is absent. Functional the day the runtime gate is cleared.
- `backend/models/chronos_client.py`  --  lazy-imports `chronos` + `torch`; same fail-open discipline.
- `backend/backtest/ensemble_blend.py`  --  pure-Python; works immediately with any scalar signal per (ticker, date).
- `tests/models/`  --  37 tests gate future regressions on the scaffolds.
- `pyfinagent_data.ts_forecast_shadow_log`  --  table DDL is NOT yet created; a future cycle can add the migration when shadow-logging starts.

## 5. What stays disabled

- No autonomous-loop integration with TimesFM / Chronos.
- No backtest-engine wiring  --  `backend/backtest/backtest_engine.py` is unchanged; MDA remains the production signal source.
- No live capital exposure. Phase-4.9 Immutable Core and Risk Guard continues to block any non-approved model.

## 6. Re-evaluation triggers

Re-open the promote/reject decision when ALL of the following hold:

1. A Python 3.11 sub-env or docker-based inference runtime is live in this repo.
2. At least one fine-tuned financial variant is available (e.g. `pfnet/timesfm-1.0-200m-fin`, or a Chronos-2 financial fine-tune, or an internally fine-tuned TimesFM 2.5 on pyfinagent's BQ price warehouse).
3. A shadow-log table captures >= 60 trading days of (forecast, realised) pairs.
4. Ensemble cv_ic over those 60+ days shows IR_blended > IR_mda by a non-trivial margin (>= 0.10).

Meeting condition 4 without 2 is unlikely: zero-shot TSFMs have been tested and rejected in the published record.

## 7. Explicit non-decisions

- **MDA baseline stays.** Phase-1's MDA-driven ensemble remains the live signal source.
- **Scaffold stays.** No code is deleted; the 660 lines of phase-8 code are load-bearing for re-evaluation.
- **Shadow-log DDL deferred.** Creating `ts_forecast_shadow_log` now would be dead weight; defer to when condition 1 above is met.
- **Fine-tuning not attempted here.** Fine-tuning TimesFM 2.5 on pyfinagent's ~5 years of BQ prices is phase-8.5 territory (Karpathy autonomous research loop) or a future phase-8.x.

## 8. References

- `handoff/current/phase-8.1-research-brief.md` (7 sources in full, 17 URLs)  --  TimesFM
- `handoff/current/phase-8.2-research-brief.md` (6 sources in full, 14 URLs)  --  Chronos-Bolt
- `handoff/current/phase-8.3-research-brief.md` (6 sources in full, 16 URLs)  --  ensemble blend
- `handoff/current/phase-8.4-research-brief.md` (closure synthesis)
- `backend/models/timesfm_client.py`, `backend/models/chronos_client.py`, `backend/backtest/ensemble_blend.py`
- `tests/models/test_timesfm_client.py` (11/11), `test_chronos_client.py` (11/11), `test_ensemble_blend.py` (15/15)
- arXiv 2511.18578 (Rahimikia et al., Nov 2025), arXiv 2412.09880 (Dec 2024), Preferred Networks blog, AWS Chronos-Bolt blog

---

**Decision signed off in the harness:** 2026-04-20 -- qa_84_v1 verification pending.
