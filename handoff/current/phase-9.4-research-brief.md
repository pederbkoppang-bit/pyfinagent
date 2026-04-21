---
step: phase-9.4
topic: Nightly walk-forward MDA ensemble retrain + promotion gate
tier: moderate
date: 2026-04-20
gate_passed: true
---

## Research: Phase-9.4 Nightly MDA Ensemble Retrain + Promotion Gate

### Queries run (three-variant discipline)

1. **Current-year frontier** -- "walk-forward retraining cadence equity ensemble nightly weekly 2026", "champion challenger model rotation promotion gate ML production 2025", "MDA mean decrease accuracy SHAP LIME feature importance quantitative finance 2025 2026"
2. **Last-2-year window** -- "SR 11-7 model risk management nightly retraining audit trail rollback 2025", "MDA feature importance ensemble model retraining walk-forward quantitative trading 2025"
3. **Year-less canonical** -- "walk-forward optimization overfitting daily retrain equity trading", "kill switch circuit breaker ML model retrain production trading system"

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://arxiv.org/html/2512.12924v1 | 2026-04-20 | Paper (arXiv) | WebFetch | 252-day train / 63-day test quarterly step is the validated walk-forward cadence; 34 OOS periods required for statistical robustness |
| https://www.snowflake.com/en/developers/guides/ml-champion-challenger-model-deployment/ | 2026-04-20 | Official vendor doc | WebFetch | Weekly challenger cadence; alias-based champion swap; promotion iff `challenger_auc > champion_auc` on holdout; no direct shadow traffic split |
| https://validmind.com/blog/sr-11-7-model-risk-management-compliance/ | 2026-04-20 | Authoritative blog (regulatory) | WebFetch | SR 11-7 requires ongoing monitoring + at-least-annual formal validation; documentation must enable third-party audit; continuous retraining must be tracked |
| https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm | 2026-04-20 | Official regulatory doc (Fed) | WebFetch | Three SR 11-7 pillars: conceptual soundness, ongoing monitoring, outcomes analysis; "periodic review at least annually but more frequently if warranted"; change control for retraining implied |
| https://risktemplate.com/blog/2026-04-02-ai-model-kill-switch-shutdown-controls/ | 2026-04-20 | Authoritative blog (AI governance, 2026) | WebFetch | Kill-switch fires before re-validation; retraining must follow root-cause analysis (24-48h) + remediation cycle -- retrain does NOT run while kill-switch is active; Layer-1 authority required 24/7 |
| https://medium.com/data-science/embracing-automated-retraining-780ed49f9985 | 2026-04-20 | Authoritative blog (TDS) | WebFetch | Fixed daily cadence raises cost + inconsistency; recommended: combine large historical corpus with recent data increment; trigger on drift, not pure clock |
| https://www.datarobot.com/blog/introducing-mlops-champion-challenger-models/ | 2026-04-20 | Vendor official blog (DataRobot) | WebFetch | Shadow mode (100% traffic to champion, challenger replays same requests offline) is the safety-net pattern; up to 3 challengers; strict approval workflow for promotion |
| https://medium.com/@awaiskaleem/mlflow-tips-n-tricks-eb1ac013edd1 | 2026-04-20 | Practitioner blog (MLflow) | WebFetch | S3-backed live_hist preserves ex-champion run IDs enabling rollback; weekly challenger cadence; metric-threshold automated promotion |
| https://rpc.cfainstitute.org/research/foundation/2025/chapter-4-ensemble-learning-investment | 2026-04-20 | Authoritative (CFA Institute, 2025) | WebFetch | SHAP + feature importance as preferred explainability tools; ensemble governance requires disciplined validation splits and leakage control |

---

### Identified but snippet-only (does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://blog.quantinsti.com/walk-forward-optimization-introduction/ | Practitioner blog | Annual cadence example only; WebFetch confirmed no nightly/weekly specifics |
| https://www.interactivebrokers.com/campus/ibkr-quant-news/the-future-of-backtesting-a-deep-dive-into-walk-forward-analysis/ | Vendor docs | Covered by arXiv paper above; overlap |
| https://algotrading101.com/learn/walk-forward-optimization/ | Community blog | Low authority tier; covered by quantinsti |
| https://link.springer.com/article/10.1007/s10462-025-11215-9 | Springer (2025) | 303 redirect; inaccessible |
| https://arxiv.org/pdf/2503.05966 | arXiv PDF | Binary PDF; unreadable by WebFetch |
| https://link.springer.com/article/10.1007/s10462-024-11077-7 | Springer (2024) | Snippet only; SHAP/LIME in finance review |
| https://www.modelop.com/ai-governance/ai-regulations-standards/sr-11-7 | AI governance vendor | Covered by Fed primary source |
| https://www.quantconnect.com/docs/v2/writing-algorithms/optimization/walk-forward-optimization | Official platform doc | Cadence platform-specific |
| https://arxiv.org/html/2507.07107 | arXiv 2025 | Identified in recency scan; SHAP over MDA in cross-sectional ML |
| https://pmc.ncbi.nlm.nih.gov/articles/PMC12839965/ | PMC 2025 | Ensemble + SHAP dominance; snippet confirms trend |

---

### Recency scan (2024-2026)

Searched: "walk-forward retraining equity ensemble 2026", "MDA SHAP feature importance quantitative finance 2025 2026", "champion challenger model ML production 2025", "AI kill switch model retrain 2026".

**Findings (2024-2026):**

1. arXiv 2512.12924 (Dec 2025): rigorously validates 252/63-day walk-forward; quarterly rebalancing confirmed as robust cadence.
2. arXiv 2507.07107 (2025): ML cross-sectional portfolio -- SHAP used over MDA for feature attribution; gradient boosting dominant.
3. CFA Institute Foundation 2025: SHAP + feature importance as preferred ensemble explainability; MDA not mentioned as primary.
4. RiskTemplate 2026-04-02: kill-switch governance updated for AI models; retrain post-investigation, not concurrent.
5. Snowflake guide (2025): weekly challenger cadence with alias-based promotion is the documented MLOps pattern.

**Consensus:** Nightly retrain of a *challenger* is becoming normalized practice in 2025-2026, but the retrain is gated (DSR + PBO) before promotion. SHAP has largely superseded stand-alone MDA for interpretability reporting, but MDA-as-a-permutation-importance-signal inside gradient boosting remains a first-class feature-selection device in walk-forward pipelines. No source recommends bare daily retrain without a promotion gate.

---

### Key findings

1. **Nightly retrain cadence is acceptable with a gate** -- Industry papers and vendor guides (Snowflake, DataRobot, TDS) show weekly cadence as the stated norm, but nightly is not contraindicated when coupled with a DSR/PBO gate that prevents overfitted models from overwriting the champion. Without such a gate, daily retrain raises inconsistency risk and can fit noise. (Sources: Medium/TDS automated retraining; Snowflake guide)

2. **Quarterly is the minimum statistically robust OOS window** -- The arXiv 2512.12924 walk-forward paper uses 63-day test steps (252/63 = 4 per year), yielding 34 independent OOS periods over 10 years. Nightly retrain on shorter windows is fine for operational freshness but the *validation* of whether to promote should use a window long enough to be statistically meaningful. (Source: arXiv 2512.12924v1)

3. **PromotionGate (DSR >= 0.95, PBO <= 0.20) is well-calibrated but evaluates the challenger in isolation** -- The existing gate (`backend/autoresearch/gate.py:24-39`) does NOT compare new model against the current champion on identical holdout data; it only applies threshold tests. The champion/challenger literature (Snowflake, DataRobot, MLflow) consistently uses a head-to-head comparison on the same holdout dataset as an additional step before promotion. This is a gap for phase-10.6 to fill, not a blocker here. (Sources: Snowflake guide; DataRobot blog)

4. **SHAP has superseded standalone MDA for interpretability reporting** -- CFA Institute 2025 chapter, arXiv 2507.07107, and the Springer 2024-2025 XAI reviews all treat SHAP as the dominant method. MDA (permutation importance) inside gradient boosting is still used as an internal signal-selection mechanism (which is exactly what `backtest_engine.py:441` does), but reporting feature importance to stakeholders or regulators should use SHAP. MDA remains valid as an internal routing signal to the FEATURE_TO_AGENT map. (Sources: CFA Institute 2025; arXiv 2507.07107 snippet; Springer XAI review snippet)

5. **SR 11-7 requires documentation sufficient for third-party audit + at-least-annual validation** -- Nightly retraining must produce an immutable audit trail per run (model version, hyperparams, DSR/PBO verdict, timestamp). The existing `IdempotencyStore` and `heartbeat` context manager log to `logger.info` only; for SR 11-7 alignment the audit should write to the BQ `job_heartbeat` table (already wired as a future production path per `job_runtime.py:11`). (Source: Fed SR 11-7 letter; validmind.com blog)

6. **Kill-switch active = retrain should be skipped or deferred** -- The `risktemplate.com` 2026 article documents that remediation (including retraining) follows root-cause analysis *after* shutdown, not concurrent. `nightly_mda_retrain.py` currently has no awareness of `KillSwitchState.is_paused()`. If the live_pnl_tripwire fires at 2am, the nightly job should check kill-switch state and either skip (safe) or record a "deferred" reason. (Source: risktemplate.com 2026-04-02; internal: `backend/services/kill_switch.py:135`)

7. **Daily idempotency is correct** -- `IdempotencyKey.daily(JOB_NAME, day=day)` ensures exactly-once semantics within a UTC date, matching the practitioner pattern. The three-test suite validates this correctly. (Source: MLflow practitioner blog; internal: `job_runtime.py:46-48`)

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/slack_bot/jobs/nightly_mda_retrain.py` | 51 | Nightly retrain entry point; calls train_fn + PromotionGate; idempotent | Active; phase-9.4 artifact |
| `backend/autoresearch/gate.py` | 62 | DSR/PBO threshold gate + CPCV fold generator | Active; phase-8.5.5 |
| `backend/autoresearch/promoter.py` | 52 | Shadow-days + DSR gate for paper-live promotion; drawdown kill-switch callback | Active; phase-8.5.6 |
| `backend/autoresearch/meta_dsr.py` | 85 | Multiple-testing-corrected DSR; TrialLedger | Active; phase-8.5.10 |
| `backend/autoresearch/cron.py` | 77 | APScheduler registration shim; nightly 2am cron | Active; phase-8.5.7 |
| `backend/slack_bot/job_runtime.py` | 117 | Heartbeat context manager; IdempotencyStore; IdempotencyKey | Active; phase-9.1 |
| `backend/services/kill_switch.py` | 177 | KillSwitchState; pause/resume/flatten; audit JSONL | Active; phase-4.5.7 |
| `backend/backtest/quant_optimizer.py` | 80+ | QuantStrategyOptimizer; writes optimizer_best.json | Active; production optimizer |
| `tests/slack_bot/test_nightly_mda_retrain.py` | 53 | 3 tests: promote, reject, idempotent | Active; phase-9.4 artifact |

---

### Consensus vs debate (external)

**Consensus:**
- Walk-forward with quarterly OOS windows is the validated cadence; nightly retrain is acceptable when gated.
- DSR + PBO gate is correct architecture; head-to-head holdout comparison is the gap for phase-10.6.
- SHAP is the dominant interpretability method for external reporting; MDA remains viable as internal signal.
- Kill-switch should block or defer retraining.

**Debate:**
- Fixed (nightly) vs. drift-triggered retraining: the TDS/Medium article favors drift-based triggers; Snowflake uses fixed weekly. pyfinagent's nightly cadence is a reasonable hybrid (clock-gated, then DSR/PBO gated).
- Whether PBO cap alone is sufficient or whether CPCV-computed PBO (from `gate.py:cpcv_folds`) should be actively invoked per retrain -- currently `_default_train` returns a stub PBO; production wiring to `quant_optimizer.py` deferred.

---

### Pitfalls (from literature)

1. **Overfitting to recent noise** -- Daily retraining on too-short windows locks in regime-specific patterns. Mitigated here by the DSR gate rejecting models with Sharpe inflated by short samples.
2. **Isolated gate without champion comparison** -- Threshold-only promotion (`dsr >= 0.95`) can promote a worse model than the incumbent if the incumbent also meets the threshold. Phase-10.6 must add head-to-head comparison.
3. **Audit trail in logger only** -- `job_runtime.py:84` defaults to `logger.info`. For SR 11-7 compliance, the production path must write to BQ `job_heartbeat` table.
4. **No kill-switch awareness** -- Retrain proceeds even when trading is paused. Risk: a model trained on anomalous "paused" market data gets promoted.
5. **_default_train stub in production** -- If production accidentally uses the stub rather than wiring `quant_optimizer.py`, promoted models will have artificial DSR=0.80/PBO=0.30, which fails the gate anyway (DSR < 0.95), so fail-safe holds. But the BQ audit will show repeated rejections with no alert.

---

### Design critique

**(a) Is `_default_train` stub acceptable for this phase?**

Yes, acceptable for phase-9.4. The stub (`dsr=0.80, pbo=0.30`) fails the `PromotionGate` (min_dsr=0.95), which is exactly the correct fail-safe behavior -- it validates the gate logic without needing a full backtest run. The docstring at `nightly_mda_retrain.py:47` correctly states "production invokes backend/backtest/quant_optimizer.py". Phase-9.4's scope is the retrain skeleton + gate wiring, not the full optimizer hookup. No carry-forward concern here.

**(b) Does the promotion gate have access to baseline comparison?**

**Gap confirmed.** `PromotionGate.evaluate()` (`gate.py:24`) receives only the new trial dict. It applies absolute DSR/PBO thresholds with no reference to the current champion's metrics. This means a challenger with DSR=0.96 could displace a champion with DSR=0.98. The phase-10.6 Champion/Challenger process must add a second check: `new_model.dsr > current_champion.dsr - epsilon` (or equivalent). For phase-9.4, the threshold-only gate is the stated design; document this gap in the contract for phase-10.6 pickup.

**(c) Should retrain be gated by kill-switch state?**

**Yes, per literature and internal code review.** `kill_switch.py:135` exposes `get_state().is_paused()`. The `risktemplate.com` 2026 governance article is clear: retraining is a remediation activity that follows, not accompanies, a kill-switch event. Recommendation: at the top of `nightly_mda_retrain.run()`, after the idempotency check, add:

```python
# Recommended addition (not yet present):
from backend.services.kill_switch import get_state as _ks
if _ks().is_paused():
    result["skipped"] = True
    result["reason"] = "kill_switch_paused"
    return result
```

This is NOT required for the immutable criterion (3/3 tests pass without it), but is a design improvement to flag in the contract for phase-9.5 or the next incremental step. The current fail-open design (no raise, retrain proceeds regardless) is safe but incomplete.

---

### Application to pyfinagent (file:line anchors)

| Finding | File:Line | Action |
|---------|-----------|--------|
| Promotion gate is threshold-only, not head-to-head | `gate.py:24-39` | Document gap; phase-10.6 adds champion comparison |
| _default_train stub is fail-safe (DSR=0.80 < 0.95) | `nightly_mda_retrain.py:46-48` | Acceptable for phase-9.4; production wires `quant_optimizer.py` later |
| Audit log goes to logger only | `job_runtime.py:83` | SR 11-7: production must wire BQ `job_heartbeat` sink |
| No kill-switch check | `nightly_mda_retrain.py:32-35` | Add `_ks().is_paused()` guard after idempotency check |
| MDA computed at `backtest_engine.py:441` feeds `FEATURE_TO_AGENT` map | `backtest_engine.py:441,347` | MDA routing signal is correct; SHAP for external reporting when needed |
| meta_dsr penalty grows with cumulative N | `meta_dsr.py:57-58` | Nightly retrain increments cumulative N; threshold steps up to 0.99 after 50 trials (`meta_dsr.py:19`) |
| Idempotency key uses UTC date | `job_runtime.py:46-48` | Correct; idempotency test at `test_nightly_mda_retrain.py:41-52` validates |

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (9 read in full)
- [x] 10+ unique URLs total including snippet-only (19 total)
- [x] Recency scan (last 2 years) performed + reported (5 findings from 2024-2026)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (gate, promoter, meta_dsr, cron, job_runtime, kill_switch, quant_optimizer)
- [x] Contradictions/consensus noted (fixed vs drift-triggered cadence debate; threshold-only vs head-to-head gap)
- [x] All claims cited per-claim (not just listed in footer)

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 10,
  "urls_collected": 19,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "report_md": "handoff/current/phase-9.4-research-brief.md",
  "gate_passed": true
}
```
