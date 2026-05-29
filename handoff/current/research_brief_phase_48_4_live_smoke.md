# Research Brief — phase-48.4 live rotation bake-off SMOKE

**Tier:** moderate (internal-heavy). External floor honored (>=5 read-in-full,
recency scan). Methodology (PBO/DSR/CSCV) reused from 48.1/48.2/48.3 — NOT
re-derived. Most effort = INTERNAL run-shape design.

**Goal:** RUN `rotation_runner.run_rotation_bakeoff(...)` for real (a SMALL smoke)
to (a) prove the 48.1-48.3 machinery works end-to-end on REAL backtests
(everything so far is $0 mock-tested), (b) produce the first REAL per-strategy
{dsr,pbo,sharpe} + selector verdict + the persisted rotation_log row.
AUDIT-ONLY (allocation_pct=0, no deploy). $0 LLM (quant-only), real BQ + real
compute.

---

## TL;DR — the recommended smoke

| Knob | Value | Why |
|------|-------|-----|
| **window** | `start_date="2022-01-01"`, `end_date="2024-06-30"` | default 12/3 windows -> **6 walk-forward windows** (real per-window Sharpe variance for DSR); ~367 NAV rows -> T~366 >> the 32-row PBO floor. BQ-confirmed dense (511,960 rows / 502 tickers). |
| **train/test override** | NONE (keep engine default 12/3/5) | The 2.5y window already yields 6 windows at the default; no need to shrink. (Shrinking to 6/2 is the speed lever IF cold-run is too slow — see Risk.) |
| **seeds** | **2** = `tb_baseline` + `qm_trend_tilt` (or `mr_short_horizon`) | 1 seed = no cross-strategy ranking. 2 seeds exercises a REAL N>=2 cross-strategy selector rank + a real incumbent-vs-challenger verdict. (Full 4-seed set DEFERRED.) |
| **num_param_variants** | **2** | The per-strategy PBO matrix needs N>=2 columns. 2 is the floor that produces a REAL (non-degenerate) pbo. (8 is the prod default but ~4x the runtime.) |
| **persist** | `True` | Produces the live_check artifact: one rotation_log.jsonl row at allocation_pct=0. |
| **scale** | 2 seeds x 2 variants = **4 real backtests** | Minimal to exercise both N>=2 (PBO) and a 2-way ranking. |

**Runtime:** cold first backtest ~5-10 min (model train + BQ preload), warm
<30s each thereafter (shared BQ cache, `skip_cache_clear=True`). 4 backtests =>
**~6-12 min wall** -> **MUST be backgrounded** (`run_in_background`).

---

## The exact invocation (copy-paste)

```python
from backend.config.settings import get_settings
from backend.db.bigquery_client import BigQueryClient
from backend.backtest.cache import clear_cache
from backend.autoresearch.rotation_runner import run_rotation_bakeoff

settings = get_settings()
bq = BigQueryClient(settings)            # NOTE: ctor takes settings, NOT no-args

verdict = run_rotation_bakeoff(
    settings,
    bq,
    seeds=[
        {"id": "tb_baseline",   "param_overrides": {}},
        {"id": "qm_trend_tilt", "param_overrides": {"strategy": "quality_momentum", "holding_days": 120}},
    ],
    num_param_variants=2,                # >=2 -> a REAL PBO matrix (not a degenerate 0.0)
    start_date="2022-01-01",
    end_date="2024-06-30",
    persist=True,
    clear_cache_fn=clear_cache,          # canonical warm-cache discipline
    # log_path defaults to backend/backtest/experiments/rotation_log.jsonl
)
print(verdict)
```

**Wiring notes (all source-verified):**
- `BigQueryClient(settings)` — ctor signature is `__init__(self, settings: Settings)`
  (`backend/db/bigquery_client.py:22`). The prompt's example `BigQueryClient()` is
  WRONG; pass `settings`. Precedent: `run_harness.py:1095-1096`.
- `engine_factory` left default → `make_rotation_engine(variant, settings, bq,
  start_date, end_date)` closes over settings+bq correctly (rotation_runner.py:251-254).
  The full-kwarg factory threads `target_vol` etc. (the 48.3 fix).
- `start_date`/`end_date` thread through `make_rotation_engine` → the
  `BacktestEngine` ctor (rotation_runner.py:119-120) and OVERRIDE the seed's own
  `start_date/end_date` (which would be optimizer_best's 2018-01-01..2025-12-31).
- `clear_cache_fn=clear_cache` is called ONCE in the adapter's `finally`
  (strategy_backtest_adapter.py:217-228). Warm cache is shared across all 4
  backtests via the module-level BQ cache.
- `load_promoted_params(bq)` (called by `_resolve_incumbent`) is **read-only** —
  a SELECT that returns the row or falls back to `load_best_params()`
  (autonomous_loop.py:46-29). Safe READ confirmed.
- **Macro preload is INSIDE `run_backtest`** (`backtest_engine.py:299-302`:
  `preload_prices` + `preload_fundamentals` + `preload_macro`). NO separate
  `cache.preload_macro()` call needed; the ~40min-hang rule only bites paths that
  BYPASS `run_backtest`, which this does not.

---

## Internal code inventory (the chain, source-verified)

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/autoresearch/rotation_runner.py` | 225-274 `run_rotation_bakeoff` | Entry point; default full-kwarg engine_factory → adapter → producer; persists verdict @ alloc=0 | LIVE (48.3) |
| `backend/autoresearch/rotation_runner.py` | 76-142 `make_rotation_engine` | Full-ctor-kwarg BacktestEngine; maps target_annual_vol→target_vol; raises on unknown strategy | LIVE (48.3) |
| `backend/autoresearch/strategy_backtest_adapter.py` | 167-256 `make_engine_backtest_fn` | Runs K variants/strategy, builds (T x K) PBO matrix, reads DSR from seed variant; **omits pbo if matrix undersized** | LIVE (48.2) |
| `backend/autoresearch/strategy_candidate_producer.py` | 127-150 `run_strategy_bakeoff` | registry→producer→selector spine; **SKIPS any strategy missing dsr\|pbo** | LIVE (48.1) |
| `backend/autoresearch/strategy_registry.py` | 67-128 `SEED_STRATEGIES` | 4 seeds: tb_baseline, mr_short_horizon, qm_trend_tilt, tb_risk_managed | LIVE (48.1) |
| `backend/autoresearch/strategy_selector.py` | 60-131 `select_best_strategy` | Gate DSR>=0.95 AND PBO<=0.20, DSR-desc/PBO-asc rank, anti-churn hysteresis | LIVE (47.6) |
| `backend/backtest/analytics.py` | 536-568 `generate_report` | DSR from `compute_deflated_sharpe(observed_sr=aggregate_sharpe, variance_of_srs=var(window_sharpes))` | LIVE |
| `backend/backtest/analytics.py` | 184-236 `compute_pbo` | CSCV PBO; **silent 0.0 when N<2 or T<S*2 (=32)** | LIVE |
| `backend/backtest/walk_forward.py` | 44-89 `generate_windows` | Expanding window; first test ends ~15mo after start, +3mo each | LIVE |
| `backend/backtest/backtest_engine.py` | 266-379 `run_backtest` | macro preload INSIDE (299-302); `full_reset()` per run; `skip_cache_clear` honored | LIVE |
| `backend/backtest/experiments/optimizer_best.json` | — | base params: strategy=triple_barrier, dsr=0.9526, sharpe=1.1705, holding_days=90 | LIVE |

---

## Net-new finding 1 — VIABLE WINDOW (the critical answer)

The window-count math (`walk_forward.py:44-89`, computed live):
- first `train_end = start + train_window_months − 1d`; each window:
  `test_start = train_end + embargo+1`, `test_end = test_start + test_window_months − 1`;
  loop while `test_start <= end_date`, then `train_end` advances to `test_end`.

**Window counts (computed, not estimated):**

| start..end | train/test | windows | verdict |
|------------|-----------|---------|---------|
| 2024-01-01..2024-06-30 | 12/3 | **0** | DEGENERATE — the trap in the prompt. <15mo → zero windows → generate_report falls to `sr_variance=0.5`, DSR degenerates. |
| 2023-01-01..2024-06-30 | 12/3 | 2 | minimal viable (DSR variance real but only 2 points; T~121). |
| **2022-01-01..2024-06-30** | **12/3** | **6** | **RECOMMENDED** — 6 windows → robust per-window Sharpe variance; T~367. |
| 2022-01-01..2024-06-30 | 6/2 | 12 | speed alternative (smaller train → faster per-window; 12 windows). |

**Why 6 windows (not the bare 2):** `generate_report` (analytics.py:549) computes
`sr_variance = var(window_sharpes)` only `if len(window_sharpes) > 1` ELSE falls
back to `0.5`. With only 2 windows the variance is a 2-point estimate (fragile,
can be ~0 if both windows are similar → DSR distortion). 6 windows gives a real
distribution. The DSR is the whole point of the rotation gate, so the smoke must
NOT feed it a degenerate variance.

**BQ data-range confirmation** (`financial_reports.historical_prices`, live query
2026-05-29):
- Full range **2017-01-03 .. 2026-05-28**, 1,831,731 rows / 507 tickers.
- 2022-01..2024-06 window: **511,960 rows / 502 tickers** (dense; no gap risk).
- 756-day `global_start` lookback (engine line 297 pulls `start − 756d` = ~2020):
  **406,632 rows in 2020-01..2021-12** → feature-building is NOT starved.

So `2022-01-01..2024-06-30` is fully covered with margin on both the window and
its training-lookback.

---

## Net-new finding 2 — SMOKE SCALE + RUNTIME

- **num_param_variants = 2 (floor for a REAL PBO).** The adapter
  (`strategy_backtest_adapter.py:132-152`) assembles a (T x N) matrix from the K
  variants' daily returns; `_assemble_pbo_matrix` returns `None` (→ pbo OMITTED →
  producer SKIPS) when `len(cols) < 2` OR `min_len < 32`. With
  num_param_variants=2 we get N=2 columns; if both succeed → a non-degenerate
  `compute_pbo`. **N=1 would yield a degenerate 0.0 that FALSE-PASSES the
  pbo<=0.20 gate** — exactly the failure 48.2 was built to prevent, so do not run
  the smoke at N=1.
- **T >= 32 confirmed:** the PBO matrix rows come from each variant's FULL
  `nav_history` daily returns (across ALL windows, not per-window). For
  2022..2024-06 the 6 test spans sum to ~367 business days → T~366 (10x the
  32-row floor). Even the 2-window minimal case gives T~121. **No T-starvation
  risk.**
- **seeds = 2** (tb_baseline + qm_trend_tilt). 1 seed cannot exercise a
  cross-strategy ranking; 2 produces a real N>=2 selector rank + an
  incumbent-vs-challenger verdict. The 3rd/4th seeds add runtime without changing
  "does the machinery work" — DEFER them.
- **Runtime:** cold first backtest ~5-10 min (GradientBoosting train across 6
  windows + first BQ preload); warm <30s each (48.2/48.3 briefs; the warm-cache
  `skip_cache_clear=True` loop is the quant_optimizer precedent that drops
  per-run time from ~5-10min to <30s). 4 backtests (2 seeds x 2 variants) =>
  **~6-12 min wall**. **BACKGROUND it** (`run_in_background: true`); poll for
  completion.
- macro preload INSIDE `run_backtest` → no separate preload, no ~40min hang.

---

## Net-new finding 3 — SANE-OUTPUT VALIDATION CRITERIA (for Q/A)

The smoke is about MECHANICS, not strategy quality (smoke-test doctrine: "prove
the pipeline runs … not … prove the model is good" — MLOps Community). The PASS
bar for "the machinery works live":

**Per-strategy metrics (for at least the seeds that were NOT skipped):**
- `dsr` is finite and in **[0, 1]** (it is a probability; compute_deflated_sharpe
  clamps via norm.cdf).
- `pbo` is finite and in **[0, 1]** AND **NOT a degenerate 0.0 produced by an
  undersized matrix**. With N=2 / T~366 a real PBO is computable; if the adapter
  log shows "PBO matrix undersized/degenerate … emitting NO pbo" for BOTH seeds,
  the smoke is INVALID (re-check N>=2 and that both variants produced nav_history).
- `sharpe` is finite (may be negative — that is fine; it is a real number, not a
  crash).
- `n_windows >= 2` on the seed variant (proves the window scheduler produced a
  real walk-forward, not the 0-window degenerate).

**Verdict shape (`select_best_strategy` output):**
- A dict with `selected_id`, `switched` (bool), `reason` (one of:
  `first_selection` / `incumbent_is_top` / `dsr_improvement` /
  `below_min_improvement` / `no_candidate_passed_gate`), `ranked` (list),
  `num_trials` (int), `delta_dsr`.
- **`no_candidate_passed_gate` is a VALID smoke outcome.** The gate is strict
  (DSR>=0.95 AND PBO<=0.20). On a 2.5y window with only 2 param variants it is
  LIKELY no seed clears DSR>=0.95 → `reason="no_candidate_passed_gate"`,
  `selected_id` = the incumbent (or None). This still PROVES the machinery works
  end-to-end — do NOT treat "no passer" as a failure. The smoke validates the
  PLUMBING, not that a rotation candidate won.
- If a seed DOES pass, the selector treats the best seed as a CHALLENGER to the
  incumbent's recorded DSR (it will not return `incumbent_is_top` because the
  incumbent is keyed by strategy NAME `triple_barrier`, not seed id `tb_baseline`
  — documented in `_resolve_incumbent`, rotation_runner.py:155-164).

**Persistence (the live_check artifact):**
- Exactly ONE new line appended to
  `backend/backtest/experiments/rotation_log.jsonl` with
  `allocation_pct == 0.0`, `status == "bakeoff_verdict"`, and `selected_id`
  matching the returned verdict, plus `num_param_variants:2` and
  `window:"2022-01-01..2024-06-30"` in the `extra` block.
- **NO deploy side effects:** no `promoted_strategies` MERGE, no
  `settings.paper_*` mutation, no `strategy_decisions` write (the deployment
  bridge is DEFERRED — confirmed by the 48.1/48.2/48.3 deferreds + the
  allocation_pct=0 design).

**Process PASS bar:** the run completes (does NOT hang past ~15 min, does NOT
crash), returns a verdict dict, and persists exactly one alloc=0 row whose
selected_id matches the verdict. Degenerate red flags: 0 windows, pbo omitted for
all seeds, a RuntimeError "all variants failed", or a non-finite dsr/sharpe.

---

## Recency scan (last 2 years, 2024-2026)

Searched 2024-2026 literature on walk-forward minimum-window count, MinBTL, and
smoke-test practice. **Findings:**
- **arXiv:2512.12924 (Dec 2025)** — a 2025 walk-forward validation framework runs
  **34 independent OOS test periods** (W=252d train, H=63d test, step 63d) and
  explicitly states that even 34 folds achieves "power of only 12%" vs the "~540
  folds" needed for 80% power. This SUPERSEDES nothing in our methodology but
  CONFIRMS the direction: more windows = more statistical validity, and a tiny
  window count is underpowered. For a SMOKE (mechanics, not inference) our 6
  windows is appropriate; for a real bake-off, more is better (noted for the
  deferred full run).
- **No new finding supersedes** Bailey-Borwein-LdP (2014/2015) on PBO/CSCV or
  Bailey-LdP (2014) on DSR/MinBTL — they remain the canonical basis the 48.x
  stack already encodes. The 2025 paper cites them as the standard.
- Smoke-test doctrine (MLOps Community / CircleCI, both 2024-2025-current) is
  stable: tiny representative slice, end-to-end, verify mechanics not quality,
  go/no-go gate, fast.

Net: no methodology change; the recency scan reinforces "6 windows for a smoke,
more for the real run" and the smoke-as-mechanics-gate framing.

---

## External sources

### Read in full (>=5 — counts toward the gate)
| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://arxiv.org/html/2512.12924v1 | 2026-05-29 | paper (Dec 2025) | WebFetch (HTML) | "34 independent test periods"; "required sample size ~540 folds" for 80% power, "our sample of 34 achieves power of only 12%"; W=252/H=63/step=63. More OOS windows = more validity. |
| https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf | 2026-05-29 | paper (Bailey/Borwein/LdP) | pdfplumber (binary chain) | CSCV: "the strategy configuration that delivers maximum performance in sample (IS) must systematically underperform … OOS" = overfitting; PBO/MinBTL canonical. Confirms columns=configurations (our N>=2 param-variant design). |
| https://mlops.community/smoke-testing-for-ml-pipelines/ | 2026-05-29 | blog (MLOps) | WebFetch | "The goal isn't to prove the model is good — it's to prove the pipeline still runs"; "tiny, synthetic datasets"; "guarantee that your pipeline is alive and dependable." Defines smoke = mechanics not quality. |
| https://circleci.com/blog/smoke-tests-in-cicd-pipelines/ | 2026-05-29 | blog (CircleCI) | WebFetch | "basic tests that verify the most critical functions"; "first-pass check to catch major issues before running more comprehensive test suites"; fast feedback / go-no-go. |
| https://blog.quantinsti.com/walk-forward-optimization-introduction/ | 2026-05-29 | blog (QuantInsti) | WebFetch | Walk-forward gives "greater statistical validity" via multiple OOS periods; "too short a training window … unstable parameters"; illustrative 5y train / 1y test. (Thin on hard numbers but read in full.) |

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253 | paper (PBO SSRN) | Same content as the davidhbailey PDF read in full. |
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 | paper (DSR SSRN) | DSR canonical — already encoded in analytics.compute_deflated_sharpe; reused not re-derived. |
| https://stefan-jansen.github.io/machine-learning-for-trading/08_ml4t_workflow/01_multiple_testing/ | book/site | MinBTL summary; corroborates Bailey, snippet sufficient. |
| https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio | encyclopedia | Tertiary; DSR already settled. |
| https://www.buildalpha.com/walk-forward-optimization/ | blog | Window-count tradeoff; snippet sufficient. |
| https://www.harness.io/.../integrating-smoke-testing-into-your-ci-cd-pipeline | blog | Smoke-as-gate; CircleCI+MLOps cover it. |
| https://sealos.io/blog/smoke-testing-for-ml-pipelines-... | blog | Duplicate of MLOps smoke doctrine. |
| https://citeseerx.ist.psu.edu/document?...PBO | paper mirror | Mirror of the PBO paper already read. |

### Search-query variants run (three-variant discipline)
- Current/recent frontier: "walk-forward backtest minimum number of out-of-sample
  windows statistical validity"; "minimum backtest length deflated sharpe ratio
  Bailey Lopez de Prado overfitting 2024 2025" (year-scoped recency).
- Canonical year-less: "probability of backtest overfitting CSCV minimum number
  of trials configurations"; "smoke test integration end-to-end pipeline minimal
  subset data validation best practice".

---

## Application to pyfinagent (mapping → file:line)

1. **External smoke doctrine → the PASS bar.** "Prove the pipeline runs, not that
   the model is good" (MLOps) ⇒ `no_candidate_passed_gate` is a VALID smoke
   outcome; the gate is `strategy_selector.py:75` / `gate.py`. Q/A must judge
   PLUMBING (verdict shape + alloc=0 row + n_windows>=2 + non-degenerate pbo), not
   whether a seed won.
2. **CSCV columns = configurations (Bailey) → num_param_variants>=2.** The N>=2
   floor in `_assemble_pbo_matrix` (strategy_backtest_adapter.py:147) is the
   direct implementation; N=1 → silent 0.0 false-pass (the bug 48.2 prevents). The
   smoke MUST set num_param_variants>=2.
3. **More OOS windows = more validity (arXiv 2025 + QuantInsti) → 6-window
   window.** `walk_forward.generate_windows` + `generate_report`'s
   `var(window_sharpes)` fallback (analytics.py:549) ⇒ pick 2022-01-01..2024-06-30
   (6 windows) so DSR variance is real, not the `0.5` degenerate.
4. **Smoke = fast / minimal / backgrounded (CircleCI).** 4 backtests, background
   the run, poll. Full 4-seed x 8-variant (~32 backtests, tens of min) stays
   DEFERRED.

---

## Risks (bounded)

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Cold backtest slow (>10 min) | medium | Background it; if too slow, drop to train/test 6/2 (12 windows, smaller train per window, faster) OR 1 seed x 2 variants first to time a single strategy. |
| Hang (>15 min, no progress) | low | macro preload is INSIDE run_backtest (no 40-min hang). If it hangs, kill and re-run a SINGLE seed to isolate. `_quiet_progress` suppresses spam — watch logs for "Walk-forward: N windows". |
| Degenerate PBO (0.0 for all) | low (N=2,T~366) | If adapter logs "matrix undersized", a variant failed → check the per-variant warning; ensure both variants produced nav_history. Q/A flags as INVALID. |
| BQ cost balloon | low | Bounded window (2.5y, 502 tickers) + warm cache (2 preload queries total, reused across all 4 backtests). Well under any practical ceiling; no unbounded historical scan. |
| Quality-momentum seed errors | low | If qm_trend_tilt raises, the producer SKIPS it (warning) and the bake-off still completes with tb_baseline alone — smoke still proves the spine. (Swap to mr_short_horizon if qm is flaky.) |
| Incumbent name != seed id quirk | n/a (documented) | Selector treats best seed as challenger; will not emit incumbent_is_top. Expected, not a bug (rotation_runner.py:160-163). |

---

## Completable-one-cycle slice

**DO this cycle:** run the 2-seed x 2-variant smoke on 2022-01-01..2024-06-30
(backgrounded), capture the REAL verdict + the per-strategy {dsr,pbo,sharpe} from
logs, and the persisted alloc=0 rotation_log row — that triple IS the live_check.

**DEFER (explicitly):**
- The FULL 4-seed x 8-variant bake-off (~32 backtests, tens of minutes).
- The DEPLOYMENT bridge (params → settings.paper_* + promoted_strategies MERGE) —
  the keystone; its own cycle.
- The weekly rotation cron.
- Re-enabling the reverted trailing/vol-target engine readers (tb_risk_managed's
  trailing half is inert).
- Effective-N (ONC) DSR clustering; CPCV multi-path PBO.

---

## Research Gate Checklist

Hard blockers — `gate_passed` true only if all checked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch/pdfplumber (5)
- [x] 10+ unique URLs total (13)
- [x] Recency scan (last 2 years) performed + reported (arXiv:2512.12924 Dec 2025)
- [x] Full papers/pages read (not abstracts) — incl. Bailey PBO via pdfplumber chain
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered the full chain (runner→adapter→producer→registry→selector→engine→analytics→walk_forward)
- [x] Contradictions/consensus noted (DSR/PBO canonical unchanged; 2025 paper confirms more-windows direction)
- [x] Claims cited per-claim with file:line + URLs

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 8,
  "urls_collected": 13,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "gate_passed": true
}
```
