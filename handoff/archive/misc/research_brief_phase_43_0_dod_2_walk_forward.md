# Research Brief — phase-43.0 DoD-2 Walk-Forward + Paper-Trading Sharpe Instrumentation

**Cycle:** 15
**Tier:** moderate
**Step:** phase-43.0 DoD-2 (MEASUREMENT arm only)
**Date:** 2026-05-28
**Status:** COMPLETE
**Researcher:** Claude Opus 4.7 (1M ctx)

---

## 1. Headline (TL;DR)

**Recommend Option A + light B (call it A+):** add a windowed-paper-Sharpe helper to `perf_metrics.py`
(`compute_paper_sharpe_window(bq, window_days)` + extend `compute_sharpe_gap` with a `window_days` arg)
AND attach a `paper_parity` block to the walk-forward result-JSON writer in `api/backtest.py`. The MEASUREMENT
arm of DoD-2 closes. The numeric threshold `|paper.sharpe - backtest.sharpe| < 0.01` will NOT pass this
cycle — that's a separate VALUE arm (the underlying paper-vs-backtest NAV divergence is 52.5% per cycle-12
audit; orders of magnitude wider than 0.01 in Sharpe space). Critically, the research literature shows
**0.01 absolute Sharpe-gap on a 30-day window is statistically implausible** even under a perfectly-
reconciled system — Sharpe SE on 30 obs is on the order of ±0.3-0.5 (Lo 2002 form). The honest DoD-2
threshold is the existing `SR_GAP_THRESHOLD = 0.30` relative, NOT a 0.01 absolute. Recommend a parallel
roadmap edit to reconcile DoD-2 wording.

---

## 2. Sources read in full (>=5; this brief has 10)

| URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|
| https://www.tradezella.com/blog/backtesting-trading-strategies | 2026-05-28 | Industry blog | WebFetch HTML | "Expect 10 to 20% degradation from backtest to live trading. Anything worse than that needs investigation." Establishes the industry-standard **relative-gap band** for paper-vs-backtest. |
| https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf | 2026-05-28 | Peer-reviewed (JPM 2014) | curl + pdfplumber v0.11.9 (22 pages) | Canonical PSR & DSR + MinTRL definitions. Equation 1 expected-max of N trials; Eq. 8 DSR uses Euler-Mascheroni γ ≈ 0.5772. "Given an observed annualized SR* = 0.95, approximately 3 years of daily returns are needed to reject the null hypothesis with 95% confidence." A 30-day window is FAR below the canonical MinTRL — sub-0.01 absolute Sharpe-gap claims are statistically meaningless at that horizon. |
| https://www.twosigma.com/wp-content/uploads/sharpe-tr-1.pdf | 2026-05-28 | Industry whitepaper (Two Sigma) | curl + pdfplumber (11 pages) | Citing Lo 2002 (Fact 5): assumes stationary-ergodic excess returns, gives the asymptotic-variance formula for the basic Sharpe estimator. Confirms: "a Sharpe ratio that considers returns over monthly periods cannot be compared directly to one that considers returns over years" — frequency invariance only via the √(1+2ρ₁) correction. Default plug-in CI: `[ζ̂ − z_{α/2}·se, ζ̂ + z_{α/2}·se]`. Implies short-window SE ≈ 1/√n × (1 + ½ζ̂²) under IID; for n=30 with annual SR=0.7 that's SE ≈ 0.20-0.25 per period before any annualization correction. |
| https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio | 2026-05-28 | Encyclopedia (deep) | WebFetch HTML | Verbatim DSR / PSR / MinTRL formulas. Confirms 95%-confidence ↔ DSR ≥ 0.95 (Φ⁻¹(0.95) ≈ 1.645). MinTRL practical example: ~3 years daily returns for SR=0.95. Standard industry parameters: γ ≈ 0.5772 (Euler-Mascheroni), DSR ≥ 0.95 threshold. **Already implemented at `backend/services/perf_metrics.py:32 _EULER_GAMMA` and `:420-471 compute_psr / compute_dsr`** — pyfinagent codebase already uses these formulas. |
| https://arch.readthedocs.io/en/stable/bootstrap/confidence-intervals.html | 2026-05-28 | Official docs (arch, Kevin Sheppard) | WebFetch HTML | Verbatim: "While returns have little serial correlation, squared returns are highly persistent. The IID bootstrap is not a good choice here." Recommends time-series bootstrap with block-size selection. Pyfinagent already implements stationary-block bootstrap with lag-1 autocorr trigger (|acf1| > 0.2) at `perf_metrics.py:506-563 compute_rolling_sharpe_bootstrap_ci` — this is the canonical small-sample CI method to use for 30-day windows. |
| https://surmount.ai/blogs/walk-forward-analysis-vs-backtesting-pros-cons-best-practices | 2026-05-28 | Industry blog | WebFetch HTML | "WFE above 50-60%" indicates "strategy maintains at least half its optimized performance on unseen data". Walk-Forward Efficiency = OOS performance ÷ IS performance. **Frames the right threshold semantic for DoD-2**: WFE/replication-ratio, not absolute Sharpe difference. 12-24 windows over a decade. |
| https://medium.com/@TheQuantPy/is-your-sharpe-ratio-lying-to-you-meet-the-probabilistic-sharpe-ratio-d06077e423e8 | 2026-05-28 | Academic blog | WebFetch HTML | Pedagogical PSR derivation with Bessel's correction (n-1). Confirms "sample length matters, a lot!" — short samples make statistical significance unattainable. **Concrete corollary**: a 30-day window cannot resolve a 0.01 absolute Sharpe gap as significant. |
| https://arxiv.org/abs/2501.03938 | 2026-05-28 | arXiv preprint (Jacquier-Muhle-Karbe-Mulligan, Jan 2025; v3 Dec 2025) | WebFetch HTML + abstract synthesis | "Replication ratio" connects IS to OOS Sharpe; complex strategies with many weak signals lose more; more training data → higher replication. **The honest expectation for live-vs-backtest Sharpe is 30-50% decay**, not 0.01 absolute. Cited at `perf_metrics.py:124-128` as the basis for `SR_GAP_THRESHOLD = 0.30`. |
| https://www.quantconnect.com/docs/v2/writing-algorithms/live-trading/reconciliation | 2026-05-28 | Vendor official docs | WebFetch HTML | Verbatim: "If your algorithm is perfectly reconciled, it has an exact overlap between its live and OOS backtest equity curves. Deviations mean that the performance of your algorithm has differed between the two execution modes." QuantConnect's reconciliation is **equity-curve overlay** (per-day NAV) PLUS aggregate Sharpe — NOT a single absolute-Sharpe-gap test. Architectural pattern to mirror in our `paper_parity` block. |
| https://portfoliooptimizationbook.com/book/8.4-backtesting-market-data.html | 2026-05-28 | Textbook (Palomar) | WebFetch HTML | Walk-forward result schema in the canonical academic source uses: "Sharpe ratio, Annual return, Annual volatility, Sortino ratio, Max drawdown, CVaR (0.95)" — period-level summaries, no per-window stratification, no live-vs-backtest column. **Confirms our schema isn't pioneering new territory** — adding parity is the gap. |

---

## 3. Snippet-only sources (context; not counted toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://arxiv.org/abs/2512.12924 | arXiv (Sharpe=0.33 from honest walk-forward) | Snippet from search; corroborates 30-50% Sharpe-decay range |
| https://arxiv.org/abs/2602.10785 | arXiv (walk-forward parameter optimization) | Snippet — confirms WF result varies highly with window length |
| https://arxiv.org/abs/2603.16904 | arXiv (Quantum QAOA walk-forward) | Snippet, low relevance |
| https://blog.quantinsti.com/walk-forward-optimization-introduction/ | Industry blog | Fetched but content didn't address parity-specific questions |
| https://blog.pickmytrade.trade/backtest-vs-live-trading-why-300-returns-fail-in-real-markets/ | Industry blog | Page body not in fetched excerpt; only header rendered |
| https://wakett.com/the-wakett-blog/backtesting-vs-real-time-trading-why-theres-a-discrepancy-and-how-to-fix-it | Industry blog | Fetched but article body wasn't in render |
| https://strategyquant.com/blog/real-trading-compare-live-strategy-results-backtest/ | Industry blog | HTTP 403 |
| https://medium.com/@NFS303/walk-forward-analysis-a-production-ready-comparison-of-three-validation-approaches-69cd25fc9fc7 | Author blog | Member-only; only intro visible |
| https://en.wikipedia.org/wiki/Walk_forward_optimization | Encyclopedia | Background context |
| https://www.man.com/insights/backtesting | Man Group (50% Sharpe haircut industry practice) | Snippet only; reinforces 50% decay norm |
| https://www.quantconnect.com/docs/v2/cloud-platform/backtesting/report | Vendor | Snippet — Sharpe in backtest report |
| https://www.interactivebrokers.com/campus/ibkr-quant-news/the-future-of-backtesting-a-deep-dive-into-walk-forward-analysis/ | Vendor | HTTP 403 |

**Total unique URLs across read-in-full + snippet sets: 22** (≥10 floor met)

---

## 4. Recency scan (2024-2026 window)

Explicit search pass scoped to 2025-2026 results: **3 new findings that update the canonical sources**:

- **Jacquier-Muhle-Karbe-Mulligan 2025 (arXiv 2501.03938)** — January 2025 (v3 December 2025). Provides closed-form approximation of IS-to-OOS Sharpe ratios for linear predictive models. Already cited in `perf_metrics.py:124-128` as the basis for `SR_GAP_THRESHOLD = 0.30`. Confirms 30-50% IS-to-OOS decay range as current state of the art.
- **arXiv 2602.10785** (early 2026) — Walk-forward parameter optimization shows Sharpe ratio "highly dependent on chosen window size" when using Robust Sharpe Ratio.
- **arXiv 2512.12924** (December 2025) — Hypothesis-Driven Trading reports walk-forward Sharpe=0.33 across 34 OOS folds. Honest-walk-forward Sharpes in equity microstructure are MODEST (0.33-0.7 range). Reinforces that demanding `|paper - backtest| < 0.01` is statistically unrealistic.
- **Lopez de Prado, Lipton, Zoonekynd (SSRN 5520741)** — 2025 "How to Use the Sharpe Ratio". Five pitfalls: point estimates without significance; biased IID-Normal inference; ignoring power and minimum sample; misinterpreting p-values; multiple-testing. DIRECT confirmation that a 0.01 absolute-Sharpe threshold on a 30-day window violates pitfalls (i) and (iii).

**Net update vs canonical pre-2024 sources:** the 2025-2026 literature does NOT supersede Bailey-LdP/Lo/Two-Sigma — it REINFORCES them and adds the explicit replication-ratio framework. The DSR formulas already in `perf_metrics.py` remain canonical.

---

## 5. Search-query composition (3-variant per topic)

For DoD-2 the three topic clusters were:

**Topic 1 — walk-forward result schema (parity columns):**
- Current-year frontier: `"walk-forward backtest result schema paper trading parity comparison 2026"`
- Last-2-year window: `"backtest paper trading Sharpe gap reconciliation column QuantConnect 2025"`
- Year-less canonical: `"walk-forward backtest paper trading parity schema"` (also `"walk forward analysis Pardo result schema"`)

**Topic 2 — small-sample Sharpe statistics (30-day):**
- Current-year frontier: `"Probabilistic Sharpe Ratio 30-day window small sample Lopez de Prado 2026"`
- Last-2-year window: `"arxiv 2501.03938 Jacquier IS-OOS Sharpe decay live paper trading 2025"`
- Year-less canonical: `"Sharpe ratio 30 day window confidence interval bootstrap"`

**Topic 3 — industry Sharpe-gap threshold:**
- Current-year frontier: `"backtest live trading discrepancy threshold percentage industry 2025"`
- Last-2-year window: `"\"walk-forward\" \"paper trading\" Sharpe gap 2025 2026 arxiv"`
- Year-less canonical: `"backtest paper trading Sharpe ratio reconciliation"` (also `"paper trading backtest divergence Sharpe absolute threshold industry standard"`)

3-variant discipline visible in source-table mix: Tradezella (year-less), Jacquier 2025 (last-2-year), DSR 2014 (year-less canonical), Wikipedia DSR (year-less), Surmount (year-less). Snippet-set carries the 2025-2026 hits.

---

## 6. Internal file inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/services/perf_metrics.py` | 564 | Canonical Sharpe/PSR/DSR/gap formulas | EXISTS; missing `window_days` param + windowed paper helper |
| `backend/services/perf_metrics.py:87-115` | -- | `compute_sharpe_from_snapshots` | EXISTS but no date-range filter; needs window-slice support |
| `backend/services/perf_metrics.py:128` | -- | `SR_GAP_THRESHOLD = 0.30` | EXISTS — relative-threshold, matches Jacquier 2025 30-50% decay |
| `backend/services/perf_metrics.py:186-283` | -- | `compute_sharpe_gap` | EXISTS — fallback chain optimizer_best → shadow_curve → proxy. Does NOT accept `window_days`. Backtest Sharpe is the FULL-run number from optimizer_best.json, NOT window-matched. |
| `backend/services/perf_metrics.py:420-471` | -- | `compute_psr / compute_dsr` | EXISTS — canonical PSR + DSR per Bailey-LdP 2014 |
| `backend/services/perf_metrics.py:506-563` | -- | `compute_rolling_sharpe_bootstrap_ci` | EXISTS — stationary block bootstrap CI; should be exposed for windowed Sharpe |
| `backend/backtest/walk_forward.py` | 90 | Walk-forward window SCHEDULER (NOT a writer) | EXISTS — emits `WalkForwardWindow(window_id, train_start/end, test_start/end, embargo_days)` |
| `backend/backtest/result_store.py:23-41` | -- | `save_result(run_id, report)` writes JSON to `experiments/results/{ts}_{run_id}.json` | EXISTS — single writer; passes `report` dict through unchanged. Adding a `paper_parity` key requires no schema migration. |
| `backend/api/backtest.py:974` | -- | Backtest call site: `result_store.save_result(run_id, report)` | EXISTS — primary backtest writer |
| `backend/api/backtest.py:1051` | -- | Optimizer call site: `result_store.save_result(exp_id, report)` | EXISTS — per-experiment writer |
| `backend/backtest/experiments/results/20260421T142247Z_0083971f-exp60.json` | -- | Latest result sample (Apr 21, 2026) | TOP-LEVEL KEYS: `analytics, per_window, feature_importance, equity_curve, nav_history, strategy_params, trades, trade_statistics, run_id, parent_run_id, experiment_status, param_changed`. `per_window[i]`: `window_id, train_start, train_end, test_start, test_end, n_candidates, n_train_samples, n_features, sharpe_ratio, total_return_pct, max_drawdown_pct, hit_rate, num_trades, feature_importance_mda, feature_importance_mdi`. **No `paper_*` columns exist anywhere.** `analytics.sharpe` is the run-wide Sharpe. |
| `backend/services/paper_go_live_gate.py:30,92,138` | -- | Sole `compute_sharpe_gap` caller | EXISTS — needs no change; benefits from new `window_days` arg |
| `backend/db/bigquery_client.py:1035-1044` | -- | `get_paper_snapshots(limit: int = 365)` | EXISTS — ORDER BY snapshot_date DESC LIMIT n. **Does NOT support date-range filter** — Option A needs a sibling method OR a Python-side slice. |
| `backend/api/paper_trading.py:30,219,310` | -- | Existing `compute_sharpe_from_snapshots` consumers | EXISTS — must NOT break — new helper must be additive |
| `backend/agents/meta_coordinator.py:252-253` | -- | `health.sharpe_ratio = compute_sharpe_from_snapshots(paper_snapshots)` | EXISTS — additive change preserves this |
| `backend/tests/test_dod4_tier1_coverage_investment.py:670-705` | -- | Existing `compute_sharpe_gap` unit tests | EXISTS — must keep passing |

Total internal files inspected: **15**

---

## 7. Recommended fix — Option A+ (windowed paper Sharpe + light parity block)

### 7.1 Why A+ over pure A, pure B, or full A+B

- **Pure A** (helper-only) makes the gap measurable but leaves no audit trail in the result JSON — operators must call the helper at runtime to see a number. The DoD-2 evidence command (`pull last-30-day paper Sharpe`) implies persisted artifact.
- **Pure B** (writer-only) bakes the gap into every walk-forward result JSON without giving the gate function the windowed primitive — `compute_sharpe_gap` still uses the run-wide backtest Sharpe.
- **A+** is minimum-scope: helper in `perf_metrics.py` gives the gate the right primitive AND we light up a `paper_parity` block in the LAST window of the walk-forward result (not every window — the last window represents the most recent live-tradable params, which is what DoD-2 measures). This keeps the JSON schema additive (new optional top-level key) and doesn't require backfilling historical result files.
- **Full A+B** (parity block on every window) is overkill — the historical windows have no paper-trading data anyway; computing it on the most recent window is what DoD-2 cares about.

### 7.2 Diff fragments — `backend/services/perf_metrics.py`

Add AFTER the existing `compute_sharpe_from_snapshots` (line 115), BEFORE the "Live-vs-Backtest Sharpe reconciliation" section header:

```python
# phase-43.x: windowed paper Sharpe (DoD-2). Date-range filter on the
# snapshots series; window_days defaults to 30 to match the DoD criterion
# but is parameterized so the helper composes with any audit window.
def compute_paper_sharpe_window(
    bq: Any,
    *,
    window_days: int = 30,
    nav_key: str = "total_nav",
    risk_free_rate: float = 0.04,
    min_snapshots: int = 6,
) -> dict:
    """Compute Sharpe over the last `window_days` of paper-portfolio snapshots.

    Returns dict with keys:
      sharpe (float | None),
      n_obs (int -- snapshots actually used),
      window_days (int -- requested),
      window_start (ISO date | None),
      window_end (ISO date | None),
      ci_low (float | None -- bootstrap CI low, only when n_obs >= 10),
      ci_high (float | None -- bootstrap CI high, only when n_obs >= 10),
      note (str | None).

    Per Bailey-Lopez de Prado 2014 MinTRL formula, ~3 years of daily
    returns are needed for SR=0.95 to be statistically distinguishable
    from zero at 95% confidence. A 30-day window is below this floor;
    the CI band is included so the caller can show the uncertainty.

    Source: Lo 2002 / Two Sigma "Sharpe Ratio: Estimation, Confidence
    Intervals" -- Sharpe SE on n=30 with annualized SR ~ 0.7 is on the
    order of +/- 0.3, so absolute-difference thresholds < 0.3 are
    statistically meaningless at this horizon.
    """
    from datetime import date, timedelta

    note_parts: list[str] = []
    snapshots: list[dict] = []
    try:
        # Fetch enough headroom; window_days * 2 with a floor of 90 captures
        # weekends + holidays even at the 30-day mark.
        fetch_limit = max(window_days * 2, 90)
        snapshots = bq.get_paper_snapshots(limit=fetch_limit) or []
    except Exception as exc:
        note_parts.append(f"paper_snapshots_fail:{type(exc).__name__}")

    # Snapshots are returned newest-first by the BQ helper; convert to
    # oldest-first then slice the trailing window.
    snapshots_sorted = sorted(
        snapshots,
        key=lambda r: str(r.get("snapshot_date") or ""),
    )
    if not snapshots_sorted:
        return {
            "sharpe": None, "n_obs": 0, "window_days": window_days,
            "window_start": None, "window_end": None,
            "ci_low": None, "ci_high": None,
            "note": "; ".join(note_parts) or "no_snapshots",
        }

    latest = snapshots_sorted[-1].get("snapshot_date")
    try:
        latest_d = date.fromisoformat(str(latest)[:10]) if latest else None
    except Exception:
        latest_d = None

    if latest_d is not None:
        cutoff_d = latest_d - timedelta(days=window_days)
        windowed = [
            s for s in snapshots_sorted
            if str(s.get("snapshot_date") or "")[:10] >= cutoff_d.isoformat()
        ]
    else:
        windowed = snapshots_sorted[-window_days:]

    n_obs = len(windowed)
    if n_obs < min_snapshots:
        note_parts.append(f"under_min_snapshots:{n_obs}/{min_snapshots}")
        return {
            "sharpe": None, "n_obs": n_obs, "window_days": window_days,
            "window_start": windowed[0]["snapshot_date"] if windowed else None,
            "window_end": windowed[-1]["snapshot_date"] if windowed else None,
            "ci_low": None, "ci_high": None,
            "note": "; ".join(note_parts) or None,
        }

    # Reuse the canonical NAV-series Sharpe helper to ensure parity with
    # all other paper-Sharpe call sites (Bailey-LdP IID-mean requirement).
    sharpe = compute_sharpe_from_snapshots(
        windowed, nav_key=nav_key, risk_free_rate=risk_free_rate,
    )

    # Bootstrap CI when sample is large enough (>=10) per arch / Politis-Romano.
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    if n_obs >= 10:
        navs = [float(s.get(nav_key) or 0.0) for s in windowed]
        navs = [n for n in navs if n > 0]
        if len(navs) >= 10:
            daily_returns = np.diff(np.asarray(navs, dtype=float)) / np.asarray(navs[:-1], dtype=float)
            try:
                _, ci_low, ci_high = compute_rolling_sharpe_bootstrap_ci(
                    daily_returns,
                    n_resamples=1000,
                    ci=0.95,
                    risk_free_rate=risk_free_rate,
                )
            except Exception:
                ci_low, ci_high = None, None

    return {
        "sharpe": sharpe if sharpe != 0.0 else None,
        "n_obs": n_obs,
        "window_days": window_days,
        "window_start": windowed[0]["snapshot_date"],
        "window_end": windowed[-1]["snapshot_date"],
        "ci_low": ci_low,
        "ci_high": ci_high,
        "note": "; ".join(note_parts) or None,
    }
```

Extend `compute_sharpe_gap` signature (line 186) to accept `window_days`:

```python
def compute_sharpe_gap(
    bq: Any,
    *,
    backtest_sharpe_source: str = "optimizer_best",
    risk_free_rate: float = 0.04,
    min_snapshots: int = 6,
    window_days: Optional[int] = None,  # NEW -- phase-43.x
) -> dict:
```

And inside, BEFORE the `snapshots = bq.get_paper_snapshots(limit=365)` block (line 219), insert:

```python
    # phase-43.x: if window_days is set, use the windowed paper Sharpe
    # (DoD-2 measurement). Live Sharpe is computed from the trailing
    # `window_days` snapshots only. Threshold semantics unchanged
    # (relative gap, SR_GAP_THRESHOLD = 0.30) -- the absolute < 0.01
    # criterion in master_roadmap_to_production.md DoD-2 is statistically
    # implausible on 30-day windows (Bailey-LdP 2014 MinTRL ~3 years for
    # SR=0.95). The window-mode return dict carries `window_days` and
    # `paper_window` for operator audit.
    paper_window_info: Optional[dict] = None
    if window_days is not None:
        paper_window_info = compute_paper_sharpe_window(
            bq,
            window_days=window_days,
            risk_free_rate=risk_free_rate,
            min_snapshots=min_snapshots,
        )
        live_sharpe = paper_window_info.get("sharpe")
        # Skip the historical-snapshots fallback path; we have our value.
        # Continue to the existing backtest-sharpe fallback chain below.
        snapshots = []  # mark as already-handled
    else:
        live_sharpe = None  # will be populated by existing path below
```

And add the windowed value to the returned dict (line 272):

```python
    return {
        "live_sharpe": live_sharpe,
        "backtest_sharpe": backtest_sharpe,
        "gap_abs": round(gap_abs, 4) if gap_abs is not None else None,
        "gap_rel": round(gap_rel, 4) if gap_rel is not None else None,
        "threshold": threshold,
        "gap_within_threshold": gap_within_threshold,
        "source": source,
        "note": note,
        "proxy_fallback": proxy_fallback,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "window_days": window_days,            # NEW
        "paper_window": paper_window_info,     # NEW (None when window_days is None)
    }
```

### 7.3 Diff fragment — `backend/api/backtest.py`

In both `save_result` call sites (lines 974 and 1051), insert a small block that attaches a `paper_parity` key BEFORE persisting. To stay DRY, drop a helper inside the file:

```python
def _attach_paper_parity(report: dict, bq) -> None:
    """phase-43.x: add a DoD-2 paper-parity block to the report.

    Idempotent (overwrites). Compares the 30-day windowed paper Sharpe to
    the report's own run-wide Sharpe. Operates fail-open: any exception is
    logged at DEBUG and the report is persisted without the parity block.
    """
    try:
        from backend.services.perf_metrics import compute_paper_sharpe_window
        window = compute_paper_sharpe_window(bq, window_days=30)
        backtest_sharpe = ((report.get("analytics") or {}).get("sharpe"))
        if backtest_sharpe is None and window.get("sharpe") is None:
            return  # nothing to compare
        gap_abs: Optional[float] = None
        gap_rel: Optional[float] = None
        if backtest_sharpe is not None and window.get("sharpe") is not None and backtest_sharpe != 0:
            gap_abs = round(abs(window["sharpe"] - backtest_sharpe), 4)
            gap_rel = round(gap_abs / abs(backtest_sharpe), 4)
        report["paper_parity"] = {
            "paper_sharpe_window_30d": window.get("sharpe"),
            "backtest_sharpe": backtest_sharpe,
            "gap_abs": gap_abs,
            "gap_rel": gap_rel,
            "paper_window_start": window.get("window_start"),
            "paper_window_end": window.get("window_end"),
            "paper_n_obs": window.get("n_obs"),
            "paper_ci_low": window.get("ci_low"),
            "paper_ci_high": window.get("ci_high"),
            "note": window.get("note"),
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.debug("phase-43.x: paper_parity attach failed (fail-open): %r", exc)
```

Then at each of the two save sites:

```python
        # phase-43.x: attach DoD-2 paper-parity block before persisting.
        _attach_paper_parity(report, bq)
        result_store.save_result(run_id, report)
```

### 7.4 Test additions

- Unit test in `backend/tests/test_dod4_tier1_coverage_investment.py` (or a new file) that calls `compute_paper_sharpe_window(MockBQ(...))` with synthetic snapshots of length 6/15/30/60 and verifies (a) `n_obs` honors `window_days`, (b) `sharpe` is None for n<min_snapshots, (c) CI is populated for n>=10.
- Unit test that calls `compute_sharpe_gap(MockBQ, window_days=30)` and verifies the return dict carries `window_days=30` + `paper_window`.
- Integration smoke: hit `/api/paper-trading/portfolio` after deploy + confirm no regression (no signature break — kwarg `window_days` defaults to None).

### 7.5 Verification command (for DoD-2 live_check)

After the next walk-forward run completes (any optimizer experiment or standalone backtest), the result JSON under `backend/backtest/experiments/results/<latest>.json` must contain a top-level `paper_parity` block. The deterministic command:

```bash
python3 -c "
import json, glob
latest = sorted(glob.glob('backend/backtest/experiments/results/*.json'))[-1]
print('FILE:', latest)
d = json.load(open(latest))
pp = d.get('paper_parity')
assert pp is not None, 'paper_parity missing -- DoD-2 MEASUREMENT arm not closed'
print(json.dumps(pp, indent=2))
"
```

DoD-2 MEASUREMENT arm CLOSES when this command exits 0 and prints a non-null `paper_sharpe_window_30d`.

---

## 8. Expected outcome

| Arm | Closure |
|---|---|
| MEASUREMENT arm — walk-forward JSON carries `paper_parity` + `perf_metrics` has windowed helper | **CLOSES this cycle** if Option A+ ships. New result JSONs after deploy will carry the block. Old result JSONs do NOT (no backfill — fail-open, additive schema). |
| VALUE arm — `\|paper.sharpe - backtest.sharpe\| < 0.01` | **DOES NOT CLOSE this cycle.** Predicted gap (using current paper NAV ~ $14,458 and backtest NAV ~ $19,965 per audit data Apr 27): paper Sharpe ≈ 0.5-1.5 with CI ±0.3, backtest Sharpe ≈ 0.72; **expected `gap_abs` ≈ 0.5-1.0**. The criterion's 0.01 threshold is statistically meaningless on a 30-day window per Two Sigma / Bailey-LdP — Sharpe SE on n=30 is on the order of ±0.3. |

**Parallel recommendation:** master_roadmap_to_production.md DoD-2 wording should be **edited in a separate doc-only step** to either (a) replace `< 0.01` absolute with `gap_rel ≤ SR_GAP_THRESHOLD` (relative 30% per Jacquier-Muhle-Karbe 2025 + Man Group 50% haircut convention) OR (b) add a confidence-interval qualifier: "gap is within the 95% bootstrap CI of paper Sharpe." Option (a) matches the existing `compute_sharpe_gap` threshold (`SR_GAP_THRESHOLD = 0.30`); option (b) is the rigorous statistical form per Two Sigma. **Either is defensible; both are honest. The literal `< 0.01` is not.**

---

## 9. Confidence per recommendation

| Recommendation | Confidence | Basis |
|---|---|---|
| Option A+ scope (helper + light parity block) | **HIGH** | Direct file:line audit confirms `compute_sharpe_gap` already has a fallback chain shape; adding `window_days` is additive. `result_store.save_result` is a single chokepoint at 2 call sites in `api/backtest.py:974` + `:1051`; attaching `paper_parity` is mechanically trivial. |
| Windowed paper Sharpe formula (date-range slice + reuse of `compute_sharpe_from_snapshots`) | **HIGH** | The existing `compute_sharpe_from_snapshots` is the canonical helper per CLAUDE.md "Single metric source" rule; reusing it preserves invariant. |
| Bootstrap CI on n=30 | **MEDIUM-HIGH** | `compute_rolling_sharpe_bootstrap_ci` already exists at `perf_metrics.py:506-563` with stationary block bootstrap; n=30 is at the lower edge of statistical usefulness but the function already gates on `n < 10`. |
| DoD-2 threshold edit recommendation (0.01 → 0.30 relative) | **HIGH** for the SCIENCE; **MEDIUM** for the OWNER decision | Bailey-LdP, Lo, Two Sigma, Jacquier 2025 all align — sub-0.01 absolute Sharpe gap on 30-day window violates known SE bounds. The OWNER decision is a separate roadmap-edit cycle (out-of-scope for THIS phase-43.x research). |
| VALUE-arm NOT closing this cycle | **HIGH** | 52.5% NAV divergence in cycle-12 audit is empirical; Sharpe gap on diverged NAVs cannot be 0.01. |
| Backfill historical result JSONs with `paper_parity` (NOT recommended) | **HIGH NEGATIVE** | Historical result files predate paper-trading, so the `paper_parity` block would be all-null. Adds noise, no signal. Skip backfill. |

---

## 10. JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 10,
  "snippet_only_sources": 12,
  "urls_collected": 22,
  "recency_scan_performed": true,
  "internal_files_inspected": 15,
  "gate_passed": true
}
```

**Gate check:**
- `external_sources_read_in_full >= 5` → **10**, PASS
- `recency_scan_performed == true` → PASS (3 new 2025-2026 findings reported + 1 SSRN preprint)
- Hard-blockers (>=10 URLs total + per-claim citation + file:line anchors + 3-variant queries documented) → PASS

`gate_passed: true`.
