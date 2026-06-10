# Research Brief — phase-47.6: Dynamic strategy rotation (per-strategy DSR + promote highest earner)

Tier: **complex** | Cycle 7 of production-ready+money push | Researcher: Layer-3 MAS
North Star (verbatim): "dynamically shift capital to whichever strategy is making the most money."

---

## TL;DR (read this first)

**Most of the rotation machinery ALREADY EXISTS.** The gap is narrow and well-bounded:

1. **The QuantOptimizer already rotates `strategy` as a categorical param** across the 5
   (+`blend`) strategies (`backend/backtest/quant_optimizer.py:108` `_CATEGORICAL_PARAMS["strategy"] = AVAILABLE_STRATEGIES`). It can already *search* for the best strategy.
2. **A full DSR/PBO promotion pipeline exists**: `promoter.py::Promoter.promote` (DSR>=0.95
   gate), `gate.py::PromotionGate` (DSR>=0.95 AND PBO<=0.20), `friday_promotion.py`
   (weekly, ranks by DSR desc / PBO asc, top-N, writes BQ), `promoter.write_to_registry`
   (atomic supersession + active row), and the loop reads it via
   `autonomous_loop.load_promoted_params` (`backend/services/autonomous_loop.py:46`).
3. **`compute_deflated_sharpe(observed_sr, num_trials, ...)` exists** with the multiple-
   testing `num_trials` deflation term (`backend/backtest/analytics.py:239`).
4. **A `promoted_strategies` BQ table** (MERGE by `(week_iso, strategy_id)`) + read/write/
   supersede methods exist (`bigquery_client.py:702-845`).

**What is MISSING (the actual phase-47.6 increment):** there is NO function that takes the
5 named STRATEGY_REGISTRY strategies, computes a per-strategy DSR (deflated by N=5
candidates), and selects the top-DSR strategy with (a) the DSR>=0.95 guard, (b) an
anti-churn min-improvement vs the incumbent, and (c) an incumbent-tie rule. The existing
pipeline promotes *optimizer-trial params* (one search trajectory), NOT a *bake-off across
the 5 named strategies*. The loop only reads `strategy` from `params` for a heartbeat log
(`autonomous_loop.py:1082`); the comment at line 1077 literally says "Full router
activation deferred to phase-31."

**Decision on backtests:** existing `quant_results.tsv` does NOT carry a per-strategy
column (header is `timestamp/run_id/param_changed/.../dsr/.../params_json`), and
`optimizer_best.json` holds only the single incumbent (`triple_barrier`, Sharpe 1.17,
DSR 0.9526). Per-strategy DSR is NOT readable from existing artifacts. **Either** parse
`params_json` per TSV row to bucket by `strategy` (sparse — most rows are triple_barrier),
**or** run 5 quant-only ($0-LLM) backtests this cycle. See "Backtests: required?" below.

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/backtest/backtest_engine.py` | 32-38 | `STRATEGY_REGISTRY` — the 5 strategies → label-method names | EXISTS. `meta_label` reuses TB labels. |
| `backend/backtest/backtest_engine.py` | 199 | `self.strategy = strategy if strategy in STRATEGY_REGISTRY else "triple_barrier"` | EXISTS — engine accepts a `strategy` arg, defaults to TB. |
| `backend/backtest/quant_optimizer.py` | 64,108 | `AVAILABLE_STRATEGIES` (5 + `blend`); `_CATEGORICAL_PARAMS["strategy"]` | EXISTS — optimizer already rotates strategy as a categorical search dim. `lock_strategy` flag at :409 can pin it. |
| `backend/backtest/analytics.py` | 239-285 | `compute_deflated_sharpe(observed_sr, num_trials, variance_of_srs=0.5, skewness, kurtosis, T)` | EXISTS — `num_trials` IS the multiple-testing deflation. Returns P[SR>=E[max SR]] in [0,1]. |
| `backend/autoresearch/gate.py` | 19-39 | `PromotionGate(min_dsr=0.95, max_pbo=0.20).evaluate(trial)` → `{promoted, reason}` | EXISTS — pure, fail-open. |
| `backend/autoresearch/promoter.py` | 29-59 | `Promoter.promote` (DSR>=0.95, shadow_min_days>=5), `position_size` (DSR-linked), `on_dd_breach` | EXISTS. |
| `backend/autoresearch/promoter.py` | 61-179 | `Promoter.write_to_registry` — atomic supersede prior active → write new active → P0 Slack | EXISTS — the auto-switch path (phase-25.R "red-line goal-c"). |
| `backend/autoresearch/friday_promotion.py` | 32-182 | `run_friday_promotion(week_iso, candidates, top_n=1, max_n=3, ...)` — weekly gate, ranks DSR desc/PBO asc, writes BQ rows | EXISTS — weekly cadence already implemented. Operates on `candidates` list (trials), not the 5 named strategies. |
| `backend/autoresearch/monthly_champion_challenger.py` | 43-216 | `run_monthly_sortino_gate` — Sortino delta>=0.3, PBO<0.2, DD ratio<=1.2, 48h HITL | EXISTS — monthly champion/challenger with human approval. |
| `backend/db/bigquery_client.py` | 702-845 | `save_promoted_strategy` (MERGE on week_iso+strategy_id), `get_latest_promoted_strategy` (status filter, DSR-desc tiebreak), `update_promoted_strategy_status` (supersede) | EXISTS — typed `promoted_strategies` table I/O. |
| `backend/services/autonomous_loop.py` | 46-74 | `load_promoted_params(bq)` — 3-tier fallback: BQ promoted → optimizer_best.json → {} | EXISTS — the loop's strategy/params read path. **This is where the selected strategy is consumed.** |
| `backend/services/autonomous_loop.py` | 209-214 | `best_params = load_promoted_params(bq)`; stamps `summary["strategy_params"]` | EXISTS — call site. |
| `backend/services/autonomous_loop.py` | 1070-1101 | strategy_decisions heartbeat (phase-30.7); reads `best_params.get("strategy")` ONLY for logging | EXISTS — comment line 1077: "Full router activation deferred to phase-31." |
| `backend/db/bigquery_client.py` | 403-427 | `save_strategy_decision(record)` → `pyfinagent_data.strategy_decisions` | EXISTS. |
| `scripts/migrations/add_strategy_decisions_table.py` | 10-17 | `strategy_decisions` schema: decided_strategy, prior_strategy, trigger, decay_signal, decay_attribution, rationale | EXISTS (phase-26.5). Partitioned DATE(ts), clustered (trigger, decided_strategy). |
| `backend/backtest/experiments/optimizer_best.json` | — | Single incumbent only: `strategy=triple_barrier`, Sharpe 1.1705, DSR 0.9526, stamped 2026-04-06 | EXISTS — NO per-strategy breakdown. |
| `backend/backtest/experiments/quant_results.tsv` | — | Cols: timestamp/run_id/param_changed/metric_before/after/delta/status/dsr/top5_mda/**params_json**/parent_run_id | EXISTS — `strategy` only inside `params_json`; NO dedicated strategy column. |

### Existing tests (the test idiom to follow)
- `tests/autoresearch/test_friday_promotion.py`, `test_monthly_champion_challenger.py`,
  `test_slot_accounting.py`, `test_rollback.py` — pure-function behavioral tests on the
  promotion gates (inject candidates dicts, assert `promoted_ids`/verdict).
- `tests/verify_phase_25_R.py`, `_C3.py`, `_B3.py`, `_A3.py` — wiring tests for
  promoter→registry→loop path. **These are the exact harness to extend.**

---

## External research

### Search-query variants run (3-variant discipline)
1. **Current-frontier / recency (2024-2026):** "regime-based strategy rotation hysteresis
   transaction costs whipsaw avoid overtrading 2025"; "Explainable Regime Aware Investing 2026".
2. **Last-2-year window:** jump-model regime switching (arXiv:2402.05272, 2024);
   regime-aware investing (arXiv:2603.04441, 2026).
3. **Year-less canonical:** "Deflated Sharpe Ratio strategy selection multiple testing
   Bailey Lopez de Prado 2014"; "strategy ensemble selection deflated sharpe ratio
   overfitting best of N backtests"; "deflated sharpe ratio python implementation
   effective number of trials".

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio | 2026-05-29 | doc (encyclopedic, cites primary) | WebFetch full | N = count of candidate strategies; E[max SR] grows with N; **DSR>=0.95 threshold**; correlated trials must be clustered to an *effective* N (ONC / hierarchical) — raw count over-deflates. |
| https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf | 2026-05-29 | paper (Bailey & LdP 2014, primary) | WebFetch full (PDF text extracted) | "DSR corrects for selection bias when the best strategy is chosen among many independent alternatives." Penalty term increases with N. Track-record length reduces the relative penalty. |
| https://arxiv.org/html/2402.05272 | 2026-05-29 | paper (Statistical Jump Model, 2024) | WebFetch full (arXiv HTML) | **Jump penalty λ is the anti-churn lever.** λ=50 → 0.8 regime shifts/yr (vs HMM 8.5). JM turnover 44% vs HMM 141%. Survives 10bps cost: Sharpe 0.48→0.68 (+42%), MDD 55%→27%. 1-day execution delay. |
| https://arxiv.org/html/2603.04441v1 | 2026-05-29 | paper (Explainable Regime-Aware Investing, 2026) | WebFetch full (arXiv HTML) | Regime allocation Sharpe **2.18 vs 1.18 SPX buy&hold** (+1.0), MDD −5.43% vs −14.62%. Turnover control via L1 weight-change penalty `−τ‖w−w_{t-1}‖₁` → daily turnover 56.65%→0.79%. Weekly model-order re-selection. |
| https://www.volatilitytradingstrategies.com/blog/hysteresis-and-the-defensive-rotation-strategy-part-2 | 2026-05-29 | industry blog (practitioner) | WebFetch full | Hysteresis = asymmetric thresholds + dead zone (exit at 20% vol, re-enter at 30% → 10% buffer). **"Stay unless conditions clearly improve" incumbent bias.** Buffer trades responsiveness for stability; reduces whipsaw with negligible return cost. |
| https://medium.com/balaena-quant-insights/deflated-sharpe-ratio-dsr-33412c7dd464 (via search synthesis) + Bailey & Borwein "Probability of Backtest Overfitting" https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf | 2026-05-29 | paper/industry | WebSearch full-snippet synthesis | "Most important missing info from virtually all published backtests is the number of trials." CPCV gives lower PBO + higher DSR than walk-forward — pyfinagent's `gate.py::cpcv_folds` already implements this. |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 | paper (DSR SSRN) | SSRN landing page = abstract only; the davidhbailey.com PDF is the same paper read in full. |
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2465675 | paper ("Deflating the Sharpe Ratio", LdP) | Companion; SSRN gated; primary content covered by the read-in-full DSR paper. |
| https://jpm.pm-research.com/content/40/5/94.abstract | paper (JPM published DSR) | Paywalled journal abstract. |
| https://gmarti.gitlab.io/qfin/2018/05/30/deflated-sharpe-ratio.html | blog (Python DSR recipe) | 302-redirects to a GitLab auth page (not a content host); declined to follow into auth. Implementation covered by analytics.py + Wikipedia effective-N note. |
| https://github.com/rubenbriones/Probabilistic-Sharpe-Ratio | code (PSR/DSR Python) | Reference impl with `num_independent_trials`/`expected_maximum_sr`/`deflated_sharpe_ratio` — confirms analytics.py shape; not needed in full (we have our own). |
| https://github.com/esvhd/pypbo | code (PBO Python) | PBO reference; pyfinagent already has CPCV in gate.py. |
| https://www.mdpi.com/2079-9292/15/6/1334 | paper (Regime-Aware LightGBM walk-forward) | Recency-scan candidate; jump-model + regime-aware papers already cover the mechanism. |
| https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110 | paper (backtest overfitting ML era) | Paywalled; CPCV-superiority point captured via search synthesis + Bailey-Borwein PDF. |
| https://abovethegreenline.com/whipsaw-trading/ | community blog | Lower-tier; hysteresis covered by the practitioner blog read in full. |
| https://arongroups.co/forex-articles/market-regime-trading/ | community blog | Lower-tier forex content. |

### Recency scan (2024-2026)

**Searched** for 2024-2026 literature on per-strategy DSR selection + regime/strategy
rotation. **Result: found 2 strong new findings that COMPLEMENT (do not supersede) the
canonical Bailey & LdP 2014 DSR work:**

1. **Statistical Jump Model (arXiv:2402.05272, 2024)** — supplies the empirically-validated
   anti-churn mechanism the North Star needs. A switch-penalty (jump penalty λ) cut regime
   shifts from ~9/yr to **<1/yr** and turnover from 141%→44%, while *improving* Sharpe
   (0.48→0.68) net of 10bps costs. Directly validates "min-improvement + holding period"
   over naive best-of-N rotation.
2. **Explainable Regime-Aware Investing (arXiv:2603.04441, 2026)** — current-frontier;
   regime allocation hit **Sharpe 2.18 vs 1.18 SPX** with MDD nearly 3x smaller, using an
   L1 weight-change transaction-cost penalty to crush turnover (56.65%→0.79%). The +1.0
   Sharpe uplift brackets the roadmap's +0.3-0.7 estimate as plausible (their baseline is
   raw SPX; pyfinagent's incumbent is already Sharpe 1.17, so expect the lower end).

The 2014 DSR remains the canonical SELECTION-bias tool; the 2024/2026 papers add the
SWITCHING-discipline (penalty/hysteresis) layer. Both are reflected in the plan below.

### Key findings (per-claim cited)

1. **DSR is the exactly-correct tool for best-of-N strategy selection.** "The deflated
   Sharpe ratio corrects for selection bias when the best strategy is chosen among many
   independent alternatives tested." (Bailey & LdP 2014, davidhbailey.com PDF). Picking the
   top of 5 STRATEGY_REGISTRY strategies IS this exact scenario — so the selector MUST pass
   `num_trials >= 5` (not 1) into `compute_deflated_sharpe`.
2. **N must be the EFFECTIVE number of independent trials, not the raw count.** "Many trials
   are not independent due to overlapping features... cluster similar strategies to estimate
   the effective number of independent trials." (Wikipedia DSR, citing LdP). pyfinagent's 5
   strategies are correlated (`meta_label` literally reuses `triple_barrier` labels; QM/MR/FM
   share the same feature vector). For v1, N=5 is a *conservative* (over-deflating) choice —
   acceptable, and documented as such. Clustering to effective-N is a DEFERRED refinement.
3. **The number of trials is the single most important missing backtest datum.** "The most
   important missing information from virtually all published backtests is the number of
   trials attempted." (Bailey & Borwein, backtest-prob.pdf). The selector must therefore
   RECORD the N it used in `strategy_decisions.rationale` / the promoted row.
4. **A switch penalty / min-improvement threshold is empirically essential, not optional.**
   Without it, regime models flip ~9x/yr (HMM) destroying returns via cost; with λ they flip
   <1x/yr and beat buy-and-hold net of cost. (arXiv:2402.05272). → the anti-churn
   min-improvement (Δ-DSR) + holding-period guard is load-bearing, not gold-plating.
5. **Incumbent bias / "stay unless clearly better" is the canonical hysteresis design.**
   Asymmetric thresholds with a dead zone prevent whipsaw at negligible return cost.
   (volatilitytradingstrategies.com). → the tie/near-tie rule must FAVOR THE INCUMBENT.
6. **A holding period + execution delay still captures the benefit.** Even with a 1-day
   execution delay and ~15-day detection lag, the persistent strategy "effectively prevents
   ~20% drawdown." (arXiv:2402.05272). → a weekly cadence (the North Star's "weekly
   promotion") is well within the regime that still works; no need for intraday reactivity.

### Consensus vs debate (external)

- **Consensus:** (a) DSR is the right selection metric under multiple testing; (b) raw N
  over-deflates when trials are correlated → use effective-N; (c) a switching penalty /
  hysteresis is mandatory to avoid cost-destroying whipsaw; (d) regime/strategy rotation can
  add meaningful Sharpe (the 2024/2026 papers show +42% to +85%).
- **Debate / open:** how to set the min-improvement threshold (papers tune λ via CV; we will
  use a fixed conservative Δ for v1). Whether to cluster the 5 strategies for effective-N
  (LdP says yes; v1 defers and uses conservative N=5). Daily vs weekly cadence (papers vary;
  North Star says weekly).

### Pitfalls (from literature)

- **P1 — Passing N=1 to DSR after a 5-way bake-off.** This is the classic selection-bias
  error the DSR exists to prevent. The selector MUST use N>=5. (Bailey & LdP).
- **P2 — Over-deflation from raw N on correlated strategies.** Acceptable for v1 (conservative
  = harder to promote = safer), but must be DOCUMENTED so a future cycle adds clustering.
- **P3 — Churn / whipsaw.** Selecting a new strategy on a razor-thin DSR edge → flip-flop →
  transaction costs eat the alpha. Mitigated by the Δ-min-improvement + incumbent-tie rule.
- **P4 — Look-ahead in the bake-off.** Each per-strategy DSR must come from a walk-forward /
  CPCV backtest (no future leakage). pyfinagent's engine already enforces walk-forward +
  5-day embargo, so reusing `BacktestEngine` per strategy inherits this.
- **P5 — Sample too short for DSR.** DSR needs adequate T relative to N. The engine produces
  ~500+ daily NAV snapshots over 2018-2025, so T >> N=5 — fine.

### Application to pyfinagent (external → internal anchors)

| External finding | pyfinagent action | Internal anchor |
|---|---|---|
| DSR for best-of-N (F1) | Selector calls `compute_deflated_sharpe(sr, num_trials=5, ...)` per strategy | `backend/backtest/analytics.py:239` |
| Effective-N caveat (F2) | v1 uses conservative N=5; document the over-deflation; defer clustering | new selector module + brief note |
| Record N (F3) | Write `num_trials` into the promoted row / `strategy_decisions.rationale` | `bigquery_client.py:702`, `:403` |
| Switch penalty (F4) | Anti-churn: promote new strategy only if `DSR_new - DSR_incumbent >= Δ` (Δ default 0.05) AND new DSR>=0.95 | new selector + reuse `gate.py:19` DSR>=0.95 |
| Incumbent bias (F5) | Tie/near-tie → keep incumbent (the `>=` Δ test already does this; exact tie returns incumbent) | new selector |
| Holding period / weekly (F6) | Wire selection into the EXISTING weekly `friday_promotion` cadence; loop reads via `load_promoted_params` | `friday_promotion.py:32`, `autonomous_loop.py:46` |

---

## MINIMAL-VIABLE shippable + testable increment (ONE cycle)

### What already exists (DO NOT rebuild)
- DSR computation with `num_trials` deflation — `analytics.py:239`.
- DSR>=0.95 gate (+PBO<=0.20) — `gate.py:PromotionGate`.
- Weekly promotion cadence, top-N ranking by DSR — `friday_promotion.py`.
- Atomic supersession + active-row write + P0 Slack — `promoter.py:write_to_registry`.
- `promoted_strategies` BQ table I/O — `bigquery_client.py:702-845`.
- Loop consumes the promoted choice — `autonomous_loop.py:46` `load_promoted_params`.
- Optimizer already rotates `strategy` categorically — `quant_optimizer.py:108`.
- `strategy_decisions` audit table + writer — `bigquery_client.py:403`, migration phase-26.5.

### What must be BUILT (the 47.6 delta — small, pure, testable)
A single pure selection function plus its wiring. Proposed shape:

```
backend/autoresearch/strategy_selector.py  (NEW, pure functions, ASCII-only)

def select_best_strategy(
    per_strategy: list[dict],          # [{"strategy": "...", "sharpe": float, "dsr": float, "pbo": float?}, ...]
    incumbent: str | None,
    *,
    min_dsr: float = 0.95,             # reuse the project DSR-guard
    min_improvement: float = 0.05,     # anti-churn Δ-DSR vs incumbent (F4/F5)
    num_trials: int | None = None,     # if dsr not pre-deflated, recompute with N=len(per_strategy)
) -> dict:
    # Returns {"selected": str, "reason": str, "switched": bool,
    #          "incumbent": str|None, "num_trials": int, "candidates_ranked": [...]}
```

Selection logic (all behaviorally testable, NO live cycle needed):
1. Filter to candidates with `dsr >= min_dsr` (and `pbo <= 0.20` if present). Reuse
   `PromotionGate.evaluate` so the guard stays single-sourced.
2. If none pass → keep incumbent (or `triple_barrier` default), `switched=False`,
   `reason="no_candidate_cleared_dsr_guard"`.
3. Rank passers by DSR desc, then PBO asc (mirror `friday_promotion.py:107` tie-break).
4. Top candidate `c*`. If `c* == incumbent` → keep, `switched=False`.
5. If `c* != incumbent`: switch ONLY if `c*.dsr - incumbent_dsr >= min_improvement`
   (incumbent_dsr=0 / -inf when no incumbent → first selection always allowed).
   Else keep incumbent (`reason="below_min_improvement"`). **Incumbent-tie favors incumbent.**
6. Always set `num_trials = len(per_strategy)` and surface it (pitfall P1/F3).

### Where the per-strategy DSRs come from — **5 backtests required THIS cycle? NO for the test; YES for a real selection.**

- **For the shippable+testable increment:** the selection FUNCTION is pure and takes
  per-strategy DSRs as input → unit-tested with synthetic dicts. **No backtests needed to
  ship + test the logic.** This is the MVP and what the behavioral test covers.
- **To produce REAL per-strategy DSRs:** existing artifacts are insufficient —
  `optimizer_best.json` holds only the incumbent; `quant_results.tsv` has `strategy` only
  inside `params_json` and is sparse/triple_barrier-dominated. So a real selection needs **5
  quant-only ($0-LLM) walk-forward backtests** (one per STRATEGY_REGISTRY entry, same
  windows/universe, then `compute_deflated_sharpe(sr, num_trials=5)` each). This is a thin
  driver: loop over `STRATEGY_REGISTRY`, call `BacktestEngine(strategy=s).run_backtest()`,
  collect `(sharpe, dsr@N=5)`. **Recommendation: ship the selector + its unit test THIS
  cycle (the immutable success criterion); run the 5-backtest driver as an in-cycle
  integration smoke IF time permits, else DEFER the live 5-backtest sweep to a follow-up.**
  The 5-backtest sweep is $0-LLM but multi-minute (cache.preload_macro mandatory), so it is
  the natural deferral boundary.

### Wiring (minimal)
- The selector's chosen strategy flows into the EXISTING promotion row: build a
  `candidate` dict for `c*` and pass it to `friday_promotion`/`save_promoted_strategy` so
  `load_promoted_params` (`autonomous_loop.py:46`) picks it up at next cycle start — **no new
  read path in the loop.** The loop already consumes `promoted_strategies`.
- Log the decision to `strategy_decisions` (`bigquery_client.py:403`) with
  `trigger="performance_threshold"`, `decided_strategy=c*`, `prior_strategy=incumbent`,
  `rationale` including `num_trials` and the Δ-DSR. The heartbeat writer at
  `autonomous_loop.py:1082` already targets this table — extend it from heartbeat-only to
  carry a real decision when a selection runs.

### The behavioral test (verification WITHOUT a multi-hour live cycle)
`tests/autoresearch/test_strategy_selector.py` (mirror `test_friday_promotion.py` idiom):
1. **Picks the highest-DSR passer.** Given 5 strategies with DSRs `[0.99, 0.97, 0.96, 0.80,
   0.50]`, all PBO ok → selects the 0.99 one; `num_trials==5`.
2. **Respects the DSR>=0.95 guard.** Given all DSRs < 0.95 → keeps incumbent,
   `switched=False`, `reason="no_candidate_cleared_dsr_guard"`.
3. **Respects anti-churn min-improvement.** Incumbent DSR 0.96; best challenger 0.98 with
   `min_improvement=0.05` → Δ=0.02 < 0.05 → keeps incumbent, `reason="below_min_improvement"`.
   Then challenger 0.96→0.97 vs incumbent 0.90, Δ=0.07 → switches.
4. **Incumbent-tie favors incumbent.** Incumbent and challenger both DSR 0.97 → keeps
   incumbent, `switched=False`.
5. **First selection (no incumbent) always allowed** when a candidate clears the guard.
6. **PBO veto** (if present): a 0.99-DSR candidate with PBO 0.30 is filtered out.
7. **(Optional dry wiring check)** assert `load_promoted_params` reads `strategy` from the
   promoted row shape the selector produces (construct the dict, monkeypatch
   `bq.get_latest_promoted_strategy` to return it, assert the returned params carry the
   selected `strategy`). This is the "loop consumes the promoted value" dry check — no live
   cycle.

### Explicitly DEFERRED (out of scope for 47.6)
- **Cron scheduling** of the 5-backtest sweep (APScheduler weekly job) — defer; the
  `friday_promotion` cadence + manual/QA-triggered sweep is enough for v1.
- **Live 5-strategy backtest sweep as a committed artifact** — may run as an in-cycle smoke
  but the *committed* deliverable is the selector + unit test; a scheduled real sweep is a
  follow-up.
- **Effective-N clustering** (ONC / hierarchical) to replace conservative N=5 — documented
  pitfall P2; defer to a refinement cycle.
- **Real-capital activation** — `monthly_champion_challenger` keeps `actual_replacement`
  gated on `Settings.real_capital_enabled` (default False, SR 11-7). Paper-only stays.
- **Removing the `lock_strategy` path / unifying with the optimizer's categorical search** —
  the optimizer's in-search rotation and this explicit bake-off are complementary; do not
  refactor them together this cycle.

### Is the +0.3-0.7 Sharpe plausible?
Yes, with the lower end most likely. The 2026 regime paper shows +1.0 Sharpe vs raw SPX
(arXiv:2603.04441) and the 2024 jump model +0.20 net of costs (arXiv:2402.05272). Since
pyfinagent's incumbent is already a strong Sharpe 1.17 (not raw SPX), expect the conservative
end of the roadmap range, and ONLY if a non-incumbent strategy actually clears DSR>=0.95 by
the Δ margin on real backtests — which the 5-backtest sweep (deferred or in-cycle smoke) will
reveal.

---

## Research Gate Checklist

Hard blockers — `gate_passed` is false if any unchecked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6: Wikipedia DSR, Bailey
      DSR PDF, jump-model arXiv HTML, regime-aware arXiv HTML, hysteresis practitioner blog,
      Bailey-Borwein PBO synthesis)
- [x] 10+ unique URLs total (19 across read-in-full + snippet-only tables)
- [x] Recency scan (last 2 years) performed + reported (2024 jump model + 2026 regime-aware)
- [x] Full papers / pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (engine, optimizer, analytics, gate,
      promoter, friday_promotion, monthly_champion, bigquery_client, autonomous_loop)
- [x] Contradictions / consensus noted (daily-vs-weekly cadence; effective-N debate)
- [x] All claims cited per-claim (Key findings section)

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 19,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "report_md": "handoff/current/research_brief_phase_47_6_strategy_rotation.md",
  "gate_passed": true
}
```
