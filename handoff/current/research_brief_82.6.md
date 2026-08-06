# Research Brief -- phase-82.6: registry-to-live selection bridge (DESIGN ONLY)

**Tier:** complex | **Audit-class:** true (loop-until-dry, K=2) | **Date:** 2026-08-06
**Status:** COMPLETE -- `gate_passed: true` (6 sources read in full, 34 URLs,
recency scan done, 13 coverage rounds, 2 dry). Envelope at the tail.

> **Two headline corrections to the step's MEASURED claim** -- read Q1 first.
> The step says the registry's exit params cross into the live cycle and that
> `strategy` is label-only. Measured: the exit params do NOT cross either (they
> land in a summary key with zero readers), and a strategy NAME *does* gate a
> live risk control at `backend/services/paper_trader.py:1427` -- currently inert
> only because `paper_positions.entry_strategy` is NULL for every row.

## Objective

DESIGN (do not build) the bridge between the strategy registry
(`STRATEGY_REGISTRY` / `optimizer_best.json`) and the LIVE selection path.
The live book is working (+18.86%); a selection-path change is the
highest-regression-risk edit in the system. This step must NOT wire selection.

Seven questions from the caller:
- Q1 re-derive the MEASURED claim's line numbers; is `strategy` label-only?
- Q2 enumerate STRATEGY_REGISTRY; structural predicate for criterion-3 test
- Q3 the exact insertion point(s) in the live cycle
- Q4 which promotion gates exist in code, thresholds, live vs dark
- Q5 rollback path from existing machinery
- Q6 what `strategy_decisions` records; is decided==prior every cycle
- Q7 prior art in-repo (phase-31 deferral)

---

## Search-query composition (3-variant discipline)

- **Year-less canonical:** `strategy rotation live deployment promotion gate deflated Sharpe ratio probability of backtest overfitting`; `champion challenger model deployment shadow mode canary rollout guardrail metrics rollback`
- **Last-2-year (2024/2025):** `strategy rotation switching penalty transaction costs regime jump model 2025 2024 turnover net-of-cost Sharpe`
- **Current-year frontier (2026):** `automated trading strategy promotion pipeline 2026 walk-forward deflated Sharpe gate live deployment`

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://arxiv.org/html/2603.20319v1 | 2026-08-06 | preprint (peer-tier 1) | WebFetch arXiv HTML (full text) | **Rotation strategies are the WORST case for implementation divergence.** Zero-cost: "every one of the 10 pairwise comparisons yields a relative difference of exactly 0.0000%". With costs, large rotation BM03 = 3.6091%, BM04 = **3.7077%** (largest). "divergence rises monotonically with cost intensity (Spearman rho=0.93, p<0.001)". Recommends "at least two independent validators, chosen to maximise implementation diversity ... with an explicit audit of each engine's cost model against a reference specification." |
| 2 | https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf | 2026-08-06 | peer-reviewed (Bailey & Lopez de Prado, DSR) | WebFetch returned binary -> **pdfplumber chain per research-gate.md Step 3**, 45,402 chars extracted | "The Deflated Sharpe Ratio (DSR) corrects for two leading sources of performance inflation: Selection bias under multiple testing and non-Normally distributed returns." Eq. (1) gives E[max Sharpe] after N independent trials via the Euler-Mascheroni constant. **"Appendix 3 shows how N can be determined when the trials are not independent."** Holdout critique: "If we apply the holdout method enough times (say 20 times for a 95% confidence level), false positives are no longer unlikely: They are expected." Type-I asymmetry: "they would rather exclude a true strategy than risking the addition of a false one." |
| 3 | https://arxiv.org/html/2402.05272v2 | 2026-08-06 | peer-reviewed (J. Asset Mgmt 2024) | WebFetch arXiv HTML (full text) | **Directly validates the in-repo hysteresis claim.** "With a zero jump penalty, the model reduces to a k-means clustering algorithm... As lambda increases, state transitions become less frequent." S&P 500 turnover **HMM 141% -> JM 44%**; DAX 246->170; Nikkei 290->72. "we impose a conservative transaction cost of 10 basis points for each one-way trade". Net Sharpe S&P 0.68 vs 0.48 buy-and-hold. Penalty tuned by "time-series cross-validation... updating the optimal jump penalty monthly". |
| 4 | https://atlan.com/know/shadow-deployment-for-ml-models/ | 2026-08-06 | industry doc | WebFetch (full page) | "The shadow model processes the same input data, and its outputs are logged for offline comparison against the champion." Limits: "doubled infrastructure costs during the shadow window"; "the possibility that shadow conditions do not fully replicate production behavior if the model has side effects or depends on stateful interactions". Promotion: "Promote when the shadow model meets predefined thresholds across accuracy, latency, throughput, and prediction-distribution alignment over a statistically significant observation window." |
| 5 | https://www.systemoverflow.com/learn/ml-infrastructure-mlops/ci-cd-ml/shadow-and-canary-deployment-for-models | 2026-08-06 | industry doc | WebFetch (full page) | Concrete staged ladder: "1% traffic for 30 min (catch obvious regressions), then 5% for 2 hours, then 25%, then full rollout." Guardrails: "p95 latency under 50ms, error rate under 0.1%, online metric delta within 2%." Rollback: "p95 > 50ms for 15 consecutive minutes, or CTR drop > 2% for 30 minutes"; must "revert to prior model in under 2 minutes". |
| 6 | https://arxiv.org/html/2512.12924v1 | 2026-08-06 | preprint | WebFetch arXiv HTML (full text) | **[ADVERSARIAL / contrarian]** A rigorous walk-forward framework that DECLINES the deflation gate: "All tests reported without adjusting for multiple comparisons to maintain transparency about statistical limitations." Honest nulls: "p-value = 0.34 (two-sided)"; "95% bootstrap confidence interval is [-0.12%, +0.43%], which includes zero"; "approximately 12% power" vs 80% required. Provides **no explicit promotion gate**; recommends conditional/zero allocation instead: "allocation to the strategy should be minimal or zero" in low-vol regimes. |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://papers.ssrn.com/sol3/Delivery.cfm/7104418.pdf?abstractid=7104418&mirid=1 | paper ("Build the Judge Before the Strategy") | **HTTP 403** -- SSRN blocks WebFetch. Highly relevant (pre-registered DSR/PBO gates frozen before returns are computed); flagged for manual retrieval. |
| https://www.calibreos.com/learn/mlsd-canary-deployment | industry doc | HTTP 403 |
| https://getnadir.com/blog/shadow-testing-canary-rollout-llm-model-swap/ | blog | HTTP 403 |
| https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110 | peer-reviewed | paywalled abstract only |
| https://link.springer.com/article/10.1057/s41260-024-00376-x | peer-reviewed | journal version of source 3 (arXiv read instead) |
| https://arxiv.org/abs/2402.05272 | preprint abstract | superseded by the HTML full text (source 3) |
| https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID4719989_code5886757.pdf | preprint | SSRN mirror of source 3 |
| https://collaborate.princeton.edu/en/publications/downside-risk-reduction-using-regime-switching-signals-a-statisti/ | listing | metadata only |
| https://ideas.repec.org/a/pal/assmgt/v25y2024i5d10.1057_s41260-024-00376-x.html | listing | metadata only |
| https://www.researchgate.net/publication/384072302_Downside_risk_reduction... | listing | metadata only |
| https://www.mdpi.com/2079-9292/15/6/1334 | peer-reviewed (2026) | recency-scan hit; DSR=0.69 reported as failing significance |
| https://arxiv.org/pdf/2602.10785 | preprint (2026) | parameter-optimization; adjacent, not load-bearing |
| https://www.bscapitalmarkets.com/statistical-jump-models-for-regime-switching.html | industry | secondary summary of source 3 |
| https://alpha-suite.org/blog/overfitting-backtesting | blog | community tier |
| https://www.turbinefi.com/blog/why-backtests-lie-prediction-market-overfitting-2026 | blog | community tier |
| https://research.mental-momentum.ai/r/backtest-overfitting-trading-strategy-ju55g3 | blog | community tier |
| https://www.backtester.run/backtesting/overfitting | blog | community tier |
| https://fortraders.com/blog/how-to-avoid-bias-in-backtesting | blog | community tier |
| https://fortraders.com/blog/use-ai-optimize-trading-strategy | blog | community tier |
| https://medium.com/@trading.dude/the-truth-about-backtesting-... | blog | community tier |
| https://medium.com/@fraidoonomarzai99/deployment-evaluation-strategies-in-mlops-c208585aa3bd | blog | community tier |
| https://metricgate.com/blogs/shadow-deployment-vs-canary/ | blog | community tier |
| https://dagshub.com/blog/model-deployment-types-strategies-and-best-practices/ | industry blog | duplicative of sources 4-5 |
| https://www.clarifai.com/blog/ai-model-deployment-strategies | industry blog | duplicative of sources 4-5 |
| https://blog.pickmytrade.trade/walk-forward-optimization-backtesting/ | blog | community tier |
| https://clearedge.trading/post/walk-forward-optimization-futures-strategy-validation | blog | community tier |
| https://www.luxalgo.com/blog/turning-trading-concepts-into-automated-strategies-2/ | blog | community tier |
| https://automatedtradingstrategies.substack.com/p/the-ats-portfolio-management-process | newsletter | community tier |

**URLs collected: 34** (6 read in full + 28 snippet-only).

## Recency scan (2024-2026)

Performed. Two dedicated year-scoped passes were run (see the 3-variant list
above): a last-2-year pass (2024/2025) and a current-year (2026) frontier pass.

**Result: 3 new findings that COMPLEMENT (do not supersede) the canonical
Bailey & Lopez de Prado basis.**

1. **Shu, Yu & Mulvey (2024), *J. Asset Management* 25(5)** -- the statistical
   jump model. This is the **primary source behind the "jump-model 2024" citation
   already in `strategy_selector.py:9-11`**, and the numbers check out exactly:
   the docstring's "a switch penalty cut turnover 141% -> 44% while improving
   net-of-cost Sharpe" matches the paper's S&P 500 HMM-vs-JM figures verbatim.
   The in-repo citation is CORRECT -- worth recording, since an unverified
   docstring citation is a common defect class.
2. **arXiv 2603.20319 (2026)** -- NEW and directly material: rotation strategies
   show the largest cross-engine implementation divergence (up to 3.71%), i.e.
   the *class of strategy this bridge would rotate among* is exactly the class
   where backtest numbers are least reproducible. Not available when the
   phase-47.6 selector was designed.
3. **arXiv 2512.12924v1 (2025/26) + MDPI Electronics 15(6):1334 (2026)** --
   contemporary walk-forward work that reports DSR **failing** significance and
   declines to gate on it. Useful as a live counter-example to gate optimism.

**No source found that supersedes DSR>=0.95 / PBO<=0.20 as the promotion
criterion.** The canonical basis stands.

---

## Q1 -- Re-derived line numbers + HEADLINE FINDING

### All five cited anchors RESOLVE (2026-08-06, `backend/services/autonomous_loop.py`, 3228 lines)

| Step's claim | Re-derived | Verdict |
|---|---|---|
| `:431` loads optimizer_best.json | `:431` = `best_params = load_promoted_params(bq)` | **PARTIALLY REFUTED** -- see below |
| `:433` sharpe display field | `:433` = `summary["best_params_sharpe"] = best_params.get("sharpe", "?")` | CONFIRMED |
| `:434-437` tp/sl/holding into summary | `:434-437` = `summary["strategy_params"] = {k: best_params[k] for k in ["tp_pct","sl_pct","holding_days"] if k in best_params}` | CONFIRMED |
| `:1649` strategy as LABEL ONLY | `:1649` = `current_strategy = (best_params.get("strategy", "unknown")` | CONFIRMED |
| `:1644` comment "strategy router (deferred to phase-31)" | `:1644` = `# strategy router (deferred to phase-31). Dead-man's-switch` | CONFIRMED |

### Correction 1 (minor): `:431` does NOT read the JSON file directly

`:431` calls `load_promoted_params(bq)` (defined `autonomous_loop.py:46-74`), a
**three-tier** loader added in phase-25.B3:
1. `bq.get_latest_promoted_strategy()` row with non-empty `params` -> return BQ params;
2. BQ empty -> `load_best_params()` (`:33-43`, reads `_OPTIMIZER_BEST_PATH`, `:30`);
3. BQ raises -> same fallback. Never raises (`:46-74`).

So the live registry source of truth is **BQ `promoted_strategies` first,
`optimizer_best.json` only as fallback**. A bridge design that keys off the JSON
file alone would target the fallback, not the live source. `optimizer_best.json`
is NOT referenced anywhere in `backend/services/autonomous_loop.py` except via
`_OPTIMIZER_BEST_PATH` at `:30`.

### Correction 2 (HEADLINE): the EXIT PARAMS do NOT cross over either

The step's "accurate framing" says *"the registry's EXIT PARAMS cross over; its
SELECTION LOGIC does not."* The first half is **REFUTED**.

`best_params` has exactly **five** occurrences in the whole 3228-line file
(exhaustive grep): `:431` (bind), `:432` (truthiness), `:433`, `:435`, `:1649`.
There is no sixth. And:

- `summary["strategy_params"]` (`:434`) has **ZERO readers** repo-wide. The only
  hit for that key in `backend/services/` is the WRITE at `:434`. (Other
  `strategy_params` hits -- `backtest_engine.py:238,370,583,698`,
  `quant_optimizer.py:300,713-728,1074-1078` -- are a *different* variable,
  `BacktestEngine._strategy_params`, in the offline optimizer.)
- `summary["best_params_sharpe"]` (`:433`) likewise has zero readers.
- `tp_pct` / `sl_pct` appear **nowhere else in `backend/services/`** -- the only
  occurrence is `:435` itself.

How live exits ACTUALLY get their levels (independent of the registry):
- `portfolio_manager.py:842-880` `_extract_stop_loss(...)` -- 3-tier: explicit
  `risk_assessment.risk_limits.stop_loss` (`:861`), then `stop_loss_pct`
  (`:868`), then `settings.paper_default_stop_loss_pct` (`:877`).
- `paper_trader.py:298-304` -- if no `stop_loss_price` supplied, defaults to
  `settings.paper_default_stop_loss_pct` (8.0) below entry.
- `paper_trader.py:784-793` `check_stop_losses()`; scale-out R-multiples at
  `:815-878` (`take_profit_2R` `:850`, `take_profit_3R` `:878`) are computed from
  `paper_default_stop_loss_pct` (`:815`), NOT from registry `tp_pct`.
- `portfolio_manager.py:129-133` -- the `stop_loss` SELL reason.

**Therefore: nothing from `best_params` crosses into live behaviour today.**
All three crossings (`:433`, `:434-437`, `:1649`) are telemetry-only, and
`autonomous_loop.py:1649` is the ONLY read of a strategy name in that file.
The "exit params already cross so selection is the only gap" premise is wrong.

Corroboration that these are not even *display* fields: `grep` across
`frontend/src/` for `decided_strategy` / `promoted_strateg` /
`best_params_sharpe` / `strategy_params` returns **zero hits**, and no
`backend/api/` module references `promoted_strateg` or `strategy_decision`.
The cycle `summary` IS returned to the API (`backend/api/paper_trading.py:1126`,
`:1380`, `:1456`), so the keys are reachable over HTTP but nothing renders them.

### Correction 3 (HEADLINE #2): a strategy NAME *does* drive live behaviour -- elsewhere

Found in round 11, outside the step's stated scope. **`backend/services/paper_trader.py:1421-1445`**
(phase-32.2, Kaminski-Lo Proposition 2 guard):

```python
if pos.get("stop_advanced_at_R"):
    entry_strategy = (pos.get("entry_strategy") or "").lower().strip()
    if entry_strategy in {"mean_reversion", "pairs"}:
        return (None, None)          # <-- SKIPS the HWM-trailing branch entirely
    trail_pct = float(getattr(self.settings, "paper_trailing_stop_pct", 8.0))
```

`mean_reversion` is a **STRATEGY_REGISTRY key** (`backtest_engine.py:71`). So a
strategy name, when present on a position, **suppresses the trailing stop** --
a live risk-control behaviour, not telemetry. This is exactly the
"selection-affecting consumption the step missed" the caller asked about.

**It is currently INERT, and provably so:**
- Live BQ (`financial_reports.paper_positions`, queried 2026-08-06):
  `entry_strategy` is **NULL for every row**.
- No writer exists. `entry_strategy` appears only at `paper_trader.py:1426`
  (read), `:1443` (log), `:1470` (`_POSITION_RT_FIELDS` passthrough), plus the
  migration and a `settings.py:560` comment. Nothing assigns it.
- The branch is deliberately **fail-CLOSED-conservative** (`:1421-1423`): unknown
  -> treat as momentum -> trail IS applied ("more protection" is the safe side).

**And the migration already names this as the bridge's landing point** --
`scripts/migrations/phase_32_2_add_entry_strategy.py:16-17`:
> "Going forward, `paper_trader.execute_buy` is the canonical write-site for
> this field; wiring it to read from `strategy_decisions.decided_strategy` at
> BUY time is a phase-32.x followup -- **NOT in scope for this cycle**."

**Three consequences the design MUST record:**
1. The bridge has **two** landing points, not one: `promoted_strategies` ->
   `load_promoted_params` (params), and `strategy_decisions.decided_strategy` ->
   `paper_positions.entry_strategy` (per-position behaviour).
2. **Highest-regression-risk detail in the whole step:** the moment anything
   starts populating `entry_strategy` with `mean_reversion`, trailing stops
   SILENTLY STOP FIRING on those positions. That is a live risk-control change
   wearing the costume of a metadata write. It must be flagged, flagged
   default-OFF, and given its own live_check.
3. **Criterion-3 scope gap:** criterion 3 tests only that registry label methods
   stay unreferenced from `autonomous_loop.py`. A bridge could change live
   behaviour entirely via `paper_trader.py` without ever touching
   `autonomous_loop.py` -- criterion 3 would stay green. The criterion is
   immutable and must not be edited, but the design should note the gap and the
   test SHOULD additionally assert `paper_positions.entry_strategy` has no new
   writer (belt-and-braces beyond the letter of the criterion).

**Implication for criterion 2** ("the design states the measured current
behaviour (strategy consumed as a label only) with file:line references that
resolve"): the criterion's own wording is satisfied and is *stronger* than the
step name -- `strategy` IS label-only. But the design doc must correct the
step's parenthetical claim about exit params or it will encode a false premise.

---

## Q2 -- STRATEGY_REGISTRY structure + the criterion-3 predicate

### The registry (`backend/backtest/backtest_engine.py:69-82`) -- 6 keys, 5 unique methods

| Strategy key | Label method (the VALUE) | Method def |
|---|---|---|
| `triple_barrier` | `_compute_triple_barrier_label` | `:1066` |
| `mean_reversion` | `_compute_mean_reversion_label` | `:1469` |
| `meta_label` | `_compute_triple_barrier_label` (**shared**) | `:1066` |
| `stretch_regime` | `_compute_stretch_regime_label` | `:1715` |
| `qarp` | `_compute_qarp_label` | `:1747` |
| `reversion_sigma` | `_compute_reversion_sigma_label` | `:1794` |

Derived set is **non-empty (6 keys / 5 distinct values)** -- asserted, not assumed.

Also present but **DEMOTED** out of the registry (`NON_COMPARABLE_STRATEGIES`,
`:55-67`, phase-82.16): `quality_momentum` (`_compute_quality_momentum_label`,
`:1448`) and `factor_model` (`_compute_factor_label`, `:1532`). The methods are
kept so the demotion is reversible.

> **Config drift (dead doc):** `.claude/rules/backend-backtest.md` says
> *"5 strategies in STRATEGY_REGISTRY"* and its "5 Strategies" table still lists
> `quality_momentum` + `factor_model` as members. Both statements are now false.
> Not in 82.6's scope, but flag it: a bridge design that reads that rules file
> would select a demoted strategy.

### What a "label method" actually does

It is **not** a selection function. Signature is uniformly
`(self, ticker: str, entry_date: str) -> int | None` -- it returns a **training
label** (+1/0/-1) for one (ticker, date) pair, used to build the supervised
training set for the `GradientBoostingClassifier`. Dispatch is
`backtest_engine.py:1440-1444`:

```python
def _compute_label(self, ticker: str, entry_date: str) -> int | None:
    method_name = STRATEGY_REGISTRY.get(self.strategy, "_compute_triple_barrier_label")
    method = getattr(self, method_name)
    return method(ticker, entry_date)
```

This matters for the design: "selecting a strategy" does **not** mean calling a
label method in the live cycle. Label methods are offline-training-only and can
never legitimately appear in `autonomous_loop.py`. A strategy choice changes
which *trained model / params* the live cycle uses, not which label fn it calls.

Also note the silent-coercion path: `resolve_strategy()` (`:84-121`) coerces any
unregistered name to `triple_barrier` and returns `was_demoted`; the `getattr`
default at `:1442` does the same coercion *without* the warning.

### Exact structural predicate for criterion 3

Criterion 3: *"a test asserts the live selection path is UNCHANGED by this step:
STRATEGY_REGISTRY label methods remain unreferenced from
backend/services/autonomous_loop.py"*. Recommended predicate:

1. `from backend.backtest.backtest_engine import STRATEGY_REGISTRY` in the TEST
   (derive, never hardcode the names -- so a 7th strategy is auto-covered).
2. `assert STRATEGY_REGISTRY` and `assert len(set(STRATEGY_REGISTRY.values())) >= 5`
   -- guard against a vacuous sweep over an empty/shrunken registry.
3. Read `backend/services/autonomous_loop.py` source ONCE; assert **no value**
   (the `_compute_*_label` strings) occurs in it, and also sweep
   `NON_COMPARABLE_STRATEGIES.values()` (a demoted method is still a label method).
4. Assert the module does **not** import `backtest_engine` at all:
   parse with `ast`, walk `ast.Import` / `ast.ImportFrom`, assert no module name
   containing `backtest_engine` and no imported alias named `STRATEGY_REGISTRY`.
   Include function-local imports -- `autonomous_loop.py` uses them heavily
   (e.g. `:395`, `:454`, `:1313`), so a module-header-only check is a false negative.

### False-positive / false-negative shapes (heed the 82.59 lesson)

- **Low FP risk here, unlike 82.59.** The 82.59 gate's ~10 false positives came
  from sweeping *common verbs*. These values are `_compute_<name>_label` --
  underscore-prefixed, unique, no English-verb collision. A literal substring
  sweep on the VALUES is safe. Do **not** instead sweep the KEYS: `meta_label`,
  `qarp`, `stretch_regime` are fine, but a key sweep invites collisions and,
  worse, `"strategy"` / `"triple_barrier"` DO legitimately appear in
  `autonomous_loop.py` (`:1649`, and `triple_barrier` is the live params value)
  -- a key-based test would fail on correct code.
- **FN-1 (transitive import).** A source-text grep of `autonomous_loop.py` alone
  misses `autonomous_loop -> X -> backtest_engine`. Real risk: `strategy_selector`
  imports `gate`, and `rotation_runner.py:54` DOES
  `from backend.backtest.backtest_engine import STRATEGY_REGISTRY`. If a future
  edit imports `rotation_runner` into the loop, a file-local test stays green.
  Mitigation: also assert the import-closure, or at minimum assert
  `autonomous_loop` imports nothing from `backend.autoresearch`.
- **FN-2 (computed getattr).** `getattr(engine, "_compute_" + name + "_label")`
  evades any literal sweep. Accept as a known limit and state it in the design.
- **FP-trap in the assertion itself:** `len(set(values)) == len(keys)` is FALSE
  (5 vs 6, because `meta_label` shares `triple_barrier`'s method). A test written
  that way fails on correct code. Use `>=` against a floor, not equality.
- The test must **fail** if someone wires selection -- so mutate it: temporarily
  inject a `_compute_qarp_label` reference into a copy of the source string and
  assert the predicate flags it. A guard that cannot fail does not count.

---

## Q3 -- The insertion point(s)

Cycle stages in `run_daily_cycle()` (`backend/services/autonomous_loop.py`),
re-derived:

| Stage | Anchor |
|---|---|
| params load | `:429-437` |
| Step 1 Screen | `:447-449` (`screen_universe` via `backend/tools/screener.py`) |
| Step 2+ Analyze | overlays `:451-509+` |
| Kill-switch enforce | `:1313-1322` |
| Step 9 Learn | `:1601-1607` |
| Step 10 MetaCoordinator | `:1609-1635` |
| Step 10.5 heartbeat | `:1637-1668` |
| Done / summary | `:1670-1689` |

**There is not ONE insertion point -- there are three, and only one is real.**

1. **The params seam (`:431`) -- THE insertion point.** `best_params =
   load_promoted_params(bq)`. This is the single place where an
   externally-selected strategy already enters the process. It is where a
   selection *would* have to take effect. But today it is **inert**: the loaded
   dict reaches only `:433`, `:435`, `:1649` (all telemetry). So the bridge is
   not "add a call at :431" -- `:431` is already correct. **The missing half is
   downstream: nothing consumes `best_params` into behaviour.**
2. **The behavioural seam (where it must land to change trades).** For a
   selection to change *which trades are taken*, it must reach either
   (a) `screen_universe` / `rank_candidates` (candidate set), or
   (b) the decide/size path in `portfolio_manager.py` (`:215` `_extract_stop_loss`,
   `:269`, `:447`, `:793`) and `paper_trader.py:238-304` (exit levels).
   Corroborated by three independent in-repo statements:
   `strategy_candidate_producer.py:36-39`, `strategy_registry.py:40`, and
   `strategy_backtest_adapter.py:43` -- all say *"flipping a promoted_strategies
   row alone changes only the heartbeat, not live orders"* because the live path
   is `settings.paper_*`-driven.
3. **The heartbeat (`:1649`) -- NOT an insertion point.** Writing a different
   `decided_strategy` there changes only the audit row.

**Design consequence:** the bridge's real work is a *parameter-application*
layer between `:431` and the decide path, i.e. mapping registry params onto the
`settings.paper_*` values that `portfolio_manager` / `paper_trader` actually
read. `rotation_runner.py:37` names exactly this and marks it DEFERRED:
*"The DEPLOYMENT bridge: params -> settings.paper_* + a promoted_strategies MERGE."*

---

## Q4 -- The promotion gates that ACTUALLY exist

Do not invent a gate. Four distinct gate objects exist; two are live.

| Gate | Location | Thresholds | Status |
|---|---|---|---|
| `PromotionGate` | `backend/autoresearch/gate.py:21-30` | `min_dsr=0.95`, `max_pbo=0.20`, `min_pbo_trials=10` | **LIVE** -- the one weekly promotion uses |
| `evaluate_stage` (staged rollout) | `backend/services/promotion_gate.py:34-63` | `STAGES=[0.05,0.25,1.0]`, `MIN_LIVE_DAYS=[14,30]`, `PSR_PARITY=0.0`, `PBO_CEILING=0.5` | **REACHABLE but off-cycle** -- callers are `scripts/risk/promotion_gate.py:159` + `scripts/audit/promotion_gate_audit.py:48-70` (CLI/audit only, not the daily loop) |
| `select_best_strategy` | `backend/autoresearch/strategy_selector.py:60-131` | composes `PromotionGate` + `min_improvement=0.01` hysteresis | **DARK** -- zero production callers (only `tests/autoresearch/test_strategy_selector.py`) |
| Auto-demote | `backend/autoresearch/rollback.py:33+` | `DD_TRIGGER` (from `promoter.py`) | LIVE-capable, HITL-free by design |

**Threshold reconciliation is already written down** -- `backend/backtest/analytics.py:184-201`:
`PBO_CEILING_LIVE = 0.20` ("backend/autoresearch/gate.py, enforced weekly"),
`PBO_CEILING_CANONICAL = 0.50` (Bailey/Borwein/Lopez de Prado/Zhu, SSRN 2326253),
`PBO_MIN_TRIALS_GATE_GRADE = 10`. The caller's brief said the 0.50 lives at
"promotion_gate.py:37" -- **re-derived: correct line, but the file is
`backend/services/promotion_gate.py:37`, NOT `backend/autoresearch/promotion_gate.py`
(which does not exist).**

Note `analytics.py:189-190` calls `evaluate_promotion` dead code that "defaults a
missing pbo to 0.0, which PASSES". That is accurate for `evaluate_promotion`, but
`evaluate_stage` in the same file is NOT dead (2 script callers). The design must
not treat all of `services/promotion_gate.py` as dead.

**Prerequisite dependency (state precisely):**
- **82.23 (PBO term never computed)** -- `PromotionGate` is fail-closed on a
  missing `pbo` (`gate.py:36-39` -> `missing_dsr_or_pbo`). So until 82.23 lands,
  **every** candidate is rejected and the selector can only ever return
  `no_candidate_passed_gate` (retain incumbent). The bridge is therefore
  *safe-but-inert* until 82.23. Blocking for any real selection.
- **82.26 (trial floor)** -- `min_pbo_trials=10` (`gate.py:30`) plus
  `PBO_MIN_TRIALS_GATE_GRADE=10` mean a PBO computed from <10 independent trials
  is rejected as `pbo_trials_below_min`. Per prior memory the producer whitelists
  only 5 keys, making 10 unreachable -- so 82.26 must raise the producer's K
  before the gate can pass anything. Blocking.

Ordering: **82.23 -> 82.26 -> (82.6 design) -> any build step.** 82.6 is design-only
so it does not *require* them to be closed first, but the design MUST record them
as build-time prerequisites.

---

## Q5 -- Rollback path (derived from existing machinery only)

Five existing mechanisms; no new machinery needed.

1. **Kill switch (fastest, live).** `backend/services/kill_switch.py`;
   enforced in-cycle at `autonomous_loop.py:1313-1322`
   (`trader.check_and_enforce_kill_switch()`, `summary["kill_switch"]`,
   `summary["steps"].append("kill_switch_halted")`). Halts trading outright --
   the blunt instrument, not selection-aware.
2. **Revert the source row (targeted).** `promoted_strategies` is MERGE-upserted
   on `(strategy_id, week_iso)` (`bigquery_client.py:739-754`) and read by
   `get_latest_promoted_strategy` (`:800`). Writing the prior strategy_id back
   restores the previous selection on the next cycle -- and because
   `load_promoted_params` (`:46-74`) is fail-open, deleting/deactivating the row
   falls back to `optimizer_best.json` automatically. **That fallback IS a
   rollback path and it already exists.**
3. **Auto-demote on drawdown (no HITL).** `rollback.py::auto_demote_on_dd_breach`
   -- fires on `DD_TRIGGER`, writes 3 sinks (`handoff/demotion_audit.jsonl`,
   `handoff/logs/monthly_approval_state.json`, weekly ledger notes).
4. **Staged allocation regress.** `evaluate_stage` (`services/promotion_gate.py:40`)
   returns advance/hold/**regress**/demote over `STAGES=[0.05,0.25,1.0]` -- a
   selection can be rolled back to 5% rather than off. This is the natural
   canary/rollback ladder and it is already written.
5. **Audit trail.** `strategy_decisions` (below) records what was decided and why,
   so a rollback is reconstructable.

Plus the standing repo idiom: **default-OFF settings flag** for any behaviour
change (dark launch), which the design should adopt as the outermost switch.

---

## Q6 -- What `strategy_decisions` actually records (LIVE BQ, queried 2026-08-06)

Table `sunny-might-477607-p8.pyfinagent_data.strategy_decisions`. Row shape
written at `autonomous_loop.py:1651-1661`: `ts, cycle_id, decided_strategy,
prior_strategy, trigger, decay_signal, decay_attribution, rationale`.
Writer `bigquery_client.py:430-452` (`save_strategy_decision`).

Measured (not asserted):

| Metric | Value |
|---|---|
| Total rows | **51** |
| `decided == prior` | **50** |
| `decided != prior` | **1** |
| Distinct `decided_strategy` | 2 |
| Distinct `trigger` | 2 |
| First ts | 2026-05-16T16:06:04Z |
| Last ts | **2026-07-31T18:48:08Z** |

Breakdown: 50 rows `(triple_barrier, triple_barrier, cycle_heartbeat)`; 1 row
`(reduce_position, triple_barrier, decay_signal)`.

**Verdict on the step's claim:** *substantially TRUE but not literally true.*
Every `cycle_heartbeat` row (50/50) has `decided == prior` -- and by
construction, since `:1654-1655` assign the SAME `current_strategy` variable to
both fields. It is not an empirical coincidence; it is impossible for a heartbeat
row to differ. But the table is not 100% identical: one `decay_signal` row
(`reduce_position`) exists, so a test asserting "every row has decided == prior"
would FAIL. Phrase any such assertion as *"every `cycle_heartbeat` row"*.

**Two bonus findings:**
- `reduce_position` is **not** a member of `STRATEGY_REGISTRY` -- the column
  mixes strategy names with *actions*. The design must state the column's domain.
- **Last write 2026-07-31, today 2026-08-06 -> ~6 days stale.** The heartbeat is
  a dead-man's-switch (`:1644-1647`); no row for 6 days means either no cycle ran
  or the fail-open `except` at `:1664-1668` is swallowing writes. Out of 82.6's
  scope but worth queueing as its own defect step.

**What it would carry once the bridge exists:** `prior_strategy` = the
`strategy_id` from the previous active `promoted_strategies` row;
`decided_strategy` = `select_best_strategy(...)["selected_id"]`; `trigger` =
the selector's `reason` (`first_selection` / `incumbent_is_top` /
`dsr_improvement` / `below_min_improvement` / `no_candidate_passed_gate`);
`rationale` = `delta_dsr` + ranked list. The selector already returns exactly
these fields (`strategy_selector.py:101-111`) -- the schema and the verdict dict
line up with no new columns.

---

## Q7 -- Prior art IN THIS REPO (do not re-design)

**The bridge has already been designed.** Findings:

1. **`backend/autoresearch/strategy_selector.py` (phase-47.6) IS the selection
   logic**, already written, tested (`tests/autoresearch/test_strategy_selector.py`),
   and dark. Its docstring (`:13-17`) names the intended bridge verbatim:
   > "This is the SELECTION layer over the EXISTING promotion infra: it REUSES
   > `backend/autoresearch/gate.py::PromotionGate` ... The chosen strategy is
   > meant to flow to the live loop through the existing `promoted_strategies`
   > BQ row that `autonomous_loop.load_promoted_params` already consumes -- **no
   > new read path**."

   So the architecture question is **already decided**: selection writes a
   `promoted_strategies` row; the loop reads it at `:431`. 82.6 should ratify and
   document this, not invent an alternative.
2. **What `strategy_selector.py:26-31` explicitly DEFERRED** (the real remaining
   gap): live per-strategy DSR population via 5 quant-only walk-forward
   backtests; the weekly cron that drives selection; real-capital activation;
   effective-N (ONC) DSR clustering.
3. **`rotation_runner.py:20-42`** deferred the other half:
   *"The DEPLOYMENT bridge: params -> settings.paper_* + a promoted_strategies
   MERGE. This runner records the verdict at allocation_pct=0 ONLY."*
   Plus a DEAD-KEY warning (`:20-29`): `trailing_stop_enabled /
   trailing_trigger_pct / trailing_distance_pct / vol_barrier_multiplier` are
   written into `engine._strategy_params` but their engine readers were REVERTED
   in commit `9fbd9cd6` -- **nothing reads them today.** A bridge that maps those
   params to live behaviour would map to nothing.
4. **The phase-31 deferral** referenced at `:1644` is the *Layer-2 strategy
   router*, a different (larger) thing than the selector. The heartbeat was
   deliberately built as observability-only "WITHOUT activating the full Layer-2
   strategy router (deferred to phase-31)" (`:1643-1644`).
5. Supporting infra already built: `promoter.py` (writes `status="active"`,
   supersedes prior), `friday_promotion.py:146-167` (weekly promotion ->
   `save_promoted_strategy`), `monthly_champion_challenger.py:240-265`
   (challenger flip), `weekly_ledger.py`, `slot_accounting.py`, `meta_dsr.py`.

**Net for the contract:** 82.6 is a *documentation-and-ratification* step over
~6 already-built components, plus a precise statement of the 4 deferred gaps.
Re-designing selection from scratch would duplicate phase-47.6.

### Round-4 addition: the SOURCE of the step's MEASURED claim (and its rot)

`docs/strategy/incumbent_live_strategy_spec.md:31-36` is where the step's
wording originates:

> ":31" -- "strategy router (deferred to phase-31)."
> ":33-36" -- "**No `STRATEGY_REGISTRY` label method executes in the live path.**
> The registry's five strategies (`backend/backtest/backtest_engine.py:32`) score
> research runs only. The registry's *exit parameters* cross over **as display
> fields**; its *selection logic* does not."

Three observations:

1. **The spec doc is MORE precise than the masterplan step.** It says exit params
   cross over *"as display fields"*. The step name dropped that qualifier and
   reads "the registry's EXIT PARAMS cross over", which a builder will read as
   behavioural. The design doc must restore the qualifier (see Q1 Correction 2).
2. **Two stale anchors in the spec doc:** "five strategies" (now **six**) and
   "`backtest_engine.py:32`" (the registry is at **`:69-82`**). The same stale
   `:32` anchor is propagated in `backend/meta_evolution/archetype_library.py:4`.
   Criterion 2 demands "file:line references that resolve" -- so 82.6 must NOT
   copy these two anchors forward.
3. `docs/strategy/phase82_design_pack.md:21-26` repeats the same claim and `:429`
   carries the one-line 82.6 row ("Design the registry->live bridge -- without it
   no backtest result can ever change live behaviour"), with `:418` noting 82.6
   is P2 (an "earlier draft labelled 82.6 as P1"). **Neither doc is the bridge
   design** -- so 82.6 has genuine work; it just must build on 47.6/48.x rather
   than start over.

### Round-4 addition: completed prior-art masterplan steps

| Step | Status | What it delivered |
|---|---|---|
| 30.7 | done | "P3: MAS strategy-router production wiring audit" (the heartbeat) |
| 47.6 | done | "Dynamic strategy rotation -- per-strategy DSR selector with anti-churn" |
| 48.1 | done | config-driven seed registry + per-strategy DSR/PBO producer |
| 48.2 | done | real-engine adapter (`make_engine_backtest_fn`), K-variant PBO matrix |
| 48.3 | done | "Live rotation runner ... verdict persisted **AUDIT-ONLY (allocation_pct=0, no deploy); live bake-off + cron + deployment bridge DEFERRED**" |
| 48.4 | done | first REAL bake-off smoke on actual backtests |

48.3's own name is the cleanest statement of the remaining gap: **the deployment
bridge is the deferred piece, and 82.6 is its design step.**

**No production caller** of `rotation_runner` exists: the only importers are
`scripts/diag_all_strategies.py:20`, `scripts/diag_label_pin.py:17`,
`scripts/diag_rotation_backtest.py:17`, `scripts/run_rotation_smoke.py:29`.
Nothing in `backend/api/`, `backend/services/`, or `backend/main.py`.

**A third registry mirror exists:** `backend/meta_evolution/archetype_library.py:31`
`IMPLEMENTED_STRATEGY_IDS` -- self-described at `:42` as a "mirror of
STRATEGY_REGISTRY". Three places now enumerate strategies (registry, archetype
library, `.claude/rules/backend-backtest.md`); two are stale. The design should
name the registry as the single source of truth.

---

## Internal code inventory

| File | Lines / anchors | Role | Status |
|---|---|---|---|
| `backend/services/autonomous_loop.py` | 3228; `:30`, `:33-43`, `:46-74`, `:281`, `:429-437`, `:1313-1322`, `:1461`, `:1637-1668` | The live cycle. Loads params, writes heartbeat | LIVE; registry-inert |
| `backend/backtest/backtest_engine.py` | 1862; `:55-67`, `:69-82`, `:84-121`, `:1440-1444` | STRATEGY_REGISTRY + label dispatch | LIVE (offline research only) |
| `backend/backtest/experiments/optimizer_best.json` | 7 top keys; 23 param keys | Fallback params snapshot | LIVE fallback; **no `pbo` key**; 4 dead keys |
| `backend/services/portfolio_manager.py` | 883; `:66-73`, `:97`, `:129-133`, `:215`, `:842-880` | `decide_trades` + stop extraction | LIVE; no strategy input |
| `backend/services/paper_trader.py` | 1581; `:238-304`, `:784-793`, `:815-878`, **`:1421-1445`** | Execution + exits + trailing guard | LIVE; **strategy-name branch, inert (NULL)** |
| `backend/services/risk_overrides.py` | `:41-42`, `:57`, `:65-70`, `:134`, `:148` | File-backed live-tunable risk limits + JSONL audit | LIVE; 4-key allowlist |
| `backend/autoresearch/gate.py` | `:21-30`, `:34-62` | `PromotionGate` DSR/PBO | LIVE (weekly), fail-closed |
| `backend/autoresearch/strategy_selector.py` | 135; `:13-17`, `:26-31`, `:60-131` | The selection logic | **BUILT, DARK, zero prod callers** |
| `backend/autoresearch/rotation_runner.py` | `:20-42`, `:54`, `:73`, `:161-196` | Bake-off runner | BUILT; scripts-only callers |
| `backend/autoresearch/promoter.py` | `:37`, `:46`, `:105`, `:133-142` | Writes `promoted_strategies` | Reachable; **`:134` defaults missing pbo to 0.0** |
| `backend/autoresearch/friday_promotion.py` | `:59`, `:108`, `:146-167` | Weekly promotion | **No caller anywhere -- unscheduled** |
| `backend/services/promotion_gate.py` | `:34-37`, `:40-63`, `:125` | Staged rollout ladder | Reachable via `scripts/risk/promotion_gate.py:159` |
| `backend/autoresearch/rollback.py` | `:24`, `:33+` | Auto-demote on DD | Built |
| `backend/db/bigquery_client.py` | `:430-452`, `:739-754`, `:800` | `save_strategy_decision`, `save_promoted_strategy` (MERGE), `get_latest_promoted_strategy` | LIVE |
| `backend/backtest/analytics.py` | `:184-201` | PBO threshold reconciliation constants | LIVE doc-as-code |
| `backend/meta_evolution/archetype_library.py` | `:4`, `:31`, `:42`, `:89` | `IMPLEMENTED_STRATEGY_IDS` registry mirror | **STALE anchor (`:32`)** |
| `docs/strategy/incumbent_live_strategy_spec.md` | 279; `:31-36` | Source of the step's claim | **2 stale anchors** |
| `docs/strategy/phase82_design_pack.md` | 431; `:21-26`, `:418`, `:429` | Phase-82 plan | Repeats the claim |
| `scripts/migrations/phase_32_2_add_entry_strategy.py` | `:5`, `:9-17`, `:68-69` | `entry_strategy` column + intent | Applied; **names the bridge wire** |
| `frontend/src/**` | -- | UI | **Zero references to any of these keys** |

**20 internal files/artefacts inspected** (plus live BQ on 2 tables).

---

## Key findings (each cited)

1. **The step's own "exit params cross over" framing is wrong.** `best_params`
   has 5 references total; `summary["strategy_params"]` has zero readers; live
   exits come from `settings.paper_default_stop_loss_pct` and the Risk Judge
   (`portfolio_manager.py:842-880`, `paper_trader.py:298-304`). The source doc
   said "as display fields" (`incumbent_live_strategy_spec.md:35`) -- the
   masterplan step dropped the qualifier.
2. **A strategy name DOES gate a live risk control** -- `paper_trader.py:1427`
   skips the trailing stop for `mean_reversion`/`pairs`. Inert only because
   `paper_positions.entry_strategy` is NULL for every row (measured, BQ,
   2026-08-06). See Q1 Correction 3.
3. **The selection logic already exists and is dark** --
   `strategy_selector.py:60-131` (phase-47.6), zero production callers, and its
   docstring (`:13-17`) already specifies the bridge: write a
   `promoted_strategies` row, which `load_promoted_params` (`:46-74`) already
   reads. Do not re-design this.
4. **The incumbent cannot pass its own gate today.** `optimizer_best.json` has
   **no `pbo` key** (measured), and `PromotionGate` is fail-closed on a missing
   pbo (`gate.py:36-39`). So 82.23 is a hard build-time prerequisite; the bridge
   is safe-but-inert until it lands. `min_pbo_trials=10` (`gate.py:30`) makes
   82.26 the second prerequisite.
5. **`strategy_decisions` is a dead record by construction** -- `:1654-1655`
   assign the same variable to `decided_strategy` and `prior_strategy`. Measured:
   50/50 heartbeat rows identical; 1 non-heartbeat row differs; last write
   2026-07-31 (~6 days stale as of 2026-08-06).
6. **Rotation is the worst-case strategy class for backtest reproducibility** --
   arXiv 2603.20319 measures up to **3.71%** cross-engine divergence on large
   rotation strategies vs **0.0000%** at zero cost, with Spearman rho=0.93 against
   cost intensity. A rotation bridge should therefore not trust a single engine's
   numbers; the paper recommends "at least two independent validators".
7. **The anti-churn hysteresis in `strategy_selector.py:9-11` is correctly
   cited** -- Shu/Yu/Mulvey (2024) measure S&P 500 turnover 141% -> 44% under a
   jump penalty, with net Sharpe 0.68 vs 0.48 buy-and-hold at 10bps one-way.
   Verified against the primary source.
8. **`promoter.py:134` defaults a missing `pbo` to 0.0**, which PASSES any
   ceiling -- the same fail-open defect `analytics.py:189` documented for the
   dead `evaluate_promotion`, but here in a reachable writer.
9. **Three registry enumerations exist and two are stale** --
   `STRATEGY_REGISTRY` (6 keys), `archetype_library.py:31`, and
   `.claude/rules/backend-backtest.md` ("5 strategies", lists two demoted names).

---

## Consensus vs debate (external)

**Consensus.** (a) Multiple-testing deflation is mandatory before promotion --
Bailey & Lopez de Prado; (b) staged exposure beats a binary cutover -- the
shadow -> canary -> full ladder is uniform across sources 4 and 5; (c) an
automated, time-bounded rollback with the prior champion kept warm is standard
(source 5: "revert to prior model in under 2 minutes"); (d) hysteresis /
switch-penalty is the correct cure for rotation whipsaw (source 3).

**Debate.** Source 6 is the dissent: a rigorous 2025/26 walk-forward framework
that **declines** multiple-testing correction ("All tests reported without
adjusting for multiple comparisons to maintain transparency") and offers **no
promotion gate at all**, preferring regime-conditional zero allocation. Read
against Bailey's Type-I asymmetry ("they would rather exclude a true strategy
than risking the addition of a false one"), the disagreement is about *where*
conservatism is applied -- in the gate, or in the allocation. pyfinagent already
has both levers (`PromotionGate` + `STAGES=[0.05,0.25,1.0]`), so the design can
sidestep the debate by using them together rather than choosing.

---

## Pitfalls (from literature, mapped)

1. **Winner's curse / regression to the mean** (source 2) -- selecting max Sharpe
   over N trials inflates it. Mitigated by DSR, but only if N is honest.
   Bailey's **Appendix 3 covers non-independent trials**, which is precisely the
   effective-N clustering `strategy_selector.py:29-31` defers; the 6 registry
   strategies are correlated, so plain N over-deflates (the SAFE direction).
2. **Holdout is not a defence** (source 2): "If we apply the holdout method
   enough times (say 20 times for a 95% confidence level), false positives are
   no longer unlikely: They are expected."
3. **Cost-model divergence dominates for rotation** (source 1) -- and pyfinagent
   has a known fee-table staleness issue, so a rotation bridge inherits it.
4. **Shadow mode has blind spots** (source 4): it cannot catch cases where the
   model "has side effects or depends on stateful interactions". A trading
   strategy is *definitionally* stateful (positions persist), so a pure shadow
   comparison will understate real divergence -- argues for the staged
   allocation ladder over shadow-only.
5. **Guardrails must be pre-committed and time-bounded** (source 5) -- a
   threshold decided after seeing the metric is not a gate.
6. **Fail-open defaults silently disarm a gate** -- an in-repo instance
   (`promoter.py:134`) rather than a literature one, but the same class.

---

## Application to pyfinagent (design guidance for the contract)

**Shape of the deliverable.** Per the `docs/design/phase-28.x-*.md` precedent, a
single doc (suggest `docs/design/phase-82.6-registry-live-bridge.md`) satisfying
criterion 1's three named elements.

**Criterion 1 element A -- the exact insertion point.** Two, stated explicitly:
- *Params channel:* `backend/services/autonomous_loop.py:431`
  (`load_promoted_params(bq)`) is ALREADY the correct read seam. The missing work
  is downstream: `decide_trades` (`portfolio_manager.py:66-73`) takes **no params
  argument** -- its only config channel is `settings` and
  `risk_overrides.get_effective(...)`. So the bridge must map registry params
  onto that channel. `risk_overrides` (`:57` `ALLOWED_KEYS`, 4 numeric keys) is
  the proven live-tunable, file-backed, JSONL-audited pattern but **cannot carry
  a string strategy name without extending `BOUNDS`**.
- *Per-position channel:* `paper_trader.execute_buy` writing
  `paper_positions.entry_strategy`, consumed at `paper_trader.py:1427`. Already
  named as the intended wire by `phase_32_2_add_entry_strategy.py:16-17`.

**Criterion 1 element B -- the promotion gate.** Compose only what exists:
`PromotionGate` (DSR>=0.95, PBO<=0.20, `min_pbo_trials`=10) -> `select_best_strategy`
(hysteresis `min_improvement`=0.01) -> `evaluate_stage` staged allocation
(`STAGES=[0.05,0.25,1.0]`, `MIN_LIVE_DAYS=[14,30]`, `PSR_PARITY=0.0`). Record
82.23 and 82.26 as hard build-time prerequisites with the reason (fail-closed on
missing pbo; unreachable trial floor). Do not restate 0.50 as a live ceiling --
`analytics.py:195-196` already fixes 0.20 as live and 0.50 as canonical-literature.

**Criterion 1 element C -- the rollback path.** Derive, do not invent: (1)
kill switch (`autonomous_loop.py:1313-1322`); (2) revert / deactivate the
`promoted_strategies` row -- `load_promoted_params` fail-open fallback to
`optimizer_best.json` IS a rollback; (3) `rollback.py::auto_demote_on_dd_breach`;
(4) `evaluate_stage` regress-to-5%; (5) `strategy_decisions` audit trail. Plus a
default-OFF settings flag as the outermost switch. Borrow the *form* of source
5's guardrails (pre-committed thresholds + bounded time window + fast revert),
not its latency numbers.

**Criterion 2.** State the measured behaviour with the anchors re-derived here,
and explicitly correct the "exit params cross over" wording plus the two stale
anchors (`five strategies`, `backtest_engine.py:32`).

**Criterion 3.** Use the structural predicate in Q2. Watch the specific traps:
`meta_label` shares `triple_barrier`'s method (so `len(set(values)) != len(keys)`);
`autonomous_loop.py` legitimately imports `backend.backtest.universe_lists`
(`:552`) and `backend.backtest.markets` (`:563`), so a blanket "no
`backend.backtest` import" assertion is a FALSE POSITIVE -- forbid
`backtest_engine` specifically; and `perf_metrics.py:151` mentions
`backtest_engine` in a COMMENT, so a package-wide text sweep needs comment
handling. Mutate the guard to prove it can fail.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (**6**; source 2 via the documented pdfplumber chain after a binary-PDF response)
- [x] 10+ unique URLs total (**34**)
- [x] Recency scan (last 2 years) performed + reported (3 findings; canonical basis stands)
- [x] Full papers / pages read (not abstracts) for the read-in-full set (the 2603.20319 abstract fetch was discarded and re-fetched as HTML full text)
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module in scope (+ `risk_overrides`, `promoter`, `archetype_library`, migrations, frontend, live BQ)
- [x] Contradictions / consensus noted (source 6 tagged ADVERSARIAL)
- [x] All claims cited per-claim
- [~] **Gap:** the SSRN "Build the Judge Before the Strategy" paper (403) is the
  single most on-topic external source and could not be retrieved; flagged for
  manual pull. Does not block the gate (6/5 floor met without it).

### Adaptive coverage log (audit-class, K=2)

| Round | Focus | New material findings? |
|---|---|---|
| 1 | Step verbatim + anchor re-derivation | YES |
| 2 | `best_params` trace, live exit mechanism | YES (headline #1) |
| 3 | Registry, selector, gates, BQ, external fetches | YES |
| 4 | docs/ + masterplan prior art | YES (source doc + 47.6/48.x stack) |
| 5 | frontend/API surface, decide path, test file | YES (import FP trap) |
| 6 | `decide_trades` signature, summary consumers | YES (`risk_overrides` seam) |
| 7 | `risk_overrides` allowlist, optimizer_best shape | YES (**no `pbo` key**) |
| 8 | services-wide registry check, scheduler, BOUNDS | **DRY** |
| 9 | existing tests, archived designs, PBO in schema | YES (`promoter.py:134`) |
| 10 | promoter reachability, cron, PromotionGate callers | YES (minor) |
| 11 | Slack-bot gate caller + strategy-name catch-all | YES (**headline #2**) |
| 12 | `entry_strategy` verification + live BQ | YES |
| 13 | exhaustive strategy-name branch sweep | **DRY** |

`rounds=13`, `dry_rounds=2`, `K_required=2`, `new_findings_last_round=0` ->
`coverage.dry = true`.

---

## Queued discovered defects (out of 82.6 scope -- file as their own steps)

1. `strategy_decisions` heartbeat last wrote **2026-07-31** (~6 days stale); the
   dead-man's-switch is not alarming (`autonomous_loop.py:1664-1668` swallows).
2. `promoter.py:134` defaults a missing `pbo` to `0.0` -> passes any ceiling.
3. Three registry enumerations, two stale (`archetype_library.py:31`,
   `.claude/rules/backend-backtest.md`); plus stale anchors
   `backtest_engine.py:32` and "five strategies" in
   `docs/strategy/incumbent_live_strategy_spec.md:34`.
4. `optimizer_best.json` carries 4 params nothing reads
   (`trailing_stop_enabled`, `trailing_trigger_pct`, `trailing_distance_pct`,
   `target_annual_vol`) -- readers reverted in commit `9fbd9cd6`
   (`rotation_runner.py:20-29`).
5. `run_friday_promotion` has **no caller anywhere** -- the weekly promotion
   path is unscheduled.

---

## JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 28,
  "urls_collected": 34,
  "recency_scan_performed": true,
  "internal_files_inspected": 20,
  "coverage": {
    "audit_class": true,
    "rounds": 13,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "summary": "Both of the step's premises are wrong. (1) The registry's exit params do NOT cross into live behaviour: best_params has 5 refs, summary['strategy_params'] has zero readers, and live exits come from settings.paper_default_stop_loss_pct + the Risk Judge. (2) A strategy NAME does gate a live risk control -- paper_trader.py:1427 skips the trailing stop for mean_reversion/pairs -- inert only because paper_positions.entry_strategy is NULL for every row. The selection logic already exists and is dark (strategy_selector.py, phase-47.6) and its docstring already specifies the bridge; 48.3 named the deployment bridge as the deferred piece. optimizer_best.json has no pbo key and PromotionGate is fail-closed, so 82.23 and 82.26 are hard prerequisites. All five cited anchors resolve; the source doc's 'five strategies' and 'backtest_engine.py:32' do not.",
  "brief_path": "handoff/current/research_brief_82.6.md",
  "gate_passed": true
}
```
