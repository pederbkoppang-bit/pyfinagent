# Research: Phase 4.2.2 Signal Accuracy Tracking (SignalsServer)

**Step:** Pure-deterministic signal-accuracy subset of Phase 4.2 Paper Trading Evaluation
**Target code:** `backend/agents/mcp_servers/signals_server.py`
**Date:** 2026-04-14
**Gate:** MANDATORY Research Gate (pyfinAgent protocol, see `.claude/context/research-gate.md`)
**Status:** PASS (17 URLs across 7 categories; 4 sources read in full)

## Scope

Implement pure-stdlib deterministic methods on `SignalsServer`:
1. Real `get_signal_history(limit, since_date)` backed by in-memory list (currently a stub).
2. `publish_signal` must append successful publishes to `self.signal_history`.
3. `track_signal_accuracy(signal_id, exit_price, exit_date)` -- record outcome, compute signed return pct + hit/miss.
4. `get_accuracy_report(group_by=None)` -- aggregate count, hit_rate, mean/median return, Wilson-CI, grouped by signal type / ticker.

Constraints: stdlib only, tolerant of missing fields, never raises, idempotent where possible.

---

## URL inventory (17 unique across 7 categories)

### Category A: Academic / peer-reviewed (5)
- https://arxiv.org/pdf/2010.08601 -- Chincoli & Boukerche, "Information Coefficient as a Performance Measure of Stock Selection Models" (arXiv, 2020)
- https://arxiv.org/html/2509.16707v1 -- "Increase Alpha: Performance and Risk of an AI-Driven Trading Framework" (arXiv, 2025)
- https://www.sciencedirect.com/science/article/pii/S2199853124001288 -- Quantitative finance + market microstructure synergy
- https://joim.com/wp-content/uploads/emember/downloads/p0543.pdf -- Ding & Martin, "Rethinking the Fundamental Law of Active Management" (JoIM)
- https://www.tandfonline.com/doi/full/10.1080/23311975.2024.2428781 -- Predictability of technical analysis using forward return (2024)

### Category B: Framework / platform docs (4)
- https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/insight-manager -- QuantConnect InsightManager docs
- https://github.com/QuantConnect/Lean/blob/master/Common/Algorithm/Framework/Alphas/Analysis/InsightManager.cs -- LEAN source
- https://pyfolio.ml4trading.io/ -- pyfolio docs (tear sheet reference)
- https://zipline-trader.readthedocs.io/en/latest/notebooks/Alphalens.html -- Alphalens factor evaluation

### Category C: Practitioner / quant firm (3)
- https://macrosynergy.com/research/how-to-measure-the-quality-of-a-trading-signal/ -- Macrosynergy: signal quality measurement (READ IN FULL)
- https://extractalpha.com/2025/07/01/top-7-trading-signals-every-quant-should-track/ -- ExtractAlpha: top signals quants track
- https://blankcapitalresearch.com/learn/grinold-fundamental-law-active-management -- Grinold's law in practice

### Category D: Regulatory (2)
- https://www.finra.org/rules-guidance/guidance/interpretations-financial-operational-rules/sea-rule-17a-4-and-related-interpretations -- FINRA SEA Rule 17a-4 interpretations
- https://www.law.cornell.edu/cfr/text/17/240.17a-4 -- 17 CFR 240.17a-4 (Cornell LII)

### Category E: Statistical methodology (2)
- https://en.wikipedia.org/wiki/Brier_score -- Brier score definition + applicability
- https://www.gabormelli.com/RKB/Wilson_Score_Interval -- Wilson Score Interval (READ IN FULL)

### Category F: Educational / knowledge base (1)
- https://www.bajajamc.com/knowledge-centre/information-coefficient -- IC explained + calculation

---

## Sources read in full (4)

1. **Macrosynergy -- "How to measure the quality of a trading signal"**
   Key takeaways: (a) hit rate = ratio of correctly-classified return directions; intuitive, gives equal weight to TP/TN. (b) IC = Pearson/Spearman correlation between signal and forward return; captures magnitude quality, needs N >= 30 to be meaningful. (c) Relationship: `IC = 2*hit_rate - 1` for equal-weighted binary calls. (d) Practical rule: IC > 0.05 is "good"; 0.02-0.10 is a meaningful persistent edge. (e) Both metrics should be reported -- they answer different questions.

2. **Wikipedia -- "Brier score"**
   Key takeaways: (a) Brier score is a strictly proper scoring rule for probabilistic predictions; range [0,1] where 0 is perfect. (b) Applicable to binary or mutually-exclusive categorical outcomes (BUY/SELL/HOLD qualifies). (c) Inappropriate for ordinal variables. (d) Requires the predictor to assign probabilities summing to 1 -- our current signals are hard BUY/SELL/HOLD calls with a confidence scalar, NOT a full probability vector, so we cannot compute a true multi-class Brier without reshaping the signal format. DEFER.

3. **Gabor Melli -- "Wilson Score Interval"**
   Key takeaways: (a) stable from n = 10; safe for small samples and proportions near 0/1. (b) Beats Wald/normal-approximation and Clopper-Pearson exact for small N. (c) Closed-form formula using only math.sqrt. (d) Widely used in A/B testing and ML evaluation -- the right default CI for trading signal hit rates on small paper-trading samples.

4. **QuantConnect InsightManager (docs + LEAN source)**
   Key takeaways: (a) InsightManager is an in-memory dict keyed by Symbol with list-of-Insight values -- a similar shape to what we need. (b) IInsightScoreFunction is the pluggable scorer; QC's default scores a signed-return magnitude. (c) Insights have a fixed expiry period; expired insights are removed from the active set but retained for analysis. (d) Evaluation happens at every timestep. For our use case (paper trading, ~1 signal per ticker per day), per-step scoring is unnecessary -- we score at exit time (track_signal_accuracy call) instead.

---

## Design decisions driven by research (contract-quotable)

**D1. Primary metric: hit rate (directional accuracy).** Simple, interpretable, computable with stdlib only, meaningful at any N. IC requires pairs + correlation compute and only becomes informative at N >= 30 (Macrosynergy, arXiv 2010.08601). We include `mean_forward_return_pct` alongside hit_rate to capture the "size of wins" dimension pyfolio tear sheets emphasize.

**D2. Secondary metric: Wilson Score Interval (95%) on hit rate.** Per Gabor Melli, Wilson is the correct CI for small-sample binomial proportions. Stable from n=10; handles proportions near 0 or 1 (where Wald gives negative or >1 bounds). Pure `math.sqrt` implementation. Reported as `hit_rate_ci_low`, `hit_rate_ci_high`.

**D3. Holding period is caller-supplied, NOT hardcoded.** `track_signal_accuracy(signal_id, exit_price, exit_date)` takes the exit the caller decides. The "canonical holding period" is a per-strategy choice; paper_trader already exposes `holding_days` config. Passing it through keeps `signals_server` invariant-free. We record `holding_days` (computed from signal.date to exit_date) for later grouping.

**D4. HOLD signals are recorded but NOT scored for hit rate.** HOLD is a no-op decision; there is no "correct direction" to measure. We append HOLD events to `signal_history` for audit and count them in totals, but exclude from the hit_rate denominator. `get_accuracy_report()` returns both `total_count` (all signals) and `scored_count` (BUY/SELL only). Precedent: QuantConnect Insight system treats a Flat insight as un-scored.

**D5. Neutral (epsilon) band: 0.20% default, configurable.** Forward returns inside [-0.20%, +0.20%] are tagged `neutral` and excluded from hit/miss counting (still counted in totals and mean_return). Below typical round-trip transaction cost (~10bps) a directional call is indistinguishable from noise.

**D6. Equal-weighted hit_rate, return-weighted mean_return.** Both reported; each answers a different question. Hit rate = "how often is the direction right"; mean_return = "what's the average P&L per signal." Pyfolio/Alphalens standard pattern. Median return also reported to flag skewed distributions.

**D7. Data structure: in-memory list + dict, both populated by publish_signal.** `self.signal_history: list[dict]` is the append-only time series. `self._signals_by_id: dict[str, dict]` is the O(1) lookup index for `track_signal_accuracy`. Both keyed off the existing `_signal_id()` sha1 prefix. When `track_signal_accuracy` updates a signal, it mutates the dict entry; the list entry is a reference to the same dict so both views stay in sync. Mirrors QuantConnect's InsightManager shape.

**D8. No durable persistence this phase.** Cross-restart retention of signal history is explicitly Phase 4.2.4 (BQ `signals_log` table + schema migration). In-memory only this session, documented in docstring. Matches the `_peak_equity` precedent from Phase 4.3.

**D9. Regulatory note.** 17 CFR 240.17a-4 requires 6-year retention for trade blotters at broker-dealers; does not directly apply to paper trading by individuals. We document the pattern the durable BQ table should follow (immutable append, sha1 signal_id as primary key) so the Phase 4.2.4 migration is straightforward.

**D10. Grouping support: 'signal_type' and 'ticker' only.** Per-sector and per-factor attribution DEFERRED -- sector requires a lookup service we don't have here, and per-factor requires signals to carry per-factor weights. Grouping via stdlib `collections.defaultdict`.

---

## Out of scope / deferred

- **Information Coefficient (IC / Pearson / Spearman)**: requires pandas/numpy AND N >= 30. Revisit Phase 4.2.4.
- **Brier score**: requires full probability vector over {BUY, SELL, HOLD}. Out of scope.
- **Per-factor attribution**: signals lack per-factor weights. Phase 3.2 follow-up.
- **Per-sector accuracy**: needs sector lookup service. Phase 4.2.2 follow-up.
- **Slack weekly report**: `slack_bot/formatters.py` work. Exposes `get_accuracy_report()` as data source.
- **Durable BQ persistence**: Phase 4.2.4.
- **Full pyfolio/Alphalens tear sheets**: pandas-dep; out of scope for stdlib-only MCP surface.

---

## Anti-leniency rules (for GENERATE + QA)

1. No pandas/numpy imports. Stdlib only (`math`, `collections`, `datetime`, `statistics`).
2. `track_signal_accuracy` must be idempotent: calling twice with same signal_id updates in place, never duplicates.
3. Never raise. All methods return structured error dicts or empty defaults.
4. No mutation of input dicts. Deepcopy on entry to each new public method.
5. HOLD signals appended to history but excluded from hit_rate denominator.
6. Wilson CI must handle n=0 (return (0.0, 0.0)) and n=1 (return degenerate interval).
7. Mean/median return computed with `statistics.mean` / `statistics.median`, not hand-rolled.
8. Preserve existing 4.1 + 4.3 public API byte-identically. Only extend.
9. `get_signal_history` return shape additively compatible with existing stub (`month`, `count`, `signals` preserved; new keys may be added).
10. Diff budget: `<350` added lines, `<60` net new logic lines beyond docstrings.
