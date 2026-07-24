# Research Brief — phase-73.5 "D2e JUDGED PILOTS" (tier: simple)

Status: IN PROGRESS (write-first; incremental). Researcher = Layer-3.
Goal: VERIFY the three pilot verdicts from 73.0's frontier map still hold,
detail the ONE build (champion-bridge, BUILD-dark), draft the un-freeze
token + validation plan. 5-source floor still applies (may be depth re-reads).

## The three judged pilots (from task + frontier_map_73.md)
1. CHAMPION-BRIDGE — BUILD-dark (per #5 ADOPT-build-dark). best_params ->
   settings.paper_* -> decide_trades, default-OFF flag, 69.3 byte-identical
   pattern. Reads NO historical_macro (freeze-safe).
2. NEWS-RAG WEIGHTING — DEFER behind #2 (PiT-RAG mechanism, needs
   per-evidence-source attribution + residual-return labels).
3. FACTOR-MINING — DEFER behind #3; smallest honest step = OOS rank-IC/ICIR
   gate, NOT a miner; heavy frameworks REJECTED (cost/infra).

(sections filled incrementally below)

---

## INTERNAL LEG (main) — champion-bridge BUILD scope, code-verified

### Verified gap (2 files, verbatim, still holds 2026-07-18)
- `strategy_registry.py:37-41` (DEFERRED note): "the deployment switch + the
  params->settings.paper_* bridge: per the deploy audit, `best_params` is NOT
  threaded into `decide_trades`/`paper_trader`; live risk/sizing/turnover is
  driven by `settings.paper_*`. Flipping a `promoted_strategies` row alone
  changes only the heartbeat, not live orders."
- `strategy_backtest_adapter.py:42-44` (same, verbatim): "the deployment
  params->settings.paper_* bridge (best_params is NOT threaded into
  decide_trades -- flipping a promoted_strategies row alone changes only the
  heartbeat, not live orders)".
- Heartbeat-only load: `autonomous_loop.py:402` `best_params =
  load_promoted_params(bq)` -> `:404-408` writes ONLY `summary["best_params_sharpe"]`
  + `summary["strategy_params"] = {tp_pct, sl_pct, holding_days}` (display keys).
  NEVER reaches the trade path.
- `decide_trades(...)` signature (`portfolio_manager.py:66-74`) takes
  `settings: Settings` and NO `best_params` arg. Call site
  `autonomous_loop.py:1406-1414` passes `settings=settings` only. CONFIRMED:
  the champion params never enter the order decision.

### DECISIVE refinement (new this gate): the live exit model DIVERGES from the backtest params
The frontier map framed the bridge as "pure params->settings plumbing." It IS
plumbing, but the honest mapped surface is MUCH narrower than the full
best_params dict — most keys have NO live consumer. Verified by exhaustive grep:
- `backtest_tp_pct` / `backtest_sl_pct` / `backtest_holding_days` /
  `backtest_max_positions` (settings.py:247-254): **ZERO** reads in
  `backend/services/` or `backend/agents/` (grep clean). These settings fields
  are backtest-engine-only.
- The LIVE exit model is structurally different from the triple-barrier
  tp/sl/time backtest: it uses `paper_default_stop_loss_pct` (8%, settings.py:552)
  as the **R unit** driving (a) the synthesized default stop
  (paper_trader.py:165/798), (b) the **2R/3R scale-out take-profit ladder**
  (paper_trader.py:659-740, reasons `take_profit_2R`/`take_profit_3R`), and (c)
  the stop threshold (paper_trader.py:1180); plus `paper_trailing_stop_pct` (8%,
  settings.py:546) for the HWM trailing stop after the +1R ratchet
  (paper_trader.py:1161). There is **no live `tp_pct`, no live fixed-`holding_days`
  time-barrier exit, and no live `target_annual_vol` vol-targeting sizer**
  (grep for target_annual_vol/vol_target across services = EMPTY).
- `holding_days` in the live path (paper_trader.py:454/478/500,
  autonomous_loop.py:2986) is an OUTPUT/observability field (days-held on the
  trade+reflection record), never an exit input.

### best_params -> settings.paper_* mapping table (only clean, live-consumed keys bridge)
optimizer_best.json params: market, start_date, end_date, train_window_months,
test_window_months, embargo_days, holding_days, mr_holding_days, tp_pct, sl_pct,
frac_diff_d, starting_capital, max_positions, top_n_candidates, strategy,
n_estimators, max_depth, min_samples_leaf, learning_rate, target_annual_vol,
trailing_trigger_pct, trailing_distance_pct, trailing_stop_enabled.

| best_params key | live settings field | live consumer | bridge verdict |
|---|---|---|---|
| `max_positions` | `paper_max_positions` | paper_trader.py:221, portfolio_manager.py:346 | **BRIDGE (clean, direct, live-consumed)** — the one unambiguous win. SEAM NUANCE: portfolio_manager.py:346 reads it via `risk_overrides.get_effective("paper_max_positions", settings.paper_max_positions)` — an operator runtime-override lever. The bridge must DEFER to a live operator override (operator wins), never clobber `risk_overrides`; write only the settings default. |
| `sl_pct` | `paper_default_stop_loss_pct` | paper_trader.py:165/677/798/1180 (R unit) | **SEMANTIC-CAVEAT** — moves the whole 2R/3R ladder + trailing base, not just "stop%"; behaviorally large. Bridge ONLY behind flag+validation; arguably operator-owned |
| `trailing_stop_enabled` / `trailing_distance_pct` | `paper_trailing_stop_pct` (+ on/off) | paper_trader.py:1161 | **SEMANTIC-CAVEAT** — backtest trailing distance vs live HWM-trailing distance; partial map |
| `tp_pct` | (none) | live TP is R-multiple, not fixed pct | **NEVER-BRIDGE** — no live consumer; bridging = no-op OR builds a new live TP path (scope creep) |
| `holding_days` | (none) | live is signal+stop driven, no time-barrier | **NEVER-BRIDGE** — no live consumer |
| `target_annual_vol` | (none) | no live vol-targeting sizer | **NEVER-BRIDGE** — no live consumer |
| `strategy` (triple_barrier/mean_reversion/...) | (none — GBM training label) | live orders come from the 20-agent pipeline, not the GBM strategy | **NEVER-BRIDGE** — GBM-training construct, not a live order selector |
| start_date/end_date/train_window_months/test_window_months/embargo_days/frac_diff_d/n_estimators/max_depth/min_samples_leaf/learning_rate/top_n_candidates/market/starting_capital | (backtest-only) | BacktestEngine only | **NEVER-BRIDGE** — backtest/training-only |

### NEVER-BRIDGE list (risk/cap params the OPERATOR owns — not in best_params, flagged)
These live risk levers must never be auto-set from a champion row (operator
tokens own them; several have runtime override via `risk_overrides.get_effective`):
`paper_min_cash_reserve_pct` (portfolio_manager.py:97), per-sector NAV-pct cap,
FF3 factor cap, the $50 position floor, `paper_daily_loss_limit_pct` +
`paper_trailing_dd_limit_pct` (kill-switch, settings.py:527-529), position-count
cap, price-tolerance gate. The bridge writes ONLY the mapped strategy knobs
above; every cap/kill-switch clips strictly downstream and stays
non-bypassable (the phase-73.3.3 sizing-seam guarantee generalizes).

### Champion-bridge BUILD scope (executor-tagged, DARK)
- **Flag**: `paper_champion_bridge_enabled: bool = False` (settings.py; phase-69.3
  dark-flag pattern — OFF => live orders byte-identical, best_params stays
  heartbeat-only exactly as today).
- **Seam**: a pure `apply_champion_params(settings, best_params) -> Settings`
  (or an overlay dict decide_trades/paper_trader consult) that, WHEN ON, copies
  ONLY `{max_positions -> paper_max_positions}` in v1 (the one clean key), with
  `{sl_pct, trailing_*}` as explicit opt-in sub-flags given their semantic
  caveats. Reads NO historical_macro (freeze-safe — what the freeze gates is the
  optimizer/BacktestEngine that PRODUCES+VALIDATES best_params, not this consumer).
- **Validation-at-flip**: the un-freeze token (below) — a fresh
  DSR/PBO/CPCV/net-of-cost bakeoff proving champion beats incumbent OOS and
  clears all costs incl. tokens.
- **$0**, no deps, no metered spend. `promoted_strategies` row shape
  (bigquery_client.py:706-765) already carries `params` (PARSE_JSON), `dsr`,
  `pbo`, `status`, `allocation_pct` — the bridge reads `row['params']` via the
  existing `load_promoted_params(bq)`; no new BQ surface.

### #6/#7 DEFER premises re-verified against 73.1/73.2/73.3 designs
- **#6 news-RAG weighting — DEFER HOLDS; readiness advanced one notch.** The
  73.2 learn-loop design (b_learn_loop_v2.md component #5) ADDS
  `evidence_source_family STRING` to `agent_memories` (the additive nullable
  migration, populated from `enrichment_signals`) — this is exactly the
  per-evidence-source-family attribution axis PiT-RAG needs (frontier NAMED
  CHECK 1). So one of #6's three prerequisites (source-attribution field) moves
  from "net-new plumbing" to "designed into 73.2.2". BUT the field is only the
  substrate; #6 still owns the Beta-Bernoulli source-reliability table + the
  retrieval rerank-by-source-reliability on top. And the OTHER TWO prerequisites
  are unchanged: (a) #2 must RUN GREEN >=1 quarter — the loop is still DARK
  behind `paper_learn_loop_enabled=False` AND the DC1 crash-fix (73.2.1) is not
  yet built; (b) true market-model residual returns (eps_h = R_h -
  (alpha+beta*R_market)) still don't exist — the loop uses realized_pnl_pct.
  Flip bar MOVES (attribution now on a concrete build path) but does not CLEAR.
- **#7 factor-mining — DEFER HOLDS; bar unchanged by 73.1.** #7's premise is "no
  positive-EV factor-mining before #3 (clean backtests) exists; smallest honest
  step = an OOS rank-IC/ICIR gate bolted onto DSR>=0.95/PBO<=0.20, sequenced
  behind #3." The 73.1 design does NOT clear that: (3a) the price-leak on the
  quant-GBM walk-forward was already remediated in 69.2 (purge+embargo), but
  73.1's genuinely-open components (post-cutoff eval harness 73.1.2 + the
  FactFin-PC counterfactual audit 73.1.3 [METERED pilot] + CPCV wiring 73.1.4)
  are DESIGNED, not built/validated. Mining factors against a scorer whose
  LLM-side leakage guards (3b) are still un-built would industrialize
  overfitting (NEGATIVE EV — a miner searches until something scores). Heavy
  frameworks stay REJECTED (QuantaAlpha ~$45-450/run + Qlib + CN universe;
  QuantEvolve $112-450/run + 80B/160GB VRAM — both break $0-metered/local).
  Flip bar unchanged: build the OOS rank-IC/ICIR eval GATE first, AFTER 73.1's
  LLM-side guards are built+validated, THEN (optionally) an LLM emitting <=3
  candidate factors with AlphaAgent's originality+decay regularizer.

---

## EXTERNAL LEG — read in full (>=5 floor; 6 fetched)

### Read-in-full table (counts toward gate)
| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | ar5iv.labs.arxiv.org/html/2402.03755 (QuantAgent) | 2026-07-18 | preprint | ar5iv HTML | Self-improve outer loop is a FIXED-K loop (Alg 2) with **NO explicit stopping rule, NO champion-vs-challenger promotion gate, NO overfitting warning**; cost O(KT²H) §4.2; Thm 4.6 sublinear Bayesian regret is theoretical, not a practical halt. Confirms our DEFER of the self-evolution branch: QuantAgent auto-adopts refinements without a validation gate — exactly the industrialized-overfitting risk our un-freeze token guards against. |
| 2 | arxiv.org/html/2510.05533 (The New Quant) | 2026-07-18 | preprint survey | arXiv HTML | §7.1 "strict publication cutoffs per evaluation fold" + "rationales that cite evidence published before the decision timestamp" + embargo. §7.5 "amortized compute cost per basis point of excess return". §7.10 "report turnover, capacity, and the effect of transaction costs on net performance"; **"compare against strong trend and factor baselines to quantify incremental value"** (= champion-vs-incumbent OOS). Directly grounds the un-freeze validation plan. |
| 3 | arxiv.org/html/2510.07920 (Profit Mirage) | 2026-07-18 | preprint | arXiv HTML | Pre/post-cutoff decay 50.18-71.85% TR, 51.48-62.23% Sharpe (§2.1/Fig 1); FactFin PC/CI/IDS leakage metrics (Eq 4-6, Alg 1); incumbent showing >50% Sharpe decay = leakage signal, champion should show <20%. Mechanism-only (our leakage-skepticism rule). Grounds the "prove champion beats incumbent OOS without leakage" leg. |
| 4 | docs.databricks.com/aws/en/machine-learning/mlops/mlops-workflow | 2026-07-18 | official docs | HTML | Champion-challenger promotion: **"confirm that it performs at least as well as the current production model"** via **"an offline comparison [that] evaluates both models against a held-out data set"**; only if challenger performs better does the alias flip. The exact dark-until-validated pattern for the bridge flag. |
| 5 | wallaroo.ai/validating-ml-models-...-shadow-deployments | 2026-07-18 | industry | HTML | Shadow mode: all models get all data, all inferences logged, **only champion output is used**; challenger accumulates real data for comparison before a "hot swap" to champion. Maps to bridge-OFF (best_params heartbeat-only = shadow) → validate → flip. |
| 6 | arxiv.org/html/2512.22305 (PDx, Dec 2025) | 2026-07-18 | preprint | arXiv HTML | Financial-services (credit) champion-challenger: challenger evaluated on an **out-of-time-validation (OTV) set** vs champion on identical windows; promote iff **S† > S⋆**; enforced time-gaps between cohorts (leakage guard); human oversight on promotion; champion persists if challenger fails OTV. 2025 recency + cross-domain corroboration of the OOS-beat-incumbent gate. |

### Snippet-only (context; year-less canonical prior-art + does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| snowflake.com/.../ml-champion-challenger-model-deployment | vendor doc | canonical prior-art (auto-retrain weekly, replace-if-outperforms); snippet sufficient |
| datarobot.com/blog/introducing-mlops-champion-challenger-models | vendor blog | shadow-mode challenger canonical; snippet |
| docs.datarobot.com/.../challengers.html | vendor doc | "replay stored predictions ... compare metrics" |
| support.minitab.com/.../champion-challenger | vendor doc | single-champion + up-to-2-challenger pattern |
| spatialedge.ai/.../champion-challenger-...-financial-services | case study | finance champion-challenger, snippet |
| medium.com/.../mlops-2025 (Xin Cheng) | blog | MLOps-2025 capability list (shadow/canary/blue-green) |
| mljar.com/ai-prompts/mlops/.../prompt-shadow-mode | blog | shadow-mode definition |
| metricgate.com/blogs/shadow-deployment-vs-canary | blog | shadow vs canary distinction |

### Query variants run (3-variant discipline)
1. Frontier: "champion challenger model deployment shadow mode trading strategy promotion **2026**"
2. Last-2-year: "shadow deployment challenger promotion out-of-sample validation MLOps **2025**"
3. Year-less canonical: "champion challenger machine learning model deployment" (surfaced Snowflake/DataRobot/Minitab prior-art → snippet-only table)

### Recency scan (last 2 years, 2024-2026) — PERFORMED
New findings in-window that complement (not supersede) the canonical map: (a)
**PDx arXiv:2512.22305 (Dec 2025)** — a fresh financial-services champion-challenger
that independently ratifies the OTV-hold-out + "S†>S⋆" + human-gate + time-gap
promotion discipline the un-freeze plan encodes. (b) **The New Quant §7.5/7.10
(Oct 2025)** — net-of-cost + cost-per-bp reporting is now an explicit field
minimum-standard, corroborating 73.4.1. (c) MLOps-2025 practitioner consensus
(Databricks/Wallaroo/DataRobot) converges on shadow→held-out-compare→flip, which
is precisely the phase-69.3 dark-flag pattern the bridge reuses. NO in-window
source SUPERSEDES the DEFER verdicts on #6/#7 or the BUILD-dark on champion-bridge;
they REINFORCE them. QuantAgent (2024) remains the canonical self-improve-scope
reference and its NO-stopping-rule/NO-gate gap is the standing argument for our
DEFER of self-evolution.

### Consensus vs debate
- CONSENSUS (5 of 6 sources): a challenger/champion strategy is promoted ONLY
  after beating the incumbent on a held-out / out-of-time set, on identical
  windows, with the challenger dark (shadow / heartbeat-only) until it passes.
  Databricks, Wallaroo, PDx, The New Quant §7.10, and our own 73.3.4 A/B design
  all agree. This is the deployment-pattern spine of the champion-bridge.
- DEBATE / [ADVERSARIAL to naive self-improvement]: QuantAgent auto-adopts
  self-refinements with no promotion gate and no overfitting warning — the
  literature's self-improvement branch is exactly the thing WITHOUT the discipline
  the MLOps/champion-challenger sources mandate. This tension is the core reason
  the bridge is BUILD-dark-behind-a-token, not auto-promote.

### Pitfalls (from literature, applied)
- Profit Mirage: a champion that looks great in-sample can be 50-72% memorized;
  the OOS-vs-incumbent leg MUST run post-leakage-guard (the GBM backtest is
  quant-only so this binds the LIVE-signal path, not the walk-forward).
- The New Quant §7.10: a P&L-only win is insufficient — must be net-of-cost AND
  beat trend/factor baselines. Matches our charter "never maximize a single raw
  metric".
- QuantAgent §4.2: unbounded self-refine is O(KT²H) — defer many-round loops
  until #4's cost objective meters them.

---

## UN-FREEZE TOKEN (champion-bridge live-flip prerequisite)

### Context
`historical_macro` freeze is a DOCTRINAL operator boundary (masterplan phase-69/70
boundaries "historical_macro stays frozen"; no code-level freeze flag exists —
grep-confirmed; enforced via the operator-token system + away-ops-rules.md). The
freeze gates the OPTIMIZER/BacktestEngine that PRODUCES + VALIDATES `best_params`.
The champion-bridge CONSUMER reads no historical_macro, so building it dark is
freeze-safe; FLIPPING it live needs a fresh bakeoff that runs the validation
machinery, hence the token. This is one of the phase-69 owed operator tokens
("historical_macro un-freeze") — 73.5 gives it concrete wording + a validation plan.

### Proposed verbatim operator token
`HISTORICAL MACRO UNFREEZE: CHAMPION-VALIDATION-BATCH`

(Grammar matches the project `<KEY>: <value>` token form per
`backend/slack_bot/operator_tokens.py:10`; recorded to
`handoff/operator_tokens.jsonl` by the 62.2 handler. NOT yet in
`KNOWN_TOKEN_ENV_MAP` — register the key when 73.5's bridge step ships, mapping it
to a one-shot validation-batch authorization, NOT a standing `.env` flag.)

A SECOND, downstream token flips the bridge live only AFTER the batch passes:
`CHAMPION BRIDGE: ON` (maps to `PAPER_CHAMPION_BRIDGE_ENABLED`). The un-freeze
token authorizes RUNNING the validation, never DEPLOYING the result.

### Validation plan that runs at un-freeze (in order, with pass bars)
1. **Incumbent revalidation on purged data (69.2 machinery).** Re-run the
   incumbent (current `optimizer_best.json`, strategy=triple_barrier) through the
   walk-forward with the AFML purge (`_label_overlaps_test`
   backtest_engine.py:570-582, horizon 135d) + 5-day embargo (walk_forward.py:36)
   under the corrected gates. Establishes the incumbent's CLEAN OOS DSR/PBO
   baseline (this is the step 69.2 explicitly deferred behind this token).
2. **CPCV robustness distribution (73.1.4).** Run the champion candidate through
   the CPCV path (`cpcv_folds` gate.py:42; φ=C(N,k) paths over STORED prices;
   purge+embargo per fold) → OOS-Sharpe distribution (mean/std/worst-path/%paths
   Sharpe>0) as a robustness COMPLEMENT reported ALONGSIDE the K-variant CSCV PBO
   scalar. gate.py byte-unchanged.
3. **Champion-vs-incumbent OOS comparison.** Both on IDENTICAL OOS windows
   (Databricks "at least as well as the current production model" held-out gate;
   PDx "S†>S⋆" on the OTV set; The New Quant §7.10 "compare against strong trend
   and factor baselines"). Champion must BEAT the incumbent OOS on net
   risk-adjusted P&L.
4. **Net-of-cost DSR (73.4.1).** Compute DSR on the NET-of-cost return series
   `r_net = r_gross - tx/NAV - slippage/NAV - token_cost/NAV` via the existing
   `compute_dsr` (perf_metrics.py:518); log BOTH dsr_gross + dsr_net for the
   transition window. (The New Quant §7.5 amortized cost/bp + §7.10 net-of-cost.)
5. **Pass bars = the IMMUTABLE gates (never edited, validation runs THROUGH them):**
   DSR>=0.95 on the NET-of-cost series (gate.py:21), PBO<=0.20 promotion gate
   (0.5 = the advisory veto cap, risk_server.py:28 — two nested gates per 73.4.2),
   champion beats incumbent OOS, clears ALL costs incl. tokens. Charter:
   improvement must be net risk-adjusted OOS WITHOUT worsening DSR or max DD — a
   P&L-only win is REJECTED.

### Scope limits (what stays FROZEN even after this token)
- **One bounded validation BATCH only** — NOT a standing optimizer resume. The
  weekly rotation cron + the open-ended parameter-SEARCH/mutation loop that
  GENERATES candidates stay frozen (a separate authorization). historical_macro
  RE-FREEZES after the batch.
- **The bridge FLAG stays OFF.** `paper_champion_bridge_enabled` remains False;
  going live is the separate `CHAMPION BRIDGE: ON` token, gated on this batch
  PASSING all immutable gates. Un-freeze ≠ deploy.
- **The immutable gates themselves stay byte-untouched** (DSR>=0.95, PBO
  0.20/0.5, the go-live booleans). Validation runs through them; it never edits
  them (masterplan immutable-criteria rule).
- **Risk/cap params stay operator-owned + never bridged** — sector caps,
  kill-switch (paper_daily_loss_limit_pct / paper_trailing_dd_limit_pct), $50
  floor, paper_min_cash_reserve_pct. The bridge writes only the mapped strategy
  knobs (v1: `max_positions` only).
- **The macro-consuming LIVE regime path is untouched** — this un-freeze is about
  the backtest/optimizer validation surface, not the live regime prompt.

---

## Application to pyfinagent (external → internal mapping)
- Champion-bridge = a champion-challenger deployment where the incumbent (live
  paper book) is the champion and best_params is the challenger held in SHADOW
  (heartbeat-only, autonomous_loop.py:404-408) until the un-freeze bakeoff proves
  it beats the incumbent OOS. Databricks/Wallaroo/PDx patterns → the dark-flag +
  held-out-compare + flip sequence, reusing the phase-69.3 pattern verbatim.
- The one honest BUILD (max_positions → paper_max_positions, dark) + the
  never-bridge list is the "challenger must not silently change risk knobs the
  operator owns" discipline the MLOps sources imply and PDx's human-gate encodes.

## Research Gate Checklist
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6)
- [x] 10+ unique URLs total (14 in tables + search hits)
- [x] Recency scan (last 2 years) performed + reported (PDx Dec-2025 + New Quant/Profit Mirage Oct-2025)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim
- [x] Internal exploration covered the champion-bridge surface + #6/#7 premise substrates
- [x] Contradictions / consensus noted (self-improve-no-gate vs champion-challenger-gate)
- [x] All claims cited per-claim

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 8,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 15,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "phase-73.5 pilot verdicts VERIFIED. Champion-bridge = BUILD-dark, but with a decisive refinement: the live exit model diverges from the backtest triple-barrier, so most best_params keys have NO live consumer (grep-confirmed: tp_pct/holding_days/target_annual_vol read nowhere live; backtest_* settings unread by services). The ONE clean live-consumed bridge is max_positions->paper_max_positions; sl_pct/trailing_* are semantic-caveat; everything else is NEVER-BRIDGE (backtest-only or GBM-training). News-RAG (#6) DEFER holds — 73.2's evidence_source_family field advances one of three prerequisites but #2-green-1qtr + residual-return labels remain. Factor-mining (#7) DEFER holds — 73.1's LLM-side guards are designed-not-built, so the OOS rank-IC/ICIR gate stays sequenced behind #3. Un-freeze token drafted (HISTORICAL MACRO UNFREEZE: CHAMPION-VALIDATION-BATCH) with a 5-step validation plan (incumbent-revalidation-on-purged-data + CPCV distribution + champion-vs-incumbent-OOS + net-of-cost-DSR, pass bars = immutable gates) and scope limits (one batch, flag stays OFF, gates byte-untouched, risk/caps operator-owned). 6 sources read in full incl. QuantAgent (no self-improve stopping rule = DEFER basis), New Quant 7.10, Profit Mirage, Databricks/Wallaroo champion-challenger, PDx 2025 recency.",
  "brief_path": "handoff/current/research_brief_73.5.md",
  "gate_passed": true
}
```
