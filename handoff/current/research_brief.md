# Research Brief — phase-60.2: Churn-engine fix (swap sentinel + re-eval/stamp mismatch + delta scale, AW-5, P0)

Tier: COMPLEX (caller-stated). Date: 2026-06-11. Agent: researcher (Layer-3 MAS, merged Explore).
Prior 60.1 brief snapshotted at handoff/archive/phase-60.1/research_brief.md.
Disclosed overrun: audit tables push past the 1500-word ceiling; prose kept tight.

## 1. Executive summary

- **Leg 1 (sentinel):** Replace conviction-0.0 with **LOCF age-capped valuation** — value an unanalyzed holding at its last persisted `final_score` from `financial_reports.analysis_results` (reader already exists: `bigquery_client.get_report(ticker)` backend/db/bigquery_client.py:303-358, `ORDER BY analysis_date DESC LIMIT 1`, SELECT * includes final_score), capped at ~7 days; **no score within cap → EXCLUDE from displacement** (never 0.0). Grounding: institutional buy alpha "declines gradually over twelve months following the original trade" (Di Mascio/Lines/Naik, *Alpha Decay*, accessed 2026-06-11) — a 1-day-old score retains ~all its information; staleness cost grows progressively, no cliff (Maven Securities, accessed 2026-06-11); don't carry indefinitely ("fixed relevance decay during structural breaks" risk, arXiv:2603.27539).
- **Leg 2 (re-eval/stamp):** Keep the BUY-time stamp (it is semantically true — a fresh analysis did exist at buy time); fix the COMPARISON: flag ON → a swap displacement requires same-cycle scores on both sides, achieved by **targeted re-eval injection** (pre-compute sectors at count-cap with candidates queued; add their weakest holdings to `reeval_tickers` same-cycle, ~1-3 lite analyses ≈ <$0.50/cycle inside the $25 cap), with LOCF (leg 1) as the $0 fallback when injection fails/budget-capped. Grounding: evaluation standard "net-of-cost returns + identical-universe paired comparison" (arXiv:2603.27539; FINSABER arXiv:2505.07078).
- **Leg 3 (delta scale):** Restore the DOCUMENTED formula — settings.py:293 already documents `max(abs(holding_score), 1.0)` while code uses `0.01` on a false "[0,1]" premise (portfolio_manager.py:526-531). Fix denominator clamp to 1.0, keep `paper_swap_min_delta_pct=25.0` UNCHANGED, document effective integer-scale semantics (bar = ceil(0.25*h) points). No new stickiness, no threshold re-tuning — formula correction only, per Boyd et al.: the bar is a net-gain-vs-cost comparison, not a retention band.
- All three legs behind ONE default-OFF flag (e.g. `paper_swap_evidence_fix`); OFF = byte-identical; ON-vs-OFF measured by a NEW decision-replay event study (Section 7) because **no existing tool replays the swap path** (Section 5.5).

## 2. External findings

### A. Stale-signal / missing-score handling
1. Alpha from institutional buys "declines gradually over twelve months following the original trade"; managers "continue to buy a stock in small increments for as long as the alpha persists" (Di Mascio, Lines & Naik, *Alpha Decay*, WP 2015/JF, https://jhfinance.web.unc.edu/wp-content/uploads/sites/12369/2016/02/Alpha-Decay.pdf, read in full via pdfplumber 2026-06-11). A day-old buy-time score is near-full-strength evidence; valuing it 0.0 contradicts the measured decay horizon by ~2 orders of magnitude.
2. Signal staleness cost is progressive, not a cliff: delayed execution of a mean-reversion signal costs on average 5.6% (US) / 9.9% (Europe) of strategy value, growing ~36bps/yr US (Maven Securities, https://www.mavensecurities.com/alpha-decay-what-does-it-look-like-and-what-does-it-mean-for-systematic-traders/, read in full 2026-06-11). Supports age-capped carry-forward over binary "fresh-or-worst".
3. "Layered temporal memory risks assuming fixed relevance decay during structural breaks" — carry-forward needs a cap/circuit-breaker, not indefinite LOCF (arXiv:2603.27539, read in full 2026-06-11).
4. Smart/partial rebalancing trades only the strongest signals and skips weak/no-signal names, cutting turnover while preserving factor premia (Smart Rebalancing, FAJ 2024, tandfonline 403 — snippet-only). Missing evidence = don't trade, not "treat as worst".

### B. Turnover/cost-aware swap thresholds
1. Canonical structure: trade fires iff estimated gain clears costs — "maximize r̂ᵀz − φtrade(z) − φhold(w+z) − γψ(w+z)" (Boyd et al., *Multi-Period Trading via Convex Optimization*, https://ar5iv.labs.arxiv.org/html/1705.00109, read in full 2026-06-11). A swap bar is a NET-IMPROVEMENT comparison; its size should reflect cost+noise, not incumbent protection.
2. LLM agents structurally overtrade: FinMem "commission ratio is five to nine times higher than FinAgent's", "excessive turnover... persistent value destruction", negative alpha in all scenarios; B&H Sharpe 0.703 vs FinAgent 0.241 on volatility selection (arXiv:2505.07078v5, KDD 2026 / FINSABER, read in full 2026-06-11). This is the source already cited at settings.py:282-284 for conservative swap defaults — keep `max_per_cycle=2` and the 25% bar's intent.
3. "Round-trip costs of 10 to 20 basis points can compound to 25 to 50 percentage points of annual drag for daily-trading systems"; only 2 of the surveyed systems model costs at all (arXiv:2603.27539). Our recorded fee is 0.1%/leg (settings.py:323) = 20bps round trip — the measured 81.4% weekly turnover sits exactly in this drag regime.
4. The classical no-trade-band literature (Constantinides-style regions; e.g. tandfonline Stochastics 2011, Springer FMPM 2022 — snippet-only) is the family 53.1 REJECTED as a tuning lever here; Section 6 draws the boundary.

### C. ON-vs-OFF replay evaluation methodology
1. "A key weakness of Market Replay is that the simulated market does not substantially adapt to or respond to the presence of the experimental strategy" (Balch et al., JPMorgan AI Research, https://www.jpmorgan.com/content/dam/jpm/cib/complex/content/technology/ai-research-publications/pdf-12.pdf, read in full via pdfplumber 2026-06-11). For PAPER trading the price-impact leg vanishes (fills don't move markets) — the residual path dependence is PORTFOLIO-STATE and DECISION-INPUT divergence, which must be disclosed per cycle, not hidden.
2. Five minimum evaluation standards: contamination control, point-in-time universe, rolling windows, **net-of-cost returns**, regime coverage (arXiv:2603.27539). The replay must hold the recorded candidate stream fixed for both arms (identical-universe discipline, FINSABER arXiv:2505.07078).
3. Sharpe-DELTA claims need the paired Ledoit-Wolf + stationary-bootstrap test — already implemented at backend/backtest/analytics.py:239-249 (`sharpe_diff_test`, n_boot=2000, Politis-Romano). At T≈5-12 daily cycles it is underpowered: report turnover/round-trips/P&L descriptively as primary; do NOT claim significant Sharpe improvement off one week (phase-52.3 methodology memory; McLean-Pontiff haircut).
4. Boyd et al. backtest reporting: portfolio value path + sensitivity ("randomly perturb the model parameters") — supports reporting the delta-threshold sensitivity (e.g. ±1 score point) rather than tuning it.

## 3. Recency scan (2024-2026)

Performed. Findings: the 2024-2026 literature CONVERGES on overtrading + cost-blindness as the dominant LLM-trading failure mode — arXiv:2505.07078v5 (KDD 2026), arXiv:2603.27539 (2026 evaluation taxonomy), arXiv:2512.02227 (Dec 2025, already the basis for our sector caps), arXiv:2507.08584 (2025, "To Trade or Not to Trade" — explicit risk-estimation before trading improves decisions), arXiv:2510.07920 (2025, Profit Mirage — leakage inflates LLM backtests), Smart Rebalancing (FAJ 2024 — turnover-prioritized partial rebalancing). No 2024-2026 finding contradicts the classical net-of-cost trading principle (Boyd et al. 2017); the new work strengthens the case for evidence-gated, cost-bounded swap rules. No new finding supersedes Ledoit-Wolf 2008 for paired Sharpe deltas.

## 4. Search queries run (3-variant discipline)

1. Year-less canonical (topic A): "stale signal handling portfolio rebalancing alpha decay last observation carried forward quantitative trading".
2. Year-less canonical (topic B): "transaction cost aware portfolio rebalancing no-trade region swap threshold net of cost alpha improvement".
3. Recent-window (topics B+C, shared): "LLM trading agents overtrading turnover transaction costs 2025 2026".
4. Year-less canonical (topic C): "counterfactual evaluation trading execution rule change replay event study path dependence backtest".

Disclosed deviation: no separate "2026"-suffixed query per topic A/B was run (tool budget); the recency obligation is covered by query 3 plus organic 2024-2026 hits in queries 1/2/4 (FAJ 2024, Springer 2024, arXiv 2510/2512/2603). Hard-blocker recency scan: satisfied and reported in Section 3.

## 5. Internal audit findings (CURRENT HEAD; audit cites were snapshot 70a8242b)

### 5.1 The sentinel (backend/services/portfolio_manager.py, 682 lines)
- `holding_lookup` built ONLY from same-cycle `holding_analyses` at :91-95; nothing persisted is consulted. Caller: autonomous_loop.py:1155-1163 (`holding_analyses=` :1158) built at :892-896 from the re-eval gather (persisted via `_persist_analysis` :875-881 when `_path` lite/full).
- Swap path gate :407-422 (`paper_swap_enabled` AND `sector_blocked` AND `max_per_sector>0`) → `_compute_swap_candidates` :439.
- **THE SENTINEL :476-483**: `analysis = holding_lookup.get(pos["ticker"], {}) or {}` → `score = analysis.get("final_score")` → `if score is None: score = 0.0` ("No fresh analysis => unknown conviction. Treat as worst").
- Weakest-holding pick: per-sector ascending sort :495-496, first non-swapped :516-521. SELL reason="swap_for_higher_conviction" :582; BUY reason="swap_buy" :591.
- 57.1 binding-REJECT gate (F-3) :186-212 sits at the candidate-BUILD chokepoint upstream of both buy loop and `sector_blocked` (:284,:319) — REJECTs never reach the swap path when ON. A 60.2 flag inside `_compute_swap_candidates` composes cleanly (disjoint regions); test both flags ON together.

### 5.2 Delta computation + threshold
- :525-532 `denom = max(abs(holding_score), 0.01)`; `delta_pct = ((cand_score - holding_score)/denom)*100.0`; gate `if delta_pct < min_delta: continue` :534.
- False-premise comment :526-531: "do NOT clamp the denominator to 1.0 -- final_score lives in [0,1]". **Lite scores are 1-10 integers** (59.3 audit AW-5, BQ-confirmed).
- `paper_swap_min_delta_pct` read :464 (fallback 25.0); settings.py:289-294 default 25.0. **INCONSISTENCY: the settings description (:293) documents `max(abs(holding_score), 1.0)`** — the code drifted from its own spec on the [0,1] premise. `paper_swap_max_per_cycle` settings.py:295-300 default 2; `paper_swap_enabled` :285-288 default True. No `.env` overrides (grep) → defaults are live.
- Effective semantics (verified): cand 7.0 vs SENTINEL 0.0 → denom 0.01 → **delta = 70,000%** (auto-clears 25% by ~3.5 orders of magnitude). cand 7.0 vs real 5.0 → 40% fires. 1 integer point fires at holding<=4 (5v4 = 25%); 2 points fire at holding<=8 (9.0 vs 8.0 = 12.5% blocked; 10 vs 8 = 25% fires) — i.e., the bar sits INSIDE lite-score run-to-run noise. With the documented 1.0 clamp: sentinel case 700% (still auto-fires → sentinel fix is the load-bearing repair); ALL real-score cases identical (every real score >= 1.0).

### 5.3 The BUY-time stamp (backend/services/paper_trader.py)
- `"last_analysis_date": now` stamped in `execute_buy` at :304 (top-up) and :328 (new position). :476 is `execute_sell` partial-exit re-insert PRESERVING the field (not a stamp).
- Readers: autonomous_loop.py:794 (re-eval gate) and frontend/src/lib/types.ts:645 (display). No other consumers — semantics can be fixed without ripple.

### 5.4 The re-eval gate (backend/services/autonomous_loop.py)
- :791-804: due when `days_since >= settings.paper_reeval_frequency_days`; missing/unparseable date → re-eval (safe default). `paper_reeval_frequency_days=3` at settings.py:322 (audit cite :308 drifted).
- **`.days` truncation + cycle-time drift gotcha**: cycles ran 19:04 (06-05) then 18:11 (06-08) → 2d23h07m → `days=2` → DELL not re-evaluated at the 3-calendar-day mark; effective cadence is 3-4 days.
- Candidate filter :776 (`new_candidates` excludes held tickers) — counterfactual-replay divergence source (Section 7).

### 5.5 Replay machinery inventory — **NO EXISTING TOOL REPLAYS THE SWAP PATH**
- `decide_trades`/`_compute_swap_candidates` consumers (repo-wide grep): autonomous_loop.py:1155 (live), backend/tests/* (synthetic fixtures), scripts/go_live_drills/zero_orders_drill.py + scripts/smoketest_stages_5_through_13.py (synthetic drills). backend/autoresearch/strategy_backtest_adapter.py:43 docstring EXPLICITLY notes best_params "is NOT threaded into decide_trades".
- The 52.x/53.1 "$0 replay" = monthly-rebalance universe machinery: backend/backtest/rebalance_band.py:22 `apply_no_trade_band(prev_holdings, ranked_tickers, top_n, band_pct)` (the REJECTED 53.1 lever, machinery still present) + backtest_engine/walk_forward. None invokes the swap engine.
- Reusable for 60.2: analytics.py:239 `sharpe_diff_test` (LW 2008 + stationary bootstrap), :338 `compute_round_trips(all_trades)`, :392 `compute_trade_statistics`. The 57.1 event-study precedent (REJECT-trades reconstruction from paper_trades + analysis_results; out-channel at autonomous_loop.py:1151-1169) is the pattern to follow.

### 5.6 Away-week fixture data — BQ-CONFIRMED (query run 2026-06-11 via python BQ client)
`SELECT ticker, action, quantity, price, reason, created_at FROM financial_reports.paper_trades WHERE ticker IN ('MU','SNDK','DELL','STX') AND DATE(created_at) BETWEEN '2026-06-04' AND '2026-06-11'`:
| Ticker | Leg | TS (UTC) | Price | Reason |
|---|---|---|---|---|
| DELL | SELL | 06-04 19:00:38 | 425.08 | swap_for_higher_conviction |
| MU | SELL | 06-05 19:02:46 | 887.30 | stop_loss_trigger |
| SNDK | SELL | 06-05 19:03:20 | 1553.615 | stop_loss_trigger |
| DELL | BUY | 06-05 19:04:09 | 394.00 | new_buy_signal |
| STX | BUY | 06-05 19:04:24 | 856.65 | swap_buy |
| STX | SELL | 06-08 18:11:20 | 882.69 | swap_for_higher_conviction |
| DELL | SELL | 06-08 18:11:34 | 400.15 | swap_for_higher_conviction |
| SNDK | BUY | 06-08 18:11:51 | 1634.73 | swap_buy |
| MU | BUY | 06-08 18:12:05 | 954.385 | swap_buy |
| MU | SELL | 06-09 18:12:08 | 894.5299 | swap_for_higher_conviction |
| SNDK | SELL | 06-09 18:12:22 | 1627.9865 | swap_for_higher_conviction |
| DELL | BUY | 06-09 18:12:55 | 370.70 | swap_buy |
| DELL | SELL | 06-10 18:39:40 | 378.335 | swap_for_higher_conviction |
| SNDK | BUY | 06-10 18:40:09 | 1656.3701 | swap_buy |

All 3 named round trips CONFIRMED: **MU 06-08→06-09** (1d 00:00:03; 0.750944 sh; −$44.95, −6.27%), **SNDK 06-08→06-09** (−$2.46, then re-bought 06-10 at 1656.37 = +1.74% ABOVE the 06-09 exit), **DELL 06-05→06-08** (2d23h07m < 3 days → no re-eval → sentinel; +$11.17) **and 06-09→06-10** (+$14.73). Every exit reason is literally `swap_for_higher_conviction`. Fees 0.1%/leg additional. (06-05 MU/SNDK exits were stop_loss_trigger — different mechanism, OUT of 60.2 scope.)

### 5.7 Existing tests matching the -k net (immutable command collects these; must pass post-change)
`pytest backend/tests -k 'swap or sentinel or reeval' --collect-only -q` → **9 tests**: test_portfolio_swap.py (4: fills_zero_buy_gap / disabled_reproduces_zero_buy / skips_below_threshold / respects_max_per_cycle), test_phase_57_1_reject_binding.py::test_reject_binding_swap_path_off_emits_on_blocks, test_cycle_heartbeat_alarm.py (2 — "sentinel" NAME COLLISION, heartbeat-file sentinel, unrelated), test_agent_map_live_model.py (2 — "swappable" collision, unrelated).

### 5.8 56.2/57.1 fixture reuse
- test_portfolio_swap.py helpers `_make_settings/_holding/_holding_analysis/_candidate_analysis` (:21-70) — **all fixtures use [0,1] scores (0.55-0.85), encoding the same false premise as the code**. Reuse the helpers for the 06-09 regression scenario but with REAL 1-10 integer scores (MU holding absent from holding_lookup vs DELL 7.0 candidate → flag OFF: swap fires [reproduces bug]; flag ON: no swap).
- test_phase_57_1_reject_binding.py: both-path (flag OFF emits / flag ON blocks) regression pattern — copy this shape for the 60.2 flag. test_phase_56_2_ops_fixes.py: degraded-scoring guard fixtures (cycle-level), useful for the "all-degraded cycle" edge.

## 6. Binding-ruling boundary analysis (53.1 LW-REJECT / 55.3 auto-FAIL family)

**Why the three legs are CORRECTNESS repairs, not band-family levers:**
1. The sentinel FABRICATES data: absence of a same-cycle analysis is converted into the strongest possible adverse signal (0.0 on a 1-10 scale — below the scale minimum). No literature supports missing-as-worst; the measured alpha-decay horizon (months, A1/A2) says a 1-day-old score is near-full-strength. Removing fabricated evidence is not incumbent protection — it is making the comparison use evidence at all.
2. The stamp/re-eval fix makes the swap comparison apples-to-apples (cand_score(t) vs holding_score(t), or explicitly age-capped LOCF) — comparison-validity, not stickiness.
3. The delta-scale fix restores the formula ALREADY DOCUMENTED at settings.py:293 (1.0 clamp) and keeps the threshold VALUE untouched at 25.0. Nothing is widened.
**Where the forbidden family begins** (do NOT cross in GENERATE): raising min_delta above 25; adding an absolute score-point floor beyond the documented formula; any holding-AGE/tenure shield ("can't swap holdings younger than N days") — that is a no-trade band in time, i.e., 53.1's rejected family wearing a calendar. Note the emergent look-alike: under LOCF, MU (buy-time score 7.0, 1 day old) vs DELL 7.0 → delta 0% → no swap. This LOOKS like tenure protection but is evidence symmetry — the holding is valued at its freshest real score, and an honestly BETTER candidate (e.g. 9 vs 7 = 28.6%) still displaces it next cycle. State this distinction verbatim in the contract.
**Evaluation bar unchanged:** the fix is NOT exempt from 53.1's measurement discipline — ON-vs-OFF replay + LW where T permits (Section 7). It is exempt only from the CATEGORY auto-FAIL, because it adds no retention band.

## 7. Replay design for criterion 4 ($0, production universe, away-week window)

**No existing tool replays the swap path (5.5) — build a minimal decision-replay event study (57.1 precedent):**
1. **Inputs (all persisted, $0):** per cycle 2026-05-26→06-10 (swap path live since phase-cycle-1): paper_trades rows (executed orders + prices), analysis_results rows (candidate + holding analyses with final_score/recommendation/risk_assessment per cycle, incl. BUY-time candidate analyses — persisted at autonomous_loop.py:875-881), positions reconstructed by rolling the paper_trades ledger forward.
2. **Validation arm (flag OFF):** re-run `decide_trades` on reconstructed inputs; assert emitted orders == recorded orders for each cycle (calibrates the reconstruction; pin `risk_overrides` to recorded/default values for determinism — decide_trades reads runtime overrides at :83-84,:254-261,:554-558).
3. **Counterfactual arm (flag ON):** same inputs, fix enabled. PRIMARY metric = per-cycle one-step order diff: each of the 3 named round trips suppressed or surviving WITH the numeric reason (LOCF score used, delta computed, threshold result). Expected (hypothesis for contract, to be verified): MU 06-09 suppressed (LOCF 7.0 vs DELL 7.0 → 0% < 25%), SNDK 06-09 suppressed, DELL 06-08 suppressed (LOCF from 06-05 analysis).
4. **Secondary (clearly labeled):** compounded counterfactual NAV: prices are exogenous in paper trading (no market impact — the Balch critique's impact leg does not apply), so the ON portfolio can be marked to market with recorded/yfinance prices. DISCLOSE divergence honestly: (a) once portfolios diverge, recorded candidate streams embed the REAL portfolio's held-ticker filter (autonomous_loop.py:776) — e.g. MU appeared as a 06-08 candidate only because the real system had stop-lossed it 06-05; flag any cycle where a counterfactual holding appears in the recorded candidate stream; (b) missing re-eval analyses for counterfactual-only holdings are valued by the SAME LOCF rule as live (consistent by construction).
5. **Metrics:** weekly turnover %NAV, round-trip count + within-3-day round trips (`compute_round_trips` analytics.py:338), per-round-trip P&L net of 0.1%/leg, NAV delta, maxDD; Sharpe delta via `sharpe_diff_test` (analytics.py:239) computed but reported as UNDERPOWERED at T≈12 cycles — the week answers "are the named round trips suppressed", not "is Sharpe improved at p<0.05" (52.3 methodology; McLean-Pontiff). Promotion = OPERATOR decision on the descriptive evidence.

## 8. Risks / gotchas for GENERATE
1. **Flag-OFF byte-identity:** all 3 legs behind one default-OFF settings flag; existing 9 collected tests must pass unchanged (immutable -k net). Follow the 52.2 dark-launch idiom (`getattr(settings, ..., False)`).
2. **Fixture scale trap:** test_portfolio_swap.py fixtures are [0,1]; under the 1.0-clamp the fixture TECH1 swap (0.82 vs 0.58 → clamped delta 24% < 25%) would STOP firing — do NOT apply the new formula to the OFF path; new ON-path tests use 1-10 integers.
3. **LOCF reader cost:** `get_report` is SELECT * per ticker — bound to displacement-exposed holdings only (sectors at cap with queued candidates), not all positions; financial_reports is us-central1; 30s timeout rule.
4. **Targeted re-eval injection** must respect `paper_max_daily_cost_usd` (the per-cycle budget checks at autonomous_loop.py:851-860) and must not double-analyze a ticker already in `reeval_tickers`.
5. **`.days` truncation + cycle drift** makes the 3-day gate effectively 3-4 days; do not "fix" globally (cost impact) — leg 2's injection addresses the displacement case; leave cadence semantics else byte-identical.
6. **57.1 composition:** REJECT candidates are filtered upstream (:194-212) — test 60.2 flag ON x 57.1 flag ON to prove no interaction.
7. **Replay determinism:** decide_trades reads risk_overrides (runtime store) — pin/patch in the replay harness; sort stability matters (orders.sort :432).
8. **Honest reporting:** DELL 06-09→06-10 round trip was PROFITABLE (+$14.73) and STX +$18.13 — the replay must report suppressed-but-profitable trades as a COST of the fix, not hide them (criterion: "suppressed or surviving with stated reasons").
9. Do not conflate the 06-05 stop_loss_trigger exits with the swap churn — out of scope.
10. settings description (:293) vs code (:531) divergence is itself evidence the formula drifted unreviewed — cite in the contract as the correctness basis.

## 9. Source table

### Read in full (6 — gate-counting)
| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| https://arxiv.org/html/2505.07078v5 (KDD 2026, FINSABER) | 2026-06-11 | peer-reviewed | WebFetch HTML | FinMem commission 5-9x; overtrading = "persistent value destruction"; bias-controlled paired evaluation protocol |
| https://arxiv.org/html/2603.27539 | 2026-06-11 | preprint (2026) | WebFetch HTML | 10-20bps round trip → 25-50pp annual drag; 5 minimum evaluation standards; memory relevance-decay risk |
| https://ar5iv.labs.arxiv.org/html/1705.00109 (Boyd et al.) | 2026-06-11 | peer-tier monograph | WebFetch ar5iv | trade iff gain > φtrade+φhold+risk; backtest reporting + parameter-perturbation sensitivity |
| https://jhfinance.web.unc.edu/wp-content/uploads/sites/12369/2016/02/Alpha-Decay.pdf (Di Mascio/Lines/Naik) | 2026-06-11 | peer-reviewed (JF) | curl+pdfplumber (56pp) | buy alpha decays GRADUALLY over ~12 months — day-old scores near-full-strength |
| https://www.jpmorgan.com/content/dam/jpm/cib/complex/content/technology/ai-research-publications/pdf-12.pdf (Balch et al.) | 2026-06-11 | industry research | curl+pdfplumber (10pp) | "Market Replay... does not substantially adapt to or respond to the presence of the experimental strategy" — replay-limitation framing |
| https://www.mavensecurities.com/alpha-decay-what-does-it-look-like-and-what-does-it-mean-for-systematic-traders/ | 2026-06-11 | industry practitioner | WebFetch HTML | staleness cost progressive (5.6% US / 9.9% EU avg), no cliff → age-cap not zeroing |

### Snippet-only (29 unique — context, non-gate; selected)
| URL | Kind | Why not fetched |
|---|---|---|
| https://www.tandfonline.com/doi/full/10.1080/0015198X.2024.2317323 (Smart Rebalancing, FAJ 2024) | peer-reviewed | HTTP 403 on fetch; used via search synopsis |
| https://www.econ.uzh.ch/static/wp_iew/iewwp320.pdf + https://www.zora.uzh.ch/id/eprint/52220/1/iewwp320.pdf (Ledoit-Wolf 2008) | peer-reviewed | BOTH mirrors returned stub files (808/146 bytes); methodology read in full in phase-52.3 and implemented at analytics.py:239 |
| https://arxiv.org/html/2512.02227v1 | preprint | already project-adopted (sector caps); cited internally |
| https://arxiv.org/pdf/2507.08584 ("To Trade or Not to Trade") | preprint | recency-scan corroboration only |
| https://arxiv.org/pdf/2510.07920 (Profit Mirage) | preprint | recency-scan corroboration only |
| https://optimization-online.org/wp-content/uploads/2015/02/4785.pdf (multi-period alpha decay) | preprint | redundant with Boyd et al. (same framework family) |
| https://link.springer.com/article/10.1007/s11408-022-00419-6 ; https://www.tandfonline.com/doi/full/10.1080/17442508.2011.651219 ; https://jpm.pm-research.com/content/29/4/49 ; https://arxiv.org/pdf/1203.4153 ; https://nr.no/en/publication/924706/ | peer-reviewed | no-trade-band family — context for Section 6 boundary only |
| https://www.bu.edu/econ/files/2011/01/KothariWarner2.pdf (event studies) ; https://link.springer.com/article/10.1007/s11408-019-00325-4 ; https://www.emergentmind.com/topics/counterfactual-replay ; https://microalphas.com/signal-decay-patterns/ ; https://www.top1000funds.com/wp-content/uploads/2021/05/SSRN-id2580551.pdf ; https://tradingagents-ai.github.io/ ; https://arxiv.org/pdf/2409.08357 ; https://arxiv.org/pdf/2407.21791 ; https://arxiv.org/html/2606.08283 ; https://arxiv.org/pdf/2603.10092 ; +ResearchGate mirrors (3), MIT WP, Springer s10614-024-10555-y, sciencedirect 000169189290059M, PMC12816306 | mixed | lower marginal value vs the 6 read-in-full; budget discipline |

## 10. JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 29,
  "urls_collected": 35,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
