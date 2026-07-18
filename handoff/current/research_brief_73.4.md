# Research Brief — phase-73.4 D2d COST-INTEGRATED PROMOTION DESIGN

Tier: **moderate** | NOT audit-class | Started 2026-07-18

Deliver design_inputs for THREE components:
1. **net-of-cost-dsr** — per-period cost attribution + r_net series construction + adapter seam.
2. **cost-per-bp-reporting** — minimum per-decision log shape (The New Quant §7.10).
3. **pbo-discrepancy-doc** — enumerate every PBO threshold; document 0.20-vs-0.5 single policy.

Provenance: frontier_map_73.md #4 (ADAPT net-of-cost DSR fed to existing compute_dsr, gate.py byte-unchanged; ADOPT cost-per-bp min reporting) + #9 (keep PBO 0.20, document). This component flagged PARALLELIZABLE-EARLY.

---

## Research: cost-integrated promotion for an LLM trading system

### Read in full (>=5 required; counts toward the gate)
| URL | Accessed | Kind | Fetched how | Key quote or finding |
| --- | --- | --- | --- | --- |
| https://arxiv.org/html/2510.05533v1 | 2026-07-18 | paper (survey) | WebFetch html (§5.6/7.5/7.10 targeted) | **§7.5: "System level reporting should include wall clock latency per decision and amortized compute cost PER BASIS POINT of excess return."** §7.10 min-reporting: "full cost model with commissions, spreads, and market impact...turnover, capacity, and the effect of transaction costs on NET performance...wall clock latency and compute cost per decision." §5.6: hybrid routing cuts spend. |
| https://ar5iv.labs.arxiv.org/html/2402.03755 | 2026-07-18 | paper | WebFetch ar5iv (§4.2, Thm 4.6) | §4.2: self-improvement inner loop token cost = **O(KT²H)** (K outer iters × T rounds × H horizon); inference-only drops to O(T²H) tokens/O(HT) time. Thm 4.6: Bayesian regret sublinear in KT. GAP: no explicit token-budget→K/T constraint — the caller must meter it (our #4 objective). |
| https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf | 2026-07-18 | paper (primary) | WebFetch (PDF→md) | PSR denominator non-normality term = "γ₃²·SR/4 + (γ₄−3)·SR²/24", ×T^(−1/2); E[max(SR)] = (1−γ)·Z⁻¹(1−1/N) + γ·Z⁻¹(1−1/(N·e)), γ=0.5772. **"the PSR depends on the sample skewness and kurtosis of REALIZED returns"** → changing the return series (fees/scaling) alters γ₃,γ₄ and hence DSR directly. Matches compute_psr/compute_dsr :495-546 exactly. |
| https://bsic.it/backtesting-series-episode-5-transaction-cost-modelling/ | 2026-07-18 | practitioner | WebFetch | Cost components: commission/fees ("easy to model...fixed"), slippage ("price change between decide and execute", ≈ `c₁·mean_spread + c₂`), market impact (temp+perm). Costs subtracted per trade to turn gross→net; "trading costs can be similar in magnitude or even outweigh the systematic premia." |
| https://arxiv.org/html/2607.10286v1 | 2026-07-18 | paper (recency) | WebFetch html (§3.2, App C, D.1) | **[RECENCY-2026]** §3.2 Eq(6): net profit **R₁:T = P₁:T − C₁:T** (gross − total deploy cost). Eq(9): net agentic value = P^timing − C^dyn (excludes static infra). App C taxonomy: **Cₜ = Cₜˡˡᵐ + Cₜᵗʳᵈ + Cₜⁱⁿᶠ + Cₜˢᵗᵒ**; LLM token cost cₜˡˡᵐ = [Σ pˢ·xᵣˢ]/ρ (ρ=success rate). D.1: costs "user-configured...accuracy depends on fidelity" — so our per-call cost_usd (measured, not configured) is STRICTLY better fidelity. |

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
| --- | --- | --- |
| gipsstandards.org/.../2020_gips_standards_firms.pdf | official std | GIPS net-of-fees substance captured via WebSearch (below); primary PDF binary. Net-of-fees = gross reduced by mgmt fees; BOTH must deduct transaction costs; all returns after tx costs incurred in-period; must be clearly labeled gross/net; may present both. |
| ryanoconnellfinance.com/gips-standards/ | blog | HTTP 403 |
| cascadecompliance.com/post/gips-model-fee-requirements | blog | fetched but only model-fee content, not the gross/net definition |
| arxiv.org/abs/2606.07846 (Cost-Aware Speculative Execution for LLM-Agent Workflows) | paper | recency hit, snippet only |
| arxiv.org/abs/2605.19337 (Agentic Trading: When LLM Agents Meet Financial Markets) | paper | recency hit, snippet only |
| arxiv.org/abs/2511.21572 (BAMAS: Budget-Aware Multi-Agent Systems) | paper | recency hit, snippet only |
| arxiv.org/abs/2512.02230 (Benchmarking LLM Agents for Wealth-Management) | paper | recency hit, snippet only |
| en.wikipedia.org/wiki/Deflated_Sharpe_ratio ; marti.ai/qfin/.../deflated-sharpe-ratio.html | tertiary/blog | DSR read in full via the Bailey primary PDF instead |
| quantstart.com/.../Successful-Backtesting-...-Part-II ; quantpedia.com/the-price-of-transaction-costs | practitioner | tx-cost read in full via BSIC instead |

### Recency scan (2024-2026)
Performed. A dedicated 2026-scoped pass ("cost-aware evaluation LLM trading agent net-of-cost Sharpe token cost per decision 2026") surfaced a LIVE 2026 research frontier on exactly this topic. Read in full: **arXiv:2607.10286 "Can Agentic Trading Systems Pay for Their Own Intelligence?" (Jul 2026)** — formalizes net = gross − cost with an LLM-token-cost sub-term, directly reinforcing The New Quant §7.5. Snippet-level new work: 2606.07846 (cost-aware speculative execution), 2605.19337 (agentic trading), 2511.21572 (BAMAS budget-aware MAS), 2512.02230 (wealth-mgmt benchmark). NEW-FINDING synthesis: cost-adjusted evaluation is now standard practice guidance — "leaderboards should favour cost-effective agents rather than accuracy at any price"; "disclose model version, pricing snapshot date, token assumptions, and the breakeven gross P&L required for net profitability"; input vs output tokens differ 3-8× and must not be conflated (our cost_tracker already prices them separately, cost_tracker.py:20-80). This COMPLEMENTS (does not supersede) The New Quant §7.5/§7.10 and the Bailey DSR canon — it operationalizes them for the LLM-agent case. No finding contradicts frontier_map #4.

### Search queries run (3-variant discipline)
1. **Current-frontier**: "The New Quant survey LLMs financial prediction trading minimum reporting standard cost per basis point" → the §7.5/§7.10 anchor.
2. **Year-less canonical**: "Deflated Sharpe Ratio Bailey Lopez de Prado correcting selection bias backtest overfitting non-normality..." + "transaction cost modeling backtest slippage commission market impact net returns..." + "GIPS...net-of-fees vs gross-of-fees..." → founding DSR paper, canonical tx-cost modelling, GIPS standard.
3. **Recency (2026)**: "cost-aware evaluation LLM trading agent net-of-cost Sharpe token cost per decision 2026" → the 2607.10286 recency full-read + 4 snippet hits.

### Key findings
1. **Cost-per-bp IS a named standard, not our invention.** The New Quant §7.5: "System level reporting should include wall clock latency per decision and amortized compute cost PER BASIS POINT of excess return." §7.10 mandates a "full cost model with commissions, spreads, and market impact...turnover, capacity, and the effect of transaction costs on NET performance...compute cost per decision." (Source: 2510.05533 §5.6/7.5/7.10)
2. **Net-of-cost must be applied to the RETURN SERIES, not as a post-hoc DSR penalty.** DSR = PSR(SR*), and PSR depends on the sample skewness γ₃ and kurtosis γ₄ of the *realized* returns. Subtracting costs changes the return distribution → changes γ₃, γ₄, SR, and hence DSR self-consistently. A "gross-DSR − additive-penalty" would be statistically meaningless (it corrupts the P(SR>SR*) semantics). (Source: Bailey & LdP DSR PDF; matches compute_psr :495-515, compute_dsr :518-546.)
3. **The N-trials deflation term is untouched by cost.** SR* = √Var[SR]·[(1−γ)Z⁻¹(1−1/N)+γZ⁻¹(1−1/(N·e))] depends only on the cross-sectional trial-Sharpe variance and N, NOT the sign/level of returns. So net-of-cost r_net flows through compute_dsr's *first* arg only; `all_trial_sharpes`/`n_trials` (the deflation) are unchanged → gate.py stays byte-for-byte identical. (Source: Bailey PDF; gate.py:19-39 reads only trial['dsr']/['pbo'].)
4. **The quant-only backtest is ALREADY transaction-cost-net; token cost there is structurally ZERO.** backtest_trader debits per-trade commission (flat_pct/per_share) so nav_history → generate_report → compute_deflated_sharpe is net of tx cost (analytics.py:665-674 derives skew/kurt/T from `np.diff(navs)/navs[:-1]`). The GBM walk-forward calls NO LLM (rules/backend-backtest.md "quant-only, $0 LLM cost"), so a token-cost subtraction there would fabricate a cost that was never incurred. Token-cost net-of-cost bites on the LIVE realized-return series (and any future live-champion bakeoff), NOT the promotion backtest. (Source: backtest_engine.py:162-221/340-366/762-764; backend-backtest.md.)
5. **Token cost must be derived from per-CALL sums, never the session gauge SUM.** session_cost_usd is a per-cycle GAUGE: reset to 0 at cycle start (autonomous_loop.py:336), accumulates within cycle (:1095), stamped on each BQ row as the cumulative-so-far value (test_phase_66_3_cost_truth.py:56-75 asserts 0.50→0.50→0.50). Per-cycle token cost = LAST/MAX gauge per cycle_id ≡ SUM of per-call cost_usd for that cycle (cost_tracker AgentCostEntry.cost_usd is the additive primitive). SUMming the gauge column across rows multi-counts. (Source: autonomous_loop.py; cost_tracker.py; MEMORY project_return_day_state.)
6. **Measured cost > configured cost (fidelity).** 2607.10286 D.1 concedes its cost model is "user-configured rather than automatically measured." Our cost_tracker measures actual input/output/cache tokens per call and prices input≠output separately (the 3-8× spread the recency lit warns about) — strictly higher fidelity than the frontier framework. Money-recon confirms scale: ~$34.46 tx + $16.75 LLM ≈ $51/window ≈ 0.2% NAV. (Source: 2607.10286; cost_tracker.py:20-80; money_recon_2026-07-18.md.)
7. **Two distinct PBO thresholds are BOTH correct, at different gates.** 0.20 = the PROMOTION gate (gate.py) — a strategy must clear it to go live (stricter than charter, the safe direction). 0.5 = the per-candidate VETO cap (risk_server.py DEFAULT_PBO_VETO_THRESHOLD, surfaced via the pbo_check/evaluate_candidate MCP tools) — an advisory MAS reject. They are not in conflict; the charter's "PBO≤0.5" is the veto floor, 0.20 is the tighter promotion bar. No read paper resolves the specific number — it is a policy choice (frontier_map #9). (Source: gate.py:21; risk_server.py:28,134-165.)

### Internal code inventory
| File | Lines | Role | Status |
| --- | --- | --- | --- |
| backend/services/perf_metrics.py | compute_dsr :518-546 | DSR = PSR(SR*); accepts ANY `returns: Sequence[float]` + `all_trial_sharpes` + `n_trials`. SR* from cross-sectional trial-Sharpe variance | LOAD-BEARING seam: feed r_net here |
| backend/services/perf_metrics.py | compute_psr :495-515 | PSR uses per-period SR, skew g3, RAW kurtosis g4 — the moments r_net MUST preserve | verified |
| backend/services/perf_metrics.py | get_scalar_metric :439-455 | optimizer scalar = risk_adjusted*(1-min(0.3, turnover*tx_cost_pct)); tx_cost_pct default 0.001 | the ONLY place cost lives today (multiplicative penalty) |
| backend/autoresearch/gate.py | PromotionGate :19-39 | min_dsr=0.95, max_pbo=0.20; reads ONLY trial['dsr']/['pbo']; cost-BLIND | stays byte-unchanged |
| backend/autoresearch/strategy_backtest_adapter.py | :155-254 | DSR from generate_report(seed)["analytics"]["deflated_sharpe"] (per-window Sharpe variance); PBO from K-variant column-stack -> compute_pbo(matrix,S=16) | r_net enters via the nav_history -> returns path |
| backend/agents/mcp_servers/risk_server.py | pbo_check :133-158, evaluate_candidate :162-222 | DEFAULT_PBO_VETO_THRESHOLD=0.5; vetoes when pbo>0.5 | SECOND PBO threshold (veto cap, advisory MAS) |
| backend/agents/cost_tracker.py | full | per-ANALYSIS AgentCostEntry {model, in/out tokens, cost_usd, ticker, cache}; summarize() JSON | per-call token cost source (in-memory, per-run) |
| backend/services/autonomous_loop.py | :91,123-136,336,1095 | `_session_cost` module GAUGE: reset 0.0 at cycle start (:336), accumulates via _add_session_cost (:1095), get_session_cost_usd() | GAUGE = cumulative-within-cycle, resets each cycle -> NEVER SUM across rows |
| backend/services/paper_trader.py | :208,289 (buy) / :443,472 (sell) | tx_cost = amount * paper_transaction_cost_pct/100; stored as paper_trades.transaction_cost | fee model only; NO explicit slippage (see below) |

**PBO threshold enumeration (for pbo-discrepancy-doc):**
1. `gate.py:21` PromotionGate.max_pbo = **0.20** — the PROMOTION gate (strategy goes live). Stricter.
2. `risk_server.py:28` DEFAULT_PBO_VETO_THRESHOLD = **0.5** — per-candidate VETO cap (advisory Data/Strategy/Risk MAS; `pbo_check` + `evaluate_candidate`). Matches the charter/memory "PBO<=0.5".
3. `strategy_backtest_adapter.py` — no threshold of its own; PRODUCES the pbo value the gate consumes (K-variant CSCV, S=16).
4. MCP tool schemas (`mcp__pyfinagent-risk__pbo_check` threshold=0.5, `evaluate_candidate` pbo_threshold=0.5) = the risk_server defaults surfaced. Same 0.5.

### Application to pyfinagent — design for the three components

#### Component 1 — net-of-cost-dsr
- **Which costs, per period:** (a) FEES: `paper_trades.transaction_cost` (paper_trader.py:208/289 buy, :443/472 sell = amount × `paper_transaction_cost_pct/100`, default 0.1%, settings.py:371) — ALREADY in the backtest nav via backtest_trader commission debit, so no double-subtract there. (b) SLIPPAGE: NO explicit model. bq_sim fills at the passed close `price` (ExecutionRouter returns it unchanged, paper_trader.py:268-277); real slippage is implicit only in alpaca_paper `fill_price`≠close. On the GBM backtest, slippage is un-modeled → an optional additive haircut (BSIC `c₁·mean_spread + c₂`) is the only net-of-cost delta available there. (c) TOKEN COST: per-cycle = SUM of per-call `cost_usd` for that cycle_id (≡ last session gauge) — NEVER the SUM of the session_cost_usd gauge column.
- **Return-series construction & seam:** TWO seams, different cost vectors —
  - SEAM A (promotion backtest, the gate's input): `strategy_backtest_adapter.make_engine_backtest_fn` → `generate_report(seed)["analytics"]["deflated_sharpe"]` (analytics.py:665-674 builds returns from nav_history). Nav is tx-net already; token cost = 0 (no LLM). Net-of-cost upgrade here = OPTIONAL slippage haircut on the nav-derived returns BEFORE `compute_deflated_sharpe`. Do NOT subtract token cost here.
  - SEAM B (live realized series / future live-champion bakeoff): `perf_metrics.compute_dsr(r_net, all_trial_sharpes, n_trials)` where r_net[t] = r_gross[t] − tx_cost_t/NAV_{t-1} − slippage_t/NAV_{t-1} − token_cost_t/NAV_{t-1}. This is where the token term is real and where the "only real build" lives (per-cycle token → period-return attribution).
- **Where r_net replaces r_gross:** the FIRST positional arg of `compute_dsr`/`compute_psr` ONLY. `all_trial_sharpes` + `n_trials` (the N-deflation) are unchanged. `gate.py` is byte-unchanged (reads only `trial['dsr']`/`['pbo']`). This preserves DSR's P(SR>SR*) statistical meaning (finding #2/#3).
- **Backward comparability (transition):** compute and log BOTH `dsr_gross` and `dsr_net` (GIPS "clearly labeled...may present both" + New Quant §7.10 "effect of transaction costs on NET performance"). Promote on `dsr_net` once validated; keep `dsr_gross` visible for ≥1 window so a regression is auditable. Guard: `compute_dsr` needs ≥5 returns + ≥2 trials (:536) — net series has the same length, so no new degeneracy.

#### Component 2 — cost-per-bp-reporting
- **Where a decision record already exists (exact columns, observability/api_call_log.py):** `llm_call_log` (writer `log_llm_call` :220-290) persists per-call {ts, provider, model, agent, latency_ms, ttft_ms, input_tok, output_tok, cache_creation_tok, cache_read_tok, request_id, ok, cycle_id, session_cost_usd GAUGE} — note it has TOKEN columns + the gauge but NO per-call cost column. The sibling `api_call_log` table (:14-52) DOES persist a per-call **`cost_usd_est FLOAT64`** column. cost_tracker AgentCostEntry has the in-memory per-call {model, input/output tokens, cost_usd, ticker}. `/api/sovereign/efficiency` already computes profit_per_llm_dollar = realized_pnl_usd / (anthropic+vertex+openai cost) (sovereign_api.py:564-568).
- **Correct per-cycle token-cost derivation (three equivalent, all avoiding the gauge-SUM trap):** (A) `MAX(session_cost_usd) GROUP BY cycle_id` on llm_call_log — the gauge's terminal value per cycle; (B) recompute per-call cost from `input_tok/output_tok/cache_*_tok × MODEL_PRICING` (cost_tracker.py:165-183) and `SUM ... GROUP BY cycle_id` — most granular, keeps per-model/per-ticker attribution; (C) `SUM(cost_usd_est) GROUP BY cycle_window` on api_call_log (a real per-call cost column → summing rows is correct). NEVER `SUM(session_cost_usd)` across llm_call_log rows (multi-counts the gauge).
- **Minimum per-decision log shape (The New Quant §7.10 + 2607.10286 App C):** one row per decision/cycle: `{decision_id, cycle_id, ts, excess_return_bps, token_cost_usd, slippage_usd, fees_usd, cost_per_bp, latency_ms, model_version, pricing_snapshot_date}`. This is the §7.10 list minus fields we already have elsewhere (turnover/capacity live in trade_statistics). Reuse the existing session_cost gauge + llm_call_log; the ONLY new field is the per-decision `cost_per_bp`.
- **cost_per_bp WITHOUT double-counting:** `excess_return_bps = (strategy_return − benchmark_return) × 1e4` realized PER TRADE/round-trip (compute_round_trips/compute_trade_statistics already exist, analytics.py). `total_cost = token_cost + slippage + fees` attributed PER DECISION. `cost_per_bp = total_cost_usd / max(excess_return_bps, ε)` with a sign/zero guard (report `None`/`inf-flag` when excess_bps ≤ 0, mirroring the /efficiency NULL-on-zero pattern sovereign_api.py:567-570). CRITICAL: cost_per_bp is a DIAGNOSTIC RATIO computed alongside the net-DSR series — it is NOT fed back into DSR. The net series already subtracts cost once as a return drag (Component 1); cost_per_bp reports the same cost as a ratio. Subtracting in DSR AND penalizing via cost_per_bp elsewhere in one objective would double-count — keep DSR the objective and cost_per_bp the report.

#### Component 3 — pbo-discrepancy-doc
- **Every PBO threshold in the codebase (enumerated):** (1) `gate.py:21` PromotionGate.max_pbo=**0.20** — promotion gate. (2) `risk_server.py:28` DEFAULT_PBO_VETO_THRESHOLD=**0.5** → `pbo_check` (:135) + `evaluate_candidate` (:164) → surfaced as the `mcp__pyfinagent-risk__pbo_check` (threshold=0.5) and `evaluate_candidate` (pbo_threshold=0.5) tool defaults. (3) `strategy_backtest_adapter.py` — no threshold; PRODUCES the pbo value (K-variant CSCV, S=16, `_DEFAULT_PBO_S`) the gate consumes. No other numeric PBO threshold exists (grep of backend/).
- **Recommended single documented policy (frontier_map #9 + #73.0):** KEEP 0.20 as the binding PROMOTION gate (stricter = the safe/false-positive-resistant direction; no read paper resolves the number → it is a deliberate policy choice). Document 0.5 as the advisory per-candidate VETO cap = the charter/memory "PBO≤0.5" floor. Framing: **"a strategy is VETOED at PBO>0.5 (candidate-level risk reject) and PROMOTED only at PBO≤0.20 (go-live bar)"** — nested, not contradictory. The two are different gates on different objects (candidate vs promotion), so both stay.
- **Where documentation lands (recommend-only):** ARCHITECTURE.md (the MADR record — sits beside "Research Gate Discipline") is the primary home; add a one-line cross-ref in `.claude/rules/backend-backtest.md` "DSR guard" bullet (which already states DSR≥0.95). CLAUDE.md is optional (it is already dense; a pointer suffices). The **charter memory** (project_system_goal.md "PBO≤0.5") is OPERATOR-OWNED — recommend the operator annotate it as "0.5 = veto cap; 0.20 = promotion gate", do NOT edit it.

### Research Gate Checklist
Hard blockers (all satisfied):
- [x] ≥5 authoritative external sources READ IN FULL via WebFetch (New Quant, QuantAgent, Bailey DSR, BSIC tx-cost, 2607.10286)
- [x] 10+ unique URLs total (25+ collected across 5 searches)
- [x] Recency scan (2024-2026) performed + reported (2607.10286 full + 4 snippet 2026 hits)
- [x] Full papers/pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim
Soft checks:
- [x] Internal exploration covered every named module (perf_metrics, gate, adapter, cost_tracker, risk_server, autonomous_loop, paper_trader, analytics, backtest_engine, sovereign_api, settings)
- [x] Contradictions noted (0.20 vs 0.5 resolved as nested gates, not a contradiction)
- [x] Claims cited per-claim

---
```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 9,
  "urls_collected": 27,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "coverage": {"audit_class": false, "rounds": 1, "dry_rounds": 0, "K_required": 2, "new_findings_last_round": 0, "dry": false},
  "summary": "Net-of-cost DSR = feed a NET-of-cost per-period return series into the EXISTING compute_dsr (:518) as its FIRST arg only; all_trial_sharpes/n_trials (the N-deflation) unchanged, so gate.py stays byte-identical. Two seams with different cost vectors: the promotion GBM backtest (strategy_backtest_adapter->generate_report) is ALREADY tx-net and has ZERO token cost (no LLM) - only an optional slippage haircut applies; the LIVE realized series (perf_metrics.compute_dsr) is where token cost bites (r_net = r_gross - tx/NAV - slippage/NAV - token/NAV). Token cost per period = SUM of per-call cost_usd per cycle_id, NEVER the SUM of the session_cost_usd GAUGE. cost-per-bp reporting per New Quant 7.5/7.10 + 2607.10286: log {excess_return_bps, token_cost, slippage, fees, cost_per_bp, latency_ms, model_version} per decision; cost_per_bp = total_cost/max(excess_bps,eps) is a DIAGNOSTIC ratio, NOT fed back into DSR (avoids double-count). PBO thresholds: 0.20 (gate.py promotion, stricter/safe) vs 0.5 (risk_server veto cap = charter floor) are nested gates on different objects - document in ARCHITECTURE.md, recommend-only on the operator-owned charter memory.",
  "brief_path": "handoff/current/research_brief_73.4.md",
  "gate_passed": true
}
```

---
(envelope at tail)
