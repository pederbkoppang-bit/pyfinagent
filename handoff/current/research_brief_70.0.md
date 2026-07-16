# Research Brief — phase-70.0: Trade diversity + changeable fund (design pack)

**Tier:** complex · **Date:** 2026-07-16 · **Author:** Layer-3 Researcher subagent
**Step objective (offline, $0, NO production code):** produce the research basis + design for
(a) SOFT profit-aware sector diversification of the analyzed top-N set (respecting the
2026-06-01 replay that HARD sector-neutral HURTS long-only returns); (b) ATOMIC, cash-bounded,
cross-sector-capable swap/rotation execution; (c) BUY-gate OBSERVABILITY (hidden $1 session
budget vs visible $2 daily cap; diagnosable silent skip-reasons).

**Binding context (north star):** risk-adjusted OOS P&L — diversification must NOT lower it →
design must be SOFT + flag-gated + backtest-validated (DARK-until-token). $0 metered; paper-only;
`historical_macro` FROZEN; hysteresis BANNED; do-no-harm (no risk-limit thresholds moved).

---

## 1. Three-variant query disclosure (research-gate mandatory)

Per `.claude/rules/research-gate.md`, ≥3 query variants per topic: current-year 2026 frontier,
last-2-year 2025/2024 window, year-less canonical.

**Topic A — sector diversification vs concentration in long-only momentum/quant**
- 2026 frontier: `sector diversification constraint long-only momentum portfolio alpha cost 2026`
- Year-less canonical: `sector neutralization long-only equity momentum returns reduce concentration`
- 2025 window: `soft sector constraint penalty portfolio optimization diversification 2025`

**Topic B — atomic / transactional multi-leg order execution + rollback**
- Year-less canonical: `atomic multi-leg order execution rollback saga pattern trading system`
- 2026 frontier: `compensating transaction order execution failure recovery trading engine 2026`
- (2025 window covered by SagaLLM Mar-2025 + Baeldung/microservices canonical hits.)

**Topic C — observability / limit-transparency for autonomous trading loops**
- 2026 frontier: `observability autonomous trading loop budget limit logging skip reason 2026`
- Year-less canonical: `algorithmic trading order rejection observability logging audit trail structured`
- (2025 window covered by VeritasChain Dec-2025 + Arize/oneuptime 2026 hits.)

8 searches run; all three variants represented per topic (source table mixes current-year,
last-2-year, and year-less canonical hits).

---

## 2. Source table

### Read IN FULL via WebFetch (7 — floor is 5)
| # | Source | Tier | Topic | Key takeaway |
|---|--------|------|-------|--------------|
| 1 | Harvey, Ehsani & Li, *Is Sector Neutrality in Factor Investing a Mistake?* — Research Affiliates / **Financial Analysts Journal, May 2023** | 1 (peer-reviewed) | A | Long-only investors are **more likely to benefit from investing in the factor as it stands** (keep sector exposure); long–short benefit from hedging out sector bets. Categorical distinction by portfolio structure. |
| 2 | QuantPedia, *Should Factor Investors Neutralize the Sector Exposure?* | 4 (practitioner) | A | Summarizes Ehsani-Harvey-Li: keeping sector exposure gives **better long-only factors in 78% of trials**; neutralizing helps long–short (across-component better in only 20% long-short trials). Decision rule: **neutralize iff Sharpe-ratio(across÷within) < their correlation** — "frequently met in long-short, unlikely in long-only." |
| 3 | microservices.io — *Pattern: Saga* (Richardson) | 2 (authoritative reference) | B | Multi-step cross-store txn = sequence of local txns each with a **compensating transaction**; **no automatic rollback** (dev must design compensation); **lack of isolation** needs countermeasures (semantic lock, pivot txn). Orchestration vs choreography. |
| 4 | *SagaLLM: Transaction Guarantees for Multi-Agent LLM Planning* — arXiv 2503.11951 (Mar 2025) | 1 (peer-reviewed preprint) | B | Saga for LLM plans: `S={T1..Tn,Cn..C1}`; guarantees **either a fully-committed S′ or a coherent rollback to S — avoiding partial/inconsistent outcomes**; traverses a dependency graph for the **minimal compensation set**; independent **GlobalValidationAgent** (rejects agent self-validation — aligns with harness Q/A doctrine). |
| 5 | VeritasChain, *Building Tamper-Evident Audit Trails for Algorithmic Trading* — DEV.to (Dec 2025) | 4 (practitioner) | C | Event schema (`event_id`/`trace_id`/`timestamp`/`event_type`); **log the FULL decision lifecycle** — SIG/ORD/REJ/EXE as distinct events; **rejections & skipped orders are REJ events carrying `skip_reason` + `decision_factors`**, never omitted; `trace_id` correlates signal→order→reject. |
| 6 | *Portfolio Optimization with Physical Decision Variables … Diversification Challenge* — arXiv 2601.08717v1 (Jan 2026) | 1 (peer-reviewed preprint) | A | **Soft diversification penalty**: `max[(1−w)·ROI − w·Risk − w_d·θ₁·HHI]`; tunable `w_d∈[0,1]` (0 = ignore diversification, byte-identical to no-penalty); **θ₁ auto-rescales** the HHI term to the magnitude of ROI/risk; explicit concentration-vs-diversification trade-off. |
| 7 | oneuptime, *AI Agents Are Breaking Your Observability Budget* (Mar 2026) | 4 (practitioner) | C | Low-yield but one usable principle: **treat budget/spend as a first-class observable metric** — "add spend as a metric, set alerts on cost anomalies, track cost-per-interaction" — i.e. surface budget consumption rather than tracking it silently. |

### Snippet-only (search hits evaluated, not read in full)
| Source | Topic | Why not read in full |
|--------|-------|----------------------|
| AlphaArchitect, *Is Sector-neutrality in Factor Investing a Mistake?* | A | **WebFetch 403 Forbidden.** Snippet confirms Ehsani/Harvey/Li: long-only degrades from neutralizing in 78% of trials; **long–short momentum specifically degrades from sector neutrality**. Corroborates #1/#2. |
| Ehsani-Harvey-Li, FAJ Vol 79 No 3 (tandfonline abstract) | A | Paywalled abstract; primary already covered via #1 full read. |
| QuantRocket, *Sector Neutralization: Why It Matters* | A | Practitioner how-to; redundant with #2. |
| MOSEK *Portfolio Optimization Cookbook* v1.6 (Nov 2025) | A | Snippet: "weight/trade/leverage/turnover limits **should be softened**"; "**risk limits softened with a penalty term subtracted from the objective**, allowing small violations"; "soft max/min sector allocation via **slack variables**." Corroborates #6's soft-penalty formulation (official-docs tier). |
| arXiv 2507.07107 — ML multi-factor cross-sectional optimization | A | Confirms long-only constraint leaves long-short alpha on the table; not central. |
| MDPI *Sector Rotation Strategies TSX 60 (2000–2025)* | A | Sector-rotation ML/OOS validation; tangential. |
| Baeldung, *Saga Pattern in Microservices* | B | Redundant with #3/#4 canonical. |
| USPTO 8,601,479 *Multi-leg transaction processing* | B | Patent; look-ahead conflict resolution for multi-venue legs; corroborates atomicity challenge. |
| Wikipedia / academia.edu — *Compensating transaction* | B | Canonical definition; covered by #3/#4. |
| Temporal + AI Agents (DEV.to) | B | Workflow-orchestration durability for agentic systems; corroborates orchestration-saga choice. |
| arXiv 2607.02830 — *Precision Auditing of Filter Rules in DEX Trading: 2,400 Rejection Events* | C | Directly on classifying **rejection events by outcome rule** — corroborates the skip-reason ledger; not read in full (DEX-specific). |
| Arize, *Best AI Observability Tools for Autonomous Agents in 2026* | C | Tooling survey; per-task budget + tracing spans. |
| codieshub, *Prevent Infinite Loops and Spiraling Costs in Autonomous Agents* | C | "Hard caps on iterations/tokens/time/spend are non-negotiable; per-task budgets make runaway visible." |
| Superblocks, *AI Audit Trail: 7 Things to Log for Compliance in 2026* | C | Log rejections/refusals, not just successes. |

**URLs collected:** ~48 unique across 8 searches (floor 10). 7 read in full; ~14 recorded snippet-only above; remainder duplicative.

---

## 3. Recency scan (last 2 years) — MANDATORY SECTION

**New findings in the 2024–2026 window that complement/supersede older canon:**
1. **Soft-penalty diversification is the current frontier** (arXiv 2601.08717, Jan 2026; MOSEK Cookbook v1.6, Nov 2025): the modern treatment adds diversification (HHI) as a **weighted penalty term with auto-scaled magnitude**, or softens caps via **slack variables**, rather than imposing hard sector caps. This is newer than and directly supports a soft-tilt over the hard `sector_neutral` percentile re-score.
2. **Saga guarantees extended to LLM-driven multi-step plans** (SagaLLM, arXiv 2503.11951, Mar 2025): the classical Saga/compensating-transaction pattern (Garcia-Molina & Salem, 1987 — the year-less canon) now has a 2025 formalization for **agentic pipelines**, with an explicit "**no partial/inconsistent outcome**" guarantee and **independent validation** — highly apt for our LLM-orchestrated SELL+BUY swap.
3. **Rejection-event auditing is an active 2025–2026 topic** (VeritasChain Dec-2025 tamper-evident trails; arXiv 2607.02830 auditing 2,400 rejection events): the emphasis on **logging every skip/rejection as a first-class structured event** is recent practice, superseding the "log only successful orders" default that the pyfinagent silent gates currently exhibit.

**Canonical prior art still valid:** Ehsani, Harvey & Li (2021 working paper / FAJ 2023) remains the authoritative source on long-only vs long-short sector neutralization — no 2025–2026 work overturns its long-only conclusion; the newer soft-penalty papers operationalize *how* to keep sector exposure while shading concentration. Saga pattern (Garcia-Molina & Salem, 1987) remains the foundational reference.

**Internal recency anchor:** the project's own **2026-06-01 replay** (`scripts/ablation/sector_neutral_replay.py`) measured **hard sector-neutral = −0.166 long-only Sharpe** (screener.py:71-73) — the binding in-house evidence the soft design must beat.

---

## 4. Internal code audit (grounds the design)

Files inspected: `handoff/current/confirmed_findings.json` (full 481-line register: 17 confirmed / 8 refuted),
`goal_phase70_trade_diversity_DRAFT.md`, `backend/services/autonomous_loop.py`,
`backend/services/portfolio_manager.py`, `backend/tools/screener.py`, `backend/services/paper_trader.py`.

### (a) Diversification funnel
- **`screener.py:249` `rank_candidates`** — pure momentum composite `mom_1m*0.40 + mom_3m*0.35 + mom_6m*0.25` (:299-303) with RSI/vol penalties; sort+truncate to `top_n` at **:483-484**. In a semis-led regime the top-N is monosector (S2).
- **`screener.py:450-474` `sector_neutral`** — the EXISTING hard lever: replaces `composite_score` with the **within-sector percentile rank [0,1]** (`_apply_pct_rank`), raw preserved on `composite_score_raw`. This **destroys the across-sector momentum signal** (a +40% semi and a +2% utility each top their sector → both ≈1.0). **Default OFF.**
- **`screener.py:64-73` `build_sector_map`** — gives candidates a GICS sector AT rank time; docstring states verbatim: *"a 2026-06-01 replay found HARD sector-neutral HURTS long-only Sharpe (-0.166); the flag stays OFF -- this wiring keeps the lever live-measurable **for a future SOFT-tilt variant**."* ← phase-70.2 is that variant.
- **`autonomous_loop.py:837-838`** — `analyze_tickers = new_candidates[:settings.paper_analyze_top_n]` (default 5). Sector enrichment (`:744-761`) currently runs and is available; the slice is NOT sector-aware. (findings #1, #2)

### (b) Swap / rotation path
- **`portfolio_manager.py:339-458`** main buy loop: count cap `:359-369` (queues overflow to `sector_blocked`), `<$50` floor `:384`, `buy_amount = min(target, available_cash)` `:381`, NAV-pct cap `:394-408`, factor-corr cap `:413-424` (default 0=off), running `available_cash -= buy_amount` `:450`.
- **`portfolio_manager.py:498-709` `_compute_swap_candidates`** — **SAME-SECTOR ONLY**: `sector_holdings = holdings_by_sector.get(cand_sector, [])` (:594) → can never rotate into a NEW sector. Sentinel churn: flag-OFF assigns un-reeval'd holding `score=0.0` (:562-566) with `denom=0.01` (:620) → `delta_pct ≈ cand_score*10000` ≫ 25% bar → any candidate clears (finding #3, AW-5). Swap BUY sized `buy_amount = nav*position_pct/100` (:675) with **NO `min(available_cash)` cap and NO `<$50` floor** — unlike the main loop. `available_cash` is **not passed** into the function (finding #9).
- **`autonomous_loop.py:1262-1320` execution — NON-ATOMIC**: ALL SELLs (`:1262-1275`) then ALL BUYs (`:1284-1320`). `paper_trader.py:197-199` returns None when `total_cost > cash` (WARN-only, no rollback) → a swap SELL fires but its paired BUY silently drops → **net −1 position, sector hole** (finding #9). `trades_executed` only increments on success (:1274, :1320); no summary counter for drops.
- Existing churn mitigation `paper_swap_churn_fix_enabled` (default OFF): clamps `denom=max(abs(score),1.0)` (:620) + excludes un-reeval'd holdings from displacement (:539-561). The atomic-swap design must ship on top of this.

### (c) BUY-gate observability
- **`autonomous_loop.py:90`** `_SESSION_BUDGET_USD = float(os.getenv("PYFINAGENT_SESSION_BUDGET_USD","1.0"))` — **half** the visible `paper_max_daily_cost_usd=2.0`. `_check_session_budget` (:95-105) **raises BudgetBreachError with NO log**. Raised in `_run_and_persist_one` (:925), which runs under `asyncio.gather(..., return_exceptions=True)` (:966-969); the result filter `isinstance(r, dict)` (:970) **silently discards the exception**. The intended clean cycle-halt (`status='budget_breach'`, :1436-1448) is **defeated** by `return_exceptions=True`. Both accumulators reset per cycle (`total_analysis_cost=0`, `_session_cost=0`), so $1 is the tighter, quieter ceiling. (findings #6/#7/#8)
- **`paper_trader.py:172-188`** price-tolerance gate: WARN-only `return None`, no BQ row, no summary counter.
- **`portfolio_manager.py:346/363/385/401`** position/sector/NAV/$50 skips DO `logger.info` — but only to backend.log; none reach the cycle summary or a BQ ledger.
- **`paper_trader.py:303-308`** non-US add-on avg-entry unit mix (`new_cost` USD ÷ `new_qty` local shares) — finding #10 (money-path, non-US only; adjacent to 70.3).

---

## 5. Design recommendations

### (a) SOFT profit-aware sector diversification — **recommend a two-part soft design, flag-gated, DARK-until-token**

**Why not hard `sector_neutral`:** the within-sector percentile re-score (screener.py:450-474) discards the
across-sector momentum long-only needs. Both the **internal −0.166 Sharpe replay** and **Ehsani-Harvey-Li**
(long-only better keeping sector in **78% of trials**) say the same thing: for a long-only book, keep the
across-sector signal. Hard neutralization is the wrong tool.

**Options compared:**
| Option | Where | Pro | Con |
|--------|-------|-----|-----|
| **min-K-sectors / round-robin fill** on the analyze slice | `autonomous_loop.py:838` | Cheap; selection-stage only; leaves the momentum score untouched for sizing; guarantees ≥K sectors reach the analyzer (structurally enables S2) | Can pull one weaker cold-sector name into *analysis* — but the downstream BUY-lean + RiskJudge still filter it, so it's funnel-widening, not a forced buy |
| **Soft diversity penalty on the rank score (HHI-style)** | `screener.py` rank stage | Continuous & **profit-aware** — a dominant momentum name still wins; single tunable knob `w_d`; `w_d=0` = byte-identical; this IS the "soft-tilt variant" screener.py:72 reserves | Needs sector at rank time (already provided by `build_sector_map`); needs θ-scaling so the penalty is commensurate with the momentum composite |
| Hard sector-neutral percentile (existing) | `screener.py:450` | (reduces concentration) | **Rejected** — measured −0.166 long-only Sharpe; kills across-sector alpha |

**Recommended approach (both parts, both default-OFF):**
1. **Primary — soft diversity penalty at rank time.** Shade `composite_score` by the candidate's sector
   representation among {held positions ∪ already-higher-ranked candidates}. Concretely a rank-decayed
   penalty: the *j*-th candidate in a given sector keeps its order but is multiplied by `(1−w_d)^(j−1)`
   (or minus `λ·count_in_sector`), so later same-sector names are shaded and cross-sector names surface —
   directly analogous to arXiv 2601.08717's `−w_d·θ₁·HHI(x)` objective term with `θ₁` auto-scaled to the
   momentum magnitude. Because it only **shades, never zeroes**, a genuinely dominant sector still wins its
   share (respects −0.166; keeps the across component per Ehsani-Harvey-Li).
2. **Secondary (belt-and-suspenders) — min-K-sector round-robin on the analyze slice** at
   `autonomous_loop.py:838`: iterate sectors in momentum-rank order, take the best unpicked candidate from
   each, repeat until `paper_analyze_top_n` filled, guaranteeing the analyzed set spans ≥K GICS sectors.
   (Ensure sector enrichment runs BEFORE the slice — finding #2.)
3. **Robust sector attribution** (findings #5/#14): exempt the `"Unknown"` bucket from the count/NAV caps
   (or bucket per-ticker) so enrichment failure can't collapse distinct sectors into one shared cap and
   freeze the funnel.
4. **Validation gate (north-star do-no-harm):** run `scripts/ablation/sector_neutral_replay.py` (the SAME
   harness that produced −0.166) on the soft-penalty variant across a `w_d` / `K` grid; ship the activation
   token ONLY if OOS Sharpe ≥ incumbent AND it beats the incumbent per the promotion gates (DSR≥0.95,
   PBO≤0.5). Flag-gated, `w_d=0`/`K=0` byte-identical, DARK-until-token, ON-vs-OFF $0 diff.

### (b) ATOMIC cash-bounded cross-sector swap — **recommend pre-flight-validation saga (orchestration), depends-on churn-fix**

The SELL+BUY swap is a 2-step distributed transaction over two BQ writes with **no ACID guarantee** — the
canonical Saga problem. Design (grounded in microservices.io Saga + SagaLLM):

1. **Cash-bound the swap BUY by construction (pivot-transaction discipline).** Size
   `buy_amount = min(nav*pct/100, running_available_cash)` and apply the `<$50` floor, mirroring the main
   loop (:381,:384). Thread a **shared running `available_cash` tracker** into `_compute_swap_candidates`
   (currently not passed — finding #9): SELL credits the weakest holding's `market_value`, BUY debits.
   This makes each pair self-funding (SELL frees ≥ BUY needs) — the cheapest fix, often sufficient alone.
2. **Pre-flight aggregate validation (SagaLLM "either fully committed or coherent rollback to S").** Before
   executing ANY order, simulate the full ordered list (SELLs credit, BUYs debit, running balance ≥ min_cash).
   If a swap's paired BUY would be under-funded, **drop the WHOLE pair (both legs) pre-execution** — never
   execute a half-swap. This is an independent-validation checkpoint (SagaLLM GlobalValidationAgent analog;
   also the harness Q/A doctrine: not self-checked), applied before commit — preferred over post-hoc
   compensation for a paper engine because it avoids executing-then-reversing fills and keeps the
   sell-first-then-buy invariant.
3. **Compensating-transaction fallback (true saga)** if any half-swap ever executes: tag each swap BUY with
   its paired SELL id; on `execute_buy → None`, emit a compensating **BUY-back** re-opening the just-sold
   holding (or, simpler, treat the SELL as the pivot and only commit it once the BUY confirms). Keep this as
   the defense-in-depth layer behind the pre-flight check.
4. **Cross-sector capability (the "changeable fund" intent).** When the book is at `max_positions` and a
   strong new-sector candidate is blocked, allow a **cross-sector** swap: SELL the weakest-overall holding
   (by TRUE same-cycle score) and BUY the new-sector name, IF it (i) lowers portfolio HHI (improves
   diversification) AND (ii) clears the conviction delta on clamped denom. When slots are FREE the main loop
   already buys other-sector names (per the refuted same-sector finding), so cross-sector swap only matters
   at full capacity.
5. **Churn-safety dependency (must-carry).** Ship the atomic/cross-sector swap ONLY on top of the churn-fix
   (`denom=max(abs(score),1.0)` + exclude un-reeval'd holdings, `paper_swap_churn_fix_enabled`), else
   cross-sector rotation becomes a bigger churn engine. Recommend the new swap flag **depends on churn-fix
   ON** (or fold both into one flag). Flag-gated default-OFF; paper-only; do-no-harm (no risk-limit moves);
   ON-vs-OFF $0 diff.

### (c) BUY-gate observability — **recommend a structured skip-reason ledger + budget reconciliation**

Grounded in VeritasChain REJ-event schema + oneuptime "budget as observable metric" + SagaLLM durability.

1. **Structured skip-reason ledger (the REJ pattern).** Every BUY-gate that drops a candidate emits a
   structured event `{cycle_id, ticker, stage, skip_reason, decision_factors:{…}, ts}` into BOTH the cycle
   `summary["skips"]` and a durable BQ ledger (e.g. `paper_trades.buy_gate_skips` or a JSONL), so **0-buy
   cycles are diagnosable without log forensics**. Enumerated `skip_reason`s: `session_budget`,
   `daily_cost_cap`, `position_cap`, `sector_count_cap`, `sector_nav_cap`, `factor_corr_cap`, `min_50_floor`,
   `price_tolerance`, `insufficient_cash`, `hold_or_non_buy`, `degraded_analysis`. This is exactly
   VeritasChain's "rejections & skips are distinct REJ events carrying skip_reason + decision_factors, never
   omitted" and arXiv 2607.02830's rejection-event classification.
2. **Fix the swallowed budget breach.** (a) `logger.warning` inside `_check_session_budget` BEFORE it raises
   (currently silent, :99-105); (b) set `summary["budget_truncated"]=True` + dropped-ticker count; (c) either
   detect `BudgetBreachError` in the gather results and set the summary flag, or (cleaner) **check the budget
   before dispatch** so the breach halts the fan-out with a logged, summarized reason instead of being eaten
   by `return_exceptions=True` (:966-970).
3. **Reconcile the two budgets (the actual bug).** Make the hidden session budget **derive from / never be
   lower than** the visible daily cap: default `_SESSION_BUDGET_USD = paper_max_daily_cost_usd` (single knob),
   OR expose `PYFINAGENT_SESSION_BUDGET_USD` as a real UI-visible setting with a validation `session ≤ daily`
   and a surfaced warning when `session < daily`. Surface BOTH `session_cost_usd` and `daily_cost_usd` on the
   cycle summary + Harness/Paper UI (oneuptime: "add spend as a metric; alert on anomalies").
4. **Do-no-harm split.** Logging + surfacing are **read-only/additive** (no trade-decision change) → can ship
   un-flagged. Only the **ceiling change** (raising the session budget) alters effective spend → **flag-gated
   default-OFF** and subject to the $0/cost check + Peder approval per the phase-70 boundaries.

**North-star framing:** "more trades" = more diversified, higher-quality deployment + more clean round-trips
for the learning loop — NOT churn of the same 20 names. (a)+(b)+(c) together widen the funnel, rotate the
fund honestly, and make starvation diagnosable, WITHOUT buying diversification with returns (every activation
is gated on a paper/backtest do-no-harm check).

---

## 6. Gate envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 14,
  "urls_collected": 48,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "gate_passed": true
}
```
