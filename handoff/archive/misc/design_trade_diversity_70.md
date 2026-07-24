# Design pack — phase-70 (Trade diversity + changeable fund)

Step 70.0 GENERATE. Offline, $0, **no production code changed** — this is the implementation design for
the downstream steps 70.1–70.5. Grounded in `research_brief_70.0.md` (gate_passed=true, 7 sources in full)
and `confirmed_findings.json` (17 confirmed). Every live-loop behavior change below ships **flag-gated,
default-OFF (DARK-until-token)** with an ON-vs-OFF `$0` diff; the risk-sector-caps as *risk limits*, stops,
kill-switch and DSR≥0.95/PBO≤0.5 promotion gates are **byte-untouched**.

---

## (a) SOFT, profit-aware sector diversification  →  step 70.2 (S2)

### Why NOT hard sector-neutralization
The existing `screener.rank_candidates(sector_neutral=…)` lever does WITHIN-sector percentile re-scoring
(`screener.py:450-474`), which discards the *across-sector* momentum that a long-only book earns from.
- Internal 2026-06-01 replay (`scripts/ablation/sector_neutral_replay.py`, comment at `screener.py:71-73`):
  hard sector-neutral measured **-0.166 long-only Sharpe**.
- Ehsani, Harvey & Li, *"Is Sector Neutrality in Factor Investing a Mistake?"* (FAJ, May 2023): long-only is
  more likely to BENEFIT from KEEPING sector exposure (78% of trials); neutralize only iff
  `Sharpe(across/within) < corr(across,within)`, which rarely holds long-only.
So we do **NOT** flip the hard lever. We add a SOFT tilt.

### Two-part soft design (both flag-gated, default-OFF = byte-identical)
1. **PRIMARY — soft diversity penalty at rank time.** Shade (never zero) `composite_score` by the
   candidate's sector representation among {held ∪ higher-ranked candidates}: the j-th same-sector name is
   multiplied by `(1 - w_d)^(j-1)`. This is the profit-aware analog of arXiv 2601.08717's `-w_d·θ₁·HHI`
   objective term, with `θ₁` auto-scaled to the momentum magnitude so a genuinely dominant sector still
   wins its share. `w_d = 0` ⇒ byte-identical to today. Flag: `paper_soft_sector_diversity_enabled`
   (default OFF) + `paper_soft_sector_diversity_w` (default 0.0). Touches `backend/tools/screener.py`
   (`rank_candidates`) — a NEW soft-penalty path, the hard `sector_neutral` lever left as-is.
2. **SECONDARY — min-K-sector round-robin on the analyze slice.** Move sector enrichment BEFORE the
   top-N truncation at `backend/services/autonomous_loop.py:838` and guarantee ≥K distinct GICS sectors
   reach the analyzer (round-robin fill), so cross-sector BUY candidates can even be generated (the
   structural enabler of S2). Flag: `paper_min_k_sectors_analyzed` (default 0 = today's behavior).
3. **"Unknown" bucket fix (findings #5/#14):** exempt the `Unknown` sector from the count/NAV caps
   (`portfolio_manager.py:272/319/360`) so a ticker-meta enrichment failure can't collapse all candidates
   into one bucket and freeze the funnel at 2.

### Validation gate (before ANY activation token)
Run `scripts/ablation/sector_neutral_replay.py` (the same harness that produced -0.166) over a `w_d × K`
grid. Grant the activation token ONLY if OOS Sharpe ≥ incumbent AND the change clears DSR≥0.95 / PBO≤0.5.
`historical_macro` stays FROZEN (replay uses existing cached data; no BQ writes, no optimizer run).

---

## (b) ATOMIC, cash-bounded, cross-sector swap / rotation  →  step 70.3 (S3 + money-path)

### The bug class
The sector-blocked swap is a **two-leg** (SELL→BUY) distributed transaction over two BigQuery writes with
no ACID guarantee — the canonical **Saga** problem (microservices.io; SagaLLM, arXiv 2503.11951). Current
defects:
- `_compute_swap_candidates` is **same-sector only** (`portfolio_manager.py:594`) → churn within tech,
  never rotation into a new sector.
- The swap BUY is **uncapped by cash** and skips the $50 floor (`portfolio_manager.py:675`;
  `available_cash` not threaded in).
- Execution is **non-atomic**: `autonomous_loop.py:1262-1320` does all SELLs then all BUYs; `execute_buy`
  returns `None` when `total_cost > cash` (`paper_trader.py:197-199`), so a SELL fires and its paired BUY
  silently drops → **net −1 position** (finding #9).
- Still fires on the fabricated 0.0-conviction / 0.01-denominator churn sentinel (finding #3, AW-5).

### Design (flag-gated, default-OFF, paper-only, sell-first invariant preserved)
1. **Pre-flight aggregate validation (the Saga "either fully committed S′ or coherent rollback to S").**
   Simulate the full ordered SELL/BUY list BEFORE executing anything; if the paired BUY would be
   under-funded, **DROP THE WHOLE PAIR (both legs)** — never execute a half-swap. This is an independent
   pre-check (not agent self-validation), consistent with SagaLLM's GlobalValidationAgent and our Q/A
   independence doctrine. No reversal of already-written paper fills.
2. **Cash-bound the swap BUY by construction:** `min(nav·pct/100, shared_running_available_cash)`, apply the
   $50 floor, and thread the SAME running-cash tracker the main loop uses so each pair is self-funding.
3. **Compensating BUY-back** as defense-in-depth if a half-swap ever executes despite (1) (Saga
   compensation / rollback).
4. **Cross-sector rotation at `max_positions`:** SELL weakest-overall (true score) + BUY a new-sector name
   iff it lowers portfolio HHI and clears the conviction delta on a clamped denominator — the "changeable
   fund" rotation the operator wants.
5. **Depends on `paper_swap_churn_fix`** (denominator `max(abs(score),1.0)` + exclude un-re-evaluated
   holdings): the new atomic-swap flag must ship on top of it or declare a dependency.
Flags: `paper_atomic_swap_enabled`, `paper_cross_sector_rotation_enabled` (both default OFF).
Touches `backend/services/portfolio_manager.py` (`_compute_swap_candidates`, swap sizing) +
`backend/services/autonomous_loop.py:1262-1320` (the execute ordering / pre-flight). No risk-limit moves.

---

## (c) BUY-gate observability + budget reconciliation  →  step 70.4 (S3)

### The bug
`_SESSION_BUDGET_USD = 1.0` (`autonomous_loop.py:90`) is HALF the operator-visible
`paper_max_daily_cost_usd = 2.0`. `_check_session_budget` raises `BudgetBreachError` with **no log**
(`:99-105`), raised inside `_run_and_persist_one` (`:925`) under `asyncio.gather(return_exceptions=True)`
(`:966-969`) then filtered out by `isinstance(r, dict)` (`:970`) — **silently swallowed**; the intended
clean cycle-halt (`:1436-1448`) is defeated. Other silent gates: price-tolerance WARN-only `return None`
(`paper_trader.py:182`); position/sector/NAV/$50 skips log only to `backend.log`, never the cycle summary
or BQ.

### Design (grounded in VeritasChain REJ-event schema + arXiv 2607.02830 rejection-auditing)
1. **Structured skip-reason ledger.** Every dropped candidate emits
   `{cycle_id, ticker, stage, skip_reason, decision_factors, ts}` into `summary['skips']` AND a durable BQ
   ledger. Enumerate `skip_reason` ∈ {session_budget, daily_cost_cap, position_cap, sector_count_cap,
   sector_nav_cap, factor_corr_cap, min_50_floor, price_tolerance, insufficient_cash, hold_or_non_buy,
   degraded_analysis} so 0-buy cycles are diagnosable without log forensics. (This is the "gate visibility"
   / observability lever.)
2. **Fix the swallowed breach:** log a WARNING before raising, set `summary['budget_truncated']` + the
   dropped count, and either check the budget BEFORE dispatch or detect `BudgetBreachError` in the gather
   results instead of letting `return_exceptions=True` eat it.
3. **Reconcile the budgets:** default the session budget to `paper_max_daily_cost_usd` (single knob) OR
   expose it as a UI setting with `session ≤ daily` validation and a surfaced warning; surface both
   `session_cost_usd` and `daily_cost_usd` on the summary/UI.

### Do-no-harm SPLIT
Logging + surfacing the skip-reason ledger and the budget values are **read-only / additive** → ship
**un-flagged** (they change no trading decision). ONLY the budget-ceiling change (raising the hidden $1
toward $2) is **flag-gated default-OFF** + needs the `$0` cost check + Peder's LLM-cost approval.

---

## Downstream step map (design → implementation)

| Design block | Step | Symptom | Flags (all default-OFF) |
|---|---|---|---|
| S1 UI: clear-snapback, risk_overrides surface/clear, knob editors | 70.1 | S1 | (frontend; no live-loop flag) |
| (a) soft diversification + Unknown-bucket fix | 70.2 | S2 | paper_soft_sector_diversity_enabled / _w, paper_min_k_sectors_analyzed |
| (b) atomic cross-sector swap + non-US avg-entry fix | 70.3 | S3/money | paper_atomic_swap_enabled, paper_cross_sector_rotation_enabled (dep: paper_swap_churn_fix) |
| (c) skip-reason ledger + budget reconciliation | 70.4 | S3 | ledger/logging un-flagged; ceiling change flag-gated |
| Starting-capital display + cron reschedule | 70.5 | general | (no live-loop flag) |

## Boundaries reaffirmed (binding)

Flag-gated **DARK-until-token** on every live-loop behavior change; **$0 metered**, free APIs only,
**paper-only**; **no change to the risk-sector-caps as risk limits**, stops, kill-switch, or DSR/PBO gates;
**a paper/backtest gate (Sharpe ≥ incumbent + DSR≥0.95/PBO≤0.5) before any diversification activation
token**; hysteresis banned; `historical_macro` frozen; harness stays exactly 3 agents (Main + Researcher +
Q/A). 70.0 itself changes none of these — it is design + research only.

## Rejected alternatives (from the research)
- **Hard sector-neutralization** (flip the existing lever): rejected — -0.166 replay Sharpe + FAJ 2023.
- **Two-phase-commit for the swap:** rejected — heavyweight/locking vs the lightweight Saga pre-flight
  validation that fits a single-process paper loop.
- **Raising the session budget un-gated:** rejected — it is a cost/behavior change, must be flag-gated +
  approved; only the *observability* of it ships un-flagged.
