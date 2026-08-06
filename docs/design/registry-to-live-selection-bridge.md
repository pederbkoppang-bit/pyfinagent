# Registry-to-live selection bridge -- DESIGN ONLY

**Status:** design ratified, **NOT built**. No selection is wired by the step
that produced this document (phase-82.6).
**Date:** 2026-08-06.
**Research basis:** `handoff/archive/phase-82.6/research_brief.md`
(audit-class, 13 rounds, dry; 6 sources read in full, 34 URLs, 20 files).

> The live paper book is working. A selection-path change is the
> highest-regression-risk edit in this system. This document exists so that when
> the bridge is built, it is built deliberately -- and so that the two hazards in
> §2 and §5 are decisions rather than side effects.

---

## 1. Measured current behaviour

Every claim below was re-derived from source on 2026-08-06. Line numbers rot;
re-derive before trusting them. Paths are given in full: there are TWO modules
named `autonomous_loop.py` -- `backend/autonomous_loop.py` is the phase-3.3
planner/evaluator harness, and `backend/services/autonomous_loop.py` is the live
trading cycle. Everything below means the latter.

| What | Where | Reality |
|------|-------|---------|
| Params load | `backend/services/autonomous_loop.py:431` | `load_promoted_params(bq)` (`:46-74`) -- a 3-tier loader preferring BQ `promoted_strategies`, with `optimizer_best.json` only as **fallback**. It does not read the JSON directly. |
| Sharpe | `:433` | `summary["best_params_sharpe"]` -- display only |
| Exit params | `:434-437` | `summary["strategy_params"]` -- **ZERO readers repo-wide**, frontend included |
| Strategy name | `:1649` | audit label written to `strategy_decisions` |
| Router | `:1644` comment | *"strategy router (deferred to phase-31)"* |

**Nothing the optimizer produces changes live trading behaviour today.** One
display number and one audit label. `tp_pct` / `sl_pct` appear nowhere else in
`backend/services/`; live exits come from
`settings.paper_default_stop_loss_pct` (`backend/services/paper_trader.py:298-304`) and the Risk
Judge (`backend/services/portfolio_manager.py:842-880`).

**A correction worth recording**, because it caused a real misreading: the
source spec (`incumbent_live_strategy_spec.md:35`) said the params cross over
**"as display fields"**. The phase-82.6 masterplan step dropped the qualifier and
asserted "the registry's EXIT PARAMS cross over". The qualifier was the whole
meaning.

`strategy_decisions` records `decided_strategy == prior_strategy`, but note this
is true **by construction** (`:1654-1655` assign the same variable), not as an
empirical finding. Measured live: 51 rows, 50 equal (all `cycle_heartbeat`), 1
unequal (`reduce_position` via `decay_signal`). Last write 2026-07-31.

## 2. HAZARD -- the bridge's natural first wire changes live risk behaviour

`backend/services/paper_trader.py:1425-1428`:

```python
        if pos.get("stop_advanced_at_R"):
            entry_strategy = (pos.get("entry_strategy") or "").lower().strip()
            if entry_strategy in {"mean_reversion", "pairs"}:
                return (None, None)
```

This is **deliberate and cited**, not a defect -- Kaminski-Lo Proposition 2:
mean-reverting strategies and cointegrated pairs lose expected return when
trailing-stop cumulative-loss thresholds fire. Its default is already fail-safe:
unknown strategy is treated as momentum, so the trail **is** applied.

**But the branch is unreachable today.** `paper_positions.entry_strategy` is
NULL on every row (measured live 2026-08-06; the table holds 1 row) and no writer
exists -- while
`scripts/migrations/phase_32_2_add_entry_strategy.py:16-17` already names that
column as the intended wire from `strategy_decisions.decided_strategy`.

**Note the precondition, because it bounds the blast radius.** The skip sits
under `if pos.get("stop_advanced_at_R")` -- it applies ONLY to positions that
have already advanced past the breakeven ratchet, not to every position. An
earlier draft of this document quoted the block without that first line and
concluded "every position is trailed"; that overstated it, and the corrected
scope is: *positions past the breakeven ratchet* are trailed today.

**Therefore:** the first, most natural bridge wire activates a dormant live
risk-behaviour change as a side effect of a change whose stated purpose is
selection. The day `entry_strategy` is populated, `mean_reversion` and `pairs`
positions **that have advanced past breakeven** stop being trailed.

**Design requirement:** populating `entry_strategy` MUST be a separately flagged,
separately reviewed change from wiring selection -- never the same commit, and
never an implicit consequence. Whoever builds this states the expected change in
trailing-stop coverage *before* the first population run.

## 3. Insertion points -- there are TWO, not one

**(i) The params seam, `backend/services/autonomous_loop.py:431`.** Already correct and inert.
But there is nowhere to hand a strategy today: `decide_trades`
(`backend/services/portfolio_manager.py:66-73`) takes **no params argument**. Its only config
channels are `settings` and `risk_overrides.get_effective()`, and
`risk_overrides.ALLOWED_KEYS` (`:57`) holds 4 numeric keys with bounds -- it
cannot carry a string strategy name without extending `BOUNDS`.

**(ii) `paper_trader.execute_buy` -> `paper_positions.entry_strategy`.** This is
the §2 hazard, and it is per-position rather than per-cycle.

A selection that changes *which trades are taken* must land at (i). A selection
that changes *how a position is managed* lands at (ii). Conflating them is the
main design error available here.

## 4. Promotion gate -- compose what exists, invent nothing

| Gate | Where | Thresholds | Status |
|------|-------|-----------|--------|
| `PromotionGate` | `backend/autoresearch/gate.py:21-30` | `min_dsr 0.95`, `max_pbo 0.20`, `min_pbo_trials 10` | **the live one** |
| `evaluate_stage` | `backend/services/promotion_gate.py:34-63` | stages `[0.05, 0.25, 1.0]`, `MIN_LIVE_DAYS [14, 30]`, `PBO_CEILING 0.5` | script-reachable only |

Note `backend/autoresearch/promotion_gate.py` **does not exist**; the 0.50
ceiling is in `backend/services/promotion_gate.py:37` and is a separate staging
gate, not the research gate.

**A strategy may select live trades only after it clears `PromotionGate` and
then walks `evaluate_stage`'s capital ladder** (5% -> 25% -> 100%, with the
minimum live-day dwell at each rung). The ladder is the mechanism that keeps a
selection error small enough to survive.

### HARD PREREQUISITES, measured -- this cannot be built today

- `optimizer_best.json` has **no `pbo` key at all**.
- `PromotionGate` is **fail-closed** on a missing pbo.

So **82.23** (the promotion gate's PBO term is never computed) and **82.26** (the
PBO trial floor) are **build-time blockers**, not advisories. Building the bridge
before them yields a gate that either rejects everything or is bypassed.

## 5. Rollback -- entirely existing machinery

1. **Kill switch** -- `backend/services/autonomous_loop.py:1313-1322`. Stops the cycle outright.
2. **Deactivate the `promoted_strategies` row.** `load_promoted_params` is
   fail-open to `optimizer_best.json`, so *removing the promotion IS the
   rollback* -- no new machinery, no deploy.
3. **`rollback.py::auto_demote_on_dd_breach`** -- automatic on drawdown.
4. **`evaluate_stage` regress-to-5%** -- shrink exposure without full revert.
5. **`strategy_decisions`** -- the audit trail for what was decided and why.
6. **Default-OFF flag**, per the standing project idiom.

Rollback (2) does **not** unwind §2: positions already written with an
`entry_strategy` keep it, so their trailing-stop treatment persists after the
selection is reverted. **Any rollback plan must state what happens to in-flight
positions**, not just to the selector.

## 6. What is already built, and dark

`backend/autoresearch/strategy_selector.py` (phase-47.6) is **complete, tested,
and has zero callers ON THE LIVE TRADING PATH.** Be precise here, because the
looser claim is false: `select_best_strategy` IS called from
`backend/autoresearch/strategy_candidate_producer.py:181`, reached via
`backend/autoresearch/rotation_runner.py:53` (`run_strategy_bakeoff`). Those are
production modules. What none of them has is a path into the live cycle -- which
is the claim that matters, and the one that makes this a deployment problem. Its docstring (`:13-17`) specifies this exact
bridge ("no new read path"), and its Shu/Yu/Mulvey (2024) citation checks out:
S&P 500 turnover 141% -> 44% under a jump/switch penalty, net Sharpe 0.68 vs
0.48 buy-and-hold at 10bps one-way.

Steps 47.6 / 48.1 / 48.2 / 48.3 / 48.4 are all `done`; 48.3's own name records
the *deployment bridge* as the deferred piece. `run_friday_promotion` has **no SCHEDULED caller** -- **25** invocations exist across the tracked repo (987 `.py` files via `git ls-files`): 12 in `tests/autoresearch/test_friday_promotion.py`, 7 in `scripts/harness/phase10_friday_promotion_test.py`, 4 in `tests/autoresearch/test_slot_usage_wiring.py`, 2 in `tests/verify_phase_25_A3.py` -- every one a test or harness call, none a scheduler or production caller.
`cron_budget.yaml`'s `friday_promotion_gate` and `sprint_calendar.yaml`'s
`fri_promotion` are ledger/plan slots, and `slot_accounting.py` carries the
name only as a logged string -- none of them invokes it.

**So this is a deployment problem, not a design problem.** Do not re-design the
selector.

## 7. External evidence that raises the bar

arXiv 2603.20319 measures **rotation strategies as the worst case for
cross-engine backtest divergence** -- up to 3.71% at realistic costs versus
0.0000% at zero cost, with Spearman rho 0.93 between divergence and cost
intensity -- and recommends **>= 2 independent validators** before trusting a
rotation result. This bridge rotates strategies, so a single backtest engine
agreeing with itself is not sufficient evidence to promote.

## 8. Build order (when 82.23 and 82.26 are closed)

1. Close 82.23 + 82.26 so `PromotionGate` has a real PBO to judge.
2. Wire `strategy_selector` in **shadow mode**: decide, record to
   `strategy_decisions`, change nothing.
3. Compare shadow decisions against the incumbent for >= `MIN_LIVE_DAYS`.
4. **Separately**, and behind its own flag, decide the §2 trailing-stop
   question and populate `entry_strategy`.
5. Only then let a decision reach seam (i), entering at the 5% rung.

Steps 4 and 5 are deliberately in that order and deliberately separate.
