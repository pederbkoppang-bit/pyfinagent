# Contract -- phase-82.6

**Step:** 82.6 (P2) -- DESIGN (not build) the registry-to-live selection bridge.
**Date:** 2026-08-06. **Cycle:** 1.
**Research gate:** PASSED -- `handoff/current/research_brief_82.6.md`,
`gate_passed: true`, **audit_class** with `dry: true` after 13 rounds / 2 dry,
6 sources read in full, 34 URLs, 20 internal files.

---

## 1. BOTH halves of the step's premise are refuted

The step says: *"the registry's EXIT PARAMS cross over; its SELECTION LOGIC does
not."* Measured by me, then independently by the gate:

**(a) The exit params do NOT cross over either.** `best_params` has exactly 5
references in `autonomous_loop.py` (`:431,:432,:433,:435,:1649`).
`summary["strategy_params"]` (`:434-437`) has **zero readers repo-wide**,
frontend included. `tp_pct` / `sl_pct` appear nowhere else in
`backend/services/`. Live exits come from `settings.paper_default_stop_loss_pct`
(`paper_trader.py:298-304`) and the Risk Judge via
`portfolio_manager.py:842-880`.

The provenance of the error is instructive: the source doc
(`incumbent_live_strategy_spec.md:35`) said the params cross over **"as display
fields"**. The masterplan step dropped the qualifier, and the qualifier was the
whole meaning.

**So today, nothing the optimizer produces changes live trading behaviour.**
One display number and one audit label.

**(b) Minor:** `:431` does not read `optimizer_best.json` directly. It calls
`load_promoted_params(bq)` (`:46-74`), a 3-tier loader preferring BQ
`promoted_strategies`, with the JSON only as fallback. All five cited anchors do
resolve.

## 2. THE HEADLINE -- a strategy name already gates a LIVE RISK CONTROL

The gate found what the step missed, and it inverts the risk framing.

`backend/services/paper_trader.py:1425-1428` branches on
`entry_strategy in {"mean_reversion", "pairs"}` and **returns early, skipping the
HWM trailing stop.** `mean_reversion` is a `STRATEGY_REGISTRY` key. The branch is
unreachable today: `paper_positions.entry_strategy` is NULL on every row
(measured live 2026-08-06 -- the table currently holds **1** row), and no writer
exists. Yet `scripts/migrations/phase_32_2_add_entry_strategy.py:16-17`
**already names that column as the intended bridge wire** from
`strategy_decisions.decided_strategy`.

**I read the branch before characterising it, and it is NOT a defect** -- the
gate's "silently disarms a risk control" framing overstates it, and I am not
going to repeat that. The in-code comment is explicit and cited:

> *Kaminski-Lo Proposition 2: mean-reverting strategies (and cointegrated pairs)
> lose expected return when trailing-stop cumulative-loss thresholds fire; SKIP
> for those. Fail-CLOSED-conservative: when entry_strategy is None/unknown, treat
> as momentum (trail IS applied) -- forgetting to flag a mean-reversion entry
> should err toward "more protection", not "no protection".*

So the skip is deliberate, research-backed, and its default is already fail-safe.

**The real hazard is one of sequencing, and it is still the most important thing
this design must say:** the bridge's natural first wire -- populating
`entry_strategy` -- **activates a dormant live risk-behaviour change as a SIDE
EFFECT of a change whose stated purpose is strategy selection.** Today every
position is trailed; the day that column is populated, `mean_reversion` and
`pairs` positions stop being trailed. That is intended behaviour arriving
unintentionally. The design must make it an explicit, separately-gated decision
rather than a consequence, which is the opposite of the step's assumption that
selection logic is the risky part and params are safe.

## 3. The bridge was already designed -- ratify, do not re-design

`backend/autoresearch/strategy_selector.py` (phase-47.6) is **complete, tested,
and DARK** with zero production callers; its docstring `:13-17` specifies the
exact bridge ("no new read path"). Steps 47.6 / 48.1 / 48.2 / 48.3 / 48.4 are
all `done`, and 48.3's own name says the *deployment bridge* is the deferred
piece. `run_friday_promotion` has no caller anywhere.

So this step **documents and ratifies an existing design**, adds the §2 hazard
it does not currently carry, and states the prerequisites. Re-designing would
duplicate settled work.

## 4. Immutable success criteria (verbatim)

1. "docs/ contains a bridge design naming the exact insertion point in the live
   cycle, the promotion gate a strategy must clear before it may select trades,
   and the rollback path"
2. "the design states the measured current behaviour (strategy consumed as a
   label only) with file:line references that resolve"
3. "a test asserts the live selection path is UNCHANGED by this step:
   STRATEGY_REGISTRY label methods remain unreferenced from
   backend/services/autonomous_loop.py"

**Verification command (immutable):**
`source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_6_bridge_design.py -q`

## 5. Design content, all derived rather than invented

- **Insertion points: TWO, not one.** (i) the params seam at `:431` -- already
  correct and inert, but `decide_trades` (`portfolio_manager.py:66-73`) takes
  **no params argument**, so there is nowhere to hand a strategy today; its only
  config channels are `settings` and `risk_overrides.get_effective()`, whose
  `ALLOWED_KEYS` (`:57`) holds 4 numeric keys and cannot carry a string.
  (ii) `paper_trader.execute_buy` -> `paper_positions.entry_strategy`, which is
  the §2 hazard.
- **Promotion gate: compose what exists.** `PromotionGate`
  (`backend/autoresearch/gate.py:21-30`, `min_dsr 0.95` / `max_pbo 0.20` /
  `min_pbo_trials 10`) is THE live one. `evaluate_stage`
  (`backend/services/promotion_gate.py:34-63`, stages `[0.05, 0.25, 1.0]`,
  `MIN_LIVE_DAYS [14, 30]`, `PBO_CEILING 0.5`) is script-reachable only. Note
  the 0.50 lives at `backend/services/promotion_gate.py:37` --
  `backend/autoresearch/promotion_gate.py` **does not exist**.
- **HARD PREREQUISITES, measured:** `optimizer_best.json` has **no `pbo` key at
  all**, and `PromotionGate` is fail-closed on a missing pbo. So **82.23** (PBO
  term never computed) and **82.26** (trial floor) are build-time blockers, not
  advisories.
- **Rollback: entirely existing machinery** -- kill switch
  (`autonomous_loop.py:1313-1322`); deactivating the `promoted_strategies` row
  (the fail-open fallback in `load_promoted_params` **is** a rollback);
  `rollback.py::auto_demote_on_dd_breach`; `evaluate_stage` regress-to-5%; the
  `strategy_decisions` audit trail; plus the standing default-OFF flag idiom.

## 6. Test traps for criterion 3, each measured by the gate

- **`len(set(values)) == len(keys)` FAILS on correct code.**
  `STRATEGY_REGISTRY` (`backtest_engine.py:69-82`) has **6 keys / 5 distinct
  method values** -- `meta_label` shares `triple_barrier`'s.
- **A blanket "no `backend.backtest` import" assertion is a FALSE POSITIVE.**
  `autonomous_loop.py` legitimately imports `backend.backtest.universe_lists`
  (`:552`) and `backend.backtest.markets` (`:563`).
- **Sweep the METHOD VALUES, not the KEYS.** `triple_barrier` legitimately
  appears in the live path as a label string; the label *method names* do not.
- **`perf_metrics.py:151` names `backtest_engine` in a COMMENT** -- a text scan
  would trip on it.
- **A "`decided == prior` on every row" assertion FAILS.** Measured live: 51
  rows, 50 equal (all `cycle_heartbeat`), **1 unequal** (`reduce_position` via
  `decay_signal`). And the equality is true **by construction**
  (`:1654-1655` assign the same variable), not empirically. Scope any such
  assertion to `cycle_heartbeat` rows -- or better, assert the construction.

## 7. Non-scope

**No selection is wired.** The live book is working; this step ships a document
and a guard that the live path is unchanged. No edit to `autonomous_loop.py`,
`paper_trader.py`, `portfolio_manager.py`, or `strategy_selector.py`. No
activation of `run_friday_promotion`. No live positions touched.

## 8. Discovered defects -- to queue, not fix

1. **`promoter.py:134` defaults a missing `pbo` to `0.0`, which PASSES any
   ceiling** -- a gate that fails OPEN on absent data, and `optimizer_best.json`
   has no `pbo` key. Highest-value of the four.
2. `strategy_decisions` heartbeat is **~6 days stale** (last write 2026-07-31)
   with the write swallowed at `:1664-1668`.
3. Three registry enumerations, **two stale** (`archetype_library.py:31` and
   `.claude/rules/backend-backtest.md` both say "five strategies"; there are 6).
4. `optimizer_best.json` carries 4 params nothing reads (readers reverted in
   `9fbd9cd6`); `run_friday_promotion` is unscheduled.

## 9. References

- `handoff/current/research_brief_82.6.md` (audit-class, 13 rounds, dry)
- arXiv 2603.20319 -- rotation strategies are the WORST case for cross-engine
  backtest divergence (3.71% at realistic costs vs 0.0000% at zero cost;
  Spearman rho 0.93 with cost intensity); recommends >=2 independent validators
- Shu, Yu & Mulvey (2024), *J. Asset Management* -- verifies the citation
  already in `strategy_selector.py:9-11`: turnover 141% -> 44% under a switch
  penalty, net Sharpe 0.68 vs 0.48 at 10bps one-way
- Internal: `backend/services/autonomous_loop.py:431-437,1644-1668`,
  `backend/services/paper_trader.py:1421-1445`,
  `backend/autoresearch/strategy_selector.py:9-17`,
  `backend/autoresearch/gate.py:21-30`,
  `backend/services/promotion_gate.py:34-63`,
  `scripts/migrations/phase_32_2_add_entry_strategy.py:16-17`
