# Contract — step 70.3 (S3 + money-path: atomic cross-sector swap + non-US avg-entry fix)

**Phase:** phase-70 | **Step:** 70.3 | **Priority:** P1 | harness_required: true
**Cycle:** 1 | Date: 2026-07-17 | **Type:** backend money-path, flag-gated default-OFF (double-gated behind
`paper_swap_enabled` too), $0, paper-only, DARK-until-token, fail-safe. live_check: none (no UI).

## Research-gate summary (gate PASSED)

Researcher via Workflow structured-output (Opus 4.8, $0). Envelope: **gate_passed=true**, tier=complex,
**8 external sources read in full**, 14 snippet-only, 45 URLs, recency scan performed, 6 internal files
re-anchored on HEAD ec64e4ea (70.0's 594/620/675 refs drifted +9). Brief: `research_brief_70.3.md`.

Grounding: SagaLLM arXiv 2503.11951 (pre-execution validation + "either fully committed S' or coherent
rollback"); microservices.io Saga + Temporal (compensation is a designed inverse, idempotent IfPresent);
multi-currency weighted-average-cost accounting (unit consistency).

Confirmed bugs on HEAD: (a) swap BUY sizing `buy_amount=nav*(pct/100.0)` (portfolio_manager.py:684) has NO
`min(available_cash)` and NO $50 floor, and the SELL is appended (:677) BEFORE the BUY amount is known → a SELL
that executes while its paired BUY drops = net -1 position. (b) add-on `new_avg=new_cost(USD)/new_qty(LOCAL)`
(paper_trader.py:308) mixes USD cost with local shares → corrupts avg_entry_price for non-US add-ons (the
first lot :338 stores the LOCAL price; execute_sell:472 treats avg_entry as LOCAL).

## Hypothesis / design (all flag-gated default-OFF; OFF ⇒ byte-identical)

1. **Atomic swap (criterion 1)** — new `paper_atomic_swap_enabled`. LAYER 1 (emit): thread `available_cash`
   into `_compute_swap_candidates`; size the swap BUY `min(nav*pct/100, available_cash + freed)` (freed =
   weakest.market_value), apply the $50 floor, and only emit BOTH legs together (tagged with a shared
   `swap_group_id`) when fundable — else drop the whole pair (never a lone SELL). LAYER 2 (execution): a
   `_execute_swap_pair` helper runs the pair **BUY-first with reserved cash** after a **SELL-feasibility
   pre-check** (position exists + FX available): pre-check → BUY(reserved_cash=freed) → if BUY drops, SELL is
   never attempted (atomic); the SELL (pre-validated) then executes. The SELL-fails-after-BUY branch is
   unreachable given the pre-check (defensive compensation only: delete the just-created BUY + LOUD log). This
   avoids any ledger reversal. Design note: I use BUY-first (vs the brief's SELL-first+compensation) because a
   paper ledger makes BUY-first strictly simpler and equally atomic — the common failure (BUY drops) needs NO
   compensation, and the SELL pre-check removes the rare one. `execute_buy` gains `reserved_cash`.
2. **Cross-sector rotation (criterion 2)** — new `paper_cross_sector_rotation_enabled`. OFF → same-sector-only
   (portfolio_manager.py:603 unchanged, byte-identical). ON → also consider the weakest-OVERALL holding across
   all sectors, REUSING the churn-fix exclusion + clamped denom + untouched 25% delta bar; fire only if
   projected portfolio HHI strictly drops; RE-VALIDATE the destination-sector count + NAV-pct caps on the
   projected composition (a fail-safe block, not a threshold move). Hard dependency: requires
   `paper_swap_churn_fix_enabled` ON (else no-op + WARN) so it inherits the anti-churn safety.
3. **avg_entry FX fix (criterion 3)** — new `paper_avg_entry_fx_fix_enabled`. ON →
   `new_avg=(old_qty*avg_entry + quantity*price)/new_qty` (LOCAL-share-weighted LOCAL prices); cost_basis stays
   USD. Byte-identical for US (quantity*price == amount_usd at fx=1). OFF → the legacy USD/LOCAL formula.
4. **Fail-safe (criterion 4)** — every new path holds/drops on failure; NO risk-limit threshold moved; the
   destination-cap re-validation only ADDS a default-OFF guard.

## Immutable success criteria (verbatim from masterplan.json 70.3)

1. The swap path cannot leave the book smaller than before the swap: either both legs execute or neither does
   (atomic/rollback); the BUY leg is bounded by available_cash and honors the $50 floor -- proven by a
   red->green test covering the SELL-executes-BUY-drops scenario
2. The swap can rotate into a DIFFERENT sector (not only same-sector churn), gated behind the
   diversification/churn flag; with the flag OFF, behavior is byte-identical to today
3. Add-to-existing-position computes avg_entry_price in consistent units for non-US tickers (no USD-cost /
   local-share mix) -- proven by a test with a non-USD ticker
4. All changes are fail-safe (a failure blocks/holds rather than corrupting the book); no live risk-limit
   thresholds moved

Verification command (immutable):
`bash -c 'ls backend/tests/ | grep -Eqi "70_3|swap|atomic" && python -c "import ast; ast.parse(open(\'backend/services/portfolio_manager.py\').read()); ast.parse(open(\'backend/services/paper_trader.py\').read())"'`

## Plan
2 (this contract, before code). 3. GENERATE: settings.py (3 flags); TradeOrder.swap_group_id; execute_buy
`reserved_cash` param + avg_entry gated fix; `_compute_swap_candidates` cash-bound + $50 floor + swap_group_id
+ cross-sector rotation (available_cash threaded at the :480 call); `_execute_swap_pair` helper + autonomous_loop
wiring (atomic-ON groups via the helper, rest via the flat loops; OFF byte-identical); test
`test_phase_70_3_atomic_swap.py` (red->green atomic; non-US avg_entry; OFF byte-identical). Verify: command +
import-smoke + pytest. 4. Q/A (Workflow). 5. LOG. 6. FLIP.

## Boundaries (binding)
$0, paper-only; flag-gated default-OFF + double-gated behind paper_swap_enabled (DARK-until-token); NO risk
threshold moved; historical_macro FROZEN; hysteresis BANNED; fail-safe; harness stays 3 agents.

## References
research_brief_70.3.md; design_trade_diversity_70.md (b); confirmed_findings.json (#3/#9/#10). Code:
portfolio_manager.py:20/480/507/603/629/677-702, paper_trader.py:119/197/217/305-338/365-404, autonomous_loop.py:1313-1374, settings.py:328/344.
