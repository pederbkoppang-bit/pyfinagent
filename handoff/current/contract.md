# Contract — step 70.4 (S3: un-gate throughput — surface + reconcile the silent BUY-gates)

**Phase:** phase-70 | **Step:** 70.4 | **Priority:** P2 | harness_required: true
**Cycle:** 1 | Date: 2026-07-17 | **Type:** backend observability (always-on, $0) + 2 flag-gated behavior knobs
(default-OFF). live_check: none (no UI).

## Research-gate summary (gate PASSED)

Researcher via Workflow structured-output (Opus 4.8, $0). Envelope: **gate_passed=true**, tier=complex,
**8 external sources read in full**, 26 snippet-only, 34 URLs, recency scan performed, 6 internal files
re-anchored on HEAD d0efa50d. Brief: `research_brief_70.4.md`. Grounding: SEC Rule 15c3-5 (rejections are
auditable first-class events); arXiv 2603.07752 (rejection tolerance is a first-class MEASURED state; silent
rejections hurt throughput); LLM cost-observability practice (surface budget truncation, never silent).

## Confirmed on HEAD

- **Session budget**: `_SESSION_BUDGET_USD=$1.00` (autonomous_loop.py:90, hidden module const) is HALF the
  operator-visible `paper_max_daily_cost_usd=$2.00` and fires first; `_check_session_budget` (:95-105) raises
  `BudgetBreachError` with NO log; captured under `gather(return_exceptions=True)` (:1090/:1097) then dropped by
  the `isinstance(r,dict)` filter (:1094/:1101) → silent truncation (never reaches the clean halt at :1585).
- **Price-tolerance** (paper_trader.py:169-193): ALREADY logs ticker+drift (:188) AND is ALREADY tunable via
  `paper_price_tolerance_pct` (settings.py:557). Gap: the WARN-only `return None` is UN-COUNTED; the caller
  (autonomous_loop.py:1468 `if trade:`) is a silent no-op → 0-trade cycles un-attributable at the summary layer.
- **Lite parse-fail** (autonomous_loop.py:2399): the parse-fail else-branch defaults `{HOLD, score:5}` with no
  log; the returned lite dict has no top-level `confidence`, so `_degraded_scoring_check` (:2108) sees score 5 ≠ 0
  and conf None → it EVADES the degraded guard and masquerades as a genuine HOLD (silently suppresses a BUY).

## Hypothesis / design

ALWAYS-ON observability ($0, no threshold moved): **G1-A** log at the budget raise; **G1-B** scan the raw gather
results for `BudgetBreachError` before the isinstance filter → `summary['session_budget_breach'/…]` + WARN;
**G2-A** a `PaperTrader.buy_rejections` accumulator (append `{ticker,reason,divergence_pct,…}` at the
price-tolerance `return None` + the other None-exits) folded into `summary['buy_rejections']` + by-reason count;
**G3-A** log WARN at the parse-fail + set `_parse_failed=True` on the returned dict + fix the mislabeled INFO log
(:2401); **G3-B** extend `_degraded_scoring_check` to count `_parse_failed`/`_degraded` + `summary['lite_parse_failures']`
(affects ONLY the P1 degraded-scoring alert, NOT any trade).

FLAG-GATED behavior changes (default-OFF → byte-identical): **G1-C** new `paper_session_budget_reconcile_enabled`
→ effective session budget = `paper_max_daily_cost_usd` when ON (single knob, session==daily==$2), else $1.00
(cost knob only, NO risk threshold moved); **G3-C** reuse the EXISTING `paper_synthesis_integrity_enabled` → set
`_degraded=True` on a parse-fail so the unconditional guard (:1080) drops it from decide_trades input (fail-safe:
removing a spurious neutral can never create a BUY; a genuine parsed HOLD is untouched).

## Immutable success criteria (verbatim from masterplan.json 70.4)

1. The per-cycle session budget and the operator-visible daily cost cap are reconciled (no hidden budget that is
   a fraction of the visible cap) OR the hidden budget is surfaced to the operator and logged on breach -- a
   cost-cut cycle is never silent
2. Price-tolerance rejections are logged with ticker + drift so 0-trade cycles are diagnosable, and the tolerance
   is tunable via settings
3. A lite-analyzer parse/rail failure is logged and counted as degraded (does not masquerade as a legitimate HOLD
   via a default score=5) so it cannot silently suppress BUYs
4. Any threshold that changes trading behavior is flag-gated default-OFF; observability additions are always-on and $0

Verification command (immutable):
`bash -c 'grep -Eqi "session budget|per-cycle|cost cap" backend/services/autonomous_loop.py && ls backend/tests/ | grep -Eqi "70_4|budget|tolerance|gate"'`

## Plan
2 (this contract). 3. GENERATE: settings.py (new paper_session_budget_reconcile_enabled); autonomous_loop.py
(G1-A/B/C, G3-A/B/C, fold buy_rejections into summary); paper_trader.py (G2-A buy_rejections accumulator);
test test_phase_70_4_gate_observability.py. Verify: command + import-smoke + pytest. 4. Q/A (Workflow). 5. LOG. 6. FLIP.

## Boundaries (binding)
$0 metered; observability always-on; behavior changes flag-gated default-OFF (byte-identical OFF); NO risk-limit
threshold moved; historical_macro FROZEN; hysteresis untouched; fail-safe; harness stays 3 agents.

## References
research_brief_70.4.md; design_trade_diversity_70.md (c); confirmed_findings.json (#6/#7/#8). Code:
autonomous_loop.py:90/95-105/335/1037-1101/1103-1131/2108-2135/2395-2401/2493-2516, paper_trader.py:169-193/1445-1469,
settings.py:371/557.
