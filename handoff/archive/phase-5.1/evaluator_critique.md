---
step: phase-5.1
cycle_date: 2026-04-26
qa_agent: qa (merged qa-evaluator + harness-verifier)
verdict: PASS
---

# Q/A Critique -- phase-5.1 Broker Abstraction Layer

## 5-item harness-compliance audit

1. **Researcher spawn** -- PASS. `handoff/current/phase-5.1-research-brief.md`
   exists with `step: phase-5.1`, tier `moderate`. Header self-reports
   gate_passed=true (7 external sources read in full, 17 URLs, recency
   scan, 6 internal files). Cited inside broker_base.py docstring.
2. **Contract pre-commit** -- PASS. `handoff/current/contract.md` has
   `step: phase-5.1` and `verification:` field is verbatim equal to the
   masterplan immutable command. Harness clobber/restoration disclosed.
3. **Results document** -- PASS. `handoff/current/experiment_results.md`
   captures both halves of the verification command including the
   "ok" sentinel and "HARNESS COMPLETE -- 1 cycles finished" marker.
4. **Log-last** -- N/A for this cycle. The cycle-1 harness_log.md
   entry at 08:55 UTC is the dry-run optimizer's own append, not the
   masterplan-cycle log; that log-append is Main's NEXT step after
   this Q/A returns PASS.
5. **No-verdict-shopping** -- PASS. First Q/A spawn for phase-5.1
   (no prior `phase-5.1-evaluator-critique.md` in handoff/current/
   or handoff/archive/phase-5.1/).

## Deterministic checks

| Check | Command | Result |
|-------|---------|--------|
| Verification half 1 (subclass + import) | `python -c "from backend.markets.alpaca_broker import AlpacaBroker; from backend.markets.broker_base import BrokerClient; assert issubclass(AlpacaBroker, BrokerClient); print('ok')"` | exit 0; stdout=`ok` |
| Verification half 2 (harness dry-run) | NOT re-run (would clobber contract.md); evidence in harness_log.md L76-80 | "HARNESS COMPLETE -- 1 cycles finished" |
| Unit tests | `pytest tests/markets/test_broker_base.py -v` | 15/15 PASS in 0.78s |
| Module files exist | `ls backend/markets/ tests/markets/` | All 5 expected files present |
| No service wiring (scope) | `grep -l 'from backend.markets' backend/services/ -r` | Empty (correct -- no premature wiring) |
| paper_trader regression | `python -c "from backend.services.paper_trader import PaperTrader"` | exit 0 (clean import) |

## Spec-alignment review

**broker_base.py:**
- `BrokerClient(abc.ABC)` with `@abc.abstractmethod` on all 6 required
  methods (submit_order, cancel_order, get_account, get_positions,
  get_orders, get_quote). Verified by reading L87-137.
- `FillResult` is RE-EXPORTED from `backend.services.execution_router`
  (L25 `from backend.services.execution_router import FillResult`),
  NOT redefined. Test `test_fillresult_is_not_duplicated` asserts
  identity. PASS.
- 4 frozen dataclasses (AccountInfo / PositionInfo / OrderInfo /
  QuoteInfo) all `@dataclass(frozen=True)` with `raw: dict` field
  for broker-specific overflow.

**alpaca_broker.py:**
- `AlpacaBroker(BrokerClient)` with all 6 methods implemented.
- `__init__` is purely `self._client = None` -- no network, no env
  read. Lazy pattern verified.
- `submit_order` delegates to `execution_router._alpaca_real_fill`
  when creds present (preserves max-notional clamp + live-key guard
  chain in the existing service). Falls through to `_alpaca_mock_fill`
  when creds absent OR real-fill raises. Fail-open verified by
  `test_submit_order_no_creds_returns_mock_fill`.
- All other methods fail-open: `_trading_client()` returns None when
  creds absent, callers return safe defaults (`_empty_account()`,
  `[]`, `False`, `_empty_quote()`) and log warnings. No exceptions
  propagate. Verified by 6 creds-absent unit tests + manual code review.
- Module-level imports are stdlib + `backend.markets.broker_base`
  only -- no env reads, no network.

**__init__.py:**
- `_REGISTRY = {("US", "equity"): AlpacaBroker}` at L36-38.
- `get_broker(market, asset_class)` normalizes case (`upper()` /
  `lower()`) and raises `ValueError` with informative message on
  miss. Tests `test_get_broker_case_insensitive` and
  `test_get_broker_unknown_raises` confirm.

## LLM-judgment leg

- **Solves masterplan intent:** Yes. The ABC + registry pattern
  cleanly admits OandaBroker (5.7) / IBKRBroker (5.8) as one-line
  registry additions without disturbing AlpacaBroker.
- **abc.ABC vs Protocol:** Defensible. pyfinagent owns all
  subclasses; runtime TypeError on incomplete subclass (per test
  `test_incomplete_subclass_raises`) is the desired behavior.
- **Delegation correctness:** `submit_order` correctly delegates to
  the existing `_alpaca_real_fill` rather than re-implementing the
  Alpaca call -- this preserves the phase-17 max-notional clamp and
  the ALPACA_TRADING_MODE live-key guard. Excellent restraint;
  avoids drift between two parallel order paths.
- **Fail-open consistency:** Uniform across all 6 methods. Every
  branch that could touch the network is wrapped in try/except with
  a `logger.warning(...fail-open: %r...)` line and a safe default
  return. ASCII-only logger messages (security.md compliant).
- **Scope honesty:** No service wiring yet -- ExecutionRouter still
  references `_alpaca_real_fill` directly, not `AlpacaBroker`.
  Correct: phase-5.1 is the abstraction, phase-5.7+ does the wiring.
- **Harness-clobber disclosure:** Honest and operationally accurate.
  Disclosed in experiment_results.md L116-125; matches the known
  pre-existing harness behavior; contract.md was re-restored.
- **Material defects:** None identified.

## Verdict

```json
{
  "ok": true,
  "verdict": "PASS",
  "violated_criteria": [],
  "violation_details": "",
  "certified_fallback": null,
  "checks_run": [
    "harness_compliance_audit_5_items",
    "verification_command_half_1",
    "verification_command_half_2_via_harness_log",
    "unit_tests_15_of_15",
    "file_existence",
    "module_shape_broker_base",
    "module_shape_alpaca_broker",
    "module_shape_init_registry",
    "no_service_wiring_grep",
    "paper_trader_import_regression",
    "llm_judgment_intent_design_delegation_failopen_scope_disclosure"
  ]
}
```

Main may proceed to: append `harness_log.md` cycle entry for
phase-5.1, then flip masterplan status to `done`.
