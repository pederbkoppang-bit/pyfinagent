---
step: phase-5.1
title: Broker Abstraction Layer (foundational for multi-asset expansion)
cycle_date: 2026-04-26
harness_required: true
verification: 'source .venv/bin/activate && python -c "from backend.markets.alpaca_broker import AlpacaBroker; from backend.markets.broker_base import BrokerClient; assert issubclass(AlpacaBroker, BrokerClient); print(''ok'')" && python scripts/harness/run_harness.py --dry-run --cycles 1'
research_brief: handoff/current/phase-5.1-research-brief.md
---

# Contract -- phase-5.1

## Step ID

`phase-5.1` -- "Broker Abstraction Layer" (`.claude/masterplan.json` phase-5).

## Research-gate summary

Spawned `researcher` (moderate tier). Brief at
`handoff/current/phase-5.1-research-brief.md`. Gate: 7 external sources
read in full via WebFetch (alpaca-py docs, ABC vs Protocol guides 2026,
quantpython BaseBroker pattern), 17 URLs total, recency scan present
(2024-2026), 6 internal files inspected. `gate_passed: true`.

Decisive findings:
- Use `abc.ABC` (NOT Protocol) -- pyfinagent owns all broker subclasses; ABC catches incomplete subclasses at instantiation time
- 6 abstract methods: `submit_order`, `cancel_order`, `get_account`, `get_positions`, `get_orders`, `get_quote`
- Reuse existing `FillResult` from `execution_router.py:43-59` -- DO NOT duplicate
- Add 4 new dataclasses in `broker_base.py`: `AccountInfo`, `PositionInfo`, `OrderInfo`, `QuoteInfo`
- Factory `get_broker(market, asset_class)` lives in `backend/markets/__init__.py` (registry dict)
- `AlpacaBroker.submit_order` delegates to `_alpaca_real_fill` to preserve max-notional clamp + live-key guard
- `run_harness.py --dry-run --cycles 1` does NOT import broker modules; safe as long as no module-level side effects

NOTE: Running `scripts/harness/run_harness.py --dry-run --cycles 1` (the
second half of the immutable verification command) clobbers
`handoff/current/contract.md` with its own auto-generated cycle
sprint-contract. This Main re-restored this file after the verification
ran. The harness exited cleanly with "HARNESS COMPLETE -- 1 cycles
finished" (proven by the trailing log lines captured before the
clobber). Pre-existing harness behavior; not introduced by this cycle.

## Hypothesis

Introducing an abstract `BrokerClient` ABC with concrete `AlpacaBroker`
implementation + `get_broker` factory establishes the contract that
future multi-asset broker integrations (5.7 OANDA / 5.8 IBKR / 5.9
international) plug into. Existing `ExecutionRouter` and `PaperTrader`
remain untouched -- this is additive net-new code, not a refactor.

## Immutable success criteria (verbatim from masterplan)

```
source .venv/bin/activate && python -c "from backend.markets.alpaca_broker import AlpacaBroker; from backend.markets.broker_base import BrokerClient; assert issubclass(AlpacaBroker, BrokerClient); print('ok')" && python scripts/harness/run_harness.py --dry-run --cycles 1
```

Plus 4 success_criteria from masterplan:
1. `backend/markets/broker_base.py` defines abstract `BrokerClient`
2. `backend/markets/alpaca_broker.py` implements `BrokerClient`
3. `get_broker('US','equity')` returns `AlpacaBroker` instance
4. no regression on existing paper-trading smoke test

## Plan steps

1. Create `backend/markets/broker_base.py` (~140 LOC):
   - `from __future__ import annotations`
   - Re-export `FillResult` from `execution_router`
   - 4 frozen dataclasses: `AccountInfo`, `PositionInfo`, `OrderInfo`, `QuoteInfo`
   - `class BrokerClient(abc.ABC)` with 6 abstract methods (signatures from research brief)

2. Create `backend/markets/alpaca_broker.py` (~200 LOC):
   - `class AlpacaBroker(BrokerClient)`
   - `submit_order` delegates to `execution_router._alpaca_real_fill` (preserves max-notional clamp + live-key guard)
   - `cancel_order` / `get_account` / `get_positions` / `get_orders` call `alpaca.TradingClient` directly with `paper=True`
   - `get_quote` calls `alpaca.data.StockHistoricalDataClient`
   - All public methods fail-open (return safe defaults / log warning) when env creds absent so the verification command works in any environment
   - No module-level network calls or env reads

3. Create `backend/markets/__init__.py` (~55 LOC):
   - Import `BrokerClient`, `AlpacaBroker`
   - `_REGISTRY: dict[tuple[str, str], type[BrokerClient]] = {("US", "equity"): AlpacaBroker}`
   - `def get_broker(market: str, asset_class: str) -> BrokerClient` -- normalize keys + lookup + raise `ValueError` on miss

4. Create `tests/markets/__init__.py` (empty) and `tests/markets/test_broker_base.py` (~200 LOC, 12+ tests per research plan).

5. Run immutable verification command (the python -c part + the harness dry-run).

## References

- `handoff/current/phase-5.1-research-brief.md`
- `backend/services/execution_router.py:43-329` (FillResult + _alpaca_real_fill + ExecutionRouter to preserve)
- `backend/services/paper_trader.py:109-213` (regression target -- no changes)
- `scripts/harness/run_harness.py` (regression target via --dry-run --cycles 1)
- alpaca-py SDK: https://alpaca.markets/sdks/python/trading.html
- ABC vs Protocol 2026: https://tiendu.github.io/2026/02/27/modern-python-oop-eng.html

## Out of scope

- Changes to `ExecutionRouter` / `PaperTrader` / any active service
- Live alpaca-py calls (tests use monkeypatched `_alpaca_real_fill`)
- 5.7 OandaBroker / 5.8 IBKRBroker (separate masterplan steps)
- BQ schema changes (no tables added)
- Frontend changes
- Wiring `AlpacaBroker` into `ExecutionRouter` (separate cycle when 5.7+ comes online)
