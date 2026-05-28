# Experiment Results — phase-47.3: Opus 4.8 cost_tracker pricing regression

**Cycle:** 3 of the production-ready+money push (FREE — no project LLM spend).
**Step:** 47.3 | **Result:** ready for Q/A.

## What was built / changed (3 files + 1 new test)
1. `backend/agents/cost_tracker.py:26` — added `"claude-opus-4-8": (5.00, 25.00),` above the 4-7
   entry. Previously 4.8 fell through to `_DEFAULT_PRICING=(0.10,0.40)` -> ~50x in / ~62.5x out
   understatement. The cost path is `CostTracker.record() -> MODEL_PRICING.get(model, _DEFAULT_PRICING)`
   (cost_tracker.py:161), so the new entry fixes every 4.8 call's recorded cost.
2. `backend/api/settings_api.py` (allowlist ~:31) — added `"claude-opus-4-8"` to the current-GA line.
3. `backend/api/settings_api.py` (display-pricing ~:214) — added a
   `{"model":"claude-opus-4-8","provider":"Anthropic","input_per_1m":5.00,"output_per_1m":25.00}` row.
4. NEW `tests/agents/test_cost_tracker_pricing.py` — 2 guards:
   - `test_opus_4_8_priced_same_as_4_7_and_not_default`: dict-entry guard (== 4.7, == (5,25), != default).
   - `test_opus_4_8_recorded_cost_uses_real_rate_not_default`: BEHAVIORAL — builds a fake response
     (1M in + 1M out, no cache), calls `CostTracker.record(model="claude-opus-4-8")`, asserts
     `cost_usd == 30.00` (real rate) and `!= 0.50` (the 50x-low default). Tests the actual lookup path.

`model_tiers.py` already 4-8 (no action). `governance.py:84-85` stale model-agnostic estimate left
out of scope (research). max_tokens-at-xhigh clamp deferred (Gemini-locked today; separate follow-up).

## Verbatim verification output
```
$ ast.parse cost_tracker.py + settings_api.py + test  -> ast OK (3 files)
$ python -m pytest tests/agents/test_cost_tracker_pricing.py -q  -> 2 passed in 0.01s
$ python -c "...MODEL_PRICING['claude-opus-4-8']==['claude-opus-4-7']!=_DEFAULT..." \
   && test $(grep -c 'claude-opus-4-8' backend/api/settings_api.py) -ge 2
PASS 4.8 pricing (5.0, 25.0)
settings_api 4.8 entries OK
EXIT_CODE=0
```

## Success-criteria mapping (masterplan phase-47.3)
1. MODEL_PRICING['claude-opus-4-8'] == ['claude-opus-4-7'] == (5.00,25.00) != _DEFAULT_PRICING — **MET** (assert exit 0).
2. claude-opus-4-8 in settings_api allowlist AND display table (>=2 occurrences) — **MET** (grep -c >= 2).
3. pytest regression guard asserts 4.8 pricing + passes; ast clean — **MET** (2 passed; ast OK on all 3).

## Scope honesty
Cost-telemetry-only change; no trade-execution / risk / perf-metrics path touched; no LLM spend; no
BQ write. Restores the accuracy of the Compute term of Net System Alpha (relevant before API-key
migration where cost telemetry drives the budget guard). The behavioral test exercises the real
`record()` lookup path, so it is not tautological.

## Files
backend/agents/cost_tracker.py, backend/api/settings_api.py,
tests/agents/test_cost_tracker_pricing.py, .claude/masterplan.json (phase-47.3 added; 47.2->in-progress),
handoff/current/{contract.md, research_brief_phase_47_3_cost_tracker.md}.
