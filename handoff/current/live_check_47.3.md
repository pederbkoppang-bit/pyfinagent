# Live Check — phase-47.3: Opus 4.8 cost_tracker pricing

Deterministic step (no live trading system needed). Verbatim command output, 2026-05-29.

## 1. Immutable verification command — EXIT 0
```
$ source .venv/bin/activate && python -c "from backend.agents.cost_tracker import MODEL_PRICING, _DEFAULT_PRICING as D; o8=MODEL_PRICING.get('claude-opus-4-8'); assert o8==MODEL_PRICING['claude-opus-4-7'] and o8!=D, o8; print('PASS 4.8 pricing', o8)" && test $(grep -c 'claude-opus-4-8' backend/api/settings_api.py) -ge 2 && echo 'settings_api 4.8 entries OK'
PASS 4.8 pricing (5.0, 25.0)
settings_api 4.8 entries OK
EXIT_CODE=0
```

## 2. Regression test — 2 passed
```
$ python -m pytest tests/agents/test_cost_tracker_pricing.py -q
..                                                                       [100%]
2 passed in 0.01s
```
- `test_opus_4_8_priced_same_as_4_7_and_not_default` (dict guard)
- `test_opus_4_8_recorded_cost_uses_real_rate_not_default` (behavioral: real `CostTracker.record()` -> cost_usd==30.00, != default 0.50)

## 3. settings_api.py — both tables patched
```
backend/api/settings_api.py:31  (allowlist)   "claude-opus-4-8", "claude-opus-4-7", ...
backend/api/settings_api.py:214 (display)      {"model":"claude-opus-4-8", ... "input_per_1m":5.00, "output_per_1m":25.00}
```

Before: every claude-opus-4-8 call cost-tracked at _DEFAULT_PRICING (0.10, 0.40) -> ~50x in / ~62.5x
out understatement. After: tracked at the real $5/$25, restoring the Compute term of Net System Alpha.
