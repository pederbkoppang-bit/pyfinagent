---
step: phase-16.36
title: Anthropic-fallback hardening bundle (#43, #44, #45, #46)
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-16
deliverables:
  - 21-site datetime.utcnow() -> datetime.now(timezone.utc) sweep
  - module-level reset_anthropic_client() in multi_agent_orchestrator.py
  - Gemini token-usage extraction in _gemini_text_call
  - backend/tests/test_anthropic_fallback.py (5 tests)
---

# Sprint Contract -- phase-16.36

## Research-gate summary

`handoff/current/phase-16.36-research-brief.md`. tier=moderate, 6 in-full,
16 URLs, recency scan present, gate_passed=true. 14 internal files
inspected.

## Bundled scope (4 task-list items, single cycle)

| # | Task | Surface |
|---|------|---------|
| #43 | `datetime.utcnow()` deprecation cleanup | 21 sites in 7 files (including 1 test fixture) |
| #44 | `reset_anthropic_client()` for mid-session key rotation | 1 new module-level function |
| #45 | Mock test for AuthError fallback path | new file `backend/tests/test_anthropic_fallback.py` (5 tests) |
| #46 | Gemini token usage extraction on fallback | 1 helper change in `_gemini_text_call` |

## Hypothesis

All four are mechanical hardening on the Anthropic/Gemini fallback path
added in 16.31. Bundle keeps the cycle small + the test suite caches a
single mock-anthropic shim once instead of four times.

## Concrete plan

### #43: datetime sweep (21 sites)

Standard replacement: `datetime.utcnow()` -> `datetime.now(timezone.utc)`.
20 of 21 are mechanical. ONE special case:

`backend/services/outcome_tracker.py:48`:
```python
holding_days = (datetime.utcnow() - rec_date).days
```
`rec_date` is naive (from `datetime.fromisoformat(analysis_date)`).
Aware-naive subtraction raises TypeError. Fix:
```python
holding_days = (datetime.now(timezone.utc).replace(tzinfo=None) - rec_date).days
```

L107-108 of the same file is already guarded (the `replace(tzinfo=None)`
helper from 16.30 fix). Plain swap works there.

Files to touch (file:line per researcher's audit):
- `backend/tools/sec_insider.py`: L160, L239
- `backend/tools/fred_data.py`: L29
- `backend/agents/memory.py`: L92, L100
- `backend/agents/skill_optimizer.py`: L599
- `backend/backtest/data_ingestion.py`: L96, L188, L280, L340
- `backend/backtest/spot_checks.py`: L37, L56, L434
- `backend/slack_bot/governance.py`: L89
- `backend/db/bigquery_client.py`: L141, L378, L449
- `backend/services/outcome_tracker.py`: L48 (SPECIAL), L108
- `backend/tests/test_outcome_tracker.py`: L123 (test fixture)

For each file: ensure `from datetime import datetime, timezone` is in
imports (most likely already imports `datetime` only; need to add
`timezone`).

### #44: `reset_anthropic_client()` module-level function

Append after `run_orchestrated_round` (~L1480) in
`backend/agents/multi_agent_orchestrator.py`:

```python
def reset_anthropic_client() -> None:
    """phase-16.36: clear cached Anthropic client + unavailability flag.

    Used after the operator rotates the Anthropic key in `backend/.env`
    so the next orchestrator invocation re-reads `settings.anthropic_api_key`
    instead of using the stale singleton.

    Also clears `get_settings.lru_cache` so the new env value propagates
    through the pydantic Settings layer.
    """
    global _orchestrator
    if _orchestrator is not None:
        _orchestrator._client = None
        _orchestrator._anthropic_unavailable = False
    try:
        from backend.config.settings import get_settings
        get_settings.cache_clear()
    except (ImportError, AttributeError):
        pass
```

Note: real attribute names per code inspection are `_client` (not
`_anthropic_client` as the brief guessed) and `_anthropic_unavailable`.

### #46: Gemini token-usage extraction

`_gemini_text_call` at L222-241. Currently returns `{"input": 0, "output": 0}`
hardcoded. Replace with:

```python
text = (getattr(resp, "text", "") or "").strip() or "No response."
_umeta = getattr(resp, "usage_metadata", None)
_in = int(getattr(_umeta, "prompt_token_count", 0) or 0)
_out = int(getattr(_umeta, "candidates_token_count", 0) or 0)
return text, {"input": _in, "output": _out}
```

`LLMResponse.usage_metadata` is a `UsageMeta` dataclass with both fields
already populated by `GeminiClient` (see `backend/agents/llm_client.py:226-241`,
`568-578`). Same `getattr`-safe pattern as `cost_tracker.py:128-129`.

Apply same fix in the exception branch (currently also returns
`{"input": 0, "output": 0}`) -- leave that one as-is since the
exception path has no response object.

### #45: Mock test file

`backend/tests/test_anthropic_fallback.py` (~150 LOC, 5 tests):

1. `test_call_agent_auth_error_triggers_gemini_fallback` -- mock
   `_client.messages.create` to raise; assert `_gemini_text_call`
   was called.
2. `test_call_agent_with_tools_auth_error_triggers_gemini_fallback`
   -- same for the tools variant.
3. `test_anthropic_unavailable_flag_persists_after_first_401`
   -- after first AuthError, `_anthropic_unavailable=True` and
   subsequent calls skip the Anthropic path entirely.
4. `test_reset_anthropic_client_clears_flags_and_settings_cache`
   -- after `reset_anthropic_client()`, the flag is False and
   `_client` is None.
5. `test_gemini_usage_dict_populated` -- when the fallback fires,
   the returned `usage` dict has non-zero `input`/`output` keys
   (mock the Gemini client to return a `LLMResponse` with
   `UsageMeta(prompt_token_count=100, candidates_token_count=50)`).

Anthropic SDK's real `AuthenticationError` requires `httpx.Response`
in its constructor, so use the sys.modules patch approach (researcher
brief): patch `sys.modules['anthropic']` with a fake module where
`AuthenticationError = Exception`.

## Success Criteria (verbatim, immutable)

```
cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && \
python -c "from backend.agents.multi_agent_orchestrator import reset_anthropic_client; reset_anthropic_client(); print('reset ok')" && \
test -z "$(grep -rn 'datetime.utcnow()' backend/tools backend/agents backend/backtest backend/db backend/slack_bot backend/services backend/tests/test_outcome_tracker.py)" && \
python -m pytest backend/tests/test_anthropic_fallback.py -v
```

Plus:
- `datetime_audit_clean`: zero hits of `datetime.utcnow()` in the 7
  audited directories.
- `reset_anthropic_client_imports`: function importable + callable.
- `gemini_usage_populated`: `_gemini_text_call` returns non-zero
  `input`/`output` when `usage_metadata` present.
- `mock_tests_pass`: 5/5 tests in `test_anthropic_fallback.py`.
- `outcome_tracker_test_pass`: existing
  `backend/tests/test_outcome_tracker.py` still passes (regression).
- `no_other_regressions`: `python -m pytest backend/tests/ tests/meta_evolution/ -v --no-header -q 2>&1 | tail -3` shows green.

## What Q/A must audit

1. Immutable verification command exits 0.
2. Zero `datetime.utcnow()` in the 7 audited dirs (deterministic grep).
3. `outcome_tracker.py:48` special-case applied (aware-naive guard).
4. `reset_anthropic_client()` function exists at module level + clears
   both `_client` and `_anthropic_unavailable` AND calls
   `get_settings.cache_clear()`.
5. `_gemini_text_call` extracts `prompt_token_count` and
   `candidates_token_count` from `usage_metadata` via getattr-safe
   pattern.
6. 5 tests in `test_anthropic_fallback.py` (or more if useful);
   all PASS; sys.modules patch pattern used (not real
   AuthenticationError construction).
7. `test_outcome_tracker.py` still passes (regression -- the
   datetime sweep touched its fixtures at L123).
8. No mutation to engine code outside the anthropic/gemini fallback
   path + the datetime sweep targets.
