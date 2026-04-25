---
step: phase-16.36
cycle_date: 2026-04-25
forward_cycle: true
expected_verdict: PASS
deliverables:
  - 21-site datetime.utcnow -> datetime.now(timezone.utc) sweep (10 files)
  - reset_anthropic_client() module function in multi_agent_orchestrator.py
  - Gemini token-usage extraction in _gemini_text_call
  - backend/tests/test_anthropic_fallback.py (6 tests)
---

# Experiment Results -- phase-16.36

## What was done

Bundle of 4 follow-ups (#43, #44, #45, #46) on the Anthropic/Gemini
fallback machinery added in 16.31. All four shipped in a single cycle:

### #43: datetime.utcnow() audit + replacement (21 sites, 10 files)

| File | Sites |
|------|-------|
| `backend/tools/fred_data.py` | 1 (L29) |
| `backend/tools/sec_insider.py` | 2 (L160, L239) |
| `backend/agents/memory.py` | 2 (L92, L100) |
| `backend/agents/skill_optimizer.py` | 1 (L599) |
| `backend/backtest/data_ingestion.py` | 4 (L96, L188, L280, L340) |
| `backend/backtest/spot_checks.py` | 3 (L37, L56, L434) |
| `backend/slack_bot/governance.py` | 1 (L89) |
| `backend/db/bigquery_client.py` | 3 (L141, L378, L449) |
| `backend/services/outcome_tracker.py` | 2 (L48, L108) |
| `backend/tests/test_outcome_tracker.py` | 1 (L123) |

All replaced `datetime.utcnow()` -> `datetime.now(timezone.utc)`.
`timezone` added to `from datetime import` line in 9 files (the test
file already had it from 16.30).

**Special case:** `outcome_tracker.py:48` and L108 both subtract a
naive `rec_date` from the new aware `now()`. Wrapped in
`.replace(tzinfo=None)` to preserve the existing arithmetic semantics.
Also updated the comment at L105 to reflect the new code shape and
add the phase-16.36 anchor.

### #44: reset_anthropic_client() module function

Appended at end of `backend/agents/multi_agent_orchestrator.py` (~L1494).
Clears `_orchestrator._client`, `_orchestrator._anthropic_unavailable`,
and `get_settings.cache_clear()`. Safe no-op when no orchestrator
instance exists. Verified via:

```
$ python -c "from backend.agents.multi_agent_orchestrator import reset_anthropic_client; reset_anthropic_client(); print('reset ok')"
reset ok
```

### #46: Gemini token usage extraction

Replaced hardcoded `{"input": 0, "output": 0}` in `_gemini_text_call`
(L237 success branch) with getattr-safe extraction from
`LLMResponse.usage_metadata.prompt_token_count` and
`.candidates_token_count`. Exception branch (L241) intentionally still
returns zeros (no response object on the failure path).

### #45: Mock test file (6 tests, target was 5)

`backend/tests/test_anthropic_fallback.py` (192 lines, 6 tests):

1. `test_call_agent_auth_error_triggers_gemini_fallback` -- Anthropic
   raises -> Gemini called -> usage dict populated.
2. `test_call_agent_with_tools_auth_error_triggers_gemini_fallback`
   -- same for tools-enabled call path.
3. `test_anthropic_unavailable_flag_persists_after_first_401` --
   second call short-circuits (Anthropic call count stays at 1).
4. `test_reset_anthropic_client_clears_flags_and_settings_cache` --
   `get_settings.cache_clear()` is called; `_client` -> None;
   `_anthropic_unavailable` -> False.
5. `test_gemini_usage_dict_populated` -- regression for #46
   (input=137, output=42 from the mocked LLMResponse).
6. `test_reset_is_safe_when_no_orchestrator_exists` -- no-op when
   `_orchestrator is None`.

Pattern used: monkeypatch `sys.modules['anthropic']` so
`AuthenticationError = Exception` (the real `AuthenticationError`
requires `httpx.Response` in its constructor; sys.modules patching
sidesteps that).

### Files touched

| Path | Action | LOC delta |
|------|--------|-----------|
| `backend/tools/fred_data.py` | edited | +1 import +1 call |
| `backend/tools/sec_insider.py` | edited | +1 import +2 calls |
| `backend/agents/memory.py` | edited | +1 import +2 calls |
| `backend/agents/skill_optimizer.py` | edited | +1 import +1 call |
| `backend/backtest/data_ingestion.py` | edited | +1 import +4 calls |
| `backend/backtest/spot_checks.py` | edited | +1 import +3 calls |
| `backend/slack_bot/governance.py` | edited | +1 import +1 call |
| `backend/db/bigquery_client.py` | edited | +1 import +3 calls |
| `backend/services/outcome_tracker.py` | edited | +1 import +2 calls + comment |
| `backend/tests/test_outcome_tracker.py` | edited | +1 call |
| `backend/agents/multi_agent_orchestrator.py` | edited | +6 line _gemini_text_call (#46) +21 line reset_anthropic_client (#44) |
| `backend/tests/test_anthropic_fallback.py` | CREATED | 192 lines |
| `handoff/current/contract.md` | rewrite (rolling) | -- |
| `handoff/current/experiment_results.md` | rewrite (this) | -- |

## Verification (verbatim, immutable)

```
$ python -c "from backend.agents.multi_agent_orchestrator import reset_anthropic_client; reset_anthropic_client(); print('reset ok')" && \
  test -z "$(grep -rn 'datetime.utcnow()' backend/tools backend/agents backend/backtest backend/db backend/slack_bot backend/services backend/tests/test_outcome_tracker.py)" && \
  echo "utcnow audit clean" && \
  python -m pytest backend/tests/test_anthropic_fallback.py -v
reset ok
utcnow audit clean
collected 6 items
test_anthropic_fallback.py::test_call_agent_auth_error_triggers_gemini_fallback PASSED
test_anthropic_fallback.py::test_call_agent_with_tools_auth_error_triggers_gemini_fallback PASSED
test_anthropic_fallback.py::test_anthropic_unavailable_flag_persists_after_first_401 PASSED
test_anthropic_fallback.py::test_reset_anthropic_client_clears_flags_and_settings_cache PASSED
test_anthropic_fallback.py::test_gemini_usage_dict_populated PASSED
test_anthropic_fallback.py::test_reset_is_safe_when_no_orchestrator_exists PASSED
6 passed in 0.02s
```

**Result: PASS.** Compound `&&` exits 0. Zero remaining `datetime.utcnow()`
in the 7 audited dirs. 6/6 mock tests PASS in 0.02s.

## No-regressions

```
$ python -m pytest backend/tests/test_outcome_tracker.py backend/tests/test_anthropic_fallback.py tests/meta_evolution/ -v 2>&1 | tail -3
============================== 52 passed in 3.67s ==============================
```

52/52 PASS across:
- `test_outcome_tracker.py` (5 tests, includes the L123 datetime fixture
  that was migrated)
- `test_anthropic_fallback.py` (6 new tests)
- `tests/meta_evolution/` (alpha_velocity, archetype_library,
  cron_allocator, directive_rewriter -- 41 tests)

## Success criteria assessment

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | datetime_audit_clean | PASS | grep returns empty string |
| 2 | reset_anthropic_client_imports | PASS | importable + callable |
| 3 | gemini_usage_populated | PASS | test_gemini_usage_dict_populated PASS |
| 4 | mock_tests_pass | PASS | 6/6 (target was 5) |
| 5 | outcome_tracker_test_pass | PASS | 5/5 PASS (includes L123 migrated fixture) |
| 6 | no_other_regressions | PASS | 52/52 across tracked test dirs |

## Honest disclosures

1. **6 tests, contract said 5.** Added a sixth test
   (`test_reset_is_safe_when_no_orchestrator_exists`) for the no-op
   guard path. Exceeds floor; not a violation.

2. **`outcome_tracker.py:48` AND `:108`** both needed the
   `.replace(tzinfo=None)` guard. The contract called out only L48
   explicitly, but L108 had the same pattern (it was already guarded
   for the rec_date side via the L107-109 helper, but the new aware
   `now()` would still TypeError without stripping its own tzinfo).
   Fixed both.

3. **Comment at outcome_tracker.py:105** also referenced
   `datetime.utcnow()`. Updated to match the new code shape (so the
   verification grep stays clean) AND added the phase-16.36 anchor.

4. **Exception branch of `_gemini_text_call` (L241) NOT changed.**
   No response object on the failure path; hardcoded zeros are
   correct. Documented in code via the existing pattern.

5. **`MagicMock(name=...)` collision** caught on first test run --
   `name` is a reserved kwarg of `MagicMock.__init__`. Refactored to
   set `config.name = "TestAgent"` after construction.

6. **isoformat() output now includes `+00:00` suffix.** Per researcher
   pitfall #2: BigQuery TIMESTAMP accepts both naive and aware ISO
   strings. No downstream breakage in the affected sites (they all
   feed BQ insert_rows_json or strftime, both of which handle the
   change).

7. **Import-side change is +1 line per file.** The `timezone` import
   addition is the largest visible diff -- intentional and minimal.

## Closes

- Task list items #43, #44, #45, #46 (4 follow-ups)
- masterplan step **phase-16.36** (immutable verification PASS)

## Next

Spawn Q/A to audit deterministic checks + LLM judgment. If PASS:
log + flip + continue with the next follow-up bundle or masterplan
phase-10.7.5.
