---
step: phase-16.36
cycle_date: 2026-04-25
agent: qa (merged qa-evaluator + harness-verifier)
verdict: PASS
---

# Phase-16.36 Q/A critique -- Anthropic-fallback hardening bundle

## Critical: harness-compliance audit (5/5)

| # | Check | Result |
|---|-------|--------|
| 1 | Research gate (`phase-16.36-research-brief.md` exists; envelope shows `gate_passed:true`, `external_sources_read_in_full:6`, `urls_collected:16`, `recency_scan_performed:true`) | PASS |
| 2 | Contract-before-GENERATE (`handoff/current/contract.md` line 2 = `step: phase-16.36`) | PASS |
| 3 | Experiment results present (`handoff/current/experiment_results.md` line 2 = `step: phase-16.36`) | PASS |
| 4 | Log-last (`grep -c phase-16.36 handoff/harness_log.md` = 0) | PASS |
| 5 | No verdict-shopping (prior critique was pinned at phase-10.7.4 PASS until this overwrite) | PASS |

## Deterministic checks

```
python -c "from backend.agents.multi_agent_orchestrator import reset_anthropic_client; \
           reset_anthropic_client(); print('reset ok')" \
  && test -z "$(grep -rn 'datetime.utcnow()' backend/tools backend/agents \
       backend/backtest backend/db backend/slack_bot backend/services \
       backend/tests/test_outcome_tracker.py)" && echo "utcnow audit clean" \
  && python -m pytest backend/tests/test_anthropic_fallback.py -v
```
- `reset ok` -- module import + reset_anthropic_client() succeeds.
- `utcnow audit clean` -- zero `datetime.utcnow()` hits across the 7 sweep targets.
- pytest: **6 passed in 0.02s**.
- Compound `&&` exit code: **0** (immutable verification met).

Repo-wide audit: `grep -rn 'datetime.utcnow()' backend/` returns zero hits — sweep is complete, not just scoped.

Regression sweep: `pytest backend/tests/test_outcome_tracker.py backend/tests/test_anthropic_fallback.py tests/meta_evolution/` -- **52 passed in 3.42s**.

## Code spot-checks (LLM judgment leg)

- **`reset_anthropic_client`** (multi_agent_orchestrator.py): 19 lines, signature `() -> None`, contains all three required clears: `_client`, `_anthropic_unavailable`, and `get_settings.cache_clear()`. Matches researcher's lru_cache invalidation guidance.
- **`_gemini_text_call` #46 patch** (multi_agent_orchestrator.py L222-241): uses getattr-safe extraction `getattr(_umeta, "prompt_token_count", 0) or 0` -- mirrors `cost_tracker.py:128-129` pattern. Returns same `{"input": _, "output": _}` shape as the Anthropic branch. Won't crash on missing/None UsageMeta.
- **`outcome_tracker.py` special case** (L48 and L110-112): both subtraction sites strip tzinfo via `.replace(tzinfo=None)` and L110 also normalizes the BQ-sourced `rec_date` defensively. Comments cite phase-16.36 and explain the aware/naive TypeError this prevents. Correct pattern; no functional behavior change.
- **`fred_data.py` spot-check**: clean `from datetime import datetime, timedelta, timezone` + `datetime.now(timezone.utc)` usage. No residual `utcnow()`.
- **Mock test pattern** (test_anthropic_fallback.py:47): `monkeypatch.setitem(sys.modules, "anthropic", fake_mod)` is the documented researcher recommendation -- sidesteps the real `AuthenticationError` requiring `httpx.Response` plumbing. 6 tests (plan said 5; +1 defensive `reset_is_safe_when_no_orchestrator_exists`) -- exceeds floor.

## Scope honesty

Files-touched table in `experiment_results.md` enumerates 12 phase-16.36 files (10 datetime sweep + 1 orchestrator + 1 new test). Other entries in `git status` are accumulated rolling state from prior phases (calendar→econ_calendar rename, archive moves, untracked meta_evolution dirs) and are not mutated by this bundle. Scope disclosure is honest.

## Anti-rubber-stamp

Mutation resistance: `test_anthropic_unavailable_flag_persists_after_first_401` and `test_reset_anthropic_client_clears_flags_and_settings_cache` directly probe the #43+#44 surface (set flag, assert persistence, then reset, assert clear). The #46 fix is exercised by `test_gemini_usage_dict_populated`. Real behavior tested, not just stubs.

## Verdict

PASS. All four bundle items (#43 sweep, #44 reset helper, #45 mock test pattern, #46 usage extraction) land cleanly. Immutable verification exits 0; 6/6 fallback tests + 52/52 regression sweep green; harness-compliance audit 5/5; scope honest.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "Immutable verification cmd exit=0; utcnow sweep clean repo-wide (0 hits in backend/); 6/6 test_anthropic_fallback PASS; 52/52 regression sweep PASS; reset_anthropic_client clears _client+_anthropic_unavailable+get_settings.cache_clear; _gemini_text_call uses getattr-safe usage extraction mirroring cost_tracker.py:128-129; outcome_tracker special case correctly normalizes tz at L48 and L110-112; mock pattern uses monkeypatch.setitem(sys.modules,'anthropic',...) per researcher recommendation; harness-compliance audit 5/5; scope honest (12 phase-16.36 files match experiment_results table).",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["research_gate_envelope", "contract_step_id", "experiment_results_step_id", "log_last_zero_hits", "no_verdict_shopping", "syntax_import", "verification_command_compound", "utcnow_repo_wide_audit", "pytest_anthropic_fallback", "pytest_regression_sweep", "reset_function_introspection", "gemini_usage_extraction_pattern", "outcome_tracker_special_case", "mock_pattern_review", "scope_honesty"]
}
```
