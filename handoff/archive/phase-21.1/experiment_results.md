---
step: phase-21.1
cycle_date: 2026-04-26
forward_cycle: true
expected_verdict: PASS
deliverables:
  - backend/config/settings.py (+1 field: apply_model_to_all_agents)
  - backend/config/model_tiers.py (+_GEMINI_LOCKED_ROLES + override branch in resolve_model)
  - backend/tests/test_apply_model_to_all_agents.py (NEW, 10 tests)
---

# Experiment Results -- phase-21.1

## What was done

When operator toggles "Apply to all agents" in settings, the Standard
model selector (`gemini_model` field) now propagates to ALL Anthropic-routed
roles in `_BUILD_TIER`. Gemini-locked roles (RAG, Search Grounding,
structured output) still use their hardcoded `gemini-2.0-flash` /
`gemini-2.5-flash` values regardless.

## Deliverables

### `backend/config/settings.py`

```python
+ apply_model_to_all_agents: bool = Field(False, description="phase-21.1: ...")
```

### `backend/config/model_tiers.py`

```python
+ _GEMINI_LOCKED_ROLES: frozenset[str] = frozenset({
+     "gemini_enrichment",
+     "gemini_deep_think",
+ })

  def resolve_model(role, tier=None):
+     # Validate role FIRST so unknown roles always raise.
+     if role not in _BUILD_TIER:
+         raise KeyError(...)

      if tier is None:
          s = get_settings()
+         override_active = bool(getattr(s, "apply_model_to_all_agents", False))
+         override_model = (getattr(s, "gemini_model", "") or "").strip()
+         if override_active and override_model and role not in _GEMINI_LOCKED_ROLES:
+             return override_model
      ...
```

### `backend/tests/test_apply_model_to_all_agents.py` (NEW, 10 tests)

1. `test_default_off_returns_per_role_models` -- baseline, no regression
2. `test_default_off_gemini_roles_unchanged`
3. `test_override_applies_to_mas_main`
4. `test_override_applies_to_all_anthropic_roles` -- 7 roles
5. `test_override_skips_gemini_enrichment` -- still returns gemini-*
6. `test_override_skips_gemini_deep_think`
7. `test_gemini_locked_roles_set_is_correct` -- structural invariant
8. `test_override_on_but_no_gemini_model_value_falls_through` -- empty string fallback
9. `test_explicit_tier_arg_bypasses_settings` -- tier kwarg unaffected
10. `test_unknown_role_still_raises` -- the role-validation-first fix

## Verification (verbatim, immutable from masterplan)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_apply_model_to_all_agents.py -v
============================== 10 passed in 0.02s ==============================
```

## Files touched

| Path | Action |
|------|--------|
| backend/config/settings.py | edit (+1 field) |
| backend/config/model_tiers.py | edit (+10 lines: _GEMINI_LOCKED_ROLES + role-validation-first refactor + override branch) |
| backend/tests/test_apply_model_to_all_agents.py | CREATED (~135 LOC, 10 tests) |
| handoff/current/{contract,experiment_results,phase-21.1-research-brief}.md | rewrite |

## Success criteria

| # | Criterion | Result |
|---|-----------|--------|
| 1 | Override flag added to settings.py | PASS |
| 2 | _GEMINI_LOCKED_ROLES defined | PASS (2 entries) |
| 3 | resolve_model honors override for non-Gemini roles | PASS |
| 4 | Gemini roles bypass override | PASS |
| 5 | Unknown role still raises with override on | PASS (caught a bug; fixed) |
| 6 | 10/10 tests pass | PASS |
| 7 | No regression on existing resolve_model behavior (override off) | PASS |

## Honest disclosures

1. **Real bug caught by test #10.** First implementation put the override
   branch BEFORE role validation. With the flag on, an unknown role
   silently returned the override model. Fixed by hoisting role validation
   above the override branch. Test #10 is the regression guard.
2. **No code-side changes to existing callers** (agent_definitions, autoresearch). They keep calling `resolve_model("mas_main")` etc.; the indirection picks up the override automatically.
3. **No frontend changes** (phase-21.2 scope).
4. **Cycle-2 not needed** (the bug fix happened during the same generate cycle, before Q/A).
5. **Forward note:** the field name `gemini_model` is misleading (it's actually the Standard model selector for ANY provider). Refactor to `default_standard_model` is a separate cycle to avoid breaking the existing Settings UI binding.

## Closes

UAT-21.1.

## Next

Spawn Q/A. After PASS: log + flip + archive. Then phase-21.2 (frontend toggle).
