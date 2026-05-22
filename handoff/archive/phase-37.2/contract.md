# phase-37.2 -- gemini deep-think source default = production (OPEN-17)

**Step id:** `phase-37.2`
**Date:** 2026-05-22
**Mode:** EXECUTION (backend config alignment).
**Cycle:** Cycle 18 (after Cycle 17 phase-35.2).

---

## North-star delta

**Terms:** B (defensive Burn-protection) + R (defensive regression-prevention).

**B:** Eliminates silent quality regression on fresh checkout / restart. Without the fix, a new operator clone or a backend restart that loses the `DEEP_THINK_MODEL=gemini-2.5-pro` env override silently falls back to `claude-opus-4-7` (Anthropic API) -- triggering credit-exhaustion errors identical to phase-34.1's blocker. Estimate: prevents 1-2 days of degraded cycles per regression event. Conservative since the operator's `.env` is sticky; the failure mode is rare but high-impact.

**R:** OWASP LLM v2 + 12-Factor §III config-from-code: production parity discipline. Source-default-trails-production is the documented anti-pattern (OneUptime 2026-01-30 + 12-Factor canonical). N* delta: small but defensive -- protects the 6-cycle progress from being undone by a fresh checkout regression.

**P:** N/A (no trading-logic change).

**How measured:** `Settings().deep_think_model == "gemini-2.5-pro"` (regression test); fresh shell `unset DEEP_THINK_MODEL && python -c "from backend.config.settings import Settings; print(Settings().deep_think_model)"` returns `gemini-2.5-pro` (post-fix).

---

## Research-gate compliance

**Researcher SPAWNED** per operator override `feedback_never_skip_researcher` (2026-05-22). Simple-tier brief at `handoff/current/research_brief_phase_37_2.md`:
- gate_passed: true
- external_sources_read_in_full: 5 (5-source floor met for simple tier)
- internal_files_inspected: 9
- recency scan performed (Gemini 3 series Q1 2026 noted but out-of-scope; within 2.5 generation, Pro is the deep-think choice)
- 3-variant queries logged

Key finding: settings.py:30 Field default = `"claude-opus-4-7"` is the LOAD-BEARING site (used by `get_settings().deep_think_model` and routes via `llm_client.py::make_client`). model_tiers.py:62 (`gemini_deep_think = "gemini-2.5-flash"`) is currently dead code (no callsite resolves that role) but should be aligned for consistency.

---

## Hypothesis

> If we change `backend/config/settings.py:30` Field default from
> `"claude-opus-4-7"` to `"gemini-2.5-pro"` AND
> `backend/config/model_tiers.py:62` `_BUILD_TIER["gemini_deep_think"]`
> from `"gemini-2.5-flash"` to `"gemini-2.5-pro"`, AND add a regression
> test asserting `Settings().deep_think_model == "gemini-2.5-pro"`,
> THEN a fresh checkout / restart without the env override will resolve
> to production-parity gemini-2.5-pro (vs the silent claude-opus-4-7
> regression that today triggers Anthropic credit-exhaustion).

---

## Immutable success criteria (verbatim from masterplan 37.2.verification)

1. `model_tiers_py_line_62_default_is_gemini_2_5_pro`
2. `settings_py_deep_think_model_field_default_is_gemini_2_5_pro`
3. `get_settings_without_env_override_resolves_to_gemini_2_5_pro`

Plus /goal integration gates 1-10.

---

## Plan steps

| # | Step | Status |
|---|---|---|
| 1 | Live-loop health check + locate fix sites | DONE |
| 2 | Researcher (simple tier, 5+ external sources, gate_passed=true) | DONE |
| 3 | Write contract | IN FLIGHT |
| 4 | Edit settings.py:30 + model_tiers.py:62 + add regression test | NEXT |
| 5 | pytest + verify total count >= 323 | NEXT |
| 6 | live_check_37.2.md + Q/A + harness_log Cycle 18 + flip 37.2 | NEXT |

---

## Files this step touches

- `backend/config/settings.py:30` -- Field default change (1 line + docstring refresh)
- `backend/config/model_tiers.py:62` -- dict-value change (1 line)
- `backend/tests/test_phase_37_2_default_alignment.py` (NEW, ~50 lines, 3 tests)

**NOT changed:** anywhere else; no migration; no env-var change (the operator's existing `DEEP_THINK_MODEL=gemini-2.5-pro` becomes redundant but harmless).

---

## References

- research_brief_phase_37_2.md (cycle 18 researcher output, 5 sources in full)
- closure_roadmap.md §3 OPEN-17 + §5 N* delta table
- backend/config/settings.py:30 (LOAD-BEARING site)
- backend/config/model_tiers.py:62 (cosmetic consistency site)
- phase-34.1e archive: where env override was first set
- /goal directive (10 integration gates; researcher mandatory per feedback_never_skip_researcher)
