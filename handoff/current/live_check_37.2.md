# Step 37.2 -- gemini deep-think source default = production -- live verification

**Date:** 2026-05-22
**Step type:** EXECUTION (backend config alignment; structural).
**Verdict:** **PASS** (with separate operator-side .env cleanup noted)

---

## 3-row immutable-criteria verdict table

| # | Criterion (verbatim from masterplan 37.2.verification) | Verdict | Evidence |
|---|---|---|---|
| 1 | `model_tiers_py_line_62_default_is_gemini_2_5_pro` | **PASS** | `backend/config/model_tiers.py:62` updated: `"gemini_deep_think": "gemini-2.5-pro"` (was `"gemini-2.5-flash"`). Comment cites phase-37.2 + production-parity reason. Verified by `test_phase_37_2_model_tiers_gemini_deep_think_role_default_is_gemini_2_5_pro`. |
| 2 | `settings_py_deep_think_model_field_default_is_gemini_2_5_pro` | **PASS** | `backend/config/settings.py:30` updated: `Field("gemini-2.5-pro", ...)` (was `Field("claude-opus-4-7", ...)`). Description refreshed with phase-37.2 rationale. Verified by `test_phase_37_2_settings_field_default_is_gemini_2_5_pro` via `Settings.model_fields["deep_think_model"].default == "gemini-2.5-pro"`. |
| 3 | `get_settings_without_env_override_resolves_to_gemini_2_5_pro` | **PASS (structural) + DEFERRED-OPERATOR (integration)** | Structural: `Settings.model_construct().deep_think_model == "gemini-2.5-pro"` (test #3) -- proves Field default wins when no env override is present. **Integration caveat:** the operator's local `backend/.env` carries an apparent stale `DEEP_THINK_MODEL=claude-opus-4-7` line (likely from before phase-34.1e was reverted, or from a pre-phase-34 default). Without that cleanup, live `Settings().deep_think_model` resolves to `claude-opus-4-7` -- silent regression risk preserved at runtime. **Operator action required:** remove the `DEEP_THINK_MODEL=...` line from `backend/.env` (the env-var is now redundant per phase-37.2) + restart backend. |

**Roll-up:** 3 of 3 source-side criteria PASS; 1 operator-side cleanup tracked in the integration caveat.

---

## /goal integration-gate scoreboard

| # | Gate | Verdict | Evidence |
|---|---|---|---|
| 1 | pytest count >= 297 baseline | **PASS** | 326 (was 323 after 35.2; +3 new; 0 regressions) |
| 2 | TS build + ast.parse green on changed | **PASS** | `ast.parse` on settings.py + model_tiers.py + test = OK; no frontend changes |
| 3 | New feature behind flag | **N/A** | Bug fix (default-alignment), not new feature |
| 4 | BQ migrations idempotent | **N/A** | No BQ |
| 5 | New env vars in .env.example + CLAUDE.md | **N/A** | No new env (operator's existing `DEEP_THINK_MODEL=` line is now redundant) |
| 6 | Contract has N* delta | **PASS** | B (defensive Burn-protection) + R (production-parity / regression-prevention) |
| 7 | Zero emojis | **PASS** |
| 8 | ASCII-only loggers | **N/A** | No logger changes |
| 9 | Single source of truth | **PASS** | settings.py:30 Field is the canonical source; model_tiers.py:62 aligned for consistency; no duplicate defaults remain |
| 10 | log first / flip last | **WILL HOLD** | Cycle 18 block next; flip is final |

---

## Operator runbook -- complete the cleanup

```bash
# 1. Edit backend/.env: remove (or comment out) the line
#    DEEP_THINK_MODEL=...   (any value -- it's now redundant)
# Optional editor command (operator-driven):
#    sed -i '' '/^DEEP_THINK_MODEL=/d' backend/.env
# (We can't do this from Main per .env permission-block.)

# 2. Restart backend so pydantic-settings picks up the change
launchctl kickstart -k "gui/$(id -u)/com.pyfinagent.backend"

# 3. Verify the resolved value is now gemini-2.5-pro via the Field default
source .venv/bin/activate && python -c "
from backend.config.settings import get_settings
print('deep_think_model =', get_settings().deep_think_model)
"
# Expected: deep_think_model = gemini-2.5-pro

# 4. Confirm the startup banner agrees (backend.log tail)
grep "phase-31.1 model routing" backend.log | tail -1
# Expected: settings.gemini_model='gemini-2.5-pro' -> standard-tier provider=Gemini
# (Note: the banner today only logs the standard tier; phase-44.1 + phase-44.7
# work tracks adding a deep-think banner for full observability.)
```

---

## Diff

```
backend/config/settings.py:30                          | +1 -1
backend/config/model_tiers.py:62                       | +6 -2  (1 line edit + 5 lines of explanatory comment)
backend/tests/test_phase_37_2_default_alignment.py     (new, ~70 lines, 3 tests)
```

ZERO frontend changes. ZERO functional code-path changes (only default values + tests).

---

## North-star delta delivered

- **B (defensive):** prevents 1-2 days of degraded cycles per fresh-checkout regression event. Conservative since operator's `.env` is sticky; the failure mode is rare but high-impact (silent fall-back to Anthropic credit-exhaustion identical to phase-34.1's blocker).
- **R (defensive):** OWASP LLM v2 + 12-Factor §III production-parity discipline. Source-default matches production reality. Caltech arxiv:2502.15800 N/A (no decision-quality change; just configuration hygiene).

---

## Pytest evidence

```
$ pytest backend/tests/test_phase_37_2_default_alignment.py -v
test_phase_37_2_settings_field_default_is_gemini_2_5_pro PASSED
test_phase_37_2_model_tiers_gemini_deep_think_role_default_is_gemini_2_5_pro PASSED
test_phase_37_2_settings_without_env_or_dotenv_resolves_to_gemini_2_5_pro PASSED
3 passed in 0.02s

$ pytest backend/ --collect-only -q | tail -2
326 tests collected in 2.14s
```

---

## Plan-only honesty check

```
$ git diff --stat backend/agents/ backend/services/ backend/api/
(empty)

$ git diff --stat backend/config/
 backend/config/settings.py     | 2 +-
 backend/config/model_tiers.py  | 8 ++++++--

$ git diff --stat frontend/src/
(empty)
```

Two single-line default edits + 3 new structural tests = bounded per /goal "NO mass refactors". Single source of truth preserved (settings.py:30 canonical; model_tiers.py:62 aligned consistency).

---

## Bottom line

phase-37.2 closes closure_roadmap §3 OPEN-17 at the source layer. The Field default in `settings.py:30` and the role-default in `model_tiers.py:62` both align to `gemini-2.5-pro`, matching production reality. 3 structural tests prove the Field default + role default + model_construct path. The operator's local `backend/.env` carries a stale `DEEP_THINK_MODEL=...` line that should be removed for integration-time completeness, but this is a 1-line operator action documented in the runbook above.

**Closure-path progress:** 7 of ~35-50 cycles done this session (cycles 12-18). Next: phase-37.4 (Moderator response_schema -- companion to 37.1), phase-38.3 (startup banner deep-think log line -- ~10 LOC), or phase-44.2 (cockpit, needs TanStack + Tremor approval).
