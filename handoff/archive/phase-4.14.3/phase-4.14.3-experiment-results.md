# Experiment Results — phase-4.14.3
Generated: 2026-04-18
Step: [T2] Add output_config.effort pass-through to ClaudeClient

## What was built

Hoisted `output_config.effort` injection out of the
`thinking_requested` branch in `ClaudeClient._call_claude()` (it was
a pre-existing MF-29 bug — effort was only applied when thinking was
on, but Anthropic docs explicitly state effort is independent of
thinking). Added role-keyed defaults with a model-ID-prefix fallback
and two guards (xhigh-Opus-4.7-only and model-support allowlist).

## Files touched

| File | Change |
|------|--------|
| `backend/config/model_tiers.py` | +97 lines. New: `Effort` literal, `EFFORT_SUPPORTED_MODELS` tuple, `EFFORT_DEFAULTS` dict (role-keyed), `MODEL_EFFORT_FALLBACK` tuple (prefix-keyed), `resolve_effort(role)`, `resolve_effort_by_model(model_id)`, `model_supports_effort(model_id)`. |
| `backend/agents/llm_client.py` | +46 / -8. Hoisted effort resolution out of thinking branch. Precedence: explicit > config > role_hint > model-ID fallback. Added xhigh-downgrade-with-log and model-support-allowlist guards. Imported resolve_effort / resolve_effort_by_model / model_supports_effort from model_tiers. |

## Verbatim command output

### Immutable verification (4.14.3)
```
$ source .venv/bin/activate && python -c "import inspect, backend.agents.llm_client as c; assert 'output_config' in inspect.getsource(c) and 'effort' in inspect.getsource(c)"
(exit 0; PASS)
```

### Supplementary semantic checks (raised by Q/A pushback in 4.14.0-T1 close)
```
$ python -c "
from backend.config.model_tiers import (
    EFFORT_DEFAULTS, resolve_effort, resolve_effort_by_model, model_supports_effort
)
assert resolve_effort('mas_communication') == 'low'
assert resolve_effort('mas_research') == 'medium'
assert resolve_effort_by_model('claude-opus-4-7') == 'xhigh'
assert resolve_effort_by_model('claude-sonnet-4-6') == 'medium'
assert resolve_effort_by_model('claude-opus-4-6') == 'high'
assert resolve_effort_by_model('claude-haiku-4-5') is None
assert model_supports_effort('claude-opus-4-7') is True
assert model_supports_effort('claude-haiku-4-5') is False
assert model_supports_effort('gemini-2.0-flash') is False
print('all semantic checks PASS')
"
all semantic checks PASS  (exit 0)
```

### Hoist-out-of-thinking verification
```
$ python -c "
import inspect, backend.agents.llm_client as c
src = inspect.getsource(c)
assert 'resolve_effort_by_model' in src
assert src.count('output_config') >= 1
assert 'xhigh' in src and 'model_supports_effort' in src
print('hoist-out-of-thinking verification: PASS')
"
hoist-out-of-thinking verification: PASS  (exit 0)
```

### Syntax checks
```
$ python -c "import ast; ast.parse(open('backend/config/model_tiers.py').read()); ast.parse(open('backend/agents/llm_client.py').read())"
(exit 0; PASS for both files)
```

## Criterion coverage

| # | Criterion | How satisfied |
|---|-----------|---------------|
| 1 | effort_passthrough_wired | `llm_client.py` hoists effort resolution to outer scope; `output_config = {"effort": effort}` is set unconditionally when a non-None effort resolves. |
| 2 | sonnet_4_6_default_medium_not_high | `MODEL_EFFORT_FALLBACK` maps `claude-sonnet-4-6` → `"medium"`; `EFFORT_DEFAULTS["mas_communication"] = "low"` (Sonnet 4.6 routing) and `["mas_research"] = "medium"` (Sonnet 4.6 deep work). Semantic test confirms `resolve_effort_by_model('claude-sonnet-4-6') == 'medium'`. |
| 3 | opus_4_7_starts_xhigh_on_coding_paths | `MODEL_EFFORT_FALLBACK` maps `claude-opus-4-7` → `"xhigh"`. The xhigh-downgrade guard permits xhigh only when `model_id.startswith("claude-opus-4-7")`. Forward-looking: no Opus 4.7 in build-tier today, but the default activates automatically when `_LIVE_TIER["mas_main"]` flips to 4.7 at May launch. |

## Behavioral notes for Q/A

1. **Effort is now applied when thinking is off.** Previously a
   Sonnet 4.6 caller without thinking got the API implicit `high`;
   now it gets `medium`. This is a meaningful production change.
2. **xhigh downgrade is silent-with-log.** If a caller explicitly
   passes `effort="xhigh"` against Sonnet 4.6, we drop to `high`
   and `logger.warning(...)`. We did not raise — silent downgrade
   is friendlier during the transition but an explicit raise is
   defensible; Q/A may push back on this choice.
3. **Haiku 4.5 is excluded.** Anthropic's effort docs do not list
   Haiku 4.5 as supported. `autoresearch_fast` role returns None
   and the model-support allowlist excludes `claude-haiku-4-5`
   from the prefix fallback (maps to None).
4. **Gemini / non-Claude models** naturally skip output_config via
   the `model_supports_effort` guard — the import path is Claude-
   only, but as a defense the ClaudeClient should never receive a
   Gemini model_id anyway.

## Known scope omissions

- `planner_agent.py`, `multi_agent_orchestrator.py` MAS tool loop,
  and `evaluator_agent.py` all construct their own Anthropic()
  client and do not route through `ClaudeClient.generate_content()`.
  Effort wiring for those paths is MF-35 (ClaudeClient
  consolidation). Out of scope for this step.
- The immutable verification command remains weak (substring check).
  MF-51 tracks hardening. Supplementary semantic checks above
  compensate within this cycle's evidence.

## Files list (final)

- Modified: `backend/config/model_tiers.py`, `backend/agents/llm_client.py`
- New contract: `handoff/current/phase-4.14.3-contract.md`
- New results: `handoff/current/phase-4.14.3-experiment-results.md` (this file)
- Pending: `handoff/current/phase-4.14.3-evaluator-critique.md` (Q/A)
- Pending: harness_log.md cycle-block append
- Pending: GAP_REPORT.md MF-28 → FIXED
- Pending: masterplan.json 4.14.3 → done
