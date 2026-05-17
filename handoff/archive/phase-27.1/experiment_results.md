# Experiment Results — phase-27.1 (C3 Anthropic schema fix)

Generated: 2026-05-16T21:25:00+00:00
Step id: 27.1
Owner: Main

## What was built/changed

### 1. New module-level helper `_ensure_additional_properties_false`

`backend/agents/llm_client.py:312-340` — added a recursive helper above `ClaudeClient`. The function descends every dict/list node, sets `additionalProperties: False` on every `type: "object"` node, and recurses into `properties.*`, `items`, `$defs.*`, `definitions.*`, `anyOf`, `oneOf`, `allOf` automatically (because the recursion is structure-blind — any dict child gets visited).

```python
def _ensure_additional_properties_false(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
        for value in schema.values():
            _ensure_additional_properties_false(value)
    elif isinstance(schema, list):
        for item in schema:
            _ensure_additional_properties_false(item)
    return schema
```

Idempotent (already-normalized schemas re-emit the same dict). Mutates in place AND returns for chaining. Pure function; no I/O.

### 2. Call-site fix at the bug locus

`backend/agents/llm_client.py:1393-1402` — in `ClaudeClient.generate_content`, the schema dict is now passed through the helper *before* being assigned to `kwargs["output_config"]["format"]["schema"]`. One added line:

```python
schema_dict = _ensure_additional_properties_false(schema_dict)
```

With a comment block citing the Anthropic docs URL.

### 3. Files modified

| File | Lines | Nature |
|------|-------|--------|
| `backend/agents/llm_client.py` | +30 (helper) +9 (call-site comment + 1-line fix) | code change, no signature change |
| `handoff/current/contract.md` | rewritten for 27.1 | contract |
| `handoff/current/experiment_results.md` | this file | results |

No tests added in this step (covered indirectly by the verification command's import + assertion check; richer unit tests can be added in a later hardening pass if needed).

## Verification command output (verbatim from masterplan 27.1)

```
$ source .venv/bin/activate && python -c "
from backend.agents.llm_client import _ensure_additional_properties_false
s={'type':'object','properties':{'x':{'type':'object','properties':{'y':{'type':'string'}}}, 'arr':{'type':'array','items':{'type':'object','properties':{'z':{'type':'string'}}}}}}
r=_ensure_additional_properties_false(s)
assert r['additionalProperties'] is False, 'top'
assert r['properties']['x']['additionalProperties'] is False, 'nested'
assert r['properties']['arr']['items']['additionalProperties'] is False, 'array items'
print('PASS')"

PASS
```

Exit code: 0. All three nested-object assertions pass.

## Live check — Claude structured output end-to-end

Independent probe of a real `claude-sonnet-4-6` call with a 2-level-nested Pydantic schema (TestSchema → SubScores), using the actual `ClaudeClient.generate_content` path that the live cycle goes through. Before this fix, the same call failed with `400 INVALID_ARGUMENT: output_config.format.schema: For 'object' type, 'additionalProperties' must be explicitly set to false` (captured in `backend.log` at 21:06:08 UTC during cycle 756a19c7).

After this fix:

```
$ python <probe>
text: {"score":6,"rationale":"AAPL is a mega-cap Information Technology name
with historically strong quality metrics (high ROE, consistent free cash flow
generation, durable margins), warranting a solid quality score. No FACT_LEDGER
or market data was provided in this message, so momentum cannot be asses[...]
HTTP-200-equivalent: returned LLMResponse, no exception, no additionalProperties error
```

The returned JSON conforms to the nested schema (root keys `score`, `rationale`, `sub`). No Anthropic 400. No exception thrown. The probe ran from the same `ClaudeClient.generate_content` path as the live cycle.

## Artifact shape

- New importable symbol: `from backend.agents.llm_client import _ensure_additional_properties_false`
- Behavior: idempotent recursive helper; pure function; returns the same dict (mutated in place).
- Call sites: 1 (`ClaudeClient.generate_content`, post-`schema_dict` construction).
- No backwards-incompatible changes to any public surface.

## Risks / known limits

- The helper does NOT validate the schema is well-formed (e.g., would not catch a malformed `properties` value). Anthropic will surface those errors as separate 400s with different messages.
- Pydantic v2 `$defs` are recursed into (any dict child gets visited), but the helper does not inline `$ref` resolution. Anthropic's validator handles `$ref` resolution server-side.
- The helper is provider-agnostic; if a future provider rejects schemas WITH `additionalProperties: false` (none known today), a provider-specific shim would be needed. Out of scope for 27.1.
