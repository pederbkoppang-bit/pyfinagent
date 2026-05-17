# Sprint Contract — phase-27.1 (C3: Anthropic strict-mode schema additionalProperties:false)

Generated: 2026-05-16T21:18:00+00:00
Owner: Main (this Claude Code session)
Step id: 27.1
Depends on: 27.0 (done — research gate)

## Research-gate summary

Source: `handoff/current/research_brief.md` §"C3 — Anthropic strict-mode schema" (lines 83-163). Authoritative citation: `https://platform.claude.com/docs/en/docs/build-with-claude/structured-outputs` — Anthropic docs require `additionalProperties: false` on EVERY object-type node (not just the root) when using `output_config.format.type=json_schema`. Q/A independently verified this URL HTTP 200 with corroborating page content (additionalProperties + nested + required).

## Hypothesis

Adding a `_ensure_additional_properties_false(schema)` helper that recursively descends into every dict-valued node of a JSON schema and sets `additionalProperties: False` on every `type: "object"` node will satisfy Anthropic's strict-mode validator without breaking any other provider's schema handling (Gemini/Vertex's `Schema` proto silently ignores the field; OpenAI's strict mode requires the same field). Calling this helper at `backend/agents/llm_client.py:1387` (the exact bug locus per the research brief) before assigning `schema_dict` to `kwargs["output_config"]["format"]["schema"]` will unblock every claude-direct structured-output call.

Falsifier: if the helper mutates a schema in a way that makes Gemini reject it, the unified-schema-prep approach is wrong and we need provider-specific schema pipelines.

## Immutable success criteria (verbatim from `.claude/masterplan.json` step 27.1)

```bash
source .venv/bin/activate && python -c "
from backend.agents.llm_client import _ensure_additional_properties_false
s={'type':'object','properties':{'x':{'type':'object','properties':{'y':{'type':'string'}}},
  'arr':{'type':'array','items':{'type':'object','properties':{'z':{'type':'string'}}}}}}
r=_ensure_additional_properties_false(s)
assert r['additionalProperties'] is False, 'top'
assert r['properties']['x']['additionalProperties'] is False, 'nested'
assert r['properties']['arr']['items']['additionalProperties'] is False, 'array items'
print('PASS')"
```

Plus the live_check: one claude-sonnet-4-6 generate call with a nested-object response_format schema returns HTTP 200 (captured via the next full-cycle run when standard model is flipped to claude in 27.6).

## Plan steps

1. Add `_ensure_additional_properties_false(schema)` as a module-level helper in `backend/agents/llm_client.py` (above `ClaudeClient.generate_content`). Pure function: returns the mutated dict (also mutates in place — fine since callers don't share). Recurses into every dict-typed value. Sets `additionalProperties: False` on every `type: "object"` node.
2. At the bug locus (line 1387 — just before the `kwargs["output_config"]["format"]["schema"] = schema_dict` assignment), wrap: `schema_dict = _ensure_additional_properties_false(schema_dict)`.
3. Run the immutable verification command above.
4. Append a unit test row to a docstring or quick smoke test confirming the helper.
5. Q/A spawn for independent verification.
6. harness_log append.
7. Flip 27.1 to done.

## Anti-patterns to avoid

- Do NOT add per-Pydantic-class `ConfigDict(extra="forbid")` everywhere — that's brittle and gives the runtime no defense against schemas authored elsewhere. The runtime helper is the right enforcement point.
- Do NOT change the call signature of `ClaudeClient.generate_content`.
- Do NOT touch the Gemini path's `_flatten_schema` (which already strips unsupported keys for Vertex).

## References

- `handoff/current/research_brief.md` lines 83-163 (C3 section + code skeleton)
- `backend/agents/llm_client.py:1379-1395` (current bug locus)
- Anthropic docs: `https://platform.claude.com/docs/en/docs/build-with-claude/structured-outputs`
- `.claude/masterplan.json` phase-27 step 27.1 verification command (immutable)
