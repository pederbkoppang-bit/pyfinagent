# Contract -- Claude as default LLM provider (task #49)

## Research gate

- Researcher spawn: 2026-04-24. Brief at `handoff/current/claude-default-research-brief.md`.
- JSON envelope: tier=moderate, external_sources_read_in_full=5 (floor 5), urls_collected=13, recency_scan=true, internal_files_inspected=8, gate_passed=true.
- Routing layer already supports Claude via `backend/agents/llm_client.py::make_client` (claude-* prefix -> ClaudeClient). Only defaults + one hard-coded model name in autonomous_loop.py need to change.
- Recommended defaults: `standard_model = claude-sonnet-4-6`, `deep_think_model = claude-opus-4-6`.
- Monday fallback confirmed robust: `_run_single_analysis` catches any Claude failure (including 401 on the current OAuth-token key) and falls through to `AnalysisOrchestrator.run_full_analysis` on Gemini.
- Three features stay Gemini-only regardless of selection: RAG (Vertex AI Search), Google Search Grounding, Vertex structured-output schemas. Already guarded by `_resolve_gemini` at `orchestrator.py:310-315`.
- Stay model-name-driven (no new `default_provider` enum) per LiteLLM/Portkey production pattern.

## Hypothesis

Today the defaults are Gemini, the autonomous_loop hard-codes `claude-sonnet-4-6` without touching settings, and the settings page's primary-model picker lacks Claude options. Two net changes land Claude-as-default without breaking the Monday Gemini fallback:

1. Flip the default model names in `settings.py` to Claude.
2. Make `autonomous_loop._run_claude_analysis` honor `settings.gemini_model` (renamed conceptually to "standard model" but the field name stays for backward compat) instead of hard-coding a Claude model.
3. Expose Claude models in the settings UI + keep the Gemini toggle.

## Planned change (MINIMUM scope)

### 1. `backend/config/settings.py` -- flip defaults

```python
gemini_model: str = Field("claude-sonnet-4-6", description="Standard-tier model for enrichment + debate. Claude default; Gemini still available via settings UI. Field name kept for backward compat.")
deep_think_model: str = Field("claude-opus-4-6", description="Deep-think-tier model for Moderator/Critic/Synthesis/RiskJudge. Claude default. Gemini still selectable.")
```

### 2. `backend/services/autonomous_loop.py::_run_claude_analysis` -- honor settings

Replace hard-coded `model="claude-sonnet-4-6"` and the raw `anthropic.Anthropic` client with routing via `make_client(settings.gemini_model)`. If `make_client` returns a non-Anthropic client (user switched to Gemini), the function still works because `generate_content` is provider-agnostic. Robust fallback stays intact (the `except Exception` wrapper at `autonomous_loop.py:_run_single_analysis` catches any failure and falls through to Gemini orchestrator).

### 3. `backend/api/settings_api.py` -- ensure `_VALID_MODELS` lists Claude

Already lists them per research. Verify + leave.

### 4. `frontend/src/app/settings/page.tsx` -- surface Claude options + Gemini-only notice

- Ensure the primary-model + deep-think dropdowns include `claude-opus-4-6`, `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5` from the backend `/api/settings/models` catalog.
- Add a one-paragraph info banner under the model pickers: "Claude is the default LLM provider. Switch to Gemini by selecting a gemini-* model above. Three features always use Gemini regardless of selection: RAG (Vertex AI Search), Google Search Grounding, and Vertex structured-output schemas -- these are Google-only APIs."

### 5. NOT in scope

- Fixing the OAuth-token-vs-API-key issue in backend/.env. That's a Peder-only task (generate real key at console.anthropic.com). Code-side fallback to Gemini keeps the system functional.
- Adding a `default_provider` enum field. Staying model-name-driven per LiteLLM production pattern.
- Changing Gemini-only features (RAG/grounding/structured-output).

## Immutable success criteria

1. `settings.py` `gemini_model` default is `claude-sonnet-4-6` (grep-able).
2. `settings.py` `deep_think_model` default is `claude-opus-4-6` (grep-able).
3. `autonomous_loop.py::_run_claude_analysis` does NOT contain the literal string `"claude-sonnet-4-6"` as an arg to `client.messages.create(model=...)` (i.e., uses the settings value, not a hardcoded model).
4. `settings.py` imports clean: `python -c "from backend.config.settings import Settings; s=Settings(); print(s.gemini_model, s.deep_think_model)"` prints the Claude defaults.
5. `autonomous_loop.py` imports clean after edit.
6. `frontend/src/app/settings/page.tsx` contains the string "Claude is the default" (from the info banner) OR a clearly visible "Claude" label in the dropdown defaults.
7. Zero-orders drill still PASSes (`python scripts/go_live_drills/zero_orders_drill.py` prints PASS) -- behavior unchanged on the code path.
8. Gemini fallback still fires on ANTHROPIC_API_KEY 401: a synthetic test that invokes `_run_single_analysis` with bad creds must still return a non-None analysis (via Gemini fallback path).
9. Frontend build passes: `npm run build` in `frontend/` returns exit 0.
10. All syntax: `ast.parse` on each edited .py file exits 0.

## Verification command (Q/A reproduces)

```bash
source .venv/bin/activate
grep -c 'Field("claude-sonnet-4-6"' backend/config/settings.py
grep -c 'Field("claude-opus-4-6"' backend/config/settings.py
grep -c '"claude-sonnet-4-6"' backend/services/autonomous_loop.py
# -> 0 for hardcoded; acceptable to match as default-fallback literal if unavoidable, but not as client.messages.create model arg
python -c "from backend.config.settings import Settings; s=Settings(); print('std=',s.gemini_model,' deep=',s.deep_think_model)"
python -c "import ast; ast.parse(open('backend/config/settings.py').read()); ast.parse(open('backend/services/autonomous_loop.py').read()); print('SYNTAX_OK')"
python -c "from backend.services import autonomous_loop; print('IMPORT_OK')"
python scripts/go_live_drills/zero_orders_drill.py
grep -c "Claude is the default" frontend/src/app/settings/page.tsx
cd frontend && npm run build 2>&1 | tail -5
```

All must succeed (counts per criterion, drill PASS, build green).

## References

- `handoff/current/claude-default-research-brief.md` (research deliverable)
- `backend/config/settings.py` (target)
- `backend/services/autonomous_loop.py` (target)
- `frontend/src/app/settings/page.tsx` (target)
- Model catalog: `backend/config/model_tiers.py`
