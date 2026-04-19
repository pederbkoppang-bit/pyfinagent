# Sprint Contract -- phase-11.2 Migrate Trivial Callers

**Written:** 2026-04-19 PRE-commit.
**Step id:** `11.2` in phase-11.
**Immutable verification:** `source .venv/bin/activate && pytest backend/tests/test_evaluator_agent.py -q && python -W error::DeprecationWarning -c "from backend.agents import evaluator_agent"` exit 0.

## Research-gate summary

Researcher envelope `{tier: moderate, external_sources_read_in_full: 6, snippet_only_sources: 4, urls_collected: 10, recency_scan_performed: true, internal_files_inspected: 7, gate_passed: true}`. Three-query-variant discipline confirmed.

**Critical finding not in the migration doc:** `debate.py` has a **DEAD `GenerativeModel` import at line 18** — nothing inside the module instantiates `GenerativeModel`; all calls route through the existing `LLMClient` abstraction. Real work in `debate.py` is just to remove the dead import, not to re-wire it. The migration doc's "moderate" label for `debate.py` was overstated by the phase-11.0 research.

**Secondary finding:** `ModeratorConsensus` / `CriticVerdict` schemas use `Field(default_factory=list)`, which hits the still-open python-genai GitHub issue #699 when passed as `response_schema=`. Not in scope for phase-11.2 because `debate.py` doesn't actually call the new SDK — but this is a known-landmine to document for phase-11.3 (when `orchestrator.py` + `llm_client.py` migrate structured-output paths for real).

Staked migration order adopted: **evaluator_agent.py → skill_optimizer.py → debate.py**.

## Hypothesis

With the shim in place (phase-11.1), swapping three files' `vertexai.generative_models.GenerativeModel` imports to `backend.agents._genai_client.get_genai_client` + updating the evaluator test's `VERTEX_AVAILABLE` patching to `GENAI_AVAILABLE` removes three of the eight call sites and lands the first DeprecationWarning suppression.

## Success criteria

**Functional:**
1. `backend/agents/evaluator_agent.py` — remove the conditional `try: from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration` block + the `VERTEX_AVAILABLE` flag. Replace with:
   - `from backend.agents._genai_client import get_genai_client` at module top.
   - `GENAI_AVAILABLE` flag set via `get_genai_client() is not None` (lazy probe).
   - `self.model = get_genai_client()` in `__init__`; a None client triggers the mock-evaluator path.
   - `_call_model` uses `self.model.models.generate_content(model=self.model_name, contents=prompt, config=...)`.
2. `backend/agents/skill_optimizer.py` — replace `_get_model()` lines 98-110 with a thin call to `get_genai_client()` that returns `(client, model_name)`. Call sites using `self._model.generate_content(...)` update to `self._model.models.generate_content(model=name, ...)` — but only IF they exist. The researcher says `_get_model()` is called from `compute_metric`; audit + update.
3. `backend/agents/debate.py` — remove the DEAD `from vertexai.generative_models import GenerativeModel` at line 18. Verify no caller is broken by the removal (it's dead code per research; pre-Q/A self-check will grep for `GenerativeModel` across the file to confirm).
4. `backend/tests/test_evaluator_agent.py` — rename every `VERTEX_AVAILABLE` monkeypatch to `GENAI_AVAILABLE`. Plus any assertion on the attribute name. Existing tests must still pass; no new tests required for this step.
5. **DeprecationWarning suppression target**: `python -W error::DeprecationWarning -c "from backend.agents import evaluator_agent"` must exit 0 (no `vertexai` import path reached during evaluator_agent's module-load).
6. 3 vertexai.generative_models imports removed; inventory drops from 8 to 5. Q/A grep-verifies.

**Correctness verification commands:**
- `python -c "import ast; ast.parse(open('backend/agents/evaluator_agent.py').read()); ast.parse(open('backend/agents/skill_optimizer.py').read()); ast.parse(open('backend/agents/debate.py').read()); print('ok')"` exit 0.
- Immutable: `pytest backend/tests/test_evaluator_agent.py -q && python -W error::DeprecationWarning -c "from backend.agents import evaluator_agent"` exit 0.
- Inventory: `grep -rn "from vertexai.generative_models\|from vertexai import generative_models" backend/ --include="*.py" | grep -v __pycache__ | wc -l` should be `5` (was 8; -3 this cycle: evaluator_agent, skill_optimizer, debate).
- Regression: `pytest backend/tests/test_skill_optimizer.py backend/tests/test_regime_detector.py backend/tests/test_planner_agent.py backend/tests/test_evaluator_agent.py backend/tests/test_autonomous_loop_integration.py backend/tests/test_bq_writer.py backend/tests/test_observability.py backend/tests/test_sentiment_ladder.py backend/tests/test_calendar_watcher.py backend/tests/test_genai_client.py -q` — zero failures (77-79 passed expected; the 1 skipped is the vaderSentiment dep absence).

**Non-goals:**
- NOT migrating `orchestrator.py` or `llm_client.py` (phase-11.3).
- NOT migrating `risk_debate.py` (phase-11.3).
- NOT fixing `Field(default_factory=list)` / issue #699 (phase-11.3 or later; not blocking 11.2).
- NOT removing the `vertexai` top-level dep (phase-11.4).

## Plan

1. Read evaluator_agent.py + skill_optimizer.py + debate.py fully; lock the exact callsites before editing.
2. evaluator_agent.py first: swap init + _call_model path; update test's monkeypatch.
3. skill_optimizer.py second: _get_model + any callers.
4. debate.py third: delete dead import.
5. Run verification commands + regression.
6. Capture output into experiment_results.md.

## References

- `handoff/current/phase-11.2-research-brief.md`
- `backend/agents/_genai_client.py` (shim, phase-11.1)
- `docs/VERTEX_AI_GENAI_MIGRATION.md` ("Per-step breakdown — 11.2")
- External read-in-full: googleapis.github.io/python-genai docs, python-genai GitHub issues #60 + #699, DeepWiki Pydantic Model Integration, python-genai repo v1.73.1 tag, Migrate to the Google GenAI SDK (ai.google.dev).

## Researcher agent id

`afd62b08494779389`
