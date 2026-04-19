# Experiment Results -- phase-11.2 Migrate Trivial Callers

**Step:** 11.2 (3rd in phase-11).
**Date:** 2026-04-19.

## What was built

Migrated 3 of the 5 live Vertex import sites (remaining 2 are `risk_debate.py` + `orchestrator.py`, reserved for phase-11.3). Zero test additions — the existing test_evaluator_agent.py was rebadged. Zero behavior changes other than swapping the underlying SDK.

**`backend/agents/evaluator_agent.py`:**
- Removed the `try: from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration / except ImportError` block + `VERTEX_AVAILABLE` flag.
- Added `from backend.agents._genai_client import get_genai_client` + `GENAI_AVAILABLE = True`.
- `__init__`: `self.model = get_genai_client()`; None-return from the shim triggers the mock-evaluator path (preserves the same graceful-degrade contract).
- `_call_model`: `self.model.generate_content(prompt)` → `self.model.models.generate_content(model=self.model_name, contents=prompt)`.

**`backend/agents/skill_optimizer.py`:**
- `_get_model()` now returns a `(client, model_name)` tuple because the new SDK is client-per-call (`client.models.generate_content(model=name, ...)`), not model-per-instance like the legacy `GenerativeModel(name)`.
- Two callers (`propose_skill_modification` at :358 and `think_harder` at :528) updated to unpack the tuple, guard `if client is None: return None`, and build `types.GenerateContentConfig(temperature=..., max_output_tokens=...)` instead of the dict `generation_config=`.

**`backend/agents/debate.py`:**
- Removed the DEAD `from vertexai.generative_models import GenerativeModel` at line 18 (researcher confirmed it was never instantiated inside the module; all debate calls route through `LLMClient`). Replaced with a comment explaining the removal.

**`backend/tests/test_evaluator_agent.py`:**
- Renamed every `VERTEX_AVAILABLE` occurrence to `GENAI_AVAILABLE` (one monkeypatch + one docstring reference).

## File list

Modified:
- `backend/agents/evaluator_agent.py` (~15 lines net)
- `backend/agents/skill_optimizer.py` (~30 lines net; tuple unpack + 2 caller updates)
- `backend/agents/debate.py` (-1 line import, +6 lines comment)
- `backend/tests/test_evaluator_agent.py` (trivial rename)

Created: none. Non-goal preserved: no new tests (existing 6 tests still cover the evaluator behavior).

## Verification command output

### Immutable (from contract)

```
$ source .venv/bin/activate && pytest backend/tests/test_evaluator_agent.py -q && python -W error::DeprecationWarning -c "from backend.agents import evaluator_agent"
......                                                                   [100%]
6 passed, 1 warning in 0.57s
```

Exit 0 on both legs. The `-W error::DeprecationWarning` import succeeded — **the Vertex `vertexai.generative_models` DeprecationWarning is gone from the evaluator_agent load path**. (The one remaining warning in pytest is from `google/genai/types.py:42` — Python 3.17 pre-deprecation noise, not our path.)

### Syntax + imports

```
$ python -c "import ast; ast.parse(open('backend/agents/evaluator_agent.py').read()); ast.parse(open('backend/agents/skill_optimizer.py').read()); ast.parse(open('backend/agents/debate.py').read()); print('SYNTAX OK')"
SYNTAX OK

$ python -c "from backend.agents.evaluator_agent import EvaluatorAgent, EvaluationVerdict, GENAI_AVAILABLE; from backend.agents.skill_optimizer import SkillOptimizer, OPTIMIZABLE_AGENTS; import backend.agents.debate; print('imports ok')"
imports ok
```

### Inventory

```
$ grep -rn "from vertexai.generative_models\|from vertexai import generative_models" backend/ --include="*.py" | grep -v __pycache__
backend/agents/debate.py:18:# phase-11.2: removed dead `from vertexai.generative_models import GenerativeModel`.
backend/agents/risk_debate.py:23:from vertexai.generative_models import GenerativeModel
backend/agents/orchestrator.py:29:from vertexai.generative_models import GenerativeModel, Tool, grounding, GenerationConfig
```

3 matches. Breakdown:
- `debate.py:18` — a **comment** referencing the removed import (not a live import; grep matched my deletion-note text).
- `risk_debate.py:23` — live import, **phase-11.3** scope.
- `orchestrator.py:29` — live import, **phase-11.3** scope.

Live imports remaining: **2** (risk_debate + orchestrator). This cycle removed 3 live imports (evaluator_agent + skill_optimizer + debate). Contract's expected number of `5` was a miscount from a broader phase-11.0 grep that included `GenerativeModel(...)` constructor calls + `vertexai.init` calls; the narrower "import line only" grep returns 2 live + 1 comment = 3. Pre-Q/A self-check catches this discrepancy so Q/A can confirm the math without flagging it as a contract violation.

### Regression

```
$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
79 passed, 1 skipped, 1 warning in 5.68s
```

Same 79p/1s as phase-11.1. Zero regressions.

## Contract criterion check

| # | Criterion | Status |
|---|-----------|--------|
| 1 | evaluator_agent.py migrated to shim, VERTEX_AVAILABLE→GENAI_AVAILABLE | PASS |
| 2 | skill_optimizer._get_model returns (client, model_name), 2 callers updated | PASS |
| 3 | debate.py dead import removed | PASS |
| 4 | test_evaluator_agent.py rename | PASS (6 tests still pass) |
| 5 | `python -W error::DeprecationWarning -c "from backend.agents import evaluator_agent"` exit 0 (no vertexai warning on load) | PASS |
| 6 | Inventory grep matches expected count | PASS-with-clarification (contract said 5 but narrow grep returns 3; see "Inventory" above — 2 live imports remain for phase-11.3, exactly as planned) |

## Known caveats

1. **Contract inventory-count was off-by-two.** Contract predicted `grep | wc -l` would show 5 after this cycle; actual shows 3. Root cause: phase-11.0 inventory grep matched both imports AND `GenerativeModel(` constructor calls AND `vertexai.init` calls; phase-11.2 contract narrowed the grep to imports only but carried over the broader 8-baseline math. Pre-Q/A self-check caught this; flagging for Q/A so the critique notes the arithmetic error in the contract, not a migration miss.
2. **`skill_optimizer._get_model` contract change**: return type went from `GenerativeModel` to `tuple[Client, str]`. Internal-only — both callers (`:358`, `:528`) updated. Any external caller would break; the grep-based audit shows no external callers.
3. **Live genai calls still not exercised against real Gemini** — same caveat as phase-11.1. The migration correctness is verified by (a) test_evaluator_agent still passing (which exercises `_mock_response` because the client returns None in test env), (b) no DeprecationWarning on evaluator_agent import, (c) successful syntactic + import smoke.
4. **Pre-Q/A self-check ran the migration target imports** (`from backend.agents.evaluator_agent import ...`) before declaring done — caught no bugs this cycle.
5. **debate.py comment takes up a grep match** — if Q/A runs the inventory grep literally, they'll see 3 lines. That's correct; 2 are live imports, 1 is my comment. Decided to leave the comment because phase-11.3 will need to know that debate.py WAS a `vertexai.generative_models` caller at some point in the archaeology. Alternative: convert the comment to a non-matching form (e.g., escape the dotted path). Not doing that; the dead-import removal is a notable event in the file history.
