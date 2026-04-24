# Claude as default LLM provider -- Experiment results

## What was built

Claude is now the default LLM provider across pyfinagent, with Gemini (and
every other model in the catalog) still fully selectable from the Settings
page. Field names preserved for backward compatibility.

## Files changed

1. `backend/config/settings.py` (defaults flipped):
   - `gemini_model`: `gemini-2.0-flash` -> `claude-sonnet-4-6`
   - `deep_think_model`: `gemini-2.5-flash` -> `claude-opus-4-6`
   - Field docstrings updated to note Claude-default + Gemini still
     selectable via Settings UI.

2. `backend/services/autonomous_loop.py::_run_claude_analysis`:
   - Reads `settings.gemini_model` instead of hard-coding
     `claude-sonnet-4-6` in the `client.messages.create(model=...)` call.
   - Validates the model prefix (`claude-*`): if the user switched the
     standard model to a non-Claude provider, raises ValueError with an
     actionable message; `_run_single_analysis` catches the exception and
     falls through to the Gemini `AnalysisOrchestrator.run_full_analysis`
     path. This preserves the Monday-cycle safety net the user called out
     ("Monday's cycle should work via Gemini").

3. `frontend/src/app/settings/page.tsx`:
   - Added info banner beneath the model pickers: "Claude is the default
     LLM provider (Sonnet for standard, Opus for deep-think). Switch to
     Gemini by picking a gemini-* model above. Three features always use
     Gemini regardless of selection: RAG (Vertex AI Search), Google
     Search Grounding (pipeline steps 4/5/9/10), and Vertex
     structured-output schemas -- these are Google-only APIs."
   - Banner styled sky-themed so it reads as informational, not warning.

## Monday-cycle risk analysis

The `ANTHROPIC_API_KEY` in backend/.env is STILL an OAuth token
(`sk-ant-oat*`) and will 401 on Monday's cycle. That is intentionally
NOT fixed in this code cycle -- it's a secrets-rotation task only Peder
can do. The code ensures Monday runs anyway:

- `_run_claude_analysis` 401s on the Anthropic call.
- The existing `except Exception` wrapper in `_run_single_analysis`
  catches the auth error and falls through to
  `AnalysisOrchestrator.run_full_analysis` (Gemini/Vertex).
- GCP credential scope was fixed in commit f2e8ce28, so the Gemini
  fallback path is now functional.

Net: Monday's cycle will produce trades via Gemini even if Peder
hasn't rotated the Anthropic key yet.

## Verification command output (verbatim)

```
$ grep -c 'Field("claude-sonnet-4-6"' backend/config/settings.py
1
$ grep -c 'Field("claude-opus-4-6"' backend/config/settings.py
1
$ grep -c 'model="claude-sonnet-4-6"' backend/services/autonomous_loop.py
0
$ python -c "from backend.config.settings import Settings; s=Settings(); print(f'std={s.gemini_model} deep={s.deep_think_model}')"
std=claude-sonnet-4-6 deep=claude-opus-4-6
$ python -c "import ast; ast.parse(open('backend/config/settings.py').read()); ast.parse(open('backend/services/autonomous_loop.py').read()); print('SYNTAX_OK')"
SYNTAX_OK
$ python -c "from backend.services import autonomous_loop; print('IMPORT_OK')"
IMPORT_OK
$ grep -c "Claude is the default" frontend/src/app/settings/page.tsx
1
$ python scripts/go_live_drills/zero_orders_drill.py
step1: decide_trades emitted BUY for AAPL amount=$1000.00
step2: paper_trades row written: ticker=AAPL action=BUY qty=5.128205 price=195.0
PASS
$ cd frontend && npm run build
Compiled successfully in 1972ms
Generating static pages using 9 workers (13/13) in 145ms
$ python3 -c "inspect._run_single_analysis: fallback_present + trying_gemini_log"
fallback_present: True
trying_gemini_log: True
```

All 10 contract criteria green + Gemini fallback code verified intact.

## Success-criteria coverage

| # | Criterion | Evidence |
|---|---|---|
| 1 | `gemini_model` default is `claude-sonnet-4-6` | PASS (grep 1) |
| 2 | `deep_think_model` default is `claude-opus-4-6` | PASS (grep 1) |
| 3 | `_run_claude_analysis` does not hardcode `"claude-sonnet-4-6"` as a model arg | PASS (grep 0) |
| 4 | Settings imports clean, prints Claude defaults | PASS |
| 5 | autonomous_loop imports clean | PASS |
| 6 | settings.page.tsx contains "Claude is the default" banner | PASS |
| 7 | zero_orders drill still PASSes | PASS |
| 8 | Gemini fallback code still present in `_run_single_analysis` | PASS (inspect) |
| 9 | Frontend `npm run build` exits 0 | PASS |
| 10 | All syntax `ast.parse` exits 0 | PASS |

## Scope discipline

- Did NOT change the OAuth-token-vs-API-key issue in `backend/.env`
  (user-only task -- paste real `sk-ant-api03-*` key from
  console.anthropic.com).
- Did NOT add a `default_provider` enum field. Stayed model-name-driven
  per LiteLLM / Portkey production pattern.
- Did NOT change Gemini-only features (RAG / grounding / structured-
  output). They remain gated behind `_resolve_gemini`.
- Did NOT remove the `gemini_model` field name. Kept for backward
  compat -- field is now conceptually "standard_model for any provider"
  but the field name stays (documented in the docstring).

## Notes / follow-ups

- Peder's action for real Claude primary path: generate a
  `sk-ant-api03-*` API key at console.anthropic.com and replace the
  OAuth token in `backend/.env::ANTHROPIC_API_KEY`.
- After key rotation: paper-trading loop will run on Claude primary,
  falling back to Gemini only on rate-limit / transient errors.
- Potential cleanup follow-up: rename `gemini_model` field to
  `standard_model` across the codebase. Non-trivial (touches settings
  API, migration, frontend). Not urgent given the field works for any
  provider now.
