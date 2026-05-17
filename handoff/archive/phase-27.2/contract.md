# Sprint Contract — phase-27.3 (C2: Provider-aware lite fallback)

Generated: 2026-05-16T21:50:00+00:00
Owner: Main
Step id: 27.3
Depends on: 27.0 (done — research gate). Independent of 27.1 / 27.2.

## Research-gate summary

`handoff/current/research_brief.md` §"C2 — Multi-provider fallback design" (lines 254-325). Recommended pattern: thin callable factory `_select_lite_analyzer(model_name) → Callable`. LiteLLM Router and LangChain `init_chat_model` rejected as over-engineered for a single call site. Existing `create_llm_client()` factory at `llm_client.py:1740` is the right reuse target.

Authoritative source: pyfinagent's own `make_client()` precedent (post-27.1 fix) PLUS LiteLLM docs (consulted as reference pattern, not adopted).

## Hypothesis

Adding `_run_gemini_analysis(ticker, settings)` that mirrors `_run_claude_analysis`'s output dict shape EXACTLY (same keys, `_path: "lite"` marker), plus a `_select_lite_analyzer(model_name) → Callable` factory that dispatches based on the `gemini_model` setting prefix, will let the lite path run end-to-end on Gemini. The factory replaces the hardcoded `_run_claude_analysis` call at `_run_single_analysis:766` AND at the last-resort fallback at `:806`.

The two-LLM-call pattern (trader prompt + independent risk-judge prompt) is preserved for Gemini — the lite Risk Judge system prompt `_LITE_RISK_JUDGE_SYSTEM` is model-agnostic and reusable verbatim.

Falsifier: if Gemini's structured-output for the small JSON trader response is less reliable than Claude's, the `re.search(r'\{.*\}', text, re.DOTALL)` regex extractor we share with the Claude path may need provider-specific parsing. We'd find out at the live_check step and address as a 27.3 follow-up.

## Immutable success criteria (verbatim from `.claude/masterplan.json` step 27.3)

```bash
source .venv/bin/activate && python -c "
from backend.services.autonomous_loop import _select_lite_analyzer
g=_select_lite_analyzer('gemini-2.5-flash')
c=_select_lite_analyzer('claude-sonnet-4-6')
assert callable(g) and callable(c), 'both return callables'
assert g is not c, 'distinct implementations'
print('PASS')"
```

Live_check: `run-now` with `standard=gemini-2.5-flash` completes a cycle without any `'Both full and lite paths failed'` log lines. (Verified in 27.5, not here.)

## Plan steps

1. Add `_run_gemini_analysis(ticker, settings)` in `backend/services/autonomous_loop.py` right below `_run_claude_analysis`. Reuses `_LITE_RISK_JUDGE_SYSTEM` / `_LITE_RISK_JUDGE_TEMPLATE` / `_LITE_RISK_DEFAULT`. Output dict shape IDENTICAL to `_run_claude_analysis` (every key the cycle loop's `_persist_analysis` reads). Routes through `make_client('gemini-...', vertex_model=None, settings)` from llm_client.
2. Add `_select_lite_analyzer(model_name) → Callable` module-level factory at the top of the lite-analyzer block. Returns `_run_gemini_analysis` if `model_name.startswith("gemini-")`, else `_run_claude_analysis`.
3. Update `_run_single_analysis` at line 766 + line 806: replace hardcoded `_run_claude_analysis(ticker, settings)` with `_select_lite_analyzer(settings.gemini_model)(ticker, settings)`.
4. Remove the ValueError gate inside `_run_claude_analysis` (the model-prefix gate at lines 893-898 was the "B-9" bug — the factory now routes correctly). Keep a soft check.
5. Run the immutable verification command.
6. Q/A spawn.
7. harness_log append.
8. Flip 27.3 to done.

## Anti-patterns to avoid

- Do NOT refactor the shared yfinance-fetch + prompt-template into a base class — duplication of ~30 lines is acceptable here; abstraction tax > maintenance cost.
- Do NOT remove `_run_claude_analysis` — keep it; the factory still routes to it for Claude models.
- Do NOT introduce LiteLLM or LangChain — research brief explicitly rejected them as overkill.
- Do NOT change `_LITE_RISK_JUDGE_SYSTEM` — it's tuned for the right axis judgments (volatility/concentration/valuation) and works on both providers.

## References

- `handoff/current/research_brief.md` lines 254-325 (C2 section + code skeleton)
- `backend/services/autonomous_loop.py:764-769` (lite-mode call site)
- `backend/services/autonomous_loop.py:804-808` (last-resort fallback call site)
- `backend/services/autonomous_loop.py:854-1058` (`_run_claude_analysis` to mirror)
- `backend/agents/llm_client.py::make_client` (the routing layer Gemini lite reuses)
- `.claude/masterplan.json` phase-27 step 27.3 verification command (immutable)
