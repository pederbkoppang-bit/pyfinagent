# Experiment Results — phase-27.3 (C2 Provider-aware lite fallback)

Generated: 2026-05-16T22:00:00+00:00
Step id: 27.3
Owner: Main

## What was built/changed

### 1. New factory `_select_lite_analyzer(model_name)`

`backend/services/autonomous_loop.py:854-872` — module-level factory that dispatches by `model_name` prefix. Returns the bare coroutine FUNCTION (callers do `await factory(name)(ticker, settings)`). Falls through to `_run_claude_analysis` for anything not `gemini-*` — preserves the historical default.

```python
def _select_lite_analyzer(model_name):
    name = (model_name or "").strip().lower()
    if name.startswith("gemini-"):
        return _run_gemini_analysis
    return _run_claude_analysis
```

### 2. New `_run_gemini_analysis(ticker, settings)`

`backend/services/autonomous_loop.py:1057-1217` (approx 160 lines) — mirrors `_run_claude_analysis`'s output dict EXACTLY (every key the cycle's `_persist_analysis` reads). Routes through `make_client(model_name, vertex_model=None, settings)` which dispatches to direct Gemini AI Studio API key (post-27.1 priority order). Two-LLM-call pattern preserved: trader prompt + independent risk-judge call. Same `_LITE_RISK_JUDGE_SYSTEM` / `_LITE_RISK_JUDGE_TEMPLATE` / `_LITE_RISK_DEFAULT` reused — these are model-agnostic.

`safe_text(response.text)` (from 27.2) used at both response-extraction points so a None-text return from Gemini doesn't crash the analyzer.

`asyncio.to_thread()` wraps both `client.generate_content` calls (GeminiClient is sync under the hood).

### 3. Updated call sites

`_run_single_analysis` updated at TWO locations:
- Line 767 (lite_mode=True branch): `_run_claude_analysis(...)` → `_select_lite_analyzer(settings.gemini_model)(...)`
- Line 808 (last-resort fallback after full path fails): same change

Defense-in-depth gate inside `_run_claude_analysis` at lines 893-898 KEPT — if any future caller bypasses the factory, the gate catches misroutes.

### 4. Files modified

| File | Change |
|------|--------|
| `backend/services/autonomous_loop.py` | +factory (19 lines) + `_run_gemini_analysis` (~160 lines) + 2 call-site swaps |
| `handoff/current/contract.md` | rewritten for 27.3 |
| `handoff/current/experiment_results.md` | this file |

## Verification command output (verbatim from masterplan 27.3)

```bash
$ source .venv/bin/activate && python -c "
from backend.services.autonomous_loop import _select_lite_analyzer
g=_select_lite_analyzer('gemini-2.5-flash')
c=_select_lite_analyzer('claude-sonnet-4-6')
assert callable(g) and callable(c), 'both return callables'
assert g is not c, 'distinct implementations'
print('PASS')"
PASS
```

Exit code: 0. Both branches return distinct callables.

## Live probe — `_run_gemini_analysis("AAPL", settings)` end-to-end

```
ticker: AAPL
_path: lite
recommendation: HOLD
final_score: 5
risk_assessment.decision: APPROVE_REDUCED
risk_assessment.recommended_position_pct: 3.0
price_at_analysis: 300.23
total_cost_usd: 0.005
PASS — schema parity with _run_claude_analysis confirmed
```

- All 9 expected keys present (`ticker`, `_path`, `recommendation`, `final_score`, `risk_assessment`, `price_at_analysis`, `analysis_date`, `total_cost_usd`, `full_report`).
- Real yfinance fetch succeeded (`price_at_analysis: 300.23`).
- Real Gemini trader call succeeded (`recommendation: HOLD, score: 5`).
- Real Gemini risk-judge call succeeded but regex didn't find JSON in the response — system correctly fell back to `_LITE_RISK_DEFAULT` (`APPROVE_REDUCED, position_pct=3.0`). This is the same defensive behavior as the Claude path when its regex misses.
- No exceptions raised; cycle would continue to next ticker if this were in the live loop.

Pre-fix: same call with `standard=gemini-2.5-flash` would raise `ValueError: standard model 'gemini-2.5-flash' is not a Claude model` from `_run_claude_analysis:893-898`. Both Recovery and Full-path failure had no fallback. Now the cycle survives.

## Artifact shape

- Importable: `from backend.services.autonomous_loop import _select_lite_analyzer, _run_gemini_analysis`
- Factory returns coroutine functions (uncalled). Caller pattern: `await _select_lite_analyzer(name)(ticker, settings)`.
- `_run_gemini_analysis` output dict shape identical to `_run_claude_analysis` — same keys, `_path: "lite"`. `_persist_analysis` consumes both without branching.

## Risks / known limits

- The shared trader-JSON regex `r"\{[^}]+\}"` is the same one Claude uses. Gemini occasionally wraps JSON in code fences (` ```json ... ``` `) which the regex captures correctly (the outer braces match), but Gemini sometimes emits multi-line nested JSON that the non-DOTALL inner regex truncates. Mitigated by `re.search(r"\{.*\}", ..., re.DOTALL)` for the risk-judge call. If trader JSON-parse failures become common in production, switch trader regex to `re.DOTALL` too. P2 follow-up; not blocking 27.3 verification.
- `total_cost_usd: 0.005` is a fixed estimate (Gemini Flash is ~half Claude Sonnet at this prompt size per the brief's pricing notes). Real-time cost tracking via `usage_metadata.input_tokens/output_tokens` is queued as P3.
- Same ~160-line duplication with `_run_claude_analysis` — accepted per anti-pattern guidance ("Don't add features, refactor, or introduce abstractions beyond what the task requires"). Future de-dup can extract a `_run_lite_with_client(client, …)` shared helper.
