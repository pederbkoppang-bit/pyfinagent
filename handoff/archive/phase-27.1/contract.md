# Sprint Contract — phase-27.2 (C1: Gemini null-text safety)

Generated: 2026-05-16T21:35:00+00:00
Owner: Main
Step id: 27.2
Depends on: 27.0 (done — research gate); independent of 27.1.

## Research-gate summary

`handoff/current/research_brief.md` §"C1 — Gemini null-text safety" (lines 165-253). Authoritative: `https://ai.google.dev/api/generate-content` + `python-genai#1039` (active bug: `response.text` returns None when `finish_reason=MAX_TOKENS` with structured output) + Google AI dev forum `t/81874` (None vs ValueError dual-mode confirmed by Google staff). Known unfixed upstream; defensive None-check is the only mitigation.

## Hypothesis

Two-layer defense closes the bug class:

1. **At the boundary:** add a `safe_text(text)` module-level helper returning `"" if text is None else text`. Add a `__post_init__` on `LLMResponse` that coerces `text=None → ""`. This makes the LLMResponse contract honest (it claims `text: str`, now enforces it).
2. **At the Gemini accessor:** add `if text is None: text = ""` after the existing try/except on `response.text` in `GeminiClient.generate_content`. Even though `__post_init__` would catch it downstream, an explicit guard at the Gemini boundary keeps the failure mode visible.

After both layers, every existing `response.text.strip()` call site (in `debate.py`, `risk_debate.py`, etc.) becomes safe by construction — no need to refactor 7 caller sites.

Falsifier: if any caller depends on `text is None` semantics (e.g., distinguishing "empty response" from "no response"), the coercion would silently change behavior. Read those callers; if they have such a branch, switch them to check `usage_metadata.candidates_token_count == 0` instead.

## Immutable success criteria (verbatim from `.claude/masterplan.json` step 27.2)

```bash
source .venv/bin/activate && python -c "
from backend.agents.llm_client import LLMResponse
r=LLMResponse(text=None, input_tokens=0, output_tokens=0)
assert r.text == '' or r.text is None, 'text should be empty or None'
from backend.agents.llm_client import safe_text
assert safe_text(None) == '', 'safe_text(None)'
assert safe_text('  hi  ').strip() == 'hi', 'safe_text str'
print('PASS')" && \
! grep -rE '\.text\.strip\(\)' backend/agents/orchestrator.py | grep -v 'safe_text' | head -1
```

Note: the verification command uses kwargs `input_tokens=0, output_tokens=0` which don't currently exist on `LLMResponse` (only `usage_metadata`). Either I extend `LLMResponse` with these as convenience fields (default 0) AND make `__post_init__` coerce text=None → "" — which satisfies both legs — OR the assertion `r.text == '' or r.text is None` accepts either coerce-to-str OR keep-as-None. Picking the coerce path because it's the right contract and keeps downstream `.strip()` calls safe without N grep-and-fix passes.

## Plan steps

1. Add `safe_text(text)` module-level helper at `backend/agents/llm_client.py` (next to `_ensure_additional_properties_false`, same architectural tier).
2. Extend `LLMResponse` dataclass: add `input_tokens: int = 0` and `output_tokens: int = 0` convenience fields. Add `__post_init__` that runs `self.text = safe_text(self.text)`.
3. In `GeminiClient.generate_content` text-extraction block (line ~918-927), add explicit `if text is None: text = ""` after the try/except to keep the failure mode visible at the Gemini boundary.
4. Run the immutable verification command.
5. Smoke a real Gemini call that would have crashed pre-fix (use a `max_output_tokens=1` to provoke MAX_TOKENS truncation).
6. Q/A spawn for independent verification.
7. harness_log append.
8. Flip 27.2 to done.

## Anti-patterns to avoid

- Do NOT refactor every `.text.strip()` call site individually — that's brittle and the `__post_init__` enforcement is the right point.
- Do NOT change Anthropic or OpenAI client text extraction — Anthropic returns `content[0].text` which is already always str. Bug is Gemini-specific.
- Do NOT alter `UsageMeta` — it stays as the canonical place for token counts; `input_tokens`/`output_tokens` on LLMResponse are convenience aliases only.

## References

- `handoff/current/research_brief.md` lines 165-253 (C1 section)
- `backend/agents/llm_client.py:609-633` (LLMResponse dataclass)
- `backend/agents/llm_client.py:918-927` (Gemini text-extraction bug locus)
- `backend/agents/debate.py:224,240`; `backend/agents/risk_debate.py:198,215,231` (downstream `.text.strip()` callers protected by post_init coercion)
- `.claude/masterplan.json` phase-27 step 27.2 verification command (immutable)
