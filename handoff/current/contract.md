# Contract — phase-47.9: Priority-3 completion (Opus-4.8 max_tokens-at-xhigh floor + driver-pin finish)

**Cycle:** 10 (Priority 3 "[OPUS-4.8] /claude-api sweep; audit per-agent max_tokens at xhigh" — the two codeable remainders after 47.3 pricing + 47.8 backend sweep). **LLM spend:** $0 (static/structural edits + unit test, no live LLM call).

## Research-gate summary
`researcher` `aea7fbf69095873c1` (gate **PASSED**): 6 sources read in full, 7 snippet-only, 13 URLs, recency scan present, 8 internal files. Brief: `handoff/current/research_brief_phase_47_9_opus48_finish.md`.

Key findings:
- **Governing fact (resolved a doc contradiction):** on the Opus-4.8 **adaptive** thinking path this project uses, `max_tokens` is a HARD ceiling on **thinking + visible text combined** (adaptive-thinking doc verbatim: "Use `max_tokens` as a hard limit on total output (thinking + response text)"). The older extended-thinking page's `budget_tokens < max_tokens` framing is the manual-mode lens and does NOT govern 4.8. At high/xhigh/max effort Claude "may exhaust the `max_tokens` budget"; Anthropic's floor guidance is "set a large `max_tokens`... starting at 64k" for 4.8 at xhigh/max.
- **Code reality (REAL latent bug):** every Layer-2 Opus-4.8 agent runs at `effort=max` (`model_tiers.py:221-225`) but the orchestrator's adaptive call sets `max_tokens = agent_config.max_tokens + 2048` (`multi_agent_orchestrator.py:1075`), i.e. ~2548–5048 for the 500–3000 configured agents. Adaptive thinking can exhaust that and starve the visible answer. The plain-**text** `stop_reason=="max_tokens"` path is **swallowed** (retry at `:1177` fires only on a `tool_use` tail; the text tail at `:1198-1202` logs a warning + returns partial).
- **Three** stale `claude-opus-4-6` driver pins (researcher corrected my `--include=*.py`-only grep): `scripts/harness/run_autonomous_loop.py:73` (overrides the now-correct 4-8 default at `backend/autonomous_loop.py:74`), `scripts/mas_harness/run_cycle.sh:63` (`claude -p --model claude-opus-4-6` CLI flag). PlannerAgent runs on whatever the driver passes.
- **PlannerAgent safety:** `planner_agent.py:146-153` + `:253-260` call the raw client with ONLY model+max_tokens+system+messages — no temperature/thinking/budget_tokens — so bumping to 4-8 will NOT 400. But `response.content[0].text` (`:156`, `:262`) is fragile if 4.8 ever returns a thinking block first.

## Hypothesis
Flooring the adaptive-branch `max_tokens` to a thinking-safe value (root-cause fix), bumping the three stale 4-6 driver pins to 4-8, and hardening the PlannerAgent text extraction (so the pin bump can't trip a content[0] failure) will eliminate the silent-output-starvation risk on the Opus-4.8 MAS path and finish the Priority-3 sweep — all at $0 (max_tokens is a ceiling, not a target; flat-fee Max).

## Immutable success criteria (verbatim from .claude/masterplan.json phase-47.9)
1. Opus-4.8/4.7 adaptive branch in multi_agent_orchestrator.py floors max_tokens to a thinking-safe value via a pure, unit-tested helper (low configured budgets raised to the floor; large configured budgets respected); the non-adaptive ELSE branch (manual budget_tokens=2048) is unchanged; the tool_use retry cap stays above the new floor
2. the three stale claude-opus-4-6 driver pins are bumped to claude-opus-4-8 (run_autonomous_loop.py planner_model, run_cycle.sh --model); no remaining operative claude-opus-4-6 default in scripts/
3. PlannerAgent response parsing is hardened to extract the first text block robustly (tolerant of a leading thinking block), so running the planner on Opus 4.8 cannot raise on content[0]
4. a pytest guard asserts the floor helper (floors low / respects high), the orchestrator branch uses it, the 3 pins are 4-8, and the planner text-extraction is thinking-block tolerant; ast.parse clean on all edited files; pytest green

## Plan steps
1. `multi_agent_orchestrator.py`: add `_OPUS_ADAPTIVE_MIN_MAX_TOKENS = 16384` + pure helper `_adaptive_max_tokens(configured)` = `max(configured + 2048, FLOOR)`. In the IF (Opus-4.8/4.7 adaptive) branch set `_max_tokens = _adaptive_max_tokens(agent_config.max_tokens)`; in the ELSE branch set `_max_tokens = agent_config.max_tokens + 2048`. Use `_max_tokens` at the create call (:1075) and the retry (`_retry_max = min(_max_tokens * 2, 32768)`), so the retry stays above the floor.
2. `scripts/harness/run_autonomous_loop.py:73` + `scripts/mas_harness/run_cycle.sh:63`: `claude-opus-4-6` → `claude-opus-4-8`.
3. `planner_agent.py`: add `_first_text(response)` helper (`"".join(b.text for b in response.content if getattr(b,"type",None)=="text")` with a content[0] fallback); replace the two `response.content[0].text` sites.
4. `tests/agents/test_phase_47_9_max_tokens_floor.py`: `_adaptive_max_tokens(500)==16384`, `(4096)==16384`, `(30000)==32048`; orchestrator branch references the helper (source); 3 pins are 4-8 (incl. run_cycle.sh); `_first_text` returns the text when a fake thinking block precedes it.
5. Verify: `ast.parse` edited .py files + `python -m pytest tests/agents/test_phase_47_9_max_tokens_floor.py -q`.

## Out-of-scope (FLAGGED follow-ups, NOT fixed — disclosed to Q/A)
- `llm_client.generate_content` (:1285) max_tokens=2048 default + its own effort/thinking injection — separate Layer-1 path (primarily Gemini); floor there is a follow-up.
- COMMUNICATION router at `effort=max` + `max_tokens=500` (worst case) — collides with owner directive on that agent; effort-lowering is an operator call.
- Making the silent **text** `stop_reason=="max_tokens"` path retry (not just tool_use) — a behavior/cost change (risk of double-billing legitimate long outputs); deferred. The floor makes text truncation rare regardless.

## References
- `handoff/current/research_brief_phase_47_9_opus48_finish.md`
- Anthropic adaptive/extended-thinking docs + Opus 4.8 effort doc (max_tokens = thinking+text ceiling; 64k floor at xhigh/max; 128k hard cap)
- phase-47.8 (the backend sweep this completes) + `backend/autonomous_loop.py:74` (the good 4-8 default the scripts override)
