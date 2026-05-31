# Contract -- phase-51.1: SecretStr unwrap (resurrect 4 dead alpha overlays)

**Step id:** 51.1 | **Priority:** P1 (money -- resurrects 4 alpha sources) | **depends_on:** 50.5
**Date:** 2026-06-01 | **harness_required:** true | **$0 LLM** (live cycle verify is operator-gated) | no pip

## Research-gate summary (PASSED)
`handoff/current/research_brief.md` (researcher `a25bfaa8d602dd4ed`: gate_passed=true, tier moderate, 5 external sources read in full, 12 URLs, recency scan, 8 internal files). Decisive findings:
- **Hypothesis CONFIRMED.** A non-empty pydantic `SecretStr` is **truthy**, so `getattr(settings,"anthropic_api_key","") or ""` returns the WRAPPER, not the plaintext. `ClaudeClient.__init__` stores it verbatim (`llm_client.py:1222`); `_get_client` (`:1238`) passes it to `Anthropic(api_key=...)`, which injects it into the `X-Api-Key` header with NO coercion -> httpx raises `Header value must be str or bytes, not <class 'pydantic.types.SecretStr'>`. Each service swallows it and returns a fallback (empty signals / 0.85 macro haircut / identity meta-score).
- **4 buggy sites (verified line numbers):** news_screen.py:258 (ctor :264), macro_regime.py:427 (:432), pead_signal.py:248 (:278), meta_scorer.py:166 (:184). All: `anthropic_key = getattr(settings,"anthropic_api_key","") or ""` then `ClaudeClient(api_key=anthropic_key)`.
- **Regression PINNED:** commit `d3f34caf` "phase-25.B10: SecretStr migration" (2026-05-13) flipped `anthropic_api_key: str -> SecretStr` (settings.py:104) without updating these phase-23.1.x sites. **Dead since 2026-05-13** (macro_regime.json computed_at 2026-04-24 is the last-good cache, not the break date).
- **Already-guarded (do NOT touch):** call_transcript_gpr.py:91-95 + analyst_narrative_scorer.py:111-115 unwrap per-site (added phase-28.11/13 -- later authors hit the bug). `make_client._unwrap` (llm_client.py:1893-1896) is why the US pure-quant pipeline is UNAFFECTED (it unwraps before constructing ClaudeClient).
- **Sibling raw-store sites (defense-in-depth):** OpenAIClient (:1088/1090), BatchClient (:1781/1783/1790), SkillFileIdCache path (prompts.py:113 -> _get_client). Not live bugs today but same pattern -> include in the self-unwrap hardening.
- **CRITICAL pitfall (pydantic #4217):** `str(SecretStr("abc"))` returns `'**********'` WITHOUT erroring -> a `str()`-based fix would silently inject the mask as the API key (new 401). The fix MUST use `.get_secret_value()`, never `str()`.

## Hypothesis
Promoting `make_client`'s local `_unwrap` to a module-level `unwrap_secret(v) -> str` and (a) self-unwrapping in `ClaudeClient.__init__` (the root-cause fix; no-op for the plain str make_client passes -> no double-unwrap, no regression to the working US path) + (b) using `unwrap_secret` at the 4 overlay sites (replacing the `or ""` truthiness footgun) makes the 4 LLM alpha overlays send a plain-str api_key -> the SDK header error is gone -> the overlays produce real signals again. The US pure-quant engine (which uses none of these overlays) is unchanged.

## Success criteria (IMMUTABLE -- verbatim from masterplan step 51.1)
1. the 4 overlay services (news_screen, macro_regime, pead_signal, meta_scorer) pass a plain str (NOT a pydantic SecretStr) as the Anthropic api_key, so the SDK no longer raises 'Header value must be str or bytes, not SecretStr'
2. ClaudeClient.__init__ self-unwraps a SecretStr api_key (defense-in-depth; a no-op for the plain str make_client already passes, so no existing caller double-unwraps or breaks)
3. a $0 unit test proves a SecretStr key reaching the ClaudeClient boundary is stored/used as a str (and the working US pure-quant path, which uses none of these overlays, is unchanged)
4. live_check_51.1.md records the $0 proof; plus, IF operator approves LLM spend, a live cycle log showing 'News screen produced N>0 ticker signals' or 'meta_scorer scored' replacing the SecretStr warning

**Verification command:** `pytest backend/tests/test_phase_51_1_secretstr.py` + `ast.parse(llm_client.py)` + `test -f live_check_51.1.md`.
**live_check:** REQUIRED -- the $0 unit/integration proof (SDK boundary receives a str); a live cycle-log signal only if LLM-spend approved.

## Plan steps (GENERATE)
1. **llm_client.py:** promote `_unwrap` to a module-level `unwrap_secret(v) -> str` (use `.get_secret_value()` when present; else `str(v)` ONLY for non-SecretStr; empty for None). Keep make_client using it.
2. **llm_client.py ClaudeClient.__init__ (:1222):** `self._api_key = unwrap_secret(api_key)` (root-cause self-unwrap). Apply the same to OpenAIClient + BatchClient ctors (defense-in-depth; no-op for the str they already get).
3. **The 4 overlay sites:** replace `anthropic_key = getattr(settings,"anthropic_api_key","") or ""` with `unwrap_secret(getattr(settings,"anthropic_api_key",""))` at news_screen.py:258, macro_regime.py:427, pead_signal.py:248, meta_scorer.py:166. (Belt-and-suspenders with step 2; also fixes the `if not key` guard to test the real string.)
4. **backend/tests/test_phase_51_1_secretstr.py (NEW):** ClaudeClient(api_key=SecretStr("sk-ant-test")) -> `_api_key == "sk-ant-test"` AND `isinstance(str)` AND `not hasattr(get_secret_value)`; plain-str case -> no double-unwrap (still "sk-ant-test"); `unwrap_secret(SecretStr(...))`/`(str)`/`(None)` unit cases; assert `str()` is NOT used (mask-injection guard) by checking a SecretStr does not become '**********'.
5. **Verify:** pytest (new + a regression sweep of the overlay-adjacent tests); ast.parse(llm_client.py). Capture a $0 proof into live_check_51.1.md (construct the boundary with a SecretStr, show the stored key is a real str). FLAG: a live cycle proof needs Peder's LLM-spend approval -- do NOT run a paid cycle in GENERATE.
6. **EVALUATE:** fresh qa. Then harness_log.md (LAST), then flip masterplan 51.1 -> done.

## Safety / scope notes
- **US pure-quant path byte-identical:** the overlays are additive (default-OFF ship; the live US screener uses none of them). `make_client` already unwraps, so its callers are unchanged; ClaudeClient self-unwrap is a no-op for a plain str (proven by the no-double-unwrap test).
- **Never `str()` a SecretStr** (mask-injection footgun) -- use `.get_secret_value()`.
- $0 LLM in GENERATE; no pip; no spend; no DROP/DELETE. Live cycle verification is operator-LLM-spend-gated (flagged, not run here).
- After 51.1: the operator-sequenced EU+KR go-live flip, then 51.2 (sector diversification), 51.3, 51.4.

## References
- handoff/current/research_brief.md (51.1 gate); regression commit d3f34caf (settings.py:104)
- backend/agents/llm_client.py:1222,1238,1893-1896 (_unwrap/ClaudeClient), :1088-1090 (OpenAIClient), :1781-1790 (BatchClient)
- backend/services/news_screen.py:258,264; macro_regime.py:427,432; pead_signal.py:248,278; meta_scorer.py:166,184
- backend/services/call_transcript_gpr.py:91-95 + analyst_narrative_scorer.py:111-115 (the correct guarded pattern to mirror)
- pydantic #4217 (str() mask footgun); anthropic-sdk _client.py (X-Api-Key no coercion)
