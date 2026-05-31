# experiment_results -- phase-51.1: SecretStr unwrap (resurrect 4 dead alpha overlays)

**Step:** 51.1 | **Date:** 2026-06-01 | **$0 LLM** (live cycle operator-gated, not run) | no pip | GENERATE complete

## What was built / changed

Fixes the regression (`d3f34caf`, 2026-05-13) where `anthropic_api_key` was migrated
`str -> SecretStr` without updating 4 services that construct `ClaudeClient` directly with
the raw key. A non-empty SecretStr is TRUTHY, so `getattr(...) or ""` returned the wrapper,
which the Anthropic SDK put into the `X-Api-Key` header -> `Header value must be str or
bytes, not SecretStr` -> each overlay silently fell back. Fixed at the boundary + edges.

| File | Change |
|------|--------|
| `backend/agents/llm_client.py` | **NEW module-level `unwrap_secret(v) -> str`** (promoted from make_client's local `_unwrap`; uses `.get_secret_value()`, never `str()`). `ClaudeClient.__init__` self-unwraps (root-cause fix). `OpenAIClient` + `BatchClient` ctors self-unwrap (defense-in-depth). `make_client` now calls the module helper. |
| `backend/services/news_screen.py:258` | `unwrap_secret(getattr(settings,"anthropic_api_key",""))` (was `... or ""`) |
| `backend/services/macro_regime.py:427` | same edge unwrap |
| `backend/services/pead_signal.py:248` | same edge unwrap |
| `backend/services/meta_scorer.py:166` | same edge unwrap |
| `backend/tests/test_phase_51_1_secretstr.py` | **NEW** 7 tests ($0, no network) |

**NOT touched** (already correctly guarded -- mirror pattern): `call_transcript_gpr.py:91-95`,
`analyst_narrative_scorer.py:111-115`.

## Why both fixes
- `ClaudeClient.__init__` self-unwrap = the **root cause** (covers any direct-construction caller, incl. the latent `SkillFileIdCache` path). No-op for the plain str `make_client` passes -> no double-unwrap, no US-path regression.
- Edge unwrap at the 4 sites = belt-and-suspenders + repairs the `if not anthropic_key:` guard (which previously tested a truthy wrapper).

## US byte-identity (the working pure-quant engine is untouched)
The 4 overlays are additive Signal-Stack flags; the live US screener (momentum/RSI/vol, $0 LLM) uses none of them. `make_client` ALREADY unwrapped the key before constructing ClaudeClient (proof step 4), so every existing make_client caller is unchanged. ClaudeClient self-unwrap is a verified no-op for a plain str (test_claude_client_plain_str_no_double_unwrap + proof step 6).

## Verification command output (verbatim)

### Syntax (all modified files)
```
OK  backend/agents/llm_client.py
OK  backend/services/news_screen.py
OK  backend/services/macro_regime.py
OK  backend/services/pead_signal.py
OK  backend/services/meta_scorer.py
OK  backend/tests/test_phase_51_1_secretstr.py
```

### pytest (phase-51.1 -- 7 tests)
```
$ python -m pytest backend/tests/test_phase_51_1_secretstr.py -q
.......                                                                  [100%]
7 passed in 0.23s
```

### Regression (existing llm_client tests)
```
$ python -m pytest test_anthropic_fallback.py test_claude_code_client.py \
      test_phase_31_1_fixes.py test_phase_37_3_budget_tokens.py test_phase_51_1_secretstr.py -q
32 passed, 1 xfailed in 2.43s
```

### make_client SecretStr path (regression smoke)
```
make_client ->  ClaudeClient | key is str: True | value ok: True
overlay services import OK; make_client SecretStr path OK
```

### $0 boundary proof -> handoff/current/live_check_51.1.md (full verbatim there)
Key lines: `str(SecretStr('sk-ant-x')) = '**********'` (the footgun we avoid) vs `unwrap_secret(..) = 'sk-ant-x'`; ClaudeClient stored `_api_key` `isinstance(str): True`, `has get_secret_value? False`, `is mask? False`.

## Artifact shape
- `unwrap_secret(v) -> str` (module-level, importable: `from backend.agents.llm_client import unwrap_secret`)
- `ClaudeClient(model_name, api_key, ...)._api_key` is always a plain `str` (SecretStr unwrapped, str passthrough)

## Operator decision flagged (not actioned here)
A LIVE cycle proof (non-empty news/meta/macro signals in backend.log) invokes Claude Haiku (~$0.10/day) -> needs Peder's LLM-spend approval per CLAUDE.md. The $0 type-assertion proof is the gate evidence; the live confirmation is optional + operator-gated.

## Next (per operator's 2026-05-31 sequencing)
After 51.1 lands: the EU+KR go-live flip (paper_markets -> ['US','EU','KR'] + backend restart), then 51.2 (sector diversification), 51.3 (digest guard), 51.4 (cron repairs).
