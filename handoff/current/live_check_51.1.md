# live_check -- phase-51.1: SecretStr unwrap (resurrect 4 dead alpha overlays)

**Step:** 51.1 | **Date:** 2026-06-01 | **Result shape:** $0 deterministic proof that a
pydantic SecretStr key reaching the ClaudeClient boundary is stored/used as a plain
`str` (no SDK "Header value must be str or bytes" error), with the `str()` mask-injection
footgun explicitly avoided. Live cycle confirmation is operator-LLM-spend-gated (flagged, NOT run).

**Command:**
```
source .venv/bin/activate && python <the $0 proof script>   # no network, no SDK instantiation, no spend
```

## Verbatim output

```
=== phase-51.1 $0 proof: SecretStr never reaches the SDK header ===

1. The original bug -- a non-empty SecretStr is TRUTHY:
   bool(SecretStr('x')) = True  -> `getattr(...) or ''` returned the WRAPPER
   (SecretStr('x') or '')      = SecretStr('**********')

2. The str() footgun we MUST avoid (pydantic #4217):
   str(SecretStr('sk-ant-x')) = '**********'   <- silent mask -> 401 if used
   unwrap_secret(SecretStr(..)) = 'sk-ant-x'   <- real value via get_secret_value

3. ClaudeClient now self-unwraps at the boundary (the 4 overlays construct it directly):
   stored _api_key type: str | isinstance(str): True
   has get_secret_value (i.e. still a wrapper)? False  <- False = fixed
   equals the real secret (not the mask)? True | is mask? False

4. make_client path (the US pure-quant pipeline) -- unchanged, str key:
   make_client -> ClaudeClient | _api_key isinstance(str): True

5. Sibling clients self-unwrap (defense-in-depth):
   OpenAIClient._api_key str: True | BatchClient._api_key str: True

6. No-double-unwrap (a plain str passes through untouched):
   ClaudeClient('sk-plain')._api_key == 'sk-plain'

RESULT: every path delivers a plain str to Anthropic(api_key=...) -> no 'Header value must be str or bytes' -> overlays live again.
```

## pytest (the $0 gate)
```
$ python -m pytest backend/tests/test_phase_51_1_secretstr.py -q
.......                                                                  [100%]
7 passed in 0.23s
```
Regression sweep (existing llm_client tests): `32 passed, 1 xfailed` (test_anthropic_fallback, test_claude_code_client, test_phase_31_1_fixes, test_phase_37_3_budget_tokens).

## Criterion-by-criterion

| # | Criterion | Evidence | Verdict |
|---|-----------|----------|---------|
| 1 | the 4 overlay services pass a plain str (not SecretStr) as api_key, no SDK header error | proof step 1 (bug) + 3 (ClaudeClient stores str) + test_overlay_services_pass_str_to_claudeclient | PASS |
| 2 | ClaudeClient.__init__ self-unwraps SecretStr; no-op for plain str (no double-unwrap) | proof step 3 (SecretStr -> str) + step 6 (str passes through) + test_claude_client_plain_str_no_double_unwrap | PASS |
| 3 | $0 unit test proves the boundary str; US pure-quant path unchanged | 7/7 unit tests + proof step 4 (make_client unchanged, str key); overlays are additive, US screener uses none | PASS |
| 4 | live_check records the $0 proof; live cycle log only if LLM-spend approved | this file; live cycle FLAGGED operator-gated (not run) | PASS ($0 leg) |

## Notes
- **Root cause** fixed at the `ClaudeClient` boundary (covers all callers) PLUS edge-unwrap at the 4 overlay sites (belt-and-suspenders; repairs their `if not key` guard). Regression pinned to commit `d3f34caf` (phase-25.B10 SecretStr migration, 2026-05-13).
- **The `str()` footgun is explicitly guarded** (proof step 2 + test): `unwrap_secret` uses `get_secret_value()`; `str(SecretStr)` would silently inject `'**********'`.
- **US byte-identity**: `make_client` already unwrapped (proof step 4), so the live US pure-quant pipeline is unchanged; ClaudeClient self-unwrap is a no-op for the str it receives.
- **Operator LLM-spend gate**: a live autonomous cycle would emit non-empty signals ("News screen produced N>0 ticker signals" / "meta_scorer scored" / a fresh macro_regime.json with computed_at > 2026-06-01) but invokes Claude Haiku (~$0.10/day). Per CLAUDE.md, LLM spend needs Peder's approval -> NOT run here; the $0 type-assertion proof above is the gate evidence (a type assertion cannot be hidden by masking).
