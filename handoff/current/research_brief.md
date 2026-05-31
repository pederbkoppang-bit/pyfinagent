# research_brief -- phase-51.1: SecretStr unwrap (resurrect dead alpha overlays)

**Tier:** moderate (assumed -- caller did not override; internal-audit-heavy with a focused external secret-handling question)
**Date:** 2026-06-01
**Step:** Fix the SecretStr regression that silently killed 4 LLM alpha overlays (worldwide-news screen, macro-regime/sector-event, PEAD, LLM-as-judge meta-scorer) since the 2026-05-13 SecretStr migration. Resurrects 4 alpha sources. The working US pure-quant path must NOT regress.

## Hypothesis under test (REVALIDATED below -- CONFIRMED)
4 LLM alpha overlay services construct `ClaudeClient` directly with a raw
pydantic `SecretStr` (bypassing `make_client()`'s `_unwrap`), causing the
Anthropic SDK to raise `Header value must be str or bytes, not
<class 'pydantic.types.SecretStr'>`. Each service catches the error and
returns a fallback -> overlays enabled-in-UI but produce ZERO effect.

---

## PART A -- INTERNAL CODE AUDIT (VERIFIED against current code)

### A1. The mechanism (CONFIRMED)
- `backend/config/settings.py:104` -- `anthropic_api_key: SecretStr = Field(SecretStr(""), ...)`. TYPE IS SecretStr. CONFIRMED.
- `backend/agents/llm_client.py:1206` -- `class ClaudeClient(LLMClient)`.
- `backend/agents/llm_client.py:1220-1223` -- `__init__(self, model_name, api_key, enable_prompt_caching=True)` stores `self._api_key = api_key` **VERBATIM** (line 1222). No unwrap. CONFIRMED.
- `backend/agents/llm_client.py:1238` -- `_get_client()` returns `_anthropic_sdk.Anthropic(api_key=self._api_key, max_retries=3)`. Passes the stored key straight to the SDK. If it's a SecretStr -> SDK raises `Header value must be str or bytes`. CONFIRMED path.
- `backend/agents/llm_client.py:1893-1896` -- `make_client()`'s `_unwrap(v)`: returns `""` for None/empty, else `v.get_secret_value() if hasattr(v,"get_secret_value") else str(v)`. The CORRECT boundary unwrap. Applied at lines 1898-1901 to anthropic/openai/gemini/github keys. So the factory path (`make_client -> ClaudeClient` at line 1963) passes a PLAIN STR and works fine. (This is why the main pure-quant pipeline -- which routes Claude analysis through make_client -- is unaffected.)

### A2. The truthiness footgun (CONFIRMED)
`getattr(settings, "anthropic_api_key", "") or ""` does NOT unwrap. A non-empty `SecretStr` is **truthy** (its `__bool__`/`__len__` reflect the wrapped string's length), so `X or ""` short-circuits and returns the SecretStr object **itself**, unwrapped. The `or ""` only helps if the key were falsy (empty). The subsequent `if not anthropic_key:` guard ALSO passes (truthy), so each service proceeds to construct ClaudeClient with the wrapped object. CONFIRMED -- this is the exact bug.

### A3. The 4 BUGGY construction sites (caller-listed, all CONFIRMED current line numbers)
| # | Service | File:line (key read) | File:line (ClaudeClient ctor) | Pattern |
|---|---|---|---|---|
| 1 | worldwide-news screen | `backend/services/news_screen.py:258` | `:264` | `getattr(...,"") or ""` -> passes SecretStr |
| 2 | macro-regime / sector-event | `backend/services/macro_regime.py:427` | `:432` | same |
| 3 | PEAD earnings overlay | `backend/services/pead_signal.py:248` | `:278` | same |
| 4 | LLM-as-judge meta-scorer | `backend/services/meta_scorer.py:166` | `:184` | same |

All 4 read `anthropic_key = getattr(settings, "anthropic_api_key", "") or ""` then `ClaudeClient(model_name=..., api_key=anthropic_key, enable_prompt_caching=False)`. NO unwrap between. CONFIRMED.

### A4. Other direct `ClaudeClient(` sites (full grep -- the fix must cover the class)
`grep -rn "ClaudeClient(" backend/` returns 8 hits:
- `llm_client.py:1206` -- the class def (not a call).
- `llm_client.py:1963` -- inside `make_client`, receives `anthropic_key` already `_unwrap`-ed. SAFE.
- `services/call_transcript_gpr.py:113` -- **ALREADY GUARDED**: lines 91-95 do `if hasattr(anthropic_key,"get_secret_value"): anthropic_key = anthropic_key.get_secret_value()`. NOT buggy. (phase-28.13, created AFTER the SecretStr migration.)
- `services/analyst_narrative_scorer.py:136` -- **ALREADY GUARDED**: lines 111-115 same unwrap guard. NOT buggy. (phase-28.11, created AFTER migration.)
- The 4 buggy sites above (news_screen:264, macro_regime:432, pead_signal:278, meta_scorer:184).

So: exactly **4 buggy sites**, 2 already-locally-guarded sites, 1 safe factory call. The two guarded sites are evidence that later authors hit this bug and worked around it per-site instead of fixing root cause -- a duplication smell that argues for the defense-in-depth fix.

**A4b. Sibling Anthropic client classes with the SAME raw-store pattern (defense-in-depth scope).** Two OTHER classes in `llm_client.py` store the key verbatim and pass it straight to the SDK, identical to `ClaudeClient`:
- `OpenAIClient.__init__` (`:1088`, `self._api_key = api_key` `:1090`) -> `_get_client` `:1090`-ish. (OpenAI SDK; same "header must be str" class of error if handed a SecretStr.)
- `BatchClient.__init__` (`:1781`, `self._api_key = api_key` `:1783`) -> `_get_client` -> `Anthropic(api_key=self._api_key)` `:1790`. **NOTE:** `BatchClient(self, model_name, api_key)` requires both args, but the only construction site `orchestrator.py:848` calls `BatchClient()` with NO args -> that path would `TypeError` BEFORE the SecretStr issue (it's a documented "25.C9.1 follow-up" wire, likely never reached today). So BatchClient is NOT a live SecretStr failure, but it shares the latent pattern.
These are NOT in scope as live bugs (neither is constructed from a settings SecretStr today -- OpenAIClient gets its key via make_client's `_unwrap` at `:1899`/`:1968`; BatchClient's call site is unreached). But IF the fix promotes a shared `unwrap_secret()` helper, applying the same `hasattr`-guarded self-unwrap to `OpenAIClient.__init__` and `BatchClient.__init__` is a cheap, consistent hardening that future-proofs the whole client family. RECOMMEND including them in the defense-in-depth pass (low cost, no behavior change for str callers).

### A5. SkillFileIdCache "SecretStr error" (investigated)
Source: `backend/config/prompts.py:36` `class SkillFileIdCache`. Its `get_or_upload` (`:113-119`) calls `claude_client_wrapper.upload_file_to_anthropic_files_api(...)`. The wrapper is a `ClaudeClient` instance; `upload_file_to_anthropic_files_api` (llm_client.py:1240) internally calls `self._get_client()` (line 1238) -> `Anthropic(api_key=self._api_key)`. So IF the ClaudeClient wrapper passed to SkillFileIdCache was constructed with a raw SecretStr, the file upload ALSO raises "Header value must be str or bytes" and is swallowed by `except Exception` at prompts.py:120-125 (logs "upload failed", returns None -> falls back to inline skill injection). SAME root cause -- ANY ClaudeClient built with an unwrapped SecretStr breaks BOTH `messages.create` AND `files.upload`. The wrapper handed to SkillFileIdCache comes from the orchestrator path which uses make_client (unwrapped today), so this is LATENT rather than currently-firing -- but the defense-in-depth `ClaudeClient.__init__` self-unwrap closes it permanently.

### A6. REGRESSION CONFIRMED -- commit + date pinned
`git log -S "anthropic_api_key: SecretStr" -- backend/config/settings.py` and `git blame -L 104,104`:
- **Commit `d3f34caf`** "phase-25.B10: SecretStr migration for API keys/tokens", **2026-05-13 10:45:11 +0200**, author Ford. Diff flips `anthropic_api_key: str = Field("")` -> `anthropic_api_key: SecretStr = Field(SecretStr(""))` (also openai_api_key, github_token). **This is the regression-introducing commit.**
- The 4 buggy services were all created EARLIER, in phase-23.1.x (against the plain-str field), so they worked when written:
  - `news_screen.py` -- `76d89aa4` phase-23.1.3
  - `meta_scorer.py` -- `35ff8f59` phase-23.1.5
  - `macro_regime.py` -- `743d65e5` phase-23.1.1
  - `pead_signal.py` -- `5a6a6e17` phase-23.1.2
  Commit d3f34caf (2026-05-13) silently broke all 4 by changing the field type WITHOUT updating these call sites -- only `make_client`'s `_unwrap` was added in/around that migration.
- **Corroborating live evidence**: `backend/services/_cache/macro_regime.json` has `"computed_at": "2026-04-24T00:00:00Z"` (file mtime Apr 27 2026). This is the last successful non-fallback macro-regime LLM computation. It PREDATES the 2026-05-13 migration -- meaning the overlay produced its last good result before the break, and every attempt since 2026-05-13 has fallen back. The caller's "~2026-04-24 boundary" is the last-good CACHE timestamp; the code-level break is 2026-05-13. **The overlays have been dead since 2026-05-13.**

### A7. The 2 already-guarded sites confirm the class-level fix is right
`git log -S "get_secret_value" -- call_transcript_gpr.py analyst_narrative_scorer.py` shows the guards arrived in phase-28.11 (`ac5a5b3c`) and phase-28.13 (`6e88f91a`) -- both AFTER the 2026-05-13 migration. Later authors independently rediscovered the bug and patched their own site. This duplication is exactly the failure mode a single root-cause fix in `ClaudeClient.__init__` prevents going forward.

### A8. RECOMMENDED FIX SHAPE -- defense-in-depth (BOTH), justified
**Recommendation: do BOTH (a) self-unwrap in `ClaudeClient.__init__` AND (b) fix the 4 sites to unwrap at the edge. (a) is the load-bearing root-cause fix; (b) is hygiene + consistency.**

- **(a) `ClaudeClient.__init__` self-unwrap (PRIMARY, root cause).** Change line 1222 from `self._api_key = api_key` to:
  ```python
  self._api_key = api_key.get_secret_value() if hasattr(api_key, "get_secret_value") else api_key
  ```
  - **Why safe for ALL existing callers (no double-unwrap risk):** the guard is `hasattr(api_key, "get_secret_value")`. A plain `str` does NOT have that attr, so for the make_client path (already passes an unwrapped str, line 1963) and the 2 already-guarded sites (also pass str) this is a **pure no-op** -- returns the str unchanged. NO double-unwrap is possible: `get_secret_value()` returns a plain str, which has no `get_secret_value`, so it cannot be called twice. CONFIRMED safe for every current caller.
  - **Why it's the right layer:** the SDK boundary is `_get_client()` (line 1238) where the key becomes an HTTP header. Unwrapping in `__init__` guarantees NO caller -- present or future -- can hand the SDK a SecretStr. Closes the 4 sites AND the latent SkillFileIdCache path AND any future direct-construction site in ONE place. This is the structural fix and the minimal change that resurrects the 4 overlays.

- **(b) Per-call `.get_secret_value()` at the 4 sites (SECONDARY, hygiene).** Replace `getattr(settings, "anthropic_api_key", "") or ""` with an unwrap, e.g.:
  ```python
  _raw = getattr(settings, "anthropic_api_key", "")
  anthropic_key = _raw.get_secret_value() if hasattr(_raw, "get_secret_value") else (_raw or "")
  ```
  - **Why ALSO do this:** (1) keeps the 4 sites correct even if someone later "simplifies" ClaudeClient.__init__; (2) makes `if not anthropic_key:` test the unwrapped string (today it tests the wrapper -- works but misleading); (3) unifies all 6 service sites to ONE idiom, killing the divergence between the 4 buggy and 2 guarded sites.
  - **Best form:** promote make_client's local `_unwrap` closure to a module-level reusable `unwrap_secret(v) -> str` helper and use it in ALL 7 places (ClaudeClient.__init__, make_client's 4 keys, the 4 service sites -> 2 already guarded can migrate too). ONE unwrap implementation. (Caller's option (a)-vs-(b): the answer is BOTH; the helper makes "both" a single source of truth.)

**Net: (a) is mandatory (root cause, no-op for str callers). (b) is strongly recommended for consistency/future-proofing. The MINIMAL fix that resurrects the 4 overlays is (a) alone; the BEST fix is (a)+(b) via a shared helper.**

### A9. $0 VERIFICATION PLAN (no LLM spend for the primary proof)
**Primary proof = $0 unit test (no network, no LLM):**
- Construct `ClaudeClient(model_name="claude-haiku-4-5", api_key=SecretStr("sk-ant-test"))` and assert `client._api_key == "sk-ant-test"`, `isinstance(client._api_key, str)`, and `not hasattr(client._api_key, "get_secret_value")`. Proves the stored key is a plain str -> the SDK header can never receive a SecretStr. **No `Anthropic()` instantiation, no network, no spend.**
- Construct with a plain `str` and assert it's unchanged (no double-unwrap; identity preserved). Optionally assert the same str object passes through.
- For the 4 service sites: monkeypatch `settings.anthropic_api_key = SecretStr("sk-ant-test")`, then assert the site's `anthropic_key` local (or the shared `unwrap_secret` helper) yields a `str`. Easiest if the unwrap is the shared helper -> unit-test the helper directly.
- (Optional, still $0) `ClaudeClient(..., api_key=SecretStr("sk"))._get_client()` should NOT raise the SecretStr TypeError -- but this instantiates the real `Anthropic` SDK object (constructor only, no HTTP) which is $0; prefer the pure `_api_key` type assertion to avoid SDK-import fragility in CI.

**Secondary proof = live cycle-log signal (REQUIRES small LLM spend -- NEEDS OPERATOR APPROVAL):**
- A real autonomous cycle invokes Claude Haiku for these overlays (Signal Stack note ~ $0.10/day Haiku). Live success signals: `news_screen` logs `News screen: N raw -> M deduped headlines` THEN returns a NON-empty signal dict (today it logs headlines then returns `{}` post-error); `meta_scorer` logs scored conviction (not "fallback (no API key)"); `macro_regime` writes a FRESH `_cache/macro_regime.json` with `computed_at` > 2026-06-01; `pead_signal` returns a non-fallback PeadSignalOutput.
- **FLAG: live verification needs operator LLM-spend approval** (CLAUDE.md: "LLM API costs require Peder's explicit approval"). The ~$0.10/day Haiku cost is small but non-zero. **Make the $0 unit test the PRIMARY/required gate evidence; the live cycle-log is OPTIONAL confirmation gated on operator approval.** Do NOT bake a live LLM run into the GENERATE verification command -- use the unit test as the deterministic gate (consistent with the project's "$0 proof first" discipline). The masterplan step's `verification.live_check` can be satisfied by the unit-test output + a cycle-log line IF the operator approves a run; otherwise the unit test stands alone.

### Part A SUMMARY
- **4 buggy sites CONFIRMED** (news_screen:264, macro_regime:432, pead_signal:278, meta_scorer:184), keys read at :258/:427/:248/:166 with the `or ""` truthiness footgun.
- **2 other sites already guarded** (call_transcript_gpr:113, analyst_narrative_scorer:136) -- not buggy; optionally migrate to the shared helper.
- **1 latent path** (SkillFileIdCache via ClaudeClient.upload, prompts.py:113) closed by the self-unwrap.
- **Regression = commit d3f34caf, 2026-05-13** (SecretStr migration); overlays dead since then; last-good macro cache computed_at=2026-04-24.
- **Fix = (a) ClaudeClient.__init__ self-unwrap [mandatory root cause] + (b) edge unwrap at the 4 sites [hygiene], ideally via a shared `unwrap_secret()` helper.** Self-unwrap is a no-op for plain-str callers (no double-unwrap) -> US pure-quant path unaffected.
- **$0 verification:** unit test asserting `ClaudeClient(api_key=SecretStr(...))._api_key` is a plain `str`. Live cycle-log proof needs operator LLM-spend approval (~$0.10/day Haiku).

---

## PART B -- EXTERNAL RESEARCH

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/_client.py | 2026-06-01 | official SDK source (tier-1) | WebFetch (full) | `api_key` typed `str | None`; stored **as-is, no coercion** (`self.api_key = api_key`); `_api_key_auth` property -> `{"X-Api-Key": api_key}`; merged into `auth_headers`. A non-str (SecretStr) flows straight into the header dict -> httpx raises "Header value must be str or bytes". **CONFIRMS the exact failure mechanism.** |
| https://pydantic.dev/docs/validation/2.0/usage/types/secrets/ (from docs.pydantic.dev/2.0 301) | 2026-06-01 | official docs (tier-1) | WebFetch (full) | `repr` masks: `password=SecretStr('**********')`; **"Use get_secret_value method to see the secret's content"**; `get_secret_value()` returns plaintext (`#> IAmSensitive`); `model_dump`/`model_dump_json` return the MASKED repr by default. Confirms you MUST call get_secret_value() for the real value. |
| https://github.com/fastapi-users/fastapi-users/discussions/700 | 2026-06-01 | community (maintainer guidance) | WebFetch (full) | Maintainer @frankie567: **"you can indeed unwrap the value manually when instantiating your authentication backend"** -- i.e. extract the str yourself at the trust boundary rather than relying on the consuming library to handle SecretStr. Endorses the EDGE-UNWRAP pattern (= the make_client `_unwrap` + the recommended ClaudeClient self-unwrap). |
| https://github.com/pydantic/pydantic/discussions/4217 | 2026-06-01 | official discussion (maintainer, tier-1) | WebFetch (full) | **THE footgun proof.** `SecretStr("abc").get_secret_value()` -> `'abc'` but `str(SecretStr("abc"))` -> `'**********'` (the MASKED value, not plaintext). Maintainer Samuel Colvin: "if you don't care about security, just use `str`." Masking is intentional; you MUST call get_secret_value() explicitly. This is exactly why `getattr(...) or ""` (which keeps the wrapper) then handing it to the SDK fails -- the wrapper is NOT its plaintext. |
| https://www.getorchestra.io/guides/pydantic-secret-types-handling-sensitive-data-securely-with-secretstr-and-secretbytes | 2026-06-01 | industry guide | WebFetch (full) | Plaintext via `.get_secret_value()`; **"unwrap at the point of actual use"**; "ensure the actual content is not displayed when printing or converting to strings"; pattern shows NOT passing the wrapper object to external clients. Reinforces edge-unwrap-before-SDK. |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://docs.pydantic.dev/2.0/usage/types/secrets/ | official docs | 301-redirects to the pydantic.dev mirror (read in full via redirect target, counted above) |
| https://github.com/pydantic/pydantic/issues/9139 | official issue | SecretStr leaks plaintext on a validation error -- repr-masking caveat (not load-bearing here) |
| https://docs.pydantic.dev/2.7/examples/secrets/ | official docs | 301-redirects to the 2.7 API/types mirror; superseded by 2.0 + maintainer-discussion reads |
| https://pydantic.dev/docs/validation/latest/concepts/types/ | official docs | "Types" concept page; mentions SecretStr exists but the dedicated secrets/discussion pages carry the get_secret_value + masking detail |
| https://medium.com/@raydebra89/dont-hardcode-your-api-key-modern-python-config-management-with-pydantic-s-secret-handling-31526e556bd8 | industry blog | pydantic-settings + SecretStr config pattern; shows masking only, no unwrap detail (snippet sufficient) |
| https://pypi.org/project/anthropic/ | official | SDK package page; the _client.py source is the authoritative read |
| https://platform.claude.com/docs/en/api/sdks/python | official docs | Python SDK usage; api_key via str/env -- corroborates str-typing |

### Search-query variants run (3-variant discipline)
1. **Current-year frontier (2026):** "pydantic v2 SecretStr get_secret_value when to unwrap trust boundary 2026"
2. **Last-2-year window (2025/2024):** "pydantic SecretStr defense-in-depth secret handling LLM client 2025" (Recency scan below)
3. **Year-less canonical:** "Anthropic Python SDK Header value must be str or bytes api_key error" (surfaced the SDK _client.py source) + the bare "pydantic SecretStr secrets docs" / "cast SecretStr to str".

### Recency scan (2024-2026) -- PERFORMED
Searched the last-2-year window on (a) pydantic v2 SecretStr handling, (b) Anthropic SDK secret/header typing, (c) defense-in-depth secret unwrapping in LLM client wrappers. **Findings (COMPLEMENT prior art; none overturn the recommended fix):**
1. **pydantic v2 SecretStr semantics are STABLE 2024-2026.** `get_secret_value()` is still the only sanctioned plaintext accessor; `str()`/`repr()` still mask to `'**********'`; this is unchanged from the v2.0 (2023) docs through the latest (2026) docs and the maintainer discussion. The fix idiom (`.get_secret_value()` / `hasattr(...,"get_secret_value")` guard) is current best practice, not a deprecated one. No newer pydantic API supersedes it.
2. **Anthropic Python SDK still types `api_key: str | None` with no coercion** (live `_client.py` on `main`, read 2026-06-01). No SDK-side change makes it tolerate a SecretStr; the unwrap MUST happen caller-side. This is stable across the SDK's 2024-2026 versions (the project pins `anthropic>=0.96.0`).
3. **2024-2026 secret-handling consensus = unwrap at the trust boundary / point of use** (fastapi-users maintainer #700; Orchestra guide). The "unwrap once at the edge, never pass the wrapper to an external client" pattern that `make_client._unwrap` already implements is the recommended one; the bug is that 4 sites bypass that edge. No 2024-2026 source argues for passing SecretStr objects into SDKs.
4. **No source found that contradicts** the recommendation to (a) self-unwrap in the client wrapper as defense-in-depth AND (b) unwrap at the call sites. The only nuance is the masking footgun itself (discussion #4217): because `str(SecretStr)` silently returns `'**********'` rather than erroring, a naive `str(key)` "fix" would inject a literal `'**********'` as the API key (a DIFFERENT, equally-broken failure) -- so the fix MUST use `get_secret_value()`, NOT `str()`. This is a real adversarial consideration captured below.

### Key findings (per-claim, cited)
1. **The SDK stores api_key uncoerced and puts it directly in a header** -- `self.api_key = api_key`; `_api_key_auth -> {"X-Api-Key": api_key}` (Source: anthropic-sdk-python `_client.py`, https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/_client.py, accessed 2026-06-01). A SecretStr reaching this dict is what raises "Header value must be str or bytes".
2. **You MUST call get_secret_value() for the plaintext; str() gives the mask** -- `SecretStr("abc").get_secret_value()` -> `'abc'`; `str(SecretStr("abc"))` -> `'**********'` (Source: pydantic maintainer discussion #4217, https://github.com/pydantic/pydantic/discussions/4217, accessed 2026-06-01; and pydantic 2.0 secrets docs "Use get_secret_value method to see the secret's content", https://pydantic.dev/docs/validation/2.0/usage/types/secrets/).
3. **A non-empty SecretStr is truthy** -- masking affects only `__str__`/`__repr__`/serialization, not boolean/length; so `getattr(...) or ""` returns the wrapper, and `if not key:` passes (Source: pydantic 2.0 secrets docs + discussion #4217 behavior; the wrapper holds the value and only hides it on display).
4. **Unwrap at the trust boundary / point of use, do not pass the wrapper to external code** -- maintainer guidance "unwrap the value manually when instantiating your authentication backend" (Source: fastapi-users #700, https://github.com/fastapi-users/fastapi-users/discussions/700) and "unwrap at the point of actual use ... not passing the wrapper object to external clients" (Source: Orchestra guide, https://www.getorchestra.io/guides/pydantic-secret-types-handling-sensitive-data-securely-with-secretstr-and-secretbytes). This is precisely what `make_client._unwrap` does and what the 4 buggy sites omit.

### Consensus vs debate (external)
- **Consensus:** (a) `get_secret_value()` is the ONE sanctioned plaintext accessor; (b) `str(SecretStr)` returns the mask, not the value (intentional); (c) unwrap once at the edge / point of use and never hand the wrapper to an external SDK/HTTP layer; (d) the consuming SDK (Anthropic) will not coerce -- caller must unwrap.
- **Debate / nuance:** Where to unwrap -- at each call site vs centrally in the wrapper. The literature (fastapi-users, Orchestra) favors "at the point of use", but a defense-in-depth wrapper-level unwrap is a strictly-safer superset (a no-op for str inputs). The pyfinagent answer is BOTH (A8): wrapper self-unwrap as the root-cause guarantee + edge unwrap for site-level clarity, ideally via one shared helper.

### Pitfalls (from literature) -> applied to phase-51.1
1. **The `str()` "fix" trap (adversarial).** Because `str(SecretStr)` returns `'**********'` WITHOUT erroring (discussion #4217), a careless fix using `str(key)` would silently send the literal mask `'**********'` as the API key -- the SDK would accept it as a str (no TypeError) and then get a 401 from the API, a NEW silent failure. **The fix MUST use `get_secret_value()` (or the `hasattr` guard), NEVER `str()`.** make_client's `_unwrap` correctly uses `get_secret_value()` and only falls back to `str()` for non-Secret non-empty values -- preserve that ordering.
2. **Double-unwrap.** Calling `.get_secret_value()` on an already-unwrapped str raises AttributeError. The `hasattr(v, "get_secret_value")` guard prevents this; do NOT unconditionally call `.get_secret_value()` at a site that might already hold a str (e.g. after the wrapper self-unwraps). The guard-based idiom is double-unwrap-safe.
3. **Masking hides the bug in logs.** Because the wrapper masks itself, a `logger.info(f"key={key}")` shows `'**********'` whether or not it's unwrapped -- so logs can't distinguish the wrapper from the plaintext. The $0 unit test (A9) asserts the TYPE (`isinstance(_api_key, str)`), which masking can't hide -- the correct verification.
4. **Truthiness false-confidence.** The existing `if not anthropic_key:` guards LOOK like they validate the key, but a wrapper is always truthy, so they never catch the wrapped-but-unusable state. Don't rely on truthiness as a "key is usable" check; assert the type or attempt-and-catch.

### Application to pyfinagent (external -> internal anchors)
- SDK uncoerced-header mechanism (anthropic _client.py) -> the failure at `ClaudeClient._get_client` llm_client.py:1238 (`Anthropic(api_key=self._api_key)`); fix at __init__ llm_client.py:1222.
- `get_secret_value()` is the only plaintext accessor (pydantic #4217 + 2.0 docs) -> the unwrap expression in the A8 fix uses `.get_secret_value()`, mirroring make_client._unwrap (llm_client.py:1893-1896).
- "unwrap at the point of use, never pass the wrapper" (fastapi-users #700, Orchestra) -> edge unwrap at news_screen.py:258, macro_regime.py:427, pead_signal.py:248, meta_scorer.py:166.
- `str()` returns the mask (pydantic #4217) -> pitfall guard: fix must use get_secret_value(), not str(); the $0 test asserts `isinstance(_api_key, str)` AND value equality to catch a mask-injection regression.

### Research Gate Checklist
Hard blockers -- all satisfied:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5: anthropic-sdk-python _client.py [official SDK source], pydantic 2.0 secrets docs [official], pydantic discussion #4217 [official, maintainer], fastapi-users #700 [community/maintainer], Orchestra SecretStr guide [industry]). Hierarchy honored: 1 official SDK source + 2 official pydantic + 1 maintainer-community + 1 industry.
- [x] 10+ unique URLs total (12: 5 full + 7 snippet-only)
- [x] Recency scan (last 2 years) performed + reported (4 findings; pydantic + Anthropic SDK secret semantics stable 2024-2026; the `str()`-mask-injection adversarial nuance surfaced)
- [x] Full pages/sources read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (Part A: settings.py:104; llm_client.py:1206/1220-1223/1238/1893-1896/1963; news_screen.py:258/264; macro_regime.py:427/432; pead_signal.py:248/278; meta_scorer.py:166/184; call_transcript_gpr.py:91-95/113; analyst_narrative_scorer.py:111-115/136; prompts.py:36/113-125; commit d3f34caf; macro_regime.json computed_at)

Soft checks:
- [x] Internal exploration covered every relevant module (settings, llm_client [ClaudeClient + make_client], all 6 service construction sites, prompts/SkillFileIdCache, git history of the migration + the 4 services + the 2 guards)
- [x] Contradictions/consensus noted (where-to-unwrap debate; the str()-mask adversarial trap)
- [x] All claims cited per-claim with file:line or URL

### Research-gate JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 7,
  "urls_collected": 12,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "report_md": "handoff/current/research_brief.md (phase-51.1)",
  "gate_passed": true
}
```
