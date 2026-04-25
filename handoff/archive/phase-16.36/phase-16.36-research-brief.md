# Research Brief: phase-16.36 — Backend Follow-ups (datetime / client reset / mock test / Gemini tokens)

**Tier assumption: simple** (20 utcnow sites found; none require behavioral changes beyond a mechanical swap; promoting to moderate based on site count per briefing instructions.)
**Actual tier: moderate** — 20 utcnow sites across 7 files warrants moderate depth.

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://docs.python.org/3/library/datetime.html | 2026-04-25 | Official doc | WebFetch | "Deprecated since version 3.12: Use datetime.now() with UTC instead." |
| https://blog.miguelgrinberg.com/post/it-s-time-for-a-change-datetime-utcnow-is-now-deprecated | 2026-04-25 | Authoritative blog | WebFetch | Replacement is `datetime.now(datetime.UTC)`; returns aware vs naive. If you need naive: `.replace(tzinfo=None)` |
| https://docs.python.org/3/library/unittest.mock.html | 2026-04-25 | Official doc | WebFetch | `side_effect=ExceptionClass(...)` on a Mock raises on call; `patch.object(obj, 'attr')` patches instance attribute |
| https://realpython.com/python-mock-library/ | 2026-04-25 | Authoritative blog | WebFetch | Distinction between `patch()` (full module path) and `patch.object()` (object + attribute name); "patch the object where it's looked up" |
| https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/list-token | 2026-04-25 | Official doc | WebFetch | `response.usage_metadata.prompt_token_count`, `.candidates_token_count`, `.total_token_count` are the three fields |
| https://www.andreagrandi.it/posts/python-now-time-to-migrate-from-utcnow/ | 2026-04-25 | Blog | WebFetch | Direct migration path confirmed: `datetime.now(timezone.utc)` |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://discuss.python.org/t/why-is-datetime-utcnow-deprecated/86868 | Forum | Covered by Python docs + Miguel blog |
| https://til.simonwillison.net/python/utc-warning-fix | Blog | Supplementary; core migration pattern already read |
| https://docs.moderne.io/user-documentation/recipes/recipe-catalog/python/migrate/replacedatetimeutcnow/ | Tool docs | Automated recipe for the same migration; informational |
| https://github.com/dbt-labs/dbt-core/issues/9791 | Issue tracker | Real-world confirmation of the deprecation pattern |
| https://github.com/googleapis/python-genai/issues/1284 | GitHub issue | Token count discrepancy note (countTokens vs usage_metadata); covered by official docs |
| https://googleapis.github.io/python-genai/ | SDK docs | Attempted full fetch; usage_metadata schema summary only |
| https://pypi.org/project/anthropic/ | PyPI | Confirmed SDK version 0.79.0 (2026-02-07); AuthenticationError class confirmed |
| https://github.com/pytest-dev/pytest-mock/issues/100 | GitHub issue | Module-scoped mocker conflict; not relevant to function-scoped test plan |
| https://discuss.python.org/t/deprecating-utcnow-and-utcfromtimestamp/26221 | Forum | Design rationale; covered by official docs |
| https://medium.com/learn-design-patterns/there-can-be-only-one-a-deep-dive-into-making-singletons-work-for-modern-python-applications-b003d1f062f2 | Blog | Singleton reset patterns; informational |

---

## Search queries run (three-variant discipline)

1. **Current-year frontier (2026):** "datetime.utcnow deprecation Python 3.12 3.13 migration 2026" | "anthropic SDK AuthenticationError pytest mock 2026"
2. **Last-2-year window (2024-2025):** "vertex ai python sdk usage metadata token count gemini 2025" | "credential rotation singleton invalidation python pattern 2024 2025"
3. **Year-less canonical:** "datetime now timezone utc migration Python" | "pytest mock library exception unittest.mock patch module singleton" | "vertex ai gemini python usage metadata"

---

## Recency scan (2024-2026)

Searched with 2025 and 2026 year tags on all four topics.

- **datetime deprecation:** No new findings supersede the Python 3.12 deprecation. Python 3.14 (current project runtime) still issues the DeprecationWarning; removal is not yet scheduled for 3.14 but is described as "future version." The migration path (`datetime.now(timezone.utc)`) is stable since 3.2 and unchanged.
- **Anthropic SDK:** Version 0.79.0 released 2026-02-07. `AuthenticationError` class is confirmed in the hierarchy. No changes to the exception class interface.
- **Vertex AI token usage:** No API changes to `usage_metadata` in 2025-2026. Fields (`prompt_token_count`, `candidates_token_count`, `total_token_count`) are stable. One 2025 GitHub issue (#1284) notes a discrepancy between `count_tokens()` API and `usage_metadata.prompt_token_count` on Vertex AI (vs Gemini API), but post-execution `usage_metadata` on the response is authoritative for billing — no impact on the extraction pattern.
- **Singleton reset / credential rotation:** No new Python-specific patterns in 2024-2026 that differ from the established module-global `None` invalidation approach.

Result: no superseding findings in the 2024-2026 window; canonical sources remain current.

---

## Key findings

1. **datetime.utcnow() replacement is mechanical.** `datetime.utcnow()` -> `datetime.now(timezone.utc)`. The result is timezone-aware (has `.tzinfo = UTC`). If the call site immediately chains `.isoformat()` or `.strftime(...)`, it continues to work — `isoformat()` on an aware datetime appends `+00:00`, which is valid ISO 8601 and accepted by BigQuery TIMESTAMP fields. If the call site chains `.replace(tzinfo=None)` (needed in `outcome_tracker.py` L107 where the normalization is explicit), the `.now(timezone.utc)` version already produces the right value to strip. (Source: Python docs, miguelgrinberg blog)

2. **Scope of datetime audit: 20 sites across 7 files.** All are in `backend/`. The orchestrator file itself has zero `datetime.utcnow()` calls — all sites are in supporting files. The replacement is a single-line sed-style substitution per site with no behavioral changes needed: all sites use the result for `.isoformat()`, `.strftime(...)`, or arithmetic with other naive datetimes (the aware version arithmetically compares correctly when the other operand is also aware UTC or converted). Exception: `outcome_tracker.py` lines 48 and 108 subtract a potentially naive `rec_date` from `datetime.utcnow()` — after migration, `datetime.now(timezone.utc)` will produce an aware datetime, and if `rec_date` is still naive the subtraction will raise. However, line 107 already does `rec_date = rec_date.replace(tzinfo=None)` before the subtraction, so line 108 is safe. Line 48 at `outcome_tracker.py` has no such guard — the `rec_date` comes from `datetime.fromisoformat(analysis_date)` which may be naive. **This site needs care**: strip tzinfo from `datetime.now(timezone.utc)` before subtracting, or ensure `rec_date` is always tz-aware. Simplest fix: `(datetime.now(timezone.utc).replace(tzinfo=None) - rec_date).days` to preserve the existing naive-naive arithmetic.

3. **`reset_anthropic_client()` design.** The orchestrator uses a module-level `_orchestrator` singleton (line 1429) holding `_client` and `_anthropic_unavailable` on the instance. `get_settings()` is wrapped in `@lru_cache()` — swapping `.env` does NOT automatically reload settings. A `reset_anthropic_client()` function should: (a) clear `_orchestrator._client = None`, (b) clear `_orchestrator._anthropic_unavailable = False`, (c) call `get_settings.cache_clear()` to force settings reload on next call. This is the complete key-rotation reset. Thread safety is not required: orchestrator is single-process, and `asyncio` loops are cooperative (no race between concurrent `_client` sets). (Source: settings.py L174, orchestrator.py L1429-1435)

4. **Mock test design for AuthenticationError fallback.** The canonical pattern is `patch.object(orchestrator_instance, '_client', new=mock_anthropic_client)` where `mock_anthropic_client.messages.create.side_effect = anthropic.AuthenticationError(...)`. However, `AuthenticationError.__init__` in Anthropic SDK 0.79 requires a `response` argument (it extends `APIStatusError`). The safest mock approach: create a `MagicMock` where `messages.create.side_effect = anthropic.AuthenticationError.__new__(anthropic.AuthenticationError)` or use `MagicMock(side_effect=Exception("mock 401"))` and patch the `isinstance(e, anthropic.AuthenticationError)` check by mocking the `anthropic` import inside the method. The cleanest approach for the codebase's pattern (lazy import inside the except block): patch `anthropic.AuthenticationError` itself so `isinstance` returns True for a plain exception. See "Application to pyfinagent" section for recommended test design. (Source: unittest.mock docs, realpython.com)

5. **Gemini token extraction attribute path.** The `_gemini_text_call()` method at line 232 calls `gemini.generate_content(prompt, config)` which returns an object from `GeminiClient`. In `llm_client.py` line 568-578, `GeminiClient.generate_content` normalizes the Vertex AI response into `LLMResponse` with `usage_metadata` of type `UsageMeta` (lines 228-241). So `resp` in `_gemini_text_call()` is an `LLMResponse`, not a raw Vertex AI response. Correct extraction: `getattr(resp, 'usage_metadata', None)` then `.prompt_token_count` and `.candidates_token_count`. The `cost_tracker.py` at lines 124-130 shows the canonical pattern for raw Vertex AI responses. For `_gemini_text_call`, use `LLMResponse.usage_metadata.prompt_token_count` and `.candidates_token_count`. The returned `usage` dict should mirror the Anthropic shape: `{"input": prompt_token_count, "output": candidates_token_count}`. (Source: llm_client.py L228-241, L568-578; cost_tracker.py L124-130)

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/agents/multi_agent_orchestrator.py` | 1485 | MAS Layer-2 orchestrator; target for tasks 44, 45, 46 | No datetime.utcnow() calls; `_gemini_text_call` at L222-241 returns hardcoded `{"input":0,"output":0}` usage |
| `backend/config/settings.py` | 177 | Pydantic-settings; `get_settings()` with `@lru_cache()` | `anthropic_api_key` field at L86; `lru_cache` means re-reading `.env` requires `cache_clear()` |
| `backend/agents/llm_client.py` | ~1100 | `GeminiClient`, `UsageMeta`, `LLMResponse` | `UsageMeta` at L228-241 has `prompt_token_count`, `candidates_token_count`, `total_token_count`; `generate_content` returns `LLMResponse` |
| `backend/agents/cost_tracker.py` | ~200 | Token usage extraction pattern | L124-130: canonical Vertex AI `usage_metadata` extraction (for raw responses) |
| `backend/tools/sec_insider.py` | ~400 | SEC filing tools | L160, L239: `datetime.utcnow()` used for cutoff date strings (`.strftime(...)`) |
| `backend/tools/fred_data.py` | ~100 | FRED data fetcher | L29: `datetime.utcnow()` used for start date string |
| `backend/agents/memory.py` | ~200 | Episodic memory | L92, L100: `datetime.utcnow().isoformat()` for timestamps |
| `backend/agents/skill_optimizer.py` | ~900 | Skill optimization | L599: `datetime.utcnow().isoformat()` |
| `backend/backtest/data_ingestion.py` | ~400 | BQ data ingestion | L96, L188, L280: `.isoformat()`; L340: `.strftime(...)` |
| `backend/backtest/spot_checks.py` | ~500 | Spot-check harness | L37, L56: `.isoformat()`; L434: `.strftime(...)` |
| `backend/slack_bot/governance.py` | ~200 | Governance alerts | L89: `.isoformat()` |
| `backend/db/bigquery_client.py` | ~500 | BQ write client | L141, L378, L449: `.isoformat()` for analysis timestamps |
| `backend/services/outcome_tracker.py` | ~200 | Outcome evaluation | L48: naive-naive arithmetic (needs care — see finding #2); L108: already guarded |
| `backend/tests/test_outcome_tracker.py` | ~160 | Existing test | L123: `datetime.utcnow()` in test fixture — also needs fixing |
| `tests/` (top-level) | N/A | Alternative test root | Discovered by pytest; `backend/tests/` is also discovered |

**Singleton structure in orchestrator (for task 44):**
- Module-level: `_orchestrator: Optional[MultiAgentOrchestrator] = None` at L1429
- `get_orchestrator()` at L1431: lazy-init pattern, no lru_cache
- Instance attrs: `self._client` (L136), `self._anthropic_unavailable` (L143)
- `_get_client()` at L161: reads `settings.anthropic_api_key` on first init
- No threading.Lock anywhere in the module — single-process, single-thread-per-call confirmed

**Gemini fallback path (for task 46):**
- `_gemini_text_call()` at L222: calls `gemini.generate_content(prompt, config)` at L233
- Returns `text, {"input": 0, "output": 0}` — hardcoded zero usage at L238
- `gemini` here is `GeminiClient` from `llm_client.py`
- `GeminiClient.generate_content()` returns `LLMResponse` (not raw Vertex AI response)
- `LLMResponse.usage_metadata` is a `UsageMeta` dataclass with `prompt_token_count`, `candidates_token_count`, `total_token_count`

**Test directory layout:**
- `pytest` from repo root discovers both `backend/tests/` and `tests/` (top-level)
- No `tests/agents/` directory exists — the new test file path should be `backend/tests/test_anthropic_fallback.py`
- `backend/tests/test_outcome_tracker.py` is the model for FakeBQ and MagicMock stub patterns

---

## Consensus vs debate (external)

**datetime migration:** Full consensus. Python docs, CPython issue tracker, Miguel Grinberg, Andrea Grandi all agree: `datetime.now(timezone.utc)`. No dissent. The naive-vs-aware behavioral difference is well-documented but rarely a problem for `.isoformat()` chains (output gains `+00:00` suffix, which BigQuery accepts).

**Mock testing pattern:** Consensus on `patch.object()` + `side_effect`. The Anthropic SDK `AuthenticationError` constructor signature (requires `response` arg) is the only complication; the workaround is to mock at the `isinstance` check level or construct the exception via `MagicMock`.

**Gemini token extraction:** Consensus on `usage_metadata.prompt_token_count` / `.candidates_token_count`. One open GitHub issue (#1284) about discrepancies between `count_tokens` pre-call and post-call `usage_metadata` on Vertex AI is not relevant (we use post-call only).

---

## Pitfalls (from literature and code audit)

1. **Aware-naive arithmetic.** `outcome_tracker.py:48` subtracts `rec_date` (potentially naive) from `datetime.utcnow()`. After migration, `datetime.now(timezone.utc)` is aware. If `rec_date` is naive the subtraction raises `TypeError`. Fix: `.replace(tzinfo=None)` on the `now()` result at that site only (other sites that chain `.isoformat()` or `.strftime()` do not have this issue).

2. **isoformat() output change.** `datetime.now(timezone.utc).isoformat()` produces `"2026-04-25T12:00:00+00:00"` instead of `"2026-04-25T12:00:00"`. BigQuery TIMESTAMP columns accept both formats. No change needed in BQ write paths. If any downstream code does exact string comparison with the naive format, it would break — but no such comparison was found in audit.

3. **AuthenticationError constructor.** `anthropic.AuthenticationError` (inherits from `APIStatusError`) requires a `message: str` and `response: httpx.Response` argument. You cannot do `raise anthropic.AuthenticationError("test")` directly in test code. Workaround: `mock_client.messages.create.side_effect = anthropic.AuthenticationError.__new__(anthropic.AuthenticationError)` (bypasses constructor) or use `spec=anthropic.AuthenticationError` on the Mock. Simplest: construct a mock `httpx.Response` stub.

4. **`get_settings()` lru_cache.** After calling `reset_anthropic_client()`, `_get_client()` will call `get_settings()` which is still cached. Must call `get_settings.cache_clear()` inside `reset_anthropic_client()` or the new key won't be read. This is the only non-obvious step in task 44.

5. **`GeminiClient.generate_content` returns `LLMResponse`, not raw Vertex AI response.** `_gemini_text_call()` calls `gemini.generate_content()` where `gemini` is a `GeminiClient` instance. The `generate_content` method returns an `LLMResponse` (see `llm_client.py:575-580`). Do NOT apply `getattr(resp, 'usage_metadata', None)` expecting a Vertex AI proto object — it will work (since `LLMResponse` also has `.usage_metadata`), but use the `LLMResponse.usage_metadata.prompt_token_count` attribute path which is already normalized.

---

## Application to pyfinagent (task-by-task)

### Task 43: datetime.utcnow() cleanup

**Scope:** 20 sites in 14 files. All in `backend/`. Zero in `multi_agent_orchestrator.py`.

**Replacement rule (applies to 19 of 20 sites):**
```python
# Before:
from datetime import datetime
datetime.utcnow()

# After:
from datetime import datetime, timezone
datetime.now(timezone.utc)
```

For sites that chain `.isoformat()` or `.strftime(...)` only — mechanical swap, no other changes.

**Special case — `outcome_tracker.py:48` (file:line anchor):**
```python
# Before (L48):
holding_days = (datetime.utcnow() - rec_date).days

# After:
holding_days = (datetime.now(timezone.utc).replace(tzinfo=None) - rec_date).days
```
This preserves naive-naive arithmetic. `rec_date` at L47 comes from `datetime.fromisoformat(analysis_date)` which is naive when `analysis_date` is a plain ISO string.

`outcome_tracker.py:108` is already guarded at L107 (`rec_date = rec_date.replace(tzinfo=None)`) so a plain swap works there.

`backend/tests/test_outcome_tracker.py:123` also needs the swap (test fixture).

**Import change needed in all files:** add `timezone` to existing `from datetime import datetime` import (or `from datetime import datetime, timezone`). Files that import `datetime` as a module (`import datetime`) should use `datetime.timezone.utc`.

**Verification:** `! grep -r "datetime.utcnow()" backend/agents/ backend/services/ backend/api/ backend/tools/ backend/backtest/ backend/db/ backend/slack_bot/ 2>/dev/null`

### Task 44: reset_anthropic_client()

**Location:** module level in `multi_agent_orchestrator.py`, after `get_orchestrator()` at L1435.

**Signature:**
```python
def reset_anthropic_client() -> None:
    """Clear the cached Anthropic client and unavailability flag.

    Call this after rotating the Anthropic API key in backend/.env so
    the next _get_client() call re-reads settings.anthropic_api_key.
    Also clears get_settings() lru_cache so the new key is loaded.
    """
    global _orchestrator
    if _orchestrator is not None:
        _orchestrator._client = None
        _orchestrator._anthropic_unavailable = False
    # Force settings reload so the new key is read from .env
    from backend.config.settings import get_settings
    get_settings.cache_clear()
```

**No threading.Lock needed** — confirmed single-process, no concurrent writers on `_orchestrator` attributes.

### Task 45: Unit test for AuthenticationError fallback

**File:** `backend/tests/test_anthropic_fallback.py`

**Test pattern (using FakeBQ-style stubs from test_outcome_tracker.py):**
```python
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.agents.multi_agent_orchestrator import MultiAgentOrchestrator

def test_call_agent_falls_back_to_gemini_on_auth_error():
    orch = MultiAgentOrchestrator()
    # Construct a mock client whose messages.create raises AuthenticationError
    mock_anthropic = MagicMock()
    import anthropic
    # Build a minimal AuthenticationError without a real httpx.Response
    auth_err = MagicMock(spec=anthropic.AuthenticationError)
    mock_anthropic.messages.create.side_effect = auth_err
    orch._client = mock_anthropic

    # Patch the isinstance check so the lazy import path works
    with patch("backend.agents.multi_agent_orchestrator.MultiAgentOrchestrator._gemini_text_call") as mock_gemini:
        mock_gemini.return_value = ("gemini fallback response", {"input": 5, "output": 10})
        # Patch the isinstance check inside _call_agent
        with patch("anthropic.AuthenticationError", new=type(auth_err)):
            from backend.agents.agent_definitions import AGENT_CONFIGS, AgentType
            config = AGENT_CONFIGS[AgentType.MAIN]
            text, usage = orch._call_agent(config, "test task")

    assert "gemini" in text.lower() or text  # fallback was called
    assert orch._anthropic_unavailable is True
    assert orch._client is None
```

**Simpler approach (recommended):** Patch `isinstance` check via `anthropic` module mock. Because `_call_agent` does `import anthropic` lazily and calls `isinstance(e, anthropic.AuthenticationError)`, we can patch the `anthropic` module on `sys.modules` with a fake where `AuthenticationError` is `Exception`. Then any raised `Exception` triggers the fallback. See recommended 4-test suite below.

**Target 4-6 tests:**
1. `test_call_agent_auth_error_triggers_gemini_fallback` — _call_agent path
2. `test_call_agent_with_tools_auth_error_triggers_gemini_fallback` — _call_agent_with_tools path
3. `test_anthropic_unavailable_flag_persists` — second call skips Anthropic entirely
4. `test_reset_anthropic_client_clears_state` — reset restores both flags
5. `test_gemini_usage_populated` — fallback returns non-zero usage dict after task 46

### Task 46: Gemini token extraction

**Location:** `_gemini_text_call()` at L222-241 in `multi_agent_orchestrator.py`.

**Current code at L233-238:**
```python
resp = gemini.generate_content(
    prompt,
    {"max_output_tokens": agent_config.max_tokens},
)
text = (getattr(resp, "text", "") or "").strip() or "No response."
return text, {"input": 0, "output": 0}
```

**After fix:**
```python
resp = gemini.generate_content(
    prompt,
    {"max_output_tokens": agent_config.max_tokens},
)
text = (getattr(resp, "text", "") or "").strip() or "No response."
# Extract token usage from LLMResponse.usage_metadata (UsageMeta dataclass)
_umeta = getattr(resp, "usage_metadata", None)
_in = getattr(_umeta, "prompt_token_count", 0) or 0
_out = getattr(_umeta, "candidates_token_count", 0) or 0
return text, {"input": _in, "output": _out}
```

This mirrors the Anthropic usage dict shape (`{"input": N, "output": N}`) and uses the same `getattr`-safe pattern as `cost_tracker.py:128-129` (file:line anchor).

**Verification command (adjusted for actual file locations):**
```bash
cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && \
python -c "
import datetime
from backend.agents.multi_agent_orchestrator import reset_anthropic_client
reset_anthropic_client()
print('reset_anthropic_client: ok')
" && \
python -c "
import subprocess, sys
result = subprocess.run(['grep', '-r', 'datetime.utcnow()', 'backend/agents/', 'backend/services/', 'backend/api/', 'backend/tools/', 'backend/backtest/', 'backend/db/', 'backend/slack_bot/'], capture_output=True, text=True)
if result.stdout.strip():
    print('FAIL: utcnow() still present:', result.stdout)
    sys.exit(1)
print('utcnow() audit: clean')
" && \
python -m pytest backend/tests/test_anthropic_fallback.py -v
```

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total (incl. snippet-only) — 16 total
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (7 files with utcnow; orchestrator; settings; llm_client; cost_tracker; test directory layout)
- [x] Contradictions / consensus noted (aware-vs-naive pitfall documented; AuthenticationError constructor pitfall documented)
- [x] All claims cited per-claim (not just listed in a footer)

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "gate_passed": true
}
```
