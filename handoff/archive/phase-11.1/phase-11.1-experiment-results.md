# Experiment Results -- phase-11.1 Pin google-genai + shim

**Step:** 11.1 (pin + shim).
**Date:** 2026-04-19.

## What was built

Tight scope: 2 new files + 1 line in `backend/requirements.txt`. No call-site migrations (phase-11.2+).

**Dep pin:**
- `google-genai==1.73.1` added to `backend/requirements.txt` with an exact-equals phase-11.1 comment referencing the 2026-06-24 deprecated-SDK removal.
- `google-cloud-aiplatform==1.142.0` left as-is (still needed for BigQuery + Vertex AI Search non-generative + `google.oauth2.service_account.Credentials`).

**New module `backend/agents/_genai_client.py`** (~120 lines):
- `get_genai_client() -> Any` — double-checked-lock singleton. Fast path no-lock when already built; slow path holds lock once and calls `_build_client`. Wraps the slow-path call in a defense-in-depth `try: ... except Exception: ...` per the fail-open contract test that exposed a gap during this cycle (Q/A-caught pre-Q/A).
- `_build_client()` internal — imports `from google import genai` lazily; reads `gcp_project_id`/`gcp_location`/`gcp_credentials_json` from Settings; passes explicit `credentials=service_account.Credentials.from_service_account_info(...)` when json is non-empty, else ADC. Returns `genai.Client(vertexai=True, project=..., location=..., credentials=...)`. Every failure mode (SDK absent, settings error, credentials parse error, Client init error) logs a WARNING and returns `None`.
- `close_genai_client()` — drops the singleton + calls `.close()` on the prior client when present. Safe to call with nothing built. Safe across threads.
- `reset_for_test()` — explicit alias so test intent reads cleanly.

**New tests `backend/tests/test_genai_client.py`** (6 tests): singleton identity, project + location kwargs, fail-open when build raises, `_build_client` returns None on settings failure, close drops the singleton, reset_for_test alias. All monkeypatched — zero real API calls, zero real GCP auth.

## Pre-Q/A self-check finding

My first version of `get_genai_client` did NOT wrap the `_build_client` call in a try/except. A test I wrote specifically to check this fail-open behavior caught it and failed — the outer fail-open guard was missing. Added the guard before Q/A. Documented in `_genai_client.py` with a `pragma: no cover -- defense-in-depth` comment on the exception handler (since `_build_client` itself never raises by construction, the outer guard is defensive).

## File list

Created:
- `backend/agents/_genai_client.py`
- `backend/tests/test_genai_client.py`

Modified:
- `backend/requirements.txt` (+1 line)

NOT touched: any existing Python source. NO call-site migration (phase-11.2+). NO `vertexai` removal (phase-11.4).

## Verification command output

### Immutable (from contract)

```
$ source .venv/bin/activate && python -c "from google import genai; print('google-genai importable')"
google-genai importable
```

Exit 0. Package is installed in the active venv (`pip install google-genai==1.73.1` ran during this cycle).

### Syntax + public API

```
$ python -c "import ast; ast.parse(open('backend/agents/_genai_client.py').read()); ast.parse(open('backend/tests/test_genai_client.py').read()); print('OK')"
OK

$ python -c "from backend.agents._genai_client import get_genai_client, close_genai_client, reset_for_test; print('shim importable')"
shim importable
```

### Pin grep

```
$ grep -E "^google-genai" backend/requirements.txt
google-genai==1.73.1         # exact pin (phase-11.1) -- replaces deprecated vertexai.generative_models (removal 2026-06-24)
```

Exact-equals, commented.

### Unit tests

```
$ pytest backend/tests/test_genai_client.py -x -q
......                                                                   [100%]
6 passed, 1 warning in 0.49s
```

(The warning is a Python 3.17 pre-deprecation noise from the SDK's types.py; unrelated to this cycle.)

### Regression across all phase-3 + phase-6 + phase-11 tests

```
$ pytest backend/tests/test_skill_optimizer.py backend/tests/test_regime_detector.py backend/tests/test_planner_agent.py backend/tests/test_evaluator_agent.py backend/tests/test_autonomous_loop_integration.py backend/tests/test_bq_writer.py backend/tests/test_observability.py backend/tests/test_sentiment_ladder.py backend/tests/test_calendar_watcher.py backend/tests/test_genai_client.py -q
79 passed, 1 skipped, 5 warnings in 9.97s
```

Zero regressions; +6 from this cycle.

## Contract criterion check

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `google-genai==1.73.1` exact pin in requirements.txt with phase-11 comment | PASS |
| 2 | `google-cloud-aiplatform` pin preserved | PASS (unchanged) |
| 3 | `vertexai` pin NOT removed (11.4 owns that) | PASS (unchanged) |
| 4 | `_genai_client.py` exports 3 public names with the documented factory shape | PASS |
| 5 | Fail-open at all 4 failure modes (SDK absent, settings error, creds parse error, Client init error) | PASS (every path has try/except + WARNING + returns None) |
| 6 | Tests >=3 covering singleton, kwargs, fail-open, close | PASS (6 tests) |
| 7 | Immutable verify exits 0 | PASS |

## Known caveats

1. **Live `genai.Client()` not exercised against a real Vertex project** in this cycle — no live credentials in the test venv. The factory is verified via monkeypatch; the real-Client path will be exercised when phase-11.2 migrates the first call site (evaluator_agent.py).
2. **One `DeprecationWarning` from the SDK itself** (`google/genai/types.py:42` — `_UnionGenericAlias` deprecated for Python 3.17). This is upstream, unrelated to pyfinagent. Logging it so Q/A doesn't mistake it for a new issue we introduced.
3. **`close_genai_client` calls `.close()` on the prior client when present**, but the python-genai 1.73.1 Client may or may not have a `close()` method depending on transport (REST vs gRPC). Guarded by `hasattr` — no crash path either way.
4. **Pre-Q/A self-check caught a real bug** (missing outer fail-open guard in `get_genai_client`). Noted explicitly because the cross-cycle discipline of "run the tests you just wrote BEFORE Q/A" is paying off.
