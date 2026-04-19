# Sprint Contract -- phase-11.1 Pin google-genai + _genai_client.py shim

**Written:** 2026-04-19 PRE-commit.
**Step id:** `11.1` in phase-11.
**Immutable verification:** `source .venv/bin/activate && python -c "from google import genai; print('google-genai importable')"` exit 0.

## Research-gate summary

Researcher envelope `{tier: moderate, external_sources_read_in_full: 5, snippet_only_sources: 8, urls_collected: 13, recency_scan_performed: true, internal_files_inspected: 4, gate_passed: true}`. Brief: `handoff/current/phase-11.1-research-brief.md`. **Three-query-variant discipline confirmed** (year-less + 2026 + 2025 all issued).

Key locked facts:
- `google-genai==1.73.1` (2026-04-14 stable). Python 3.10-3.14 compatibility range — our 3.14 venv qualifies.
- Zero breaking changes in 1.70.x → 1.73.x series (constructor + Vertex AI auth path unchanged).
- No transitive conflict with `google-cloud-aiplatform==1.142.0` (independent packages).
- Thread safety NOT documented in SDK source — double-checked lock singleton is mandatory for FastAPI/uvicorn.
- `gcp_project_id` / `gcp_location` / `gcp_credentials_json` at `backend/config/settings.py:24-27` map 1:1 to `genai.Client` kwargs.
- `_genai_client.py` does not currently exist (clean creation).

Staked rec adopted: `get_genai_client() -> genai.Client` — double-checked lock singleton; explicit `project`/`location` from `Settings`; optional `credentials` kwarg (service-account JSON when `gcp_credentials_json` is set; ADC fallback otherwise); `close_genai_client()` for FastAPI lifespan shutdown.

## Hypothesis

A thin `backend/agents/_genai_client.py` factory module with a double-checked-lock singleton + a matching pin in `backend/requirements.txt` is sufficient for phase-11.1 — every subsequent call-site migration (11.2, 11.3) imports through this one surface, and removing `vertexai` in 11.4 becomes a grep-verifiable operation.

## Success criteria

**Functional:**
1. `backend/requirements.txt` gets `google-genai==1.73.1` added (exact-equals per supply-chain pin policy; matches the `google-cloud-aiplatform==1.142.0` convention). Comment references phase-11 + the 2026-06-24 deadline for the removal of the deprecated SDK.
2. `google-cloud-aiplatform` pin preserved — NOT removed in this step (still needed for BigQuery + Vertex AI Search non-generative APIs + `google.oauth2.service_account.Credentials`).
3. `vertexai` top-level package pin NOT yet removed (phase-11.4 owns that).
4. `backend/agents/_genai_client.py` (new) exports:
   - `get_genai_client() -> genai.Client` — returns a process-singleton, double-checked-lock guarded. On first call builds via `genai.Client(vertexai=True, project=settings.gcp_project_id, location=settings.gcp_location, credentials=...)`. Credentials: if `settings.gcp_credentials_json` is non-empty, parse to service-account credentials via `google.oauth2.service_account.Credentials.from_service_account_info`; else ADC.
   - `close_genai_client() -> None` — drops the singleton so next call rebuilds. Safe to call when client missing.
   - `reset_for_test() -> None` — test helper alias for `close_genai_client` with explicit name so test intent is visible.
5. Factory fail-open: if `google-genai` is absent (pip not yet updated in dev env) OR `genai.Client(...)` raises on init, `get_genai_client()` logs a WARNING and returns `None` (NOT raise). Callers that require a real client must guard with `if client is None: ...`.
6. `backend/tests/test_genai_client.py` (new) with >=3 tests:
   - `test_get_genai_client_returns_singleton` — same object returned on successive calls after reset.
   - `test_get_genai_client_passes_project_and_location` — monkeypatch `genai.Client` constructor; assert kwargs.
   - `test_get_genai_client_fail_open_when_sdk_absent` — monkeypatch to simulate ImportError; assert returns None.
   - `test_close_genai_client_drops_singleton` — call close, next get instantiates again.
7. Immutable verify command runs cleanly (`from google import genai` succeeds).

**Correctness verification commands:**
- Syntax: `python -c "import ast; ast.parse(open('backend/agents/_genai_client.py').read()); ast.parse(open('backend/tests/test_genai_client.py').read()); print('ok')"` exit 0.
- Import smoke: `python -c "from backend.agents._genai_client import get_genai_client, close_genai_client, reset_for_test; print('ok')"` exit 0.
- SDK import: `python -c "from google import genai; print('google-genai', genai.__version__ if hasattr(genai,'__version__') else 'importable')"` exit 0.
- Unit tests: `pytest backend/tests/test_genai_client.py -x -q` — all pass.
- Regression: `pytest backend/tests/test_skill_optimizer.py backend/tests/test_regime_detector.py backend/tests/test_planner_agent.py backend/tests/test_evaluator_agent.py backend/tests/test_autonomous_loop_integration.py backend/tests/test_bq_writer.py backend/tests/test_observability.py backend/tests/test_sentiment_ladder.py backend/tests/test_calendar_watcher.py backend/tests/test_genai_client.py -q` — same 73 passing + 4 new = 77 passed / 1 skipped expected.
- Grep: `grep -E "^google-genai" backend/requirements.txt` matches exactly `google-genai==1.73.1` with a phase-11 comment.

**Non-goals:**
- NOT migrating any call site (phase-11.2-11.3).
- NOT touching `evaluator_agent.py` / `skill_optimizer.py` / `debate.py` / `risk_debate.py` / `orchestrator.py` / `llm_client.py`.
- NOT removing any existing dep.
- NOT writing the ThinkingConfig fix (phase-11.3).
- NOT instantiating the client at import time — lazy is the pattern.

## Plan

1. Install `google-genai==1.73.1` in the active venv.
2. Add to `backend/requirements.txt`.
3. Write `backend/agents/_genai_client.py`.
4. Write `backend/tests/test_genai_client.py`.
5. Run all verification commands.

## References

- `handoff/current/phase-11.1-research-brief.md`
- `handoff/archive/phase-11.0/phase-11.0-experiment-results.md` (migration plan doc)
- `docs/VERTEX_AI_GENAI_MIGRATION.md` (step 11.1 section)
- `backend/config/settings.py:24-27` (credentials plumbing)
- External read-in-full: google-genai PyPI, googleapis.github.io/python-genai, python-genai releases, python-genai/google/genai/client.py, Google Cloud SDK overview.

## Researcher agent id

`a74563d373552a785`
