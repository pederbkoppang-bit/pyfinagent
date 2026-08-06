---
name: cc-rail-e2e-smoke-4000-2
description: Step 4000.2 CC-rail E2E smoke research -- CLAUDE_CODE_BINARY loses to PATH and is captured at import; llm_call_log is buffered so a zero read is not evidence; PUT /api/settings writes .env durably; rail-guard state has no out-of-process reader
metadata:
  type: project
---

Findings from the phase-4000.2 research gate (2026-08-06). Builds on
[[cc-rail-e2e-4000-1]]. Re-derive line numbers before citing -- they move.

**FACT 1 -- an env-var binary override that loses to PATH is a false stub hook.**
`backend/agents/claude_code_client.py::_resolve_claude_binary` tries
`shutil.which(binary)` BEFORE the `_DEFAULT_SEARCH_PATHS` list that holds
`CLAUDE_CODE_BINARY`, and that list is a MODULE-LEVEL literal evaluated at import
time. So (a) while a real `claude` is on PATH the env var is dead, and (b) a
`monkeypatch.setenv` after import cannot reach it either. A test that "stubs the
binary" via the env var silently invokes the REAL CLI -- a live rail call from a
unit test. The working mechanism is pytest's documented
`monkeypatch.setenv("PATH", str(tmp_path), prepend=os.pathsep)` with a stub file
literally named `claude`.

**Why:** the spawn prompt asserted "the rail honors CLAUDE_CODE_BINARY" as if
unconditional. It honors it only on the fallthrough branch. Generalise: any
"env override" for a resolved path is worthless until you read the resolution
ORDER and check whether the container is built at import time.

**How to apply:** whenever a test needs to intercept a binary the code under test
execs, read the resolver top-to-bottom and assert the NEGATIVE (env-only does not
redirect) so a refactor cannot reintroduce the live path.

**FACT 2 -- `llm_call_log` writes are BUFFERED, so a zero row-count is not
evidence.** `backend/services/observability/api_call_log.py`: `log_llm_call`
buffers; the flush is PIGGYBACKED on the next call and only fires at
`_FLUSH_ROWS=100` or `_FLUSH_SECONDS=60`. No timer thread, no HTTP endpoint
forces it (only two tests and `scripts/smoketest/phase6_e2e.py` call
`flush_llm()`). The tail of any window can sit in backend memory indefinitely.
Consequences: BQ cannot drive a real-time counter/abort; and "0 metered rows"
is indistinguishable from "not flushed yet" -- always pair a zero with a
POSITIVE control (N>0 of the rows you DO expect) before calling it a pass.
Same class as the fail-open zeros in `spend.py::fetch_llm_spend` (returns
`(0.0, 0.0)` on any exception).

**FACT 3 -- `PUT /api/settings/` writes `backend/.env` on disk**, then clears the
settings cache. A partial body is fine (`model_dump(exclude_none=True)`). So a
crashed flag-flip run leaves the flag mutated ACROSS a backend restart -- the
restore-in-`finally` must itself be a PUT, not an in-memory reset. GET is
server-cached (~300s) but the PUT invalidates it, so a post-PUT GET is fresh.

**FACT 4 -- rail-guard state is process-local with NO out-of-process reader.**
`claude_code_client.py::rail_guard_status()` reads a module global; its only
production consumer is the autonomous loop, which copies it into the cycle-history
row. There is no API endpoint. Any check asserting `consecutive_failures == 0`
from outside the backend is fabricated-SAFE unless it uses surrogates
(in-window `ok=false` rail rows, the `rail_guard_skipped:` marker in
`LLMResponse.thoughts`, the `breaker_open` P1 page).

**FACT 5 -- the analysis pipeline is ALL-Claude today, so rail volume is high.**
Live pins measured from `GET /api/settings/`: `gemini_model=claude-sonnet-4-6`,
`deep_think_model=claude-opus-4-8`. The orchestrator builds its clients from those
SETTINGS, not from `model_tiers.resolve_model()` -- `_BUILD_TIER`'s claude roles
are Layer-2 only and are NOT the analysis-path bound. With ~39 LLM calls in a full
single-ticker analysis, one analysis plausibly exceeds a 30-rail-call cap. Also:
there is no cancel endpoint for a running analysis, so a "mid-analysis abort" can
only abort the observer, never the backend's in-flight work.

**Testing shape that works here:** `pytest-subprocess` is USELESS when the code
under test is a child process (it hooks `subprocess.Popen` in-process).
Use `pytest-httpserver` (real localhost socket; `assert_request_made(..., count=0)`
proves a request was NOT made) or copy the in-repo template
`backend/tests/test_phase_76_9_2_max_bridge.py` (ThreadingHTTPServer stub +
`importlib.util.spec_from_file_location` to import a `scripts/` file).
