# Research Brief -- phase-4000.2 (CC rail E2E smoke script + test harness)

**Tier:** `simple` (caller-specified). Depth is capped; the >=5-sources-read-in-full
floor still applies at every tier per `.claude/rules/research-gate.md`.
**Builds on:** `handoff/archive/phase-4000.1/research_brief_4000.1.md` (CLI envelope,
Max-plan billing, `llm_call_log` writer shapes) -- NOT re-derived here.
**Status:** IN PROGRESS (write-first; appended incrementally).
**Started:** 2026-08-06

## Question

4000.2 builds `scripts/qa/smoke_cc_rail_e2e.py` (`--dry` / `--live`) plus
`backend/tests/test_phase_4000_2_cc_rail_smoke.py`. Research needed on:
1. pytest patterns for stubbing a CLI binary invoked by absolute path
2. stub HTTP backends for a script exercised via `subprocess`
3. crash-safe config toggle / restore-in-finally
4. polling an async REST task endpoint with deadline + backoff
5. current `claude` CLI flags for a scripted PROBE call (recency re-check only)

Internal audit (i)-(vii) per the spawn prompt.

---

## Search queries run (three-variant discipline, per research-gate.md)

| Variant | Query |
|---------|-------|
| year-less canonical | `pytest stub external CLI binary subprocess fake executable PATH fixture` |
| year-less canonical | `pytest fixture local http.server threading stub HTTP backend subprocess integration test` |
| current-year frontier (2026) | `integration test CLI script subprocess stub binary environment variable override 2026 best practice` |
| current-year frontier (2026) | `"restore in finally" feature flag toggle test crash safe context manager python 2026` |
| last-2-year window (2025) | `pytest 2025 stub subprocess CLI binary tmp_path executable PATH monkeypatch integration test pattern` |

## Read in full (>=5 required; counts toward the gate) -- 7 fetched

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://docs.pytest.org/en/stable/how-to/monkeypatch.html | 2026-08-06 | official doc | WebFetch | The documented PATH-stub pattern, verbatim: *"Use `monkeypatch.setenv("PATH", value, prepend=os.pathsep)` to modify `$PATH`"* for subprocess interaction with fake binaries; *"All modifications will be undone after the requesting test function or fixture has finished."* |
| 2 | https://pytest-subprocess.readthedocs.io/en/latest/usage.html | 2026-08-06 | official doc | WebFetch | *"The plugin hooks on the `subprocess.Popen()`"* -- i.e. IN-PROCESS only. **Cannot intercept a subprocess spawned by a separate process**, so it is the WRONG tool for 4000.2's `subprocess.run(script)` test shape. `fp.any()`, `allow_unregistered(True)`, `pass_command()` documented. |
| 3 | https://pytest-httpserver.readthedocs.io/en/latest/howto.html | 2026-08-06 | official doc | WebFetch | *"By default, the server run by pytest-httpserver will listen on localhost on a random available port"* -- a REAL socket, so a child process can reach it. `assert_request_made(RequestMatcher("/bar"), count=0)` proves a request was **not** made. Caveat: *"serves the request in a single-threaded, blocking way"* unless `threaded=True`; handler assertions must be surfaced via `check_assertions()`. |
| 4 | https://docs.python.org/3/library/contextlib.html | 2026-08-06 | official doc | WebFetch | *"If an unhandled exception occurs in the block, it is reraised inside the generator at the point where the yield occurred. Thus, you can use a `try`...`except`...`finally` statement ... to ensure that some cleanup takes place."* Plus: a generator CM that swallows an exception silently marks it handled -- *"the generator must reraise that exception."* `ExitStack` unwinds callbacks in reverse registration order. |
| 5 | https://code.claude.com/docs/en/cli-reference | 2026-08-06 | official vendor doc | WebFetch | Current scripted flags. `--max-budget-usd` = *"Maximum dollar amount to spend on API calls before stopping (print mode only)."* `--max-turns` = *"Limit the number of agentic turns (print mode only). Exits with an error when the limit is reached."* `--bare` = *"Minimal mode: skip auto-discovery of hooks, skills, plugins, MCP servers, auto memory, and CLAUDE.md so scripted calls start faster."* -- still OPT-IN, not the `-p` default. |
| 6 | https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/ | 2026-08-06 | industry (AWS Architecture) | WebFetch | Full Jitter `sleep = random(0, min(cap, base * 2**attempt))`; no-jitter is *"the clear loser. It not only takes more work, but also takes more time than the jittered approaches."* Recommendation: jittered backoff *"should be considered a standard approach for remote clients."* |
| 7 | https://google.aip.dev/151 | 2026-08-06 | official doc (Google API standards) | WebFetch | Async-task contract: return *"some kind of promise to the user and allow the user to check back in later"*, rule of thumb ~10s+; the id *"**must** be set, to allow clients to poll the long-running ... operation until it has completed."* Notably gives **no** polling-interval guidance -- that gap is why source 6 is needed alongside it. |

## Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://github.com/aklajnert/pytest-subprocess | repo | superseded by its own docs (source 2) |
| https://til.simonwillison.net/pytest/pytest-subprocess | blog | same content, lower tier |
| https://github.com/pytest-dev/pytest-localserver | repo | alternative to source 3; same shape, fewer assertion helpers |
| https://pypi.org/project/pytest-simplehttpserver/ | package | static-file only; no request-count assertions |
| https://github.com/sdebruyn/fabric-dw-mcp-cli/issues/393 | issue (2026) | recency evidence, cited in the recency scan |
| https://github.com/cloudfoundry/cli/blob/master/integration/README.md | repo doc | real-binary integration convention, corroborating |
| https://nexte.st/docs/configuration/env-vars/ | tool doc | Rust-side env conventions; not transferable |
| https://devgex.com/en/article/00013847 | blog | `os.environ.copy()` + `env=` guidance, community tier |
| https://dev.to/kaushikcoderpy/... context-managers-2026 | blog (2026) | community tier; source 4 is authoritative for the same claim |
| https://oneuptime.com/blog/post/2026-01-24-feature-toggles-python/view | blog (2026) | feature-toggle restore pattern; community tier |
| https://www.flagsmith.com/blog/python-feature-flag | vendor blog | vendor SDK, not applicable to a PUT-based flag |
| https://builder.aws.com/content/3EumjoZascWd1oZiEgL8ORlv3qE/... | doc | AWS Builders' Library redirect target is a JS SPA; curl+tag-strip yielded 20 chars. Substituted source 6. |

---

## Internal code inventory (audit items (i)-(vii))

| File | Lines cited | Role | Status |
|------|-------------|------|--------|
| `backend/api/analysis.py` | 350-381, 384-416, 419-455 | POST start + GET poll | LIVE |
| `backend/api/auth.py` | 135-153 | `get_current_user` localhost bypass | LIVE |
| `backend/agents/llm_client.py` | 2108-2131, 2136-2151 | rail routing seam | LIVE |
| `backend/agents/claude_code_client.py` | 49-76, 91-212, 215-291 | binary resolve + rail guard + model resolve | LIVE |
| `backend/config/model_tiers.py` | 96-144, 154-168 | `_BUILD_TIER` role->model map | LIVE (Layer-2 only) |
| `backend/agents/orchestrator.py` | 517-519, 571-659, 717-737 | pipeline client construction | LIVE |
| `backend/services/observability/spend.py` | 194-250 | `fetch_llm_spend` | LIVE |
| `backend/services/observability/api_call_log.py` | 221 | `log_llm_call` writer | LIVE |
| `backend/services/autonomous_loop.py` | 1736-1771 | only reader of `rail_guard_status()` | LIVE |
| `backend/services/cycle_health.py` | 302-329 | rail flags -> cycle-history row | LIVE |
| `scripts/qa/*.py` (5 sweeps + 1 provenance check) | -- | check-script prior art | LIVE |

### (i) POST /api/analysis/ + poll shape + terminal states

- **Start:** `POST /api/analysis/` (`analysis.py:350`, note the TRAILING SLASH --
  un-slashed 307-redirects, same trap the 4000.1 baseline hit on `/api/settings`).
  Body = `AnalysisRequest`; the handler reads only `req.ticker` and uppercases it
  (`:353`). Response `AnalysisResponse{analysis_id, ticker, status}` with
  `status=PENDING` (`:381`). Non-Celery mode mints a `uuid4` task id and fires an
  `asyncio.create_task` (`:360`, `:375-378`) -- fire-and-forget, in-memory `_tasks`
  dict, so a backend restart LOSES the task.
- **Poll:** `GET /api/analysis/{analysis_id}` (`:384`). 404 if unknown (`:394`).
  Returns `AnalysisStatusResponse{analysis_id, ticker, status, current_step,
  steps_completed, message, step_log[], report, error}`.
- **Terminal states:** `AnalysisStatus.COMPLETED` (`:399`, `:331`) and
  `AnalysisStatus.FAILED` (`:343`, `:372`). Non-terminal: `PENDING`, `RUNNING`.
  So the poll predicate is `status in {COMPLETED, FAILED}`, NOT a `done` boolean.
- **Degradation shape for E4:** on COMPLETED the handler parses
  `task["result"]["final_synthesis"]` into `SynthesisReport`; a parse failure logs
  `"Could not parse synthesis into SynthesisReport model"` and returns
  `report=None` WITHOUT changing status (`:401-404`). **A COMPLETED-with-null-report
  is therefore a silent-degradation shape the E4 check must treat as FAIL.**

### (ii) DEV_LOCALHOST_BYPASS covers POST and PUT (method-agnostic)

`backend/api/auth.py:150-153`: the bypass is inside `get_current_user`, keyed
ONLY on `os.getenv("DEV_LOCALHOST_BYPASS") == "1"` AND
`request.client.host in ("127.0.0.1", "::1", "localhost")`. **No method check
exists anywhere in the function** -- it returns `{"email": "dev@localhost",
"localhost_bypass": True}` before any token parsing. So a localhost `curl -X PUT
/api/settings/` needs no NextAuth token, same as GET. Corroborated by
`.claude/rules/security.md` ("Localhost tooling ... relies on the
`DEV_LOCALHOST_BYPASS=1` + client-is-127.0.0.1 rail ... both conditions required").
**Two caveats the smoke must handle:** (1) the env var must be set on the RUNNING
backend process, not on the smoke's shell -- verify by probing, do not assume;
(2) Starlette `TestClient` is NOT covered (client.host is `testclient`) --
`backend/tests/auth_helper.py:8`, `backend/tests/api/test_sovereign.py:29`. The
unit tests therefore cannot rely on the bypass; they must stub the endpoints.

### (iii) Rail-calls-per-analysis -- BOUNDED, and the bound BREAKS the <=30 cap

This is the highest-value finding of this gate. Two facts compose:

1. **The pipeline's model pins are Claude TODAY.** Measured live 2026-08-06 from
   `GET /api/settings/` on the running backend:
   `gemini_model = "claude-sonnet-4-6"`, `deep_think_model = "claude-opus-4-8"`,
   plus `macro_regime_model / meta_scorer_model / news_screen_model /
   pead_signal_model = "claude-haiku-4-5"`, `apply_model_to_all_agents = false`.
2. **The orchestrator builds its clients from those settings, NOT from
   `resolve_model()`.** `orchestrator.py:652-659`:
   `general_client = make_client(settings.gemini_model, ...)`,
   `deep_think_client` / `synthesis_client = make_client(deep_model_name, ...)`,
   `quant_exec_client = make_client(settings.gemini_model, ...)`. `make_client`
   routes any `claude-*` name to `ClaudeCodeClient` when the flag is on
   (`llm_client.py:2115-2126`). **`model_tiers._BUILD_TIER` (`model_tiers.py:96-144`)
   is Layer-2 only** -- its claude roles (`mas_main`, `mas_qa`, `mas_research`,
   `mas_communication`, `autoresearch_*`) reach production through
   `agent_definitions.py:130/182/230`, NOT through the analysis pipeline. Do not
   cite `_BUILD_TIER` as the analysis-path bound.

Bound (config-dependent, NOT static -- flipping either settings pin to a
`gemini-*` value drops the count to ~0):
`self.general_client` is referenced **24x**, `deep_think_client` **8x**,
`synthesis_client` **5x**, `quant_exec_client` **4x** in `orchestrator.py`
(static reference counts, an upper bound on distinct call sites, not a per-run
execution count -- loops and conditionals move the runtime number both ways).
Against `.claude/rules/backend-agents.md` ("Lite Mode: ~39 -> ~20 LLM calls"),
a FULL single-ticker analysis is ~39 LLM calls, of which the Gemini-locked ones
(RAG step 3 via `rag_model`, `orchestrator.py:607-618`; Search Grounding steps
4/5/9/10 via `_grounded_vertex`, `:717`) stay on Gemini and the rest hit the rail.

**Consequence for 4000.2:** ONE full-mode single-ticker analysis plausibly exceeds
the <=30-rail-call cap on its own. The cap is still correct as a safety rail, but
the script MUST (a) abort mid-analysis rather than post-hoc, and (b) the `--dry`
pass MUST record the observed per-analysis count before any live window -- exactly
what 4000.1(e) demanded. Recommend the contract state the cap is a SAFETY CEILING,
and that a cap-abort in `--live` is a legitimate, loudly-reported outcome, not a
test bug.

**Second-order problem the contract must solve:** the rail subprocesses are spawned
by the BACKEND process, not by the smoke script, so the counter CANNOT be an
in-process `subprocess.run` wrapper. It has to be an out-of-process observation
(llm_call_log rows under the 4000.1(c) rule, polled during the run). See (v).

### (iv) E2 spend metric -- `fetch_llm_spend`

`backend/services/observability/spend.py:194` --
`def fetch_llm_spend() -> tuple[float, float]` returning `(daily_usd,
monthly_usd)`. **It takes NO window arguments**: the window is hardcoded in SQL --
month-to-date `ts >= TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), MONTH)` (`:226`) with a
same-day split via `IF(DATE(ts) = CURRENT_DATE(), ...)` (`:216-219`). Exclusions
(`:227-230`): `AND ok` (failed calls excluded), `AND provider != 'claude-code'`,
`AND (agent IS NULL OR (agent != 'cc_rail' AND agent NOT LIKE 'cc_rail:%'))`.
Prices raw token columns in Python via `_price_llm_tokens` (`:237`), NOT by summing
a stored cost column -- consistent with the "never SUM `session_cost_usd`" rule.
Fail-open to `(0.0, 0.0)` on ANY exception (`:248-250`).
**Binding on the E2 design:** because there is no window arg, a "delta over the
smoke window" must be a BEFORE/AFTER difference of the `daily_usd` value (two
calls bracketing the window), and the fail-open makes `0.0` ambiguous between
"no metered spend" and "the query blew up". E2 must therefore ALSO assert the
in-window metered-complement ROW COUNT is 0 (4000.1(c) second clause) -- the row
count distinguishes the two, the dollar figure alone does not.

### (v) Reading rail-guard state for E5 from OUTSIDE the process

**There is no API endpoint, and this is the gap 4000.2 must design around.**
`rail_guard_status()` (`claude_code_client.py:137-147`) reads
module-global `_RAIL_GUARD` under a lock -- **process-local state**. Its ONLY
production reader is `backend/services/autonomous_loop.py:1739-1747`, which
copies `rail_skipped` / `breaker_tripped` / `skipped_calls` into the cycle summary
and then into `cycle_health.py:302-329`'s cycle-history row. That path fires on the
autonomous CYCLE, which the smoke deliberately does not run (4000.1(e) rejected
the lite path). Three consequences:
- An out-of-process reader of `consecutive_failures` **does not exist**. The
  smoke cannot assert `consecutive_failures == 0` by observation.
- Surrogates that DO cross the process boundary: (1) `llm_call_log` rows with
  `ok = false` in-window under the 4000.1(c) rule -- a real-failure proxy;
  (2) the `rail_guard_skipped: ...` marker string that `claude_code_client.py:741`
  puts in `LLMResponse.thoughts` when blocked; (3) the P1 page raised on the
  breaker transition (`:194-210`, source `claude_code_rail`, error_type
  `breaker_open`) -- absence of a page in the window is weak evidence.
- **Recommend the contract restate E5 as a measurable predicate** (zero in-window
  `ok=false` rail rows + zero `rail_guard_skipped` markers + no `breaker_open`
  page) and record the direct-`consecutive_failures` read as a known gap, OR
  queue "expose rail_guard_status on an observability endpoint" as its own step.
  Silently asserting an unreadable value would be a fabricated-SAFE check.

### (vi) Where a completed analysis persists (E4 evidence)

Two surfaces, both needed:
- **In-memory:** `_tasks[task_id]["result"]` (`analysis.py:331-335`), served by the
  poll endpoint. Lost on restart.
- **Durable:** `bq.save_report(...)` at `analysis.py:210` via
  `BigQueryClient` (imported `:46`, constructed `:75`) into the
  `analysis_results` table (88-column ML schema per
  `.claude/rules/backend-api.md`). **The save is wrapped in a bare
  `try/except` that only logs** (`"Failed to save report to BigQuery: {e}"`,
  `analysis.py:~312`) -- so a COMPLETED status does NOT prove persistence. E4's
  "persisted analysis row exists" leg must query the table, not infer it from the
  poll response.

### (vii) `scripts/qa/` prior art + conventions

Existing: `ascii_logger_check.py`, `check_optimizer_best_provenance.py`,
`coverage_tier_check.py`, `env_syntax_check.py`, `sweep_absent_verification_paths.py`,
`sweep_ascii_logger{,_v2,_v3}.py`, plus two bash verifiers
(`verify_phase_4000_1_baseline.sh`, `verify_qa_roster_live.sh`). Style to mirror:
executable bit + shebang on the CLI-facing ones, module docstring naming the
phase, non-zero exit on failure. **No prior `scripts/qa/*.py` drives a live HTTP
backend or a subprocess-stubbed binary** -- 4000.2's smoke is a new shape here.
In-repo prior art for the subprocess+stub pattern lives in
`backend/tests/test_phase_76_9_2_max_bridge.py` (caller-flagged) and
`backend/tests/test_phase_66_1_rail_guard.py` (guard-state isolation fixture,
`_isolated_guard`, `:53-55` calling `rail_guard_reset`).

**Correction to the "no prior art" line above -- a SECOND smoke tree exists:**
`scripts/smoketest/` (`phase6_e2e.py`, `intel_e2e.py`, `rainbow_rehearsal.py`,
`aggregate.sh`, `steps/`). `scripts/smoketest/phase6_e2e.py:208` already calls
`flush_llm()` -- the closest existing analogue to 4000.2's script. Mirror its
conventions, and decide deliberately whether `smoke_cc_rail_e2e.py` belongs in
`scripts/qa/` (as the step name mandates) or `scripts/smoketest/`; the step name
is binding, so `scripts/qa/` it is, but the contract should note the split so a
future reader is not surprised.
`test_phase_76_9_2_max_bridge.py:1-45` is the exact template for this step:
it runs the REAL script as a subprocess against a **stub upstream** using
`http.server.ThreadingHTTPServer` + `threading` + `stat` (chmod) + a
`ANTHROPIC_BRIDGE_UPSTREAM` env override, and loads the script under test via
`importlib.util.spec_from_file_location` (`:39-42`) -- the idiom for importing a
`scripts/` file that is not a package.

### (viii) Three measured constraints not in the spawn prompt (all binding)

**(viii-a) `CLAUDE_CODE_BINARY` is NOT a reliable stub hook -- the spawn
prompt's premise is only half true.** `claude_code_client.py:57-76`:

```python
def _resolve_claude_binary(binary: str) -> str:
    if binary and (os.path.isabs(binary) and os.path.isfile(binary)):
        return binary
    resolved = shutil.which(binary)          # <-- PATH WINS HERE
    if resolved:
        return resolved
    for candidate in _DEFAULT_SEARCH_PATHS:  # <-- CLAUDE_CODE_BINARY only reached if PATH missed
        if candidate and os.path.isfile(candidate):
            return candidate
    return binary
```

Two independent defects for a test author:
1. `shutil.which("claude")` at `:70` is consulted **before** `_DEFAULT_SEARCH_PATHS`
   (`:73`), and the operator's real binary IS on PATH (`/Users/ford/.local/bin/claude`,
   measured). So setting `CLAUDE_CODE_BINARY` to a stub does **nothing** -- the test
   would silently invoke the REAL CLI, i.e. a live rail call from a "unit" test.
2. `_DEFAULT_SEARCH_PATHS` is a **module-level list** (`:49-54`) whose first element
   is `os.environ.get("CLAUDE_CODE_BINARY")` evaluated **at import time**. A
   `monkeypatch.setenv("CLAUDE_CODE_BINARY", ...)` inside a test therefore cannot
   affect it at all unless the module is re-imported.

**Correct stub mechanism** = the pytest-documented one (source 1): write an
executable file literally named `claude` into `tmp_path`, `chmod 0o755`, then
`monkeypatch.setenv("PATH", str(tmp_path), prepend=os.pathsep)`. `shutil.which`
then resolves the stub first, and the resolution logic is exercised **as imported**
(satisfying the step criterion "imported, not reimplemented"). Recommend the test
ALSO assert the negative: with only `CLAUDE_CODE_BINARY` set and no PATH prepend,
resolution does NOT return the stub -- that pins the finding so a future refactor
cannot silently reintroduce a live-call path.

**Production argv the stub must parse** (`claude_code_client.py:373-395`):
`[<resolved>, "--print", "--output-format", "json", "--disallowedTools", <csv>]`
plus optional `--append-system-prompt`, `--json-schema`, `--model`. **The prompt
arrives on STDIN, never as argv** (`:404`, `input=prompt`; documented at `:365-372`
as a phase-cycle-4 bugfix). The env is scrubbed of `ANTHROPIC_API_KEY` /
`ANTHROPIC_AUTH_TOKEN` only (`:405-411`), so `PATH` propagates to the child.
`--max-tokens` is explicitly NOT sent (`:397-403`: it is an SDK option, not a CLI
flag; ~63% of calls once failed with `unknown option '--max-tokens'`).

**(viii-b) `llm_call_log` rows are BUFFERED, not written per call.**
`api_call_log.py:221` -- *"Buffer a llm_call_log row. Never raises."* Flush is
piggybacked on the NEXT `log_llm_call` and fires only when
`len(_llm_buffer) >= _FLUSH_ROWS` (100) or `>= _FLUSH_SECONDS` (60) have elapsed
(`:38-39`, `:299-304`); `flush_llm()` (`:309`) then does one
`client.insert_rows_json` (`:345`). **There is no timer thread and no HTTP
endpoint that forces a flush** (only callers: the two test files and
`scripts/smoketest/phase6_e2e.py:208`). Consequences the contract must absorb:
- The **tail rows of the smoke window can sit in the backend's memory
  indefinitely** after the analysis ends -- E1/E2/E3 querying BQ immediately would
  read a truncated window and could report a false "0 metered rows" pass.
- Any BQ-derived rail-call counter has >=60s lag, so it **cannot** be the
  mechanism for a mid-analysis `<=30` abort.
- `flush_llm()` also hard-returns 0 when `PYFINAGENT_TEST_NO_BQ=1` (`:322`), which
  `backend/tests/conftest.py:21` sets suite-wide -- so the unit tests are already
  BQ-dark by default. Good: assert it rather than re-set it.

**(viii-c) `PUT /api/settings/` writes `backend/.env` on disk.**
`settings_api.py:402-465`: body is a **partial** `SettingsUpdate`
(`model_dump(exclude_none=True)`, `:418`), each field mapped through
`_FIELD_TO_ENV` and written by `_update_env_var` (`:437-448`), booleans
lower-cased (`:441`), then `get_settings.cache_clear()` +
`get_api_cache().invalidate("settings:*")` (`:451-452`). So the minimal flip is
`{"paper_use_claude_code_route": true}` and the minimal restore is the same key
with the pre-captured value. **Because the write is durable to `.env`, a crashed
`--live` run leaves the flag flipped across a backend restart** -- the
restore-in-finally is not a nicety, it is the only thing standing between a crash
and a permanently mutated operator config. The `finally` must issue a PUT, not
merely reset in-memory state, and should re-GET to confirm (the GET is cached, but
the PUT invalidates `settings:*`, so a post-PUT GET is fresh).

---

## Recency scan (2024-2026) -- PERFORMED

Searched the 2026 frontier and the 2025 window (queries listed above). **Three new
findings, all complementary rather than superseding:**

1. **(2026) "run the real installed binary" is the current integration-test
   consensus**, not in-process runners: a 2026 production suite states the rule as
   run the real binary as a subprocess *"exactly how an end user invokes it -- NOT
   through in-process methods like CliRunner"*, because the outermost binary covers
   entry point, arg parsing and process startup/teardown
   (github.com/sdebruyn/fabric-dw-mcp-cli issue #393). This **validates** 4000.2's
   `subprocess.run(...).returncode` shape over an in-process click/typer runner, and
   it is the same reason source 2 (pytest-subprocess) is the wrong tool here.
2. **(2026) child-env hardening**: same source recommends running the child with
   `PYTHONWARNINGS=error::ResourceWarning` and `PYTHONDEVMODE=1` so leaked sockets /
   unclosed files become hard failures. Cheap to adopt for a script that opens an
   HTTP stub + subprocesses.
3. **(2026) Claude Code CLI additions since the phase-78 snapshot**:
   `--max-budget-usd`, `--max-turns`, `--effort` (now incl. `ultracode`),
   `--no-session-persistence`, `--tools`. Verified against the INSTALLED binary
   (`/Users/ford/.local/bin/claude`, `2.1.223 (Claude Code)`, `--help` grep,
   2026-08-06) -- `--bare`, `--effort`, `--max-budget-usd`, `--model`,
   `--no-session-persistence`, `--output-format` all present. **`--bare` is still
   opt-in, NOT the `-p` default** -- the W1 vendor-watch risk carried from 4000.1 has
   NOT materialised as of CLI 2.1.223. Everything else about the envelope shape,
   `modelUsage` semantics and Max-plan billing is unchanged from the 4000.1 brief and
   is NOT re-derived here.

No 2024-2026 work supersedes sources 1, 3, 4, 6 or 7 (all are living official docs
or a still-canonical 2015 result that the 2026 literature continues to cite).

---

## Key findings

1. **pytest-subprocess cannot be used for this step.** *"The plugin hooks on the
   `subprocess.Popen()`"* -- in-process only (source 2, 2026-08-06). 4000.2's tests
   spawn the script as a child, so the child's own `subprocess.run` is invisible to
   the fixture. Use a real stub binary on PATH instead.
2. **The pytest-documented stub-binary mechanism is a PATH prepend**, not an env
   override: *"Use `monkeypatch.setenv("PATH", value, prepend=os.pathsep)` to modify
   `$PATH`"* (source 1). This is also the only mechanism that beats
   `_resolve_claude_binary`'s `shutil.which` (finding viii-a).
3. **Stub the backend with a real listening socket.** pytest-httpserver *"will
   listen on localhost on a random available port"* (source 3) -- reachable from the
   child process, unlike `requests-mock`/`responses` which patch in-process.
   `assert_request_made(..., count=0)` is the direct proof-obligation for the
   step's "zero PUT requests during a dry run" criterion. Set `threaded=True`.
4. **Crash-safe restore is a `try/finally` around a `yield`,** per the CPython docs
   (source 4); a CM that catches-and-logs must re-raise or it silently marks the
   exception handled. For multiple resources (stub server + flag + spawned analysis)
   `ExitStack` gives reverse-order unwinding.
5. **Poll with a deadline + jittered backoff.** AIP-151 supplies the async-task
   contract (poll until done; ~10s+ tasks return a promise) but *no* interval
   guidance (source 7); AWS supplies the interval: Full Jitter
   `sleep = random(0, min(cap, base * 2**attempt))`, and no-jitter is *"the clear
   loser"* (source 6). For a single-client smoke, contention is nil, so the
   load-bearing part is the **overall deadline/budget**, not the jitter.
6. **Cap the probe with `--max-budget-usd` and `--max-turns`** (source 5) -- both
   are print-mode flags that make the preflight probe bounded by construction rather
   than by convention. Neither is currently passed by `claude_code_invoke`.

## Consensus vs debate (external)

- **Consensus:** real-binary/real-socket integration testing over in-process
  monkeypatching (sources 1, 3 + the 2026 recency finding); `try/finally` cleanup
  (source 4); jittered exponential backoff (source 6).
- **Debate / genuine tension:** *in-process mocking vs real subprocess*.
  pytest-subprocess exists precisely because real subprocesses are slow and
  flaky, and its docs make no cross-process caveat -- a reader could easily adopt
  it here and get a test that silently calls the real `claude`. The 2026 source
  takes the opposite line. **Resolution for this step is forced, not chosen:** the
  code under test is a *separate process*, so only the real-stub side is even
  available. Record the reasoning so the next author does not re-litigate it.
- **Gap, not debate:** neither AIP-151 nor the AWS piece addresses *aborting* a
  long-running remote operation mid-flight. Nothing in the external literature
  tells us how to stop a running backend analysis; `analysis.py` exposes no cancel
  route. See the pitfalls.

## Pitfalls (from literature + measurement)

- **Silent live calls from a "unit" test** -- the single biggest risk here
  (finding viii-a). Mitigation: PATH-prepend + an explicit negative assertion.
- **False-green from a truncated BQ window** (finding viii-b): buffered rows make
  "0 metered rows" indistinguishable from "not flushed yet". Mitigation: assert a
  positive control (N>0 rail rows) before trusting any zero, per the E1 "N-of-N
  under the stated rule" shape from 4000.1(f).
- **Fail-open masquerading as a pass**: `fetch_llm_spend` returns `(0.0, 0.0)` on
  ANY exception (`spend.py:248-250`) and `log_llm_call` swallows all errors. A
  zero from either is not evidence.
- **pytest-httpserver assertion swallowing**: handler-thread assertion failures
  do not fail the test unless `check_assertions()` / `check()` is called (source 3).
- **A COMPLETED analysis with `report=None`** is a real degradation shape
  (`analysis.py:401-404`) that the naive `status == COMPLETED` check would pass.
- **No cancel path**: there is no endpoint to stop a running analysis, so a
  "mid-analysis abort" can only abort the SMOKE, not the backend's work. The rail
  calls already in flight will continue. The contract must say this plainly rather
  than imply the cap can stop the spend.

## Application to pyfinagent (external finding -> internal anchor)

| Finding | Anchor | Action for 4000.2 |
|---|---|---|
| PATH-prepend stub (source 1) | `claude_code_client.py:57-76`, `:49-54` | stub named `claude` in `tmp_path`, `chmod 0o755`, `monkeypatch.setenv("PATH", ..., prepend=os.pathsep)`; assert `CLAUDE_CODE_BINARY`-only does NOT work |
| stub must read STDIN | `claude_code_client.py:373-395`, `:404` | stub reads `sys.stdin`, ignores argv order, emits a canned envelope incl. a duplicate-`canonicalModel` `modelUsage` map (4000.1 D3 / step 4000.7) |
| real listening socket (source 3) | `analysis.py:350/384`, `settings_api.py:389/402` | stub `POST /api/analysis/`, `GET /api/analysis/{id}`, `GET|PUT /api/settings/`; `assert_request_made(PUT, count=0)` for `--dry` |
| `try/finally` + `ExitStack` (source 4) | `settings_api.py:437-452` (writes `.env`) | restore via PUT inside `finally`; `--keep-on` cancels the restore via `stack.pop_all()` |
| deadline + jitter (sources 6, 7) | `analysis.py:396-416` terminal states | poll `status in {COMPLETED, FAILED}`; Full-Jitter sleep capped; hard overall deadline; treat deadline-hit as FAIL |
| `--max-budget-usd` / `--max-turns` (source 5) | `claude_code_invoke` argv `:373-395` | use for the preflight probe only; do NOT change production routing (non-scope) |
| rail-call bound is config-dependent | live `gemini_model=claude-sonnet-4-6`, `deep_think_model=claude-opus-4-8`; `orchestrator.py:652-659` | contract states the cap is a SAFETY CEILING likely to bind on a full analysis; `--dry` records the observed count first |
| E5 has no out-of-process surface | `claude_code_client.py:137-147`; only reader `autonomous_loop.py:1739-1747` | restate E5 as measurable predicates, or queue an observability-endpoint step; do not assert an unreadable value |

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **7**
- [x] 10+ unique URLs total (incl. snippet-only) -- **19**
- [x] Recency scan (last 2 years) performed + reported -- 3 findings, incl. a live
      `claude --help` re-verification on CLI 2.1.223
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered all seven requested modules ((i)-(vii)) plus three
      unrequested constraints ((viii-a/b/c))
- [x] Contradictions / consensus noted (in-process mock vs real subprocess; the
      spawn prompt's `CLAUDE_CODE_BINARY` premise corrected)
- [~] **Known gap, stated honestly:** the E5 `consecutive_failures` value has no
      out-of-process reader. This brief proposes surrogates but does not resolve it;
      4000.2's contract must choose (surrogate predicate vs a queued endpoint step).
- [x] All claims cited per-claim

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 12,
  "urls_collected": 19,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Two spawn-prompt premises corrected. (1) CLAUDE_CODE_BINARY is NOT a usable stub hook: _resolve_claude_binary consults shutil.which BEFORE _DEFAULT_SEARCH_PATHS, and that list captures the env var at IMPORT time -- a monkeypatched value can never win while a real claude is on PATH, so the naive stub would fire live rail calls from a unit test. Use pytest's documented PATH-prepend. (2) pytest-subprocess cannot be used at all (hooks Popen in-process; the script under test is a child). Use pytest-httpserver (real localhost socket, assert_request_made count=0 proves the dry-run no-PUT criterion). Measured constraints: llm_call_log is BUFFERED (flush at 100 rows / 60s, piggybacked, no endpoint) so BQ cannot drive a mid-analysis abort and a zero-row read is not evidence; PUT /api/settings writes .env durably, making restore-in-finally load-bearing across restarts; live pins are gemini_model=claude-sonnet-4-6 + deep_think_model=claude-opus-4-8, so one full analysis plausibly exceeds the <=30 cap; and rail-guard state is process-local with no out-of-process reader (E5 gap).",
  "brief_path": "handoff/current/research_brief_4000.2.md",
  "gate_passed": true
}
```
