# Research Brief — phase-66.1: cc_rail pre-cycle health-probe gate + circuit breaker + single P1 page

Tier: MODERATE. Status: COMPLETE — gate_passed: true.
Date: 2026-07-07. Author: researcher (Layer-3).

## Question

cc_rail (`backend/agents/claude_code_client.py`) failed 100% 2026-06-15 -> 07-06 (~162 doomed calls/cycle, 2,400+ total, zero pages). Step 66.1 adds: (1) pre-cycle probe gate using `claude_code_health_probe()`; (2) per-cycle circuit breaker (default threshold <=20 consecutive failures) with exactly ONE deduped P1 via bot-token path; (3) degraded-mode = hold (documented), Gemini fallback config-gated default OFF.
Immutable test: `python -m pytest backend/tests/test_phase_66_1_rail_guard.py -q`

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/agents/claude_code_client.py` | 453 total; probe :236-281, error :76, invoke :80-233, adapter :313-441, log writer :340-365 | cc_rail client (subprocess `claude --print`) | live; probe UNGATED |
| `backend/services/autonomous_loop.py` | run_daily_cycle :124; probe block :209-234; meta_scorer :725-760; fan-out :777-904; degraded guards :906-975; record_cycle_end :1393; _run_single_analysis :1482 | daily 18:00 UTC cycle | live; 4 BROKEN alert imports |
| `backend/agents/llm_client.py` | make_client :1908; cc-route gate :1963-1977; routing-breach ValueError :1987-1996 | model->client routing choke point | live |
| `backend/agents/orchestrator.py` | clients :592-599; _generate_with_retry :755-862 | full pipeline per ticker | live |
| `backend/agents/debate.py` / `risk_debate.py` | _generate_with_retry :58 / :53 (max_retries=3) | debate legs' own retry helpers | live |
| `backend/services/observability/alerting.py` | 289 total; deduper :55-98; _CRITICAL_SEVERITIES :46; _bot_token_fallback :123-163; raise_cron_alert :166-237; sync :240-274 | THE paging path (62.7) | live, proven |
| `backend/services/cycle_health.py` | record_cycle_start :264; record_cycle_end :292-330; heartbeat alarm :234-253 | cycle_history.jsonl writer | live |
| `backend/config/settings.py` | gemini_model :30; deep_think_model :31; cc-route flag :172; churn-fix flag :311; fallback_alarm_threshold; model_config :554 | flag conventions | live |
| `backend/config/model_tiers.py` | EFFORT_SUPPORTED_MODELS :206; MODEL_EFFORT_FALLBACK :262 | effort tables (NOT touched by 66.1) | live |
| `backend/tests/test_phase_60_4_observability.py` | :40-113 cc_rail writer tests | monkeypatch pattern to copy | live |
| `backend/tests/test_phase_62_2_operator_tokens.py` | :15-19 autouse isolate_paths fixture | test-isolation pattern | live |

## Internal audit findings

### 1. Call topology + the ~162 calls/cycle

Entry: APScheduler job id `paper_trading_daily` (registry `backend/services/cron_control.py:37`; job id defined `backend/api/paper_trading.py:38`) -> `run_daily_cycle()` at `backend/services/autonomous_loop.py:124`. Inside the cycle:

- **Conviction overlay**: `meta_score_candidates(candidates, ...)` at :725-728 (LLM-scored; on rail-down it degrades to the "conviction 10.00" no-LLM fallback, `_all_conviction_fallback` :1857, stamped `meta_scorer_degraded` :745).
- **Per-ticker fan-out**: `analyze_tickers = new_candidates[:settings.paper_analyze_top_n]` :777; `summary["new_to_analyze"]` :814; concurrency-capped `_run_and_persist_one` :847 dispatched via two `asyncio.gather` calls :893-904 (new + reeval). Every analysis flows through `_run_single_analysis` :1482: full orchestrator (`AnalysisOrchestrator(settings)` :1538, `run_full_analysis` :1539) -> on failure falls back to lite Claude analyzer :1582-1593.
- **Which calls are cc_rail**: `settings.gemini_model` DEFAULTS to `"claude-sonnet-4-6"` (settings.py:30) -> orchestrator's `general_client` :592 and `quant_exec_client` :599 are claude-*; `deep_think_model` defaults `gemini-2.5-pro` (settings.py:31) -> Moderator/Critic/Synthesis on Gemini. `make_client` (llm_client.py:1963-1972) routes ANY `claude-*` model to `ClaudeCodeClient` when `paper_use_claude_code_route=True` (settings.py:172). So per ticker: ~10 enrichment + debate legs + 2 quant + deep-dive Q&A on the cc rail (~20-30 claude-routed calls/ticker full path), plus the lite fallback (also claude-sonnet -> cc rail) and meta_scorer calls. ~162/cycle = this per-ticker fan-out times analyzed+reeval tickers; the exact decomposition is queryable from `llm_call_log` rows `agent LIKE 'cc_rail:%'` with `ok=false` (writer at claude_code_client.py:340-365, phase-60.4).
- **The SINGLE choke point**: `claude_code_invoke()` (claude_code_client.py:80) is the one function every cc_rail call passes through (orchestrator clients, lite analyzer, meta_scorer alike). A module-level rail-state flag checked at its top (set by the pre-cycle probe, reset per cycle) gates ALL cc calls in one place. Loop-level precedent for module cycle-state already exists: `_current_cycle_id` module state set/reset at autonomous_loop.py:203-206. Secondary (policy-level) gate: the probe block :209-234 already runs FIRST in the cycle -- extend it to set the skip flag; per-ticker skip belongs in `_run_and_persist_one` :857 or `_run_single_analysis` :1482 (both new+reeval pass through them).

### 2. Retry logic + where the failure counter lives

- `orchestrator._generate_with_retry` :755-862: `max_retries=3`, `delay=5` doubling backoff, per-step timeout resolved via `_resolve_step_timeout` (phase-60.1 lifts to `recommended_step_timeout=150` for the cc rail, claude_code_client.py:326-333). Retries fire only on TimeoutError (:841-845) and transient exceptions (:846-862). `debate.py:58` and `risk_debate.py:53` have their OWN `_generate_with_retry(max_retries=3)` copies.
- **Load-bearing subtlety**: `ClaudeCodeClient.generate_content` CATCHES `ClaudeCodeError` (raised claude_code_client.py:76, thrown :179/:187/:197/:204/:213/:223) and returns an EMPTY `LLMResponse` (:399-413) -- failures propagate as empty text, NOT exceptions. So retry helpers do NOT retry cc failures (1 doomed subprocess per call site); the ~162 is fan-out width, not retry amplification. It also means a breaker cannot count exceptions upstream -- the natural counter home is INSIDE claude_code_client.py, either in the `except ClaudeCodeError` arm of `generate_content` (:399) or in `claude_code_invoke` itself, next to the existing `_log_cc_call(ok=False)` failure bookkeeping (:404-408). A module-level counter + `reset` called from the cycle-start probe block is per-cycle in effect (one process, one cycle at a time; cycle_lock serializes runs).

### 3. Paging path (phase-62.5/62.7 bot-token, dedupe) -- AND THE ZERO-PAGES ROOT CAUSE

- **Canonical in-process path**: `backend/services/observability/alerting.py` -- async `raise_cron_alert(source, error_type, severity, title, details)` :166, sync wrapper `raise_cron_alert_sync` :240. P1 is in `_CRITICAL_SEVERITIES` :46 (phase-62.7 R-1): page-class severities BYPASS the 3-in-5-min consecutive threshold AND the 1h repeat window (`should_fire` returns True unconditionally :75-80). With `slack_webhook_url` empty (this machine), critical severities route to `_bot_token_fallback` :123-163 -> `chat.postMessage` with `settings.slack_bot_token` (SecretStr unwrapped via `.get_secret_value()` :136-137), channel default `C0ANTGNNK8D` :138, `urllib` + `asyncio.to_thread`, fail-open bool return. This is the 62.5-drill-proven delivery leg.
- **Exactly-once semantics**: because P1 bypasses ALL dedupe (every P1 call fires a page), exactly-once-per-cycle MUST be enforced by the CALLER -- i.e., the breaker latches open on the counting transition (fail_count == threshold) and calls `raise_cron_alert` exactly once; once open it stops calling (and stops spawning subprocesses). Do NOT rely on the deduper for P1 rate-limiting.
- **ROOT CAUSE FINDING (explains "zero pages")**: `backend/services/alerting.py` DOES NOT EXIST (`ls`: No such file or directory; `backend/services/__init__.py` is empty -- no shim). The real module is `backend/services/observability/alerting.py`. Yet FOUR in-cycle alarm sites import the nonexistent path `from backend.services.alerting import raise_cron_alert`:
  - autonomous_loop.py:220 -- phase-56.2 rail health-probe P1 (`rail_down`)
  - autonomous_loop.py:751 -- conviction-overlay-degraded P1
  - autonomous_loop.py:923 -- degraded-scoring P1 (56.2 F-5 guard)
  - autonomous_loop.py:957 -- fallback-rate P1 (60.1 AW-4 alarm)
  Each import sits INSIDE a fail-open `try/except` (:233-234, :935-936, ...) so the `ModuleNotFoundError` is swallowed as "guard errored (non-fatal)". The probe itself HAS been wired into the cycle since 56.2 (:215-216 -- contra the step framing "unused in the cycle path": it runs and stamps `summary["claude_rail_healthy"]` :217, but (a) its result gates nothing, and (b) its page is dead code). The two sites using the CORRECT path (autonomous_loop.py:1415/:1438, plus cycle_health.py:109/:234, kill_switch.py:169/:348, drawdown_alarm.py:112, slack_bot/scheduler.py:48) all say `backend.services.observability.alerting`. 66.1 must fix these four imports (or route new gate/breaker alerts through the correct path) -- otherwise the new P1 would be dead on arrival too. A regression test asserting `importlib.util.find_spec` or simply importing the module in the test is cheap insurance.

### 4. Config conventions (default-OFF flags, effort tables)

- Naming/shape convention (settings.py): `paper_<feature>_enabled: bool = Field(False, description="phase-X (...): ..., DEFAULT OFF (do-no-harm)... Promotion to ON is an OPERATOR decision recorded in live_check_X.md ...")` -- exemplars `paper_swap_churn_fix_enabled` :311 and `paper_data_integrity_enabled` :42. Numeric knobs use bounded Fields (`paper_swap_max_per_cycle: int = Field(2, ge=0, le=10, ...)` :305). Env vars = UPPERCASED field name via pydantic-settings (`model_config = {"env_file": backend/.env, ...}` :554; no env_prefix). Suggested: `paper_cc_breaker_threshold: int = Field(20, ge=1, le=200, ...)` + `paper_cc_gemini_fallback_enabled: bool = Field(False, ...)`.
- Threshold-knob precedent that reads from settings with getattr-default: `fallback_alarm_threshold: float = Field(0.5, ...)` consumed via `float(getattr(settings, "fallback_alarm_threshold", 0.5))` at autonomous_loop.py:946.
- **Effort tables untouched**: `EFFORT_SUPPORTED_MODELS` (model_tiers.py:206) and `MODEL_EFFORT_FALLBACK` (:262) affect API-request effort params in SDK clients; `ClaudeCodeClient` never consumes effort (CLI rail, no effort flag). 66.1 gates call DISPATCH, not model/effort resolution -- no interplay, assert-only in tests if desired.
- Gemini fallback, if built: `make_client` (llm_client.py:1963) is where a config-gated rail-down fallback would route claude-* -> Gemini; note the existing routing-breach ValueError :1987-1996 intentionally hard-fails claude-*+cc-route to prevent silent direct-API billing -- any fallback must go to Gemini (free-tier Vertex), NEVER Anthropic-direct.

### 5. Test conventions (monkeypatch pattern, .env-bleed isolation)

- `test_phase_60_4_observability.py` (:26-113) is the closest template: builds a fake CLI `_envelope()` dict; `monkeypatch.setattr(ccc, "claude_code_invoke", lambda *a, **k: _envelope())` to fake the subprocess boundary; `monkeypatch.setattr(acl, "log_llm_call", ...)` to capture observability writes; `REPO_ROOT` sys.path insert at top; immutable `-k` selector documented in the module docstring with test names embedding the selector terms (66.1's immutable command is a direct file path, so name tests `test_phase_66_1_*` / embed `rail_guard`).
- `test_phase_62_4_sentinel.py` runs bash scripts via `subprocess.run(..., env=env)` -- relevant only if 66.1 probes real subprocess behavior; prefer monkeypatching `claude_code_invoke`/`claude_code_health_probe` instead (no real `claude` binary in CI path).
- **.env-bleed avoidance (phase-61.1 finding)**: never construct `get_settings()`-backed Settings in tests (reads live `backend/.env`); use the 62.2 autouse-fixture pattern (`test_phase_62_2_operator_tokens.py:15-19`): `@pytest.fixture(autouse=True) def isolate_paths(tmp_path, monkeypatch)` that monkeypatches module paths/state; for settings pass a `SimpleNamespace`/stub with only the fields under test, or `Settings(_env_file=None)`. Also reset module-global state (breaker counter, deduper via `reset_default_deduper()` alerting.py:277) in the autouse fixture -- the deduper and any rail-state flag are process-global.

### 6. Cycle summary state (cycle_history.jsonl)

- Writer: `CycleHealthLog` in cycle_health.py -- start row :264-290, terminal row `record_cycle_end` :292-330 writing `handoff/cycle_history.jsonl` (`_HISTORY_PATH` :36). Fields are a FIXED dict :309-323; phase-60.4 added `meta_scorer_degraded: bool` via a new kwarg (:301, :322) -- follow that exact precedent to add `rail_skipped` / `breaker_tripped` (kwargs default False so old callers unaffected). Called from run_daily_cycle's finally at autonomous_loop.py:1393.
- In-cycle `summary` dict already carries rail state: `summary["claude_rail_healthy"]` :217, `summary["degraded"]` :917, `summary["fallback_rate"]` :951 -- add `summary["rail_skipped"]`/`summary["breaker_tripped"]` beside them so 66.2's funnel diagnosis can read both the summary and the JSONL row. Live row shape verified from tail (2026-07-06 cycle e188ffaa: completed, 319s, n_trades=0, `meta_scorer_degraded: true` -- note: still degraded on 07-06 18:00 UTC, i.e. post-restore verification for 66.2).

## External research

### Search queries run (3 variants x 3 topics, per research-gate.md)

Topic 1 (circuit breaker canonical): bare `circuit breaker pattern half-open Nygard "Release It" Fowler threshold`; +2026 `circuit breaker pattern health probe microservices guidance 2026`; +2025 `circuit breaker LLM API client resilience 2025`.
Topic 2 (page-once / probe-before-batch): bare `alert deduplication single page per incident PagerDuty best practice`; +2026 `alert fatigue deduplication page once per incident 2026`; +2025 `fail fast health check before batch expensive API fan-out 2025`.
Topic 3 (Claude Code CLI reliability): bare `Claude Code CLI headless exit code error handling`; +2026 `claude CLI authentication failure exit code non-interactive 2026`; +2025 `claude --print output-format json error subtype 2025`.

### Read in full (>=5 required; counts toward the gate) — 7 fetched

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://martinfowler.com/bliki/CircuitBreaker.html | 2026-07-07 | authoritative blog (canonical) | WebFetch full | "Once the failures reach a certain threshold, the circuit breaker trips, and all further calls ... return with an error, without the protected call being made at all." "Any change in breaker state should be logged and breakers should reveal details of their state for deeper monitoring." Self-reset = retry protected call after interval. |
| https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker | 2026-07-07 | official doc (updated 2026-07-02) | WebFetch full | "The failure counter for the Closed state is time based. It automatically resets at periodic intervals ... The failure threshold triggers the Open state only when a specified number of failures occur during a specified interval" (= per-window reset). "Failed operations testing: In the Open state, rather than using a timer ... a circuit breaker can periodically ping the remote service ... or use a special health-check operation" (-> Health Endpoint Monitoring). "alert an administrator when a circuit breaker switches to the Open state." Types-of-exceptions: may "require a larger number of time-out exceptions to trigger ... compared to the number of failures caused by the unavailable service" (auth-down should trip faster than timeouts). Manual override recommended. |
| https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/circuit-breaker.html | 2026-07-07 | official doc | WebFetch full | Pattern "popularized by Michael Nygard in his book, Release It (Nygard 2018)". "Force open or close the circuit: System administrators should have the ability to open or close a circuit." "Multithreaded calls ... ensure that subsequent calls do not move the expiration timeout endlessly." Observability: "logging set up to identify the calls that fail when the circuit breaker is open." Circuit state persisted in a table with ExpiryTimeStamp (state-store precedent). |
| https://learn.microsoft.com/en-us/azure/architecture/patterns/health-endpoint-monitoring | 2026-07-07 | official doc | WebFetch full | Probe the REAL dependency path; "determine whether a 200 (OK) status code is sufficient ... Checking the status code is the minimum implementation"; check response CONTENT not just code; "Consider caching the endpoint status. Running the health check frequently might be expensive"; probes feed circuit breakers ("a circuit breaker can test the health of a service by sending a request to an endpoint that the service exposes"). |
| https://code.claude.com/docs/en/headless | 2026-07-07 | official doc (Anthropic) | WebFetch full | `-p/--print` non-interactive; `--output-format json` envelope carries `result`/`session_id`/`total_cost_usd`; stream-json `system/api_retry` events carry `error` category enum: `authentication_failed`, `oauth_org_not_allowed`, `billing_error`, `rate_limit`, `overloaded`, `invalid_request`, `model_not_found`, `server_error`, `max_output_tokens`, `unknown` -- the ONLY documented machine-readable auth-vs-network classification. `--bare` "skips OAuth and keychain reads. Anthropic authentication must come from ANTHROPIC_API_KEY" -- confirms the codebase's never-use---bare rule for the Max rail (claude_code_client.py:117-126) even though the doc now recommends --bare for scripts. Piped stdin capped 10MB (v2.1.128+, "exits with a clear error and a non-zero status"). |
| https://support.pagerduty.com/main/docs/event-management | 2026-07-07 | official doc | WebFetch full | "Subsequent alerts with a matching dedup_key deduplicate into the same incident" while unresolved -- one INCIDENT per underlying problem, repeats append to the alert log instead of re-paging. The caller-supplied stable key is the exactly-once mechanism. |
| https://portkey.ai/blog/retries-fallbacks-and-circuit-breakers-in-llm-apps/ | 2026-07-07 | industry (LLM gateway; already the codebase's failover canonical, claude_code_client.py:19) | WebFetch full | "Retries are designed for temporary glitches"; fallbacks are reactive (+latency); circuit breakers "automatically cut off traffic to unhealthy components before the rest of the system is affected"; breakers monitor "specific failure status codes (e.g., 429, 502, 503)"; retry storms: "At scale, this turns into a retry storm, stacking up requests." No concrete threshold numbers given (honest gap). |

### Identified but snippet-only (does NOT count toward gate; ~73 further unique URLs across 9 searches)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://microservices.io/patterns/reliability/circuit-breaker.html | authoritative (Richardson) | redundant with Fowler/Azure/AWS trio |
| https://docs.aws.amazon.com/builders-library/... (timeouts/retries) | official | retry-budget angle secondary to breaker focus |
| https://institute.sfeir.com/en/claude-code/claude-code-headless-mode-and-ci-cd/errors/ (+cheatsheet/FAQ) | community training | key claims captured in snippets: exit 0 success / non-zero error; "claude auth status exiting 0 when logged in and 1 when not"; "no complete enumerated exit-code table ... branch on zero versus non-zero"; "silent failure ... exit code 0 but the response is empty" |
| https://techcommunity.microsoft.com/.../improve-llm-backend-resiliency-with-load-balancer-and-circuit-breaker-rules... | official blog 2025 | gateway-level (APIM) -- confirms practice, wrong layer for a local CLI rail |
| https://markaicode.com/circuit-breakers-llm-api-reliability/ | blog 2025 | tertiary to Portkey |
| https://oneuptime.com/blog/post/2026-01-30-alert-deduplication/view (+2026-02-02, 2026-03-31) | vendor blog 2026 | "deduplication groups repeated firings into a single active incident so on-call engineers get one page, not hundreds" -- captured in snippet |
| https://incident.io/blog/sre-alerting-best-practices | vendor blog | PagerDuty official doc covers the mechanism |
| https://dev.to/waxell/ai-agent-circuit-breakers... / https://n1n.ai blog 2026-02 / https://ve3.global/... | blogs 2025-26 | LLM-breaker recency evidence; low marginal content |
| https://github.com/anthropics/claude-code/issues/46845 | GitHub issue | non-TTY first-run wizard exits 1 -- adjacent failure class, noted |
| https://support.claude.com/en/articles/14552646-troubleshoot-claude-code-installation-and-authentication | official support | headless doc supersedes for our need |
| https://code.claude.com/docs/en/errors | official doc | surfaced by search; headless page already covered classification |
| https://microservices.io/patterns/observability/health-check-api.html | authoritative | Azure health-endpoint pattern covers it |
| https://www.groundcover.com/learn/performance/circuit-breaker-pattern, https://layra4.dev/pattern/circuit-breaker, https://singhajit.com/circuit-breaker-pattern/, https://amquesteducation.com/blog/circuit-breaker-pattern/, others | blogs | consensus repetition of the canonical trio |

### Recency scan (2024-2026)

Performed (the +2026 and +2025 query variants above). Findings: (1) 2025-2026 produced a dense LLM-SPECIFIC circuit-breaker literature (Portkey, Azure APIM LLM circuit-breaker rules 2025, markaicode 2025, n1n.ai 2026-02, dev.to/VE3 2025-26) -- the pattern is now standard production practice for LLM rails; new nuance vs the 2007/2014 canon: LLM breakers must also treat QUALITY degradation (empty/hallucinated 200-OK responses) as failure signal, matching our empty-`LLMResponse` failure mode. (2) Alert-dedupe canon unchanged; 2026 posts (OneUptime Jan-Mar 2026) restate dedup_key/one-page-per-incident. (3) Claude Code docs added the `system/api_retry` error-category enum (incl. `authentication_failed`) and the 10MB stdin cap (v2.1.128), and now recommend `--bare` for scripts -- which this project must CONTINUE TO REJECT for the Max-OAuth rail (bare skips keychain OAuth). No finding supersedes the canonical breaker mechanics; the 2024-2026 window refines failure classification and placement (gateway vs client).

### Key findings (external)

1. Breaker = counter + threshold + fail-fast; trip => calls return immediately without invoking the dependency (Fowler; Azure). Distinct from retry: "The Circuit Breaker pattern prevents an application from performing an operation that's likely to fail" and retry logic "should be sensitive to any exceptions that the circuit breaker returns and stop" (Azure).
2. Per-window reset matters: the Closed-state failure counter is TIME-BASED and resets periodically so occasional failures don't trip it; threshold means N failures within an interval (Azure). In pyfinagent the natural window IS the cycle (per-cycle reset at probe time) -- one cycle/day makes burn-rate windows degenerate, consistent with the loop's own SRE note (autonomous_loop.py:943-944).
3. Probe-instead-of-timer is a sanctioned Open->recovery mechanic: "use a special health-check operation that the remote service provides" (Azure -> Health Endpoint Monitoring). `claude auth status` is exactly that: free, token-less, exercises the same OAuth path (claude_code_client.py:236-249; auth status exits 0 logged-in / 1 not, SFEIR). Probe content-check (loggedIn flag) matches "status code is the minimum implementation" guidance.
4. Alert on the TRANSITION, not the failures: "alert an administrator when a circuit breaker switches to the Open state" (Azure); "Any change in breaker state should be logged" (Fowler); PagerDuty dedup_key = one incident per problem while unresolved. => exactly-one P1 per cycle = fire on the closed->open transition only; the deduper cannot do this for P1 (bypass), so the breaker latch is the dedupe.
5. Failure classification: auth-down is non-transient -- Azure explicitly allows fewer failures to trip for hard-down vs timeouts; Claude Code's documented error categories (`authentication_failed` vs `rate_limit`/`server_error`) ground the split, though in plain `--output-format json` mode only exit-code + stderr + envelope subtype are available; do NOT hard-code specific non-zero exit values (undocumented; branch zero-vs-nonzero and read detail text).
6. Fallback discipline: fallbacks are reactive and add latency (Portkey); AWS/Azure both stress admin manual override + observability of calls failed-while-open. A default-OFF Gemini fallback flag matches Azure's "Open state can return a default value that's meaningful" only as an explicit, gated choice -- degraded-hold is the conservative default.

## Consensus vs debate

- Consensus (all 7 full-reads + snippets): 3-state breaker, threshold-within-window, fail-fast on open, health-probe-driven recovery, log/alert state transitions, one page per incident.
- Debate/nuance: (a) concrete thresholds -- literature refuses universal numbers ("five consecutive failures or 30% of calls" appears only as an example); per-dependency tuning to "match the likely recovery pattern" (Azure). The step's <=20 default is defensible: >20 consecutive failures within one cycle is unambiguous hard-down for a 162-call fan-out, while tolerating transient blips. (b) Half-open probing mid-cycle is canon for long-lived services; for a once-daily batch cycle the next cycle's pre-cycle probe IS the half-open trial -- simpler and adequate (supported by Azure's probe-instead-of-timer variant). (c) 2025-26 LLM literature pushes gateway-level breakers; not applicable to a local subprocess rail -- client-level is correct here.

## Pitfalls (from literature, mapped)

1. Retry x breaker interaction: upstream retries must stop when the breaker is open (Azure) -- satisfied structurally here because cc failures return empty responses (no retry), but a breaker that raises a NEW exception type could TRIGGER the 3x retry loops (orchestrator.py:853-862 catches generic Exception; "unavailable" substring marks it transient => 3x subprocess spawns). Breaker-open short-circuit must return the SAME empty-LLMResponse shape (or a non-transient-named exception), not a transient-looking one.
2. Alert-on-every-failure = 40 pages/incident (OneUptime/PagerDuty). P1 bypasses this project's deduper entirely (alerting.py:75-80) -- calling raise_cron_alert per failure would page ~162 times. Latch first, page once.
3. Probe passes but calls fail (silent 200-OK class; SFEIR "exit code 0 does not guarantee a usable response"): auth status can be ok while invoke fails (limits, model access). Hence probe gate AND in-cycle breaker are BOTH required -- neither alone suffices.
4. Timer-moving under concurrency (AWS): concurrent `_run_and_persist_one` tasks all increment the counter -- use a lock or atomic int; don't let each failure extend/reset windows.
5. Health check must be cheap/cached (Azure health-endpoint): probe once per cycle, not per call (15s subprocess).
6. Doc-vs-project divergence: headless doc now recommends `--bare`, which would BREAK the OAuth rail (skips keychain) -- keep the existing no---bare rule (claude_code_client.py:122-126).

## Application to pyfinagent (file:line mapping)

1. **Pre-cycle gate**: extend the EXISTING probe block autonomous_loop.py:209-234 -- on `not _rail_ok`: set `summary["rail_skipped"]=True`, set the rail-down latch (new module fn in claude_code_client.py, mirroring `_current_cycle_id` module-state precedent autonomous_loop.py:203-206), and page P1 via the CORRECT import `backend.services.observability.alerting.raise_cron_alert` (fix :220; also :751, :923, :957 -- `backend.services.alerting` is unimportable, proven via find_spec, and each is swallowed by its fail-open except).
2. **Breaker**: counter + latch inside claude_code_client.py beside `_log_cc_call(ok=False)` (:399-413 catch arm) or in `claude_code_invoke` (:80); threshold from `settings.paper_cc_breaker_threshold` (default 20, `Field(20, ge=1, ...)` per settings.py:305 bounded-Field + :311 flag-description conventions; consume via `getattr(settings, ..., 20)` per autonomous_loop.py:946). Open => skip subprocess, return the standard failure path; page exactly once on the tripping transition via bot-token P1 (alerting.py:123-163; webhook empty => `_bot_token_fallback` auto-selected :197-205). Reset the counter+latch in the cycle-start probe block (per-window reset, Azure).
3. **Degraded mode**: rail-down => analyses fail/fall back exactly as today (hold; no new trading behavior). Optional Gemini fallback would live at make_client llm_client.py:1963-1977 behind `paper_cc_gemini_fallback_enabled` default OFF; NEVER fall through to Anthropic-direct (routing-breach guard :1987-1996 stays).
4. **Funnel visibility for 66.2**: `summary["rail_skipped"]` / `summary["breaker_tripped"]` beside :217; persist via new kwargs on `record_cycle_end` (cycle_health.py:292-330) following the `meta_scorer_degraded` precedent (:301/:322); doomed-call forensics already queryable from llm_call_log `agent LIKE 'cc_rail:%' AND ok=false` (claude_code_client.py:340-365).
5. **Tests** (`backend/tests/test_phase_66_1_rail_guard.py`): monkeypatch `ccc.claude_code_invoke` / `claude_code_health_probe` / captured `raise_cron_alert` per test_phase_60_4_observability.py:40-113; autouse tmp_path/state-reset fixture per test_phase_62_2_operator_tokens.py:15-19 (+ `reset_default_deduper()` alerting.py:277 and breaker-state reset); never read the live `.env` -- stub settings objects (61.1 finding). Include a regression test that `importlib.util.find_spec("backend.services.observability.alerting")` is the import used by the gate (or simply import the gate module and assert the alert fn identity).

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7)
- [x] 10+ unique URLs total (~80 unique across 9 searches)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (client, loop, routing, alerting, cycle-history, settings, tiers, tests)
- [x] Contradictions / consensus noted (thresholds unprescribed; --bare doc-vs-project divergence)
- [x] All claims cited per-claim
- Note: tool-call usage exceeded the moderate-tier soft budget (~30 vs <=18) -- forced by the caller-mandated 3x3 search matrix + 6-question internal audit; source floor and depth honored.

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 73,
  "urls_collected": 80,
  "recency_scan_performed": true,
  "internal_files_inspected": 15,
  "report_md": "handoff/current/research_brief_66.1.md",
  "gate_passed": true
}
```
