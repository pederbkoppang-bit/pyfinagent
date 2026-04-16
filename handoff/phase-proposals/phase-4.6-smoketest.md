# Phase 4.6 Proposal: Full-Stack Smoketest (step-by-step E2E verification)

## Goal

Establish a reproducible, ASCII-only, one-command smoketest that verifies
every externally observable surface of pyfinAgent end-to-end before any
promotion to "trusted build" (the harness gate, a release tag, or a
production-style deploy). The smoketest MUST be:

1. Runnable from a clean checkout at `/Users/ford/.openclaw/workspace/pyfinagent`
   after `source .venv/bin/activate` with no additional setup.
2. Deterministic about what "pass" means: every step has explicit,
   machine-checkable success criteria plus an expected-output snippet.
3. Actionable on failure: every step lists a concrete recovery step (restart
   uvicorn, reseed fixtures, re-run one sub-command, file a ticket) so the
   next agent/operator does not have to re-derive what went wrong.
4. Cheap enough to run on every harness cycle: target wall-clock budget
   ~8 minutes against real APIs, ~3 minutes against VCR cassettes.

The smoketest is a black-box synthetic probe in the Google SRE sense -- it
exercises public interfaces and measures end-to-end behavior, rather than
replacing the white-box unit tests that already live in `tests/`
([Google SRE book, Ch. 6](https://sre.google/sre-book/monitoring-distributed-systems/);
[SRE School black-box monitoring guide](https://sreschool.com/blog/black-box-monitoring/)).

## Success criteria

- The new driver script `scripts/smoketest/run_smoketest.py` exits with code 0
  on a healthy stack and a non-zero code on any step failure, naming the
  failing step id (e.g. `4.6.4`).
- Each of the 10 ordered steps below has:
  - a `verification.command` that is a real shell one-liner,
  - a `success_criteria[]` list of at least two independent checks,
  - an `expected_output` fenced block the driver greps for,
  - a `recovery` action on failure.
- Coverage includes: backend `/api/health`, 3 MCP servers, the 12 enrichment
  signals for AAPL, `paper-trading/run-now` dry-run, frontend build + 5 paper
  trading tabs rendering, Slack digest delivery, and a simulated watchdog
  failure.
- The driver appends a one-line pass/fail row to `handoff/harness_log.md`
  using the existing cycle format, so the Harness tab on the backtest page
  shows smoketest history alongside other cycles.
- `python scripts/smoketest/run_smoketest.py --dry-run` prints the plan
  without executing, so the harness can cost-estimate before running it.
- The JSON snippet in this file validates as a valid `phase-4.6` block with
  the required masterplan schema keys.

## Step-by-step plan

Each step below is a black-box synthetic probe (Google SRE,
[Prober / blackbox_exporter](https://sre.google/sre-book/monitoring-distributed-systems/))
with golden-signal checks -- latency, correctness, errors.

### 4.6.0 -- Preflight: venv + dependency sanity

- Verify `.venv` exists, Python is 3.14, critical deps import.
- `verification.command`:
  `python -c "import sys,fastapi,httpx,google.cloud.bigquery,anthropic,yfinance; assert sys.version_info[:2]==(3,14); print('PREFLIGHT_OK')"`
- `success_criteria`:
  - stdout contains `PREFLIGHT_OK`
  - exit code 0
- `expected_output`: `PREFLIGHT_OK`
- `recovery`: `source .venv/bin/activate && pip install -r requirements.txt`

### 4.6.1 -- Backend boot + `/api/health` returns 200

Start uvicorn in the background on an ephemeral port (8765 to avoid clashing
with an already-running dev server), then poll `/api/health` with 30s
timeout. Pattern per
[FastAPI testing docs](https://fastapi.tiangolo.com/tutorial/testing/) and
[pytest-uvicorn background fixture recipe](https://www.pythontutorials.net/blog/how-to-start-a-uvicorn-fastapi-in-background-when-testing-with-pytest/).

- `verification.command`:
  `python scripts/smoketest/steps/boot_backend.py --port 8765 --timeout 30`
- `success_criteria`:
  - `curl -sf http://127.0.0.1:8765/api/health` returns JSON with
    `"status":"ok"` and `"version"` matching `v\d+\.\d+\.\d+`
  - response body contains `mcp_servers` with all three of `data`,
    `backtest`, `signals` marked `status:"ok"`
  - end-to-end latency < 5s (golden signal: latency)
- `expected_output`:
  ```
  {"status":"ok","service":"pyfinagent-backend"
  ```
- `recovery`: kill stale workers
  (`pkill -9 -f "uvicorn backend.main:app"`), inspect `backend.log`,
  re-run step.

### 4.6.2 -- MCP servers respond to ping

The three FastMCP servers (`data_server`, `backtest_server`,
`signals_server`) under `backend/agents/mcp_servers/` MUST respond to a
JSON-RPC `ping` within the MCP spec's contract
([fast.io MCP health-check pattern](https://fast.io/resources/implementing-mcp-server-health-checks/);
[MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector)).

- `verification.command`:
  `python scripts/smoketest/steps/mcp_ping.py --servers data,backtest,signals --timeout 10`
- `success_criteria`:
  - all three servers return an empty-result JSON-RPC response to `ping`
  - `list_tools` returns >=1 tool for each server (proves the handshake
    past ping)
- `expected_output`: `MCP_PING_OK data=ok backtest=ok signals=ok`
- `recovery`: restart the MCP subprocesses via
  `scripts/start_services.sh --mcp-only`; confirm no zombie children with
  `ps -ef | grep mcp_servers`.

### 4.6.3 -- 12 enrichment signals return for AAPL

Hit `GET /api/signals/AAPL` and assert all 12 documented signal keys are
present in the response payload. The 12 keys are drawn directly from
`backend/api/signals.py::get_all_signals`: `insider`, `options`,
`social_sentiment`, `patent`, `earnings_tone`, `fred_macro`, `alt_data`,
`sector`, `nlp_sentiment`, `anomalies`, `monte_carlo`, `quant_model`.

- `verification.command`:
  `curl -sf -m 60 "http://127.0.0.1:8765/api/signals/AAPL" | python -c "import sys,json; d=json.load(sys.stdin); keys=['insider','options','social_sentiment','patent','earnings_tone','fred_macro','alt_data','sector','nlp_sentiment','anomalies','monte_carlo','quant_model']; miss=[k for k in keys if k not in d]; print('SIGNALS_OK' if not miss else f'MISSING:{miss}'); sys.exit(0 if not miss else 1)"`
- `success_criteria`:
  - all 12 signal keys present in response
  - at least 8 of 12 signals return a non-ERROR payload (degraded-tolerant;
    individual `_safe` wrappers can return `signal:"ERROR"` on transient
    vendor failure)
  - total wall-clock < 60s
- `expected_output`: `SIGNALS_OK`
- `recovery`: inspect `backend.log` for per-signal errors, confirm
  `.env` has `ALPHAVANTAGE_API_KEY`, `FRED_API_KEY`, `API_NINJAS_KEY`;
  re-run with `LOG_LEVEL=DEBUG`.

### 4.6.4 -- Paper trading `/run-now` dry-run

POST to `/api/paper-trading/run-now` with a dry-run flag so no BQ writes
occur. Confirms the full autonomous-cycle code path (gate check, ticket
queue, portfolio manager, fill simulator) executes end-to-end without a
real market-moving write. This validates the "simulation calibration"
principle from
[Portfolio123 live-trading validation guide](https://community.portfolio123.com/uploads/short-url/3WHpAUOzhCG8QAUez71HpoWnA62.pdf)
and the CFA
[Investment Model Validation Guide](https://rpc.cfainstitute.org/sites/default/files/-/media/documents/article/rf-brief/investment-model-validation.pdf).

- `verification.command`:
  `curl -sf -m 120 -X POST "http://127.0.0.1:8765/api/paper-trading/run-now?dry_run=true" -H "Content-Type: application/json" -d '{}'`
- `success_criteria`:
  - HTTP 200 with JSON body containing `"status":"ok"` or `"started":true`
  - response includes a `cycle_id` or `run_id`
  - follow-up `GET /api/paper-trading/status` shows `last_run_ts` updated
    within last 120s
- `expected_output`: `"status":"ok"` or `"started":true`
- `recovery`: check kill-switch state (`/api/paper-trading/kill-switch`),
  resume if tripped (`/api/paper-trading/resume`); verify BQ creds via
  `bq ls sunny-might-477607-p8:pyfinagent_pms`.

### 4.6.5 -- Frontend build succeeds

Run `npm run build` in `frontend/` and confirm exit code 0 and no TypeScript
errors. Playwright would be heavier but build alone catches 90% of real
regressions ([Playwright + Next.js guide](https://nextjs.org/docs/pages/guides/testing/playwright);
[makerkit smoke testing guide](https://makerkit.dev/blog/tutorials/smoke-testing-saas-playwright)).

- `verification.command`:
  `cd /Users/ford/.openclaw/workspace/pyfinagent/frontend && npm run build 2>&1 | tail -30`
- `success_criteria`:
  - exit code 0
  - output contains `Compiled successfully` or `Generating static pages`
  - no `Type error:` lines in the last 200 lines
- `expected_output`: `Compiled successfully`
- `recovery`: run `npm ci` to rebuild node_modules from lockfile,
  re-run build with `--debug`; if a type error appears, grep
  `frontend/src/lib/types.ts` for the offending interface.

### 4.6.6 -- Frontend paper-trading 5-tab render check

Use Playwright in headless mode (or an HTTP-only check against the
server-rendered HTML) to walk the 5 tabs defined in
`frontend/src/app/paper-trading/page.tsx`: `positions`, `trades`, `chart`,
`reality-gap`, `exit-quality`. Each tab must render without a console
error and without the error boundary banner (rose-950/50 bg per
`frontend.md` rules).

- `verification.command`:
  `python scripts/smoketest/steps/frontend_tabs.py --base http://localhost:3000 --tabs positions,trades,chart,reality-gap,exit-quality`
- `success_criteria`:
  - all 5 tab routes return HTTP 200
  - page HTML contains the tab label text
  - no `TypeError` or `ReferenceError` in captured console logs
  - no element with class `border-rose-500` in the rendered DOM
- `expected_output`: `TABS_OK 5/5`
- `recovery`: check `frontend.log`, verify backend is reachable from the
  Next.js server (Bearer token, CORS), rebuild with
  `npm run build && npm run start`.

### 4.6.7 -- Slack digest posts to test channel

Trigger the Slack scheduler's `post_signal_digest` path against a dedicated
`SLACK_TEST_CHANNEL_ID` (falls back to `SLACK_CHANNEL_ID`). The Events API
auto-disables after 95% delivery failure in a 60-minute window
([Slack webhook best-practices guide](https://hookdeck.com/webhooks/platforms/guide-to-slack-webhooks-features-and-best-practices)),
so we MUST verify delivery not just 200-response.

- `verification.command`:
  `python -m backend.slack_bot.digest_test --channel-env SLACK_TEST_CHANNEL_ID --text "smoketest-4.6.7-$(date +%s)" --verify-delivery`
- `success_criteria`:
  - Slack returns `ok:true`
  - subsequent `conversations.history` call returns a message whose text
    matches the timestamp we sent (end-to-end delivery, not just accept)
  - elapsed < 10s
- `expected_output`: `SLACK_DELIVERED`
- `recovery`: verify `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN` in
  `backend/.env`, check `backend_slack.log` for socket-mode disconnects,
  re-run `python -m backend.slack_bot.app` in background.

### 4.6.8 -- Watchdog alert fires on simulated failure (chaos probe)

This is the Netflix / Chaos Monkey
([origin story](https://www.gremlin.com/chaos-monkey/the-origin-of-chaos-monkey);
[Gremlin chaos engineering principles](https://en.wikipedia.org/wiki/Chaos_engineering))
step: deliberately induce a failure (SIGKILL the paper-trading cycle
process) and assert that within 60s the SLA monitor / cycle_health service
logs an alert row. Validates that monitoring itself is not silently broken,
which is the bug class the Coralogix Netflix FIT post identifies as the
most common
([Coralogix: How Netflix uses fault injection](https://coralogix.com/blog/how-netflix-uses-fault-injection-to-truly-understand-their-resilience/)).

- `verification.command`:
  `python scripts/smoketest/steps/chaos_watchdog.py --timeout 90`
- `success_criteria`:
  - script finds a running `paper_trader` subprocess and SIGKILLs it
  - within 90s, `backend/services/sla_monitor.py` writes an alert row to
    `pyfinagent_data.harness_learning_log` or emits a WARN to
    `backend.log`
  - process is restarted by the supervisor (uvicorn reloader or
    `start_services.sh`) automatically
- `expected_output`: `WATCHDOG_FIRED`
- `recovery`: manually restart via `scripts/start_services.sh`; if alert
  did not fire, inspect `backend/services/sla_monitor.py` cadence config
  and ensure the heartbeat table is receiving writes.

### 4.6.9 -- Harness log row appended + exit cleanup

Final step: append a one-line row to `handoff/harness_log.md` matching the
existing cycle format, then shut down the uvicorn process started in 4.6.1.

- `verification.command`:
  `python scripts/smoketest/steps/finalize.py --log handoff/harness_log.md --port 8765`
- `success_criteria`:
  - `handoff/harness_log.md` grew by exactly one row
  - the new row includes `phase=4.6`, `result=PASS|FAIL`, `duration_s`,
    and a per-step status grid
  - no `uvicorn` process remains bound to port 8765 (clean shutdown)
  - `ps aux | grep mcp_server` shows zero extra PIDs vs. pre-test snapshot
- `expected_output`: `SMOKETEST_DONE result=PASS`
- `recovery`: if stray workers remain, run
  `pkill -9 -f "uvicorn.*8765"`; if the log was not appended, manually add
  a row to preserve auditability per
  [CLAUDE.md rule "ALWAYS append to handoff/harness_log.md"].

## Research findings

### Synthetic probes and black-box monitoring (Google SRE)

- [Google SRE book, Ch. 6 -- Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)
  defines the Four Golden Signals (latency, traffic, errors, saturation).
  Every step above explicitly measures at least latency (timeout) and
  errors (exit-code + body grep).
- [O'Reilly SRE book Ch. 6 print edition](https://www.oreilly.com/library/view/site-reliability-engineering/9781491929117/ch06.html)
  distinguishes black-box (probe external interface) from white-box
  (instrument internals). Our smoketest is deliberately black-box --
  white-box coverage already exists in `tests/test_mcp_servers.py`,
  `tests/test_end_to_end.py`.
- [SRE School -- Black-box monitoring guide 2026](https://sreschool.com/blog/black-box-monitoring/):
  "exercise public interfaces, measure end-to-end behavior through
  external synthetic and real user probes."
- [Google SRE workbook -- Monitoring with advanced analytics](https://sre.google/workbook/monitoring/)
  reinforces "treat monitoring config as code" -- hence the
  deterministic driver script instead of an ad-hoc checklist.
- [dm03514 tech blog -- SRE Probing 101 with Cloudprober](https://medium.com/dm03514-tech-blog/sre-availability-probing-101-using-googles-cloudprober-8c191173923c)
  shows the dual-probe pattern (behind-LB + front-door) we echo in
  steps 4.6.1 (uvicorn direct) and 4.6.6 (Next.js-rendered front door).

### Chaos engineering (Netflix)

- [Gremlin -- Chaos Monkey origin](https://www.gremlin.com/chaos-monkey/the-origin-of-chaos-monkey)
  and [Wikipedia -- Chaos engineering](https://en.wikipedia.org/wiki/Chaos_engineering)
  establish the principle "the best way to avoid failure is to fail
  constantly." Step 4.6.8 is the minimum viable chaos probe for this
  codebase.
- [Coralogix -- How Netflix uses fault injection testing (FIT)](https://coralogix.com/blog/how-netflix-uses-fault-injection-to-truly-understand-their-resilience/):
  FIT's real contribution is precision -- you can assert which component
  fails and which alert fires. Our chaos step asserts both (SIGKILL
  target + expected alert row).
- [Splunk -- What is Chaos Monkey](https://www.splunk.com/en_us/blog/learn/chaos-monkey.html)
  and [OneUptime -- Chaos Monkey for resilience](https://oneuptime.com/blog/post/2026-01-28-chaos-monkey-resilience-testing/view)
  both emphasize "test and confirm recovery, not just cause disruption" --
  our step verifies auto-restart, not just the SIGKILL.
- [BrowserStack -- Chaos Monkey guide for engineers](https://www.browserstack.com/guide/chaos-monkey-testing)
  on measuring "observability precision" -- whether your monitoring
  actually detects the injected fault. Step 4.6.8 asserts on the alert
  row, not just the process death.
- [arXiv 1702.05843 -- Chaos Engineering (Basiri et al. 2017)](https://arxiv.org/pdf/1702.05843)
  formalizes the four principles: hypothesis of steady state, vary
  real-world events, run experiments in production, automate experiments
  to run continuously. This smoketest intentionally runs in a dev-port
  sandbox, not production, matching the spirit of the harness gate.

### Verification-first / evals (Anthropic + community)

- [Anthropic -- Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents):
  "automated evals pre-launch and in CI/CD are the first line of defense
  against quality problems" -- exactly the role this phase plays in the
  harness cycle.
- [Anthropic -- Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents):
  minimum viable tool set; our driver exposes one command per step, not a
  generic "run-everything" tool, to keep the planner's context lean.
- [Anthropic -- Prompting best practices (Claude 4)](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices):
  explicit success criteria enable higher-quality agent retries on
  failure -- our `recovery` per step is the failure-mode prompt.

### Synthetic monitoring tooling (Datadog, Honeycomb)

- [Datadog -- UX smoke tests with Synthetic Monitoring](https://www.datadoghq.com/blog/smoke-testing-synthetic-monitoring/)
  advocates "tag subsets of tests as smoke" so CI runs only the fast lane.
  Our driver exposes `--tag smoke` to support this.
- [Datadog -- Best practices for continuous testing](https://www.datadoghq.com/blog/best-practices-datadog-continuous-testing/)
  and
  [Datadog Synthetic docs](https://docs.datadoghq.com/synthetics/)
  emphasize self-healing locators. We use stable JSON keys and CSS class
  checks rather than brittle XPath.
- [NoBS Datadog best practices](https://www.nobs.tech/blog/datadog-synthetic-monitoring-best-practices):
  "a 5-minute test cadence means a 1-minute outage can look like 5
  minutes of downtime." The smoketest runs on demand (per harness cycle),
  not on a fixed schedule, to avoid this paradox.
- [Capterra -- Datadog vs Honeycomb 2026](https://www.capterra.com/compare/135453-198473/Datadog-Cloud-Monitoring-vs-Honeycomb)
  and
  [CubeAPM -- Datadog vs Honeycomb comparison](https://cubeapm.com/blog/datadog-vs-honeycomb-vs-cubeapm/)
  confirm Honeycomb's sweet spot is high-cardinality trace debugging,
  not smoke-test orchestration. We stay with bash + Python rather than
  introducing an observability vendor just for this phase.

### FastAPI + MCP health-check patterns

- [FastAPI testing docs](https://fastapi.tiangolo.com/tutorial/testing/)
  (TestClient + pytest + httpx) is the canonical pattern for steps
  like 4.6.1 and 4.6.3.
- [fast.io -- Implementing MCP server health checks](https://fast.io/resources/implementing-mcp-server-health-checks/)
  documents the JSON-RPC `ping` method + separate deep-check tool
  pattern -- we implement both in step 4.6.2 (ping) and 4.6.3 (deep
  check via `list_tools` + real signal call).
- [MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector)
  and [openstatus -- How to monitor an MCP server](https://docs.openstatus.dev/guides/how-to-monitor-mcp-server/)
  provide the industry-standard verification pattern we adapt.
- [pytest-uvicorn background fixture recipe](https://www.pythontutorials.net/blog/how-to-start-a-uvicorn-fastapi-in-background-when-testing-with-pytest/)
  gives us the boot-and-poll pattern used in step 4.6.1.
- [Safir -- spawning uvicorn for tests](https://safir.lsst.io/user-guide/uvicorn.html)
  is the LSST pattern for a separate-process uvicorn under pytest -- we
  use it for port-isolation from any running dev server.

### Smoke testing in CI and deployment gates

- [GitHub deployment gates docs](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/control-deployments)
  and
  [OneUptime -- Deployment gates in GitHub Actions](https://oneuptime.com/blog/post/2025-12-20-deployment-gates-github-actions/view)
  frame the smoketest as a deploy gate. Our driver's exit code can be
  wired into a future `.github/workflows/smoketest.yml`.
- [makerkit -- Smoke testing your SaaS with Playwright](https://makerkit.dev/blog/tutorials/smoke-testing-saas-playwright):
  "focus on critical paths -- auth, payments, features users pay for."
  For us that is: login (covered implicitly by step 4.6.6), paper-trading
  status, signal retrieval, Slack digest.
- [URL Health Check GitHub Action](https://github.com/marketplace/actions/url-health-check)
  is the minimum-viable probe -- we replicate its logic inline with
  `curl -sf` to stay dependency-free.
- [pronextjs.dev -- E2E testing with Playwright](https://www.pronextjs.dev/workshops/next-js-production-project-setup-and-infrastructure-fq4qc/e2-e-testing-with-playwright-d3eyw)
  and
  [Next.js Playwright docs](https://nextjs.org/docs/pages/guides/testing/playwright)
  for how to drive the Next.js 15 App Router in headless mode.

### Paper trading / quant validation

- [Portfolio123 -- Out-of-sample cohort analysis of algos](https://community.portfolio123.com/uploads/short-url/3WHpAUOzhCG8QAUez71HpoWnA62.pdf)
  makes the case that paper trading is precisely the calibration layer
  between backtest and live, so a smoketest that walks the
  paper-trading path is load-bearing for the rest of the product.
- [CFA Institute -- Investment Model Validation guide](https://rpc.cfainstitute.org/sites/default/files/-/media/documents/article/rf-brief/investment-model-validation.pdf)
  articulates the "simulation, backtest, forward test, live" ladder;
  this phase is the forward-test gate.

### Slack delivery verification

- [Hookdeck -- Guide to Slack webhooks best practices](https://hookdeck.com/webhooks/platforms/guide-to-slack-webhooks-features-and-best-practices):
  Slack auto-disables subscriptions after 95% failure in 60 minutes.
  Step 4.6.7 verifies end-to-end delivery (reads the posted message back)
  rather than trusting a 200.
- [platformOS -- Testing Slack notifications](https://www.platformos.com/blog/post/how-to-test-slack-notifications)
  and
  [Testkube -- Slack webhook integration](https://docs.testkube.io/articles/slack-integration)
  give the round-trip pattern (post + `conversations.history` lookup)
  we adopt.

## Proposed masterplan.json snippet

```json
{
  "id": "phase-4.6",
  "name": "Full-stack smoketest (E2E synthetic probe)",
  "status": "pending",
  "depends_on": ["phase-4.5"],
  "gate": null,
  "steps": [
    {
      "id": "4.6.0",
      "name": "Preflight: venv + critical imports",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "python -c \"import sys,fastapi,httpx,google.cloud.bigquery,anthropic,yfinance; assert sys.version_info[:2]==(3,14); print('PREFLIGHT_OK')\"",
        "success_criteria": [
          "stdout contains PREFLIGHT_OK",
          "exit code 0"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "4.6.1",
      "name": "Backend boot + /api/health returns 200",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "python scripts/smoketest/steps/boot_backend.py --port 8765 --timeout 30",
        "success_criteria": [
          "curl /api/health returns status=ok",
          "response includes mcp_servers with data, backtest, signals all ok",
          "latency under 5s"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "4.6.2",
      "name": "MCP servers respond to ping + list_tools",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "python scripts/smoketest/steps/mcp_ping.py --servers data,backtest,signals --timeout 10",
        "success_criteria": [
          "all three servers respond to JSON-RPC ping",
          "list_tools returns at least one tool per server"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "4.6.3",
      "name": "12 enrichment signals return for AAPL",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "curl -sf -m 60 http://127.0.0.1:8765/api/signals/AAPL | python -c \"import sys,json; d=json.load(sys.stdin); keys=['insider','options','social_sentiment','patent','earnings_tone','fred_macro','alt_data','sector','nlp_sentiment','anomalies','monte_carlo','quant_model']; miss=[k for k in keys if k not in d]; print('SIGNALS_OK' if not miss else f'MISSING:{miss}'); sys.exit(0 if not miss else 1)\"",
        "success_criteria": [
          "all 12 documented signal keys present",
          "at least 8 of 12 return non-ERROR payload",
          "total wall clock under 60s"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "4.6.4",
      "name": "Paper trading run-now dry-run succeeds",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "curl -sf -m 120 -X POST \"http://127.0.0.1:8765/api/paper-trading/run-now?dry_run=true\" -H \"Content-Type: application/json\" -d '{}'",
        "success_criteria": [
          "HTTP 200",
          "response contains status=ok or started=true",
          "subsequent /status shows last_run_ts within 120s"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "4.6.5",
      "name": "Frontend npm run build succeeds",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "cd /Users/ford/.openclaw/workspace/pyfinagent/frontend && npm run build 2>&1 | tail -30",
        "success_criteria": [
          "exit code 0",
          "output contains Compiled successfully",
          "no Type error lines in last 200 lines"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "4.6.6",
      "name": "Paper-trading 5 tabs render without error",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "python scripts/smoketest/steps/frontend_tabs.py --base http://localhost:3000 --tabs positions,trades,chart,reality-gap,exit-quality",
        "success_criteria": [
          "all 5 tab routes return HTTP 200",
          "each tab label text present in rendered HTML",
          "no TypeError or ReferenceError in console logs",
          "no rose-500 error banner element in DOM"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "4.6.7",
      "name": "Slack digest delivered end-to-end",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "python -m backend.slack_bot.digest_test --channel-env SLACK_TEST_CHANNEL_ID --text smoketest-4.6.7 --verify-delivery",
        "success_criteria": [
          "Slack API returns ok=true",
          "conversations.history lookup returns the posted message",
          "round trip under 10s"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "4.6.8",
      "name": "Watchdog alert fires on simulated process kill",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "python scripts/smoketest/steps/chaos_watchdog.py --timeout 90",
        "success_criteria": [
          "paper_trader subprocess SIGKILLed",
          "sla_monitor writes alert row within 90s",
          "process auto-restarts"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "4.6.9",
      "name": "Append harness log row + clean shutdown",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "python scripts/smoketest/steps/finalize.py --log handoff/harness_log.md --port 8765",
        "success_criteria": [
          "handoff/harness_log.md gained exactly one row",
          "new row contains phase=4.6 and result field",
          "no uvicorn process bound to 8765 after exit",
          "no stray mcp_server PIDs vs pre-test snapshot"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    }
  ]
}
```

## Implementation notes

- Create `scripts/smoketest/` with:
  - `run_smoketest.py` -- the top-level driver. Accepts
    `--dry-run`, `--only 4.6.3`, `--skip 4.6.8`, `--tag smoke`. Imports
    each step module, runs it, aggregates results, writes the harness
    log row.
  - `steps/boot_backend.py`, `steps/mcp_ping.py`, `steps/frontend_tabs.py`,
    `steps/chaos_watchdog.py`, `steps/finalize.py`. Each exposes a
    `main(argv)` returning exit code 0 on PASS.
  - `conftest.py` -- shared fixtures (port allocator, process cleanup).
- Wire `scripts/harness/run_harness.py` to invoke the smoketest before
  any promotion to the "trusted" state; failure blocks the cycle.
- Do NOT replace `tests/test_end_to_end.py` -- white-box tests stay.
  The smoketest is the additional black-box layer.
- The `/api/paper-trading/run-now` endpoint currently does not accept a
  `dry_run=true` query parameter -- Phase 4.6 implementation MUST add it
  and gate BQ writes off when set. This is a one-line change to
  `backend/api/paper_trading.py::run_now`.
- `SLACK_TEST_CHANNEL_ID` is a new env var; add to `.env.example` and
  `backend/config/settings.py`. Must be a private channel the bot is in.
- Follow the ASCII-only-logger rule from `.claude/rules/security.md`:
  no arrows or em dashes in `logger.info()` / `print()` -- use `--` and
  `->` instead.
- All `open()` calls in new scripts MUST pass `encoding="utf-8"` per
  `.claude/rules/backend-api.md`.
- Budget: wall-clock <= 8 minutes with live APIs. Step 4.6.3 dominates
  (~45s for 12 vendor fetches) and step 4.6.5 is ~40s. Parallelize
  nothing -- ordering matters because 4.6.8 kills a process 4.6.4 spun up.
- The frontend Next.js server is assumed already running on 3000 for step
  4.6.6; if not, the driver boots `npm run start` in a subprocess and
  tears it down in `finalize.py`. Do not conflate this with the dev
  server on the same port.
- Masterplan immutability: once this phase is merged, the
  `verification.command` and `success_criteria[]` are frozen per
  CLAUDE.md rule "Never edit verification criteria in masterplan.json."
  Add new criteria only by creating phase-4.6.x sub-steps.

## References

- [Google SRE book -- Monitoring Distributed Systems (Ch. 6)](https://sre.google/sre-book/monitoring-distributed-systems/)
- [Google SRE book -- Practical Alerting](https://sre.google/sre-book/practical-alerting/)
- [Google SRE workbook -- Monitoring](https://sre.google/workbook/monitoring/)
- [O'Reilly SRE book print -- Ch. 6 Monitoring Distributed Systems](https://www.oreilly.com/library/view/site-reliability-engineering/9781491929117/ch06.html)
- [SRE School -- Black-box monitoring (2026 guide)](https://sreschool.com/blog/black-box-monitoring/)
- [dm03514 -- SRE Probing 101 with Cloudprober](https://medium.com/dm03514-tech-blog/sre-availability-probing-101-using-googles-cloudprober-8c191173923c)
- [SentinelOne / Scalyr -- Black box monitoring for opaque systems](https://www.sentinelone.com/blog/black-box-monitoring-track-opaque-systems/)
- [Gremlin -- Chaos Monkey origin story](https://www.gremlin.com/chaos-monkey/the-origin-of-chaos-monkey)
- [Wikipedia -- Chaos engineering](https://en.wikipedia.org/wiki/Chaos_engineering)
- [arXiv 1702.05843 -- Chaos Engineering (Basiri et al.)](https://arxiv.org/pdf/1702.05843)
- [Coralogix -- How Netflix uses fault injection testing](https://coralogix.com/blog/how-netflix-uses-fault-injection-to-truly-understand-their-resilience/)
- [Splunk -- What is Chaos Monkey](https://www.splunk.com/en_us/blog/learn/chaos-monkey.html)
- [BrowserStack -- Chaos Monkey guide for engineers](https://www.browserstack.com/guide/chaos-monkey-testing)
- [OneUptime -- Chaos Monkey for resilience testing](https://oneuptime.com/blog/post/2026-01-28-chaos-monkey-resilience-testing/view)
- [Google Cloud -- Getting started with chaos engineering](https://cloud.google.com/blog/products/devops-sre/getting-started-with-chaos-engineering)
- [Datadog -- Smoke testing with synthetic monitoring](https://www.datadoghq.com/blog/smoke-testing-synthetic-monitoring/)
- [Datadog -- Best practices for continuous testing](https://www.datadoghq.com/blog/best-practices-datadog-continuous-testing/)
- [Datadog Synthetic docs](https://docs.datadoghq.com/synthetics/)
- [NoBS -- Datadog synthetic monitoring best practices](https://www.nobs.tech/blog/datadog-synthetic-monitoring-best-practices)
- [Capterra -- Datadog vs Honeycomb 2026](https://www.capterra.com/compare/135453-198473/Datadog-Cloud-Monitoring-vs-Honeycomb)
- [CubeAPM -- Datadog vs Honeycomb vs CubeAPM comparison](https://cubeapm.com/blog/datadog-vs-honeycomb-vs-cubeapm/)
- [Anthropic -- Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- [Anthropic -- Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Anthropic -- Prompting best practices (Claude 4)](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices)
- [FastAPI testing docs](https://fastapi.tiangolo.com/tutorial/testing/)
- [fast.io -- Implementing MCP server health checks](https://fast.io/resources/implementing-mcp-server-health-checks/)
- [MCP Inspector docs](https://modelcontextprotocol.io/docs/tools/inspector)
- [openstatus -- How to monitor an MCP server](https://docs.openstatus.dev/guides/how-to-monitor-mcp-server/)
- [pytest-uvicorn background fixture recipe](https://www.pythontutorials.net/blog/how-to-start-a-uvicorn-fastapi-in-background-when-testing-with-pytest/)
- [Safir -- spawning uvicorn for tests](https://safir.lsst.io/user-guide/uvicorn.html)
- [Next.js Playwright testing docs](https://nextjs.org/docs/pages/guides/testing/playwright)
- [makerkit -- Smoke testing your SaaS with Playwright](https://makerkit.dev/blog/tutorials/smoke-testing-saas-playwright)
- [pronextjs.dev -- E2E testing with Playwright](https://www.pronextjs.dev/workshops/next-js-production-project-setup-and-infrastructure-fq4qc/e2-e-testing-with-playwright-d3eyw)
- [GitHub deployment gates docs](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/control-deployments)
- [OneUptime -- Deployment gates in GitHub Actions](https://oneuptime.com/blog/post/2025-12-20-deployment-gates-github-actions/view)
- [URL Health Check GitHub Action](https://github.com/marketplace/actions/url-health-check)
- [Hookdeck -- Guide to Slack webhooks best practices](https://hookdeck.com/webhooks/platforms/guide-to-slack-webhooks-features-and-best-practices)
- [platformOS -- How to test Slack notifications](https://www.platformos.com/blog/post/how-to-test-slack-notifications)
- [Testkube -- Slack webhook integration](https://docs.testkube.io/articles/slack-integration)
- [Portfolio123 -- Out-of-sample cohort analysis](https://community.portfolio123.com/uploads/short-url/3WHpAUOzhCG8QAUez71HpoWnA62.pdf)
- [CFA Institute -- Investment Model Validation guide](https://rpc.cfainstitute.org/sites/default/files/-/media/documents/article/rf-brief/investment-model-validation.pdf)
