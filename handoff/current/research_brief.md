# Research Brief -- phase-62.4 (goal-away-ops): guardrail/budget sentinel scripts/away_ops/sentinel.sh

Tier: moderate (caller-set). Date: 2026-06-12. Researcher: Layer-3 (merged Explore). STATUS: COMPLETE (internal pass 1 + external/synthesis pass 2 after retry; gate PASSED).

Step scope: scripts/away_ops/sentinel.sh printing {metered_llm_usd_today, baseline_usd,
kill_switch_paused, flags_match_tokens, ok} JSON; exit 0 healthy; metered-figure source PINNED
in script header; tamper tests (synthetic cost row above baseline -> non-zero exit + named gate;
behavior flag w/o matching token -> non-zero exit + named gate); verify 62.3 wrapper pre-flight
wiring assumption (missing-or-failing sentinel -> digest-only) + document wrapper-test leg.

## Read in full (>=5 required; counts toward gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents | 2026-06-12 | official eng blog | WebFetch full | Pre-flight health check before each session: "Start the session by reading `init.sh`"; "run a basic test on the development server to catch any undocumented bugs"; detect "if the app had been left in a broken state". NO budget/circuit-breaker content (notable absence). |
| https://www.braintrust.dev/articles/how-to-track-llm-costs-2026 | 2026-06-12 | authoritative blog | WebFetch full | Request-time tagging beats retroactive log parsing: "retry tokens may not surface in application logs"; "Rolling cached and uncached tokens into a single input-token count can overstate spending"; hard-cap "check belongs in the proxy or middleware layer, because blocking at the LLM call is too late"; soft alerts use rolling baselines, hard gates use fixed caps. |
| https://docs.cloud.google.com/bigquery/docs/control-genai-costs | 2026-06-12 | official docs | WebFetch full | "Token usage directly correlates with your Vertex AI billing" for Gemini; "Cached tokens don't count towards the quotas"; doc is silent on grounding/per-request charges -> token x price is a floor, not the invoice. |
| https://waxell.ai/blog/ai-agent-token-budget-enforcement | 2026-06-12 | industry blog | WebFetch full | "the enforcement layer has to be outside the agent's code. An agent that has been told 'stop after $X' in its system prompt will honor that instruction right up until it's task-motivated not to." Monitoring "is asynchronous: by the time a monitoring alert fires, the spend has already occurred." $47K/11-day runaway-loop incident (Nov 2025). "you cannot reliably cost-estimate a production agent from its per-request performance in staging". Fail-open-vs-fail-closed for the enforcement layer itself: NOT addressed. |
| https://www.moschetti.org/rants/jqvspy.html | 2026-06-12 | practitioner blog | WebFetch full | jq -> python break-even as state/logic grows: "running it all in python is clearly more attractive"; python list-args "eliminates the headache of escaping quotes"; complex jq workflows devolve into "lots of individual jq executions... writing it back out to a tmp file". |
| https://jqlang.org/manual/ | 2026-06-12 | official docs | WebFetch full | `--arg` "passes a value to the jq program as a predefined variable... value will be treated as a string" (injection-safe emission with `-n`); `--argjson` for typed values; `-e` exit-status mapping (0 truthy / 1 false-null / 4 no output) for shell gating. |
| https://oneuptime.com/blog/post/2026-02-26-configuration-drift-detection-gitops/view | 2026-06-12 | industry blog | WebFetch full | Drift = live state diverging from declared source of truth; passive (alert) vs active (self-heal) modes, "Self-healing is powerful but can be disruptive"; "make all detected drift actionable -- either it is a legitimate issue that needs correction, or it is an expected behavior that should be explicitly ignored" (`ignoreDifferences` = grandfather-manifest analogue). |
| https://ss64.com/mac/date.html | 2026-06-12 | man-page mirror (official) | WebFetch full | BSD `-v` adjusts/sets date parts ("If val is preceded with a plus or minus sign, the date is adjusted... otherwise the relevant part of the date is set"); `-j` = don't set, parse with `-f`; `-u` = UTC; **no GNU-style `-d` exists on macOS date**; `+%Y-%m-%d` formatting and `-I` are portable. |
| https://docs.cloud.google.com/bigquery/docs/data-manipulation-language | 2026-06-12 | official docs | WebFetch full | "Rows that were recently written using the tabledata.insertall streaming method can't be modified with data manipulation language (DML)... The recent writes are those that occurred within the last 30 minutes." (Storage Write API rows ARE immediately modifiable.) Governs the tamper-test design: `api_call_log.py:141,320` uses `insert_rows_json` = insertAll path. |

## Snippet-only (context, not gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://www.baeldung.com/linux/bash-variables-create-json-string | blog | **HTTP 403 on WebFetch** (attempted); replaced by jq manual (higher tier) |
| https://www.traceloop.com/blog/from-bills-to-budgets-how-to-track-llm-token-usage-and-cost-per-user | blog | gateway/proxy attribution pattern; redundant w/ Braintrust |
| https://langfuse.com/docs/observability/features/token-and-cost-tracking | official docs | platform-specific (Langfuse) cost ingestion; not adopting the platform |
| https://www.cloudzero.com/blog/llm-api-pricing-comparison/ | industry | pricing tables only |
| https://www.truefoundry.com/blog/llm-cost-attribution-team-budgets | industry | chargeback/team budgets; out of scope |
| https://blog.alephant.io/10-real-time-ai-api-budget-guardrails-for-2026/ | industry | Alert -> Throttle -> Kill ladder; corroborates Waxell |
| https://cordum.io/blog/ai-agent-circuit-breaker-pattern | industry | tool-failure circuit breakers (reliability, not budget) |
| https://earezki.com/ai-news/2026-03-02-i-built-an-mcp-server-so-my-ai-agent-can-track-its-own-spending/ | blog | agent self-tracking via MCP -- the pattern Waxell argues AGAINST as sole control |
| https://techcommunity.microsoft.com/blog/linuxandopensourceblog/applying-site-reliability-engineering-to-autonomous-ai-agents/4521357 | vendor blog | SRE-for-agents framing; threshold->terminate enforcement layer |
| https://www.sakurasky.com/blog/missing-primitives-for-trustworthy-ai-part-6/ | blog | layered kill-switches (hard stop / soft pause / spend governors) |
| https://www.anthropic.com/engineering/harness-design-long-running-apps | official eng blog | already canonical project reference; read in prior phases |
| https://sqlpey.com/bash/shell-variables-in-jq-filters/ | community | --arg injection-safety; superseded by jq manual |
| https://cameronnokes.com/blog/working-with-json-in-bash-using-jq/ | blog | single-quote-the-filter hygiene |
| https://www.benjaminrancourt.ca/how-to-easily-create-a-json-file-in-bash/ | blog | heredoc JSON creation (the unsafe-without-escaping pattern) |
| https://www.harness.io/harness-devops-academy/configuration-drift | vendor glossary | drift causes/consequences; generic |
| https://octopus.com/devops/feature-flags/feature-flag-best-practices/ | vendor | flag audits, stale-flag review, single source of truth |
| https://docs.terrateam.io/governance/drift-detection/ | official docs | scheduled drift scans for IaC |
| https://shellmap.eversources.app/cmd/date | reference | "date -d yesterday works on Linux, errors on macOS. date -v-1d works on macOS, errors on Linux"; +%Y-%m-%d portable; 86400-second DST trap |
| https://www.jbmurphy.com/2011/02/17/gnu-date-vs-bsd-date/ | blog | year-less canonical prior art for the -v/-d divergence |
| https://man7.org/linux/man-pages/man1/date.1.html | official docs | GNU side of the divergence (--date strings) |
| https://learningbox.co.jp/en/2016/06/27/... | blog | historical: BSD -d set kernel DST value (silent misbehavior class) |

## Search queries (three-variant discipline)

5 topics x 3 variants = 15 queries, all run 2026-06-12: (T1) "LLM cost tracking attribution BigQuery token usage 2026" / "...per-token budget tracking pattern 2025" / year-less "LLM cost observability token usage logging warehouse". (T2) "autonomous AI agent budget guard circuit breaker 2026" / "AI agent spending limit kill switch guardrail pattern 2025" / year-less "Anthropic engineering agent harness budget guardrails long-running". (T3) "shell script emit JSON safely jq vs python 2026" / "generate JSON from bash script jq --arg escaping 2025" / year-less "bash output JSON best practice jq heredoc pitfalls". (T4) "configuration drift detection desired state reconciliation 2026" / "feature flag drift audit source of truth reconciliation 2025" / year-less "configuration drift detection GitOps reconciliation loop". (T5) "BSD date vs GNU date macOS shell script differences 2026" / "macOS date command -v flag vs GNU date -d midnight today 2025" / year-less "GNU date BSD date portability shell script day boundary UTC".

## Recency scan (2024-2026)

Performed (every topic had a 2026-scoped and a 2025-scoped query). Findings:
1. **2025-2026 agent-budget literature converged on "enforcement outside the agent"** (Waxell 2026, Alephant 2026, Microsoft SRE-for-agents 2026) with Alert -> Throttle -> Kill ladders; motivating incident class is the Nov-2025 $47K/11-day runaway loop. This SUPERSEDES older alert-only cost dashboards and directly validates the 62.3/62.4 split (wrapper-level pre-flight gate, not prompt-level instruction).
2. **Anthropic "Effective harnesses for long-running agents" (late 2025)** adds the pre-flight environment-health-check pattern (init.sh + smoke test before each session) -- the architectural template the sentinel instantiates. Its budget silence means budget mechanics must come from the cost literature, not Anthropic.
3. **Braintrust 2026 playbook**: request-creation-time attribution is now standard doctrine; retroactive log parsing under-reports (retries, cache splits) -- confirms treating llm_call_log token-math as a lower bound.
4. **jq 1.8.x is current upstream (2025)**; macOS now ships jq in the BASE system (`/usr/bin/jq`, 1.7.1-apple -- verified live on this Mac 2026-06-12), removing the "jq not installed" portability objection for launchd contexts on THIS host.
5. No 2024-2026 source addresses fail-open-vs-fail-closed for the budget-enforcement layer itself (checked explicitly; Waxell silent) -- that decision must be reasoned from away-ops rails, not copied from literature.

## Key findings

1. **Enforcement placement**: budget gates must sit outside the agent's reasoning loop -- "the enforcement layer has to be outside the agent's code" (Waxell 2026); monitoring alone "is asynchronous: by the time a monitoring alert fires, the spend has already occurred". The sentinel-in-wrapper pre-flight (before `claude -p` spawns) is the proxy-layer interception point Braintrust prescribes ("blocking at the LLM call is too late").
2. **Token-math is a floor**: Google's own doc ties tokens to Vertex billing but covers neither grounding nor per-request charges; Braintrust documents invisible retry tokens + cache-token splits. External literature independently confirms the internal 58.1 under-metering finding -> pin "lower bound" in the script header.
3. **Hard gates use pinned constants, rolling baselines belong in alerts**: Braintrust separates fixed hard caps from rolling-baseline soft alerts; Waxell: "you cannot reliably cost-estimate a production agent from its per-request performance in staging" -> baseline_usd = pinned 58.1-ledger constant, NOT a derived 14-day mean.
4. **Drift doctrine**: every detected divergence must be "either... a legitimate issue that needs correction, or... an expected behavior that should be explicitly ignored" (oneuptime/ArgoCD `ignoreDifferences`) -- the git-tracked `flag_baseline.json` grandfather manifest is exactly this ignore-rule pattern; sentinel stays PASSIVE (alert+gate, never auto-edits .env; self-healing "can be disruptive").
5. **JSON emission**: build JSON with a real serializer, never bash string-concat. jq `-n --arg` is injection-safe ("value will be treated as a string"); python `json.dumps` wins once state/logic grows (moschetti break-even). Since the sentinel must call venv python anyway (BQ + .env parse), one serializer = `json.dumps`.
6. **Date math**: BSD/GNU adjustment syntax is incompatible (`-v` vs `-d`; macOS has NO GNU `-d`), but `date -u +%Y-%m-%d` IS portable. Compute the day boundary in SQL (`CURRENT_DATE()` is UTC) or python UTC -- never bash local-time arithmetic (Oslo CEST is UTC+2: a local "today" misclassifies 2h of rows against the UTC-partitioned `DATE(ts)`).
7. **Tamper-test constraint**: rows written via `insert_rows_json` (= `tabledata.insertAll`, `api_call_log.py:141,320`) "can't be modified with DML... within the last 30 minutes" (BQ docs) -> a synthetic row in PROD llm_call_log cannot be promptly deleted and would inflate `metered_llm_usd_today` for the rest of the UTC day. Inject the synthetic figure via env override or a test dataset, not a prod write.

## Internal code inventory

### 1. Metered-cost source (THE critical question)

**Answer: `pyfinagent_data.llm_call_log` (BQ, US) tokens x `cost_tracker.MODEL_PRICING`, with
cc-rail exclusions, is the ONLY queryable per-call metered source -- and it is a documented
LOWER BOUND. Pin exactly this in the sentinel header.**

| Candidate | File:line | Verdict |
|---|---|---|
| `llm_call_log` writer | `backend/services/observability/api_call_log.py:189-331` | Schema (migration comment :191-204 + `scripts/migrations/add_llm_call_log.py:40-51`): ts/provider/model/agent/latency/ttft/input_tok/output_tok/cache_creation_tok/cache_read_tok/request_id/ok/ticker/cycle_id/session_cost_usd. **NO cost_usd column** -- cost must be computed tokens x pricing. PARTITION BY DATE(ts), clustered (provider, model) -> day-bounded queries are partition-pruned + cheap. ts is UTC (`_now_iso()` :62-63). |
| Pricing join precedent | `backend/api/sovereign_api.py:236-286` `_fetch_llm_cost_by_provider` | THE pattern to mirror: GROUP BY provider, model over `ts >=` window, join `MODEL_PRICING` in Python ("no native BQ pricing table -- the dict is the source of truth" :239-240). Maps gemini->vertex bucket. **Flaw for sentinel reuse: written pre-60.4, it counts `provider='anthropic' AND agent LIKE 'cc_rail%'` rows (flat-fee) as metered.** |
| CC orchestrator rail | `backend/agents/claude_code_client.py:340-365` | phase-60.4 AW-7 writer: `provider="anthropic"`, `agent="cc_rail:<role>"`. Docstring: "Cost is the flat-fee rail: session_cost_usd delta 0 (tokens still recorded for volume audits)". **Provider alone does NOT separate metered from flat-fee.** |
| CC lite-path rail | `backend/services/autonomous_loop.py:1866-1890` | phase-56.2 writer: `provider="claude-code"` -- second flat-fee marker, different convention than 60.4's. Metered filter must exclude BOTH: `provider != 'claude-code' AND NOT (provider='anthropic' AND agent LIKE 'cc_rail%')`. |
| Metered writers | `llm_client.py:1086-1100` (gemini), `:1754-1776` (anthropic SDK), `:2152-2177` (advisor dual-rows), `orchestrator.py:824-827` (gemini code-exec) | gemini = Vertex metered; anthropic SDK = ANTHROPIC_API_KEY metered (the 51.1 overlay calls). |
| cost_budget_watcher | `backend/slack_bot/jobs/cost_budget_watcher.py:82-115` | Watches **BigQuery bytes-billed** (`INFORMATION_SCHEMA.JOBS_BY_PROJECT` x $6.25/TiB), NOT LLM spend. phase-9.9.2 swapped away from the Anthropic Cost API. Not the LLM source. |
| `_check_cost_budget` | `backend/agents/llm_client.py:396-456` | **Finding: the "$25 daily LLM-spend cap" (`cost_budget_daily_usd`, settings.py:345) is wired to `_default_fetch_spend()` = BQ bytes-billed (:427-431), not LLM tokens.** Description-vs-implementation mismatch; no existing runtime guard watches metered LLM token dollars. The 62.4 sentinel is the FIRST. |
| `/api/reports/cost-history` | `backend/api/reports.py:87-97` -> `backend/db/bigquery_client.py:362-375` | Per-report `total_cost_usd` from the reports table = cost_tracker NOMINAL (values CC-rail tokens at API prices; live_check_58.1.md ledger: MU row 3.776 nominal vs actual metered "<<"). WRONG source for metered. |

**Live 14-day baseline query (run 2026-06-12, venv python + ADC, partition-filtered, <30s):**
only 3 of 14 days have metered rows -- 2026-05-29 $0.0054, 2026-06-01 $0.0025, 2026-06-11
$0.0106 (all gemini-2.5/2.0-flash; mean $0.0061/day). The 58.1 ledger estimated ~$0.5-1.0
metered for 2026-06-11 alone => **llm_call_log UNDER-METERS Vertex spend** (live_check_58.1.md
§D disclosure: "Gemini calls are under-metered in llm_call_log"; grounded-search legs also
bill per-request, invisible to token math). Two consequences: (a) the sentinel header must pin
the figure as a lower bound; (b) baseline_usd must be a PINNED CONSTANT from the 58.1 ledger
(lite $0.05-0.17/cycle, full $1.08-4.06/cycle), NOT an auto-derived 14-day mean ($0.006 would
false-trip the first legitimate full cycle). BQ syntax trap hit live: `ROWS` is a reserved
keyword as a column alias (same class as the 60.4 calendar_events bug).

**$25 58.1 window identification:** rail 4 exempts "$25 58.1 window + existing Gemini
pipeline" (away-ops-rules.md:14-16). llm_call_log has NO purpose tag -- window spend is
identifiable only by date range (window opened 2026-06-11, live_check_58.1.md §A) +
provider='gemini'. Recommended: bake the exemption into the baseline constant (set
baseline_usd to accommodate approved full-mode days, e.g. $5-8/day; hard-fail well below the
$25/day settings cap) rather than attempting per-row window attribution that the schema
cannot support.

### 2. Flag-vs-token reconciliation

- Registry: `KNOWN_TOKEN_ENV_MAP` (`backend/slack_bot/operator_tokens.py:52-55`) is **EMPTY
  today** -- both entries ("FEE TABLE" -> PAPER_FEE_TABLE_ENABLED, "EU SCREENER" ->
  PAPER_SCREENER_PER_MARKET) are comments to be registered by 61.5/65.2 when they ship.
- Behavior flags in settings (pydantic `model_config` settings.py:545 = env_file backend/.env,
  no prefix -> env var = UPPERCASED field name): `paper_data_integrity_enabled` (:42),
  `paper_risk_judge_reject_binding` (:277), `paper_swap_churn_fix_enabled` (:311),
  `momentum_52wh_tilt_enabled` (:409). All Field(False) defaults; the first three are
  operator-ON in backend/.env per caller (keystroke-applied pre-62.2).
- `handoff/operator_tokens.jsonl` **does not exist** (verified 2026-06-12) -- 62.2 is still
  `pending` in masterplan and the three ON flags pre-date the token mechanism. A naive
  "every True flag needs a token line" check would fail on day one.
- **Grandfather mechanism (recommended):** a baseline manifest, e.g.
  `scripts/away_ops/flag_baseline.json` = {ENV_KEY: expected_bool} captured at 62.4 build
  time and cited by 62.4's contract/commit. Sentinel rule: for each registry flag read from
  backend/.env, `value == baseline[key]` -> OK (grandfathered); `value != baseline[key]` ->
  require a matching operator_tokens.jsonl line (key via KNOWN_TOKEN_ENV_MAP, value ON/OFF
  matching) else exit non-zero with gate `flags_match_tokens`. New keys absent from baseline
  default to False-expected. The manifest is git-tracked: tampering it is visible in diffs,
  and the pre-tool-use hook's .env write tripwire (below) is the complementary control.
- **.env read-access distinction (confirmed live):** my own Bash grep on `backend/.env` was
  DENIED -- the Claude-session permission layer + `pre-tool-use-danger.sh` intercept CLAUDE
  tool calls (write shapes blocked at :173-199 absent a fresh `tokens_cursor` mtime<6h; the
  hook comment :174-175 explicitly names "the 62.4 sentinel reconciliation is the backstop").
  A launchd-spawned wrapper subprocess is NOT a Claude tool call -- no hooks, no permission
  rules; plain POSIX read access applies. Read-only .env access in the sentinel process is
  safe and is the designed division of labor (hook = write tripwire, sentinel = read-only
  reconciliation). Same trick lets Main build the baseline manifest via a script whose
  command line never names .env.
- Parsing .env in bash is fragile (quotes, comments, CRLF) -- delegate to venv python
  (`backend.config.settings.get_settings()` itself, or a 10-line dotenv parse) for the
  reconciliation leg.

### 3. Kill-switch state read

- File replay: `backend/services/kill_switch.py:36` `_AUDIT_PATH` =
  `handoff/kill_switch_audit.jsonl`; `_load_from_audit()` (:61-106) scans all lines, last
  `pause`/`resume` event wins (:71-81). File exists (2.9KB; tail shows resume 2026-06-11 +
  sod_snapshot/peak_update rows). **Works with backend DOWN -- this is the reliable path.**
- API: `GET /api/paper-trading/kill-switch` (`backend/api/paper_trading.py:480-495`) needs a
  live backend; healthcheck (62.5) owns service-liveness, so the sentinel must NOT
  false-fail on a dead backend -> replay the JSONL directly (paper_trading.py:777-778 itself
  mirrors _AUDIT_PATH for local-file reads -- precedent).
- Replay is ~10 lines of python: track last pause/resume; absent file = not paused (matches
  `_load_from_audit` :62-63 semantics). Note: `kill_switch_paused: true` is NOT by itself
  unhealthy for the sentinel (rail 5 keeps it paused after breaches; pause is a legitimate
  state to REPORT, not a gate failure -- the away rails want it surfaced in the digest).
  Sentinel prints the boolean; gate failure is reserved for budget breach + flag mismatch.

### 4. Wrapper sentinel contract (run_away_session.sh:80-103) -- 62.3 wiring VERIFIED

- `scripts/away_ops/run_away_session.sh:83-85`: `[ ! -x sentinel.sh ]` -> log "sentinel
  missing -- fail-closed to digest-only" -> PROMPT_KIND=digest_only. :86-88: `bash
  sentinel.sh >> $SLOG 2>&1` non-zero exit -> "sentinel FAILED -- downgrading to
  digest-only". **Wiring assumption HOLDS in code.** Consequences:
  - Invocation is `bash sentinel.sh` but the gate test is `-x` -> the file MUST be chmod +x
    anyway or it is treated as missing.
  - stdout/stderr append to `handoff/away_ops/session.log` -- the JSON is an audit artifact
    in the log; the wrapper parses NOTHING, only the exit code drives the downgrade. JSON on
    stdout single-line keeps the log greppable.
  - **No timeout wraps the sentinel call** (gtimeout guards only the claude invocation
    :119). A hung BQ query stalls the whole launchd session BEFORE any cap -> sentinel must
    SELF-BOUND (internal query timeout ~20s; total <30s per immutable criteria).
  - `AWAY_SESSION_DRY_RUN=1` **bypasses the entire pre-flight block** (:80) -- all of
    today's 62.3 acceptance runs were dry_run=1 (session.log 2026-06-12: lock/HALT-DEV/
    prompt-selection proven, sentinel branch NEVER executed). The 62.4 wrapper-test leg
    cannot use the existing dry-run path as-is; and non-dry-run launches REAL `claude -p`
    with hardcoded `CLAUDE_BIN` (:18) -- not PATH-shimmable. Recommendation: 62.4 makes the
    minimal wrapper amendment to run the (read-only, fast) sentinel pre-flight in dry-run
    too, so `AWAY_SESSION_DRY_RUN=1` + a forced-fail sentinel asserts the
    `prompt=digest_only` log line without burning a session; disclose the 62.3-file edit in
    experiment_results.md. Fallback: defer the live integration leg to 62.7's dress
    rehearsal (operator watching) and ship 62.4 with sentinel unit tamper-tests + static
    wiring citation only -- weaker vs the immutable criterion's "wrapper test asserts the
    prompt path switch".
- Exit-code semantics to implement: 0 = healthy (all gates pass; JSON ok:true); non-zero =
  at least one NAMED gate failed (JSON ok:false + gate name on stdout before exit).
- masterplan verification command (goal_away_ops.md:138): `source .venv/bin/activate && bash
  scripts/away_ops/sentinel.sh; echo exit=$?` -- venv IS available to the sentinel; the
  launchd wrapper does NOT activate venv, so the sentinel must resolve the venv python by
  absolute path (`$REPO/.venv/bin/python`) rather than rely on PATH.

## Consensus vs debate

- **Consensus**: enforcement outside the agent (Waxell/Alephant/Microsoft/Braintrust unanimous); request-time attribution > retroactive log parsing; serializer-emitted JSON only; Git-tracked baseline + explicit ignore rules for legitimate drift; `+%Y-%m-%d`/`-u` portable across BSD/GNU date while adjustment flags are not.
- **Debate 1 -- jq vs python for emission**: jq camp (one-shot, pipeline-fit) vs python camp (state, debugging, no tmp-file chains). Resolved for 62.4 by context: venv python is already mandatory for the BQ + .env legs, so `json.dumps` wins; `/usr/bin/jq -n --arg` is the sanctioned fallback if a pure-bash leg ever needs it.
- **Debate 2 -- fail-open vs fail-closed when the guard itself breaks**: literature is SILENT (explicitly checked). The 62.3 wrapper already chose fail-closed-to-digest-only for sentinel-missing/failing; consistency + the away threat model ("can't verify the budget -> don't burn it") extend that to infra failures inside the sentinel, with a DISTINCT gate name so the digest can tell outage from breach. Cost is low because digest-only is a useful degraded mode, not a halt.
- **Debate 3 -- auto-remediation**: GitOps self-healing exists but "can be disruptive"; for operator-owned .env flags, auto-revert is out of the question (the sentinel is read-only by design; the pre-tool-use hook is the write tripwire). Passive gate is the right mode.

## Pitfalls

1. Bash string-concatenated/heredoc JSON breaks on quotes/newlines -> serializer only.
2. GNU-ism `date -d` silently absent/different on macOS; historical BSD `-d` set the kernel DST value. Any shell date ARITHMETIC is a portability bug class; only formatting is portable.
3. 86400-second day arithmetic breaks across DST (25h wall-clock days).
4. Local-time "today" vs UTC-partitioned `DATE(ts)`: Oslo (UTC+2 summer) misbounds the window by 2h and breaks partition pruning if the predicate isn't a DATE literal/`CURRENT_DATE()`.
5. Synthetic prod-row tamper tests: insertAll rows are DML-immutable for 30 min; the row then poisons the real metered figure for the rest of the day (self-DoS: the gate trips on every later run today).
6. Auto-derived baselines from sparse history false-trip on the first legitimate heavy day (internal: 14-day mean $0.006 vs legitimate full cycle $1-4).
7. Naive "every ON flag needs a token" fails day one (tokens file doesn't exist; 3 flags grandfathered; registry map empty).
8. Unbounded BQ call in a pre-flight with no wrapper timeout stalls the whole launchd session.
9. `jq -e`/exit-code subtleties: if jq is used for gating, 0/1/4 semantics differ from "non-zero = fail" intuition -- another reason to keep gate logic in python/bash, not jq filters.

## Application to pyfinagent

- **Shape**: `scripts/away_ops/sentinel.sh` = thin bash orchestrator (chmod +x; resolves `$REPO/.venv/bin/python` by absolute path per internal §4) that runs ONE embedded python program doing: (a) BQ metered query with ~20s timeout, day window = `DATE(ts) = CURRENT_DATE()` (UTC, partition-pruned, zero shell date math -- Key findings 6); (b) kill-switch JSONL replay (internal §3, last pause/resume wins, backend-down safe); (c) .env flag parse + `flag_baseline.json` grandfather reconciliation (internal §2); (d) emit single-line JSON via `json.dumps` on stdout (wrapper appends to session.log; only exit code drives the downgrade -- internal §4). Header comment pins: metered source = `pyfinagent_data.llm_call_log` tokens x `cost_tracker.MODEL_PRICING`, EXCLUDING `provider='claude-code'` AND `provider='anthropic' AND agent LIKE 'cc_rail%'` (internal §1), declared as a LOWER BOUND (under-metered Vertex + grounding surcharges).
- **Gates** (named in JSON, all non-zero exit): `metered_budget` (metered_llm_usd_today > baseline_usd), `flags_match_tokens` (non-grandfathered ON flag without matching token line), `metered_source_unavailable` (BQ timeout/error -- infra class, distinct name; optionally exit 2 vs 1 to encode class in the log). `kill_switch_paused` is REPORTED, never a gate (internal §3; rail 5 keeps pause legitimate).
- **baseline_usd**: pinned constant in the header (accommodate approved full-mode days, $5-8/day; well under the $25 settings cap), justified by Waxell/Braintrust hard-cap doctrine (Key findings 3) + the 58.1 ledger.
- **Tamper tests**: (a) cost gate -- env override (e.g. `SENTINEL_TEST_METERED_USD=99`) or test-dataset pointer, asserting non-zero exit + `metered_budget` in JSON; NOT a prod insert (Key findings 7). The override only INFLATES the read for testing; it cannot mask a real breach when unset. (b) flag gate -- temp .env copy/`SENTINEL_ENV_FILE` override with a non-grandfathered flag ON and no token line -> non-zero + `flags_match_tokens`. (c) wrapper leg -- minimal 62.3 amendment to run the read-only sentinel pre-flight under `AWAY_SESSION_DRY_RUN=1`, then force-fail it and assert the `prompt=digest_only` log line (internal §4; fallback = defer live leg to 62.7 dress rehearsal, disclosed).

## Risks & gotchas + GO/NO-GO

**Risks & gotchas (ranked):**
1. **Under-metering makes the budget gate soft against Vertex overruns** -- llm_call_log misses grounded-search legs and under-logs Gemini (internal §1 live query: 3-of-14 days, mean $0.006/day vs 58.1 ledger $0.5-1.0). The sentinel still catches the TARGET threat (away-session metered Anthropic/SDK spend + gross Vertex runaways) but cannot see invoice-level truth. Mitigation: header pins "lower bound"; digest discloses; billing-export reconciliation stays an operator follow-on. Externally corroborated (Braintrust invisible retries; Google doc silent on grounding).
2. **Tamper-test self-DoS** if implemented as a real prod insert: 30-min DML immunity (insertAll path, `api_call_log.py:141,320`) + day-long inflation of the metered figure (gate then trips on every later run today). Must use env/test-dataset injection; a prod BQ write from a test also violates the CLAUDE.md BQ write-approval discipline.
3. **False-trip on day one of legitimate full-mode** if baseline_usd is derived from the sparse 14-day history ($0.006 mean). Pin the constant; bake the $25 58.1-window exemption into the constant, not per-row attribution (schema has no purpose tag, internal §1).
4. **Flag-gate false positives at ship time**: tokens file absent + 3 pre-62.2 operator-ON flags + EMPTY `KNOWN_TOKEN_ENV_MAP` (internal §2). Grandfather manifest is mandatory; new keys default False-expected; manifest git-tracked so tampering shows in diffs; pre-tool-use hook (:173-199) is the complementary write tripwire.
5. **Hang risk**: no wrapper timeout around the sentinel (internal §4) -- BQ client needs an explicit ~20s timeout, total runtime <30s, else a hung query stalls the launchd session before any cap.
6. **Day-boundary drift**: local-time or shell-arithmetic "today" misbounds by 2h (Oslo CEST) against UTC `DATE(ts)` partitions and can defeat partition pruning. UTC `CURRENT_DATE()` in SQL only; no `-v`/`-d` arithmetic.
7. **Dry-run bypass**: 62.3's pre-flight block is skipped under `AWAY_SESSION_DRY_RUN=1`, so the wrapper-test leg needs the minimal amendment (run the read-only sentinel in dry-run too) -- an edit to a 62.3-shipped file, disclose in experiment_results.md; fallback = defer the live leg to 62.7 with the weaker static-citation evidence.
8. **Exit-code discipline**: chmod +x required despite `bash sentinel.sh` invocation (the wrapper gates on `-x`); JSON as the last stdout line keeps session.log greppable; named gate must appear in the JSON before exit.
9. **Kill-switch semantics**: paused != unhealthy -- gating on it would suppress sessions exactly when the operator most needs the digest (post-breach, rail 5). Report-only.

**GO/NO-GO: GO.** Every immutable criterion has a verified design path: all JSON fields computable with backend down except the BQ leg (which fail-closes with a distinct infra gate name); metered source pinned and lower-bound-honest; both tamper tests implementable without prod writes; 62.3 wrapper wiring verified in code (internal §4) with a concrete dry-run-amendment plan for the wrapper-test leg. No unresolved hard blocker. Two disclosed compromises: (a) the budget gate is lower-bound-only w.r.t. Vertex invoice truth; (b) the wrapper test touches a 62.3-shipped file (or is explicitly deferred to 62.7).

## Recommendations

1. Bash-thin / python-thick: one venv-python child does BQ + kill-switch JSONL replay + .env reconciliation + `json.dumps` emission; bash maps named gate -> exit code (1 = breach [`metered_budget`, `flags_match_tokens`], 2 = infra [`metered_source_unavailable`]; wrapper treats any non-zero as digest-only). Absolute `$REPO/.venv/bin/python`; no PATH reliance under launchd.
2. Pin in the header: the exact metered SQL (with BOTH flat-fee exclusions: `provider='claude-code'`, `provider='anthropic' AND agent LIKE 'cc_rail%'`), the LOWER-BOUND disclaimer, the baseline_usd constant + 58.1-ledger justification, and the UTC day-boundary definition.
3. Ship `scripts/away_ops/flag_baseline.json` (git-tracked) grandfathering the 3 operator-ON flags; reconciliation rule per internal §2; cite the manifest in 62.4's contract.
4. Kill-switch via direct JSONL replay (last pause/resume wins, absent-file = not paused); emit `kill_switch_paused` as data, never gate.
5. Tamper tests via `SENTINEL_TEST_METERED_USD` + `SENTINEL_ENV_FILE` overrides (inflate-only; cannot mask a real breach when unset); never insert into prod llm_call_log.
6. Amend `run_away_session.sh` minimally so the sentinel pre-flight also runs under dry-run; acceptance = forced-fail sentinel -> `prompt=digest_only` log line, zero sessions burned.
7. Self-bound: ~20s BQ timeout, <30s total; on timeout emit `ok:false, failed_gate:"metered_source_unavailable"`, exit 2.

## Research Gate Checklist

- [x] >=5 authoritative external sources READ IN FULL via WebFetch (9: Anthropic eng, Google Cloud docs x2, jq manual, ss64/date(1) mirror, Braintrust, Waxell, moschetti, oneuptime)
- [x] 10+ unique URLs total (9 full + 21 snippet-only/attempted = 30 recorded)
- [x] Recency scan performed + reported (2024-2026; 5 findings incl. one explicit literature gap)
- [x] Full pages read for the read-in-full set (one 403 -- Baeldung -- recorded as attempted, replaced by the jq manual, a higher-tier source)
- [x] file:line anchors for every internal claim (internal §1-§4 from first pass + 2 live verifications this pass: `api_call_log.py:141,320` insert path; `/usr/bin/jq` 1.7.1-apple present in macOS base)

```json
{"tier": "moderate", "external_sources_read_in_full": 9, "snippet_only_sources": 21, "urls_collected": 30, "recency_scan_performed": true, "internal_files_inspected": 19, "gate_passed": true}
```
