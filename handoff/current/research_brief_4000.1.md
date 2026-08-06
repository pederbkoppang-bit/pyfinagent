# Research Brief -- phase-4000.1 (CC-rail E2E smoke)

Tier: **moderate** (caller-specified). Coverage: audit_class = false.
Started: 2026-08-06. Status: IN PROGRESS (write-first; grows incrementally).

Mission: prove app Claude-role LLM traffic runs E2E on the Claude Code
Max-plan rail (`claude --print` subprocess, flag
`settings.paper_use_claude_code_route`) with the metered Anthropic key
dark, to revive trade throughput.

LENGTH DISCLOSURE: this brief exceeds the nominal `moderate` 700-word guide.
The caller scoped FOUR internal sub-questions (a)-(d) plus five external
topics, and the per-claim file:line citation tables are structurally required
by the gate. Repo norm for briefs is 26-79 KB. Flagging rather than trimming
evidence.

## Search queries run (three-variant discipline)

| Variant | Query |
|---|---|
| Current-year frontier | `Claude Code headless mode --print --output-format json envelope fields modelUsage 2026` |
| Current-year frontier | `Claude Max plan usage limits Claude Code weekly limits 2026` |
| Year-less canonical | `Claude Code CLI reference --print scripting exit codes --model flag` |
| Year-less canonical | `Anthropic "non-interactive usage" claude -p Agent SDK subscription monthly usage credits Max plan` |
| Year-less canonical | `LLM provider failover routing production agent systems gateway reliability` |
| Year-less, domain-scoped | `"Use the Claude Agent SDK with your Claude plan" support.claude.com credits` (allowed_domains: anthropic/claude) |
| Last-2-year window | `LLM multi-agent trading system model routing rail abstraction 2025 TradingAgents arXiv` |

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://code.claude.com/docs/en/headless | 2026-08-06 | Official doc (tier 2) | WebFetch, full | `--bare` "will become the default for `-p` in a future release" and "doesn't use your subscription login" -> latent rail-fatal change (F1). Exit 0/non-zero; in-run failures print to **stdout** (F2). `--output-format json` payload includes `total_cost_usd` + "a per-model cost breakdown" (F3). SIGTERM -> exit 143. |
| 2 | https://code.claude.com/docs/en/cli-reference | 2026-08-06 | Official doc (tier 2) | WebFetch, full | `--model` "accepts an alias ... or a model's full name. Overrides the `model` setting and `ANTHROPIC_MODEL`" (F5). `--max-budget-usd` + `--max-turns` exist, print-mode only (F4). Exit codes 0 success / 1 failure. |
| 3 | https://code.claude.com/docs/en/costs | 2026-08-06 | Official doc (tier 2) | WebFetch, full | On subscriptions the dollar figure is NOTIONAL: "Claude Max and Pro subscribers have usage included in their subscription, so the session cost figure isn't relevant for billing" and it is "computed locally from token counts priced at standard list rates". Cache lifetime "an hour on a subscription ... five minutes once you're drawing on usage credits". `/usage` figures are local-machine-only and it is not offered in `-p`. |
| 4 | https://support.claude.com/en/articles/12429409-extra-usage-for-paid-claude-plans | 2026-08-06 | Official support (tier 2) | WebFetch, full | Usage credits = "consumption-based pricing at standard API rates" past the included limit. "Usage credits apply to both Claude conversations and Claude Code terminal usage." No separate non-interactive pool described here. |
| 5 | https://support.claude.com/en/articles/11145838-use-claude-code-with-your-pro-or-max-plan | 2026-08-06 | Official support (tier 2) | WebFetch, full | "Both Pro and Max plans offer usage limits that are shared across Claude and Claude Code, meaning all activity in both tools counts against the same usage limits." Points at the Agent-SDK article (source 6). `/status` for remaining allocation. |
| 6 | https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan | 2026-08-06 | Official support (tier 2) | WebFetch, full | **DECISIVE (F0).** "Update June 15: We're pausing the changes to Claude Agent SDK usage described below." Paused credit would have covered "The `claude -p` command in Claude Code (non-interactive mode)" at Pro $20 / Max 5x $100 / Max 20x $200. "Teams running shared production automation should use Claude Platform with an API key for predictable pay-as-you-go billing." |

**Measured evidence (not a web source; listed for auditability):** one authorized
`claude --print --output-format json` probe on CLI **v2.1.223**, 2026-08-06 --
see "MEASURED: the CURRENT envelope shape" below. This is the primary evidence
for external topic 1 and supersedes any doc description of the envelope.

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|--------------------------|
| https://code.claude.com/docs/en/agent-sdk/typescript | Official doc | FETCH ATTEMPTED -- page truncated before the `SDKResultMessage`/`ModelUsage` type declarations. Yielded only that `maxBudgetUsd` is "compared against the same estimate as `total_cost_usd`" (corroborates F3/source 3). Superseded by the live probe. |
| https://venturebeat.com/technology/anthropic-reinstates-openclaw-and-third-party-agent-usage-on-claude-subscriptions-with-a-catch | Tech press | FETCH ATTEMPTED -- HTTP 429. Snippet corroborates the pause. |
| https://thenewstack.io/anthropic-agent-sdk-credits/ | Tech press | FETCH ATTEMPTED -- returned newsletter/nav chrome, no article body. |
| https://www.morphllm.com/claude-code-usage-limits | Community | Asserts the non-interactive credit pool as LIVE -- CONTRADICTED by source 6; recorded as the stale claim, not relied on. |
| https://www.explainx.ai/blog/claude-usage-limits-2026-timeline-explained | Community | Same class; secondary tracker. |
| https://www.truefoundry.com/blog/claude-code-limits-explained | Community | Secondary limits tracker. |
| https://ccforeveryone.com/guides/claude-code-limits-and-pricing | Community | Secondary limits tracker. |
| https://techsy.io/en/blog/claude-2x-usage-limits-explained | Community | Secondary limits tracker. |
| https://www.jdhodges.com/blog/claude-ai-usage-limits/ | Community | Secondary limits tracker. |
| https://www.techtimes.com/articles/317625/20260602/anthropic-ends-subscription-subsidy-agents-june-15-credit-pool-replaces-flat-rate-access.htm | Press | Pre-pause reporting (2026-06-02); superseded by source 6. |
| https://zed.dev/blog/anthropic-subscription-changes | Vendor blog | Third-party impact analysis of the same change. |
| https://www.digitalapplied.com/blog/anthropic-claude-credit-overhaul-june-15-2026 | Community | Corroborates the pause. |
| https://devtoolpicks.com/blog/anthropic-splits-claude-subscriptions-agent-sdk-credit-june-2026 | Community | Corroborates amounts. |
| https://claudefa.st/blog/guide/development/agent-sdk-credit | Community | Corroborates amounts. |
| https://portkey.ai/blog/failover-routing-strategies-for-llms-in-production/ | Industry | The rail's own cited canonical (`claude_code_client.py:19`); year-less check found no superseding version. |
| https://www.getmaxim.ai/articles/top-5-llm-failover-routing-gateways-in-2026/ | Industry | Gateway landscape; not applicable (single-rail, local-only deployment). |
| https://www.truefoundry.com/blog/llm-failover-load-balancing-provider-outages | Industry | Failover patterns (retry/hedge/weighted). |
| https://arxiv.org/abs/2412.20138 | Peer-reviewed preprint | TradingAgents -- the rail's cited prior art (`claude_code_client.py:17`). Still the canonical reference; no rail-routing update. |
| https://arxiv.org/html/2509.05080v1 | Preprint | MM-DREX: dynamic routing of LLM experts for financial trading (2025). |
| https://arxiv.org/html/2605.19337v1 | Preprint | "Agentic Trading: When LLM Agents Meet Financial Markets" (2026). |
| https://github.com/TauricResearch/TradingAgents | Code | Reference implementation. |
| https://hidekazu-konishi.com/entry/claude_code_cicd_and_headless_automation.html | Blog | Headless CI patterns; superseded by source 1. |
| https://amux.io/guides/claude-code-headless/ | Blog | Headless self-hosting guide; superseded by source 1. |

## Recency scan (2024-2026)

**Performed.** Explicit last-2-year passes run (see the query table: the 2026
frontier queries plus the `...2025 TradingAgents arXiv` window query), plus
year-less canonical passes to catch prior art the year-locked queries bias away
from.

**Result: 3 new findings in the window that MATERIALLY change the plan, and 1
that confirms no change.**

1. **(2026-06-15, supersedes everything) The `claude -p` billing-model change
   and its pause.** F0 above. This did not exist when the rail was built
   (2026-05-26) or when phase-78.2 measured it (2026-07-25). It is the single
   highest-impact new fact and adds a watch item + a second E6 figure.
2. **(CLI v2.1.223, measured today) Envelope drift.** Seven new top-level keys
   vs the in-code docstring, and the alias-vs-snapshot `canonicalModel`
   collapse (Finding P1). Additive and non-breaking, but it invalidates any
   assumption that the phase-78 envelope sample is still complete.
3. **(2026, doc) `--bare` becoming the `-p` default.** F1 -- a scheduled,
   documented change that would break Max-subscription auth for this rail.
4. **No new finding on the failover/rail-abstraction prior art.** The rail's
   own citations (TradingAgents arXiv:2412.20138, Portkey) remain current;
   2025-2026 work (MM-DREX arXiv:2509.05080, "Agentic Trading"
   arXiv:2605.19337, MASR routing) addresses *which model to pick per task*,
   not *how to fail over between billing rails*, so it does not supersede the
   existing design. pyfinagent is single-rail + local-only
   (auto-memory `project_local_only_deployment`), so the gateway literature's
   hedging/weighted-distribution patterns are explicitly NOT applicable --
   pushing back on fleet-shaped infra is the standing rule.

## Key findings (external)

### F1 -- `--bare` will become the DEFAULT for `-p`, and `--bare` KILLS Max-subscription auth (rail-fatal time bomb)

Anthropic headless doc, accessed 2026-08-06: *"`--bare` is the recommended
mode for scripted and SDK calls, and **will become the default for `-p` in a
future release**."* And: *"Set `ANTHROPIC_API_KEY` before running it, because
bare mode doesn't use your subscription login"* / *"In bare mode, Claude Code
never reads OAuth credentials or the system keychain."*
(https://code.claude.com/docs/en/headless)

The rail's own code already knows `--bare` is forbidden
(`claude_code_client.py:364-369`: *"Do NOT add `--bare` ... --bare rejects
OAuth + keychain reads and requires ANTHROPIC_API_KEY, which would break the
Max-subscription rail"*). What is NEW since that comment was written is that
Anthropic intends to make it the DEFAULT. When that ships, this rail breaks
silently in the worst possible way: `ANTHROPIC_API_KEY` is scrubbed from the
subprocess env at `claude_code_client.py:408-411`, so the CLI would have NO
credential at all and every call would fail -- exactly the "162 doomed calls
per cycle" shape the 66.1 guard was built for. RECOMMENDATION: 4000.1 should
record the observed `claude --version` in the baseline so a future breakage
can be bisected against a CLI upgrade, and 4000.4 should carry a standing
watch item.

### F2 -- exit-code + failure-channel semantics, confirming the code's phase-66.2 hard-won behaviour

Headless doc verbatim: *"Claude Code exits with code 0 on success and a
non-zero code when the run fails ... If you pass an invalid flag, Claude Code
reports the error to stderr before the run starts. **When a failure happens
inside the run, such as missing authentication, Claude Code prints the failure
as the result on stdout.**"* CLI reference: exit `0` success, `1` failure
(incl. not-logged-in, max turns/budget exceeded); SIGTERM aborts the turn and
exits **143**.
(https://code.claude.com/docs/en/headless, https://code.claude.com/docs/en/cli-reference)

This independently CORROBORATES the phase-66.2 comment at
`claude_code_client.py:440-446` (the 07-07 quota burst logged 65 failures with
empty stderr because the diagnostic was on stdout). The smoke's preflight must
therefore treat exit-0 as necessary-not-sufficient and assert the ENVELOPE
shape -- which is what the masterplan text already demands.

### F3 -- the envelope is officially documented to carry a per-model cost breakdown

Headless doc: *"With `--output-format json`, the response payload includes
`total_cost_usd` and a **per-model cost breakdown**, so scripted callers can
track spend per invocation without consulting the usage dashboard."* This is
the documented basis for E6's quota math and confirms `modelUsage`'s map
nature is intentional, not an accident.

### F4 -- `--max-budget-usd` exists and is print-mode only: a HARD in-CLI budget rail the smoke should use

CLI reference verbatim: *"`--max-budget-usd` -- Maximum dollar amount to spend
on API calls before stopping (print mode only). Spend from subagents counts
toward the cap. Once spend reaches the cap, spawning another subagent fails
with `Budget limit reached` ... the cap-enforcement behaviors require Claude
Code v2.1.217 or later."* `--max-turns` likewise exists and *"Exits with an
error when the limit is reached."*
(https://code.claude.com/docs/en/cli-reference)

pyfinagent does NOT pass either flag today (argv built at
`claude_code_client.py:371-385` -- only `--print`, `--output-format`,
`--disallowedTools`, optional `--append-system-prompt`, `--json-schema`,
`--model`). Note `claude_code_client.py:386-393` already records that
`--max-tokens` is an SDK-only option that the CLI rejects, and points at
`--max-budget-usd` as the CLI's actual budget knob. For 4000.2's "<=30 rail
calls" budget this is a defence-in-depth option (a per-call ceiling in
addition to the call counter), but note it caps DOLLARS, and on the
subscription rail the dollar figure is notional -- so the call counter stays
the primary control.

### F0 -- HEADLINE: `claude -p` was scheduled to move OFF the flat-fee subscription onto a capped monthly credit. Anthropic PAUSED it on 2026-06-15. The pause is the only thing making this goal's premise true today.

This is the single most decision-relevant external finding and it is NOT in
the goal draft. Anthropic's own support article
(https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan,
accessed 2026-08-06) carries a dated notice at the top, verbatim:

> "Update June 15: We're pausing the changes to Claude Agent SDK usage
> described below."

The change that is paused would have moved these surfaces onto a separate
monthly credit, verbatim from the same page:

> "Claude Agent SDK usage in your own projects (Python or TypeScript)" +
> "**The `claude -p` command in Claude Code (non-interactive mode)**" +
> "The Claude Code GitHub Actions integration" + "Third-party apps that
> authenticate with your Claude subscription"

`claude -p` is EXACTLY what `claude_code_client.py:371-376` shells. The
pre-announced credit amounts were **Pro $20 / Max 5x $100 / Max 20x $200**
per month, billed at standard API list rates. The same page states the
current state:

> "Interactive Claude Code in the terminal or your IDE continues to use your
> subscription usage limits exactly as before."

and today the paused-state means non-interactive draws from the subscription
too. Anthropic's own guidance for this exact use case is a warning shot:

> "Teams running shared production automation should use Claude Platform with
> an API key for predictable pay-as-you-go billing."

WHY THIS MATTERS TO 4000: the whole goal rests on "the Max plan is flat-fee
and already paid for". That is TRUE TODAY and was verified against the
primary source this session. But it rests on a **paused**, already-drafted,
already-priced change with named dollar figures -- not on a stable policy.
If the pause lifts, pyfinagent's rail traffic silently starts drawing a
$100-$200/mo metered-at-list-rates pool, which breaks the standing away-ops
`$0 metered` constraint (auto-memory `project_away_ops_plan`) with no code
change and no log line -- the same silent-re-tiering failure class as
phase-78.2.

CONCRETE CONSEQUENCE FOR SECTION (f): E6 should be expressed as TWO numbers
under one stated rule, not one -- (i) %-of-weekly-Max-pool per day (the
operator's current question) and (ii) the same extrapolated `costUSD` as a
%-of-$100-monthly-Agent-SDK-credit (the Max-5x figure; state the plan tier
assumption explicitly). Number (ii) costs nothing extra to compute from the
same envelopes and is the pre-registered answer to "what happens if the pause
lifts". Recommend 4000.4 carry an explicit watch item on this URL.

NOTE ON SOURCE CONFLICT (see "Consensus vs debate"): secondary trackers still
describe the credit pool as LIVE. They are wrong / stale. The primary source
above is authoritative and dated.

### F5 -- `--model` accepts aliases OR full ids (confirms the phase-78.2 threading is well-formed)

CLI reference verbatim: *"Sets the model for the current session with an alias
for the latest model (`sonnet`, `opus`, `haiku`, or `fable`) or a model's full
name. **Overrides the `model` setting and `ANTHROPIC_MODEL`**."* This confirms
the precedence chain documented in the `model` arg docstring at
`claude_code_client.py:322-334` and that passing `--model` closes off the
`~/.claude/settings.json` leak that phase-78.2 measured.

## Internal code inventory

### (a) THE CC-RAIL ROW MARKER -- headline finding: it is THREE shapes, not one

There is no single "provider" that identifies a rail row. **Three distinct
writers** stamp rail traffic into `pyfinagent_data.llm_call_log`, and they
disagree on `provider`:

| # | Writer (file:line) | provider | agent | Reaches |
|---|---|---|---|---|
| W1 | `backend/agents/claude_code_client.py:656-668` (`ClaudeCodeClient._log_cc_call`) | `"anthropic"` | `f"cc_rail:{agent}"` if `_role`/`_agent` set, **else bare `"cc_rail"`** (ternary at :659) | the 6 signal overlays + all Layer-1 orchestrator traffic routed through `make_client` |
| W2 | `backend/services/autonomous_loop.py:2347-2358` (`_log_claude_code_call`) | `"claude-code"` | caller-supplied, e.g. `"lite_trader"` | lite trader (B1) + lite risk judge (B2) |
| W3 | `backend/services/ticket_queue_processor.py:228-238` (`_meter_rail`) | `"anthropic"` | `f"cc_rail:ticket_{agent_id}"` | ticket-queue agents |

`log_llm_call` itself is at `backend/services/observability/api_call_log.py:221`;
the row dict it builds is at `:279-294` (`ts, provider, model, agent,
latency_ms, ttft_ms, input_tok, output_tok, cache_creation_tok,
cache_read_tok, request_id, ok, ticker, cycle_id, session_cost_usd`).
There is **no per-call cost column** -- `session_cost_usd` is a per-cycle
cumulative GAUGE (`api_call_log.py:256-266`; never SUM it).

**Do not derive the rule yourself -- the codebase already contains the
canonical enumeration, and it is load-bearing.** `backend/services/
observability/spend.py:26-39` states verbatim: *"The rail produces exactly
THREE row shapes, all of which this query must exclude (phase-75.5.12;
derived from the writers, not assumed)"*, listing (a) `provider='claude-code'`
with arbitrary agent, (b) `provider='anthropic', agent='cc_rail:<role>'`,
(c) `provider='anthropic', agent='cc_rail'` (BARE). Critically it adds:
*"Shape (c) is the DOMINANT production shape, not an edge case ...
orchestrator.py:826-835 sets only `_ticker` and never `_role` ... Measured
30d 2026-07-25: 2,549 bare-'cc_rail' calls (~4.87M tokens) vs 7 in the colon
shape."*

**E1 row-selection RULE (executable verbatim as a WHERE clause) -- the exact
logical complement of the spend.py exclusion at `spend.py:228-230`:**

```sql
-- A row IS a CC-rail (flat-fee Max) row iff:
provider = 'claude-code'
OR agent = 'cc_rail'
OR agent LIKE 'cc_rail:%'
```

and a row is **Anthropic-direct (metered)** iff the negation holds while
`provider = 'anthropic'`:

```sql
provider = 'anthropic'
AND (agent IS NULL OR (agent != 'cc_rail' AND agent NOT LIKE 'cc_rail:%'))
```

TRAPS, each measured this session:
1. `provider` alone CANNOT separate the rails -- W1/W3 write
   `provider='anthropic'`, the same string the metered SDK client writes.
   A naive `provider='claude-code'` rule silently misses the DOMINANT
   shape and would report a false "rail not used".
2. Do NOT simplify to `agent LIKE 'cc_rail%'`. `spend.py:37-38` records the
   exact `!=` was chosen over a prefix ON PURPOSE (a prefix would swallow a
   future `cc_railway`); `claude_code_client.py:631-637` repeats the warning
   and notes an earlier revision of that very comment paraphrased it wrongly.
3. `model` is NOT a marker: `_log_cc_call` logs the **resolved** model
   (`resolve_rail_model`, `claude_code_client.py:215-294`, :649) which is a
   normal `claude-*` id -- identical in shape to a metered row's model.
4. DOC DRIFT: `spend.py:30` cites `claude_code_client.py:504` for shape (b);
   the ternary actually lives at **:659** today. Re-derive line numbers.
5. Rail-guard skips write **NO row at all** (`claude_code_client.py:733-743`,
   deliberate -- avoids phantom-row spam), so "0 rail rows" is ambiguous
   between "rail off" and "rail blocked". E5 (guard status) disambiguates.

### (b) Rail call sites reachable from the RUNNING app + cheapest bounded entrypoint

Two mechanisms reach the CLI. Both are gated on the same flag.

**Mechanism 1 -- via `make_client` (the `ClaudeCodeClient` adapter).** Seam at
`backend/agents/llm_client.py:2108-2131`: `if model_name.startswith("claude-")
and settings.paper_use_claude_code_route` -> `ClaudeCodeClient(model_name,
timeout_s=claude_code_timeout_s)`. **Note the `claude-` prefix condition** --
Gemini-configured pipeline steps do NOT become rail calls, so a full analysis
produces rail traffic only for the Claude-pinned roles. Fallthrough to
Anthropic-direct while the flag is ON raises a hard routing-breach
`ValueError` at `llm_client.py:2148-2157` (a second breach guard exists on
`advisor_call`). Reaches the six overlays, which each carry the same phase-78.1
"CC rail needs NO Anthropic key" note: `meta_scorer.py:204`,
`news_screen.py:262`, `pead_signal.py:273`, `macro_regime.py:502`,
`analyst_narrative_scorer.py:107`, `call_transcript_gpr.py:96`.

**Mechanism 2 -- direct `claude_code_invoke` calls** (bypass the adapter, so
they do NOT get the rail-guard check in `generate_content`):
`backend/services/autonomous_loop.py:2498/:2503` (lite trader) and `:2581/:2586`
(lite risk judge); `backend/services/ticket_queue_processor.py:241` (ticket
agents, flag-gated at `:195`).

**Recommended cheapest bounded entrypoint: `POST /api/analysis/`**
(`backend/api/analysis.py:350-351` `start_analysis`), single ticker. It is
async -- it returns a `task_id` and the caller polls
`GET /api/analysis/{analysis_id}` (`analysis.py:384-385`); the smoke must poll,
not assume synchronous completion. The work runs through `_run_sync_analysis`
(`analysis.py:43`) into the orchestrator and hence `make_client`.

**REJECTED alternative -- the lite path.** `_run_claude_analysis`
(`autonomous_loop.py` ~:2430-2460, rail-selection log at :2453-2461) is
reachable ONLY by running a full autonomous paper-trading cycle. A cycle is
budgeted at `paper_cycle_max_seconds=7200` (`settings.py:33`), and that field's
own description records the reason: *"cycle 6 (2026-05-26) found the Claude
Code CLI rail ... ~30s per claude_code_invoke ... push a 13-ticker full-
orchestrator cycle past 3600s"*. That is neither cheap nor bounded, and it
would trade real money. Do not use it for the smoke.

**OPEN RISK the executor MUST measure, not assume:** the rail-call count for
one `POST /api/analysis/` is NOT knowable from code alone -- it depends on how
many pipeline roles are currently pinned to `claude-*` models. The masterplan's
`<=30 rail calls` budget could be breached by a single full analysis. 4000.2's
counter must therefore abort mid-analysis, and 4000.1 should record the
observed count from the `--dry` pass before the live window opens.

### (c) How `/api/settings` GET and PUT authenticate for a localhost curl

Router: `backend/api/settings_api.py:17`, `APIRouter(prefix="/api/settings")`.
It is **NOT** in the auth middleware's `_PUBLIC_PATHS` allowlist (the exact
list is enumerated in `.claude/rules/security.md`), so both verbs are behind
the auth middleware. A localhost curl authenticates through the dev rail at
`backend/api/auth.py:135-153`: `get_current_user` returns
`{"email": "dev@localhost", "localhost_bypass": True}` iff **BOTH**
`os.getenv("DEV_LOCALHOST_BYPASS") == "1"` **AND** `request.client.host` is in
`("127.0.0.1", "::1", "localhost")`. Both conditions required -- the smoke must
run on the host, and 4000.1 must MEASURE that `DEV_LOCALHOST_BYPASS=1` is
actually set in the RUNNING process's env (the researcher sandbox is denied
`backend/.env`; delegate that read to Main).

Three traps found by reading the handlers:

1. **GET is CACHED for 300 s.** `settings_api.py:393-398` reads/writes cache
   key `"settings:full"` with `ENDPOINT_TTLS["settings:full"] = 300.0`
   (`backend/services/api_cache.py:136`). A GET is therefore a read of a
   <=5-minute-old snapshot, not necessarily the live value. For baseline (a)
   this is fine because nothing else is writing, but the smoke's post-window
   GET must follow the PUT (which invalidates) or wait out the TTL.
2. **PUT WRITES `backend/.env`.** `update_settings` (`settings_api.py:402-403`)
   loops the updates and calls `_update_env_var(env_key, env_value)`
   (`:437-447`); `_update_env_var` at `:316-331` regex-substitutes or appends
   the key in the .env file. The mapped key is
   `"paper_use_claude_code_route": "PAPER_USE_CLAUDE_CODE_ROUTE"`
   (`settings_api.py:307`). So the goal draft's *"flip the flag ON via the
   real API, never by editing .env"* is subtler than it reads: **the API's own
   implementation edits .env.** The real distinction is that the API path ALSO
   runs `get_settings.cache_clear()` + `get_api_cache().invalidate("settings:*")`
   (`:449-452`), which a manual .env edit does not -- and a manual edit
   therefore produces exactly the **"lru_cache desync"** named verbatim in the
   routing-breach error text at `llm_client.py:2155-2156`.
3. **THE FLIP IS DURABLE ACROSS A BACKEND RESTART**, because it is persisted to
   .env. A crashed smoke leaves the rail ON permanently, not until the next
   restart. This is the strongest possible argument for 4000.2's
   restore-in-`finally` requirement, and the restore must go through PUT (not
   a .env edit) so the cache is cleared too.

### (d) `financial_reports.paper_trades` columns for the cadence query

`_pt_table` at `backend/db/bigquery_client.py:548-549` returns
`f"{gcp_project_id}.{bq_dataset_reports}.{name}"` -- confirming the masterplan's
claim that the paper tables live in `financial_reports`, not `pyfinagent_pms`.

Schema from the migration `scripts/migrations/migrate_paper_trading.py:58-68`:

| Column | Type | Mode |
|---|---|---|
| `trade_id` | STRING | REQUIRED |
| `ticker` | STRING | REQUIRED |
| `action` | STRING | REQUIRED |
| `quantity` | FLOAT64 | REQUIRED |
| `price` | FLOAT64 | REQUIRED |
| `total_value` | FLOAT64 | NULLABLE |
| `transaction_cost` | FLOAT64 | NULLABLE |
| `reason` | STRING | NULLABLE |
| `analysis_id` | STRING | NULLABLE |
| `risk_judge_decision` | STRING | NULLABLE |
| **`created_at`** | **STRING** | REQUIRED |

**TRAP 1 -- `created_at` is a STRING, not a TIMESTAMP** (`:68`). This is the
known repo defect class (auto-memory
`reference_vacuous_type_guards_on_bq_string_columns`: `historical_macro.date`
is STRING so an `isinstance(v, date)` guard never fires). The trailing-4-week
date filter in (d) must therefore be either a lexicographic comparison on an
ISO-8601 string or an explicit `PARSE_TIMESTAMP`/`SAFE_CAST` -- and the stated
rule must say which. The repo's own reader already does the string form:
`get_paper_trades` binds `since` as `ScalarQueryParameter("since", "STRING",
since_iso)` and orders `created_at DESC` (`bigquery_client.py:711-735`).

**TRAP 2 -- the writer DROPS None-valued columns.** `save_paper_trade`
(`bigquery_client.py:693-708`) begins `row = {k: v for k, v in row.items() if
v is not None}` and builds an INSERT from the surviving keys only, so nullable
columns are absent per-row. Any pairing rule must be NULL-safe.

**TRAP 3 / BIGGEST FIND -- a round-trip table may ALREADY EXIST; do not invent
a pairing rule before checking.** `scripts/migrations/add_round_trip_schema.py:66-75`
creates a schema with `round_trip_id` (REQUIRED), `ticker` (REQUIRED),
`buy_trade_id`, `sell_trade_id`, `entry_date` (TIMESTAMP), `exit_date`
(TIMESTAMP), `entry_price`, `exit_price`, `quantity`, `realized_pnl_usd`.
If that table is populated, the round-trips/week half of (d) is a direct
`COUNT(*)` on `exit_date`, not a self-join heuristic on `paper_trades`.
The executor MUST check whether it exists and is populated (`describe-table`
+ a bounded count) BEFORE writing a pairing rule -- per the standing
verify-never-rebuild rule. Note the two halves then use DIFFERENT date columns
(`created_at` STRING vs `exit_date` TIMESTAMP), which auto-memory
`feedback_normalization_rule_must_be_stated_with_the_ratio` makes a hard
requirement to state explicitly: **write down the one window rule and show
both counts under it**, including how the STRING/TIMESTAMP difference is
reconciled.

Useful existing helper: `get_paper_trades_in_window(window_days)` at
`bigquery_client.py:982-995`. Only writer of paper_trades:
`backend/services/paper_trader.py:1474` and `:1479`.
Live schema may exceed the migration (cf. `add_sector_to_paper_positions.py`)
-- confirm with `describe-table`, do not trust the migration alone.

## MEASURED: the CURRENT envelope shape (one authorized probe, 2026-08-06)

The step permits *"at most ONE probe invocation of the claude CLI to capture a
current envelope shape sample"*. Executed exactly once, mirroring production
argv (`--print --output-format json --disallowedTools "Bash,Edit,Write,Read,
Glob,Grep,Agent" --model claude-haiku-4-5`, prompt via stdin, with
`ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN` unset per
`claude_code_client.py:408-411`). Raw capture retained at
`/private/tmp/claude-501/-Users-ford--openclaw-workspace-pyfinagent/504c8eb0-efe8-436d-937a-903936d9a2d6/scratchpad/probe.json`.

- **CLI version: `2.1.223 (Claude Code)`**. Binary `/Users/ford/.local/bin/claude`
  (the first `_DEFAULT_SEARCH_PATHS` candidate, `claude_code_client.py:49-54`).
- exit 0, empty stderr, `subtype: "success"`, `is_error: false`,
  `stop_reason: "end_turn"`, `num_turns: 1`.

**Top-level keys observed (21):** `api_error_status, duration_api_ms,
duration_ms, fast_mode_disabled_reason, fast_mode_state, is_error, modelUsage,
num_turns, permission_denials, result, session_id, stop_reason, subtype,
terminal_reason, time_to_request_ms, total_cost_usd, ttft_ms, ttft_stream_ms,
type, usage, uuid`.

**ENVELOPE DRIFT since the docstring at `claude_code_client.py:351-358`:** SEVEN
keys now present that the docstring does not list -- `api_error_status`,
`fast_mode_disabled_reason`, `fast_mode_state`, `permission_denials`,
`terminal_reason`, `time_to_request_ms`, `ttft_stream_ms`. The drift is
ADDITIVE, so the parser (which reads only known keys) is unaffected -- but
`structured_output` was absent here because no `--json-schema` was passed, so
4000.2 must not assert its unconditional presence.

**`modelUsage` verbatim (the E3-critical capture):**

```json
{
 "claude-haiku-4-5": {
  "inputTokens": 9, "outputTokens": 42, "cacheReadInputTokens": 0,
  "cacheCreationInputTokens": 45580, "webSearchRequests": 0,
  "costUSD": 0.091379, "contextWindow": 200000, "maxOutputTokens": 32000,
  "canonicalModel": "claude-haiku-4-5", "provider": "firstParty"
 },
 "claude-haiku-4-5-20251001": {
  "inputTokens": 3570, "outputTokens": 1591, "cacheReadInputTokens": 148551,
  "cacheCreationInputTokens": 28776, "webSearchRequests": 0,
  "costUSD": 0.062350100000000006, "contextWindow": 200000,
  "maxOutputTokens": 32000, "canonicalModel": "claude-haiku-4-5",
  "provider": "firstParty"
 }
}
```

### FINDING P1 -- `modelUsage` is a map, and TWO KEYS CAN SHARE ONE `canonicalModel`. `resolve_rail_model` COLLAPSES them.

Both entries above carry `canonicalModel: "claude-haiku-4-5"`. Now read
`resolve_rail_model` at `claude_code_client.py:274-277`:

```python
named: dict = {}
for key, entry in usage_map.items():
    named[(entry.get("canonicalModel") or key)] = entry
```

Two distinct map keys collapse onto ONE dict key, and **the last one iterated
wins** -- here the `...-20251001` entry (costUSD 0.0624) silently displaces the
higher-cost entry (costUSD 0.0914) BEFORE `max(..., key=_weight)` ever runs at
`:291`. In this alias-vs-snapshot case the surviving label is still correct by
luck, so E3's model-name answer is unaffected. But the consequence is sharp and
must be pre-registered in section (f):

**Any cost/token total derived from `resolve_rail_model`'s collapsed map
UNDER-COUNTS.** E6 must sum the RAW `modelUsage` map, never the collapsed one.
This is a genuinely new instance of the phase-78 "never take the first key"
defect class -- the earlier fix addressed *which key you pick*; this is *how
many keys survive to be picked from*. It is a correctness note on the E6
definition, not a live-trading defect (the logger only wants a label).

### FINDING P2 -- `total_cost_usd` == the SUM over ALL `modelUsage` entries (arithmetic identity the smoke can assert)

`0.091379 + 0.0623501 = 0.1537291` == `total_cost_usd: 0.15372910000000004`
(float noise only). Reading only the FIRST key yields `$0.0914` = **59% of the
truth**. This gives 4000.2 a free, self-checking E3/E6 assertion:
`abs(total_cost_usd - sum(e["costUSD"] for e in modelUsage.values())) < 1e-6`.
A fixture that reads one key fails it -- exactly the mutation the masterplan's
E3 criterion demands.

### FINDING P3 -- the per-call FIXED OVERHEAD is ~45.6K cache-creation tokens and ~$0.15, for a 9-token prompt. This dominates E6.

The probe prompt was 9 input tokens ("Reply with exactly the word: ok") and
still booked `cacheCreationInputTokens: 45580` on the first entry (plus 28,776
+ 148,551 cache-read on the second). Cause: the rail passes **no `--bare`**
(deliberately -- see F1), so per the headless doc *"`claude -p` loads the same
context an interactive session would, including anything configured in the
working directory or `~/.claude`"*: CLAUDE.md, hooks, skills, plugins, MCP
servers, auto-memory. `claude_code_invoke` passes `cwd` through unset from the
production adapter (`claude_code_client.py:747-760` never sets it), so the
subprocess inherits the BACKEND PROCESS's cwd -- if that is the repo root, every
rail call re-loads pyfinagent's own CLAUDE.md and `.mcp.json` servers.

Consequences to pre-register:
- E6's per-call cost is **overhead-dominated, not prompt-dominated**. Estimating
  from prompt length alone will be wrong by an order of magnitude.
- The `/docs/en/costs` cache-lifetime rule bites: *"The lifetime is an hour on a
  subscription and drops to five minutes once you're drawing on usage credits."*
  Rail calls spaced more than an hour apart (i.e. across cycles) each re-pay the
  full cache-creation cost. E6 must state the assumed intra-cycle vs inter-cycle
  cache-hit posture.
- 4000.1 baseline (b) should ALSO record the running backend's **cwd**, since it
  determines the overhead. This is a cheap addition to the process capture the
  masterplan already asks for.
- A legitimate future optimisation (NOT this phase) is a dedicated minimal cwd
  for rail calls. Out of scope; note it and move on.

## Application to pyfinagent -- 4000.1 sections (a)-(f)

- **(a) live flag value.** GET `http://127.0.0.1:8000/api/settings/` (trailing
  slash; router prefix `settings_api.py:17` + route `"/"` at `:389`). Auth via
  `DEV_LOCALHOST_BYPASS=1` + 127.0.0.1 (`auth.py:150-153`) -- verify the env var
  is set in the RUNNING process. Remember the 300 s cache
  (`api_cache.py:136`). Code default is `False` (`settings.py:176`), but .env
  may override it -- measure.
- **(b) process start time + cwd.** Add **cwd** to the capture (Finding P3). The
  phase-78.2 change to compare against is the `model=self.model_name` line at
  `claude_code_client.py:759`; if the running process predates it, the rail runs
  the CLI's session-default model and the smoke would measure pre-78.2 behaviour.
- **(c) rail row rule.** Use the three-shape WHERE clause in section (a) above.
  It is the exact complement of `spend.py:228-230`; do NOT re-derive it as a
  single-provider test.
- **(d) cadence.** Check `paper_round_trips` FIRST (Trap 3). State one window
  rule; reconcile `created_at` STRING vs `exit_date` TIMESTAMP explicitly.
- **(e) entrypoint.** `POST /api/analysis/` + poll; record the measured rail-call
  count from the dry pass (the `<=30` budget is at genuine risk).
- **(f) E1-E7 definitions.** Adopt: E1 = the three-shape rule, reported N-of-N;
  E2 = metered delta via the SAME exclusion `fetch_llm_spend` uses
  (`llm_client.py:425-440`; note `cost_budget_use_llm_spend_enabled` defaults
  OFF, so confirm which metric is live) -- never SUM `session_cost_usd`;
  E3 = iterate the RAW `modelUsage` map + assert the P2 sum identity;
  E6 = TWO figures (%-weekly-Max-pool/day AND %-of-$100-Agent-SDK-credit per F0),
  computed from the raw map, with the P3 overhead and cache posture stated.

## Consensus vs debate (external)

**One live contradiction, resolved.** Secondary trackers (morphllm, explainx,
truefoundry, ccforeveryone) state as CURRENT fact that non-interactive usage --
"Agent SDK, `claude -p`, GitHub Actions, third-party apps on your subscription"
-- "draws from a separate monthly credit: $20 on Pro, $100 on Max 5x, $200 on
Max 20x". Anthropic's own dated notice
(https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan,
accessed 2026-08-06) says the change is **paused as of 2026-06-15** and that
those surfaces still draw from subscription limits.

RESOLUTION: the primary, dated, first-party source wins; the trackers are
reporting the announcement, not the pause. Corroborated independently by
digitalapplied, devtoolpicks and the VentureBeat headline ("reinstates ...
with a catch"). **Practical upshot: the goal's flat-fee premise is TRUE today
and CONDITIONAL tomorrow** -- do not write "the Max plan is flat-fee" into the
contract as a standing fact; write it as a fact verified on 2026-08-06 against
a paused policy, with the watch URL.

**Consensus elsewhere:** every source agrees Claude Code CLI usage shares ONE
pool with claude.ai chat, that switching models does not restore a hit window,
and that subscription dollar figures are locally-computed estimates at list
rates rather than bills.

## Pitfalls (from literature + docs)

1. **Exit 0 is not success; success is `subtype == "success"`.** The headless
   doc confirms in-run failures (including missing auth) are printed as the
   RESULT on stdout. The repo already encodes this
   (`claude_code_client.py:475-483` checks `subtype`, and the docstring at
   `:357-358` warns `is_error` "has known mis-flag history"). The smoke's
   preflight must not accept exit-0 alone -- the masterplan already says so;
   this is the external corroboration.
2. **Reading only the first `modelUsage` key under-reports by ~41%** in the
   measured probe. Sum all entries (Finding P2).
3. **Cost figures on a subscription are notional.** Never present
   `total_cost_usd` as spend; present it as a pool-consumption proxy.
4. **Cache-miss cliff.** Subscription cache lifetime is one hour; rail calls
   spaced further apart re-pay the full ~45K-token context load.
5. **Gateway-literature overreach.** The failover literature's hedging /
   weighted-distribution patterns assume a fleet; this deployment is one Mac
   (auto-memory `project_local_only_deployment`). Do not import them.

## Risks / gotchas for the smoke

| # | Risk | Evidence | Mitigation for 4000.2/4000.3 |
|---|---|---|---|
| R1 | **Envelope drift** -- 7 new top-level keys since the in-code docstring; `structured_output` present only with `--json-schema`. | Probe, CLI v2.1.223 | Assert on the keys you USE, never on an exact key set; record `claude --version` in the baseline so a future break bisects to a CLI upgrade. |
| R2 | **`--bare` becoming the `-p` default would kill the rail** (no OAuth/keychain, and the rail scrubs `ANTHROPIC_API_KEY`). | Source 1; `claude_code_client.py:364-369`, `:408-411` | Standing watch item in 4000.4; the 66.1 breaker + probe gate already contain the blast radius to one cycle. |
| R3 | **Billing-model change un-pauses** -> rail traffic moves to a $100/$200 metered credit, breaking `$0 metered`. | Source 6 (F0) | Add the %-of-$100-credit figure to E6; watch item; kill criterion. |
| R4 | **Auth** -- `/api/settings` needs BOTH `DEV_LOCALHOST_BYPASS=1` AND a 127.0.0.1 client. | `auth.py:150-153` | Measure the running process's env (Main must read `backend/.env`; researcher sandbox is denied it). Fail the preflight loudly, not with a 401 mistaken for "backend down". |
| R5 | **GET /api/settings is 300 s cached** -> a "live value" read can be stale. | `settings_api.py:393-398`, `api_cache.py:136` | Take the baseline GET before anything writes; take the post-window GET after the restore PUT (which invalidates). |
| R6 | **The flag flip is DURABLE across restart** (PUT writes `backend/.env`). A crashed smoke leaves the rail ON forever. | `settings_api.py:437-447`, `:316-331`, `:307` | restore-in-`finally` is mandatory; restore via PUT (not a .env edit) so the lru_cache is cleared. |
| R7 | **Manual .env edits cause lru_cache desync** -> the routing-breach `ValueError`. | `llm_client.py:2148-2157` | Never edit .env directly during the window. |
| R8 | **Rail-call budget may be breached by ONE analysis.** The `<=30` cap is not derivable from code. | `analysis.py:350`; `settings.py:33` (~30s/call, 13-ticker cycle >3600s) | Counter must abort mid-analysis; record the measured count in the dry pass first. |
| R9 | **Zero rail rows is ambiguous** (rail off vs guard-blocked -- skips write no row). | `claude_code_client.py:733-743` | E1 must be read together with E5's `rail_guard_status()`. |
| R10 | **Concurrency** -- another session draining the masterplan; the auto-commit hook stages the whole tree. | Phase CONCURRENCY RAIL; auto-memory `feedback_audit_the_commit_not_the_diff` | `git add -An` before every flip; single-writer window for 4000.3. |
| R11 | **Latency** -- the probe took 36 s for a 9-token prompt; production budgets 150 s/call. | Probe; `settings.py` `claude_code_timeout_s=150` | Size the smoke's own timeouts above 150 s; do not mistake slowness for a hang. |
| R12 | **E2 metric ambiguity** -- two different spend metrics exist and the selector defaults OFF. | `llm_client.py:425-440`; `cost_budget_use_llm_spend_enabled` | State WHICH metric E2 reads and confirm the live flag value; never SUM `session_cost_usd` (a gauge). |

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **6**, all
      tier-2 official Anthropic docs/support (no community-tier source counted).
- [x] 10+ unique URLs total -- **29** (6 full + 3 attempted + 20 snippet-only).
- [x] Recency scan (last 2 years) performed + reported -- 3 material new
      findings + 1 explicit no-change.
- [x] Full pages read (not abstracts) for the read-in-full set.
- [x] file:line anchors for every internal claim.

Soft checks:
- [x] Internal exploration covered every module the caller named (a)-(d), plus
      the routing seam, spend metric, migrations and the three rail writers.
- [x] Contradictions noted and resolved with a primary source (see Consensus).
- [x] Claims cited per-claim inline, not in a footer.
- [~] Brief length exceeds the `moderate` guide -- disclosed at the top.
- [~] Not verified (out of researcher scope / needs Main or the executor):
      whether `DEV_LOCALHOST_BYPASS=1` is set in the running backend; whether
      `paper_round_trips` is populated; the running backend's cwd and start
      time; the live value of `paper_use_claude_code_route`. All four are
      MEASUREMENT tasks 4000.1 already assigns -- flagged so nobody mistakes
      this brief for having done them.

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 23,
  "urls_collected": 29,
  "recency_scan_performed": true,
  "internal_files_inspected": 17,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 3,
    "dry": false
  },
  "summary": "Six official Anthropic docs read in full plus one authorized live CLI probe (v2.1.223). HEADLINE: Anthropic announced then PAUSED (2026-06-15) moving `claude -p` off the flat-fee subscription onto a capped monthly credit (Max 5x $100) -- the goal's flat-fee premise is true today but rests on a paused policy, so E6 needs a second figure (%-of-$100-credit) and a watch item. `--bare` is documented to become the `-p` default and it disables subscription auth -- rail-fatal, latent. Probe measured: modelUsage is a map whose keys can share one canonicalModel, which `resolve_rail_model` COLLAPSES (last-wins), so E6 must sum the RAW map; total_cost_usd equals the sum over all entries (first key alone = 59%), giving a free assertion. Per-call overhead is ~45.6K cache-creation tokens / ~$0.15 for a 9-token prompt because the rail deliberately omits --bare. Internally: the CC-rail row marker is THREE shapes, not one (provider='claude-code' OR agent='cc_rail' OR agent LIKE 'cc_rail:%'); paper_trades.created_at is STRING; a paper_round_trips table may already exist; PUT /api/settings writes .env so the flip survives restart.",
  "brief_path": "handoff/current/research_brief_4000.1.md",
  "gate_passed": true
}
```
