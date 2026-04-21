# Claude Docs Alignment -- Consolidated v3 (phase-4.12)

Produced: 2026-04-18. Scope: audit-only. Extends
`handoff/audit/phase-4.11/CONSOLIDATED_REPORT_v2.md` with findings
from the gap-closure phase-4.12 (Test-and-evaluate, Strengthen
guardrails, Prompting tail, Skills tail, MCP remote, Resources,
Managed Agents sub-pages). Net ~22 pages NOT covered in v2.

**v3 is a diff report, not a full rewrite.** All MUST-FIX and
NICE-TO-HAVE items from v2 still apply; items below are ADDITIONS
or refinements based on the new reads.

## Coverage update

- v2 total: ~130 pages.
- v3 adds: **22 pages** (7 Test+Guardrails, 7 Prompting/Skills/MCP/
  Resources, 8 Managed Agents sub-pages).
- Running total: **~152 pages** verified read-in-full.
- URL drift observed:
  - `/strengthen-guardrails/*` 404s; canonical stem is
    `/docs/en/docs/test-and-evaluate/strengthen-guardrails/`.
  - `/test-and-evaluate/reducing-latency` 404s; resolves via
    `docs.anthropic.com/claude/docs/reducing-latency`.
  - `/release-notes/claude-platform` 404s; canonical is
    `/release-notes/overview`.
  - `/managed-agents/cloud-environment-setup` → canonical
    `/managed-agents/environments`; `/container-reference` →
    `/cloud-containers`; `/session-event-stream` →
    `/events-and-streaming`; `/outcomes` → `/define-outcomes`.
  - **"Claude API skill" is NOT a real page** -- skills on Managed
    Agents are filesystem packs; nested LLM calls are
    `callable_agents` (multiagent).
  - Use-case guides index has **4 entries, not 3** -- missed
    "ticket routing" in prior audits.

## Additions to MUST FIX

Append to v2's 20-item list (continuing numbering):

**21. Migrate Claude JSON output to Structured Outputs API.**
     Anthropic's "increase consistency" doc explicitly says "For
     guaranteed JSON schema conformance use Structured Outputs,
     not prompt engineering"; prefill is **NOT SUPPORTED on Opus
     4.7, Opus 4.6, Sonnet 4.6** -- the exact models pyfinAgent
     uses. Grep: 22 `json.loads` calls across 16 agent files with
     ZERO `JSONDecodeError` handlers in
     `multi_agent_orchestrator.py`. This elevates v2's N5 from
     NICE-TO-HAVE to MUST-FIX -- the current pattern is
     doc-unsupported, not merely suboptimal.

**22. Per-call latency instrumentation.** `backend/agents/
     llm_client.py` has 33 token/cost refs and ZERO
     `time.perf_counter`/`latency_ms`/`ttft_ms`/`stream=True`
     references. The "reducing latency" doc prescribes TTFT +
     baseline latency as first-class metrics. Concrete anchors
     from the audit:
     (a) wrap every `client.messages.create` / `generate_content`
         in `time.perf_counter()` deltas; emit `{provider, model,
         input_tokens, output_tokens, latency_ms, ttft_ms}`
         alongside the cost record;
     (b) extend `backend/services/perf_tracker.py` schema with
         `latency_ms` per step;
     (c) persist to BQ `pyfinagent_data.llm_call_log`;
     (d) surface p95 on the Harness tab;
     (e) add `latency_p95_ms` success criterion to
         `.claude/masterplan.json` verification_criteria for
         production-path steps (MAS orchestrator, Slack bot).

**23. Haiku 4.5 harmlessness pre-screen on user-facing surfaces.**
     Zero live references to `jailbreak`/`prompt_injection`/
     `harmlessness` in backend (only one archived phase-3.2
     contract). `backend/slack_bot/assistant_handler.py` forwards
     raw Slack text to the agent pipeline unchecked. "Mitigate
     jailbreaks" doc prescribes a Haiku 4.5 pre-screen with
     Structured Outputs `{is_harmful: boolean, reason: string}`
     before the main call, ethical system prompt with canned
     refusal string, and repeat-offender throttle. Apply to Slack
     assistant_handler + any paper-trader inbound free-text path.

**24. Prompt-leak defences on Slack streaming.**
     `backend/slack_bot/streaming_integration.py` plausibly
     exfiltrates system prompts via "repeat your instructions
     verbatim" attacks. Doc prescribes: (a) regex/keyword
     post-filter on agent output, (b) LLM-based leak detector
     for nuanced cases, (c) periodic red-team audits. Since our
     skill prompts contain proprietary trading logic + risk
     weights + EBITDA-style formulas, zero-defence is below
     baseline.

**25. "I don't know" permission in high-stakes skill prompts.**
     The reduce-hallucinations doc calls this the basic-tier
     first step. Grep shows the phrase appears only in
     CHANGELOG + one archived contract -- ZERO live occurrences
     across 28 `backend/agents/skills/*.md` files. Retrofit the
     top 10 highest-stakes prompts: `risk_judge`,
     `synthesis_agent`, `neutral_analyst`, `quant_model_agent`,
     `scenario_agent`, `moderator_agent`, `deep_dive_agent`,
     `critic_agent`, `bias_detector`, `devils_advocate_agent`.
     Single-sentence retrofit per file.

## Refinements to existing v2 items

**Item 10 (exception classes)** -- the Managed Agents audit
confirms the doc's errors guidance: the Python SDK provides
`anthropic.RateLimitError` / `APIStatusError` /
`APIConnectionError` classes that already carry `retry-after` in
`e.response.headers`. Deleting the 4 string-match retry loops +
adopting `max_retries` fixes items 9 + 10 together and closes the
v3 finding on unsupported prompt-engineering for JSON.

**Item 19 (cron_budget.yaml)** -- refined: phase-4.12's Managed
Agents audit confirms routines are the right surface for slots
6-15 but routines have no `define_outcome`-style approval gate,
so slots 1-5 trading-ops need an EXTERNAL gate (PR, ticket queue,
Slack approve-reply). Decision should be made before phase-10.7
wiring.

**v2 NICE-TO-HAVE N4 (Advisor tool)** -- Claude Platform release
notes confirm it is now public beta (`advisor-tool-2026-03-01`,
2026-04-09). Specifically the phase-4.12 audit recommends piloting
on the MAS Moderator path (Haiku/Sonnet executor + Opus advisor)
as the most cost-sensitive single agent.

**v2 NICE-TO-HAVE N1 (prompt caching on MAS hot path)** -- Release
notes 2026-02-19 shipped **automatic cache-control** -- the API
now caches the last cacheable block without manual breakpoints.
Adopt this variant instead of manual cache_control wiring;
simpler + handles growing conversations automatically.

## Additions to NICE TO HAVE

### Cluster K -- Evaluation discipline (net-new)

K1. **Held-out eval suite.** Stand up
    `backend/backtest/experiments/eval_suite/` with ~200 labelled
    cases per skill (code/string-matched where possible; LLM-
    binary for judgement calls). Re-run on every prompt change.
    Scorecard in TSV parallel to `quant_results.tsv`. Addresses
    "volume over quality" doctrine from `define-success`.

K2. **`{{VARIABLE}}` syntax in skill `.md` files.** Required for
    Workbench Evaluate-tool import. Enables A/B prompt-version
    testing without custom tooling.

K3. **Cross-model eval matrix in `skill_optimizer.py`.**
    Best-practices mandates Haiku/Sonnet/Opus rows. Currently we
    evaluate only the provider configured for that agent.

K4. **Verbatim-quote-first pattern** on `rag_agent`,
    `deep_dive_agent`, `scenario_agent` -- the three agents that
    routinely ingest >20k-token documents. Doc threshold.

K5. **Post-hoc claim verification + `[]` retraction** pass on
    `synthesis_agent` output.

K6. **Best-of-N (N=3) consistency check** on the allocation
    decision endpoint. Disagreement triggers escalation.

### Cluster L -- Prompt engineering polish (net-new)

L1. **Run 4 Claude MAS skills through Console prompt-improver.**
    `bull_agent.md`, `bear_agent.md`, `devils_advocate_agent.md`,
    `moderator_agent.md`. Adds XML-tagged sections + CoT
    scaffolding with no code change. A/B via `skill_optimizer`.

L2. **Rename `backend/agents/skills/`** to `prompt_templates/`,
    OR clarify in CLAUDE.md that these are local templates, not
    Anthropic Agent Skills. Terminology collision risks reviewer
    confusion at go-live.

L3. **Split `quant_strategy.md` (239 L)** and
    **`synthesis_agent.md` (179 L)** into SKILL.md + one-level-
    deep reference files (progressive-disclosure pattern,
    <500-line body guidance).

L4. **Document "MCP connector intentionally unused"** in
    `.claude/rules/backend-agents.md` or `ARCHITECTURE.md` so
    go-live reviewers don't flag. Beta header `mcp-client-
    2025-11-20` is not ZDR-eligible; all our MCP is local stdio
    or in-process FastMCP by design.

### Cluster M -- Managed Agents pilot (refined with concrete scope)

M1. **Submit Managed Agents access-form NOW** requesting memory +
    multi-agent Research Preview access.

M2. **Pilot scope for #1 (CONDITIONAL GO from audit):**
    - ONE masterplan step (propose: phase-4.6-style parameter-
      optimization cycle -- small, well-rubriced, no FastAPI
      call-outs).
    - Orchestrator = managed agent with `callable_agents:
      [qa-evaluator, harness-verifier]` (parallel threads, own
      context windows).
    - Outcome = rubric copied VERBATIM from the step's
      `contract.md` success criteria.
    - No custom MCP; only `agent_toolset_20260401` + the
      built-in GitHub MCP.
    - `networking: limited` with `allowed_hosts: ["api.github.
      com", "api.anthropic.com"]` only.
    - Defer: BigQuery MCP, FastAPI reachability, HMAC migration.

M3. **Blockers for broader rollout (must resolve before pilot #2):**
    - Permission policies are binary (`always_allow`/
      `always_ask`) -- strictly less expressive than our HMAC
      capability tokens. Either (a) shift enforcement to the
      host app's `user.tool_confirmation` loop, or (b) shift
      into each (public) MCP server.
    - In-process FastMCP servers unreachable -- managed MCP is
      remote-HTTPS streamable-HTTP only. Need Cloud-Run-style
      deployment for BigQuery MCP, paper-trading, ticket-queue.
    - FastAPI on `:8000` unreachable from containers (no VPC
      peering). Need public HTTPS endpoint + auth layer.
    - No generic secrets injection. Vaults = MCP auth only;
      `authorization_token` = GitHub-specific. `gcloud` SA JSON
      has no documented channel. Request clarification from
      Anthropic via the access form.

M4. **Opportunity: `define_outcome` as contract.md replacement.**
    Rubric grader runs in a SEPARATE context window -- literal
    structural defense against the "agents confidently praise
    their own work" anti-pattern we already ban. Maps cleanly to
    PASS/CONDITIONAL/FAIL via `satisfied / needs_revision /
    max_iterations_reached / failed / interrupted`. Caveat:
    grader reasoning is OPAQUE ("you see that it's working, not
    what it's thinking") -- we'd lose the transparent
    `evaluator_critique.md` audit trail that our human review
    depends on. Non-trivial tradeoff.

M5. **`session_thread_id` echo** on tool-confirmation + custom-
    tool-result replies when the request originated in a
    subagent. If we adopt multiagent, application code must
    preserve this -- easy to miss and silently breaks threading.

### Cluster N -- Release-notes adoption (net-new deltas since last read)

N1. **Automatic cache-control (2026-02-19).** Drop manual
    `cache_control` breakpoints in `ClaudeClient`; API caches the
    last cacheable block. Complements v2 A1.

N2. **Code execution free with web search/fetch (2026-02-17).**
    If we ever adopt server-side web-search (v2 B4), pair with
    code-execution at no extra cost.

N3. **Web search + programmatic tool calling GA (2026-02-17).**
    Both features moved out of beta since our last read; no
    `anthropic-beta` header needed.

N4. **`ant` CLI + Managed Agents public beta (2026-04-08).**
    Awareness; pilot path (Cluster M above).

N5. **1M-context beta retiring for Sonnet 4.5/4 on 2026-04-30.**
    We're on Sonnet 4.6 which is GA 1M, safe; flag if we ever
    roll back to 4.5.

## Decisions now required (expanded from v2's 5 to 8)

1. Approve the 25 MUST-FIX items (v2 items 1-20 + v3 items 21-25).
2. Pick 3-5 NICE-TO-HAVE clusters for phase-4.13+ scope.
3. Routines-vs-cron decision for phase-10.7 (v2 + v3 refined).
4. Agent SDK adoption: port planner + evaluator only, or defer?
5. **Managed Agents pilot: submit access form now** with the
   narrow scope in M2. Yes/no?
6. **(NEW) Structured Outputs migration priority:** blocking
   go-live (given 22 doc-unsupported `json.loads` sites)?
7. **(NEW) Rename `skills/` -> `prompt_templates/`** before
   go-live review, yes/no?
8. **(NEW) Eval suite (Cluster K) phasing:** phase-4.13 or defer?

All implementation remains blocked pending approval per
audit-only mandate.

## References

- v2 consolidated report: `handoff/audit/phase-4.11/
  CONSOLIDATED_REPORT_v2.md`
- Phase-4.12 topic audits:
  - `handoff/audit/phase-4.12/evals_and_guardrails.md`
  - `handoff/audit/phase-4.12/prompting_skills_mcp_resources.md`
  - `handoff/audit/phase-4.12/managed_agents_gaps.md`
- All 14 topic audits under `handoff/audit/phase-4.10/` and
  `handoff/audit/phase-4.11/`.

Documentation accessed 2026-04-18 across platform.claude.com,
code.claude.com, and docs.anthropic.com (the last hosts the
"reducing latency" page that 404s on platform.claude.com).
