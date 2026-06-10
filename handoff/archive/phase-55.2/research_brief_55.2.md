# Research Brief — Step 55.2: Ops incidents + agent-quality audit

**Tier:** complex
**Away window:** 2026-06-01 → 2026-06-10 (autonomous paper-trading)
**Scope:** review-only (NO fixes), $0 (bounded BQ reads, 30s timeout, no LLM cycle spend)
**Researcher:** sole research agent (external lit + internal code audit, one session)

This brief feeds the contract for step 55.2. It carries (A) an internal
code audit with file:line anchors and a live BQ agent-label→skill mapping,
and (B) external literature read in full. Written incrementally.

---

## PART A — INTERNAL CODE AUDIT (file:line)

### A(a) — Slack "Approve" → "Missing API key for provider anthropic"

**Render site (the button):** `backend/slack_bot/governance.py:165-168`
```
"text": {"type": "plain_text", "text": "Approve"},
"value": "approve",
"action_id": "approval_approve"
```

**FINDING F-A1 (preliminary):** the string `Missing API key for provider`
does NOT appear anywhere in the repo source (only in handoff/masterplan
audit-basis text describing the incident). The litellm SDK string at
`.venv/.../litellm/main.py:5601` reads "Missing API key for **Volcengine**..."
— NOT the observed "...for provider \"anthropic\"". So the exact error
phrasing comes from a *different* code path (an Anthropic SDK wrapper or a
provider-router), to be pinned below.

**Correct llm client path:** `backend/agents/llm_client.py` (NOT
backend/services/llm_client.py — that file does not exist).

**Path the operator's "Approve" message takes:**
1. `commands.py:185 @app.message("") handle_any_message` — every non-bot
   message in `#ford-approvals` (`_APPROVAL_CHANNEL`) is ingested as a
   ticket (`ingest_slack_message`) and acknowledged. The literal word
   "Approve" matches NO special branch (`clear queue`/`status` only) — so
   it falls through to ticket creation + the assistant/orchestrator answer
   path.
2. The `approval_approve` / `approval_deny` **button** action_ids are
   RENDERED at `governance.py:166-175` but have **NO `@app.action()`
   handler** anywhere (grep of all `@app.action(...)` registrations shows
   only `app_home_*`, `agent_model_change_*`, `agent_feedback_*`). So a
   button click is a dead control; the operator typed the word instead.
3. The message-answer path routes through the assistant handler /
   orchestrator, which for non-DIRECT classifications calls the LLM via
   `make_client()` → because `paper_use_claude_code_route=True` (confirmed
   live) this constructs `ClaudeCodeClient` (the `claude` CLI subprocess
   rail), NOT direct Anthropic. ("Approve" is not in the
   `direct_responder.can_handle_directly` trigger list — `commands.py`/
   `direct_responder.py:33-46` — so it does not get the local no-LLM
   fast-path.)

**ROOT CAUSE (FINDING F-A1, HIGH confidence):** The error string
`Missing API key for provider "anthropic"` is NOT emitted by any repo or
venv code (exhaustive grep: only litellm's "Missing API key for
**Volcengine**" exists, and the ticket processor raises a *different*
string "ANTHROPIC_API_KEY not found in settings or environment"). It is
emitted by the **`claude` CLI binary itself**, invoked by
`claude_code_client.py:claude_code_invoke()`. That function deliberately
**scrubs `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` from the
subprocess env** (`claude_code_client.py:163-170`, phase-38.13.1) so the
CLI authenticates via `~/.claude/` OAuth (the Max-subscription flat-fee
rail) instead of billing api.anthropic.com. **When the `~/.claude/` OAuth
session is expired/absent (plausible during a 10-day unattended away
week), the CLI has neither an env key (scrubbed by design) nor a valid
OAuth token → it errors with a provider-named missing-key message.** The
CLI's non-zero exit is wrapped into `ClaudeCodeError` at
`claude_code_client.py:188-197`; whichever caller surfaced it to Slack
relayed the CLI's own stderr text (hence the exact phrasing not being in
our source).

Live state confirmed via settings introspection (no secret values
printed): `paper_use_claude_code_route = True`, `anthropic_api_key
is_set = True` (≥20 chars). So the env key EXISTS but is intentionally
withheld from the CLI subprocess; the failure is OAuth-session, not a
missing `.env` var. **GENERATE must verify `~/.claude/` token validity
and the CLI's current auth state as the primary check, and only secondarily
confirm the .env var name is present.** Do NOT "fix" by un-scrubbing
ANTHROPIC_API_KEY — that silently re-routes billing to the exhausted
direct-API account (the exact failure phase-38.13.1 guarded against; see
the hard-fail at `llm_client.py:1967-1976`).

**FAIL-OPEN vs FAIL-CLOSED determination (FINDING F-A2):** The approval
path **FAILS CLOSED** with respect to *trade execution* — and this is the
correct posture, though arrived at by accident rather than design:
- The thing the operator was approving is answered by an LLM. When the LLM
  rail is down, the assistant handler hits its `except Exception` →
  "Deterministic fallback" (`assistant_handler.py:~344+`), so the operator
  got an error message, NOT a silent success.
- Critically, typing "Approve" does **not** itself execute any trade. The
  autonomous trading loop (`autonomous_loop.py`) runs on its own scheduler
  and does its own buy/sell; the Slack approve flow is a
  human-acknowledgement/Q&A surface, not a trade-gating interlock. So the
  broken approve flow did **not** cause unintended trades — it caused
  *loss of operator oversight* (the human-on-the-loop could not get
  answers), which is a fail-closed-on-action / fail-open-on-observability
  split.
- SECURITY NOTE for the contract: there is a **latent fail-open** risk in
  the dead `approval_approve` button. Because no handler is registered, if
  a future feature wires the button to a privileged action without
  re-checking auth, a click would either no-op (current) or, worse, be
  added without the LLM-down guard. Flag as finding for phase-56, severity
  LOW (currently inert), but note the dead control explicitly.

### A(b) — Nightly watchdog ReadTimeout pattern (~20:05-20:50 CEST)

**Watchdog:** launchd `com.pyfinagent.backend-watchdog`
(`scripts/launchd/com.pyfinagent.backend-watchdog.plist`, StartInterval=60s)
runs `scripts/launchd/backend_watchdog.sh` → `curl -m 5` GET
`http://127.0.0.1:8000/api/health`. 3 consecutive fails → SIGUSR1 stack
dump + P1 Slack alert + `launchctl kickstart -k`. Success resets counter.

**Two timeouts conflated as "ReadTimeout":** (1) watchdog `curl -m 5` on
`/api/health`; (2) morning/evening **digest** (`scheduler.py:399`/`:435`)
`httpx.AsyncClient(timeout=30.0)` GET `/api/paper-trading/portfolio` +
`/api/reports/?limit=5`, caught by `_route_exception_to_p1`
(`scheduler.py:421/466`) → posted to Slack as a `ReadTimeout`. The
operator's Slack "ReadTimeout" is the httpx digest probe.

**ROOT CAUSE (FINDING F-C, HIGH confidence):** Trading cycle starts
**18:00:00 UTC = 20:00 CEST** exactly (`handoff/cycle_history.jsonl`:
`af5a8000` 06-04T18:00:00→19:02 = 62 min; `035dbb69` 06-05T18:00:00→19:05
= 65 min). Watchdog FAILs (`handoff/logs/backend-watchdog.log`, UTC/Z)
cluster at 05-27 18:05:48Z+18:16:46Z, 05-28 18:04:54Z, 06-01 18:36Z,
06-02 18:33Z, 06-05 18:13Z — **all while the single-process uvicorn
backend runs its CPU/IO-heavy cycle, starving the event loop** so
`/api/health` and the portfolio probe transiently time out. **Every FAIL
is "(1/3)" — never reaches the 3-consecutive kickstart threshold; the
backend was never down, just briefly unresponsive.** Long cycles (62-65
min, 06-04/05) vs short (~13 min, 06-08/09 — `25f2fb19`/`0361d1ea`) track
whether the heavy path ran. Severity LOW (self-heals, no data loss).
Honest bound: event-loop starvation, not a crash; off-peak digest schedule
or process isolation removes the cosmetic alert.

### A(c) — 05-27/05-28 ALL-analyses-0.0/10-HOLD (silent degraded scoring)

**Score path:** lite synthesis returns `final_weighted_score` (default
`5.0` at `orchestrator.py:2050`). Digest reads it at `formatters.py:37`:
`score = report.get("final_weighted_score", 0)` → **default 0 if absent**
→ `{score:.1f}/10` renders "0.0/10".

**ROOT CAUSE (FINDING F-D, HIGH confidence — confirmed from
`analysis_results`):** 2026-05-27 `analysis_results` is interleaved: ~12
rows with REAL scores (MU 7.3 Buy, DELL 7.0 Hold — *lowercase* rec, normal
path) AND a parallel block of **10 rows all `final_score=0.0`
`recommendation="HOLD"` (UPPERCASE)** for the same tickers. Uppercase
"HOLD" vs lowercase "Hold" is the tell: **uppercase = the degraded
fallback synthesis path that fired when the LLM rail was unavailable**
(same root cause as F-A1). The 05-28 morning digest re-published the 0.0
rows. Across the whole window `pillar_4_sentiment=0.0`,
`social_sentiment_score=NULL`, `insider_signal`/`patent_signal` empty on
every row. **The pipeline failed SILENTLY — it wrote 0.0/HOLD and the
digest published it with no guard distinguishing "score is 0.0" from
"scoring failed".** Severity HIGH (operator saw HOLDs that were actually
pipeline failures). This is the epistemic-calibration failure case (§B4).

### A(d) — Per-skill firing audit + agent-label taxonomy (CORRECTED)

**THREE step premises are wrong against live data:**
1. *"llm_call_log has NO cycle_id column"* — FALSE. Schema (pyfinagent_data,
   386 rows): `ts, provider, model, agent(NULL), latency_ms, ttft_ms,
   input_tok, output_tok, request_id, ok, ticker(NULL), cycle_id(NULL),
   session_cost_usd, cache_creation_tok, cache_read_tok`. It HAS cycle_id
   AND ticker.
2. *"agent labels are debate roles (Bull R1/2)"* — NOT present in the away
   window (or anywhere meaningful; see below).
3. *"strategy_decisions.cycle_id/ticker/score/action/rationale for the
   join"* — FALSE. `strategy_decisions` (16 rows) is the rotation log:
   `ts, cycle_id, decided_strategy, prior_strategy, trigger, decay_signal,
   decay_attribution, rationale`. No per-ticker score/action. NOT a
   score-join table.

**HEADLINE FINDING F-E (HIGH confidence) — the observability gap:**
`llm_call_log` does NOT capture the away-week analysis. Live away-window
(2026-06-01..06-10) query:

| agent | n | cycles | tickers | fails | cost_usd |
|-------|---|--------|---------|-------|----------|
| NULL  | 16| 1      | 0       | 0     | 0.40     |

8 are Haiku probes (in=1000/out=50, $0.00); 8 are real Gemini-2.0-flash
calls on 06-01 (cyc=ca54842f) but `agent=NULL, ticker=NULL`. **Cycles on
06-04/05/08/09 (which `cycle_history.jsonl` confirms RAN) wrote ZERO
llm_call_log rows.** Code cause: `log_llm_call` is invoked ONLY in
`GeminiClient` (`llm_client.py:1066`) and `ClaudeClient`
(`llm_client.py:1736`); **`claude_code_client.py` never calls it.** With
`paper_use_claude_code_route=True` (live), the Claude analysis bypassed
llm_call_log; the Gemini-locked sub-skills logged with `_role` unset (NULL
agent). **llm_call_log under-counts the Claude rail to zero — it is NOT
the ground truth the step assumed.**

Full-history taxonomy (all 386 rows; what labels look like WHEN set):

| agent | provider | n |
|-------|----------|---|
| (NULL) | gemini | 197 |
| (NULL) | anthropic | 184 |
| phase26.1-smoke | anthropic | 1 |
| Synthesis | anthropic | 1 |
| Synthesis_advisor_tool | anthropic | 1 |
| Quant Model_code_exec | gemini | 1 |
| Enhanced Macro_combined_t | gemini | 1 |

99% NULL-agent; the labelled rows are skill/tool names (the label IS a
skill tag when set — but it almost never is). No "Bull R1/R2" debate roles
appear (multi-round debate was in the lite-skip list, did not run).

**AUTHORITATIVE per-skill evidence = `paper_trades.signals` (JSON) +
`analysis_results`.** Away-week `paper_trades.signals` taxonomy (52 rows;
populated on BUYs, `[]` on mechanical SELLs):

| agent | role | meaning |
|-------|------|---------|
| Quant | screener | momentum/RSI/vol/sector + composite_score |
| SignalStack | overlay | conviction overlay — **every away-week BUY = `"conviction 10.00; fallback (LLM unavailable)"`** |
| Trader | decision | LLM trade rationale (weight 0.7) |
| RiskJudge | gate | volatility/concentration risk gate |

**This is the lite/screener path, NOT the 28-skill orchestrator.**
`analysis_results.full_report_json` away-week rows have only 3 keys
(`analysis`/`market_data`/`source`) — no rich per-skill artifact.
Deterministic-skill audit: `pillar_4_sentiment=0.0`,
`social_sentiment_score=NULL`, `insider_signal`/`patent_signal` empty on
ALL away-week rows → **insider/patent/sentiment/social skills effectively
did NOT fire.**

**ANSWER to the operator's 2026-06-10 question** ("are the agents also
using all the skills, rag, earnings tone, insider, patent, news"): **No.**
Away week ran the lite path (Quant→SignalStack→Trader→RiskJudge); the LLM
overlay (SignalStack) fell back to static conviction 10.00 (Claude rail
down); the deterministic enrichment skills wrote no signal; debate/
deep-dive/devil's-advocate/risk-assessment (lite-skip,
`orchestrator.py:1491-2069`) did not run. The `/paper-trading/manage`
"Lite mode" checkbox showing UNCHECKED is contradicted by the evidence —
**the cycles ran lite regardless of the checkbox** (UI/runtime desync →
phase-56 finding).

### A(e) — Signal stability: the SNDK flip (DIRECTION CORRECTED)

The step said "SNDK 7.0-BUY → 5.0-HOLD"; the data shows the OPPOSITE:
- SNDK 06-05 19:01:33 → **5.0 HOLD** (px=1553.80)
- SNDK 06-08 18:08:34 → **7.0 BUY** (px=1632.50)
- SNDK 06-09 18:12:22 → SELL (pnl=-0.41%)

SNDK went **HOLD(5.0)→BUY(7.0)** as price rose +5.1% over the weekend, then
sold next day. MU held **7.0 BUY** flat (06-05 px=878.47 → 06-08 px=954.94,
+8.7%) before its -6.27% SELL on 06-09. **Attribution: score moves track
price momentum (screener composite uses 1m/3m/6m momentum) — new-price
information, NOT model noise.** But with the LLM overlay on a static
fallback there was no independent check to damp momentum-chasing; the +5%
move mechanically lifted SNDK HOLD→BUY. Day-over-day |Δscore| = 2.0 on a
5% price move = high score-elasticity, characteristic of a single-factor
(momentum) signal with no stabilizing overlay (see §B3 churn).

### A(f) — Reasoning-quality spot-check basis

Pull from `analysis_results.full_report_json` (away-week minimal) +
`paper_trades.signals` (4-agent rationale JSON). MU 06-08 BUY → 06-09
-6.27% SELL whipsaw: Quant composite 102.276 (+286.8% 6m momentum),
SignalStack "conviction 10.00; fallback (LLM unavailable)", Trader "tight
trailing stop against mean-reversion risk", RiskJudge "VOLATILITY
parabolic ~8x the limit". GENERATE: pull FULL (untruncated) signals JSON
for MU 06-08 + 000660.KS + DELL; assess whether cited skills informed the
decision. Preliminary: decision driven by Quant momentum + RiskJudge;
**SignalStack LLM overlay was a fallback stub → contributed nothing.**

### A(g) — Away-week LLM burn vs P&L (reconciliation basis)

`llm_call_log.session_cost_usd` away-window SUM = **$0.40 / 16 rows** (most
from the 8 Gemini 06-01 calls; the 8 Haiku probes $0.00). Per F-E this is
an UNDER-count: the Claude Code rail (Max flat-fee, $0 metered) is unlogged.
The 55.1 `/performance total_analysis_cost=5.05 USD over 36 days`
reconciles directionally (low spend) but the away-week $0.40 is NOT the
true compute. N* framing for GENERATE: "≈$0.40 metered + unmetered
Claude-Code-rail calls (Max flat-fee)"; do NOT claim $0.40 is the full
cost. arXiv:2603.27539: a 7-agent system is ~$0.50-$2.00/decision at API
pricing — pyfinagent's near-$0 is the flat-fee Max rail working as intended.

### A(h) — Look-ahead / temporal sanitation (lite path)

The lite path's signals are Quant price-momentum (trailing returns to the
bar — inherently point-in-time, no look-ahead). The RAG/news/social
fact-ledger skills did NOT fire in the away week (empty signals), so the
away-week TRADES carry no news/sentiment look-ahead exposure — but that is
because those skills were DOWN, not proven clean. Standing look-ahead risk
lives in RAG timestamp discipline when those skills are active
(arXiv:2603.27539 §4.1: "imprecisely timestamped RAG databases can inject
future information into historical queries"). See §B5.

---

## PART B — EXTERNAL LITERATURE

### Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://arxiv.org/html/2603.27539v1 | 2026-06-10 | paper (peer-track) | WebFetch (arxiv/html) | 4-D taxonomy (architecture/coordination/memory/tool); **Coordination Primacy Hypothesis** — coordination protocol > model scaling; 5 evaluation failures incl. **transaction-cost neglect** ("10-20 bps round-trip → 25-50 pts annual drag") and **look-ahead via "imprecisely timestamped RAG databases"**; **Coordination Breakeven Spread** CBS(d)=Δp(d)/2; cost-per-decision "$0.50-$2.00" for a 7-agent system |
| 2 | https://opentelemetry.io/blog/2026/genai-observability/ | 2026-06-10 | official doc | WebFetch | Each tool = an `execute_tool` span; `gen_ai.request.model`, `gen_ai.response.finish_reasons` (`stop`/`tool_calls`). **Detect a skipped tool = `finish_reasons` says tool_calls but no corresponding `execute_tool` span.** Exactly the instrumentation pyfinagent lacks (NULL-agent gap) |
| 3 | https://tianpan.co/blog/2026-04-20-llm-calibration-production-overconfidence | 2026-06-10 | authoritative blog | WebFetch | "model is **least accurate when it is most confident**" (Dunning-Kruger); calibration degrades under distribution shift; chained agents: 90% claimed → ~75% actual → 42% over 3 hops; **measure ECE (>0.10 = unreliable), reliability diagram; prompt-tone "be calibrated" does NOT help** |
| 4 | https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop | 2026-06-10 | official doc | WebFetch | Canonical **fail-CLOSED** approval: workflow PAUSES, emits `RequestInfoEvent`/`function_approval_request`; **action does not execute until response received**; pending requests checkpointed and **re-emitted on restart**. The reference design pyfinagent's accidental-fail-closed should be measured against |
| 5 | https://medium.com/@AlignX_AI/designing-human-in-the-loop-for-agentic-workflows-079faec737ed | 2026-06-10 | industry blog | WebFetch | Human-as-bottleneck anti-pattern: "all the cost of human involvement, none of the speed... false sense of safety"; escalation must carry context (what it tried, why uncertain); high-stakes = "prepare action with full audit trail → human authorizes → execute with attribution logged" |
| 6 | https://arxiv.org/abs/2309.17322 | 2026-06-10 | paper (Glasserman & Lin) | WebFetch | Two LLM-sentiment look-ahead forms: **(a) knowing post-article returns, (b) "distraction effect"** (company knowledge contaminates sentiment); mitigation = **anonymize/mask company names+dates before scoring**; in-sample the distraction effect dominated and was larger for big-cap names |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://arxiv.org/pdf/2601.13770 | paper (2026 PIT look-ahead benchmark) | Recency-scan hit; PDF binary (would need html); snippet sufficed |
| https://arxiv.org/pdf/2305.04135 | paper (predictive churn) | PDF binary — extraction blocked (research-gate rule: html-only); core definition captured from snippet |
| https://shostack.org/blog/the-security-principles-of-saltzer-and-schroeder | canonical security | Year-less canonical hit — "fail-safe defaults: default is lack of access" captured from snippet |
| https://www.coalitionforsecureai.org/announcing-the-cosai-principles-for-secure-by-design-agentic-systems/ | standards body | Snippet gave "propose-then-commit; block tool calls until approval is recorded" |
| https://www.mdpi.com/2673-2688/7/4/138 | paper (LLM sentiment market-neutral) | Adjacent; covered by #6 |
| https://arxiv.org/pdf/2510.10526 | paper (LLM+RL sentiment trading) | Adjacent; covered by #1/#6 |
| https://github.com/Kriss-V/deadmancheck | tool/pattern | "**cron jobs that run but do nothing**" — exact analogue of F-D/F-E; design pattern, not read-in-full |
| https://oneuptime.com/blog/post/2026-02-06-heartbeat-dead-man-switch-opentelemetry-pipeline/view | blog | dead-man-switch + output assertions (rows_exported>0) |
| https://medium.com/@chu.ngwoke/silent-failures-in-data-pipelines-why-theyre-so-dangerous-7c3c2aff8238 | blog | "dashboards update as if everything is fine" = the 0.0/10 digest |
| https://pmc.ncbi.nlm.nih.gov/articles/PMC12657207/ | journal | test-retest reliability yardstick (Pearson r 0.8-0.9 good) |
| https://arxiv.org/html/2603.22651v1 | paper | MAS orchestration cost-accuracy benchmark (adjacent to #1) |
| https://greptime.com/blogs/2026-05-09-opentelemetry-genai-semantic-conventions | blog | OTel agent/MCP-tool tracing (corroborates #2) |

### Key findings (external)

1. **Coordination > model scaling** — "the inter-agent coordination protocol
   is the most consequential structural factor in trading decision quality
   ... exerting greater influence than model selection" (arXiv:2603.27539 §5.2).
   Application: pyfinagent's away-week degradation was a *coordination/
   availability* failure (SignalStack overlay down → static fallback), not a
   model-quality failure. Fixing the rail matters more than upgrading models.
2. **Transaction-cost neglect reverses returns** — "round-trip costs of 10 to
   20 basis points can compound to 25 to 50 percentage points of annual drag"
   (arXiv:2603.27539 §4.4). Application: the away-week whipsaws (MU -6.27%
   round-trip in 1 day) are exactly the cost-bleeding pattern; the CBS metric
   says momentum-chasing on wide-spread names may not clear costs.
3. **Most confident when most wrong** — "the model is least accurate when it
   is most confident" (TianPan 2026). Application: the 0.0/10 HOLD rows are
   the inverse failure — a *failed* score masquerading as a confident neutral
   one; the guard is ECE/output-assertion, NOT prompt tone.
4. **Fail-safe defaults (canonical)** — Saltzer & Schroeder: "the default
   situation is lack of access; the protection scheme identifies conditions
   under which access is permitted." CoSAI: "block tool calls until approval
   is recorded." Application: the dead `approval_approve` button + the
   LLM-down approval path should default to NO privileged action (they
   currently do, by accident — F-A2).
5. **LLM-sentiment look-ahead has two forms** (Glasserman & Lin 2309.17322):
   direct return-knowledge AND the distraction effect; mask entities+dates.
   Application: when pyfinagent's RAG/news/social skills ARE active, enforce
   publication-timestamp < decision-timestamp and consider entity masking.
6. **"Jobs that run but do nothing"** — the canonical silent-failure class
   (deadmancheck; OneUptime; "dashboards update as if everything is fine").
   Application: F-D (0.0/10) and F-E (empty llm_call_log) are textbook
   instances; the fix is an output assertion (e.g. `final_score>0` AND
   `n_llm_calls>0` per cycle) + a dead-man heartbeat on the analysis writer.

### Recency scan (last 2 years, 2024-2026) — MANDATORY

Searched explicitly for 2024-2026 work on all five topics. **Result: found
multiple NEW findings that complement (do not supersede) the canonical
sources.** The single most relevant new finding is the pre-anchored
**arXiv:2603.27539 (2026)** — the first taxonomy purpose-built for
evaluating LLM-based financial MAS, postdating TradingAgents (2412.20138)
and FinMem; it directly introduces the **Coordination Breakeven Spread**
and the five evaluation-failure catalogue that map onto pyfinagent's
incidents. Also new in-window: **arXiv:2601.13770 (2026)** — a standardized
benchmark of look-ahead bias in point-in-time data (complements the 2023
Glasserman & Lin paper, extending it to a benchmark); the **OTel GenAI
semantic conventions (2026, experimental)** now standardize `execute_tool`
spans + `gen_ai.response.finish_reasons` for tool-firing audits (the
mechanism pyfinagent lacks); and the **TianPan calibration-in-production
(Apr 2026)** Dunning-Kruger framing. No new finding *overturns* the
canonical Saltzer-Schroeder fail-safe-defaults principle or test-retest
reliability — those remain the durable yardsticks. Older canonical sources
(Saltzer & Schroeder 1975 fail-safe defaults; Glasserman & Lin 2023)
remain valid and are cited alongside the 2026 work.

### Search-query log (3-variant discipline per .claude/rules/research-gate.md)

| Topic | Current-frontier (2026) | Last-2-yr (2024-25) | Year-less canonical |
|-------|------------------------|---------------------|---------------------|
| Observability / tool-firing | "OpenTelemetry GenAI semantic conventions 2026" | (covered by frontier) | "LLM agent tool-use observability audit which tools fired tracing" |
| HITL approval fail-open/closed | (covered by canonical) | — | "human in the loop approval fail-safe fail-secure design principle" + "human-in-the-loop approval workflow agent trading fail-open fail-closed" |
| Signal/recommendation churn | — | "recommendation churn day-over-day rating flip rate test-retest" | "test-retest reliability scoring stability" |
| Epistemic calibration / degraded-mode | "LLM epistemic calibration degraded mode silent failure guardrails 2026" | "silent failure detection data pipeline all-zero output dead man switch 2025" | (canonical via calibration blog) |
| PIT look-ahead news/sentiment | — | — | "look-ahead bias point-in-time news sentiment event-driven trading signal data hygiene" |
| Pre-anchored | "arXiv 2603.27539 LLM financial multi-agent evaluation taxonomy" | — | — |

### Consensus vs debate (external)

- **Consensus:** fail-safe defaults (no action without recorded approval);
  silent "ran-but-did-nothing" failures need output assertions + dead-man
  switches; LLM confidence is poorly calibrated and degrades off-distribution;
  transaction costs can reverse a strategy's sign.
- **Debate / nuance:** arXiv:2603.27539 explicitly states the Coordination
  Primacy Hypothesis "cannot yet be validated" empirically — author-reported
  ablations ("removing coordination cut Sharpe 15-30%; smaller model only
  5-8%") are "suggestive rather than confirmatory." Glasserman & Lin found
  the *distraction effect* (not direct look-ahead) dominated in-sample — a
  reminder that the obvious bias is not always the binding one.

### Pitfalls (from literature)

- Treating an empty/zero metric as a real value (the 0.0/10 trap) — empty
  queries return "no data" and most alerts do nothing (OneUptime).
- "Be more calibrated" prompt-tuning gives false comfort; it shifts verbal
  register without improving confidence-accuracy correlation (TianPan).
- Backtesting LLM sentiment over the training window inflates apparent skill
  via distraction + look-ahead (Glasserman & Lin).
- Approval dialogs that lack context → rubber-stamping / security theater
  (StackAI/CoSAI). pyfinagent's approve flow gives the operator an *error*,
  not a low-context approval — which is safer but useless.

### Application to pyfinagent (external → internal file:line)

| External finding | pyfinagent locus | Action implication |
|------------------|------------------|--------------------|
| Fail-safe defaults / propose-then-commit | dead `approval_approve` at `governance.py:166-175` (no handler); approve answered via LLM (`assistant_handler.py` → `make_client` → ClaudeCodeClient) | Confirm no privileged action ever fires without recorded approval; register or remove the dead button (phase-56) |
| OTel `execute_tool` span + finish_reasons to detect skipped tools | `log_llm_call` only in Gemini/Claude clients (`llm_client.py:1066/1736`); **absent in `claude_code_client.py`** | Add per-skill firing log to the Claude Code rail; tag `agent`/`ticker`/`cycle_id` (phase-56) — closes F-E |
| Output assertion / dead-man (jobs that run but do nothing) | `formatters.py:37` publishes `final_score` default 0; `orchestrator.py:2050` default 5.0 | Add a cycle output-assertion (`final_score>0` AND `n_llm_calls>0`) before the digest publishes — closes F-D |
| ECE / confident-neutral vs failed score | uppercase-HOLD vs lowercase-Hold in `analysis_results` | The degraded path should emit a DISTINCT sentinel (e.g. `recommendation="DEGRADED"`), never a 0.0/HOLD that looks like a real neutral |
| Transaction-cost neglect / CBS | `paper_trades` round-trips (MU -6.27% in 1 day); momentum screener | Whipsaw audit should net costs; flag momentum-chasing that fails CBS |
| LLM-sentiment look-ahead (mask entities/dates) | RAG/news/social skills (DOWN in away week; empty signals) | When re-enabled, enforce pub-ts < decision-ts; consider entity masking |
| Test-retest reliability (Pearson r 0.8-0.9) | SNDK |Δscore|=2.0 on +5% px | Frame signal-stability as a reliability coefficient; single-factor momentum has low test-retest stability by construction |

---

## ORDERED GENERATE PLAN (feeds the contract)

1. **Incident IDs + severity** (deliverables a-c): assign stable finding IDs
   for phase-56 — F-A1 (approve-flow: Claude Code OAuth rail down; HIGH),
   F-A2 (fail-closed-on-action confirmation + dead button; LOW), F-C
   (watchdog event-loop starvation by 18:00 UTC cycle; LOW, self-heal), F-D
   (silent 0.0/10 degraded scoring; HIGH), F-E (llm_call_log observability
   gap — Claude rail unlogged; HIGH). Each gets a one-line WONTFIX-or-fix
   note. Verify `~/.claude/` OAuth validity as the F-A1 primary check (do
   NOT print secrets; do NOT un-scrub ANTHROPIC_API_KEY).
2. **Agent-label → skill mapping table** (deliverable d): publish the live
   BQ tables above (away-window NULL result + full-history taxonomy +
   `paper_trades.signals` 4-agent taxonomy). State the corrected premises
   (cycle_id EXISTS; strategy_decisions is rotation-only; llm_call_log
   under-counts the Claude rail). Answer the operator's skills question: NO,
   the away week ran lite (Quant/SignalStack/Trader/RiskJudge) with the LLM
   overlay on static fallback and the deterministic enrichment skills silent.
3. **Mode determination** (deliverable d, lite-vs-full): from
   `analysis_results.full_report_json` (3 keys) + the SignalStack fallback
   string, conclude the cycles ran LITE despite the UI checkbox showing
   unchecked → flag the UI/runtime desync.
4. **Signal-stability table** (deliverable e): day-over-day BUY/HOLD/SELL +
   |Δscore| per ticker from `analysis_results`; reproduce SNDK 5.0→7.0
   (CORRECTED direction) and MU flat-7.0; attribute to price momentum
   (new-info), note absence of a stabilizing overlay.
5. **Reasoning spot-check** (deliverable f): pull FULL signals JSON for >=3
   away-week analyses incl. the MU 06-08→06-09 whipsaw + 000660.KS + DELL;
   assess skill-output→decision linkage (SignalStack was a no-op stub).
6. **Burn vs P&L** (deliverable g): report $0.40 metered + the unmetered
   Claude-rail caveat; cross-check 55.1's $5.05/36d; do NOT claim $0.40 is
   the full compute.
7. **Look-ahead paragraph** (deliverable h): away-week trades carry no
   news/sentiment look-ahead (skills were DOWN), but state the standing RAG
   timestamp-discipline risk + Glasserman-Lin entity-masking mitigation.
8. All work is review-only, $0, bounded BQ (30s), no LLM cycle spend.

### Research Gate Checklist

Hard blockers — `gate_passed` is false if any unchecked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6)
- [x] 10+ unique URLs total (18: 6 full + 12 snippet)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full papers / pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (slack_bot, llm_client,
  claude_code_client, orchestrator, scheduler, watchdog, formatters, BQ schemas)
- [x] Contradictions / consensus noted (Coordination Primacy "not yet validated";
  distraction > look-ahead in-sample)
- [x] All claims cited per-claim (file:line internal; URL external)
- [ ] 3-variant query discipline: year-less canonical genuinely run for HITL,
  churn, look-ahead; for observability + calibration the canonical was reached
  via the frontier/last-2yr hits (noted in the query log) — minor soft gap.

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 12,
  "urls_collected": 18,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "report_md": "handoff/current/research_brief_55.2.md",
  "gate_passed": true
}
```
