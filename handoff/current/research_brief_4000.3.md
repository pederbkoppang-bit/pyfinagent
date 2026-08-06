# Research Brief -- phase-4000.3 (CC-rail E2E live window)

Tier: **simple** (caller-stated). Write-first: this file created before any
source was read; appended incrementally.

Scope note (budget): 4000.1 + 4000.2 briefs already cover the CLI envelope,
billing, `llm_call_log` writers, endpoint shapes, counters and guard surfaces
(`handoff/archive/phase-4000.1/research_brief_4000.1.md`,
`handoff/archive/phase-4000.2/research_brief_4000.2.md`). This gate answers
FOUR narrow questions for the live window and clears the >=5-read-in-full
floor. No re-derivation of 4000.1/4000.2 material.

## Questions

- Q1 Ticker choice + live-book interference
- Q2 Window timing (deadline sizing + scheduler collision)
- Q3 Analysis cost shape (Gemini-locked vs rail-routed)
- Q4 Failure handling (poll body + residual state)

---

## STATUS: COMPLETE -- gate_passed: true (6 read in full, 20 URLs, recency scan done)

---

## Q1 -- Ticker choice + live-book interference

### Persistence seam (what a manual analysis WRITES)

`POST /api/analysis/` runs the full Layer-1 pipeline in-process
(`backend/api/analysis.py:350-381` -> `_run_sync_analysis` at `:43`), and on
SUCCESS calls `bq.save_report(...)` at `backend/api/analysis.py:210-311`,
writing ONE row to `financial_reports.analysis_results` tagged
`report["_path"] = "full"` (`backend/api/analysis.py:208-209`). It also fires
a best-effort Slack "Analysis Complete" notification
(`backend/api/analysis.py:316-328`) -- expected noise during the window, not a
defect.

It writes NO `paper_positions`, NO `paper_trades`, NO
`strategy_decisions` row. The `strategy_decisions` heartbeat is written only
by the autonomous cycle (`backend/services/autonomous_loop.py:1637-1666`).

### Consumption seam (does the trade path READ it?)

**No -- not for the decision.** `decide_trades` is called once per cycle at
`backend/services/autonomous_loop.py:1461-1469` with `candidate_analyses` /
`holding_analyses` that the cycle builds IN MEMORY in the same run:
candidates from `rank_candidates(screen_data, ...)` at
`autonomous_loop.py:890` (`backend/tools/screener.py:249`), analyses from the
cycle's own per-ticker runs. `backend/tools/screener.py` contains ZERO
references to `analysis_results` (grep: no matches). So a manual analysis row
cannot become a candidate, a BUY, or a re-evaluation.

**One real (indirect) seam exists and it is metadata-only.**
`_fetch_ticker_meta` (`backend/api/paper_trading.py:1159-1240`) UNIONs
`paper_positions` (priority 1) with `analysis_results` (priority 2) and
returns `{company_name, sector}`. The autonomous cycle calls it twice:
- candidate enrichment, `autonomous_loop.py:928-950`;
- legacy-position sector backfill, `autonomous_loop.py:1417-1430`, guarded by
  `max_per_sector > 0` and applied ONLY to positions whose `sector` is blank.

`decide_trades` reads `pos.get("sector")` to seed the sector cap
(`autonomous_loop.py:1417-1419` comment). So the ONLY way a manual analysis
touches the decision is by supplying a `sector` string for a ticker that has
none -- and `paper_positions` outranks `analysis_results` (priority 1 vs 2,
`paper_trading.py:1200-1215`), so a HELD ticker's sector is already canonical.

### Live book (measured 2026-08-06, GET /api/paper-trading/portfolio)

One open position: **NTAP** (Technology, qty 5.346643, opened
2026-07-31T18:47:37Z) -- matches the 4000.1-era note. Universe knobs:
`paper_screen_top_n = 10`, `paper_analyze_top_n = 5`
(`backend/config/settings.py:380-381`).

### RECOMMENDATION: run the window on **AAPL**

Reasons, in the order they bind:
1. **Not the held name.** NTAP is the only open position; analyzing it would
   put a fresh `analysis_results` row on the exact ticker whose
   sector/company metadata the cycle re-reads at
   `autonomous_loop.py:1417-1430`. AAPL removes that seam entirely.
2. **Sector already canonical.** Even in the metadata seam, AAPL is a name the
   pipeline has analyzed repeatedly, so `analysis_results.sector` is already
   populated -- the write is idempotent in content.
3. **Screen membership is emergent, not fixed.** `paper_screen_top_n=10` is a
   ranked cut of the screen (`autonomous_loop.py:886-900`), so no ticker can
   be guaranteed "outside the universe"; the interference argument therefore
   has to rest on the CONSUMPTION seam (which is metadata-only), not on
   universe membership. Choosing a mega-cap the screener rarely selects for
   momentum is a second-order comfort, not the load-bearing reason.
4. **Data-fetch steps will not stall.** AAPL has complete yfinance
   fundamentals, options chain, insider filings and news coverage -- the
   enrichment steps that time out on thin names.

Runner-up if the operator prefers a name with zero screen affinity: **JNJ**
(mega-cap, different sector from the single holding, deep data coverage).

---

## Q2 -- Window timing

### Per-step budget arithmetic (cite chain)

- `claude_code_timeout_s = 150` (`backend/config/settings.py:186-191`) -- the
  CLI subprocess timeout.
- `ClaudeCodeClient.recommended_step_timeout = 150`
  (`backend/agents/claude_code_client.py:591`); the settings description at
  `settings.py:190` states the instance value is derived as `timeout_s + 30`
  (test asserts both forms: `backend/tests/test_phase_61_2_decision_integrity.py:269`
  `== 230` for a 200s timeout, `:274` `_timeout_s == 150 and
  recommended_step_timeout == 180`). Either way the STEP budget sits ABOVE the
  subprocess timeout by design.
- `_resolve_step_timeout` (`backend/agents/orchestrator.py:380-402`): grounded
  calls at the 90s default are lifted to **180s**; any rail
  `recommended_step_timeout` above the caller's value raises the budget
  (never lowers it).
- `_generate_with_retry` (`backend/agents/orchestrator.py:815`):
  `max_retries = 3`, `timeout = 90` default, `future.result(timeout=timeout)`
  at `:884`, raising `TimeoutError` after the loop at `:945-948`.
- `claude_code_empty_retry_max = 2` (`backend/config/settings.py:~193`) --
  up to 2 EXTRA attempts on errored-empty rail responses, but ONLY when
  `paper_synthesis_integrity_enabled` is True (`settings.py` field below).

**Worst case per LLM step on the rail: 3 attempts x ~180s = ~540s**, and the
27% timeout-shaped failure rate measured in 4000.1 means a multi-step full
analysis has a materially non-trivial chance of hitting that on at least one
step.

### Verdict on `--analysis-deadline`

The smoke default is `1800` (`scripts/qa/smoke_cc_rail_e2e.py:482`). That is
**~3 worst-case steps' worth of budget for a pipeline with far more than 3 LLM
steps**. A single unlucky step chain (3 x 180s) consumes 30% of the deadline
before any other step runs. RECOMMENDATION: **pass `--analysis-deadline 3600`**
explicitly for the live window and state the reason in the contract. The
deadline is a client-side poll bound only (`poll_analysis`,
`scripts/qa/smoke_cc_rail_e2e.py:240-251`) -- raising it does NOT raise any
server-side budget, does NOT increase rail calls (the `<=30` cap is enforced
independently at `smoke_cc_rail_e2e.py:~330` post-window), and only prevents
the window from throwing `TimeoutError` on an analysis that is still healthy
and in flight. Raising it is therefore strictly safer than leaving 1800.

### Scheduler collision -- MEASURED, and it is real

`GET /api/jobs/all` on the running backend (2026-08-06):

```json
{"id": "paper_trading_daily", "source": "main_apscheduler",
 "schedule": "cron[day_of_week='mon-fri', hour='14', minute='0']",
 "next_run": "2026-08-06T14:00:00-04:00", "last_run": null,
 "status": "scheduled", "controllable": true}
```

Registered at `backend/api/paper_trading.py:45-46` (`_scheduler_job_id =
"paper_trading_daily"`) into the main `AsyncIOScheduler` created at
`backend/main.py:309-314`. **Next fire: 2026-08-06 14:00 ET = 18:00 UTC.**

The cycle and the manual analysis share ONE process and ONE event loop
(`AsyncIOScheduler` on the API loop -- see the note at
`autonomous_loop.py:610`), so an overlap does not just muddy attribution: it
puts cycle rail calls INSIDE the smoke's `llm_call_log` window, which would
blow the `<=30` rail-call cap check (`smoke_cc_rail_e2e.py` post-window gate)
and pollute E1/E6 with rows the window did not cause.

**RECOMMENDATION: open the window so it CLOSES before 18:00 UTC**, and record
the measured `next_run` in the live_check as the collision-avoidance evidence.
If the window cannot fit, the clean alternative is
`POST /api/paper-trading/stop` (`backend/api/paper_trading.py:110-120`, which
removes the job) before the window and re-`start` after -- but that mutates
scheduler state, so it needs its own operator note; **the timing route is
preferred and requires no state change.**

---

## Q3 -- Analysis cost shape (Gemini-locked vs rail-routed)

### MEASURED live pins (GET /api/settings/, 2026-08-06)

```
gemini_model                 = claude-sonnet-4-6
deep_think_model             = claude-opus-4-8
paper_use_claude_code_route  = True     <-- ALREADY ON
paper_screen_top_n = 10 ; paper_analyze_top_n = 5
```

**The rail flag is already True on the running backend.** The window's flip is
therefore a no-op PUT, and the smoke's restore target `pre_flag`
(`scripts/qa/smoke_cc_rail_e2e.py:262`) is **True** -- so the operator's
end-state rule ("flag returns to True on every path") is satisfied by the
`ExitStack` restore alone (`smoke_cc_rail_e2e.py:288-290`). **Do NOT pass
`--keep-on`**: it only cancels the restore (`smoke_cc_rail_e2e.py:410-412`),
which is unnecessary here and weakens the crash-safety property.

### Routing rule

`make_client` routes to the rail iff `model_name.startswith("claude-")` AND
the flag is True (`backend/agents/llm_client.py:2113-2126`). Both live pins
are `claude-*`, so:

| Client | Constructed | Model | Route |
|---|---|---|---|
| `general_client` | `orchestrator.py:652` | claude-sonnet-4-6 | **RAIL** |
| `deep_think_client` | `orchestrator.py:653` | claude-opus-4-8 | **RAIL** |
| `synthesis_client` | `orchestrator.py:654` | claude-opus-4-8 | **RAIL** |
| `quant_exec_client` | `orchestrator.py:659` | claude-sonnet-4-6 | **RAIL** (see drift) |
| `rag_client` | `orchestrator.py:662` | Gemini workhorse | **GEMINI-LOCKED** |
| `grounded_client` | `orchestrator.py:664+` | Gemini | **built but UNUSED** (see below) |

Rail-routed steps by anchor: Macro `:1196`; Insider `:1287`; Options `:1294`;
Social `:1301`; Patent `:1308`; Earnings Tone `:1315`; Alt Data `:1331`;
Sector `:1338`; NLP `:1345`; Anomaly `:1352`; Scenario `:1365`; Quant Model
`:1378`; debate/critic/synthesis/revision `:1408`, `:1557`, `:1589`, `:1615`,
`:1670`.

### Two findings that change the E6 attribution

1. **Grounding is OFF, and the "grounded" steps silently land on the rail.**
   `self.supports_grounding` is read off `general_client`
   (`orchestrator.py:746`), and `ClaudeCodeClient.supports_grounding = False`
   (`backend/agents/claude_code_client.py:583`). So the four grounded call
   sites -- Market `:1178`, Competitor `:1187`, Deep Dive (Questions) `:1211`,
   Enhanced Macro `:1322` -- take the `else` branch to `self.general_client`
   (= the rail) with `is_grounded=False`. Consequence for Q2: the 180s
   grounded lift in `_resolve_step_timeout` (`orchestrator.py:397-398`) does
   NOT apply; the budget instead comes from `rail_min`
   (`orchestrator.py:399-401`). Consequence for E6: **no Google Search
   grounding happens in this window**, and the window's output is
   ungrounded-pipeline quality -- disclose it rather than presenting the
   analysis as representative of the grounded path.

2. **Gemini's share is at most the RAG step, and may be ZERO.** `rag_client`
   is the only client built OUTSIDE `make_client`
   (`orchestrator.py:662`, `GeminiClient(self.rag_model, _gemini_standard)`;
   `_gemini_standard` resolves through `_resolve_gemini`,
   `orchestrator.py:517-519`, to `_GEMINI_FALLBACK` at `:503` because the pin
   is not a `gemini-*` name). The step is gated by `self._rag_available`
   (`orchestrator.py:596`, `:613`, early-return at `:1153`), so if the Vertex
   data store is unavailable the RAG step is skipped entirely.

   **E6 conclusion: attribute essentially 100% of the window's LLM tokens to
   the Max pool.** Do not model a Gemini/Claude split.

3. **DOC/BEHAVIOR DRIFT (queue, do not fix here).** The comment at
   `orchestrator.py:654-658` asserts `quant_exec_client` "is Gemini-only
   (code_execution is Gemini-specific)... When settings.gemini_model points to
   a non-Gemini model, this still routes to Gemini via the bundle." Under the
   live pins that is FALSE: `make_client` sees `claude-sonnet-4-6` and returns
   a `ClaudeCodeClient` (`llm_client.py:2113-2126`), so Scenario `:1365` and
   Quant Model `:1378` run on the rail WITHOUT `code_execution`. This is a
   pre-existing condition, not caused by the window; it belongs in the
   defect queue alongside 4000.6/4000.7/4000.8.

---

## Q4 -- Failure handling

### What the poll body carries on FAIL

Two failure legs, both terminal and both leaving a readable body:

- In-pipeline exception: `backend/api/analysis.py:336-345` sets
  `status=FAILED` and `error = f"[{err_type}] Step '{last_step}': {e}"`
  (exception type + the last step name -- actionable).
- Unhandled task exception: `backend/api/analysis.py:368-374` (`_on_error`)
  sets `status=FAILED`, `error="[Type] unhandled: ..."`.

`GET /api/analysis/{id}` returns `AnalysisStatusResponse`
(`analysis.py:406-416`) with `status`, `current_step`, `steps_completed`,
`step_log`, `error`, and `report=None`.

**The smoke handles this correctly and loudly.** `poll_analysis` treats
`"failed"` as terminal and RETURNS the body rather than raising
(`scripts/qa/smoke_cc_rail_e2e.py:246-247`); E4 then appends it to `bad` and
goes red (`smoke_cc_rail_e2e.py:387-388`). No code change needed.

### Residual state the window must (not) clean up

| State | Anchor | Left behind on FAIL? | Cleanup needed |
|---|---|---|---|
| `_tasks[task_id]` | `analysis.py:33`, `:361` | YES -- **never evicted** (no `pop`/`del`/`clear`/TTL in the file) | None. Process-local, a few entries. **But do NOT restart the backend before evidence capture** -- the id 404s (`analysis.py:393-394`) |
| `analysis_results` row | `analysis.py:210` | **NO** -- `save_report` is inside the try, after `run_full_analysis` returns; `orchestrator.py` has ZERO `save_report` calls (grep count 0) | None |
| Partial signals | -- | **NO** -- nothing persists mid-pipeline | None |
| Slack notification | `analysis.py:316-328` | **NO** -- success path only | None |
| `llm_call_log` rows | rail writer | **YES, and that is the good property** -- every rail call up to the failure persists, so E1/E2/E3/E6 stay measurable on a failed window; only E4 goes red | None |
| Rail flag | `smoke_cc_rail_e2e.py:288-290` | Restored to `pre_flag` (= True) by the `ExitStack` callback registered BEFORE the flip, so a crash unwinds through it | None |

### Two operational constraints found in the smoke's argparse

- `--expected-calls-per-analysis` is **REQUIRED** for `--live`
  (`smoke_cc_rail_e2e.py:501-505`): measure it with `--dry` first.
- `--max-rail-calls` may only be **lowered**, never raised
  (`smoke_cc_rail_e2e.py:489-492`).
- On a cap trip the watcher aborts and the script returns 3, but **in-flight
  backend calls continue -- there is no cancel endpoint**
  (`smoke_cc_rail_e2e.py:313-317`). That is the anticipated loud outcome; the
  live_check should quote this line rather than treating a cap trip as a
  clean stop.

---

## External research

### Read in full (>=5 required; counts toward the gate) -- 6 sources

| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| https://code.claude.com/docs/en/costs | 2026-08-06 | Official doc | WebFetch (full page) | "Claude Max and Pro subscribers have usage included in their subscription, so the session cost figure isn't relevant for billing purposes"; "Claude Code computes the dollar figure locally from token counts priced at standard list rates, so it doesn't reflect promotional pricing... and may differ from your actual bill." |
| https://support.claude.com/en/articles/11145838-use-claude-code-with-your-pro-or-max-plan | 2026-08-06 | Official doc (dated 2026-06-11) | WebFetch (full page) | "Both Pro and Max plans offer usage limits that are shared across Claude and Claude Code, meaning all activity in both tools counts against the same usage limits." No headless/CLI carve-out. |
| https://www.anthropic.com/news/higher-limits-spacex | 2026-08-06 | Vendor announcement (2026-05-06) | WebFetch (full page) | RECENCY DELTA: "doubling Claude Code's five-hour rate limits" and "removing the peak hours limit reduction on Claude Code", "all effective today" (2026-05-06). |
| https://sre.google/workbook/canarying-releases/ | 2026-08-06 | Authoritative practitioner (Google SRE Workbook) | WebFetch (full chapter) | Canarying = "a partial and time-limited deployment of a change in a service and its evaluation"; "Stack-rank the metrics you want to evaluate..."; "Select the top few metrics... (perhaps no more than a dozen)"; abort: "we should pause and roll back the deployment". |
| https://principlesofchaos.org/ | 2026-08-06 | Canonical (year-less prior art) | WebFetch (full text) | "Build a Hypothesis around Steady State Behavior"; "Minimize Blast Radius -- ...it is the responsibility and obligation of the Chaos Engineer to ensure the fallout from experiments are minimized and contained." |
| https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work | 2026-08-06 | Official doc | WebFetch (full page) | **Negative result, recorded honestly:** the page does NOT carry 5-hour/weekly mechanics, token denominations, or the all-models-vs-Sonnet split -- it defers to linked articles. Confirms the 4000.1/4000.2 finding that **no token-denominated weekly-pool figure is published**, which is exactly what `smoke_cc_rail_e2e.py:234-235` already discloses. |

### Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://support.claude.com/en/articles/11049741-what-is-the-max-plan | Official doc | Plan marketing; 4000.1 already read the Max-plan surface |
| https://claude.com/pricing | Official | Pricing table, no window-relevant delta |
| https://support.claude.com/en/articles/12429409-manage-extra-usage-for-paid-claude-plans | Official doc | Credits mechanics covered by the costs doc read in full |
| https://support.claude.com/en/articles/9797557-usage-limit-best-practices | Official doc | Advisory, not gate-relevant |
| https://platform.claude.com/docs/en/api/rate-limits | Official doc | API-metered rail, out of scope (window is $0-metered) |
| https://www.anthropic.com/news/max-plan | Vendor | Superseded by the 2026-05-06 limits change |
| https://support.claude.com/en/articles/8325606-what-is-the-pro-plan | Official doc | Wrong plan tier |
| https://www.getunleash.io/blog/canary-release-vs-smoke-test | Industry blog | Lower tier than the SRE Workbook chapter read in full |
| https://www.optimizely.com/optimization-glossary/canary-testing | Industry glossary | Definitional only |
| https://www.harness.io/harness-devops-academy/integrating-smoke-testing-into-your-ci-cd-pipeline-what-devops-needs-to-know | Vendor blog | "Smoke tests... in minutes, not hours" -- noted, not authoritative |
| https://www.mida.so/blog/canary-testing | Industry blog | Community tier |
| https://nerdleveltech.com/ai-agent-reliability-verification-loops-guardrails | Blog (2026-07) | Secondary reporting on July-2026 guardrail papers; not primary |
| https://www.getmaxim.ai/articles/top-5-ai-guardrail-solutions-for-production-llm-applications-in-2026/ | Vendor listicle | Commercial, low weight |
| https://futureagi.com/blog/llm-guardrails-safeguarding-ai-2025/ | Vendor blog | Commercial, low weight |

Unique URLs collected: **20** (6 read in full + 14 snippet-only).

### Search-query composition (3-variant discipline)

1. Current-year frontier: `Anthropic Claude Max plan usage limits weekly 2026 Claude Code` (domain-scoped to Anthropic properties).
2. Year-less canonical: `testing in production bounded blast radius abort criteria smoke test canary`.
3. Last-2-year window: `production experiment guardrails pre-registered stopping rule 2025 2026 reliability`.

### Recency scan (2024-2026)

Performed. **One finding that supersedes the 4000.1/4000.2 reads, and it is
favourable:** Anthropic's 2026-05-06 announcement states it is "doubling
Claude Code's five-hour rate limits" and "removing the peak hours limit
reduction on Claude Code" for Pro/Max, "all effective today"
(https://www.anthropic.com/news/higher-limits-spacex). The 4000.1/4000.2
briefs' headroom assumptions were therefore CONSERVATIVE, not optimistic --
the window has more 5-hour headroom than those briefs assumed, and there is no
peak-hours penalty to schedule around. **No change to the $0-metered
constraint**: the shared-limits statement
(support.claude.com/.../11145838) and the credits mechanism
(code.claude.com/docs/en/costs) are unchanged, so past the ceiling the rail
still moves to metered credits. Second finding, negative: still **no published
token-denominated weekly-pool number** -- the smoke's existing
`weekly_pool_note` disclosure (`smoke_cc_rail_e2e.py:234-235`) remains the
honest framing and must not be replaced by a fabricated percentage.

### Key findings (external -> window)

1. **The CLI's `costUSD` is a locally computed list-rate estimate, not a
   bill.** "Claude Code computes the dollar figure locally from token counts
   priced at standard list rates... and may differ from your actual bill"
   (code.claude.com/docs/en/costs, accessed 2026-08-06). E6 must present its
   USD figure as an *extrapolated proxy*, and E2's $0-metered claim must rest
   on the `llm_call_log` metered-complement rule, never on the envelope's
   `total_cost_usd`.
2. **No headless carve-out.** "all activity in both tools counts against the
   same usage limits" (support.claude.com/.../11145838) -- the window's
   subprocess calls draw the same pool as the operator's interactive session,
   so a concurrent second Claude Code session inflates the pool draw the E6
   number is trying to measure. This is an independent argument for the
   single-writer precondition the step already requires.
3. **Pre-register the metric set and keep it small.** "Select the top few
   metrics to use in canary evaluations (perhaps no more than a dozen)"
   (Google SRE Workbook). E1-E6 (six checks) is already inside that bound and
   was frozen in 4000.1 -- do not add checks mid-window.
4. **Hypothesise the steady state before the experiment; minimize blast
   radius** (principlesofchaos.org). Maps to: the AAPL-not-NTAP ticker choice
   (Q1), the `<=2` tickers / `<=30` rail-calls caps, and the pre-registered
   kill criteria (E3 mismatch, any metered increment).
5. **A canary is time-limited and its interval must fit the metric interval**
   ("Make sure the intervals of your metrics are either the same as or less
   than your canary duration", SRE Workbook). Maps to Q2: the window must
   close before the 18:00 UTC cycle, and `--flush-wait-s` (default 65) exists
   because `llm_call_log` is buffered -- a shorter window than the buffer
   would read an empty table (the 4000.2 finding).

### Consensus vs debate

Consensus across the SRE/chaos sources: production experiments are legitimate
and preferable to staging inference, PROVIDED blast radius is bounded, metrics
are pre-registered, and abort criteria are written before the run. No source
disagrees with running a bounded live test; the debate in the literature is
only about canary POPULATION sizing (percentage-of-traffic), which does not
transfer to a one-shot n=1 analysis -- here the analogue of "1-5% of traffic"
is "1 ticker, not the held one, outside the decision path", which Q1
establishes is achievable.

### Pitfalls (from literature + this audit)

- Presenting a locally-computed `costUSD` as a bill (costs doc).
- Deploying at a non-representative time -- here the inverse risk: deploying at
  a time that OVERLAPS the 18:00 UTC cycle and contaminates the window.
- Adding metrics mid-experiment (SRE Workbook's stack-rank-first guidance is
  the countermeasure; 4000.1 already froze E1-E6).
- Assuming a Gemini/Claude token split that does not exist under the live pins
  (Q3).

## Internal code inventory

| File | Anchors used | Role | Status |
|---|---|---|---|
| `backend/api/analysis.py` | 33, 43, 210-311, 316-328, 336-345, 350-381, 393-394, 406-416 | Manual-analysis entrypoint; the window's tool | Live; `_tasks` never evicted (minor, benign) |
| `backend/services/autonomous_loop.py` | 610, 890, 928-950, 1417-1430, 1461-1469, 1637-1666 | Autonomous cycle; the thing to avoid | Live |
| `backend/services/portfolio_manager.py` | 66-90 | `decide_trades`; consumes in-cycle analyses only | Live |
| `backend/api/paper_trading.py` | 45-46, 110-120, 1159-1240 | Scheduler job + the one `analysis_results` read seam (metadata) | Live |
| `backend/agents/orchestrator.py` | 380-402, 517-519, 652-664, 746, 815, 884, 945-948, 1153, 1178-1352, 1408-1670 | Layer-1 pipeline; step budgets + client routing | Live; stale comment at 654-658 |
| `backend/agents/llm_client.py` | 2113-2126 | Rail routing decision | Live |
| `backend/agents/claude_code_client.py` | 583, 591 | Rail client; `supports_grounding=False`, `recommended_step_timeout` | Live |
| `backend/config/settings.py` | 176, 186-191, 380-381 | Flag + timeouts + universe sizes | Live |
| `backend/main.py` | 309-314 | Scheduler construction | Live |
| `scripts/qa/smoke_cc_rail_e2e.py` | 234-235, 240-251, 262, 288-290, 299-317, 387-388, 410-412, 478-505 | The window's tool | Built (4000.2) |
| `backend/tools/screener.py` | 249 | `rank_candidates`; zero `analysis_results` references | Live |

## Application to pyfinagent (the five things 4000.3's contract should carry)

1. Ticker: **AAPL** (Q1). Not NTAP. Runner-up JNJ.
2. Pass **`--analysis-deadline 3600`** (Q2); client-side bound only, cannot
   raise rail calls or server budgets.
3. Close the window before **18:00 UTC 2026-08-06** (measured `next_run`), or
   stop/start the scheduler with its own operator note (Q2).
4. Do **not** pass `--keep-on`; `pre_flag` is already True so the restore
   satisfies the operator's end-state rule by itself (Q3).
5. E6 attributes ~100% of window tokens to the Max pool and must label its USD
   figure a list-rate proxy; disclose that grounding was OFF (Q3, external #1).

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6)
- [x] 10+ unique URLs total incl. snippet-only (20)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (11 files)
- [x] Contradictions / consensus noted (incl. the stale `quant_exec_client`
      comment and the "flag already True" divergence from the step name's
      framing of a flip)
- [x] Claims cited per-claim
- Gap: tier is `simple`, so no multi-pass / adversarial sourcing was done; the
  brief exceeds the ~300-word `simple` guide length because the caller asked
  four anchor-dense internal questions. Analysis DEPTH is simple-tier.

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 14,
  "urls_collected": 20,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Ticker: AAPL (NTAP is the only holding and the sole analysis_results read seam is metadata-only via _fetch_ticker_meta; decide_trades never reads BQ analyses). Timing: paper_trading_daily next_run measured at 2026-08-06T14:00 ET = 18:00 UTC -- close the window before it; raise --analysis-deadline 1800 -> 3600 (3 attempts x ~180s step budget on one step alone). Cost shape: paper_use_claude_code_route is ALREADY True and both pins are claude-*, so ~100% of window tokens hit the Max pool; grounding is OFF (ClaudeCodeClient.supports_grounding=False) so the 4 grounded steps silently run ungrounded on the rail; Gemini's only possible share is the RAG step, which may be skipped. Failure: a FAILed analysis returns a terminal poll body with [Type] Step 'name' error; E4 catches it; no BQ row, no partial signals, no Slack; only the in-memory _tasks entry persists (never evicted) so do not restart before evidence capture. Do NOT pass --keep-on: pre_flag is True so the restore already satisfies the end-state rule.",
  "brief_path": "handoff/current/research_brief_4000.3.md",
  "gate_passed": true
}
```

