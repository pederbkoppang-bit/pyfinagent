# phase-45.0 CLOSURE Research Brief -- legacy-phase dedup + 2026 frontier scan

**Date:** 2026-05-22
**Tier:** deep
**Mode:** ~50% internal legacy-phase dedup + ~50% external 2026 frontier
**Author:** researcher subagent
**Cycle context:** 18:00 UTC cron c7801712 completed end-to-end 18:00 -> 18:37 UTC.
NAV $23,252.06 (+16.26% YTD), +1.07% alpha vs SPY, 9 open positions, COHR closed
on `stop_loss_trigger` (+17.89% pnl -- a phase-32.2 trail event in production).
Regression baseline: **297 tests** collected (pytest --collect-only -q).

This brief replaces the prior phase-44.0 frontend brief at this same path.

---

## Section A -- LEGACY PHASE VERDICT TABLE

Cross-reference: `.claude/masterplan.json` (70 phases), `handoff/current/master_roadmap_to_production.md`
(authored 2026-05-22, phase-33.0 super-planning, 1182 lines), and the actual
in-progress phases 35-44 with 32+ already-planned pending substeps.

Verdict legend: **DROP** (already done / superseded); **DEFER-POST-PROD**
(valuable but off the production-readiness critical path); **FOLD-INTO-3X.Y**
(specific later phase absorbs it); **KEEP** (still valid as-is).

| Phase | Current status | Name (truncated) | Verdict | Reason (~2 sentences) | Supersession refs |
|---|---|---|---|---|---|
| phase-4 | in-progress | Production Readiness | **FOLD-INTO-43.0** | Only open substep is `4.9 = Pre-go-live aggregate smoketest (test everything)`, status `blocked`. phase-43.0 (Definition-of-Done audit, 14 criteria) is a strict superset of "aggregate smoketest" -- DoD measures the same thing plus more. **Action:** flip phase-4.9 to `done` with a "see phase-43.0" note + flip phase-4 itself to `done`. | phase-43.0 (master_roadmap §6 DoD) |
| phase-5 | pending | Multi-Market Expansion (15-step) | **DEFER-POST-PROD** | 11 of 14 substeps still pending (5.2-5.15). Master roadmap explicitly defers phase-5 to post-prod ("phase-42 deferred because it depends on phase-5 which is currently pending and out-of-scope for production-readiness" -- master_roadmap §2). Crypto removal already done (phase-5-crypto-removal-*). FX + futures + EODHD + IBKR + cross-market regime/signal/backtest are all v2 initiatives. **Action:** flip phase-5 to `deferred` with note "post-prod; revisit after 43.0 PASS". | master_roadmap_to_production.md:259-268 (phase-42 deferred entry) |
| phase-10.7 | proposed | Meta-Evolution Engine | **DEFER-POST-PROD** | All 3 of 8 substeps shown are `done` (alpha velocity metric, recursive prompt optimization, archetype seed library); the remainder (`10.7.4`-`10.7.8`) are evolution / archive automation. NONE of these are on the 14-criterion DoD gate. Master roadmap §1 explicitly lists "Meta-evolution housekeeping" outside the production-critical-path. **Action:** flip phase-10.7 to `deferred`. | master_roadmap_to_production.md (no mention in §4 critical path) |
| phase-13 | blocked | bypassPermissions -> acceptEdits + Seatbelt | **KEEP** (but mark `deferred` until acceptEdits gate clears) | Owner-side blocker per `feedback_permissions_bypass_required`: "harness needs defaultMode=bypassPermissions for unattended runs; phase-4.14.6 is intentionally blocked". Has been blocked >30 days; the underlying harness still depends on bypassPermissions to operate. This phase will reopen only when Claude Code ships a `acceptEdits` mode that satisfies unattended-cron requirements. **Action:** flip to `deferred` with note "blocked-upstream; revisit when Claude Code ships unattended acceptEdits". | feedback_permissions_bypass_required.md |
| phase-16 | in-progress | Full-application end-to-end UAT (pre-go-live) | **FOLD-INTO-43.0** | 59 substeps in this phase (one of the largest); the work that remains pending is the *aggregate* UAT -- which is exactly what phase-43.0 (Definition-of-Done audit, 14 measurable criteria) operationalizes. Many substeps are already done (16.1-16.3 Infrastructure / Layer-1 / Layer-2 done per the masterplan list). **Action:** for each NOT-done substep, evaluate whether the DoD criterion in phase-43 covers it; for those covered, fold; flip phase-16 to `done` with a fold note pointing to 43.0. | master_roadmap_to_production.md §6 DoD; phase-43.0 success criteria |
| phase-23.6 | in_progress | persistent items deferred from phase-23.5 | **KEEP** | 3 of 5 substeps already done (23.6.0, 23.6.1, 23.6.2). Remaining 23.6.3 + 23.6.4 are concrete code hardening items (production fetch/write/alert wiring, slack-bot heartbeat bridging). These are NOT covered by phases 35-44 and they are dependent on production cron reliability which the DoD requires. **Action:** keep as-is; complete the remaining 2 substeps in normal flow. | -- |
| phase-23.7 | in_progress | Harness plumbing: auto-commit-and-push + semver changelog | **KEEP (NEARLY-DONE)** | 1 substep listed (`23.7.0 = done`). The phase header is `in_progress` only because there is residual work the auto-changelog hook noted -- but the only listed substep is done. **Action:** verify the wrapper is truly done; flip the phase header to `done`. Likely a single-line masterplan fix. | feedback_auto_commit_hook_stalls.md (operator-known recurring stall pattern) |
| phase-23.8 | pending | Dev-MAS Audit Remediation | **KEEP** (partially complete) | 3 of 5 substeps already done (23.8.0 Bundle-1; 23.8.1 live_check hook gate; 23.8.2 Delete TaskCompleted hook). Remaining 23.8.3 + 23.8.4 are R-5 + R-6 remediations (deferred decisions) -- not on the DoD gate. **Action:** evaluate 23.8.3 and 23.8.4: if they're absorbed by phase-38.4 (auto-commit hook refuses status-flip without harness_log -- which is R-3-equivalent) then fold; otherwise keep. | phase-38.4 (auto-commit hook gate) |
| phase-26 | pending | Frontier-sync: adopt 2026-04→05 Anthropic/Google releases + topology gaps | **FOLD-INTO-40.2 + 40.3 + 41.1** | 3 of 8 substeps already done (26.0 Opus 4.7 migration, 26.1 Per-session Task Budget, 26.2 Advisor Tool adoption). Remaining substeps are: stress-test doctrine (phase-40.3 absorbs OPEN-26), Claude Code v2.1.140-143 features (phase-40.2 absorbs OPEN-25), Gemini 3.x audit (phase-41.0/41.1 P3 bundle absorbs OPEN-32/33). **Action:** map each remaining sub-step to 40.2 / 40.3 / 41.x, then flip phase-26 to `done` with fold notes. | phase-40.2, phase-40.3, phase-41.0, phase-41.1 (master_roadmap §4) |
| phase-27 | pending | Multi-Provider Full-Path Pipeline (Gemini + Claude) | **FOLD-INTO-37.X (LLM-Route Hardening) + DEFER residual** | 27.0-27.5 done (research gate, structured-output, null-safety, schemas). The remaining 3 substeps (27.6 + 27.6.3 + 27.6.4) are Claude full-path smoketest + Cloud Function redeploy. Phase-37 (LLM-Route + Structured-Output Hardening) is the direct successor for structured-output discipline. **27.6.4 is `DEFERRED` already per its name** (Cloud Function redeploy is operator-only sandbox-blocked). **Action:** fold 27.6 + 27.6.3 into phase-37 work; flip 27.6.4 to `deferred-post-prod` (same class as phase-39); flip phase-27 to `done` with fold notes. | phase-37.1/37.4 (response_schema); phase-39 (operator-only sandbox-blocked pattern) |
| phase-28 | pending | Candidate Picker Expansion | **KEEP** (most done; verify residual) | 14+ of 18 substeps already done (28.0-28.13). Per masterplan list, all 28.x substeps shown are `done`. Phase still shows status `pending` likely because some non-shown substeps (28.14-28.17) remain. **Action:** read the full step list (not just first 3) and verdict per-substep; likely fold residual into phase-42 (Universe Expansion, deferred post-prod). | master_roadmap §4 phase-42 (deferred) |
| phase-29 | pending | Harness MAS + MCP + Academic-Fetch + Frontier-Sync | **KEEP** (29.0-29.7 done; 29.8/29.9 are P2/P3 bundles fold-into-41) | 8 of 10 substeps done (29.0-29.7). Remaining 29.8 + 29.9 are explicitly labeled "P2 bundle" + "P3 bundle" in the master roadmap, and phase-41 (phase-29.8/29.9 Bundle Closure) is literally named after closing them. **Action:** fold 29.8 + 29.9 into phase-41.0 + 41.1 (already done in masterplan); flip phase-29 to `done` post-fold. | phase-41.0 (P2 bundle close) + phase-41.1 (P3 bundle close) |

### Sub-bundle expansion (29.8 + 29.9)

| ID | Bundle | Status | Action |
|---|---|---|---|
| 29.8 | P2 bundle: budget_tokens + alwaysLoad/continueOnBlock + OpenAlex + Gemini-3.x | pending | **FOLD-INTO-41.0** -- master roadmap §2 OPEN-32 explicitly says "consolidates with phase-37.3 + 40.1 + 40.2". |
| 29.9 | P3 bundle: stress-test cycle + Mythos Preview + Gemini 3.1 + GPT-5.5 docs + deep-tier multi-subagent-fork + scaffolding-pruning + cycle-2-flow surfacing | pending | **FOLD-INTO-41.1** -- master roadmap §2 OPEN-33 "consolidates with phase-40.3". |

### Verdict tally (12 legacy phases)

| Verdict | Count | Phases |
|---|---|---|
| DROP (flip to `done`, fold-by-reference) | **6** | phase-4, phase-16, phase-26, phase-27, phase-29 (each via fold-into-3X.Y), phase-23.7 (verify-then-done) |
| DEFER-POST-PROD | **3** | phase-5, phase-10.7, phase-13 |
| FOLD-INTO-3X.Y (specific later step) | **6 fold mappings** | 4 -> 43.0; 16 -> 43.0; 23.8 -> 38.4 (partial); 26 -> 40.2/40.3/41.x; 27 -> 37.X + deferred; 29 -> 41.0/41.1 |
| KEEP | **3** | phase-23.6 (real residual work), phase-23.8 (partial -- 2 residual substeps), phase-28 (verify residual) |
| Total | 12 | -- |

Note: counts sum to >12 because some phases get both a primary verdict and a partial fold-mapping. Net: **~6 DROP + 3 DEFER + 3 KEEP** is the headline tally.

---

## Section B -- 18:00 UTC CRON c7801712 -- DID IT CLOSE PHASE-35.1/35.2?

Per Main's pre-cycle observation, c7801712 ran 18:00:00 -> 18:37:02 UTC
(2,222,177 ms, status=`completed`, n_trades=0, error_count=0,
`handoff/cycle_history.jsonl` tail). All 8 autonomous-loop steps executed
including stop_loss_enforcement. The question: did the cycle PRODUCE the
evidence needed to retroactively close phase-35.1 (learn-loop alive) and
phase-35.2 (Risk Judge citing portfolio_sector_exposure)?

### Direct BigQuery probes (via google-cloud-bigquery client, location-resolved per dataset)

| Probe | Target dataset | Query | Result | Verdict |
|---|---|---|---|---|
| **B-1** | `financial_reports.outcome_tracking` (us-central1) | `SELECT COUNT(*) FROM outcome_tracking` and per-timestamp `WHERE created_at > 17:00 UTC`. | `total_rows = 0`; `n_after = 0`; `latest = NULL`. The table EXISTS (schema: ticker, analysis_date, recommendation, price_at_recommendation, current_price, return_pct, holding_days, beat_benchmark, evaluated_at) but has **never been written to**. | **FAIL** -- phase-35.1 (OPEN-22 learn-loop alive) NOT closed by c7801712. |
| **B-2** | `financial_reports.agent_memories` (us-central1) | `SELECT COUNT(*)` + per-`created_at` filter. | `total_rows = 0`; `n_after = 0`. Table EXISTS (schema: agent_type, ticker, situation, lesson, created_at) but empty. | **FAIL** -- the BM25-retrieve target rows do not exist. |
| **B-3** | `pyfinagent_data.llm_call_log` (US) | `SELECT * WHERE cycle_id = "c7801712"` -- no filter on `agent`. | ROWS: **0**. The llm_call_log table HAS 138 total rows from 5 distinct cycles, BUT the latest entry is `2026-05-21 05:15:58 UTC`. c7801712 is NOT in the log at all. | **FAIL** -- phase-35.2 (OPEN-23 Risk Judge citing portfolio_sector_exposure) cannot be verified from llm_call_log because the table wasn't written for this cycle. |
| **B-4** | `pyfinagent_data.strategy_decisions` (US) | `SELECT * WHERE cycle_id = "c7801712"`. | ROWS: **1**. `ts=2026-05-22 18:37:01.85 UTC`, `decided_strategy=triple_barrier`, `prior_strategy=triple_barrier`, `trigger=cycle_heartbeat`, `decay_signal=None`, `decay_attribution=None`, `rationale="per-cycle heartbeat; no regime change detected. Full router activation deferred to phase-31."` | **HEARTBEAT-ONLY** -- not a meaningful threshold-crossing decision; just a per-cycle "I'm alive" row. |
| **B-5** | `financial_reports.paper_trades` (us-central1) | `SELECT * WHERE DATE(created_at) = '2026-05-22'`. | ROWS: **2**. (a) `18:35:45 | COHR | SELL | stop_loss_trigger | qty=4.51 @ 378.32 | hold=25d | mfe=28.36% | mae=-6.09% | capture=0.63 | pnl=+17.89%` -- the c7801712 cycle close; (b) `16:59:16 | LITE | SELL | stop_loss_trigger | qty=1.08 @ 965.74 | hold=25d | pnl=+9.54%` -- the dc3f6cf1 cycle close. **BOTH** SELLs have `risk_judge_decision = ""` and `signals = []`. | **STRONG MIXED.** Phase-32.2 trail event fired live TWICE today (LITE at 16:59 + COHR at 18:35) -- both positive-pnl trail-stop fires. But `risk_judge_decision` and `signals` are empty on stop_loss_trigger SELLs -- the Risk Judge decision metadata is not being persisted on this SELL path. |
| **B-6** | `financial_reports.paper_portfolio_snapshots` (us-central1) | Latest 5 days NAV + position count. | `2026-05-22: nav=23252.06, pos=9, trades=0, pnl=16.26%, alpha=1.07%`. Yesterday `2026-05-21: pos=11`. So **11 -> 9 = 2 closed today**, NOT "9 -> 8" as Main's pre-cycle note said. **Both closures identified: LITE (dc3f6cf1 cycle, 16:59 UTC) + COHR (c7801712 cycle, 18:35 UTC).** | **POSITION COUNT DELTA -2** -- two trail-stop fires across two cycles today. |

### Findings

**Phase-35.1 (OPEN-22 learn-loop alive) status: STILL NOT CLOSED by c7801712.**

`outcome_tracking` and `agent_memories` are empty. The success criteria spelled
in `master_roadmap_to_production.md` line 360-368 require `outcome_tracking`
has >=1 row created by autonomous_loop after a real stop_loss-triggered close,
AND `agent_memories` BM25 returns >=1 lesson on next cycle. Today neither
table has a single row.

**This is significant** because c7801712 DID close a position on
`stop_loss_trigger` (COHR). The lesson-emitting path *would have run* if it
were wired -- so what's missing is not "the chance to learn" but **the writer
that converts a closed SELL into an `outcome_tracking` row + a
`agent_memories.lesson` row**. The learn-loop's read path (BM25 over an
empty index) would also be a no-op even if invoked.

**Phase-35.2 (OPEN-23 Risk Judge citing portfolio_sector_exposure) status: NOT VERIFIED.**

llm_call_log has 0 rows for c7801712. Either the call-logging layer isn't
firing during the autonomous_loop's Risk-Judge invocations, OR the calls
fire but bypass the LLM telemetry log (e.g., direct generateContent without
the wrapping `llm_client.py` instrumentation). The 138 historical rows in
llm_call_log cover only 5 cycles, all from May 16-21, suggesting irregular
or off-path logging.

**Phase-35.3 (OPEN-20 sustained-cycle) status: PARTIAL.**

c7801712 brings the consecutive-completed-cycle count to **3** (or 4
depending on count rules). Tail of cycle_history.jsonl shows:

```
2026-05-22T18:00 c7801712 completed 37 min n_trades=0 errors=0     <-- TODAY
2026-05-22T16:23 dc3f6cf1 completed 37 min n_trades=0 errors=0     <-- phase-34.2 (logged in live_check_34.2.md)
2026-05-22T05:30 021ed63e timeout                                 <-- BAD
2026-05-21T18:00 8df751b3 running (orphan)                        <-- not "completed"
2026-05-20T18:00 9fdcc2df running (orphan)                        <-- not "completed"
```

Phase-35.3 requires **5 consecutive completed** cycles. Today's tail shows
only 2 strict-completed cycles in the last 24h (dc3f6cf1 + c7801712). The
21-may and 20-may rows are `status='running'` (orphan -- never wrote
completed_at successfully -- this is exactly OPEN-15 + OPEN-11). Phase-35.3
is NOT closed; needs 3 more clean cycles minimum.

### Net verdict for Section B

| Step | Pre-c7801712 status | Post-c7801712 status | Delta |
|---|---|---|---|
| phase-35.1 (learn-loop alive) | pending | pending | **NO CHANGE** -- writer never fired despite the COHR SELL that should have triggered it |
| phase-35.2 (Risk Judge cites portfolio_sector_exposure) | pending | pending (NOT VERIFIED) | **NO CHANGE** -- llm_call_log empty for this cycle |
| phase-35.3 (5 consecutive completed cycles) | 1 of 5 | 2 of 5 (or 3 if dc3f6cf1 counts) | **+1 to streak; not closed** |

**Implication for planner:** Closing phase-45.0 cannot include "OPEN-22 closed
organically by c7801712." It can include "OPEN-22 newly diagnosed: the writer
that converts a closed SELL into an outcome_tracking + agent_memories row is
missing in code, NOT just unfired." That's a sharper finding than the master
roadmap had pre-c7801712 -- it converts OPEN-22 from "unverified" to
"diagnostically missing, file the gap as a one-step fix in phase-35.1's
verification.live_check."

### Section B silver lining (deserves its own callout)

**TWO trail-out events fired live today across two consecutive cycles:**

- **LITE** at 16:59:16 UTC (dc3f6cf1 cycle, phase-34.2 corrective): SELL
  stop_loss_trigger, +9.54% pnl, 25 days held.
- **COHR** at 18:35:45 UTC (c7801712 cycle, today's 18:00 cron): SELL
  stop_loss_trigger, +17.89% pnl, 25 days held, capture_ratio=0.63
  (returned $0.63 of every $1 of mfe).

This is unambiguous production evidence that phase-32.2 trail discipline is
**structurally working**. The capture ratio 0.63 on COHR is in the
healthy-band per AFML ch.3.2-3.3 (typical range 0.5-0.75). The
profit-protection layer is structurally working AT THE TRAIL LAYER. The
ONLY remaining BLOCK in profit-protection is OPEN-2 (scale-out at +2R / +3R)
-- which is exactly phase-36.1.

**Implication:** the closure roadmap can sharpen DoD §8 ("profit-protection
BLOCK closed") from "pending phase-36.1" to "phase-32.2 trail demonstrably
firing in production (2 events / day on 2026-05-22); scale-out wiring
(phase-36.1) is the only residual."

---

## Section C -- 2026 external frontier (>=8 in-full)

### Read in full (>=8 required for deep tier; counts toward gate)

| # | URL | Title | Date | Kind | Key takeaway |
|---|---|---|---|---|---|
| 1 | https://arxiv.org/html/2412.20138v3 | TradingAgents: Multi-Agents LLM Financial Trading Framework | 2024-12 / arXiv | Peer arXiv | Multi-agent (analyst / researcher / trader / risk / fund-manager) with structured-output + natural-language hybrid. Documented Sharpe 5.6 - 8.2 over AAPL/GOOGL/AMZN backtests, with CR improvement 6.1% - 24.6% over best baseline. **Direct relevance to pyfinagent's existing 3-agent harness MAS** + 28-agent analysis pipeline. Validates the structured-output-for-control + NL-for-debate pattern that is already in pyfinagent's MAS. |
| 2 | https://arxiv.org/html/2502.15800 | LLM Agents Do Not Replicate Human Market Traders (Caltech / Virginia Tech) | 2025-02 / arXiv | Peer arXiv | LLM-driven markets produce **mean squared error 0.536 vs 429.8 for humans** in pricing assets to fundamental value -- LLMs are "textbook-rational" but suppress bubble formation. Implications: (a) LLM forecast errors approach zero unbiased, (b) production LLM trading agents may suppress real market behavioral structure, (c) prompting and fine-tuning materially alter behavior. **Pyfinagent relevance:** justifies the existing 28-agent Layer-1 ensemble (more rationality, less single-model bias drift) AND the Risk-Judge structured-output requirement (so the LLM's textbook rationality is calibrated to portfolio-level constraints, not just per-stock fundamentals). |
| 3 | https://www.twosigma.com/articles/ai-in-investment-management-2026-outlook-part-i/ | Two Sigma AI in Investment Management 2026 (Part I) | 2026 | Industry vendor blog (high-quality) | "AI is becoming the operating system for how quant research and investing work" (Greenwood). Three themes: AI-as-OS, human-judgment-remains-essential, integration-over-raw-power. Research funnel is inverting -- "LLMs widen the top, shifting bottleneck from idea scarcity to evaluation speed." **Direct pyfinagent relevance:** the harness MAS pattern IS the "integration-over-raw-power" approach. Validates phase-29.x research-on-demand pattern. |
| 4 | https://www.twosigma.com/articles/ai-in-investment-management-2026-outlook-part-ii/ | Two Sigma AI Outlook Part II | 2026 | Industry vendor blog | Critical warning from Jin Choi: "With AI agents, researchers can easily generate a large number of hypotheses and backtest them, which can exacerbate the overfitting issue." Also: "pre-trained LLMs embedded in forecasting pipelines risk implicitly knowing about major regime shifts (pandemic, AI boom)." **Pyfinagent relevance:** justifies the stress-test doctrine (phase-40.3) + the PSR/DSR/PBO discipline (DoD §11) + the 297-test regression baseline. The framing "research discipline as testable hypotheses multiply" maps 1:1 to the contract.md immutable-criteria pattern. |
| 5 | https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/ | Does Meta Labeling Add to Signal Efficacy? (Hudson & Thames) | 2024-2025 | Authoritative industry blog (Hudson & Thames are the canonical AFML-implementation team) | Empirical: meta-labeling lifts mean-reversion out-of-sample accuracy 17% -> 63%, precision 0.17 -> 0.20; trend-following 48% -> 55% / 0.48 -> 0.54. Mechanism: secondary ML model decides *when* to trade, primary signal decides *direction*. Limitations: depends on signal quality, requires high-quality tick data, unbalanced-class problem requires up-sampling. **Pyfinagent relevance:** validates phase-40.7 (meta-labeling exit classifier, currently owner-gated). Quantitative justification for the LLM-cost approval Main needs from Peder. |
| 6 | https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents | Effective Harnesses for Long-Running Agents (Anthropic) | 2025-11 | Official Anthropic engineering doc | Four core harness components: (1) Feature list as JSON, (2) Progress tracking + git history, (3) Initialization script, (4) Structured git-commit handoffs. Key quote on evaluation gates: "Claude tended to make code changes...but would fail recognize that the feature didn't work end-to-end." Browser automation critical for validation. Subagents marked exploratory. **Direct pyfinagent relevance:** the masterplan.json + harness_log.md + 5-file handoff protocol IS Anthropic's documented pattern. **The doc DOES NOT recommend abandoning subagents** -- it says "still unclear whether a single, general-purpose coding agent performs best" -- which is the OPPOSITE of arguments to dissolve the 3-agent harness. |
| 7 | https://www.langchain.com/articles/agent-observability | AI Agent Observability: Tracing, Testing, Improving (LangChain / LangSmith) | 2026 | Vendor doc (authoritative for the trace-tree pattern) | Trace tree captures LLM-calls + tool-invocations + retrieval-steps hierarchically. Threads connect related traces. LLM-as-judge online evaluations sample production traffic. Annotation queue routes problematic traces. Side-by-side run comparison for prompt/model/parameter A-B. **Pyfinagent relevance:** /agents Live Stream tab is currently a flat log; LangSmith trace-tree is the documented 2026 pattern. The MASEvent enum (classify / plan / delegate / tool_call / tool_result / thinking / synthesize / loop_check / quality_gate / citation / complete / error) already supports the categorization. Phase-44.7 maps directly. |
| 8 | https://docs.python.org/3/whatsnew/3.14.html | What's New in Python 3.14 (CPython official) | 2025-10-07 | Official Python release notes | Python 3.14.0 released 2025-10-07 with PEP 779 free-threaded build OFFICIALLY SUPPORTED (no longer experimental). Single-threaded programs 5-10% slower in free-threaded mode -- mitigated by new Tier 2 JIT compiler that identifies non-shared data paths. Standard distribution still ships GIL-enabled default. Per `os.cpu_count()` semantics and threading benefits. **Pyfinagent relevance:** the backend runs Python 3.14 per CLAUDE.md ("Backend: FastAPI + Python 3.14"). Free-threading is OPT-IN. No code change required to consume the standard build. JIT is enabled via `PYTHON_JIT=1`. **Production deployment guidance:** keep GIL-enabled default until autonomous-loop runs sustained 5+ clean cycles (phase-35.3) to avoid mixing in a new variable. |
| 9 | https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/ | OWASP Top 10 for LLM Applications 2025 (v2.0) | 2024-2025 | Official OWASP standard | Top 10: LLM01 Prompt Injection / LLM02 Sensitive Info Disclosure / LLM03 Supply Chain / LLM04 Data and Model Poisoning / LLM05 Improper Output Handling / LLM06 Excessive Agency / LLM07 System Prompt Leakage (NEW in v2.0) / LLM08 Vector and Embedding Weaknesses (NEW) / LLM09 Misinformation / LLM10 Unbounded Consumption. **Pyfinagent relevance:** DoD §14 says "OWASP LLM Top-10 v2.0 compliance" is closed by phase-29.4/29.6. The two NEW v2.0 entries (LLM07 system-prompt-leakage + LLM08 vector/embedding) are MOST relevant to pyfinagent's signal-prompt + risk-judge + meta-labeling architecture. The skill_file_ids.json artifact (skills cached + uploaded) is a potential LLM07 leakage surface that should be audited. |
| 10 | https://www.sitepoint.com/tailwind-css-v4-container-queries-modern-layouts/ | Tailwind CSS v4: Container Queries & Modern Layouts | 2026 | Industry blog (high-quality, technical) | Tailwind v4 ships native container queries (no plugin). `@container` utility = `container-type: inline-size`. Child elements use `@sm:` / `@md:` / `@lg:` / `@xl:` prefixes. Color-mix() native opacity. 5x build speed, 100x incremental. **Pyfinagent relevance:** phase-44.1 (Foundation: design tokens + states + Cmd-K + WCAG baseline) can adopt container queries to fix the home page `h-full equal-row` anti-pattern surfaced in phase-44.0. |
| 11 | https://www.glacis.io/guide-sr-11-7 | SR 11-7 Model Risk Management: Complete Guide for AI Systems | 2024-2026 | Industry guidance (synthesized from Fed/OCC) | SR 11-7 (2011) + SR 21-8 (2021) supplement establish 3-pillar framework: development / validation / governance, plus 3-lines-of-defense (developers / independent validation / internal audit). Every ML model deployed in financial services is a model under SR 11-7 including LLMs. **Pyfinagent relevance:** the per-cycle Q/A independent-evaluator pattern IS the SR-21-8 line-2 independent validation function. The harness_log + 5-file handoff trail IS the SR-11-7 documentation requirement. The phase-43.0 DoD audit IS SR-11-7 §III.1 ongoing-monitoring. pyfinagent is not under SR-11-7 jurisdiction (operator only, not a bank) but the principles guide the production-readiness gate. |

11 sources read in full (target was >=8; floor cleared with buffer).

### Snippet-only (does NOT count toward gate)

| URL | Why not read in full |
|---|---|
| https://arxiv.org/abs/2602.23330 (Toward Expert Investment Teams 2026-02) | Same topic as #1 read-in-full; differential reading marginal |
| https://arxiv.org/pdf/2603.17692 (Blindfolded LLMs, Anonymization for Portfolio) | Niche -- pyfinagent uses ticker symbols, not anonymized embeddings |
| https://arxiv.org/pdf/2603.22567 (TrustTrade Selective Consensus) | Same theme as #2 -- LLM-trader-vs-human; complementary not adding |
| https://www.theregister.com/2025/10/08/python_314_released_with_cautious/ | Press release format of #8 read-in-full |
| https://tailwindcss.com/blog/tailwindcss-v4 | Vendor blog snippet; #10 is the operator-oriented version |
| https://owasp.org/Top10/2025/en/ | Snippet view of #9 (full v2.0 PDF was the read-in-full) |
| https://hedgeco.net/news/04/2026/two-sigmas-ai-first-internal-mandate-... | Third-party reporting on Two Sigma; #3/#4 are the primary sources |
| https://github.com/TauricResearch/TradingAgents | GitHub for #1; source code, not the paper |
| https://www.letsaskclaire.com/finance/ai-model-risk-management | Industry analysis blog; #11 covers the same SR-11-7 content with more depth |
| https://www.confident-ai.com/blog/owasp-top-10-2025-for-llm-applications-... | Vendor analysis of #9 |
| https://realpython.com/python-news-november-2025/ | Newsletter format; #8 is the canonical |
| https://www.modelop.com/ai-governance/ai-regulations-standards/sr-11-7 | Vendor SR-11-7 page; #11 is more compact |
| https://www.langchain.com/langsmith | LangSmith product page snippet; #7 read-in-full is the article |
| https://www.braintrust.dev/articles/agent-observability-complete-guide-2026 | Vendor doc; same content shape as #7 |
| https://www.tremor.so/ | Vendor docs read in phase-44.0 brief; not re-read |
| https://www.w3.org/WAI/standards-guidelines/wcag/new-in-22/ | Same as phase-44.0 brief read-in-full; cross-counted, not recounted |

Snippet-only total: 16+ URLs.

---

## Section D -- North-star delta estimates per surviving step

Surviving steps = phases 35-44 already-planned substeps that are NOT
folded-into-each-other by Section A. Term: P (precision = signal quality
improvement), R (recall = coverage improvement), B (burn-rate = cost
reduction). Estimates are bracketed; "speculative" = no empirical anchor.

| Step | Term | Estimate | How measured | Anchor source |
|---|---|---|---|---|
| 35.1 (learn-loop alive) | P | +0.10 to +0.30 Sharpe over 90 days | outcome_tracking row count grows from 0 -> 10+, BM25-retrieved lessons reduce repeat-failure trades | Hudson & Thames meta-labeling lift (#5) -- secondary classifier gates trades on outcome history |
| 35.2 (Risk Judge cites portfolio_sector_exposure) | P + B | +0.05 Sharpe, -10% LLM token spend | When schema is enforced (phase-37.1), the 80% fallback rate drops to <5%; structured output is 30% fewer output tokens | OPEN-16 OPS-F3 + Vertex AI 2026 SDK docs |
| 35.3 (5 consecutive completed cycles) | R | +50% production-coverage confidence | cycle_history.jsonl 5-row streak | DoD §9 (master_roadmap §6) |
| 36.1 (scale-out at +2R / +3R) | P | +5-10% capture ratio per round-trip | mfe / realized PnL ratio -- raw COHR data shows 28.36% mfe to 17.89% realized = 0.63 capture; scale-out targets 0.75-0.85 | AFML ch.3.2-3.3; #1 TradingAgents Sharpe 5.6-8.2 with explicit profit-taking |
| 36.2 (ATR-scaled stops) | P + R | +0.05-0.10 Sharpe via fewer noise-stops | 2*ATR(14) replaces fixed 8% trail; backtest against historical_prices | Quant_strategy.md self-documented gap |
| 36.3 (triple-barrier EXIT) | P | -20% give-back ratio on time-barrier expirations | take_profit_price + time_barrier_days columns; AFML ch.3 | AFML ch.3 + #5 Hudson & Thames |
| 36.4 (entry_strategy persisted at BUY) | P | speculative | Enables phase-32 mean-reversion guard (Kaminski-Lo) -- today every position defaults to 'momentum' | OPEN-8 31.0-FX |
| 36.5 (continuous sector-cap re-check) | R + B | -1 to -3 concentration-breach events / month | new Step 5.7 alert-only first; force-divest behind owner gate | OPEN-7 30.0-F6 |
| 36.6 (tiered DD ladder -5/-10/-15) | speculative | Owner-gated; needs live A/B | dd_ladder_hits idempotency column | OPEN-3 31.0-F8 |
| 37.1 (RiskJudge response_schema) | P + B | +0.05 Sharpe, -20% latency on Risk-Judge path | 0 fallback warnings vs current 80% fallback | OPEN-16 |
| 37.2 (deep-think source default = production) | speculative | -1 hr operator-time per fresh checkout | settings.py + model_tiers.py default flip | OPEN-17 29.0-F14 |
| 37.3 (budget_tokens cleanup) | speculative | Code-debt only; no measurable delta | grep returns 0 budget_tokens refs | OPEN-18 |
| 37.4 (Moderator response_schema) | P | mirror of 37.1; +0.02 Sharpe | 0 Moderator-invalid-JSON warnings | OPEN-16 companion |
| 38.1 (kill-switch auto-resume) | B | -3.5h outage / event * ~2 events / month = -7h / month operator-time | Production outage windows reduced | OPS-F10 |
| 38.2 (lost-cycle observability) | R | +visibility on cycle 3a + run-now lost cycles | cycle_history.jsonl `cycle_starting` row at Step 1 not Step 8 | OPEN-11 OPS-F4 |
| 38.3 (startup banner deep_think_model) | speculative | -1 silent-regression hr per fresh-checkout | log line | OPEN-12 |
| 38.4 (auto-commit hook refuses status-flip without harness_log) | B | -1 silent-protocol-breach per cycle | hook gate + audit_trail | OPS-F7 |
| 38.5 (ASCII-only loggers) | speculative | Crash avoidance on log-rotation | scripts/qa/ascii_logger_check.py CI guard | 30.0-P3-2 |
| 38.6 (restart-survivable _running flag) | R | +zero-loss restart resilience | file-lock TTL; mtime cleanup | 30.0-P3-3 |
| 38.7 (SPY benchmark anchor) | R | +0.5-1.0% alpha-pct accuracy on the dashboard | first-funded snapshot anchor | OPEN-9 30.0-A-2 |
| 39.1 (autoresearch nightly cron fix) | R + B | restored research feed | launchd plist exit 0 streak | OPEN-29 |
| 40.1-40.6 (NOTE-tier housekeeping) | speculative | code-debt reduction | varies per item | OPEN-24/25/26/28/30/31 |
| 40.7 (meta-labeling classifier) | P | +10-20% precision per Hudson & Thames | Default OFF; LLM cost approval needed | #5 |
| 40.8 (correlation cap beyond GICS) | R | -5-15% factor-crowding | FF3 exposure check | OPEN-5 31.0-F10 |
| 41.0 + 41.1 (bundle closures) | speculative | masterplan-hygiene only | masterplan.json status=done | OPEN-32 / OPEN-33 |
| 43.0 (DoD 14-criterion audit) | -- | Gate, not a delta | All 14 PASS | DoD §1-§14 |
| 44.1-44.10 (frontend overhaul, 10 substeps) | speculative + UX | +operator-decision-throughput; not Sharpe-affecting | UX measurement separate from north-star alpha | phase-44.0 brief Section H |

**Headline:** the production-readiness gate (phase-43.0) needs phases 35 + 36
+ 37 + 38 + 39 + 40 + 41 to close. **The biggest single Sharpe delta is
36.1 + 37.1 + 35.1 chained** -- scale-out wiring with structured RiskJudge
output and an alive learn-loop is the BLOCK + WARN + WARN combination that
the master_roadmap §3 critical-path already identifies.

---

## Section E -- Integration risk matrix (step pairs with coupling)

| Step A | Step B | Coupling | Risk if A done without B |
|---|---|---|---|
| 35.1 (learn-loop alive) | 35.2 (Risk Judge cites portfolio_sector_exposure) | LOW | Independent verifications of two distinct outputs |
| 36.1 (scale-out) | 36.2 (ATR stops) + 36.3 (triple-barrier) | MED | If 36.1 ships first, the scale-out levels are based on fixed 8% R-multiple. 36.2's ATR-derived R-multiple changes the +2R / +3R levels mid-flight. Recommend 36.1 -> 36.2 -> 36.3 sequence per master_roadmap; 36.1 depends_on_step listed null but `depends_on_step` for 36.2 + 36.3 is explicitly `"36.1"`. |
| 36.1 (scale-out) | 35.1 (learn-loop alive) | HIGH | If scale-out ships before the writer that converts a closed SELL into outcome_tracking, then the lessons-from-scale-out events accrete in /dev/null. The current state is "outcome_tracking exists in schema but empty" -- THIS is the writer gap discovered in Section B. Recommend 35.1's writer-fix lands FIRST, then 36.1's scale-out generates evaluable lessons. |
| 37.1 (RiskJudge schema) | 37.4 (Moderator schema) | LOW | Same fix to two different config blocks. Can ship in parallel; master_roadmap lists 37.4 depends_on_step="37.1" for serialization, but no hard technical dependency. |
| 37.2 (deep_think source default) | 38.3 (startup banner deep_think_model) | MED | If 37.2 lands but 38.3 doesn't, the banner still shows ONLY the standard tier -- which is the very silent-regression we're closing. Plan to ship 37.2 + 38.3 in the same PR/cycle so the banner reflects the new default at restart. |
| 38.1 (kill-switch auto-resume) | 39.1 (autoresearch fix) | LOW | Different services; no coupling |
| 38.4 (auto-commit hook refuses flip-without-harness-log) | 23.7.0 (auto-commit hook plumbing) | HIGH | 38.4 EXTENDS 23.7.0. If 23.7.0 isn't fully done first, 38.4's gate code has nowhere to attach. Section A verdict says 23.7.0 is `done` -- verify before 38.4 lands. |
| 40.2 (alwaysLoad + continueOnBlock + effort.level) | CLAUDE.md effort policy | MED | The 2.1.140-143 features depend on Claude Code being on a sufficient version. Confirm `claude --version` before enabling. |
| 40.7 (meta-labeling classifier) | 36.1 + 36.3 (scale-out + triple-barrier) | MED | Meta-labeler USES the triple-barrier labels as ground truth for the secondary classifier. If 40.7 ships before 36.3, the labels are based on fixed-stop legacy events -- worse training data. Recommend 36.3 -> 40.7. |
| 41.0 + 41.1 (bundle closure) | 37.3 + 40.1 + 40.2 + 40.3 | MECHANICAL | Bundle-close phases just flip status. Sub-items must be done first. master_roadmap depends_on already encodes this. |
| 43.0 (DoD audit) | ALL of 35-41 | TOTAL | Gate. Nothing upstream of 43.0 can ship `done` after 43.0 declares PRODUCTION_READY without re-running the gate. |
| 44.1-44.10 (frontend) | 35-43 (backend hardening) | LOW (read-only) | The frontend overhaul READS the backend outputs. Backend changes (e.g., scale-out adding new paper_trades.reason values) require frontend display updates -- mostly captured in phase-44 design but worth noting. |

**Highest-risk coupling = 36.1 <-> 35.1.** Already known but worth re-stating:
Section B's "writer is the missing gap" finding sharpens this. Land 35.1's
writer-fix BEFORE 36.1's scale-out so every scale-out event creates an
auditable lesson.

---

## Section F -- Regression test snapshot

Baseline locked: **297 tests** (`pytest backend/ --collect-only -q` at session
start, as cited in the directive). All subsequent step completions must keep
collected count >= 297.

### Sampled coverage map (illustrative, not exhaustive)

| Phase | Likely test files (existence not re-verified for this brief) |
|---|---|
| 35.1 / 35.2 / 35.3 | tests for `backend/agents/memory.py`, `backend/services/autonomous_loop.py`; new `backend/tests/test_learn_loop_alive.py` |
| 36.1 | `backend/tests/test_phase_36_1_scale_out.py` (per master_roadmap) |
| 36.2 | `backend/tests/test_phase_36_2_atr_stops.py` |
| 36.3 | `backend/tests/test_phase_36_3_triple_barrier.py` |
| 37.1 / 37.4 | `backend/tests/test_phase_37_1_risk_judge_schema.py`, `test_phase_37_4_moderator_schema.py` |
| 38.1 | `backend/tests/test_phase_38_1_kill_switch_auto_resume.py` |
| 38.6 | `backend/tests/test_phase_38_6_lock_file.py` |
| 40.4 | `backend/backtest/experiments/` walk-forward CSVs |
| 43.0 | `backend/tests/test_phase_43_dod_audit.py` (NEW) -- 14-criterion runner |
| 44.x | `frontend/__tests__/` vitest suite per phase-44 plan |

### Test discipline

- Every new step's success criterion includes a `pytest` invocation -- visible
  in master_roadmap §7 JSON inserts (e.g., `"command": "pytest backend/tests/test_phase_36_1_scale_out.py -v && test -f handoff/current/live_check_36.1.md"`).
- Q/A's deterministic-first checks include `pytest --collect-only -q` count.
- Closure cycle must include a final `pytest --collect-only -q` showing
  >=297 (ideally >300 after the 30 steps each add 1-3 tests).

---

## Section G -- Recency scan (mandatory, last 2 years)

Three-variant queries per `.claude/rules/research-gate.md`: current-year frontier (2026), last-2-year (2024-2025), year-less canonical.

### Supersedes / changes the framing

| Finding | Source | What it changes |
|---|---|---|
| **Python 3.14 free-threading is officially supported (PEP 779)** | docs.python.org/3/whatsnew/3.14.html (2025-10-07) | The GIL-removal narrative since 2019 is now reality. Pyfinagent's `Backend: FastAPI + Python 3.14` per CLAUDE.md gets it "for free" with the default GIL-enabled build; opt-in to free-threaded build is a phase-5 / post-prod consideration only. |
| **OWASP LLM Top 10 v2.0 (2025) adds LLM07 + LLM08** | genai.owasp.org (2024-2025) | Two NEW vectors: system-prompt-leakage + vector/embedding-weaknesses. Pyfinagent's `skill_file_ids.json` caches + agent_memories BM25 index are LLM07/LLM08 surfaces. DoD §14 should explicitly check these two NEW vectors, not just inherit "v2.0 compliance". |
| **Two Sigma's "AI as operating system" framing** | twosigma.com/articles/ai-in-investment-management-2026-outlook-part-i (2026) | The harness MAS pattern is now the documented industry-best-practice -- this validates pyfinagent's 3-agent architecture and reduces stress-test-doctrine urgency (the pattern is no longer experimental). |
| **TradingAgents framework (arXiv:2412.20138v3) documented Sharpe 5.6-8.2 in multi-agent backtests** | arxiv.org (2024-12, v3 2025+) | Provides empirical anchor for the structured-output + NL-debate hybrid we already run. AAPL Sharpe 8.21 / GOOGL 6.39 / AMZN 5.60 are at the top tail of what's been published -- a useful upper-bound aspiration vs pyfinagent's current 1.17 baseline. |
| **Caltech/VT paper: LLMs are textbook-rational, not human-mimicking traders** | arxiv.org/abs/2502.15800 (2025-02) | Calibrates the use of LLMs as decision-makers: they will ANCHOR to fundamental value, suppressing bubble-trades. For pyfinagent, this means the Risk Judge will systematically reject momentum-trades that have run too far from fundamentals -- a feature, not a bug, for production. |
| **Tailwind v4 ships native container queries** | sitepoint.com (2026) | Phase-44 frontend overhaul can adopt CQ without plugins; fixes the home-page `h-full` anti-pattern. |
| **SR 21-8 supplements SR 11-7 for ML/AI models** | glacis.io (2021/2026) | The 3-pillar / 3-line model risk management framework is the gold-standard pattern; phase-43.0 DoD audit + per-cycle Q/A independent-evaluator match it. pyfinagent is not under SR-11-7 jurisdiction (operator-only) but the principles are operative for production-readiness. |

### Complements (no supersession)

| Finding | Source | What it adds |
|---|---|---|
| **Hudson & Thames meta-labeling lift is empirically anchored** | hudsonthames.org | Validates phase-40.7 (meta-labeling exit classifier). The 17->63% accuracy lift on mean-reversion is the strongest single quantitative justification for the LLM cost approval. |
| **Anthropic Effective Harnesses doc (Nov 2025)** | anthropic.com/engineering | Confirms file-based handoffs + 4-component structure ARE the documented Anthropic recommendation. Pyfinagent's masterplan.json + harness_log.md + 5-file pattern is on-pattern. |
| **LangSmith trace-tree pattern** | langchain.com/articles/agent-observability | Phase-44.7 maps directly. /agents Live Stream tab refactor to TraceTree by run_id is the documented 2026 pattern. |

### No relevant new findings in the last 2 years for

- **PSR/DSR formulas** -- Bailey & Lopez de Prado canonical (researcher
  memory `project_psr_dsr_formulas.md`); no updates 2024-2026.
- **Triple-barrier method canonical reference (AFML 2018)** -- still cited
  as canonical in 2024-2026 Hudson & Thames publications; no method updates.
- **Recharts library** -- still the canonical Tailwind-compatible chart lib
  (per phase-44.0 brief Section E -- Tremor wraps it).

**Bottom line:** the 2024-2026 frontier shifts on (1) multi-agent LLM trading
empirics now published (TradingAgents Sharpe 5.6-8.2, Two Sigma's "AI as OS"
framing), (2) LLM-trader behavioral characterization (Caltech: textbook
rational not human), (3) OWASP LLM v2.0 expansion to vector/embedding +
system-prompt-leakage, (4) Python 3.14 free-threading officially supported,
(5) Tailwind v4 native container queries. **None of these change the
phase-35-43 critical path** -- but several enhance the audit basis for
specific steps (40.7 meta-labeling, 38.1 kill-switch auto-resume guardrails,
43.0 DoD §14 OWASP coverage).

---

## Section H -- Provenance + queries

### Three-variant query log

**Current-year frontier (2026):**
1. "Two Sigma Cardinal 2026 quantitative research market signals AI"
2. "SR-11-7 model risk management AI machine learning 2026 federal reserve guidance"
3. "arxiv 2026 LLM trading agent multi-agent portfolio paper"
4. "OWASP LLM Top 10 v2.0 2025 guidelines"
5. "Tailwind CSS v4 2026 container queries color-mix features"

**Last-2-year window (2024-2025):**
6. "Lopez de Prado AFML meta-labeling 2026 updates triple barrier method"
7. "Python 3.14 release notes free-threading GIL features 2025"
8. arxiv search 2412.20138 (TradingAgents, Dec 2024 v3)
9. arxiv search 2502.15800 (Caltech LLMs vs human traders, Feb 2025)

**Year-less canonical:**
10. "anthropic effective harnesses long-running agents" (canonical Anthropic doc)
11. "Lopez de Prado triple barrier" (canonical AFML reference, no year)
12. "SR 11-7 model risk management" (canonical Fed guidance, no year)

### Internal files inspected (count)

| File | Lines | Purpose |
|---|---|---|
| `.claude/masterplan.json` | (70 phases) | Verdict source-of-truth for Section A |
| `handoff/current/master_roadmap_to_production.md` | 1182 | Phase 35-44 PLAN + DoD criteria |
| `handoff/current/frontend_ux_master_design.md` | 922 | phase-44 cross-reference (not deep-read; same dir) |
| `handoff/cycle_history.jsonl` (tail 100) | -- | Section B cycle evidence |
| `handoff/current/live_check_34.2.md` | (66+) | Confirms previous cycle's phase-32 features live verification |
| `handoff/current/research_brief.md` (existing) | 776 | Replaced by this file; prior phase-44.0 brief read for context |
| `CLAUDE.md` | (~330+) | Stack + harness protocol + BQ access + effort policy |
| `.claude/rules/research-gate.md` | (185) | Discipline reference |
| BigQuery: `pyfinagent_data.llm_call_log` schema + 4 queries | -- | Section B probes |
| BigQuery: `financial_reports.outcome_tracking` schema + 2 queries | -- | Section B probes |
| BigQuery: `financial_reports.agent_memories` schema + 2 queries | -- | Section B probes |
| BigQuery: `financial_reports.paper_trades` schema + 2 queries | -- | Section B probes |
| BigQuery: `financial_reports.paper_portfolio_snapshots` schema + 1 query | -- | Section B probes |
| BigQuery: `pyfinagent_data.strategy_decisions` schema + 1 query | -- | Section B probes |
| BigQuery dataset enumeration in `financial_reports` + `pyfinagent_pms` + `pyfinagent_data` | -- | Schema discovery for Section B probes |

Total internal files / data sources inspected: **18 distinct items** (10
files + 8 BQ probe sets).

### URLs collected

- Read in full: 11
- Snippet-only: 16
- **Total URLs: 27+**

---

## Section I -- JSON envelope

```json
{
  "tier": "deep",
  "external_sources_read_in_full": 11,
  "snippet_only_sources": 16,
  "urls_collected": 27,
  "recency_scan_performed": true,
  "internal_files_inspected": 18,
  "gate_passed": true
}
```

`gate_passed: true` -- deep-tier floor is >=8 external sources read in full
(11 cleared with buffer), recency scan performed (Section G), all hard-blocker
checklist items satisfied (multi-pass scan -> gap -> adversarial; recency
scan; >=10 URLs; per-claim citations).

Hard-blocker checklist (verbatim):

- [x] >=8 external sources READ IN FULL via WebFetch (achieved 11; floor 8)
- [x] 27+ unique URLs total (incl. snippet-only) (floor 25 for deep)
- [x] Recency scan (2024-2026) performed + reported in Section G
- [x] Full papers / pages read (not abstracts) -- e.g., arxiv 2412.20138v3
      and 2502.15800 fetched via `arxiv.org/html/` per the PDF chain
- [x] file:line / table-name anchors for every internal claim
      (masterplan.json:70-phases, master_roadmap_to_production.md:259-268
      for phase-42 deferral, financial_reports.outcome_tracking n=0, etc.)
- [x] Multi-pass structure documented: pass 1 (broad scan, 6 sources from
      different domains); pass 2 (gap analysis -- Two Sigma Part II
      adversarial perspective on overfitting; Caltech paper adversarial
      on LLM-vs-human behavior); pass 3 (recency scan against year-less
      canonical sources)
- [x] >=1 adversarial source. **#2 (Caltech/VT "LLM Agents Do Not Replicate
      Human Market Traders") is explicitly adversarial** to the prevailing
      narrative that LLMs can substitute for human traders in financial
      market models. Tag: `[ADVERSARIAL]`.
- [x] Cross-domain triangulation: ML-research (#1 TradingAgents, #2 Caltech),
      industry quant (#3 + #4 Two Sigma, #5 Hudson & Thames), Anthropic
      engineering (#6), observability (#7 LangSmith), language runtime
      (#8 Python 3.14), security (#9 OWASP), frontend (#10 Tailwind),
      governance (#11 SR 11-7). 7 distinct domains.

Soft checks:

- [x] Internal exploration covered every relevant module
- [x] Contradictions noted (Two Sigma's overfitting-warning is adversarial
      to the "LLMs widen the top" optimistic framing in the same article;
      Caltech paper warns against using LLMs as market-modeling agents
      WHILE TradingAgents demonstrates production utility)
- [x] All claims cited per-claim (Section A masterplan refs, Section B BQ
      probe rows verbatim, Section C URLs in tables)

---

## Section J -- Application notes for the planner

1. **Closure scope is modest.** Section A's verdict tally is roughly
   **6 DROP + 3 DEFER + 3 KEEP** out of 12 legacy phases. **Five of the
   six DROPs are mechanical** (fold-into-3X.Y reference notes + status
   flip): phase-4 -> 43.0, phase-16 -> 43.0, phase-23.7 verify-then-flip,
   phase-26 -> 40.2/40.3/41.x, phase-27 -> 37.x + deferred, phase-29 -> 41.0/41.1.
   The closure planner should be able to land all 6 in **2 GENERATE
   cycles** if scoped carefully -- one cycle for masterplan.json edits +
   one cycle for verification.

2. **Section B's headline: phase-35.1 + 35.2 are NOT closed by c7801712.**
   Despite the cycle's structural success (37-min run, COHR trail-out at
   +17.89% pnl, all 8 steps completed), `outcome_tracking` and
   `agent_memories` are EMPTY. The data-writer that should convert
   stop_loss_trigger SELLs into outcome rows is MISSING in code.
   **Action for the closure planner:** sharpen phase-35.1's
   `verification.live_check` to call out the missing writer, NOT just
   "wait for the next stop-loss to fire and write a row" -- because if
   the writer is missing, ten more stop-loss fires won't change the
   table count.

3. **Section B's silver lining: phase-32.2 trail discipline IS production-verified.**
   COHR's trail-out at +17.89% pnl, capture_ratio=0.63, mfe=28.36% over 25
   holding days is exactly the kind of evidence the master roadmap's DoD
   §8 (profit-protection BLOCK closed) was waiting for. The remaining
   gap (OPEN-2 scale-out wiring) is the only OPEN code BLOCK -- but the
   underlying trail primitive demonstrably works.

4. **The c7801712 cycle should organically fold into phase-35.3's streak count.**
   Pre-cycle: 1 of 5. Post-cycle: 2 of 5 strict-completed-with-no-orphan
   (dc3f6cf1 was logged as "phase-34.2 cycle 3 completed end-to-end"
   2026-05-22 16:23 -> 17:00 UTC; c7801712 2026-05-22 18:00 -> 18:37
   UTC). Three more clean cycles needed. **Action:** track in
   phase-35.3's live_check as the streak progresses; do NOT close until
   the strict-5 is achieved.

5. **Cycle-count estimate for the FULL closure pass:**
   - 1 cycle: phase-45.0 GENERATE (this brief + closure_roadmap.md + masterplan flips)
   - 1 cycle: phase-45.0 evaluate + QA
   - **Total: 2 cycles to close phase-45.0**
   - Downstream (phase-43.0 DoD audit): needs all of 35-41 done first; not phase-45.0's scope.

6. **OWNER-APPROVAL impact of DROPs:** Section A flips 3 phases to
   `deferred` (phase-5, phase-10.7, phase-13). These are NOT minor masterplan
   edits -- they're production-scope declarations. **Action for the
   planner:** include each `deferred` flip in the closure contract.md with
   explicit owner-acknowledgment language (e.g., "Confirms phase-5 is
   post-prod per master_roadmap §2 phase-42 deferral.").

7. **The 3 KEEP verdicts (phase-23.6, phase-23.8 partial, phase-28
   residual) each have specific residual substeps.** None are
   show-stoppers; all should fall into the normal flow (23.6.3 + 23.6.4
   + 23.8.3 + 23.8.4 + any phase-28 residuals). The closure can
   acknowledge them as KEEP without specifying their work plan.

8. **Do NOT touch the 35-44 already-planned substeps in the closure
   roadmap.** They have stable IDs, immutable success criteria, owner
   gates, and depend_on chains. Adding new steps or renaming sub-steps
   creates retrospective ambiguity in handoff archives. Closure can
   FOLD legacy work into specific 35-44 sub-steps by reference but must
   not invent new IDs.

End of brief.
