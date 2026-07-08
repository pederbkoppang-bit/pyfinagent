# live_check 66.2 -- Redeploy capital via the NORMAL path (prep phase, 2026-07-07)

Required shape: "live_check_66.2.md with the BUY BQ row or the 5-day funnel
diagnosis, plus both integrity-check verdicts."

## 1. Criterion 1 -- clock RUNNING (66.1 closed 2026-07-07 ~18:30 UTC)

### Day 1 of <=5: 2026-07-07 cycle 0725d2aa -- ZERO BUYs, cause DIAGNOSED (not gates)

Funnel row (the new persisted funnel field's first cycle -- working):
```
| 2026-07-07 | 0725d2aa | 583/577/10/5 | rail 58/65 | rail_skip False | breaker False | 5 analyses (5 degraded) | HOLD:5 | trades - | ALL-HOLD COLLAPSE (pipeline defect: rail down; gates never evaluated) |
```
ROOT CAUSE (evidence-confirmed): **Max-plan session-limit exhaustion on the shared
credential** -- 65 of 123 rail calls failed (0 tokens, empty stderr, bursts; max
streak 12 < breaker threshold 20, correctly no trip), concentrated at the
decision phase (18:42-18:44Z) after live debates had produced consensus=BUY 0.62;
the 20:00 UTC away PM session then died with the smoking gun: "You've hit your
session limit - resets 1am (Europe/Oslo)". The heavy return-day dev workload
(this session + subagents + drills) shares the Max quota with the trading rail.
07-08 morning: probe healthy, AM session rc0 -- quota reset confirmed.

OPERATIONAL RULE derived (register + operator advisory): keep dev-session usage
light in the 17:00-20:00 UTC window on trading days, or the rail starves.
Follow-up fix shipped 07-08: rail failures now log the CLI's STDOUT snippet
(the limit message lives there, not stderr) so this diagnosis takes minutes,
not hours, next time.

First post-deploy scheduled cycle: 2026-07-07 18:00 UTC (backend restarted
16:31:55 UTC holding all phase-66 fixes). Evidence lands here when it exists:
either the first ordinary-pipeline BUY row, or per-cycle funnel reports
accumulating toward the >=5-healthy-rail-day diagnosis.

Funnel tooling ACCEPTANCE (scripts/diagnostics/funnel_report.py, read-only, over
the collapse window):

```
| day        | cycle    | rail ok/fail | analyses (deg) | rec mix            | non-HOLD | trades (by reason)                              | verdict |
| 2026-06-09 | 0361d1ea | 0/0          | 5 (0)  | BUY:4 HOLD:1       | 4 | BUY:swap_buy=2 SELL:swap_for_higher_conviction=2 | GATES EVALUATED, trades executed |
| 2026-06-10 | d2a9e92b | 0/0          | 5 (0)  | BUY:4 HOLD:1       | 4 | BUY:new_buy_signal=1 BUY:swap_buy=1 SELL:stop_loss_trigger=1 SELL:swap_for_higher_conviction=1 | GATES EVALUATED, trades executed |
| 2026-06-11 | 78d253f5 | 3/2          | 8 (7)  | BUY:1 HOLD:5 N/A:2 | 3 | - | GATES EVALUATED, zero BUYs (check decide_trades per-gate logs) |
| 2026-06-12 | 5f15fdbe | 36/45        | 5 (3)  | BUY:2 HOLD:3       | 2 | - | GATES EVALUATED, zero BUYs (check decide_trades per-gate logs) |
```

The tool exposes exactly the criterion-b structure AND the known gap (universe/
screener/decide_trades counters are log-only -- reported explicitly, never
silently omitted). Refinement over the brief: on 06-11/06-12 a few BUY recs
SURVIVED the degraded scorer and were rejected inside decide_trades -- the
log-only stage; post-66.1 cycles will carry rail_skipped/breaker_tripped columns.

## 2. Criterion 3 -- CLOSED: Alpaca short exposure EXPLAINED with evidence

Read-only inspection (alpaca-py, TradingClient paper=True; NO orders placed),
account PA3VQZZLAKE2 ACTIVE, 2026-07-07 ~16:20 UTC:

```
equity=99273.30 cash=102778.62 long_mv=10909.12 short_mv=-14414.44
positions: 20 (10 long, 10 short); sum(short mv) = -14414.46 == short_market_value
shorts: ADBE AMD AVGO CSCO GOOGL IBM META MSFT PYPL UBER (qty -4..-5 each)
closed orders 2026-06-10 13:51-13:52Z: alternating 1-share buy/sell pairs with
client_order_id prefix d4-<SYM>-<n> (drill batch) + probe-alp-1
```

ROOT CAUSE: pre-departure MCP-validation drill orders (2026-06-10) -- 1-share
SELLs on symbols the account did not hold, which a margin-default Alpaca paper
account fills as SHORT OPENS. The autonomous loop CANNOT have caused it: its
launchd env has no EXECUTION_BACKEND/ALPACA keys -> ExecutionRouter defaults
bq_sim (execution_router.py:65-71; pre-prod audit 2026-05-16:161 confirms); BQ is
the authoritative ledger and the Alpaca account is a disconnected mirror.
(The -13,842.89 in the Cycle-58 finding vs -14,414.44 today = mark-to-market
drift on the same 10 shorts.) HYGIENE (operator, optional): reset the paper
account in the Alpaca dashboard; no BQ correction needed; filed for 63.3.

## 3. Criterion 4 -- CLOSED: single portfolio row is BY DESIGN; KR conversion intact

Design citation: paper_portfolio is one aggregate USD row keyed
portfolio_id='default' (bigquery_client.py:521-534, upsert :550-571); the 50.2
multi-market design puts `market`/`base_currency` on POSITIONS and converts FX
at trade/mark time (paper_trader.py:312-313/:333-334; _fx_local_to_usd :515-523;
sell credit :416). EU/KR exposure surfaces in paper_positions.market +
paper_trades, not extra portfolio rows.

USD-magnitude check on ALL 10 KR trade rows (BQ verbatim): prices are local KRW
by design (e.g. 000660.KS SELL @ 2,425,000) while total_value is correctly
USD-converted (that row: $560.80, NOT 857,843) -- the 56.1 conversion has NOT
regressed. paper_positions GROUP BY market/currency: empty (100% cash),
consistent. VERDICT: intended design; no defect. (Cosmetic: execute_sell log
line prints KRW with a '$' -- register note.)

## 4. Bonus fix shipped during prep (alert integrity)

Phantom "-61.51% drawdown" P1 (fired 2026-07-06 20:05Z on a book UP 20%):
DESC-order trap in compute_drawdown_from_snapshots (navs[-1]=OLDEST row as
"current"). Fixed (date-key ordering; refuses to guess when unknowable),
5 tests, live verification: real drawdown -2.76% (below the -3% tier,
correctly silent). Deployed in the 16:31 UTC restart.

## 5. Day-2 pre-cycle evidence sweep (2026-07-08 ~07:30 UTC)

Provenance: ultracode Workflow wf_5ec0a566-4e2 -- 5 read-only finders + 30
adversarial per-claim verifiers (35 agents, 448 tool calls, 0 errors; 27
CONFIRMED, 3 REFUTED-and-corrected). Every claim below survived independent
reproduction. Full dossier: workflow journal + scratchpad sweep_full.txt.
READ-ONLY sweep: zero repo/table writes; criterion 2 untouched.

### 5a. GO/NO-GO tonight: rail GO, BUY path structurally open

- stdout-logging fix LOADED: PID 3937 started 09:14:26 CEST > commit 399fdad4
  09:14:24; code at claude_code_client.py:333-342; no uncommitted diff.
- probe ok=True (07:31 UTC), route=True, breaker=20 armed, per-cycle
  rail_guard_reset at autonomous_loop.py:221. 0 cc_rail rows today (expected;
  single combined US+EU+KR cycle, APScheduler 14:00 ET = 18:00 UTC, no earlier
  EU/KR cycles -- paper_trading.py:1299-1322).
- decide_trades admission = rec in {BUY, STRONG_BUY} ONLY (portfolio_manager.py
  :50,:161); NO confidence/score floor anywhere; ~$22.8k spendable after 5%
  reserve; sector/count/NAV caps at zero utilization; RiskJudge REJECT gate
  default OFF (settings.py:283); kill-switch paused=false (daily 0.0%/4%,
  trailing 0.53%/10%). Sizing: ~$720 at APPROVE_REDUCED 3%, ~$2,400 at 10%.
- Criterion-2 audit CLEAN over c1e6050b..HEAD: only 3 backend commits
  (drawdown-alarm fix, funnel persistence, stdout logging); zero touches of
  thresholds/caps/limits/entry/sizing (aggregate diff grep = 0 hits).

### 5b. HEADLINE: yesterday's BUYs EXISTED and died at synthesis, not gates

Debate consensus BUY/0.62 for 000660.KS (18:17:49Z) and SNDK (18:20:26Z); all
5 analyses (000660.KS SNDK 009150.KS DELL MU) ran the FULL path (zero lite
rows) and persisted HOLD/0.0 with final_synthesis.error='Failed to parse final
report.' -- the orchestrator fallback (orchestrator.py:2172) converted rail
starvation into synthetic HOLDs. Root enabler (61.2-class, live-confirmed at
5/5 scale): ClaudeCodeError -> EMPTY LLMResponse (claude_code_client.py:555-570)
which _generate_with_retry never retries (orchestrator.py:797-862 retries
exceptions only). The deep-path Risk Judge also starved (DELL/MU 18:42-18:44Z)
and fail-opened to APPROVE_REDUCED/3%. A 0.62-confidence BUY would have cleared
decide_trades -- there is no threshold. This is the primary pipeline-defect
(vs-gates) confound for criterion 1(b); fix is masterplan 61.2 (pending,
post-66.2 per goal sequencing).

### 5c. CORRECTION: the 5-day healthy-rail clock is at DAY 0, not day 1

Section-1 above headered 07-07 as "Day 1 of <=5". The sweep found NO codified
healthy-rail-day definition anywhere, and the three existing signals CONFLICT
on 07-07: rail_skipped/breaker_tripped both false (pass; max fail streak 12 <
20), but funnel_report.py's own verdict = "ALL-HOLD COLLAPSE (pipeline defect:
rail down)" and research_brief_66.2.md:102's falsifier (ok-rate >90%) fails at
47.2% (58/123). Honest ruling: 07-07 does NOT count as a healthy-rail day.
PROPOSED definition for Q/A + operator ratification: healthy-rail day :=
rail_skipped=false AND breaker_tripped=false AND cycle cc_rail ok-rate >= 90%.
Clock stands at DAY 0 pending tonight's cycle.

### 5d. NEW live degradation: direct-API Anthropic credits dead since ~07-03

CORRECTION (2026-07-08 ~09:35 UTC, BQ-verified): the death is FAR older than
~07-03. Last GENUINE direct-API success = 2026-05-17 (51-52 real calls
05-16/17 with real token counts). Every "ok" non-rail anthropic row since
06-01 (30 rows) is a TEST FIXTURE (identical 1000/50/123.4 signature; writer:
backend/tests/test_observability.py:230 -- tests pollute the prod table,
register item). Direct-API failures are never written to llm_call_log at all,
and the conviction-overlay alert site was one of the four dead alerting.py
imports until 66.1 (07-07) -- three independent blinders explain ~7 weeks of
invisible fallback. The 61.2 criterion-4 "06-03..06-10 unavailability" window
is subsumed: it is one continuous credit-death span since ~2026-05-18.

'credit balance is too low' on the DIRECT Anthropic SDK (not the rail, not the
Max plan): meta_scorer (meta_scorer.py:168-190, fires once per cycle at Step 1)
falls back to raw composite ranking every cycle; compute_macro_regime's LLM leg
fails the same way -> regime=unknown conviction=0.00 mult=0.85 applied
uniformly (rank-preserving, macro_regime.py:526-542; FRED HTTP itself healthy,
9x HTTP 200 on 07-07). Neither forces HOLD nor blocks BUY, but candidate
ranking + regime context run degraded EVERY cycle regardless of rail health.
OPERATOR DECISION REQUIRED (metered spend): top up API credits, or accept/repin.

### 5e. historical_macro staleness verdict: (c) backtest/reporting-only

- The live funnel NEVER reads historical_macro; sole service-layer consumer is
  cycle_health monitoring -- i.e. the staleness alert itself. Live macro flows
  via compute_macro_regime -> direct FRED HTTP. NOT a BUY suppressor.
- Root cause: the table NEVER had a scheduled writer. Only writer =
  DataIngestionService.ingest_macro via run-once migration
  extend_historical_data.py (end_date hardcoded '2025-12-31'); exactly two
  write days ever (2026-03-22, 2026-03-25). max data date 2025-12-31 (189d),
  max ingested_at 2026-03-25 (105d), 4412 rows / 7 series.
- weekly_fred_refresh is triple-dead-but-green: no FRED_API_KEY in slack-bot
  env, writes nonexistent pyfinagent_data.fred_observations, zero readers,
  reports status=ok written=0. (Backend's own FRED key WORKS -- rotation ask
  is hygiene, not the root cause.)
- REAL impact: backtests/optimizer -- preload_macro refuses stale cache
  (189d>35d, cache.py:243-251) -> slow per-cutoff path with NO staleness guard
  (cache.py:399-421) silently serving 2025-12-31 macro features for 2026
  cutoffs. DO NOT run the optimizer until macro ingestion is repaired
  (register: promoted params would inherit stale-macro bias).

### 5f. Register additions (file into 63.3's defect register)

1. Probe blind spot: auth-status probe cannot detect Max session-limit
   exhaustion (07-07's actual failure mode) -- probe passes while quota is dead.
2. Breaker rate gap: 65 interleaved failures never trip the consecutive-20
   breaker (max streak 12); needs a per-cycle failure-RATE alarm.
3. 61.2-class synthetic-HOLD persistence + empty-response-no-retry (see 5b).
4. Fail-open risk gate: unparseable Risk Judge output yields APPROVE_REDUCED/3%
   -- unsafe if paper_risk_judge_reject_binding ever flips ON.
5. Debate evidence loss: debate_consensus/debate_confidence columns empty when
   synthesis fails (BUY/0.62 recoverable only from full_report_json).
6. cc_rail llm_call_log rows carry ticker=NULL on the full path -- per-ticker
   starvation attribution requires backend.log parsing.
7. weekly_fred_refresh triple-dead-but-green (5e).
8. sortino.py:101-121 tier-1 MAR: wrong dataset + never-present series ->
   permanent 404, fail-open to tier 2/3.
9. mcp data_server.py:142,:172 hardcode cutoff '2025-12-31'.
10. cycle_health P1 staleness alert fires for a non-live-path table
    (alert fatigue; pairs with the 07-07 hotfix's repeat-window).
11. No codified healthy-rail-day definition (5c) -- ratification pending.
12. Degraded-scoring guard is observability-only AND decide_trades has no
    floor: a PARSED BUY with confidence 0 would trade (sorts last, tradeable).

### 5g. Operator asks (pre-18:00 UTC today)

1. DONE 2026-07-08 ~08:45 UTC -- `claude setup-token` run + wired into 4
   plists (backend/away-am/away-pm/away-watchdog), jobs reloaded, backend
   healthy (PID 24910, /docs 200); validated end-to-end: auth status under
   the wired env = loggedIn=true authMethod=oauth_token. 1-year credential,
   renewal ~2027-07. NOTE: same Max quota pool -- ask 4 unchanged.
2. Anthropic direct-API credit decision (5d) -- metered, needs your approval.
3. Ratify (or amend) the healthy-rail-day definition in 5c.
4. Keep dev-session Claude usage light 16:30-20:30 UTC (quota guard).

## 6. Money-engine audit findings bearing on tonight's evidence (2026-07-08 ~10:00 UTC)

Provenance: ultracode Workflow wf_e26ca01b-6c6 -- 5 read-only auditors + 20
adversarial verifiers (25 agents, 0 errors; 18 CONFIRMED, 2 REFUTED-corrected).
Full dossier: handoff/current/money_engine_audit_2026-07-08.md. READ-ONLY.

### 6a. P0 -- full-path RiskJudge consumption is broken (affects criterion 1a TONIGHT)

VERIFIED by direct invocation: risk_debate.py:306-313 nests the judge verdict
under risk_assessment['judge'], but portfolio_manager._extract_position_pct
(:655-668) and the REJECT gate (:194) read TOP-LEVEL only (api/analysis.py:158
+ tasks/analysis.py:162 read 'judge' correctly -- the consumer drifted).
Consequences on the CURRENT book if tonight's cycle emits a full-path BUY:
- sized at the hardcoded 10% default ($2,399.77), NOT the judge's pct;
- a judge REJECT/0% still buys full size (binding flag irrelevant -- '' != 'REJECT');
- paper_trades.risk_judge_decision persists '' -- the criterion-1(a) text
  requires "risk_judge_decision recorded", so a ''-decision BUY tonight is
  NOT clean 1(a) evidence; it is itself pipeline-defect evidence for the (b)
  arm. Detection rule: alert on ''-decision BUY rows (lite-era BUYs 06-01..10
  all carry decisions, $238-736 sizes = correct 1-3% consumption).
OPERATOR DECISION REQUIRED: (i) accept tonight as-is (any BUY = 10%-sized,
''-decision; documented, capped by sector/NAV caps at $2.4k, stops apply), or
(ii) authorize a same-day hotfix (one-line judge-aware fallback + zero-falsy
fix) + pre-16:30 restart -- a trading-behavior change on cycle day, hence not
done unilaterally. Related verified: fail-open judge defaults trade
APPROVE_REDUCED/3% with no alarm; `if pct:` zero-falsy inverts judge 0% into
the 10%/3% defaults.

### 6b. Already-fired money defect: KR realized P&L overstated

paper_trader.py:443 revalues the ENTIRE price move at exit-date FX, dropping
the FX leg on entry notional: real 07-03 000660.KS stop sell recorded
realized_pnl_usd 87.18 vs true 84.20 (+3.5%). The correct decomposition
fx_pnl_attribution() exists in-file with ZERO production callers. Fold into
61.3 (money-display/currency step) -- register.

### 6c. Latent money-risk registers (verified; for 61.3/61.5/63.3)

- Add-on-buy USD-into-LOCAL averaging (61.3's headline) CONFIRMED latent with
  precise triggers; post-corruption effect = stop-ratchet FREEZE (:1128).
- upsert_paper_portfolio DELETE-then-INSERT: crash window silently re-seeds
  cash to $20,000 (vs $23,997.71).
- execute_sell: no positive-price guard; price=None -> hash-synthetic $50-550
  router fill credited as real cash.
- Fee truth for 61.5: 0.1%/side IS charged (45/45 live trades) -- the gap is
  slippage=0, per-market fees, and LLM cost never debited from NAV.
- Alpha axis: dividend-adjusted SPY vs dividend-less book (~1.2%/yr
  understatement) + EU/KR FX beta misattributed as alpha.
- Dual divergent Sortino implementations both claiming canonical.

### 6d. Prod-table pollution (corrects section 5d's count)

Fixture contamination is 106 rows since 2026-05-19 (not 30 -- the 30 was the
June-July window only; writer test_observability.py:226-235, zero conftest
isolation, 60s auto-flush). ALSO: pyfinagent_data.api_call_log does not exist
-- ALL production API telemetry silently dropped today; sla_alerts is 100%
drill output. Marker convention + conftest no-op flush + operator-approved
cleanup DELETEs drafted in the dossier (BQ deletes need explicit approval).

## 7. Day-2 SCHEDULED-CYCLE evidence -- 2026-07-08 cycle 9a8720b3 (completed 19:21:33 UTC)

Scheduled-run evidence (39.1 doctrine): the persistent cycle monitor caught
9a8720b3 start 18:00:00.3Z -> complete 19:21:33.6Z. NO manual rerun.

### Funnel row 2 (canonical, funnel_report.py)
```
| 2026-07-08 | 9a8720b3 | 583/577/10/5 | rail 87/46 | rail_skip False | breaker False | 5 analyses (4 deg) | HOLD:5 | non-HOLD 0 | trades - | ALL-HOLD COLLAPSE (pipeline defect) |
```

### Result: ZERO BUYs (n_trades=0); NOT a healthy-rail day
- cc_rail ok-rate **65.4% (87 ok / 46 fail of 133)** -- BETTER than day-1's 47.2%
  but BELOW the proposed >=90% healthy-rail threshold (healthy_rail_day_
  definition_PROPOSAL.md). By the proposed rule (and funnel_report's own
  ALL-HOLD-COLLAPSE verdict), 07-08 does NOT count. **Clock stays DAY 0.**
- rail_skipped False, breaker_tripped False (max streak < 20; success-reset held).
- 0 paper_trades today (BQ verified); kill-switch not involved.

### Cause: the 61.2 defect fired AGAIN, live, 2nd consecutive cycle
BQ analysis_results for 9a8720b3 (verbatim):
- DELL / MU / SNDK: `_path=full`, `final_synthesis.error='Failed to parse final
  report.'` -> synthetic **HOLD/0.0** (the exact 61.2 fabrication).
- **MU had debate consensus=BUY** destroyed at synthesis; **009150.KS** full-path
  debate=BUY but persisted "Hold"/5.0 (falsy-recommendation -> the _persist
  `or "Hold"` coercion, 61.2's second fabrication site).
- 000660.KS: fell back to `_path=lite` (rail failure on its full path) -> HOLD.
- meta_scorer_degraded=true again (direct-API credits dead; operator chose
  leave-degraded 2026-07-08).
So >=2 tickers carried a BUY-side debate signal that the synthesis/persist
fabrication converted to HOLD. This is criterion-1(b) "pipeline-defect (NOT
gates-correctly-reject)" evidence, live-reproduced at scale twice.

### NEW diagnostic finding (rail failure class differs from day 1)
The 46 failures (18:44 UTC decision-phase burst) log **empty stdout AND empty
stderr** under the 66.1 stdout-capture logging -- so they are NOT the
"session-limit message on stdout" class that explained day 1. Empty-both exit=1
is consistent with subprocess TIMEOUT (the 120s cap 61.2 raises to 150s) or
concurrency, NOT quota. The setup-token wiring (08:44 UTC) removed the shared-
credential quota risk (ok-rate improved 47->65%), exposing a SECOND failure
class underneath. This directly motivates deploying 61.2 (150s timeout +
retry-on-empty). Register: instrument the empty-both exit-1 class (add rc/
timeout discriminator to the 66.1 logging) -- follow-on.

### Disposition
NO status flip. 66.2 stays pending. The 61.2 build (Cycle 74, dark) is the
targeted fix; tonight re-confirms its necessity. Next: deploy 61.2 ungated
fixes at the post-cycle restart; the integrity flag (retry-on-empty + synthesis
routing) is the operator promotion that would have saved MU/009150.KS tonight.
