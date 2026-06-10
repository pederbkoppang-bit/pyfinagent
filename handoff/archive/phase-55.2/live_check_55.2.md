# live_check_55.2 — Ops + skills audit: live-system evidence

**Step:** 55.2. **Date:** 2026-06-10. **Required shape (masterplan):** the ops+skills section of the post-mortem: llm_call_log query results (fire-count table), incident evidence excerpts, and the signal-stability table.

Audit: `handoff/current/55.2-ops-skill-audit.md`. Raw dumps: `/tmp/55_1/{analysis_scores,spot_signals,llm_call_log_away}.json`.

## A. llm_call_log fire-count tables (verbatim query results)

Away window — `SELECT agent, provider, model, COUNT(*) n, COUNTIF(NOT ok) fails, ROUND(SUM(session_cost_usd),4) cost, MIN(ts), MAX(ts) FROM pyfinagent_data.llm_call_log WHERE ts >= '2026-06-01' AND ts < '2026-06-10' GROUP BY 1,2,3`:

| agent | provider | model | n | fails | cost | first_ts | last_ts |
|---|---|---|---|---|---|---|---|
| NULL | gemini | gemini-2.0-flash | 8 | 0 | 0.40 | 2026-06-01 18:03:39Z | 2026-06-01 19:11:06Z |
| NULL | anthropic | claude-haiku-4-5 | 4 | 0 | 0.00 | 2026-06-01 15:31:19Z | 2026-06-01 15:46:14Z |

**12 rows for the whole away week; ZERO rows for the 06-02..06-09 cycles** (which `cycle_history.jsonl` + 59 `analysis_results` rows prove ran). The 4 direct-Anthropic Haiku successes at 15:31-15:46Z on 06-01 — minutes before the operator's failed 15:47 Approve attempts (kill_switch_audit.jsonl manual pause/resume pairs) — prove the env API key was valid while the CLI OAuth rail was not.

Full-history agent-label taxonomy — `SELECT agent, provider, COUNT(*) FROM llm_call_log GROUP BY 1,2`:

| agent | provider | n |
|---|---|---|
| NULL | gemini | 197 |
| NULL | anthropic | 184 |
| Quant Model_code_exec | gemini | 1 |
| phase26.1-smoke | anthropic | 1 |
| Synthesis_advisor_tool | anthropic | 1 |
| Enhanced Macro_combined_tools | gemini | 1 |
| Synthesis | anthropic | 1 |

(99.5% NULL-agent; labels are skill/tool tags when set; no "Bull R1/2" debate roles exist. Schema check: `cycle_id` and `ticker` columns EXIST — NULL-valued.)

## B. Incident evidence excerpts

**F-A1 — error-string provenance:** `grep -rn "Missing API key for provider" backend/ .venv/` → 0 hits in repo code; nearest venv hit is litellm "Missing API key for Volcengine" (different provider, different phrasing). Env-scrub site verbatim (`backend/agents/claude_code_client.py:163-170`): the subprocess env drops `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN` (phase-38.13.1 comment: force `~/.claude/` OAuth, never bill the metered key). Error wrap at `:188-197` (`ClaudeCodeError` carries CLI stderr verbatim).

**F-A1 — live CLI auth state (post-return; no secret values):**
```
claude auth status -> loggedIn: true, authMethod: claude.ai, apiProvider: firstParty, email: peder.bkoppang@hotmail.no
```

**F-A2 — dead button:** `governance.py:166-175` renders `action_id: approval_approve / approval_deny`; `grep -rn "@app.action" backend/slack_bot/` registrations contain only `app_home_*`, `agent_model_change_*`, `agent_feedback_*` — no approval handler.

**F-C — watchdog excerpts (`handoff/logs/backend-watchdog.log`, UTC):**
```
2026-05-27T17:02:22Z health FAIL (1/3)   2026-05-27T18:05:48Z health FAIL (1/3)
2026-05-27T18:16:46Z health FAIL (1/3)   2026-05-28T18:04:54Z health FAIL (1/3)
2026-06-01T18:36:02Z health FAIL (1/3)   2026-06-02T18:33:21Z health FAIL (1/3)
2026-06-05T18:13:34Z health FAIL (1/3)
```
All "(1/3)" — kickstart threshold (3) never reached. Cycle windows from `cycle_history.jsonl`: `af5a8000` 2026-06-04T18:00:00→19:02 (62 min), `035dbb69` 2026-06-05T18:00:00→19:05 (65 min); short cycles 06-08/09 (~13 min: `25f2fb19`, `0361d1ea`). 18:05-18:50Z ≡ the operator's "20:05-20:50 CEST".

**F-D — the 0.0 block (`analysis_results`, 2026-05-27):**
```
DELL 2026-05-27 06:12 score=7.0  rec=Hold   cost=0.10   <- real (lowercase, fractional path)
DELL 2026-05-27 18:15 score=0.0  rec=HOLD   cost=0.10   <- degraded (uppercase)
MU   2026-05-27 05:54 score=7.35 rec=Buy    cost=0.10
MU   2026-05-27 18:12 score=0.0  rec=HOLD
also GEV/GLW/HPE/INTC/KEYS/ON/SNDK/STX/WDC @ 18:02-18:20Z all 0.0/HOLD
```
Publisher site: `formatters.py:37` `score = report.get("final_weighted_score", 0)` → "0.0/10". The 05-28 morning digest carried these 05-27-evening rows.

**F-F — REJECT executed (BQ `paper_trades`):**
```
2026-06-03T19:05 BUY DELL qty=0.5816 px=424.15 risk_judge_decision=REJECT  -> executed; sold 06-04 (+0.22%)
```
Non-enforcement site: `portfolio_manager.py:185` (decision recorded), `:194-198` (log-only branch); no code path drops a REJECT candidate.

**F-G — prompt/config divergence:** RiskJudge rationales (signals JSON) cite "10% single-sector cap"; config `paper_max_per_sector_nav_pct=30.0` (settings.py:237-241). Rationales also state "no current portfolio sector breakdown was provided".

**F-H — lite checkbox desync:** Playwright capture `handoff/current/captures_55.1/55_1_manage_markets_toggle.png` (Lite mode UNCHECKED) vs §C lite-path signatures on all away-week cycles.

## C. Skill-firing + spot-check evidence (paper_trades.signals, verbatim excerpts)

SignalStack fallback on EVERY away-week BUY (all 10 BUY rows 06-01..06-09):
```json
{"agent": "SignalStack", "role": "overlay", "rationale": "conviction 10.00; fallback (LLM unavailable)", "weight": 1.0}
```
Fallback producer: `meta_scorer.py:254` (`_fallback_all` → "fallback (LLM unavailable)").

MU BUY 06-08 (whipsaw spot-check 1): Quant `1m +47.6%; 3m +145.2%; 6m +286.8%; RSI14 68.5; composite 102.276`; Trader "…+124% 60-day run and 45x P/E warrant a tight trailing stop against mean-reversion risk"; RiskJudge "VOLATILITY (HIGH/extreme)… parabolic at ~8x the limit… CONCENTRATION (MODERATE): no portfolio sector weights were supplied…" → APPROVE_REDUCED. Exit 06-09: −6.27%.

000660.KS BUY 06-04 (spot-check 2): RiskJudge "…VALUATION (UNRELIABLE): P/E of 0.0 and a nonsensical $1.63-quadrillion market cap are corrupted…" → APPROVE_REDUCED. Stop-out 06-05: −9.92%.

DELL BUY 06-03 (spot-check 3): RiskJudge 3-axis EXTREME/Elevated/High → decision **REJECT** → trade executed anyway (F-F).

Enrichment-skill silence (59 away-week `analysis_results` rows): `earnings_confidence=NULL`, `nlp_sentiment_score=NULL`, `deep_think_calls=NULL`, `insider_signal=''`, `patent_signal=''`, `pillar_4_sentiment=0.0`, `social_sentiment_score=NULL`, `debate_rounds_count=NULL`, `devils_advocate_challenges=NULL` on every row; `full_report_json` keys = {analysis, market_data, source}.

## D. Signal-stability table

Embedded in `55.2-ops-skill-audit.md` §4 (per-ticker flips + |Δscore| + paths; TOTAL 16 flips / 46 pairs = 35%, mean |Δscore| 1.15; SNDK 06-05 5.0-HOLD → 06-08 7.0-BUY reproduced with px 1,553.80 → 1,632.50).

## E. Burn vs P&L

`llm_call_log` away-week metered: **$0.40**; `analysis_results` away-week self-reported: `{n: 59, cost_sum: 0.59}`; `/performance total_analysis_cost: 5.05` (36 days, inception-to-date). Against realized: churn RTs −$132.00, week NAV −$551.55 (06-01→06-09). Burn ≈ 0.8% of the churn loss.

## F. Tool reruns (criterion 4)

```
$ python scripts/risk/tca_report.py
{"wrote": ".../handoff/tca_last_week.json", "rows": 70, "median_bps_liquid": 5.9964, "alert_triggered": false}   [SYNTHETIC seeder — tool-smoke only]
$ python scripts/harness/paper_execution_parity.py
alpaca.common.exceptions.APIError: {"code":40010001,"message":"client_order_id must be unique"}   [FAILED — honest report; B13]
```

## G. Constraint compliance

NO fixes (no source modified); $0 (BQ reads bounded with date filters/LIMIT; log/code reads; no LLM trading-cycle spend); no secret values printed.
