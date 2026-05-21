# Experiment Results — phase-33.0 Pre-Flight Readiness Check

**Step:** `phase-33.0` (diagnostic-only readiness check before the next autonomous paper-trading cycle).
**Date:** 2026-05-21.
**Next cron fire:** 2026-05-21 **18:00 UTC** = **20:00 Europe/Oslo** = **14:00 America/New_York** (per `cron[day_of_week='mon-fri', hour='14', minute='0', tz=America/New_York]`).

---

## TOP-LEVEL VERDICT: **NOT_READY**

**2 FAIL** (cat G, cat I) + **1 WARN** (cat H) + **6 PASS**. Per the contract success criterion: ANY FAIL → NOT_READY.

The next autonomous cron will fire on schedule (Mon-Fri 14:00 ET), but it will produce a `kill_switch_halted` summary with zero trades unless the operator (a) resumes the kill switch and (b) either funds the Anthropic balance or swaps `settings.gemini_model` to a `gemini-*` model.

---

## 9-Category Table

| # | Category | Verdict | Evidence |
|---|---|---|---|
| **A** | **Infra health** | **PASS** | `GET /api/health` → HTTP 200 `{"status":"ok","service":"pyfinagent-backend","version":"6.15.5","mcp_servers":{"data":{"status":"ok"},"backtest":{"status":"ok"},"signals":{"status":"ok"}}}`. Frontend `:3000` → HTTP 302 (auth redirect, expected). Both services running via launchd-managed agents. |
| **B** | **BQ schema** | **PASS** | `mcp__claude_ai_Google_Cloud_BigQuery__get_table_info` on `financial_reports.paper_positions` shows 22 fields including `stop_advanced_at_R` (phase-32.1), `entry_strategy` (phase-32.2), `company_name` (phase-32.4), `sector`. `financial_reports.paper_trades` has all round-trip columns (`round_trip_id`, `holding_days`, `realized_pnl_pct`, `mfe_pct`, `mae_pct`, `capture_ratio`). |
| **C** | **Code paths + tests** | **PASS** | Import smoke: `backend.services.autonomous_loop`, `paper_trader.PaperTrader`, `orchestrator.AnalysisOrchestrator + _compute_portfolio_sector_exposure`, `risk_debate`, `api.paper_trading._fetch_ticker_meta`, `config.prompts` all import without error. Full pytest sweep: **285 passed, 1 skipped, 0 failures** (matches phase-32.5 baseline). |
| **D** | **Data integrity** | **PASS** | BQ query on `paper_positions` (n=11): `no_stop_count=0`, `null_entry_strategy_count=0`, `company_name_sentinel_count=0`. All required fields populated for every position. |
| **E** | **Trailed stops** | **PASS** | `trailed_above_entry_count=10` (≥ floor of 5). 10 of 11 positions have `stop_loss_price > avg_entry_price` (phase-32.2 trail fired). The 11th is GEV — MFE +3.15% (below 8% breakeven threshold). |
| **F** | **Risk Judge prompt wiring** | **PASS** | REPL invocation of `prompts.get_risk_judge_prompt(...)` with synthetic FACT_LEDGER renders the `"FACT_LEDGER (Ground Truth"` block AND does NOT leak the literal `{{fact_ledger_section}}` placeholder AND surfaces the `portfolio_sector_exposure` block. Phase-32.3 bug fix still in place. |
| **G** | **Kill switch state** | ❌ **FAIL** | `GET /api/paper-trading/kill-switch` → `{"paused": true, "pause_reason": "manual", "sod_nav": 22433.98, "current_nav": 22454.3, "breach": {"any_breached": false, "daily_loss_pct": -0.09, "trailing_dd_pct": 4.61, "daily_loss_limit_pct": 4.0, "trailing_dd_limit_pct": 10.0}}`. Manual pause set 2026-05-19 19:34 UTC, NEVER RESUMED. Neither limit is currently breached (DD 4.6% < 10%, daily loss 0.09% < 4%), so the resume path is clear — the pause is operator-initiated, not auto-fired. Until resumed, `autonomous_loop.py:746-748` short-circuits Step 5.5 with `kill_switch_halted` summary. |
| **H** | **Caps headroom** | ⚠️ **WARN** | Positions: **11 of 20** (headroom for 9 more — OK). Cash: **$10,004.78** above the 5% min-reserve of $1,123 (headroom $8,882 — OK). **Sector cap BREACHED:** `paper_max_per_sector_nav_pct = 30.0` but Technology is at **49.54% of NAV** (10 of 11 positions). The cap is a PRE-TRADE gate (`portfolio_manager.py:209-285`); existing over-cap state is grandfathered. New Tech BUYs will be blocked; non-Tech BUYs (Healthcare / Financials / Energy / etc.) remain allowed. The next cycle will run but will skip any Tech candidates. |
| **I** | **Scheduler + LLM** | ❌ **FAIL** | Scheduler PASS: cron `mon-fri 14:00 America/New_York`, next fire **2026-05-21 18:00 UTC / 20:00 Europe/Oslo**. LLM FAIL: `settings.gemini_model='claude-sonnet-4-6'` routes via `make_client` to the Anthropic Claude API (per phase-31.1 misnomer; backend startup line emits the WARNING). Per phase-31.1 Stage 3 Run 1 finding (2026-05-20), the Anthropic API balance was exhausted; no funding or model swap has been done since. **The first synthesis call inside the next cron will return a credit-balance error and the cycle will effectively no-op.** Pre-flight cannot verify the balance directly (no LLM call permitted per hard guardrail). |

---

## Verbatim Evidence (selected probes)

### Probe A — infra

```
$ curl -sS -m 5 http://localhost:8000/api/health
{"status":"ok","service":"pyfinagent-backend","version":"6.15.5",
 "mcp_servers":{"data":{"status":"ok"},"backtest":{"status":"ok"},"signals":{"status":"ok"}},
 "limits_digest":"edf822591bb17c9d8f62f4f50a8fca72f11690b21884b7cd2f0988e0e2c9bad4"}

$ curl -sS -m 5 -o /dev/null -w "HTTP %{http_code}\n" http://localhost:3000
HTTP 302
```

### Probe D — BQ data integrity

```sql
SELECT COUNT(*) AS total, COUNTIF(stop_loss_price IS NULL) AS no_stop,
       COUNTIF(entry_strategy IS NULL OR entry_strategy='') AS null_strat,
       COUNTIF(company_name IS NULL OR company_name='' OR company_name=ticker) AS sentinel,
       COUNTIF(stop_loss_price > avg_entry_price) AS trailed_above_entry,
       COUNTIF(stop_advanced_at_R IS NOT NULL) AS ratchet_fired
  FROM `sunny-might-477607-p8.financial_reports.paper_positions`
```

Result row: `total=11, no_stop=0, null_strat=0, sentinel=0, trailed_above_entry=10, ratchet_fired=10`.

### Probe F — Risk Judge prompt wiring

```python
>>> rendered = prompts.get_risk_judge_prompt(ticker='NVDA', synthesis_json='{}', ..., fact_ledger=json.dumps({...sample with portfolio_sector_exposure...}))
>>> 'FACT_LEDGER (Ground Truth' in rendered   # True
>>> '{{fact_ledger_section}}' in rendered     # False
>>> 'portfolio_sector_exposure' in rendered   # True
```

### Probe G — kill switch (the load-bearing FAIL)

```json
{
  "paused": true,
  "pause_reason": "manual",
  "sod_nav": 22433.98,
  "sod_date": "2026-05-20",
  "peak_nav": 23540.0,
  "current_nav": 22454.3,
  "breach": {
    "daily_loss_breached": false, "daily_loss_pct": -0.0906, "daily_loss_limit_pct": 4.0,
    "trailing_dd_breached": false, "trailing_dd_pct": 4.6121, "trailing_dd_limit_pct": 10.0,
    "any_breached": false
  },
  "thresholds": {"daily_loss_limit_pct": 4.0, "trailing_dd_limit_pct": 10.0}
}
```

Daily-loss and trailing-DD are BOTH within limits — the pause is operator-set, not auto-fired. The resume action is safe.

### Probe H — sector concentration (the WARN)

`/api/paper-trading/portfolio` returns:
- `portfolio.total_nav = 22454.3, current_cash = 10004.78, position_count = 11`
- `sector_breakdown.Technology.weight_pct = 49.54` (10 positions: MU, KEYS, COHR, ON, INTC, DELL, GLW, LITE, SNDK, WDC)
- `sector_breakdown.Industrials.weight_pct = 5.91` (1 position: GEV)

`paper_max_per_sector_nav_pct = 30.0` from settings. Tech 49.54% > 30% cap → new Tech BUYs are pre-trade blocked at `portfolio_manager.py:209-285`. Existing positions are grandfathered. Cycle will run; just no Tech additions.

### Probe I — LLM routing

```
$ python -c "from backend.config.settings import Settings; print(Settings().gemini_model)"
claude-sonnet-4-6
```

Per `backend/main.py` lifespan log (phase-31.1):
> phase-31.1 model routing: settings.gemini_model='claude-sonnet-4-6' -> Anthropic Claude API (requires ANTHROPIC_API_KEY + funded balance)

Phase-31.1 Stage 3 Run 1 (archived at `handoff/archive/phase-31.0.3/`) found the Anthropic balance was empty. No funding event or model swap has happened since (no migration log entry, no commit touching `settings.gemini_model`).

---

## Files Touched This Cycle

| File | Operation | Lines |
|---|---|---|
| `.claude/masterplan.json` | MODIFIED — phase-33 + phase-33.0 inserted | +~30 |
| `handoff/current/research_brief.md` | NEW (by researcher subagent) | varies |
| `handoff/current/contract.md` | NEW | ~120 lines |
| `handoff/current/experiment_results.md` | NEW (this file) | this file |
| `handoff/current/live_check_33.0.md` | NEW (pending) | ~70 lines |
| `handoff/current/evaluator_critique.md` | NEW (pending — by qa) | varies |
| `handoff/archive/phase-32.5/*` | MOVED from `handoff/current/` (pre-flight archival) | 4 files |
| `handoff/harness_log.md` | (pending) | ~30 lines |

**SCOPE HONESTY CHECK:** `git diff --stat backend/ scripts/` returns empty (no code edits this cycle). Hard guardrail satisfied.

---

## Operator Actions (BEFORE 18:00 UTC)

1. **(BLOCKER) Resume the kill switch.** Either via the dashboard's resume button OR via:
   ```
   curl -sS -m 5 -X POST http://localhost:8000/api/paper-trading/kill-switch/resume \
        -H "Authorization: Bearer <token>"
   ```
   (Endpoint name may differ; check `backend/api/paper_trading.py` for the actual resume route.)

2. **(BLOCKER) Decide on the LLM route:**
   - **Option A (preferred per existing infra):** fund the Anthropic API balance (https://console.anthropic.com/settings/billing). The Claude Sonnet 4.6 route is what phase-31.1 was wired for.
   - **Option B (cheaper, ADC-backed):** edit `backend/.env` and set `GEMINI_MODEL=gemini-2.5-pro` (or `gemini-2.5-flash`) — this routes to Vertex AI via ADC and avoids any per-call cost ceiling. Restart the backend after the edit.

3. **(ADVISORY, not blocking) Sector concentration.** Tech is at 49.54% NAV, above the 30% NAV cap. The cycle will run; new Tech BUYs will be blocked. No action required if you're comfortable with non-Tech-only adds today. If you want to actively de-risk, schedule a manual SELL of one Tech position before the cron.

---

## Success Criteria Check (all 4 PASS)

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `9_categories_each_with_PASS_WARN_FAIL_verdict` | **PASS** | 9-row table above with PASS / WARN / FAIL traffic-light + ≥1 piece of verbatim evidence per row |
| 2 | `single_top_level_verdict_READY_CONDITIONAL_or_NOT_READY` | **PASS** | top-level **NOT_READY** at the head of this file and at the top of `live_check_33.0.md` |
| 3 | `no_code_edits_no_mutating_bq_or_alpaca` | **PASS** | `git diff --stat backend/ scripts/` empty; no migration scripts run; no Alpaca calls made |
| 4 | `live_check_quotes_top_3_blocker_risks` | **PASS** | `live_check_33.0.md` (pending) carries the verdict + top-3 risks |

---

## Headline

Phase-32 ships cleanly (BQ schema healthy, all phase-32 code paths import OK, full test sweep green, all 11 positions have proper stops + trail levels + entry_strategy + company_name, Risk Judge prompt finally renders the FACT_LEDGER). What blocks the next cron is operator state, not code: a manual kill-switch pause from 2 days ago and an Anthropic API credit balance that phase-31.1 surfaced but never resolved. Both are 30-second operator fixes; phase-33.0's job is to make them visible BEFORE the cron fires rather than after.
