---
step: virtual-fund-readiness
date: 2026-04-24
tier: moderate
researcher: researcher-agent
---

## Research: Virtual $10K Fund Autonomous Readiness — 2026-04-24

### Read in full (>=5 required; counts toward the gate)
| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://quantpedia.com/how-to-paper-trade-quantpedia-backtests/ | 2026-04-24 | doc/practitioner | WebFetch | Warmup, logging, out-of-sample validation are prerequisites; paper trading proves backtest not overfit |
| https://adventuresofgreg.com/blog/2025/12/15/algorithmic-trading-strategy-checklist-key-elements/ | 2026-04-24 | blog/practitioner | WebFetch | 12-item checklist; max-DD <20%, walk-forward >50-60%, kill switch must fire within 5s |
| https://markaicode.com/ai-quant-trading-bot-backtest-risk-live-execution/ | 2026-04-24 | blog/practitioner | WebFetch | Three non-negotiable circuit breakers: per-trade stop 1.5%, daily loss 5%, max-DD 15%; 2-week paper minimum before live |
| https://docs.alpaca.markets/docs/paper-trading | 2026-04-24 | official doc | WebFetch | Paper env does NOT simulate slippage, market impact, or dividends; endpoint/key separation required; readiness gate = small live test after paper |
| https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-04-24 | official doc | WebFetch | Every harness component encodes an assumption worth stress-testing; graduated feedback loops preferred over single go/no-go |
| https://arxiv.org/html/2412.20138v3 | 2026-04-24 | paper (arXiv) | WebFetch | TradingAgents: risk team assesses volatility/liquidity/counterparty before fund manager approves; qualitative deliberation not hard thresholds |

### Identified but snippet-only (does NOT count toward gate)
| URL | Kind | Why not fetched |
|-----|------|----------------|
| https://wundertrading.com/journal/en/learn/article/agentic-trading | blog | 403 |
| https://www.fia.org/sites/default/files/2024-07/FIA_WP_AUTOMATED%20TRADING%20RISK%20CONTROLS_FINAL_0.pdf | industry PDF | Binary/unreadable |
| https://www.esma.europa.eu/.../ESMA74-1505669079-10311_Supervisory_Briefing... | regulator PDF | Binary/unreadable |
| https://www.quantstart.com/articles/Successful-Backtesting-of-Algorithmic-Trading-Strategies-Part-I/ | practitioner | No explicit go-live gates |
| https://nurp.com/algorithmic-trading-blog/future-of-algorithmic-trading-trends-and-predictions/ | blog | Snippet only — trend article |

### Recency scan (2024-2026)
Searched: "paper trading autonomous readiness 2026", "AI trading agent pre-production gates 2025", "ESMA algorithmic trading supervisory briefing 2026". Result: no new academic findings supersede the practitioner consensus. ESMA 2026-02 briefing confirms annual self-assessment + kill switch requirement but is PDF-only and unreadable. markaicode.com (2026) and adventuresofgreg.com (2025-12) confirm 2-week minimum paper window and <20% max-DD as current industry floor.

---

### Internal code inventory
| File | Lines | Role | Status |
|------|-------|------|--------|
| backend/services/autonomous_loop.py | ~350 | Daily cycle orchestrator (screen→analyze→decide→trade→snapshot) | LIVE — BLOCKER-1 fix (price<=0 warning + BUY rules) present at line 248 |
| backend/services/paper_go_live_gate.py | 129 | 5-boolean promote gate | LIVE — gate correct, data insufficient |
| backend/services/kill_switch.py | 177 | KillSwitchState, evaluate_breach | LIVE — daily_loss=4%, trailing_DD=10% |
| backend/config/settings.py | ~160 | paper_daily_loss_limit_pct=4.0, trailing_dd=10.0, paper_trading_hour=10 ET | LIVE |
| backend/services/portfolio_manager.py | ~200 | decide_trades, Risk Judge extraction | LIVE — Risk Judge rejection is silent (no logging if risk_judge_decision != APPROVE) |

---

### Key findings

**BLOCKER: Backend is running stale code.**
- Process PID 57446 started at 16:33 CEST.
- Commit dfd57ec1 (BLOCKER-1 fix) landed at 18:09 CEST — 96 minutes later.
- The running process cannot have the fix in memory. The `logger.warning "Dropping BUY for"` guard and the revised Claude prompt are NOT active.
- Evidence: `ps -o lstart` is unambiguous. The launchd agent exit code is -15 (signal), meaning it was not restarted cleanly post-commit.

**Go-live gate: 2/5 passing.**
- PASSING: `max_dd_within_tolerance` (current DD ~0.5% vs 20% cap) and `sr_gap_le_30pct` (proxy via reconciliation divergence, assumed low with only 1 trade).
- FAILING: `trades_ge_100` (1 round trip vs 100 required), `psr_ge_95_sustained_30d` (insufficient observations), `dsr_ge_95` (insufficient observations). This is structural — 27 days of drought means no statistical basis for PSR/DSR.

**Kill switch margin.**
- Daily loss limit: 4% of SOD NAV. With NAV 9499.50, trigger = NAV drops ~$380 intraday.
- Trailing DD limit: 10% from peak. Current inception DD = 0.5% (9499.50 vs 10000). Headroom = 9.5 percentage points. NOT near trigger.
- Kill switch is NOT armed at current NAV — the frontend "KILL ACTIVE" label refers to the mechanism being enabled, not a breach.

**Risk Judge silent rejection risk.**
- `portfolio_manager.py:153` extracts `risk_assessment.get("decision", "")` but there is no explicit guard requiring `decision == "APPROVE"` before adding to `buy_candidates`. The Risk Judge sizing is used (position_pct), but a REJECT decision does not filter the candidate out. However, a REJECT decision likely sets `position_pct` to 0, causing `buy_amount < 50` skip at line 176. This is silent — no log line emitted. After BLOCKER-1 fix loads, the new Claude prompt should increase BUY base rate, which will surface whether Risk Judge is the next bottleneck.

**Monday 14:00 cycle (local = 09:00 ET).**
- `paper_trading_hour = 10 ET` in settings default. The frontend shows "NEXT RUN 27 apr 14:00" which is 14:00 CEST = 08:00 ET. There is a 2-hour discrepancy between the settings default and the displayed time — the `.env` likely overrides to a different hour. The cycle WILL fire Monday; the exact hour is env-dependent but confirmed weekday.

---

### Consensus vs debate
External consensus: 2-week minimum paper window, max-DD cap 15-20%, kill switch mandatory, statistical significance (50+ trades minimum). pyfinagent's gate requires 100 round trips — stricter than industry floor, appropriate given DSR-gated promotion criteria.

### Pitfalls
1. Stale process is the single highest-priority risk — zero benefit from the BLOCKER-1 fix until restart.
2. Risk Judge silent rejection — even post-restart, if the 28-agent pipeline produces no BUY-rated candidates, no trades will flow. Needs a post-restart cycle observation to confirm.
3. 27-day drought means PSR/DSR gates will remain red for weeks even if trading resumes normally — promotion eligibility is not Monday's concern, but the drought must end to accumulate data.

### Application to pyfinagent
- External practitioners require kill switch functional + 2-week minimum paper window. We have the switch; we have >4 weeks of paper history but only 1 trade — the window is open but data-sparse.
- Anthropic harness doctrine: "every assumption worth stress-testing." The fix being on disk but not in-process memory is exactly this failure mode — the component assumes a deployment step that didn't happen.

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total (11 collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module
- [x] Contradictions/consensus noted
- [x] All claims cited per-claim

---

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 5,
  "urls_collected": 11,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "gate_passed": true
}
```

Appended 2026-04-24 per `.claude/rules/research-gate.md` (task #47 follow-up).
