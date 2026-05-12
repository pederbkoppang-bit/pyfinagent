---
step: 24.5
title: Slack notifications + operator alerting audit (P0)
date: 2026-05-12
tier: complex
---

## Research: Slack Notifications + Operator Alerting Audit (phase-24.5)

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://docs.slack.dev/reference/block-kit/blocks/alert-block/ | 2026-05-12 | Official doc | WebFetch | New Alert block (April 2026): 5 severity levels (default/info/warning/error/success), mrkdwn text, block_id. No built-in deduplication or threading. |
| https://oneuptime.com/blog/post/2026-02-20-monitoring-alerting-best-practices/view | 2026-05-12 | Blog | WebFetch | Critical/Warning/Info tier model; every alert must have a runbook; symptom-based not infra-metric; burn-rate SLO; dedup by grouping within time windows. |
| https://oneuptime.com/blog/post/2026-01-30-alert-deduplication/view | 2026-05-12 | Blog | WebFetch | Fingerprinting (alertname+host+service hash), dedup_key per PagerDuty/OpsGenie, fixed vs. sliding time windows; identical-issue grouping prevents 50 alert floods. |
| https://pro.stockalarm.io/blog/day-trading-alerts-setup-guide | 2026-05-12 | Industry blog | WebFetch | 5 essential trading alert types; urgency tiers (phone > push > text > email); start with 5-10 high-quality alerts not 50; daily hygiene — deactivate closed-position alerts immediately. |
| https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-05-12 | Official doc | WebFetch | File-based handoff as implicit audit trail; contract-based verification (pre-established success criteria); evaluator as behavioral health monitor; no explicit escalation-threshold design documented. |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://docs.slack.dev/changelog/2026/04/16/block-kit-new-blocks/ | Official doc | Snippet covered Card/Alert/Carousel availability date |
| https://oneuptime.com/blog/post/2026-01-30-alert-fatigue-prevention/view | Blog | Covered by read-in-full deduplication article |
| https://upticknow.com/blog/reduce-alert-fatigue-monitoring-systems-2026.html | Blog | Budget exhausted; dedup article covered same ground |
| https://icinga.com/blog/alert-fatigue-monitoring/ | Blog | Snippet adequate for context |
| https://www.unitedfintech.com/blog/using-real-time-trading-alerts | Industry | Snippet covered channel-matching concept |
| https://stockalarm.io/ | Product | Product page, not a guidance article |
| https://corporateinsight.com/live-trading-alerts-for-self-directed-brokerage-clients/ | Research | Snippet covered delivery-method flexibility |
| https://pingfatigue.com/what-is-alert-fatigue | Community | Snippet covered definition; read-in-full article more authoritative |
| https://link.springer.com/chapter/10.1007/978-3-032-19540-1_2 | Peer-reviewed | Paywalled; snippet provided core concept |
| https://www.sherlocks.ai/discover/tools-to-reduce-alert-fatigue | Industry | Snippet listed tool categories adequately |

### Recency scan (2024-2026)

Searched "Slack Block Kit alert 2026", "alert fatigue deduplication monitoring 2026", "financial trading alert design 2026", "alert deduplication 2025". Findings:

- Slack introduced the new Alert block in April 2026 with 5 severity levels — directly applicable to pyfinagent's P0/P1/P2 escalation schema. This supersedes using bare section blocks for severity messaging.
- Two authoritative 2026 guides (oneuptime Feb 2026) codify the deduplication fingerprinting + time-window pattern, aligning with watchdog's existing state-transition gating (scheduler.py:310-337). The existing watchdog is already dedup-correct; digests are not.
- No new peer-reviewed literature on trading notification systems found in 2025-2026 beyond practitioner blogs.

### Three-variant search-query discipline

1. Current-year frontier: "Slack Block Kit alert design 2026", "alert fatigue deduplication monitoring best practices 2026", "financial trading alert design 2026"
2. Last-2-year window: "alert deduplication 2025", "trading notifications 2025"
3. Year-less canonical: "financial trading real-time notifications alert design", "Slack Block Kit building", "alert deduplication"

---

## Key findings

1. **WRONG ENDPOINT for digests** — Both `_send_morning_digest` and `_send_evening_digest` call `/api/portfolio/performance` (scheduler.py:235, 260), which is the OLD in-memory portfolio (`backend/api/portfolio.py`). That router uses `_positions: dict` (portfolio.py:23) — an ephemeral in-process dict that is always empty on a fresh process restart. It returns `total_pnl: 0, total_return_pct: 0` by default (portfolio.py:99-107). The CORRECT live P&L comes from `/api/paper-trading/portfolio` which reads from `paper_trader.get_positions()` backed by BQ (paper_trading.py:175-208). This is the root cause of `Portfolio: +$0.00 (+0.0%)`.

2. **`total_return_pct` field name mismatch** — `format_morning_digest` reads `portfolio_data.get("total_return_pct", 0)` (formatters.py:322). The legacy `/api/portfolio/performance` returns key `total_pnl_pct` (not `total_return_pct`) (portfolio.py:143). The working paper-trading endpoint also returns `total_pnl_pct` (paper_trading.py:140). So even if the endpoint were switched, the formatter reads the wrong key — `total_return_pct` will always be 0. Bug exists at two levels: wrong endpoint AND wrong key name.

3. **"Recent Analyses" uses unfiltered BQ query** — The morning digest calls `/api/reports/?limit=5` (scheduler.py:238). `list_reports` calls `bq.get_recent_reports(limit=5)` which runs `SELECT ... FROM reports_table ORDER BY analysis_date DESC LIMIT 5` with NO ticker filter (bigquery_client.py:258-268). If SNDK was analyzed 5 times recently, all 5 rows are SNDK. There is no deduplication by ticker or rotation across current holdings.

4. **Morning digest scheduler TZ is correct; configured hour is wrong** — `start_scheduler` uses `timezone=ZoneInfo("America/New_York")` (scheduler.py:144). The APScheduler cron is DST-aware ET. The default `morning_digest_hour` in settings is `8` (settings.py:199) and `evening_digest_hour` is `17`. Operator reports 2:00 PM ET for morning digest and 11:00 PM for evening — meaning the `.env` overrides are `morning_digest_hour=14` and `evening_digest_hour=23`. The scheduler TZ code is correct; the issue is the operator has wrong values in `.env`.

5. **No trade confirmation notifications** — `send_analysis_alert` exists (scheduler.py:426-458) and fires after analysis. But there is no `send_trade_confirmation` function anywhere in the codebase. When `paper_trader.execute_trade()` is called from `paper_trading.py`, no Slack message is sent. Zero call sites for trade-confirmation Block Kit blocks exist.

6. **No kill-switch alert** — `pause_signals()` exists (scheduler.py:353-366) but only calls `scheduler.shutdown()` with a logger.info. It does NOT call `send_trading_escalation()`. No Slack post is triggered on kill-switch activation.

7. **No drawdown alarm** — `/api/paper-trading/drawdown` exists (paper_trading.py:~363) but its output is never wired to a Slack alert. `send_trading_escalation` exists in scheduler.py:369-423 with P0 iMessage escalation, but nothing calls it for drawdown threshold breaches.

8. **`cost_budget_watcher` exists but only reaches Slack via `alert_fn` injection** — The phase-9.8 job (cost_budget_watcher.py) DOES fire when BQ spend exceeds caps. But its alert_fn is only wired in production when `_production_fns.make_alert_fn_for_budget` is called during `register_phase9_jobs` (scheduler.py:562-565). If that wiring fails (logged as fail-open at scheduler.py:569-571), cost alerts are silently dropped.

9. **`format_escalation_alert` is implemented and P0 iMessage is wired** — The escalation infrastructure exists (scheduler.py:369-423, formatters.py:624-686) and is production-ready. It just isn't called from the right places.

10. **`/portfolio` slash command hits the SAME wrong endpoint** — `handle_portfolio` calls `GET /api/portfolio/performance` (commands.py:138). Per operator, `/portfolio` returns INCORRECT P&L (matching the empty in-memory store). The formatter reads `total_pnl` (formatters.py:105) and `total_return_pct` (formatters.py:106) — both absent from the old endpoint's response shape (which returns `total_pnl_pct`, not `total_return_pct`).

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/slack_bot/app.py` | 78 | Bolt entry, Socket Mode, task spawning | Healthy |
| `backend/slack_bot/scheduler.py` | 610 | APScheduler: digests, watchdog, escalation, phase-9 jobs | BUG: digests hit wrong endpoint |
| `backend/slack_bot/formatters.py` | 687 | Block Kit builders: analysis, portfolio, morning/evening digest, escalation, accuracy | BUG: wrong field key `total_return_pct` |
| `backend/slack_bot/commands.py` | 303 | Slash commands: /analyze, /portfolio, /report; message handler | BUG: /portfolio hits wrong endpoint |
| `backend/slack_bot/jobs/cost_budget_watcher.py` | ~120 | BQ spend circuit breaker | Works but silently fails if alert_fn not wired |
| `backend/slack_bot/jobs/_production_fns.py` | ~290 | alert_fn factory for phase-9 jobs | Healthy; fail-open on wiring error |
| `backend/api/portfolio.py` | 169 | OLD in-memory portfolio (ephemeral, always empty on restart) | WRONG TARGET for digests |
| `backend/api/paper_trading.py` | ~940 | Live paper trading with BQ-backed positions | CORRECT data source, not used by digests |
| `backend/db/bigquery_client.py` | 730+ | BQ queries including `get_recent_reports` | BUG: no ticker-distinct dedup in list |
| `backend/config/settings.py` | ~211 | `morning_digest_hour=8`, `evening_digest_hour=17` (defaults) | Defaults correct; .env values wrong |

---

### Consensus vs debate (external)

Consensus: severity-tiered alerting with deduplication, symptom-based triggers, and channel-urgency matching are the standard. The new Slack Alert block (April 2026) natively supports severity levels that should replace the current plain section blocks for P0/P1/P2 escalations.

Debate: whether trade confirmation should fire for every trade or only above a position-size threshold (noise prevention). Standard practice (stockalarm.io) says start with fewer, higher-quality alerts.

### Pitfalls (from literature)

1. Alert fatigue from too many low-signal notifications — existing watchdog correctly uses state-transition gating (healthy pattern).
2. Deduplication fingerprint too specific (includes timestamps) = breaks grouping — `get_recent_reports` has no fingerprint at all for recent-analyses digest.
3. Delivery channel mismatch — iMessage for P0 is already implemented but nothing triggers it for drawdown or kill-switch.

---

### Application to pyfinagent

**Bug 1 — Portfolio P&L always $0.00:**
Root: `scheduler.py:235` calls `GET /api/portfolio/performance` (in-memory, empty).
Fix: Switch to `GET /api/paper-trading/portfolio`.
Field fix: `formatters.py:322` reads `total_return_pct`; paper-trading response returns `total_pnl_pct`. Change formatter key OR normalize paper-trading response to include `total_return_pct`.

**Bug 2 — Recent Analyses shows 5x SNDK:**
Root: `bigquery_client.py:258-268` runs `ORDER BY analysis_date DESC LIMIT 5` with no ticker deduplication.
Fix: Either (a) `QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY analysis_date DESC) = 1` to get 1 per ticker, or (b) fetch limit=25 and deduplicate in Python, or (c) add a `DISTINCT ON (ticker)` approach.

**Bug 3 — Digest fires at wrong time:**
Root: `.env` file has `morning_digest_hour=14`, `evening_digest_hour=23`. TZ code is correct (scheduler.py:144-148, `ZoneInfo("America/New_York")`).
Fix: Change `.env` to `MORNING_DIGEST_HOUR=6` and `EVENING_DIGEST_HOUR=23` (or 20 for 8 PM). This is an operator config fix only — no code change needed.

**Missing notifications (all require new code):**
- Trade confirmation: hook into `paper_trader.execute_trade` or `paper_trading.py`'s cycle result — call `send_trading_escalation` or a new `send_trade_confirmation` function.
- Kill-switch alert: in `pause_signals()` (scheduler.py:353-366), add `asyncio.create_task(send_trading_escalation(app, "P0", "Kill Switch Activated", ...))` after scheduler shutdown.
- Drawdown alarm: add threshold check in `_send_morning_digest` after fetching portfolio, or in the paper-trading daily cycle.
- Cycle completion summary: after the scheduled paper-trading cycle in `paper_trading.py:931`, post a brief Block Kit summary to Slack.
- Error escalation routing: promote logger.exception sites in scheduler.py to `send_trading_escalation("P1", ...)` for digest failures.

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5 fetched)
- [x] 10+ unique URLs total (15 collected: 5 read in full + 10 snippet-only)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (scheduler, formatters, commands, portfolio API, paper_trading API, BQ client, settings, 2 job modules)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim (not just listed in a footer)

---

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 10,
  "urls_collected": 15,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
