---
name: nav-asof-killswitch-86-12
description: Step 86.12 -- current_nav is a STORED BQ figure on all 5 kill-switch paths; the mark asof (paper_portfolio.updated_at) is already in the dict every path reads and is thrown away; baseline freshness IS checked, NAV freshness is NOT
metadata:
  type: project
---

**Fact.** `evaluate_breach(current_nav, ...)` never marks anything. All five
production producers of `current_nav` read the persisted
`paper_portfolio.total_nav` column (or, on the MCP path, accept an arbitrary
caller value):

- `backend/api/paper_trading.py:513-517` (GET /kill-switch) -- STORED + `starting_capital` fallback
- `backend/api/paper_trading.py:569-580` (POST /resume) -- STORED, **no** starting_capital fallback
- `backend/services/paper_trader.py:1269-1272` (`roll_daily_anchor`, Step 0) -- STORED
- `backend/services/paper_trader.py:1342-1343` (`check_and_enforce_kill_switch`, Step 5.5) -- STORED but fresh by ordering (Step 5 marks first)
- `backend/agents/mcp_servers/risk_server.py:64-82` -- **caller-supplied float, zero provenance**

**The asymmetry.** `kill_switch.py:865` disarms the daily leg on a stale
*baseline* (`_sod_date_is_stale`, `:961-986`, whose docstring says "freshness is a
claim that must be provable"). `kill_switch.py:887` validates `current_nav` only for
`None`/`<=0`. So `(sod - current_nav)/sod` at `:916` pairs a provably-today
denominator with a possibly-days-old numerator -- the classic *nonsynchronous
trading* pair (Getmansky/Lo/Makarov JFE 2004: prices "recorded at different times but
erroneously treated as if recorded simultaneously"), which biases estimated variance
DOWNWARD, i.e. **the error direction is FIRE LATE, not fire early.**

**Why the fix needs no migration.** `mark_to_market` already writes
`"updated_at": now(utc)` into the portfolio row (`paper_trader.py:789-795`) and
`get_paper_portfolio` is `SELECT *` (`bigquery_client.py:553-566`) -- so the asof is
sitting in the dict all four BQ paths read and drop. In-repo prior art for the exact
pattern one level down: the per-position `marked_at` stamp (phase-61.3), explicitly
"Observability only: no order, stop, or size depends on it."

**Second-order staleness inside a "fresh" mark.** `mark_to_market` falls back to the
previous mark / entry price when `_get_live_price` returns None
(`paper_trader.py:705-708`) and **keeps the last-known USD market value** when FX is
unavailable for a non-USD position (`:713-721`), both silently. And
`nav = portfolio["current_cash"] + total_positions_value` (`:780`) sums stored cash
with possibly-stale marks -- nonsynchronous by construction.

**Measured from `handoff/kill_switch_audit.jsonl`.** Live-file event counts
`{pause:44, resume:10, sod_snapshot:10}` -- **zero `peak_update` rows in the live
file** (archives only). The `sod_snapshot` rows are stamped 18:47-20:58 UTC, i.e. at
or after the US close, so the "start-of-day" anchor is really a prior-close mark
(`kill_switch.py:7` says "4% of start-of-day NAV" -- doc drift). Row schema is
`{ts, event, nav, date, provisional?}`: `ts` is the WRITE time, never the NAV's asof,
so mark-age is not derivable from the trail. On 2026-08-09 the 85.6
provisional->final upgrade wrote 23833.94 -> 23833.94 five minutes apart; the row
cannot distinguish "mark ran, book flat" from "mark did not refresh".

**Why:** step 86.12 asks what a daily-loss limit should be evaluated against. The
answer turned out to be an observability gap, not a threshold question.

**How to apply:** any NAV-freshness guard must (a) default to STALE on a missing
`updated_at` (mirroring `_sod_date_is_stale`'s unparseable-is-stale choice at
`:968-972`), (b) use the data-side timestamp not `datetime.now()` at read time, (c)
follow the per-leg `*_missing`/`armed` shape and NEVER set `any_breached=True`
(`kill_switch.py:826-831` documents why that would flatten a healthy book), (d) cover
BOTH legs since the trailing leg at `:921-923` reads the same `current_nav`, and (e)
carry a reachable unblock condition or it re-creates the 85.6 resume deadlock
(template wording at `paper_trading.py:602-663`).

External anchors: BCBS 239 Principle 5 (timeliness co-equal with accuracy, scaled to
"the potential volatility of the risk being measured") + Principle 3(d) "single
authoritative source"; dbt source-freshness (data-side `loaded_at_field`, gate runs
first, downstream steps do not run). SEC 15c3-5 and EU RTS 6 are prescriptive about
thresholds and real-time monitoring but **silent on the freshness of the value
compared against them** -- the obligation comes from BCBS 239, not the trading rules.
Adversarial 2026 finding: prop firms have been REMOVING daily-loss limits once
real-time marking exists (Apex post-Mar-2026, TPT Jan-2025), relying on trailing
equity alone.

Doc drift found: `kill_switch.py:12` cites "FINRA Rule 15c3-5" -- it is an **SEC**
rule (17 CFR 240.15c3-5).

See also [[project_kill_switch_36_9_armed_semantics]],
[[project_kill_switch_deadlock_85_6]], [[project_stale_anchor_disarm_85_5_1]].
