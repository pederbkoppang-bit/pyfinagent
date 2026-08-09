# Research Brief — step 36.17

**Topic:** RISK POLICY — should protective stop-loss exits keep enforcing while the kill switch has HALTED the cycle?
**Tier:** moderate (caller-specified)
**Audit-class:** NO (coverage reported for information only)
**Started:** 2026-08-09
**Status:** IN PROGRESS (write-first; this file grows incrementally)

## Scope

Five questions from the caller:
1. Standard semantics of kill switch / trading halt / risk pause w.r.t. protective, risk-reducing, exit-only orders vs new entries. Is "liquidation-only" / "exit-only" / "risk-reducing-only" a NAMED standard state?
2. Failure literature: halting risk enforcement while retaining exposure.
3. Invariants required if protective exits continue during a pause; is `backfill_missing_stops` a NEW risk decision or a protective default?
4. Evaluate three options (a) move 5.6 above the halt, (b) stop-loss-only pass inside the halt branch, (c) accept + document.
5. Ordering hazard: Step 5.4 scale-out runs BEFORE the halt evaluation. Same defect class or different?

Constraints on recommendation: no threshold changes; no flatten on the blocked path; nothing that lets a paused book place BUYs.

---

## Search queries run (three-variant discipline)

(recorded as executed — see below)

---

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|

(populated incrementally)

---

## Internal code inventory

(populated incrementally)

---
## Internal code inventory (RE-DERIVED at source 2026-08-09 — the step text's :1334/:1336 are STALE)

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/services/autonomous_loop.py` | 3540 | daily cycle; Step 5.4/5.5/5.6 | LIVE (the only production stop-loss caller) |
| `backend/autonomous_loop.py` | — | SECOND file of the same name | present in repo; NOT the services module the cycle uses |
| `backend/services/kill_switch.py` | 1138 | pause/resume/flatten singleton | LIVE |
| `backend/services/paper_trader.py` | 1739 | order primitives | LIVE |
| `backend/config/settings.py` | 653 | flags | LIVE |

### Exact line anchors (all re-derived, not copied)

- `cycle_halt_reason()` — `backend/services/autonomous_loop.py:139-165` (def at :139, docstring :140-153, `triggered`->"breach" :155-156, `blocked`->block_reason :157-162, `is_paused`->"paused" :163-164, `return None` :165). Precedence breach > blocked > paused.
- Step 5.4 scale-out — `autonomous_loop.py:1370-1393`. `trader.check_scale_out_fires` called at **:1381**, wrapped in try/except that fails OPEN (:1390-1393, comment "Stop-loss enforcement at Step 5.6 still provides the floor").
- Step 5.5 halt — `autonomous_loop.py:1395-1437`. `check_and_enforce_kill_switch` at **:1400**; `cycle_halt_reason(ks_check, _ks_state().is_paused())` at **:1402**; `if halt_reason:` at **:1403**; `summary["halted"]=True` :1409; `summary["status"]="halted_kill_switch"` :1425; `halt_reason` recorded :1426; `_log_cycle_signals_to_bq` :1428; `mark_to_market` :1429; `save_daily_snapshot` :1430-1434; **`return summary` at :1437**.
- Step 5.6 stop-loss — `autonomous_loop.py:1439-1490` (header comment :1439-1452, `backfill_missing_stops` at **:1458**, `check_stop_losses` at **:1471**, `execute_sell(reason="stop_loss_trigger")` at **:1474-1481**, `closed_tickers.append` :1484). Company-name backfill :1492-1511 (cosmetic).
- **Every `return summary` site in `autonomous_loop.py`: :1437 (the halt), :1795, :1801, :1819.** Only :1437 precedes Step 5.6.

### The confirmed defect

Step 5.6 (:1439) is textually AFTER the halt's `return summary` (:1437). Therefore on ANY halted cycle — `breach`, `blocked`, or `paused` — neither `backfill_missing_stops` nor `check_stop_losses` nor the stop-out `execute_sell` runs. Confirmed by reading, not inferred.

### Pause semantics

`backend/services/kill_switch.py:13-16` (verbatim, re-counted):
```
Breach semantics (FINRA Rule 15c3-5 "hard block" pattern):
  - Pause = halt new entries; existing positions kept.
  - Flatten = close every open position at market; cancel pending orders.
  - Limit breach => auto-flatten + auto-pause + audit log; explicit human
```
So the module's OWN stated contract is "existing positions kept" — i.e. the book still holds risk while paused. Nothing in that contract says exits stop.

### Does flatten happen on every halt path? NO — only one branch

`paper_trader.py:1468-1473`:
```python
if breach["any_breached"] and not state.is_paused():
    logger.warning(...)
    flatten_result = self.flatten_all(reason="kill_switch_auto_flatten")
    state.pause(trigger="limit_breach", details=...)
    return {"triggered": True, "blocked": False, ...}
```
- `flatten_all` fires ONLY on `any_breached AND not already paused` → the `breach` halt reason.
- `blocked` branch (`paper_trader.py:1505-1543`, returns `{"triggered": False, "blocked": True, "block_reason": "kill_switch_disarmed_lost_history"}`) — NO flatten. Its own comment (:1499-1504) says the refusal "is per-cycle and non-latching, and existing positions are untouched, matching the module's documented semantics".
- `paused` (a later cycle after a prior breach) — `any_breached and not state.is_paused()` is False because `is_paused()` is True, so NO flatten. Also, a breach that RE-occurs while already paused takes no action at all.

**Net:** on `blocked` and `paused` cycles the book carries full exposure with stop enforcement switched OFF, every cycle, until the halt clears. That is the caller's premise and it is CORRECT as read.

### Does anything re-check the pause on the SELL path? NO — and that asymmetry is DELIBERATE and DOCUMENTED

- `execute_buy` IS gated: `paper_trader.py:282-294` calls `_kill_switch_refusal_for_buy()` (defined :183-234) and returns `None` on refusal. It FAILS CLOSED (:225-233) — an unreadable switch refuses.
- `execute_sell` (`paper_trader.py:531`) contains NO kill-switch/pause check. Verbatim rationale at `paper_trader.py:278-281`:
```
# execute_sell is deliberately NOT gated and that asymmetry is load-bearing:
# the switch performs its flatten THROUGH execute_sell, so gating sells on
# `paused` would deadlock the pause -- the switch could never close the
# positions it just decided to close. Selling is the safe direction.
```
**This is the single most important internal finding for the recommendation.** The "no new BUYs" invariant is enforced at the ORDER CHOKE POINT (`execute_buy:282`), not merely by the cycle's early return. `paper_trader.py:268-276` calls it "complete mediation for BUYs" (CWE-424 alternate path / CWE-638 remedy). So running a stop-loss-only pass during a halt CANNOT open or increase a position: any BUY reaching `execute_buy` while paused is refused at :282-294.

Caveat: the BUY gate keys on `state.is_paused()` and `baselines_present_in`. On the **`blocked`** path the switch is NOT paused and (post-anchor) baselines ARE present, so `_kill_switch_refusal_for_buy` returns None — the blocked path's BUY suppression comes ONLY from the cycle's early return at :1437. Any option that runs code after the halt must therefore keep the blocked path's `return` before Step 6 decide/execute.

### backfill_missing_stops — what it actually does

`paper_trader.py:926-1000`. For each open position with falsy `stop_loss_price`: `stop = round(avg_entry_price * (1 - default_pct/100), 4)`, persisted via `bq.save_paper_position`. `default_pct` defaults to `settings.paper_default_stop_loss_pct` (8.0). Skips positions that already have a stop and positions with `avg_entry_price <= 0`. It is idempotent (second run reports them as `skipped`). It writes to BQ but places NO order.

### check_stop_losses — what it actually does

`paper_trader.py:797-806`. Pure read: `if stop and current and current <= stop: triggered.append(...)`. No side effects, no orders. Note `if stop and current` — a NULL stop or a 0/None current price silently yields no trigger (this is the same NaN/None-suppression class recorded in prior memory).

### THE SEVERITY QUESTION: is there any OTHER stop-enforcement path?

**NO.** Grep proving it (repo-wide, Python only, node_modules excluded):
```
grep -rn "check_stop_losses" . --include="*.py" | grep -v /node_modules/
```
Production (non-test, non-script) hits are exactly:
- `backend/services/paper_trader.py:797` — the definition
- `backend/services/autonomous_loop.py:1471` — the ONLY production call site
(plus comment mentions at `autonomous_loop.py:1440,1441,1446,1452,1468,1494`)

Everything else is tests (`backend/tests/test_autonomous_loop_step_5_6.py`, `test_dod4_tier1_coverage_investment.py:1010/1023`, `test_phase_36_12_...:413`, `test_phase_61_3_addon_currency.py`), verifiers (`tests/verify_phase_25_1.py`), or `scripts/smoketest_stages_5_through_13.py`. **No API route, no scheduler job, no MCP tool calls `check_stop_losses`.** The daily cycle is the sole enforcement path, so a halted cycle = zero stop enforcement, full stop.

`backfill_missing_stops` has ONE additional non-cycle caller: `scripts/maintenance/backfill_stops.py:75` — a MANUAL maintenance CLI, not a scheduled job. It backfills stops but does not enforce them.

Corollary: `portfolio_manager.py` also emits stop-loss SELLs inside `decide_trades()` (per `handoff/harness_log.md:175`), but `decide_trades` is Step 6 — which is likewise behind the :1437 return. So that path is halted too.

---
## EXISTING TEST COVERAGE (contract-critical — enumerated, node IDs measured)

### `run_daily_cycle` IS drivable end-to-end in a test. Canonical example:
**`backend/tests/test_phase_36_12_kill_switch_trading_path_block.py::test_phase_36_12_a_blocked_cycle_really_places_no_orders`** (def at **:225**, drives `asyncio.run(al.run_daily_cycle(settings=settings))` at **:291**). Its own docstring (:226-247) calls it "THE COMPOSITION, EXECUTED".

**Exact mock surface required (copy this shape) — `test_phase_36_12...py:248-289`:**
```python
monkeypatch.setattr(al, "BigQueryClient", lambda *a, **k: bq)
monkeypatch.setattr(al, "PaperTrader",   lambda *a, **k: trader)
monkeypatch.setattr(al, "screen_universe",       lambda *a, **k: [])
monkeypatch.setattr(al, "rank_candidates",       lambda *a, **k: [])
monkeypatch.setattr(al, "get_sp500_tickers",     lambda *a, **k: [])
monkeypatch.setattr(al, "get_russell1000_tickers", lambda *a, **k: [])
monkeypatch.setattr(al, "_log_cycle_signals_to_bq", lambda *a, **k: 0)
monkeypatch.setattr(al, "AnalysisOrchestrator", MagicMock())
monkeypatch.setattr(al, "decide_trades", decide)
monkeypatch.setattr(al, "_running", False)
monkeypatch.setattr(cycle_health, "get_log", lambda *a, **k: fake_log)   # keeps GIT-TRACKED files clean
settings.news_screen_enabled = False                                     # COST hazard
trader.check_and_enforce_kill_switch.return_value = ks_blocked           # the halt injection point
```
Plus THREE fixtures in that module:
- `ks_isolated` — redirects `kill_switch._AUDIT_PATH` to tmp.
- autouse `_live_audit_file_is_write_protected` (**:38-60**) — byte-compares `handoff/kill_switch_audit.jsonl` before/after; fails the test if anything wrote to LIVE safety state.
- autouse `captured_alerts` (**:63-90**) — intercepts `alerting.raise_cron_alert_sync`. Its docstring records, MEASURED, that without it the suite **posts a false P1 to the operator's real Slack** ("alert bot-token fallback delivered=True ... 'Kill-switch DISARMED at cycle start'"). **Any new test in this area MUST carry this fixture** (channel-isolation class, phase-86.3).

Two documented hazards, both quoted from :238-246: **COST** — "An unstubbed run makes a REAL 150s LLM call"; **TRACKED STATE** — `handoff/.cycle_heartbeat.json` + `handoff/cycle_history.jsonl` are git-tracked.

### Full node-ID inventory

| File | Node ID (line) | Drives `run_daily_cycle`? | What it pins |
|---|---|---|---|
| `test_phase_36_12_kill_switch_trading_path_block.py` | `test_phase_36_12_blocked_cycle_halts_the_autonomous_loop` (:203) | no (predicate only) | `cycle_halt_reason` on blocked |
| " | `test_phase_36_12_a_blocked_cycle_really_places_no_orders` (:225) | **YES** (:291) | blocked path halts, no orders |
| " | `test_phase_36_12_an_already_paused_cycle_still_halts` (:317) | **YES** (:371) | paused path halts, no orders |
| " | `test_phase_36_12_a_quiet_cycle_actually_proceeds_and_trades` (:380) | **YES** (:435) | proceed direction |
| " | `test_phase_36_12_halt_precedence_is_breach_then_block_then_paused` (:447) | no | precedence |
| " | `test_phase_36_12_the_loop_actually_calls_the_halt_predicate` (:460) | no (AST) | branch SHAPE |
| `test_autonomous_loop_step_5_6.py` | `test_step_5_6_backfill_runs_before_check_stop_losses` (:67) + 3 more | **NO — surrogate** | ordering only |
| `test_phase_85_4_cycle_loudness.py` | `test_c3_killswitch_halt_records_a_real_terminal_status_not_running` (:300); `test_c3_halted_status_is_not_counted_as_a_completion` (:350) | **YES** (:213) | `status="halted_kill_switch"` |
| `test_phase_85_6_anchor_deadlock.py` | drives at :374/:400/:429/:441 | **YES** | anchor deadlock |
| `test_dod4_tier1_coverage_investment.py` | `test_paper_trader_check_stop_losses_returns_tickers_below_stop` (:1010); `..._backfill_missing_stops_skips_positions_with_stops` (:640); `..._skips_zero_entry_price` (:653) | no | unit-level trader |

### THE EXISTING STEP-5.6 COVERAGE IS STRUCTURALLY BLIND TO THIS DEFECT

`backend/tests/test_autonomous_loop_step_5_6.py` does **not** import or drive `run_daily_cycle`. It defines a LOCAL reproduction `async def _step_5_6_under_test(trader, summary)` at **:44-61**, whose own docstring says: *"Keeping the reproduction in the test file avoids importing the 1700-line module under test."* The only contact with production source is a line-ORDER scan (**:175-199**) over a 50-line window starting at the "Step 5.6" header. **Neither can observe that the halt at :1437 returns before Step 5.6 ever runs.** This is the "guard stops one seam short" class: 4 green tests assert the ordering *within* a block that, on a halted cycle, is never entered.

### COLLISION ANALYSIS for option (b) — MEASURED, not assumed

1. **The AST guard at :460-521 does NOT block option (b).** It walks every `body` list and requires that the statement *immediately following* `halt_reason = cycle_halt_reason(...)` is an `ast.If` testing that exact name (:503-513). It constrains what sits **between** the assignment and the `if` — **not** the contents of the `if` body. Adding statements *inside* `if halt_reason:` is invisible to it.
2. **`summary["steps"][-1] == "kill_switch_halted"` IS a hard collision.** Asserted at **:298** (blocked) and **:374** (paused). If option (b) appends any step name (e.g. `"stop_loss_enforcement"`) inside the halt branch, **both tests go RED.** The contract must either append nothing to `summary["steps"]` inside the halt branch (record under a distinct key such as `summary["halt_stop_loss_triggered"]`), or deliberately update those two assertions with a written justification. Do not discover this during GENERATE.
3. **`trader.execute_sell.called is False` (:304, :377) does NOT collide — measured.** `trader` is a `MagicMock`, so `trader.check_stop_losses()` returns a `MagicMock`, and I measured `list(MagicMock()) == []` (0 iterations). A `for ticker in triggered_stops:` loop therefore never reaches `execute_sell`, and both assertions stay green. This is a latent VACUITY though: those two tests would keep passing even if the new pass were broken. The new reproduce-first test must set `trader.check_stop_losses.return_value = ["WDC"]` explicitly.
4. **Measured trap if the backfill block is copied verbatim:** `backfill_result.get("count_backfilled", 0) > 0` raises `TypeError: '>' not supported between instances of 'MagicMock' and 'int'`. Production swallows it at `autonomous_loop.py:1466`, so it fails silently — another reason not to copy backfill into the halt branch (see Q3).

---

## External research

### Search queries run (three-variant discipline)
- **Year-less canonical:** `"liquidation only" mode exchange risk kill switch exit-only orders risk-reducing`; `MiFID II RTS 6 Article 12 kill functionality cancel unexecuted orders algorithmic trading`
- **Last-2-year / recency:** `stop-loss orders disabled during trading halt outage broker losses unable to close positions 2025 2026`
- Current-frontier hits arrived inside the recency query and via the 2026-dated primary documents (ESMA 2026-02, NYSE Pillar v4.7 2026-03-19, BusinessToday 2026-06-10).

### Read in full (6; floor is 5)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://eur-lex.europa.eu/eli/reg_del/2017/589/oj/eng | 2026-08-09 | Regulation (RTS 6) | WebFetch, full | Art. 12 verbatim: kill functionality = "cancel immediately... any or all of its **unexecuted orders**". Scoped to orders in flight, NOT to open positions. Art. 17(1): a triggered post-trade control may lead to "adjusting or shutting down the relevant trading algorithm... **or an orderly withdrawal from the market**". Art. 14(2)(g): BCP must include "alternative arrangements... to manage outstanding orders **and positions**". |
| 2 | https://www.esma.europa.eu/sites/default/files/2026-02/ESMA74-1505669079-10311_Supervisory_Briefing_on_Algorithmic_Trading_in_the_EU.pdf | 2026-08-09 | Supervisory briefing (2026-02-26) | curl + pdfplumber, 19pp full | ¶93: remedial actions form a LADDER — "the cancellation of orders, **a withdrawal from the market or** the use of the kill functionality" (three distinct actions). ¶11(5): "Risk Management Adjustments • Adjusting trading behaviour based on risk metrics (e.g. VaR, exposure limits, **stop-loss triggers**)" is itself classified as ALGORITHMIC TRADING. ¶70-74: mandatory "hard blocks"; a hard block is a *default limit traders cannot overcome*. |
| 3 | https://www.nyse.com/publicdocs/nyse/NYSE_Pillar_Risk_Controls.pdf | 2026-08-09 | Exchange spec v4.7 (2026-03-19) | WebFetch(binary)+pdfplumber, 40pp full | Breach Action menu (§6.7): "Notification only" / "**Block only - accept cancels; reject order, modify, cancel/replace**" / "Cancel Non-Auction orders & Block". A blocked entity STILL ACCEPTS risk-reducing messages. Kill Switch is itself a MENU of separable actions: "Block new order entry / Unblock / Cancel Non-Auction Orders / Cancel Auction Orders / Cancel GTC Orders". §2: "The risk controls are **not designed to be a firm's sole means of risk control** and should not be relied upon as such." |
| 4 | https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457088324/Kill+Switch | 2026-08-09 | Exchange spec | WebFetch, full | The **counter-example**: CME Globex Kill Switch blocks "**All order entry**" and cancels "All working orders". It provides **NO** liquidation-only / close-only mode, and the doc says nothing about what the firm must then do with its positions. |
| 5 | https://www.finra.org/rules-guidance/notices/15-09 | 2026-08-09 | Regulator guidance | WebFetch, full | Only disablement language: "providing mechanisms by which the firm may quickly disable the algorithm or supporting platform with a minimal number of steps." **NEGATIVE FINDING (load-bearing):** contains NO guidance on open positions after disablement, no risk-reducing-vs-risk-increasing distinction, no unwind responsibility. The literature silently assumes a human desk inherits the risk. |
| 6 | https://www.businesstoday.in/opinion/columns/story/trapped-at-the-terminal-where-even-your-stop-loss-wont-work-536134-2026-06-10 | 2026-08-09 | Incident column (2026-06-10) | WebFetch, full | MCX outages 2024-02, 2025-07, 2025-10-28: "Stop-loss orders are set, margins are placed, and the trade is active. However, **once the terminal goes blank, all protective measures become worthless**." "A retail participant's stop-loss was triggered twice while the exchange remained dark." Traders absorbed 100% of the loss. |

### Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://www.cmegroup.com/tools-information/webhelp/globex-credit-controls/Content/Kill-Switch.html | exchange doc | WebFetch timeout; curl returned "IP address is blocked due to suspected web scraping". Substituted by the CME Confluence wiki (#4 above). |
| https://support.ninjatrader.com/s/article/Liquidation-Only-Status-Due-to-Risk-Settings | broker doc | Salesforce SPA — WebFetch returned only a CSS-error loading shell. Snippet: liquidate-only means "you will not be able to enter into new positions until the start of the next trading session". |
| https://vendor-support.ninjatrader.com/s/article/Liquidation-Only-Status-Risk-Tradovate | broker doc | Same SPA shell failure. |
| https://www.sec.gov/rules/final/2010/34-63241.pdf (Rule 15c3-5) | SEC rule | HTTP 403; the `.htm` variant 404s (386 bytes). NOT read — see gate note. |
| https://www.sec.gov/litigation/admin/2013/34-70694.pdf (Knight Capital) | SEC order | HTTP 403, then "SEC.gov Request Rate Threshold Exceeded". **NOT read — the Knight Capital sub-question is therefore answered only indirectly.** |
| https://www.federalregister.gov/documents/2015/12/17/2015-30533/regulation-automated-trading | CFTC Reg AT | 302 cross-host redirect to `unblock.federalregister.gov`; not followed. |
| https://www.eurex.com/ex-en/support/initiatives/risk-protection | exchange | HTTP 404. |
| https://www.interactivebrokers.com/campus/ibkr-quant-news/stop-loss-orders-and-trading-halts/ | broker | HTTP 403. |
| https://www.finra.org/rules-guidance/notices/16-21 | regulator | Fetched but OFF-TOPIC (Series 57 registration); pointed to 15-09, which was then read in full. |
| https://work-club.com/liquidation-only-stock-meaning/ | community | Community tier. Snippet: "you can reduce or exit the position, but you cannot open or add to it." |
| https://optionx.trade/blogs/what-is-kill-switch-trading-platform | community | Community tier. |
| https://positioned.app/traders-glossary/kill-switch | community | Glossary tier. |
| https://www.mexc.com/.../liquidation-and-risk-limit-360044646391 | crypto venue | Community/venue tier. |
| https://www.ice.com/publicdocs/circulars/17191%20attach.pdf | exchange circular | Redundant with RTS 6 primary. |
| https://service.betterregulation.com/document/273576 | regulation mirror | Superseded by EUR-Lex primary (#1). |
| https://www.hoganlovells.com/.../mifid-ii-algorithmic-trading-29122016.pdf | law firm | Secondary to #1. |
| https://www.kroll.com/en/publications/financial-compliance-regulation/algorithmic-trading-under-mifid-ii | consulting | Secondary to #1. |
| https://www.fia.org/fia/articles/mifid-ii-rts-published-eu-official-journal | industry body | Announcement only. |
| http://www.mfaalts.org/.../ESMA-QA-on-MiFID-II-and-MiFIR-Market-Structures-Topics.pdf | ESMA Q&A mirror | Superseded by #2. |
| https://www.emissions-euets.com/internal-electricity-market-glossary/1133-algorithmic-trading | glossary | Community tier. |
| https://raseedinvest.com/en/learn/stock-halts-explained-... | broker education | Community tier. |
| https://www.business-standard.com/markets/news/bse-to-discontinue-stop-loss-market-orders-from-oct-9-...-123092200889_1.html | press | Recency context: BSE discontinued SL-M orders after a freak trade. |
| https://www.forex.com/en-us/help-and-support/trading-halts/ | broker | Community/vendor tier. |

**Totals: 29 unique URLs collected; 6 read in full; 23 snippet-only or fetch-failed.**

### Recency scan (last 2 years, 2024-2026) — PERFORMED

Searched explicitly for 2024-2026 material on protective exits during halts/outages. **Result: 3 new findings that COMPLEMENT (do not supersede) the canonical sources.**

1. **ESMA Supervisory Briefing on Algorithmic Trading, 2026-02-26** (read in full). Post-dates RTS 6 by nine years and is the current supervisory expectation. Two things matter here: it makes the remedial ladder explicit (¶93 — cancellation vs withdrawal-from-market vs kill functionality are three DIFFERENT actions), and it classifies **stop-loss triggers as algorithmic trading** (¶11(5)) — directly relevant to whether synthesizing a stop during a halt is a "new decision".
2. **NYSE Pillar Risk Controls v4.7, 2026-03-19** (read in full). The current spec still ships "Block only — **accept cancels**" as a first-class breach action, confirming the risk-reducing carve-out is live practice in 2026, not legacy.
3. **MCX outage series (2024-02, 2025-07, 2025-10-28), reported 2026-06-10** (read in full). The freshest documented instance of exactly this failure mode: exposure retained, protective logic frozen, loss borne by the position holder.

No source found in the window ARGUES FOR freezing protective exits while retaining exposure. Nearest thing to a counter-argument is CME's kill switch (#4), which is a full block with no exit carve-out — addressed in "Consensus vs debate" below.

---

## Key findings

1. **"Kill" is scoped to ORDERS, not to POSITIONS.** "An investment firm shall be able to cancel immediately, as an emergency measure, any or all of its **unexecuted orders**" (RTS 6 Art. 12, EUR-Lex, accessed 2026-08-09). Nothing in the kill-functionality mandate stops a firm from managing risk it already holds; a separate provision (Art. 14(2)(g)) requires arrangements "to manage outstanding orders **and positions**".
2. **Halting is a graduated ladder, not a binary.** ESMA ¶93 (2026-02): remedial actions "may also lead to the cancellation of orders, a withdrawal from the market **or** the use of the kill functionality." Three distinct rungs.
3. **Risk-reducing messages are explicitly carved out of a blocked state in live exchange practice.** NYSE Pillar v4.7 §6.7 breach action: "**Block only — accept cancels; reject order, modify, cancel/replace**." The blocked entity may still act to reduce risk.
4. **Liquidation-only IS a named standard state, but at the BROKER/PROP layer, not the exchange-gateway layer.** Tradovate/NinjaTrader "Liquidate-Only status due to Risk Settings" (snippet): "you will not be able to enter into new positions until the start of the next trading session." CME's exchange kill switch has no such mode (#4) — because an exchange gateway cannot manage your book; it can only stop your flow.
5. **The exchange explicitly disclaims being your risk system.** NYSE Pillar §2: "The risk controls are **not designed to be a firm's sole means of risk control** and should not be relied upon as such." pyfinagent's kill switch is the FIRM-side risk system — the layer that IS supposed to keep managing positions.
6. **Regulatory guidance is SILENT on post-kill position management, and that silence assumes a human.** FINRA 15-09 mentions only "mechanisms by which the firm may quickly disable the algorithm... with a minimal number of steps" and says nothing about the resulting exposure. The assumption is a staffed desk. pyfinagent runs unattended for weeks (away-ops).
7. **The failure mode is documented and recent.** MCX 2024-2025: "once the terminal goes blank, all protective measures become worthless"; "a retail participant's stop-loss was triggered twice while the exchange remained dark" (BusinessToday, 2026-06-10). Losses compounded precisely because exposure persisted while protection did not.
8. **A stop-loss trigger is itself an algorithmic trading decision under EU law.** ESMA ¶11(5) lists "Adjusting trading behaviour based on risk metrics (e.g. VaR, exposure limits, stop-loss triggers)" as algorithmic trading. Consequence for Q3: *setting* a stop level is a trading decision; *enforcing an already-set* level is executing a prior decision. That distinction is the whole answer to the backfill question.

---

## Consensus vs debate

**Consensus (5 of 6 sources):** a risk halt suppresses *risk-increasing* order flow, and risk-*reducing* action is either explicitly permitted (NYSE), explicitly a separate rung (ESMA), or simply out of scope of the kill (RTS 6). No source treats "stop managing what you hold" as the intended meaning of a halt.

**The dissent — and it is real:** CME's Globex Kill Switch blocks **all** order entry with no exit carve-out (#4). Two things reconcile it: (a) CME's switch protects the MARKET from a malfunctioning participant, where an uncontrolled algo's "protective" orders are exactly what you must stop; (b) it is a *gateway* control, and the firm retains other routes (voice desk, GCC, other venues). pyfinagent's switch has neither property — it protects the operator from his own book, and it is the ONLY route. So the CME analogy does not transfer.

**Genuine open question the literature does not settle:** whether an *unattended* system should self-execute protective exits during a halt, or halt-and-page. The literature assumes a human; pyfinagent has none for weeks at a time. That is the operator's call and the contract should surface it as such.

---

## Answers to the five questions

**Q1 — Standard semantics.** Yes: "liquidation-only" / "close-only" / "reduce-only" is a named, standard state, distinct from a full halt, and it lives at the broker/prop-risk layer (Tradovate/NinjaTrader), with the same *shape* visible at NYSE ("Block only — accept cancels"). Regulation scopes "kill" to unexecuted orders (RTS 6 Art. 12) and treats withdrawal-from-market as a separate action (ESMA ¶93). CME is the pure-block counter-example, and is a gateway control, not a firm risk system. pyfinagent's `kill_switch.py:14` "Pause = halt new entries; existing positions kept" is *already* the liquidation-only semantic in prose — the code just fails to deliver the second half.

**Q2 — Failure literature.** Yes, documented and recent. MCX 2024-02 / 2025-07 / 2025-10-28: protective orders became "worthless" while positions stayed live and prices moved; losses fell entirely on the position holder (BusinessToday 2026-06-10). The Knight Capital 2012 order could NOT be fetched (SEC 403 / rate-limit) — I am not citing it. FINRA 15-09's *silence* on post-disable position management is itself evidence: the regime presumes a human desk absorbs the residual risk, an assumption pyfinagent violates.

**Q3 — Invariants, and the backfill sub-question.**
Invariants that must hold if exits continue during a halt:
- **No BUYs / no position increases.** Already independently enforced at the choke point: `paper_trader.py:282-294` refuses every BUY when paused, and FAILS CLOSED (:225-233). *But* on the `blocked` path the switch is not paused and baselines are present, so that gate returns None — blocked-path BUY suppression comes ONLY from the `return summary` at :1437. **Therefore any halt-branch code must still return before Step 6.**
- **No re-entry / no new sizing decisions** — `decide_trades` stays behind the return.
- **Idempotency** — `execute_sell` is naturally idempotent (`get_position` returns None if already sold; `autonomous_loop.py:1442-1443`).
- **Do not clear the halt** — no `state.resume()`, no mutation of `_paused_at`; the halt must remain exactly as latching (breach/paused) or non-latching (blocked) as it is.
- **Stay loud** — do not disturb `summary["status"]="halted_kill_switch"` (:1425) or `summary["halt_reason"]` (:1426); phase-85.4 depends on them.
- **Exit-only means SELL-only** — the pass must reach `execute_sell` and nothing else.

**Is `backfill_missing_stops` a NEW risk decision or a protective default? It is a NEW RISK DECISION.** Three reasons:
(i) It *synthesizes a price that never existed*: `stop = avg_entry_price * (1 - 8%)` (`paper_trader.py:~980`). On a book that has already breached a daily-loss or trailing-DD limit, entry prices are stale and a synthesized stop can land ABOVE the current mark — so backfill-then-check converts "this position has no stop" into "sell this position at market, now". On the `blocked` and `paused` paths that is a **flatten by side effect**, on exactly the branches the design deliberately does not flatten (`paper_trader.py:1468`), and it would violate the caller's constraint against introducing a flatten on the blocked path.
(ii) ESMA ¶11(5) classifies stop-loss triggers as algorithmic trading — *choosing* the level is the algorithm determining an order parameter. Enforcing a level chosen earlier is executing a prior decision; inventing one now is a fresh decision made while the system is supposed to be halted.
(iii) It is a BQ WRITE (`bq.save_paper_position`) that mutates durable risk state during a halt.
**Recommendation: split them.** During a halt, run `check_stop_losses` (enforce pre-existing commitments) and NOT `backfill_missing_stops`. A NULL-stop position during a halt is an operator-visible gap that should be *reported* (it already can be — `backfill_missing_stops` is reachable manually via `scripts/maintenance/backfill_stops.py:75`), not silently invented. The existing coupling at :1458→:1471 is correct for a normal cycle and wrong for a halted one.

**Q4 — The three options.**
- **(a) Move Step 5.6 above the Step 5.5 halt.** *For:* simple; protection becomes unconditional. *Against:* it changes the HEALTHY path too (stop enforcement would run before the kill-switch evaluation), so on a `breach` cycle you would stop-out and *then* `flatten_all` — two exit paths, duplicated fee/learn-loop events, and a race over the same positions. It also runs the backfill (a new risk decision, per Q3) on every path including breach. Blast radius far exceeds the defect. **Not supported.**
- **(b) A stop-loss-only pass inside the halt branch, before the return.** *For:* this is precisely "Block only — accept cancels" (NYSE §6.7) and the ESMA ¶93 middle rung, implemented at the right layer. The healthy path stays byte-identical. The no-BUY invariant is independently guaranteed at `execute_buy:282`. *Against / must be handled:* scope it to `blocked` and `paused` (on `breach`, `flatten_all` has already run at :1470, so the pass is a no-op at best); exclude `backfill_missing_stops`; do not append to `summary["steps"]` (collision with :298/:374); keep `return summary` last. **This is the option the literature supports.**
- **(c) Accept and document.** *For:* zero code risk; a literal reading of "existing positions kept". *Against:* defensible only with a human watching. MCX is the empirical refutation; FINRA 15-09 presumes a desk; the `blocked` path is per-cycle and NON-latching so it can recur indefinitely. **And (c) is not free:** if chosen, the operator's instruction would be "on any halted cycle, manually check stops and exit breached positions" — and **no tool exists to do that** (`scripts/maintenance/backfill_stops.py` backfills only; nothing else calls `check_stop_losses`). Choosing (c) therefore *creates* work: build the operator-facing stop-check tool plus a halted-with-open-positions alert. **Not supported as-is.**

**Q5 — Step 5.4 ordering hazard: DIFFERENT defect class, and milder.**
- 5.6 is an **omission** — a *protective* action withheld on halted cycles.
- 5.4 is a **commission** — a *discretionary* profit-taking partial close placed on a cycle that is about to halt, because `check_scale_out_fires` (:1381) runs before `check_and_enforce_kill_switch` (:1400).
- Mitigants, measured: `paper_scale_out_enabled` defaults **False** (`settings.py:35`), so it is DARK today (zero live impact); a scale-out SELL is risk-reducing so it cannot violate "no new BUYs"; and it fails open (:1390-1393).
- Residual real risk once the flag is ever enabled: on a **breach** cycle, 5.4 could place a partial close and 5.5 then `flatten_all` the remainder — two exits, two fee events, and `scale_out_levels_hit` written for a position that no longer exists.
- **Recommendation: do NOT bundle it into 36.17.** Queue it as its own research-gated masterplan step (moving 5.4 below 5.5 changes the documented MTM-freshness ordering at :1377-1379 and needs its own analysis).

---

## Application to pyfinagent (mapping)

| External finding | pyfinagent anchor | Implication |
|---|---|---|
| NYSE "Block only — accept cancels" | halt branch `autonomous_loop.py:1403-1437` | The halt should block risk-increasing flow and still permit the risk-reducing pass. |
| RTS 6 Art. 12 (kill = unexecuted orders) | `kill_switch.py:14` "existing positions kept" | The docstring already states the liquidation-only semantic; the code does not implement its second half. |
| ESMA ¶93 ladder | `cycle_halt_reason` :139-165 | Three halt reasons already exist; they can carry different remedial depth (breach already flattened; blocked/paused should still enforce stops). |
| ESMA ¶11(5) stop-loss triggers = algo trading | `backfill_missing_stops` :926 vs `check_stop_losses` :797 | Synthesizing a level is a new decision; enforcing one is not. Split them in the halt branch. |
| NYSE §2 "not the firm's sole means of risk control" | `autonomous_loop.py:1471` is the ONLY caller | pyfinagent has no second layer — the cycle IS the sole means, which raises severity. |
| MCX: protection frozen, exposure retained | `paper_trader.py:1468` (no flatten on blocked/paused) | Exactly the documented failure shape, currently reproduced. |

### Severity verdict
**High.** `check_stop_losses` has exactly ONE production caller (`autonomous_loop.py:1471`) — no API route, no scheduler job, no MCP tool. On `blocked` or `paused` cycles the book holds full exposure with zero stop enforcement and zero backfill, indefinitely, and `blocked` is non-latching so it can repeat every cycle without an operator resume ever being required.

---

## Pitfalls (from literature + internal)

1. Do not read CME's all-block kill switch as authority for freezing exits — it is a *venue gateway* protecting the market, not a firm risk system, and the firm retains other routes.
2. Do not let the exit pass become a flatten: excluding `backfill_missing_stops` is what keeps option (b) inside the caller's "no flatten on the blocked path" constraint.
3. Do not append to `summary["steps"]` inside the halt branch — measured collision with `test_phase_36_12...:298` and `:374`.
4. Do not trust the existing Step 5.6 suite as coverage: it never drives `run_daily_cycle` (`test_autonomous_loop_step_5_6.py:44-61`).
5. Any new test in this area MUST carry the `captured_alerts` fixture, or it pages the operator's real Slack (measured, `test_phase_36_12...:70-77`), and must keep `cycle_health.get_log` stubbed or it dirties git-tracked handoff files.
6. `check_stop_losses` silently no-ops on a NULL stop or a 0/None price (`if stop and current`, :804) — a halted book with NULL stops gets nothing even with option (b). That is the argument for *alerting* on NULL stops during a halt rather than backfilling them.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL (6: EUR-Lex RTS 6; ESMA 2026 briefing; NYSE Pillar v4.7; CME wiki; FINRA 15-09; BusinessToday MCX)
- [x] 10+ unique URLs total (29)
- [x] Recency scan (2024-2026) performed + reported (3 findings)
- [x] Full pages/PDF text read, not abstracts (2 PDFs via curl+pdfplumber, 40pp + 19pp)
- [x] file:line anchors for every internal claim (all re-derived at source)

Soft checks:
- [x] Internal exploration covered every module in the caller's scope
- [x] Contradictions noted (CME vs NYSE/ESMA — see Consensus vs debate)
- [x] Claims cited per-claim
- [ ] GAP, disclosed: SEC Rule 15c3-5 adopting release and the Knight Capital administrative order were both blocked (HTTP 403 / SEC rate-limit). The 15c3-5 "hard block" reference in `kill_switch.py:13` is therefore NOT corroborated against the primary rule text in this brief. CFTC Reg AT also unread (cross-host redirect). Q2's Knight sub-question is answered from MCX + the FINRA-silence finding instead.
- [ ] NOTE: tool-call budget for `moderate` (<=18) was exceeded (~32 calls). The >=5 read-in-full floor plus the caller's explicit test-enumeration scope required it; disclosed rather than trading away the floor.

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 23,
  "urls_collected": 29,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 3,
    "dry": false
  },
  "brief_path": "handoff/current/research_brief_36.17.md",
  "gate_passed": true
}
```
