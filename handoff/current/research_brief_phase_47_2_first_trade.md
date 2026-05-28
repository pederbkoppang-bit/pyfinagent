# Research Brief — phase-47.2: First autonomous trade end-to-end

**Step:** phase-47.2 — "First autonomous trade end-to-end (empty new_candidates set + sod_date roll)"
**Tier:** moderate-to-complex (touches the trade-decision path)
**Researcher session:** 2026-05-28
**Status:** IN PROGRESS (incremental write)

---

## Objective

Validated root cause + SMALLEST SAFE fix to make the autonomous paper-trading
cycle execute >=1 legitimate trade end-to-end, without disabling any safety
control (kill-switch, stop-loss, max-position, sector caps).

Operator's #1 concern: "we didn't have any trades today" — the loop records
`n_trades:0` every cycle.

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/services/autonomous_loop.py` | 124-160 | `run_daily_cycle` entry; `dry_run=True` SHORT-CIRCUITS (no funnel) | dry_run is a no-op smoketest — useless for funnel diagnosis |
| `backend/services/autonomous_loop.py` | 324-329 | `screen_universe` (live yfinance, 6mo) | working — returns candidates |
| `backend/services/autonomous_loop.py` | 578 | rank → `top_n=paper_screen_top_n` (=10) | working |
| `backend/services/autonomous_loop.py` | 655-658 | `held_tickers` / `new_candidates` subtraction / `analyze_tickers` | NOT empty — 4 new flow through per cycle (log-confirmed) |
| `backend/services/autonomous_loop.py` | 751-755 | `candidate_analyses` gather | working — analyses produced |
| `backend/services/autonomous_loop.py` | 803-808 | `check_and_enforce_kill_switch` + pause gate | WIRED; sod_date roll lives here |
| `backend/services/autonomous_loop.py` | 943-950 | `decide_trades` call | THE COLLAPSE POINT |
| `backend/services/portfolio_manager.py` | 43-47 | `_BUY_RECS={BUY,STRONG_BUY}`, `_SELL_RECS`, `_DOWNGRADE_RECS` | — |
| `backend/services/portfolio_manager.py` | 241-271 | buy-loop: position-cap break (242), sector-COUNT cap (261-271) | **SECTOR-COUNT CAP blocks 100% of buys** |
| `backend/services/portfolio_manager.py` | 286-303 | per-sector NAV-pct cap (=30%) | secondary blocker |
| `backend/services/portfolio_manager.py` | 357-372 | swap-path guard + call | swap_enabled=True but produces NO orders |
| `backend/services/portfolio_manager.py` | 389-563 | `_compute_swap_candidates` (delta>=25%, max 2/cycle) | structurally cannot clear a 5-over book; also no log output |
| `backend/services/paper_trader.py` | 935-967 | `check_and_enforce_kill_switch` — sod_date idempotent roll at 955-956 | **ALREADY WIRED** (contra the prompt hypothesis) |
| `backend/services/kill_switch.py` | 196-210 | `update_sod_nav(nav, date)` | sod_date set here; NO `reset_sod_if_needed` method exists anywhere |

## Live dry-run ground truth

**NOTE:** `POST /api/paper-trading/run-now -d '{"dry_run": true}'` is a NO-OP
(`autonomous_loop.py:156-160` short-circuits — stamps `_last_run`, returns
`{"status":"ok","dry_run":true}`, runs zero funnel work). It does NOT reveal
stage counts. Ground truth instead came from `backend.log` (307 MB) cycle
traces + live API state.

### Funnel stage counts (from backend.log, multiple recent cycles)
| Stage | Count | Source |
|-------|-------|--------|
| Universe screened | full S&P500 | `Step 1 -- Screening universe` |
| Meta-scorer ranked candidates | **10** (top conviction=10) | `Meta-scorer ranked 10 candidates` |
| `new_candidates` (after held subtraction) | **4** | `Step 3 -- Analyzing 4 new + 9 re-evals` |
| `candidate_analyses` (BUY-rec'd) | >=3 reach buy_candidate w/ APPROVE | `buy_candidate risk_judge decision=...` for STX/CIEN/AMD/QCOM |
| Orders built by `decide_trades` | **0 sells, 0 buys** | `Trade decisions: 0 sells, 0 buys` |
| Trades executed | **0** | `Step 7 -- Executing 0 trades` |

### The exact rejection lines (smoking gun)
```
02:24:18 buy_candidate risk_judge decision=APPROVE_REDUCED ticker=STX  position_pct=2.5 final_score=7.0
02:24:18 buy_candidate risk_judge decision=APPROVE_REDUCED ticker=CIEN position_pct=2.5 final_score=7.0
02:24:18 buy_candidate risk_judge decision=APPROVE_HEDGED  ticker=AMD  position_pct=2.5 final_score=8.0
02:24:18 Skipping BUY AMD:  sector Technology at cap (10/2)
02:24:18 Skipping BUY STX:  sector Technology at cap (10/2)
02:24:18 Skipping BUY CIEN: sector Technology at cap (10/2)
18:59:28 Skipping BUY QCOM: sector Technology at cap (9/2)
20:36:00 Skipping BUY QCOM: sector Technology at cap (8/2)
02:24:18 Trade decisions: 0 sells, 0 buys
```

### Live state (curl, 2026-05-28)
- NAV **$23,654**, cash **$13,773 (58% — NOT capital constrained)**, 8 positions.
- `paper_max_positions=20` (effective, .env override) → **12 open slots** — book NOT saturated by count.
- **Held book sectors: Technology x7 (MU, KEYS, ON, INTC, DELL, SNDK, WDC) + Industrials x1 (GEV).**
- Kill switch: `paused:false`, no breach (daily -0.47%, trailing -0.62%). **Safety is NOT the blocker.**
- `sod_date="2026-05-27"` (yesterday) — rolls via `check_and_enforce_kill_switch` (already wired); today's cycle simply hasn't run yet (`last_run:null` post-restart).

### Effective settings (.venv get_settings, respects .env)
`paper_max_positions=20`, `paper_screen_top_n=10`, `paper_analyze_top_n=5`,
`paper_reeval_frequency_days=3`, **`paper_max_per_sector=2`**,
`paper_max_per_sector_nav_pct=30.0`, `paper_swap_enabled=True`,
`paper_swap_min_delta_pct=25.0`, `paper_swap_max_per_cycle=2`,
`paper_max_daily_cost_usd=2.0`, `lite_mode=False`, `meta_scorer_enabled=True`.

### Why the swap path (the intended rescue) does not fire
1. The most recent `Skipping BUY` log lines use the OLD message text
   (no "-- queued for swap check" suffix added at `portfolio_manager.py:266`),
   confirming the running cycles in the log predate commit `69c710ec`
   (2026-05-26 22:35, "cycle 1 position-swap framework"). The swap code is
   in the repo but had not been exercised by a fresh cycle in these traces.
2. Even when it runs it is structurally insufficient: Technology is **5 over**
   its count cap (7 held vs cap 2); `paper_swap_max_per_cycle=2` can displace
   at most 2 holdings/cycle, and only when a candidate beats the weakest
   same-sector holding by >=25% relative `final_score`. The held positions'
   re-eval `final_score` is frequently absent on the lite path (falls to the
   0.0 sentinel), and the NAV-pct reduction-clause guard adds another gate.
   The swap path is a partial palliative, not a fix for a structurally
   single-sector book + single-sector candidate stream.

### SCALE-MISMATCH note (secondary bug, worth flagging to Main)
`final_score` in the live `buy_candidate` logs is on a **0-10 scale** (7.0, 8.0),
but `_compute_swap_candidates` comments (`portfolio_manager.py:478-480`) assert
"final_score lives in [0,1]". The swap delta math still works numerically
(relative %), but the 0.01 epsilon denominator + the [0,1] assumption are
inconsistent with the actual 0-10 scale — a latent correctness smell, not the
primary cause.

## External research

### Read in full (>=5 — counts toward the gate)
| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://arxiv.org/html/2603.20319v1 | 2026-05-28 | paper (peer-track preprint) | WebFetch HTML | Implementation Risk in Portfolio Backtesting (Yin, Miki, Lesnichenko, Gural 2026). Backtrader **fill-ordering bug**: "triggers an immediate margin rejection" when cash temporarily goes negative during sequential order processing — "rejecting buy orders whose funding depends on later sell proceeds. This infrastructure flaw prevents valid rebalances from executing entirely." Directly analogous to a sell-first-then-buy ordering failure. 5 failure modes; recommends >=2 independent validators + cost-model audit. |
| https://arxiv.org/html/2601.05428v1 | 2026-05-28 | paper | WebFetch HTML | Dynamic Inclusion & Bounded Multi-Factor Tilts (Garrone, Jan 2026). Concentration handled by **bounded multiplicative tilt around equal-weight** `m_i=clip(1+λz_i, m_min, m_max)` — "factor signals influence allocations without allowing any asset to dominate." Crucially: bounds **trim/cap**, they do NOT hard-block eligibility — "This prevents starvation: assets meeting liquidity thresholds remain eligible; bounds apply symmetrically." Semi-annual rebalance, "moderate turnover, consistently below naive equal-weight." |
| https://en.wikipedia.org/wiki/Hierarchical_Risk_Parity | 2026-05-28 | doc (canonical / López de Prado 2016) | WebFetch | HRP allocates "inversely proportional to cluster variance"; "ensures that assets only compete with similar assets for representation" — prevents one sector dominating WITHOUT a hard count cap. CLA put 92.66% in top-5; HRP 62.57% — diversification by *graduated weighting*, not binary exclusion. +31.3% OOS Sharpe vs CLA. |
| https://www.tradingheroes.com/mt4-strategy-tester-fix-zero-trades/ | 2026-05-28 | blog (practitioner) | WebFetch | Zero-trades-despite-signals checklist. Most relevant cause: **"Incomplete Automation — some EAs only handle entries OR exits, not both."** Maps to: system generates SELLs only via re-eval downgrade/stop, never frees a slot to let a blocked BUY in → a one-sided pipeline that cannot turn over. |
| https://www.guardfolio.ai/blog/portfolio-risks-2026 | 2026-05-28 | industry (2026) | WebFetch | "Limit sector exposure to 25% maximum." On breach: **"Rebalance when positions exceed thresholds, not on a calendar"** — i.e., TRIM/ROTATE the overweight sector. Notably NOT "block new buys": "rather than restricting new capital allocation." This is the consensus remediation the current code inverts. |

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://arxiv.org/pdf/1509.08110 | paper (Performance v. Turnover, 4000 alphas) | Snippet sufficed for turnover-floor point; PDF not equation-critical here |
| https://arxiv.org/pdf/1406.0044 | paper (Can Turnover Go to Zero?) | "if tradable instruments are finite, turnover cannot go to zero" — corroborates need for rotation; snippet sufficient |
| https://dev.to/qcautomation/why-your-paper-trading-backtests-are-lying-to-you-... | blog | Optimistic-fill defaults; lower tier than the arXiv implementation-risk paper |
| https://www.wisdomtree.com/.../the-2026-market-a-world-of-constant-rotation | industry | "constant rotation" 2026 thesis; corroborates rotation cadence; snippet only |
| https://www.ainvest.com/news/assessing-2026-sector-rotation-... | industry | 2026 sector-rotation context; snippet only |
| https://privatebanking.one/portfolio-rebalancing-2026/ | industry | 25% sector cap + 5% stop corroboration; snippet only |
| https://www.quantresearch.org/Lectures.htm | author (López de Prado lectures) | Canonical year-less hit; index page, not a single readable doc |
| https://arxiv.org/pdf/2508.11856 | paper (RL-BHRP) | Adjacent HRP extension; not needed for this fix |

### Search-query variants run (3-variant discipline)
1. **Current-year frontier (2026):** "portfolio rebalancing sector concentration limit prevents new entries sell-to-rotate quantitative trading 2026"
2. **Last-2-year window:** "paper trading backtest live execution silently zero fills no trades debugging candidate funnel" (surfaced 2026 arXiv implementation-risk + 2025/2026 practitioner)
3. **Year-less canonical:** "Lopez de Prado portfolio construction sector concentration constraint rebalancing turnover advances financial machine learning" + "trading system candidate universe minus current holdings anti-pattern position rotation rule turnover starvation" (surfaced HRP 2016, AFML, the turnover-floor arXiv classics 1404.5050 / 1406.0044 / 1509.08110)

## Recency scan (2024-2026)

Searched 2026 + 2025 windows on backtest-to-live execution defects, sector-cap
rotation, and candidate-funnel starvation. **2 new findings supersede/complement
the canonical sources:**
1. **arXiv:2603.20319 (2026)** — first formal quantification of "implementation
   risk": engine-level fill-ordering / cost-model bugs that silently zero out
   or reject fills. Directly elevates the diagnostic discipline this step needs
   (audit the order-build path, not just the signal). Newer than and complements
   López de Prado's backtest-overfitting canon.
2. **arXiv:2601.05428 (Jan 2026)** — modern statement of the *bounded-tilt-not-
   hard-block* principle for concentration: "prevents starvation … bounds apply
   symmetrically." This is the academically-current articulation of why a hard
   per-sector COUNT cap with no rotation valve is the wrong primitive.
No 2024-2026 finding contradicts the recommended fix; the consensus (graduated
trim/rotate over binary exclusion) is stable from López de Prado 2016 through 2026.

## Validated root cause

**The prompt's diagnostic hypothesis is WRONG on two counts, and the real cause
is proven by live logs:**

1. **`new_candidates` is NOT empty.** Every cycle analyzes "4 new + 9 re-evals"
   (`autonomous_loop.py:658`). The subtraction at :657 leaves a healthy set.
2. **`sod_date` IS already wired.** `check_and_enforce_kill_switch`
   (`paper_trader.py:935`, called at `autonomous_loop.py:804` in Step 6) performs
   an idempotent daily SOD roll at lines 955-956. There is **no
   `reset_sod_if_needed` symbol anywhere** in the backend; the function the
   prompt references does not exist. The live API confirms `sod_date` rolls
   (currently "2026-05-27"); it only looks stale because today's cycle hasn't
   run post-restart (`last_run:null`). **No sod_date wiring fix is needed** —
   the only nuance is that the roll happens in Step 6, AFTER decide_trades, so
   the *daily-loss* anchor for the CURRENT cycle uses yesterday's SOD. That is
   harmless for the zero-trades problem (kill switch is not paused) but is noted
   below as an optional ordering hardening.

**ACTUAL ROOT CAUSE — `decide_trades` blocks 100% of BUYs on the per-sector
COUNT cap.** Live log (`portfolio_manager.py:264-271`):
```
Skipping BUY AMD:  sector Technology at cap (10/2)
Skipping BUY STX:  sector Technology at cap (10/2)
Skipping BUY CIEN: sector Technology at cap (10/2)
Skipping BUY QCOM: sector Technology at cap (9/2)
```
The held book is **7 Technology + 1 Industrials**, and the screener's candidate
stream is **entirely Technology semis** (AMD, STX, CIEN, QCOM, MU, WDC, …).
With `paper_max_per_sector=2`, Technology is **3.5x over cap**, so every
BUY candidate — all of which pass the LLM gate AND the Risk Judge
(APPROVE_REDUCED/APPROVE_HEDGED, final_score 7-8) — is sector-blocked. Result:
`Trade decisions: 0 sells, 0 buys` every cycle. There are 12 free position
slots and 58% cash; the book is starved purely by the COUNT cap.

**Why the intended rescue (swap path) does not save it:**
- The swap path (`paper_swap_enabled=True`, commit `69c710ec` 2026-05-26) was
  added AFTER the log traces shown (their "Skipping BUY" lines lack the new
  "-- queued for swap check" suffix at :266), so it had not run in those cycles.
- Even when it runs it is structurally insufficient: it does at most
  `paper_swap_max_per_cycle=2` swaps and only when a candidate beats the weakest
  *same-sector* holding by `paper_swap_min_delta_pct=25%` relative `final_score`.
  Held positions frequently lack a fresh re-eval `final_score` on the lite path
  (sentinel 0.0), and the NAV-pct reduction-clause adds another gate. It cannot
  clear a sector that is 5 positions over cap, and it never *adds net exposure*
  to a sector — it only 1-for-1 rotates within Technology. The first legitimate
  trade can come from it, but it is fragile and unproven (zero swap log lines).

**Upstream contributory cause (the deeper "why"):** the screener has no sector
diversification — it surfaces the same single hot sector (semis/Tech) every day,
so the candidate stream and the held book are both ~90% Technology. The
per-sector cap was doing exactly its job (preventing a 100%-Tech book); the
defect is that there is **no rotation valve and no cross-sector candidate
supply**, so "cap reached" degrades to "trade nothing" — the precise anti-pattern
the literature warns against (Guardfolio 2026: rotate, don't block;
arXiv:2601.05428: bounds must not starve; arXiv:1406.0044: finite instruments →
turnover must stay positive).

## Smallest-safe fix

**Goal: >=1 legitimate trade this cycle WITHOUT disabling any safety control.**
Ranked smallest-first. Recommend **Fix A as the primary**, with Fix B as the
durable follow-up.

### Fix A (SMALLEST SAFE — verify the swap path actually fires; one-cycle proof)
The rotation machinery already exists and is the literature-endorsed primitive
(trim/rotate over block). It simply has **never been exercised by a live cycle**.
Smallest action:
1. Restart the backend so the running process carries commit `69c710ec`'s swap
   code (the log proves the old message text is live → stale process).
2. Trigger a REAL cycle: `POST /api/paper-trading/run-now` (NO dry_run — dry_run
   short-circuits). Watch for `Swap fired (n/2): SELL … -> BUY …` lines.
3. If the swap fires, you get >=1 trade with **all caps intact** (sector count
   unchanged, NAV-pct rechecked, sell-first-then-buy preserved). Done.

**Risk:** swap may still emit 0 if (a) no candidate beats the weakest Tech
holding by 25%, or (b) holding `final_score` is the 0.0 sentinel and the
NAV-pct guard blocks the add. If 0 swaps after a real cycle, apply Fix B.

### Fix B (durable, still safe — give the cap a rotation valve / minor loosening)
The single safest *parameter* change that guarantees turnover without removing a
control: **make the swap path robust + lower its activation friction**, OR allow
a controlled count-cap headroom. Two sub-options, pick one:
- **B1 (preferred, no cap change):** Fix the swap path's holding-score
  resolution so a held position with no fresh re-eval uses a real prior score
  instead of the 0.0 sentinel, and lower `paper_swap_min_delta_pct` from 25 → 10
  (a 10% relative conviction edge is a defensible rotation trigger;
  arXiv:2601.05428 supports graduated tilts).
  **CAVEAT (verified):** `paper_positions` does NOT store `final_score` (schema
  check — only `analysis_results` carries it, keyed by ticker+analysis_date;
  `bigquery_client.py:41,142,264`). So "use the entry score" means joining the
  held ticker back to its latest `analysis_results.final_score`, OR simply
  treating the 0.0-sentinel holding as the weakest (which it already is) so ANY
  credible candidate (score 7-8) clears even a 10% delta. The simplest correct
  B1 is therefore: ensure `reeval_tickers` analyses populate `final_score` so
  `holding_lookup` has a real number, and lower the delta threshold. If the
  re-eval path genuinely cannot yield a holding score, the 0.0 sentinel ALREADY
  makes `delta_pct` huge → swap fires; in that case the real blocker is the
  NAV-pct reduction-clause or `max_per_cycle`, which Fix A's live run will
  reveal. Main must confirm WHICH via a real cycle's swap log lines before
  picking the exact one-line change.
  This keeps the sector COUNT constant (1-for-1), keeps NAV-pct + stop-loss +
  kill-switch fully intact, and produces the first trade by rotating the weakest
  Tech holding into the highest-conviction Tech candidate.
- **B2 (only if B1 still yields 0):** raise `paper_max_per_sector` 2 → 3. This
  is a *deliberate, bounded* concentration loosening (still well inside the
  NAV-pct 30% cap, which remains the real risk ceiling per Guardfolio's 25-30%
  guidance). It lets exactly one more Tech name in per sector-overage and is
  trivially reversible. Document it as a testing-phase setting.

**Do NOT** disable `paper_max_per_sector` (→ 0) or raise it wildly — that would
let the book go 100% Technology, violating the safety mandate. The NAV-pct cap
(30%) must stay as the backstop.

### Optional hardening (not required for this step)
Move the `check_and_enforce_kill_switch` SOD roll so the daily-loss anchor is
set at cycle START (Step 1) rather than Step 6, so the current cycle's daily-loss
math uses today's open. Harmless today (not paused); tidy for correctness.

## Verification

Immutable success shape for the step:
1. `POST /api/paper-trading/run-now` (real, not dry_run) → after completion, the
   cycle summary / `backend.log` shows **`Executing N trades` with N>=1** and
   **`Trade decisions: X sells, Y buys` with X+Y>=1** (a swap pair counts as
   1 SELL + 1 BUY).
2. A **fresh row in `financial_reports.paper_trades`** dated 2026-05-28 with a
   real ticker, action, and fill price (query via BQ MCP
   `mcp__bigquery__execute-query`: `SELECT ticker, action, executed_at FROM
   financial_reports.paper_trades WHERE DATE(executed_at)=CURRENT_DATE()
   ORDER BY executed_at DESC LIMIT 5`).
3. **All safety controls still present and unflipped:** kill switch
   `paused:false` (or correctly paused only on a real breach), stop_loss_price
   set on the new BUY, the sector COUNT cap unchanged for count-neutral swaps
   (or +1 only if B2 applied), NAV-pct cap not exceeded.
4. Confirm via log that the trade went through the documented path
   (`Swap fired …` for Fix A/B1, or `new_buy_signal` BUY for B2), not by
   bypassing a gate.

## Research Gate Checklist

Hard blockers — all satisfied:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5: arXiv
      2603.20319, arXiv 2601.05428, HRP/Wikipedia, MT4 zero-trades, Guardfolio)
- [x] 10+ unique URLs total (13 incl. snippet-only)
- [x] Recency scan (last 2 years) performed + reported (2 new 2026 findings)
- [x] Full papers/pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered the full funnel (screen → rank → subtract →
      analyze → decide_trades → swap → execute) + kill_switch sod path
- [x] Contradictions noted (prompt hypothesis refuted with log evidence)
- [x] Claims cited per-claim with URLs + file:line

---

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 8,
  "urls_collected": 13,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "report_md": "handoff/current/research_brief_phase_47_2_first_trade.md",
  "gate_passed": true
}
```
