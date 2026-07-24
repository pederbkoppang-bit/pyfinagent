# Research Brief — phase-68 step 68.0 (Real-Fill Runway cutover)

**Status: COMPLETE — gate_passed: true**
**Tier: complex** (13 read-in-full, ~48 URLs, recency scan done, three query variants)
**Date: 2026-07-10** | Researcher session (Layer-3)
**HEADLINE: the AMD/MU "price defect" premise is FALSE — fills were real 2026-07-09 market prices (AMD closed $546.72, MU $991.64; web-corroborated). 68.5's immutable criteria 1-2 + 4 need operator attention before that step spawns. All other topic-1/2/3 findings feed the cutover design as planned.**

Four mandated topics:
1. Env propagation to the launchd process (internal-heavy)
2. Shadow-mode true-order semantics (internal, code-traced)
3. alpaca-py paper order + reconciliation (external + internal)
4. AMD/MU price-defect hypothesis tree (internal-heavy)

Plus: DON'T-RE-FIX verification (alerting imports 66.1, cc-rail guard 66.1/66.4).

---

## Topic 1 — Env propagation to the launchd process (internal trace) [DRAFT-COMPLETE]

### How execution_router resolves its backend today

- `backend/services/execution_router.py:39` — `DEFAULT_MODE: BackendMode = "bq_sim"`.
- `backend/services/execution_router.py:65-71` — `_current_mode()` reads **`os.getenv("EXECUTION_BACKEND")`** directly (NOT settings). Unknown value -> WARN + fall back to `bq_sim`.
- `backend/services/execution_router.py:268-269` — `ExecutionRouter.__init__` calls `_current_mode()` per instantiation (module docstring says "selected at import time" — that is STALE: resolution is per-constructor, which is good news for env-flip rollback, bad news only if os.environ never has the key).
- Instantiation sites: `backend/services/paper_trader.py:255` (execute_buy) and `:396` (execute_sell) — a FRESH `ExecutionRouter()` per trade, so an env change takes effect on the next trade without restart *if* os.environ were mutated in-process; with launchd the env is fixed at process spawn.

### Why EXECUTION_BACKEND never arrives (root-cause chain, verified)

1. `backend/config/settings.py:584` — `model_config = {"env_file": str(_ENV_FILE), ...}` (`_ENV_FILE = backend/.env` per `settings.py:12`). pydantic-settings loads `.env` into the `Settings` **object only**; it does NOT export keys to `os.environ`. Confirmed: no `load_dotenv` anywhere in backend (66.2 brief; re-verified this session via grep — only pydantic env_file).
2. `settings.py` has **no `execution_backend` field at all** (grep `execution` in settings.py: only the phase-57.1 REJECT-gate description matches). So even the settings path could not carry it today.
3. `~/Library/LaunchAgents/com.pyfinagent.backend.plist:5-16` — `EnvironmentVariables` carries exactly four keys: `CLAUDE_CODE_OAUTH_TOKEN` (the 07-08 setup-token wiring), `DEV_LOCALHOST_BYPASS=1`, `PATH` (venv-first), `PYTHONUNBUFFERED=1`. **No EXECUTION_BACKEND, no ALPACA_* keys.**
4. Net: `os.getenv("EXECUTION_BACKEND")` in the launchd-started uvicorn is always None -> `bq_sim` forever. Matches researcher memory `project_funnel_zero_trade_66_2.md` ("loop is bq_sim FOREVER, mock alpaca fills even if flipped").

### Config-precedence options (for the design doc)

- (a) **plist EnvironmentVariables** — the only mechanism that reaches `os.getenv` in the launchd process today; precedent: the 07-08 CLAUDE_CODE_OAUTH_TOKEN key (66-era plist edits). Edit + `launchctl bootout/bootstrap` or `kickstart -k` to apply.
- (b) **settings field read via get_settings()** — idiomatic for this codebase (every other toggle is a `Settings` field, e.g. `paper_price_tolerance_pct`, `paper_position_recommendation_fix_enabled`), sourced from `backend/.env` WITHOUT plist edits or os.environ export; requires changing `_current_mode()` to consult settings (fallback chain: os.environ > settings > default). pydantic-settings itself already implements precedence "init args > os.environ > .env > defaults" internally.
- (c) **os.environ injection at startup** (e.g. `main.py` exporting selected settings) — anti-idiomatic here; nothing in the codebase does this.
- Codebase idiom = (b); launchd-reachability today = (a). Design likely: router consults `os.environ` first (operational override + rollback flip), then a new `settings.execution_backend` field (.env-persistent), then `DEFAULT_MODE` — with the 68.1 startup log printing mode+source.

### Precedent for plist edits (66.1/66.4 era)

The plist's `CLAUDE_CODE_OAUTH_TOKEN` key is the 2026-07-08 setup-token wiring (66.4 credential recovery). Mechanism: edit plist, then `launchctl bootout gui/$UID/com.pyfinagent.backend && launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.pyfinagent.backend.plist` (or `kickstart -k`); exit 113 = booted-out -> bootstrap fallback (memory `project_away_watchdog_p1_path.md`).

## Topic 2 — Shadow-mode TRUE order semantics (code-traced) [DRAFT]

### Current fill path end-to-end

- `paper_trader.py:255-261` (execute_buy): `router = ExecutionRouter(); fill = router.submit_order(symbol, qty=quantity, side="buy", client_order_id=trade_id, close_price=price)`. `trade_id = str(uuid.uuid4())` (:254) doubles as `client_order_id`.
- `paper_trader.py:396-402` (execute_sell): mirror path.
- `execution_router.py:274-289` (`submit_order`): bq_sim -> `_bq_sim_fill` (:83, fills at the close_price passed in; hash-synthesizes `50.0 + (h % 500)` ONLY when close_price is None, :96-98); alpaca_paper with creds -> `_alpaca_real_fill` (:176), without creds -> `_alpaca_mock_fill` (:128, bq_sim close +0.3% slippage, source="mock_alpaca"); **shadow (:281-289): bq_sim fill is authoritative + returned; alpaca real fill fired inside try/except with `logger.warning` on failure — shadow failure already cannot fail the cycle; alpaca fill result is DISCARDED (not returned, not persisted)**.
- `execution_router.py:292-314` (`shadow_submit`): returns the PAIR (bq, alpaca) but "does not write to any ledger" — used only by the parity harness (`scripts/harness/paper_execution_parity.py`), NOT by paper_trader.
- Persistence: `paper_trader.py:282` `self._safe_save_trade(trade)` — trade dict (:264-281) carries trade_id/ticker/action/quantity/price/total_value/…; **no `source` column is persisted today** (fill.source only reaches the log line :359-360).

### Hook point + paired-fill data (for design)

- The natural shadow hook is `submit_order`'s shadow branch (:281-289): it already computes both fills in-line; today it throws the alpaca FillResult away. A shadow-fill persistence hook there (fire-and-forget, try/except-wrapped) has per-order access to: client_order_id (=trade_id, joinable to paper_trades), symbol, qty, side, both fill prices, both latencies, status, ts. Cycle id is NOT in scope inside the router — it lives with the autonomous_loop cycle; join via trade_id -> paper_trades row (created_at/cycle attribution) or thread a cycle_id param.
- Isolation invariants already present to KEEP: (1) bq_sim result is the only one returned/persisted in shadow mode; (2) alpaca exceptions swallowed with WARN (:287-288); (3) `shadow_submit` never writes. New invariant needed: shadow-fill BQ write must itself be try/except fire-and-forget so a BQ error cannot fail the trade.

### Persistence + schema (traced)

- `bigquery_client.py:516-517` — `_pt_table(name)` = `{gcp_project_id}.{bq_dataset_reports}.{name}` -> **financial_reports (us-central1)** dataset for all paper_* tables.
- `bigquery_client.py:660-676` — `save_paper_trade`: None-values dropped, then a DYNAMIC parameterized `INSERT INTO paper_trades (cols...)` built from the dict keys. An unknown key -> BQ rejects the INSERT (confirmed by the phase-40.8.1 comment at paper_trader.py:283-286, which attaches factor_loadings only AFTER save for exactly this reason). `_safe_save_trade` (paper_trader.py:1166) wraps it.
- `paper_trades` schema: base columns trade_id/ticker/action/quantity/price/total_value/transaction_cost/reason/analysis_id/risk_judge_decision/created_at (scripts/migrations/migrate_paper_trading.py:54-66) + round-trip columns (add_round_trip_schema.py) + signals. **NO `source` column exists today** — `fill.source` reaches only the execute_buy log line (paper_trader.py:359-360) and is then dropped. 68.3's criterion ">=3 cycles with fills recorded source='alpaca_paper'" therefore requires a schema migration (ALTER TABLE ADD COLUMN IF NOT EXISTS `source` STRING via scripts/migrations/, per the CLAUDE.md BQ rule: MCP for inspection, migrations for change) + adding `source: fill.source` to the trade dict (paper_trader.py:264-281 and :425-446).
- Shadow-fill persistence options for the design: (i) same-table rows with `source='shadow_alpaca'` — pollutes trade history consumed by digests/metrics (get_paper_trades has no source filter today; every consumer would need one); (ii) **a NEW `paper_shadow_fills` table** (cycle_id, trade_id join key, ticker, side, qty, bq_price, alpaca_price, alpaca_order_id, alpaca_status, latency_ms, submitted_at) — isolates real-vs-shadow cleanly, zero consumer changes, drift report = one JOIN. Internal precedent for the pair shape: `shadow_submit` returns exactly this pair (execution_router.py:292-314); `scripts/go_live_drills/alpaca_shadow_drill.py` already used `uat-shadow-*` client_order_id prefixes (66.2 brief).
- Cycle-id availability: paper_trader has no cycle context; autonomous_loop's `_cycle_id` exists at :221. Options: thread cycle_id into execute_buy/sell (signature churn) or join shadow rows to paper_trades via trade_id==client_order_id + created_at (no churn). The trade_id join is sufficient for the 68.2 drift report ("same cycle, same ticker" pairing) since the bq_sim trade row IS the cycle's fill.

## Topic 4 — AMD/MU price-defect hypothesis tree [RESOLVED — headline finding]

### HEADLINE: there is NO price defect. The fills were REAL market prices; the "~$150/~$110 real levels" reference in live_check_66.2.md:402 was a stale-knowledge error made at the 66.2 close.

Evidence chain (each item independently verifiable):

1. **Recorded fills** — backend.log (dateless CompactFormatter, cycle 603e287c evening 2026-07-09): `21:17:46 I [paper_trader] BUY 0.8360 x MU @ $1004.70 (source=bq_sim) = $839.92` and `21:17:59 ... BUY 1.3200 x AMD @ $545.42 (source=bq_sim) = $719.93` (log lines 124393/124398). 21:17 CEST = 15:17 ET = mid-session.
2. **yfinance history fetched this session (2026-07-10)**: AMD last-3 closes `[516.11, 517.41, 546.72]`, MU `[938.38, 948.80, 991.64]`.
3. **Date alignment pinned by independent web source**: Micron "closed at $948.80 on July 8, climbing from $938.38" (CNN/Robinhood/Investing.com search results, accessed 2026-07-10) -> yf's 948.80 = 07-08, therefore the LAST yf bar (991.64 MU / 546.72 AMD) = **2026-07-09 close**.
4. So on fill day 2026-07-09: **AMD closed $546.72** (fill $545.42, -0.24% vs close — a normal intraday print); **MU closed $991.64** (fill $1004.70, +1.3% vs close on a +4.5% day — normal intraday print).
5. Corroboration of magnitude: AMD 52-week range $137.59-$584.73 (CNN, accessed 2026-07-10); MU all-time-high close **$1213.37 on 2026-06-25** (MacroTrends), driven by the Anthropic $22B HBM supply deal. The "$150 / $110" anchors are early-2025 price levels (AMD's 52wk LOW is $137.59 — exactly the stale anchor).
6. Internal consistency (already noted): qty = amount_usd/price (paper_trader.py:217) -> notional correct at ~3%/3.5% NAV; avg_entry = the same real price. **The book is CORRECT. avg_entry does NOT corrupt trailing-stop math. There are no rows to fix.**

### Consequence for 68.5 (IMMUTABLE-CRITERIA CONFLICT — Main must surface to operator)

68.5's immutable criteria demand "AMD/MU price defect root-caused to file:line with a deterministic repro" + "corrupted BQ rows corrected AUDITABLY". Both are unsatisfiable as written: the deterministic repro of the *incident* is `yf.Ticker('AMD').history()` returning the same magnitudes (run this session), and "correcting" the rows would CORRUPT a correct book. The true root cause is **live_check_66.2.md:402's un-sourced sanity check against the closer's world-knowledge price levels** (a process defect in evidence review, not a code defect). 68.5's OTHER deliverables remain fully valid (see ranked tree below: the sanity-gate closes real latent holes). Recommend operator-level re-scope of 68.5 criteria 1-2 before that step spawns; do NOT silently reinterpret them.

### Ranked hypothesis tree (as mandated — with confirm/kill evidence per branch)

- **(e*) Reference-expectation error / no defect — CONFIRMED (rank 1).** Confirm: items 1-6 above. Kill: an independent broker quote showing AMD ~=150 on 2026-07-09 (none exists; web cross-check agrees with yfinance). Repro: `source .venv/bin/activate && python -c "import yfinance as yf; print(yf.Ticker('AMD').history(period='5d')['Close']); print(yf.Ticker('MU').history(period='5d')['Close'])"`.
- **(b) Hash-fallback price synthesis — KILLED for this incident (rank n/a), REAL as a latent class.** execution_router.py:96-98 synthesizes `50.0 + (sha1(symbol)[:8] % 500)` only when close_price is None. Computed this session: AMD -> 297.0, MU -> 264.0 — wrong magnitudes AND integer+.0 shape (observed fills have cents). BUT the class is live on the SELL side: execute_sell has NO positive-price guard and a None price reaches the router's hash fallback (money_engine_audit_2026-07-08.md execution-sim F3). The 68.5 sanity gate MUST cover this path regardless of the AMD/MU exoneration.
- **(a) Close-cache stale/wrong-ticker close — KILLED for this incident.** The BUY fill reference is `_get_live_price()` = fresh `yf.Ticker(t).history(period="1d")` last Close (paper_trader.py:1220-1242), fetched at autonomous_loop.py:1263; no BQ close-cache is in the BUY fill path. SELLs do use the Step-5 MTM-stamped current_price (stale-by-one-cycle by design; same audit, F3 family) — gate-relevant, not AMD/MU-relevant.
- **(c) FX/scale confusion — KILLED.** AMD/MU are US tickers; market="US" -> `_fx_usd_to_local`/`_fx_local_to_usd` = 1.0 (paper_trader.py:212-217). The real FX defect family is FX-1..FX-4 in money_engine_audit_2026-07-08.md (see Topic-4 appendix below) and is non-US-only.
- **(d) Wrong-column/DESC-order BQ read — KILLED for avg_entry.** avg_entry is written at BUY time from the fill (paper_trader.py:338), not read back from an ordered query. The DESC class was real but elsewhere: drawdown alarm (fixed — see DON'T-RE-FIX section).
- **(e) Split-adjustment absence — KILLED.** Neither AMD nor MU split in the window; yfinance history is auto-adjusted; magnitudes match independent quotes.

### Topic-4 appendix — genuinely-open price-integrity defects the 68.5 gate should target (from money_engine_audit_2026-07-08.md, all [CONFIRMED], none AMD/MU)

- F3: unguarded SELL fill price (None -> hash-synthetic $50-550 fill; 0 -> position deleted for $0); no re-fetch, no tolerance gate on SELLs.
- F1/F2: non-atomic portfolio upsert (cash re-seed to $20k on crash window) + trade-row-before-position-write.
- FX-1 (parked 61.3, HAND-OFF only): add-on BUY to open non-USD position stores USD-scale value into LOCAL-unit avg_entry_price (paper_trader.py:290-291 vs :338 unit contract); LATENT (0 non-US BUY-after-BUY sequences ever). FX-2: realized_pnl_usd revalues entry leg at exit FX (paper_trader.py:443; proven +$2.98 overstatement on the real 07-03 KR stop sell). FX-3: LOCAL-as-USD fallbacks (paper_trader.py:521, :288/:462/:524). FX-4: correct `fx_pnl_attribution()` exists (paper_trader.py:44-53) with zero production callers.
- The phase-30.6 BUY tolerance gate fails OPEN when price_at_analysis is None (paper_trader.py:175-188 requires it non-None; comment :168 "lite-Claude path can lack it") — an independent-quote gate must not share the fill's data source (yfinance vs yfinance compares a source to itself; AMD/MU proves live+analysis prices agree even when world-knowledge screams — the gate needs a SECOND source, e.g. Alpaca latest-quote, to be meaningful).

## DON'T-RE-FIX verification (mandated)

- **Alerting imports (66.1) — FIXED, stands.** autonomous_loop.py:233, :764, :971, :1005 all import `backend.services.observability.alerting` with the phase-66.1 comment ("was backend.services.alerting (module DOES NOT EXIST...)"); zero bare `backend.services.alerting` imports remain in backend/ (grep this session — only the guard test references the string). Guard test: backend/tests/test_phase_66_1_rail_guard.py:62-68 asserts `find_spec("backend.services.alerting") is None` and the import string is absent.
- **cc-rail guard (66.1/66.4) — DONE, stands.** autonomous_loop.py:209-230: per-cycle `rail_guard_reset(_cycle_id)` (:221) BEFORE the probe, `claude_code_health_probe` (:222), failed probe gates the rail (:224-230, phase-66.1 criterion 1) with breaker-dedupe.
- **DESC-order phantom drawdown — ALREADY FIXED (66.2 hotfix, commit 9262ed36)**, not waiting for 68.5: backend/services/drawdown_alarm.py:65-100 sorts by the snapshot's own date key and refuses to guess when undated; regression tests exist at backend/tests/test_phase_66_2_drawdown_order.py (incl. `test_desc_order_growing_nav_is_zero_drawdown_phantom_regression`). 68.5's "DESC phantom fix + regression test" criterion is therefore ALREADY SATISFIED by 66.2 evidence — second immutable-criteria collision for Main to surface. Remaining DESC-class consumers already handle order: perf_metrics.py:101 + paper_go_live_gate.py:46 carry explicit phase-47.4 newest-first comments (re-sort/handle in place).

---

## Topic 3 — alpaca-py paper order + reconciliation (external + internal)

### External facts (per-claim cited; all accessed 2026-07-10)

- **paper=True semantics**: "paper=True enables paper trading" on the TradingClient constructor; SDK routes to the paper environment (alpaca-py Trading API reference, https://alpaca.markets/sdks/python/trading.html). REST base URL for paper = `https://paper-api.alpaca.markets` ("In most cases, you need to set an environment variable APCA_API_BASE_URL = https://paper-api.alpaca.markets" — Paper Trading docs, https://docs.alpaca.markets/us/docs/paper-trading). With alpaca-py the `paper=True` flag replaces the env var; the repo already hardcodes it (execution_router.py:228, alpaca_broker.py:74).
- **client_order_id idempotency**: POST /v2/orders `client_order_id` = "A unique identifier for the order. Automatically generated if not sent. (<= 128 characters)" (https://docs.alpaca.markets/reference/postorder). Duplicate submission against another ACTIVE order returns **HTTP 422 "client_order_id must be unique"** (official Alpaca learn guide "How to Fix 30 Common Errors...", https://alpaca.markets/learn/how-to-fix-common-trading-api-errors-at-alpaca). Retrieval: `get_order_by_client_id('...')` (https://docs.alpaca.markets/us/docs/working-with-orders). NOTE the qualifier: uniqueness is enforced against ACTIVE orders; Alpaca has no Stripe-style idempotency-key header (open forum ask, no staff answer: https://forum.alpaca.markets/t/idempotency-on-order-create/15801, Jan 2025). Design consequence: client_order_id = trade_id (uuid4) gives us dedupe on crash-retry (422 on the retry -> fetch by client id instead of double-submitting), and the repo already passes it (execution_router.py:233).
- **Order lifecycle**: primary states `new`, `partially_filled`, `filled` ("no further updates will occur"), `done_for_day`, `canceled`, `expired`, `replaced`, `pending_cancel`, `pending_replace`; uncommon: `accepted`, `pending_new`, `accepted_for_bidding`, `stopped`, `rejected`, `suspended`, `calculated` (https://docs.alpaca.markets/us/docs/orders-at-alpaca). The router's 2s poll loop for terminal status (execution_router.py:239-244) checks only `filled`/`partially_filled` — a `rejected`/`canceled` order exits the loop with fill_price None -> FillResult.fill_price=0.0, which execute_buy then REPLACES with the passed-in price (`exec_price = fill.fill_price if ... else price`, paper_trader.py:260) — a real-cutover trap: an UNFILLED alpaca order would be booked at the reference price. Reconciliation design must treat non-filled terminal states explicitly.
- **Paper fill simulation**: fills "based on real-time quotes... only when they become marketable"; "partial fills for a random size 10% of the time"; matched "against the best available current market price (NBBO)"; does NOT simulate market impact, latency slippage, queue position, price improvement, regulatory fees, or dividends; "order quantity is not checked against the NBBO quantities" (Paper Trading docs). Drift vs bq_sim close-fills is therefore EXPECTED and bounded (68.3's <2% criterion is realistic for liquid US names at market orders during RTH); timing deltas dominated by the 21:17-CEST fill time.
- **Fractional shares**: market orders support fractional `qty` or `notional` (SDK reference) — matters because bq_sim positions are fractional (AMD 1.32 sh); the shadow submit can mirror exact quantities. TimeInForce: `day, gtc, opg, cls, ioc, fok` (orders-at-alpaca); **fractional orders must be DAY** (known Alpaca constraint; router already uses TimeInForce.DAY, execution_router.py:232).
- **Flatten**: `close_all_positions(cancel_orders=True)` "will also cancel all open orders" (SDK reference) — the 68.2 flatten primitive. Per-position `close_position` also exists. Account reset alternative: paper accounts can be deleted/recreated with arbitrary balance, but "generate new API keys for any newly created account" (Paper Trading docs) — reset would INVALIDATE stored keys; prefer close_all_positions to preserve creds.
- **Errors relevant to cutover**: 403 "insufficient buying power", 403 PDT protection ($25k rule applies on paper too), 403 "Insufficient qty available" (shares reserved by open orders), 422 duplicate client id (learn guide). PDT note: bq_sim NAV ~$24k is UNDER $25k — a real paper account making 4+ day trades in 5 business days can trip PDT; the Alpaca paper account balance should be set >= $25k or day-trade cadence reviewed (daily cycle = 1 trade/day, low risk, but stop-loss same-day exits could count).
- **PKLIVE vs PKTEST prefixes — honest negative**: the PK(paper)/AK(live) key-prefix convention is NOT documented anywhere official (searched official docs + forum threads 7509/4055; the authentication docs only say paper and live keys are distinct per-domain). It is practitioner folklore that paper key IDs start "PK". Consequence: the repo's `key.startswith("PKLIVE")` guard (execution_router.py:74-76) matches a pattern that (per the folklore convention) no real Alpaca key uses — live keys would start "AK", so the PKLIVE check is close to vacuous. The load-bearing guards are `paper=True` (SDK-level base-URL pin) + `ALPACA_PAPER_TRADE` flag + key SOURCE separation. 68.1's "PKLIVE-class rejected" criterion should be implemented as "reject any key NOT matching the expected paper shape (startswith 'PK')" plus keep the PKLIVE literal for continuity — and must NOT claim the prefix convention is Alpaca-documented.
- **alpaca-py version state (PyPI JSON, fetched 2026-07-10)**: pinned 0.43.2 released 2025-11-04; newer patch releases 0.43.3 (2026-04-24), 0.43.4 (2026-04-29), 0.43.5 (2026-07-02, latest; adds "Tolerate Alpaca's deprecated PDT/DTBP account fields" — relevant if get_account fields are consumed). No breaking API changes in 0.43.x patches (tax-id types, enum fix, repr fix, PDT-field tolerance). Pin stays valid for 68.x; note the PDT/DTBP-field deprecation tolerance only exists in 0.43.5 if account-config parsing errors appear.

### Internal (creds + guards to KEEP)

- Router creds channel: `os.environ["ALPACA_API_KEY_ID"]` / `ALPACA_API_SECRET_KEY` read directly (execution_router.py:193-194, :277, :284, :305; alpaca_broker.py:35-38). Settings ALSO carry `alpaca_api_key_id`/`alpaca_api_secret_key` as SecretStr (settings.py:128-129) but those are the NEWS-adapter channel; SecretStr in settings does NOT populate os.environ, so even a populated backend/.env leaves the router creds-blind (same class as EXECUTION_BACKEND; researcher sandbox cannot read backend/.env to confirm which keys it holds — Main must verify). **Beware the SecretStr truthiness trap when unifying: never `str()` a SecretStr; use unwrap (memory: SecretStr killed 4 alpha overlays).**
- Guards to KEEP (name them in the design): `_refuse_live_keys()` (execution_router.py:74-81: PKLIVE prefix + `ALPACA_PAPER_TRADE=false` refusal), `paper=True` hardcoded (execution_router.py:228, alpaca_broker.py:74), `_max_notional_usd` clamp default $10k with worst-case 1e6 price fallback (execution_router.py:157-226), phase-30.6 price-tolerance gate (paper_trader.py:164-188), idempotency 30-min duplicate-BUY guard (paper_trader.py:219-246), `.mcp.json` alpaca server pins `ALPACA_PAPER_TRADE=true` (mcp.json alpaca env block) and `.claude/settings.json:167-177` denies all trade-mutation MCP tools (place/cancel/replace/close/exercise) — the BACKEND path uses alpaca-py directly and is NOT covered by the MCP deny-list (correct: deny-list is session-protection, not runtime-protection).
- Existing shadow drill: `scripts/go_live_drills/alpaca_shadow_drill.py` made REAL 1-share paper BUYs with `uat-shadow-*` order ids (66.2 brief — its "+1 SELL" docstring is stale); `scripts/harness/mcp_ab_test.py` HAS a live sell path (:159). These two + MCP-session drills created the stray shorts (-13,842.89 short_market_value, SIGNED — real short positions on a margin-type paper account where a naked SELL opens a short). 68.2's flatten must use close_all_positions and re-snapshot.

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://docs.alpaca.markets/us/docs/paper-trading | 2026-07-10 | official doc | WebFetch full | paper base URL; fills on real-time quotes, 10% random partials; reset invalidates keys; no fees/dividends simulated |
| 2 | https://docs.alpaca.markets/us/docs/working-with-orders | 2026-07-10 | official doc | WebFetch full | get_order_by_client_id; timeout rule: "should not attempt to resend the order... until confirmed"; websocket status stream |
| 3 | https://docs.alpaca.markets/us/docs/orders-at-alpaca | 2026-07-10 | official doc | WebFetch full | full order-status enumeration (new/filled/canceled/rejected/...); client_order_id auto-generation; TIF set |
| 4 | https://alpaca.markets/sdks/python/trading.html | 2026-07-10 | official SDK doc | WebFetch full | paper=True; submit_order request models; close_all_positions(cancel_orders=True); fractional qty/notional |
| 5 | https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/ | 2026-07-10 | official doc | WebFetch full (301-followed) | precedence init > env > dotenv > secrets; dotenv loaded into model, NOT os.environ; "environment variables will always take priority over values loaded from a dotenv file" |
| 6 | https://docs.alpaca.markets/reference/postorder | 2026-07-10 | official API ref | WebFetch full | client_order_id "<= 128 characters", auto-generated if not sent |
| 7 | https://alpaca.markets/learn/how-to-fix-common-trading-api-errors-at-alpaca | 2026-07-10 | official learn guide | WebFetch full | 422 "client_order_id must be unique" (active orders); 403 buying-power/PDT/insufficient-qty messages |
| 8 | https://github.com/alpacahq/alpaca-py/releases | 2026-07-10 | vendor GitHub | WebFetch full | 0.43.x change list (dates unreliable in render; corrected via #9) |
| 9 | https://pypi.org/pypi/alpaca-py/json | 2026-07-10 | registry data | curl + parse (counts per gcloud-docs precedent) | 0.43.2 = 2025-11-04; 0.43.5 latest 2026-07-02; no 2026 breaking changes |
| 10 | https://www.launchd.info/ | 2026-07-10 | authoritative reference | WebFetch full | EnvironmentVariables dict, no shell expansion; "always (re)load a job definition after changing it"; bootstrap/bootout/kickstart on 10.10+ |
| 11 | https://martinfowler.com/articles/feature-toggles.html | 2026-07-10 | authoritative blog (the router's own cited pattern) | WebFetch full | Ops Toggles need reconfiguration without redeploy; kill switch = "manually-managed Circuit Breaker" |
| 12 | https://forum.alpaca.markets/t/idempotency-on-order-create/15801 | 2026-07-10 | community | WebFetch full | negative result: no idempotency-key header exists; no staff answer (Jan 2025 thread) |
| 13 | https://forum.alpaca.markets/t/questions-1-build-scanner-2-paper-vs-live-api-key/7509 | 2026-07-10 | community | WebFetch full | negative result: PK/AK prefix NOT confirmed anywhere official |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.cnn.com/markets/stocks/AMD + /MU | market data | search snippet sufficed (price cross-check: AMD $516.80 07-0x, 52wk 137.59-584.73) |
| https://www.macrotrends.net/stocks/charts/MU/micron-technology/stock-price-history | market data | snippet: MU ATH close $1213.37 on 2026-06-25 |
| https://www.tradingkey.com/analysis/stocks/us-stocks/262015351-... | analysis | snippet: MU Anthropic $22B HBM deal context |
| https://www.fool.com/investing/2026/07/04/prediction-micron-technology-stock-will-hit-at-lea/ | analysis | magnitude corroboration only |
| https://alpaca.markets/learn/connect-to-alpaca-api | official learn | key-generation UI walkthrough; no prefix info (updated 2026-03-25 per snippet) |
| https://alpaca.markets/learn/start-paper-trading | official learn | duplicative of #1 |
| https://github.com/alpacahq/alpaca-docs/.../paper-trading.md | official repo | duplicative of #1 |
| https://github.com/alpacahq/alpaca-py | vendor GitHub | README duplicative of #4 |
| https://github.com/alpacahq/alpaca-trade-api-python/blob/master/README.md | vendor GitHub (deprecated SDK) | APCA_API_BASE_URL env precedent only |
| https://docs.alpaca.markets/us/v1.1/docs/authentication-1 | official doc | auth headers; no prefix info |
| https://forum.alpaca.markets/t/getting-422-when-trying-to-supply-client-order-id/12223 | community | 422 corroboration |
| https://forum.alpaca.markets/t/all-orders-containing-client-order-id/13694 | community | client-id query patterns |
| https://github.com/alpacahq/alpaca-trade-api-python/issues/401 | community | id vs client_order_id distinction |
| https://forum.alpaca.markets/t/how-to-get-the-apca-api-key-id-and-then-apca-api-secret-key/4055 | community | no prefix info |
| https://forum.alpaca.markets/t/unprocessable-entity-422/6417, .../12316 | community | 422 class corroboration |
| https://apispine.com/alpaca/authentication | third-party 2026 | unofficial; not authoritative for prefixes |
| https://alpaca.markets/deprecated/docs/api-documentation/api-v2/orders/ | official (deprecated) | superseded by #3/#6 |
| https://www.datons.ai/..., https://wire.insiderfinance.io/..., https://www.npmjs.com/package/@alpacahq/alpaca-trade-api, https://pypi.org/project/alpaca-trade-api/0.25/, https://rapidapi.com/collection/alpaca-api, https://robinhood.com/us/en/stocks/{AMD,MU}/, https://www.morningstar.com/stocks/xnas/{amd,mu}/quote, https://finance.yahoo.com/quote/{AMD,MU}/(+history), https://www.cnbc.com/quotes/AMD, https://ir.amd.com/stock-data/price-history, https://www.marketbeat.com/stocks/NASDAQ/AMD/forecast/, https://www.thestreet.com/investing/stocks/amd-...-july-2026, https://www.investing.com/equities/{adv-micro-device,micron-tech}-historical-data, https://stockinvest.us/stock/MU | tutorials/quotes | context + price corroboration only |

Unique URLs collected: ~48 (13 read in full + ~35 snippet-only).

## Search-query composition (three-variant discipline)

- Current-year: "alpaca paper trading API key prefix PK base URL **2026**", "AMD stock price today July **2026**", "Micron MU stock price July **2026**".
- Last-2-year window: covered by the 2025 forum threads (idempotency Jan 2025) + PyPI release history 2025-2026 pull.
- Year-less canonical: "alpaca-py TradingClient paper trading client_order_id idempotency", "alpaca API duplicate client_order_id 422 must be unique", "alpaca api key PK prefix paper AK live key begins".

## Recency scan (2024-2026)

Performed. Findings: (1) alpaca-py shipped three patch releases in 2026 (0.43.3/0.43.4/0.43.5, latest 2026-07-02) with no breaking trading-API changes vs the pinned 0.43.2 — 0.43.5's PDT/DTBP deprecated-field tolerance is the only cutover-adjacent change (Alpaca is deprecating those account fields server-side). (2) Alpaca's paper-trading docs moved to per-account create/delete semantics (reset invalidates API keys) — newer than older "reset button" lore. (3) pydantic-settings current docs (2.x line, 2025-2026) re-confirm the dotenv-not-exported behavior that is this phase's root cause — no change that would alter the 68.1 design. (4) MU/AMD mid-2026 price history (MU ATH 2026-06-25 $1213.37 on the Anthropic HBM deal; AMD 52wk high $584.73) — this 2026-recency data is what OVERTURNS the step's premise. No superseding findings on launchd (stable since 10.10) or Fowler ops-toggle doctrine.

## Key findings (numbered, per-claim cited above)

1. EXECUTION_BACKEND never reaches the launchd process: router reads os.environ only (execution_router.py:66); pydantic env_file loads into the Settings model without exporting (pydantic-settings docs; settings.py:584); the plist carries no such key (plist:5-16). Fix surface = plist key (operational) + optional settings-field fallback (idiomatic), logged mode+source at startup.
2. The AMD/MU "corruption" is NOT a defect — fills match real 2026-07-09 market prices (AMD close $546.72, MU $991.64; web-corroborated). The stale "$150/$110" reference in live_check_66.2.md:402 is the actual root cause. **68.5's immutable criteria 1-2 are unsatisfiable as written; the DESC-phantom criterion is ALREADY satisfied by the 66.2 hotfix (commit 9262ed36). Main must surface both to the operator before 68.5 spawns.**
3. Real latent price-integrity holes DO exist and justify the 68.5 sanity gate: unguarded SELL fill price (hash-synthetic on None, $0-delete on 0), tolerance gate fails open without price_at_analysis, and the gate's "independent quote" must come from a non-yfinance source (Alpaca latest-quote) or it compares yfinance to itself.
4. Shadow mode exists and is already isolation-correct in structure (bq_sim authoritative, alpaca try/except-swallowed, result discarded) — what's missing is persistence of the paired fill (no `source` column in paper_trades; new `paper_shadow_fills` table recommended) and creds in the launchd env.
5. client_order_id (=trade_id uuid4, <=128 chars) gives crash-retry idempotency via 422-on-duplicate-active + get_order_by_client_id recovery; the router's fill-poll treats non-filled terminal states as "use reference price" — must be fixed for real cutover (rejected order must NOT book a fill).
6. The PKLIVE prefix guard is near-vacuous (convention undocumented; live keys reputedly "AK..."): keep it, but make paper=True + base-URL pin + "key must start with PK" the tested triple-enforcement.

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| backend/services/execution_router.py | 39, 65-71, 74-81, 96-98, 128-154, 157-226, 228-244, 268-289, 292-314, 316-329 | mode resolution, guards, three fill paths, shadow, rollback | live; mode resolution per-constructor (docstring "import time" is stale); shadow discards alpaca fill |
| backend/config/settings.py | 12, 128-129, 584-589 | env_file loading; alpaca news-adapter SecretStr creds; no execution_backend field | live |
| ~/Library/LaunchAgents/com.pyfinagent.backend.plist | 5-16, 30-41 | 4 env keys (no EXECUTION_BACKEND/ALPACA); caffeinate+uvicorn launch | live |
| backend/services/paper_trader.py | 164-188, 212-217, 219-246, 249-261, 264-282, 302-353, 359-360, 379-402, 425-447, 1166, 1220-1242 | tolerance gate, FX sizing, idempotency guard, router calls, trade/position writes, live-price fetch | live; no source column persisted; SELL price unguarded |
| backend/services/autonomous_loop.py | 209-230, 233/764/971/1005, 1211, 1252-1295, 1837, 2084, 2328, 2367, 2526 | rail guard, fixed alerting imports, decide_trades call, BUY executor + live-price fill reference, lite-path price fields | live |
| backend/db/bigquery_client.py | 516-517, 660-676, 698, 1039-1042 | _pt_table -> financial_reports; dynamic INSERT (unknown col rejected); snapshots DESC | live |
| scripts/migrations/migrate_paper_trading.py | 54-66 | paper_trades base schema (no source col) | authoritative schema record |
| backend/services/drawdown_alarm.py | 65-108, 143 | DESC-phantom FIXED (date-key sort, refuse-undated); correct alerting import | fixed 66.2 (9262ed36) |
| backend/tests/test_phase_66_2_drawdown_order.py | all | DESC regression tests already exist | passing suite exists |
| backend/tests/test_phase_66_1_rail_guard.py | 62-68 | guards against bare backend.services.alerting regression | exists |
| backend/markets/alpaca_broker.py | 35-38, 61-79, 81-106, 141, 162-181 | lazy TradingClient, delegates to router fills, get_orders/positions | built (phase-5.1), not in live loop |
| scripts/go_live_drills/alpaca_shadow_drill.py + scripts/harness/mcp_ab_test.py | (66.2 brief) | origin of stray paper-account shorts; uat-shadow-* order-id precedent | historical |
| handoff/current/money_engine_audit_2026-07-08.md | FX-1..FX-4, F1-F4 registers | confirmed latent money defects feeding 68.5 gate + 61.3 handoff | evidence pack |
| handoff/current/live_check_66.2.md | 369-402 | the AMD/MU evidence + the stale-reference claim at :402 | premise overturned this brief |
| .mcp.json (alpaca server) + .claude/settings.json:167-177 | — | MCP paper pin + session deny-list on trade mutations | live; not a runtime guard |

## Consensus vs debate (external)

- Consensus: dotenv loaders feeding a settings object do not mutate the process env (pydantic-settings docs; matches the no-load_dotenv grep); launchd env changes require job reload; ops toggles should flip without redeploy (Fowler).
- Debate/uncertainty: Alpaca key-prefix convention (undocumented — treat as heuristic only); exact duplicate-client_order_id semantics for COMPLETED orders (docs only establish uniqueness against active orders; 68.1 tests should probe both) ; whether paper fills at NBBO overstate fill quality vs live (Alpaca itself lists the unsimulated effects — drift thresholds should not be tightened below ~1-2%).

## Pitfalls (from literature + code)

1. Rejected/canceled alpaca order booked at reference price (router poll gap) — reconciliation must key on terminal status, not fill_price presence.
2. Paper account reset deletes API keys — flatten with close_all_positions, never reset, or plist creds go stale.
3. PDT protection on sub-$25k paper accounts can 403 day-trade-like sequences (stop-loss same-day exits).
4. 10% random partial fills on paper — paired-fill drift rows must record filled_qty, not assume qty.
5. Shadow BQ write failure must be swallowed (fire-and-forget) or it converts a measurement tool into a cycle-killer.
6. plist EnvironmentVariables: no variable expansion; reload (bootout/bootstrap or kickstart -k) required; exit 113 -> bootstrap fallback.
7. An "independent quote" gate sourced from the same yfinance feed is self-comparison — use Alpaca latest-quote (creds now available in-process for mode=shadow/alpaca) or another second source.
8. SecretStr unwrap trap when unifying creds channels (never `or ""` / str() a SecretStr).

## Design inputs for Main (per design-doc section)

**Config precedence (env > .env > default):**
- Mechanism: add `EXECUTION_BACKEND` to plist EnvironmentVariables (precedent: CLAUDE_CODE_OAUTH_TOKEN key, 07-08) for the launchd process; optionally add `execution_backend: str = Field("bq_sim")` to Settings and change `_current_mode()` (execution_router.py:65-71) to `os.environ > settings > DEFAULT_MODE`, logging BOTH resolved mode AND source at startup (68.1 criterion). pydantic-settings already implements env>dotenv internally, so the settings field alone gives ".env support"; the os.environ check preserves the pure-env override + rollback flip.
- Byte-identical default: DEFAULT_MODE stays "bq_sim" (execution_router.py:39); with nothing set anywhere, behavior unchanged (68.1 test).
- Apply/reload: edit plist -> `launchctl bootout gui/$UID/... && launchctl bootstrap gui/$UID ...` (or `kickstart -k`); "always (re)load a job definition after changing it" (launchd.info). Restart is safe re double-fire: MemoryJobStore + forward-only scheduling (memory: backend-restart-safety); do not kickstart mid-cycle.
- LOUD missing-creds log (68.1): today mode=alpaca_paper without creds SILENTLY mock-fills (execution_router.py:277-280) — replace silence with a single unmissable startup error naming ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY.

**Shadow isolation:**
- Invariants to keep: bq_sim result is the ONLY returned/persisted fill in shadow mode (execution_router.py:283-289); alpaca exceptions logged-and-swallowed (:287-288); new shadow-fill BQ write must itself be try/except fire-and-forget.
- Persistence: NEW table `paper_shadow_fills` in financial_reports (us-central1) via scripts/migrations/ (NOT ad-hoc MCP), keyed by client_order_id==trade_id joinable to paper_trades; columns per Topic-2 list incl. alpaca_status + filled_qty (10%-random-partial reality). Avoid same-table `source='shadow'` rows (every paper_trades consumer would need filters).
- Never mutates bq_sim state: shadow branch touches no position/cash writes (all in paper_trader AFTER the router returns the bq fill) — test asserts book identical shadow on/off (68.2 criterion).

**Order-id idempotency:**
- client_order_id = trade_id (uuid4, 36 chars <=128) already wired (paper_trader.py:254-258, execution_router.py:233). Duplicate ACTIVE submission -> 422 "client_order_id must be unique"; recovery = get_order_by_client_id. Crash-retry double-BUY additionally caught by the 30-min guard (paper_trader.py:219-246 — note its >1% price-move blind spot, money-audit F4). Shadow rows may reuse trade_id with a `shadow-` prefix if Alpaca-side uniqueness collides with a later real cutover submission of the same trade_id (uat-shadow-* precedent).
- Fix-before-cutover: non-filled terminal statuses (rejected/canceled/expired) must produce NO booked fill (current poll would fall back to reference price — paper_trader.py:260).

**Rollback = env flip:**
- Flip plist EXECUTION_BACKEND back to bq_sim + reload; per-constructor mode resolution (execution_router.py:268-269) means no in-process state to unwind (router docstring's rollback claim verified); `rollback_to_bq_sim()` (:325-329) + `flip_to()` (:316-322) exist for in-process drills. State preserved in BQ via the new source column/table.
- 68.3 rollback drill: flip -> one clean bq_sim cycle -> flip forward; evidence = startup mode+source log lines.

**PKLIVE/paper-only guards kept (named):**
- `_refuse_live_keys` (execution_router.py:74-81) — keep, but 68.1's triple-enforcement should test: (a) paper base URL pinned via paper=True (:228, alpaca_broker.py:74), (b) key NOT startswith("PK") rejected (PKLIVE literal kept as belt-and-suspenders; convention undocumented — say so in the test comment), (c) ALPACA_PAPER_TRADE=false refusal + mode can never escalate beyond paper regardless of env values. Plus `_max_notional_usd` clamp (:157-226) and phase-30.6 tolerance gate (paper_trader.py:164-188) unchanged.
- $0-metered boundary intact: Alpaca paper API is free; no LLM spend in 68.1-68.3 paths.

**Operator-facing flags Main must raise (from this gate):**
1. 68.5 criteria 1-2 unsatisfiable (no defect; rows correct; "correcting" would corrupt the book) + criterion 4 (DESC) already satisfied by 66.2 — needs operator re-scope BEFORE 68.5 spawns; the sanity-gate + FX-1 handoff + 63.3 seeds remain valid scope.
2. 68.2 flatten: use close_all_positions, NOT account reset (reset invalidates the API keys the plist will carry).
3. Verify backend/.env alpaca key state (researcher sandbox denied); decide whether plist carries creds directly or a future settings-unification does (SecretStr trap).

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (13; 8 official-doc/vendor tier)
- [x] 10+ unique URLs total (~48)
- [x] Recency scan (last 2 years) performed + reported (2026 alpaca-py releases, 2026 price history — premise-overturning)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (router, trader, loop, settings, plist, BQ client, migrations, broker, drills, audits)
- [x] Contradictions / consensus noted (key-prefix folklore vs docs; live_check_66.2 premise vs market data)
- [x] All claims cited per-claim
- [ ] backend/.env contents NOT inspected (sandbox-denied) — flagged to Main rather than guessed

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 13,
  "snippet_only_sources": 35,
  "urls_collected": 48,
  "recency_scan_performed": true,
  "internal_files_inspected": 19,
  "report_md": "handoff/current/research_brief_68.0.md",
  "gate_passed": true
}
```
