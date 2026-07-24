# Research Brief — phase-70.3: ATOMIC, cash-bounded, cross-sector swap + non-US avg-entry FX fix

**Tier:** complex (focused-implementation) · **Date:** 2026-07-17 · **Author:** Layer-3 Researcher subagent
**HEAD at re-anchor:** `ec64e4ea` (`ec64e4ea chore: auto-changelog hook entry for 1d493993`)

**Step 70.3 objective (S3 + money-path):** make the sector-blocked SWAP/rotation path correct + safe,
flag-gated, fail-safe, $0, paper-only, DARK-until-token. Four criteria:
1. Swap is **ATOMIC** — BOTH legs (SELL then paired BUY) execute or NEITHER; BUY bounded by
   `available_cash` + honors the $50 floor; never a SELL that fires while its paired BUY silently drops →
   net −1 position. Proven by a red→green test of the SELL-executes-BUY-drops scenario.
2. Swap can rotate into a **DIFFERENT sector** (not only same-sector churn), gated behind a
   diversification/churn flag; OFF → byte-identical.
3. Add-to-existing **avg_entry_price** computed in **CONSISTENT units** for non-US tickers (no
   USD-cost / local-share mix); tested with a non-USD ticker.
4. All **fail-safe** (a failure blocks/holds, never corrupts the book); **NO risk-limit threshold moved.**

**Binding:** flag-gated default-OFF (byte-identical OFF); $0 metered; paper-only; NO risk threshold moved;
`historical_macro` FROZEN; fail-safe (failure blocks/holds, never corrupts the book).

**Builds on:** `research_brief_70.0.md` (gate_passed=true, 7 sources in full) + `design_trade_diversity_70.md`
section (b). This 70.3 brief is the FOCUSED implementation tier: re-anchors the drifted HEAD line numbers,
adds fresh 2025–2026 Saga / FX-weighted-average-cost sources, and hardens the atomic-execution design against
the non-cash BUY-drop reasons the 70.0 pre-flight-only sketch missed.

---

## 1. Three-variant query disclosure (research-gate mandatory)

Per `.claude/rules/research-gate.md`, ≥3 query variants per topic (current-year 2026 frontier /
last-2-year 2025-24 window / year-less canonical).

**Topic A — SAGA / compensating-transaction / atomic multi-leg order execution + rollback**
- 2026 frontier: `saga pattern compensating transaction atomic multi-leg order execution 2026`
- Last-2-year: `SagaLLM transaction guarantees multi-agent rollback 2025`
- Year-less canonical: `saga pattern compensating transaction microservices no automatic rollback`

**Topic B — multi-currency / FX weighted-average-cost accounting (unit consistency on add-on lots)**
- 2026 frontier: `multi currency weighted average cost basis foreign currency position accounting 2026`
- Last-2-year: `weighted average cost method multi-currency portfolio local currency FX 2025`
- Year-less canonical: `weighted average cost basis foreign currency shares functional currency accounting`

(Queries run + result set recorded in §2 source table; mix of current-year / last-2-year / year-less hits.)

---

## 2. Source table

### Read IN FULL via WebFetch (8 — floor is 5)
| # | Source | Tier | Topic | Key takeaway |
|---|--------|------|-------|--------------|
| 1 | microservices.io — **Pattern: Saga** (Richardson) | 2 (authoritative ref) | A | "a sequence of local transactions … If a local transaction fails … the saga executes a series of compensating transactions that undo the changes." **"Lack of automatic rollback — a developer must design compensating transactions that explicitly undo changes."** Lack of isolation → countermeasures (semantic lock, pivot txn, commutative updates). Orchestration vs choreography. |
| 2 | **SagaLLM: Transaction Guarantees for Multi-Agent LLM Planning** — arXiv 2503.11951**v3** (Chang & Geng, Stanford; Mar/Jul 2025) | 1 (peer-reviewed preprint) | A | `S={T₁..Tₙ,Cₙ..C₁}`; **"Applying O … must yield either a fully committed state S′, or trigger a coherent rollback that returns the system to S, thereby avoiding partial or inconsistent outcomes."** Minimal compensation set via dependency graph `D={(oᵢ,oⱼ,cᵢⱼ)}`. **Pre-execution Validation: GlobalValidationAgent validates inputs + dependency satisfaction BEFORE any operation** (independent, not self-validation). |
| 3 | Temporal — *Compensating Actions, Part of a Complete Breakfast with Sagas* | 3 (authoritative practitioner) | A | **"Undoing the state changes is different from rolling the database back to a previous snapshotted state"** — compensation is a deliberate application-level inverse ("put the funds back"). Register compensation **before** each forward step; run in reverse. **"Compensation methods must take into account the possibility that what you want to compensate for may or may not have executed"** → idempotent `…IfPresent`. |
| 4 | oneuptime — *How to Implement the Saga Pattern* (**2026-02-20**) | 4 (practitioner, 2026) | A | Orchestration: "a central coordinator … tells each service what to do and handles compensation on failures." **"On failure, compensate all completed steps in reverse order."** Emphasizes POST-failure compensation over pre-validation; "Make compensating actions idempotent so retries are safe"; compensation-failure escalation (retry/DLQ/alert). |
| 5 | Peter Selinger — *Tutorial on multiple currency accounting* (named academic, Dalhousie) | 3 (authoritative blog) | B | Track cost in the asset's own unit, keep **quantity separate from value**: "All accounts are kept in the reference currency … we do not actually keep any account in [barrels of oil]" — separate informational unit records. Currency trading account **isolates FX gains/losses** from the asset's own price change ("purpose … is not to perform conversions, but to calculate gains and losses"). |
| 6 | CFA Institute — *Currency Management: An Introduction* (2026 refresher) | 1/2 (authoritative) | B | **"The domestic-currency return on foreign-currency assets can be broken into the foreign-currency asset return and the return on the foreign currency … These two components … are multiplicative."** Local-asset move and FX move are **distinct, non-conflatable** components requiring separate measurement. |
| 7 | Markdale Financial Management — *Foreign Currency Cost Base Reporting* | 4 (practitioner) | B | Convention (B) for TAX: cost base tracked in HOME currency with per-lot FX at the transaction date ("the exchange rate in effect on the date of the transaction should be used"). Confirms per-lot FX discipline; must keep a **separate ledger** — brokers don't reconcile local vs home. |
| 8 | AllInvestView — *Multi-Currency Portfolio Tracking Guide* (**2026**) | 4 (practitioner, 2026) | B | Convention (B) explicit + the anti-pattern warning: **"prevents misalignment between share quantities (tracked in local currency) and aggregate costs (tracked in base currency)."** "Two purchases of the same foreign stock at the same local price could have different base-currency cost bases if the FX rate moved." |

### Snippet-only (evaluated, not read in full)
| Source | Topic | Why not read in full |
|--------|-------|----------------------|
| trackyourportfol.io — *Portfolio Performance Across Multiple Currencies* | B | **WebFetch 503**. Search snippet is directly usable: "The cleanest method is to calculate performance in layers. **First, measure each holding in its local currency. Second, translate that local value into your base currency using a consistent FX source.**" → the LOCAL-first layering = convention (A). |
| Kantox — *Weighted Average Exchange Rate* | B | Glossary: WAER = blended FX weighted by transaction size; corroborates weighted-average mechanics; redundant with #6. |
| KPMG *Handbook: Foreign currency* (2026) / PwC Viewpoint / Deloitte DART ASC 830 §3.2 | B | IAS 21 / ASC 830 primary standards; paywalled/gated; the convention split is already covered by #5–#8. |
| Intuit TTLC — foreign-bond basis at acquisition vs sale FX | B | Community-tier; confirms per-lot acquisition-date FX; low weight. |
| Azure Architecture Center — *Saga Design Pattern* | A | MS Learn canonical; redundant with #1/#4. |
| Baeldung / Conduktor / Medium (Das, Thakur, Awad, Shalash) — Saga | A | Practitioner explainers; redundant with #1–#4 (isolation countermeasures, orchestration). |
| MDPI *Enhancing Saga Pattern …* (Appl. Sci. 12/6242) | A | Academic Saga enhancement; tangential to a single-process paper loop. |
| researchgate/themoonlight SagaLLM mirrors | A | Duplicates the arXiv full read (#2). |
| USPTO 11,874,822 — multi-stream transactional event processing | A | Patent; corroborates multi-leg atomicity challenge; not read in full. |

---

## 3. Recency scan (last 2 years) — MANDATORY SECTION

**New findings in the 2024–2026 window that complement/supersede older canon:**
1. **Pre-execution validation is now a first-class Saga step, not just post-hoc compensation** (SagaLLM,
   arXiv 2503.11951v3, Mar/Jul 2025): the classical Saga (Garcia-Molina & Salem, 1987 — year-less canon) is
   extended with an **independent GlobalValidationAgent that validates inputs + dependency satisfaction BEFORE
   any operation commits**, plus the explicit "either fully committed S′ or coherent rollback to S — avoiding
   partial/inconsistent outcomes" guarantee. This directly upgrades the 70.0 design: **do BOTH** a pre-flight
   validation (catch the knowable-ahead cash drop) AND compensation (for execution-time-only drops).
2. **2026 practitioner Saga guidance still centers compensation for un-pre-validatable failures** (oneuptime
   2026-02-20): "On failure, compensate all completed steps in reverse order"; compensation must be idempotent;
   escalate on compensation-failure. Confirms compensation is the load-bearing mechanism for the failures a
   pre-flight cannot see (our price-tolerance / FX-unavailable / live-price<=0 BUY drops).
3. **Multi-currency cost-basis guidance (2025–2026) converges on: pick ONE convention and never mix units.**
   AllInvestView 2026 states the anti-pattern verbatim — misalignment between **share quantities (local)** and
   **aggregate cost (base)** — which is *exactly* the pyfinagent `new_avg = USD_cost / local_shares` bug
   (§4a). The two live conventions (A: track avg price in LOCAL, derive base value via FX — Selinger,
   trackyourportfolio, CFA multiplicative decomposition; B: track cost in BASE with per-lot FX — Markdale,
   AllInvestView) are BOTH internally consistent; the defect is having a field that is convention-A on the
   first lot (:338) and convention-mix on the add-on (:308).

**Canonical prior art still valid:** Saga / compensating-transaction (Garcia-Molina & Salem, 1987) and the
IAS 21 / ASC 830 functional-currency framework remain foundational; the 2025–2026 work operationalizes them
(agentic pre-validation + explicit unit-consistency warnings), it does not overturn them.

**Internal recency anchor:** the `paper_swap_churn_fix` (phase-60.2, settings.py:344) is the in-house
precedent — a byte-identical-OFF correctness fix that was still flag-gated; the 70.3 flags follow that pattern.

---

## 4. Internal RE-ANCHOR (exact HEAD line numbers — 70.0 refs drifted +9)

Files inspected: `backend/services/portfolio_manager.py`, `backend/services/paper_trader.py`,
`backend/services/autonomous_loop.py`, `backend/config/settings.py`, plus the 70.0 handoff pack
(`design_trade_diversity_70.md`, `research_brief_70.0.md`, `confirmed_findings.json`) and the
`paper_swap_churn_fix` history (settings.py:344 + 59.3/60.2 comments).

**DRIFT NOTICE:** the 70.0 design's `594 / 620 / 675` refs have all drifted **+9 lines** on HEAD
`ec64e4ea`. Corrected anchors below.

### (S3-a) Swap candidate computation — `backend/services/portfolio_manager.py`
| Symbol | 70.0 ref | **HEAD ec64e4ea** | Note |
|--------|---------|------------------|------|
| `_compute_swap_candidates` **call** (gated by `paper_swap_enabled`) | ~:480 | **:480** (gate at :476) | unchanged |
| `_compute_swap_candidates` **def** | ~:498/507 | **:507** | signature :507–516 has **NO `available_cash` param** (finding #9) |
| Churn sentinel `score = 0.0` for un-reeval'd holding (flag OFF) | :562–566 | **:575** | inside `if _churn_fix_on: … continue` else `score = 0.0` (:548–575) |
| Same-sector-ONLY selection: `sector_holdings = holdings_by_sector.get(cand_sector, [])` | **:594** | **:603** | ← the cross-sector blocker (criterion 2) |
| Churn denom clamp: `denom = max(abs(holding_score), 1.0 if _churn_fix_on else 0.01)` | **:620** | **:629** | `delta_pct` at :630, `< min_delta` skip at :632 |
| Projected-sector-NAV check (computes a `buy_amount` for the projection only) | — | **:657–674** | `buy_amount = nav*(position_pct/100)` at :659 (projection) |
| SELL emit (`swap_for_higher_conviction`) | — | **:677–682** | sell-first-then-buy invariant |
| **Swap BUY sizing** `buy_amount = nav * (float(position_pct) / 100.0)` | **:675** | **:684** | ← **NO `min(available_cash)`, NO `<$50` floor** (criterion 1) |
| Swap BUY emit (`TradeOrder(action="BUY", …)`) | — | **:685–702** | |
| `sector_market_values` running update | — | **:706–708** | updates sector value but NOT a cash tracker |

**Contrast — the MAIN buy loop DOES cash-bound + $50-floor** (portfolio_manager.py):
- `target_amount = nav * (position_pct / 100.0)` **:387**
- `buy_amount = min(target_amount, available_cash)` **:388** ← cash bound
- `if buy_amount < 50: … skip` **:391–397** ← $50 floor
- running `available_cash -= buy_amount` **:459**
- `available_cash = cash + estimated_freed_cash - min_cash` **:167**; `min_cash = nav * (…)` **:91**

### (S3-b) Non-atomic execution — `backend/services/autonomous_loop.py`
| Symbol | 70.0 ref | **HEAD ec64e4ea** | Note |
|--------|---------|------------------|------|
| Execute step ("Step 7") | — | **:1306–1308** | `decide_trades` returns `orders` at :1290 |
| **All SELLs first** loop | :1262–1275 | **:1313–1330** | `if order.action != "SELL": continue` :1317; `trades_executed += 1` on success :1328 |
| **Then all BUYs** loop | :1284–1320 | **:1331–1374** | `if order.action != "BUY": continue` :1339; `if trade: trades_executed += 1` :1373–1374 |
| Live-price fetch + `price <= 0` drop | — | **:1341–1349** | a swap BUY can also drop here (price<=0) AFTER its SELL fired |
| Silent BUY-drop | — | **:1373** | `if trade:` — when `execute_buy` returns None, nothing increments, **no compensating action for the already-executed paired SELL** → net −1 position (finding #9) |

### (S3-c) `execute_buy` drop paths + avg-entry — `backend/services/paper_trader.py`
| Symbol | 70.0 ref | **HEAD ec64e4ea** | Note |
|--------|---------|------------------|------|
| `execute_buy` def | — | **:119** | |
| Price-tolerance gate `return None` (WARN-only) | :182 | **:182–188** | drop reason NOT knowable pre-flight |
| **Insufficient-cash `return None`** `if total_cost > cash:` | :197–199 | **:197–199** | WARN-only, no rollback |
| Max-positions `return None` | — | **:204–206** | |
| FX-unavailable `return None` | — | **:214–216** | non-US drop reason NOT knowable pre-flight |
| `quantity = (amount_usd * _usd_to_local) / price` → **LOCAL shares** | :217 | **:217** | quantity is in LOCAL shares; `price` is LOCAL |
| Idempotency-guard `return None` | — | **:226–244** | |
| **Add-on avg-entry block** | :303–319 | **:303–332** | see formula below |
| **First-lot** `avg_entry_price = price` (LOCAL price/share) | — | **:338** | sets the invariant: avg_entry_price is in LOCAL currency |
| `execute_sell` consumes avg-entry as LOCAL: `entry_price = avg_entry_price` :422; `realized_pnl_usd = (price − entry_price) * sell_qty * _l2u` **:472** | — | **:422, :472** | CONFIRMS avg_entry_price MUST be in LOCAL units |

---

## 4a. The avg_entry_price FX-unit bug — EXACT current formula + fix

**Invariant** (established by the first-lot at paper_trader.py **:338** `avg_entry_price = price`, and consumed
by `execute_sell` at **:472** `realized_pnl_usd = (price − entry_price) * sell_qty * _l2u`): `avg_entry_price`
is denominated in the position's **LOCAL currency per share**.

**Add-on path (existing position), paper_trader.py :304–308 — CURRENT (HEAD):**
```python
old_qty  = existing["quantity"]                                     # LOCAL shares
old_cost = existing["cost_basis"] or (old_qty * existing["avg_entry_price"])   # USD
new_qty  = old_qty + quantity                                       # LOCAL shares  (quantity :217 = local)
new_cost = old_cost + amount_usd                                    # USD  (old USD + new USD)
new_avg  = new_cost / new_qty                                       # ← BUG: USD / LOCAL-shares
```
`new_avg` is written to `avg_entry_price` at **:314**. It has units **USD-per-LOCAL-share**, but the invariant
and `execute_sell` treat `avg_entry_price` as **LOCAL-per-share**. For a non-US ticker (`_usd_to_local ≠ 1`)
this corrupts the position's entry price → wrong `unrealized_pnl` (:318–319) and wrong `realized_pnl_usd`
(:472) on the next partial sell. **Money-path corruption, non-US add-ons only.**

**Why US is unaffected (byte-identical):** for US, `_usd_to_local = 1.0` ⇒ `quantity = amount_usd / price` ⇒
`quantity * price = amount_usd`, and `old_cost = old_qty * old_avg` (USD==LOCAL), so
`new_avg = (old_qty*old_avg + amount_usd)/new_qty = (old_qty*old_avg + quantity*price)/new_qty` — identical to
the unit-consistent formula below.

**Unit-consistent fix (LOCAL-share-weighted average of LOCAL prices):**
```python
new_avg = (old_qty * existing["avg_entry_price"] + quantity * price) / new_qty
```
- `old_qty (local shares) * avg_entry_price (local price)` = old lot's LOCAL notional
- `quantity (local shares) * price (local exec price)` = new lot's LOCAL notional
- `/ new_qty (local shares)` = weighted-average LOCAL price → **unit-consistent**, matches the :338 invariant.
- Keep `cost_basis = new_cost` (USD) unchanged — cost_basis is correctly documented as USD (:330/:351).
- **Byte-identical for US** (proof above). Corrects non-US only.
- Use `price` (not `exec_price`) to stay consistent with how `quantity` (:217) and the first-lot (:338) are
  computed; in bq_sim mode `exec_price == price` anyway. (Note the alternative for fill-accurate books, but
  `price` preserves byte-identity + the existing invariant.)

Recommend gating behind **`paper_avg_entry_fx_fix_enabled`** (default OFF), mirroring the
`paper_swap_churn_fix_enabled` precedent (a byte-identical-OFF correctness fix that was still flag-gated).

---

## 5. Design recommendations (each with exact file:line)

Grounded in the §2 Saga + FX reads. Each recommendation below carries the exact HEAD file:line to touch.

### Flags (all default-OFF, byte-identical OFF) — add to `backend/config/settings.py` after :344
- **`paper_atomic_swap_enabled`** (bool, default False) — cash-bound swap BUY + $50 floor + pre-flight pair
  validation + paired-BUY-drop compensation.
- **`paper_cross_sector_rotation_enabled`** (bool, default False) — allow displacing the weakest-OVERALL
  holding across sectors iff projected portfolio HHI strictly decreases (dep: churn_fix ON).
- **`paper_avg_entry_fx_fix_enabled`** (bool, default False) — the LOCAL-weighted avg_entry fix (§4a).
- **Dependency guard (fail-safe):** `paper_atomic_swap_enabled` and `paper_cross_sector_rotation_enabled`
  REQUIRE `paper_swap_churn_fix_enabled` ON. If cross-sector is ON while churn_fix is OFF → **no-op the
  cross-sector path + WARN** (never rotate the whole book on fabricated 0.0 sentinels).

### 5.1 — Cash-bound the swap BUY + $50 floor (criterion 1, sizing) — `portfolio_manager.py`
- **Thread `available_cash` into `_compute_swap_candidates`**: add param at the def (**:507–516**) and pass it
  at the call site (**:480**). The main loop already has `available_cash` (**:167**); pass the same running
  value in. Also thread a running tracker so multiple swaps in one cycle are self-funding.
- **Fix the sizing at :684** (currently `buy_amount = nav * (float(position_pct) / 100.0)` — uncapped):
  ```python
  # mirror the main loop (:387-388, :391): cash-bound + $50 floor
  target_amount = nav * (float(position_pct) / 100.0)
  freed = weakest["market_value"]            # the paired SELL frees this (sell-first)
  buy_amount = min(target_amount, available_cash + freed)
  if buy_amount < 50:                        # $50 floor (same as :391)
      logger.info("Swap skip %s -> %s: buy_amount %.2f < $50 floor", weakest["ticker"], cand["ticker"], buy_amount)
      continue                               # drop the WHOLE pair (do NOT emit the SELL)
  available_cash = available_cash + freed - buy_amount   # running self-funding tracker
  ```
  Emit SELL (**:677–682**) and BUY (**:685–702**) only AFTER this passes → guarantees the pair is funded by
  construction (SagaLLM pre-execution input validation). NOTE: the SELL must not be appended before the floor
  check — restructure so both legs are appended together or neither (currently the SELL is appended at :677
  before the BUY sizing at :683 — move the floor/cash check ABOVE the SELL append).

### 5.2 — Atomic execution: pre-flight + compensation (criterion 1, atomicity) — `autonomous_loop.py`
The research is decisive: **sizing alone is NOT atomic.** `execute_buy` can still `return None` at execution
for reasons a pre-flight cannot know — price-tolerance (paper_trader.py **:182–188**), FX-unavailable
(**:214–216**), live-price<=0 (autonomous_loop **:1347–1349**), idempotency (**:226–244**), max-positions
(**:204–206**). The current all-SELLs-then-all-BUYs loop (**:1313–1374**) fires the swap SELL first, so ANY of
these drops the paired BUY → net −1 (finding #9). Two layers, both grounded:

**Layer 1 — pre-flight aggregate validation (SagaLLM "pre-execution validation", drop the knowable-cash case).**
Before Step-7 execution (insert at **~:1312**, before the SELL loop at :1313), simulate running cash over the
ordered `orders`: SELLs credit `market_value`, BUYs debit `amount_usd*(1+tx_pct)`. For any order tagged as a
swap BUY whose debit would drive running cash below `min_cash`, **remove BOTH its legs from `orders`** (the
paired SELL + the BUY) before anything executes. Pair identification: see 5.2a.

**Layer 2 — compensating restore on paired-BUY drop (microservices.io/Temporal/oneuptime, the load-bearing
atomicity mechanism).** For the residual execution-time-only drops: in the BUY loop (**:1338–1374**), when a
swap-tagged BUY returns `None` (**:1373** `if trade:` is falsy), run a **compensating restore of the paired
sold position** — the logical inverse of the SELL (microservices.io: "compensating transaction … logical
inverse", not a snapshot restore; Temporal: idempotent, `IfPresent`). Because this is a PAPER book backed by
BQ, the safest inverse is a **direct position re-insert** (re-write the `paper_positions` row captured
pre-SELL + re-debit the freed cash), which CANNOT fail on market data — unlike a market re-buy. Record
`summary["swap_compensations"]`. Net effect: **NEITHER leg persists → criterion 1 satisfied.**

**5.2a — Pair identification.** Add a `swap_group_id: Optional[str]` field to `TradeOrder`; set the same UUID on
the paired SELL (:677) and BUY (:685) in `_compute_swap_candidates`. The executor uses it to (i) pre-flight
drop both legs and (ii) find the SELL to compensate when its BUY drops. Capture the pre-SELL position snapshot
keyed by `swap_group_id` in the SELL loop so Layer-2 can restore it.

**Red→green test (criterion 1 proof):** unit-test `_compute_swap_candidates` + the execute path with a swap
whose paired BUY is forced to drop (e.g. monkeypatch `execute_buy → None`, or set a price-tolerance breach).
RED on HEAD: SELL persists, BUY absent → position count −1. GREEN after fix: compensation restores the sold
position (or pre-flight dropped both) → position count unchanged, `summary["swap_compensations"]==1`.

### 5.3 — Cross-sector rotation (criterion 2) — `portfolio_manager.py`
Gate behind `paper_cross_sector_rotation_enabled` (OFF → **:603 unchanged**, same-sector-only, byte-identical):
- When ON and no eligible SAME-sector holding exists (or to enable rotation), consider the **weakest-overall
  holding across ALL sectors** (min `final_score` over `holdings_by_sector` flattened), reusing the SAME
  churn-fix exclusion (un-reeval'd holdings skipped, :548–570) + clamped denom (:629) + 25% delta bar (:632)
  so cross-sector inherits churn safety.
- **Fire only if it strictly lowers portfolio HHI** (diversification improvement): compute HHI over sector
  NAV-weights pre- and post-swap; require `HHI_post < HHI_pre`. This is the "changeable fund" rotation and it
  aligns with the 70.2 soft-diversity north star.
- **Re-validate the destination-sector caps** (cross-sector CHANGES sector membership, unlike same-sector):
  re-check the sector COUNT cap (main-loop logic :364–375) and sector-NAV-pct cap (the projected-NAV check
  already at :657–674) for the BUY's sector on the projected composition. Blocking a cap-breaching rotation is
  a **fail-safe addition, not a threshold move**. The 25% delta bar and all caps are UNTOUCHED.

### 5.4 — avg_entry FX-unit fix (criterion 3) — `paper_trader.py:308`
Gate behind `paper_avg_entry_fx_fix_enabled` (OFF → :308 unchanged, byte-identical). When ON, replace
`new_avg = new_cost / new_qty` (**:308**) with the LOCAL-weighted average (see §4a):
```python
if getattr(self.settings, "paper_avg_entry_fx_fix_enabled", False):
    new_avg = (old_qty * existing["avg_entry_price"] + quantity * price) / new_qty   # LOCAL/share
else:
    new_avg = new_cost / new_qty                                                     # legacy (byte-identical)
```
`cost_basis = new_cost` (USD) stays unchanged (:315). Byte-identical for US (`quantity*price == amount_usd`);
corrects non-US add-ons. **Test:** a non-US (e.g. KR/EU) ticker, two add-on BUYs at a non-1.0 FX; assert
`avg_entry_price` equals the LOCAL-weighted price (in local units), and that `execute_sell` (:472) then yields
the correct `realized_pnl_usd`. US regression test: assert byte-identical avg_entry vs legacy.

### 5.5 — Reconciliation with `paper_swap_churn_fix` (task-required)
`paper_swap_churn_fix_enabled` (settings.py:344) repairs **fabricated-evidence comparisons** — it (1) EXCLUDES
un-reeval'd holdings from displacement (:548–570) and (2) clamps the delta denominator to
`max(abs(score),1.0)` (:629). The 70.3 flags are **orthogonal and compose on top of it**:
- churn_fix decides **which** holdings are eligible to displace and by what delta; atomic-swap ensures the
  resulting pair **executes atomically + cash-bounded**; cross-sector widens the **candidate holding set**
  (whole book) while REUSING churn_fix's exclusion+clamp so it inherits the same anti-churn safety.
- Hence the **hard dependency**: cross-sector rotation over the whole book without churn_fix would re-admit the
  81.4%-turnover churn engine on 0.0 sentinels. Ship 70.3 flags declaring `paper_swap_churn_fix_enabled` ON as
  a precondition (guard no-ops the cross-sector path + WARNs if violated).
- No conflict on the OFF path: churn_fix OFF = byte-identical pre-60.2; the 70.3 flags OFF = byte-identical
  current; both compose cleanly. The 25.0 delta threshold is UNTOUCHED (53.1/55.3 anti-band rulings respected).

### 5.6 — Fail-safe + do-no-harm audit (criterion 4)
- Every new drop/skip path **HOLDS or blocks** (skip the pair, no-op the cross-sector path, keep legacy
  avg_entry) — never corrupts the book. Layer-2 compensation RESTORES on partial failure.
- **NO risk threshold moved:** 25% swap-delta bar, sector count/NAV caps, $50 floor, stops, kill-switch,
  DSR≥0.95/PBO≤0.5 gates all byte-untouched. The $50 floor + cash-bound are the SAME limits the main loop
  already applies (:387–397), applied consistently to the swap — not new/moved thresholds.
- **$0 metered, paper-only, `historical_macro` FROZEN** — pure execution-path logic; no LLM calls, no BQ
  writes beyond the existing paper_* tables, no optimizer run.
- **DARK-until-token:** all three flags default OFF; ON-vs-OFF `$0` diff; activation is an operator decision.

---

## 6. Gate envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 14,
  "urls_collected": 45,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "gate_passed": true
}
```

`gate_passed: true` — 8 sources read in full via WebFetch (floor 5; tier mix: 2× Tier-1 peer-reviewed
[SagaLLM, CFA], 1× Tier-2 canonical [microservices.io], 2× Tier-3 authoritative [Temporal, Selinger], 3×
Tier-4 practitioner incl. 2× 2026 [oneuptime, AllInvestView]); recency scan present (§3, 3 findings);
3-variant queries disclosed per topic (§1); internal RE-ANCHOR complete with corrected HEAD line numbers
(§4, drift +9 confirmed) + the exact avg_entry formula & fix (§4a); design recommendations with exact
file:line + pseudocode (§5); reconciliation with `paper_swap_churn_fix` (§5.5); brief written write-first.
