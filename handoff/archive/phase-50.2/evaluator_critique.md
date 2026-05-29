# Q/A Evaluator Critique -- phase-50.2: Multi-currency portfolio accounting

**Verdict: PASS** | Date: 2026-05-30 | Fresh Q/A (first for 50.2, no self-eval) | merged qa-evaluator + harness-verifier (deterministic-first) | MONEY-CRITICAL step (touches the live +20% paper P&L engine)

The NON-NEGOTIABLE constraint -- USD-only path byte-identical -- is met and was independently re-proven on the LIVE portfolio (NAV $24,023.58, unchanged to the cent). No live USD-path regression exists.

---

## 1. Harness-compliance audit (5 items -- ALL PASS)

1. **researcher gate** -- PASS. `handoff/current/research_brief.md` is a 50.2 multi-currency brief, gate_passed=true (JSON envelope: `external_sources_read_in_full=8`, `recency_scan_performed=true`, `urls_collected=19`, `internal_files_inspected=11`). 8 sources read in full (CFA Institute, Karnosky-Singer/Meradia, IAS 21/CPDbox, CFI, arXiv 1611.01463 via ar5iv HTML, fundcount, sharesight, NetSuite) with the recency scan (2024-2026) present. Cited by `contract.md` lines 6-7, 49.
2. **contract-before-generate** -- PASS. `git log`: `ffdb8816 phase-50.2: PLAN` precedes `c452de61 phase-50.2: GENERATE`. The 4 success_criteria in contract.md (lines 27-30) are verbatim from masterplan step 50.2 `verification.success_criteria` (confirmed by extraction).
3. **results present** -- PASS. `experiment_results.md` present with file list (paper_trader.py + test_phase_50_2_multicurrency.py), verbatim verification output (7 passed; live NAV byte-identical), and `live_check_50.2.md` present with the numeric live re-proof.
4. **log-last** -- PASS. NO `phase=50.2` entry in `handoff/harness_log.md` yet (last entry is `Cycle 19 ... phase=50.1 result=PASS` at line 26014); masterplan 50.2 still `status: in_progress`. Correct ordering preserved.
5. **no verdict-shopping** -- PASS. This is the FIRST Q/A for 50.2. The on-disk `evaluator_critique.md` before this write was for 50.1 (PASS). No prior 50.2 CONDITIONAL/FAIL exists, so no simultaneous-presentation / 3rd-CONDITIONAL concern.

---

## 2. Deterministic checks (run independently -- ALL PASS)

```
ast.parse(paper_trader.py)                            -> AST_OK
pytest backend/tests/test_phase_50_2_multicurrency.py -> 7 passed in 0.81s
get_fx_rate('USD','USD')==1.0 + import paper_trader   -> det OK
test -f handoff/current/live_check_50.2.md            -> live_check_present
```

### THE CRITICAL ONE -- independent LIVE byte-identity re-proof (read-only, no mutation)

Re-ran the byte-identity proof myself against the real BQ portfolio (NOT trusting the generator's number):

```
positions 7  NAV_new 24023.58  NAV_old 24023.58  stored 24023.58  BYTE_IDENTICAL True
```

Every one of the 7 live positions has `market` US/NULL -> `_fx_local_to_usd(market)==1.0` -> `qty*current_price*fx == qty*current_price` to the cent. NAV recomputed the NEW way equals NAV the OLD way equals the stored `total_nav` ($24,023.58). **The working +20% engine is provably untouched.**

Helper-guard re-proof (the None/blank/US safety that underpins byte-identity):
```
_fx_local_to_usd(None)=1.0   _fx_local_to_usd('US')=1.0   _fx_local_to_usd('')=1.0
_fx_usd_to_local(None)=1.0   _fx_usd_to_local('US')=1.0
USD attribution (Fe=Fc=1.0): local_pnl=262.5  fx_pnl=0.0  zero_fx=True
```
The `market or "US"` guard in both helpers fires BEFORE `fx_rates.market_currency`, so the raw `AttributeError` on `market_currency(None)` is unreachable from every money site. All live positions confirmed `fx==1.0` (pnl path unchanged).

---

## 3. Code-review heuristics (5 dimensions evaluated -- no BLOCK, no WARN)

Diff scanned: `git diff ffdb8816 c452de61 -- backend/services/paper_trader.py` (+104) and the new test (+80).

- **Security (Dim 1):** no secrets in diff; no command/SQL/path/SSRF sink; no LLM-output-to-execution path added; no dep-pin removal. CLEAN.
- **Trading-domain (Dim 2):** NO risk guard removed -- grep on removed lines for `kill_switch|is_paused|stop_loss|paper_max_positions|backfill_stop` returned NONE. `crypto` not re-enabled. `perf_metrics` not bypassed (no inline Sharpe/drawdown/alpha). `execute_buy`/`execute_sell` ordering of kill-switch + stop-loss + max-positions guards is untouched (the diff only injects FX multipliers at money arithmetic, downstream of the guards). CLEAN.
- **Code quality (Dim 3):** no new `except Exception`/bare except; no `print()`; new module helpers carry full type hints + docstrings; new logger calls are ASCII. CLEAN.
- **Anti-rubber-stamp (Dim 4):** `financial-logic-without-behavioral-test` does NOT fire -- the GENERATE commit `c452de61` ships `test_phase_50_2_multicurrency.py` (+80, 7 tests) in the SAME commit as the paper_trader money-math change. No tautological assertions (tests assert numeric equalities to 1e-9 / 0.01, not `is not None`); not over-mocked (only `fx_rates.get_fx_rate` is patched, the module under test runs real). CLEAN.
- **LLM-evaluator anti-patterns (Dim 5):** N/A -- first 50.2 Q/A, no prior verdict to flip; this critique cites file:line + command output throughout (not a no-evidence pass). CLEAN.

`frontend/**` NOT touched -> ESLint/tsc gate N/A.

---

## 4. Adversarial money-site reasoning (the regression hunt)

For each money term I verified (a) USD reduces to exactly x1.0 and (b) each term is converted in exactly ONE place (no double-conversion). Units traced as USD vs LOCAL.

| Site | file:line | USD byte-identity | Single conversion? | Currency consistency |
|------|-----------|-------------------|--------------------|----------------------|
| BUY share count | :209 `(amount_usd*_usd_to_local)/price` | `_usd_to_local=1.0` -> `amount_usd/price` exactly | yes (USD->LOCAL once to size shares) | OK |
| BUY existing-branch MV/pnl | :299-301 `new_qty*price*_local_to_usd` | `_local_to_usd=1.0` -> `new_qty*price` | yes | `new_cost` USD (`old_cost+amount_usd`, :289); MV USD -> `pnl=MV_usd-cost_usd` |
| BUY new-branch | :321-323 `cost_basis=amount_usd`, `market_value=amount_usd` | already USD, no FX factor needed | n/a | USD - USD |
| SELL proceeds -> cash | :440 `new_cash += net_proceeds*_l2u` (sell_value=sell_qty*price LOCAL, :386-388) | `_l2u=1.0` -> proceeds unchanged | yes (LOCAL->USD once before crediting USD cash) | OK |
| SELL realized_pnl_usd | :440 `(price-entry_price)*sell_qty*_l2u` | `_l2u=1.0` -> unchanged | yes (LOCAL pnl ->USD once) | OK |
| SELL partial pos_row | :457-475 `_rem_cb` proportional USD; `_rem_mv=remaining*price*_l2u` | `_l2u=1.0` -> `remaining*price`; `_rem_cb` proportional of USD `_orig_cb` -> byte-identical | yes | `pnl=_rem_mv(USD)-_rem_cb(USD)` |
| MTM market_value/pnl | :520-523 `qty*live_price*_l2u`; `pnl=market_value-cost_basis` | `_l2u=1.0` -> `qty*live_price` | yes (LOCAL->USD once) | comment `# both USD` correct: MV USD, cost_basis USD |
| NAV | accumulation of USD market_values + USD cash | sums USD | n/a | OK |

**No double-conversion** anywhere: each money term is multiplied by an FX factor in exactly one location.

**cost_basis consistency (the trap):** the shipped minimal-change model stores `cost_basis` in USD (NOT local as the brief proposed). In `mark_to_market`, `market_value` is USD and `cost_basis` is USD, so `pnl = market_value - cost_basis` is USD - USD -- no currency mixing. For a non-USD position the entry FX is implicitly baked into the USD `cost_basis`, so `MV(@current FX) - cost(@entry FX)` still captures both the local price move and the FX move (the IAS-21/CFI result), just without an explicit `entry_fx_rate` column. Internally consistent.

**partial-sell:** `_rem_cb = _orig_cb * (remaining/quantity)` -- proportional remaining of the (USD) original cost. For USD this equals the pre-50.2 `remaining*avg_entry_price` because `_orig_cb` defaults to `quantity*avg_entry_price` when `cost_basis` is null, else uses the stored USD cost_basis. Byte-identical for USD; correct proportionality for non-USD.

**FX-None fail-soft (verified acceptable, not a defect):**
- BUY non-USD with FX unavailable -> `return None` (skip the buy) at :206-207 -- never treats None as USD. Correct.
- mark_to_market FX unavailable -> keeps last-known `market_value` + WARN at :513-518 -- never mis-marks. Correct.
- SELL FX unavailable -> last-resort 1.0 + WARN at :371-374 -- accepted: fires only for a non-USD exit when the rate is genuinely unsourceable; the design choice (never block an exit) is defensible for a paper engine; the USD path has `_l2u=1.0` always so this branch is never hit live. NOTE-level, not a money bug.

**attribution (no residual):** `fx_pnl_attribution` at :40-51: `local_pnl = qty*(Pc-Pe)*Fe`, `fx_pnl = qty*Pc*(Fc-Fe)`. Sum `= qty*(Pc*Fc - Pe*Fe) = MV_usd - cost_usd` algebraically with no residual; test `test_attribution_eur_sums_to_total_usd_pnl` asserts this (1320-1100=220 == 110+110). USD: `Fe=Fc=1.0` -> `fx_pnl=0.0` (asserted). Matches CFA `(1+R_local)(1+R_fx)-1` and CFI `FX gain = Foreign Amount*(Current-Transaction rate)`.

**Could ANY non-USD change leak into the USD path?** No. Every new term is gated by an FX factor that is exactly `1.0` for `market` in {US, "", None} via the `market or "US"` guard + `market_currency(...)=="USD"` early-return in both helpers. The live re-proof (NAV $24,023.58 == stored) is the empirical confirmation; the per-site trace above is the by-inspection confirmation. The non-USD path is additionally DORMANT in the live loop today -- grep confirms no caller passes `market=` to `execute_buy` (that wiring is phase-50.3), so `market` defaults to `"US"` -> x1.0 on every live trade.

---

## 5. Scope-honesty assessment

The experiment_results discloses a minimal-change model. Assessed each disclosure:
- **cost_basis stored USD (not local) + no `entry_fx_rate` column** -- honestly disclosed (experiment_results line 26, live_check line 46) with rationale (avoids a BQ migration) and a correctness argument (entry FX derivable as `cost_usd/(qty*avg_entry_price)`). Internally consistent (USD-USD pnl). Acceptable; not a criterion violation.
- **attribution not wired into the live `_compute_attribution` endpoint** -- honestly disclosed (it would be all-zero today since every live position is USD -> fx_pnl=0). Criterion #3's decomposition is satisfied by the tested `fx_pnl_attribution` helper + numeric evidence; criteria do not require live-endpoint wiring. Deferred to 50.6. Acceptable.
- **trade-record display fields (paper_trades.total_value/transaction_cost) stay LOCAL for a non-USD trade** -- honestly disclosed as display-only, NOT a NAV/cash error (NAV uses USD positions + USD-credited cash). Acceptable as a minor follow-up.

These are honest, bounded disclosures, not overclaims. None constitutes a money bug or a criterion violation.

---

## 6. Success-criteria mapping (4/4 met)

1. **paper_trader NAV/cost_basis/market_value/realized+unrealized P&L FX-convert each position to USD via fx_rates** -- MET. Conversions at BUY share-count (:209), existing-branch MV (:299), partial-sell pos_row (:457-475), SELL proceeds (:440) + realized_pnl_usd (:440), mark_to_market MV (:520) + NAV. cost_basis is USD (minimal-change model). All via the 50.1 `fx_rates` service through the two helpers.
2. **USD-only byte-identical** -- MET. Independently re-proven LIVE: NAV_new == NAV_old == stored == $24,023.58; 7 unit tests confirm x1.0 at every site; per-site by-inspection trace confirms exact reduction.
3. **non-USD values into USD NAV at correct FX + local-vs-FX P&L decomposition** -- MET. EUR example: 5 sh @ EUR100 -> $550 USD (= 5*100*1.10); `fx_pnl_attribution` decomposes per the arXiv/CFA model with no residual (tested).
4. **live/fixture numeric evidence of a EUR USD-NAV contribution + local/FX split** -- MET. live_check_50.2.md section 3: EUR $550 NAV contribution; attribution (EUR 100->110, FX 1.10->1.20) local_pnl $110 + fx_pnl $110 == $220 == MV_usd($1320) - cost_usd($1100).

---

## 7. NOTE (non-blocking; recommended follow-up doc clarification)

`mark_to_market`'s `unrealized_pnl_pct` (:523) = `pnl/cost_basis*100` with both terms USD -> it is a **USD-return %**, whereas the contract (lines 11, 52) and brief described it as a **local-return %**. Under the shipped minimal-change model (USD cost_basis) the local-return % is not directly available without the entry FX. This is:
- **Zero live impact** -- every live position is USD, so local-return == USD-return identically.
- **NOT one of the 4 immutable success criteria** -- the criteria name NAV/cost_basis/market_value/realized+unrealized P&L conversion and the local-vs-FX *decomposition* (delivered by `fx_pnl_attribution`), not the `_pct` field's reference currency.
- A **documentation/clarification gap** between the brief's intended `_pct` semantic and the minimal model's. Recommend a one-line note when 50.6 wires attribution into the UI: either relabel `unrealized_pnl_pct` as a base-currency return, or compute a local-return % from `avg_entry_price` (LOCAL) + `current_price` (LOCAL), which are both available on the row.

Severity NOTE -> PASS-with-flag (per skill severity dispatch). Does NOT degrade the verdict.

---

## Verdict

**PASS.** All 4 immutable criteria met; the MONEY-CRITICAL byte-identity constraint independently re-proven LIVE (NAV $24,023.58 unchanged to the cent); no double-conversion, no currency mixing in any USD-feeding term; FX-None fail-soft is sound; attribution is exact (no residual); the financial-logic change ships with a behavioral test; all scope deviations are honestly disclosed and non-blocking. The one `unrealized_pnl_pct` local-vs-USD semantic is a zero-live-impact NOTE / doc-clarification follow-up, not a criterion violation or money bug.

checks_run: syntax, verification_command, live_byte_identity_reproof, fx_helper_guards, code_review_heuristics, research_brief, evaluator_critique
