# Design Pack — phase-69 Audit Burn-down (step 69.0)

**Status**: DESIGN ONLY. No production code changes in this step (69.0). This pack is the
implementation contract for the code steps 69.1 (book-safety), 69.2 (gate correctness),
69.3 (signal integrity). Every element names its exact file:line target and the do-no-harm
invariant it preserves.

**Boundaries (binding, restated)**: $0 metered, free APIs only, paper-only; kill-switch limits
(4% daily / 10% trailing / 8% stop / 30% sector), DSR≥0.95, PBO≤0.5 are **byte-untouched**;
fixes are fail-safe + ledger-math only; hysteresis stays banned; historical_macro stays frozen;
every guard-behavior change ships DARK behind an operator token.

**Research basis**: `handoff/current/research_brief_69.0.md` (gate_passed=true; 8 sources read in
full). Reference anchors cited inline.

---

## 1. FX degradation chain (book-safety; register item 1) — target 69.1

### Current defect (verified 2026-07-11)
- `fx_rates._usd_value_live` (`backend/services/fx_rates.py:78-104`): cache(6h TTL) → `_fetch_yf`
  → `_fetch_fred` → `return val`. When **both** yfinance and FRED fail, `val` is `None`. The live
  path **never** reads the `historical_fx_rates` table it write-through-persists at `:103`.
- `paper_trader.execute_sell` (`backend/services/paper_trader.py:388-392`): `_l2u =
  _fx_local_to_usd(position.market)`; `if _l2u is None: logger.warning("...crediting at 1.0"); _l2u
  = 1.0`. The `_l2u` factor multiplies into: `net_proceeds` credit (`:506`), trade `total_value`
  (`:433`), `transaction_cost` (`:434`), `realized_pnl_usd` (`:460`). So a KRW exit during a dual-FX
  outage books ~1300× phantom USD cash (and poisons the monotonic kill-switch peak → spurious
  full-book flatten); an EUR exit leaks ~14%.
- **Asymmetry**: `execute_buy` (`:212-217`) already **blocks** (`return None`, "skipping BUY") when
  FX is unavailable — the fail-closed mirror. `mark_to_market` (`:532-540`) already keeps last-known
  market value on FX outage — an in-repo last-known precedent.

### Design (fail-closed waterfall; reference: Modern Treasury native-currency posting, US Treasury "use a verifiable rate + record the source")
Introduce a **last-known-rate degradation chain** in `_usd_value_live(ccy)`:

```
_usd_value_live(ccy):
    if ccy == USD: return 1.0
    hit = api_cache.get(fx:usd_value:ccy)           # (A) fresh 6h cache — unchanged
    if hit is not None: return hit
    val = _fetch_yf(ccy)                             # (B) live primary — unchanged
    if val is None: val = _fetch_fred(ccy)          # (C) live fallback — unchanged
    if val is not None:
        api_cache.set(...); _persist(ccy, today, val, "yfinance/fred-live")   # unchanged
        return val
    # NEW (D) last-known-good, read DIRECTLY from historical_fx_rates (NOT via _usd_value_asof):
    lk = _last_known_usd_value(ccy)                  # SELECT rate ... WHERE pair=@pair ORDER BY date DESC LIMIT 1
    if lk is not None:
        logger.warning("fx_rates: %s live+FRED down; serving last-known %.6f (stale)", ccy, lk)
        return lk
    return None                                      # only when NO rate was EVER stored
```

- `_last_known_usd_value(ccy)` is a **new dedicated helper** that queries `historical_fx_rates`
  directly (mirroring the SQL already in `_usd_value_asof:164-172` but with no date bound and no
  degrade-to-live branch). **Do NOT** call `_usd_value_asof` from `_usd_value_live` — `_usd_value_asof`
  degrades back to `_usd_value_live` (`:160,176,179`), which would create **mutual recursion**. This is
  the single most important implementation pitfall.
- **execute_sell rule** (`paper_trader.py:388-392`): replace the unconditional `_l2u = 1.0` with:
  - if `_fx_local_to_usd` returns a rate (now almost always true, since any prior BUY persisted one and
    the live path serves last-known) → use it.
  - if it returns `None` (no rate EVER stored for this market) → **BLOCK the exit and PAGE P1**
    (fail-closed), never credit at 1.0. A bounded staleness error strictly dominates an unbounded
    ~1300× parity error; a stranded stop-loss exit is surfaced to the operator rather than silently
    corrupting the ledger. (For a market the engine actively trades, this residual is effectively
    unreachable — a rate is persisted on every BUY and every successful live fetch.)

### Do-no-harm invariant
No threshold changes. USD path stays byte-identical (`ccy==USD → 1.0`, `from==to → 1.0`). The change
is purely additive fallback + replacing a silent 1.0 with last-known-else-block. The `execute_buy`
behavior is unchanged; `execute_sell` moves from fail-open-at-1.0 to fail-closed-at-last-known —
strictly safer for the ledger.

### 69.1 reproduction tests (red→green)
- Monkeypatch `_fetch_yf` and `_fetch_fred` to return `None`; seed one `historical_fx_rates` KRW row;
  assert a KR SELL credits at the **last-known** rate (≈ 1/1300 USD-per-KRW), NOT 1.0 (RED today: credits 1.0).
- With NO stored rate at all, assert the SELL is **blocked + paged**, not credited at 1.0.

---

## 2. Kill-switch audited peak-reset state machine (book-safety; register items 1-2) — target 69.1

### Current defect (verified 2026-07-11)
- `update_peak` (`kill_switch.py:212-217`): "Ratchet ... upward. Never moves down." No reset event
  exists anywhere. After a flatten-to-cash, NAV ≈ cash ≪ peak, so trailing-DD stays breached forever.
- `resume()` (`:184-194`) clears `_paused` but leaves `_peak_nav`; `check_auto_resume` (`:275-333`)
  refuses while `any_breached` (`:312-314`). ⇒ both resume paths refuse forever once flattened =
  **permanent lockout**.
- `evaluate_breach` (`:230-264`) guards the denominators (`sod>0`, `peak>0`) but **not** `current_nav`;
  a caller `or 0.0` (BQ timeout) yields `daily_loss_pct=(sod-0)/sod=100%` and `trailing_dd=100%` — a
  **phantom full breach** on a transient 5s timeout.
- **Existing infrastructure to build on**: `_load_from_audit` (`:61-106`) already event-sources
  `pause / resume / auto_resume_alert / sod_snapshot / peak_update` from `handoff/kill_switch_audit.jsonl`
  and restores `_peak_nav` at `:104`. So the reset is a NEW event on the SAME append-only stream, not
  new infrastructure.

### Design (reference: Fowler "ops staff can reset; log every state change"; MS circuit-breaker "manual override resets counter + durable state")

**New audit event `peak_reset`** (guard-behavior change ⇒ DARK until operator token `KS-PEAK-RESET: APPROVED`):

- **Emit sites (2, both authorized)**:
  1. **On flatten-to-cash** (the kill-switch flatten path, after positions → cash): the pre-flatten peak
     is meaningless against a 100%-cash NAV. Emit `peak_reset(old_peak, new_peak=post_flatten_nav,
     trigger="flatten")` and set `_peak_nav = post_flatten_nav`.
  2. **On operator resume** (`resume(trigger="manual")`): re-anchor `_peak_nav = current_nav` so the
     trailing-DD denominator reflects the resumed book. Emit `peak_reset(old_peak, new_peak=current_nav,
     trigger="operator_resume", operator=<id>)`.
- **Replay branch** in `_load_from_audit`: on a `peak_reset` row, set `_peak_nav = row.new_peak`
  (processed in stream order alongside `peak_update`). This makes the reset **restart-replayable** and
  **idempotent**: replaying the same audit stream yields the same `_peak_nav` regardless of restarts.
- **Audit row shape**: `{event:"peak_reset", ts, old_peak, new_peak, trigger, operator?}` appended via
  the existing `_append_audit` so it lands in `handoff/kill_switch_audit.jsonl`.
- **DARK gating**: gate the emit sites behind a settings flag (e.g. `kill_switch_peak_reset_enabled`,
  default False) that flips only when `KS-PEAK-RESET: APPROVED` is recorded. Until then behavior is
  byte-identical to today (no reset fires). The replay branch may ship live (it is a no-op with zero
  `peak_reset` rows in the stream).

**`current_nav<=0` null-breach guard** (fail-safe data-sanity, NOT a threshold change) — `evaluate_breach`:
- At the top of `evaluate_breach`, if `current_nav is None or current_nav <= 0`: return a
  no-breach result with a diagnostic flag (e.g. `"nav_invalid": True`) instead of computing a 100%
  breach. A funded paper book's NAV is never ≤0, so this only suppresses breaches on invalid/absent
  input (the BQ-timeout `or 0.0`). This is fail-safe but suppresses a (false) breach, so it is
  **surfaced for operator awareness** in the live_check even though it is not a threshold change and
  does not require the DARK token.

### Do-no-harm invariant
The 4%/10%/8%/30% thresholds are byte-untouched. `update_peak`'s monotonic ratchet is unchanged; the
reset is an *additional*, operator-gated event. Restores the documented "human resume once healthy"
behavior. The `current_nav<=0` guard makes the switch fire on REAL breaches only — it cannot mask a
true breach (a real NAV is >0).

### 69.1 reproduction tests (red→green)
- Simulate flatten → assert (with flag ON) a `peak_reset` audit row lands and a subsequent operator
  resume leaves `any_breached=False`; assert (flag OFF, default) NO reset fires (dark-by-default).
- Feed `evaluate_breach(current_nav=0.0, ...)` → assert `any_breached=False` + `nav_invalid` (RED today:
  returns 100% daily+trailing breach).
- Kill-switch-audit replay test: append a `peak_reset` row, reload via `_load_from_audit`, assert
  `_peak_nav` == the reset value (restart-replay determinism).

---

## 3. Sign-safe overlay algebra (signal integrity; register item 6) — target 69.3

### Current defect (verified 2026-07-11)
Overlays multiply a **signed** momentum composite by a tilt:
- `news_screen.apply_news_to_score` (`:329-332`): `base_score * 1.10` (positive) / `* 0.90` (negative).
- `macro_regime.apply_regime_to_score` (`:542-549`): `score = base_score * conviction_multiplier`;
  then `score *= 1.05` (overweight) / `*= 0.95` (underweight).
- Same multiplicative-on-signed pattern in the pead / options / insider / peer_leadlag overlays.

For a **negative** base (broad drawdowns), `score*1.10` is MORE negative ⇒ a positive catalyst
DEMOTES the name and a negative catalyst PROMOTES it — the inversion the register flags.
Reference: Elastic BM25 — multiplicative boosts "implicitly assume positive base values for the ratio
to remain meaningful"; FTSE Russell factor tilts presume a non-negative score domain.

### Design (sign-aware additive tilt — ONE unified expression)

```
def sign_safe(score: float, mult: float) -> float:
    return score + abs(score) * (mult - 1.0)
```

**Proof it preserves intended ranking in both regimes.**
- `score ≥ 0`: `|score| = score` ⇒ `score + score*(mult-1) = score*mult`  → identical to today's intent.
- `score < 0`:  `|score| = -score` ⇒ `score - score*(mult-1) = score*(2-mult)`.
- Monotonicity of a boost (`mult>1`) and penalty (`mult<1`):

  | score | mult | today (`score*mult`) | sign-safe | effect |
  |------:|-----:|---------------------:|----------:|--------|
  |  +10  | 1.10 | 11.0 | 11.0 | boost raises ✓ |
  |  −10  | 1.10 | −11.0 (inverted) | −9.0 | boost raises rank ✓ |
  |  −10  | 0.90 | −9.0 (inverted) | −11.0 | penalty lowers rank ✓ |
  |  +10  | 0.90 | 9.0 | 9.0 | penalty lowers ✓ |

  In both sign regimes, `∂score_out/∂mult = |score| ≥ 0`, so a larger multiplier ("boost") never lowers
  the score and a smaller one ("penalty") never raises it. Inversion eliminated.
- Preferred over **clamp-to-no-op** (`if score<=0: return score`) because clamping discards the catalyst
  exactly in the drawdown regime where selection carries the most information.

**Application sites** (69.3, each behind a flag with an ON-vs-OFF live_check): `news_screen:329`,
`macro_regime:547` (sector tilt) and `:542` (conviction_multiplier), pead / options / insider /
peer_leadlag overlays. Chain multiple overlays by composing `sign_safe` per factor (order-independent
in sign, since each step preserves sign).

### Do-no-harm invariant
Positive-base behavior is byte-identical to today (`score*mult`). Only negative-base ranking changes —
and it changes from *provably wrong* to *provably correct*. Flag-gated + ON-vs-OFF comparison so the
live ranking shift is operator-visible before it is trusted. No threshold touched.

### 69.3 reproduction test (red→green)
Two candidates with equal negative base score; one gets a positive catalyst, one a negative catalyst.
Assert (post-fix) the positive-catalyst candidate ranks ABOVE the negative-catalyst one (RED today: the
order is inverted).

---

## 4. Promotion-gate corrections (offline; register item 5) — target 69.2

All four are in the offline backtest/analytics layer — **zero live-money surface**. Immutable
DSR≥0.95 / PBO≤0.5 thresholds are byte-untouched; these make the gate *measure what it claims*.

### 4a. DSR unit correction — `analytics.compute_deflated_sharpe` (`:292-335`), caller `:654-661`
- **Bug**: `generate_report` (`:654-661`) passes `observed_sr = aggregate_sharpe` (ANNUALIZED, `mean/std*
  sqrt(252)`, `:129-144`) with `T = len(daily_returns)` (DAILY) and `variance_of_srs` = variance of
  ANNUALIZED window Sharpes (`:642`). The SE `sqrt((1 − γ3·SR + (γ4−1)/4·SR²)/T)` (`:323-325`) and the
  E[maxSR] term require **per-period** SR and V to match a per-period T.
- **Formula (Bailey & López de Prado, JPM 2014 / SSRN 2460551, read in full)**:
  `DSR = Φ[ (SR̂ − SR*)·√(T−1) / √(1 − γ3·SR̂ + ((γ4−1)/4)·SR̂²) ]`, γ4 = RAW kurtosis (Normal=3),
  `SR* = √V·[(1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(N·e))]`, γ=0.5772. SR̂ and SR* must be per-period.
- **Fix**: de-annualize BOTH consistently before the SE/z computation —
  `SR_p = SR_ann / √ppy`, `V_p = V_ann / ppy` (V scales by 1/ppy because SR scales by √ppy); keep skew &
  raw-kurtosis from per-period (daily) returns as today; use per-period T (prefer T−1). `ppy` = the
  same `periods_per_year` used to annualize (252/250).
- **Reference pin (unit test target, independently re-derived — matches the paper's numerical example)**:
  inputs `SR_ann=2.5, T=1250, N=100, V=0.5, skew=−3, kurt=10, ppy=250` →
  `SR_p=0.158114, SR*_p=0.113163, z=1.2841 → DSR=0.9004`. Secondary: `N=46 → 0.9505`; Normal
  (skew=0,kurt=3) reaches 0.95 at `N=88`. **Assert the pre-fix path** (annualized SR + T=1250) `→
  DSR≈0.9999999` so the test both proves the fix and documents the ~√ppy≈√252 inflation.
- **Do-no-harm**: the DSR≥0.95 gate constant is untouched; only the *statistic* is corrected. (Incumbent
  re-validation under corrected gates needs the historical_macro un-freeze — a **separate operator
  token**, out of 69.2 scope; 69.2 ships code + fixtures only.)

### 4b. Purge + embargo — `_build_training_data` (`:566-598`), `walk_forward.py:61`
- **Bug**: no purge; a fixed `embargo_days=5` gap (`walk_forward.py:61`, engine `:148`) vs a triple-barrier
  label horizon of `holding_days` scanned to `holding_days×1.5` (up to ~135d, `:658`). Training samples
  with `sample_date ∈ [test_start − 1.5·holding_days, train_end]` have labels computed from prices INSIDE
  the test window ⇒ leakage. Note `exit_dates` at `:596` uses `holding_days`, UNDERSTATING the horizon.
- **Fix (AFML Ch.7, López de Prado GARP whitepaper read in full)**: purge every training sample whose
  label interval `[sample_date, sample_date + 1.5·holding_days]` overlaps `[test_start, test_end]`
  (the 3-condition overlap test); stamp `exit_dates` with the TRUE `1.5·holding_days` horizon; retain an
  embargo (≈0.01·T) as a post-test gap. `sample_dates`, `entry_dates`, `exit_dates` already exist for
  sample-weighting, so the purge reuses them.
- **Do-no-harm**: purely removes leaked training rows; no threshold, no live surface.
- **Fixture test**: assert no retained training sample's `[entry, entry+1.5·holding_days]` overlaps
  `[test_start, test_end]`.

### 4c. Boundary snap — `backtest_engine.py:486-490` (entry) + `:512-516` (liquidation)
- **Bug**: `cache.cached_prices(ticker, test_start_str, test_start_str)` exact-date lookups return empty
  on weekend/holiday boundaries ⇒ the window executes zero trades and `close_all_positions` liquidates at
  ENTRY price.
- **Fix**: business-day-snap the boundary date to the nearest prior date WITH data (or widen the lookup to
  a small `[d-5, d]` range and take the last available close). Apply at both the entry (`:488`) and
  liquidation (`:512-516`) lookups. (The daily mark-to-market loop at `:501-507` already iterates
  `pd.bdate_range` and tolerates per-day misses via last-known; the defect is specifically the boundary
  entry/exit price.)
- **Fixture test**: a Sat/Sun-bounded window now fills trades and liquidates at a real close, not entry price.

### 4d. Fracdiff-at-predict — TRAIN `:628-637` vs PREDICT `:793-801`
- **Bug**: fracdiff is applied to `_NON_STATIONARY` cols at train time (`:628-637`, `X.fillna(X.median())`
  then 0) but NOT at predict time (`:793-801`, `fv.get(f,0)` + `.fillna(0)`), and the NaN-fill policy
  differs (train=median, predict=0). The model trains on small frac-diff values and predicts on raw levels.
- **Fix**: apply the SAME fracdiff transform at predict time using the **train-time fitted** fracdiff
  weights and the **train-time medians** (persist them from the training pass), and use the same
  median-fill policy at predict. Do NOT recompute fracdiff on a single prediction row (statistically
  meaningless over one observation).
- **Do-no-harm**: makes train/predict feature distributions consistent; offline only.
- **Fixture test**: assert the predict-time transform of a known feature vector equals the train-time
  transform (same weights, same fill).

### 4e. Go-live booleans — `paper_go_live_gate.py:111` (register item 5; 69.2)
- Tighten the two under-spec booleans to their documented immutable definitions: `psr_ge_95_sustained_30d`
  → per-day PSR sustainment across 30d (not a point-in-time n≥30 check); `max_dd_within_tolerance` →
  realized-DD vs backtest-DD+5pp (not a last-30-snapshot scan). Fixture-tested. The go-live CRITERIA are
  the documented spec — this makes the code MATCH the spec, it does not relax it.

---

## 5. Do-no-harm ledger (criterion 3)

| Immutable / guard | Value | This pack's treatment |
|---|---|---|
| Daily-loss limit | 4% | **byte-untouched** (no edit anywhere in 69.x) |
| Trailing-DD limit | 10% | **byte-untouched** |
| Stop-loss | 8% | **byte-untouched** |
| Sector cap | 30% | **byte-untouched** |
| DSR promotion gate | ≥0.95 | **byte-untouched** (69.2 fixes the statistic, not the constant) |
| PBO promotion gate | ≤0.5 | **byte-untouched** |
| hysteresis | banned | not introduced |
| historical_macro | frozen | not written (69.3 INDPRO/liquidity uses a NEW cached path, separate step) |

**Guard-behavior changes → DARK until operator token:**
- Kill-switch `peak_reset` (§2) → DARK until `KS-PEAK-RESET: APPROVED`.
- Sign-safe live overlay ranking (§3) → flag-gated + ON-vs-OFF live_check (operator-visible before trusted).

**Operator-awareness (not a threshold change, no token required, but surfaced):**
- `current_nav<=0` null-breach guard (§2) — suppresses a FALSE breach on invalid NAV only.

---

## 6. Downstream step map (implementation queue)

| Step | Scope | Live surface | Sequencing |
|---|---|---|---|
| 69.1 | §1 FX chain + §2 kill-switch peak-reset/guard + clear-queue pkill removal + lock safety | money-path | **byte-coordinate with phase-68** (68.5 fill-price gate shares paper_trader) |
| 69.2 | §4 DSR + purge/embargo + boundary + fracdiff + go-live booleans | offline (zero live) | independent of phase-68; ships code+fixtures; incumbent re-validation waits on historical_macro un-freeze token |
| 69.3 | §3 sign-safe overlays + news cap/retry + QMJ + INDPRO/liquidity (new cached path) | live, flag-gated | ON-vs-OFF live_check; final IC/ablation waits on historical_macro un-freeze token |

Register items NOT owned by 69.1-69.3 (learn-loop tz TypeError → 68.4; external-flow/deposit Sharpe +
STRING/TIMESTAMP query → 68.5/68.6; FX-1 residual → parked 61.3; 30 contested + Slack/UI defects → 63.3
seeds) are filed in 69.4 with a coverage table — no execution here.
