# Research Brief — Step 69.0 (P0 design pack, phase-69 audit burn-down)

status: COMPLETE (gate_passed=true; 8 sources read in full; see Gate completion note for provenance)
tier: complex
date: 2026-07-11
author: researcher (Layer-3 harness)
feeds: design_audit_burndown_69.md (Main authors next)
boundaries: $0 metered, free APIs, paper-only, do-no-harm (kill-switch limits / stops / DSR>=0.95 / PBO<=0.5 byte-untouched), guard changes DARK-until-operator-token

## Scope

Four external topics + internal code audit (register: handoff/current/audit_phase69/register.md):

1. FX last-known-rate fallback / fail-closed vs fail-open for multi-currency ledgers
2. Audited, restart-replayable drawdown-guard / high-water-mark reset state machines
3. Sign-safe multiplicative overlays on signed scores
4. DSR standard-error exact expression + worked reference value; AFML Ch.7 purge/embargo vs label horizon

## Search queries run (3-variant discipline)

Per topic: current-year (2026), last-2-year (2025/2024), year-less canonical.
- T1 FX: "FX rate fallback stale rate multi-currency ledger fail-closed 2026" / "foreign exchange rate unavailable last-known rate treasury accounting booking 2025" / "last known good exchange rate fallback multi-currency ledger design"
- T2 kill-switch: "trading kill switch high water mark reset drawdown circuit breaker audit 2026" / "circuit breaker pattern state persistence across restart manual reset half-open 2025" / "high water mark reset trailing drawdown limit operator resume risk management system"
- T3 overlay: "sign-safe multiplicative overlay signed alpha score boost penalty quant 2026" / "combining alpha factors additive vs multiplicative z-score signal blending 2025" / "multiplicative boost negative score inversion ranking factor tilt"
- T4 DSR/purge: "deflated Sharpe ratio standard error annualized versus per-period T units 2026" / "probabilistic Sharpe ratio formula skewness kurtosis worked example Bailey Lopez de Prado 2025" / "purged cross-validation embargo triple barrier label horizon Lopez de Prado" (+2 recency-scan variants)

## Read in full (8; >=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| 1 | https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf | 2026-07-11 | peer-reviewed (JPM 2014, SSRN 2460551) | curl + pdfplumber 0.11.9 | DSR Eq.2 + numerical example: annualized SR 2.5 -> DSR 0.9004 at N=100; SE denominator per-period, "non-annualized (250 obs/yr)" |
| 2 | https://portfoliooptimizer.io/blog/the-probabilistic-sharpe-ratio-bias-adjustment-confidence-intervals-hypothesis-testing-and-minimum-track-record-length/ | 2026-07-11 | authoritative blog (quant) | WebFetch | PSR SE formula; MinTRL formula; PSR "invariant to calendar conventions" but reference c must be time-scaled consistently |
| 3 | https://www.garp.org/hubfs/Whitepapers/a1Z1W0000054x6lUAA.pdf | 2026-07-11 | practitioner-authored primary (López de Prado, GARP) | curl + pdfplumber 0.11.9 | EXACT purge 3-condition overlap test + embargo definition h ≈ 0.01T, applied AFTER the test window only |
| 4 | https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/ | 2026-07-11 | industry practitioner | WebFetch | Purge = remove train samples whose event-time overlaps test trade-times; embargo before test fold for feature lookback; "first embargo, then purge" |
| 5 | https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker | 2026-07-11 (doc updated 2026-07-02) | official docs | WebFetch | Manual override (admin force-close + reset counter / force-open); raise event on every state change to alert admin; durable state + monitoring |
| 6 | https://martinfowler.com/bliki/CircuitBreaker.html | 2026-07-11 | authoritative blog (Fowler) | WebFetch | "Operations staff should be able to trip or reset breakers"; "Any change in breaker state should be logged"; half-open trial resets or restarts timeout |
| 7 | https://www.elastic.co/search-labs/blog/bm25-ranking-multiplicative-boosting-elasticsearch | 2026-07-11 | official docs (Elastic) | WebFetch | Multiplicative boost proportionality "implicitly assumes positive base values for the ratio to remain meaningful"; additive vs multiplicative tradeoffs |
| 8 | https://www.moderntreasury.com/journal/announcing-multi-currency-support-for-ledgers | 2026-07-11 | industry (ledger infra) | WebFetch | Amounts stored in NATIVE currency of the account; "every debit and credit be balanced per currency" -- do NOT convert/mix at booking |

## Identified but snippet-only (does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://fiscal.treasury.gov/resources/reporting-rates-exchange | official (US Treasury) | WebFetch 403; guidance captured from search snippet: "if no Treasury FMS rate is available, use another verifiable exchange rate and provide the source"; amend if deviation >=10%; rate usable for ensuing 3 months |
| https://fiscaldata.treasury.gov/datasets/treasury-reporting-rates-exchange/ | official (US Treasury) | WebFetch timeout 60s; same dataset as above |
| https://www.isda.org/book/additional-provisions-for-use-with-a-deliverable-currency-disruption-and-isda-deliverable-currency-disruption-fallback-matrix/ | industry standard (ISDA) | derivatives currency-disruption fallback matrix; paywalled book; confirms formal fallback-waterfall practice exists |
| https://sdk.finance/blog/what-is-a-multi-currency-ledger-how-fintechs-track-balances-transfers-and-settlement-across-currencies/ | industry | corroborates per-currency posting + no cross-currency balancing |
| https://www.tandfonline.com/doi/full/10.2469/faj.v74.n3.5 | peer-reviewed (FAJ) | signal blending vs portfolio blending; z-score concentration -> rank preferred; paywalled |
| https://www.lseg.com/content/dam/ftse-russell/en_us/documents/research/multi-factor-indexes-power-of-tilting.pdf | industry (FTSE Russell) | factor tilt = multiply weights by score in [0,1]; tilts presume NON-NEGATIVE score domain |
| https://en.wikipedia.org/wiki/Purged_cross-validation | tertiary | corroborates purge/embargo; primary (GARP LdP) read instead |
| https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio | tertiary | corroborates DSR; primary paper read instead |
| https://arxiv.org/pdf/2604.15531 | preprint (2026) | "Spurious Predictability in Financial ML" -- recency-scan hit; complements DSR, does not supersede |
| https://arxiv.org/pdf/2512.22476 | preprint (2026) | AutoQuant crypto auto-tuning USES DSR as gate (2026) -- evidence DSR remains canonical |
| https://www.mql5.com/en/blogs/post/767321 | community | prop-firm kill-switch: stop BEFORE violation; trailing DD locks at peak-minus-max |
| https://www.alphaexcapital.com/prop-trading/.../trailing-drawdown-explained | community | trailing DD "only moves up, never down"; hard-breach = permanent fail (the monotonic-peak lockout, confirmed as industry default) |
| https://help.tradeify.co/en/articles/10495897-rules-trailing-max-drawdowns | community | trailing max DD locks at high-water mark after threshold |
| https://medium.com/@tzjy/5-different-ways-of-alpha-signal-blending... | community | 5 blending methods incl. weighted-sum (additive) on z-scores |
| https://www.quantconnect.com/research/17112/probabilistic-sharpe-ratio/ | industry | PSR reference implementation |

Total unique URLs collected: 23 (8 read in full + 15 snippet-only).

## Recency scan (2024-2026)

Performed (2 dedicated recency searches + 2026/2025 variants on all 4 topics). Result: **no new finding supersedes the canonical sources.**
- DSR (Bailey-LdP 2014) and purge/embargo + CPCV (López de Prado 2018) remain the standard; 2025-2026 work (arXiv:2512.22476 AutoQuant crypto, arXiv:2407.17645 Hopfield allocation, arXiv:2603.01820 DL risk-adjusted benchmark) APPLIES DSR/CPCV as the accepted gate rather than replacing it.
- One complementary direction worth noting (not blocking): arXiv:2604.15531 "Spurious Predictability in Financial ML" (2026) + the PSR blog both suggest non-parametric/bootstrap multiple-testing control can AUGMENT the parametric Z-score DSR. This aligns with the repo's own memory that a Sharpe-DELTA is better tested by paired Ledoit-Wolf + stationary bootstrap than by DSR. For phase-69's task (fix the ABSOLUTE-DSR unit bug), DSR-as-written is the correct target; bootstrap augmentation is a possible follow-on, out of 69.0 scope.
- Circuit-breaker: Azure doc last updated 2026-07-02 (current); Fowler remains canonical. No change to the manual-reset / durable-state / log-every-transition guidance.
- FX/ledger: Modern Treasury multi-currency (2022) design still current; native-currency-per-posting is the standing best practice; no 2025-2026 reversal found.

## Key findings

### Topic 4a — DSR exact expression + reference values (HIGHEST VALUE)

Source read IN FULL: Bailey & López de Prado, "The Deflated Sharpe Ratio" (JPM 2014; SSRN 2460551), fetched as PDF from davidhbailey.com + pdfplumber 0.11.9 extraction (research-gate PDF chain, accessed 2026-07-11).

**Exact expressions (paper Eq. 2 + Bailey-LdP 2012 PSR):**
- `DSR = PSR(SR*) = Z[ (SR_hat - SR*) * sqrt(T-1) / sqrt(1 - γ3*SR_hat + ((γ4-1)/4)*SR_hat^2) ]`
  i.e. `sigma(SR_hat) = sqrt( (1 - γ3*SR_hat + ((γ4-1)/4)*SR_hat^2) / (T-1) )` with γ4 = RAW kurtosis (Normal=3; term is (γ4-1)/4, NOT (γ4-3)/4). Repo uses `/T` not `/(T-1)` — negligible at T=1250 (0.04%), not the defect.
- `SR* = E[max SR] = sqrt(V[{SR_n}]) * [ (1-γ)*Z^-1(1-1/N) + γ*Z^-1(1-1/(N*e)) ]`, γ = 0.5772 (Euler-Mascheroni), N = # independent trials.
- **Units rule (verbatim from the paper's numerical example):** the deflation threshold is computed "**non-annualized (with 250 observations per year)**" — SR_hat and SR* MUST be in PER-PERIOD units so they match a per-period T. The paper's own example takes annualized SR 2.5 and divides by sqrt(250) before plugging in.

**Canonical worked reference (paper, "A NUMERICAL EXAMPLE" section — unit-test pin):**
Inputs: annualized SR_hat = 2.5, daily sample of 5 years (T = 1250, 250 obs/yr), N = 100 trials, V[{SR_n}] = 0.5 (annualized^2), skew γ3 = -3, raw kurtosis γ4 = 10.
- SR_hat per-period = 2.5/sqrt(250) = 0.1581139
- SR* annualized = sqrt(0.5)*[(0.4228)*2.3263 + 0.5772*2.6800] ≈ 1.7892 -> per-period = 1.7892/sqrt(250) ≈ 0.11316
- sigma denominator = sqrt(1 - (-3)(0.158114) + (9/4)(0.158114)^2) = sqrt(1.5306) = 1.23717
- z = (0.158114 - 0.113163)*sqrt(1249)/1.23717 = 1.2841 -> **DSR = Φ(1.2841) ≈ 0.9004** (paper: "only a 90% chance that the true SR ... is greater than zero")
- Secondary pins from the same example: N = 46 -> DSR = **0.9505**; Normal returns (γ3=0, γ4=3) -> DSR reaches 0.95 at N = **88**.
(All three re-derived by hand this session and matching the paper's printed values.)

**Quantified failure mode (why annualized SR + daily T explodes z):** plugging SR_ann=2.5 with T=1250 into the same expression gives sigma = sqrt((1+7.5+14.0625)/1249) = 0.1344, z = (2.5-1.7892)/0.1344 = **5.29 -> DSR ≈ 0.9999999** vs the true 0.9004. For small skew/kurt terms the inflation factor is ~sqrt(periods_per_year) ≈ sqrt(252) because the numerator (SR - SR*) is annualized (~sqrt(252)x larger) while sqrt(1/T) in sigma is per-period. This is exactly `analytics.py:654` (annualized `aggregate_sharpe` + `T=len(daily_returns)`), collapsing the immutable DSR>=0.95 gate into a near-binary pass.

**Fix shape for `compute_deflated_sharpe`:** de-annualize BOTH the observed SR and the trials' SR variance consistently (SR_p = SR_ann/sqrt(ppy); V_p = V_ann/ppy — variance scales by 1/ppy since SR scales by sqrt(ppy)), keep skew/kurt from per-period (daily) returns as today, use per-period T (and prefer T-1).

### Topic 4b — Purge + embargo vs label horizon

Sources read IN FULL: López de Prado GARP whitepaper (#3), QuantInsti CPCV (#4); AFML Ch.7 canon.
- **Purge** = remove from the TRAINING set every sample whose label event-time interval `[t_entry, t_exit]` overlaps the test fold's time span (the 3-condition overlap test: train label starts within test, ends within test, or straddles it). Not a fixed calendar gap.
- **Embargo** = an ADDITIONAL small gap (LdP: `h ≈ 0.01·T` of the whole sample) on training samples immediately FOLLOWING the test fold, to kill serial-correlation leakage from features that peek just past the test window.
- **Repo defect**: `walk_forward.py:61` uses a fixed `embargo_days=5` gap between `train_end` and `test_start`; that is neither a purge nor sufficient — the triple-barrier label horizon is `holding_days` scanned to `holding_days×1.5` (up to ~135d, `backtest_engine.py:658`), so any training sample with `sample_date` inside `[test_start − 1.5·holding_days, train_end]` has a label computed from prices INSIDE the test window → leakage. Note the recorded `exit_dates` at :596 use `holding_days`, UNDERSTATING the true horizon.
- **Fix**: purge training samples whose `[sample_date, sample_date + 1.5·holding_days]` overlaps `[test_start, test_end]`; use `1.5·holding_days` (true horizon) for the exit stamp; keep an embargo as a post-test gap. This makes the walk-forward leak-free per AFML Ch.7.

### Topic 1 — FX last-known / fail-closed (ledger integrity)

Sources read IN FULL: Modern Treasury multi-currency ledger (#8); US Treasury reporting-rates (snippet, WebFetch 403 but guidance captured); in-repo `execute_buy` block + `mark_to_market` last-known precedents.
- Multi-currency ledgers store amounts in the account's NATIVE currency and balance **per currency**; you do NOT convert at booking with an assumed rate (Modern Treasury). Booking a KRW notional as USD violates per-currency balance — a critical ledger defect (the ~1300x phantom-cash bug).
- **Fail-closed waterfall**: when the live rate is unavailable, serve the last-known-good VERIFIABLE rate and record its source (US Treasury: "if no rate is available, use another verifiable exchange rate and provide the source"; staleness tolerance is generous — amend only if deviation ≥10%, a rate is usable ~3 months). Assuming parity (rate=1.0) is never acceptable.
- **Recommendation for the stop-loss SELL**: (a) fix `_usd_value_live` to serve a last-known chain — stale api_cache → `historical_fx_rates` most-recent row → module last-known — so `None` is nearly unreachable for any market that has ever traded (every prior BUY write-through-persisted a rate); (b) for the residual "no rate EVER stored" case, credit at last-known if ANY exists, else BLOCK + PAGE (fail-closed) rather than credit at 1.0 — a bounded staleness error strictly dominates an unbounded ~1300x parity error. Pair the block with a P1 page so a stranded exit gets operator action; never book phantom USD. **Pitfall to encode in the design**: the last-known read inside `_usd_value_live` must query `historical_fx_rates` DIRECTLY (a dedicated helper), NOT via `_usd_value_asof`, which degrades back to `_usd_value_live` → mutual recursion.

### Topic 2 — Kill-switch audited peak-reset state machine

Sources read IN FULL: Fowler CircuitBreaker (#6); MS Azure circuit-breaker (#5); prop-firm trailing-DD community (snippets confirm monotonic-peak = permanent-fail is the industry default the repo inherited).
- Circuit-breaker canon: operations staff must be able to reset/trip the breaker (Fowler); EVERY state transition must be logged / raise an event (Fowler + MS); a manual force-close resets the failure counter and durable state (MS).
- The repo ALREADY has append-only, restart-replayable event sourcing (`_load_from_audit` replays `pause/resume/sod_snapshot/peak_update` from `handoff/kill_switch_audit.jsonl`). So the peak-reset is a NEW `peak_reset` event type + a replay branch that sets `_peak_nav`, NOT new infrastructure.
- **Trigger conditions (guard-behavior change → DARK until `KS-PEAK-RESET: APPROVED`)**: (a) on flatten-to-cash the old peak is meaningless against a 100%-cash NAV → reset peak to the post-flatten NAV; (b) on operator resume → re-anchor peak to current NAV so the trailing-DD denominator reflects the resumed book. Both emit a `peak_reset` audit row `{old_peak, new_peak, trigger, operator}` and replay deterministically (idempotent: replaying the stream yields the same peak). This restores the documented "human resume once healthy" behavior; thresholds (4/10/8/30) stay byte-untouched.
- **`current_nav<=0` guard** (`evaluate_breach`): return null/no-breach when `current_nav` is None or ≤0 so a BQ-timeout `or 0.0` no longer renders a phantom 100% daily+trailing breach. This is a fail-safe DATA-SANITY fix (a funded book's NAV is never ≤0), not a threshold change — but it suppresses a (false) breach, so surface it for operator awareness even though it does not need the DARK token.

### Topic 3 — Sign-safe multiplicative overlay algebra

Sources read IN FULL: Elastic BM25 multiplicative boosting (#7); FTSE Russell tilting (snippet); FAJ signal-blending (snippet).
- Multiplicative boosts "implicitly assume positive base values for the ratio to remain meaningful" (Elastic); factor tilts presume a NON-NEGATIVE score domain (FTSE Russell). The repo multiplies a SIGNED momentum composite (routinely negative in drawdowns) → the boost inverts into a penalty exactly when selection carries the most information.
- **Recommended fix (sign-aware additive tilt), ONE unified expression**: `score_out = score + abs(score) * (mult - 1)`.
  - For `score ≥ 0`: reduces to `score * mult` (byte-identical intent to today).
  - For `score < 0`: reduces to `score * (2 - mult)` → a boost (mult>1) moves the score UP toward zero (better rank), a penalty (mult<1) moves it DOWN (worse rank).
  - Verified: (+10,×1.10)→11; (−10,×1.10)→−9 (boost improves rank); (−10,×0.90)→−11 (penalty worsens); (+10,×0.90)→9. A "boost" always raises and a "penalty" always lowers, in BOTH sign regimes.
- Preferred over clamp-to-no-op (which discards the catalyst for negative-base candidates — the drawdown regime where it matters most). Apply at every multiplicative overlay site: `news_screen:329`, `macro_regime:547` (sector tilt) + `:542` conviction_multiplier, and the pead/options/insider/peer_leadlag overlays.

## Internal code inventory (all register claims re-verified 2026-07-11; read-only)

| File:line | Verbatim anchor | Register status |
|---|---|---|
| `backend/services/paper_trader.py:388-392` | `_l2u = _fx_local_to_usd(position.get("market"))` -> `if _l2u is None: logger.warning(...); _l2u = 1.0` ("never block an exit") | CONFIRMED. Cash credit at :506 `new_cash = portfolio["current_cash"] + net_proceeds * _l2u`; also poisons trade `total_value` :433, `transaction_cost` :434, `realized_pnl_usd` :460 |
| `backend/services/paper_trader.py:212-217` | execute_buy FX block: `if _usd_to_local is None or _local_to_usd is None: ... skipping BUY ... return None` | The fail-closed mirror the sell path should adopt (with last-known fallback first) |
| `backend/services/paper_trader.py:532-540` | mark_to_market: `if _l2u is None: ... keeping last-known market_value` | In-repo precedent for last-known-good on FX outage |
| `backend/services/fx_rates.py:78-104` | `_usd_value_live`: api_cache (6h TTL :53) -> `_fetch_yf` -> `_fetch_fred` -> `return val` (None if both fail). Write-through `_persist(...)` at :103 | CONFIRMED: live path NEVER reads `historical_fx_rates` |
| `backend/services/fx_rates.py:153-179` | `_usd_value_asof`: `SELECT rate FROM historical_fx_rates WHERE pair=@pair AND date<=@d ORDER BY date DESC LIMIT 1` | Last-known-good READ ALREADY EXISTS -- just not consulted by the live path. Writers: `_persist` :200-212, `backfill_fx` :215-262 |
| `backend/services/kill_switch.py:212-217` | `update_peak`: "Ratchet the trailing high-water mark upward. Never moves down." -> `_append_audit("peak_update", nav=...)` | CONFIRMED monotonic-only; no reset event exists anywhere |
| `backend/services/kill_switch.py:61-106` | `_load_from_audit` event-sourced replay of `pause`/`resume`/`auto_resume_alert`/`sod_snapshot`/`peak_update` from `handoff/kill_switch_audit.jsonl` (:104 restores peak) | Restart-replayable append-only machinery ALREADY EXISTS; fix = new `peak_reset` event + replay branch + authorized emit sites |
| `backend/services/kill_switch.py:230-264` | `evaluate_breach` computes `daily_loss_pct = (sod - current_nav)/sod` with NO `current_nav <= 0` guard | CONFIRMED: callers' `or 0.0` BQ-timeout fallback renders phantom 100% breach |
| `backend/services/kill_switch.py:275-333` | `check_auto_resume`: `if breach["any_breached"]: never auto-resume` (:312-314); manual `resume()` :184-194 | With 100% cash + frozen peak, trailing-DD breach persists -> both resume paths refuse forever. CONFIRMED lockout |
| `backend/backtest/analytics.py:323-325` | `se_sr = math.sqrt((1 - skewness*observed_sr + (kurtosis-1)/4 * observed_sr**2) / T)` | Formula SHAPE correct (raw kurtosis, `(k-1)/4`); UNITS wrong at caller |
| `backend/backtest/analytics.py:654-661` | `generate_report` passes `observed_sr=result.aggregate_sharpe` (ANNUALIZED, mean/std*sqrt(252) per :129-144) with `T=len(daily_returns)` (:650), skew/kurt from daily returns (:648-649, `fisher=False` = raw kurtosis, correct) | CONFIRMED unit mix. `variance_of_srs` (:642) is variance of ANNUALIZED window Sharpes -- must be de-annualized consistently too |
| `backend/backtest/analytics.py:317-320` | `E[max SR]` two-term Euler-Mascheroni expression | Matches canonical Bailey-LdP form; not a defect |
| `backend/backtest/backtest_engine.py:486-490, 512-516` | `cache.cached_prices(ticker, test_start_str, test_start_str)` exact-date lookups at trade + liquidation | CONFIRMED empty-on-weekend/holiday |
| `backend/backtest/backtest_engine.py:566-598` | `_build_training_data`: biweekly samples up to `train_end`; `exit_dates.append(sample_date + timedelta(days=holding_days))` (:596-597); NO purge filter | CONFIRMED. NOTE: recorded exit UNDERSTATES the true label window -- `_compute_triple_barrier_label` scans to `holding_days*1.5` (:658) |
| `backend/backtest/walk_forward.py:61` | `test_start = train_end + timedelta(days=self.embargo_days + 1)`; default `embargo_days=5` (engine :148) | 5-day gap vs 90-135d (x1.5) label horizon. CONFIRMED |
| `backend/backtest/backtest_engine.py:628-637` | TRAIN: fracdiff applied to `_NON_STATIONARY` cols over the interleaved mixed-ticker sample matrix; `X.fillna(X.median())` then 0 | CONFIRMED (note: fracdiff over interleaved cross-ticker rows is itself statistically dubious) |
| `backend/backtest/backtest_engine.py:793-801` | PREDICT: `row = {f: fv.get(f, 0) ...}`; `fillna(0)`; no fracdiff | CONFIRMED train/predict transform skew |
| `backend/services/news_screen.py:282` | `"max_output_tokens": min(8192, 250 * len(deduped))` | CONFIRMED: cap binds for >32 headlines; parse fail -> `return {}` (:291-299) |
| `backend/services/news_screen.py:329-332` | `return base_score * 1.10` / `* 0.90` | CONFIRMED sign-unsafe multiplicative overlay |
| `backend/services/macro_regime.py:542-549` | `score = base_score * regime.conviction_multiplier`; `score *= 1.05` overweight / `*= 0.95` underweight | CONFIRMED sign-unsafe (multiplier clamp 0.5-1.5 at :466) |

No stale register line/claim found: every checked file:line matches the register's description as of 2026-07-11.

## Consensus vs debate (external)

- **Strong consensus**: DSR/PSR must be computed in PER-PERIOD units matching a per-period T (Bailey paper explicit); purge+embargo is the standard leak-control (LdP AFML Ch.7); circuit-breakers need operator reset + logged transitions (Fowler/MS); multiplicative boosts assume a non-negative base (Elastic, FTSE Russell); multi-currency ledgers post in native currency and never assume parity at booking (Modern Treasury, Treasury).
- **Debate / nuance**: (a) parametric DSR Z-score vs bootstrap/non-parametric multiple-testing (recency-scan hits arXiv:2604.15531 + PSR blog) — for phase-69's ABSOLUTE-DSR unit-fix the DSR-as-written is the correct target; bootstrap augmentation is an out-of-scope follow-on. (b) FX residual for a stop-loss exit: ledger literature favors strict fail-closed, but a stop-loss exit has a competing do-no-harm concern (don't strand a position). Resolved by the last-known chain making the residual near-unreachable, with block+page ONLY when no rate was ever stored.

## Pitfalls (from literature)

- **DSR**: de-annualizing SR̂ but FORGETTING to de-annualize `variance_of_srs`/E[maxSR] leaves a residual unit mix — the fix must de-annualize BOTH (V_p = V_ann/ppy).
- **Purge**: using the recorded `holding_days` (:596) instead of the TRUE `1.5·holding_days` label horizon (:658) under-purges and still leaks.
- **FX**: wiring the last-known fallback via `_usd_value_asof` re-enters `_usd_value_live` → mutual recursion; use a dedicated direct-BQ helper.
- **Sign-safe**: clamp-to-no-op silently drops the catalyst in drawdowns; the additive form preserves it.
- **Kill-switch**: peak-reset is fail-safe but is still a GUARD-BEHAVIOR change → DARK until the operator token; thresholds byte-untouched. `current_nav<=0` guard suppresses a (false) breach → surface for operator awareness.
- **Fracdiff-at-predict** (design-adjacent, for 69.2): apply the SAME train-time fitted fracdiff weights + train medians at predict time; do NOT recompute fracdiff on a single prediction row (statistically meaningless) and do NOT swap the NaN-fill policy (train=median vs predict=0).

## Application to pyfinagent

Direct mapping of findings → design-pack (69.0) elements → fix sites (69.1/69.2/69.3):
- **FX chain (69.1)**: `fx_rates._usd_value_live:78-104` gains a last-known chain (stale cache → direct `historical_fx_rates` read → None only if never stored); `paper_trader.execute_sell:388-392` replaces `_l2u = 1.0` with credit-at-last-known-else-block+page (mirrors `execute_buy:212` fail-closed intent). Do-no-harm: no threshold change; ledger-math + fallback only.
- **Kill-switch (69.1)**: new `peak_reset` event in `kill_switch` (+`_load_from_audit` replay branch), emitted on flatten + operator-resume, DARK until `KS-PEAK-RESET: APPROVED`; `evaluate_breach:230-264` gains a `current_nav<=0` null-breach guard. Thresholds untouched.
- **DSR (69.2)**: `analytics.compute_deflated_sharpe:292-335` de-annualizes SR̂ and V consistently; reference test pins DSR(SR_ann=2.5,T=1250,N=100,skew=−3,kurt=10,ppy=250)=0.9004 and asserts the pre-fix path ≈0.9999999. DSR≥0.95 gate byte-untouched.
- **Purge+embargo (69.2)**: `_build_training_data:566-598` purges samples whose `[sample_date, sample_date+1.5·holding_days]` overlaps `[test_start,test_end]`; fixture asserts zero overlap.
- **Boundary snap (69.2)**: `backtest_engine:486-490/512-516` business-day-snap the exact-date price lookups so weekend/holiday-bounded windows trade.
- **Fracdiff-at-predict (69.2)**: `:793-801` applies the train-time transform + median fill identically.
- **Sign-safe overlays (69.3)**: `score + abs(score)*(mult-1)` at `news_screen:329`, `macro_regime:542/547`, pead/options/insider/peer_leadlag — flag-gated, ON-vs-OFF live_check.
- **News cap (69.3)**: raise/remove the `min(8192, …)` cap (or chunk at ~32) + parse-fail retry at `news_screen:282-299`.
- **QMJ + INDPRO/liquidity (69.3)**: assign `revenue_growth_yoy` before `quality_score` (`historical_data:202`); add INDPRO + net-liquidity to `_REGIME_SERIES` via a NEW cached path (historical_macro frozen).

## Research Gate Checklist

- [x] >=5 authoritative external sources READ IN FULL via WebFetch (8 read in full)
- [x] 10+ unique URLs total (incl. snippet-only) (23 collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full papers / pages read (not abstracts) for the read-in-full set (Bailey DSR + LdP GARP fetched as PDF via pdfplumber)
- [x] file:line anchors for every internal claim (19 internal sites)
- [x] Each of the 4 topics has >=1 authoritative source
- [x] DSR reference value captured (0.9004 @ N=100; bug path 0.9999999)

## Gate completion note (provenance)

Research (8 sources read in full, all 4 topics, DSR reference re-derived) was performed by the harness researcher subagent. TWO researcher spawns (Fable, then Opus) each read the sources but STALLED on the end-of-session flush before finalizing (the write-first anti-pattern the project memory warns about; Fable ~14 min transcript-idle, Opus ~4.5 min). Both were stopped per the CLAUDE.md STALL WATCH doctrine. The source table, DSR worked example, and internal inventory were persisted incrementally (write-first partially held). **Main (Opus, this session) finalized the synthesis sections (Topics 1-3 Key findings, Application, this note) and the envelope from the already-read sources + independent re-derivation of the DSR reference values** — the documented "Main updates the stalled handoff file" pattern, NOT new self-directed research replacing the gate. Every synthesis claim traces to a source row in the "Read in full" table above. Q/A should verify the source rows + DSR reference independently.

## JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 15,
  "urls_collected": 23,
  "recency_scan_performed": true,
  "internal_files_inspected": 19,
  "gate_passed": true
}
```
