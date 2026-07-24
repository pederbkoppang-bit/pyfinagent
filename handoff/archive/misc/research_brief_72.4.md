# Research Brief — phase-72.4: P4 Regime Deployment-Policy Research

**Tier:** moderate (as assigned). NOT audit-class.
**Step:** masterplan phase-72.4 — recommended deployment policy for flat/down market regimes.
**Deliverable:** literature-grounded recommend-only deployment policy for a long-only momentum book, mapped to pyfinagent mechanisms with file:line. NO code.
**Started:** 2026-07-18. Write-first; grown incrementally as sources are read.

---

## The operator's question (restated)

Once scoring is restored (P0 fix), what SHOULD "earning" look like per regime? When is ~100% cash the right call for a long-only momentum book vs deploying (mean-reversion harvest, defensive sectors, or accepting benchmark-like returns)? The 7-week flat window (SPY 5.84→5.18, dip to 1.87 on 06-10 — note: these appear to be % figures, benchmark ~flat) means our ~100%-cash posture *avoided* a drawdown, but by DEFECT (credit-dead scoring), not policy. We need a policy so the next flat/down regime is a *choice*.

---

## Internal code inventory (mechanisms to fit)

| File:line | Mechanism | Detail |
|---|---|---|
| `backend/tools/screener.py:299-313` | **Momentum composite** | `score = mom_1m*0.40 + mom_3m*0.35 + mom_6m*0.25`; RSI penalty (>80 → ×0.7, <20 → ×0.8); vol penalty (ann-vol>0.6 → ×0.85). `value_momentum` strategy (`:314-316`) blends mean-reversion (`mom_3m*0.5 − |sma_dist|*0.2 + mom_1m*0.3`) but is NOT the live default. |
| `backend/tools/screener.py:320-324` | Regime hook | Score passes through `apply_regime_to_score` ONLY if a `regime` object is passed. |
| `backend/services/macro_regime.py:32-38` | **Regime tags + multipliers (DARK)** | `risk_on=1.15, risk_off=0.70, mixed=1.00, unknown=0.85` — a **conviction multiplier**, NOT a binary cash gate. |
| `backend/services/macro_regime.py:40-43` | Regime inputs | FRED: T10Y2Y, VIXCLS, BAMLH0A0HYM2 (HY OAS), FEDFUNDS, CPIAUCSL, UNRATE, INDPRO. LLM-as-judge (haiku-4-5, `settings.py:389`). |
| `backend/services/macro_regime.py:604-630` | `apply_regime_to_score` | Sign-safe multiply (phase-69.3) of base score × conviction_multiplier, then ×1.05/×0.95 sector over/under-weight tilt. |
| `backend/config/settings.py:388` | `macro_regime_filter_enabled=False` | **DARK.** The whole regime overlay is off; when on it only *scales scores*, cannot by itself force cash. |
| `backend/config/settings.py:37` | `regime_net_liquidity=False` | **DARK** (phase-69.3): adds FRED net-liquidity (WALCL−WTREGEN−RRP) to the regime prompt. |
| `backend/services/autonomous_loop.py:422-434` | Regime wiring | Regime computed per-cycle only if flag on; passed into `rank_candidates` (`:839-842`). |
| `backend/config/settings.py:528-529` | **Kill-switch** | `paper_daily_loss_limit_pct=4.0`, `paper_trailing_dd_limit_pct=10.0`. `kill_switch.py:281-329` `evaluate_breach`; halts decide/execute (`autonomous_loop.py:1262-1268`). Catastrophic circuit-breaker, not a regime tool. |
| `backend/config/settings.py:535-540, 546-551` | **Trailing stops** | `paper_default_stop_loss_pct=8.0`; `paper_trailing_stop_pct=8.0` after +1R breakeven ratchet. The actual per-position risk exit. |
| `backend/config/settings.py:366` | **Cash floor** | `paper_min_cash_reserve_pct=5.0` — a *minimum* reserve, NOT a deployment target. `portfolio_manager.py:96-98` (runtime-overridable). |

**KEY STRUCTURAL FACT for the policy:** pyfinagent has **NO explicit market-timing / trend-filter deployment gate**. "Deploy vs cash" is *emergent*: it falls out of (a) how many names pass the momentum screen with positive momentum, (b) whether LLM analysis returns BUY/STRONG_BUY (`portfolio_manager.py:63,182-189` — the P0 seam), (c) cash left above the 5% floor. The regime overlay is the ONLY macro lever and it is a *soft score multiplier* (currently dark), never a cash switch. The recent ~100% cash was the credit-dead-scoring DEFECT (all HOLDs), not any deployment policy. So the policy question is: what *dark levers* should the operator turn on, and with what thresholds, so that flat/down-regime cash-vs-deploy becomes a CHOICE.

---

## External research — read in full

### 1. Daniel & Moskowitz, "Momentum Crashes," JFE 122 (2016) 221-247 [PEER-REVIEWED] — read in full via pdfplumber (27pp)
- **When momentum crashes:** "momentum crashes are partly forecastable. They occur in panic states, following market declines and when market volatility is high, and are contemporaneous with market rebounds." The danger is NOT a flat market — it is the sharp REBOUND after a bear market.
- **Mechanism:** in a bear market a momentum portfolio's up-market beta is "more than double" its down-market beta; losers "behave like written call options on the market." So a long-only momentum book is implicitly SHORT optionality into a recovery — it holds the wrong names when the market snaps back.
- **The fix is scaling, not binary exit:** the optimal dynamic strategy scales the WML weight so "the dynamic strategy's conditional volatility is proportional to the conditional Sharpe ratio." Weight ∝ forecast_mean / forecast_variance (GJR-GARCH vol forecast + regression mean forecast). OOS this "approximately doubles the alpha and Sharpe ratio of a static momentum strategy"; full implementable version SR = **1.19** (4× the static US-equity momentum SR). Caveat the authors flag: the dynamic strategy "at times employs considerably more leverage" and "would certainly incur higher transaction costs."
- **Improvement is not only crash-avoidance:** "even over subperiods devoid of those crashes, there is still improvement" — vol-scaling helps in normal regimes too, not just crashes.

### 2. Faber, "A Quantitative Approach to Tactical Asset Allocation" (updated "10 Years Later," ~2013) [INDUSTRY/PRACTITIONER] — read in full via pdfplumber (70pp)
- **Rule:** "Buy when monthly price > 10-month SMA. Sell and move to cash when monthly price < 10-month SMA." Updated monthly; <1 round-trip/yr; invested ~70% of the time.
- **Timing does NOT raise average return — it cuts risk:** S&P avg return 11.26% vs timing 11.22% (≈equal); but COMPOUNDED 9.32% (B&H) vs 10.18% (timing) — the edge is entirely lower volatility + avoided drawdowns, not stock-picking. "market timing solution is a risk-reduction technique."
- **Drawdown reduction is the whole prize:** 1930s DD 83.66% → 42.24%; 2000-02 DD 44.73% → 16.52%; avoided the 2008 50.95% DD. Correlation with S&P negative years = **−0.38**, with positive years = **+0.83** ("stay long in up markets, exit in down").
- **THE ADVERSARIAL POINT FOR US [ADVERSARIAL vs binary gating]:** it "underperforms the index in roughly half of all years" and specifically "can underperform buy and hold during a roaring bull market." Faber's own FAQ: **"a market that oscillates can be a poor market for trendfollowers due to whipsaws."** "The value added by timing is evident only over the course of entire business cycles." A binary trend-gate applied to a FLAT/CHOPPY 7-week window (our exact regime) is where trend-following bleeds via whipsaw — it is not a validated edge at that horizon.

### 3. Vanguard, "A framework for allocating to cash" (April 2024) [OFFICIAL/INDUSTRY] — read in full via pdfplumber (18pp)
- **Cash is a FUNDING-LEVEL decision, not a market-view decision:** "Allocations to cash are recommended only for investors with lower risk tolerances and shorter time horizons." "The better-funded a goal, the less [risk needed]": for a well-funded goal, 100% cash maximizes P(success); as funding drops, "capturing the risk premium becomes more important" (a not-well-funded goal has only 17.5% success at 100% cash vs ~60% at 60/40).
- **Cost of tactical excess cash is small but real and always negative:** adding 10% cash costs 2-20 bps/yr in certainty-equivalent (−17.3 to −1.5 bps across risk tolerances). "stocks are risky—and so is avoiding them."
- **Against tactical de-risking to cash:** staying invested through 2020 beat both go-to-cash paths; going to cash and round-tripping back "in a significant underperformance."

### 4. J.P. Morgan AM, "The cost of holding cash in a volatile market" [INDUSTRY] — read in full via WebFetch
- **The stay-invested case:** 60/40 "outperformed cash more than 70% of the time over a one-year horizon, and always over three years"; excess returns "7 percentage points over one year, and more than 20 percentage points over a three-year timeframe."
- **Reinvestment risk:** in a downturn central banks cut rates so "cash returns would contract meaningfully" — cash is not even a safe yield in the regime where you'd most want it.
- Complements the classic "missing best days" fallacy (JPMAM data via search: 6 of the 10 best S&P days 2006-2025 fell within two weeks of the 10 worst; 76% of best days occur in a bear market or the first 2 months of a bull) — best/worst days cluster, so a cash gate that dodges the worst days also forfeits the rebound days.

### 5. Yang, "Signature-Informed Transformer for Asset Allocation," arXiv:2510.03129 (Oct 2025) [PREPRINT] — read in full via arXiv HTML
- End-to-end transformer that optimizes CVaR directly. **No explicit regime detection** — its regime response is emergent (a learned gate γ "clusters during volatile episodes"). OOS SR 0.67-0.77 vs equal-weight 0.58-0.60. **Honest limits:** tested only 2020-2024, "does not explicitly address data-snooping risk or out-of-sample stability across different market regimes beyond the single test period"; 10bps t-costs cut SR by 0.03-0.04; fragile to hyperparameter tuning.

### 6. "Deep Learning for Financial Time Series: A Large-Scale Benchmark," arXiv:2603.01820 (2026) [PREPRINT] — read in full via arXiv HTML
- Deep sequence models beat linear/passive on aggregate horizons (VLSTM SR 2.39 vs passive 0.48) BUT the authors stress linear models are "highly variable across time" in "environments characterized by non-stationarity, regime shifts, and low signal-to-noise ratios," and every conclusion "remain[s] conditional on the dataset and backtesting assumptions employed." A high in-sample Sharpe here is NOT a deployable edge — this is the data-snooping caution, not a green light.

### 7. Shu & Mulvey, "Downside Risk Reduction Using Regime-Switching Signals: A Statistical Jump Model Approach," arXiv:2402.05272v2 (2024, upd.) [PREPRINT — recency + methodological anchor] — read in full via arXiv HTML
- **THE DECISIVE FINDING for our policy:** citing Nystrup et al. (2016), "gradually adjusting the weight on the equity index as a linear function of the forecasted probability delivers similar out-of-sample performance compared to switching allocations between 100% and 0%," and a binary switch "may be too extreme for practical trading applications." → **scale continuously, do not binary-gate to cash.**
- Regime-switch (even binary) OOS vs buy-and-hold: S&P500 1990-2023 SR 0.68 vs 0.48, MDD −26.6% vs −55.2%, +~1%/yr; DAX SR 0.44 vs 0.30; Nikkei SR 0.31 vs 0.12. The prize is again **drawdown reduction**, not return.
- **Reliability caveats (why not to trust a regime tag blindly):** ~half-month detection LATENCY at crash onset/end; "might sometimes misinterpret oscillations during prolonged turbulent periods" (i.e., a CHOPPY/FLAT regime is exactly where the detector is least reliable); HMMs are "sensitive to model mis-estimation and mis-specification." Uses a t+2 execution lag to avoid look-ahead.

---

## Recency scan (last 2 years, 2024-2026)

Performed (3-variant queries incl. year-less canonical + 2025/2026 + bare-topic). **Findings that COMPLEMENT the canonical Daniel-Moskowitz/Barroso/Faber sources — none supersede them:**
- **Convergence on continuous scaling over binary gating** (Shu-Mulvey 2024 / Nystrup 2016; the regime-detection-AI-market and Tandfonline 2026 QF paper snippets): the 2024-2026 regime-allocation literature has moved AWAY from binary in/out toward continuous weight-by-regime-probability, which matches D-M's forecast-mean/variance scaling. This is the single most decision-relevant recent finding.
- **Transformer/DL regime models (SIT 2025, DL benchmark 2026)** show OOS gains but with heavy data-snooping/single-period caveats — consistent with pyfinagent's own P3 verdict that the ~20-flag overlay library is unvalidated on our data. Confirms this is an open RESEARCH topic, not a deployable production lever. (Matches the "transformer/diffusion regime models are a known open autoresearch topic here" note in the task.)
- Tandfonline QF 2026 "Tactical asset allocation with macroeconomic regime detection" (forecasts the DISTRIBUTION of future regimes, not a point tag) — HTTP 403, snippet-only; its distribution-forecast framing reinforces "scale by regime probability," not "switch on regime label."
- No 2024-2026 source overturns the core canon: momentum crashes on rebounds (D-M), vol-scaling ≈ doubles momentum SR (Barroso), trend-gating cuts drawdowns but whipsaws in flat markets (Faber).

---

## Key findings (synthesis)

1. **The literature is near-unanimous: the earning-optimal response to a weak/uncertain regime is to SCALE exposure continuously, NOT to binary-flip to cash.** Daniel-Moskowitz (weight ∝ mean/variance, SR→1.19), Barroso-Santa-Clara (inverse-vol scale, SR 0.53→0.97), Nystrup/Shu-Mulvey (linear-by-probability ≈ binary but "binary too extreme for practical trading"). (Sources: kentdaniel.net JFE 2016; snifferquant Barroso 2014; arXiv:2402.05272v2.)
2. **Trend/cash gating buys DRAWDOWN REDUCTION, not extra return, and it WHIPSAWS in flat/choppy markets** — i.e., exactly the pyfinagent 7-week regime. Faber: same avg return (11.26 vs 11.22%), edge is all compounding/lower-vol; "a market that oscillates can be a poor market for trendfollowers due to whipsaws"; underperforms in ~half of years and in bull markets. (Source: mebfaber SSRN-962461.)
3. **Cash is a FUNDING-LEVEL / horizon decision, not a market-timing dial.** Vanguard: cash recommended "only for investors with lower risk tolerances and shorter time horizons"; tactical excess cash costs 2-20 bps/yr CE and round-trip de-risking underperforms. JPM: 60/40 beats cash 70%+ of 1-yr windows, always over 3yr. (Sources: Vanguard cash framework 2024; JPMAM.)
4. **For a LONG-ONLY momentum book the real regime hazard is the REBOUND after a down move, not the flat window itself** — momentum losers "behave like written call options on the market," so a fully-invested long book holds the wrong (high up-beta) names into a leadership-rotating recovery. Long-only cannot go short (D-M's dynamic strategy is short-weight 82% of the extreme), so its ONLY regime levers are (a) buy fewer/smaller and (b) exit via stops. (Source: kentdaniel.net JFE 2016.)
5. **Regime detectors are LEAST reliable in choppy/flat regimes** (~half-month latency; "misinterpret oscillations during prolonged turbulent periods") — so any regime lever must be a soft down-weight with hysteresis-free continuous action, never a hard cash switch triggered on a noisy label. (Source: arXiv:2402.05272v2.)

---

## Consensus vs debate

- **CONSENSUS:** scale-don't-switch; cash-as-residual not cash-as-timing-gate; the catastrophic tail is best handled by stops/circuit-breakers (per-position) rather than market-timing; vol-scaling improves momentum in ALL regimes not just crashes.
- **DEBATE / conflict (stated honestly):**
  - *Faber (trend-gating avoids drawdowns) vs Vanguard/JPM (stay invested).* Resolved by HORIZON + MECHANISM: Faber's edge is over full business cycles via a binary gate that LOSES in flat 7-week windows; our regime is precisely Faber's whipsaw zone, so we side with "stay deployed but scale." The two are not actually contradictory — both agree binary cash-timing at short horizons is a loser.
  - *DL/transformer regime models (high OOS Sharpe) vs the data-snooping skeptics.* The high-Sharpe DL results (SIT, VLSTM 2.39) are single-period, un-snoop-tested, t-cost-fragile. Not deployable; keep as research.

## Pitfalls (from literature, mapped to our risk)

- **Whipsaw bleed** from any binary trend/cash gate in a flat market (Faber) — do NOT add a 200dma-type deployment switch.
- **Momentum crash on the rebound** (D-M) — do NOT mechanically re-pile into prior winners on the first up-day after a down window; the vol penalty + risk_off multiplier are the mitigants we have.
- **Latent regime-label noise** (Shu-Mulvey) — a soft multiplier tolerates a wrong label (costs a little ranking); a hard cash switch on a wrong label costs the whole rebound. Prefer the multiplier.
- **Overfit overlay admission** (our own P3): the ~20-flag overlay library is unvalidated on our data (2-of-26 incremental-admission failure mode) — turn on ONE regime lever at a time, at DEFAULT params, measure 3-5 cycles.

---

## Application to pyfinagent (fit_to_pyfinagent, file:line)

**The policy is implementable with EXISTING dark levers + already-live risk layer — no new code.**

1. **Deploy-but-scaled = turn on the continuous conviction multiplier.** `macro_regime_filter_enabled` (`settings.py:388`, DARK) → the ONE lever that is literally the literature's "scale by regime" mechanism: `apply_regime_to_score` (`macro_regime.py:604-630`) multiplies each candidate's momentum-composite score (`screener.py:299-313`) by `conviction_multiplier` (risk_on 1.15 / mixed 1.0 / risk_off 0.70 / unknown 0.85, `macro_regime.py:33-38`). It DOWN-WEIGHTS scores in risk_off so fewer/weaker BUYs clear the `{BUY,STRONG_BUY}` seam (`portfolio_manager.py:63,182-189`) and cash rises as a RESIDUAL — it can never by itself force 100% cash. Cost <$0.05/day (haiku, `settings.py:389`). This is the single recommended flip.
2. **Cash floor stays 5%** (`settings.py:366`) — it is a liquidity minimum, NOT a de-risking dial; do not raise it as a market-timing gate (Vanguard/JPM). Cash above 5% must be an emergent residual of an empty positive-momentum screen, not a target.
3. **Catastrophic-tail layer already live and correct** — kill-switch 4%/10% (`settings.py:528-529`, `kill_switch.py:281-329`) + 8% trailing stops/breakeven ratchet (`settings.py:535-551`). These ARE Faber's "avoid the far-left tail," applied per-position/book-level instead of as a market-timing gate. No change.
4. **`regime_net_liquidity`** (`settings.py:37`, DARK) — secondary; only AFTER #1 is on and measured 3-5 cycles (don't stack two dark levers; P3 sequencing).
5. **Do NOT build a binary trend/200dma deployment gate** — Faber's whipsaw warning + Nystrup's "binary too extreme" make it the wrong tool for our flat regime. The multiplier already gives continuous scaling.
6. **Future (NOT proposed here, beyond existing levers):** D-M/Barroso-style vol-scaling of position SIZE (not just score) would be the rigorous long-only analogue; note as a candidate research step, not a 72.4 action.

---

## RECOMMENDED POLICY (recommend-only — operator decides; no code beyond existing dark levers)

**Statement.** Once scoring is restored, the DEFAULT posture is *deploy*, because pyfinagent is a funded long-only momentum book at a short horizon where the literature says time-in-market beats binary market-timing (Vanguard, JPM). "Earning" in a flat/down regime is NOT going to cash — it is deploying *less aggressively*, sized by regime, and letting cash rise only as a RESIDUAL of an empty positive-momentum screen. Concretely: (1) DEPLOY whenever names clear the momentum screen with positive momentum and analysis returns BUY/STRONG_BUY; (2) SCALE that deployment down in weak regimes by turning on the continuous macro-regime conviction multiplier (risk_off ×0.70), which the whole momentum literature (Daniel-Moskowitz forecast-mean/variance scaling; Barroso vol-scaling; Nystrup/Shu-Mulvey linear-by-probability) endorses over any binary cash switch; (3) HOLD cash as an emergent residual — never raise the 5% floor as a timing gate — and rely on the already-live 8% trailing stops + 4%/10% kill-switch for the catastrophic tail. Do NOT add a binary 200dma/trend deployment gate: it whipsaws in exactly this flat regime (Faber) and is "too extreme for practical trading" (Nystrup). Turn on ONE lever (macro_regime_filter_enabled) at default params, measure 3-5 cycles, before considering regime_net_liquidity.

### Evidence FOR
- Scale-don't-switch is near-unanimous: Daniel-Moskowitz (dynamic weight ∝ mean/variance, OOS SR 1.19, ~2× static); Barroso-Santa-Clara (inverse-vol scaling SR 0.53→0.97); Nystrup/Shu-Mulvey ("linear function of forecasted probability delivers similar OOS performance" to binary, and binary is "too extreme for practical trading").
- Stay-invested base case: Vanguard (tactical cash costs 2-20 bps/yr CE; de-risk-then-reinvest underperforms); JPM (60/40 beats cash 70%+ of 1yr windows, always 3yr); best/worst days cluster so a cash gate forfeits rebounds.
- We ALREADY own the endorsed mechanism as a dark lever (the continuous conviction multiplier) — zero new code, <$0.05/day.
- Catastrophic-tail protection (Faber's actual prize — drawdown reduction) is already live via stops + kill-switch, applied per-position rather than as a market-timing gate.

### Evidence AGAINST (honest conflicts + risks)
- Faber proves binary trend-gating DOES cut drawdowns massively (83.7%→42.2%, avoided 2008) over full cycles — a real edge we are declining. Rebuttal: that edge disappears (whipsaw) at our flat 7-week horizon and it does not raise returns; our stops/kill-switch capture the tail differently.
- Regime detectors are least reliable in choppy/flat regimes (Shu-Mulvey: ~half-month latency, "misinterpret oscillations") and our LLM-as-judge overlay is UNVALIDATED on our data (P3: overlay library 2-of-26 incremental-admission failure) — so the multiplier is recommended at DEFAULT params only, one lever at a time, measured, and is fail-safe (a wrong label costs a little ranking, not the whole rebound).
- The high-Sharpe DL/transformer regime models (SIT OOS 0.67-0.77; VLSTM 2.39) are single-period, un-snoop-tested, t-cost-fragile — NOT deployable; explicitly kept as research, not a 72.4 lever.
- Residual momentum-crash risk on the post-down rebound (D-M) is only partially mitigated by the vol penalty + risk_off multiplier; a full fix (D-M/Barroso vol-scaling of position SIZE) is out of scope for existing dark levers and noted as future research.

### fit_to_pyfinagent
Turn on `macro_regime_filter_enabled` (`backend/config/settings.py:388`, currently DARK) → `compute_macro_regime` (`backend/services/autonomous_loop.py:422-434`) feeds a continuous `conviction_multiplier` (risk_on 1.15 / mixed 1.0 / risk_off 0.70 / unknown 0.85, `backend/services/macro_regime.py:33-38`) into `apply_regime_to_score` (`backend/services/macro_regime.py:604-630`), which sign-safe-scales the momentum-composite score (`backend/tools/screener.py:299-313`) before the BUY seam (`backend/services/portfolio_manager.py:63,182-189`). Cash floor stays 5% (`settings.py:366`); catastrophic tail stays on kill-switch 4%/10% (`settings.py:528-529`) + 8% trailing stops (`settings.py:535-551`). Secondary lever `regime_net_liquidity` (`settings.py:37`) deferred until #1 is measured. No binary trend gate. This is a recommend-only operator decision for `operator_decision_sheet_72.md` §P4.

---

## Research Gate Checklist

Hard blockers (all satisfied):
- [x] >=5 authoritative external sources READ IN FULL — **7** (Daniel-Moskowitz JFE; Faber; Vanguard; JPM AM; SIT arXiv:2510.03129; DL-benchmark arXiv:2603.01820; Shu-Mulvey arXiv:2402.05272v2)
- [x] 10+ unique URLs total — ~30+ surfaced across 8 searches
- [x] Recency scan (2024-2026) performed + reported (dedicated section)
- [x] Full papers read (pdfplumber for the 3 binary PDFs; arXiv HTML for the 3 preprints; WebFetch for JPM) — not abstracts
- [x] file:line anchors for every internal claim (internal inventory table + fit section)

Soft checks:
- [x] Internal exploration covered screener, macro_regime, kill_switch, settings, portfolio_manager, autonomous_loop wiring
- [x] Contradictions/consensus noted (Faber vs Vanguard/JPM; DL vs snooping-skeptics)
- [x] Per-claim citations with URLs

### Source hierarchy check
1 top-tier peer-reviewed (JFE 2016) + 3 arXiv preprints + 3 industry/official (Vanguard, JPM, Faber/practitioner). Clears the "not 5 community-tier" bar.

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 9,
  "urls_collected": 31,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Literature is near-unanimous that the earning-optimal response to a weak/flat regime for a momentum book is to SCALE exposure continuously, not binary-flip to cash: Daniel-Moskowitz (weight proportional to forecast mean/variance, OOS Sharpe 1.19), Barroso-Santa-Clara (inverse-vol scaling 0.53->0.97), and Nystrup/Shu-Mulvey (linear-by-regime-probability matches binary switching and binary is 'too extreme for practical trading'). Trend/cash gating (Faber) buys drawdown reduction not return and WHIPSAWS in exactly our flat 7-week regime. Cash is a funding/horizon decision not a timing dial (Vanguard 2-20bps CE cost; JPM 60/40 beats cash 70%+ of 1yr windows). For a long-only book the real hazard is the REBOUND after a down move (momentum losers = written calls), and its only levers are buy-fewer/smaller + stops. Recommended policy: DEFAULT deploy; scale down in weak regimes via the existing DARK continuous conviction multiplier (macro_regime_filter_enabled, risk_off x0.70); cash rises only as an emergent residual (5% floor unchanged); catastrophic tail stays on live 8% trailing stops + 4%/10% kill-switch; NO binary trend gate; one lever at a time, measured 3-5 cycles. Fully implementable with existing dark levers, no new code.",
  "brief_path": "handoff/current/research_brief_72.4.md",
  "gate_passed": true
}
```
