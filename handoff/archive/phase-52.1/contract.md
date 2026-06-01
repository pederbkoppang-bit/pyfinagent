# Contract -- phase-52.1: 52-week-high momentum tilt (price-only alpha signal) -- MEASURE-FIRST

**Step id:** 52.1 | **Priority:** P1 (north-star #4) | **depends_on:** 51.2 (reuses the $0 replay harness)
**Date:** 2026-06-01 | **harness_required:** true | **$0 LLM** | no pip | **NO live engine change** (measure-first)

## Research-gate summary (PASSED)
`handoff/current/research_brief.md` (researcher `a99666872066bb1fe`: gate_passed=true, tier complex, 7 sources read in full, 19 URLs, recency scan, internal audit of screener.py + the replay). Decisive:
- **THE PICK: 52-week-high proximity (George-Hwang 2004) as a centered MULTIPLICATIVE TILT** on the existing 1/3/6-mo momentum composite. Chosen as the MEASURE-FIRST pick because the formula is ALREADY implemented (`screener.py:213-214`: `pct_to_52w = current_price / trailing-252d-high`, in (0,1]) and it fits the replay with a one-line window widening -- residual momentum (the higher-evidenced RUNNER-UP) needs a net-new OLS harness + 3.5x-longer download, so measure the cheap already-built 52wh first.
- **Blend = multiplicative tilt (preferred, NOT replace, NOT z-blend):** `score *= (1 + k*(pct_to_52w - mean_pct_to_52w_universe))`, centered so the average tilt ~= 1.0 (turnover-neutral on average), k in {0.5, 1.0}. Mirrors the existing RSI/vol multiplier idiom; PRESERVES the working momentum ranking and only nudges it (minimal regression risk). Replace/z-blend throw away the measured-good 1/3/6 signal.
- **HONEST CAVEAT (the headline risk):** Barroso-Wang 2021 -- the 52wh edge is MUTED for large caps ("price momentum explains the predictability" for larger stocks). On an S&P-500 (large-cap) book the dSharpe may be ~0. THIS IS WHY WE MEASURE, not assume. Realistic expectation: a SMALL positive dSharpe (~+0.02 to +0.10) with low turnover; treat >= +0.05 dSharpe at <= +10% turnover as PASS-worthy-to-escalate-to-a-live-gate; ~0 or negative = cleanly REJECT + pivot to residual momentum (52.2).
- **Adversarial/deprioritized (cited):** skip-month "echo" CONTRAINDICATED ("Echo disappears" 2023 + Aalto thesis: conventional recent momentum out-performs); 200-dma trend gate is a risk overlay not a ranking enhancer; low-vol fights momentum; vol-scaling already measured (+0.015 marginal).
- **Over-tuning pitfall (McLean-Pontiff):** a single replay number isn't proof; report Sharpe AND turnover; test only k in {0.5,1.0} (NOT a sweep); don't cherry-pick the best k.

## Hypothesis
Tilting the production momentum composite toward names nearer their 52-week high (a centered multiplicative tilt, price-only, reusing the existing `pct_to_52w` formula) measurably improves the forward-1mo Sharpe of the top-N basket on our S&P-500 universe WITHOUT unacceptable turnover -- OR (the Barroso-Wang large-cap-mute scenario) it does not, in which case we cleanly reject it and escalate to residual momentum with evidence. Measured $0 via the 51.2 replay harness reusing the production `rank_candidates` output; NO live engine change.

## Success criteria (IMMUTABLE -- verbatim from masterplan step 52.1)
1. a research-backed PRICE-BASED alpha-signal enhancement (cited 2025-2026 + canonical, computable from daily closes alone) is measured ON-vs-OFF against the baseline momentum ranking on the S&P 500 universe via the $0 replay, reporting Sharpe / return / turnover
2. the measurement reuses the production rank_candidates path with identical screen_data for both arms (sole delta = the new signal) + causal forward returns; the result is honestly reported (a negative result is a VALID outcome, not a failure)
3. NO live engine change in this step (measure-first; any live wiring is a separate operator-gated step); the working US momentum core is untouched
4. live_check_52.1.md records the ON-vs-OFF comparison + the cited basis + a keep/reject recommendation

**Verification command:** `pytest backend/tests/test_phase_52_1_alpha_signal.py` + `ast.parse(replay)` + `test -f live_check_52.1.md`.
**live_check:** REQUIRED -- the $0 replay ON-vs-OFF (Sharpe/return/turnover, baseline vs 52wh-tilt k=0.5/1.0) + the cited basis + a keep/reject recommendation; NO live flag flip.

## Plan steps (GENERATE)
1. **Extend `scripts/ablation/sector_neutral_replay.py` (NO live-engine file touched):**
   - `build_screen_row`: add `pct_to_52w_high = last / c.rolling(252, min_periods=20).max().iloc[-1]` (the George-Hwang formula; price-only; `c` causal window already in scope).
   - Widen `win_lo = max(0, t_idx - 200)` -> `max(0, t_idx - 260)` so 252 days are available; START 2021-06-01 stays (or push to 2021-01-01 if marginal).
   - Add `hi52_tilt` configs (k=0.5, k=1.0): rank via the PRODUCTION `rank_candidates(rows, top_n=len(rows), strategy="momentum")` to get each row's production `composite_score`, then apply the centered tilt `composite_score * (1 + k*(pct_to_52w_high - mean_pct))`, re-sort, take top_n. (Reuses the production composite verbatim; the tilt is replay-side post-processing -> zero live change.)
   - Score with the existing `basket_fwd_return` / `ann_sharpe` / turnover machinery; report dSharpe + dTurnover vs baseline (mirror the 51.2 verdict block).
2. **Test** `backend/tests/test_phase_52_1_alpha_signal.py`: (a) the tilt MECHANISM -- two rows with equal composite but different pct_to_52w_high -> the higher-52wh one ranks first (deterministic); (b) pct_to_52w_high computation from a price series; (c) centering -> a universe-mean-52wh name gets ~1.0 tilt (no change). $0, no network.
3. **Verify:** pytest; ast.parse(replay); run the replay -> capture baseline vs hi52_tilt(0.5/1.0) Sharpe/return/turnover into `live_check_52.1.md` with a keep/reject recommendation per the >=+0.05 dSharpe / <=+10% turnover bar. Report ALL k (no cherry-pick).
4. **EVALUATE:** fresh qa. Then harness_log.md (LAST), then flip masterplan 52.1 -> done.

## Safety / scope notes
- **NO live engine change.** Diff = the replay script + the new test ONLY. screener.py / autonomous_loop.py / decide_trades / risk guards are UNTOUCHED. The tilt is replay-side post-processing of the production composite. If 52wh measures PASS-worthy, the live wiring (a `strategy="momentum_52wh"` branch) is a SEPARATE operator-gated step.
- **Honest negative result is a valid outcome** (criterion #2): if dSharpe ~0/negative (Barroso-Wang large-cap mute), report it + recommend pivoting to residual momentum (52.2). The step PASSES on a sound measurement, not on the signal winning.
- **No over-tuning:** test k in {0.5,1.0} only; report both; don't cherry-pick.
- $0 LLM; no pip; no spend; no DROP/DELETE; no live trading change.

## References
- handoff/current/research_brief.md (52.1 gate)
- backend/tools/screener.py:213-214 (pct_to_52w formula -- reuse), :268-282 (momentum composite), :499-523 (multidim 52w leg, bundled -- NOT reused)
- scripts/ablation/sector_neutral_replay.py:67-94 (build_screen_row), :161 (win_lo), the config loop + basket_fwd_return/ann_sharpe
- George-Hwang 2004 (J.Finance, 52wh); Barroso-Wang 2021 (large-cap mute); Blitz-Huij-Martens 2011 (residual momentum, the runner-up); McLean-Pontiff (factor decay)
