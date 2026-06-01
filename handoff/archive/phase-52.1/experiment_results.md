# experiment_results -- phase-52.1: 52-week-high momentum tilt (price-only alpha signal)

**Step:** 52.1 | **Date:** 2026-06-01 | **$0 LLM** | no pip | **NO live engine change** | GENERATE complete

## Outcome in one line
Measured the 52-week-high multiplicative tilt (George-Hwang 2004, price-only) ON-vs-OFF on our
S&P-500 universe via the $0 replay: a **SMALL but real POSITIVE edge at k=0.5 (~+0.05 Sharpe,
turnover-neutral)**. Recommend ESCALATING k=0.5 to a live operator gate. The first measured alpha
WIN of the element-2-redirect arc (after rotation + sector-neutral were both measured-and-rejected).

## What was changed (replay-only; NO live-engine file touched)

| File | Change |
|------|--------|
| `scripts/ablation/sector_neutral_replay.py` | `build_screen_row` adds `pct_to_52w_high` (= `last / 252d-rolling-max`, George-Hwang price-only). `win_lo` 200->260 (>=252d for the high). NEW `hi52_tilt_basket(ranked_all, k, top_n)` helper -- re-ranks the PRODUCTION-composite-scored rows by a CENTERED multiplicative tilt `composite * (1 + k*(pct_to_52w - mean))`. Two configs measured: `hi52_k0.5`, `hi52_k1.0`. Results table + a 52.1 verdict (keep if dSharpe>=+0.05 AND dTurnover<=+10%). |
| `backend/tests/test_phase_52_1_alpha_signal.py` | **NEW** 5 tests: tilt breaks ties toward higher-52wh; centered (no change when all-equal-pct); gentle k can't overturn a big composite gap; handles missing pct; `pct_to_52w_high` feature math. |

## The measurement (criterion #1/#2)
```
config            ann_Sharpe   avg_fwd_mo%  avg_turnover
baseline               1.388         4.054         0.555
hi52_k0.5              1.439         4.103         0.551   -> dSharpe +0.051, dTurnover -0.004  KEEP
hi52_k1.0              1.436         4.075         0.564   -> dSharpe +0.047, dTurnover +0.009  (borderline)
(sector_neutral 1.220 / vol_scaled 1.403 carried from 51.2 for context)
```
- **k=0.5 PASSES** the gate (+0.051 Sharpe, turnover-neutral); k=1.0 is borderline (+0.047) -> use the gentle k=0.5; the k-monotonicity (k=0.5 > k=1.0) argues against over-tilting/over-tuning.
- **Honestly small + noisy:** a preview run showed +0.057/+0.054; this run +0.051/+0.047 (live-yfinance drift). True edge ~+0.05 ann Sharpe -- modest, exactly the Barroso-Wang large-cap-mute prediction (52wh helps but is muted on large caps). NOT a game-changer; reported without overselling.

## Research basis (gate PASSED)
`research_brief.md` (researcher `a99666872066bb1fe`, 7 sources, gate_passed). 52wh chosen as the MEASURE-FIRST pick (already-implemented formula + one-line replay change); residual momentum (Blitz-Huij-Martens) is the higher-evidenced runner-up (bigger build) if 52wh measured flat. Echo/skip-month CONTRAINDICATED ("Echo disappears" 2023). Over-tuning guarded (k a priori, not swept).

## Verification command output (verbatim)
```
SYNTAX OK
$ python -m pytest backend/tests/test_phase_52_1_alpha_signal.py -q
.....                                                                    [100%]
5 passed in 0.23s
```
Full replay -> handoff/current/live_check_52.1.md.

## Scope / safety (criterion #3)
- **NO live engine change.** Diff = the replay script + the new test ONLY. screener.py / autonomous_loop / decide_trades / risk guards UNTOUCHED. The tilt is replay-side post-processing of the production composite.
- If escalated: the live wiring (a `strategy="momentum_52wh"` overlay, default-OFF) is a SEPARATE operator-gated step.

## Artifact shape
- `hi52_tilt_basket(ranked_all, k, top_n, mean_pct=None) -> [ticker]`
- `build_screen_row(...)["pct_to_52w_high"]` (price-only, (0,1])

## Recommendation / next
- **ESCALATE k=0.5 to a live operator gate** (a small, turnover-neutral, research-backed momentum overlay; honest ~+0.05 Sharpe edge -- operator weighs it vs wiring complexity).
- RUNNER-UP: residual momentum (52.2) for a larger edge.
- MEASURE Monday's first multi-market cycle before stacking live changes.
