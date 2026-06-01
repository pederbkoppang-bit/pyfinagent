# live_check -- phase-52.1: 52-week-high momentum tilt (price-only alpha signal)

**Step:** 52.1 | **Date:** 2026-06-01 | **Result shape:** a $0 replay measures the 52-week-high
multiplicative tilt ON-vs-OFF vs the baseline momentum ranking on the S&P 500 (48 monthly
rebalances). **Outcome: a SMALL but real POSITIVE edge at k=0.5 (~+0.05 Sharpe, turnover-neutral)
-> recommend ESCALATING to a live operator gate. NO live change in this step.**

**Command:**
```
source .venv/bin/activate && PYTHONPATH=. python scripts/ablation/sector_neutral_replay.py
```
Reuses the PRODUCTION `rank_candidates` composite (the tilt is replay-side post-processing -> zero
live-engine change). $0: free yfinance prices + Wikipedia sectors. The 52wh formula is the
already-in-prod `current_price / trailing-252d-high` (George-Hwang 2004).

## Verbatim result (helper version -- matches the committed script)
```
config            ann_Sharpe   avg_fwd_mo%  avg_sectors  avg_turnover
---------------------------------------------------------------------
baseline               1.388         4.054         4.73         0.555
sector_neutral         1.220         2.659        10.00         0.640
vol_scaled             1.403         2.045         4.73         0.555
hi52_k0.5              1.439         4.103         4.71         0.551
hi52_k1.0              1.436         4.075         4.69         0.564

--- VERDICT (52.1 52-week-high tilt) ---
hi52_k0.5 vs baseline: dSharpe=+0.051, dTurnover=-0.004 -> KEEP? True
hi52_k1.0 vs baseline: dSharpe=+0.047, dTurnover=+0.009 -> KEEP? False
52wh-tilt recommendation: ESCALATE to a live operator gate

N rebalances scored: 47
```

## Interpretation (criterion #1/#2 -- evidence-based, honestly reported)
- **The 52wh tilt at k=0.5 ADDS risk-adjusted value:** Sharpe 1.388 -> 1.439 (**+0.051**), forward
  return 4.054% -> 4.103%/mo, and turnover slightly LOWER (-0.004). Both legs improve at k=0.5.
- **It is a SMALL edge, and noisy run-to-run.** A preview run (live-data drift) showed +0.057/+0.054;
  this run +0.051/+0.047. So the true edge is ~**+0.05 ann Sharpe at k=0.5** -- right at the +0.05
  bar, modest. This is EXACTLY the Barroso-Wang large-cap-mute prediction: 52wh helps but is muted
  on large caps (the edge is real but small, not the small-cap-scale lift).
- **k=1.0 is borderline (+0.047, just under the bar):** over-tilting dilutes the gain -> use the
  GENTLE k=0.5. The k-robustness (both positive, k=0.5 > k=1.0) argues against over-tuning.
- **Recommendation: ESCALATE k=0.5 to a live operator gate** -- propose wiring it as a
  `strategy="momentum_52wh"` overlay (a SEPARATE operator-gated step), with the honest caveat that
  it's a ~+0.05 Sharpe edge (modest), so the operator weighs it against the wiring complexity.

## Criterion-by-criterion

| # | Criterion | Evidence | Verdict |
|---|-----------|----------|---------|
| 1 | a research-backed PRICE-BASED signal measured ON-vs-OFF reporting Sharpe/return/turnover | the table above (baseline vs hi52_k0.5/k1.0); 52wh = George-Hwang price-only formula | PASS |
| 2 | reuses production rank_candidates, identical screen_data both arms (sole delta = the tilt), causal fwd returns; honestly reported | the tilt post-processes `rank_candidates(... strategy="momentum")` output; both k reported (no cherry-pick); small-edge + drift disclosed | PASS |
| 3 | NO live engine change; US momentum core untouched | diff = the replay script + the new test ONLY (no screener.py / autonomous_loop / decide_trades change); the tilt is replay-side | PASS |
| 4 | live_check records the ON-vs-OFF comparison + cited basis + keep/reject recommendation | this file (ESCALATE k=0.5; cited George-Hwang + Barroso-Wang) | PASS |

## Honest caveats (no overselling)
- **Small edge, near the threshold + noisy** (+0.047 to +0.057 across runs). A +0.05 Sharpe lift is
  modest; the operator should decide if it's worth live-wiring vs leaving the working engine alone.
- **Single 48-rebalance sample, today's S&P-500 membership** (survivorship) -- the DELTA is what
  matters (hits both arms equally), but McLean-Pontiff factor-decay means the live edge may be
  smaller still. Do NOT over-tune k on this sample (k=0.5 chosen a priori, not swept).
- **Long-only, no short leg** -- 52wh-near names can fall hardest in a sharp reversal (the existing
  vol-penalty partially offsets).

## Recommendation / next
- **ESCALATE the k=0.5 52wh tilt to a live operator gate** as a separate step (a `momentum_52wh`
  screener overlay, default-OFF, operator-approved before live) -- a small, turnover-neutral,
  research-backed momentum enhancement.
- The higher-evidenced RUNNER-UP (single-factor residual momentum, Blitz-Huij-Martens) remains the
  bigger-build escalation if a larger edge is wanted (52.2).
- MEASURE Monday's first multi-market cycle before stacking live changes.
