# Flag-promotion decision brief -- 2026-07-09 (operator decision)

Context: phase-66.2 is the only open step; the book has earned nothing for a
week because BUYs die to the 61.2 defect under rail degradation (synthesis
failure -> synthetic HOLD/0.0). The fixes are built, unit-tested, and deployed
DARK (flags default OFF). Promotion (flag ON) is a live trading-behavior change
and is the operator's call -- this brief is the decision input. I am NOT flipping
them without your token.

## The three dark flags

| Flag | ON behavior | Risk on the CURRENT 100%-cash book | Evidence |
|---|---|---|---|
| `paper_synthesis_integrity_enabled` | retry-on-empty (bounded 2, breaker-safe) + route synthesis-error to lite fallback + NULL degraded rows (never fabricate 0.0/HOLD) + meta-scorer rank-normalization | **LOW** -- reliability/integrity only; loosens NO gate. Worst case: <=2 extra rail retries per starved call. THIS is the fix that lets a BUY survive a rail hiccup. | 33 tests (test_phase_61_2*); defect reproduced live 2 consecutive cycles |
| `paper_risk_judge_shape_fix_enabled` | full-path BUYs size at the judge's pct (smaller than the 10% default) + REJECT binds + risk_judge_decision RECORDED (not '') + 0% = no-buy | **LOW / SAFER** -- makes BUYs smaller + properly gated; required for 66.2 criterion-1(a) ("risk_judge_decision recorded") | 8 tests (test_phase_66_2_risk_judge_shape); money-audit adversarially verified |
| `paper_position_recommendation_fix_enabled` | revives signal_downgrade SELLs (needs an analysis rec on the position) | **INERT NOW** -- 0 positions, so it cannot fire until something is held; the unsafe-combo guard warns if ON without synthesis-integrity | 8 tests; blast radius only matters once positions exist |

## Why promote now (not "wait")

- The two flags that actually let the engine earn are synthesis-integrity (BUY
  survives rail hiccup) + RJ-shape (BUY sized + recorded). Both are LOW-risk on
  a cash book (they make trades SAFER, not looser).
- Today's manual test cycle (running now, flags OFF) shows the rail STILL
  failing (~15 fails early) even with the 150s timeout deployed -- i.e. the
  timeout alone is not the fix; the retry-on-empty (synthesis-integrity) is.
- position-recommendation is inert now -- promoting it is free (guarded), or
  hold it until the first BUY lands. Either is fine.

## Recommendation

**Promote `paper_synthesis_integrity_enabled` + `paper_risk_judge_shape_fix_
enabled`.** Hold or promote position-recommendation (inert either way; I lean
promote-with-the-pair since the unsafe-combo guard is satisfied once
synthesis-integrity is ON).

## Validation plan on your token (I execute immediately, no further ask)

1. Wait for the current manual cycle to finish (~10:08 UTC; can't run two -- lock).
2. Set the flag(s) in backend/.env + restart backend (bootout/bootstrap; the
   ON values are picked up at start).
3. Run ONE flags-ON test cycle via /run-now.
4. Report the ON-vs-OFF delta: did synthesis-error tickers route to lite / retry
   and produce real recommendations? did a BUY land with risk_judge_decision
   recorded + judge sizing? -- concrete BQ evidence.
5. If it produces a clean BUY: that is the money step working. The FORMAL 66.2
   close still wants a SCHEDULED-cycle BUY (39.1 doctrine), which the 18:00 UTC
   run then confirms with the flags live.

## Tokens (one word unblocks me)

- `PROMOTE SYNTHESIS-INTEGRITY` / `PROMOTE RJ-SHAPE` / `PROMOTE POSITION-REC`
  (any subset), or `PROMOTE ALL THREE`, or `HOLD` (keep dark, scheduled-only).

## Boundaries honored
No flag flipped without your token. No heavy subagent/workflow run while the
manual cycle's rail is live (quota-starvation lesson, 07-07). No fabricated
evidence. Trailing-stop untouched.
