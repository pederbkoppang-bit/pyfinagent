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

---

## ADVERSARIAL PROMOTION-READINESS REVIEW (2026-07-09, workflow wf_cff750df)

24 agents (4 per-flag/interaction reviewers + 20 verifiers). **Net: PROMOTE-
WITH-CAVEATS on all four; ZERO blockers.** Full dossier:
promo_readiness_review_2026-07-09.txt. Verified findings:

### synthesis-integrity -> SAFE TO PROMOTE (one gap FIXED this commit)
- C1 (was high->medium): lite fallback can return recommendation=None ->
  `None.upper()` crashed decide_trades. **FIXED** now (None-safe guard at
  portfolio_manager.py:135/:177, ungated crash fix, test added). Fail-safe
  even before the fix (crash = no trade = stays cash) but real.
- C2 (low): the `"scoring_matrix" not in synthesis` predicate can over-fire
  (route a healthy critic-corrected report to lite) and under-catch (empty
  `{}` still fabricates 0.0). FOLLOW-UP: tighten to `synthesis.get("error") or
  not synthesis.get("scoring_matrix")`. Non-blocking.
- C5 (low): all-degraded cycle -> rows dropped before _degraded_scoring_check
  -> the P1 "all degraded" alarm can miss (60.1 fallback-rate path still
  fires). FOLLOW-UP: count degraded-None in the guard. Non-blocking.

### rj-shape -> SAFE TO PROMOTE (strictly safer on a cash book)
- Confirmed: ON makes full-path BUYs SMALLER (judge pct 2-5% vs the 10%
  default) + records the decision. Pure risk reduction for reactivation.
- FOLLOW-UPs (both UNREACHABLE on the 100%-cash book -- swaps need positions):
  swap-path :649/:674 still `or 10.0` (0% not honored on swaps);
  _extract_stop_loss (line 210) reads flat `risk_assessment` not `_rj_view`
  so full-path BUYs use the 8% default stop, not the judge's nested stop.
  Both are flag-gated + position-dependent -> fix before positions accumulate.

### position-rec -> **HOLD (do NOT promote yet)**
- interactions-C1 (CONFIRMED high): the unsafe-combination guard is WARN-ONLY,
  not a block. ON + a synthetic-HOLD-shaped output on a HELD ticker's re-eval
  -> signal_downgrade SELL of a healthy position. And even with synthesis-
  integrity ON, a lite parse-fail returns a real HOLD (not degraded), so the
  hazard is not fully closed. Inert now (0 positions) but a real downside once
  the book holds anything. REQUIRE before promoting: harden the guard to a
  hard block (or gate signal_downgrade on the flag) + prove synthesis-integrity
  live. rollback is also not perfectly clean (rule not flag-gated; only the
  WRITE is) -- another reason to hold until positions exist.

## REFINED RECOMMENDATION
**PROMOTE `synthesis-integrity` + `rj-shape`** (the earn-again levers; safe/
safer on the cash book; the one real crash gap is fixed). **HOLD
`position-rec`** until positions exist AND its guard is hardened. Validation
plan unchanged: on token -> set flags in .env, restart, run-now flags-ON
cycle, report ON-vs-OFF delta.
