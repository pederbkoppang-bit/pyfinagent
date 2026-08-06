---
name: exit-quality-ratio-blowup-82-5
description: Phase-82.5 exit-quality tiles -- measured facts on capture/edge ratio blowup, the frontend NOT double-scaling, MFE clamped at 0 by construction, and the asymmetric degenerate-case treatment
metadata:
  type: project
---

Measured 2026-08-04 while researching phase-82.5 (Exit-quality tiles reading -4208% and 86.92).

**The mean of these ratios does not merely have outliers -- it DOES NOT EXIST.**
A ratio whose denominator can reach zero is Cauchy-like: Franz (arXiv:0710.2024, read in
full via the ar5iv chain) proves the expected value and variance do not exist, and that
*the mean of n IID Cauchy variables follows the same Cauchy distribution as one of them*.
So "collect more trades and it will settle" is false. Any fix must change the DEFINITION,
not the estimator. Winsorizing/clipping is also wrong -- it yields a finite number with no
population parameter to estimate.

**Non-obvious measured facts (re-derive line numbers, they move):**

- **The frontend does NOT double-scale.** `MfeMaeScatter.tsx:114` does
  `(avg_capture_ratio * 100).toFixed(0)`, and `capture_ratio = realized_pnl_pct / mfe_pct`
  -- percent over percent, so it is DIMENSIONLESS and the x100 is CORRECT. -42.08 x 100 =
  -4208% exactly. Corroborated by `:111` (edge ratio has NO x100) and `:121` (threshold
  0.4 -> "40%"). Do not "fix" the formatter; if the backend ever emits a value already in
  percent, that multiply becomes a real double-scale.
- **`mfe == 0` is a CLAMP ARTEFACT, not a measurement.** `paper_trader.py:718-721` seeds
  `prev_mfe = float(pos.get("mfe_pct") or 0.0)` then `new_mfe = max(prev_mfe, pnl_pct)`, so
  MFE is floored at 0 by construction. A trade whose best mark was -3% records mfe = 0. The
  true MFE for those rows is unrecoverable from the stored column. This MATCHES the industry
  convention (TradingDiaryPro: *"if the trade ... was never in the profit zone then the MFE
  is zero"*) -- so it is not a bug to fix, it is a domain restriction to respect.
- **The degeneracies are ASYMMETRIC.** `mae == 0` = "never traded against us" = a genuine,
  desirable property whose edge ratio is +inf; excluding it (the `if mae_abs > 0` filter)
  deletes the BEST trades -- survivorship bias with the sign pointing the wrong way.
  `mfe == 0` = "no exit decision to grade" -- the ENTRY failed; including it at a fabricated
  0.0 blames the exit. **Exclude for capture, KEEP for edge.**
- **A median is defined over the extended reals; a mean is not.** +inf rows can be RANKED
  rather than deleted, so a median needs no exclusion while <50% of rows are degenerate.
  The `if mae_abs > 0` filter exists ONLY because the code chose a mean.
- **TWO independent copies of the same broken mean:** the endpoint
  (`paper_trading.py:1031-1032`) and `paper_round_trips.py:157` (feeding `/performance ->
  round_trip_summary`). The second is open finding **55.3 F-10** (`avg_capture_ratio=-53.7`,
  filed 2026-06, never fixed). Patching one leaves the other blown up.
- **Second-order defect from the same root cause:** the leakage rule at
  `paper_trading.py:1027` tests `p["capture_ratio"] < 0.4`, which the fabricated 0.0 always
  satisfies -- so every `mfe == 0` row is silently eligible to be flagged an "exit leaker".
- **No promotion/sizing path reads these metrics.** `paper_go_live_gate.py:131` uses only
  `len(pair_round_trips(trades))` for `trades_ge_100`; the other five consumers read only
  `realized_pnl_usd` or prices. Grep found zero occurrences in `backend/backtest/`,
  `backend/meta_evolution/`, `backend/slack_bot/`, or any `*.sql`. **But do NOT touch the
  pairing loop** -- that WOULD move a promotion boolean.
- **The house zero-denominator precedent already exists and is being violated:**
  `sovereign_api.py:566-569` -- *"Zero-denominator contract: return None for the ratio, not
  infinity."* The exit-quality path (`else 0.0`) predates it.
- **No robust-aggregation helper exists in the repo.** `perf_metrics.py` (the documented
  single source of truth for metrics) has no median/trimmed/winsorized estimator. Closest
  reusable idiom is the hand-rolled median already inside the defective module itself
  (`paper_round_trips.py:153-154`, `median_holding_days`).
- **Published interpretive scale for capture is bounded in [0,1]:** <0.40 noise-driven
  exits, >0.50 solid, 0.60+ trend setups, 0.65-0.80 well-optimized, >0.75 excellent. A tile
  reading -4208% is off the scale entirely, not a bad score on it.
- **Correct aggregation of a SET of ratios weights each by its own denominator**, which is
  algebraically the ratio of sums (ProcessExcellenceNetwork's weighted-average procedure ==
  Wikipedia's `r = sum(y)/sum(x)`). BUT PMC2430201 warns that ratio-of-sums itself degrades
  when the denominator is *"heavily skewed towards zero"* -- which is exactly this data
  (8/32 at exactly 0). So ratio-of-sums is a SECONDARY readout, never the headline here.

**Sourcing note:** QuantifiedStrategies (the canonical Sweeney-anchored MAE/MFE page) is
bot-blocked to BOTH WebFetch and `curl` + tag-strip -- the curl fallback returns a 72-byte
interstitial. Sweeney's *Campaign Trading* (1996) is offline-only. Use TradingDiaryPro +
docs.tradingmetrics.com instead; do not burn calls retrying QuantifiedStrategies.

See [[research-gate-discipline]] and [[fabricated-safe-80-36]] -- the `else 0.0` here is the
same absence-becomes-affirmative class.
