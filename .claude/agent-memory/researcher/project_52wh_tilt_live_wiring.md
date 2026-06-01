---
name: 52wh-tilt-live-wiring
description: phase-52.2 exact wiring of the measured 52-week-high momentum tilt into LIVE screener.rank_candidates as a config-gated default-OFF post-pass; insertion point, flag pattern, byte-identity proof
metadata:
  type: project
---

phase-52.2 wires the phase-52.1-MEASURED centered multiplicative 52wh tilt
(`tilted = composite * (1 + k*(pct_to_52w - universe_mean))`, k=0.5, +0.05 ann
Sharpe turnover-neutral on the S&P-500 replay) into LIVE `rank_candidates` as a
CONFIG-GATED cross-sectional post-pass, DEFAULT OFF, byte-identical when off.
Enable (flag flip) is DEFERRED to a separate post-Monday-baseline operator action.

**Why:** operator's #1 constraint is DON'T REGRESS the working live engine (+20%
NAV) -> the wiring MUST be byte-identical when the flag is OFF (the default),
proven by a test. Wire-now/enable-later is the canonical dark-launch pattern
(Fowler, LaunchDarkly) + the AlgoXpert IS->WFA->OOS lock-after-WFA gate.

**How to apply (the exact wiring, all file:line CONFIRMED 2026-06-01):**
- INSERTION POINT = `backend/tools/screener.py:473` -- AFTER the sector_neutral
  post-pass (ends :472) and IMMEDIATELY BEFORE `scored.sort(...)` (:474). Runs
  after every composite_score + RSI/vol multipliers + both existing gated
  re-scoring passes (multidim :434-440, sector_neutral :448-472). Honoured by the
  single final sort :474.
- FLAG = new kwargs `momentum_52wh_tilt: bool = False` + `momentum_52wh_tilt_k:
  float = 0.5` on `rank_candidates` (signature :249-273, beside
  `multidim_momentum: bool = False` at :261). Guard `if momentum_52wh_tilt and
  scored:` -> SKIPPED when False -> byte-identical to today.
- HELPER mirrors `_apply_multidim_momentum` (:491-550): mutates `scored` in place,
  writes `composite_score_raw` ONLY when it runs (so its ABSENCE on OFF-path rows
  is the byte-identity witness). Faithful to the 52.1 reference
  `scripts/ablation/sector_neutral_replay.py:123-138 hi52_tilt_basket` (same
  centered formula, same mean-over-non-None-pcts, same missing->tilt 1.0 no-op).
- SETTINGS = `momentum_52wh_tilt_enabled: bool = Field(False, ...)` +
  `momentum_52wh_tilt_k: float = Field(0.5, ...)` in
  `backend/config/settings.py` beside the `multidim_momentum_*` block (:334-338).
- CALL-SITE = `backend/services/autonomous_loop.py:638-666` (the ONLY live caller;
  import :25). Add `momentum_52wh_tilt=getattr(settings,
  "momentum_52wh_tilt_enabled", False)` + `momentum_52wh_tilt_k=getattr(settings,
  "momentum_52wh_tilt_k", 0.5)` beside the multidim args at :649-654.
- FEATURE ALREADY AVAILABLE AT RANK TIME: `pct_to_52w_high` is computed INSIDE
  screen_universe's per-ticker loop (screener.py:210-214) and set on EVERY row
  (:228) for all screened names (not just top-N) -> NO threading needed; it
  already flows screen_universe -> rank_candidates. min_periods=20 guard -> None
  for short windows -> helper treats as tilt 1.0.
- BYTE-IDENTITY TEST: assert `rank_candidates(data)` ==
  `rank_candidates(data, momentum_52wh_tilt=False)` AND no `composite_score_raw`
  on any OFF row; ON path flips the ranking (sanity).

This is the SAME gated-default-OFF idiom as [[strategy-rotation-infra]] overlays
and the sector_neutral/multidim post-passes. Reason NOT to wire it: none when OFF
(zero regression by construction); residual risk (gradual momentum decay once
ENABLED, arXiv:2512.11913) belongs to the deferred enable decision, deflated via
DSR for the 5 configs tried (Bailey-Lopez de Prado).
