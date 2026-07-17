# Experiment results — step 70.2 (S2: soft, profit-aware cross-sector diversification)

**Phase/step:** phase-70 → 70.2 | **Date:** 2026-07-17 | **Type:** backend + ML, flag-gated default-OFF, $0, paper-only, DARK-until-token

## Files changed (6)

1. **`backend/config/settings.py`** — 4 flags (all default-OFF/identity): `paper_soft_sector_diversity_enabled`
   (bool F), `paper_soft_sector_diversity_w` (float 0.0, ge0 le1), `paper_min_k_sectors_analyzed` (int 0, ge0 le11),
   `paper_unknown_sector_cap_exempt` (bool F).
2. **`backend/tools/screener.py`** — `_apply_soft_sector_diversity(scored, w)`: within each sector the j-th
   (0-based) name by raw composite is shaded by `(1-w)^j` via the canonical SIGN-SAFE `overlay_math.sign_safe_mult`
   (forced enabled), so a penalty lowers rank even for a negative score (no sign inversion). Leader (j=0)
   untouched → keeps across-sector momentum → NOT hard neutralization. Params `soft_sector_diversity=False`,
   `soft_sector_diversity_w=0.0`; the block runs after all overlays, before `scored.sort()`; `w=0`/OFF → skipped.
3. **`backend/services/autonomous_loop.py`** — `_min_k_sector_slice(cands, n, k)` round-robin leader-pick on the
   deep-analyze slice (:838) when K>0 (else plain slice); added the diversity flag to the `build_sector_map` gate
   (:433) so candidates carry a sector at rank time; threaded the two soft kwargs into the `rank_candidates` call.
4. **`backend/services/portfolio_manager.py`** — `_unk_exempt` guards the count cap (:359) and NAV-pct cap (:394)
   so the "Unknown" (missing-sector) bucket is exempt when `paper_unknown_sector_cap_exempt` is ON. OFF → byte-identical.
5. **`scripts/ablation/sector_neutral_replay.py`** — soft-diversity configs (`soft_w0.10/0.20/0.30`) + a 70.2 verdict
   + a paired-monthly-returns dump (`handoff/current/_70_2_soft_diversity_replay.json`) for the DSR/PBO activation gate.
6. **`backend/tests/test_phase_70_2_soft_diversity.py`** (NEW) — 7 deterministic (network-free) tests.

## Verification command output (verbatim)

```
$ bash -c 'grep -Eqi "sector" backend/services/autonomous_loop.py && ls backend/tests/ | grep -Eqi "70_2|diversif|sector" && python -c "import ast; ast.parse(open(\"backend/services/autonomous_loop.py\").read())"'
VERIFICATION: PASS (exit 0)
$ python -m pytest backend/tests/test_phase_70_2_soft_diversity.py -q
7 passed
```
Import-smoke: settings/screener/autonomous_loop/portfolio_manager all import clean; helpers present; flags default False/0.0/0/False.

## Criterion 1 — analyzed set spans ≥2 sectors (ON) vs monosector (OFF) [live_check ON-vs-OFF]

Deterministic (`test_min_k_slice_reproduces_and_diversifies`): a monosector-heavy candidate list (5 Technology
+ 1 Energy + 1 Health Care) →
- **OFF** (plain top-5 slice) = **1 sector** — reproduces today's monosector funnel.
- **ON** (`_min_k_sector_slice`, K=3) = **3 distinct sectors** {Technology, Energy, Health Care}, best names still first.

Corroborated by the ablation basket breadth below (avg distinct sectors 4.71 → 5.96/6.73/7.31 as w rises).

## Criterion 2 — SOFT + no OOS P&L drop (it RAISES risk-adjusted P&L) [$0, macro-free ablation replay]

`scripts/ablation/sector_neutral_replay.py` — replays PRODUCTION `rank_candidates` over 47 monthly rebalances
(2022–2025, S&P 500), $0 (yfinance + Wikipedia), NO LLM/BQ/historical_macro/optimizer (historical_macro FROZEN
respected). ann_Sharpe of the equal-weight top-10 basket:

| config | ann_Sharpe | Δ vs base | avg_sectors | avg_turnover |
|---|---|---|---|---|
| baseline (OFF) | 1.344 | — | 4.71 | 0.557 |
| **soft_w0.10** | **1.520** | **+0.176** | 5.96 (+1.25) | 0.543 |
| **soft_w0.20** | **1.543** | **+0.200** | 6.73 (+2.02) | 0.545 |
| **soft_w0.30** | **1.578** | **+0.234** | 7.31 (+2.60) | 0.549 |
| sector_neutral (HARD — rejected) | 1.226 | **-0.117** | 10.00 | 0.638 |

The SOFT penalty **raises** OOS Sharpe at every tested w (does not lower risk-adjusted P&L) AND increases sector
breadth, with turnover slightly LOWER than baseline. HARD sector-neutralization **lowers** Sharpe (-0.117) —
re-confirming the 2026-06-01 replay and justifying the soft (not hard) design. Replay verdict:
"ESCALATE to operator activation gate (then DSR>=0.95 + PBO<=0.5 on the dumped paired returns)". Paired monthly
returns dumped for the activation-gate DSR/PBO computation.

## Criterion 3 — Unknown-bucket enrichment failure no longer freezes the funnel

Deterministic (`test_unknown_exempt_off/on`): 2 held positions with a MISSING sector + a new missing-sector
candidate, cap=2 →
- **OFF** (default): the new BUY is BLOCKED (Unknown counts as one bucket at cap) — byte-identical to today.
- **ON**: the BUY is ALLOWED — the Unknown (missing-data) bucket is exempt, so an enrichment outage can't
  collapse N real sectors into one bucket and starve the funnel.

## Criterion 4 — flag OFF → byte-identical

Deterministic (`test_soft_off_and_w0_byte_identical`): `rank_candidates` with the soft flag OFF (and with
`w=0.0` ON) yields identical ticker order + identical composite scores as the no-param call, and no
`composite_score_raw` side-channel is written. Min-K K=0 → plain slice; Unknown-exempt OFF → caps enforced as
today. Every lever defaults to the identity path.

## Do-no-harm / scope
Backend + ablation-script only; $0 metered (free yfinance + Wikipedia); paper-only; NO risk-limit threshold /
stop / kill-switch / DSR/PBO gate moved; historical_macro FROZEN (ablation is macro-free); hysteresis untouched;
hard sector-neutralization rejected. All live-loop behavior is DARK until the operator flips the flags —
activation is gated on OOS Sharpe >= incumbent (met: +0.18..+0.23) + DSR>=0.95 + PBO<=0.5 (computed at the token
from the dumped paired returns). No operator config mutated.
