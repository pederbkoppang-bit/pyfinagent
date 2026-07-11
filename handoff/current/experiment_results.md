# Experiment Results — Step 69.3 (P1 signal integrity + $0 free-data lift)

- **Phase / step**: phase-69 → 69.3
- **Date**: 2026-07-11
- **Type**: LIVE signal-integrity fixes + a $0 free-data lift, all flag-gated default-OFF (live engine byte-identical). historical_macro untouched.

## What was changed

1. **Sign-safe overlays (criterion 1)** — all **14** `apply_*_to_score` overlays now route through the shared
   flag-gated `sign_safe_mult(base, mult)` helper (`backend/services/overlay_math.py`, committed in 69.1):
   news_screen, macro_regime (conviction + sector tilt), pead_signal, options_flow_screen,
   insider_signal_screen, peer_leadlag_screen, analyst_revisions, call_transcript_gpr, sector_momentum,
   analyst_narrative_scorer, social_velocity_screen, ma_preannounce_screen, defense_signal, sector_calendars.
   Flag `sign_safe_overlays` default-OFF = byte-identical legacy `base*mult`. (The 3 inline base-score
   penalties at screener:306/308/311 are LEFT for operator decision, per the research brief — NOT changed.)
2. **News token-cap + retry (criterion 2)** — `news_screen.py`: the truncating `min(8192, 250*len(deduped))`
   (froze at 8192 for >32 headlines → JSON truncated → `{}`) is replaced with `min(48000, max(8192, ...))`
   (claude-haiku-4-5 max output = 64k) + a parse-fail retry (2 attempts).
3. **QMJ Growth (criterion 3)** — `historical_data.py`: `revenue_growth_yoy` is now assigned BEFORE the QMJ
   Growth read (was ~51 lines after → always None → Growth dimension dead). Deps in scope earlier; safe move.
4. **INDPRO + net-liquidity (criterion 4)** — INDPRO added to `fred_data.SERIES` (was referenced in
   `_REGIME_SERIES` but never fetched); new `macro_regime._fetch_net_liquidity` (WALCL−WTREGEN−RRPONTSYD×1000,
   24h file cache, existing free FRED key, **writes NO BQ**) + a net-liq regime-prompt line. The INDPRO +
   net-liq regime-prompt INCLUSION is gated behind `regime_net_liquidity` (default-OFF → the live regime
   prompt is BYTE-IDENTICAL; verified `off == pre-fix prompt`).

## Verification command output (verbatim)

```
$ python -m pytest backend/tests/test_signal_integrity_69.py -q --timeout=180
............                                                             [100%]
12 passed in 0.04s
```

Ruff gate (qa.md §1a) on all 17 touched files + the test: **All checks passed!** (exit 0). (8 PRE-EXISTING
F401 unused-imports — confirmed in HEAD, not a 69.3 regression — were auto-removed via `ruff --fix -select
F401`; the 6 affected modules re-import OK; 1054 tests collect.)

## $0 live ON-vs-OFF check (criteria 1 + 4) — no metered LLM call

```
ON-vs-OFF live ranking (base=-10.0; AAA=+catalyst boost, BBB=-catalyst penalty):
  sign_safe_overlays=False -> AAA=-11.00  BBB=-9.00  higher-rank=BBB(-catalyst)     # INVERTED
  sign_safe_overlays=True  -> AAA=-9.00   BBB=-11.00 higher-rank=AAA(+catalyst)     # FIXED
  => OFF ranks the NEGATIVE catalyst higher; ON ranks the POSITIVE catalyst higher (inversion eliminated).

Regime prompt:
  OFF: INDPRO present=False  NET_LIQUIDITY present=False  (byte-identical to pre-fix prompt: True)
  ON : INDPRO present=True   NET_LIQUIDITY present=True
    - INDPRO (IP): current=103.200 previous=102.9 trend=rising as_of=2026-07-01
    - NET_LIQUIDITY (Fed WALCL-TGA-RRP, $M): current=6100000 trend=rising as_of=2026-07-01 [rising -> risk_on lean]
```

## Do-no-harm evidence

- **Live engine byte-identical when flags OFF**: `sign_safe_mult` OFF = `base*mult` (fixture over a grid);
  the regime prompt OFF == the pre-fix (INDPRO-absent) prompt (fixture). Both flags default-OFF.
- **historical_macro FROZEN**: the net-liq path writes NO BQ (`_fetch_net_liquidity` has no `insert_rows` /
  `bigquery` / `_bq(` / `.query(` — fixture-asserted); uses a file cache + the existing free FRED key.
- **No regressions**: 1054 tests collect; the 6 ruff-fixed modules re-import OK.
- **No conflict with phase-68**: overlays/regime ≠ fills.

## Deferred (operator tokens)
- Final IC / ablation / optimizer validation of the sign-safe ranking + the net-liquidity feature is DEFERRED
  behind the historical_macro un-freeze token. The code + flags + the $0 ON-vs-OFF proof do NOT require it.
- Activating the live behavior change is the operator's call: flip `sign_safe_overlays` and/or
  `regime_net_liquidity` after reviewing this ON-vs-OFF evidence.
