# Experiment Results — phase-28.9 — Options-flow OI-surge filter

**Step ID:** phase-28.9
**Date:** 2026-05-17
**Cycle:** 1

---

## What was built / changed

### Files modified
| File | Change |
|---|---|
| `backend/config/settings.py` | Added 9 fields after russell1000 block: `options_flow_screen_enabled` (False) + 8 calibration knobs (otm threshold, dte range, vol multipliers, boost magnitudes, cache). |
| `backend/tools/screener.py` | Added `options_surge_signals=None` kwarg to `rank_candidates`. Apply block in per-stock loop AFTER sector_momentum (mirror of phase-28.1 pattern). |
| `backend/services/autonomous_loop.py` | Added flag-conditional pre-fetch of surge signals for top 2*paper_screen_top_n candidates (cost-bounded). Passes to rank_candidates. |

### Files created
| File | Purpose |
|---|---|
| `backend/services/options_flow_screen.py` | New 165-line module. `OptionsSurgeSignal` Pydantic model + `fetch_oi_surge_signals(...)` async (per-ticker yfinance, Semaphore(4), 0.3s throttle, DTE+OTM+vol filters) + `apply_options_surge_to_score(...)` helper. |

---

## Verification — verbatim output

### 1. Immutable verification command

```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/options_flow_screen.py').read()); from backend.services.options_flow_screen import fetch_oi_surge_signals; print('importable')" && grep -q 'options_flow_screen_enabled' backend/config/settings.py && echo "MASTERPLAN VERIFICATION: PASS"
importable
MASTERPLAN VERIFICATION: PASS
```

EXIT 0. **PASS.**

### 2. Live `fetch_oi_surge_signals` — 5 large-caps, real yfinance options chains

```
INFO options_flow_screen: options_flow_screen: 5/5 tickers flagged (strong>=2 surges +0.06; moderate=1 +0.03)

  NVDA: n_surges=1  max_vol_oi=4.48     max_vol_avg=5.39   boost=1.03  strikes=[270.0]
  TSLA: n_surges=11 max_vol_oi=2750.0   max_vol_avg=17.10  boost=1.06  strikes=[427.5, 430.0, 430.0, 435.0, 455.0, 427.5, 430.0, 447.5, 605.0, 430.0]
  AAPL: n_surges=5  max_vol_oi=17.76    max_vol_avg=20.70  boost=1.06  strikes=[305.0, 305.0, 315.0, 305.0, 305.0]
  MSFT: n_surges=6  max_vol_oi=43.36    max_vol_avg=17.37  boost=1.06  strikes=[440.0, 430.0, 435.0, 440.0, 450.0, 455.0]
  META: n_surges=7  max_vol_oi=1000000  max_vol_avg=17.24  boost=1.06  strikes=[642.5, 765.0, 667.5, 675.0, 695.0, 625.0, 667.5]

--- apply smoke ---
  NVDA: base=10.000 -> 10.300 ( +3.0%)
  TSLA: base=10.000 -> 10.600 ( +6.0%)
  AAPL: base=10.000 -> 10.600 ( +6.0%)
  MSFT: base=10.000 -> 10.600 ( +6.0%)
  META: base=10.000 -> 10.600 ( +6.0%)
```

**Result:** All 5 large-caps flagged today. NVDA at moderate (1 surge → +3%); TSLA/AAPL/MSFT/META all at strong (multiple surges → +6%). Signal is firing on real elevated OTM near-expiry call activity.

### Calibration observation (honest disclosure)

5/5 mega-caps flagged might suggest the thresholds are LOOSE for high-volume names. The Wayne State signal is about strike-level UNUSUAL activity; mega-caps always have liquid OTM calls, so the surge predicate triggers more easily. **Operator should consider tightening `options_vol_avg_multiplier` to 8.0 or `options_vol_oi_multiplier` to 5.0 for mega-cap-heavy universes.** Default 5.0 / 3.0 is the Wayne State practitioner default for the median-cap stock; not changed because default-OFF means operator A/B-tests before flipping.

---

## Success criteria mapping

| Criterion (immutable) | Evidence | Result |
|---|---|---|
| `options_flow_screen_module_created` | `backend/services/options_flow_screen.py` exists, importable | PASS |
| `OTM_near_expiry_volume_threshold_documented` | 8 settings fields with descriptions citing Wayne State; module docstring documents the full predicate | PASS |
| `feature_flag_options_flow_screen_enabled_default_false` | `Settings().options_flow_screen_enabled == False` | PASS |
| `wired_into_rank_candidates_or_meta_scorer` | `rank_candidates` accepts `options_surge_signals` kwarg; apply block in per-stock loop calls `apply_options_surge_to_score` | PASS |
| `live_check_lists_OI_surge_candidates_for_one_cycle` | live_check_28.9.md lists 5/5 surge candidates with strike numbers and boost multipliers | PASS |

---

## Next

Q/A pass. On PASS: append Cycle 24, flip phase-28.9. Post-launch tier: 4/7.
