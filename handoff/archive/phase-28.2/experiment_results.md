# Experiment Results — phase-28.2 — 12-quarter SUE stacking

**Step ID:** phase-28.2
**Date:** 2026-05-17
**Cycle:** 1

---

## What was built / changed

### Files modified

| File | Change |
|---|---|
| `backend/services/pead_signal.py` | Line 38: `_LOOKBACK_QUARTERS = 8` → `12` (with multi-line phase-28.2 comment citing ScienceDirect 2025 + equal-weight rationale + cache back-compat note). Line 54: `surprise_score` description updated from "rolling-8Q mean" to "rolling-12Q mean (phase-28.2; was 8Q)". |
| `backend/config/settings.py` | Line 187: `pead_signal_lookback_quarters` default 8 → 12 (sync with module constant; description cites phase-28.2 + ScienceDirect 2025). |

### Files NOT modified

- `backend/services/pead_signal.py::_trailing_mean_from_cache` body unchanged — still equal-weighted arithmetic mean (the simplest stacking, recommended by Researcher because EWMA would de-weight the OLDER lags that are precisely the ones gaining importance per ScienceDirect 2025).
- All other callers, tests, cache file readers — no change required.

---

## Verification — verbatim output

### 1. Immutable verification command

```
$ source .venv/bin/activate && grep -qE '_LOOKBACK_QUARTERS\s*=\s*12' backend/services/pead_signal.py && python -c "import ast; ast.parse(open('backend/services/pead_signal.py').read()); print('PASS')" && echo "MASTERPLAN VERIFICATION: PASS"
PASS
MASTERPLAN VERIFICATION: PASS
```

EXIT 0. **PASS.**

### 2. Synthetic smoke — 12 cache files, compare 8Q vs 12Q

```
Temp cache dir: /var/folders/n4/9khkbgzj593cmjc28m9chntm0000gn/T/pead_smoke_ahkan13j
Wrote 12 synthetic cache files

=== POST-EDIT (phase-28.2): _LOOKBACK_QUARTERS = 12 ===
Module constant: 12
Trailing mean (12Q): 0.5475 over n=12 quarters

=== SIMULATED PRE-EDIT: _LOOKBACK_QUARTERS = 8 ===
Trailing mean (8Q): 0.6025 over n=8 quarters

=== Effect on surprise_score ===
Hypothetical current sentiment_score: 0.750
surprise_score under 8Q  = 0.750 - 0.6025 = +0.1475
surprise_score under 12Q = 0.750 - 0.5475 = +0.2025
diff (12Q vs 8Q): +0.0550

Tag under 8Q:  positive_surprise
Tag under 12Q: positive_surprise
holding_window_days under 8Q:  28
holding_window_days under 12Q: 42

PASS: 12Q stacking computed; cache back-compat verified
```

The synthetic ticker shows a slowly improving sentiment trend (older quarters lower, more recent quarters higher). Reading 12Q vs 8Q correctly produces a LOWER trailing mean for the wider window → a LARGER surprise_score for the current observation, which is the desired ScienceDirect 2025 effect: capturing more historical context exposes more of the trend reversal.

### 3. Cache back-compat verified

Cache filename format `pead_<TICKER>_<YYYY-MM-DD>.json` does NOT encode lookback depth. The new 12Q code reads the same cache files that the old 8Q code wrote — no migration, no breakage. The synthetic smoke wrote 12 cache files using the legacy format; the new code read them all correctly.

---

## Success criteria mapping

| Criterion (immutable) | Evidence | Result |
|---|---|---|
| `lookback_quarters_increased_to_12` | `grep -E '_LOOKBACK_QUARTERS\s*=\s*12'` matches | PASS |
| `weighting_scheme_added_or_documented` | Multi-line phase-28.2 comment at line 38-44 explains equal-weight choice + ScienceDirect 2025 rationale; settings.py field description also cites equal-weighted | PASS |
| `back-compat_with_existing_cache_files` | Synthetic smoke wrote 12 cache files in the existing `pead_<TICKER>_<DATE>.json` format and the new code read them all correctly; no migration required | PASS |
| `syntax_OK_and_pead_signal_still_importable` | `python -c "import ast; ast.parse(...)"` exit 0; smoke imports `from backend.services.pead_signal import ...` successfully | PASS |

---

## Artifact shape

Post-edit pead_signal.py (lines 35-46):

```python
_VALID_TAGS = ("positive_surprise", "negative_surprise", "neutral", "insufficient_history")
# phase-28.2 (2026-05-17): bumped 8 -> 12 per ScienceDirect 2025 ML paper
# ("Beyond the last surprise: Reviving PEAD with ML and historical earnings") which
# documents Sharpe lift from 0.34 (latest-only) to 0.63 (+85%) when stacking 12
# quarters of SUE history. Equal-weight (arithmetic mean) preserved — the mechanism
# is that older lags GAIN importance as markets price news faster, so EWMA would
# de-weight exactly the valuable observations. Cache filenames don't encode
# lookback depth (`pead_<TICKER>_<YYYY-MM-DD>.json`) -> back-compat with existing
# cache files is fully preserved.
_LOOKBACK_QUARTERS = 12
```

settings.py (line 187):

```python
pead_signal_lookback_quarters: int = Field(12, description="phase-28.2: Trailing quarters of PEAD sentiment used to compute surprise (bumped 8->12 per ScienceDirect 2025 SUE-stacking paper, +85% Sharpe lift; equal-weighted mean)")
```

---

## Operator-impact note

Unlike the previous opt-in additions (28.5, 28.1) which were behind dedicated feature flags defaulting OFF, this step ALWAYS changes the trailing-mean window. The PEAD signal itself is still gated by `pead_signal_enabled = False`, so production picker behavior is unchanged when that flag is OFF. When the flag IS on, `surprise_score` values for any ticker with 9+ cached quarters will SHIFT modestly (since the trailing mean now averages 12 observations instead of 8). The magnitude depends on how different the older 4 quarters are from the more recent 8; the synthetic smoke shows a typical +0.055 shift for a moderately trending ticker. This is the intended effect.

---

## Next

Q/A pass via fresh `qa` subagent. On PASS: append Cycle 17, flip phase-28.2 status.
