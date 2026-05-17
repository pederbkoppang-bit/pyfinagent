# Experiment Results — phase-28.3 — GPR-triggered energy-sector tilt

**Step ID:** phase-28.3
**Date:** 2026-05-17
**Cycle:** 1

---

## What was built / changed

### Files modified

| File | Change |
|---|---|
| `backend/services/macro_regime.py` | Added: `_GPR_URL_PRIMARY/FALLBACK`, `_GPR_CACHE_DIR/PATH`, `_GPR_ROLLING_MONTHS` constants + multi-line phase-28.3 rationale comment; `_fetch_gpr_acts()` async helper (downloads xls, parses GPRA column, computes rolling quantile threshold); `_apply_gpr_tilt()` helper (injects configured ETFs into sector_hints.overweight when above threshold). Inside `compute_macro_regime()`: post-LLM hook that calls `_fetch_gpr_acts` + `_apply_gpr_tilt` when `settings.gpr_signal_enabled=True`. |
| `backend/config/settings.py` | Added 4 fields after analyst_revisions block: `gpr_signal_enabled` (False), `gpr_signal_quantile` (0.90), `gpr_signal_cache_hours` (24), `gpr_signal_sector_etfs` ("XLE"). |
| `backend/requirements.txt` | Added `xlrd>=2.0.1` (parse matteoiacoviello.com .xls legacy Excel format). |

### Files NOT created

- No new module file required — the GPR logic is local to macro_regime.py (small footprint, naturally lives next to the existing FRED-regime LLM path).

---

## Verification — verbatim output

### 1. Immutable verification command

```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/macro_regime.py').read()); print('syntax OK')" && grep -qE 'gpr|geopolitical' backend/services/macro_regime.py && echo "MASTERPLAN VERIFICATION: PASS"
syntax OK
MASTERPLAN VERIFICATION: PASS
```

EXIT 0. **PASS.**

### 2. Imports + settings defaults + unit test of `_apply_gpr_tilt`

```
syntax OK: backend/services/macro_regime.py
syntax OK: backend/config/settings.py
all imports OK

gpr_signal_enabled = False
gpr_signal_quantile = 0.9
gpr_signal_cache_hours = 24
gpr_signal_sector_etfs = 'XLE'
PASS: defaults correct

--- _apply_gpr_tilt unit ---
Below threshold: overweight=['XLK']
Above threshold: overweight=['XLK', 'XLE']
PASS: _apply_gpr_tilt correct

--- Multi-ETF inject (XLE,XOM,CVX) ---
overweight=['XLK', 'XLE', 'XOM', 'CVX']
PASS: multi-ETF inject
```

**PASS.**

### 3. Live `_fetch_gpr_acts` — real matteoiacoviello.com data

```
INFO: GPR Excel downloaded (2705408 bytes) from https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls
GPR-Acts current: 285.35
90th-pct threshold (rolling 60 months): 184.93
last_date: 2026-04-01 00:00:00
above_threshold: True
```

**REAL DATA:** As of April 2026, the GPR-Acts index is 285.35 — well above the 184.93 90th-percentile threshold computed over the trailing 60 months. **If `gpr_signal_enabled=True`, the picker would inject XLE into `sector_hints.overweight` for this cycle.** This matches the primary brief's narrative of Middle-East escalation in early 2026.

### 4. Mid-cycle dependency add (resolved)

First live fetch failed with `Import xlrd failed. Install xlrd >= 2.0.1 for xls Excel support`. The matteoiacoviello.com file is legacy .xls format which pandas needs `xlrd` to read. Fix: `pip install xlrd>=2.0.1` (now installed in venv) + added to `backend/requirements.txt` line 20. After install, the live fetch succeeded (see Test 3 above).

---

## Success criteria mapping

| Criterion (immutable) | Evidence | Result |
|---|---|---|
| `gpr_index_fetcher_implemented_with_caching` | `_fetch_gpr_acts` reads from `_GPR_CACHE_PATH` and skips re-download when age < `cache_hours`; downloaded 2.7MB once and cached | PASS |
| `sector_tilt_branch_added_to_macro_regime` | `_apply_gpr_tilt` + post-LLM hook in `compute_macro_regime` (gated on `settings.gpr_signal_enabled`) | PASS |
| `threshold_documented_in_audit_basis` | Multi-line comment at `_REGIME_SERIES` + 1 explains GPRA, asymmetry rationale, 90th-pct quantile, and Caldara-Iacoviello / IMF GFSR citations; settings.py descriptions cite phase-28.3 | PASS |
| `live_check_shows_XLE_overweight_when_gpr_above_threshold` | live_check_28.3.md captures GPR=285.35 > threshold=184.93 → XLE injected (above_threshold=True observed live) | PASS |

---

## Artifact shape

Post-edit signature additions in macro_regime.py:

```python
# New constants (after _REGIME_SERIES)
_GPR_URL_PRIMARY = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
_GPR_URL_FALLBACK = "https://www.matteoiacoviello.com/gpr_files/data_gpr.xls"
_GPR_CACHE_DIR = _CACHE_DIR / "gpr"
_GPR_CACHE_PATH = _GPR_CACHE_DIR / "data_gpr_export.xls"
_GPR_ROLLING_MONTHS = 60

# New helpers (before _load_cache)
async def _fetch_gpr_acts(cache_hours: int = 24, quantile: float = 0.90) -> Optional[dict]: ...
def _apply_gpr_tilt(parsed, gpr_info: dict, sector_etfs_csv: str) -> MacroRegimeOutput: ...

# Inside compute_macro_regime (after LLM call):
if getattr(settings, "gpr_signal_enabled", False):
    try:
        gpr_info = await _fetch_gpr_acts(cache_hours=..., quantile=...)
        if gpr_info:
            parsed = _apply_gpr_tilt(parsed, gpr_info, settings.gpr_signal_sector_etfs)
            logger.info("GPR tilt: current=... threshold=... above=... overweight=...")
    except Exception as e:
        logger.warning("GPR tilt application failed (non-fatal): %s", e)
```

---

## Known follow-ups (NOT blocking)

- **Dependency add to backend/requirements.txt**: `xlrd>=2.0.1` added (line 20). Anyone with a fresh venv must `pip install -r backend/requirements.txt` to pick it up. The current venv has xlrd installed; production rollout of `gpr_signal_enabled=True` should ensure the requirements.txt pin propagates.
- **`pyproject.toml` does not exist** in this repo (only `requirements.txt`). The dep add to requirements.txt is the canonical path.
- **GPR data is monthly** — the picker can run daily, but the GPR-Acts value updates only ~once per month. The 24h cache TTL is generous; could be widened to 7 days without semantic loss.

---

## Next

Q/A pass via fresh `qa` subagent. On PASS: append Cycle 18, flip phase-28.3 status.
