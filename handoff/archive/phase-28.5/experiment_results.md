# Experiment Results — phase-28.5 — Short-interest exclusion filter

**Step ID:** phase-28.5
**Date:** 2026-05-17
**Cycle:** 1

---

## What was built / changed

### Files modified

| File | Change |
|---|---|
| `backend/config/settings.py` | Added 3 fields after `meta_scorer_max_batch`: `short_interest_filter_enabled` (bool, default False), `short_interest_threshold` (float, default 0.10), `short_interest_cache_days` (int, default 14). |
| `backend/tools/screener.py` | Added 2 new kwargs to `screen_universe`: `short_interest_lookup` (Optional[dict[str, float]], default None) and `short_interest_threshold` (float, default 0.10). Inserted exclusion block immediately after the basic price/volume filter (post line 121). Updated docstring with phase-28.5 explanation. |
| `backend/services/autonomous_loop.py` | Added flag-conditional pre-fetch of `short_interest_lookup` immediately after the `sector_events` block (line ~246); modified `screen_universe()` call to pass `short_interest_lookup` and `short_interest_threshold`. |

### Files created

| File | Purpose |
|---|---|
| `backend/services/short_interest.py` | New module. `fetch_short_interest_lookup()` returns `dict[ticker, shortPercentOfFloat]`. Primary path: FINRA bimonthly CSV download (probes recent settlement dates, caches locally for 14 days). Fallback path: per-ticker `yfinance.Ticker.info["shortPercentOfFloat"]` (throttled 0.5s/call). Returns empty dict on any error (default-OFF safety). |
| `handoff/current/phase-28.5-research-brief.md` | Research-gate brief (Researcher subagent; `gate_passed: true`; 6 sources read in full). |
| `handoff/current/contract.md` | This step's contract (rolling). |
| `handoff/current/experiment_results.md` | This file (rolling). |
| `handoff/current/live_check_28.5.md` | Live exclusion evidence with real yfinance shortPercentOfFloat values. |

---

## Verification — verbatim output

### 1. Immutable verification command (from `.claude/masterplan.json::phase-28.steps[5].verification.command`)

```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/tools/screener.py').read()); print('syntax OK')" && grep -qE 'short.{0,30}(ratio|interest|exclusion)' backend/tools/screener.py && echo "MASTERPLAN VERIFICATION: PASS"
syntax OK
MASTERPLAN VERIFICATION: PASS
```

EXIT 0. **PASS.**

### 2. Multi-file syntax + import + signature + settings field check

```
$ source .venv/bin/activate && python -c "..."
syntax OK: backend/tools/screener.py
syntax OK: backend/services/short_interest.py
syntax OK: backend/services/autonomous_loop.py
syntax OK: backend/config/settings.py

--- Import check ---
all imports OK

--- Signature check ---
screen_universe params: ['tickers', 'min_avg_volume', 'min_price', 'period', 'sector_lookup', 'short_interest_lookup', 'short_interest_threshold']
PASS: new kwargs present

--- Settings fields ---
short_interest_filter_enabled = False (must be False)
short_interest_threshold = 0.1 (must be 0.10)
short_interest_cache_days = 14 (must be 14)
PASS: defaults correct
```

**PASS.**

### 3. Smoke tests — back-compat + exclusion + edge cases

```
--- Smoke test 1: back-compat (no lookup) ---
Returned 3 results: ['AAPL', 'MSFT', 'NVDA']
PASS: back-compat (no lookup)

--- Smoke test 2: exclusion with synthetic lookup ---
Returned: ['AAPL', 'MSFT']
PASS: exclusion works
# TSLA at 0.15 > 0.10 -> excluded; AAPL at 0.05 -> kept; MSFT not in lookup -> kept

--- Smoke test 3: empty dict lookup -> no exclusion ---
Returned: ['AAPL', 'MSFT', 'TSLA']
PASS: empty lookup is no-op
```

**PASS.**

### 4. Live data-path test — `fetch_short_interest_lookup` end-to-end

```
INFO httpx: HTTP Request: GET https://cdn.finra.org/equity/regsho/monthly/shrt20260515.csv "HTTP/1.1 403 Forbidden"
INFO httpx: HTTP Request: GET https://cdn.finra.org/equity/regsho/monthly/shrt20260430.csv "HTTP/1.1 403 Forbidden"
INFO httpx: HTTP Request: GET https://cdn.finra.org/equity/regsho/monthly/shrt20260415.csv "HTTP/1.1 403 Forbidden"
WARNING backend.services.short_interest: FINRA short-interest CSV: no recent settlement date returned 200 (last 35 days tried)
INFO backend.services.short_interest: yfinance fallback: fetched shortPercentOfFloat for 5/5 tickers
INFO backend.services.short_interest: Short-interest lookup: 5 tickers total

=== Lookup size: 5 ===
  TSLA: shortPercentOfFloat=0.023
  GME: shortPercentOfFloat=0.145
  AMC: shortPercentOfFloat=0.175
  AAPL: shortPercentOfFloat=0.0092
  MSFT: shortPercentOfFloat=0.0107
```

**Result:**
- FINRA bulk path **NOT working** in this environment — the `cdn.finra.org/equity/regsho/monthly/shrt<date>.csv` URL pattern returns HTTP 403. The actual FINRA download URL convention requires either authenticated portal access or a different CDN path. **Follow-up:** correct the URL probe to use FINRA's documented `https://www.finra.org/sites/default/files/...` or `https://api.finra.org/data/group/otcMarket/name/equityShortInterest` API endpoint.
- **yfinance fallback works correctly.** Fetched valid `shortPercentOfFloat` for all 5 test tickers (incl. high-short meme stocks GME 14.5% and AMC 17.5% — both would be excluded at threshold 0.10).
- The default-OFF safety pattern is honored: feature flag remains False; cycle continues normally even when both data paths fail.

---

## Success criteria mapping

| Criterion (immutable from masterplan) | Evidence | Result |
|---|---|---|
| `short_interest_field_collected_in_screen_universe` | `short_interest_lookup` kwarg added to `screen_universe` signature (`inspect.signature` confirms); per-ticker `shortPercentOfFloat` consumed in exclusion check inside the per-ticker loop | PASS |
| `exclusion_filter_added_with_documented_threshold` | New exclusion block at screener.py after line 127; threshold = 0.10 documented in code comment, docstring, settings.py field description, and contract.md | PASS |
| `feature_flag_short_exclusion_enabled_default_false` | `settings.short_interest_filter_enabled = False` confirmed by live `Settings()` instantiation | PASS |
| `live_check_lists_excluded_tickers_for_one_cycle` | `handoff/current/live_check_28.5.md` shows real yfinance shortPercentOfFloat data for 5 tickers + simulated cycle log with GME + AMC excluded | PASS |

---

## Artifact shape

Post-edit `screen_universe` signature:

```python
def screen_universe(
    tickers: Optional[list[str]] = None,
    min_avg_volume: int = 100_000,
    min_price: float = 5.0,
    period: str = "6mo",
    sector_lookup: Optional[dict] = None,
    short_interest_lookup: Optional[dict[str, float]] = None,
    short_interest_threshold: float = 0.10,
) -> list[dict]:
```

Exclusion block in body (immediately after basic price/volume filter):

```python
# phase-28.5: short-interest exclusion (high-short underperforms ~1.16%/mo per Boehmer-Jones-Zhang 2008).
# Lookup is opt-in; built by caller via FINRA bimonthly CSV (preferred) or yfinance per-ticker fallback.
# No exclusion fires when lookup is None or empty dict, preserving back-compat.
if short_interest_lookup:
    short_pct = short_interest_lookup.get(ticker)
    if short_pct is not None and short_pct > short_interest_threshold:
        logger.debug(
            "Excluding %s: shortPercentOfFloat=%.3f > %.3f (phase-28.5)",
            ticker, short_pct, short_interest_threshold,
        )
        continue
```

Settings.py addition (after `meta_scorer_max_batch` at line ~198):

```python
short_interest_filter_enabled: bool = Field(False, description="phase-28.5: Exclude tickers with shortPercentOfFloat > short_interest_threshold from screener. Boehmer-Jones-Zhang 2008: high-short stocks underperform 1.16%/mo. Default OFF.")
short_interest_threshold: float = Field(0.10, description="phase-28.5: shortPercentOfFloat cutoff above which a ticker is excluded (default 10% = approximate top-decile for S&P 500 large-caps).")
short_interest_cache_days: int = Field(14, description="phase-28.5: Days to cache the FINRA bimonthly CSV before re-downloading (FINRA publishes bimonthly, so 14 days matches their cadence).")
```

Autonomous_loop.py addition (right after sector_events block):

```python
# phase-28.5: short-interest exclusion lookup (FINRA bimonthly CSV preferred, yfinance fallback)
short_interest_lookup: dict[str, float] = {}
if getattr(settings, "short_interest_filter_enabled", False):
    try:
        from backend.services.short_interest import fetch_short_interest_lookup
        short_interest_lookup = await fetch_short_interest_lookup()
        logger.info(
            "Short-interest lookup loaded: %d tickers (threshold=%.3f)",
            len(short_interest_lookup), settings.short_interest_threshold,
        )
        summary["short_interest_tickers_loaded"] = len(short_interest_lookup)
    except Exception as e:
        logger.warning("Short-interest lookup failed (non-fatal): %s", e)

screen_data = screen_universe(
    period="6mo",
    short_interest_lookup=short_interest_lookup or None,
    short_interest_threshold=getattr(settings, "short_interest_threshold", 0.10),
)
```

---

## Known follow-up

- **FINRA URL pattern needs correction.** The `cdn.finra.org/equity/regsho/monthly/shrt<DATE>.csv` URL returns 403. The actual FINRA API/download path requires investigation (likely the `api.finra.org/data/group/...` REST endpoint described in the research brief). This is a separate small ticket; the yfinance fallback handles current operation. **Tracked as phase-28.5-followup-finra-url** (not in masterplan; can be done inline or as a tiny patch).
- **yfinance per-ticker cost.** With S&P 500 the fallback would be 500 × 0.5s = 250s plus 429 risk. Acceptable if FINRA fixed; recommend NOT enabling the feature flag in production until the FINRA path works.

---

## Next

Q/A pass via fresh `qa` subagent. On PASS: append harness_log Cycle 15 entry → flip phase-28.5 status to `done`.
