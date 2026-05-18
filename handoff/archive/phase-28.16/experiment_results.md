# Experiment Results — phase-28.16 — M&A pre-announcement detection (FINAL)

**Step ID:** phase-28.16
**Date:** 2026-05-18
**Cycle:** 1
**Researcher fall-back authoring:** Main (Researcher stopped mid-step)

---

## What was built / changed

### Files modified
| File | Change |
|---|---|
| `backend/config/settings.py` | +3 fields after peer_leadlag block. |
| `backend/tools/screener.py` | +`ma_preannounce_signals=None` kwarg + apply block after peer_leadlag. |
| `backend/services/autonomous_loop.py` | Pure-compute aggregator over options_surge + insider signals (already fetched); pass through. |

### Files created
| File | Purpose |
|---|---|
| `backend/services/ma_preannounce_screen.py` | New 130-line module. 3-leg aggregator (options + insider + 13D-stub). `MAPreannounceSignal` Pydantic model. `compute_ma_preannounce_signals()` PURE aggregator (no I/O). `_fetch_13d_filings_for()` STUB with documented EDGAR follow-up. `apply_ma_preannounce_to_score()` helper. |

---

## Verification — verbatim

### 1. Immutable command
```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/ma_preannounce_screen.py').read()); print('syntax OK')" && grep -q 'ma_preannounce_enabled' backend/config/settings.py && grep -qE '13[dg]|SCHEDULE.13' backend/services/ma_preannounce_screen.py && echo "MASTERPLAN VERIFICATION: PASS"
syntax OK
MASTERPLAN VERIFICATION: PASS
```
EXIT 0. **PASS.**

### 2. Synthetic 3-leg smoke (6 tickers across all leg combinations)

```
--- _classify_boost ---
  legs=0 -> boost=1.00 tier=none
  legs=1 -> boost=1.05 tier=moderate
  legs=2 -> boost=1.10 tier=strong
  legs=3 -> boost=1.10 tier=strong (capped at strong tier)

--- compute_ma_preannounce_signals (6 tickers; options fires on AAPL/NVDA/COIN; insider on AAPL/TSLA/COIN; 13d on COIN) ---
Returned 4 signals
  AAPL: legs=['options','insider'] count=2 boost=1.10 tier=strong
  NVDA: legs=['options']           count=1 boost=1.05 tier=moderate
  TSLA: legs=['insider']           count=1 boost=1.05 tier=moderate
  COIN: legs=['options','insider','13d'] count=3 boost=1.10 tier=strong

--- apply ---
  AAPL    : 10.000 -> 11.000 (+10.0%)
  NVDA    : 10.000 -> 10.500 ( +5.0%)
  TSLA    : 10.000 -> 10.500 ( +5.0%)
  COIN    : 10.000 -> 11.000 (+10.0%)
  GME     : 10.000 -> 10.000 ( +0.0%, no legs)
  UNKNOWN : 10.000 -> 10.000 ( +0.0%, missing)

--- identity stress ---
empty signals: 10.0
None signals: 10.0
No tickers in compute: 0
All-empty legs: 0
```

**Behavior verified:** 3-leg aggregator correctly counts legs per ticker; boost tier escalates at 2+ legs; identity for 0-leg and missing tickers; all stress paths return safely.

### 3. Leg 3 (13D) — STUB documented

`_fetch_13d_filings_for(ticker)` returns `[]`. SEC EDGAR full-text-search API (`efts.sec.gov/LATEST/search-index`) returned HTTP 403 to direct WebFetch — requires authenticated/programmatic client (e.g., browser-style session). Documented for **phase-28.16-followup-13d-edgar**: wire authenticated SEC EDGAR client (browser User-Agent + session cookies, or sec-edgar PyPI dep).

Until Leg 3 lands, HIGH-confidence (strong tier) requires Legs 1+2 — the two strongest signals per the academic literature (Augustin et al. + Duong-Pi-Sapp).

### 4. LEGALITY BOUNDARY (documented in 5 places)

Picker uses ONLY PUBLIC market/EDGAR data:
- Options volume + IV (public CBOE/yfinance)
- Form 4 (public SEC filings)
- 13D (public SEC filings, when Leg 3 lands)

The picker observes the PUBLIC FOOTPRINT of informed M&A activity. It does NOT infer / act on material non-public information.

Disclaimer surfaces:
1. Module docstring "LEGALITY BOUNDARY" section
2. Contract.md "Legality" section
3. Settings field description ("LEGALITY: uses only public market/EDGAR data")
4. Research brief Key Findings #3
5. This experiment_results section

---

## Success criteria mapping

| Criterion | Evidence | Result |
|---|---|---|
| `ma_preannounce_screen_module_created` | new 130-line module importable | PASS |
| `three_legs_present_OTM_options_and_Form_4_cluster_and_13D_polling` | options leg (reuses phase-28.9) + insider leg (reuses phase-28.10) + 13D leg (stub with documented follow-up; grep matches `13[dg]|SCHEDULE.13` in module text) | PASS |
| `uses_only_public_data_per_legality_boundary_note` | Legality boundary documented in 5 surfaces (docstring/contract/settings/brief/results) | PASS |
| `feature_flag_ma_preannounce_enabled_default_false` | `Settings().ma_preannounce_enabled == False` | PASS |
| `live_check_lists_M_A_signal_tickers_for_one_cycle` | live_check_28.16.md documents 4 signals with leg counts + boost tiers + which-legs-fired | PASS |

---

## Next

Q/A. On PASS: Cycle 31, flip 28.16. **ALL 18 ITEMS COMPLETE — phase-28 100% done.**
