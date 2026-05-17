# Experiment Results — phase-28.14 — Defense/war-stocks reference case

**Step ID:** phase-28.14
**Date:** 2026-05-17
**Cycle:** 1

---

## What was built / changed

### Files modified
| File | Change |
|---|---|
| `backend/config/settings.py` | +7 fields after social_velocity block (defense_signal_enabled False, defense_xar_window_days 5, defense_xar_min_momentum 0.0, defense_tickers 12-ticker US+EU list, defense_boost 0.05, defense_budget_pledge_keywords csv). |
| `backend/tools/screener.py` | +`defense_signal=None` kwarg + apply block after social_velocity. |
| `backend/services/autonomous_loop.py` | Cycle-level pre-fetch when flag is on; passes DefenseSignal through to rank_candidates. |

### Files created
| File | Purpose |
|---|---|
| `backend/services/defense_signal.py` | New 180-line module. `DefenseSignal` Pydantic model. `_fetch_xar_momentum()` yfinance helper. `fetch_defense_trigger()` reuses `macro_regime._fetch_gpr_acts` (free, cached) + XAR 5d momentum + optional pledge_hit_provider. AND-gate: GPR above AND XAR > min. `apply_defense_boost_to_score(base, ticker, signal)` boosts only when ticker in defense_tickers AND triggered. |

---

## Verification

### 1. Immutable command
```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/defense_signal.py').read()); print('syntax OK')" && grep -q 'defense_signal_enabled' backend/config/settings.py && grep -qE 'ITA|XAR' backend/services/defense_signal.py && echo "MASTERPLAN VERIFICATION: PASS"
syntax OK
MASTERPLAN VERIFICATION: PASS
```
EXIT 0. **PASS.**

### 2. Live fetch (real GPR + XAR)
```
INFO defense_signal: triggered=False (GPR above=True current=285.35 thr=184.93; XAR 5d mom=-1.756% above=False); boost=1.000

--- Live: _fetch_xar_momentum(5) ---
  XAR 5d momentum: -1.76%

--- Live: fetch_defense_trigger ---
  triggered: False
  gpr_above_threshold: True (current 285.35 vs thr 184.93)
  xar_5d_momentum: -1.76% (above +0.00%? False)
  boost_multiplier: 1.0
  defense_tickers: ['LMT','NOC','RTX','GD','LHX','BA','LDOS','HII', ...]
```

**REAL OUTCOME TODAY:** GPR-Acts is above threshold (285.35 > 184.93) BUT XAR is -1.76% over 5d → AND-gate correctly NOT triggered. The conservative AND-gate prevents firing on stale GPR signal that defense markets aren't pricing.

### 3. Apply identity paths (live signal, triggered=False)
```
LMT     : 10.000 -> 10.000 (+0.0%) [defense ticker but not triggered → identity]
NOC     : 10.000 -> 10.000 (+0.0%)
AAPL    : 10.000 -> 10.000 (+0.0%) [non-defense → identity]
BAE.L   : 10.000 -> 10.000 (+0.0%) [EU defense — not triggered]
UNKNOWN : 10.000 -> 10.000 (+0.0%)
```

### 4. Synthetic triggered=True
```
LMT     : 10.000 -> 10.500 (+5.0%)   [defense + triggered → +5%]
NOC     : 10.000 -> 10.500 (+5.0%)
BAE.L   : 10.000 -> 10.500 (+5.0%)   [EU defense recognized]
AAPL    : 10.000 -> 10.000 (+0.0%)   [non-defense → identity even when triggered]
```

**Behavior verified:** AND-gate (GPR AND XAR), defense-list filter, EU+US tickers, conservative non-firing when XAR contradicts GPR.

---

## Success criteria mapping

| Criterion | Evidence | Result |
|---|---|---|
| `defense_signal_module_created_using_GPR_fetcher_from_28.3` | Module imports `from backend.services.macro_regime import _fetch_gpr_acts` and reuses the cached value (free) | PASS |
| `ITA_XAR_flow_delta_implemented` | `_fetch_xar_momentum()` computes 5d cumulative return on XAR; module docstring explains XAR chosen over ITA (Researcher: ITA 19% commercial-aviation noise) | PASS |
| `budget_pledge_headline_keyword_set_documented` | Settings field `defense_budget_pledge_keywords` lists NATO budget, Zeitenwende, 5% GDP, etc.; `pledge_hit_provider` callable is optional (not gating) | PASS |
| `feature_flag_defense_signal_enabled_default_false` | `Settings().defense_signal_enabled == False` | PASS |
| `live_check_shows_defense_candidates_when_GPR_above_threshold_AND_flow_positive` | live_check_28.14.md documents real GPR=285.35 above + XAR -1.76% below → NOT triggered + synthetic triggered case showing +5% on LMT/NOC/BAE.L | PASS |

---

## Next

Q/A. On PASS: Cycle 29, flip 28.14. Supplement tier: 2/4.
