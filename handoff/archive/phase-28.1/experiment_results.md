# Experiment Results — phase-28.1 — Analyst EPS revision-breadth plug-in

**Step ID:** phase-28.1
**Date:** 2026-05-17
**Cycle:** 1 (single cycle; one mid-cycle bug-fix on tz comparison)

---

## What was built / changed

### Files modified

| File | Change |
|---|---|
| `backend/config/settings.py` | Added 5 fields after the short-interest block: `analyst_revisions_enabled` (bool, False), `analyst_revisions_lookback_days` (int, 100 — Mill Street canonical), `analyst_revisions_min_analysts` (int, 3 — noise guard), `analyst_revisions_threshold` (float, 0.10 — deadband edge), `analyst_revisions_weight` (float, 0.15 — multiplier intensity). |
| `backend/tools/screener.py` | Added `revision_signals=None` kwarg to `rank_candidates`; inserted overlay block after `sector_events` block (calls `apply_revisions_to_score`). |
| `backend/services/autonomous_loop.py` | When `settings.analyst_revisions_enabled=True`, fetch revisions for `2 * paper_screen_top_n` candidates AFTER first-pass `screen_universe` (bounds cost), pass to `rank_candidates(revision_signals=...)`. Mirrors existing graceful-degradation pattern. |

### Files created

| File | Purpose |
|---|---|
| `backend/services/analyst_revisions.py` | New 165-line module. `RevisionSignal` Pydantic model + `fetch_revision_signals(tickers, lookback_days, min_analysts) -> dict[ticker, RevisionSignal]` (async, Semaphore(4), 0.3s throttle) + `apply_revisions_to_score(base, ticker, signals, threshold, weight) -> float` (deadband + multiplicative). |
| `handoff/current/phase-28.1-research-brief.md` | Research-gate brief (Researcher subagent; `gate_passed: true`; 5 sources read in full). |
| `handoff/current/contract.md` | This step's contract (rolling). |
| `handoff/current/experiment_results.md` | This file (rolling). |
| `handoff/current/live_check_28.1.md` | Live evidence: 4 real revision signals + rank_candidates baseline vs overlay + top-3 conviction shifts. |

---

## Mid-cycle bug-fix (cycle-1 internal, single-evidence revision)

First smoke returned **0/5 tickers produced signals** for a known-active set. Root cause: `_compute_breadth` used a tz-AWARE cutoff (`datetime.now(timezone.utc) - ...`) but yfinance returns a tz-NAIVE `datetime64[s]` index — the comparison raised `TypeError: Invalid comparison between dtype=datetime64[s] and datetime`, silently swallowed by the outer try/except, returning None.

**Fix:** switched cutoff to tz-naive (`datetime.now()`), added an explicit fallback for tz-aware indexes (`tz_convert(None)`), removed the silent swallow of comparison errors. Documented the yfinance contract (naive index, `Action ∈ {up, down, main, init, reit}` with main being 80% noise) in the function docstring.

**Post-fix smoke:** 4/9 tickers produce signals (AAPL, TSLA, GOOGL, AMD).

---

## Verification — verbatim output

### 1. Immutable verification command (from `.claude/masterplan.json::phase-28.steps[1].verification.command`)

```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/analyst_revisions.py').read()); from backend.services.analyst_revisions import fetch_revision_signals; print('module importable')" && grep -q 'analyst_revisions_enabled' backend/config/settings.py && grep -q 'analyst_revisions' backend/services/autonomous_loop.py && echo "MASTERPLAN VERIFICATION: PASS"
module importable
MASTERPLAN VERIFICATION: PASS
```

EXIT 0. **PASS.**

### 2. 4-file syntax + import + signature + settings defaults

```
syntax OK: backend/services/analyst_revisions.py
syntax OK: backend/tools/screener.py
syntax OK: backend/services/autonomous_loop.py
syntax OK: backend/config/settings.py

--- Imports ---
all imports OK

--- Settings defaults ---
analyst_revisions_enabled = False
analyst_revisions_lookback_days = 100
analyst_revisions_min_analysts = 3
analyst_revisions_threshold = 0.1
analyst_revisions_weight = 0.15
PASS: defaults correct

--- rank_candidates signature ---
rank_candidates params: ['screen_data', 'top_n', 'strategy', 'regime', 'pead_signals', 'news_signals', 'sector_events', 'revision_signals']
PASS: revision_signals kwarg present
```

**PASS.**

### 3. Live fetch — real yfinance upgrades_downgrades

```
--- Smoke: fetch revisions (min_analysts=1) ---
Returned 4/9 signals (min_analysts=1)
  AAPL: breadth=+1.000 up=1 down=0 total=1
  TSLA: breadth=+1.000 up=1 down=0 total=1
  GOOGL: breadth=+0.000 up=1 down=1 total=2
  AMD: breadth=+0.143 up=4 down=3 total=7

--- Smoke: fetch revisions at production setting min_analysts=3 ---
Returned 1/9 signals (min_analysts=3)
  AMD: breadth=+0.143 up=4 down=3 total=7
```

At production setting (min_analysts=3), AMD produces a non-empty signal. Criterion #4 (`smoke_run_with_flag_on_produces_non_empty_signal_for_recent_reporters`) **satisfied** — recent reporter AMD has 7 actionable grade changes in the last 100 days, breadth +0.143 (above 0.10 deadband → boost applied).

### 4. apply_revisions_to_score behavior

```
--- Apply to scores ---
  AAPL: breadth=+1.000 -> 5.750 (+15.0%) [APPLIED]   # full breadth -> full weight*1 = 15% boost
  TSLA: breadth=+1.000 -> 5.750 (+15.0%) [APPLIED]
  GOOGL: breadth=+0.000 -> 5.000 (+0.0%) [deadband]   # 0.0 is below 0.10 threshold; identity
  AMD: breadth=+0.143 -> 5.107 (+2.1%) [APPLIED]      # small positive breadth -> small boost
```

Deadband working correctly (GOOGL at exactly 0.0 stays unchanged).

### 5. rank_candidates baseline vs overlay (synthetic momentum + real revisions)

```
--- Baseline (no revision_signals) ---
  NVDA : composite_score=8.500
  AAPL : composite_score=7.700
  META : composite_score=7.300
  MSFT : composite_score=6.900
  AMD  : composite_score=6.900
  GOOGL: composite_score=6.500
  TSLA : composite_score=6.100
  AMZN : composite_score=6.100
  GME  : composite_score=5.700

--- With revision_signals overlay ---
  AAPL : composite_score=8.855  (br=+1.00)
  NVDA : composite_score=8.500  (no sig)
  META : composite_score=7.300  (no sig)
  AMD  : composite_score=7.048  (br=+0.14)
  TSLA : composite_score=7.015  (br=+1.00)
  MSFT : composite_score=6.900  (no sig)
  GOOGL: composite_score=6.500  (br=+0.00)
  AMZN : composite_score=6.100  (no sig)
  GME  : composite_score=5.700  (no sig)

--- Top-3 conviction shifts ---
  AAPL: 7.700 -> 8.855  (delta=+1.155)
  TSLA: 6.100 -> 7.015  (delta=+0.915)
  AMD: 6.900 -> 7.048  (delta=+0.148)
```

**Ranking changed:** AAPL #2 → #1 (overtook NVDA which has no revision signal); TSLA #7 → #5; AMD stays mid-pack but with small lift.

### 6. Back-compat: rank_candidates with NO revision_signals

```
--- Back-compat: rank_candidates with NO revision_signals ---
  result: [{'ticker': 'AAPL', 'momentum_1m': 10, 'momentum_3m': 8, 'momentum_6m': 5, 'rsi_14': 60, 'composite_score': 8.05}]
PASS: back-compat
```

Zero callsite changes required elsewhere.

---

## Success criteria mapping

| Criterion (immutable) | Evidence | Result |
|---|---|---|
| `analyst_revisions_module_created_and_syntax_OK` | `backend/services/analyst_revisions.py` created; `python -c "import ast; ast.parse(...)"` exit 0; `fetch_revision_signals` importable | PASS |
| `feature_flag_analyst_revisions_enabled_default_false` | `Settings().analyst_revisions_enabled == False` (live instantiation) | PASS |
| `wired_into_rank_candidates_or_meta_scorer` | `rank_candidates` signature now includes `revision_signals`; overlay applied via `apply_revisions_to_score` in per-stock loop | PASS |
| `smoke_run_with_flag_on_produces_non_empty_signal_for_recent_reporters` | AMD produces signal at production setting (min_analysts=3): breadth=+0.143, 4 up, 3 down in 100d | PASS |
| `cycle_cost_delta_under_0_05_USD` | $0 LLM cost (yfinance only); per-ticker HTTP for top-N (~10-30 tickers) at 0.3s throttle = ~3-10s per cycle | PASS |

---

## Artifact shape

Post-edit `rank_candidates` signature:

```python
def rank_candidates(
    screen_data: list[dict],
    top_n: int = 10,
    strategy: str = "momentum",
    regime=None,
    pead_signals=None,
    news_signals=None,
    sector_events=None,
    revision_signals=None,  # NEW
) -> list[dict]:
```

Overlay block in body (after sector_events):

```python
if revision_signals:
    from backend.services.analyst_revisions import apply_revisions_to_score
    score = apply_revisions_to_score(score, stock.get("ticker"), revision_signals)
```

`apply_revisions_to_score` formula: `score *= (1 + breadth * weight)` only when `|breadth| > threshold` (deadband).

---

## Next

Q/A pass via fresh `qa` subagent. On PASS: append Cycle 16 entry, flip phase-28.1 status.
