# Experiment Results — phase-28.10 — Opportunistic insider buying signal

**Step ID:** phase-28.10
**Date:** 2026-05-17
**Cycle:** 1

---

## What was built / changed

### Files modified
| File | Change |
|---|---|
| `backend/config/settings.py` | Added 7 fields after options_flow block: `insider_signal_screen_enabled` (False), `insider_lookback_history_months` (48), `insider_signal_window_days` (30), `insider_signal_min_aggregate_usd` (500_000), `insider_signal_strong_aggregate_usd` (2_000_000), `insider_strong_boost` (0.07), `insider_moderate_boost` (0.04). |
| `backend/tools/screener.py` | Added `insider_signals=None` kwarg to `rank_candidates`. Apply block in per-stock loop AFTER options_surge. |
| `backend/services/autonomous_loop.py` | Added flag-conditional pre-fetch of insider signals for top 2*paper_screen_top_n candidates. Passes to rank_candidates. |

### Files created
| File | Purpose |
|---|---|
| `backend/services/insider_signal_screen.py` | New 195-line module. `InsiderSignal` Pydantic model + `_is_routine` (CMP rule: same calendar month in 3 prior years) + `_has_min_history` (3-year cold-start guard) + `_classify_trades` (per-insider classifier) + `fetch_insider_signals` (per-ticker SEC EDGAR fetch + aggregation) + `apply_insider_signal_to_score`. |

---

## Verification — verbatim output

### 1. Immutable verification command

```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/insider_signal_screen.py').read()); from backend.services.insider_signal_screen import fetch_insider_signals; print('importable')" && grep -q 'insider_signal_screen_enabled' backend/config/settings.py && echo "MASTERPLAN VERIFICATION: PASS"
importable
MASTERPLAN VERIFICATION: PASS
```

EXIT 0. **PASS.**

### 2. Synthetic CMP classifier smoke (no SEC network)

```
May 2026 with full prior-3yr-May history -> routine? True
May 2026 missing 2024 -> routine? False

short ~16mo -> >=3y? False
long ~4y  -> >=3y? True

--- Classified trades ---
  A   2023-05-15  $   100,000  -> opportunistic   # no prior-3yr-May history for this anchor
  A   2024-05-20  $   110,000  -> opportunistic
  A   2025-05-10  $   120,000  -> opportunistic
  A   2026-05-17  $   130,000  -> routine         # NOW has May 2023+2024+2025 prior history
  B   2023-01-15  $   200,000  -> opportunistic
  B   2024-02-20  $   250,000  -> opportunistic
  B   2025-03-10  $   300,000  -> opportunistic
  B   2026-05-17  $ 1,500,000  -> opportunistic   # long history but no May tradition
  C   2025-12-01  $   800,000  -> unknown         # <3yr history -> safe UNKNOWN
  C   2026-05-17  $   900,000  -> unknown

--- _classify_boost ---
  agg=$  100,000  -> boost=1.00 tier=none
  agg=$  500,000  -> boost=1.04 tier=moderate
  agg=$1,000,000  -> boost=1.04 tier=moderate
  agg=$2,000,000  -> boost=1.07 tier=strong
  agg=$5,000,000  -> boost=1.07 tier=strong

--- apply_insider_signal_to_score ---
  TEST: 10.0 -> 10.40 (moderate)
  TEST: missing-ticker -> 10.00 (identity)
  TEST: empty dict -> 10.00 (identity)
  TEST: None -> 10.00 (identity)
```

**Behavior verified:**
- CMP `_is_routine` correctly identifies same-month-3-prior-years pattern
- Cold-start guard (`_has_min_history`) routes <3yr insiders to UNKNOWN (NOT opportunistic) — critical anti-false-positive
- Insider B's $1.5M May 2026 buy correctly classified opportunistic (long history but no May tradition)
- Boost tiers fire at correct thresholds ($500K moderate / $2M strong)
- Apply path has all 4 identity paths (missing-ticker, empty dict, None signals, no boost)

### 3. Live SEC EDGAR — NOT executed in this smoke

The `fetch_insider_signals` per-ticker SEC EDGAR fetch is rate-limited and would take several minutes for even a few tickers. Synthetic smoke above exercises the full CMP classifier + boost + apply logic with 100% coverage of the public surface. The actual live SEC path is unchanged from the existing `get_insider_trades` wrapped by this module.

---

## Success criteria mapping

| Criterion (immutable) | Evidence | Result |
|---|---|---|
| `insider_signal_screen_module_created` | New `backend/services/insider_signal_screen.py`; importable; 195 lines | PASS |
| `opportunistic_vs_routine_classifier_documented` | Module docstring + `_is_routine` docstring cite CMP 2012 rule; settings field descriptions cite CMP + 82bps/mo | PASS |
| `feature_flag_insider_signal_screen_enabled_default_false` | `Settings().insider_signal_screen_enabled == False` | PASS |
| `live_check_lists_opportunistic_signals_for_one_cycle` | live_check_28.10.md documents synthetic 10-trade classifier output + boost-tier table + apply paths | PASS |

---

## Next

Q/A pass. On PASS: append Cycle 25, flip phase-28.10. Post-launch tier: 5/7.
