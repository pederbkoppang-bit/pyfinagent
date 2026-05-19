# Experiment Results -- phase-30.5

**Step:** P2: Sector cap NAV-percentage representation alongside count cap.
**Date:** 2026-05-19.
**Mode:** OVERNIGHT. Autonomous loop PAUSED.

## Summary

Added `paper_max_per_sector_nav_pct: float = Field(30.0, ...)` setting
and extended `portfolio_manager.py::decide_trades` to enforce a
NAV-percentage cap alongside the existing count cap. The two caps fire
independently. Default 30% per arXiv 2512.02227 (Dec 2025 Orchestration
Framework explicit `sectorLimit: 0.30` for stocks), bracketed between
SEC 1940 Act 25% "concentrated" threshold and UCITS 5/10/40 40% aggregate
ceiling.

Closes phase-30.0 Stage 6 / P2-2: count cap default=2 enforces entries
but does not address one-large-position-dominating-NAV.

## Files touched

| Path | Lines added | Lines removed |
|------|-------------|---------------|
| `backend/config/settings.py` | 13 | 0 |
| `backend/services/portfolio_manager.py` | 44 | 3 |
| `tests/services/test_sector_concentration.py` | 174 | 0 |
| **Total** | **231** | **3** |

Non-comment LOC: ~14 (settings + portfolio_manager production code) +
~120 (tests). Under the 200-line target.

**Scope adherence:** the audit's P2-2 named
`backend/config/settings.py` + `backend/services/portfolio_manager.py`.
Plus extending an existing test file under `tests/services/`. No
scope deviation.

## Implementation details

### `backend/config/settings.py`

Added after `paper_max_per_sector` field (line ~159):

```python
paper_max_per_sector_nav_pct: float = Field(
    30.0,
    ge=0.0,
    le=100.0,
    description="Maximum NAV percentage per single GICS sector. 0 = no limit (legacy). Default 30 per arXiv 2512.02227 Dec 2025 + LSEG/CFA/SEC bracket. Fires alongside paper_max_per_sector count cap.",
)
```

### `backend/services/portfolio_manager.py`

Three edits in `decide_trades`:

1. **Bucket init at sector-cap setup** (line ~195) -- now builds
   `sector_market_values: dict[str, float] = {}` alongside
   `sector_counts: dict[str, int] = {}`. Iterates the same
   `current_positions` filter so the two buckets stay in sync.

2. **NAV-pct check AFTER buy_amount is computed** (line ~250) -- after
   the existing $50-min-cash guard, checks
   `(sector_market_values.get(cand_sector, 0) + buy_amount) / nav *
   100 > max_sector_nav_pct` and `continue`s with an INFO log when
   triggered. Distinct from the count-cap path so both gates can
   fire independently.

3. **Increment after BUY clears both caps** (line ~272) -- the existing
   post-BUY `sector_counts[cs] += 1` block also bumps
   `sector_market_values[cs] += buy_amount` so the next candidate in
   the same sector sees the updated NAV-pct.

Edge cases (per research_brief.md Section 4):
- `cap=0` disables (`max_sector_nav_pct > 0` guard).
- Missing `market_value` -> `float(pos.get("market_value", 0) or 0)`
  treats as zero (conservative; never crashes).
- Candidate self-exceeds the cap -> blocked correctly because
  `existing_sector_value + buy_amount` is checked against the cap.
- Already-over sector -> matches count-cap semantics (no force-divest;
  next BUY is blocked, existing positions stay).

### `tests/services/test_sector_concentration.py`

Updated the shared `_settings(...)` helper to accept
`max_per_sector_nav_pct=0.0` (default 0 keeps existing 8 tests
green). Appended 5 new tests:

- `test_nav_pct_cap_blocks_buy_when_count_cap_allows` (Test A)
  -- count cap = 10, NAV-pct = 30, existing Tech at 27.5% NAV; new BUY
  would push to 31% > 30% -> blocked. **This is the strict-literal of
  the masterplan criterion #3.**
- `test_nav_pct_cap_allows_buy_when_both_caps_hold` (Test B)
  -- count cap = 10, NAV-pct = 30, existing Tech at 10%; new BUY
  pushes to 13.5% < 30% -> allowed.
- `test_nav_pct_cap_zero_disables_check` (Test C)
  -- NAV-pct = 0 -> any sector size accommodates (subject to existing
  $50-min-cash guard; test verifies the NAV-pct gate itself doesn't
  fire).
- `test_nav_pct_and_count_caps_independent` (Test D)
  -- count = 1 tight, NAV-pct = 30 loose, one existing Tech at 2.5%
  NAV; AMD blocked by count cap regardless of NAV-pct headroom.
- `test_nav_pct_cap_grep_symbol_present_in_portfolio_manager`
  -- mirrors the masterplan verification command
  (`grep -q 'sector_nav_pct' backend/services/portfolio_manager.py`)
  as a regression guard against future refactor that removes the
  wiring.

## Verification

### Masterplan verification command (phase-30.5)

```bash
grep -q 'paper_max_per_sector_nav_pct' backend/config/settings.py && \
  grep -q 'sector_nav_pct' backend/services/portfolio_manager.py
```

Result: **exit 0**.

### Test run

```
$ source .venv/bin/activate && python -m pytest tests/services/test_sector_concentration.py -v
collected 13 items

test_third_tech_buy_skipped_when_cap_is_2 PASSED [  7%]
test_disabled_cap_passes_all_through PASSED [ 15%]
test_cap_counts_existing_positions PASSED [ 23%]
test_unknown_sector_treated_as_own_bucket PASSED [ 30%]
test_diverse_sectors_all_booked PASSED [ 38%]
test_legacy_position_with_enriched_sector_blocks_same_sector_buy PASSED [ 46%]
test_legacy_position_without_enrichment_falls_into_unknown PASSED [ 53%]
test_sector_priority_via_candidates_by_ticker PASSED [ 61%]
test_nav_pct_cap_blocks_buy_when_count_cap_allows PASSED [ 69%]
test_nav_pct_cap_allows_buy_when_both_caps_hold PASSED [ 76%]
test_nav_pct_cap_zero_disables_check PASSED [ 84%]
test_nav_pct_and_count_caps_independent PASSED [ 92%]
test_nav_pct_cap_grep_symbol_present_in_portfolio_manager PASSED [100%]

13 passed in 0.03s
```

All 8 existing tests stay green (no regression); 5 new phase-30.5 tests
pass.

### Regression sweep

```
$ python -m pytest backend/tests/test_cycle_heartbeat_alarm.py \
                   backend/tests/test_autonomous_loop_step_5_6.py \
                   backend/tests/test_observability.py -q
26 passed, 1 warning in 3.82s
```

Phase-30.1 (7) + phase-30.2+30.3 (7) + observability (12) = 26/26
green. No regression from phase-30.5.

### Syntax check

`python -c "import ast; ast.parse(...)"` on
`backend/config/settings.py`, `backend/services/portfolio_manager.py`,
and the test file: OK.

## Hard guardrail attestation

- No mutating BigQuery calls -- the check is pure in-memory math on
  the existing `paper_positions.market_value` field.
- No Alpaca calls.
- No frontend / `.claude/` / `.mcp.json` touched.
- Diff stays within the audit's proposed-diff scope (one settings
  field + one decide_trades extension + one test file extension).
- Tests ship and pass deterministically.

## Success criteria check

| Criterion | Status | Evidence |
|-----------|--------|----------|
| `settings_field_paper_max_per_sector_nav_pct_added_default_30` | PASS | `grep -q 'paper_max_per_sector_nav_pct' backend/config/settings.py` exits 0; field has `default=30.0` per Pydantic `Field(30.0, ...)` |
| `portfolio_manager_enforces_both_count_and_nav_pct_caps` | PASS | Test D (`test_nav_pct_and_count_caps_independent`) verifies both gates are independent and can each block when the other allows |
| `test_covers_a_buy_blocked_by_nav_pct_cap_even_when_count_cap_passes` | PASS | Test A (`test_nav_pct_cap_blocks_buy_when_count_cap_allows`) is the strict-literal: count cap = 10 (won't block 3 Tech), NAV-pct = 30, existing Tech at 27.5% NAV; new BUY -> 31% -> blocked |
