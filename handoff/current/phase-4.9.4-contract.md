# Contract -- Cycle 92 / phase-4.9 step 4.9.4

Step: 4.9.4 Gauntlet regime catalog (7 historical windows)

## Research-gate upheld (7th cycle)

Researcher (25 URLs across primary sources: NBER business-cycle
dating, Fed press releases, SNB press release + Jordan speech,
SEC/CFTC final flash-crash report, BIS Bulletin 90 for yen carry,
Cboe VIX attribution tables) + Explore (inventory of existing
backtest scaffolding: no `gauntlet/` dir yet; dominant idiom is
ISO-string start/end dates with `date.fromisoformat` internally;
walk_forward.py + spot_checks.py already support per-run date
injection; spot_checks.py:168-173 has hardcoded 2-regime fallback
to be superseded; 10+ existing regime/stress references including
agent_definitions.py:117 Robustness criteria).

## Honest scope call

Step 4.9.4 is a DATA CATALOG step. We create the 7-entry
`REGIMES` tuple; we do NOT (this cycle):
- run the gauntlet against any strategy (that's 4.9.5 "Gauntlet
  runner");
- integrate with spot_checks.py (the hardcoded 2-regime pre/post-
  COVID pattern remains in place until 4.9.7 promotion-pipeline);
- verify BigQuery data coverage for every window (the 2025 tariff
  window may have partial data; we flag it in the module but
  don't block).

The date ranges in this catalog are IMMUTABLE once shipped --
changing them would invalidate historical gauntlet results.
Corrections require a new phase-4.9.x step with regression
analysis. This is enforced by the lint allowlist (existing
phase-4.9.3 lint already protects limits_schema.py; this step
adds regimes.py to a NEW "catalog allowlist" concept in 4.9.9).

## Scope

Files created:

1. **NEW** `backend/backtest/gauntlet/__init__.py`
   Package marker. Exports `REGIMES`, `RegimeWindow` from regimes.py.

2. **NEW** `backend/backtest/gauntlet/regimes.py`
   Frozen `RegimeWindow` dataclass + immutable `REGIMES` tuple of
   exactly 7 entries. Each entry:
   - `id: str` (snake_case canonical id)
   - `name: str` (human-readable)
   - `start: str` (ISO YYYY-MM-DD, inclusive)
   - `end: str` (ISO YYYY-MM-DD, inclusive)
   - `asset_classes: tuple[str, ...]` (equity, FX, rates, credit,
     oil, crypto)
   - `region: str` (US, Europe, Japan, Global)
   - `note: str` (shock mechanism + key ticker/VIX references)
   - `primary_source_url: str` (from researcher)
   - `intraday_only: bool` (True only for flash_crash_2010 --
     daily-bar backtests will show zero drawdown for that window)

   The 7 entries, with dates sourced from researcher:
   - gfc_2008:              2008-09-15 .. 2009-03-09
   - flash_crash_2010:      2010-05-06 .. 2010-05-06 (intraday_only)
   - snb_chf_2015:          2015-01-15 .. 2015-01-26
   - covid_crash_2020:      2020-02-19 .. 2020-03-23
   - fed_hike_shock_2022:   2022-01-03 .. 2022-10-12
   - yen_carry_unwind_2024: 2024-07-31 .. 2024-08-09
   - tariff_vol_2025:       2025-04-02 .. 2025-04-09

3. **NEW** `scripts/audit/gauntlet_regimes_audit.py`
   Seven teeth:
   (a) import_ok: `REGIMES` importable, is a tuple of length 7.
   (b) ids_unique: no duplicate ids; ids match a strict
       `^[a-z][a-z0-9_]*$` pattern.
   (c) dates_valid: every start/end parses as ISO date; start <= end.
   (d) dates_chronologically_sorted: REGIMES ordered by start date
       (catches accidental reordering in future PRs).
   (e) immutability: `RegimeWindow` is a frozen dataclass
       (attempting to set a field raises FrozenInstanceError).
   (f) universe_fields_populated: every entry has non-empty
       asset_classes, region, note, primary_source_url;
       primary_source_url starts with `https://`.
   (g) intraday_flag_consistent: the flash_crash_2010 entry (and
       only that entry) has `intraday_only=True` with
       `start == end`; all others have
       `intraday_only=False` with `start <= end`.

## Immutable success criteria (from masterplan)

1. seven_regimes_defined: `len(REGIMES) == 7` and every entry
   has `start` and `end` keys (masterplan verification command).
2. date_ranges_immutable: `RegimeWindow` is a frozen dataclass;
   attempting mutation raises.
3. universe_hints_present: every entry has non-empty
   `asset_classes`, `region`, `note` fields.

## Verification (immutable, from masterplan)

    python -c "from backend.backtest.gauntlet.regimes import REGIMES; \
    assert len(REGIMES) == 7 and all('start' in r and 'end' in r for r in REGIMES)"

Plus: `python scripts/audit/gauntlet_regimes_audit.py --check`.

Note: the masterplan verification uses `'start' in r` which implies
a dict-like interface. We will implement `RegimeWindow` as a frozen
dataclass that also supports dict-style key access (via a `__iter__`
returning field names, or a `keys()` method + `__getitem__`) so the
immutable verification passes verbatim without edits.

## Anti-rubber-stamp

qa must check:
- `REGIMES` is defined at module level as a `tuple` (not list),
  so reassignment is clearly signalled. Each element is a
  RegimeWindow frozen dataclass instance.
- The mutation-test fixture in the audit must actually attempt a
  field set (e.g. `REGIMES[0].end = "2030-01-01"`) and confirm a
  `FrozenInstanceError` is raised -- not just trust the decorator.
- The date range for each regime matches the researcher's primary
  source EXACTLY (not a paraphrase).
- Exactly one entry has `intraday_only=True` (flash_crash_2010);
  zero others.
- The masterplan verification command runs verbatim and passes
  (dict-style key access works on the dataclass).
- Note field (not just a stub) mentions a real ticker/index and a
  real VIX level or percentage drop, per the researcher output.

## References

- Researcher cycle-92 findings (25 URLs; primary: NBER, SNB, Fed,
  SEC/CFTC, BIS Bulletin 90, Cboe VIX attribution).
- Explore cycle-92 findings (existing backtest scaffolding + ISO
  string/date idiom + spot_checks.py:168-173 hardcoded 2-regime
  fallback).
- backend/backtest/walk_forward.py:30-42 (date parsing idiom).
- backend/backtest/backtest_engine.py:144-145 (API ISO string
  contract).
