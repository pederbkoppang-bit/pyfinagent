# Contract -- Phase 4.2.3.2 / SN4 `since_date` Lexicographic Trap Fix

**Step ID:** 4.2.3.2 (SN4 micro-fix, follow-on to Phase 4.2.2 soft note SN4)
**Target file:** `backend/agents/mcp_servers/signals_server.py`
**Target function:** `SignalsServer.get_signal_history` (+ new private helper)
**Base commit:** `2494d10` (origin/main HEAD as of 2026-04-14T2200Z)

## Problem statement

Phase 4.2.2 `get_signal_history` compares `since_date` against stored
signal dates using Python string `>=`. This is a **lexicographic**
compare, which is only equivalent to chronological compare when BOTH
sides are strictly zero-padded canonical ISO-8601 (`YYYY-MM-DD`).

Concrete failure cases:

- `since_date="2026-4-1"` (unpadded), stored `date="2026-04-15"`
  (padded): `"2026-04-15" >= "2026-4-1"` is **False** (because `"0" < "4"`
  at index 5), so a valid April 15 record is **spuriously excluded**.
- `since_date="2026-04-01"` (padded), stored `date="2026-1-15"`
  (unpadded): `"2026-1-15" >= "2026-04-01"` is **True** (because `"1" > "0"`
  at index 5), so a January 15 record is **spuriously included** when
  the caller asked for "since April 1".

Documented as soft note SN4 in the Phase 4.2.2 QA critique, and again
in the Phase 4.2.3 / 4.2.3.1 session logs. This contract closes SN4.

## Fix approach (locked by research.md)

1. Add `date` to the existing `from datetime import datetime, timezone`
   line (one-word addition, no new import statement).
2. Add a private `@staticmethod _parse_iso_date(s) -> Optional[date]`
   on `SignalsServer` that:
   - Returns `None` for any non-string or empty input.
   - Tries `date.fromisoformat(s)` first (canonical fast path).
   - On failure, splits on `"-"`, re-pads each component with
     `f"{int(x):02d}"`, retries `date.fromisoformat`.
   - Catches `ValueError` + `TypeError`, returns `None` on any failure.
3. Replace the lex-compare block in `get_signal_history` with a
   `date`-object compare using the new helper on both `since_date`
   (once) and each record's `date` field (per iteration).

## Success criteria (25 deterministic assertions)

### Group A -- Helper happy path (SC1-SC6)

- **SC1:** `_parse_iso_date("2026-04-01") == date(2026, 4, 1)`
- **SC2:** `_parse_iso_date("2026-04-15") == date(2026, 4, 15)`
- **SC3:** `_parse_iso_date("2000-01-01") == date(2000, 1, 1)`
- **SC4:** `_parse_iso_date("2026-4-1") == date(2026, 4, 1)` (unpadded)
- **SC5:** `_parse_iso_date("2026-4-15") == date(2026, 4, 15)` (unpadded month)
- **SC6:** `_parse_iso_date("2026-12-1") == date(2026, 12, 1)` (unpadded day)

### Group B -- Helper reject path (SC7-SC12)

- **SC7:** `_parse_iso_date(None) is None`
- **SC8:** `_parse_iso_date("") is None`
- **SC9:** `_parse_iso_date("not-a-date") is None`
- **SC10:** `_parse_iso_date(20260401) is None` (int, wrong type)
- **SC11:** `_parse_iso_date("2026-13-01") is None` (invalid month)
- **SC12:** `_parse_iso_date("2026-04-32") is None` (invalid day)

### Group C -- get_signal_history SN4 semantics (SC13-SC18)

- **SC13:** With `signal_history = [{"date": "2026-04-15", ...}]` and
  `since_date="2026-4-1"`, the result includes the April 15 signal.
  (Previously excluded by lex compare.)
- **SC14:** With `signal_history = [{"date": "2026-1-15", ...}]` and
  `since_date="2026-04-01"`, the result **excludes** the January 15
  signal. (Previously included by lex compare.)
- **SC15:** With a canonical padded since_date and padded stored dates,
  the filter behavior is unchanged from pre-fix (back-compat).
- **SC16:** `since_date=None` returns all signals (no filter applied).
- **SC17:** `since_date="not-a-date"` returns all signals (parse failure
  degrades to unfiltered, never raises).
- **SC18:** A stored signal with `date="not-a-date"` is silently
  dropped from the filtered result when `since_date` is a valid date.

### Group D -- Never-raise + edge cases (SC19-SC22)

- **SC19:** `get_signal_history(since_date={"x": 1})` does not raise,
  returns a valid dict with all keys.
- **SC20:** `get_signal_history(since_date=42)` does not raise
  (int input).
- **SC21:** A signal record that is not a dict (e.g. `None`) is
  tolerated by the filter loop and skipped.
- **SC22:** Calling `get_signal_history(limit=5, since_date="2026-4-1")`
  returns a dict with the correct `month`, `count`, `signals`,
  `total_count` keys.

### Group E -- Scope discipline (SC23-SC25)

- **SC23:** The only lines changed in `signals_server.py` are (a) the
  `from datetime import ...` line (adds `date`), (b) the new
  `_parse_iso_date` helper, and (c) the `since_date` block in
  `get_signal_history`. No other function changes.
- **SC24:** `ast.dump()` of every `SignalsServer` method other than
  `get_signal_history` (and `_parse_iso_date` which is new) is
  byte-identical pre-fix vs post-fix. Preserved method list (min):
  `_signal_id`, `_empty_response`, `_remember`, `_risk_response`,
  `generate_signal`, `validate_signal`, `risk_check`, `size_position`,
  `check_stop_loss`, `track_drawdown`, `get_portfolio`,
  `get_risk_constraints`, `publish_signal`, `track_signal_accuracy`,
  `get_accuracy_report`, `_wilson_ci`, `_append_signal_history`.
- **SC25:** Diff bound: `<= 50` added / `<= 15` deleted lines. One
  file touched. Zero new top-level imports beyond adding `date` to
  the existing `from datetime import ...`.

## Adversarial probes (10, for qa-evaluator)

1. `_parse_iso_date("2026-4-1T00:00:00")` -- datetime form, should
   return `None` (out of scope per research).
2. `_parse_iso_date("2026/04/01")` -- wrong separator, should return
   `None`.
3. `_parse_iso_date("20260401")` -- basic ISO form. Python 3.11+
   `fromisoformat` accepts this in the strict path, so it should
   parse successfully. (Documented expected behavior.)
4. `_parse_iso_date("  2026-04-01  ")` -- whitespace-padded. Strict
   path fails, split path succeeds because `int()` strips whitespace
   on each component; returns `date(2026, 4, 1)`. (Lenient but safe:
   whitespace is never a chronological ambiguity.)
5. `_parse_iso_date("2026-4-1 ")` -- trailing space. Split path
   `int("1 ")` succeeds for the same reason; returns `date(2026, 4, 1)`.
6. 100-record fuzz: mix of padded / unpadded / garbage records +
   unpadded `since_date` -- filter correctly selects only records
   with `parsed_date >= since_dt`, never raises.
7. `since_date` passed as a `date` object directly (not a string) --
   `_parse_iso_date` returns `None` (isinstance check fails),
   degrades to unfiltered. No raise.
8. `since_date = "2026-04-01"` and one signal has
   `date = "2026-04-01"` -- inclusive boundary, should be INCLUDED
   (`>=`, not `>`).
9. `get_signal_history(since_date="2026-1-1")` called twice in a row
   on the same state -- deterministic, same result both times.
10. Mutation safety: the returned `signals` list is a new list; mutating
    it does not affect `self.signal_history`.

## Anti-leniency rules

- **Only the helper + the one call-site may change.** If QA finds a
  touched line outside these two regions, REJECT.
- **No emoji, no Unicode.** ASCII only throughout.
- **No new top-level imports except `date`.** The existing
  `from datetime import datetime, timezone` line becomes
  `from datetime import datetime, timezone, date`.
- **Never add a `raise` statement.** All date-parse errors are caught
  and degrade to `None`.
- **No dateutil, no pendulum, no arrow, no third-party date lib.**
- **Helper name is exactly `_parse_iso_date`.**
- **Diff bound hard-capped.** `<= 50` added / `<= 15` deleted.
- **Byte-identity preserved** for every `SignalsServer` method other
  than `get_signal_history` and the new `_parse_iso_date`.

## Out of scope

- Hardening `_append_signal_history` to normalize date-on-write.
- Parsing full datetime strings or other locale formats.
- Any Phase 4.2.4 BQ persistence work.
- Any re-touch of formatters.py (Phase 4.2.3.1 scaffold).
- masterplan.json status sync (separate task).
