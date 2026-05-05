---
step: phase-23.2.19
cycle_date: 2026-05-05
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_2_19.py'
---

# Experiment Results — phase-23.2.19

## Hypothesis recap

User screenshot showed `KILL ACTIVE -81.8%` (impossible drawdown) and
`GATE 1/5 NOT ELIGIBLE` (uninformative tooltip). Live API confirmed
`sod_nav=$9499.50, current_nav=$17270.87` -- a stale SOD anchor from
2026-04-20 because `paper_trader.check_and_enforce_kill_switch` had an
`else: pass` stub for the daily-roll branch. `today` was computed at
line 547 but never used. After 15 days the audit log still held a
single sod_snapshot row at $9499.50.

## What was changed

### Fix A -- KillSwitchState tracks sod_date
`backend/services/kill_switch.py`:
- New `_sod_date: Optional[str]` field on `KillSwitchState`.
- Boot replay reads explicit `date` from the audit row when present;
  falls back to parsing `ts` for legacy rows (`fromisoformat` ->
  UTC date string). Existing 04-20 row maps to `_sod_date="2026-04-20"`.
- `update_sod_nav(nav, date=None)` now accepts an optional `date` kwarg
  and writes both `nav` and `date` into the audit row. `date=None`
  defaults to today's UTC date.
- `_snapshot_locked()` returns `sod_date` alongside the existing keys.

### Fix B -- daily roll in paper_trader
`backend/services/paper_trader.py:546-554`:
Replaced the `if/else: pass` stub with a single guard:
```python
if snap.get("sod_nav") is None or snap.get("sod_date") != today:
    state.update_sod_nav(nav, date=today)
```
First cycle of a new UTC calendar day re-anchors SOD to that day's
open NAV. Same-day re-calls are no-ops (the comparison is False, no
new audit row written). Restart-idempotent: boot replay restores
`_sod_date`, so a mid-day restart preserves the morning's anchor.

### Fix C -- API exposes sod_date
`backend/api/paper_trading.py:355-371`:
`/api/paper-trading/kill-switch` response now includes `sod_date` so
the UI / operator can see when daily-loss% was last re-anchored.

### Fix D -- Per-criterion gate tooltip
`frontend/src/components/OpsStatusBar.tsx`:
`GateSegment` now builds a multi-line title string mirroring the
labels from `GoLiveGateWidget.tsx:92-123`. Each line:
`[PASS|FAIL] <label> (<actual>)`. All 5 booleans
(`trades_ge_100`, `psr_ge_95_sustained_30d`, `dsr_ge_95`,
`sr_gap_le_30pct`, `max_dd_within_tolerance`) referenced. Native
multi-line `title=` per WCAG 1.4.13 native-attr exemption (operator
UI, mouse-driven). No new component, no new CSS, no JS.

### Tests
- `tests/services/test_sod_daily_roll.py` -- 8 tests:
  - `test_snapshot_now_includes_sod_date` -- snapshot shape contract
  - `test_update_sod_nav_stamps_explicit_date_in_audit_row` -- writer
  - `test_update_sod_nav_default_date_is_today` -- default kwarg
  - `test_paper_trader_rolls_sod_on_new_day` -- the core bug fix
  - `test_paper_trader_does_not_roll_same_day` -- idempotency
  - `test_boot_replay_restores_sod_date_from_explicit_field` -- forward
  - `test_boot_replay_falls_back_to_ts_for_legacy_rows` -- backward compat
  - `test_legacy_row_then_new_day_rolls_correctly` -- exact prod path
- `tests/verify_phase_23_2_19.py` -- 5-check immutable verifier.

## Files modified / added

```
backend/services/kill_switch.py             -- _sod_date field + boot replay + update_sod_nav signature + snapshot
backend/services/paper_trader.py            -- daily-roll guard
backend/api/paper_trading.py                -- /kill-switch endpoint exposes sod_date
frontend/src/components/OpsStatusBar.tsx    -- multi-line per-criterion tooltip
tests/services/test_sod_daily_roll.py       -- NEW, 8 regression tests
tests/verify_phase_23_2_19.py               -- NEW, 5-check verifier
handoff/current/contract.md                 -- updated for phase-23.2.19
handoff/current/phase-23.2.19-external-research.md      -- researcher output
handoff/current/phase-23.2.19-internal-codebase-audit.md -- researcher output
```

## Verification (verbatim output)

```
$ source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_2_19.py
OK backend/services/kill_switch.py
OK backend/services/paper_trader.py
OK backend/api/paper_trading.py
OK frontend/src/components/OpsStatusBar.tsx
OK tests/services/test_sod_daily_roll.py

phase-23.2.19 verification: ALL PASS (5/5)

$ PYTHONPATH=. pytest tests/services/test_sod_daily_roll.py -q
........                                                                 [100%]
8 passed in 0.02s

$ PYTHONPATH=. pytest tests/services/test_cycle_failure_alerts.py \
                     tests/services/test_kill_switch_no_deadlock.py \
                     tests/services/test_snapshot_upsert.py -q
14 passed in 0.97s

$ cd frontend && npx --no-install tsc --noEmit
(no output - clean)
```

## Research-gate evidence

Researcher (a0d307cf3d1e4f3c3) returned `gate_passed: true`:
- 7 sources read in full via WebFetch (24a11y title-attribute trials,
  Sarah Higley tooltips-WCAG-2.1, W3C SC 1.4.13 -- confirms native
  title= EXEMPT from hoverable/dismissable, MDN tooltip role updated
  Nov-2025, Trading Technologies SOD doctrine, flook.co tooltip
  accessibility, W3C/WAI ARIA APG tooltip pattern).
- 17 URLs collected; 10 in snippet-only.
- Recency scan 2024-2026 performed; no breaking changes (WCAG 2.2
  Oct-2023 unchanged 1.4.13).
- 7 internal files inspected with file:line anchors.

## Backwards compatibility

- `update_sod_nav(nav)` keeps its existing single-arg call shape
  (`date` defaults to today). Existing callers untouched.
- Boot replay handles legacy audit rows lacking `date` via `ts`
  fallback. Production's lone 04-20 row is parsed correctly.
- Kill-switch endpoint adds a new key `sod_date`; all existing fields
  unchanged. UI consumers are forward-compatible.
- OpsStatusBar `GateSegment` outer rendering unchanged (same chips,
  counts, IconInfo). Only the `title=` attribute string changed.

## Honest disclosures

- **First-cycle behavior:** the next cycle to run will re-anchor SOD
  to that cycle's NAV. The displayed daily-loss% will jump from -81.8%
  to ~0% as soon as the cycle fires. Operator should expect this as
  the visible signal that the fix is working.
- **Pure-UTC date comparison:** rolls every UTC midnight regardless of
  weekend / market holiday. For paper trading this is correct (cycles
  run on cron mon-fri 14:00 EDT = ~18:00 UTC, so each cycle's first
  anchor is that day's open). If the operator wants weekend-skip
  semantics in the future, that's a follow-up.
- **No backfill of legacy audit rows.** The lone 04-20 row stays
  unchanged; boot replay handles it via `ts` fallback. New rows
  written post-fix have explicit `date`.
- **WCAG posture:** native multi-line `title=` is informational-only.
  Mouse-only reveal; no keyboard or screen-reader access. WCAG 1.4.13
  exempts native browser title= explicitly. If a screen-reader user
  joins the team, upgrade to `role="tooltip"` + `aria-describedby`
  with a small dedicated component.
- **Live `current_nav` was $17270.87 at fix time.** The next cycle (or
  next backend restart's first kill-switch poll) will write a new
  sod_snapshot with today's NAV. Operator can verify in
  `handoff/kill_switch_audit.jsonl` -- new row should appear with
  `date: "2026-05-05"` and a non-stale nav.
- **The frontend tooltip uses native `title`.** Tooltip newlines
  render in most browsers but Safari historically rendered them as
  one line until ~Safari 17. Acceptable degradation for an internal
  operator UI on Chrome.
