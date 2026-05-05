---
step: phase-23.2.19
title: SOD NAV daily-roll fix + Go-Live Gate per-criterion tooltip
cycle_date: 2026-05-05
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_2_19.py'
research_brief: handoff/current/phase-23.2.19-external-research.md (also see phase-23.2.19-internal-codebase-audit.md)
---

# Contract — phase-23.2.19

## Hypothesis

User screenshot of the Ops Status Bar shows two issues at 2026-05-05 20:44 CEST:

1. **`KILL ACTIVE -81.8% / -0.0%`** — the "-81.8%" is a stale-SOD display
   bug. Live `/api/paper-trading/kill-switch` returned
   `sod_nav=9499.50, current_nav=17270.87, daily_loss_pct=-81.81`.
   `handoff/kill_switch_audit.jsonl` contains exactly ONE
   `sod_snapshot` event ever — `2026-04-20T12:01:03.965687+00:00 nav=9499.5`.
   Every cycle since has skipped the daily roll because of the
   `else: pass` stub at `backend/services/paper_trader.py:550-554`:
   ```python
   if snap.get("sod_nav") is None:
       state.update_sod_nav(nav)
   else:
       # idempotent daily roll -- TODO
       pass
   ```
   `today` is computed at line 547 but never compared. Once `_sod_nav`
   is non-None it stays frozen forever. The kill-switch breach gate
   itself is unaffected (`any_breached: false` because -81.8% is a
   gain, sign convention positive=loss); only the rendered % is wrong.

2. **`GATE NOT ELIGIBLE 1/5`** — by-design (paper-to-live promotion gate),
   but the tooltip on hover only says `"1/5 checks passing"`
   (`frontend/src/components/OpsStatusBar.tsx:164`). User wants to see
   WHICH criterion is the 1 passing and which 4 are blocking, without
   leaving the page. The full breakdown already exists in
   `GoLiveGateWidget.tsx:92-123` (labels + thresholds + actuals); we
   just need a multi-line `title=` string in the OpsStatusBar segment
   reusing those labels.

## Research-gate summary

Researcher (a0d307cf3d1e4f3c3) returned `gate_passed: true`:
- 7 sources read in full via WebFetch (24a11y title-attribute,
  Sarah Higley tooltips-WCAG-2.1, W3C SC 1.4.13 — confirms title=
  EXEMPT from hoverable/dismissable requirements, MDN tooltip role
  Nov-2025, Trading Technologies SOD docs, flook.co tooltip
  accessibility, W3C/WAI ARIA APG tooltip pattern).
- 17 URLs collected; 10 in snippet-only.
- Recency scan 2024-2026 — no breaking changes (WCAG 2.2 Oct-2023
  did not change 1.4.13; MDN tooltip page updated Nov 2025).
- 7 internal files inspected with file:line anchors.
- Key external finding: WCAG 1.4.13 explicitly exempts native browser
  `title=` from its requirements. For internal operator UI used with
  a mouse, native multi-line `title=` is the proportionate fix; full
  ARIA tooltip would require new component + CSS + JS.
- Key internal finding: `GateSegment` already receives the full
  `GoLiveGate` object — all data for a richer tooltip is in scope.

## Immutable success criteria (verbatim — DO NOT EDIT)

1. `KillSwitchState` exposes `sod_date` (UTC ISO date string) alongside
   `sod_nav` in its snapshot payload. Field is set whenever
   `update_sod_nav` is called and is restored on boot from the latest
   `sod_snapshot` audit row's `ts`.
2. `paper_trader.check_and_enforce_kill_switch` calls
   `update_sod_nav(current_nav)` whenever `snap.get("sod_date")`
   does NOT equal today's UTC date. The first cycle of each new UTC
   calendar day re-anchors SOD to that day's open NAV.
3. The kill-switch endpoint `/api/paper-trading/kill-switch` returns
   `sod_date` in its response payload (so the UI can show the actual
   anchor date if needed).
4. `OpsStatusBar` GateSegment tooltip lists each of the 5 booleans
   with PASS/FAIL + threshold + actual, one criterion per line.
   Backwards-compat: existing `eligible / passes / total` rendering
   on the segment itself is unchanged.
5. Regression test `tests/services/test_sod_daily_roll.py` (new):
   - first-ever cycle: `sod_nav` and `sod_date` get written
   - same-day re-call: NO new audit row written (idempotent)
   - new-day re-call: new audit row, `sod_nav` updated to today's NAV,
     `sod_date` updated to today
6. The frontend build passes (`cd frontend && npm run build` exits 0
   OR `tsc --noEmit` passes if a full build is too slow).
7. `python -c "import ast; ast.parse(open(P).read())"` passes for
   every modified `.py` file.
8. `python tests/verify_phase_23_2_19.py` exits 0.

## Plan steps

1. `backend/services/kill_switch.py`:
   - Add `_sod_date: Optional[str] = None` field to `KillSwitchState.__init__`.
   - In boot replay loop (around line 70), when an `sod_snapshot` row
     is read, store `_sod_date` from the row's `date` field if present,
     else parse it from `ts` (`datetime.fromisoformat(ts).date().isoformat()`).
     Backwards compat: existing audit rows without an explicit `date`
     field still work via the `ts` fallback.
   - Update `update_sod_nav(nav, date=None)` to accept optional date
     and append `date` to the audit row + store in `_sod_date`. If
     `date` is None, default to today's UTC date.
   - Expose `sod_date` in `_snapshot_locked()` and `snapshot()`.
2. `backend/services/paper_trader.py:546-554`:
   - Replace the `if snap.get("sod_nav") is None: ... else: pass` block
     with `today = datetime.now(timezone.utc).date().isoformat()` (already
     computed) and a single condition:
     `if snap.get("sod_nav") is None or snap.get("sod_date") != today:
         state.update_sod_nav(nav, date=today)`.
3. `backend/api/paper_trading.py:355-371`:
   - Add `sod_date` to the kill-switch endpoint response (read from the
     existing `state.snapshot()` dict).
4. `frontend/src/components/OpsStatusBar.tsx`:
   - In `GateSegment` (around line 158), build a multi-line tooltip
     string from `gate.booleans`, `gate.details`, `gate.thresholds`.
     Each line: `[PASS|FAIL] <label> (<actual> vs <threshold>)`.
   - Replace the existing `title="${passes}/${total} checks passing"`
     with the new multi-line string.
   - No new component, no new CSS, no JS. Native multi-line `title=`
     supports `\n` separators.
5. Regression test `tests/services/test_sod_daily_roll.py` per criterion 5.
6. Verifier `tests/verify_phase_23_2_19.py`:
   - `ast.parse` modified .py files
   - assert `_sod_date` field exists in kill_switch.py
   - assert `sod_date` keyword in update_sod_nav signature
   - assert `sod_date` returned from snapshot
   - assert paper_trader uses `sod_date` comparison
   - assert OpsStatusBar tooltip references each of the 5 boolean keys
7. Append `handoff/harness_log.md` cycle entry AFTER Q/A PASS, BEFORE
   any masterplan flip.

## Out of scope

- New ARIA tooltip component (`role="tooltip"` + `aria-describedby`):
  research confirmed native `title=` is sufficient for an internal
  operator UI under WCAG 1.4.13's exemption. Phase-2 if a screen-reader
  user joins the team.
- Backfill of historical `kill_switch_audit.jsonl` to add `date` to
  the lone 04-20 sod_snapshot row. The boot replay's `ts`-fallback
  handles legacy rows; no migration needed.
- Market-calendar awareness (skip SOD update on weekends/holidays):
  pure-UTC date comparison rolls every day. If the operator wants
  weekend-skip, that's a follow-up — for current paper-trading use
  the daily roll is correct.

## Backwards compatibility

- `update_sod_nav(nav)` keeps its existing single-arg call shape
  (default `date=None` -> today). Existing callers untouched.
- Boot replay handles legacy audit rows lacking `date` via `ts`
  fallback. The lone 04-20 row will be picked up as
  `_sod_date = "2026-04-20"`, and the next cycle will re-anchor it.
- Kill-switch endpoint adds a new key `sod_date`; existing fields
  unchanged. UI consumers are forward-compatible.
- OpsStatusBar swap is internal to `GateSegment`; the segment's
  outer rendering, color, and counts are unchanged.

## References

- Researcher: `handoff/current/phase-23.2.19-external-research.md`,
  `handoff/current/phase-23.2.19-internal-codebase-audit.md`
- `backend/services/paper_trader.py:533-566`
- `backend/services/kill_switch.py:45-72, 149-153, 188-192`
- `backend/api/paper_trading.py:355-371`
- `frontend/src/components/OpsStatusBar.tsx:158-180`
- `frontend/src/components/GoLiveGateWidget.tsx:92-123`
- W3C WCAG 1.4.13 native-title= exemption
- Trading Technologies SOD reset doctrine (time-governed, not
  event-governed; restart-idempotent)
