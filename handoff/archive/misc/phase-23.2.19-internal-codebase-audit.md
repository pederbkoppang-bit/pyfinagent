# Phase-23.2.19 Internal Codebase Audit

Generated: 2026-05-05 by Researcher agent

## 1. kill_switch.py — Full Audit

**File:** `backend/services/kill_switch.py` (208 lines)

### State fields on KillSwitchState (lines 45-51)

| Field | Type | Default |
|-------|------|---------|
| `_lock` | `threading.Lock` | new lock |
| `_paused` | `bool` | `False` |
| `_pause_reason` | `Optional[str]` | `None` |
| `_sod_nav` | `Optional[float]` | `None` |
| `_peak_nav` | `Optional[float]` | `None` |

**Finding: There is NO `_sod_date` field.** There is no `_last_sod_date`, no `_sod_day`, no date tracking anywhere in `KillSwitchState`. The only way to know when `_sod_nav` was set is to read the audit log (`kill_switch_audit.jsonl`), which carries a `"ts"` timestamp on each `"sod_snapshot"` row. The in-memory object has no mechanism to determine whether the stored `_sod_nav` is from today or a prior day.

### Boot replay (lines 53-74)

`_load_from_audit()` iterates EVERY line in the audit log sequentially. For each line:
- `event == "pause"` → sets `_paused = True`, captures `pause_reason`
- `event == "resume"` → sets `_paused = False`
- `event == "sod_snapshot"` → sets `_sod_nav = float(row["nav"])`
- `event == "peak_update"` → sets `_peak_nav = float(row["nav"])`

**Root cause confirmed:** With exactly one `sod_snapshot` row in the log (`2026-04-20T12:01:03.965687+00:00, nav=9499.5`), the boot replay always lands on `_sod_nav = 9499.5`. Every subsequent process restart re-loads this stale value. No date comparison occurs during replay.

### update_sod_nav (lines 149-153)

```python
def update_sod_nav(self, nav: float) -> None:
    with self._lock:
        self._sod_nav = float(nav)
        self._append_audit("sod_snapshot", nav=self._sod_nav)
```

This method writes the value and appends to the audit log. It does NOT check whether today's SOD has already been written. It is the caller's responsibility to call it at most once per day. There is only one callsite.

### Sole callsite of update_sod_nav

`backend/services/paper_trader.py:549` — the only callsite across the entire codebase:

```python
snap = state.snapshot()
today = datetime.now(timezone.utc).date().isoformat()
if snap.get("sod_nav") is None:
    state.update_sod_nav(nav)
else:
    # idempotent daily roll -- reset when the audit log's latest sod
    # is older than today. The audit log is append-only; peek via the
    # snapshot date via a best-effort check on the JSONL tail.
    pass
```

The `else` branch is a stub — `today` is computed but unused. Once `sod_nav` is non-None (after first set on 2026-04-20), this branch always executes as a no-op. No daily roll ever fires.

### No other callsites

Confirmed by grep: `grep -rn "update_sod_nav"` returns exactly one hit in the production code at `paper_trader.py:549`. No test exercises the daily-roll path.

### evaluate_breach (lines 173-207)

Uses `_sod_nav` directly from `_state.snapshot()`. With `sod_nav = 9499.5` and `current_nav = 17270.87`:

```
daily_loss_pct = (9499.5 - 17270.87) / 9499.5 * 100 = -81.8%
```

The formula computes `(sod - current) / sod * 100`. When current > sod (account has grown), this yields a large negative number — meaning "no daily loss, you are up 81.8%". The sign convention is: positive value = loss. Negative value = gain. The display renders the raw signed float including the minus sign, so "-81.8%" appears in the UI. The kill-switch logic (`daily_loss_breached = daily_loss_pct >= daily_loss_limit_pct`) correctly does NOT trigger (a gain is not a breach). Only the display is wrong.

---

## 2. OpsStatusBar.tsx — Tooltip Pattern Audit

**File:** `frontend/src/components/OpsStatusBar.tsx` (361 lines)

### Existing tooltip implementations

| Location | Pattern | Content |
|----------|---------|---------|
| Line 164 (GateSegment div) | `title="N/M checks passing"` | e.g. `"1/5 checks passing"` |
| Line 216 (KillSegment span) | `title="Daily: X% of Y% | Trailing: X% of Y%"` | pipe-separated values |
| Line 287 (CycleSegment div) | `title={bands.map(b => b.name + ": " + b.band).join(" | ")}` | pipe-separated band names |

**Pattern in use:** All three existing tooltips use the native HTML `title=` attribute on the wrapper div/span. No project-level Tooltip primitive exists — all "Tooltip" hits in the frontend are Recharts `<Tooltip>` chart components, not a custom floating-tooltip component for informational overlays.

### GateSegment structure (lines 151-178)

```tsx
<div className="flex items-center gap-2" title={`${passes}/${total} checks passing`}>
  <SegmentLabel>Gate</SegmentLabel>
  <span className={...}>{eligible ? "ELIGIBLE" : "NOT ELIGIBLE"}</span>
  <span className="font-mono text-xs text-slate-400">{passes}/{total}</span>
  <IconInfo size={12} className="text-slate-600" />
</div>
```

The tooltip is on the outermost div. Hovering anywhere over the Gate segment shows "1/5 checks passing" — but does NOT show WHICH checks pass or fail.

### GoLiveGateWidget.tsx — Full criterion labels

**File:** `frontend/src/components/GoLiveGateWidget.tsx` (193 lines)

The full breakdown table on the paper-trading page uses these human-readable labels (lines 92-123):

| Key | Label format | Current value |
|-----|-------------|--------------|
| `trades_ge_100` | `>=${t.trades} trades` | `>=100 trades`, `n_round_trips=0` |
| `psr_ge_95_sustained_30d` | `PSR >= ${t.psr.toFixed(2)} (${t.psr_sustained_days}d)` | `PSR >= 0.95 (30d)` |
| `dsr_ge_95` | `DSR >= ${t.dsr.toFixed(2)}` | `DSR >= 0.95` |
| `sr_gap_le_30pct` | `Reality gap <= ${(t.sr_gap * 100).toFixed(0)}%` | `Reality gap <= 30%` |
| `max_dd_within_tolerance` | `Max DD <= ${t.max_dd_pct.toFixed(0)}%` | `Max DD <= 25%` |

And hint values that pair with each:
- trades: `n_round_trips` count
- psr: `psr` value + `n_obs`
- dsr: `dsr` value
- sr_gap: `latest_reconciliation_divergence_pct`
- dd: `realized_max_dd_pct`

### No custom Tooltip component

The grep confirms all `Tooltip` references in `frontend/src/components/` are Recharts chart tooltips. There is no `<Tooltip>` primitive for informational overlays. The established project pattern for non-chart tooltips is native `title=` on the wrapper element.

---

## 3. Kill Switch API Response Shape

**File:** `backend/api/paper_trading.py:355-371`

The `/api/paper-trading/kill-switch` endpoint returns:
```json
{
  "paused": false,
  "pause_reason": null,
  "sod_nav": 9499.5,
  "peak_nav": 17265.72,
  "current_nav": 17270.87,
  "breach": {
    "daily_loss_pct": -81.81,
    "daily_loss_limit_pct": 4.0,
    "trailing_dd_pct": -0.03,
    "trailing_dd_limit_pct": 10.0,
    "any_breached": false
  }
}
```

The frontend `KillSegment` renders `{kill.breach.daily_loss_pct.toFixed(1)}% / {kill.breach.trailing_dd_pct.toFixed(1)}%` → "-81.8% / -0.0%". The existing title tooltip on line 216 includes the stale `daily_loss_pct` and the correct `trailing_dd_pct`.

---

## 4. Test Coverage for SOD Daily Roll

**File:** `tests/services/test_kill_switch_no_deadlock.py` (93 lines)

Tests cover: pause/resume deadlock regression (phase-23.1.22), cycle speed benchmark, and source-level structural guards. There is NO test for:
- Daily SOD roll (else branch in `paper_trader.py:550-554` is never executed in tests)
- Mid-day system restart SOD persistence
- Stale SOD detection from audit log timestamps

The fixture pattern is: instantiate `KillSwitchState()` directly (not via `get_state()` singleton), call methods, assert state. Tests use `threading.Event` for concurrency checks. The `_AUDIT_PATH` write happens at each test but to the real file — tests do NOT mock the audit path.

---

## 5. Integration Points for the Fix

### SOD daily-roll fix (kill_switch.py)

The fix requires either:

**Option A — Date field in KillSwitchState:**
Add `_sod_date: Optional[str]` field. `update_sod_nav` stores the date alongside the nav. Boot replay reads `"date"` from `sod_snapshot` rows. `check_and_enforce_kill_switch` compares `state._sod_date` to today and calls `update_sod_nav` if different.

**Option B — Parse timestamps in boot replay:**
During `_load_from_audit()`, for each `sod_snapshot` row, store not just `nav` but also the parsed date from `"ts"`. Expose it via `snapshot()`. The caller in `paper_trader.py` reads `snap.get("sod_date")` and compares.

**Option C — Parse audit log tail at callsite:**
The existing TODO comment in `paper_trader.py:552-554` describes this: peek the JSONL tail to find the latest `sod_snapshot` ts, compare to today. Most fragile — race-condition prone.

Option A is cleanest: it adds minimal state, keeps audit log backward-compatible (the `sod_snapshot` row gains an optional `date` field), and the daily-roll check in `paper_trader.py` becomes a simple string comparison.

### Gate tooltip fix (OpsStatusBar.tsx)

The fix must add per-criterion breakdown to the `GateSegment` title. Following the existing project pattern (native `title=`), the tooltip content mirrors the GoLiveGateWidget labels:

```
GATE CHECKS (1/5 passing)
PASS: Reality gap <= 30% (11.05%)
FAIL: >= 100 trades (0 closed round trips)
FAIL: PSR >= 0.95 (30d) (n_a, n_obs=12)
FAIL: DSR >= 0.95 (n/a)
FAIL: Max DD <= 25% (44.997%)
```

The `GateSegment` component already receives the full `GoLiveGate` object (including `booleans`, `details`, and `thresholds`). All data needed for the tooltip is available without additional API calls.

---

## 6. Internal File Summary

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/services/kill_switch.py` | 208 | Kill-switch state, audit log, breach eval | Bug: no `_sod_date` field; SOD never refreshed |
| `backend/services/paper_trader.py:533-566` | 34 | SOD roll callsite (check_and_enforce_kill_switch) | Bug: `else: pass` stub never fires daily roll |
| `backend/api/paper_trading.py:355-371` | 17 | Kill-switch API endpoint, returns stale sod_nav | Correct — exposes what state holds |
| `frontend/src/components/OpsStatusBar.tsx` | 361 | Ops status bar, GateSegment tooltip | Bug: title only shows count, not per-criterion detail |
| `frontend/src/components/GoLiveGateWidget.tsx` | 193 | Full gate breakdown table on paper-trading page | Correct — has all label/hint data to mirror |
| `tests/services/test_kill_switch_no_deadlock.py` | 93 | Deadlock regression tests | Gap: no SOD daily-roll test |
| `handoff/kill_switch_audit.jsonl` | N/A | Append-only audit trail | Has 1 sod_snapshot from 2026-04-20, stale |
