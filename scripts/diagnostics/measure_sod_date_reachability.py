#!/usr/bin/env python3
"""phase-85.5.1 criterion 1 -- can `sod_date` be None or STALE in PRODUCTION?

The step forbids answering this by reasoning: the measurement has to be recorded
verbatim. So this script *reaches* each candidate state through the production
`KillSwitchState` replay and prints what the switch actually does there.

Every case runs against a throwaway audit journal in a temp directory. The
operator's `handoff/kill_switch_audit.jsonl` is never read or written, and the
script asserts that isolation before measuring anything -- an isolation that is
assumed rather than proven is not isolation.

Usage:
    source .venv/bin/activate && python scripts/diagnostics/measure_sod_date_reachability.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import backend.services.kill_switch as ks  # noqa: E402

DAILY_LIMIT = 4.0
TRAILING_LIMIT = 10.0


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _yesterday() -> str:
    return (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()


def _row(**kw) -> str:
    return json.dumps(kw)


def _measure(label: str, rows: list[str], current_nav: float, note: str = "") -> dict:
    """Write `rows` to an isolated journal, replay it through the REAL state, and
    evaluate a REAL breach against it."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "kill_switch_audit.jsonl"
        path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        orig = ks._AUDIT_PATH
        ks._AUDIT_PATH = path
        try:
            assert str(ks._AUDIT_PATH).startswith(td), "ISOLATION FAILED"
            # `_audit_source_paths` reads `_AUDIT_PATH` at CALL time and derives
            # the archive dir from its .parent, so one redirect moves every
            # replayed source. Prove it -- the live journal must not appear.
            srcs = [str(x) for x in ks._audit_source_paths()]
            assert all(x.startswith(td) for x in srcs), (
                f"ISOLATION FAILED -- a live source is still in the replay: {srcs}"
            )
            state = ks.KillSwitchState()
            snap = state.snapshot()
            prev_state = getattr(ks, "_state", None)
            ks._state = state
            try:
                breach = ks.evaluate_breach(current_nav, DAILY_LIMIT, TRAILING_LIMIT)
            finally:
                if prev_state is not None:
                    ks._state = prev_state
        finally:
            ks._AUDIT_PATH = orig

    drop_pct = None
    if snap.get("sod_nav"):
        drop_pct = round((snap["sod_nav"] - current_nav) / snap["sod_nav"] * 100, 4)

    print(f"\n{'='*76}\nCASE {label}")
    if note:
        print(f"  {note}")
    print(f"  replayed snapshot : sod_nav={snap.get('sod_nav')!r} "
          f"sod_date={snap.get('sod_date')!r} peak_nav={snap.get('peak_nav')!r}")
    print(f"  current_nav       : {current_nav}   (drop vs sod: {drop_pct}%)")
    print(f"  armed             : {breach.get('armed')}")
    print(f"  daily_baseline_missing={breach.get('daily_baseline_missing')} "
          f"daily_baseline_stale={breach.get('daily_baseline_stale')}")
    print(f"  daily_loss_breached   : {breach.get('daily_loss_breached')}  "
          f"({breach.get('daily_loss_pct')}%)")
    print(f"  trailing_dd_breached  : {breach.get('trailing_dd_breached')}  "
          f"({breach.get('trailing_dd_pct')}%)")
    print(f"  >>> any_breached      : {breach.get('any_breached')}")
    return {"snapshot": snap, "breach": breach}


def main() -> int:
    print("phase-85.5.1 -- production reachability of a None/STALE daily anchor")
    print(f"measured {datetime.now(timezone.utc).isoformat()}  "
          f"limits: daily {DAILY_LIMIT}% / trailing {TRAILING_LIMIT}%")
    print("Every case replays an ISOLATED journal through the real KillSwitchState.")

    results = {}

    # ── C: the UTC rollover. No fault required; happens every single day. ──────
    results["C"] = _measure(
        "C -- UTC ROLLOVER (yesterday's anchor, before today's first roll)",
        [_row(ts=f"{_yesterday()}T18:00:00+00:00", event="sod_snapshot",
              nav=100.0, date=_yesterday()),
         _row(ts=f"{_yesterday()}T18:00:00+00:00", event="peak_update", nav=100.0)],
        current_nav=80.0,
        note="THE SEVERITY DRIVER. Reachable every UTC day with no fault at all.",
    )

    # ── A/B: legacy row, no `date`, unparseable `ts` -> sod_date stays None ────
    results["AB"] = _measure(
        "A/B -- LEGACY ROW (no `date` key, unparseable `ts`)",
        [_row(ts="not-a-timestamp", event="sod_snapshot", nav=100.0),
         _row(ts=f"{_today()}T00:00:00+00:00", event="peak_update", nav=100.0)],
        current_nav=80.0,
        note="Mechanism is live; no row of this shape exists in the live journal.",
    )

    # ── F: startup before any anchor -> MISSING, deliberately not "stale" ──────
    results["F"] = _measure(
        "F -- STARTUP, no anchor has ever been written",
        [_row(ts=f"{_today()}T00:00:00+00:00", event="peak_update", nav=100.0)],
        current_nav=80.0,
        note="Absence is named `missing`, NOT `stale` (kill_switch.py:922-923).",
    )

    # ── E: oversized int -> OverflowError aborts the WHOLE replay ─────────────
    results["E"] = _measure(
        "E -- OVERSIZED INT aborts the entire audit replay (NEW, found by the gate)",
        # ORDER MATTERS and my first construction got it wrong: with the
        # oversized row LAST, the good rows have already applied and the abort
        # strands nothing -- the run printed armed=True and both legs firing,
        # contradicting the claim. The row has to come FIRST (or, as here, be the
        # first row in `ts` order) for the abort to swallow the rest.
        ['{"ts": "%sT00:00:00+00:00", "event": "peak_update", "nav": %s}'
         % (_today(), "1" + "0" * 400),
         _row(ts=f"{_today()}T00:01:00+00:00", event="sod_snapshot",
              nav=100.0, date=_today()),
         _row(ts=f"{_today()}T00:02:00+00:00", event="peak_update", nav=100.0)],
        current_nav=80.0,
        note="_coerce_nav catches only (TypeError, ValueError); OverflowError is "
             "swallowed at :394 and aborts the replay, so every LATER row is lost.",
    )

    # ── HEALTHY control: the guard must still fire when it CAN evaluate ───────
    results["HEALTHY"] = _measure(
        "HEALTHY CONTROL -- same-day anchor, a real 20% drawdown",
        [_row(ts=f"{_today()}T00:00:00+00:00", event="sod_snapshot",
              nav=100.0, date=_today()),
         _row(ts=f"{_today()}T00:00:00+00:00", event="peak_update", nav=100.0)],
        current_nav=80.0,
        note="If this does NOT breach on both legs, the measurement above is "
             "meaningless -- the guard would be broken for every state.",
    )

    print(f"\n{'='*76}\nVERDICT")
    c, ab, f, e, h = (results[k]["breach"] for k in ("C", "AB", "F", "E", "HEALTHY"))
    print(f"  C  UTC rollover     : sod_date STALE, daily leg {'DISARMED' if not c['daily_loss_breached'] else 'fires'}, "
          f"any_breached={c['any_breached']}  <-- reachable EVERY DAY")
    print(f"  A/B legacy row      : sod_date={results['AB']['snapshot'].get('sod_date')!r}, "
          f"any_breached={ab['any_breached']}")
    print(f"  F  startup          : missing={f['daily_baseline_missing']} "
          f"stale={f['daily_baseline_stale']}, any_breached={f['any_breached']}")
    e_snap = results["E"]["snapshot"]
    e_stranded = e_snap.get("sod_nav") is None and e_snap.get("peak_nav") is None
    print(f"  E  oversized int    : sod_nav={e_snap.get('sod_nav')!r} "
          f"peak={e_snap.get('peak_nav')!r}, any_breached={e['any_breached']}"
          f"  <-- {'BOTH legs stranded' if e_stranded else 'legs survived (rows before the bad one still applied)'}")
    print(f"  HEALTHY control     : daily={h['daily_loss_breached']} "
          f"trailing={h['trailing_dd_breached']} any={h['any_breached']}")

    print("\n  ANSWER TO CRITERION 1: YES -- production CAN reach a None/stale")
    print("  sod_date, and case C needs no fault at all. Exposure is bounded to")
    print("  drawdowns in [daily_limit, trailing_limit) because the trailing leg")
    if e_stranded:
        print("  is date-independent -- EXCEPT in case E, where the same fault strands")
        print("  the peak too and NOTHING fires for any drawdown.")
    else:
        print("  is date-independent. Case E aborts the replay, but only rows AFTER")
        print("  the malformed one are lost -- see the case-E block for what survived.")
    if not h["any_breached"]:
        print("\n  *** CONTROL FAILED -- do not trust the rows above. ***")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
