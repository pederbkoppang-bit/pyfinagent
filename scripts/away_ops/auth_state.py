#!/usr/bin/env python3
"""phase-85.3: auth-state derivation seam for the away-ops watchdog.

Extracted from the healthcheck.sh python heredoc so the logic is testable
(criterion 1). Prints exactly two tokens to stdout -- "<auth_ok> <detail>"
-- preserving the call-site contract; the caller keeps its
`|| echo "unknown probe_error"` fail-open wrapper.

WHY THIS EXISTS (the 28-day false alarm): the previous derivation latched
auth_ok=false on ANY 401 in the newest session file newer than cleared_at.
With cleared_at never written (the clear arm was unreachable -- the
auth_ok=false branch always preceded it) and session production paused by
the open incident, a 2026-07-10 401 held the alarm ok:false every 30
minutes for weeks about a credential that `claude auth status` verified
healthy on every run. Alarm-fatigue literature (AHRQ Making Healthcare
Safer III: 72-99% false-alarm rates drive desensitization) says such an
alarm is worse than none.

THE THREE LEGS (contract D3):
  L1  EVIDENCE MAX-AGE: a 401 older than --window-s (default 129600s = 36h)
      no longer holds the latch. Derivation of 36h: away sessions start
      07:30 and 22:00 local, worst-case inter-start gap 14.5h; session files
      are written at session END with a 4h cap => worst-case normal mtime
      gap ~18.5h; 36h ~= 2x tolerates one fully missed slot without
      flapping. (Nagios freshness practice: explicit thresholds, stale
      evidence forces an ACTIVE check -- which is leg L2 here. The 26h
      Nagios worked example was considered and rejected: only ~7.5h
      headroom over the normal gap.)
  L2  ACTIVE RE-CHECK: `claude auth status` rc (passed by the caller as
      --status-ok) remains an independent, un-aged failure leg -- a dead
      credential fails THIS leg regardless of session-file staleness.
  L3  THE LATCH STAYS EXPLICIT (Yokogawa latch semantics: never
      self-resolving from the paging side): --apply performs the clear
      transition ONLY on a strictly-healthy derivation ("true", never
      "unknown" -- the previous elif arm cleared on probe errors too,
      which could close a REAL incident on a transient failure; fixed
      here, DD-5).

Fail-open (criterion 6): any unexpected error prints "unknown probe_error"
and exits 0 -- the watchdog must never manufacture a P1 out of its own
error.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os

DEFAULT_WINDOW_S = 129600  # 36h -- see module docstring for the derivation


def derive(ops: str, state_path: str, status_ok: bool, now: datetime.datetime,
           window_s: int) -> tuple[str, str]:
    """Return (auth_ok, detail) as the two stdout tokens."""
    ok, detail = True, "ok"
    if not status_ok:
        ok, detail = False, "auth_status_rc_nonzero"
    cleared_at = None
    try:
        st = json.load(open(state_path))
        if st.get("cleared_at"):
            cleared_at = datetime.datetime.fromisoformat(st["cleared_at"])
    except Exception:
        pass
    sessions = sorted(glob.glob(os.path.join(ops, "session_*.json")),
                      key=os.path.getmtime)
    if sessions and ok:
        newest = sessions[-1]
        try:
            body = open(newest, encoding="utf-8", errors="replace").read()
        except Exception:
            body = ""
        if '"api_error_status": 401' in body or '"api_error_status":401' in body:
            mt = datetime.datetime.fromtimestamp(os.path.getmtime(newest),
                                                 datetime.timezone.utc)
            age_s = (now - mt).total_seconds()
            fresh = age_s <= window_s
            after_clear = cleared_at is None or mt > cleared_at
            if fresh and after_clear:
                ok, detail = False, "401_in_" + os.path.basename(newest)
            elif not fresh and after_clear:
                # stale evidence: does NOT hold the latch (L1); leg L2 has
                # already vouched for the credential this run.
                detail = "stale_401_ignored_" + os.path.basename(newest)
    return ("true" if ok else "false"), detail


def apply_transition(state_path: str, auth_ok: str, now: datetime.datetime) -> None:
    """The latch CLEAR transition (the only writer on the healthy side).

    Fires ONLY on a strictly-healthy derivation -- never on "unknown"
    (DD-5: a probe error must not close a real incident). Opening the
    latch stays with the caller's paging arm, unchanged."""
    if auth_ok != "true":
        return
    try:
        st = json.load(open(state_path))
    except Exception:
        return
    if st.get("incident_open"):
        json.dump(
            {"incident_open": False,
             "cleared_at": now.isoformat(),
             "cleared_by": "healthcheck_healthy"},
            open(state_path, "w"),
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--status-ok", required=True, choices=["true", "false"])
    ap.add_argument("--now", default=None,
                    help="ISO timestamp for tests; default = real UTC now")
    ap.add_argument("--window-s", type=int, default=DEFAULT_WINDOW_S)
    ap.add_argument("--apply", action="store_true",
                    help="perform the latch clear transition on healthy")
    args = ap.parse_args()
    now = (datetime.datetime.fromisoformat(args.now)
           if args.now else datetime.datetime.now(datetime.timezone.utc))
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)
    auth_ok, detail = derive(args.ops, args.state, args.status_ok == "true",
                             now, args.window_s)
    if args.apply:
        apply_transition(args.state, auth_ok, now)
    print(auth_ok, detail)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        # fail-open (criterion 6): the watchdog must never manufacture a P1
        # out of this seam's own error.
        print("unknown probe_error")
        raise SystemExit(0)
