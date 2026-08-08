#!/usr/bin/env python3
"""phase-85.4 C1/C2 -- measure the autonomous cycle's ANALYSIS phase from the
orchestrator's own logs, so the 7200s budget can be judged against a number
instead of an opinion.

Answers, per cycle window found in the log:

  * when the analysis phase started (`Paper trading: Step 3 -- Analyzing ...`)
    and when it ended (`Paper trading: Step 5 -- Mark to market`, or the
    cycle's terminal line if it never got there),
  * how many tickers were dispatched and how many finished,
  * per-ticker wall-clock, from `Orchestrator pre-dispatch ticker=X` to
    `Lite analysis persisted to analysis_results for X`,
  * the serial ticker-seconds, the observed effective parallelism, and the
    wall-clock the phase WOULD have needed had every dispatched ticker
    finished at the cycle's own mean per-ticker cost,
  * the cc_rail subprocess-call latency distribution in the window, including
    how many "successes" landed within 5s of the 150s subprocess cap (a
    truncated distribution means the cap is censoring, not protecting).

Read-only. Touches no service, starts no cycle. Safe to run at any time.

Usage:
    python scripts/diagnostics/measure_analysis_phase.py [--log backend.log] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

_TS_FMT = "%Y-%m-%d %H:%M:%S,%f"

_RE_PREDISPATCH = re.compile(r"Orchestrator pre-dispatch ticker=([A-Z0-9._-]+)")
_RE_PERSISTED = re.compile(r"analysis persisted to analysis_results for ([A-Z0-9._-]+)")
_RE_STEP3 = re.compile(r"Step 3 -- Analyzing (\d+) new \+ (\d+) re-evals")
_RE_CONCURRENCY = re.compile(r"per-provider concurrency cap = (\d+)")
_RE_RAIL_START = re.compile(r"claude_code_invoke: args=\d+ prompt_len=(\d+) timeout_s=(\d+)")
_RE_RAIL_TIMEOUT = re.compile(r"claude_code_invoke: subprocess timeout after (\d+)s")
_RE_RAIL_OK = re.compile(r"claude_code_invoke: ok .*?(\d+(?:\.\d+)?)s")


def _parse_ts(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(s, _TS_FMT)
    except Exception:
        return None


def _pct(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def load_events(log_path: Path) -> list[dict[str, Any]]:
    """Yield {ts, module, message} for every parseable JSON log line."""
    events: list[dict[str, Any]] = []
    with log_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            ts = _parse_ts(str(row.get("timestamp", "")))
            if ts is None:
                continue
            events.append({"ts": ts, "module": row.get("module"), "message": str(row.get("message", ""))})
    return events


def split_cycles(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cut the event stream into cycle windows on `Step 1 -- Screening universe`."""
    cycles: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None
    for ev in events:
        msg = ev["message"]
        if ev["module"] == "autonomous_loop" and "Step 1 -- Screening universe" in msg:
            if current is not None:
                cycles.append(current)
            current = {"started_at": ev["ts"], "events": [], "terminal": None, "terminal_at": None}
        if current is None:
            continue
        current["events"].append(ev)
        if ev["module"] == "autonomous_loop":
            if "cycle TIMED OUT" in msg:
                current["terminal"], current["terminal_at"] = "timeout", ev["ts"]
            elif "cycle complete" in msg:
                current["terminal"], current["terminal_at"] = "completed", ev["ts"]
            elif "cycle failed" in msg:
                current["terminal"], current["terminal_at"] = "error", ev["ts"]
    if current is not None:
        cycles.append(current)
    return cycles


def measure_cycle(cycle: dict[str, Any]) -> dict[str, Any]:
    events = cycle["events"]
    analysis_start: Optional[datetime] = None
    analysis_end: Optional[datetime] = None
    analysis_end_reason = "never_reached"
    n_new = n_reeval = 0
    concurrency: Optional[int] = None
    dispatch: dict[str, datetime] = {}
    finish: dict[str, datetime] = {}
    rail_timeouts: list[int] = []
    rail_starts = 0
    rail_timeout_s: Optional[int] = None

    for ev in events:
        msg, mod = ev["message"], ev["module"]
        if mod == "autonomous_loop":
            m = _RE_STEP3.search(msg)
            if m and analysis_start is None:
                analysis_start = ev["ts"]
                n_new, n_reeval = int(m.group(1)), int(m.group(2))
            m = _RE_CONCURRENCY.search(msg)
            if m and concurrency is None:
                concurrency = int(m.group(1))
            m = _RE_PREDISPATCH.search(msg)
            if m:
                dispatch.setdefault(m.group(1), ev["ts"])
            m = _RE_PERSISTED.search(msg)
            if m:
                finish[m.group(1)] = ev["ts"]
            if "Step 5 -- Mark to market" in msg and analysis_end is None:
                analysis_end, analysis_end_reason = ev["ts"], "reached_mark_to_market"
        elif mod == "claude_code_client":
            m = _RE_RAIL_START.search(msg)
            if m:
                rail_starts += 1
                rail_timeout_s = int(m.group(2))
            m = _RE_RAIL_TIMEOUT.search(msg)
            if m:
                rail_timeouts.append(int(m.group(1)))

    if analysis_end is None and cycle["terminal_at"] is not None:
        analysis_end, analysis_end_reason = cycle["terminal_at"], f"cycle_{cycle['terminal']}"

    # Per-ticker wall-clock (only tickers that both dispatched AND finished).
    per_ticker: dict[str, float] = {}
    for tkr, t0 in dispatch.items():
        t1 = finish.get(tkr)
        if t1 is not None and t1 >= t0:
            per_ticker[tkr] = (t1 - t0).total_seconds()
    unfinished = sorted(set(dispatch) - set(per_ticker))

    serial_sec = sum(per_ticker.values())
    phase_sec = (
        (analysis_end - analysis_start).total_seconds()
        if analysis_start and analysis_end
        else None
    )
    # Effective parallelism = serial ticker-seconds actually retired / wall-clock
    # spent retiring them. Under-counts when tickers are still in flight at the
    # cut, which is exactly the timeout case -- so it is a LOWER bound there.
    eff_par = (serial_sec / phase_sec) if phase_sec and phase_sec > 0 else None
    mean_ticker = statistics.mean(per_ticker.values()) if per_ticker else None

    # Projection: what the phase would have cost had every DISPATCHED ticker
    # finished at this cycle's own mean per-ticker wall-clock, at the observed
    # effective parallelism.
    projected_phase_sec = None
    if mean_ticker and eff_par and eff_par > 0:
        projected_phase_sec = (mean_ticker * len(dispatch)) / eff_par

    screening_sec = (
        (analysis_start - cycle["started_at"]).total_seconds() if analysis_start else None
    )
    projected_cycle_sec = (
        screening_sec + projected_phase_sec
        if screening_sec is not None and projected_phase_sec is not None
        else None
    )

    return {
        "cycle_started_at": cycle["started_at"].isoformat(sep=" "),
        "terminal_status": cycle["terminal"],
        "terminal_at": cycle["terminal_at"].isoformat(sep=" ") if cycle["terminal_at"] else None,
        "cycle_wall_sec": (
            (cycle["terminal_at"] - cycle["started_at"]).total_seconds()
            if cycle["terminal_at"]
            else None
        ),
        "screening_sec": screening_sec,
        "analysis_start": analysis_start.isoformat(sep=" ") if analysis_start else None,
        "analysis_end": analysis_end.isoformat(sep=" ") if analysis_end else None,
        "analysis_end_reason": analysis_end_reason,
        "analysis_phase_sec": phase_sec,
        "tickers_planned": n_new + n_reeval,
        "tickers_dispatched": len(dispatch),
        "tickers_finished": len(per_ticker),
        "tickers_unfinished": unfinished,
        "concurrency_cap": concurrency,
        "per_ticker_wall_sec": {k: round(v, 1) for k, v in sorted(per_ticker.items())},
        "per_ticker_mean_sec": round(mean_ticker, 1) if mean_ticker else None,
        "per_ticker_median_sec": (
            round(statistics.median(per_ticker.values()), 1) if per_ticker else None
        ),
        "serial_ticker_sec": round(serial_sec, 1),
        "effective_parallelism": round(eff_par, 2) if eff_par else None,
        "projected_analysis_sec_if_all_finished": (
            round(projected_phase_sec) if projected_phase_sec else None
        ),
        "projected_cycle_sec_if_all_finished": (
            round(projected_cycle_sec) if projected_cycle_sec else None
        ),
        "rail_calls_started": rail_starts,
        "rail_calls_timed_out": len(rail_timeouts),
        "rail_timeout_rate": (
            round(len(rail_timeouts) / rail_starts, 4) if rail_starts else None
        ),
        "rail_subprocess_timeout_s": rail_timeout_s,
    }


def measure_agent_latency(events: list[dict[str, Any]], t0: datetime, t1: datetime) -> dict[str, Any]:
    """Per-agent latency from the `trace` module inside [t0, t1].

    The Trace lines carry the ACTUAL per-agent latency in ms, which is the only
    place a rail call's *successful* duration is recorded -- the claude_code_client
    lines record a duration only on TIMEOUT.
    """
    lat: list[float] = []
    for ev in events:
        if ev["module"] != "trace" or not (t0 <= ev["ts"] <= t1):
            continue
        m = re.search(r"latency=(\d+)ms", ev["message"])
        if m:
            lat.append(int(m.group(1)) / 1000.0)
    if not lat:
        return {"n": 0}
    cap = 150.0
    return {
        "n": len(lat),
        "min_s": round(min(lat), 1),
        "median_s": round(statistics.median(lat), 1),
        "p90_s": round(_pct(lat, 0.90) or 0, 1),
        "max_s": round(max(lat), 1),
        "n_within_5s_of_150s_cap": sum(1 for v in lat if cap - 5 <= v <= cap),
        "n_at_or_above_cap": sum(1 for v in lat if v >= cap),
        "pct_within_5s_of_cap": round(
            100.0 * sum(1 for v in lat if cap - 5 <= v <= cap) / len(lat), 1
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="backend.log")
    ap.add_argument("--json", default=None, help="write the full measurement as JSON here")
    ap.add_argument("--budget-sec", type=float, default=7200.0)
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"FATAL: log not found: {log_path}", file=sys.stderr)
        return 2

    events = load_events(log_path)
    cycles = split_cycles(events)
    # Only cycles that actually reached the analysis phase are measurable.
    measured = []
    for c in cycles:
        m = measure_cycle(c)
        if m["analysis_start"] is None:
            continue
        t0 = _parse_ts(m["analysis_start"] + "000")
        t1 = _parse_ts((m["analysis_end"] or m["analysis_start"]) + "000")
        if t0 and t1:
            m["agent_latency_in_analysis_window"] = measure_agent_latency(events, t0, t1)
        measured.append(m)

    if not measured:
        print("No cycle in this log reached the analysis phase -- nothing to measure.")
        return 1

    print(f"log={log_path}  lines_parsed={len(events)}  cycles_with_analysis_phase={len(measured)}")
    print(f"budget_sec={args.budget_sec:.0f}\n")
    for m in measured:
        print("=" * 78)
        print(f"CYCLE  started={m['cycle_started_at']}  terminal={m['terminal_status']}"
              f"  wall={m['cycle_wall_sec']}s")
        print(f"  screening        : {m['screening_sec']}s")
        print(f"  analysis phase   : {m['analysis_phase_sec']}s  (end reason: {m['analysis_end_reason']})")
        print(f"  tickers          : planned={m['tickers_planned']} dispatched={m['tickers_dispatched']}"
              f" finished={m['tickers_finished']} unfinished={m['tickers_unfinished']}")
        print(f"  concurrency cap  : {m['concurrency_cap']}")
        print(f"  per-ticker wall  : {m['per_ticker_wall_sec']}")
        print(f"  per-ticker mean  : {m['per_ticker_mean_sec']}s  median={m['per_ticker_median_sec']}s")
        print(f"  serial ticker-s  : {m['serial_ticker_sec']}s   effective parallelism={m['effective_parallelism']}")
        print(f"  PROJECTED analysis if all dispatched tickers finished : "
              f"{m['projected_analysis_sec_if_all_finished']}s")
        print(f"  PROJECTED cycle   (screening + analysis)              : "
              f"{m['projected_cycle_sec_if_all_finished']}s  vs budget {args.budget_sec:.0f}s")
        pc = m["projected_cycle_sec_if_all_finished"]
        if pc is not None:
            verdict = "OVER BUDGET" if pc > args.budget_sec else "within budget"
            print(f"  VERDICT          : {verdict} (delta {pc - args.budget_sec:+.0f}s)")
        print(f"  cc_rail calls    : started={m['rail_calls_started']} timed_out={m['rail_calls_timed_out']}"
              f" rate={m['rail_timeout_rate']} subprocess_timeout_s={m['rail_subprocess_timeout_s']}")
        print(f"  agent latency    : {m.get('agent_latency_in_analysis_window')}")

    if args.json:
        Path(args.json).write_text(json.dumps(measured, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
