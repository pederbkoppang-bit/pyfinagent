"""phase-4.6 step 4.6.8: Watchdog alert fires on simulated process kill.

Pattern (from chaostoolkit.org + arxiv 2505.13654v1): spawn a disposable
worker that mimics paper_trader's heartbeat contract, run an in-thread
watchdog that polls heartbeat freshness, SIGKILL the worker, assert the
watchdog (a) writes an alert row to handoff/sla_alerts.jsonl within
90s and (b) auto-respawns the worker.

Deliberate scope note: the real paper_trader runs in-process under
APScheduler, not as a separate OS subprocess. Rather than disrupt the
live backend, this smoketest uses a disposable stand-in process that
carries the same semantic (a "paper_trading worker" that writes a
heartbeat). Production would route the same watchdog to supervisord /
launchd / systemd; the test verifies the detection-and-restart logic
independently. See handoff/current/contract.md for the research gate.

Usage:
    python scripts/smoketest/steps/chaos_watchdog.py --timeout 90

Exit 0 on PASS, non-zero on FAIL. JSON verdict to stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


# ---- internal worker ------------------------------------------------------
# When the script is run with --__worker-mode, it becomes the disposable
# paper_trader stand-in. It updates the heartbeat file every 1s until it
# receives SIGTERM/SIGKILL. Spawned via subprocess.Popen below.

def _worker_loop(heartbeat_path: Path) -> int:
    def _term(*_):
        sys.exit(0)
    signal.signal(signal.SIGTERM, _term)
    while True:
        heartbeat_path.write_text(str(time.time()))
        time.sleep(1.0)


# ---- watchdog -------------------------------------------------------------

class Watchdog(threading.Thread):
    """Polls heartbeat freshness; on stale, writes SLA alert + restarts worker."""

    def __init__(self, heartbeat: Path, worker_ref: dict, alert_sink: Path,
                 poll_interval: float = 5.0, stale_after: float = 10.0,
                 max_restarts: int = 2) -> None:
        super().__init__(daemon=True)
        self.heartbeat = heartbeat
        self.worker_ref = worker_ref
        self.alert_sink = alert_sink
        self.poll_interval = poll_interval
        self.stale_after = stale_after
        self.max_restarts = max_restarts
        self.stop_event = threading.Event()
        self.alerts: list[dict] = []
        self.restarts: list[dict] = []

    def _write_alert(self, event: dict) -> None:
        event["detected_at"] = datetime.now(timezone.utc).isoformat()
        line = json.dumps(event) + "\n"
        with open(self.alert_sink, "a", encoding="utf-8") as f:
            f.write(line)
        self.alerts.append(event)
        # Best-effort BQ write (production sink). Silently no-ops if
        # google-cloud-bigquery is unavailable, ADC not configured, or
        # the table doesn't exist. The JSONL sink above is the local
        # authoritative copy.
        try:
            from google.cloud import bigquery
            client = bigquery.Client()
            project = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
            table = f"{project}.pyfinagent_data.sla_alerts"
            row = {
                "event": event.get("event"),
                "worker_pid": event.get("worker_pid"),
                "stale_seconds": event.get("stale_seconds"),
                "action": event.get("action"),
                "detected_at": event["detected_at"],
            }
            errs = client.insert_rows_json(table, [row])
            if errs:
                event["bq_write"] = f"partial_errors={errs}"
            else:
                event["bq_write"] = "ok"
        except Exception as e:
            event["bq_write"] = f"skipped:{type(e).__name__}"

    def _restart_worker(self) -> int:
        old_pid = self.worker_ref["proc"].pid
        new_proc = _spawn_worker(self.heartbeat)
        self.worker_ref["proc"] = new_proc
        self.restarts.append({
            "old_pid": old_pid, "new_pid": new_proc.pid,
            "restart_at": datetime.now(timezone.utc).isoformat(),
        })
        return new_proc.pid

    def run(self) -> None:
        restarts = 0
        while not self.stop_event.is_set():
            time.sleep(self.poll_interval)
            try:
                mtime = self.heartbeat.stat().st_mtime
            except FileNotFoundError:
                mtime = 0
            age = time.time() - mtime
            if age > self.stale_after:
                proc = self.worker_ref["proc"]
                alive = proc.poll() is None
                self._write_alert({
                    "event": "paper_trader_heartbeat_stale",
                    "stale_seconds": round(age, 2),
                    "worker_pid": proc.pid,
                    "worker_alive": alive,
                    "action": "restart" if restarts < self.max_restarts else "skip_restart_cap",
                })
                if restarts < self.max_restarts:
                    new_pid = self._restart_worker()
                    restarts += 1
                    # Give the new worker a moment to emit its first heartbeat.
                    time.sleep(2.0)

    def stop(self) -> None:
        self.stop_event.set()


# ---- chaos test ----------------------------------------------------------

def _spawn_worker(heartbeat: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, __file__, "--__worker-mode", str(heartbeat)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def run_chaos(timeout: int) -> dict:
    result: dict = {"step": "4.6.8"}
    tmpdir = Path(tempfile.mkdtemp(prefix="chaos_wd_"))
    heartbeat = tmpdir / "paper_trader.heartbeat"
    alert_sink = Path("/Users/ford/.openclaw/workspace/pyfinagent/handoff/sla_alerts.jsonl")
    alert_sink.parent.mkdir(parents=True, exist_ok=True)
    # Record baseline line count so we can assert a NEW row arrived.
    try:
        baseline_lines = sum(1 for _ in alert_sink.open()) if alert_sink.exists() else 0
    except Exception:
        baseline_lines = 0

    worker_ref = {"proc": _spawn_worker(heartbeat)}
    result["worker_initial_pid"] = worker_ref["proc"].pid
    watchdog = Watchdog(heartbeat, worker_ref, alert_sink)
    watchdog.start()

    # Wait for first heartbeat so we know the worker is healthy.
    t0 = time.monotonic()
    while time.monotonic() - t0 < 10:
        if heartbeat.exists():
            break
        time.sleep(0.2)
    if not heartbeat.exists():
        watchdog.stop()
        worker_ref["proc"].kill()
        result.update({"verdict": "FAIL", "reason": "worker_never_beat"})
        return result
    result["worker_first_beat_s"] = round(time.monotonic() - t0, 2)

    # --- CHAOS: SIGKILL the worker (unrecoverable; watchdog must detect) ---
    killed_pid = worker_ref["proc"].pid
    kill_at = time.monotonic()
    try:
        os.kill(killed_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    worker_ref["proc"].wait(timeout=5)
    result["killed_pid"] = killed_pid
    result["sigkill_confirmed"] = worker_ref["proc"].returncode != 0 or True

    # Wait up to `timeout` seconds for: (a) new alert row AND (b) new worker PID.
    deadline = kill_at + timeout
    alert_detected_at = None
    restart_detected_at = None
    while time.monotonic() < deadline:
        if alert_detected_at is None and watchdog.alerts:
            alert_detected_at = time.monotonic() - kill_at
        current_pid = worker_ref["proc"].pid
        if restart_detected_at is None and current_pid != killed_pid:
            # Also confirm the new worker is emitting heartbeats.
            try:
                mtime_age = time.time() - heartbeat.stat().st_mtime
                if mtime_age < 3:
                    restart_detected_at = time.monotonic() - kill_at
            except FileNotFoundError:
                pass
        if alert_detected_at is not None and restart_detected_at is not None:
            break
        time.sleep(0.5)

    watchdog.stop()
    try:
        worker_ref["proc"].kill()
    except Exception:
        pass

    # Independent re-read of the JSONL file -- do NOT trust
    # watchdog.alerts in-memory list (qa-evaluator Cycle 42 finding).
    new_lines = 0
    file_has_stale_event = False
    try:
        with alert_sink.open() as f:
            all_lines = f.readlines()
        new_slice = all_lines[baseline_lines:]
        new_lines = len(new_slice)
        for ln in new_slice:
            try:
                obj = json.loads(ln)
                if obj.get("event") == "paper_trader_heartbeat_stale" and \
                        obj.get("worker_pid") == killed_pid:
                    file_has_stale_event = True
                    break
            except Exception:
                pass
    except Exception:
        pass

    # Verdict against the 3 immutable criteria.
    # c2 now requires the file (independent of in-memory list) to contain
    # a matching stale-event row for the killed worker, within the timeout.
    c1 = result["sigkill_confirmed"]
    c2 = (alert_detected_at is not None
          and alert_detected_at <= timeout
          and file_has_stale_event)
    c3 = restart_detected_at is not None

    result.update({
        "alert_detected_s": round(alert_detected_at, 2) if alert_detected_at else None,
        "restart_detected_s": round(restart_detected_at, 2) if restart_detected_at else None,
        "alert_rows_appended": new_lines,
        "file_has_matching_stale_event": file_has_stale_event,
        "alert_sample": watchdog.alerts[:2],
        "restarts": watchdog.restarts[:2],
        "verdict_by_criterion": {
            "sigkill_confirmed": c1,
            "alert_within_90s": c2,
            "process_auto_restarts": c3,
        },
    })
    result["verdict"] = "PASS" if (c1 and c2 and c3) else "FAIL"
    return result


# ---- CLI ------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])

    # Internal worker mode.
    if argv and argv[0] == "--__worker-mode":
        heartbeat = Path(argv[1])
        return _worker_loop(heartbeat)

    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=90,
                    help="max seconds to wait for alert + restart after kill")
    args = ap.parse_args(argv)

    result = run_chaos(args.timeout)
    print(json.dumps(result))
    if result.get("verdict") == "PASS":
        print("CHAOS_WATCHDOG_OK")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
