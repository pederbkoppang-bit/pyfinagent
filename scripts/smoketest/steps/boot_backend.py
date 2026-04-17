"""phase-4.6 step 4.6.1: boot backend, hit /api/health, assert mcp_servers OK.

Usage:
    python scripts/smoketest/steps/boot_backend.py --port 8765 --timeout 30

Exits 0 on PASS, non-zero on FAIL. Emits JSON to stdout in both cases.

Design rationale (evidence from handoff/current/contract.md research gate):
- Out-of-process uvicorn subprocess: matches LSST Safir spawn_uvicorn pattern;
  exercises real middleware + lifespan that TestClient would bypass.
- DEVNULL for stdout/stderr: prevents 64KB macOS pipe-buffer deadlock when
  caller never drains. uvicorn --log-level warning emits little anyway.
- Early exit on proc.poll() != None: if uvicorn dies at startup (import error,
  port in use), no point polling for 30s.
- time.monotonic(): NTP-safe latency measurement.
- subprocess.TimeoutExpired-only catch in finally: don't swallow KeyboardInterrupt.
- 127.0.0.1 not localhost: avoids macOS ::1-first IPv6 resolution stall.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


def _poll_health(proc: subprocess.Popen, port: int, deadline: float) -> tuple[dict | None, float | None]:
    url = f"http://127.0.0.1:{port}/api/health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return None, None  # uvicorn died; stop polling
        try:
            t0 = time.monotonic()
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    body = json.loads(resp.read().decode("utf-8"))
                    return body, time.monotonic() - t0
        except (urllib.error.URLError, ConnectionRefusedError, OSError, TimeoutError):
            pass
        time.sleep(0.5)
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--timeout", type=int, default=30)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app",
         "--port", str(args.port), "--host", "127.0.0.1",
         "--log-level", "warning"],
        cwd=str(repo),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    result: dict = {"step": "4.6.1", "port": args.port}
    try:
        deadline = time.monotonic() + args.timeout
        body, latency = _poll_health(proc, args.port, deadline)
        if body is None:
            exit_code = proc.poll()
            reason = "backend_exited_early" if exit_code is not None else "health_timeout"
            result.update({"verdict": "FAIL", "reason": reason, "backend_exit_code": exit_code})
            print(json.dumps(result))
            return 1

        mcp = body.get("mcp_servers", {}) or {}
        missing = [n for n in ("data", "backtest", "signals") if mcp.get(n, {}).get("status") != "ok"]
        if missing:
            result.update({"verdict": "FAIL", "reason": f"mcp_servers_not_ok:{missing}",
                           "health_body": body, "latency_s": latency})
            print(json.dumps(result))
            return 1
        if latency > 5:
            result.update({"verdict": "FAIL", "reason": "latency_over_5s",
                           "latency_s": latency, "health_body": body})
            print(json.dumps(result))
            return 1

        result.update({"verdict": "PASS", "latency_s": latency, "health_body": body})
        print(json.dumps(result))
        print("BOOT_BACKEND_OK")
        return 0
    finally:
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
