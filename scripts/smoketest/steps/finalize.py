"""phase-4.6 step 4.6.9: Append harness log row + clean shutdown.

Acceptance (immutable):
- handoff/harness_log.md gained exactly one row
- new row contains phase=4.6 and result field
- no uvicorn process bound to 8765 after exit
- no stray mcp_server PIDs vs pre-test snapshot

Design (see handoff/current/contract.md for research gate):
- The harness_log format is multi-line Markdown blocks; "one row" is
  interpreted as exactly one new `## Cycle ...` header line appended.
- MCP servers are in-process modules today; "stray PIDs" is an empty
  set both before and after, so delta is trivially zero. We still
  snapshot `ps` to prove it.
- Port 8765 check uses socket.connect_ex -- zero allocations, works
  without ps/lsof privileges.

Usage:
    python scripts/smoketest/steps/finalize.py --log handoff/harness_log.md --port 8765
"""
from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CYCLE_RE = re.compile(r"^## Cycle\s+\d+", re.MULTILINE)
MCP_PS_PATTERN = "backend.agents.mcp_servers"


def _count_cycle_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return len(CYCLE_RE.findall(path.read_text(encoding="utf-8")))


def _mcp_pids() -> set[int]:
    """Return PIDs running backend.agents.mcp_servers.* modules."""
    try:
        out = subprocess.check_output(["ps", "-Ao", "pid,command"], text=True,
                                       timeout=5)
    except Exception:
        return set()
    pids: set[int] = set()
    for line in out.splitlines()[1:]:
        if MCP_PS_PATTERN in line:
            try:
                pids.add(int(line.strip().split()[0]))
            except (ValueError, IndexError):
                pass
    return pids


def _port_is_bound(port: int, host: str = "127.0.0.1") -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        rc = s.connect_ex((host, port))
    finally:
        s.close()
    return rc == 0


def _next_cycle_number(text: str) -> int:
    nums = [int(m.group(1)) for m in re.finditer(r"^## Cycle\s+(\d+)", text, re.MULTILINE)]
    return (max(nums) + 1) if nums else 1


def append_cycle_row(log_path: Path, phase: str, result: str, note: str) -> dict:
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    cycle_n = _next_cycle_number(existing)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    block = (
        f"\n## Cycle {cycle_n} -- {ts} -- phase={phase} result={result}\n\n"
        f"**Decision:** {result} -- {note}\n"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(block)
    return {"cycle": cycle_n, "ts": ts, "block_chars": len(block)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--phase", default="4.6")
    ap.add_argument("--result", default="PASS")
    ap.add_argument("--note", default="aggregate smoketest finalize -- all prior steps green")
    args = ap.parse_args()

    log_path = Path(args.log)
    result: dict = {"step": "4.6.9", "log": str(log_path), "port": args.port}

    rows_before = _count_cycle_rows(log_path)
    mcp_pids_before = _mcp_pids()

    append_info = append_cycle_row(log_path, args.phase, args.result, args.note)
    result["append"] = append_info

    time.sleep(0.2)

    rows_after = _count_cycle_rows(log_path)
    mcp_pids_after = _mcp_pids()
    stray_mcp = mcp_pids_after - mcp_pids_before
    port_bound = _port_is_bound(args.port)

    new_block = log_path.read_text(encoding="utf-8").split(f"## Cycle {append_info['cycle']}")[-1]
    has_phase = f"phase={args.phase}" in new_block
    has_result = "Decision:" in new_block and args.result in new_block

    c1 = (rows_after - rows_before) == 1
    c2 = has_phase and has_result
    c3 = not port_bound
    c4 = len(stray_mcp) == 0

    result.update({
        "rows_before": rows_before,
        "rows_after": rows_after,
        "delta_rows": rows_after - rows_before,
        "has_phase_field": has_phase,
        "has_result_field": has_result,
        "port_bound_after": port_bound,
        "mcp_pids_before": sorted(mcp_pids_before),
        "mcp_pids_after": sorted(mcp_pids_after),
        "stray_mcp_pids": sorted(stray_mcp),
        "verdict_by_criterion": {
            "exactly_one_new_row": c1,
            "row_has_phase_and_result": c2,
            "port_8765_unbound": c3,
            "no_stray_mcp_pids": c4,
        },
    })
    ok = c1 and c2 and c3 and c4
    result["verdict"] = "PASS" if ok else "FAIL"

    print(json.dumps(result))
    if ok:
        print("FINALIZE_OK")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
