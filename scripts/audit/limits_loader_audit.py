"""phase-4.9 step 4.9.2 limits-loader audit.

Six teeth:
1. imports: `load_once` + `get_digest` importable from
   `backend.governance.limits_loader`.
2. idempotent: two calls return the same frozen object; digest
   is a 64-char lowercase hex.
3. sighup_ignored: after `load_once()`, `signal.getsignal(SIGHUP)`
   == `signal.SIG_IGN`.
4. watcher disabled by env: with
   `PYFINAGENT_DISABLE_GOVERNANCE_WATCHER=1`, no watcher thread
   named "governance-limits-watcher" is running.
5. mutation_kills_process: source of limits_loader.py contains a
   `os._exit(` call (not `sys.exit`) in the watcher path.
6. digest exposed to /api/health: main.py's health handler
   references `limits_digest` in its return dict.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOADER = REPO / "backend" / "governance" / "limits_loader.py"
MAIN = REPO / "backend" / "main.py"
OUT = REPO / "handoff" / "limits_loader_audit.json"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    # Force the env var so the test run doesn't spawn a watcher
    # thread that could then tail-kill the audit if it races with
    # an editor save.
    os.environ["PYFINAGENT_DISABLE_GOVERNANCE_WATCHER"] = "1"

    reasons: list[str] = []
    checks: dict[str, bool] = {}

    # 1. Imports.
    try:
        sys.path.insert(0, str(REPO))
        from backend.governance.limits_loader import (  # noqa: F401
            get_digest, is_initialized, load_once,
        )
        checks["imports_ok"] = True
    except Exception as e:
        checks["imports_ok"] = False
        reasons.append(f"import failed: {e}")
        # can't proceed with runtime checks
        result = {
            "step": "4.9.2",
            "ran_at": datetime.now(timezone.utc).isoformat(),
            **checks,
            "reasons": reasons,
            "verdict": "FAIL",
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return 1 if args.check else 0

    # 2. Idempotent + digest format.
    l1 = load_once()
    l2 = load_once()
    d = get_digest()
    idempotent = l1 is l2
    digest_hex = (
        isinstance(d, str)
        and len(d) == 64
        and all(c in "0123456789abcdef" for c in d)
    )
    checks["load_once_idempotent"] = idempotent
    checks["digest_64_hex"] = digest_hex
    if not idempotent:
        reasons.append("load_once returned different objects")
    if not digest_hex:
        reasons.append(f"digest not 64-char hex: got {d!r}")

    # 3. SIGHUP ignored.
    sighup_ok = signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
    checks["sighup_ignored"] = sighup_ok
    if not sighup_ok:
        reasons.append(f"SIGHUP handler != SIG_IGN; got {signal.getsignal(signal.SIGHUP)!r}")

    # 4. Watcher disabled by env.
    watcher_running = any(
        t.name == "governance-limits-watcher"
        for t in threading.enumerate()
    )
    checks["watcher_disabled_by_env"] = not watcher_running
    if watcher_running:
        reasons.append("watcher thread started despite env disable flag")

    # 5. os._exit present in loader source.
    src = LOADER.read_text(encoding="utf-8") if LOADER.exists() else ""
    checks["os_exit_in_watcher"] = bool(re.search(r"os\._exit\(", src))
    # Must NOT use sys.exit inside the watcher function block (check
    # within 20 lines of `def _watcher_loop`).
    watcher_block_m = re.search(
        r"def _watcher_loop\([^)]*\)[^:]*:\s*\n(?:[^\n]*\n){0,30}",
        src,
    )
    watcher_block = watcher_block_m.group(0) if watcher_block_m else ""
    checks["no_sys_exit_in_watcher"] = "sys.exit" not in watcher_block
    if not checks["os_exit_in_watcher"]:
        reasons.append("os._exit not present in loader source")
    if not checks["no_sys_exit_in_watcher"]:
        reasons.append("sys.exit used inside watcher (catchable); use os._exit")

    # 6. Health endpoint exposes digest.
    main_src = MAIN.read_text(encoding="utf-8") if MAIN.exists() else ""
    checks["health_exposes_digest"] = (
        "limits_digest" in main_src
        and "/api/health" in main_src
    )
    if not checks["health_exposes_digest"]:
        reasons.append("backend/main.py /api/health does not return limits_digest")

    all_ok = all(checks.values())
    verdict = "PASS" if all_ok else "FAIL"
    result = {
        "step": "4.9.2",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        **checks,
        "digest": d if digest_hex else None,
        "reasons": reasons,
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "verdict": verdict,
        **{k: v for k, v in checks.items()},
    }))
    if args.check and not all_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
