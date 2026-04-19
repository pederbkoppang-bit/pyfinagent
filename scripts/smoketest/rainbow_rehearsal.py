#!/usr/bin/env python
"""phase-12.4 Rainbow rehearsal -- end-to-end color-flip-only smoketest.

Composes the pieces phase-12 delivered (manifests + promote.py +
rollback.py + canary SLO diff) into a 5-stage serial rehearsal that
proves the Rainbow machinery works without a real cluster or migration.

Per the phase-12.0 cycle-2 scope-reassignment note: the Vertex migration
shipped directly in phase-11 without Rainbow, so phase-12.4's original
"first real migration" candidate is gone. This dummy rehearsal is the
replacement — operators can run it before any future migration as a
preflight drill.

Stages:
    S1. promote dry-run       -- verify promote.py emits the expected
                                 kubectl patch command + JSON.
    S2. canary equal latency  -- synthetic 20 blue + 20 green @ 100ms
                                 each; SLO diff must report regression=False.
    S3. canary regression     -- synthetic green @ 250ms; SLO diff must
                                 report regression=True + ratio > 2.0.
    S4. rollback dry-run      -- verify rollback.py emits the expected
                                 patch-back-to-blue command.
    S5. audit + summary       -- append JSONL row to handoff/audit/
                                 rainbow_rehearsal.jsonl + print JSON
                                 summary.

Exit codes:
    0  all 5 stages completed (regression detection in S3 is expected).
    1  uncaught Python exception escaped a stage's fail-open boundary.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import logging
import sys
import traceback
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("rainbow_rehearsal")

_AUDIT_JSONL = _ROOT / "handoff" / "audit" / "rainbow_rehearsal.jsonl"
_SCRIPTS_DIR = _ROOT / "scripts" / "deploy" / "rainbow"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_cli(name: str):
    """Import promote.py or rollback.py via importlib (module lives outside backend/)."""
    spec = importlib.util.spec_from_file_location(
        f"rainbow_{name}", _SCRIPTS_DIR / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_audit(record: dict) -> None:
    try:
        _AUDIT_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:  # pragma: no cover -- fail-open
        logger.debug("audit write fail-open err=%r", exc)


# ------------------ stages ------------------


def _stage_promote_dry_run() -> dict:
    """S1: call promote.py --dry-run --to green; verify output."""
    promote = _load_cli("promote")
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = promote.main(["--dry-run", "--to", "green"])
        output = buf.getvalue()
        ok = rc == 0 and "DRY-RUN" in output and "color=green" in output
        return {
            "name": "promote_dry_run",
            "ok": ok,
            "rc": rc,
            "has_dry_run_line": "DRY-RUN" in output,
            "has_color_green": "color=green" in output,
        }
    except Exception as exc:
        return {
            "name": "promote_dry_run",
            "ok": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def _stage_canary_equal_latency() -> dict:
    """S2: 20 blue + 20 green @ 100ms -> regression=False."""
    try:
        from backend.services.observability.api_call_log import (
            log_api_call,
            reset_buffer_for_test,
        )
        from backend.services.observability.rainbow_canary import (
            canary_snapshot_from_buffer,
        )

        reset_buffer_for_test()
        for _ in range(20):
            log_api_call(
                source="pyfinagent-blue",
                endpoint="/api/rehearsal",
                http_status=200,
                latency_ms=100.0,
                response_bytes=42,
                ok=True,
            )
        for _ in range(20):
            log_api_call(
                source="pyfinagent-green",
                endpoint="/api/rehearsal",
                http_status=200,
                latency_ms=100.0,
                response_bytes=42,
                ok=True,
            )
        diff = canary_snapshot_from_buffer(
            is_blue=lambda r: r.get("source") == "pyfinagent-blue",
            is_green=lambda r: r.get("source") == "pyfinagent-green",
        )
        reset_buffer_for_test()
        ok = diff.reason == "ok" and diff.regression is False
        return {
            "name": "canary_equal_latency",
            "ok": ok,
            "reason": diff.reason,
            "regression": diff.regression,
            "ratio": round(diff.ratio, 3),
            "blue_samples": diff.blue_samples,
            "green_samples": diff.green_samples,
        }
    except Exception as exc:
        return {
            "name": "canary_equal_latency",
            "ok": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def _stage_canary_regression() -> dict:
    """S3: green @ 250ms vs blue @ 100ms -> regression=True, ratio > 2.0."""
    try:
        from backend.services.observability.api_call_log import (
            log_api_call,
            reset_buffer_for_test,
        )
        from backend.services.observability.rainbow_canary import (
            canary_snapshot_from_buffer,
        )

        reset_buffer_for_test()
        for _ in range(20):
            log_api_call(
                source="pyfinagent-blue",
                endpoint="/api/rehearsal",
                http_status=200,
                latency_ms=100.0,
                response_bytes=42,
                ok=True,
            )
        for _ in range(20):
            log_api_call(
                source="pyfinagent-green",
                endpoint="/api/rehearsal",
                http_status=200,
                latency_ms=250.0,
                response_bytes=42,
                ok=True,
            )
        diff = canary_snapshot_from_buffer(
            is_blue=lambda r: r.get("source") == "pyfinagent-blue",
            is_green=lambda r: r.get("source") == "pyfinagent-green",
        )
        reset_buffer_for_test()
        ok = diff.reason == "ok" and diff.regression is True and diff.ratio > 2.0
        return {
            "name": "canary_regression",
            "ok": ok,
            "reason": diff.reason,
            "regression": diff.regression,
            "ratio": round(diff.ratio, 3),
            "threshold": diff.threshold,
        }
    except Exception as exc:
        return {
            "name": "canary_regression",
            "ok": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def _stage_rollback_dry_run() -> dict:
    """S4: rollback.py --dry-run; verify prints "blue" as rollback target."""
    rollback = _load_cli("rollback")
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = rollback.main(["--dry-run"])
        output = buf.getvalue()
        ok = rc == 0 and "blue" in output and "DRY-RUN" in output
        return {
            "name": "rollback_dry_run",
            "ok": ok,
            "rc": rc,
            "has_dry_run_line": "DRY-RUN" in output,
            "has_blue": "blue" in output,
        }
    except Exception as exc:
        return {
            "name": "rollback_dry_run",
            "ok": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


# ------------------ orchestrator ------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Rainbow deploy end-to-end rehearsal (no real cluster, no "
            "production changes). Drives promote + canary SLO diff + "
            "rollback on synthetic data."
        )
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Unused; the whole rehearsal is inherently dry-run. Retained for "
        "CLI shape parity with phase6_e2e.py.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose logging.",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    summary = {
        "ts": _now_iso(),
        "dry_run": args.dry_run,
        "stages": [],
        "overall_ok": False,
    }

    try:
        stages = [
            _stage_promote_dry_run(),
            _stage_canary_equal_latency(),
            _stage_canary_regression(),
            _stage_rollback_dry_run(),
        ]
        summary["stages"] = stages
        summary["overall_ok"] = all(s.get("ok", False) for s in stages)
    except Exception as exc:
        summary["fatal_exception"] = repr(exc)
        summary["traceback"] = traceback.format_exc()
        summary["overall_ok"] = False
        _write_audit(summary)
        print(json.dumps(summary, default=str, indent=2))
        return 1

    _write_audit(summary)
    print(json.dumps(summary, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
