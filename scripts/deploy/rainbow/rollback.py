#!/usr/bin/env python
"""phase-12.2 Rainbow rollback CLI.

Reverts the Service selector's `color:` label to the previous color.
Reads the current color from the cluster (`kubectl get service -o
jsonpath='{.spec.selector.color}'`), then flips to the default "other"
color in the blue/green 2-color MVP palette. For a larger palette,
pass `--to <previous-color>` explicitly.

Usage:
    # Dry-run: print the kubectl patch command + JSON; do nothing.
    python scripts/deploy/rainbow/rollback.py --dry-run

    # Live rollback (auto-detects the other color from the blue/green pair).
    python scripts/deploy/rainbow/rollback.py

    # Live rollback to an explicit color (for >2-color palettes).
    python scripts/deploy/rainbow/rollback.py --to indigo

Exit codes:
    0  rollback succeeded (or --dry-run completed).
    1  kubectl exited non-zero or raised.
    2  argparse / input validation error.
    3  could not read current color from the cluster.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys


# 2-color MVP toggle. For larger palettes, operator passes --to explicitly.
_TOGGLE = {"blue": "green", "green": "blue"}


def read_current_color(service: str, timeout: int = 10) -> str | None:
    """Read the Service's current selector color. Returns None on failure."""
    try:
        result = subprocess.run(
            [
                "kubectl",
                "get",
                "service",
                service,
                "-o",
                "jsonpath={.spec.selector.color}",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return None
    if result.returncode != 0:
        return None
    color = (result.stdout or "").strip()
    return color or None


def build_patch_json(color: str) -> str:
    return json.dumps({"spec": {"selector": {"color": color}}})


def build_kubectl_cmd(service: str, patch_json: str) -> list[str]:
    return ["kubectl", "patch", "service", service, "-p", patch_json]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Roll back the pyfinagent Rainbow Service selector to the "
            "previous color. Auto-detects from blue/green 2-color palette, "
            "or pass --to COLOR explicitly."
        )
    )
    ap.add_argument(
        "--to",
        default=None,
        help="Target color (optional; auto-detected for blue/green palette).",
    )
    ap.add_argument(
        "--service",
        default="pyfinagent-backend",
        help="Kubernetes Service name (default: pyfinagent-backend).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the kubectl patch command + JSON; do NOT execute.",
    )
    args = ap.parse_args(argv)

    target = args.to

    if target is None:
        # Auto-detect from the cluster.
        if args.dry_run:
            # In dry-run we try the cluster read but fall back gracefully
            # so the command can be demonstrated without a live cluster.
            current = read_current_color(args.service)
            if current is None:
                # No cluster: assume "currently green, roll back to blue".
                # This matches the README's default rollout recipe (green
                # is the new color; blue was the prior).
                target = "blue"
                print(
                    "[rollback] DRY-RUN: no cluster reachable; assuming "
                    "current=green and rolling back to blue",
                )
            else:
                target = _TOGGLE.get(current)
                if target is None:
                    print(
                        f"[rollback] DRY-RUN: current color {current!r} is not "
                        "in the blue/green toggle; pass --to explicitly",
                        file=sys.stderr,
                    )
                    return 2
                print(f"[rollback] DRY-RUN: current={current} target={target}")
        else:
            current = read_current_color(args.service)
            if current is None:
                print(
                    f"[rollback] could not read current color from "
                    f"service={args.service}; pass --to explicitly",
                    file=sys.stderr,
                )
                return 3
            target = _TOGGLE.get(current)
            if target is None:
                print(
                    f"[rollback] current color {current!r} is not in the "
                    "blue/green toggle; pass --to explicitly",
                    file=sys.stderr,
                )
                return 2
            print(f"[rollback] current={current} target={target}")

    if not target or not target.replace("-", "").replace("_", "").isalnum():
        print(
            f"[rollback] invalid target color {target!r}; must be alphanumeric",
            file=sys.stderr,
        )
        return 2

    patch_json = build_patch_json(target)
    cmd = build_kubectl_cmd(args.service, patch_json)

    if args.dry_run:
        print(f"[rollback] DRY-RUN: would patch service={args.service} to color={target}")
        print(f"[rollback] kubectl command:")
        print("  " + " ".join(
            [c if not any(ch in c for ch in " {}\"") else f"'{c}'" for c in cmd]
        ))
        print(f"[rollback] patch JSON: {patch_json}")
        return 0

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        print("[rollback] kubectl not found on PATH", file=sys.stderr)
        return 1
    except subprocess.TimeoutExpired:
        print("[rollback] kubectl patch timed out (30s)", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[rollback] kubectl patch failed: {exc!r}", file=sys.stderr)
        return 1

    if result.returncode != 0:
        print(
            f"[rollback] kubectl exited {result.returncode}: {result.stderr.strip()}",
            file=sys.stderr,
        )
        return 1

    print(f"[rollback] {result.stdout.strip()}")
    print(f"[rollback] rolled back {args.service} selector color -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
