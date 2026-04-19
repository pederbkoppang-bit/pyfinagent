#!/usr/bin/env python
"""phase-12.2 Rainbow promote CLI.

Flips the Service selector's `color:` label to a new target color.
Pre-deploy the target color's Deployment BEFORE running this script (see
`deploy/rainbow/README.md`); `kubectl rollout status` should report the
target pods Ready first.

Usage:
    # Dry-run: print the kubectl patch command + JSON; do nothing.
    python scripts/deploy/rainbow/promote.py --dry-run --to green

    # Live flip.
    python scripts/deploy/rainbow/promote.py --to green

Exit codes:
    0  kubectl patch succeeded (or --dry-run completed).
    1  kubectl exited non-zero or raised.
    2  argparse / input validation error.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys


def build_patch_json(color: str) -> str:
    """Produce the exact JSON string kubectl patch expects."""
    # strategic-merge-patch format: partial spec.selector
    return json.dumps({"spec": {"selector": {"color": color}}})


def build_kubectl_cmd(service: str, patch_json: str) -> list[str]:
    return ["kubectl", "patch", "service", service, "-p", patch_json]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Flip the pyfinagent Rainbow Service selector to a new color. "
            "Pre-deploy the target color's Deployment FIRST."
        )
    )
    ap.add_argument(
        "--to",
        required=True,
        help="Target color (e.g., blue, green). Must match a label on an "
        "already-ready Deployment.",
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

    # Basic validation: target color must be a short identifier.
    if not args.to or not args.to.replace("-", "").replace("_", "").isalnum():
        print(
            f"[promote] invalid --to color {args.to!r}; must be alphanumeric",
            file=sys.stderr,
        )
        return 2

    patch_json = build_patch_json(args.to)
    cmd = build_kubectl_cmd(args.service, patch_json)

    if args.dry_run:
        print(f"[promote] DRY-RUN: would patch service={args.service} to color={args.to}")
        print(f"[promote] kubectl command:")
        # Print in a form a human can copy/paste. argparse-safe quoting.
        print("  " + " ".join(
            [c if not any(ch in c for ch in " {}\"") else f"'{c}'" for c in cmd]
        ))
        print(f"[promote] patch JSON: {patch_json}")
        return 0

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        print("[promote] kubectl not found on PATH", file=sys.stderr)
        return 1
    except subprocess.TimeoutExpired:
        print("[promote] kubectl patch timed out (30s)", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[promote] kubectl patch failed: {exc!r}", file=sys.stderr)
        return 1

    if result.returncode != 0:
        print(
            f"[promote] kubectl exited {result.returncode}: {result.stderr.strip()}",
            file=sys.stderr,
        )
        return 1

    print(f"[promote] {result.stdout.strip()}")
    print(f"[promote] flipped {args.service} selector color -> {args.to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
