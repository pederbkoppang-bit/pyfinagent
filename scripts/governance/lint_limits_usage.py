"""phase-4.9 step 4.9.3 lint: governance-limits usage guard.

AST-based scanner that walks every tracked `.py` under the repo
and fails if any file (outside an allowlist) does one of:

(a) Defines a module-level literal constant whose name matches a
    governance field (case-insensitive on `max_*_pct` style or the
    UPPER_SNAKE equivalent) with a numeric literal value.
    Example violation: `MAX_PORTFOLIO_LEVERAGE = 1.5` sitting
    outside the governance package.

(b) Reads an env-var backdoor: `os.environ.get("MAX_...")` or
    `os.getenv("MAX_...")` where the key is one of the six
    governance names (any case). No env-var override allowed;
    the snapshot is the single source.

(c) WARN: attribute access to the legacy fields
    `settings.paper_daily_loss_limit_pct` /
    `settings.paper_trailing_dd_limit_pct` in code paths that
    should instead consult the immutable snapshot. These are
    migration markers; `--strict` prints WARN but does not fail
    this cycle (a later phase-4.9 migration step will flip the
    hard fail).

Allowlist (files permitted to reference the governance names
freely because they ARE the governance layer or this lint
itself):

    backend/governance/limits_schema.py
    backend/governance/limits_loader.py
    scripts/governance/lint_limits_usage.py
    scripts/audit/limits_lint_audit.py
    scripts/audit/immutable_limits_audit.py
    scripts/audit/limits_loader_audit.py

Exit codes:
    0 -- no violations (or warnings only without --strict)
    1 -- violation + --strict set
    2 -- misuse (argparse)
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

GOVERNANCE_NAMES: tuple[str, ...] = (
    "max_position_notional_pct",
    "max_portfolio_leverage",
    "max_daily_loss_pct",
    "max_trailing_dd_pct",
    "max_gross_exposure_pct",
    "max_sector_weight_pct",
)

GOVERNANCE_NAMES_UPPER: set[str] = {n.upper() for n in GOVERNANCE_NAMES}

ALLOWLIST: frozenset[str] = frozenset({
    "backend/governance/limits_schema.py",
    "backend/governance/limits_loader.py",
    "scripts/governance/lint_limits_usage.py",
    "scripts/audit/limits_lint_audit.py",
    "scripts/audit/immutable_limits_audit.py",
    "scripts/audit/limits_loader_audit.py",
})

LEGACY_SETTINGS_ATTRS: tuple[str, ...] = (
    "paper_daily_loss_limit_pct",
    "paper_trailing_dd_limit_pct",
)

SKIP_DIRS: tuple[str, ...] = (
    ".venv", ".venv.py313.bak", "node_modules", ".git", "__pycache__",
    ".mypy_cache", ".pytest_cache", "build", "dist", ".next",
    "handoff/archive", ".claude/worktrees", ".claude/skills",
    ".claude/agents", ".claude/context", ".claude/rules",
)


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO)).replace("\\", "/")


def _iter_py_files() -> list[Path]:
    out: list[Path] = []
    for p in REPO.rglob("*.py"):
        rel = _rel(p)
        if any(rel.startswith(s + "/") or ("/" + s + "/") in rel for s in SKIP_DIRS):
            continue
        out.append(p)
    return sorted(out)


def _is_numeric_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_numeric_literal(node.operand)
    return False


def _name_is_governance(name: str) -> bool:
    return name in GOVERNANCE_NAMES or name.upper() in GOVERNANCE_NAMES_UPPER


def _env_key_governance(key: str | None) -> bool:
    if key is None:
        return False
    k = key.strip().lower()
    return k in GOVERNANCE_NAMES


def _str_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _scan_file(path: Path) -> tuple[list[dict], list[dict]]:
    """Return (violations, warnings) for this file."""
    rel = _rel(path)
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src, filename=rel)
    except SyntaxError:
        return [], []

    violations: list[dict] = []
    warnings: list[dict] = []

    allowlisted = rel in ALLOWLIST

    # (a) Module-level literal constants named after governance fields.
    if not allowlisted:
        for node in tree.body:
            targets: list[ast.AST] = []
            value: ast.AST | None = None
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
                value = node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets = [node.target]
                value = node.value
            else:
                continue

            if value is None or not _is_numeric_literal(value):
                continue

            for t in targets:
                if isinstance(t, ast.Name) and _name_is_governance(t.id):
                    violations.append({
                        "file": rel,
                        "line": node.lineno,
                        "kind": "module_constant",
                        "name": t.id,
                        "reason": (
                            f"module-level literal `{t.id}` duplicates a "
                            "governance limit; read from the immutable "
                            "snapshot instead"
                        ),
                    })

    # (b) os.environ.get / os.getenv backdoor + (c) legacy settings attr.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            key_node: ast.AST | None = None
            kind: str | None = None
            # os.environ.get("KEY")
            if (
                isinstance(fn, ast.Attribute)
                and fn.attr == "get"
                and isinstance(fn.value, ast.Attribute)
                and fn.value.attr == "environ"
                and isinstance(fn.value.value, ast.Name)
                and fn.value.value.id == "os"
            ):
                if node.args:
                    key_node = node.args[0]
                    kind = "os.environ.get"
            # os.getenv("KEY")
            elif (
                isinstance(fn, ast.Attribute)
                and fn.attr == "getenv"
                and isinstance(fn.value, ast.Name)
                and fn.value.id == "os"
            ):
                if node.args:
                    key_node = node.args[0]
                    kind = "os.getenv"

            if key_node is not None and kind is not None and not allowlisted:
                key = _str_literal(key_node)
                if _env_key_governance(key):
                    violations.append({
                        "file": rel,
                        "line": node.lineno,
                        "kind": "env_backdoor",
                        "name": key,
                        "reason": (
                            f"{kind}('{key}') would let an env-var "
                            "override a governance limit; disallowed"
                        ),
                    })

        # (c) settings.paper_daily_loss_limit_pct etc.
        if isinstance(node, ast.Attribute) and not allowlisted:
            if (
                node.attr in LEGACY_SETTINGS_ATTRS
                and isinstance(node.value, ast.Name)
                and node.value.id == "settings"
            ):
                warnings.append({
                    "file": rel,
                    "line": node.lineno,
                    "kind": "legacy_settings_attr",
                    "name": node.attr,
                    "reason": (
                        f"settings.{node.attr} should be migrated to "
                        "the immutable snapshot (phase-4.9 follow-up)"
                    ),
                })

    return violations, warnings


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strict", action="store_true",
                   help="exit 1 on any (a)/(b) violation")
    p.add_argument("--json", dest="as_json", action="store_true",
                   help="emit JSON report to stdout")
    args = p.parse_args()

    all_violations: list[dict] = []
    all_warnings: list[dict] = []
    files_scanned = 0

    for path in _iter_py_files():
        files_scanned += 1
        v, w = _scan_file(path)
        all_violations.extend(v)
        all_warnings.extend(w)

    report = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "files_scanned": files_scanned,
        "governance_names": list(GOVERNANCE_NAMES),
        "allowlist": sorted(ALLOWLIST),
        "violations": all_violations,
        "warnings": all_warnings,
    }

    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print(f"lint_limits_usage: scanned {files_scanned} py files; "
              f"violations={len(all_violations)} warnings={len(all_warnings)}")
        for v in all_violations:
            print(f"  FAIL {v['file']}:{v['line']} [{v['kind']}] {v['reason']}")
        for w in all_warnings:
            print(f"  WARN {w['file']}:{w['line']} [{w['kind']}] {w['reason']}")

    if args.strict and all_violations:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
