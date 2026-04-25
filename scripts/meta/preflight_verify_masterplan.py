"""phase-16.38 (#29) pre-flight verifier for masterplan verification commands.

Walks `.claude/masterplan.json`, extracts every step's `verification`
field, parses the command shell-style WITHOUT executing it, and checks
that referenced file paths and Python imports actually exist.

Catches drift like phase-16.22 surfaced: ~6 verification commands had
referenced files that had been renamed or deleted. The previous fix
(#9 in 16.33+16.34) was reactive -- this script is the proactive gate.

Static checks only -- no command execution. Avoids:
  - side effects (database writes, file creation)
  - cost (LLM calls, BQ queries)
  - flake (network timeouts, env-dependent tools)

Handles BOTH verification field shapes per masterplan audit:
  - String:  "verification": "python -c '...'"
  - Object:  "verification": {"command": "...", "success_criteria": [...]}

Usage:
    python scripts/meta/preflight_verify_masterplan.py .claude/masterplan.json
    python scripts/meta/preflight_verify_masterplan.py .claude/masterplan.json --quiet

Exit codes:
  0 = no broken refs detected
  1 = one or more steps reference missing paths or unimportable modules
  2 = filesystem / JSON parse error
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]

PATH_SUFFIXES = frozenset({
    ".py", ".yaml", ".yml", ".json", ".md", ".sh",
    ".tsv", ".csv", ".mjs", ".js", ".ts", ".tsx",
})

# Project subtree roots -- tokens starting with one of these are real paths.
# A bare slash-leading token like "/login" is an HTTP route, not a file.
PROJECT_ROOTS = (
    "backend/", "frontend/", "tests/", "scripts/", "docs/",
    ".claude/", "handoff/",
)

# Tokens that look like file refs but are clearly NOT (regex chars, shell
# metachars, ticker symbols, URL routes). False-positive suppressors.
NON_PATH_PATTERNS = (
    "\\.", "\\|",  # regex escapes
    "://",  # URLs
    ";",  # multi-statement blocks
    "(",  # subshell / function
    ")",
    " ",  # multi-token strings broken apart
)

VENV_PREFIX_RE = re.compile(
    r"^source\s+\.venv/bin/activate\s*&&\s*", re.MULTILINE
)

IMPORT_RE = re.compile(r"\b(?:from\s+([a-zA-Z_][\w.]*)\s+import|import\s+([a-zA-Z_][\w.]*))")


def _is_path_token(token: str) -> bool:
    """Heuristic: token looks like a file path the script should verify.

    Tightened heuristic per phase-16.38 first-run feedback:
    - Must start with one of PROJECT_ROOTS (so URL routes, ticker symbols,
      regex patterns, and quoted shell substitutions are all suppressed)
    - OR have a project-suffix and contain no shell metacharacters
    """
    if token.startswith("-"):
        return False
    if any(pat in token for pat in NON_PATH_PATTERNS):
        return False
    if "=" in token and "/" not in token:  # env var assignment
        return False
    # Strict: must start with a real project subtree root
    if any(token.startswith(root) for root in PROJECT_ROOTS):
        return True
    # OR: bare filename with a known project suffix (no slashes)
    if "/" not in token:
        suffix = Path(token).suffix
        if suffix and suffix in PATH_SUFFIXES:
            return True
    return False


def _extract_imports(cmd: str) -> set[str]:
    """Pull out `from X.Y import` and `import X.Y` module names."""
    out: set[str] = set()
    for m in IMPORT_RE.finditer(cmd):
        mod = m.group(1) or m.group(2)
        if mod:
            # `import a, b` only catches first name; that's fine for the
            # common `python -c "from X import Y"` pattern.
            out.add(mod.split(",")[0].strip())
    return out


def _strip_venv_prefix(cmd: str) -> str:
    return VENV_PREFIX_RE.sub("", cmd)


def _check_paths(tokens: Iterable[str]) -> list[str]:
    """Return list of broken path tokens (relative to REPO_ROOT)."""
    broken: list[str] = []
    for tok in tokens:
        if not _is_path_token(tok):
            continue
        # Resolve relative to repo root
        candidate = REPO_ROOT / tok
        if not candidate.exists():
            # Some commands use globs / shell expansions -- skip if obviously not a literal path
            if "*" in tok or "?" in tok or "[" in tok:
                continue
            broken.append(tok)
    return broken


def _check_imports(modules: set[str]) -> list[str]:
    """Return list of unimportable module names."""
    broken: list[str] = []
    # Ensure REPO_ROOT is on sys.path so backend.* imports resolve
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    for mod in modules:
        # Skip built-ins / single-name stdlib (find_spec triggers parent
        # imports for dotted; for single names we only verify if it looks
        # like a project module by having a dot).
        if "." not in mod:
            # bare names like `import json` are stdlib or top-level installed; skip
            continue
        try:
            spec = importlib.util.find_spec(mod)
        except (ModuleNotFoundError, ImportError, ValueError, TypeError):
            broken.append(mod)
            continue
        if spec is None:
            broken.append(mod)
    return broken


def _extract_command(verification: Any) -> str | None:
    """Handle both string + object verification shapes."""
    if verification is None:
        return None
    if isinstance(verification, str):
        return verification
    if isinstance(verification, dict):
        cmd = verification.get("command")
        if isinstance(cmd, str):
            return cmd
    return None


def _walk_steps(node: Any) -> Iterable[tuple[str, str]]:
    """Yield (step_id, raw_command) tuples for every step with verification."""
    if isinstance(node, dict):
        sid = node.get("id")
        ver = node.get("verification")
        cmd = _extract_command(ver)
        if sid and cmd:
            yield (str(sid), cmd)
        for v in node.values():
            yield from _walk_steps(v)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_steps(item)


def verify(masterplan_path: str | Path, *, quiet: bool = False) -> int:
    p = Path(masterplan_path)
    if not p.exists():
        print(f"preflight_verify_masterplan: file not found: {p}", file=sys.stderr)
        return 2
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"preflight_verify_masterplan: JSON parse error: {e}", file=sys.stderr)
        return 2

    steps = list(_walk_steps(data))
    n_steps = len(steps)
    n_broken = 0
    n_warn = 0

    for sid, raw_cmd in steps:
        cmd = _strip_venv_prefix(raw_cmd)
        try:
            tokens = shlex.split(cmd, posix=True)
        except ValueError as e:
            n_warn += 1
            print(
                f"[WARN] step={sid}: unparseable command ({e})",
                file=sys.stderr,
            )
            continue

        broken_paths = _check_paths(tokens)
        broken_imports = _check_imports(_extract_imports(cmd))

        if broken_paths or broken_imports:
            n_broken += 1
            for tok in broken_paths:
                print(
                    f"[BROKEN] step={sid}: missing path {tok!r}",
                    file=sys.stderr,
                )
            for mod in broken_imports:
                print(
                    f"[BROKEN] step={sid}: unimportable module {mod!r}",
                    file=sys.stderr,
                )

    if not quiet:
        print(
            f"preflight_verify_masterplan: scanned {n_steps} steps, "
            f"{n_broken} broken, {n_warn} unparseable"
        )

    return 1 if n_broken > 0 else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-flight: verify masterplan verification commands "
                    "reference live paths/modules without executing them."
    )
    parser.add_argument("path", help="path to masterplan.json")
    parser.add_argument("--quiet", action="store_true",
                        help="no stdout (exit code only); broken refs still go to stderr")
    args = parser.parse_args(argv)
    return verify(args.path, quiet=args.quiet)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
