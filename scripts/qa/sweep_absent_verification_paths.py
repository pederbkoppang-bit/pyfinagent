#!/usr/bin/env python3
"""phase-75.17: sweep `.claude/masterplan.json` for `status=done` steps whose
`verification` block references a filesystem path that does not exist on
disk. Such a command is unrunnable, so its recorded PASS is unreproducible
-- governance rot (ISACA "a policy cannot prove the system behaved
correctly at the moment it mattered"; see research_brief_75.17.md).

Design (strategy-agnostic adjudicator, per the 75.17 research gate):
MANY extractors pull path-shaped candidate tokens out of a command string;
ONE adjudicator (`_fp_reason`) decides genuine-vs-false-positive. The
research brief proved the genuine set is IDENTICAL whether fed by
{structure-aware, broad-regex} or all four extractors -- the census is
robust to extraction strategy, not an artifact of a lucky regex.

`classify()` is the IMPORTABLE core (dict in, dict out; no argv, no
sys.exit) -- step 75.19 is chartered to reuse it for the
`preflight_verify_masterplan.py` recalibration. Do not add nightly CI
wiring here; that is 75.19's job (see research_brief_75.17.md, "Where the
sweep tool lives").

A step already carrying a `superseded_record` sibling is treated as
DISPOSITIONED, not re-flagged -- this is what makes `classify()` return an
empty genuine set once every genuine defect has been annotated, and lets
the same function report the pre-annotation genuine set when pointed at an
older masterplan snapshot (e.g. `git show <commit>:.claude/masterplan.json`).

Verification shapes handled (a naive `.get("command")` crashes on the
list-shaped and silently drops the str-shaped):
  - dict   {"command": "...", ...}          (majority)
  - str    "..."                            (bare command)
  - list   ["...", "..."]                   (multi-command)
  - None                                    (no verification at all)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# extension-anchored token regex; tsx BEFORE ts (the 75.2.1 alternation
# caveat -- an ts|tsx alternation would truncate "component.tsx" at "component.ts").
_EXT = (
    r"(?:py|tsx|ts|jsx|js|md|json|sh|ya?ml|tsv|txt|plist|sql|html|css|cfg"
    r"|ini|toml|lock|csv|pkl|joblib|proto|parquet)"
)

_ABS_HOST_PREFIXES = (
    "/Users", "/Library", "/tmp", "/var", "/etc", "/opt",
    "/private", "/Applications", "/System", "/bin", "/usr", "/sbin",
)

_RUNTIME_TRANSIENT_PREFIXES = ("tmp/", "handoff/", "frontend/handoff/")


# ---------------------------------------------------------------------------
# Shape normalizer (the polymorphic-field discriminator)
# ---------------------------------------------------------------------------

def verif_commands(verification: object) -> list[str]:
    """Normalize any of the four `verification` shapes into a list of
    command strings. Never raises on an unexpected shape -- returns []."""
    if verification is None:
        return []
    if isinstance(verification, str):
        return [verification]
    if isinstance(verification, list):
        return [x for x in verification if isinstance(x, str)]
    if isinstance(verification, dict):
        command = verification.get("command")
        if isinstance(command, str):
            return [command]
        if isinstance(command, list):
            return [x for x in command if isinstance(x, str)]
    return []


# ---------------------------------------------------------------------------
# Extractors -- pull candidate path-shaped tokens out of a command string
# ---------------------------------------------------------------------------

_CTX_PATTERNS = [
    re.compile(r"""open\(\s*['"]([^'"]+)['"]"""),
    re.compile(r"\btest\s+(?:!\s+)?-[a-zA-Z]+\s+([A-Za-z0-9_./+-]+\.\w+)"),
    re.compile(r"!\s*test\s+-[a-zA-Z]+\s+([A-Za-z0-9_./+-]+\.\w+)"),
    re.compile(r"\bpython3?\s+(?:-m\s+\S+\s+)?([A-Za-z0-9_./+-]+\.py)\b"),
    re.compile(r"\bpytest\s+([A-Za-z0-9_./+-]+\.py)\b"),
    re.compile(r"\bsource\s+([A-Za-z0-9_./+-]+)"),
    re.compile(r"\bcat\s+([A-Za-z0-9_./+-]+\.\w+)"),
    re.compile(r"\bbash\s+([A-Za-z0-9_./+-]+\.sh)\b"),
]
_RE_EXT = re.compile(
    r"(?<![A-Za-z0-9_./+-])([A-Za-z0-9_][A-Za-z0-9_./+-]*\.%s)(?![A-Za-z0-9])" % _EXT
)
_RE_DIR = re.compile(
    r"(?<![A-Za-z0-9_./+-])((?:backend|frontend|scripts|docs|tests|\.claude)/[A-Za-z0-9_./+-]+)"
)


def _extract_structure_aware(cmd: str) -> set[str]:
    """Extractor A: context regexes anchored on shell/python verbs."""
    out: set[str] = set()
    for pattern in _CTX_PATTERNS:
        for m in pattern.finditer(cmd):
            out.add(m.group(1))
    return out


def _extract_broad_regex(cmd: str) -> set[str]:
    """Extractor B: extension-anchored + repo-dir-anchored regex."""
    out: set[str] = set()
    for pattern in (_RE_EXT, _RE_DIR):
        for m in pattern.finditer(cmd):
            out.add(m.group(1).rstrip("./"))
    return out


def _extract_quoted_and_whitespace(cmd: str) -> set[str]:
    """Extractor C: whitespace + quoted-string tokenizer (higher recall,
    lower precision -- the adjudicator's well-formedness/boundary gates
    collapse its noise back to the same genuine set as A/B)."""
    out: set[str] = set()
    for m in re.finditer(r"""['"]([^'"]+)['"]""", cmd):
        out.add(m.group(1))
    for tok in cmd.split():
        out.add(tok)
    return {
        t for t in out
        if ("/" in t and re.search(r"\.%s(?![A-Za-z0-9])" % _EXT, t))
        or re.match(r"^[A-Za-z0-9_./+-]+\.%s$" % _EXT, t)
    }


def _extract_maximal_recall(cmd: str) -> set[str]:
    """Extractor D: bare-extension scan + verb-argument scan."""
    out: set[str] = set()
    for m in re.finditer(r"[A-Za-z0-9_][A-Za-z0-9_./+-]*\.%s(?![A-Za-z0-9])" % _EXT, cmd):
        out.add(m.group(0))
    for m in re.finditer(r"(?:python3?|pytest|source|bash)\s+([A-Za-z0-9_./+-]+)", cmd):
        out.add(m.group(1))
    return out


_EXTRACTORS = (
    _extract_structure_aware,
    _extract_broad_regex,
    _extract_quoted_and_whitespace,
    _extract_maximal_recall,
)


def _extract_candidates(cmd: str) -> set[str]:
    out: set[str] = set()
    for extractor in _EXTRACTORS:
        out |= extractor(cmd)
    return out


def _clean(token: str) -> str:
    return token.strip().strip("'\"").rstrip(".,;:)").lstrip("(")


# ---------------------------------------------------------------------------
# Adjudicator -- decides genuine-absent-path vs false-positive
# ---------------------------------------------------------------------------

def _resolves_on_disk(repo_root: Path, token: str) -> bool:
    """repo-root-relative, frontend-relative, and frontend/src-relative
    resolution (the node-named false-positive family: `lib/icons.ts` only
    resolves under `frontend/src/`)."""
    return (
        (repo_root / token).exists()
        or (repo_root / "frontend" / token).exists()
        or (repo_root / "frontend/src" / token).exists()
        or (repo_root / "backend" / token).exists()
    )


def fp_reason(
    token: str,
    cmd: str,
    repo_root: Path,
    repo_basenames: set[str] | None = None,
) -> str | None:
    """Return a false-positive class string if `token` is NOT a genuine
    command-breaking absent path; return None if it IS genuine (i.e. an
    absent path the command actually depends on)."""
    t = _clean(token)
    if not t:
        return "empty"

    # well-formedness gate: a real path token is clean path chars only
    # (leading dot and glob metacharacters allowed).
    if not re.match(r"^[.A-Za-z0-9_][A-Za-z0-9_./+*?\[\]-]*$", t):
        return "malformed-token"

    # boundary gate: the token must appear as a WHOLE path in the command,
    # not a mis-split fragment of a neighboring token (guards extractors
    # C/D against `python -c` inline-code noise).
    if not re.search(r"(?<![A-Za-z0-9_./+-])" + re.escape(t) + r"(?![A-Za-z0-9./+-])", cmd):
        return "mis-split"

    if "://" in t or t.startswith(("http", "localhost", "127.0.0.1")):
        return "url"
    if re.search(r"(localhost|127\.0\.0\.1)[:/0-9A-Za-z._?=&%-]*" + re.escape(t), cmd):
        return "url"
    if t.startswith("/"):
        return "url-route" if not t.startswith(_ABS_HOST_PREFIXES) else "abs-host-path"
    if t.startswith(_RUNTIME_TRANSIENT_PREFIXES):
        return "runtime/transient"

    # glob-prefix re-resolution: the extractor truncated a glob at the
    # metacharacter (e.g. `alt_data_ic_*.tsv` -> token `alt_data_ic_`);
    # re-glob before calling it absent (the 7.12 hard case).
    escaped = re.escape(t)
    if re.search(escaped + r"[*?\[]", cmd):
        try:
            if list(repo_root.glob(t + "*")) or list((repo_root / "frontend").glob(t + "*")):
                return "glob-prefix-matches"
        except (OSError, ValueError):
            return "malformed-glob"

    # negative-assertion detection (shell + python), full-path-aware so a
    # bare basename token still matches an assertion on its full path.
    negation_patterns = (
        r"!\s*test\s+-[a-zA-Z]+\s+" + escaped,
        r"test\s+!\s+-[a-zA-Z]+\s+" + escaped,
        r"\[\s*!\s+-[a-zA-Z]+\s+" + escaped,
        r"test\s+-[a-zA-Z]+\s+" + escaped + r"\s*\|\|",
        r"assert\s+not\s+(?:any\()?os\.path\.exists\([^)]*" + escaped,
        r"not\s+os\.path\.exists\([^)]*" + escaped,
    )
    for pat in negation_patterns:
        if re.search(pat, cmd):
            return "absence-asserted"

    # tolerated-missing: an exists-guarded read with an empty-string else
    # branch (`open(p) if os.path.exists(p) else ''`).
    if re.search(r"os\.path\.exists\([^)]*\)\s+else\s+['\"]{2}", cmd) and re.search(escaped, cmd):
        return "absence-tolerated(else-empty)"

    # grep search-PATTERN: the path text appears only inside a quoted grep
    # pattern, not as a file argument.
    if re.search(r"grep[^|;&]*['\"][^'\"]*" + escaped + r"[^'\"]*['\"]", cmd):
        if not re.search(r"grep\b[^|;&]*\s" + escaped + r"(\s|$|\|)", cmd):
            return "grep-search-pattern"

    # shell variable expansion ($f.md etc.)
    if re.search(r"\$\w*" + escaped, cmd) or "$" in token:
        return "shell-var"

    # wildcard glob that matches something
    if any(c in t for c in "*?["):
        try:
            return "glob-matches" if (list(repo_root.glob(t)) or list((repo_root / "frontend").glob(t))) else None
        except (OSError, ValueError):
            return "malformed-glob"

    if _resolves_on_disk(repo_root, t):
        return "exists-on-disk"
    if "/" not in t and repo_basenames and t in repo_basenames:
        return "basename-exists-elsewhere"
    if "/" not in t and not re.search(r"\.%s$" % _EXT, t):
        return "bare-word-no-ext"
    if "%s" in token or re.search(r"%s\." + re.escape(t.rsplit(".", 1)[-1]), cmd):
        return "printf-template"

    return None  # GENUINE absent path


def _git_ls_files(repo_root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files"],
        capture_output=True, text=True, check=False,
    )
    return set(result.stdout.split())


def git_classify(path: str, repo_root: Path) -> tuple[str, str]:
    """('never-existed'|'retired'|'in-history-absent(runtime?)', commit-or-'')."""
    added = subprocess.run(
        ["git", "-C", str(repo_root), "log", "--all", "--diff-filter=A", "--oneline", "--", path],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    deleted = subprocess.run(
        ["git", "-C", str(repo_root), "log", "--all", "--diff-filter=D", "--oneline", "--", path],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    if not added:
        return "never-existed", ""
    if deleted:
        return "retired", deleted.splitlines()[0][:12]
    return "in-history-absent(runtime?)", ""


# ---------------------------------------------------------------------------
# classify() -- the importable core (75.19 preflight-gate reuse target)
# ---------------------------------------------------------------------------

def flat_steps(masterplan: dict) -> list[dict]:
    return [s for ph in masterplan.get("phases", []) for s in (ph.get("steps") or [])]


def shape_census(masterplan: dict) -> dict[str, int]:
    census = {"dict": 0, "str": 0, "list": 0, "none": 0}
    for st in flat_steps(masterplan):
        v = st.get("verification")
        if isinstance(v, dict):
            census["dict"] += 1
        elif isinstance(v, str):
            census["str"] += 1
        elif isinstance(v, list):
            census["list"] += 1
        else:
            census["none"] += 1
    return census


def classify(
    masterplan: dict,
    repo_root: Path,
    *,
    git_classify_fn=git_classify,
    repo_basenames: set[str] | None = None,
) -> dict:
    """Pure-ish core: given a parsed masterplan dict and a repo root, return
    the genuine absent-path defect census.

    A step already carrying a `superseded_record` sibling is treated as
    DISPOSITIONED and excluded -- this is what makes the genuine set go to
    empty once every defect in this census has been annotated, while the
    same function still reports the full genuine set against an older
    (pre-annotation) masterplan snapshot.

    Only `status == "done"` steps are scanned (a pending/deferred step
    naming a not-yet-created artifact is not a defect).

    Returns:
        {
          "genuine": {step_id: [{"path", "class", "retired_by_commit"}]},
          "shape_census": {"dict", "str", "list", "none"},
          "steps_scanned": int,
        }
    """
    if repo_basenames is None:
        repo_basenames = _git_ls_files(repo_root)
        repo_basenames = {p.split("/")[-1] for p in repo_basenames}

    genuine: dict[str, list[dict]] = {}
    steps_scanned = 0
    for st in flat_steps(masterplan):
        if st.get("status") != "done":
            continue
        if "superseded_record" in st:
            continue
        steps_scanned += 1
        sid = st["id"]
        for cmd in verif_commands(st.get("verification")):
            for tok in _extract_candidates(cmd):
                t = _clean(tok)
                if not t:
                    continue
                reason = fp_reason(tok, cmd, repo_root, repo_basenames)
                if reason:
                    continue
                gc, rc = git_classify_fn(t, repo_root)
                genuine.setdefault(sid, []).append(
                    {"path": t, "class": gc, "retired_by_commit": rc or None}
                )

    return {
        "genuine": genuine,
        "shape_census": shape_census(masterplan),
        "steps_scanned": steps_scanned,
    }


def load_masterplan(path: Path) -> dict:
    """No try/except around json.loads: a truncated/malformed masterplan
    must raise, never silently degrade to an empty census (M1)."""
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masterplan", default=str(REPO_ROOT / ".claude/masterplan.json"))
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root)
    masterplan = load_masterplan(Path(args.masterplan))
    result = classify(masterplan, repo_root)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        sc = result["shape_census"]
        print(
            f"phase-75.17 sweep: {result['steps_scanned']} done/unannotated steps scanned; "
            f"shape census dict={sc['dict']} str={sc['str']} list={sc['list']} none={sc['none']}",
            file=sys.stderr,
        )
        if not result["genuine"]:
            print("phase-75.17 sweep: CLEAN -- no genuine absent-path defects", file=sys.stderr)
        for sid, rows in sorted(result["genuine"].items()):
            for row in rows:
                commit = f" ret={row['retired_by_commit']}" if row["retired_by_commit"] else ""
                print(f"  {sid:10s} {row['class']:12s}{commit}  {row['path']}", file=sys.stderr)

    return 1 if result["genuine"] else 0


if __name__ == "__main__":
    sys.exit(main())
