"""Pre-flight verifier for masterplan verification commands.

phase-16.38 (#29) original: walk `.claude/masterplan.json`, extract every
step's `verification` command, and statically check that referenced file
paths and Python imports exist -- no command execution (no side effects,
no cost, no flake).

phase-75.19 RECALIBRATION. The original was status-blind, annotation-blind,
container-blind, and used an ad-hoc path heuristic; on 2026-07-24 it
reported "863 steps, 151 broken, 8 unparseable" across 222 BROKEN lines
with a true genuine residue of ZERO -- ~100% effective false positives.
A gate that noisy gets ignored (Sadowski et al., CACM 2018: checks survive
only under ~10% effective-FP; see research_brief_75.19.md). What changed:

  1. STATUS-AWARE: only `status == "done"` steps can be reported broken.
     A pending/deferred/dropped/superseded step naming a not-yet-created
     artifact is not a defect; those land in excluded buckets.
  2. ANNOTATION-AWARE: a step carrying a `superseded_record` sibling is
     DISPOSITIONED (75.2.1 shape) and excluded, mirroring the 75.17 sweep.
  3. CONTAINER-EXPLICIT walk: scans `phases[].steps[]` AND
     `phases[].subphases[]` (live steps; the old recursive walk's ids all
     resolve, but the 75.17 sweep's `flat_steps` missed subphases);
     EXCLUDES `archived_legacy_steps[]` / `archived_dropped_steps[]`
     (archive duplicates) and everything under `superseded_record`.
     Excluded containers are counted, so the narrowing is auditable.
  4. ADJUDICATION REUSED from `scripts.qa.sweep_absent_verification_paths`
     (the 75.17 charter: "classify() is the IMPORTABLE core -- step 75.19
     is chartered to reuse it"). We import the pipeline pieces
     (`verif_commands`, `_extract_candidates`, `_clean`, `fp_reason`,
     `git_classify`) rather than calling `classify()` itself because its
     `flat_steps` walk misses `subphases[]` (recall gap measured
     2026-07-24) and it has no import leg. No adjudicator is re-implemented
     here -- a second copy would drift from the 75.17 census.
  5. IMPORT LEG kept from the original (the sweep is path-only; imports
     are load-bearing), gated behind the same status/annotation filters.
  6. SHLEX-INDEPENDENT scanning: candidate extraction is regex-based, so a
     command `shlex` cannot tokenize (nested quotes) is STILL scanned; the
     failure is reported in its own `shlex-untokenizable` bucket, never as
     "broken". The commands themselves are immutable and stay untouched.
  7. INTERNALLY-CONSISTENT summary: emitted [GENUINE] lines, the distinct
     step count they span, and every excluded bucket appear in one summary
     whose counts are self-checked against the emitted rows before exit
     (`check_consistency`); any mismatch is INTERNAL-INCONSISTENCY, exit 2.
     Per-step de-duplication removes the old duplicate-line emission
     (e.g. step 8.4 was emitted twice for one path).

Verification shapes handled (via `verif_commands`): dict / str / list /
None. The original's `_extract_command` silently dropped the 13
list-shaped verifications.

Usage:
    python scripts/meta/preflight_verify_masterplan.py .claude/masterplan.json
    python scripts/meta/preflight_verify_masterplan.py .claude/masterplan.json --quiet
    python scripts/meta/preflight_verify_masterplan.py .claude/masterplan.json --json

Exit codes:
  0 = no GENUINE broken refs on done+unannotated steps
  1 = one or more GENUINE broken refs
  2 = filesystem / JSON parse error, or internal summary inconsistency
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shlex
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.qa.sweep_absent_verification_paths import (
    _clean,
    _extract_candidates,
    _git_ls_files,
    fp_reason,
    git_classify,
    verif_commands,
)

# Container semantics (measured against the live masterplan 2026-07-24:
# steps=859, subphases=13, archived_legacy_steps=3, archived_dropped_steps=1
# verification-bearing entries).
LIVE_STEP_CONTAINERS = frozenset({"steps", "subphases"})
ARCHIVE_CONTAINERS = frozenset({"archived_legacy_steps", "archived_dropped_steps"})

VENV_PREFIX_RE = re.compile(r"^source\s+\.venv/bin/activate\s*&&\s*", re.MULTILINE)

IMPORT_RE = re.compile(
    r"\b(?:from\s+([a-zA-Z_][\w.]*)\s+import|import\s+([a-zA-Z_][\w.]*))"
)


def _strip_venv_prefix(cmd: str) -> str:
    return VENV_PREFIX_RE.sub("", cmd)


def _extract_imports(cmd: str) -> set[str]:
    """Pull out `from X.Y import` and `import X.Y` module names."""
    out: set[str] = set()
    for m in IMPORT_RE.finditer(cmd):
        mod = m.group(1) or m.group(2)
        if mod:
            out.add(mod.split(",")[0].strip())
    return out


def _check_imports(modules: set[str], repo_root: Path) -> list[str]:
    """Return the unimportable dotted module names (bare names skipped --
    stdlib or top-level installed packages are not this gate's business)."""
    broken: list[str] = []
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    for mod in sorted(modules):
        if "." not in mod:
            continue
        try:
            spec = importlib.util.find_spec(mod)
        except (ModuleNotFoundError, ImportError, ValueError, TypeError):
            broken.append(mod)
            continue
        if spec is None:
            broken.append(mod)
    return broken


def iter_steps(node: Any, _container: str | None = None) -> Iterator[tuple[str, dict]]:
    """Yield ("live" | "archived", step_dict) for every dict carrying both
    an `id` and a `verification`, honoring container semantics.

    - Entered via a LIVE_STEP_CONTAINERS key  -> "live"
    - Entered via an ARCHIVE_CONTAINERS key   -> "archived" (sticky)
    - `superseded_record` values are annotations, never steps: not descended.
    - Dicts outside any step container (e.g. the phase list itself) are not
      yielded; measured: zero phases carry their own verification.
    """
    if isinstance(node, dict):
        if _container is not None and node.get("id") is not None \
                and node.get("verification") is not None:
            yield (_container, node)
        for key, value in node.items():
            if key == "superseded_record":
                continue
            if _container == "archived":
                child: str | None = "archived"
            elif key in ARCHIVE_CONTAINERS:
                child = "archived"
            elif key in LIVE_STEP_CONTAINERS:
                child = "live"
            else:
                child = _container
            yield from iter_steps(value, child)
    elif isinstance(node, list):
        for item in node:
            yield from iter_steps(item, _container)


def build_report(
    masterplan: dict,
    repo_root: Path,
    *,
    git_classify_fn: Callable[[str, Path], tuple[str, str]] | None = None,
    repo_basenames: set[str] | None = None,
) -> dict:
    """The testable core: parsed masterplan in, structured report out.

    Returns:
      {
        "genuine": {sid: [{"kind": "path"|"import", "ref", "class",
                           "retired_by_commit"}]},
        "lines":   [formatted "[GENUINE] step=..." strings, 1:1 with entries],
        "buckets": {"archived_excluded", "annotated_excluded", "by_status",
                    "shlex_untokenizable": [(sid, err)]},
        "summary": {counts -- see check_consistency for the invariants},
      }
    """
    if git_classify_fn is None:
        git_classify_fn = git_classify
    if repo_basenames is None:
        repo_basenames = {p.split("/")[-1] for p in _git_ls_files(repo_root)}

    genuine: dict[str, list[dict]] = {}
    lines: list[str] = []
    by_status: dict[str, int] = {}
    shlex_untokenizable: list[tuple[str, str]] = []
    archived_excluded = 0
    annotated_excluded = 0
    live_total = 0
    scanned = 0

    for kind, step in iter_steps(masterplan):
        if kind == "archived":
            archived_excluded += 1
            continue
        live_total += 1
        sid = str(step.get("id"))
        status = str(step.get("status") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        if status != "done":
            continue
        if "superseded_record" in step:
            annotated_excluded += 1
            continue
        scanned += 1
        seen: set[tuple[str, str]] = set()  # per-step de-dup (kind, ref)
        for cmd in verif_commands(step.get("verification")):
            cmd_s = _strip_venv_prefix(cmd)
            try:
                shlex.split(cmd_s, posix=True)
            except ValueError as exc:
                shlex_untokenizable.append((sid, str(exc)))
            for tok in sorted(_extract_candidates(cmd_s)):
                t = _clean(tok)
                if not t or ("path", t) in seen:
                    continue
                if fp_reason(tok, cmd_s, repo_root, repo_basenames) is None:
                    seen.add(("path", t))
                    cls, commit = git_classify_fn(t, repo_root)
                    genuine.setdefault(sid, []).append(
                        {"kind": "path", "ref": t, "class": cls,
                         "retired_by_commit": commit or None}
                    )
                    lines.append(f"[GENUINE] step={sid}: absent path {t!r} ({cls})")
            for mod in _check_imports(_extract_imports(cmd_s), repo_root):
                if ("import", mod) in seen:
                    continue
                seen.add(("import", mod))
                genuine.setdefault(sid, []).append(
                    {"kind": "import", "ref": mod, "class": "unimportable",
                     "retired_by_commit": None}
                )
                lines.append(f"[GENUINE] step={sid}: unimportable module {mod!r}")

    summary = {
        "live_steps": live_total,
        "by_status": by_status,
        "scanned_done_unannotated": scanned,
        "annotated_excluded": annotated_excluded,
        "archived_excluded": archived_excluded,
        "genuine_lines": len(lines),
        "genuine_steps": len(genuine),
        "shlex_untokenizable": len(shlex_untokenizable),
    }
    return {
        "genuine": genuine,
        "lines": lines,
        "buckets": {
            "archived_excluded": archived_excluded,
            "annotated_excluded": annotated_excluded,
            "by_status": by_status,
            "shlex_untokenizable": shlex_untokenizable,
        },
        "summary": summary,
    }


def check_consistency(report: dict) -> list[str]:
    """Re-derive every summary count from the report's own rows; return the
    discrepancies (empty == internally consistent). This is the criterion-3
    guard: the summary can never again say something the rows don't."""
    problems: list[str] = []
    summary = report["summary"]
    genuine = report["genuine"]
    lines = report["lines"]
    n_entries = sum(len(v) for v in genuine.values())

    if summary["genuine_lines"] != len(lines):
        problems.append(
            f"summary.genuine_lines={summary['genuine_lines']} != emitted lines {len(lines)}"
        )
    if n_entries != len(lines):
        problems.append(f"genuine entries {n_entries} != emitted lines {len(lines)}")
    if summary["genuine_steps"] != len(genuine):
        problems.append(
            f"summary.genuine_steps={summary['genuine_steps']} != distinct steps {len(genuine)}"
        )
    line_ids = {m.group(1) for m in
                (re.search(r"step=([^:]+):", ln) for ln in lines) if m}
    if line_ids != set(genuine):
        problems.append(
            f"line step-ids {sorted(line_ids)} != genuine keys {sorted(genuine)}"
        )
    if summary["scanned_done_unannotated"] != (
        summary["by_status"].get("done", 0) - summary["annotated_excluded"]
    ):
        problems.append(
            "scanned_done_unannotated != by_status[done] - annotated_excluded"
        )
    if summary["live_steps"] != sum(summary["by_status"].values()):
        problems.append("live_steps != sum(by_status)")
    if summary["shlex_untokenizable"] != len(report["buckets"]["shlex_untokenizable"]):
        problems.append("summary.shlex_untokenizable != bucket length")
    return problems


def _format_summary(summary: dict) -> str:
    status_part = " ".join(
        f"{k}={v}" for k, v in sorted(summary["by_status"].items()) if k != "done"
    )
    return (
        "preflight_verify_masterplan (recalibrated phase-75.19): "
        f"live_steps={summary['live_steps']} "
        f"scanned(done+unannotated)={summary['scanned_done_unannotated']} "
        f"genuine={summary['genuine_lines']} lines across "
        f"{summary['genuine_steps']} steps; excluded: "
        f"archived={summary['archived_excluded']} "
        f"annotated(superseded_record)={summary['annotated_excluded']} "
        f"non-done{{{status_part}}}; "
        f"shlex-untokenizable(regex-scanned)={summary['shlex_untokenizable']}"
    )


def verify(masterplan_path: str | Path, *, quiet: bool = False,
           json_out: bool = False) -> int:
    p = Path(masterplan_path)
    if not p.exists():
        print(f"preflight_verify_masterplan: file not found: {p}", file=sys.stderr)
        return 2
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"preflight_verify_masterplan: JSON parse error: {e}", file=sys.stderr)
        return 2

    report = build_report(data, REPO_ROOT)
    problems = check_consistency(report)
    if problems:
        for prob in problems:
            print(f"[INTERNAL-INCONSISTENCY] {prob}", file=sys.stderr)
        return 2

    if json_out:
        print(json.dumps(report, indent=2))
        return 1 if report["genuine"] else 0

    for line in report["lines"]:
        print(line, file=sys.stderr)
    for sid, err in report["buckets"]["shlex_untokenizable"]:
        print(f"[NOTE] step={sid}: shlex-untokenizable ({err}); "
              "scanned via regex extractors", file=sys.stderr)
    if not quiet:
        print(_format_summary(report["summary"]))
    return 1 if report["genuine"] else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-flight: verify masterplan verification commands "
                    "reference live paths/modules without executing them "
                    "(status-aware, annotation-aware; phase-75.19)."
    )
    parser.add_argument("path", help="path to masterplan.json")
    parser.add_argument("--quiet", action="store_true",
                        help="no stdout summary (exit code only); GENUINE "
                             "lines still go to stderr")
    parser.add_argument("--json", action="store_true",
                        help="emit the full structured report as JSON on stdout")
    args = parser.parse_args(argv)
    return verify(args.path, quiet=args.quiet, json_out=args.json)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
