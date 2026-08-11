#!/usr/bin/env python3
"""phase-86.44 mutation matrix.

Three guards, three cells, and every cell is gated on its own control being
GREEN first -- a cell scored against a broken instrument is UNSCORABLE, not
KILLED (the lesson from 86.33, where a checker read stderr the hook had
redirected into a file and went red against a healthy guard).

The D1 cell drives the REAL producer under REAL concurrent processes rather
than reasoning about the code, because a read-modify-write looks completely
fine in source and only loses data when two writers interleave.
"""
from __future__ import annotations

import multiprocessing as mp
import pathlib
import re
import shutil
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUN_HARNESS = ROOT / "scripts/harness/run_harness.py"
BACKTEST_API = ROOT / "backend/api/backtest.py"
RUNBOOK = ROOT / "docs/runbooks/per-step-protocol.md"
HARNESS_LOG = ROOT / "handoff/harness_log.md"

N_WRITERS = 12
N_EACH = 6


# ── the fixture the real producer needs ──────────────────────────────
def _fixture(cycle: int) -> dict:
    return dict(
        cycle=cycle,
        plan={"hypothesis": f"h{cycle}"},
        generator_result={
            "num_trials": 1, "initial_sharpe": 0.0, "final_sharpe": 0.0,
            "delta": 0.0, "kept": False, "elapsed_sec": 0.0,
        },
        grades={
            "verdict": "PASS", "composite": 1,
            "statistical_validity": {"score": 1}, "robustness": {"score": 1},
            "simplicity": {"score": 1}, "reality_gap": {"score": 1},
            "sub_sharpes": {}, "cost_2x_sharpe": 0.0,
        },
        elapsed=0.0,
    )


def _writer(args):
    """Run in a CHILD PROCESS: import the production module and append."""
    target, lo, hi = args
    sys.path.insert(0, str(ROOT))
    import importlib
    mod = importlib.import_module("scripts.harness.run_harness")
    mod.HARNESS_LOG = pathlib.Path(target)
    # Stub ONE unrelated dependency: the real _reconciliation_log_line opens a
    # BigQuery client per call, so 72 appends took >120s and hit live services.
    # The subject under test -- the file write -- is NOT stubbed. Only the string
    # that gets interpolated into the entry body is.
    mod._reconciliation_log_line = lambda: "- Reconciliation: stubbed (86.44 matrix)"
    for c in range(lo, hi):
        try:
            mod.append_harness_log(**_fixture(c))
        except Exception:
            pass
    return True


def check_d1_concurrent_append() -> tuple[bool, str]:
    """Drive the real producer from N processes; every entry must survive."""
    with tempfile.TemporaryDirectory() as td:
        target = pathlib.Path(td) / "harness_log.md"
        # Seed with the REAL log's bulk. A read-modify-write race is only lost
        # when the read+write takes long enough for another writer to interleave,
        # and that is a function of FILE SIZE. Seeding a 14-byte file would make
        # the mutant look safe -- a false negative that flatters the fix.
        target.write_text(HARNESS_LOG.read_text(), encoding="utf-8")
        _seeded = len(re.findall(r"^## Cycle \d+ --", target.read_text(), re.M))
        jobs = [(str(target), i * N_EACH, (i + 1) * N_EACH) for i in range(N_WRITERS)]
        with mp.Pool(N_WRITERS) as pool:
            pool.map(_writer, jobs)
        got = len(re.findall(r"^## Cycle \d+ --", target.read_text(), re.M)) - _seeded
    want = N_WRITERS * N_EACH
    return got == want, (f"{got}/{want} new entries survived {N_WRITERS} concurrent "
                         f"writers against a {_seeded}-cycle seed")


def check_d2_parser_lossless() -> tuple[bool, str]:
    """Drive the REAL endpoint function against the REAL log."""
    sys.path.insert(0, str(ROOT))
    for m in [k for k in sys.modules if k.startswith("backend.api.backtest")]:
        del sys.modules[m]
    import importlib
    bt = importlib.import_module("backend.api.backtest")
    parsed = bt.get_harness_log()["cycles"]
    on_disk = len(re.findall(r"^## Cycle ", HARNESS_LOG.read_text(), re.M))
    return len(parsed) == on_disk, f"parser returned {len(parsed)} of {on_disk} headers"


# The D3 population is DERIVED by git grep, not pinned.
#
# Cycle 1 found the guard scanning one file while the class had two. I "fixed" that
# by pinning a two-file list -- and cycle 2 then derived FIVE live occurrences. A
# pinned list can only ever be as complete as the last person who edited it, and it
# cannot fail on a NEW file that acquires the literal. So the population is now the
# git-grep result minus a NAMED allowlist, and every exclusion carries its reason.
#
# The allowlist is deliberately tiny and explicit:
#   - this file: it carries the literal as mutation PAYLOADS, by construction.
#   - the dated audit record: editing a 2026-05-12 audit to tidy a template is the
#     same category error as renumbering history, which criterion 4 declined to do.
D3_ALLOWED = {
    "scripts/qa/mutation_matrix_86_44.py",
    "docs/audits/phase-24-2026-05-12/24.0-charter-findings.md",
}
D3_EXCLUDE_PATHSPECS = [
    ":!handoff/harness_log.md", ":!handoff/archive/", ":!handoff/current/",
    ":!.claude/agent-memory/", ":!.claude/masterplan.json",
]


def check_d3_runbook_placeholder() -> tuple[bool, str]:
    """Derive every live tracked file carrying the bare `## Cycle N` template."""
    import subprocess as sp
    r = sp.run(["git", "grep", "-l", "--", "## Cycle N -- YYYY-MM-DD", "."] + D3_EXCLUDE_PATHSPECS,
               capture_output=True, text=True, cwd=ROOT)
    # git grep exits 1 on no-match, which is the healthy state -- not an error.
    if r.returncode not in (0, 1):
        return False, f"git grep failed rc={r.returncode}: {r.stderr.strip()[:120]}"
    found = {ln.strip() for ln in r.stdout.splitlines() if ln.strip()}
    offenders = sorted(found - D3_ALLOWED)
    # Anti-vacuity: the allowlist must actually be present, or the derivation is
    # silently scanning nothing (a rename would make this guard pass on an empty set).
    missing_allowed = sorted(a for a in D3_ALLOWED if not (ROOT / a).exists())
    if missing_allowed:
        return False, f"allowlisted path(s) missing -- population unverifiable: {missing_allowed}"
    return not offenders, (f"{len(offenders)} unallowed live file(s) carry the bare template "
                           f"(derived from {len(found)} git-grep hits, {len(D3_ALLOWED)} allowlisted)"
                           + (f" {offenders}" if offenders else ""))


CHECKS = {
    "d1_concurrent_append": check_d1_concurrent_append,
    "d2_parser_lossless": check_d2_parser_lossless,
    "d3_runbook_placeholder": check_d3_runbook_placeholder,
}

MUTANTS = {
    "M1_revert_d1_to_read_modify_write": (
        RUN_HARNESS,
        '    _new_file = not HARNESS_LOG.exists()\n'
        '    with open(HARNESS_LOG, "a", encoding="utf-8") as fh:\n'
        '        if _new_file:\n'
        '            fh.write(\n'
        '                "# Harness Log\\n\\nAutomated three-agent harness loop. "\n'
        '                "Each cycle: Planner -> Generator -> Evaluator.\\n"\n'
        '            )\n'
        '        fh.write(entry)\n',
        '    existing = HARNESS_LOG.read_text(encoding="utf-8") if HARNESS_LOG.exists() else ""\n'
        '    HARNESS_LOG.write_text(existing + entry, encoding="utf-8")\n',
        "d1_concurrent_append",
    ),
    "M2_revert_d2_to_digits_only": (
        BACKTEST_API,
        r'parts = re.split(r"^## (Cycle [^\n]+?)\s*--\s*(.+)$", content, flags=re.MULTILINE)',
        r'parts = re.split(r"^## (Cycle \d+)\s*--\s*(.+)$", content, flags=re.MULTILINE)',
        "d2_parser_lossless",
    ),
    "M3_restore_the_trap_in_the_RUNBOOK": (
        RUNBOOK,
        "## Cycle <N> -- YYYY-MM-DD -- phase=X.Y result=PASS/CONDITIONAL/FAIL",
        "## Cycle N -- YYYY-MM-DD -- phase=X.Y result=PASS/CONDITIONAL/FAIL",
        "d3_runbook_placeholder",
    ),
    # One cell per member of the class. Without this, deleting the CLAUDE.md fix
    # leaves the matrix green -- which is exactly the state the cycle-1 Q/A found.
    "M4_restore_the_trap_in_CLAUDE_md": (
        ROOT / "CLAUDE.md",
        "`## Cycle <N> -- YYYY-MM-DD -- phase=X.Y result=PASS/CONDITIONAL/FAIL`",
        "`## Cycle N -- YYYY-MM-DD -- phase=X.Y result=PASS/CONDITIONAL/FAIL`",
        "d3_runbook_placeholder",
    ),
}


def run_all() -> dict:
    return {k: fn() for k, fn in CHECKS.items()}


def main() -> int:
    print("=" * 74)
    print("CONTROL -- every check must be GREEN before any cell is scored")
    print("=" * 74)
    control = run_all()
    for k, (ok, detail) in control.items():
        print(f"  {'GREEN' if ok else 'RED  '}  {k:26s} {detail}")
    if not all(ok for ok, _ in control.values()):
        print("\nCONTROL IS RED -> every cell is UNSCORABLE. Fix the instrument first.")
        return 2

    print()
    results = {}
    restore_failed: list[str] = []
    for name, (path, present, replacement, target) in MUTANTS.items():
        original = path.read_text()
        if present not in original:
            print(f"  ANCHOR-MISS  {name}: text not found in {path.name} -- NOT scored")
            results[name] = "ANCHOR-MISS"
            continue
        if original.count(present) != 1:
            print(f"  ANCHOR-AMBIGUOUS {name}: {original.count(present)} occurrences -- NOT scored")
            results[name] = "ANCHOR-AMBIGUOUS"
            continue
        backup = tempfile.NamedTemporaryFile(delete=False, suffix=path.suffix)
        backup.write(original.encode()); backup.close()
        try:
            path.write_text(original.replace(present, replacement, 1))
            after = run_all()
            ok, detail = after[target]
            others_ok = all(v[0] for k, v in after.items() if k != target)
            if not ok and others_ok:
                verdict = "KILLED"
            elif not ok:
                verdict = "KILLED (but other checks also RED -- MIS-ATTRIB risk)"
            else:
                verdict = "SURVIVED"
            results[name] = verdict
            print(f"  {verdict:12s} {name}\n               -> {target}: {detail}")
        finally:
            shutil.copyfile(backup.name, path)
            pathlib.Path(backup.name).unlink(missing_ok=True)
            identical = path.read_text() == original
            print(f"               restore byte-identical: {identical}")
            if not identical:
                # NOT `return` here: a return inside `finally` swallows any exception
                # propagating out of the try, which would hide the very failure that
                # left the tree dirty. Flag it and let the loop end normally.
                print("               !! RESTORE FAILED -- tree is dirty, fix before committing")
                restore_failed.append(path.name)

    print()
    post = run_all()
    print("POST-RESTORE control:", {k: v[0] for k, v in post.items()})
    if restore_failed:
        print(f"\nRESTORE FAILED on {restore_failed} -- tree is dirty")
        return 3
    all_killed = all(str(v).startswith("KILLED") for v in results.values())
    print(f"\nALL CELLS KILLED: {all_killed}")
    return 0 if all_killed and all(ok for ok, _ in post.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
