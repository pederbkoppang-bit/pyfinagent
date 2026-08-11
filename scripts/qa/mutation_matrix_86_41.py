#!/usr/bin/env python3
"""phase-86.41 mutation matrix -- is each assertion load-bearing?

A passing suite proves nothing on its own: it may be green because the guard
works, or green because nothing it asserts can fail. Each cell below breaks ONE
property of the guard and requires a NAMED test to go red. A cell that survives
is a hole in the suite, not a success.

Two disciplines this file exists to honour, both learned the hard way:

  * A cell must DISCRIMINATE. If the control answer and the mutant's fail-safe
    answer coincide, the cell "passes" while proving nothing. Every cell here
    names the specific test that must fail, and the runner rejects a cell whose
    reds are all in other tests (MIS-ATTRIB).
  * The control must be GREEN FIRST. If the unmutated suite is red, every
    subsequent red is meaningless.

Cells 2 and 3 are not hypothetical regressions: they are the two defects this
guard actually shipped with, both caught by driving the real pipeline rather
than reading the source. They are pinned here so they cannot come back.

Run:  python scripts/qa/mutation_matrix_86_41.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGET = REPO / "backend" / "agents" / "orchestrator.py"
SUITE = REPO / "backend" / "tests" / "test_phase_86_41_quant_isolation.py"

CELLS = [
    {
        "id": "M1-no-guard",
        "what": "remove the try/except entirely (the pre-86.41 state)",
        "old": """            try:
                # ONLY the sub-agent call belongs in the try. `step()` invokes a
                # caller-supplied progress callback; with it inside, a raising
                # SSE emitter was caught here, mislabelled a quant failure, and
                # silently overwrote a GOOD quant report with the yfinance
                # fallback. Caught by test_healthy_quant_is_untouched_by_the_guard.
                report["quant"] = await self.run_quant_agent(ticker)
            except Exception as _quant_exc:""",
        "new": """            if True:
                report["quant"] = await self.run_quant_agent(ticker)
            except_never = None
            if False:
                _quant_exc = None""",
        "expect_named": ["test_quant_failure_does_not_abort_the_ticker"],
    },
    {
        "id": "M2-plain-append",
        "what": "setdefault -> plain [] append (the KeyError defect, real)",
        "old": '                report.setdefault("skipped_stages", []).append({',
        "new": '                report["skipped_stages"].append({',
        "expect_named": ["test_quant_failure_does_not_abort_the_ticker"],
    },
    {
        "id": "M3-step-inside-try",
        "what": "move step() back inside the try (the over-broad-catch defect, real)",
        "old": """                report["quant"] = await self.run_quant_agent(ticker)
            except Exception as _quant_exc:""",
        "new": """                report["quant"] = await self.run_quant_agent(ticker)
                step("quant", "completed", "Financial data collected")
            except Exception as _quant_exc:""",
        "expect_named": ["test_healthy_quant_is_untouched_by_the_guard"],
    },
    {
        "id": "M4-impersonate-non-us",
        "what": "reuse phase-60.1's non-US reason for a rate-limit failure",
        "old": '                    f"Quant degraded -- yfinance fundamentals only ({type(_quant_exc).__name__})",',
        "new": '                    "Quant skipped (non-US listing) -- yfinance fundamentals only",',
        "expect_named": [
            "test_degraded_reason_does_not_impersonate_the_non_us_branch",
            "test_quant_failure_is_reported_as_degraded_and_names_the_cause",
        ],
    },
    {
        "id": "M5-default-reason-drift",
        "what": "change the phase-60.1 default reason string",
        "old": '    reason: str = "non-US listing, phase-60.1 KR-aware skip",',
        "new": '    reason: str = "non-US listing (phase-60.1)",',
        "expect_named": ["test_default_reason_is_byte_identical_to_phase_60_1"],
    },
]


def run_suite() -> tuple[bool, list[str]]:
    """Return (all_green, [names of failing tests])."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(SUITE), "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    out = proc.stdout + proc.stderr
    failed = []
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("FAILED ") or s.startswith("ERROR "):
            # "FAILED path::test_name - AssertionError: ..."
            frag = s.split(" ", 1)[1].split(" - ")[0]
            failed.append(frag.rsplit("::", 1)[-1])
    return (proc.returncode == 0 and not failed), failed


def main() -> int:
    src = TARGET.read_text()
    backup = Path(tempfile.gettempdir()) / "orchestrator_86_41_backup.py"
    backup.write_text(src)

    print("=" * 74)
    print("phase-86.41 mutation matrix")
    print("=" * 74)

    green, failed = run_suite()
    print(f"\n[control] unmutated suite green: {green}")
    if not green:
        print(f"  ABORT -- control is RED ({failed}); every later red would be meaningless.")
        return 1

    results = []
    try:
        for cell in CELLS:
            if cell["old"] not in src:
                # An anchor that does not match mutates nothing and the suite
                # stays green -- indistinguishable from a surviving mutant.
                results.append((cell["id"], "ANCHOR-MISS", cell["what"], []))
                print(f"\n[{cell['id']}] ANCHOR NOT FOUND -- cell is vacuous, not passing")
                continue
            if src.count(cell["old"]) != 1:
                results.append((cell["id"], "ANCHOR-AMBIGUOUS", cell["what"], []))
                print(f"\n[{cell['id']}] anchor matches {src.count(cell['old'])}x -- ambiguous")
                continue

            TARGET.write_text(src.replace(cell["old"], cell["new"], 1))
            _, failed = run_suite()
            named_red = [n for n in cell["expect_named"] if n in failed]
            other_red = [n for n in failed if n not in cell["expect_named"]]

            if named_red:
                status = "KILLED"
            elif failed:
                status = "MIS-ATTRIB"
            else:
                status = "SURVIVED"

            results.append((cell["id"], status, cell["what"], failed))
            print(f"\n[{cell['id']}] {status}  ({cell['what']})")
            print(f"    expected red: {cell['expect_named']}")
            print(f"    actual  red: {failed or '(none)'}")
            if status == "MIS-ATTRIB":
                print(f"    !! only unexpected tests failed: {other_red}")
    finally:
        # No `return` in this block: it would discard any in-flight exception
        # and report a restore failure as an ordinary exit code.
        TARGET.write_text(backup.read_text())
        restored = TARGET.read_text() == src
        print(f"\n[restore] target byte-identical to pre-mutation: {restored}")
        if not restored:
            print("    !! MANUAL RESTORE REQUIRED FROM", backup)

    if not restored:
        return 2

    print("\n" + "=" * 74)
    bad = [r for r in results if r[1] != "KILLED"]
    for cid, status, what, failed in results:
        print(f"  {status:<16} {cid:<24} {what}")
    print("=" * 74)
    if bad:
        print(f"RESULT: {len(bad)} of {len(results)} cells NOT killed -- the suite has holes.")
        return 1

    green_after, failed_after = run_suite()
    print(f"post-restore suite green: {green_after} {failed_after or ''}")
    print(f"RESULT: all {len(results)} cells KILLED, control green, target restored.")
    return 0 if green_after else 1


if __name__ == "__main__":
    raise SystemExit(main())
