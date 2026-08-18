#!/usr/bin/env python3
"""phase-86.116 criterion 7 -- mutation matrix against the REAL file and suite.

Scoring rules, each one paid for by a previous step:

* the CONTROL is observed GREEN first; if it is not, every cell is UNSCORABLE;
* a non-zero exit is NOT a kill -- pytest exits 5 on "no tests collected", and a
  typo in `-k` would otherwise score as a success. The mutant must exit **1**;
* the mutant must COLLECT THE SAME NUMBER OF TESTS as the control, so a mutation
  that breaks import (collecting zero) is UNSCORABLE rather than a kill;
* the NAMED test must be among the failures -- "something went red" is not
  evidence that the guard for this defect fired;
* restore is verified by SHA-256 against the pre-mutation bytes;
* SIGTERM/SIGINT/SIGHUP restore the file, and the run REFUSES TO START from a
  target that already contains a `MUTANT` marker. (A 2-minute timeout stranded a
  mutant on disk during step 86.59; `try/finally` does not run on a signal.)

An EQUIVALENT-BY-DESIGN cell is declared as such UP FRONT with its evidence, and
is not counted as a kill or a survivor. Silently omitting it would hide a real
property of the fix.

Run: ``python scripts/qa/mutation_86_116.py``
"""

from __future__ import annotations

import hashlib
import re
import signal
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGET = REPO / "backend" / "backtest" / "cache.py"
SUITE = "backend/tests/test_phase_86_116_price_dedup.py"

# (cell, old, new, named test that must fail)
CELLS: list[tuple[str, str, str, str]] = [
    (
        "M1 preload_prices stops de-duplicating -> the bulk path hands out a doubled index",
        "        _prices_full[str(ticker)] = _dedupe_index(\n"
        "            group.drop(columns=[\"ticker\"]).set_index(\"date\").sort_index(),\n"
        "            str(ticker),\n"
        "        )",
        "        _prices_full[str(ticker)] = (\n"
        "            group.drop(columns=[\"ticker\"]).set_index(\"date\").sort_index()\n"
        "        )  # MUTANT",
        "test_preload_prices_returns_a_unique_index",
    ),
    (
        "M2 cached_prices stops de-duplicating -> the per-ticker fallback path leaks",
        "        df = _dedupe_index(df.set_index(\"date\").sort_index(), ticker)",
        "        df = df.set_index(\"date\").sort_index()  # MUTANT",
        "test_cached_prices_returns_a_unique_index",
    ),
    (
        "M3 index-keyed dedup swapped for VALUE-keyed drop_duplicates()",
        "    out = df[~df.index.duplicated(keep=\"first\")]",
        "    out = df.drop_duplicates()  # MUTANT",
        "test_value_keyed_dedup_is_insufficient",
    ),
    (
        "M4 _dedupe_index made a pass-through -> the fix is present but inert",
        "    if df is None or df.empty or df.index.is_unique:\n        return df",
        "    if True:\n        return df  # MUTANT",
        "test_dedupe_removes_duplicate_index_entries",
    ),
    (
        "M5 inertness guard inverted -> the fix fires only when there is nothing to do",
        "    if df is None or df.empty or df.index.is_unique:",
        "    if df is None or df.empty or not df.index.is_unique:  # MUTANT",
        "test_dedupe_removes_duplicate_index_entries",
    ),
    (
        "M6 empty/None guard dropped -> the loader crashes on an empty result set",
        "    if df is None or df.empty or df.index.is_unique:",
        "    if df.index.is_unique:  # MUTANT",
        "test_dedupe_handles_empty_and_none",
    ),
]

# Declared UP FRONT, with the measurement that justifies it.
EQUIVALENT_BY_DESIGN: list[tuple[str, str, str, str]] = [
    (
        "E1 keep='first' -> keep='last'",
        "    out = df[~df.index.duplicated(keep=\"first\")]",
        "    out = df[~df.index.duplicated(keep=\"last\")]  # MUTANT",
        "EQUIVALENT BY DESIGN: which of two same-date rows survives is "
        "immaterial -- measured across the 394,719 duplicated keys whose close "
        "differs, the gap is 0.0% at BOTH p50 and p99 with a 0.93% maximum. The "
        "code comment says the choice is immaterial, so a cell asserting "
        "otherwise would contradict the shipped rationale. Declared rather than "
        "omitted, and scored as neither a kill nor a survivor.",
    ),
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_suite() -> tuple[int, int, str]:
    """Run the real suite. Returns (returncode, collected_count, output)."""
    r = subprocess.run(
        [sys.executable, "-m", "pytest", SUITE, "-q", "--no-header"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    out = r.stdout + r.stderr
    m = re.search(r"(\d+) passed", out)
    f = re.search(r"(\d+) failed", out)
    collected = (int(m.group(1)) if m else 0) + (int(f.group(1)) if f else 0)
    return r.returncode, collected, out


def main() -> int:
    original = TARGET.read_bytes()
    base_sha = hashlib.sha256(original).hexdigest()

    if b"MUTANT" in original:
        print("REFUSING TO RUN: the target already contains a stranded MUTANT "
              "marker. Restore it before scoring anything -- a matrix run from "
              "a poisoned baseline is not a measurement.")
        return 2

    def _restore_and_die(signum, _frame):
        TARGET.write_bytes(original)
        print(f"\nsignal {signum} -- target restored, matrix aborted")
        raise SystemExit(130)

    for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(_sig, _restore_and_die)
        except (ValueError, OSError):
            pass

    print("=" * 78)
    print("phase-86.116 -- MUTATION MATRIX (criterion 7)")
    print("=" * 78)
    print(f"target : {TARGET.relative_to(REPO)}")
    print(f"suite  : {SUITE}")
    print(f"sha256 : {base_sha[:16]}...")
    print()

    rc, control_n, out = run_suite()
    print(f"control -> rc={rc}  collected={control_n}  "
          f"{'GREEN' if rc == 0 else 'RED'}")
    if rc != 0:
        print("CONTROL IS NOT GREEN -- every cell is UNSCORABLE.")
        print(out[-1500:])
        return 1
    print()

    results = []
    for name, old, new, named in CELLS:
        text = original.decode()
        if text.count(old) != 1:
            results.append((name, "UNSCORABLE",
                            f"anchor appears {text.count(old)}x, expected 1"))
            print(f"[UNSCORABLE] {name}\n             anchor not unique")
            continue
        TARGET.write_text(text.replace(old, new, 1))
        try:
            mrc, mn, mout = run_suite()
        finally:
            TARGET.write_bytes(original)
            assert sha(TARGET) == base_sha, "RESTORE FAILED -- tree is dirty"

        if mrc == 0:
            verdict, detail = "SURVIVED", "suite stayed green under the mutation"
        elif mrc != 1:
            verdict, detail = "UNSCORABLE", (
                f"pytest exited {mrc}, not 1 -- exit 5 is 'no tests collected' "
                f"and must never score as a kill")
        elif mn != control_n:
            verdict, detail = "UNSCORABLE", (
                f"collected {mn} tests vs {control_n} in control -- the mutant "
                f"changed what runs, so the comparison is not like-for-like")
        elif named not in mout:
            verdict, detail = "UNSCORABLE", (
                f"suite went red but `{named}` is not among the failures -- "
                f"something else broke and this guard was never shown to fire")
        else:
            verdict, detail = "KILLED", f"`{named}` failed (rc=1, collected={mn})"
        results.append((name, verdict, detail))
        print(f"[{verdict}] {name}\n           {detail}")

    print()
    for name, old, new, why in EQUIVALENT_BY_DESIGN:
        print(f"[EQUIVALENT-BY-DESIGN] {name}")
        for line in (why[i:i + 72] for i in range(0, len(why), 72)):
            print(f"           {line}")

    print()
    print("-" * 78)
    killed = sum(1 for _, v, _ in results if v == "KILLED")
    survived = [n for n, v, _ in results if v == "SURVIVED"]
    unscorable = [n for n, v, _ in results if v == "UNSCORABLE"]
    print(f"KILLED {killed} / {len(results)}   SURVIVED {len(survived)}   "
          f"UNSCORABLE {len(unscorable)}   "
          f"EQUIVALENT-BY-DESIGN {len(EQUIVALENT_BY_DESIGN)} (not scored)")
    for n in survived:
        print(f"  SURVIVED: {n}")
    for n in unscorable:
        print(f"  UNSCORABLE: {n}")
    print(f"restore verified: sha256 unchanged ({sha(TARGET)[:16]}...)")
    return 0 if not survived and not unscorable else 1


if __name__ == "__main__":
    raise SystemExit(main())
