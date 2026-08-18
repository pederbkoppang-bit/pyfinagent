#!/usr/bin/env python3
"""phase-86.110 criterion 6: mutation matrix for the heartbeat-isolation work.

Same strict scoring rule as the 86.108 and 86.109 matrices: control green
FIRST, pytest exit **1** (exit 5 -- nothing collected -- is not a kill), the
mutant must collect the SAME count as the control (a mutant that cannot build
is not a killed mutant), and the **named** test must be the failing one.

Cell **P3** is the one worth reading. The sweep's first implementation checked
`re.search("_HEARTBEAT_PATH", src)`, which the fix's own explanatory comments
satisfied -- so with the fix reverted the sweep still reported zero leaks. The
guard was matching its own documentation. P3 restores that substring check and
must go red, which is the only way to know the AST rewrite actually bought
something.
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[2]
SUITE = ["backend/tests/test_phase_86_110_heartbeat_isolation.py"]
PY = str(REPO / ".venv" / "bin" / "python")

RAIL = "backend/tests/test_phase_66_1_rail_guard.py"
SWEEP = "scripts/qa/heartbeat_leak_sweep_86_110.py"
CONFTEST = "backend/tests/conftest.py"

_HB_LINE = '    monkeypatch.setattr(ch, "_HEARTBEAT_PATH", tmp_path / ".cycle_heartbeat.json")\n'

CELLS = [
    ("P1", RAIL,
     "    # phase-86.110: see the sibling test above -- record_cycle_end writes the\n"
     "    # heartbeat too, so both constants must be isolated.\n" + _HB_LINE,
     "",
     "test_the_previously_leaking_tests_no_longer_touch_the_real_heartbeat",
     "reverting the isolation at the funnel-counts site is caught BEHAVIOURALLY"),

    ("P2", SWEEP,
     '            if name == "setattr":',
     '            if name == "NEVER_MATCHES":  # MUTANT: setattr no longer detected',
     "test_the_reachability_sweep_is_clean",
     "a sweep that stops recognising monkeypatch.setattr is caught"),

    ("P3", SWEEP,
     "    for node in _ast.walk(tree):",
     '    if const in src:  # MUTANT: substring check -- comments satisfy it\n'
     "        return True\n"
     "    for node in _ast.walk(tree):",
     "test_both_leaking_sites_isolate_the_heartbeat_constant",
     "the substring check a comment satisfies -- the sweep's own original defect"),

    # P4 REMOVED as an EQUIVALENT MUTANT, and the removal is the finding.
    # Narrowing WRITERS to ("record_cycle_end",) -- the filing's own narrower
    # set -- changes NO row in this repo, because no file calls
    # record_cycle_start without also calling record_cycle_end. Measured:
    #     for every .py under backend/ and scripts/, `record_cycle_start(`
    #     present AND `record_cycle_end(` absent -> zero files.
    # So the cell cannot discriminate here and scoring it KILLED would require
    # pointing it at a test that fails for an unrelated reason. The WIDER set
    # is still correct -- it is what keeps the sweep right the day someone adds
    # a start-only caller -- but this matrix cannot prove that today, and
    # saying so is better than manufacturing a kill.

    ("P5", CONFTEST,
     "        if snap is None or now_sha is None or snap[1] != now_sha:",
     "        if False:  # MUTANT: guard never fires",
     "test_the_conftest_guard_fires_and_repairs",
     "a runtime guard that never fires is caught"),

    ("P6", CONFTEST,
     "            if snap is None:\n                p.unlink(missing_ok=True)\n            else:\n                p.write_bytes(snap[0])",
     "            pass  # MUTANT: report but do NOT repair",
     "test_the_conftest_guard_fires_and_repairs",
     "a guard that reports without repairing -- leaving most of the damage"),

    ("P7", CONFTEST,
     '_PROTECTED_HANDOFF_FILES = {\n    "handoff/.cycle_heartbeat.json": "ledger_backed",\n    "handoff/cycle_history.jsonl": "append_only",\n}',
     '_PROTECTED_HANDOFF_FILES = {}  # MUTANT: nothing is protected',
     "test_the_conftest_guard_fires_and_repairs",
     "an empty protected set -- the guard passes vacuously over no files"),
# P8-P10 guard the CONCURRENCY fix added in cycle 2. The evaluator found
    # that the original guard would blindly restore its snapshot, which on this
    # local-only deployment can DELETE a real cycle row appended by the live
    # autonomous_loop mid-suite and then blame an innocent test. Making the
    # guard tolerant of a legitimate concurrent write risks the opposite
    # failure -- a guard that is now blind -- so all three arms are mutated.
    ("P8", CONFTEST,
     "    rule = _PROTECTED_HANDOFF_FILES.get(rel)",
     "    return True  # MUTANT: every change looks legitimate -> guard is blind\n"
     "    rule = _PROTECTED_HANDOFF_FILES.get(rel)",
     "test_the_guard_STILL_catches_a_test_leak_after_the_concurrency_fix",
     "a tolerance rule that makes the guard blind to real leaks is caught"),

    ("P9", CONFTEST,
     "        return after.startswith(before) and len(after) > len(before)",
     "        return True  # MUTANT: a REWRITE or TRUNCATION counts as an append",
     "test_a_real_APPEND_to_the_cycle_ledger_is_not_reverted",
     "treating a rewrite/truncation of the append-only ledger as an append"),

    ("P10", CONFTEST,
     "            return f'\"cycle_id\": \"{cid}\"' in ledger or f'\"cycle_id\":\"{cid}\"' in ledger",
     "            return True  # MUTANT: any cycle_id counts as real",
     "test_a_real_cycle_heartbeat_write_is_not_reverted_but_a_fixture_value_is",
     "accepting a fixture cycle_id that appears in ZERO ledger rows as legitimate"),
# P11 is the sub-expression cell a Q/A found missing: P9 mutates the whole
    # compound return, which the length-only fixtures catch, so the matrix read
    # 9/9 while the `startswith` half had NO falsifying fixture. Mutating each
    # half separately is what distinguishes "the expression is tested" from
    # "one clause of it is".
    ("P11", CONFTEST,
     "        return after.startswith(before) and len(after) > len(before)",
     "        return len(after) > len(before)  # MUTANT: prefix check dropped",
     "test_a_real_APPEND_to_the_cycle_ledger_is_not_reverted",
     "dropping ONLY the prefix half -- survived before the longer-rewrite fixture"),
]


def run_suite() -> tuple[int, str]:
    p = subprocess.run(
        [PY, "-m", "pytest", *SUITE, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True,
    )
    return p.returncode, p.stdout + p.stderr


_SUMMARY = re.compile(r"^=*\s*(?:\d+ \w+(?:, )?)+.*?in [\d.]+s", re.M)


def collected(out: str) -> int:
    """Tests that actually ran in the OUTER pytest, from its LAST summary line.

    Summing every `N passed` match over the whole output is wrong here and was
    measured wrong: several tests in this suite spawn a NESTED pytest, whose
    summary is captured into the failure text, so a failing cell counted the
    inner run too and scored UNSCORABLE for the wrong reason (20 != 10). Only
    the final summary line describes the outer run.
    """
    lines = [ln for ln in out.strip().splitlines()
             if re.search(r"\b\d+ (passed|failed|error)", ln) and " in " in ln]
    if not lines:
        return 0
    return sum(int(m.group(1))
               for m in re.finditer(r"(\d+) (passed|failed|error)", lines[-1]))


def main() -> int:
    print("=" * 70)
    print("CONTROL (must be GREEN before any cell is scored)")
    print("=" * 70)
    rc, out = run_suite()
    control_n = collected(out)
    print(out.strip().splitlines()[-1] if out.strip() else "(no output)")
    print(f"CONTROL rc={rc}  collected={control_n}")
    if rc != 0 or control_n == 0:
        print("\nCONTROL IS NOT GREEN -- every cell would be UNSCORABLE. Aborting.")
        return 2

    # The heartbeat is snapshotted around the WHOLE matrix: several cells
    # deliberately break the isolation, so a cell can legitimately dirty the
    # real file while it runs.
    hb = REPO / "handoff" / ".cycle_heartbeat.json"
    hb_snap = hb.read_bytes()

    results = []
    for cid, relpath, old, new, must_fail, proves in CELLS:
        path = REPO / relpath
        original = path.read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        text = original.decode("utf-8")

        if text.count(old) != 1:
            results.append((cid, "UNSCORABLE", f"anchor matched {text.count(old)}x (need 1)", proves))
            print(f"{cid}: UNSCORABLE -- anchor matched {text.count(old)} times")
            continue

        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        try:
            m_rc, m_out = run_suite()
            m_n = collected(m_out)
            named = must_fail in m_out
            if m_rc == 5:
                verdict, why = "UNSCORABLE", "pytest exit 5 -- nothing collected"
            elif m_rc == 0:
                verdict, why = "SURVIVED", "suite stayed green under the mutation"
            elif m_rc != 1:
                verdict, why = "UNSCORABLE", f"pytest exit {m_rc} (not a test failure)"
            elif m_n != control_n:
                verdict, why = "UNSCORABLE", f"collected {m_n} != control {control_n}"
            elif not named:
                verdict, why = "UNSCORABLE", f"red, but {must_fail} was not the failing test"
            else:
                verdict, why = "KILLED", f"{must_fail} failed; collected {m_n} == control"
        finally:
            path.write_bytes(original)
            restore_ok = hashlib.sha256(path.read_bytes()).hexdigest() == digest
            hb.write_bytes(hb_snap)

        if not restore_ok:
            print(f"\nFATAL: restore of {relpath} did NOT verify. Aborting.")
            return 3
        results.append((cid, verdict, why, proves))
        print(f"{cid}: {verdict:11s} {why}")

    print("\n" + "=" * 70)
    print("MATRIX")
    print("=" * 70)
    for cid, verdict, why, proves in results:
        print(f"  {cid:4s} {verdict:11s} {proves}")
        if verdict != "KILLED":
            print(f"        -> {why}")
    survivors = [c for c, v, *_ in results if v == "SURVIVED"]
    unscorable = [c for c, v, *_ in results if v == "UNSCORABLE"]
    print(f"\nKILLED={len([1 for _, v, *_ in results if v == 'KILLED'])}/{len(results)}"
          f"  SURVIVORS={survivors or 'none'}  UNSCORABLE={unscorable or 'none'}")
    print("\nRESTORE VERIFIED: every cell re-hashed to its pre-mutation SHA-256, "
          "and the real heartbeat was restored after every cell.")
    return 0 if not survivors and not unscorable else 1


if __name__ == "__main__":
    raise SystemExit(main())
