#!/usr/bin/env python3
"""phase-86.86 (D6) mutation matrix -- prove the INGRESS-seam guards can FAIL.

    python3 scripts/qa/mutation_matrix_86_86.py

Criterion 7: "restore `or _LITE_RISK_DEFAULT['recommended_position_pct']`
byte-identically at EACH fixed site and show the new assertion goes RED, with
the control observed GREEN first and the restore verified byte-identical".

WHY THIS IS A SEPARATE FILE FROM `mutation_matrix_86_74.py`. That matrix is a
CLOSED step's artifact and is left untouched (freeze discipline). More
importantly its cells all target the CONSUMER (`portfolio_manager.py`); these
target the PRODUCER, and the 86.86 research gate's finding was precisely that
*"the existing mutation matrix is complete only over the subject it names and
has no producer cell"*. Two matrices with different subjects are honest; one
matrix claiming to cover both would be the "matching totals hide contradictory
content" failure.

WHAT A MATRIX LICENSES. Only "these N mutations were killed". It is NOT a claim
that the guards are complete, and this file never makes one. Completeness of the
CLASS is a separate, derived question answered by
`scripts/qa/verify_lite_risk_seam_86_86.py`, which AST-enumerates the members
and fails if its own positive control cannot find a member it is handed.

The scoring rules are inherited deliberately, each paid for previously:
  * control GREEN first, else every cell is UNSCORABLE;
  * a no-match `str.replace` is NOT_APPLIED, never a silent pass;
  * the `-k` selector must be proven live, because pytest exits 5 on an empty
    selection and `rc != 0` would score that as a KILL;
  * byte-identical restore verified by sha256;
  * mutate the REAL module the tests import -- a cell aimed at a site no test
    executes is UNSCORABLE by construction.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# Reuse the 86.74 harness helpers rather than re-implementing them. `selected()`
# in particular encodes the pytest-exit-5 vacuity fix, and a second copy of that
# subtlety is exactly the drift this step exists to remove.
from mutation_matrix_86_74 import ROOT, SUITE, run, selected, sha  # noqa: E402

AL = ROOT / "backend" / "services" / "autonomous_loop.py"
SEAM_CHECKER = "scripts/qa/verify_lite_risk_seam_86_86.py"

#: (id, description, old, new, the test that must go RED)
MUTATIONS = [
    (
        "D6-M1",
        "restore the falsy-zero `or _LITE_RISK_DEFAULT[...]` in the producer",
        '        "recommended_position_pct": _lite_position_pct(risk_dict, ticker),',
        '        "recommended_position_pct": float(\n'
        '            risk_dict.get("recommended_position_pct")\n'
        '            or _LITE_RISK_DEFAULT["recommended_position_pct"]\n'
        '        ),  # MUTANT: the D6 defect restored verbatim',
        "TestLiteProducerKeepsAnExplicitZero",
    ),
    (
        "D6-M2",
        "the mirror collapse: map ABSENT to 0.0 instead of the default",
        '    return float(_LITE_RISK_DEFAULT["recommended_position_pct"])',
        '    return 0.0  # MUTANT: absent collapsed onto zero -- the other collapse',
        "TestLiteProducerKeepsAnExplicitZero",
    ),
    (
        "D6-M3",
        "let an UNPARSEABLE verdict reach the default instead of failing closed",
        "        return 0.0\n    return float(_LITE_RISK_DEFAULT",
        "        return float(_LITE_RISK_DEFAULT[\"recommended_position_pct\"])  # MUTANT\n"
        "    return float(_LITE_RISK_DEFAULT",
        "TestLiteProducerUnparseableFailsClosed",
    ),
    (
        "D6-M4",
        "silence the UNPARSEABLE warning (fail closed but no longer loud)",
        '        logger.warning(\n'
        '            "Lite risk judge for %s: recommended_position_pct is present but "',
        '        logger.debug(  # MUTANT: demoted below WARNING -- silently swallowed\n'
        '            "Lite risk judge for %s: recommended_position_pct is present but "',
        "test_unparseable_is_LOUD_not_silent",
    ),
    (
        "D6-M5",
        "reintroduce the falsy-zero downstream of the producer, end-to-end",
        "    verdict = _resolve_position_pct(risk_dict, {})\n"
        "    if verdict.kind == SIZE:",
        "    verdict = _resolve_position_pct(risk_dict, {})\n"
        "    if verdict.kind == SIZE and verdict.pct:  # MUTANT: falsy -- 0.0 falls through",
        "TestLiteZeroProducesNoOrderEndToEnd",
    ),
]

SUBJECTS = {mid: AL for mid, *_ in MUTATIONS}


def run_seam_checker() -> int:
    """The AST checker is a second, independent guard on the same property."""
    import subprocess
    r = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), SEAM_CHECKER],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    return r.returncode


def main() -> int:
    baseline = {p: (p.read_text(), sha(p)) for p in set(SUBJECTS.values())}
    for p, (_, s) in sorted(baseline.items(), key=lambda kv: str(kv[0])):
        print(f"subject : {p.relative_to(ROOT)}  sha={s[:12]}")
    print()

    print("=== CONTROL (must be GREEN before any cell is scorable) ===")
    rc, tail = run(SUITE)
    control_green = rc == 0
    print(f"  suite control       : {'GREEN' if control_green else 'RED'}")
    seam_rc = run_seam_checker()
    seam_green = seam_rc == 0
    print(f"  seam-checker control: {'GREEN' if seam_green else 'RED'}\n")

    results: list[tuple[str, str, str]] = []
    for mid, desc, old, new, target in MUTATIONS:
        subject = SUBJECTS[mid]
        original, original_sha = baseline[subject]

        if not control_green:
            results.append((mid, "UNSCORABLE", "control was not green"))
            print(f"  {mid}: UNSCORABLE -- control not green")
            continue

        if old not in original:
            results.append((mid, "NOT_APPLIED", f"target absent in {subject.name}"))
            print(f"  {mid}: NOT_APPLIED -- target absent, cell not scored")
            continue

        n_sel = selected(target)
        if n_sel == 0:
            results.append((mid, "UNSCORABLE", f"-k {target!r} selected 0 tests"))
            print(f"  {mid}: UNSCORABLE -- selector matches nothing, cell not scored")
            continue

        try:
            subject.write_text(original.replace(old, new, 1))
            assert sha(subject) != original_sha, "mutation did not change the file"
            rc_m, _ = run(SUITE, "-k", target)
            killed = rc_m == 1
            verdict = "KILLED" if killed else ("UNSCORABLE" if rc_m == 5 else "SURVIVED")
            results.append((mid, verdict, f"{subject.name} [{n_sel} sel]: {desc}"))
            print(f"  {mid}: {verdict:<10} ({n_sel} tests selected)  {desc}")
        finally:
            # NOT `return` in the finally block -- that swallows an in-flight
            # exception and would report a restore failure as an orderly exit.
            subject.write_text(original)
            back = sha(subject)
        if back != original_sha:
            print(f"  !!! RESTORE FAILED for {mid}: {back[:16]} != {original_sha[:16]}")
            return 3

    # A dedicated cell for the AST checker: break the "both paths route through
    # the one producer" property and require the checker -- not the suite -- to
    # notice. Without this the checker's own guards are unproven.
    print()
    print("=== SEAM-CHECKER CELL (the AST guard must be able to go RED) ===")
    subject, (original, original_sha) = AL, baseline[AL]
    seam_cell = ("SEAM-M1", "point the Gemini lite path off the shared producer")
    old = ("    # phase-86.86 (D6): ONE producer, shared with the Claude lite path.\n"
           "    risk_assessment = _build_lite_risk_assessment(risk_dict, ticker)")
    new = ('    risk_assessment = {  # MUTANT: second parallel idiom\n'
           '        "decision": str(risk_dict.get("decision") or _LITE_RISK_DEFAULT["decision"]),\n'
           '        "reasoning": "m", "reason": "m",\n'
           '        "recommended_position_pct": float(\n'
           '            risk_dict.get("recommended_position_pct")\n'
           '            or _LITE_RISK_DEFAULT["recommended_position_pct"]\n'
           '        ),\n'
           '        "risk_level": str(risk_dict.get("risk_level") or _LITE_RISK_DEFAULT["risk_level"]),\n'
           '        "risk_limits": dict(risk_dict.get("risk_limits") or _LITE_RISK_DEFAULT["risk_limits"]),\n'
           '    }')
    if not seam_green:
        results.append((seam_cell[0], "UNSCORABLE", "seam-checker control was not green"))
        print(f"  {seam_cell[0]}: UNSCORABLE -- control not green")
    elif old not in original:
        results.append((seam_cell[0], "NOT_APPLIED", "target absent"))
        print(f"  {seam_cell[0]}: NOT_APPLIED -- target absent, cell not scored")
    else:
        try:
            subject.write_text(original.replace(old, new, 1))
            assert sha(subject) != original_sha
            rc_seam = run_seam_checker()
            verdict = "KILLED" if rc_seam != 0 else "SURVIVED"
            results.append((seam_cell[0], verdict, f"seam checker: {seam_cell[1]}"))
            print(f"  {seam_cell[0]}: {verdict:<10} {seam_cell[1]}")
        finally:
            subject.write_text(original)
            back = sha(subject)
        if back != original_sha:
            print(f"  !!! RESTORE FAILED for {seam_cell[0]}")
            return 3

    all_restored = all(sha(p) == s for p, (_, s) in baseline.items())
    print(f"\nrestore verified byte-identical for all {len(baseline)} subject(s): {all_restored}")
    if not all_restored:
        return 3

    print("\n=== MATRIX ===")
    for mid, verdict, note in results:
        print(f"  {mid:<8} {verdict:<12} {note}")
    print(f"\ncells scored: {len(results)}")

    bad = [m for m, v, _ in results if v != "KILLED"]
    if bad:
        print(f"\nRESULT: cells not KILLED: {bad}")
        return 1
    print(f"\nRESULT: all {len(results)} cells KILLED. This licenses exactly "
          f"'these {len(results)} mutations were killed' -- no global claim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
